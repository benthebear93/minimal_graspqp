from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import importlib.util
import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh as tm

from minimal_graspqp.assets import resolve_shadow_hand_asset_dir

with_torchsdf = importlib.util.find_spec("torchsdf") is not None
if with_torchsdf:
    from torchsdf import compute_sdf, index_vertices_by_faces


DEFAULT_JOINT_ORDER = [
    "robot0_WRJ1",
    "robot0_WRJ0",
    "robot0_FFJ3",
    "robot0_FFJ2",
    "robot0_FFJ1",
    "robot0_FFJ0",
    "robot0_LFJ4",
    "robot0_LFJ3",
    "robot0_LFJ2",
    "robot0_LFJ1",
    "robot0_LFJ0",
    "robot0_MFJ3",
    "robot0_MFJ2",
    "robot0_MFJ1",
    "robot0_MFJ0",
    "robot0_RFJ3",
    "robot0_RFJ2",
    "robot0_RFJ1",
    "robot0_RFJ0",
    "robot0_THJ4",
    "robot0_THJ3",
    "robot0_THJ2",
    "robot0_THJ1",
    "robot0_THJ0",
]

DEFAULT_JOINT_STATE = torch.tensor(
    [
        0.0,
        0.0,
        0.1,
        0.0,
        0.6,
        0.0,
        0.0,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.6,
        0.0,
        -0.1,
        0.0,
        0.6,
        0.0,
        0.0,
        1.2,
        0.0,
        -0.2,
        0.0,
    ],
    dtype=torch.float32,
)

FINGERTIP_LINK_NAMES = [
    "robot0_ffdistal",
    "robot0_mfdistal",
    "robot0_rfdistal",
    "robot0_lfdistal",
    "robot0_thdistal",
]


@dataclass
class ShadowHandMetadata:
    asset_dir: Path
    urdf_path: Path
    mesh_dir: Path
    contact_mesh_dir: Path
    joint_names: list[str]
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    default_joint_state: torch.Tensor
    contact_candidate_points: torch.Tensor
    contact_candidate_normals: torch.Tensor
    contact_candidate_links: list[str]
    penetration_points: dict[str, torch.Tensor]
    collision_vertices: dict[str, torch.Tensor]
    collision_faces: dict[str, torch.Tensor]
    collision_primitives: dict[str, list[dict[str, object]]]

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_contact_candidates(self) -> int:
        return int(self.contact_candidate_points.shape[0])


def _parse_joint_limits(urdf_path: Path, joint_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    root = ET.fromstring(urdf_path.read_text())
    lower = []
    upper = []
    joint_names_set = set(joint_names)
    limits_by_name = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        if name not in joint_names_set:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits_by_name[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    for joint_name in joint_names:
        lo, hi = limits_by_name[joint_name]
        lower.append(lo)
        upper.append(hi)
    return torch.tensor(lower, dtype=torch.float32), torch.tensor(upper, dtype=torch.float32)


def _farthest_point_indices(points: np.ndarray, k: int) -> np.ndarray:
    if len(points) <= k:
        return np.arange(len(points), dtype=np.int64)
    selected = np.empty((k,), dtype=np.int64)
    selected[0] = 0
    min_dist_sq = np.sum((points - points[0]) ** 2, axis=1)
    for idx in range(1, k):
        selected[idx] = int(np.argmax(min_dist_sq))
        dist_sq = np.sum((points - points[selected[idx]]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
    return selected


def _sample_surface_candidates(mesh: tm.Trimesh, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    sampled_points, face_indices = tm.sample.sample_surface_even(mesh, max(1000, 50 * n_points))
    selected = _farthest_point_indices(sampled_points, n_points)
    points = sampled_points[selected]
    normals = mesh.face_normals[face_indices[selected]]
    return points.astype(np.float32), normals.astype(np.float32)


def _load_contact_candidates(asset_dir: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    contact_points_data = json.loads((asset_dir / "contact_points.json").read_text())
    candidates = []
    normals = []
    links = []
    mesh_root = asset_dir / "meshes"
    mesh_cache: dict[Path, tm.Trimesh] = {}
    for link_name, entries in contact_points_data.items():
        for entry in entries:
            if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str):
                relative_path, num_points = entry
                mesh_path = (mesh_root / relative_path).resolve()
                if mesh_path not in mesh_cache:
                    mesh_cache[mesh_path] = tm.load(mesh_path, force="mesh", process=False)
                points, point_normals = _sample_surface_candidates(mesh_cache[mesh_path], int(num_points))
                candidates.extend(points.tolist())
                normals.extend(point_normals.tolist())
                links.extend([link_name] * len(points))
                continue
            if isinstance(entry, list) and len(entry) == 3:
                point = np.asarray(entry, dtype=np.float32)
                candidates.append(point.tolist())
                normals.append(np.zeros(3, dtype=np.float32).tolist())
                links.append(link_name)
                continue
            raise ValueError(f"Unsupported contact candidate entry for {link_name}: {entry}")
    return torch.tensor(candidates, dtype=torch.float32), torch.tensor(normals, dtype=torch.float32), links


def _load_penetration_points(asset_dir: Path) -> dict[str, torch.Tensor]:
    raw = json.loads((asset_dir / "penetration_points.json").read_text())
    return {name: torch.tensor(points, dtype=torch.float32) for name, points in raw.items()}


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _origin_transform(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    xyz = np.fromstring(element.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float32)
    rpy = np.fromstring(element.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float32)
    if xyz.size != 3:
        xyz = np.zeros(3, dtype=np.float32)
    if rpy.size != 3:
        rpy = np.zeros(3, dtype=np.float32)
    return _rpy_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2])), xyz


def _load_geometry_mesh(asset_dir: Path, geometry: ET.Element) -> tm.Trimesh:
    mesh_node = geometry.find("mesh")
    if mesh_node is not None:
        mesh_path = asset_dir / mesh_node.attrib["filename"]
        mesh = tm.load(mesh_path, force="mesh", process=False)
        scale = np.fromstring(mesh_node.attrib.get("scale", "1 1 1"), sep=" ", dtype=np.float32)
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale.reshape(1, 3)
        return mesh
    box_node = geometry.find("box")
    if box_node is not None:
        size = np.fromstring(box_node.attrib["size"], sep=" ", dtype=np.float32)
        return tm.creation.box(extents=size)
    sphere_node = geometry.find("sphere")
    if sphere_node is not None:
        return tm.primitives.Sphere(radius=float(sphere_node.attrib["radius"]))
    cylinder_node = geometry.find("cylinder")
    if cylinder_node is not None:
        return tm.primitives.Cylinder(radius=float(cylinder_node.attrib["radius"]), height=float(cylinder_node.attrib["length"]))
    raise ValueError("Unsupported geometry node in Shadow Hand URDF.")


def _load_collision_meshes(
    asset_dir: Path,
    urdf_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list[dict[str, object]]]]:
    root = ET.fromstring(urdf_path.read_text())
    vertices_by_link: dict[str, torch.Tensor] = {}
    faces_by_link: dict[str, torch.Tensor] = {}
    primitives_by_link: dict[str, list[dict[str, object]]] = {}
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        elements = link.findall("collision")
        if not elements:
            elements = link.findall("visual")
        if not elements:
            continue
        link_vertices = []
        link_faces = []
        link_primitives = []
        vertex_offset = 0
        for element in elements:
            geometry = element.find("geometry")
            if geometry is None:
                continue
            rotation, translation = _origin_transform(element.find("origin"))
            box_node = geometry.find("box")
            sphere_node = geometry.find("sphere")
            cylinder_node = geometry.find("cylinder")
            if box_node is not None:
                size = np.fromstring(box_node.attrib["size"], sep=" ", dtype=np.float32)
                link_primitives.append(
                    {
                        "type": "box",
                        "half_extents": torch.tensor(size * 0.5, dtype=torch.float32),
                        "rotation": torch.tensor(rotation, dtype=torch.float32),
                        "translation": torch.tensor(translation, dtype=torch.float32),
                    }
                )
                continue
            if sphere_node is not None:
                link_primitives.append(
                    {
                        "type": "sphere",
                        "radius": float(sphere_node.attrib["radius"]),
                        "rotation": torch.tensor(rotation, dtype=torch.float32),
                        "translation": torch.tensor(translation, dtype=torch.float32),
                    }
                )
                continue
            if cylinder_node is not None:
                link_primitives.append(
                    {
                        "type": "cylinder",
                        "radius": float(cylinder_node.attrib["radius"]),
                        "half_height": 0.5 * float(cylinder_node.attrib["length"]),
                        "rotation": torch.tensor(rotation, dtype=torch.float32),
                        "translation": torch.tensor(translation, dtype=torch.float32),
                    }
                )
                continue
            mesh = _load_geometry_mesh(asset_dir, geometry)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            vertices = vertices @ rotation.T + translation.reshape(1, 3)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            link_vertices.append(torch.tensor(vertices, dtype=torch.float32))
            link_faces.append(torch.tensor(faces + vertex_offset, dtype=torch.long))
            vertex_offset += vertices.shape[0]
        if link_vertices:
            vertices_by_link[link_name] = torch.cat(link_vertices, dim=0)
            faces_by_link[link_name] = torch.cat(link_faces, dim=0)
        if link_primitives:
            primitives_by_link[link_name] = link_primitives
    return vertices_by_link, faces_by_link, primitives_by_link


def load_shadow_hand_metadata(asset_dir: str | Path | None = None) -> ShadowHandMetadata:
    resolved_dir = resolve_shadow_hand_asset_dir(asset_dir)
    urdf_path = resolved_dir / "shadow_hand.urdf"
    chain = pk.build_chain_from_urdf(urdf_path.read_text())
    joint_names = chain.get_joint_parameter_names()
    if joint_names != DEFAULT_JOINT_ORDER:
        raise ValueError("Unexpected Shadow Hand joint order in URDF.")
    joint_lower, joint_upper = _parse_joint_limits(urdf_path, joint_names)
    contact_candidate_points, contact_candidate_normals, contact_candidate_links = _load_contact_candidates(resolved_dir)
    collision_vertices, collision_faces, collision_primitives = _load_collision_meshes(resolved_dir, urdf_path)
    return ShadowHandMetadata(
        asset_dir=resolved_dir,
        urdf_path=urdf_path,
        mesh_dir=resolved_dir / "meshes",
        contact_mesh_dir=resolved_dir / "contact_mesh",
        joint_names=joint_names,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        default_joint_state=DEFAULT_JOINT_STATE.clone(),
        contact_candidate_points=contact_candidate_points,
        contact_candidate_normals=contact_candidate_normals,
        contact_candidate_links=contact_candidate_links,
        penetration_points=_load_penetration_points(resolved_dir),
        collision_vertices=collision_vertices,
        collision_faces=collision_faces,
        collision_primitives=collision_primitives,
    )


def filter_contact_candidates(
    metadata: ShadowHandMetadata,
    allowed_links: list[str] | tuple[str, ...],
) -> ShadowHandMetadata:
    allowed = set(allowed_links)
    mask = torch.tensor(
        [link_name in allowed for link_name in metadata.contact_candidate_links],
        dtype=torch.bool,
    )
    return replace(
        metadata,
        contact_candidate_points=metadata.contact_candidate_points[mask],
        contact_candidate_normals=metadata.contact_candidate_normals[mask],
        contact_candidate_links=[
            link_name
            for link_name in metadata.contact_candidate_links
            if link_name in allowed
        ],
    )


def apply_contact_candidate_overrides(
    metadata: ShadowHandMetadata,
    overrides: dict[int, list[float] | tuple[float, float, float]],
) -> ShadowHandMetadata:
    if not overrides:
        return metadata
    points = metadata.contact_candidate_points.clone()
    for index, point in overrides.items():
        if index < 0 or index >= points.shape[0]:
            raise IndexError(f"Contact candidate override index out of range: {index}")
        point_tensor = torch.tensor(point, dtype=points.dtype)
        if point_tensor.shape != (3,):
            raise ValueError(f"Contact candidate override for index {index} must have shape (3,).")
        points[index] = point_tensor
    return replace(metadata, contact_candidate_points=points)


def load_contact_candidate_overrides(path: str | Path | None) -> dict[int, list[float]]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    overrides: dict[int, list[float]] = {}
    for key, value in raw.items():
        overrides[int(key)] = value
    return overrides


class ShadowHandModel:
    """Minimal Shadow Hand wrapper for FK and contact candidate transforms."""

    def __init__(self, metadata: ShadowHandMetadata, device: str | torch.device = "cpu", dtype=torch.float32):
        self.metadata = metadata
        self.device = torch.device(device)
        self.dtype = dtype
        self.chain = pk.build_chain_from_urdf(metadata.urdf_path.read_text()).to(dtype=dtype, device=self.device)
        self._candidate_points = metadata.contact_candidate_points.to(device=self.device, dtype=dtype)
        self._candidate_normals = metadata.contact_candidate_normals.to(device=self.device, dtype=dtype)
        self._penetration_links = list(metadata.penetration_points.keys())
        self._collision_meshes: dict[str, dict[str, object]] = {}
        collision_links = set(metadata.collision_vertices.keys()) | set(metadata.collision_primitives.keys())
        for link_name in collision_links:
            vertices = metadata.collision_vertices.get(link_name)
            faces = metadata.collision_faces.get(link_name)
            mesh_data: dict[str, object] = {
                "primitives": [],
            }
            if vertices is not None and faces is not None:
                vertices_device = vertices.to(device=self.device, dtype=self.dtype)
                faces = faces.to(device=self.device)
                mesh_data["vertices"] = vertices_device
                mesh_data["faces"] = faces
                if with_torchsdf:
                    mesh_data["face_verts"] = index_vertices_by_faces(
                        vertices_device.to(dtype=torch.float32),
                        faces,
                    )
                else:
                    mesh_data["mesh"] = tm.Trimesh(
                        vertices_device.detach().cpu().numpy(),
                        faces.detach().cpu().numpy(),
                        process=False,
                    )
            for primitive in metadata.collision_primitives.get(link_name, []):
                mesh_data["primitives"].append(
                    {
                        "type": primitive["type"],
                        "rotation": primitive["rotation"].to(device=self.device, dtype=self.dtype),
                        "translation": primitive["translation"].to(device=self.device, dtype=self.dtype),
                        **{
                            key: value.to(device=self.device, dtype=self.dtype)
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in primitive.items()
                            if key not in {"type", "rotation", "translation"}
                        },
                    }
                )
            self._collision_meshes[link_name] = mesh_data

    @classmethod
    def create(
        cls,
        asset_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
        fingertips_only: bool = False,
        contact_points_override_path: str | Path | None = None,
    ) -> "ShadowHandModel":
        metadata = load_shadow_hand_metadata(asset_dir=asset_dir)
        if contact_points_override_path is not None:
            metadata = apply_contact_candidate_overrides(
                metadata,
                load_contact_candidate_overrides(contact_points_override_path),
            )
        if fingertips_only:
            metadata = filter_contact_candidates(metadata, FINGERTIP_LINK_NAMES)
        return cls(metadata, device=device)

    def default_joint_state(self, batch_size: int = 1) -> torch.Tensor:
        state = self.metadata.default_joint_state.to(device=self.device, dtype=self.dtype)
        return state.unsqueeze(0).repeat(batch_size, 1)

    def clamp_to_limits(self, joint_values: torch.Tensor) -> torch.Tensor:
        lower = self.metadata.joint_lower.to(device=self.device, dtype=self.dtype)
        upper = self.metadata.joint_upper.to(device=self.device, dtype=self.dtype)
        return joint_values.clamp(lower, upper)

    def forward_kinematics(self, joint_values: torch.Tensor) -> dict[str, torch.Tensor]:
        fk_result = self.chain.forward_kinematics(joint_values)
        return {name: transform.get_matrix() for name, transform in fk_result.items()}

    def apply_wrist_pose(
        self,
        points: torch.Tensor,
        wrist_translation: torch.Tensor | None = None,
        wrist_rotation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if wrist_rotation is None:
            wrist_rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(points.shape[0], -1, -1)
        if wrist_translation is None:
            wrist_translation = torch.zeros((points.shape[0], 3), device=points.device, dtype=points.dtype)
        return points @ wrist_rotation.transpose(-1, -2) + wrist_translation.unsqueeze(1)

    def contact_candidates_world(
        self,
        joint_values: torch.Tensor,
        indices: torch.Tensor | None = None,
        wrist_translation: torch.Tensor | None = None,
        wrist_rotation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        transforms = self.forward_kinematics(joint_values)
        batch_size = joint_values.shape[0]
        if indices is None:
            local_points = self._candidate_points.unsqueeze(0).expand(batch_size, -1, -1)
            links = self.metadata.contact_candidate_links
        else:
            local_points = self._candidate_points[indices]
            links = [self.metadata.contact_candidate_links[idx] for idx in indices.reshape(-1).tolist()]
            links = [links[row * indices.shape[1] : (row + 1) * indices.shape[1]] for row in range(batch_size)]

        if indices is None:
            world_points = []
            for candidate_id, link_name in enumerate(links):
                transform = transforms[link_name]
                point = local_points[:, candidate_id]
                hom = torch.cat([point, torch.ones(batch_size, 1, device=self.device, dtype=self.dtype)], dim=-1)
                world_points.append((transform @ hom.unsqueeze(-1)).squeeze(-1)[..., :3])
            return self.apply_wrist_pose(torch.stack(world_points, dim=1), wrist_translation=wrist_translation, wrist_rotation=wrist_rotation)

        world_points = []
        for batch_id in range(batch_size):
            batch_points = []
            for column_id, link_name in enumerate(links[batch_id]):
                transform = transforms[link_name][batch_id]
                point = local_points[batch_id, column_id]
                hom = torch.cat([point, torch.ones(1, device=self.device, dtype=self.dtype)], dim=0)
                batch_points.append((transform @ hom.unsqueeze(-1)).squeeze(-1)[..., :3])
            world_points.append(torch.stack(batch_points, dim=0))
        return self.apply_wrist_pose(torch.stack(world_points, dim=0), wrist_translation=wrist_translation, wrist_rotation=wrist_rotation)

    def penetration_spheres_world(
        self,
        joint_values: torch.Tensor,
        wrist_translation: torch.Tensor | None = None,
        wrist_rotation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        transforms = self.forward_kinematics(joint_values)
        batch_size = joint_values.shape[0]
        centers = []
        radii = []
        link_names = []
        for link_name in self._penetration_links:
            link_points = self.metadata.penetration_points[link_name].to(device=self.device, dtype=self.dtype)
            local_centers = link_points[:, :3]
            local_radii = link_points[:, 3]
            transform = transforms[link_name]
            hom = torch.cat(
                [local_centers.unsqueeze(0).expand(batch_size, -1, -1), torch.ones((batch_size, local_centers.shape[0], 1), device=self.device, dtype=self.dtype)],
                dim=-1,
            )
            world_centers = (transform.unsqueeze(1) @ hom.unsqueeze(-1)).squeeze(-1)[..., :3]
            centers.append(world_centers)
            radii.append(local_radii.unsqueeze(0).expand(batch_size, -1))
            link_names.extend([link_name] * local_centers.shape[0])
        centers_cat = torch.cat(centers, dim=1)
        radii_cat = torch.cat(radii, dim=1)
        centers_cat = self.apply_wrist_pose(centers_cat, wrist_translation=wrist_translation, wrist_rotation=wrist_rotation)
        return centers_cat, radii_cat, link_names

    def cal_distance(
        self,
        points_world: torch.Tensor,
        joint_values: torch.Tensor,
        wrist_translation: torch.Tensor | None = None,
        wrist_rotation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._collision_meshes:
            raise ValueError("Shadow Hand collision meshes are unavailable.")

        points = points_world
        if wrist_rotation is None:
            wrist_rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(points.shape[0], -1, -1)
        if wrist_translation is None:
            wrist_translation = torch.zeros((points.shape[0], 3), device=points.device, dtype=points.dtype)
        points = (points - wrist_translation.unsqueeze(1)) @ wrist_rotation

        transforms = self.forward_kinematics(joint_values)
        distances = []
        for link_name, mesh_data in self._collision_meshes.items():
            transform = transforms[link_name]
            points_local = (points - transform[:, :3, 3].unsqueeze(1)) @ transform[:, :3, :3]
            flat_points = points_local.reshape(-1, 3)
            link_distances = []
            if "face_verts" in mesh_data:
                face_verts = mesh_data["face_verts"]
                if face_verts.dtype != flat_points.dtype:
                    face_verts = face_verts.to(dtype=flat_points.dtype)
                squared_distance, signs, _, _ = compute_sdf(flat_points, face_verts)
                link_distance = torch.sqrt(squared_distance + 1e-8) * (-signs.to(dtype=flat_points.dtype))
                link_distances.append(link_distance.reshape(points.shape[0], points.shape[1]))
            elif "mesh" in mesh_data:
                query = flat_points.detach().cpu().numpy()
                mesh = mesh_data["mesh"]
                link_distance = tm.proximity.signed_distance(mesh, query)
                link_distances.append(torch.tensor(link_distance, dtype=points.dtype, device=points.device).reshape(points.shape[0], points.shape[1]))
            for primitive in mesh_data.get("primitives", []):
                primitive_points = (points_local - primitive["translation"].view(1, 1, 3)) @ primitive["rotation"]
                if primitive["type"] == "box":
                    half_extents = primitive["half_extents"].view(1, 1, 3)
                    q = primitive_points.abs() - half_extents
                    outside = q.clamp_min(0.0).norm(dim=-1)
                    inside = torch.minimum(q.max(dim=-1).values, torch.zeros((), device=q.device, dtype=q.dtype))
                    link_distances.append(-(outside + inside))
                elif primitive["type"] == "sphere":
                    radius = primitive_points.new_tensor(float(primitive["radius"]))
                    link_distances.append(radius - primitive_points.norm(dim=-1))
                elif primitive["type"] == "cylinder":
                    radial = primitive_points[..., :2].norm(dim=-1)
                    d = torch.stack(
                        [
                            radial - primitive_points.new_tensor(float(primitive["radius"])),
                            primitive_points[..., 2].abs() - primitive_points.new_tensor(float(primitive["half_height"])),
                        ],
                        dim=-1,
                    )
                    outside = d.clamp_min(0.0).norm(dim=-1)
                    inside = torch.minimum(d.max(dim=-1).values, torch.zeros((), device=d.device, dtype=d.dtype))
                    link_distances.append(-(outside + inside))
            if link_distances:
                distances.append(torch.stack(link_distances, dim=0).max(dim=0).values)
        return torch.stack(distances, dim=0).max(dim=0).values
