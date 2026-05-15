from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh as tm

from minimal_graspqp.hands.shadow_hand import (
    DEFAULT_JOINT_ORDER,
    DEFAULT_JOINT_STATE,
    FINGERTIP_LINK_NAMES,
    resolve_contact_link_names,
    with_torchsdf,
)

if with_torchsdf:
    from torchsdf import compute_sdf, index_vertices_by_faces


LOCAL_MUJOCO_HAND_XML = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "mujoco_shadow_hand_in_hand_rolling"
    / "right_hand.xml"
)

ROBOT0_TO_RH_JOINT = {
    "robot0_WRJ1": "rh_WRJ2",
    "robot0_WRJ0": "rh_WRJ1",
    "robot0_FFJ3": "rh_FFJ4",
    "robot0_FFJ2": "rh_FFJ3",
    "robot0_FFJ1": "rh_FFJ2",
    "robot0_FFJ0": "rh_FFJ1",
    "robot0_LFJ4": "rh_LFJ5",
    "robot0_LFJ3": "rh_LFJ4",
    "robot0_LFJ2": "rh_LFJ3",
    "robot0_LFJ1": "rh_LFJ2",
    "robot0_LFJ0": "rh_LFJ1",
    "robot0_MFJ3": "rh_MFJ4",
    "robot0_MFJ2": "rh_MFJ3",
    "robot0_MFJ1": "rh_MFJ2",
    "robot0_MFJ0": "rh_MFJ1",
    "robot0_RFJ3": "rh_RFJ4",
    "robot0_RFJ2": "rh_RFJ3",
    "robot0_RFJ1": "rh_RFJ2",
    "robot0_RFJ0": "rh_RFJ1",
    "robot0_THJ4": "rh_THJ5",
    "robot0_THJ3": "rh_THJ4",
    "robot0_THJ2": "rh_THJ3",
    "robot0_THJ1": "rh_THJ2",
    "robot0_THJ0": "rh_THJ1",
}

ROBOT0_TO_RH_JOINT_SIGN = {
    "robot0_THJ1": -1.0,
    "robot0_THJ0": -1.0,
}

GRASPQP_TO_MUJOCO_BODY = {
    "robot0_palm": "rh_palm",
    "robot0_ffproximal": "rh_ffproximal",
    "robot0_ffmiddle": "rh_ffmiddle",
    "robot0_ffdistal": "rh_ffdistal",
    "robot0_mfproximal": "rh_mfproximal",
    "robot0_mfmiddle": "rh_mfmiddle",
    "robot0_mfdistal": "rh_mfdistal",
    "robot0_rfproximal": "rh_rfproximal",
    "robot0_rfmiddle": "rh_rfmiddle",
    "robot0_rfdistal": "rh_rfdistal",
    "robot0_lfproximal": "rh_lfproximal",
    "robot0_lfmiddle": "rh_lfmiddle",
    "robot0_lfdistal": "rh_lfdistal",
    "robot0_thproximal": "rh_thproximal",
    "robot0_thmiddle": "rh_thmiddle",
    "robot0_thdistal": "rh_thdistal",
}

EXTRA_BODY_TO_LINK = {
    "rh_forearm": "root_robot0_forearm",
    "rh_wrist": "robot0_wrist",
    "rh_ffknuckle": "robot0_ffknuckle",
    "rh_mfknuckle": "robot0_mfknuckle",
    "rh_rfknuckle": "robot0_rfknuckle",
    "rh_lfmetacarpal": "robot0_lfmetacarpal",
    "rh_lfknuckle": "robot0_lfknuckle",
    "rh_thbase": "robot0_thbase",
    "rh_thhub": "robot0_thhub",
}

CONTACT_COUNTS = {
    "robot0_ffproximal": 8,
    "robot0_ffmiddle": 8,
    "robot0_ffdistal": 16,
    "robot0_mfproximal": 8,
    "robot0_mfmiddle": 8,
    "robot0_mfdistal": 16,
    "robot0_rfproximal": 8,
    "robot0_rfmiddle": 8,
    "robot0_rfdistal": 16,
    "robot0_lfproximal": 8,
    "robot0_lfmiddle": 8,
    "robot0_lfdistal": 16,
    "robot0_thproximal": 8,
    "robot0_thmiddle": 8,
    "robot0_thdistal": 16,
}


def _body_to_link_name(body_name: str) -> str:
    inverse = {rh: robot0 for robot0, rh in GRASPQP_TO_MUJOCO_BODY.items()}
    if body_name in inverse:
        return inverse[body_name]
    if body_name in EXTRA_BODY_TO_LINK:
        return EXTRA_BODY_TO_LINK[body_name]
    if body_name.startswith("rh_"):
        return "robot0_" + body_name.removeprefix("rh_")
    return body_name


def _resolve_mjcf_mesh_paths(root, xml_path: Path) -> None:
    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir") if compiler is not None else None
    base_dir = xml_path.parent / meshdir if meshdir else xml_path.parent
    for mesh in root.findall("./asset/mesh"):
        filename = mesh.attrib.get("file")
        if filename is None:
            continue
        path = Path(filename)
        if not path.is_absolute():
            mesh.attrib["file"] = str((base_dir / path).resolve())
    if compiler is not None and "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]


def _load_model(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    _resolve_mjcf_mesh_paths(root, xml_path)
    tmp_path = Path("/tmp/minimal_graspqp_mjcf_shadow_hand.xml")
    ET.ElementTree(root).write(tmp_path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(tmp_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_from_pos_matrix(position: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return out


def _mesh_from_geom(model: mujoco.MjModel, geom_id: int) -> tm.Trimesh | None:
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        va = int(model.mesh_vertadr[mesh_id])
        vn = int(model.mesh_vertnum[mesh_id])
        fa = int(model.mesh_faceadr[mesh_id])
        fn = int(model.mesh_facenum[mesh_id])
        return tm.Trimesh(
            vertices=np.asarray(model.mesh_vert[va : va + vn], dtype=np.float64),
            faces=np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int64),
            process=False,
        )
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return tm.creation.box(extents=2.0 * size[:3])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return tm.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return tm.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]), sections=32)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        if hasattr(tm.creation, "capsule"):
            return tm.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]), count=[16, 16])
        return tm.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]), sections=32)
    return None


def _body_local_geom_mesh(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tm.Trimesh | None:
    mesh = _mesh_from_geom(model, geom_id)
    if mesh is None:
        return None
    body_id = int(model.geom_bodyid[geom_id])
    body_tf = _transform_from_pos_matrix(data.xpos[body_id], data.xmat[body_id])
    geom_tf = _transform_from_pos_matrix(data.geom_xpos[geom_id], data.geom_xmat[geom_id])
    mesh = mesh.copy()
    mesh.apply_transform(np.linalg.inv(body_tf) @ geom_tf)
    return mesh


def _body_local_mesh(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, groups: tuple[int, ...]) -> tm.Trimesh | None:
    meshes = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if int(model.geom_group[geom_id]) not in groups:
            continue
        mesh = _body_local_geom_mesh(model, data, geom_id)
        if mesh is not None and len(mesh.vertices) and len(mesh.faces):
            meshes.append(mesh)
    if not meshes:
        return None
    return tm.util.concatenate(meshes)


def _farthest_point_indices(points: np.ndarray, k: int) -> np.ndarray:
    if len(points) <= k:
        return np.arange(len(points), dtype=np.int64)
    selected = np.empty((k,), dtype=np.int64)
    selected[0] = int(np.argmax(points[:, 2]))
    min_dist_sq = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for idx in range(1, k):
        selected[idx] = int(np.argmax(min_dist_sq))
        dist_sq = np.sum((points - points[selected[idx]]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
    return selected


def _sample_contact_points(mesh: tm.Trimesh, count: int, normal_y_max: float, seed: int) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_indices = tm.sample.sample_surface(mesh, max(1024, count * 120))
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64)
    mask = normals[:, 1] <= normal_y_max
    if int(mask.sum()) >= count:
        points = points[mask]
    return np.asarray(points[_farthest_point_indices(points, count)], dtype=np.float32)


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(k.shape)
    angle = angle[..., None, None]
    return eye + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * (k @ k)


@dataclass
class MJCFShadowHandMetadata:
    mjcf_path: Path
    joint_names: list[str]
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    default_joint_state: torch.Tensor
    contact_candidate_points: torch.Tensor
    contact_candidate_normals: torch.Tensor
    contact_candidate_links: list[str]

    @property
    def num_contact_candidates(self) -> int:
        return int(self.contact_candidate_points.shape[0])


class MJCFShadowHandModel:
    def __init__(self, mjcf_path: str | Path = LOCAL_MUJOCO_HAND_XML, device: str | torch.device = "cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model, self.data = _load_model(Path(mjcf_path).expanduser().resolve())
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(self.model.nbody)]
        self.link_names = [_body_to_link_name(name) for name in self.body_names]
        self.link_to_body = {link: idx for idx, link in enumerate(self.link_names)}

        lower, upper = [], []
        self.rh_joint_for_robot0: list[int] = []
        self.robot0_to_rh_sign: list[float] = []
        for robot0_name in DEFAULT_JOINT_ORDER:
            rh_name = ROBOT0_TO_RH_JOINT[robot0_name]
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, rh_name)
            if joint_id < 0:
                raise ValueError(f"Missing MuJoCo joint: {rh_name}")
            sign = float(ROBOT0_TO_RH_JOINT_SIGN.get(robot0_name, 1.0))
            lo, hi = np.asarray(self.model.jnt_range[joint_id], dtype=np.float32)
            if sign < 0:
                lo, hi = -hi, -lo
            lower.append(float(lo))
            upper.append(float(hi))
            self.rh_joint_for_robot0.append(joint_id)
            self.robot0_to_rh_sign.append(sign)

        points, normals, links = self._build_contact_candidates()
        self.metadata = MJCFShadowHandMetadata(
            mjcf_path=Path(mjcf_path).expanduser().resolve(),
            joint_names=list(DEFAULT_JOINT_ORDER),
            joint_lower=torch.tensor(lower, dtype=torch.float32),
            joint_upper=torch.tensor(upper, dtype=torch.float32),
            default_joint_state=DEFAULT_JOINT_STATE.clone(),
            contact_candidate_points=torch.tensor(points, dtype=torch.float32),
            contact_candidate_normals=torch.tensor(normals, dtype=torch.float32),
            contact_candidate_links=links,
        )
        self._candidate_points = self.metadata.contact_candidate_points.to(device=self.device, dtype=self.dtype)
        self._candidate_normals = self.metadata.contact_candidate_normals.to(device=self.device, dtype=self.dtype)
        self._prepare_torch_kinematics()
        self._prepare_collision_meshes()
        self._prepare_penetration_spheres()

    @classmethod
    def create(
        cls,
        mjcf_path: str | Path = LOCAL_MUJOCO_HAND_XML,
        device: str | torch.device = "cpu",
        fingertips_only: bool = False,
        allowed_contact_links: list[str] | tuple[str, ...] | None = None,
        **_: object,
    ) -> "MJCFShadowHandModel":
        model = cls(mjcf_path=mjcf_path, device=device)
        allowed = resolve_contact_link_names(allowed_contact_links)
        if fingertips_only:
            allowed = FINGERTIP_LINK_NAMES if not allowed else [name for name in allowed if name in FINGERTIP_LINK_NAMES]
        if allowed:
            mask = torch.tensor([name in set(allowed) for name in model.metadata.contact_candidate_links], dtype=torch.bool)
            model.metadata.contact_candidate_points = model.metadata.contact_candidate_points[mask]
            model.metadata.contact_candidate_normals = model.metadata.contact_candidate_normals[mask]
            model.metadata.contact_candidate_links = [name for keep, name in zip(mask.tolist(), model.metadata.contact_candidate_links) if keep]
            model._candidate_points = model.metadata.contact_candidate_points.to(device=model.device, dtype=model.dtype)
            model._candidate_normals = model.metadata.contact_candidate_normals.to(device=model.device, dtype=model.dtype)
        return model

    def _build_contact_candidates(self, normal_y_max: float = -0.25) -> tuple[np.ndarray, np.ndarray, list[str]]:
        points, normals, links = [], [], []
        for seed, (link_name, count) in enumerate(CONTACT_COUNTS.items()):
            rh_body = GRASPQP_TO_MUJOCO_BODY[link_name]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, rh_body)
            mesh = _body_local_mesh(self.model, self.data, body_id, groups=(3, 2))
            if mesh is None:
                continue
            sampled = _sample_contact_points(mesh, count, normal_y_max, seed)
            points.append(sampled)
            normals.append(np.zeros_like(sampled, dtype=np.float32))
            links.extend([link_name] * sampled.shape[0])
        return np.concatenate(points, axis=0), np.concatenate(normals, axis=0), links

    def _prepare_torch_kinematics(self) -> None:
        body_pos = torch.tensor(np.asarray(self.model.body_pos), dtype=self.dtype, device=self.device)
        body_quat = np.asarray(self.model.body_quat)
        body_rot = torch.tensor(np.stack([_quat_wxyz_to_matrix(q) for q in body_quat]), dtype=self.dtype, device=self.device)
        body_parent = torch.tensor(np.asarray(self.model.body_parentid), dtype=torch.long, device=self.device)
        joint_axis = torch.zeros((self.model.nbody, 3), dtype=self.dtype, device=self.device)
        body_to_robot0_joint = -torch.ones((self.model.nbody,), dtype=torch.long, device=self.device)
        sign_by_body = torch.ones((self.model.nbody,), dtype=self.dtype, device=self.device)
        for robot0_idx, joint_id in enumerate(self.rh_joint_for_robot0):
            body_id = int(self.model.jnt_bodyid[joint_id])
            body_to_robot0_joint[body_id] = robot0_idx
            axis = np.asarray(self.model.jnt_axis[joint_id], dtype=np.float32) * self.robot0_to_rh_sign[robot0_idx]
            joint_axis[body_id] = torch.tensor(axis, dtype=self.dtype, device=self.device)
        self._body_pos = body_pos
        self._body_rot = body_rot
        self._body_parent = body_parent
        self._joint_axis_by_body = joint_axis
        self._robot0_joint_by_body = body_to_robot0_joint
        self._sign_by_body = sign_by_body

    def _prepare_collision_meshes(self) -> None:
        self._collision_meshes: dict[str, dict[str, torch.Tensor | tm.Trimesh]] = {}
        for body_id in range(1, self.model.nbody):
            link_name = self.link_names[body_id]
            mesh = _body_local_mesh(self.model, self.data, body_id, groups=(3,))
            if mesh is None:
                mesh = _body_local_mesh(self.model, self.data, body_id, groups=(2,))
            if mesh is None or not len(mesh.vertices) or not len(mesh.faces):
                continue
            vertices = torch.tensor(np.asarray(mesh.vertices, dtype=np.float32), dtype=self.dtype, device=self.device)
            faces = torch.tensor(np.asarray(mesh.faces, dtype=np.int64), dtype=torch.long, device=self.device)
            data: dict[str, torch.Tensor | tm.Trimesh] = {"vertices": vertices, "faces": faces}
            if with_torchsdf:
                data["face_verts"] = index_vertices_by_faces(vertices.to(dtype=torch.float32), faces)
            else:
                data["mesh"] = tm.Trimesh(mesh.vertices, mesh.faces, process=False)
            self._collision_meshes[link_name] = data

    def _prepare_penetration_spheres(self) -> None:
        centers, radii, links = [], [], []
        for link_name, mesh_data in self._collision_meshes.items():
            vertices = mesh_data["vertices"]
            assert isinstance(vertices, torch.Tensor)
            bounds_min = vertices.min(dim=0).values
            bounds_max = vertices.max(dim=0).values
            center = 0.5 * (bounds_min + bounds_max)
            radius = (bounds_max - bounds_min).norm().clamp(min=0.004, max=0.012) * 0.35
            centers.append(center)
            radii.append(radius)
            links.append(link_name)
        self._penetration_centers = torch.stack(centers, dim=0) if centers else torch.zeros((0, 3), dtype=self.dtype, device=self.device)
        self._penetration_radii = torch.stack(radii, dim=0) if radii else torch.zeros((0,), dtype=self.dtype, device=self.device)
        self._penetration_links = links

    def default_joint_state(self, batch_size: int = 1) -> torch.Tensor:
        state = self.metadata.default_joint_state.to(device=self.device, dtype=self.dtype)
        return state.unsqueeze(0).repeat(batch_size, 1)

    def clamp_to_limits(self, joint_values: torch.Tensor) -> torch.Tensor:
        lower = self.metadata.joint_lower.to(device=joint_values.device, dtype=joint_values.dtype)
        upper = self.metadata.joint_upper.to(device=joint_values.device, dtype=joint_values.dtype)
        return joint_values.clamp(lower, upper)

    def forward_kinematics(self, joint_values: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = joint_values.shape[0]
        transforms = [torch.eye(4, dtype=joint_values.dtype, device=joint_values.device).unsqueeze(0).repeat(batch_size, 1, 1)]
        for body_id in range(1, self.model.nbody):
            local = torch.eye(4, dtype=joint_values.dtype, device=joint_values.device).unsqueeze(0).repeat(batch_size, 1, 1)
            local[:, :3, :3] = self._body_rot[body_id].to(dtype=joint_values.dtype, device=joint_values.device)
            local[:, :3, 3] = self._body_pos[body_id].to(dtype=joint_values.dtype, device=joint_values.device)
            robot0_joint = int(self._robot0_joint_by_body[body_id].item())
            if robot0_joint >= 0:
                axis = self._joint_axis_by_body[body_id].to(dtype=joint_values.dtype, device=joint_values.device)
                rj = _axis_angle_matrix(axis.unsqueeze(0).expand(batch_size, -1), joint_values[:, robot0_joint])
                joint_tf = torch.eye(4, dtype=joint_values.dtype, device=joint_values.device).unsqueeze(0).repeat(batch_size, 1, 1)
                joint_tf[:, :3, :3] = rj
                local = local @ joint_tf
            parent = int(self._body_parent[body_id].item())
            transforms.append(transforms[parent] @ local)
        return {link: transforms[body_id] for body_id, link in enumerate(self.link_names)}

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
            world_points = []
            for candidate_id, link_name in enumerate(links):
                transform = transforms[link_name]
                point = local_points[:, candidate_id]
                hom = torch.cat([point, torch.ones(batch_size, 1, device=self.device, dtype=self.dtype)], dim=-1)
                world_points.append((transform @ hom.unsqueeze(-1)).squeeze(-1)[..., :3])
            return self.apply_wrist_pose(torch.stack(world_points, dim=1), wrist_translation=wrist_translation, wrist_rotation=wrist_rotation)

        local_points = self._candidate_points[indices]
        flat_links = [self.metadata.contact_candidate_links[idx] for idx in indices.reshape(-1).tolist()]
        links_by_batch = [flat_links[row * indices.shape[1] : (row + 1) * indices.shape[1]] for row in range(batch_size)]
        world_points = []
        for batch_id in range(batch_size):
            batch_points = []
            for column_id, link_name in enumerate(links_by_batch[batch_id]):
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
        centers = []
        batch_size = joint_values.shape[0]
        for center, link_name in zip(self._penetration_centers, self._penetration_links):
            hom = torch.cat([center.to(dtype=joint_values.dtype, device=joint_values.device), torch.ones(1, dtype=joint_values.dtype, device=joint_values.device)])
            centers.append((transforms[link_name] @ hom.view(1, 4, 1).expand(batch_size, -1, -1)).squeeze(-1)[..., :3])
        if centers:
            centers_cat = torch.stack(centers, dim=1)
            radii = self._penetration_radii.to(dtype=joint_values.dtype, device=joint_values.device).unsqueeze(0).expand(batch_size, -1)
        else:
            centers_cat = torch.zeros((batch_size, 0, 3), dtype=joint_values.dtype, device=joint_values.device)
            radii = torch.zeros((batch_size, 0), dtype=joint_values.dtype, device=joint_values.device)
        return self.apply_wrist_pose(centers_cat, wrist_translation=wrist_translation, wrist_rotation=wrist_rotation), radii, self._penetration_links

    def cal_distance(
        self,
        points_world: torch.Tensor,
        joint_values: torch.Tensor,
        wrist_translation: torch.Tensor | None = None,
        wrist_rotation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        points = points_world
        if wrist_rotation is None:
            wrist_rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(points.shape[0], -1, -1)
        if wrist_translation is None:
            wrist_translation = torch.zeros((points.shape[0], 3), device=points.device, dtype=points.dtype)
        points = (points - wrist_translation.unsqueeze(1)) @ wrist_rotation
        transforms = self.forward_kinematics(joint_values)
        distances = []
        for link_name, mesh_data in self._collision_meshes.items():
            if link_name not in transforms:
                continue
            transform = transforms[link_name]
            points_local = (points - transform[:, :3, 3].unsqueeze(1)) @ transform[:, :3, :3]
            flat_points = points_local.reshape(-1, 3)
            if "face_verts" in mesh_data:
                face_verts = mesh_data["face_verts"]
                assert isinstance(face_verts, torch.Tensor)
                face_verts = face_verts.to(dtype=flat_points.dtype, device=flat_points.device)
                squared_distance, signs, _, _ = compute_sdf(flat_points, face_verts)
                link_distance = torch.sqrt(squared_distance + 1e-8) * (-signs.to(dtype=flat_points.dtype))
                distances.append(link_distance.reshape(points.shape[0], points.shape[1]))
            elif "mesh" in mesh_data:
                mesh = mesh_data["mesh"]
                assert isinstance(mesh, tm.Trimesh)
                query = flat_points.detach().cpu().numpy()
                link_distance = tm.proximity.signed_distance(mesh, query)
                distances.append(torch.tensor(link_distance, dtype=points.dtype, device=points.device).reshape(points.shape[0], points.shape[1]))
        return torch.stack(distances, dim=0).max(dim=0).values
