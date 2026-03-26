from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pytorch_kinematics as pk
import torch
import trimesh as tm

from minimal_graspqp.assets import resolve_shadow_hand_asset_dir


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
    contact_candidate_links: list[str]
    penetration_points: dict[str, torch.Tensor]

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


def _load_contact_candidates(asset_dir: Path) -> tuple[torch.Tensor, list[str]]:
    contact_points_data = json.loads((asset_dir / "contact_points.json").read_text())
    candidates = []
    links = []
    mesh_root = asset_dir / "meshes"
    mesh_cache: dict[Path, tm.Trimesh] = {}
    for link_name, entries in contact_points_data.items():
        for relative_path, vertex_index in entries:
            mesh_path = (mesh_root / relative_path).resolve()
            if mesh_path not in mesh_cache:
                mesh_cache[mesh_path] = tm.load(mesh_path, force="mesh", process=False)
            vertex = mesh_cache[mesh_path].vertices[vertex_index]
            candidates.append(vertex.tolist())
            links.append(link_name)
    return torch.tensor(candidates, dtype=torch.float32), links


def _load_penetration_points(asset_dir: Path) -> dict[str, torch.Tensor]:
    raw = json.loads((asset_dir / "penetration_points.json").read_text())
    return {name: torch.tensor(points, dtype=torch.float32) for name, points in raw.items()}


def load_shadow_hand_metadata(asset_dir: str | Path | None = None) -> ShadowHandMetadata:
    resolved_dir = resolve_shadow_hand_asset_dir(asset_dir)
    urdf_path = resolved_dir / "shadow_hand.urdf"
    chain = pk.build_chain_from_urdf(urdf_path.read_text())
    joint_names = chain.get_joint_parameter_names()
    if joint_names != DEFAULT_JOINT_ORDER:
        raise ValueError("Unexpected Shadow Hand joint order in URDF.")
    joint_lower, joint_upper = _parse_joint_limits(urdf_path, joint_names)
    contact_candidate_points, contact_candidate_links = _load_contact_candidates(resolved_dir)
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
        contact_candidate_links=contact_candidate_links,
        penetration_points=_load_penetration_points(resolved_dir),
    )


class ShadowHandModel:
    """Minimal Shadow Hand wrapper for FK and contact candidate transforms."""

    def __init__(self, metadata: ShadowHandMetadata, device: str | torch.device = "cpu", dtype=torch.float32):
        self.metadata = metadata
        self.device = torch.device(device)
        self.dtype = dtype
        self.chain = pk.build_chain_from_urdf(metadata.urdf_path.read_text()).to(dtype=dtype, device=self.device)
        self._candidate_points = metadata.contact_candidate_points.to(device=self.device, dtype=dtype)

    @classmethod
    def create(cls, asset_dir: str | Path | None = None, device: str | torch.device = "cpu") -> "ShadowHandModel":
        return cls(load_shadow_hand_metadata(asset_dir=asset_dir), device=device)

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

    def contact_candidates_world(self, joint_values: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
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
            return torch.stack(world_points, dim=1)

        world_points = []
        for batch_id in range(batch_size):
            batch_points = []
            for column_id, link_name in enumerate(links[batch_id]):
                transform = transforms[link_name][batch_id]
                point = local_points[batch_id, column_id]
                hom = torch.cat([point, torch.ones(1, device=self.device, dtype=self.dtype)], dim=0)
                batch_points.append((transform @ hom.unsqueeze(-1)).squeeze(-1)[..., :3])
            world_points.append(torch.stack(batch_points, dim=0))
        return torch.stack(world_points, dim=0)
