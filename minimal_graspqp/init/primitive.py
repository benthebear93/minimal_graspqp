from __future__ import annotations

import math

import torch

from minimal_graspqp.objects import Box, Cylinder, Sphere
from minimal_graspqp.state import GraspState


def _primitive_center(primitive: Sphere | Cylinder | Box, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(primitive.center, dtype=dtype, device=device)


def _random_rotation_matrices(batch_size: int, max_angle: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    axes = torch.randn(batch_size, 3, dtype=dtype, device=device)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angles = (2.0 * torch.rand(batch_size, dtype=dtype, device=device) - 1.0) * max_angle
    K = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]
    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    angles = angles.view(-1, 1, 1)
    return eye + torch.sin(angles) * K + (1.0 - torch.cos(angles)) * (K @ K)


def initialize_grasps_for_primitive(
    hand_model,
    primitive: Sphere | Cylinder | Box,
    batch_size: int,
    distance_lower: float = 0.08,
    distance_upper: float = 0.12,
    joint_jitter_strength: float = 0.05,
    max_wrist_angle: float = math.pi / 6.0,
    num_contacts: int = 4,
) -> GraspState:
    dtype = hand_model.dtype
    device = hand_model.device

    center = _primitive_center(primitive, dtype=dtype, device=device)
    direction = torch.randn(batch_size, 3, dtype=dtype, device=device)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radii = distance_lower + (distance_upper - distance_lower) * torch.rand(batch_size, 1, dtype=dtype, device=device)
    wrist_translation = center.unsqueeze(0) + direction * radii
    wrist_rotation = _random_rotation_matrices(batch_size, max_angle=max_wrist_angle, dtype=dtype, device=device)

    default_joint = hand_model.default_joint_state(batch_size=batch_size)
    lower = hand_model.metadata.joint_lower.to(device=device, dtype=dtype)
    upper = hand_model.metadata.joint_upper.to(device=device, dtype=dtype)
    joint_span = (upper - lower).unsqueeze(0)
    jitter = (2.0 * torch.rand_like(default_joint) - 1.0) * joint_jitter_strength * joint_span
    joint_values = hand_model.clamp_to_limits(default_joint + jitter)

    num_candidates = hand_model.metadata.num_contact_candidates
    contact_indices = torch.randint(0, num_candidates, (batch_size, num_contacts), device=device)
    return GraspState(
        joint_values=joint_values,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
        contact_indices=contact_indices,
    )
