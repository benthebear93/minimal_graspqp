from __future__ import annotations

import math

import torch

from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.rotation import look_at_rotation, project_rotation_matrices
from minimal_graspqp.state import GraspState


def _random_rotation_matrices(
    batch_size: int, max_angle: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    axes = torch.randn(batch_size, 3, dtype=dtype, device=device)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angles = (
        2.0 * torch.rand(batch_size, dtype=dtype, device=device) - 1.0
    ) * max_angle
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


def _random_contact_indices(
    num_candidates: int, batch_size: int, num_contacts: int, device: torch.device
) -> torch.Tensor:
    return torch.randint(num_candidates, size=(batch_size, num_contacts), device=device)


def _object_facing_rotation(
    forward: torch.Tensor, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    up_hint = (
        torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        .unsqueeze(0)
        .expand_as(forward)
        .clone()
    )
    fallback_up = (
        torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        .unsqueeze(0)
        .expand_as(forward)
    )
    aligned = (forward * up_hint).sum(dim=-1, keepdim=True).abs() > 0.95
    up_hint = torch.where(aligned, fallback_up, up_hint)
    right = torch.linalg.cross(up_hint, forward)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    true_up = torch.linalg.cross(forward, right)
    return torch.stack([right, true_up, forward], dim=-1)


def initialize_grasps_for_primitive(
    hand_model,
    primitive: Sphere | Cylinder | Box | MeshObject,
    batch_size: int,
    distance_lower: float = 0.08,
    distance_upper: float = 0.12,
    joint_jitter_strength: float = 0.05,
    max_wrist_angle: float = math.pi / 6.0,
    num_contacts: int = 4,
    base_wrist_rotation: torch.Tensor | None = None,
) -> GraspState:
    # only use convex hull mesh
    dtype = hand_model.dtype
    device = hand_model.device

    if hasattr(primitive, "sample_init_surface"):
        surface_points, surface_normals = primitive.sample_init_surface(
            batch_size=batch_size, dtype=dtype, device=device
        )
    else:
        surface_points, surface_normals = primitive.sample_surface(
            batch_size=batch_size, dtype=dtype, device=device
        )
    radii = distance_lower + (distance_upper - distance_lower) * torch.rand(
        batch_size, 1, dtype=dtype, device=device
    )
    wrist_translation = surface_points + surface_normals * radii
    desired_forward = -surface_normals
    wrist_rotation = _object_facing_rotation(
        desired_forward, dtype=dtype, device=device
    )
    wrist_rotation = wrist_rotation @ _random_rotation_matrices(
        batch_size, max_angle=max_wrist_angle, dtype=dtype, device=device
    )
    if base_wrist_rotation is not None:
        base_wrist_rotation = base_wrist_rotation.to(device=device, dtype=dtype)
        wrist_rotation = wrist_rotation @ base_wrist_rotation.unsqueeze(0)
    wrist_rotation = project_rotation_matrices(wrist_rotation)
    current_forward = wrist_rotation[:, :, 2]
    flip_mask = (current_forward * desired_forward).sum(dim=-1) < 0
    if flip_mask.any():
        fix = torch.diag(
            torch.tensor([-1.0, 1.0, -1.0], dtype=dtype, device=device)
        ).unsqueeze(0)
        wrist_rotation[flip_mask] = wrist_rotation[flip_mask] @ fix

    default_joint = hand_model.default_joint_state(batch_size=batch_size)
    lower = hand_model.metadata.joint_lower.to(device=device, dtype=dtype)
    upper = hand_model.metadata.joint_upper.to(device=device, dtype=dtype)
    joint_span = (upper - lower).unsqueeze(0)
    jitter = (
        (2.0 * torch.rand_like(default_joint) - 1.0)
        * joint_jitter_strength
        * joint_span
    )
    joint_values = hand_model.clamp_to_limits(default_joint + jitter)

    contact_indices = _random_contact_indices(
        hand_model.metadata.num_contact_candidates,
        batch_size=batch_size,
        num_contacts=num_contacts,
        device=device,
    )
    return GraspState(
        joint_values=joint_values,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
        contact_indices=contact_indices,
    )
