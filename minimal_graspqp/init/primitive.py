from __future__ import annotations

import math

import torch

from minimal_graspqp.objects import Box, Cylinder, Sphere
from minimal_graspqp.rotation import look_at_rotation, project_rotation_matrices
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


def _select_diverse_contact_indices(
    candidate_distance: torch.Tensor,
    candidate_links: list[str],
    num_contacts: int,
) -> torch.Tensor:
    batch_size, num_candidates = candidate_distance.shape
    device = candidate_distance.device
    link_to_indices: dict[str, list[int]] = {}
    for idx, link_name in enumerate(candidate_links):
        link_to_indices.setdefault(link_name, []).append(idx)

    per_link_order = sorted(link_to_indices.keys())
    selected_rows = []
    for batch_idx in range(batch_size):
        selected: list[int] = []
        used = set()

        # First pass: take the nearest candidate from as many distinct links as possible.
        ranked_links = []
        for link_name in per_link_order:
            indices = torch.tensor(link_to_indices[link_name], device=device, dtype=torch.long)
            distances = candidate_distance[batch_idx, indices]
            best_local = indices[torch.argmin(distances)].item()
            best_distance = float(candidate_distance[batch_idx, best_local].item())
            ranked_links.append((best_distance, best_local))
        ranked_links.sort(key=lambda item: item[0])

        for _, candidate_idx in ranked_links:
            selected.append(candidate_idx)
            used.add(candidate_idx)
            if len(selected) == num_contacts:
                break

        # Second pass: fill remaining slots globally by nearest distance.
        if len(selected) < num_contacts:
            global_rank = torch.argsort(candidate_distance[batch_idx])
            for candidate_idx in global_rank.tolist():
                if candidate_idx in used:
                    continue
                selected.append(candidate_idx)
                used.add(candidate_idx)
                if len(selected) == num_contacts:
                    break

        selected_rows.append(torch.tensor(selected, device=device, dtype=torch.long))

    return torch.stack(selected_rows, dim=0)


def initialize_grasps_for_primitive(
    hand_model,
    primitive: Sphere | Cylinder | Box,
    batch_size: int,
    distance_lower: float = 0.08,
    distance_upper: float = 0.12,
    joint_jitter_strength: float = 0.05,
    max_wrist_angle: float = math.pi / 6.0,
    num_contacts: int = 4,
    base_wrist_rotation: torch.Tensor | None = None,
) -> GraspState:
    dtype = hand_model.dtype
    device = hand_model.device

    center = _primitive_center(primitive, dtype=dtype, device=device)
    direction = torch.randn(batch_size, 3, dtype=dtype, device=device)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radii = distance_lower + (distance_upper - distance_lower) * torch.rand(batch_size, 1, dtype=dtype, device=device)
    wrist_translation = center.unsqueeze(0) + direction * radii
    forward_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    up_axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    wrist_rotation = look_at_rotation(
        wrist_translation,
        wrist_translation + direction,
        forward_axis=forward_axis,
        up_axis=up_axis,
    )
    wrist_rotation = wrist_rotation @ _random_rotation_matrices(batch_size, max_angle=max_wrist_angle, dtype=dtype, device=device)
    if base_wrist_rotation is not None:
        base_wrist_rotation = base_wrist_rotation.to(device=device, dtype=dtype)
        wrist_rotation = wrist_rotation @ base_wrist_rotation.unsqueeze(0)
    wrist_rotation = project_rotation_matrices(wrist_rotation)
    desired_forward = -direction
    current_forward = wrist_rotation[:, :, 2]
    flip_mask = (current_forward * desired_forward).sum(dim=-1) < 0
    if flip_mask.any():
        fix = torch.diag(torch.tensor([-1.0, 1.0, -1.0], dtype=dtype, device=device)).unsqueeze(0)
        wrist_rotation[flip_mask] = wrist_rotation[flip_mask] @ fix

    default_joint = hand_model.default_joint_state(batch_size=batch_size)
    lower = hand_model.metadata.joint_lower.to(device=device, dtype=dtype)
    upper = hand_model.metadata.joint_upper.to(device=device, dtype=dtype)
    joint_span = (upper - lower).unsqueeze(0)
    jitter = (2.0 * torch.rand_like(default_joint) - 1.0) * joint_jitter_strength * joint_span
    joint_values = hand_model.clamp_to_limits(default_joint + jitter)

    all_candidates = hand_model.contact_candidates_world(
        joint_values,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
    )
    candidate_distance = primitive.signed_distance(all_candidates).abs()
    contact_indices = _select_diverse_contact_indices(
        candidate_distance,
        hand_model.metadata.contact_candidate_links,
        num_contacts=num_contacts,
    )
    return GraspState(
        joint_values=joint_values,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
        contact_indices=contact_indices,
    )
