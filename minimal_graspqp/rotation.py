from __future__ import annotations

import math

import torch


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=dtype,
        device=device,
    )
    ry = torch.tensor(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=dtype,
        device=device,
    )
    rz = torch.tensor(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    return rz @ ry @ rx


def palm_down_rotation(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return rotation_matrix_from_rpy(math.pi, 0.0, 0.0, dtype=dtype, device=device)


def project_rotation_matrices(rotations: torch.Tensor) -> torch.Tensor:
    u, _, v_t = torch.linalg.svd(rotations)
    projected = u @ v_t
    det = torch.linalg.det(projected)
    needs_flip = det < 0
    if needs_flip.any():
        u = u.clone()
        u[needs_flip, :, -1] *= -1.0
        projected = u @ v_t
    return projected


def look_at_rotation(
    camera_positions: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    forward_axis: torch.Tensor,
    up_axis: torch.Tensor,
) -> torch.Tensor:
    base_up_axis = up_axis.to(device=camera_positions.device, dtype=camera_positions.dtype)
    base_forward_axis = forward_axis.to(device=camera_positions.device, dtype=camera_positions.dtype)
    up = base_up_axis.unsqueeze(0).expand(camera_positions.shape[0], -1).clone()

    forward = camera_positions - target_positions
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    alignment = torch.sum(up * forward, dim=-1, keepdim=True).abs()
    fallback_up = torch.tensor([0.0, 1.0, 0.0], device=camera_positions.device, dtype=camera_positions.dtype)
    up = torch.where(alignment < 0.95, up, fallback_up.unsqueeze(0).expand_as(up))

    right = torch.linalg.cross(up, forward)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    true_up = torch.linalg.cross(forward, right)

    orientation = torch.stack([forward, true_up, right], dim=-1)
    basis = torch.stack(
        [
            base_forward_axis,
            -torch.linalg.cross(base_forward_axis, base_up_axis),
            base_up_axis,
        ],
        dim=-1,
    )
    return orientation @ basis
