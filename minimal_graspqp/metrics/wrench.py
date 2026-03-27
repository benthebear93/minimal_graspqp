from __future__ import annotations

import math

import torch


def _normalize(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)


def _orthonormal_basis(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref_x = torch.tensor([1.0, 0.0, 0.0], dtype=normals.dtype, device=normals.device)
    ref_y = torch.tensor([0.0, 1.0, 0.0], dtype=normals.dtype, device=normals.device)
    use_y = normals[..., 0].abs() > 0.9
    reference = torch.where(use_y.unsqueeze(-1), ref_y, ref_x)
    tangent_1 = _normalize(torch.linalg.cross(normals, reference))
    tangent_2 = _normalize(torch.linalg.cross(normals, tangent_1))
    return tangent_1, tangent_2


def friction_cone_edges(normals: torch.Tensor, friction: float = 0.2, num_edges: int = 4) -> torch.Tensor:
    normals = _normalize(normals)
    tangent_1, tangent_2 = _orthonormal_basis(normals)
    if num_edges == 4:
        cone_edges = [
            friction * tangent_1 + math.sqrt(1.0 - friction**2) * normals,
            friction * tangent_2 + math.sqrt(1.0 - friction**2) * normals,
            -friction * tangent_1 + math.sqrt(1.0 - friction**2) * normals,
            -friction * tangent_2 + math.sqrt(1.0 - friction**2) * normals,
        ]
    else:
        edge_angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            steps=num_edges + 1,
            device=normals.device,
            dtype=normals.dtype,
        )[:-1]
        cone_edges = []
        for angle in edge_angles:
            basis = math.cos(float(angle)) * tangent_1 + math.sin(float(angle)) * tangent_2
            cone_edges.append(friction * basis + math.sqrt(1.0 - friction**2) * normals)
    return torch.stack(cone_edges, dim=-2).flatten(-3, -2) / len(cone_edges)


def build_wrench_matrix(
    contact_points: torch.Tensor,
    force_directions: torch.Tensor,
    cog: torch.Tensor,
    torque_weight: float = 1.0,
) -> torch.Tensor:
    friction_cone_size = force_directions.shape[-2] // contact_points.shape[-2]
    repeated_points = contact_points.repeat_interleave(friction_cone_size, dim=-2)
    lever_arms = repeated_points - cog.unsqueeze(-2)
    torques = torch.linalg.cross(lever_arms, force_directions) * torque_weight
    return torch.cat([force_directions, torques], dim=-1).transpose(-1, -2)
