from __future__ import annotations

from dataclasses import dataclass

import torch


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


@dataclass
class Sphere:
    radius: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.center, dtype=points.dtype, device=points.device)
        return (points - center).norm(dim=-1) - self.radius

    def normals(self, points: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.center, dtype=points.dtype, device=points.device)
        return _normalize(points - center)


@dataclass
class Cylinder:
    radius: float
    half_height: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.center, dtype=points.dtype, device=points.device)
        p = points - center
        radial = p[..., :2].norm(dim=-1)
        d = torch.stack([radial - self.radius, p[..., 2].abs() - self.half_height], dim=-1)
        outside = d.clamp_min(0.0).norm(dim=-1)
        inside = d.max(dim=-1).values.clamp_max(0.0)
        return outside + inside

    def normals(self, points: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.center, dtype=points.dtype, device=points.device)
        p = points - center
        radial = p[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        side_normal = torch.cat([p[..., :2] / radial, torch.zeros_like(p[..., :1])], dim=-1)
        cap_normal = torch.zeros_like(p)
        cap_normal[..., 2] = torch.sign(p[..., 2])
        choose_cap = (p[..., 2].abs() - self.half_height) >= (radial.squeeze(-1) - self.radius)
        return torch.where(choose_cap.unsqueeze(-1), cap_normal, side_normal)


@dataclass
class Box:
    half_extents: tuple[float, float, float]
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.center, dtype=points.dtype, device=points.device)
        extents = torch.tensor(self.half_extents, dtype=points.dtype, device=points.device)
        q = (points - center).abs() - extents
        outside = q.clamp_min(0.0).norm(dim=-1)
        inside = q.max(dim=-1).values.clamp_max(0.0)
        return outside + inside

    def normals(self, points: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.center, dtype=points.dtype, device=points.device)
        extents = torch.tensor(self.half_extents, dtype=points.dtype, device=points.device)
        p = points - center
        abs_p = p.abs()
        outside_mask = (abs_p > extents).any(dim=-1)
        closest_face = torch.argmax(abs_p / extents.clamp_min(1e-8), dim=-1)
        normals = torch.zeros_like(points)
        normals[outside_mask] = _normalize(torch.sign(p[outside_mask]) * (abs_p[outside_mask] - extents).clamp_min(0.0))
        inside_idx = ~outside_mask
        if inside_idx.any():
            inside_face = closest_face[inside_idx]
            normals_inside = torch.zeros((inside_face.shape[0], 3), dtype=points.dtype, device=points.device)
            signs = torch.sign(p[inside_idx])
            normals_inside[torch.arange(inside_face.shape[0]), inside_face] = signs[
                torch.arange(inside_face.shape[0]), inside_face
            ]
            normals[inside_idx] = normals_inside
        return normals
