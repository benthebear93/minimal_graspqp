from __future__ import annotations

from dataclasses import dataclass

import torch


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def _random_unit_vectors(batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    vectors = torch.randn(batch_size, 3, dtype=dtype, device=device)
    return _normalize(vectors)


@dataclass
class Sphere:
    radius: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def sample_surface(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        center = torch.tensor(self.center, dtype=dtype, device=device)
        normals = _random_unit_vectors(batch_size, dtype=dtype, device=device)
        points = center.unsqueeze(0) + self.radius * normals
        return points, normals

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

    def sample_surface(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        center = torch.tensor(self.center, dtype=dtype, device=device)
        angles = 2.0 * torch.pi * torch.rand(batch_size, dtype=dtype, device=device)
        side_probs = torch.full((batch_size,), self.radius / (self.radius + self.half_height), dtype=dtype, device=device)
        choose_side = torch.rand(batch_size, dtype=dtype, device=device) < side_probs

        points = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        normals = torch.zeros(batch_size, 3, dtype=dtype, device=device)

        if choose_side.any():
            side_idx = choose_side.nonzero(as_tuple=False).squeeze(-1)
            side_angles = angles[side_idx]
            side_xy = torch.stack([torch.cos(side_angles), torch.sin(side_angles)], dim=-1)
            points[side_idx, :2] = self.radius * side_xy
            points[side_idx, 2] = (2.0 * torch.rand(side_idx.shape[0], dtype=dtype, device=device) - 1.0) * self.half_height
            normals[side_idx, :2] = side_xy

        cap_idx = (~choose_side).nonzero(as_tuple=False).squeeze(-1)
        if cap_idx.numel() > 0:
            cap_angles = angles[cap_idx]
            cap_radius = self.radius * torch.sqrt(torch.rand(cap_idx.shape[0], dtype=dtype, device=device))
            points[cap_idx, 0] = cap_radius * torch.cos(cap_angles)
            points[cap_idx, 1] = cap_radius * torch.sin(cap_angles)
            cap_sign = torch.where(
                torch.rand(cap_idx.shape[0], dtype=dtype, device=device) < 0.5,
                torch.full((cap_idx.shape[0],), -1.0, dtype=dtype, device=device),
                torch.ones(cap_idx.shape[0], dtype=dtype, device=device),
            )
            points[cap_idx, 2] = cap_sign * self.half_height
            normals[cap_idx, 2] = cap_sign

        return points + center.unsqueeze(0), normals

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

    def sample_surface(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        center = torch.tensor(self.center, dtype=dtype, device=device)
        extents = torch.tensor(self.half_extents, dtype=dtype, device=device)
        face_areas = torch.tensor(
            [
                extents[1] * extents[2],
                extents[1] * extents[2],
                extents[0] * extents[2],
                extents[0] * extents[2],
                extents[0] * extents[1],
                extents[0] * extents[1],
            ],
            dtype=dtype,
            device=device,
        )
        face_probs = face_areas / face_areas.sum()
        face_ids = torch.multinomial(face_probs, batch_size, replacement=True)

        points = torch.empty(batch_size, 3, dtype=dtype, device=device)
        points[:, 0] = (2.0 * torch.rand(batch_size, dtype=dtype, device=device) - 1.0) * extents[0]
        points[:, 1] = (2.0 * torch.rand(batch_size, dtype=dtype, device=device) - 1.0) * extents[1]
        points[:, 2] = (2.0 * torch.rand(batch_size, dtype=dtype, device=device) - 1.0) * extents[2]
        normals = torch.zeros(batch_size, 3, dtype=dtype, device=device)

        axis = torch.div(face_ids, 2, rounding_mode="floor")
        sign = torch.where(face_ids % 2 == 0, -torch.ones_like(face_ids, dtype=dtype), torch.ones_like(face_ids, dtype=dtype))
        for axis_id in range(3):
            mask = axis == axis_id
            if mask.any():
                points[mask, axis_id] = sign[mask] * extents[axis_id]
                normals[mask, axis_id] = sign[mask]
        return points + center.unsqueeze(0), normals

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
