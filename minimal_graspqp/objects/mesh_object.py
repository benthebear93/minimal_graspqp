from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import importlib.util
import numpy as np
import torch
import trimesh as tm

with_torchsdf = importlib.util.find_spec("torchsdf") is not None
if with_torchsdf:
    from torchsdf import compute_sdf, index_vertices_by_faces


def _farthest_point_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[0] <= count:
        return torch.arange(points.shape[0], device=points.device, dtype=torch.long)
    selected = torch.empty(count, device=points.device, dtype=torch.long)
    selected[0] = 0
    distances = torch.cdist(points[selected[0] : selected[0] + 1], points).squeeze(0)
    for idx in range(1, count):
        next_index = torch.argmax(distances)
        selected[idx] = next_index
        next_distances = torch.cdist(points[next_index : next_index + 1], points).squeeze(0)
        distances = torch.minimum(distances, next_distances)
    return selected


def _sample_surface_with_fps(
    mesh: tm.Trimesh,
    count: int,
    *,
    oversample_factor: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    dense_count = max(count, oversample_factor * count)
    sampled_points, face_indices = tm.sample.sample_surface(mesh, dense_count)
    sampled_points_t = torch.tensor(sampled_points, dtype=torch.float32)
    selected = _farthest_point_indices(sampled_points_t, count).cpu().numpy()
    return sampled_points[selected], face_indices[selected]


@dataclass
class MeshObject:
    mesh_path: Path
    scale: float = 1.0
    penetration_num_samples: int = 256

    def __post_init__(self):
        self.mesh_path = Path(self.mesh_path).expanduser().resolve()
        self._mesh = tm.load(self.mesh_path, force="mesh", process=True)
        self._mesh.vertices = self._mesh.vertices * self.scale
        self._convex_hull = self._mesh.convex_hull
        self.center = tuple(self._mesh.centroid.tolist())
        self._has_rtree = importlib.util.find_spec("rtree") is not None
        mesh_vertices = torch.tensor(np.asarray(self._mesh.vertices, dtype=np.float32), dtype=torch.float32)
        mesh_faces = torch.tensor(np.asarray(self._mesh.faces, dtype=np.int64), dtype=torch.long)
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._mesh_face_verts = index_vertices_by_faces(mesh_vertices, mesh_faces) if with_torchsdf else None
        hull_triangles = np.asarray(self._convex_hull.triangles, dtype=np.float32)
        self._hull_face_points = torch.tensor(hull_triangles[:, 0, :], dtype=torch.float32)
        self._hull_face_normals = torch.tensor(np.asarray(self._convex_hull.face_normals, dtype=np.float32), dtype=torch.float32)
        hull_vertices = torch.tensor(np.asarray(self._convex_hull.vertices, dtype=np.float32), dtype=torch.float32)
        hull_faces = torch.tensor(np.asarray(self._convex_hull.faces, dtype=np.int64), dtype=torch.long)
        self._hull_vertices = hull_vertices
        self._hull_faces = hull_faces
        self._hull_face_verts = index_vertices_by_faces(hull_vertices, hull_faces) if with_torchsdf else None
        penetration_points, _ = self.sample_surface(
            self.penetration_num_samples,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self._penetration_surface_points = penetration_points.detach().cpu()

    @property
    def mesh(self) -> tm.Trimesh:
        return self._mesh

    @property
    def convex_hull(self) -> tm.Trimesh:
        return self._convex_hull

    @property
    def penetration_surface_points(self) -> torch.Tensor:
        return self._penetration_surface_points

    @classmethod
    def from_code(cls, data_root: str | Path, object_code: str, scale: float = 1.0) -> "MeshObject":
        root = Path(data_root).expanduser().resolve()
        candidates = [
            root / object_code / "coacd" / "remeshed.obj",
            root / object_code / "coacd" / "decomposed.obj",
        ]
        for path in candidates:
            if path.exists():
                return cls(path, scale=scale)
        raise FileNotFoundError(f"Could not resolve mesh for object code '{object_code}' under {root}")

    def sample_surface(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        points, face_indices = _sample_surface_with_fps(self.mesh, batch_size)
        normals = self.mesh.face_normals[face_indices]
        return (
            torch.tensor(points, dtype=dtype, device=device),
            torch.tensor(normals, dtype=dtype, device=device),
        )

    def sample_init_surface(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        inflation: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hull = self.convex_hull
        if self._has_rtree:
            sampled, face_indices = _sample_surface_with_fps(hull, 100 * batch_size)
            inflated = sampled + hull.face_normals[face_indices] * inflation
            selected = _farthest_point_indices(torch.tensor(inflated, dtype=torch.float32), batch_size).cpu().numpy()
            support_points = inflated[selected]
            closest_points, _, triangle_ids = hull.nearest.on_surface(support_points)
            del closest_points
            normals = hull.face_normals[triangle_ids]
            points = support_points
        else:
            points, face_indices = _sample_surface_with_fps(hull, batch_size)
            normals = hull.face_normals[face_indices]
        return (
            torch.tensor(points, dtype=dtype, device=device),
            torch.tensor(normals, dtype=dtype, device=device),
        )

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        if self._mesh_face_verts is not None:
            flat_points = points.reshape(-1, 3)
            face_verts = self._mesh_face_verts.to(device=points.device, dtype=points.dtype)
            squared_distance, signs, normal, _ = compute_sdf(flat_points, face_verts)
            del normal
            signed_distance = torch.sqrt(squared_distance + 1e-8) * (-signs.to(dtype=points.dtype))
            return signed_distance.reshape(points.shape[:-1])
        if self._has_rtree:
            query = points.detach().cpu().reshape(-1, 3).numpy()
            sdf = -tm.proximity.signed_distance(self.convex_hull, query)
            return torch.tensor(sdf, dtype=points.dtype, device=points.device).reshape(points.shape[:-1])
        face_points = self._hull_face_points.to(device=points.device, dtype=points.dtype)
        face_normals = self._hull_face_normals.to(device=points.device, dtype=points.dtype)
        flat_points = points.reshape(-1, 3)
        plane_distances = ((flat_points.unsqueeze(1) - face_points.unsqueeze(0)) * face_normals.unsqueeze(0)).sum(dim=-1)
        sdf = plane_distances.max(dim=1).values
        return sdf.reshape(points.shape[:-1])

    def normals(self, points: torch.Tensor) -> torch.Tensor:
        if self._mesh_face_verts is not None:
            flat_points = points.reshape(-1, 3)
            face_verts = self._mesh_face_verts.to(device=points.device, dtype=points.dtype)
            _, signs, normal, _ = compute_sdf(flat_points, face_verts)
            outward = normal * signs.to(dtype=points.dtype).unsqueeze(1)
            return outward.reshape(*points.shape)
        if self._has_rtree:
            query = points.detach().cpu().reshape(-1, 3).numpy()
            _, _, triangle_ids = self.mesh.nearest.on_surface(query)
            face_normals = self.mesh.face_normals[triangle_ids]
            return torch.tensor(face_normals, dtype=points.dtype, device=points.device).reshape(*points.shape)
        face_points = self._hull_face_points.to(device=points.device, dtype=points.dtype)
        face_normals = self._hull_face_normals.to(device=points.device, dtype=points.dtype)
        flat_points = points.reshape(-1, 3)
        plane_distances = ((flat_points.unsqueeze(1) - face_points.unsqueeze(0)) * face_normals.unsqueeze(0)).sum(dim=-1)
        face_ids = plane_distances.argmax(dim=1)
        normals = face_normals[face_ids]
        return normals.reshape(*points.shape)
