from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import importlib.util
import numpy as np
import torch
import trimesh as tm


@dataclass
class MeshObject:
    mesh_path: Path
    scale: float = 1.0

    def __post_init__(self):
        self.mesh_path = Path(self.mesh_path).expanduser().resolve()
        self._mesh = tm.load(self.mesh_path, force="mesh", process=True)
        self._mesh.vertices = self._mesh.vertices * self.scale
        self._convex_hull = self._mesh.convex_hull
        self.center = tuple(self._mesh.centroid.tolist())
        self._has_rtree = importlib.util.find_spec("rtree") is not None
        hull_triangles = np.asarray(self._convex_hull.triangles, dtype=np.float32)
        self._hull_face_points = torch.tensor(hull_triangles[:, 0, :], dtype=torch.float32)
        self._hull_face_normals = torch.tensor(np.asarray(self._convex_hull.face_normals, dtype=np.float32), dtype=torch.float32)

    @property
    def mesh(self) -> tm.Trimesh:
        return self._mesh

    @property
    def convex_hull(self) -> tm.Trimesh:
        return self._convex_hull

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
        points, face_indices = tm.sample.sample_surface_even(self.mesh, batch_size)
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
            sampled, face_indices = tm.sample.sample_surface_even(hull, 100 * batch_size)
            inflated = sampled + hull.face_normals[face_indices] * inflation
            choice = np.random.choice(inflated.shape[0], size=batch_size, replace=inflated.shape[0] < batch_size)
            support_points = inflated[choice]
            closest_points, _, triangle_ids = hull.nearest.on_surface(support_points)
            del closest_points
            normals = hull.face_normals[triangle_ids]
            points = support_points
        else:
            points, face_indices = tm.sample.sample_surface_even(hull, batch_size)
            normals = hull.face_normals[face_indices]
        return (
            torch.tensor(points, dtype=dtype, device=device),
            torch.tensor(normals, dtype=dtype, device=device),
        )

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        if self._has_rtree:
            query = points.detach().cpu().reshape(-1, 3).numpy()
            sdf = -tm.proximity.signed_distance(self.mesh, query)
            return torch.tensor(sdf, dtype=points.dtype, device=points.device).reshape(points.shape[:-1])
        face_points = self._hull_face_points.to(device=points.device, dtype=points.dtype)
        face_normals = self._hull_face_normals.to(device=points.device, dtype=points.dtype)
        flat_points = points.reshape(-1, 3)
        plane_distances = ((flat_points.unsqueeze(1) - face_points.unsqueeze(0)) * face_normals.unsqueeze(0)).sum(dim=-1)
        sdf = plane_distances.max(dim=1).values
        return sdf.reshape(points.shape[:-1])

    def normals(self, points: torch.Tensor) -> torch.Tensor:
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
