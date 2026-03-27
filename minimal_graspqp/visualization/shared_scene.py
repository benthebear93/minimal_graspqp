from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh as tm
from transforms3d.euler import euler2mat

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere


@dataclass
class VisualMeshSpec:
    link_name: str
    mesh_path: Path
    scale: np.ndarray
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = euler2mat(*rpy)
    transform[:3, 3] = xyz
    return transform


def load_visual_specs(model: ShadowHandModel) -> list[VisualMeshSpec]:
    root = ET.fromstring(model.metadata.urdf_path.read_text())
    specs: list[VisualMeshSpec] = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh = geometry.find("mesh")
            if mesh is None:
                continue
            filename = mesh.attrib["filename"].replace("package://", "")
            mesh_path = (model.metadata.asset_dir / filename).resolve()
            scale_raw = mesh.attrib.get("scale", "1 1 1")
            scale = np.array([float(v) for v in scale_raw.split()], dtype=float)
            origin = visual.find("origin")
            if origin is None:
                xyz = np.zeros(3, dtype=float)
                rpy = np.zeros(3, dtype=float)
            else:
                xyz = np.array([float(v) for v in origin.attrib.get("xyz", "0 0 0").split()], dtype=float)
                rpy = np.array([float(v) for v in origin.attrib.get("rpy", "0 0 0").split()], dtype=float)
            specs.append(VisualMeshSpec(link_name=link_name, mesh_path=mesh_path, scale=scale, origin_xyz=xyz, origin_rpy=rpy))
    return specs


def mesh_cache_load(mesh_path: Path, cache: dict[Path, tm.Trimesh]) -> tm.Trimesh:
    if mesh_path not in cache:
        cache[mesh_path] = tm.load(mesh_path, force="mesh", process=False)
    return cache[mesh_path]


def primitive_mesh(primitive: Sphere | Cylinder | Box | MeshObject) -> tm.Trimesh:
    if isinstance(primitive, Sphere):
        mesh = tm.creation.icosphere(subdivisions=3, radius=primitive.radius)
        mesh.apply_translation(np.array(primitive.center))
        return mesh
    if isinstance(primitive, Cylinder):
        mesh = tm.creation.cylinder(radius=primitive.radius, height=2.0 * primitive.half_height, sections=48)
        mesh.apply_translation(np.array(primitive.center))
        return mesh
    if isinstance(primitive, Box):
        mesh = tm.creation.box(extents=2.0 * np.array(primitive.half_extents))
        mesh.apply_translation(np.array(primitive.center))
        return mesh
    if isinstance(primitive, MeshObject):
        return primitive.mesh.copy()
    raise TypeError(f"Unsupported object type: {type(primitive).__name__}")
