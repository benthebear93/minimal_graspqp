from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh as tm
from transforms3d.euler import euler2mat

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, Sphere


@dataclass
class VisualMeshSpec:
    link_name: str
    mesh_path: Path
    scale: np.ndarray
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = euler2mat(*rpy)
    transform[:3, 3] = xyz
    return transform


def _load_visual_specs(model: ShadowHandModel) -> list[VisualMeshSpec]:
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


def _mesh_cache_load(mesh_path: Path, cache: dict[Path, tm.Trimesh]) -> tm.Trimesh:
    if mesh_path not in cache:
        cache[mesh_path] = tm.load(mesh_path, force="mesh", process=False)
    return cache[mesh_path]


def _trimesh_to_mesh3d(mesh: tm.Trimesh, color: str, opacity: float, name: str) -> go.Mesh3d:
    vertices = mesh.vertices
    faces = mesh.faces
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo="skip",
    )


def _primitive_mesh(primitive: Sphere | Cylinder | Box) -> tm.Trimesh:
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
    raise TypeError(f"Unsupported primitive type: {type(primitive).__name__}")


def create_shadow_hand_primitive_figure(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box,
    joint_values: torch.Tensor | None = None,
    contact_indices: torch.Tensor | None = None,
) -> go.Figure:
    if joint_values is None:
        joint_values = hand_model.default_joint_state(batch_size=1)
    joint_values = joint_values.to(device=hand_model.device, dtype=hand_model.dtype)
    fk = hand_model.forward_kinematics(joint_values)

    figure = go.Figure()
    figure.add_trace(_trimesh_to_mesh3d(_primitive_mesh(primitive), color="lightgreen", opacity=0.45, name="object"))

    visual_specs = _load_visual_specs(hand_model)
    cache: dict[Path, tm.Trimesh] = {}
    for spec in visual_specs:
        mesh = _mesh_cache_load(spec.mesh_path, cache).copy()
        mesh.apply_scale(spec.scale)
        mesh.apply_transform(_make_transform(spec.origin_xyz, spec.origin_rpy))
        world_transform = fk[spec.link_name][0].detach().cpu().numpy()
        mesh.apply_transform(world_transform)
        figure.add_trace(_trimesh_to_mesh3d(mesh, color="lightblue", opacity=0.7, name=spec.link_name))

    contacts = hand_model.contact_candidates_world(joint_values, indices=contact_indices)
    contact_points = contacts[0].detach().cpu().numpy()
    figure.add_trace(
        go.Scatter3d(
            x=contact_points[:, 0],
            y=contact_points[:, 1],
            z=contact_points[:, 2],
            mode="markers",
            marker={"size": 3, "color": "crimson"},
            name="contacts" if contact_indices is not None else "contact candidates",
        )
    )
    figure.update_layout(
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="Shadow Hand and primitive object",
        showlegend=False,
    )
    return figure
