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
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.state import GraspState


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


def _transform_mesh(mesh: tm.Trimesh, transform: np.ndarray) -> tm.Trimesh:
    mesh_cp = mesh.copy()
    mesh_cp.apply_transform(transform)
    return mesh_cp


def _primitive_mesh(primitive: Sphere | Cylinder | Box | MeshObject) -> tm.Trimesh:
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


def create_shadow_hand_primitive_figure(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box | MeshObject,
    joint_values: torch.Tensor | None = None,
    contact_indices: torch.Tensor | None = None,
    wrist_translation: torch.Tensor | None = None,
    wrist_rotation: torch.Tensor | None = None,
) -> go.Figure:
    if joint_values is None:
        joint_values = hand_model.default_joint_state(batch_size=1)
    joint_values = joint_values.to(device=hand_model.device, dtype=hand_model.dtype)
    fk = hand_model.forward_kinematics(joint_values)
    if wrist_translation is None:
        wrist_translation = torch.zeros((1, 3), device=hand_model.device, dtype=hand_model.dtype)
    if wrist_rotation is None:
        wrist_rotation = torch.eye(3, device=hand_model.device, dtype=hand_model.dtype).unsqueeze(0)

    figure = go.Figure()
    figure.add_trace(_trimesh_to_mesh3d(_primitive_mesh(primitive), color="lightgreen", opacity=0.45, name="object"))

    visual_specs = _load_visual_specs(hand_model)
    cache: dict[Path, tm.Trimesh] = {}
    for spec in visual_specs:
        mesh = _mesh_cache_load(spec.mesh_path, cache).copy()
        mesh.apply_scale(spec.scale)
        mesh.apply_transform(_make_transform(spec.origin_xyz, spec.origin_rpy))
        world_transform = fk[spec.link_name][0].detach().cpu().numpy()
        wrist_transform = np.eye(4)
        wrist_transform[:3, :3] = wrist_rotation[0].detach().cpu().numpy()
        wrist_transform[:3, 3] = wrist_translation[0].detach().cpu().numpy()
        mesh.apply_transform(wrist_transform @ world_transform)
        figure.add_trace(_trimesh_to_mesh3d(mesh, color="lightblue", opacity=0.7, name=spec.link_name))

    contacts = hand_model.contact_candidates_world(
        joint_values,
        indices=contact_indices,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
    )
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
        title="Shadow Hand and object",
        showlegend=False,
    )
    return figure


def create_initialization_figure(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box | MeshObject,
    grasp_state: GraspState,
    max_samples: int = 6,
) -> go.Figure:
    num_samples = min(max_samples, grasp_state.batch_size)
    figure = go.Figure()
    primitive_mesh = _primitive_mesh(primitive)
    figure.add_trace(_trimesh_to_mesh3d(primitive_mesh, color="lightgreen", opacity=0.35, name="object"))

    visual_specs = _load_visual_specs(hand_model)
    cache: dict[Path, tm.Trimesh] = {}
    joint_values = grasp_state.joint_values[:num_samples].to(device=hand_model.device, dtype=hand_model.dtype)
    fk = hand_model.forward_kinematics(joint_values)

    for sample_idx in range(num_samples):
        wrist_transform = np.eye(4)
        wrist_transform[:3, :3] = grasp_state.wrist_rotation[sample_idx].detach().cpu().numpy()
        wrist_transform[:3, 3] = grasp_state.wrist_translation[sample_idx].detach().cpu().numpy()
        for spec in visual_specs:
            mesh = _mesh_cache_load(spec.mesh_path, cache).copy()
            mesh.apply_scale(spec.scale)
            mesh.apply_transform(_make_transform(spec.origin_xyz, spec.origin_rpy))
            link_transform = fk[spec.link_name][sample_idx].detach().cpu().numpy()
            mesh.apply_transform(wrist_transform @ link_transform)
            figure.add_trace(
                _trimesh_to_mesh3d(
                    mesh,
                    color="lightblue",
                    opacity=0.20,
                    name=f"init_{sample_idx}_{spec.link_name}",
                )
            )

    contacts = hand_model.contact_candidates_world(
        grasp_state.joint_values[:num_samples],
        indices=grasp_state.contact_indices[:num_samples],
        wrist_translation=grasp_state.wrist_translation[:num_samples],
        wrist_rotation=grasp_state.wrist_rotation[:num_samples],
    )
    for sample_idx in range(num_samples):
        pts = contacts[sample_idx].detach().cpu().numpy()
        figure.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 3, "color": "crimson"},
                name=f"init_contacts_{sample_idx}",
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="Shadow Hand initialization preview",
        showlegend=False,
    )
    return figure


def create_optimization_result_figure(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box | MeshObject,
    initial_state: GraspState,
    final_state: GraspState,
    sample_index: int = 0,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(_trimesh_to_mesh3d(_primitive_mesh(primitive), color="lightgreen", opacity=0.35, name="object"))

    visual_specs = _load_visual_specs(hand_model)
    cache: dict[Path, tm.Trimesh] = {}

    def add_state_meshes(state: GraspState, color: str, opacity: float, name_prefix: str):
        joint_values = state.joint_values[sample_index : sample_index + 1].to(device=hand_model.device, dtype=hand_model.dtype)
        fk = hand_model.forward_kinematics(joint_values)
        wrist_transform = np.eye(4)
        wrist_transform[:3, :3] = state.wrist_rotation[sample_index].detach().cpu().numpy()
        wrist_transform[:3, 3] = state.wrist_translation[sample_index].detach().cpu().numpy()
        for spec in visual_specs:
            mesh = _mesh_cache_load(spec.mesh_path, cache).copy()
            mesh.apply_scale(spec.scale)
            mesh.apply_transform(_make_transform(spec.origin_xyz, spec.origin_rpy))
            link_transform = fk[spec.link_name][0].detach().cpu().numpy()
            mesh.apply_transform(wrist_transform @ link_transform)
            figure.add_trace(
                _trimesh_to_mesh3d(mesh, color=color, opacity=opacity, name=f"{name_prefix}_{spec.link_name}")
            )

    add_state_meshes(initial_state, color="royalblue", opacity=0.18, name_prefix="initial")
    add_state_meshes(final_state, color="darkorange", opacity=0.65, name_prefix="final")

    initial_contacts = hand_model.contact_candidates_world(
        initial_state.joint_values[sample_index : sample_index + 1],
        indices=initial_state.contact_indices[sample_index : sample_index + 1],
        wrist_translation=initial_state.wrist_translation[sample_index : sample_index + 1],
        wrist_rotation=initial_state.wrist_rotation[sample_index : sample_index + 1],
    )[0].detach().cpu().numpy()
    final_contacts = hand_model.contact_candidates_world(
        final_state.joint_values[sample_index : sample_index + 1],
        indices=final_state.contact_indices[sample_index : sample_index + 1],
        wrist_translation=final_state.wrist_translation[sample_index : sample_index + 1],
        wrist_rotation=final_state.wrist_rotation[sample_index : sample_index + 1],
    )[0].detach().cpu().numpy()

    figure.add_trace(
        go.Scatter3d(
            x=initial_contacts[:, 0],
            y=initial_contacts[:, 1],
            z=initial_contacts[:, 2],
            mode="markers",
            marker={"size": 3, "color": "royalblue"},
            name="initial_contacts",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=final_contacts[:, 0],
            y=final_contacts[:, 1],
            z=final_contacts[:, 2],
            mode="markers",
            marker={"size": 4, "color": "darkorange"},
            name="final_contacts",
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="Optimization result: initial (blue) vs final (orange)",
        showlegend=False,
    )
    return figure
