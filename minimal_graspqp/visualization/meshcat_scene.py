from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh as tm

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.state import GraspState
from minimal_graspqp.visualization.plotly_scene import (
    _load_visual_specs,
    _make_transform,
    _mesh_cache_load,
    _primitive_mesh,
)


def _ensure_meshcat():
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf

    return meshcat, g, tf


def _color_hex(color: str) -> int:
    mapping = {
        "lightgreen": 0x90EE90,
        "lightblue": 0xADD8E6,
        "royalblue": 0x4169E1,
        "darkorange": 0xFF8C00,
        "crimson": 0xDC143C,
    }
    return mapping.get(color, 0xCCCCCC)


def _publish_trimesh(vis, path: str, mesh: tm.Trimesh, color: str, opacity: float):
    _, g, _ = _ensure_meshcat()
    vis[path].set_object(
        g.TriangularMeshGeometry(mesh.vertices, mesh.faces),
        g.MeshLambertMaterial(color=_color_hex(color), opacity=opacity, transparent=opacity < 1.0),
    )


def _publish_points(vis, path: str, points: np.ndarray, color: str, radius: float = 0.004):
    _, g, tf = _ensure_meshcat()
    material = g.MeshLambertMaterial(color=_color_hex(color), opacity=1.0)
    for idx, point in enumerate(points):
        vis[f"{path}/{idx}"].set_object(g.Sphere(radius), material)
        vis[f"{path}/{idx}"].set_transform(tf.translation_matrix(point))


def _configure_viewer(vis):
    # Remove MeshCat's default decorations so only the scene content remains.
    vis["/Grid"].set_property("visible", False)
    vis["/Axes"].set_property("visible", False)
    vis["/Background"].set_property("visible", False)


def publish_shadow_hand_primitive_meshcat(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box | MeshObject,
    joint_values,
    contact_indices=None,
    wrist_translation=None,
    wrist_rotation=None,
):
    meshcat, _, _ = _ensure_meshcat()
    vis = meshcat.Visualizer().open()
    vis.delete()
    _configure_viewer(vis)

    primitive_mesh = _primitive_mesh(primitive)
    _publish_trimesh(vis, "scene/object", primitive_mesh, color="lightgreen", opacity=0.45)

    joint_values = joint_values.to(device=hand_model.device, dtype=hand_model.dtype)
    fk = hand_model.forward_kinematics(joint_values)
    if wrist_translation is None:
        wrist_translation = joint_values.new_zeros((1, 3))
    if wrist_rotation is None:
        wrist_rotation = np.eye(3)[None]
        wrist_rotation = joint_values.new_tensor(wrist_rotation)

    visual_specs = _load_visual_specs(hand_model)
    cache: dict[Path, tm.Trimesh] = {}
    wrist_transform = np.eye(4)
    wrist_transform[:3, :3] = wrist_rotation[0].detach().cpu().numpy()
    wrist_transform[:3, 3] = wrist_translation[0].detach().cpu().numpy()
    for spec in visual_specs:
        mesh = _mesh_cache_load(spec.mesh_path, cache).copy()
        mesh.apply_scale(spec.scale)
        mesh.apply_transform(_make_transform(spec.origin_xyz, spec.origin_rpy))
        link_transform = fk[spec.link_name][0].detach().cpu().numpy()
        mesh.apply_transform(wrist_transform @ link_transform)
        _publish_trimesh(vis, f"scene/hand/{spec.link_name}", mesh, color="lightblue", opacity=0.75)

    contacts = hand_model.contact_candidates_world(
        joint_values,
        indices=contact_indices,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
    )[0].detach().cpu().numpy()
    _publish_points(vis, "scene/contacts", contacts, color="crimson")
    return vis


def publish_optimization_result_meshcat(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box | MeshObject,
    initial_state: GraspState,
    final_state: GraspState,
    sample_index: int = 0,
):
    meshcat, _, _ = _ensure_meshcat()
    vis = meshcat.Visualizer().open()
    vis.delete()
    _configure_viewer(vis)
    _publish_trimesh(vis, "scene/object", _primitive_mesh(primitive), color="lightgreen", opacity=0.35)

    visual_specs = _load_visual_specs(hand_model)
    cache: dict[Path, tm.Trimesh] = {}

    def publish_state(prefix: str, state: GraspState, color: str, opacity: float):
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
            _publish_trimesh(vis, f"scene/{prefix}/{spec.link_name}", mesh, color=color, opacity=opacity)
        contacts = hand_model.contact_candidates_world(
            state.joint_values[sample_index : sample_index + 1],
            indices=state.contact_indices[sample_index : sample_index + 1],
            wrist_translation=state.wrist_translation[sample_index : sample_index + 1],
            wrist_rotation=state.wrist_rotation[sample_index : sample_index + 1],
        )[0].detach().cpu().numpy()
        _publish_points(vis, f"scene/{prefix}_contacts", contacts, color=color)

    publish_state("initial", initial_state, color="royalblue", opacity=0.18)
    publish_state("final", final_state, color="darkorange", opacity=0.75)
    return vis
