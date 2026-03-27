from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh as tm
import viser

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.state import GraspState
from minimal_graspqp.visualization.shared_scene import (
    load_visual_specs,
    make_transform,
    mesh_cache_load,
    primitive_mesh,
)


def _rgb(color: str) -> tuple[int, int, int]:
    mapping = {
        "lightgreen": (144, 238, 144),
        "lightblue": (173, 216, 230),
        "royalblue": (65, 105, 225),
        "darkorange": (255, 140, 0),
        "crimson": (220, 20, 60),
        "goldenrod": (218, 165, 32),
        "limegreen": (50, 205, 50),
    }
    return mapping.get(color, (204, 204, 204))


def _new_server(host: str, port: int) -> viser.ViserServer:
    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights(cast_shadow=False)
    return server


def _add_mesh(server: viser.ViserServer, name: str, mesh: tm.Trimesh, color: str, opacity: float) -> None:
    server.scene.add_mesh_simple(
        name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=_rgb(color),
        opacity=opacity,
    )


def _add_points(
    server: viser.ViserServer,
    prefix: str,
    points: np.ndarray,
    color: str,
    radius: float = 0.004,
) -> None:
    for idx, point in enumerate(points):
        server.scene.add_icosphere(
            f"{prefix}/{idx}",
            radius=radius,
            color=_rgb(color),
            opacity=1.0,
            position=point,
        )


def _add_spheres(
    server: viser.ViserServer,
    prefix: str,
    centers: np.ndarray,
    radii: np.ndarray,
    color: str,
    opacity: float,
) -> None:
    for idx, (center, radius) in enumerate(zip(centers, radii)):
        server.scene.add_icosphere(
            f"{prefix}/{idx}",
            radius=float(radius),
            color=_rgb(color),
            opacity=opacity,
            position=center,
        )


def _wrist_transform(wrist_translation: torch.Tensor, wrist_rotation: torch.Tensor) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = wrist_rotation[0].detach().cpu().numpy()
    transform[:3, 3] = wrist_translation[0].detach().cpu().numpy()
    return transform


def _render_hand_meshes(
    server: viser.ViserServer,
    prefix: str,
    hand_model: ShadowHandModel,
    joint_values: torch.Tensor,
    wrist_translation: torch.Tensor,
    wrist_rotation: torch.Tensor,
    color: str,
    opacity: float,
) -> None:
    fk = hand_model.forward_kinematics(joint_values)
    visual_specs = load_visual_specs(hand_model)
    cache: dict[Path, tm.Trimesh] = {}
    wrist_tf = _wrist_transform(wrist_translation, wrist_rotation)
    for spec in visual_specs:
        mesh = mesh_cache_load(spec.mesh_path, cache).copy()
        mesh.apply_scale(spec.scale)
        mesh.apply_transform(make_transform(spec.origin_xyz, spec.origin_rpy))
        mesh.apply_transform(wrist_tf @ fk[spec.link_name][0].detach().cpu().numpy())
        _add_mesh(server, f"{prefix}/{spec.link_name}", mesh, color=color, opacity=opacity)


def publish_shadow_hand_primitive_viser(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box | MeshObject,
    joint_values: torch.Tensor,
    *,
    contact_indices: torch.Tensor | None = None,
    highlight_contact_indices: list[int] | None = None,
    wrist_translation: torch.Tensor | None = None,
    wrist_rotation: torch.Tensor | None = None,
    show_penetration_spheres: bool = True,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> viser.ViserServer:
    server = _new_server(host=host, port=port)
    joint_values = joint_values.to(device=hand_model.device, dtype=hand_model.dtype)
    if wrist_translation is None:
        wrist_translation = torch.zeros((1, 3), device=hand_model.device, dtype=hand_model.dtype)
    if wrist_rotation is None:
        wrist_rotation = torch.eye(3, device=hand_model.device, dtype=hand_model.dtype).unsqueeze(0)

    _add_mesh(server, "/object", primitive_mesh(primitive), color="lightgreen", opacity=0.45)
    _render_hand_meshes(
        server,
        "/hand",
        hand_model,
        joint_values,
        wrist_translation,
        wrist_rotation,
        color="lightblue",
        opacity=0.8,
    )

    contacts = hand_model.contact_candidates_world(
        joint_values,
        indices=contact_indices,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
    )[0].detach().cpu().numpy()
    _add_points(server, "/contacts", contacts, color="crimson")

    if highlight_contact_indices:
        highlight = contacts[np.asarray(highlight_contact_indices, dtype=np.int64)]
        _add_points(server, "/highlight_contacts", highlight, color="limegreen", radius=0.007)

    if show_penetration_spheres:
        centers, radii, _ = hand_model.penetration_spheres_world(
            joint_values,
            wrist_translation=wrist_translation,
            wrist_rotation=wrist_rotation,
        )
        _add_spheres(
            server,
            "/penetration_spheres",
            centers[0].detach().cpu().numpy(),
            radii[0].detach().cpu().numpy(),
            color="goldenrod",
            opacity=0.25,
        )
    return server


def publish_initialization_viser(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box | MeshObject,
    state: GraspState,
    *,
    spacing: float = 0.25,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> viser.ViserServer:
    server = _new_server(host=host, port=port)

    for sample_index in range(state.batch_size):
        sample_offset = np.array([sample_index * spacing, 0.0, 0.0], dtype=np.float32)
        object_mesh = primitive_mesh(primitive).copy()
        object_mesh.apply_translation(sample_offset)
        _add_mesh(server, f"/sample_{sample_index}/object", object_mesh, color="lightgreen", opacity=0.45)

        joint_values = state.joint_values[sample_index : sample_index + 1].to(device=hand_model.device, dtype=hand_model.dtype)
        wrist_translation = state.wrist_translation[sample_index : sample_index + 1].clone()
        wrist_translation[:, 0] += spacing * sample_index
        wrist_rotation = state.wrist_rotation[sample_index : sample_index + 1]
        _render_hand_meshes(
            server,
            f"/sample_{sample_index}/hand",
            hand_model,
            joint_values,
            wrist_translation,
            wrist_rotation,
            color="lightblue",
            opacity=0.3,
        )
        contacts = hand_model.contact_candidates_world(
            state.joint_values[sample_index : sample_index + 1],
            indices=state.contact_indices[sample_index : sample_index + 1],
            wrist_translation=wrist_translation,
            wrist_rotation=wrist_rotation,
        )[0].detach().cpu().numpy()
        _add_points(server, f"/sample_{sample_index}/contacts", contacts, color="crimson")

    return server


def publish_optimization_result_viser(
    hand_model: ShadowHandModel,
    primitive: Sphere | Cylinder | Box | MeshObject,
    initial_state: GraspState,
    final_state: GraspState,
    *,
    sample_index: int = 0,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> viser.ViserServer:
    server = _new_server(host=host, port=port)
    _add_mesh(server, "/object", primitive_mesh(primitive), color="lightgreen", opacity=0.35)

    def add_state(prefix: str, state: GraspState, color: str, opacity: float) -> None:
        joint_values = state.joint_values[sample_index : sample_index + 1].to(device=hand_model.device, dtype=hand_model.dtype)
        wrist_translation = state.wrist_translation[sample_index : sample_index + 1]
        wrist_rotation = state.wrist_rotation[sample_index : sample_index + 1]
        _render_hand_meshes(server, prefix, hand_model, joint_values, wrist_translation, wrist_rotation, color=color, opacity=opacity)
        contacts = hand_model.contact_candidates_world(
            joint_values,
            indices=state.contact_indices[sample_index : sample_index + 1],
            wrist_translation=wrist_translation,
            wrist_rotation=wrist_rotation,
        )[0].detach().cpu().numpy()
        _add_points(server, f"{prefix}_contacts", contacts, color=color)

    add_state("/initial", initial_state, color="royalblue", opacity=0.2)
    add_state("/final", final_state, color="darkorange", opacity=0.8)
    return server
