from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh as tm
import viser

from minimal_graspqp.assets import resolve_shadow_hand_asset_dir
from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.rotation import palm_down_rotation
from minimal_graspqp.visualization.shared_scene import (
    load_visual_specs,
    make_transform,
    mesh_cache_load,
)


def _load_overrides(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_overrides(path: Path, overrides: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")


def _wrist_rotation(hand_model: ShadowHandModel, palm_down: bool) -> torch.Tensor:
    if palm_down:
        return palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device).unsqueeze(0)
    return torch.eye(3, dtype=hand_model.dtype, device=hand_model.device).unsqueeze(0)


def _compose_wrist_transform(wrist_translation: torch.Tensor, wrist_rotation: torch.Tensor) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = wrist_rotation[0].detach().cpu().numpy()
    transform[:3, 3] = wrist_translation[0].detach().cpu().numpy()
    return transform


def _contact_world_points(
    hand_model: ShadowHandModel,
    joint_values: torch.Tensor,
    wrist_translation: torch.Tensor,
    wrist_rotation: torch.Tensor,
) -> np.ndarray:
    return hand_model.contact_candidates_world(
        joint_values,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
    )[0].detach().cpu().numpy()


def _contact_link_world_transform(
    hand_model: ShadowHandModel,
    joint_values: torch.Tensor,
    wrist_translation: torch.Tensor,
    wrist_rotation: torch.Tensor,
    index: int,
) -> np.ndarray:
    link_name = hand_model.metadata.contact_candidate_links[index]
    fk = hand_model.forward_kinematics(joint_values)
    wrist_transform = _compose_wrist_transform(wrist_translation, wrist_rotation)
    return wrist_transform @ fk[link_name][0].detach().cpu().numpy()


def _render_hand(server: viser.ViserServer, hand_model: ShadowHandModel, joint_values: torch.Tensor, wrist_transform: np.ndarray) -> None:
    visual_specs = load_visual_specs(hand_model)
    cache: dict[Path, tm.Trimesh] = {}
    fk = hand_model.forward_kinematics(joint_values)
    for spec in visual_specs:
        mesh = mesh_cache_load(spec.mesh_path, cache).copy()
        mesh.apply_scale(spec.scale)
        mesh.apply_transform(make_transform(spec.origin_xyz, spec.origin_rpy))
        link_transform = fk[spec.link_name][0].detach().cpu().numpy()
        mesh.apply_transform(wrist_transform @ link_transform)
        server.scene.add_mesh_simple(
            f"/hand/{spec.link_name}",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=(173, 216, 230),
            opacity=0.8,
        )


def _render_penetration_spheres(
    server: viser.ViserServer,
    hand_model: ShadowHandModel,
    joint_values: torch.Tensor,
    wrist_translation: torch.Tensor,
    wrist_rotation: torch.Tensor,
) -> None:
    centers, radii, _ = hand_model.penetration_spheres_world(
        joint_values,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
    )
    for idx, (center, radius) in enumerate(zip(centers[0].detach().cpu().numpy(), radii[0].detach().cpu().numpy())):
        server.scene.add_icosphere(
            f"/penetration/{idx}",
            radius=float(radius),
            color=(218, 165, 32),
            opacity=0.18,
            position=center,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit Shadow Hand contact candidates with click selection in viser.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--override-file", default="outputs/shadow_hand_contact_overrides.json")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--palm-down", action="store_true")
    parser.add_argument("--fingertips-only", action="store_true")
    parser.add_argument("--index", type=int, default=None, help="Optional initial selected index.")
    args = parser.parse_args()

    override_path = Path(args.override_file)
    overrides = _load_overrides(override_path)

    hand_model = ShadowHandModel.create(
        device=args.device,
        fingertips_only=args.fingertips_only,
        contact_points_override_path=override_path if override_path.exists() else None,
    )
    joint_values = hand_model.default_joint_state(batch_size=1)
    wrist_translation = torch.zeros((1, 3), dtype=hand_model.dtype, device=hand_model.device)
    wrist_rotation = _wrist_rotation(hand_model, args.palm_down)
    wrist_transform = _compose_wrist_transform(wrist_translation, wrist_rotation)
    world_points = _contact_world_points(hand_model, joint_values, wrist_translation, wrist_rotation)
    local_points = hand_model.metadata.contact_candidate_points.detach().cpu().numpy()
    link_names = hand_model.metadata.contact_candidate_links

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights(cast_shadow=False)

    _render_hand(server, hand_model, joint_values, wrist_transform)
    _render_penetration_spheres(server, hand_model, joint_values, wrist_translation, wrist_rotation)

    points_folder = server.scene.add_frame("/contacts")
    point_handles: list[viser.SceneNodeHandle] = []
    state: dict[str, object] = {"selected_index": args.index, "syncing_gui": False}

    selected_marker = server.scene.add_icosphere(
        "/selected_contact",
        radius=0.008,
        color=(50, 205, 50),
        opacity=1.0,
        visible=False,
    )
    gizmo = server.scene.add_transform_controls(
        "/selected_contact_gizmo",
        scale=0.12,
        disable_rotations=True,
        depth_test=False,
        visible=False,
    )

    gui_index = server.gui.add_number("Selected Index", initial_value=-1 if args.index is None else args.index, step=1)
    gui_link = server.gui.add_text("Selected Link", initial_value="", disabled=True)
    gui_local = server.gui.add_vector3("Local XYZ", initial_value=(0.0, 0.0, 0.0), step=0.0005)
    gui_world = server.gui.add_vector3("World XYZ", initial_value=(0.0, 0.0, 0.0), step=0.0005)
    gui_step = server.gui.add_number("Nudge Step", initial_value=0.001, step=0.0005)
    gui_save = server.gui.add_button("Save Overrides")
    gui_reset = server.gui.add_button("Reset Selected Override")
    gui_status = server.gui.add_text("Status", initial_value="Ready", disabled=True)

    def set_gui_value(handle, value) -> None:
        state["syncing_gui"] = True
        try:
            handle.value = value
        finally:
            state["syncing_gui"] = False

    def sync_selected(index: int | None) -> None:
        state["selected_index"] = index
        if index is None:
            set_gui_value(gui_index, -1)
            set_gui_value(gui_link, "")
            set_gui_value(gui_local, (0.0, 0.0, 0.0))
            set_gui_value(gui_world, (0.0, 0.0, 0.0))
            selected_marker.visible = False
            gizmo.visible = False
            return
        world = world_points[index]
        local = local_points[index]
        set_gui_value(gui_index, int(index))
        set_gui_value(gui_link, link_names[index])
        set_gui_value(gui_local, tuple(float(v) for v in local))
        set_gui_value(gui_world, tuple(float(v) for v in world))
        selected_marker.position = world
        selected_marker.visible = True
        gizmo.position = world
        gizmo.visible = True

    def apply_world_update(index: int, world_xyz: np.ndarray) -> None:
        link_world = _contact_link_world_transform(hand_model, joint_values, wrist_translation, wrist_rotation, index)
        local_h = np.linalg.inv(link_world) @ np.concatenate([world_xyz.astype(np.float32), np.array([1.0], dtype=np.float32)])
        local_points[index] = local_h[:3]
        world_points[index] = world_xyz.astype(np.float32)
        point_handles[index].position = world_points[index]
        selected_marker.position = world_points[index]
        set_gui_value(gui_local, tuple(float(v) for v in local_points[index]))
        set_gui_value(gui_world, tuple(float(v) for v in world_points[index]))
        overrides[str(index)] = [float(v) for v in local_points[index]]
        set_gui_value(gui_status, f"Edited index {index}")

    def nudge_selected(axis: int, sign: float) -> None:
        index = state["selected_index"]
        if index is None:
            return
        updated = world_points[index].copy()
        updated[axis] += float(gui_step.value) * sign
        gizmo.position = updated
        apply_world_update(index, updated)

    for index, (world, link_name) in enumerate(zip(world_points, link_names)):
        handle = server.scene.add_icosphere(
            f"/contacts/{index}",
            radius=0.004,
            color=(220, 20, 60),
            opacity=1.0,
            position=world,
        )
        point_handles.append(handle)

        def register_click(idx: int, h: viser.SceneNodeHandle, link: str) -> None:
            @h.on_click
            def _(_) -> None:
                sync_selected(idx)
                set_gui_value(gui_status, f"Selected {idx} ({link})")

        register_click(index, handle, link_name)

    @gizmo.on_update
    def _(_) -> None:
        index = state["selected_index"]
        if index is None:
            return
        apply_world_update(index, np.asarray(gizmo.position, dtype=np.float32))

    @gui_local.on_update
    def _(_) -> None:
        if state["syncing_gui"]:
            return
        index = state["selected_index"]
        if index is None:
            return
        local_points[index] = np.asarray(gui_local.value, dtype=np.float32)
        link_world = _contact_link_world_transform(hand_model, joint_values, wrist_translation, wrist_rotation, index)
        world_h = link_world @ np.concatenate([local_points[index], np.array([1.0], dtype=np.float32)])
        world_points[index] = world_h[:3]
        point_handles[index].position = world_points[index]
        selected_marker.position = world_points[index]
        gizmo.position = world_points[index]
        set_gui_value(gui_world, tuple(float(v) for v in world_points[index]))
        overrides[str(index)] = [float(v) for v in local_points[index]]
        set_gui_value(gui_status, f"Edited index {index} from local coordinates")

    @gui_index.on_update
    def _(_) -> None:
        if state["syncing_gui"]:
            return
        index = int(gui_index.value)
        if 0 <= index < len(point_handles):
            sync_selected(index)
        else:
            sync_selected(None)

    @gui_save.on_click
    def _(_) -> None:
        _save_overrides(override_path, overrides)
        set_gui_value(gui_status, f"Saved overrides to {override_path}")

    @gui_reset.on_click
    def _(_) -> None:
        index = state["selected_index"]
        if index is None:
            return
        overrides.pop(str(index), None)
        base_model = ShadowHandModel.create(device=args.device, fingertips_only=args.fingertips_only)
        local_points[index] = base_model.metadata.contact_candidate_points[index].detach().cpu().numpy()
        link_world = _contact_link_world_transform(base_model, joint_values, wrist_translation, wrist_rotation, index)
        world_h = link_world @ np.concatenate([local_points[index], np.array([1.0], dtype=np.float32)])
        world_points[index] = world_h[:3]
        point_handles[index].position = world_points[index]
        gizmo.position = world_points[index]
        selected_marker.position = world_points[index]
        set_gui_value(gui_local, tuple(float(v) for v in local_points[index]))
        set_gui_value(gui_world, tuple(float(v) for v in world_points[index]))
        set_gui_value(gui_status, f"Reset override for index {index}")

    with server.gui.add_folder("Nudge"):
        server.gui.add_button("+X").on_click(lambda _: nudge_selected(0, +1.0))
        server.gui.add_button("-X").on_click(lambda _: nudge_selected(0, -1.0))
        server.gui.add_button("+Y").on_click(lambda _: nudge_selected(1, +1.0))
        server.gui.add_button("-Y").on_click(lambda _: nudge_selected(1, -1.0))
        server.gui.add_button("+Z").on_click(lambda _: nudge_selected(2, +1.0))
        server.gui.add_button("-Z").on_click(lambda _: nudge_selected(2, -1.0))

    sync_selected(args.index if args.index is not None and 0 <= args.index < len(point_handles) else None)

    print(f"Shadow Hand assets: {resolve_shadow_hand_asset_dir()}")
    print(f"Override file: {override_path.resolve()}")
    print(f"Viser contact editor ready: http://{args.host}:{args.port}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
