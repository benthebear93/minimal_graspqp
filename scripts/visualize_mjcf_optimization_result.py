from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh as tm
import viser

from minimal_graspqp.hands import MJCFShadowHandModel
from minimal_graspqp.hands.mjcf_shadow_hand import _body_local_mesh
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.state import GraspState
from minimal_graspqp.visualization.shared_scene import primitive_mesh


def _to_state(payload: dict, key: str) -> GraspState:
    state = payload[key]
    return GraspState(
        joint_values=state["joint_values"],
        wrist_translation=state["wrist_translation"],
        wrist_rotation=state["wrist_rotation"],
        contact_indices=state["contact_indices"],
    )


def _primitive_from_metadata(metadata: dict):
    primitive_type = metadata["type"]
    if primitive_type == "mesh":
        return MeshObject(
            Path(metadata["mesh_path"]),
            scale=float(metadata.get("scale", 1.0)),
            rotation_rpy=tuple(metadata.get("rotation_rpy", [0.0, 0.0, 0.0])),
        )
    center = tuple(metadata.get("center", [0.0, 0.0, 0.0]))
    if primitive_type == "sphere":
        return Sphere(radius=float(metadata["radius"]), center=center)
    if primitive_type == "cylinder":
        return Cylinder(radius=float(metadata["radius"]), half_height=float(metadata["half_height"]), center=center)
    if primitive_type == "box":
        return Box(half_extents=tuple(metadata["half_extents"]), center=center)
    raise ValueError(f"Unsupported primitive: {primitive_type}")


def _apply_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    hom = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ hom.T).T[:, :3]


def _render_hand(server: viser.ViserServer, hand: MJCFShadowHandModel, state: GraspState, sample: int) -> None:
    joint_values = state.joint_values[sample : sample + 1].to(device=hand.device, dtype=hand.dtype)
    wrist_translation = state.wrist_translation[sample].detach().cpu().numpy().astype(np.float64)
    wrist_rotation = state.wrist_rotation[sample].detach().cpu().numpy().astype(np.float64)
    wrist_tf = np.eye(4, dtype=np.float64)
    wrist_tf[:3, :3] = wrist_rotation
    wrist_tf[:3, 3] = wrist_translation
    fk = hand.forward_kinematics(joint_values)
    for body_id in range(1, hand.model.nbody):
        link_name = hand.link_names[body_id]
        if link_name not in fk:
            continue
        mesh = _body_local_mesh(hand.model, hand.data, body_id, groups=(2,))
        if mesh is None:
            mesh = _body_local_mesh(hand.model, hand.data, body_id, groups=(3,))
        if mesh is None:
            continue
        transform = wrist_tf @ fk[link_name][0].detach().cpu().numpy()
        vertices = _apply_transform(np.asarray(mesh.vertices, dtype=np.float64), transform)
        server.scene.add_mesh_simple(
            f"/hand/{link_name}_{body_id}",
            vertices=vertices.astype(np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=(160, 180, 190),
            opacity=0.72,
        )


def _render_contacts(server: viser.ViserServer, hand: MJCFShadowHandModel, state: GraspState, sample: int, radius: float) -> None:
    joint_values = state.joint_values[sample : sample + 1].to(device=hand.device, dtype=hand.dtype)
    wrist_translation = state.wrist_translation[sample : sample + 1].to(device=hand.device, dtype=hand.dtype)
    wrist_rotation = state.wrist_rotation[sample : sample + 1].to(device=hand.device, dtype=hand.dtype)
    indices = state.contact_indices[sample : sample + 1].to(device=hand.device)
    contacts = hand.contact_candidates_world(
        joint_values,
        indices=indices,
        wrist_translation=wrist_translation,
        wrist_rotation=wrist_rotation,
    )[0].detach().cpu().numpy()
    for idx, point in enumerate(contacts):
        server.scene.add_icosphere(
            f"/contacts/{idx:02d}",
            radius=radius,
            color=(220, 20, 60),
            opacity=1.0,
            position=point,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an MJCF-hand GraspQP result in viser.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--contact-radius", type=float, default=0.002)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    final_state = _to_state(payload, "final_state")
    sample = int(args.sample_index) if args.sample_index is not None else int(torch.argmin(payload["final_energy"]).item())
    primitive = _primitive_from_metadata(payload["primitive"])
    hand_info = payload.get("hand", {})
    hand_kwargs = {
        "device": "cpu",
        "fingertips_only": bool(hand_info.get("fingertips_only", False)),
        "allowed_contact_links": hand_info.get("allowed_contact_links"),
    }
    if hand_info.get("mjcf_path"):
        hand_kwargs["mjcf_path"] = hand_info["mjcf_path"]
    hand = MJCFShadowHandModel.create(**hand_kwargs)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights(cast_shadow=False)

    obj_mesh = primitive_mesh(primitive)
    server.scene.add_mesh_simple(
        "/object",
        vertices=np.asarray(obj_mesh.vertices, dtype=np.float32),
        faces=np.asarray(obj_mesh.faces, dtype=np.uint32),
        color=(20, 150, 40),
        opacity=0.88,
    )
    _render_hand(server, hand, final_state, sample)
    _render_contacts(server, hand, final_state, sample, args.contact_radius)
    print(f"MJCF GraspQP visualization ready: http://localhost:{args.port}")
    print(f"sample={sample} final_energy={float(payload['final_energy'][sample]):.6f}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
