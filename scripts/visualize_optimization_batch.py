from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.state import GraspState
from minimal_graspqp.visualization import publish_optimization_batch_viser


def build_primitive_from_metadata(metadata: dict):
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
    raise ValueError(f"Unsupported object metadata: {primitive_type}")


def _to_state(payload: dict, key: str) -> GraspState:
    state = payload[key]
    if isinstance(state, GraspState):
        return state
    return GraspState(
        joint_values=state["joint_values"],
        wrist_translation=state["wrist_translation"],
        wrist_rotation=state["wrist_rotation"],
        contact_indices=state["contact_indices"],
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize all optimization batch samples in one viser scene.")
    parser.add_argument("--input", default="outputs/primitive_optimization.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--spacing", type=float, default=0.25, help="Horizontal spacing between batch samples.")
    parser.add_argument("--row-spacing", type=float, default=0.35, help="Vertical spacing between initial and final rows.")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    primitive = build_primitive_from_metadata(payload["primitive"])
    initial_state = _to_state(payload, "initial_state")
    final_state = _to_state(payload, "final_state")
    fingertips_only = bool(payload.get("hand", {}).get("fingertips_only", False))
    allowed_contact_links = payload.get("hand", {}).get("allowed_contact_links")
    hand_model = ShadowHandModel.create(
        device=args.device,
        fingertips_only=fingertips_only,
        allowed_contact_links=allowed_contact_links,
    )

    server = publish_optimization_batch_viser(
        hand_model,
        primitive,
        initial_state=initial_state,
        final_state=final_state,
        spacing=args.spacing,
        row_spacing=args.row_spacing,
        host=args.host,
        port=args.port,
    )
    batch_size = int(final_state.batch_size)
    print("Viser batch optimization visualization ready.")
    print(f"Open this URL in your browser: http://localhost:{args.port}")
    print(f"Showing {batch_size} samples: initial row at y=0.0, final row at y={args.row_spacing}.")
    if "initial_energy" in payload and "final_energy" in payload:
        initial_energy = payload["initial_energy"].detach().cpu()
        final_energy = payload["final_energy"].detach().cpu()
        best_index = int(torch.argmin(final_energy).item())
        print(f"Best final-energy sample: {best_index}")
        for sample_index in range(batch_size):
            print(
                f"sample {sample_index}: "
                f"initial={float(initial_energy[sample_index]):.6f}, "
                f"final={float(final_energy[sample_index]):.6f}"
            )
    if args.duration > 0:
        time.sleep(args.duration)
        server.stop()
        return
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
