from __future__ import annotations

import argparse
import time

import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.state import GraspState
from minimal_graspqp.visualization import publish_optimization_result_meshcat


def build_primitive_from_metadata(metadata: dict):
    primitive_type = metadata["type"]
    if primitive_type == "mesh":
        return MeshObject(metadata["mesh_path"], scale=float(metadata.get("scale", 1.0)))
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
    return GraspState(
        joint_values=state["joint_values"],
        wrist_translation=state["wrist_translation"],
        wrist_rotation=state["wrist_rotation"],
        contact_indices=state["contact_indices"],
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize optimization results in MeshCat.")
    parser.add_argument("--input", default="outputs/primitive_optimization.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    primitive = build_primitive_from_metadata(payload["primitive"])
    hand_model = ShadowHandModel.create(device=args.device)
    initial_state = _to_state(payload, "initial_state")
    final_state = _to_state(payload, "final_state")
    vis = publish_optimization_result_meshcat(hand_model, primitive, initial_state, final_state, sample_index=args.sample_index)
    print("MeshCat optimization visualization ready.")
    print(f"Open this URL in your browser: {vis.url()}")
    if "initial_energy" in payload and "final_energy" in payload:
        print(f"Initial energy[{args.sample_index}]: {float(payload['initial_energy'][args.sample_index]):.6f}")
        print(f"Final energy[{args.sample_index}]: {float(payload['final_energy'][args.sample_index]):.6f}")
    if args.duration > 0:
        time.sleep(args.duration)
        return
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
