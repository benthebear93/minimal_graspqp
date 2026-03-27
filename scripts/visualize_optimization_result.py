from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.state import GraspState
from minimal_graspqp.visualization import create_optimization_result_figure


def build_primitive_from_metadata(metadata: dict):
    primitive_type = metadata["type"]
    if primitive_type == "mesh":
        return MeshObject(Path(metadata["mesh_path"]), scale=float(metadata.get("scale", 1.0)))
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
    parser = argparse.ArgumentParser(description="Visualize initial vs final grasp state from an optimization result.")
    parser.add_argument("--input", default="outputs/primitive_optimization.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/primitive_optimization.html")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    primitive = build_primitive_from_metadata(payload["primitive"])
    initial_state = _to_state(payload, "initial_state")
    final_state = _to_state(payload, "final_state")
    hand_model = ShadowHandModel.create(device=args.device)

    figure = create_optimization_result_figure(
        hand_model,
        primitive,
        initial_state=initial_state,
        final_state=final_state,
        sample_index=args.sample_index,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path))
    print(f"Wrote optimization visualization to {output_path}")
    if "initial_energy" in payload and "final_energy" in payload:
        print(f"Initial energy[{args.sample_index}]: {float(payload['initial_energy'][args.sample_index]):.6f}")
        print(f"Final energy[{args.sample_index}]: {float(payload['final_energy'][args.sample_index]):.6f}")
    if args.show:
        figure.show()


if __name__ == "__main__":
    main()
