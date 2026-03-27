from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import MeshObject
from minimal_graspqp.rotation import palm_down_rotation
from minimal_graspqp.visualization import create_initialization_figure


def main():
    parser = argparse.ArgumentParser(description="Visualize Shadow Hand initialization against a mesh object.")
    parser.add_argument(
        "--mesh-path",
        default="/home/haegu/mj_sim/assets/syringe/assets/syringe.obj",
        help="Path to a mesh object file.",
    )
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-contacts", type=int, default=4)
    parser.add_argument("--output", default="outputs/shadow_hand_mesh_init.html")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--palm-down", action="store_true")
    parser.add_argument("--fingertips-only", action="store_true", help="Restrict contact candidates to fingertip distal links only.")
    args = parser.parse_args()

    hand_model = ShadowHandModel.create(device=args.device, fingertips_only=args.fingertips_only)
    mesh_object = MeshObject(Path(args.mesh_path), scale=args.mesh_scale)
    base_rotation = None
    if args.palm_down:
        base_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device)
    grasp_state = initialize_grasps_for_primitive(
        hand_model,
        mesh_object,
        batch_size=args.batch_size,
        num_contacts=args.num_contacts,
        base_wrist_rotation=base_rotation,
    )
    figure = create_initialization_figure(hand_model, mesh_object, grasp_state, max_samples=args.batch_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path))
    print(f"Wrote visualization to {output_path}")
    if args.show:
        figure.show()


if __name__ == "__main__":
    main()
