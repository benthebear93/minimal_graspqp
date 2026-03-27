from __future__ import annotations

import argparse
from pathlib import Path

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import Box, Cylinder, Sphere
from minimal_graspqp.rotation import palm_down_rotation
from minimal_graspqp.visualization import create_initialization_figure


def build_primitive(args):
    if args.primitive == "sphere":
        return Sphere(radius=args.radius)
    if args.primitive == "cylinder":
        return Cylinder(radius=args.radius, half_height=args.half_height)
    if args.primitive == "box":
        return Box(half_extents=(args.half_x, args.half_y, args.half_z))
    raise ValueError(f"Unsupported primitive: {args.primitive}")


def main():
    parser = argparse.ArgumentParser(description="Visualize initialized Shadow Hand grasp candidates around a primitive.")
    parser.add_argument("--primitive", choices=["sphere", "cylinder", "box"], default="sphere")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-contacts", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--half-height", type=float, default=0.08)
    parser.add_argument("--half-x", type=float, default=0.04)
    parser.add_argument("--half-y", type=float, default=0.04)
    parser.add_argument("--half-z", type=float, default=0.04)
    parser.add_argument("--distance-lower", type=float, default=0.08)
    parser.add_argument("--distance-upper", type=float, default=0.12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="shadow_hand_initialization.html")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--palm-down", action="store_true", help="Bias initialization around a palm-down wrist orientation.")
    parser.add_argument("--fingertips-only", action="store_true", help="Restrict contact candidates to fingertip distal links only.")
    args = parser.parse_args()

    hand_model = ShadowHandModel.create(device=args.device, fingertips_only=args.fingertips_only)
    primitive = build_primitive(args)
    base_wrist_rotation = None
    if args.palm_down:
        base_wrist_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device)
    grasp_state = initialize_grasps_for_primitive(
        hand_model,
        primitive,
        batch_size=args.batch_size,
        distance_lower=args.distance_lower,
        distance_upper=args.distance_upper,
        num_contacts=args.num_contacts,
        base_wrist_rotation=base_wrist_rotation,
    )
    figure = create_initialization_figure(hand_model, primitive, grasp_state, max_samples=args.batch_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path))
    print(f"Wrote initialization visualization to {output_path}")
    if args.show:
        figure.show()


if __name__ == "__main__":
    main()
