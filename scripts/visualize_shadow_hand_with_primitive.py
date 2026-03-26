from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, Sphere
from minimal_graspqp.visualization import create_shadow_hand_primitive_figure


def build_primitive(args):
    if args.primitive == "sphere":
        return Sphere(radius=args.radius)
    if args.primitive == "cylinder":
        return Cylinder(radius=args.radius, half_height=args.half_height)
    if args.primitive == "box":
        return Box(half_extents=(args.half_x, args.half_y, args.half_z))
    raise ValueError(f"Unsupported primitive: {args.primitive}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the Shadow Hand against a primitive object.")
    parser.add_argument("--primitive", choices=["sphere", "cylinder", "box"], default="sphere")
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--half-height", type=float, default=0.08)
    parser.add_argument("--half-x", type=float, default=0.04)
    parser.add_argument("--half-y", type=float, default=0.04)
    parser.add_argument("--half-z", type=float, default=0.04)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="shadow_hand_primitive.html")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--contacts", nargs="*", type=int, default=None, help="Optional contact candidate indices to highlight.")
    args = parser.parse_args()

    hand_model = ShadowHandModel.create(device=args.device)
    primitive = build_primitive(args)
    joint_values = hand_model.default_joint_state(batch_size=1)

    contact_indices = None
    if args.contacts:
        contact_indices = torch.tensor([args.contacts], dtype=torch.long, device=hand_model.device)

    figure = create_shadow_hand_primitive_figure(hand_model, primitive, joint_values=joint_values, contact_indices=contact_indices)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path))
    print(f"Wrote visualization to {output_path}")
    if args.show:
        figure.show()


if __name__ == "__main__":
    main()

