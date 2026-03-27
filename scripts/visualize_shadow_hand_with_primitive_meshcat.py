from __future__ import annotations

import argparse
import time

import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Box, Cylinder, Sphere
from minimal_graspqp.rotation import palm_down_rotation
from minimal_graspqp.visualization import publish_shadow_hand_primitive_viser


def build_primitive(args):
    if args.primitive == "sphere":
        return Sphere(radius=args.radius)
    if args.primitive == "cylinder":
        return Cylinder(radius=args.radius, half_height=args.half_height)
    if args.primitive == "box":
        return Box(half_extents=(args.half_x, args.half_y, args.half_z))
    raise ValueError(f"Unsupported primitive: {args.primitive}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the Shadow Hand against a primitive object in viser.")
    parser.add_argument("--primitive", choices=["sphere", "cylinder", "box"], default="sphere")
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--half-height", type=float, default=0.08)
    parser.add_argument("--half-x", type=float, default=0.04)
    parser.add_argument("--half-y", type=float, default=0.04)
    parser.add_argument("--half-z", type=float, default=0.04)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contacts", nargs="*", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--duration", type=float, default=0.0, help="If >0, keep the process alive for that many seconds. Default 0 keeps it alive until Ctrl+C.")
    parser.add_argument("--palm-down", action="store_true", help="Flip the default wrist orientation by 180 degrees around the x-axis.")
    parser.add_argument(
        "--hide-penetration-spheres",
        action="store_true",
        help="Hide the Shadow Hand penetration spheres.",
    )
    args = parser.parse_args()

    hand_model = ShadowHandModel.create(device=args.device)
    primitive = build_primitive(args)
    joint_values = hand_model.default_joint_state(batch_size=1)
    contact_indices = None
    if args.contacts:
        contact_indices = torch.tensor([args.contacts], dtype=torch.long, device=hand_model.device)
    wrist_rotation = None
    if args.palm_down:
        wrist_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device).unsqueeze(0)

    server = publish_shadow_hand_primitive_viser(
        hand_model,
        primitive,
        joint_values,
        contact_indices=contact_indices,
        wrist_rotation=wrist_rotation,
        show_penetration_spheres=not args.hide_penetration_spheres,
        host=args.host,
        port=args.port,
    )
    print("Viser visualization ready.")
    print(f"Open this URL in your browser: http://localhost:{args.port}")
    if args.duration > 0:
        time.sleep(args.duration)
        server.stop()
        return
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
