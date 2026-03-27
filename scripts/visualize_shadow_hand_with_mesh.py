from __future__ import annotations

import argparse
import time
from pathlib import Path

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import MeshObject
from minimal_graspqp.rotation import palm_down_rotation
from minimal_graspqp.visualization import publish_initialization_viser


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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--duration", type=float, default=0.0)
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
    server = publish_initialization_viser(
        hand_model,
        mesh_object,
        grasp_state,
        host=args.host,
        port=args.port,
    )
    print("Viser mesh initialization visualization ready.")
    print(f"Open this URL in your browser: http://localhost:{args.port}")
    if args.duration > 0:
        time.sleep(args.duration)
        server.stop()
        return
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
