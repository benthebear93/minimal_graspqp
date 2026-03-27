from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from minimal_graspqp.assets import resolve_shadow_hand_asset_dir
from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Sphere
from minimal_graspqp.rotation import palm_down_rotation
from minimal_graspqp.visualization import publish_shadow_hand_primitive_viser


def _load_overrides(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_overrides(path: Path, overrides: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")


def _print_index_summary(hand_model: ShadowHandModel, indices: list[int] | None = None) -> None:
    points = hand_model.metadata.contact_candidate_points.detach().cpu()
    links = hand_model.metadata.contact_candidate_links
    selected = indices if indices is not None else list(range(points.shape[0]))
    for idx in selected:
        point = points[idx].tolist()
        print(f"{idx:3d}  {links[idx]:20s}  local=({point[0]: .5f}, {point[1]: .5f}, {point[2]: .5f})")


def main():
    parser = argparse.ArgumentParser(description="Edit Shadow Hand contact candidate positions with viser preview.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index", type=int, help="Global contact candidate index to inspect or edit.")
    parser.add_argument("--set", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Replace the selected local point.")
    parser.add_argument("--delta", nargs=3, type=float, metavar=("DX", "DY", "DZ"), help="Offset the selected local point.")
    parser.add_argument(
        "--override-file",
        default="outputs/shadow_hand_contact_overrides.json",
        help="JSON file storing per-index local point overrides.",
    )
    parser.add_argument("--list", action="store_true", help="Print contact candidate indices and local coordinates.")
    parser.add_argument("--show", action="store_true", help="Open a viser preview.")
    parser.add_argument("--duration", type=float, default=0.0, help="Keep viser alive for N seconds. Default 0 waits until Ctrl+C.")
    parser.add_argument("--palm-down", action="store_true")
    parser.add_argument("--fingertips-only", action="store_true")
    parser.add_argument("--hide-penetration-spheres", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    override_path = Path(args.override_file)
    overrides = _load_overrides(override_path)

    base_model = ShadowHandModel.create(device=args.device, fingertips_only=args.fingertips_only)
    num_candidates = base_model.metadata.num_contact_candidates

    if args.index is not None and not (0 <= args.index < num_candidates):
        raise IndexError(f"--index must be in [0, {num_candidates - 1}]")

    if args.index is not None and (args.set is not None or args.delta is not None):
        point = base_model.metadata.contact_candidate_points[args.index].detach().cpu().tolist()
        if str(args.index) in overrides:
            point = overrides[str(args.index)]
        if args.set is not None:
            point = [float(v) for v in args.set]
        if args.delta is not None:
            point = [float(p + d) for p, d in zip(point, args.delta)]
        overrides[str(args.index)] = point
        _save_overrides(override_path, overrides)
        print(f"Updated override {args.index} -> {point}")

    hand_model = ShadowHandModel.create(
        device=args.device,
        fingertips_only=args.fingertips_only,
        contact_points_override_path=override_path if override_path.exists() else None,
    )

    print(f"Shadow Hand assets: {resolve_shadow_hand_asset_dir()}")
    print(f"Override file: {override_path.resolve()}")
    if args.index is not None:
        _print_index_summary(hand_model, [args.index])
    elif args.list:
        _print_index_summary(hand_model)

    if args.show:
        wrist_rotation = None
        if args.palm_down:
            wrist_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device).unsqueeze(0)
        server = publish_shadow_hand_primitive_viser(
            hand_model,
            Sphere(radius=0.05),
            hand_model.default_joint_state(batch_size=1),
            highlight_contact_indices=[args.index] if args.index is not None else None,
            wrist_rotation=wrist_rotation,
            show_penetration_spheres=not args.hide_penetration_spheres,
            host=args.host,
            port=args.port,
        )
        print("Viser contact preview ready.")
        print(f"Open this URL in your browser: http://localhost:{args.port}")
        if args.duration > 0:
            time.sleep(args.duration)
            server.stop()
            return
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
