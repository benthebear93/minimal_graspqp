from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_shadow_hand_dynamic_grasp import (  # noqa: E402
    DEFAULT_JOINT_ORDER,
    LOCAL_MUJOCO_HAND_XML,
    _build_external_mujoco_hand_scene,
    _launch_external_dynamic_viewer,
    _run_external_dynamic_smoke,
)

FINGER_CURL_JOINTS = [
    "robot0_FFJ2",
    "robot0_FFJ1",
    "robot0_FFJ0",
    "robot0_MFJ2",
    "robot0_MFJ1",
    "robot0_MFJ0",
    "robot0_RFJ2",
    "robot0_RFJ1",
    "robot0_RFJ0",
    "robot0_LFJ2",
    "robot0_LFJ1",
    "robot0_LFJ0",
    "robot0_THJ3",
    "robot0_THJ1",
]


FINGER_CLOSE_DIRECTIONS = {
    "robot0_FFJ2": 1.0,
    "robot0_FFJ1": 1.0,
    "robot0_FFJ0": 1.0,
    "robot0_MFJ2": 1.0,
    "robot0_MFJ1": 1.0,
    "robot0_MFJ0": 1.0,
    "robot0_RFJ2": 1.0,
    "robot0_RFJ1": 1.0,
    "robot0_RFJ0": 1.0,
    "robot0_LFJ2": 1.0,
    "robot0_LFJ1": 1.0,
    "robot0_LFJ0": 1.0,
    "robot0_THJ3": 1.0,
    "robot0_THJ1": -1.0,
    "robot0_THJ0": -1.0,
}


def _parse_samples(raw: str | None, batch_size: int, max_samples: int | None) -> list[int]:
    if raw:
        samples = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                start, end = part.split(":", maxsplit=1)
                samples.extend(range(int(start), int(end)))
            else:
                samples.append(int(part))
    else:
        samples = list(range(batch_size))
    samples = [sample for sample in samples if 0 <= sample < batch_size]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def _subset_state(state: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices].clone() for key, value in state.items()}


def _save_success_payload(payload: dict[str, Any], selected: list[int], rows: list[dict[str, Any]], output: Path) -> None:
    indices = torch.tensor(selected, dtype=torch.long)
    out = dict(payload)
    out["initial_state"] = _subset_state(payload["initial_state"], indices)
    out["final_state"] = _subset_state(payload["final_state"], indices)
    for key in ("initial_energy", "final_energy"):
        if isinstance(payload.get(key), torch.Tensor):
            out[key] = payload[key][indices].clone()
    if isinstance(payload.get("energy_trace"), torch.Tensor):
        trace = payload["energy_trace"]
        out["energy_trace"] = trace[:, indices].clone() if trace.ndim >= 2 else trace.clone()
    original = payload.get("original_indices")
    if isinstance(original, torch.Tensor):
        out["original_indices"] = original[indices].clone()
    else:
        out["original_indices"] = indices.clone()
    out["mujoco_vendored_hand_eval"] = rows
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)


def _row_success(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.require_thumb_contact and row.get("final_th_force", 0.0) < args.min_thumb_force:
        return False
    if row.get("min_contact_force", 0.0) < args.min_contact_force:
        return False
    if row.get("min_distal_contact_links", 0.0) < args.min_horizon_distal_contact_links:
        return False
    if args.fail_on_contact_loss and row.get("first_zero_contact_step", -1.0) >= 0.0:
        return False
    return (
        row["z_drop"] <= args.max_z_drop
        and row["displacement_norm"] <= args.max_displacement
        and row["final_object_contacts"] >= args.min_final_contacts
        and row["final_contact_force"] >= args.min_final_contact_force
        and row["final_distal_contact_links"] >= args.min_distal_contact_links
        and row["curl_sum"] >= args.min_curl_sum
    )


def _curl_sum(joint_values: np.ndarray) -> float:
    total = 0.0
    for joint_name in FINGER_CURL_JOINTS:
        value = float(joint_values[DEFAULT_JOINT_ORDER.index(joint_name)])
        if joint_name in {"robot0_THJ1"}:
            value = -value
        total += max(value, 0.0)
    return total


def _pressed_targets(joint_values: np.ndarray, offset: float) -> np.ndarray:
    if offset <= 0.0:
        return joint_values
    targets = joint_values.copy()
    for joint_name, direction in FINGER_CLOSE_DIRECTIONS.items():
        joint_id = DEFAULT_JOINT_ORDER.index(joint_name)
        targets[joint_id] = targets[joint_id] + direction * offset
    return targets


def _thumb_pressed_targets(joint_values: np.ndarray, offset: float) -> np.ndarray:
    if offset <= 0.0:
        return joint_values
    targets = joint_values.copy()
    for joint_name in ("robot0_THJ3", "robot0_THJ1", "robot0_THJ0"):
        direction = FINGER_CLOSE_DIRECTIONS[joint_name]
        joint_id = DEFAULT_JOINT_ORDER.index(joint_name)
        targets[joint_id] = targets[joint_id] + direction * offset
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved GraspQP final poses using the vendored MuJoCo Shadow Hand asset."
    )
    parser.add_argument("--input", required=True, help="GraspQP optimization result .pt file.")
    parser.add_argument("--out-dir", default="outputs/mujoco_vendored_hand_eval")
    parser.add_argument("--output", default=None, help="Filtered successful grasp payload. Default: <out-dir>/sim_valid_grasps.pt")
    parser.add_argument("--samples", default=None, help="Comma list or start:end range. Default evaluates all candidates.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mesh-path", default=None, help="Optional object mesh path override for legacy result files.")
    parser.add_argument("--mujoco-hand-xml", default=str(LOCAL_MUJOCO_HAND_XML))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--preload-steps", type=int, default=200)
    parser.add_argument("--free-object-during-preload", action="store_true")
    parser.add_argument("--gravity-during-preload", action="store_true")
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--friction", type=float, nargs=3, default=(5.0, 0.5, 0.05), metavar=("SLIDE", "TORSION", "ROLL"))
    parser.add_argument("--condim", type=int, default=6, choices=[1, 3, 4, 6])
    parser.add_argument("--object-mass", type=float, default=0.05)
    parser.add_argument("--contact-margin", type=float, default=0.0, help="MuJoCo geom contact margin. Use 0 for strict visual/collision contact.")
    parser.add_argument("--show-collision-geoms", action="store_true", help="Render hand collision geoms as translucent blue geoms in generated viewer scenes.")
    parser.add_argument("--show-contact-points", action="store_true", help="Render MuJoCo object contact points as red spheres in the viewer.")
    parser.add_argument("--convex-object", action="store_true", help="Use convex hull object collision. Default uses full object mesh.")
    parser.add_argument("--floor-z", type=float, default=-0.05)
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--floor-size", type=float, default=0.25)
    parser.add_argument("--max-z-drop", type=float, default=0.02)
    parser.add_argument("--max-displacement", type=float, default=0.03)
    parser.add_argument("--min-final-contacts", type=int, default=2)
    parser.add_argument("--min-final-contact-force", type=float, default=0.5)
    parser.add_argument("--min-distal-contact-links", type=int, default=3)
    parser.add_argument("--min-curl-sum", type=float, default=5.0)
    parser.add_argument("--min-contact-force", type=float, default=0.0, help="Minimum contact force required at every simulated step.")
    parser.add_argument("--min-horizon-distal-contact-links", type=int, default=0, help="Minimum distal contact links required at every simulated step.")
    parser.add_argument("--fail-on-contact-loss", action="store_true", help="Fail if the object loses all contacts during the simulated horizon.")
    parser.add_argument("--press-close-offset", type=float, default=0.0, help="Add radians along closing directions before MuJoCo evaluation.")
    parser.add_argument("--thumb-close-offset", type=float, default=0.0, help="Additional thumb-only closing offset in radians.")
    parser.add_argument("--require-thumb-contact", action="store_true", help="Require final thumb-object force to pass.")
    parser.add_argument("--min-thumb-force", type=float, default=0.1)
    parser.add_argument("--actuator-kp", type=float, default=None, help="Override all native MuJoCo position actuator kp values.")
    parser.add_argument("--actuator-force", type=float, default=None, help="Override all native MuJoCo position actuator force ranges to +/- this value.")
    parser.add_argument("--thumb-actuator-kp", type=float, default=None, help="Override MuJoCo thumb position actuator kp values.")
    parser.add_argument("--thumb-actuator-force", type=float, default=None, help="Override MuJoCo thumb position actuator force ranges to +/- this value.")
    parser.add_argument("--joint-damping", type=float, default=None, help="Override all native MuJoCo joint damping values.")
    parser.add_argument("--hold-hand-qpos", action="store_true", help="Reset hand joint qpos/qvel to the target pose every step while the object is free.")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer for each evaluated sample after metrics are computed.")
    parser.add_argument("--viewer-max-steps", type=int, default=0, help="Maximum viewer simulation steps. 0 keeps the viewer open until closed.")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    batch_size = int(payload["final_state"]["joint_values"].shape[0])
    samples = _parse_samples(args.samples, batch_size, args.max_samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    success_indices: list[int] = []
    for rank, sample in enumerate(samples):
        joint_values = payload["final_state"]["joint_values"][sample].detach().cpu().numpy().astype(np.float64)
        joint_values = _pressed_targets(joint_values, args.press_close_offset)
        joint_values = _thumb_pressed_targets(joint_values, args.thumb_close_offset)
        sample_dir = out_dir / f"sample_{sample:04d}"
        xml_path, object_center, _ = _build_external_mujoco_hand_scene(
            payload,
            sample,
            sample_dir,
            hand_xml=args.mujoco_hand_xml,
            mesh_path=args.mesh_path,
            hand_joint_values=joint_values,
            object_position=None,
            timestep=args.timestep,
            gravity=(0.0, 0.0, args.gravity_z),
            friction=tuple(float(x) for x in args.friction),
            condim=args.condim,
            object_mass=args.object_mass,
            object_convex_hull=bool(args.convex_object),
            floor_z=None if args.no_floor else args.floor_z,
            floor_size=args.floor_size,
            actuator_kp=args.actuator_kp,
            actuator_force=args.actuator_force,
            thumb_actuator_kp=args.thumb_actuator_kp,
            thumb_actuator_force=args.thumb_actuator_force,
            joint_damping=args.joint_damping,
            contact_margin=args.contact_margin,
            show_collision_geoms=args.show_collision_geoms,
        )
        metrics = _run_external_dynamic_smoke(
            xml_path,
            object_center,
            joint_values,
            args.steps,
            preload_steps=args.preload_steps,
            freeze_object_during_preload=not args.free_object_during_preload,
            zero_gravity_during_preload=not args.gravity_during_preload,
            hold_hand_qpos=args.hold_hand_qpos,
        )
        row: dict[str, Any] = {
            "rank": rank,
            "sample": sample,
            "source_sample": int(payload.get("original_indices", torch.arange(batch_size))[sample]),
            "final_energy": float(payload["final_energy"][sample]) if isinstance(payload.get("final_energy"), torch.Tensor) else float("nan"),
            "curl_sum": _curl_sum(joint_values),
            **metrics,
            "scene_xml": str(xml_path),
        }
        row["success"] = _row_success(row, args)
        rows.append(row)
        if row["success"]:
            success_indices.append(sample)
        print(
            f"{rank:03d} sample={sample} success={row['success']} "
            f"z_drop={row['z_drop']:.5f} disp={row['displacement_norm']:.5f} "
            f"contacts={row['final_object_contacts']:.0f} force={row['final_contact_force']:.3f}"
        )
        if args.viewer:
            print(f"Opening MuJoCo viewer for sample={sample}: {xml_path}", flush=True)
            _launch_external_dynamic_viewer(
                xml_path,
                object_center,
                joint_values,
                preload_steps=args.preload_steps,
                freeze_object_during_preload=not args.free_object_during_preload,
                zero_gravity_during_preload=not args.gravity_during_preload,
                max_viewer_steps=args.viewer_max_steps,
                show_contact_points=args.show_contact_points,
                hold_hand_qpos=args.hold_hand_qpos,
            )

    csv_path = out_dir / "mujoco_vendored_hand_eval.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["sample"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output = Path(args.output) if args.output else out_dir / "sim_valid_grasps.pt"
    if success_indices:
        _save_success_payload(payload, success_indices, rows, output)
    print(f"Wrote metrics to {csv_path}")
    print(f"Successful samples: {success_indices}")
    if success_indices:
        print(f"Wrote successful grasp payload to {output}")


if __name__ == "__main__":
    main()
