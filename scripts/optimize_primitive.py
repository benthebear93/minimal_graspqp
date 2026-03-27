from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minimal_graspqp.energy import compute_grasp_energy
from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.metrics import ForceClosureQP
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.optim import MalaConfig, MalaOptimizer
from minimal_graspqp.rotation import palm_down_rotation


def build_object(args):
    if args.mesh_path:
        return MeshObject(Path(args.mesh_path), scale=args.mesh_scale)
    if args.object_code:
        if not args.object_root:
            raise ValueError("--object-root is required when --object-code is used.")
        return MeshObject.from_code(args.object_root, args.object_code, scale=args.mesh_scale)
    if args.primitive == "sphere":
        return Sphere(radius=args.radius)
    if args.primitive == "cylinder":
        return Cylinder(radius=args.radius, half_height=args.half_height)
    if args.primitive == "box":
        return Box(half_extents=(args.half_x, args.half_y, args.half_z))
    raise ValueError(f"Unsupported primitive: {args.primitive}")


def serialize_state(state):
    return {
        "joint_values": state.joint_values.detach().cpu(),
        "wrist_translation": state.wrist_translation.detach().cpu(),
        "wrist_rotation": state.wrist_rotation.detach().cpu(),
        "contact_indices": state.contact_indices.detach().cpu(),
    }


def object_metadata(args, obj):
    if isinstance(obj, MeshObject):
        if args.object_code:
            return {
                "type": "mesh",
                "object_code": args.object_code,
                "object_root": str(Path(args.object_root).resolve()) if args.object_root else None,
                "mesh_path": str(obj.mesh_path),
                "scale": float(args.mesh_scale),
                "center": list(obj.center),
            }
        return {
            "type": "mesh",
            "mesh_path": str(obj.mesh_path),
            "scale": float(args.mesh_scale),
            "center": list(obj.center),
        }
    if args.primitive == "sphere":
        return {"type": "sphere", "radius": args.radius, "center": [0.0, 0.0, 0.0]}
    if args.primitive == "cylinder":
        return {
            "type": "cylinder",
            "radius": args.radius,
            "half_height": args.half_height,
            "center": [0.0, 0.0, 0.0],
        }
    if args.primitive == "box":
        return {
            "type": "box",
            "half_extents": [args.half_x, args.half_y, args.half_z],
            "center": [0.0, 0.0, 0.0],
        }
    raise ValueError(f"Unsupported primitive: {args.primitive}")


def main():
    parser = argparse.ArgumentParser(description="Run minimal GraspQP optimization on a primitive or mesh object.")
    parser.add_argument("--primitive", choices=["sphere", "cylinder", "box"], default="sphere")
    parser.add_argument("--mesh-path", default=None, help="Optional direct path to a mesh object file.")
    parser.add_argument("--object-root", default=None, help="Optional root directory for object-code lookup.")
    parser.add_argument("--object-code", default=None, help="Optional object code resolved as <object-root>/<object-code>/coacd/*.obj.")
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Uniform scale applied when loading a mesh object.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-contacts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--half-height", type=float, default=0.08)
    parser.add_argument("--half-x", type=float, default=0.04)
    parser.add_argument("--half-y", type=float, default=0.04)
    parser.add_argument("--half-z", type=float, default=0.04)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mala-star", action="store_true")
    parser.add_argument("--step-size", type=float, default=5e-3)
    parser.add_argument("--noise-scale", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=18.0)
    parser.add_argument("--temperature-decay", type=float, default=0.95)
    parser.add_argument("--contact-switch-probability", type=float, default=0.0)
    parser.add_argument("--reset-interval", type=int, default=600)
    parser.add_argument("--z-score-threshold", type=float, default=1.0)
    parser.add_argument("--stepsize-period", type=int, default=50)
    parser.add_argument("--annealing-period", type=int, default=30)
    parser.add_argument("--mu", type=float, default=0.98)
    parser.add_argument("--output", default="outputs/primitive_optimization.pt")
    parser.add_argument("--palm-down", action="store_true", help="Bias initialization around a palm-down wrist orientation.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    hand_model = ShadowHandModel.create(device=args.device)
    primitive = build_object(args)
    base_wrist_rotation = None
    if args.palm_down:
        base_wrist_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device)
    initial_state = initialize_grasps_for_primitive(
        hand_model,
        primitive,
        batch_size=args.batch_size,
        num_contacts=args.num_contacts,
        base_wrist_rotation=base_wrist_rotation,
    )
    metric = ForceClosureQP(min_force=0.0, max_force=20.0)
    optimizer = MalaOptimizer(
        MalaConfig(
            num_steps=args.num_steps,
            step_size=args.step_size,
            noise_scale=args.noise_scale,
            temperature=args.temperature,
            temperature_decay=args.temperature_decay,
            stepsize_period=args.stepsize_period,
            annealing_period=args.annealing_period,
            mu=args.mu,
            contact_switch_probability=args.contact_switch_probability,
            reset_interval=args.reset_interval,
            z_score_threshold=args.z_score_threshold,
            use_mala_star=args.mala_star,
        )
    )

    initial_losses = compute_grasp_energy(hand_model, primitive, initial_state, metric)
    final_state, history = optimizer.optimize(hand_model, primitive, initial_state, metric)
    final_losses = compute_grasp_energy(hand_model, primitive, final_state, metric)

    output = {
        "primitive": object_metadata(args, primitive),
        "initial_state": serialize_state(initial_state),
        "final_state": serialize_state(final_state),
        "initial_energy": initial_losses["E_total"].detach().cpu(),
        "final_energy": final_losses["E_total"].detach().cpu(),
        "energy_trace": torch.stack(history.energy_trace).cpu(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Wrote optimization result to {output_path}")
    mean_initial = initial_losses["E_total"].mean().item()
    mean_final = final_losses["E_total"].mean().item()
    trace_means = [trace.mean().item() for trace in history.energy_trace]
    accepted_counts = [int(mask.sum().item()) for mask in history.accepted_trace]
    reset_counts = [int(mask.sum().item()) for mask in history.reset_trace]
    print(f"Mean initial energy: {mean_initial:.6f}")
    print(f"Mean final energy: {mean_final:.6f}")
    print(f"Best mean energy in trace: {min(trace_means):.6f}")
    print(f"Accepted per step: {accepted_counts}")
    if args.mala_star:
        print(f"Resets per step: {reset_counts}")


if __name__ == "__main__":
    main()
