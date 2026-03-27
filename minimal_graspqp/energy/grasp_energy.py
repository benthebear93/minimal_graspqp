from __future__ import annotations

from typing import Any

import torch

from minimal_graspqp.metrics import ForceClosureQP


def compute_joint_limit_penalty(joint_values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    lower = lower.to(device=joint_values.device, dtype=joint_values.dtype)
    upper = upper.to(device=joint_values.device, dtype=joint_values.dtype)
    return (lower - joint_values).clamp_min(0.0).sum(dim=-1) + (joint_values - upper).clamp_min(0.0).sum(dim=-1)


def compute_self_penetration_energy(
    sphere_centers: torch.Tensor,
    sphere_radii: torch.Tensor,
    link_names: list[str],
) -> torch.Tensor:
    deltas = sphere_centers.unsqueeze(2) - sphere_centers.unsqueeze(1)
    distances = deltas.norm(dim=-1)
    thresholds = sphere_radii.unsqueeze(2) + sphere_radii.unsqueeze(1)
    overlap = (thresholds - distances).clamp_min(0.0)
    batch_size, num_spheres, _ = sphere_centers.shape
    mask = torch.ones((num_spheres, num_spheres), dtype=torch.bool, device=sphere_centers.device)
    mask.fill_diagonal_(False)
    for i in range(num_spheres):
        for j in range(num_spheres):
            if link_names[i] == link_names[j]:
                mask[i, j] = False
    return (overlap * mask.unsqueeze(0)).sum(dim=(1, 2)) * 0.5


def _surface_points_for_penetration(
    primitive: Any,
    num_samples: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cache = getattr(primitive, "_penetration_surface_cache", None)
    if cache is None:
        cache = {}
        setattr(primitive, "_penetration_surface_cache", cache)
    if num_samples not in cache:
        if not hasattr(primitive, "sample_surface"):
            raise ValueError("Object must provide sample_surface(...) for penetration evaluation.")
        points, _ = primitive.sample_surface(num_samples, dtype=torch.float32, device=torch.device("cpu"))
        cache[num_samples] = points.detach().cpu()
    return cache[num_samples].to(device=device, dtype=dtype)


def compute_grasp_energy(
    hand_model,
    primitive: Any,
    grasp_state,
    force_closure_metric: ForceClosureQP,
    w_fc: float = 1.0,
    w_dis: float = 100.0,
    w_pen: float = 100.0,
    w_spen: float = 10.0,
    w_joint: float = 1.0,
    num_penetration_samples: int = 256,
) -> dict[str, torch.Tensor]:
    contact_points = hand_model.contact_candidates_world(
        grasp_state.joint_values,
        indices=grasp_state.contact_indices,
        wrist_translation=grasp_state.wrist_translation,
        wrist_rotation=grasp_state.wrist_rotation,
    )
    signed_distance = primitive.signed_distance(contact_points)
    contact_normals = primitive.normals(contact_points)
    cog = torch.tensor(primitive.center, dtype=contact_points.dtype, device=contact_points.device).unsqueeze(0).expand(contact_points.shape[0], -1)

    e_dis = signed_distance.abs().sum(dim=-1)
    e_fc = force_closure_metric.evaluate(contact_points, contact_normals, cog)

    object_surface_points = _surface_points_for_penetration(
        primitive,
        num_samples=num_penetration_samples,
        dtype=contact_points.dtype,
        device=contact_points.device,
    ).unsqueeze(0).expand(contact_points.shape[0], -1, -1)
    e_pen = hand_model.cal_distance(
        object_surface_points,
        grasp_state.joint_values,
        wrist_translation=grasp_state.wrist_translation,
        wrist_rotation=grasp_state.wrist_rotation,
    ).clamp_min(0.0).sum(dim=-1)

    penetration_centers, penetration_radii, link_names = hand_model.penetration_spheres_world(
        grasp_state.joint_values,
        wrist_translation=grasp_state.wrist_translation,
        wrist_rotation=grasp_state.wrist_rotation,
    )
    e_spen = compute_self_penetration_energy(penetration_centers, penetration_radii, link_names)
    e_joint = compute_joint_limit_penalty(
        grasp_state.joint_values,
        hand_model.metadata.joint_lower,
        hand_model.metadata.joint_upper,
    )
    total = w_fc * e_fc + w_dis * e_dis + w_pen * e_pen + w_spen * e_spen + w_joint * e_joint

    return {
        "contact_points": contact_points,
        "contact_normals": contact_normals,
        "signed_distance": signed_distance,
        "E_fc": e_fc,
        "E_dis": e_dis,
        "E_pen": e_pen,
        "E_spen": e_spen,
        "E_joint": e_joint,
        "E_total": total,
    }
