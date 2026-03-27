import torch

from minimal_graspqp.energy import compute_grasp_energy, compute_joint_limit_penalty, compute_self_penetration_energy
from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.metrics import ForceClosureQP
from minimal_graspqp.objects import Sphere
from minimal_graspqp.state import GraspState


def test_compute_joint_limit_penalty_out_of_bounds():
    joint_values = torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.float32)
    lower = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    upper = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    penalty = compute_joint_limit_penalty(joint_values, lower, upper)
    assert torch.allclose(penalty, torch.tensor([2.0]))


def test_compute_self_penetration_energy_zero_for_separated_spheres():
    centers = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=torch.float32)
    radii = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    energy = compute_self_penetration_energy(centers, radii, ["a", "b"])
    assert torch.allclose(energy, torch.tensor([0.0]))


def test_compute_grasp_energy_returns_term_breakdown():
    hand_model = ShadowHandModel.create(device="cpu")
    primitive = Sphere(radius=0.05)
    state = initialize_grasps_for_primitive(hand_model, primitive, batch_size=2, num_contacts=4)
    metric = ForceClosureQP(min_force=0.0, max_force=10.0)
    losses = compute_grasp_energy(hand_model, primitive, state, metric)
    for key in ["E_fc", "E_dis", "E_pen", "E_spen", "E_joint", "E_total"]:
        assert key in losses
        assert losses[key].shape == (2,)
        assert torch.isfinite(losses[key]).all()


def test_compute_grasp_energy_penalizes_inside_translation():
    hand_model = ShadowHandModel.create(device="cpu")
    default_joint_values = hand_model.default_joint_state(batch_size=1)
    transforms = hand_model.forward_kinematics(default_joint_values)
    palm_vertices = hand_model._collision_meshes["robot0_palm"]["vertices"]
    palm_center_local = palm_vertices.mean(dim=0)
    palm_transform = transforms["robot0_palm"][0]
    palm_center_world = (
        palm_transform[:3, :3] @ palm_center_local.unsqueeze(-1)
    ).squeeze(-1) + palm_transform[:3, 3]
    primitive = Sphere(radius=0.03, center=tuple(float(x) for x in palm_center_world.tolist()))
    metric = ForceClosureQP(min_force=0.0, max_force=10.0)

    outside_state = GraspState(
        joint_values=default_joint_values.clone(),
        wrist_translation=torch.tensor([[0.40, 0.0, 0.0]], dtype=torch.float32),
        wrist_rotation=torch.eye(3).unsqueeze(0),
        contact_indices=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    )
    inside_state = GraspState(
        joint_values=default_joint_values.clone(),
        wrist_translation=torch.zeros((1, 3), dtype=torch.float32),
        wrist_rotation=torch.eye(3).unsqueeze(0),
        contact_indices=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    )
    outside_losses = compute_grasp_energy(hand_model, primitive, outside_state, metric)
    inside_losses = compute_grasp_energy(hand_model, primitive, inside_state, metric)
    assert inside_losses["E_pen"].item() > outside_losses["E_pen"].item()
