import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import Sphere


def test_initialize_grasps_for_primitive_shapes_and_limits():
    hand_model = ShadowHandModel.create(device="cpu")
    grasp_state = initialize_grasps_for_primitive(hand_model, Sphere(radius=0.05), batch_size=8, num_contacts=4)
    assert grasp_state.joint_values.shape == (8, 24)
    assert grasp_state.wrist_translation.shape == (8, 3)
    assert grasp_state.wrist_rotation.shape == (8, 3, 3)
    assert grasp_state.contact_indices.shape == (8, 4)
    lower = hand_model.metadata.joint_lower
    upper = hand_model.metadata.joint_upper
    assert torch.all(grasp_state.joint_values >= lower.unsqueeze(0) - 1e-6)
    assert torch.all(grasp_state.joint_values <= upper.unsqueeze(0) + 1e-6)


def test_initialize_grasps_for_primitive_wrist_distance():
    hand_model = ShadowHandModel.create(device="cpu")
    sphere = Sphere(radius=0.05)
    grasp_state = initialize_grasps_for_primitive(
        hand_model,
        sphere,
        batch_size=16,
        distance_lower=0.10,
        distance_upper=0.15,
    )
    distances = grasp_state.wrist_translation.norm(dim=-1)
    assert torch.all(distances >= 0.10 - 1e-6)
    assert torch.all(distances <= 0.15 + 1e-6)
