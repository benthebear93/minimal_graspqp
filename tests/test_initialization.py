import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import MeshObject
from minimal_graspqp.objects import Sphere
from minimal_graspqp.rotation import palm_down_rotation


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
    assert torch.all(distances >= 0.15 - 1e-6)
    assert torch.all(distances <= 0.20 + 1e-6)


def test_initialize_grasps_for_primitive_supports_palm_down_base_rotation():
    hand_model = ShadowHandModel.create(device="cpu")
    base_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device)
    grasp_state = initialize_grasps_for_primitive(
        hand_model,
        Sphere(radius=0.05),
        batch_size=4,
        max_wrist_angle=0.0,
        base_wrist_rotation=base_rotation,
    )
    assert grasp_state.wrist_rotation.shape == (4, 3, 3)
    det = torch.linalg.det(grasp_state.wrist_rotation)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_initialize_grasps_for_primitive_faces_object_when_unperturbed():
    hand_model = ShadowHandModel.create(device="cpu")
    grasp_state = initialize_grasps_for_primitive(
        hand_model,
        Sphere(radius=0.05),
        batch_size=8,
        max_wrist_angle=0.0,
    )
    wrist_to_center = -grasp_state.wrist_translation
    wrist_to_center = wrist_to_center / wrist_to_center.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    forward_world = grasp_state.wrist_rotation[:, :, 2]
    alignment = (forward_world * wrist_to_center).sum(dim=-1)
    assert torch.all(alignment > 0.95)


def test_initialize_grasps_for_primitive_supports_equalized_contact_pools():
    hand_model = ShadowHandModel.create(device="cpu", allowed_contact_links=["th", "ff", "mf"])
    mesh_object = MeshObject("/home/haegu/minimal_graspqp/assets/objects/test_tube.stl", scale=0.001)
    link_to_indices = {}
    for idx, link_name in enumerate(hand_model.metadata.contact_candidate_links):
        link_to_indices.setdefault(link_name, []).append(idx)
    contact_index_pools = [
        torch.tensor(link_to_indices["robot0_thdistal"], dtype=torch.long),
        torch.tensor(link_to_indices["robot0_thdistal"], dtype=torch.long),
        torch.tensor(link_to_indices["robot0_thdistal"], dtype=torch.long),
        torch.tensor(link_to_indices["robot0_ffdistal"], dtype=torch.long),
        torch.tensor(link_to_indices["robot0_ffdistal"], dtype=torch.long),
        torch.tensor(link_to_indices["robot0_ffdistal"], dtype=torch.long),
        torch.tensor(link_to_indices["robot0_mfdistal"], dtype=torch.long),
        torch.tensor(link_to_indices["robot0_mfdistal"], dtype=torch.long),
        torch.tensor(link_to_indices["robot0_mfdistal"], dtype=torch.long),
    ]
    grasp_state = initialize_grasps_for_primitive(
        hand_model,
        mesh_object,
        batch_size=4,
        num_contacts=9,
        contact_index_pools=contact_index_pools,
    )
    for column_id, pool in enumerate(contact_index_pools):
        assert torch.isin(grasp_state.contact_indices[:, column_id], pool).all()
