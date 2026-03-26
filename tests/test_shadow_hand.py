from pathlib import Path

import torch

from minimal_graspqp.assets import resolve_shadow_hand_asset_dir
from minimal_graspqp.hands import ShadowHandModel, load_shadow_hand_metadata


def test_resolve_shadow_hand_asset_dir():
    asset_dir = resolve_shadow_hand_asset_dir()
    assert asset_dir.exists()
    assert (asset_dir / "shadow_hand.urdf").exists()


def test_load_shadow_hand_metadata():
    metadata = load_shadow_hand_metadata()
    assert metadata.num_joints == 24
    assert metadata.default_joint_state.shape == (24,)
    assert metadata.contact_candidate_points.ndim == 2
    assert metadata.contact_candidate_points.shape[-1] == 3
    assert metadata.contact_candidate_normals.shape == metadata.contact_candidate_points.shape
    assert len(metadata.contact_candidate_links) == metadata.num_contact_candidates
    assert "robot0_ffdistal" in metadata.penetration_points
    assert metadata.num_contact_candidates == 80


def test_shadow_hand_forward_kinematics_and_contacts():
    model = ShadowHandModel.create(device="cpu")
    joint_values = model.default_joint_state(batch_size=2)
    transforms = model.forward_kinematics(joint_values)
    assert "robot0_palm" in transforms
    assert transforms["robot0_palm"].shape == (2, 4, 4)

    contacts = model.contact_candidates_world(joint_values)
    assert contacts.shape[0] == 2
    assert contacts.shape[1] == model.metadata.num_contact_candidates
    assert contacts.shape[2] == 3


def test_shadow_hand_selected_contact_candidates():
    model = ShadowHandModel.create(device="cpu")
    joint_values = model.default_joint_state(batch_size=1)
    indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    contacts = model.contact_candidates_world(joint_values, indices=indices)
    assert contacts.shape == (1, 4, 3)


def test_shadow_hand_contact_candidates_are_not_one_point_per_link():
    metadata = load_shadow_hand_metadata()
    counts = {}
    for link_name in metadata.contact_candidate_links:
        counts[link_name] = counts.get(link_name, 0) + 1
    assert counts["robot0_ffdistal"] == 8
    assert counts["robot0_ffmiddle"] == 4
    assert counts["robot0_thdistal"] == 8
