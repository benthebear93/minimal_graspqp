import pytest

pytest.importorskip("meshcat")

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Sphere
from minimal_graspqp.visualization import publish_shadow_hand_primitive_meshcat


def test_publish_shadow_hand_primitive_meshcat():
    hand_model = ShadowHandModel.create(device="cpu")
    joint_values = hand_model.default_joint_state(batch_size=1)
    vis = publish_shadow_hand_primitive_meshcat(hand_model, Sphere(radius=0.05), joint_values)
    assert vis is not None
