import pytest

pytest.importorskip("viser")

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Sphere
from minimal_graspqp.visualization import publish_shadow_hand_primitive_viser


def test_publish_shadow_hand_primitive_viser():
    hand_model = ShadowHandModel.create(device="cpu")
    joint_values = hand_model.default_joint_state(batch_size=1)
    server = publish_shadow_hand_primitive_viser(hand_model, Sphere(radius=0.05), joint_values, port=8891)
    assert server is not None
    try:
        server.stop()
    except RuntimeError:
        pass
