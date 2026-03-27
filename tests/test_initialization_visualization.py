import pytest

pytest.importorskip("viser")

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import Sphere
from minimal_graspqp.visualization import publish_initialization_viser


def test_publish_initialization_viser():
    hand_model = ShadowHandModel.create(device="cpu")
    primitive = Sphere(radius=0.05)
    grasp_state = initialize_grasps_for_primitive(hand_model, primitive, batch_size=3, num_contacts=4)
    server = publish_initialization_viser(hand_model, primitive, grasp_state, port=8892)
    assert server is not None
    try:
        server.stop()
    except RuntimeError:
        pass
