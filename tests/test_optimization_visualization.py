import pytest
import torch

pytest.importorskip("viser")

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import Sphere
from minimal_graspqp.visualization import publish_optimization_result_viser


def test_publish_optimization_result_viser():
    hand_model = ShadowHandModel.create(device="cpu")
    primitive = Sphere(radius=0.05)
    initial_state = initialize_grasps_for_primitive(hand_model, primitive, batch_size=2, num_contacts=4)
    final_state = initial_state.clone()
    final_state.wrist_translation = final_state.wrist_translation + torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    server = publish_optimization_result_viser(hand_model, primitive, initial_state, final_state, sample_index=0, port=8893)
    assert server is not None
    try:
        server.stop()
    except RuntimeError:
        pass
