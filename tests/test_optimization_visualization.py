import pytest
import torch

plotly = pytest.importorskip("plotly.graph_objects")

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import Sphere
from minimal_graspqp.visualization import create_optimization_result_figure


def test_create_optimization_result_figure():
    hand_model = ShadowHandModel.create(device="cpu")
    primitive = Sphere(radius=0.05)
    initial_state = initialize_grasps_for_primitive(hand_model, primitive, batch_size=2, num_contacts=4)
    final_state = initial_state.clone()
    final_state.wrist_translation = final_state.wrist_translation + torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    figure = create_optimization_result_figure(hand_model, primitive, initial_state, final_state, sample_index=0)
    assert isinstance(figure, plotly.Figure)
    assert len(figure.data) > 4

