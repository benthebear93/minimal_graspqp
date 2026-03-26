import pytest

plotly = pytest.importorskip("plotly.graph_objects")

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import Sphere
from minimal_graspqp.visualization import create_initialization_figure


def test_create_initialization_figure():
    hand_model = ShadowHandModel.create(device="cpu")
    primitive = Sphere(radius=0.05)
    grasp_state = initialize_grasps_for_primitive(hand_model, primitive, batch_size=3, num_contacts=4)
    figure = create_initialization_figure(hand_model, primitive, grasp_state, max_samples=3)
    assert isinstance(figure, plotly.Figure)
    assert len(figure.data) > 3
