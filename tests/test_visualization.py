import pytest

plotly = pytest.importorskip("plotly.graph_objects")

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.objects import Sphere
from minimal_graspqp.visualization import create_shadow_hand_primitive_figure


def test_create_shadow_hand_primitive_figure():
    hand_model = ShadowHandModel.create(device="cpu")
    figure = create_shadow_hand_primitive_figure(hand_model, Sphere(radius=0.05))
    assert isinstance(figure, plotly.Figure)
    assert len(figure.data) >= 3

