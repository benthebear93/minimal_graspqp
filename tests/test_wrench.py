import torch

from minimal_graspqp.metrics import build_wrench_matrix, friction_cone_edges


def test_friction_cone_edges_shape_and_norm():
    normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    edges = friction_cone_edges(normals, friction=0.5, num_edges=4)
    assert edges.shape == (1, 8, 3)
    assert torch.allclose(edges.norm(dim=-1), torch.full_like(edges[..., 0], 0.25), atol=1e-5)


def test_build_wrench_matrix_shape():
    contact_points = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float32)
    force_dirs = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    cog = torch.zeros((1, 3), dtype=torch.float32)
    wrench = build_wrench_matrix(contact_points, force_dirs, cog, torque_weight=2.0)
    assert wrench.shape == (1, 6, 4)
