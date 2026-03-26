import torch

from minimal_graspqp.objects import Box, Cylinder, Sphere


def test_sphere_signed_distance_and_normal():
    sphere = Sphere(radius=1.0)
    points = torch.tensor([[1.5, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    sdf = sphere.signed_distance(points)
    normals = sphere.normals(points)
    assert torch.allclose(sdf, torch.tensor([0.5, 0.0]))
    assert torch.allclose(normals[0], torch.tensor([1.0, 0.0, 0.0]))


def test_cylinder_signed_distance_and_normal():
    cylinder = Cylinder(radius=1.0, half_height=2.0)
    points = torch.tensor([[1.5, 0.0, 0.0], [0.0, 0.0, 2.5]], dtype=torch.float32)
    sdf = cylinder.signed_distance(points)
    normals = cylinder.normals(points)
    assert torch.allclose(sdf, torch.tensor([0.5, 0.5]), atol=1e-5)
    assert torch.allclose(normals[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(normals[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)


def test_box_signed_distance_and_normal():
    box = Box(half_extents=(1.0, 2.0, 3.0))
    points = torch.tensor([[1.5, 0.0, 0.0], [0.0, 2.5, 0.0]], dtype=torch.float32)
    sdf = box.signed_distance(points)
    normals = box.normals(points)
    assert torch.allclose(sdf, torch.tensor([0.5, 0.5]), atol=1e-5)
    assert torch.allclose(normals[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(normals[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)
