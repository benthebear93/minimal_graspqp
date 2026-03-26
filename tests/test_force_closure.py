import torch

from minimal_graspqp.metrics import ForceClosureQP


def test_force_closure_energy_shape_and_solution():
    metric = ForceClosureQP(friction=0.2, num_edges=4, min_force=0.0, max_force=10.0, torque_weight=1.0)
    contact_points = torch.tensor(
        [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
        dtype=torch.float32,
    )
    contact_normals = -torch.nn.functional.normalize(contact_points, dim=-1)
    cog = torch.zeros((1, 3), dtype=torch.float32)
    energy, solution, wrench = metric.evaluate(contact_points, contact_normals, cog, return_solution=True)
    assert energy.shape == (1,)
    assert solution.shape[0] == 1
    assert wrench.shape[1] == 6


def test_force_closure_prefers_more_diverse_contacts():
    metric = ForceClosureQP(friction=0.2, num_edges=4, min_force=0.0, max_force=10.0, torque_weight=1.0)
    good_contacts = torch.tensor(
        [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
        dtype=torch.float32,
    )
    bad_contacts = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.9, 0.0, 0.0], [1.2, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    good_normals = -torch.nn.functional.normalize(good_contacts, dim=-1)
    bad_normals = -torch.nn.functional.normalize(bad_contacts, dim=-1)
    cog = torch.zeros((1, 3), dtype=torch.float32)
    good_energy = metric.evaluate(good_contacts, good_normals, cog)
    bad_energy = metric.evaluate(bad_contacts, bad_normals, cog)
    assert good_energy.item() < bad_energy.item()
