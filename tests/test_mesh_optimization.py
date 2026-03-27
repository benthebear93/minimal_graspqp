from pathlib import Path

import torch

from minimal_graspqp.energy import compute_grasp_energy
from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.metrics import ForceClosureQP
from minimal_graspqp.objects import MeshObject
from minimal_graspqp.optim import MalaConfig, MalaOptimizer


MESH_PATH = Path("/home/haegu/minimal_graspqp/assets/objects/graspqp_test_sphere/coacd/decomposed.obj")


def test_mala_optimizer_runs_on_mesh_object():
    if not MESH_PATH.exists():
        return
    torch.manual_seed(0)
    hand_model = ShadowHandModel.create(device="cpu")
    mesh_object = MeshObject(MESH_PATH)
    initial_state = initialize_grasps_for_primitive(hand_model, mesh_object, batch_size=1, num_contacts=4)
    metric = ForceClosureQP(min_force=0.0, max_force=10.0)
    optimizer = MalaOptimizer(MalaConfig(num_steps=2, step_size=5e-3, noise_scale=5e-4, use_mala_star=False))

    initial_energy = compute_grasp_energy(hand_model, mesh_object, initial_state, metric)["E_total"].mean()
    final_state, history = optimizer.optimize(hand_model, mesh_object, initial_state, metric)
    final_energy = compute_grasp_energy(hand_model, mesh_object, final_state, metric)["E_total"].mean()

    assert len(history.energy_trace) == 3
    assert torch.isfinite(initial_energy)
    assert torch.isfinite(final_energy)
