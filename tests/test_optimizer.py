import torch

from minimal_graspqp.energy import compute_grasp_energy
from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.metrics import ForceClosureQP
from minimal_graspqp.objects import Sphere
from minimal_graspqp.optim import MalaConfig, MalaOptimizer


def test_mala_optimizer_reduces_mean_energy():
    torch.manual_seed(0)
    hand_model = ShadowHandModel.create(device="cpu")
    primitive = Sphere(radius=0.05)
    initial_state = initialize_grasps_for_primitive(hand_model, primitive, batch_size=1, num_contacts=4)
    metric = ForceClosureQP(min_force=0.0, max_force=10.0)
    optimizer = MalaOptimizer(MalaConfig(num_steps=1, step_size=5e-3, noise_scale=5e-4, use_mala_star=False))

    initial_energy = compute_grasp_energy(hand_model, primitive, initial_state, metric)["E_total"].mean()
    final_state, history = optimizer.optimize(hand_model, primitive, initial_state, metric)
    final_energy = compute_grasp_energy(hand_model, primitive, final_state, metric)["E_total"].mean()
    best_trace_energy = min(trace.mean().item() for trace in history.energy_trace)

    assert len(history.energy_trace) == 2
    assert history.accepted_trace[0].sum().item() >= 1
    assert best_trace_energy < initial_energy.item()
    assert final_energy.item() < initial_energy.item()


def test_mala_star_optimizer_runs_and_records_resets():
    torch.manual_seed(0)
    hand_model = ShadowHandModel.create(device="cpu")
    primitive = Sphere(radius=0.05)
    initial_state = initialize_grasps_for_primitive(hand_model, primitive, batch_size=1, num_contacts=4)
    metric = ForceClosureQP(min_force=0.0, max_force=10.0)
    optimizer = MalaOptimizer(
        MalaConfig(
            num_steps=1,
            step_size=5e-3,
            noise_scale=5e-4,
            use_mala_star=True,
            reset_interval=2,
            z_score_threshold=0.5,
        )
    )

    final_state, history = optimizer.optimize(hand_model, primitive, initial_state, metric)
    final_energy = compute_grasp_energy(hand_model, primitive, final_state, metric)["E_total"]
    initial_energy = compute_grasp_energy(hand_model, primitive, initial_state, metric)["E_total"]
    best_trace_energy = min(trace.mean().item() for trace in history.energy_trace)

    assert final_energy.shape == (1,)
    assert len(history.reset_trace) == 1
    assert all(mask.shape == (1,) for mask in history.reset_trace)
    assert history.accepted_trace[0].sum().item() >= 1
    assert best_trace_energy < initial_energy.mean().item()
    assert final_energy.mean().item() < initial_energy.mean().item()
