from __future__ import annotations

from dataclasses import dataclass

import torch

from minimal_graspqp.energy import compute_grasp_energy
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.state import GraspState


def _project_rotation_matrices(rotations: torch.Tensor) -> torch.Tensor:
    u, _, v_t = torch.linalg.svd(rotations)
    projected = u @ v_t
    det = torch.linalg.det(projected)
    needs_flip = det < 0
    if needs_flip.any():
        u = u.clone()
        u[needs_flip, :, -1] *= -1.0
        projected = u @ v_t
    return projected


def _clone_masked(dst: GraspState, src: GraspState, mask: torch.Tensor) -> GraspState:
    if not mask.any():
        return dst
    dst.joint_values[mask] = src.joint_values[mask]
    dst.wrist_translation[mask] = src.wrist_translation[mask]
    dst.wrist_rotation[mask] = src.wrist_rotation[mask]
    dst.contact_indices[mask] = src.contact_indices[mask]
    return dst


@dataclass
class MalaConfig:
    num_steps: int = 25
    step_size: float = 5e-3
    temperature: float = 18.0
    temperature_decay: float = 0.95
    stepsize_period: int = 50
    annealing_period: int = 30
    mu: float = 0.98
    joint_grad_scale: float = 1.0
    wrist_translation_grad_scale: float = 1.0
    wrist_rotation_grad_scale: float = 1.0
    noise_scale: float = 1e-3
    contact_switch_probability: float = 0.0
    reset_interval: int = 600
    z_score_threshold: float = 1.0
    use_mala_star: bool = False


@dataclass
class OptimizationHistory:
    energy_trace: list[torch.Tensor]
    accepted_trace: list[torch.Tensor]
    reset_trace: list[torch.Tensor]


class MalaOptimizer:
    def __init__(self, config: MalaConfig):
        self.config = config
        self._ema_joint = None
        self._ema_translation = None
        self._ema_rotation = None
        self._step = None

    def _energy(self, hand_model, primitive, state, metric):
        return compute_grasp_energy(hand_model, primitive, state, metric)["E_total"]

    def _make_differentiable_state(self, state: GraspState) -> GraspState:
        return GraspState(
            joint_values=state.joint_values.detach().clone().requires_grad_(True),
            wrist_translation=state.wrist_translation.detach().clone().requires_grad_(True),
            wrist_rotation=state.wrist_rotation.detach().clone().requires_grad_(True),
            contact_indices=state.contact_indices.clone(),
        )

    def _ensure_optimizer_buffers(self, state: GraspState):
        batch_size = state.batch_size
        if self._step is None or self._step.shape[0] != batch_size:
            device = state.joint_values.device
            self._step = torch.zeros(batch_size, dtype=torch.long, device=device)
            self._ema_joint = torch.zeros_like(state.joint_values)
            self._ema_translation = torch.zeros_like(state.wrist_translation)
            self._ema_rotation = torch.zeros_like(state.wrist_rotation)

    def _current_step_size(self) -> torch.Tensor:
        return self.config.step_size * self.config.temperature_decay ** torch.div(
            self._step,
            self.config.stepsize_period,
            rounding_mode="floor",
        )

    def _current_temperature(self) -> torch.Tensor:
        return self.config.temperature * self.config.temperature_decay ** torch.div(
            self._step,
            self.config.annealing_period,
            rounding_mode="floor",
        )

    def _propose(self, hand_model, state: GraspState) -> GraspState:
        grad_joint = state.joint_values.grad
        grad_translation = state.wrist_translation.grad
        grad_rotation = state.wrist_rotation.grad

        self._ema_joint = self.config.mu * (grad_joint.pow(2).mean(dim=0, keepdim=True)) + (1.0 - self.config.mu) * self._ema_joint
        self._ema_translation = self.config.mu * (
            grad_translation.pow(2).mean(dim=0, keepdim=True)
        ) + (1.0 - self.config.mu) * self._ema_translation
        self._ema_rotation = self.config.mu * (grad_rotation.pow(2).mean(dim=0, keepdim=True)) + (1.0 - self.config.mu) * self._ema_rotation

        step_size = self._current_step_size().unsqueeze(-1)
        joint_values = state.joint_values - (
            step_size * self.config.joint_grad_scale * grad_joint / (torch.sqrt(self._ema_joint) + 1e-6)
        )
        joint_values = hand_model.clamp_to_limits(joint_values)

        wrist_translation = state.wrist_translation - (
            step_size * self.config.wrist_translation_grad_scale * grad_translation / (torch.sqrt(self._ema_translation) + 1e-6)
        )
        wrist_translation = wrist_translation + self.config.noise_scale * torch.randn_like(wrist_translation)

        step_size_rot = self._current_step_size().view(-1, 1, 1)
        wrist_rotation = state.wrist_rotation - (
            step_size_rot * self.config.wrist_rotation_grad_scale * grad_rotation / (torch.sqrt(self._ema_rotation) + 1e-6)
        )
        wrist_rotation = wrist_rotation + self.config.noise_scale * torch.randn_like(wrist_rotation)
        wrist_rotation = _project_rotation_matrices(wrist_rotation)

        contact_indices = state.contact_indices.clone()
        switch_mask = (
            torch.rand(contact_indices.shape, device=contact_indices.device) < self.config.contact_switch_probability
        )
        if switch_mask.any():
            num_candidates = hand_model.metadata.num_contact_candidates
            contact_indices[switch_mask] = torch.randint(0, num_candidates, (int(switch_mask.sum().item()),), device=contact_indices.device)

        return GraspState(
            joint_values=joint_values.detach(),
            wrist_translation=wrist_translation.detach(),
            wrist_rotation=wrist_rotation.detach(),
            contact_indices=contact_indices.detach(),
        )

    def optimize(self, hand_model, primitive, initial_state: GraspState, metric) -> tuple[GraspState, OptimizationHistory]:
        current_state = initial_state.clone().detached()
        self._ensure_optimizer_buffers(current_state)
        current_energy = self._energy(hand_model, primitive, current_state, metric).detach()

        history = OptimizationHistory(energy_trace=[current_energy.clone()], accepted_trace=[], reset_trace=[])
        best_state = current_state.clone()
        best_energy = current_energy.clone()

        for step in range(self.config.num_steps):
            diff_state = self._make_differentiable_state(current_state)
            losses = compute_grasp_energy(hand_model, primitive, diff_state, metric)
            losses["E_total"].sum().backward()
            proposal = self._propose(hand_model, diff_state)
            proposal_energy = self._energy(hand_model, primitive, proposal, metric).detach()

            temperature_vec = self._current_temperature().to(dtype=current_energy.dtype)
            z_score = None
            if self.config.use_mala_star:
                mean = current_energy.mean()
                std = current_energy.std(unbiased=False).clamp_min(1e-6)
                z_score = (current_energy - mean) / std
                temperature_vec = temperature_vec * (1.0 + 0.5 * (1.0 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, device=z_score.device)))))

            delta = proposal_energy - current_energy
            accept_prob = torch.exp((-delta / temperature_vec).clamp(max=40.0)).clamp(max=1.0)
            accept = (delta <= 0.0) | (torch.rand_like(accept_prob) < accept_prob)

            next_state = current_state.clone()
            next_state = _clone_masked(next_state, proposal, accept)
            next_energy = current_energy.clone()
            next_energy[accept] = proposal_energy[accept]

            reset_mask = torch.zeros_like(accept)
            if (
                self.config.use_mala_star
                and self.config.reset_interval is not None
                and self.config.reset_interval > 0
                and (step + 1) % self.config.reset_interval == 0
                and step < max(0, self.config.num_steps - 2 * self.config.reset_interval)
            ):
                if z_score is not None:
                    reset_mask = z_score > self.config.z_score_threshold
                if reset_mask.any():
                    reinit_state = initialize_grasps_for_primitive(
                        hand_model,
                        primitive,
                        batch_size=reset_mask.sum().item(),
                        num_contacts=current_state.contact_indices.shape[1],
                    )
                    next_state.joint_values[reset_mask] = reinit_state.joint_values
                    next_state.wrist_translation[reset_mask] = reinit_state.wrist_translation
                    next_state.wrist_rotation[reset_mask] = reinit_state.wrist_rotation
                    next_state.contact_indices[reset_mask] = reinit_state.contact_indices
                    next_energy[reset_mask] = self._energy(hand_model, primitive, next_state, metric)[reset_mask].detach()
                    self._step[reset_mask] = 0
                    self._ema_joint[reset_mask] = 0
                    self._ema_translation[reset_mask] = 0
                    self._ema_rotation[reset_mask] = 0

            improved = next_energy < best_energy
            if improved.any():
                best_state = _clone_masked(best_state, next_state, improved)
                best_energy[improved] = next_energy[improved]

            current_state = next_state
            current_energy = next_energy
            self._step = self._step + 1
            history.energy_trace.append(current_energy.clone())
            history.accepted_trace.append(accept.clone())
            history.reset_trace.append(reset_mask.clone())

        return best_state, history
