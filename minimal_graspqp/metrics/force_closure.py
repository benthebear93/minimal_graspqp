from __future__ import annotations

import torch

from minimal_graspqp.metrics.wrench import build_wrench_matrix, friction_cone_edges
from minimal_graspqp.solvers import BoundedLeastSquaresQPSolver


class ForceClosureQP:
    def __init__(
        self,
        friction: float = 0.2,
        num_edges: int = 4,
        min_force: float = 0.0,
        max_force: float = 20.0,
        torque_weight: float = 5.0,
        svd_gain: float = 0.1,
        values_gain: float = 2.0,
        warm_start: bool = True,
    ):
        self.friction = friction
        self.num_edges = num_edges
        self.min_force = min_force
        self.max_force = max_force
        self.torque_weight = torque_weight
        self.svd_gain = svd_gain
        self.values_gain = values_gain
        self.warm_start = warm_start
        self.solver = BoundedLeastSquaresQPSolver()
        self._last_solution: torch.Tensor | None = None

    def evaluate(
        self,
        contact_points: torch.Tensor,
        contact_normals: torch.Tensor,
        cog: torch.Tensor,
        return_solution: bool = False,
    ):
        force_dirs = friction_cone_edges(contact_normals, friction=self.friction, num_edges=self.num_edges)
        wrench_matrix = build_wrench_matrix(contact_points, force_dirs, cog=cog, torque_weight=self.torque_weight)
        basis = -wrench_matrix.sum(dim=-1).unsqueeze(1)
        A = wrench_matrix.unsqueeze(1)
        b = basis
        init = 1.5
        if self.warm_start and self._last_solution is not None and self._last_solution.shape[0] == contact_points.shape[0]:
            init = self._last_solution
        residuals, solution = self.solver.solve(
            A,
            b,
            init=init,
            min_bound=self.min_force + 1.0,
            max_bound=self.max_force + 1.0,
            return_solution=True,
        )
        if self.warm_start:
            self._last_solution = solution.detach()
        svd_values = torch.linalg.svdvals(wrench_matrix)
        svd_scale = svd_values.prod(dim=-1).clamp_min(1e-12).pow(1.0 / wrench_matrix.shape[-2]).unsqueeze(-1)
        energy = self.values_gain * (residuals.mean(dim=-1) + 1e-2) * torch.exp(-self.svd_gain * svd_scale.mean(dim=-1))
        if return_solution:
            friction_cone_size = force_dirs.shape[-2] // contact_points.shape[-2]
            aggregated = solution.view(*solution.shape[:-1], -1, friction_cone_size).sum(dim=-1).squeeze(1)
            return energy, aggregated, wrench_matrix
        return energy
