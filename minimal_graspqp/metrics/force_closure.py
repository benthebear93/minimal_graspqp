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
    ):
        self.friction = friction
        self.num_edges = num_edges
        self.min_force = min_force
        self.max_force = max_force
        self.torque_weight = torque_weight
        self.svd_gain = svd_gain
        self.solver = BoundedLeastSquaresQPSolver()

    def _basis(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        eye = torch.eye(6, device=device, dtype=dtype)
        basis = torch.cat([eye, -eye], dim=0)
        return basis.unsqueeze(0).expand(batch_size, -1, -1)

    def evaluate(
        self,
        contact_points: torch.Tensor,
        contact_normals: torch.Tensor,
        cog: torch.Tensor,
        return_solution: bool = False,
    ):
        force_dirs = friction_cone_edges(contact_normals, friction=self.friction, num_edges=self.num_edges)
        wrench_matrix = build_wrench_matrix(contact_points, force_dirs, cog=cog, torque_weight=self.torque_weight)
        basis = self._basis(contact_points.shape[0], contact_points.device, contact_points.dtype)
        A = wrench_matrix.unsqueeze(1).expand(-1, basis.shape[1], -1, -1)
        b = basis
        residuals, solution = self.solver.solve(
            A,
            b,
            min_bound=self.min_force,
            max_bound=self.max_force,
            return_solution=True,
        )
        svd_values = torch.linalg.svdvals(wrench_matrix)
        svd_scale = svd_values.prod(dim=-1).clamp_min(1e-12).pow(1.0 / wrench_matrix.shape[-2])
        energy = residuals.mean(dim=-1) * torch.exp(-self.svd_gain * svd_scale)
        if return_solution:
            return energy, solution, wrench_matrix
        return energy
