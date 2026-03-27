from __future__ import annotations

import torch
from qpth.qp import QPFunction


class BoundedLeastSquaresQPSolver:
    """Solve batched bounded least-squares problems with qpth."""

    def __init__(self, eps: float = 5e-2, max_iter: int = 12):
        self.qp = QPFunction(verbose=False, eps=eps, maxIter=max_iter)

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        init: torch.Tensor | float | int | None = None,
        min_bound: float = 1.0,
        max_bound: float = 20.0,
        return_solution: bool = False,
    ):
        if A.ndim < 3:
            raise ValueError("A must have shape (..., output_dim, num_vars).")
        if b.shape != A.shape[:-1]:
            raise ValueError("b must have shape matching A without the last dimension.")

        leading_shape = A.shape[:-2]
        output_dim = A.shape[-2]
        num_vars = A.shape[-1]

        A_flat = A.reshape(-1, output_dim, num_vars)
        b_flat = b.reshape(-1, output_dim)
        device = A.device
        dtype = A.dtype

        if init is None:
            init = torch.ones((A_flat.shape[0], num_vars), device=device, dtype=dtype)
        elif isinstance(init, (int, float)):
            init = torch.full((A_flat.shape[0], num_vars), float(init), device=device, dtype=dtype)
        else:
            if init.ndim == len(leading_shape) + 1:
                init = init.reshape(-1, num_vars)
            init = init.to(device=device, dtype=dtype).clone()
            if init.shape[0] != A_flat.shape[0]:
                init = torch.full((A_flat.shape[0], num_vars), 1.5, device=device, dtype=dtype)
        init = init.clamp(min_bound + 1e-6, max_bound - 1e-6)

        eye = torch.eye(num_vars, device=device, dtype=dtype).unsqueeze(0)
        q = A_flat.transpose(-1, -2) @ A_flat + eye * 1e-4
        p = -(A_flat.transpose(-1, -2) @ b_flat.unsqueeze(-1)).squeeze(-1)
        G = torch.cat([eye.expand(A_flat.shape[0], -1, -1), -eye.expand(A_flat.shape[0], -1, -1)], dim=-2)
        h = torch.cat(
            [
                torch.full((A_flat.shape[0], num_vars), max_bound, device=device, dtype=dtype),
                torch.full((A_flat.shape[0], num_vars), -min_bound, device=device, dtype=dtype),
            ],
            dim=-1,
        )
        equality_matrix = torch.empty((A_flat.shape[0], 0, num_vars), device=device, dtype=dtype)
        equality_rhs = torch.empty((A_flat.shape[0], 0), device=device, dtype=dtype)
        del init
        solution = self.qp(q, p, G, h, equality_matrix, equality_rhs)
        residual = 0.5 * ((A_flat @ solution.unsqueeze(-1)).squeeze(-1) - b_flat).pow(2).sum(dim=-1)
        residual = residual.reshape(leading_shape)
        solution = solution.reshape(*leading_shape, num_vars)
        if return_solution:
            return residual, solution
        return residual
