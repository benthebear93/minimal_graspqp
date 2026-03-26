import torch

from minimal_graspqp.solvers import BoundedLeastSquaresQPSolver


def test_qp_solver_respects_bounds():
    solver = BoundedLeastSquaresQPSolver()
    A = torch.eye(2, dtype=torch.float32).unsqueeze(0)
    b = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    residual, solution = solver.solve(A, b, min_bound=0.0, max_bound=1.0, return_solution=True)
    assert residual.shape == (1,)
    assert solution.shape == (1, 2)
    assert torch.all(solution >= -1e-4)
    assert torch.all(solution <= 1.0 + 1e-4)


def test_qp_solver_backpropagates():
    solver = BoundedLeastSquaresQPSolver()
    A = torch.eye(2, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    b = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    residual = solver.solve(A, b, min_bound=0.0, max_bound=2.0, return_solution=False)
    residual.sum().backward()
    assert A.grad is not None
    assert torch.isfinite(A.grad).all()
