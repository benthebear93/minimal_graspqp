# GraspQP Reproduction

This repository is an implementation-focused reproduction of the paper **"GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping"** by René Zurbrügg, Andrei Cramariuc, and Marco Hutter.

The goal of this project is to build a clean, modular, and reproducible implementation of the main method proposed in the paper:

- a differentiable **force-closure energy**
- a **QP-based** formulation for bounded contact wrench optimization
- a modified **MALA\*** optimizer for diverse grasp generation
- an offline pipeline for **dexterous grasp synthesis and evaluation**

This repository currently starts from the paper and will be developed into a full reproduction codebase.

## Setup

This repository uses `uv` for environment and dependency management.

Recommended environment:

- Python `3.11`
- `uv`
- optional CUDA-enabled PyTorch environment if optimization will run on GPU

Create the virtual environment and install the base dependencies:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

Install development tools:

```bash
uv sync --extra dev
```

Install visualization dependencies:

```bash
uv sync --extra viz
```

Install everything used in the current minimal scope:

```bash
uv sync --extra dev --extra viz
```

Useful `uv` commands:

```bash
# Refresh the lockfile after editing dependencies
uv lock

# Run tests inside the uv-managed environment
uv run pytest

# Run a future script without manual activation
uv run python scripts/<script_name>.py
```

Notes:

- The dependency set follows the reduced scope in [SOFTWARE_REQUIREMENTS.md](/home/haegu/minimal_graspqp/SOFTWARE_REQUIREMENTS.md).
- The current environment is intentionally minimal: Shadow Hand only, no Isaac Lab, and primitive-object testing only.
- If you need a CUDA-specific PyTorch build, you may need to re-sync against the appropriate PyTorch index for your machine.

## Paper

- Title: *GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping*
- arXiv: `2508.15002v1`
- Authors: René Zurbrügg, Andrei Cramariuc, Marco Hutter

The paper proposes a differentiable grasp synthesis framework that improves grasp diversity and physical realism by replacing simplified force-closure surrogates with a more rigorous constrained formulation. The central idea is to optimize grasp parameters using gradients of a force-closure objective defined implicitly through a quadratic program, then improve exploration with a distribution-aware variant of MALA.

## Project Goal

The target of this repository is to reproduce the main technical claims of the paper:

1. Implement the analytical grasp energy with distance, regularization, and force-closure terms.
2. Implement the differentiable QP used to optimize bounded wrench coefficients.
3. Implement the MALA\* sampling / optimization procedure with:
   - dynamic resetting
   - adaptive temperature scaling
4. Generate diverse grasp candidates for dexterous hands.
5. Evaluate grasp quality using stability, diversity, and penetration metrics.

## Scope of the Reproduction

The intended scope is:

- reproduce the **core optimization method**, not just the final dataset
- prioritize a **readable research implementation**
- keep components modular enough to support ablations
- support future comparison against simpler baselines such as relaxed force-closure objectives

The initial focus is on the method itself:

- grasp representation
- contact and wrench construction
- force-closure QP
- differentiable gradients through the QP
- MALA\* update rule
- evaluation metrics

Large-scale dataset generation across thousands of objects is a later-stage goal, after the single-object and small-batch pipeline is stable.

## Method Summary

For a grasp configuration, the paper defines a total energy of the form

`E = E_FC + w_dis E_dis + w_reg E_reg`

where:

- `E_FC` is the force-closure energy
- `E_dis` encourages active contacts to lie on the object surface
- `E_reg` penalizes invalid or undesirable hand configurations such as penetration and joint-limit violations

The main novelty is the force-closure term. Instead of assuming equal contact magnitudes or frictionless closure, the paper solves for bounded nonnegative contact coefficients and defines force closure through a constrained optimization problem. The final formulation is implemented as a quadratic program and differentiated through the KKT system.

The second main contribution is **MALA\***, a modified Metropolis-adjusted Langevin procedure that improves diversity by:

- resetting poor samples based on the current energy distribution
- increasing acceptance temperature for underperforming samples

## Planned Implementation

The repository is expected to grow into the following structure:

```text
.
├── configs/              # experiment and optimizer settings
├── data/                 # object assets, metadata, cached preprocessing
├── graspqp/
│   ├── geometry/         # contacts, normals, wrench construction
│   ├── energy/           # distance, regularization, force-closure terms
│   ├── qp/               # differentiable quadratic program layer / solver wrapper
│   ├── optim/            # MALA and MALA* implementations
│   ├── hands/            # hand models and kinematics
│   ├── simulation/       # simulator bindings or evaluation environments
│   └── metrics/          # UGR, entropy, penetration, success metrics
├── scripts/              # training / optimization / evaluation entry points
├── tests/                # unit and regression tests
└── README.md
```

This exact layout may change, but the implementation will follow the same separation of concerns.

## Reproduction Roadmap

### Stage 1: Core Math

- implement grasp parameterization
- implement contact wrench construction
- implement friction-cone pyramid approximation
- implement the constrained force-closure objective

### Stage 2: Differentiable Optimization

- implement the QP form of the force-closure problem
- differentiate through the QP solution
- validate gradients numerically on small synthetic cases

### Stage 3: Grasp Search

- implement MALA baseline
- implement MALA\* with dynamic resetting and adaptive temperature scaling
- optimize batches of grasp candidates from coarse initializations

### Stage 4: Evaluation

- measure stability under external disturbances
- compute unique grasp rate (UGR)
- compute entropy over joint states and pose
- compute penetration depth

### Stage 5: Baselines and Ablations

- relaxed force-closure baselines
- MALA vs. MALA\*
- constrained vs. unconstrained formulations
- effect of rank / singular-value scaling

## Expected Dependencies

The current `uv` environment is centered on:

- Python 3.10+
- PyTorch
- SciPy
- trimesh
- transforms3d
- pytorch_kinematics
- qpth

Optional groups:

- `dev`: pytest, pytest-cov, ruff
- `viz`: matplotlib, plotly

## Evaluation Targets

The reproduction should eventually support the key metrics discussed in the paper:

- **Unique Grasp Rate (UGR)**: proportion of unique successful grasps
- **Entropy**: diversity over joint states and grasp pose
- **Penetration Depth**: hand-object interpenetration penalty
- **Success under Disturbance**: grasp stability against canonical-axis forces

The long-term goal is to reproduce the qualitative and quantitative behavior reported in the paper, especially:

- improved diversity over simpler force-closure approximations
- better grasp stability
- stronger scaling with more contact points
- better sample efficiency than simpler baselines under comparable optimization budgets

## Status

Current status: **repository initialization**

At the moment, this repository contains the paper PDF and a reproduction plan. Code, experiments, and results will be added incrementally.

## Notes

- This is an **unofficial reproduction** project.
- The implementation may differ from the original authors' internal codebase where details are not fully specified in the paper.
- When paper details are ambiguous, this repository will favor explicit, testable engineering decisions and document them.

## Reference

If you use this repository, please cite the original paper.

```bibtex
@article{zurbrugg2025graspqp,
  title={GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping},
  author={Zurbr{\"u}gg, Ren{\'e} and Cramariuc, Andrei and Hutter, Marco},
  journal={arXiv preprint arXiv:2508.15002},
  year={2025}
}
```
