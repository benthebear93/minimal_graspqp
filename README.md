# minimal_graspqp

`minimal_graspqp` is a reduced reproduction of **GraspQP** focused on a small, testable subset of the original method.

Current project constraints:

- Shadow Hand only
- no physics-engine validation
- no Isaac / simulator evaluation

The current codebase implements a minimal end-to-end pipeline:

- Shadow Hand loading and forward kinematics
- primitive and mesh-object signed distance and surface normals
- contact candidate loading from the original Shadow Hand asset pack
- friction-cone wrench construction
- differentiable force-closure QP
- grasp energy terms
- MALA / MALA* optimization
- viser visualization

## Setup

This repository uses `uv`.

Recommended environment:

- Python `3.11`
- `uv`
- optional CUDA-enabled PyTorch setup if you want faster optimization

Create the environment and install everything used by the current repo:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra dev --extra viser --extra mesh
```

Useful commands:

```bash
uv lock
uv run pytest -q
uv run python scripts/<script_name>.py
uv run sphinx-build -b html docs docs/_build/html
```

## Repository Layout

```text
.
├── minimal_graspqp/
│   ├── energy/           # grasp energy terms
│   ├── hands/            # Shadow Hand model and FK
│   ├── init/             # primitive / mesh-based grasp initialization
│   ├── metrics/          # wrench and force-closure utilities
│   ├── objects/          # sphere, cylinder, box, mesh objects
│   ├── optim/            # MALA / MALA*
│   ├── solvers/          # QP wrappers
│   ├── state.py          # grasp-state container
│   └── visualization/    # viser viewers
├── scripts/              # runnable examples
├── tests/                # feature-level tests
├── SOFTWARE_REQUIREMENTS.md
└── README.md
```

## Implemented Scope

Implemented now:

- Shadow Hand asset resolution from the original `graspqp` asset tree
- batched FK and contact candidate transforms
- original-style Shadow Hand contact-candidate interpretation:
  - `contact_points.json` entries are treated as surface-sampling directives
  - the current Shadow Hand candidate pool contains `80` contact candidates
- candidate normals for the Shadow Hand contact set
- primitive SDFs for `sphere`, `cylinder`, and `box`
- mesh-object loading from direct mesh paths or original-style object-code layouts
- convex-hull initialization for mesh objects
- full-mesh object SDF / normal queries via `TorchSDF`
- `E_dis`, `E_pen`, `E_spen`, `E_joint`, `E_fc`
- original-style hand-object penetration:
  - object surface samples against Shadow Hand collision-mesh signed distance
- original-style force-closure surrogate:
  - overall friction-cone span metric
  - bounded QP with warm-started coefficients
- minimal MALA / MALA* optimizer with:
  - EMA-style gradient normalization
  - annealed step size and temperature
  - MALA* z-score reset support
- optional contact-index switching during optimization
- primitive-aware initialization with:
  - object-facing wrist orientation
  - optional `--palm-down` bias
- random active contact initialization matching the original code path
- result export to `outputs/primitive_optimization.pt`
- viser live visualization

Out of scope for this repo:

- multi-hand support
- complex mesh-object benchmark suites
- Isaac / simulator validation
- full paper-scale hyperparameter reproduction

## Documentation

Sphinx documentation lives under `docs/` and is organized into:

- overview and core concepts
- task-oriented guides
- API reference by package

Build it with:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

Deployment options:

- GitHub Pages via `.github/workflows/docs.yml`
- Read the Docs via `.readthedocs.yaml`

## Run

### 1. Static Visualization

```bash
uv run python scripts/visualize_shadow_hand_with_primitive.py --primitive sphere --palm-down
```

Other supported primitives:

```bash
uv run python scripts/visualize_shadow_hand_with_primitive.py --primitive cylinder
uv run python scripts/visualize_shadow_hand_with_primitive.py --primitive box
```

### 2. Initialization Visualization

```bash
uv run python scripts/visualize_initialization.py --primitive sphere --batch-size 6 --palm-down
```

### 3. Primitive Optimization

Baseline run:

```bash
uv run python scripts/optimize_primitive.py \
  --primitive sphere \
  --palm-down \
  --batch-size 4 \
  --num-steps 10
```

MALA*:

```bash
uv run python scripts/optimize_primitive.py \
  --primitive sphere \
  --palm-down \
  --batch-size 4 \
  --num-steps 10 \
  --mala-star
```

A small deterministic smoke run:

```bash
uv run python scripts/optimize_primitive.py \
  --primitive sphere \
  --palm-down \
  --batch-size 1 \
  --num-steps 1 \
  --mala-star \
  --seed 0
```

The optimizer writes:

- `outputs/primitive_optimization.pt`

and prints:

- mean initial energy
- mean final energy
- best mean energy in trace
- accepted steps per iteration
- resets per iteration

### 3.1. Mesh Optimization

Original-style object-code layout:

```bash
uv run python scripts/optimize_primitive.py \
  --object-root /home/haegu/minimal_graspqp/assets/objects \
  --object-code core_bottle \
  --batch-size 4 \
  --num-steps 20 \
  --num-contacts 12 \
  --mala-star \
  --contact-switch-probability 0.4 \
  --output outputs/core_bottle_optimization.pt
```

Direct mesh path:

```bash
uv run python scripts/optimize_primitive.py \
  --mesh-path /home/haegu/minimal_graspqp/assets/objects/remeshed.obj \
  --batch-size 4 \
  --num-steps 20 \
  --num-contacts 12 \
  --mala-star \
  --contact-switch-probability 0.4 \
  --output outputs/mesh_optimization.pt
```

Notes:

- `--contact-switch-probability 0.4` matches the original `graspqp` default more closely
- the paper optimizes for `7000` steps; the short commands here are smoke / debug runs
- mesh initialization uses the convex hull, while distance / normal queries use the full mesh

### 4. Optimization Result Visualization

```bash
uv run python scripts/visualize_optimization_result.py \
  --input outputs/primitive_optimization.pt \
  --sample-index 0
```

An example run that currently behaves reasonably well for the minimal setup:

```bash
uv run python scripts/optimize_primitive.py \
  --object-root /home/haegu/minimal_graspqp/assets/objects \
  --object-code core_bottle \
  --mala-star \
  --num-contacts 12 \
  --num-steps 200 \
  --batch-size 4 \
  --contact-switch-probability 0.4 \
  --seed 0
```

## Testing

Run the full test suite:

```bash
uv run pytest -q
```

Tests are organized by feature, including:

- hand model and FK
- primitive geometry
- wrench construction
- QP solving
- force closure
- initialization
- energy computation
- optimizer behavior
- viser visualization smoke tests

## Notes

- This is an unofficial reproduction.
- The implementation follows the reduced scope in [SOFTWARE_REQUIREMENTS.md](/home/haegu/minimal_graspqp/SOFTWARE_REQUIREMENTS.md).
- Optimizer defaults are informed by the original `graspqp` codebase, but this repo is still a minimal implementation rather than a full paper-scale reproduction.
- Energy reduction does not yet imply a physically validated grasp. This repo currently has no simulator-based or quasi-static validation stage.
- The current initialization and contact selection are much better than the first minimal version, but they are still simplified relative to the full original pipeline.
