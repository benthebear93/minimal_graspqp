# minimal_graspqp

`minimal_graspqp` is a reduced reproduction of **GraspQP** focused on a small, testable subset of the original method.

Current project constraints:

- Shadow Hand only
- no physics-engine validation
- primitive objects only: `sphere`, `cylinder`, `box`

The current codebase implements a minimal end-to-end pipeline:

- Shadow Hand loading and forward kinematics
- primitive-object signed distance and surface normals
- contact candidate loading from the original Shadow Hand asset pack
- friction-cone wrench construction
- differentiable force-closure QP
- grasp energy terms
- MALA / MALA* optimization
- Plotly and MeshCat visualization

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
uv sync --extra dev --extra viz --extra meshcat
```

Useful commands:

```bash
uv lock
uv run pytest -q
uv run python scripts/<script_name>.py
```

## Repository Layout

```text
.
├── minimal_graspqp/
│   ├── energy/           # grasp energy terms
│   ├── hands/            # Shadow Hand model and FK
│   ├── init/             # primitive-based grasp initialization
│   ├── metrics/          # wrench and force-closure utilities
│   ├── objects/          # sphere, cylinder, box primitives
│   ├── optim/            # MALA / MALA*
│   ├── solvers/          # QP wrappers
│   ├── state.py          # grasp-state container
│   └── visualization/    # Plotly and MeshCat viewers
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
- `E_dis`, `E_pen`, `E_spen`, `E_joint`, `E_fc`
- bounded least-squares / force-closure QP
- minimal MALA / MALA* optimizer with:
  - EMA-style gradient normalization
  - annealed step size and temperature
  - MALA* z-score reset support
- primitive-aware initialization with:
  - object-facing wrist orientation
  - optional `--palm-down` bias
  - link-diverse active contact selection
- result export to `outputs/primitive_optimization.pt`
- Plotly HTML visualization
- MeshCat live visualization

Out of scope for this repo:

- multi-hand support
- complex mesh-object benchmark suites
- Isaac / simulator validation
- full paper-scale hyperparameter reproduction

## Run

### 1. Static Visualization

Plotly:

```bash
uv run python scripts/visualize_shadow_hand_with_primitive.py --primitive sphere --palm-down --show
```

MeshCat:

```bash
uv run python scripts/visualize_shadow_hand_with_primitive_meshcat.py --primitive sphere --palm-down
```

Other supported primitives:

```bash
uv run python scripts/visualize_shadow_hand_with_primitive_meshcat.py --primitive cylinder
uv run python scripts/visualize_shadow_hand_with_primitive_meshcat.py --primitive box
```

The MeshCat viewer is configured without the default grid, axes, or background.

### 2. Initialization Visualization

Plotly:

```bash
uv run python scripts/visualize_initialization.py --primitive sphere --batch-size 6 --palm-down --show
```

Save to HTML:

```bash
uv run python scripts/visualize_initialization.py \
  --primitive cylinder \
  --batch-size 8 \
  --output outputs/init_cylinder.html
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

### 4. Optimization Result Visualization

Plotly:

```bash
uv run python scripts/visualize_optimization_result.py \
  --input outputs/primitive_optimization.pt \
  --sample-index 0 \
  --show
```

MeshCat:

```bash
uv run python scripts/visualize_optimization_result_meshcat.py \
  --input outputs/primitive_optimization.pt \
  --sample-index 0
```

An example run that currently behaves reasonably well for the minimal setup:

```bash
uv run python scripts/optimize_primitive.py \
  --primitive sphere \
  --palm-down \
  --mala-star \
  --num-contacts 12 \
  --num-steps 20 \
  --batch-size 4 \
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
- Plotly visualization smoke tests
- MeshCat visualization smoke tests

## Notes

- This is an unofficial reproduction.
- The implementation follows the reduced scope in [SOFTWARE_REQUIREMENTS.md](/home/haegu/minimal_graspqp/SOFTWARE_REQUIREMENTS.md).
- Optimizer defaults are informed by the original `graspqp` codebase, but this repo is still a minimal implementation rather than a full paper-scale reproduction.
- Energy reduction does not yet imply a physically validated grasp. This repo currently has no simulator-based or quasi-static validation stage.
- The current initialization and contact selection are much better than the first minimal version, but they are still simplified relative to the full original pipeline.
