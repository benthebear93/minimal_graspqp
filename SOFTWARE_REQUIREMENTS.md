# Software Requirements for `minimal_graspqp`

## Purpose

This document defines the intended software and feature scope for `minimal_graspqp`, a reduced analytical reproduction of GraspQP centered on Shadow Hand offline grasp optimization.

## Target Scope

`minimal_graspqp` is intended to provide:

- a single-hand analytical optimization pipeline
- primitive and small local mesh-object support
- differentiable energy evaluation and optimization
- lightweight local visualization and debugging tools

It is not intended to provide:

- simulator integration
- large-scale dataset workflows
- multi-hand support
- reinforcement learning environments
- teleoperation, tracking, or retargeting workflows

## Functional Requirements

### FR-1. Shadow Hand

The system shall support Shadow Hand as the only hand model.

Requirements:

- load Shadow Hand URDF, collision meshes, contact candidates, and penetration points
- expose joint limits, default joint state, and batched forward kinematics
- support optional fingertip-only contact candidate filtering

Acceptance:

- the hand model can be instantiated on CPU or CUDA
- contact candidates and penetration points are available for optimization and visualization

### FR-2. Object Models

The system shall support both primitives and small local mesh objects.

Requirements:

- primitives: `sphere`, `cylinder`, `box`
- local mesh path loading
- original-style object-code loading from `<object_root>/<object_code>/coacd/...`
- convex-hull-based initialization for mesh objects
- full-mesh SDF and normals when `TorchSDF` is available

Acceptance:

- primitive and mesh objects can be initialized, queried, visualized, and optimized

### FR-3. Grasp Representation

The system shall represent a grasp with:

- wrist translation
- wrist rotation
- joint configuration
- active contact indices

Acceptance:

- grasp states can be initialized, optimized, serialized, and visualized

### FR-4. Energy Terms

The system shall implement the reduced total energy:

`E_total = w_fc * E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joint * E_joint`

Required terms:

- `E_fc`: force-closure term
- `E_dis`: contact distance term
- `E_pen`: hand-object penetration term
- `E_spen`: self-penetration term
- `E_joint`: joint-limit penalty

Acceptance:

- each term can be evaluated independently
- total energy is differentiable with respect to grasp parameters

### FR-5. Force Closure

The system shall implement a bounded QP-based force-closure surrogate.

Requirements:

- construct friction-cone wrench matrices
- use bounded nonnegative coefficient solves
- support warm-starting across optimization steps

Acceptance:

- the metric produces finite batched energies
- gradients propagate through contact geometry

### FR-6. Optimization

The system shall support MALA and MALA* style optimization.

Requirements:

- batched optimization
- annealed step size and temperature
- optional contact switching
- MALA* z-score temperature modulation
- optional periodic reset support

Acceptance:

- optimization runs complete end-to-end for primitive and mesh examples
- result checkpoints can be exported for later visualization

### FR-7. Visualization

The system shall provide lightweight local visualization.

Requirements:

- static primitive viewer
- initialization viewer
- optimization result viewer for a single sample
- optimization result viewer for a full batch

Acceptance:

- a user can inspect initial and final grasp configurations locally through viser

## Software Requirements

### Required

- Python `3.11`
- `uv`
- PyTorch compatible with the local CUDA stack when GPU acceleration is desired
- `qpth`
- `pytorch_kinematics`
- `trimesh`

### Strongly Recommended

- `TorchSDF`
- `PyTorch3D`
- `rtree`
- `viser`

These are operationally important:

- without `TorchSDF`, hand-object penetration and mesh SDF evaluation may fall back to much slower paths
- without `viser`, visualization scripts will not run

## Operational Expectations

- The preferred workflow is `uv sync` and `uv run`; direct `pip` and manual virtual-environment handling should be avoided unless necessary.
- The optimization scripts should print backend status at startup so missing acceleration backends are immediately visible.
- Debug runs should support lightweight timing output for identifying whether runtime is dominated by penetration, force closure, or backward computation.

## Reference Alignment

`minimal_graspqp` is intended to stay conceptually aligned with the analytical portion of `/home/haegu/graspqp`, while remaining smaller and easier to inspect.

Alignment targets:

- Shadow Hand asset usage
- contact candidate interpretation
- QP-based force-closure structure
- MALA / MALA* optimization behavior
- original-style hand-object penetration energy when acceleration backends are available

Permitted simplifications:

- smaller local object set
- reduced script surface
- reduced documentation and infrastructure
- fewer supported backends and less tuning than the full repository
