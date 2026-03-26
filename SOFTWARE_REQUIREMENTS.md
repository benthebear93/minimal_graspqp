# Software Requirements for `minimal_graspqp`

## 1. Purpose

This document defines the software requirements for `minimal_graspqp`, a reduced reimplementation of the GraspQP method described in the paper *"GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping"*.

The requirements are derived from the full implementation in `/home/haegu/graspqp`, but intentionally restrict scope to make the first implementation small, testable, and easier to validate.

## 2. Scope

`minimal_graspqp` shall implement only the core offline grasp optimization pipeline needed to reproduce the main analytical method at a reduced scale.

The reduced scope is constrained by the following project decisions:

1. Only **Shadow Hand** shall be supported.
2. No **physics-engine-based validation** shall be included.
3. Initial testing shall be limited to **simple analytic or mesh primitives**, such as:
   - sphere
   - cylinder
   - cube / box
   - square-like box variants

## 3. In-Scope Features

The system shall include:

- Shadow Hand kinematic model loading
- contact point handling for the Shadow Hand
- object representation for simple primitive objects
- signed-distance-based contact evaluation between hand contacts and object surface
- differentiable force-closure energy
- QP-based bounded coefficient solver for the GraspQP metric
- grasp energy aggregation
- gradient-based batch optimization with MALA and MALA*
- offline result export
- lightweight visualization or logging for debugging
- unit tests for the core mathematical components
- a local visualization path for manual inspection of hand-object configurations

## 4. Out-of-Scope Features

The following shall explicitly be excluded from `minimal_graspqp`:

- Isaac Lab integration
- Isaac Sim or any other simulator-based evaluation
- multi-hand support
- large-scale dataset pipelines
- DexGraspNet-scale object loading and preprocessing
- viewer web app and Blender export pipeline
- wandb dependency as a hard requirement
- support for all SDF backends from the original repository
- reinforcement learning environments
- retargeting, teleoperation, or hand pose tracking features

## 5. Reference Reduction from the Original Repository

The original repository `/home/haegu/graspqp` contains two broad subsystems:

- a standalone analytical optimization pipeline in `graspqp/src/graspqp`
- a simulator integration in `graspqp_isaaclab`

For `minimal_graspqp`, only the standalone analytical path shall be considered relevant. In particular:

- The energy structure in `core/energy.py` is relevant.
- The MALA and MALA* optimizer logic in `core/optimizer.py` is relevant.
- The Shadow Hand asset wrapper in `hands/shadow.py` is relevant.
- The differentiable QP span metric in `metrics/ops/span.py` and `metrics/solver/qp_solver.py` is relevant.
- Isaac Lab code under `graspqp_isaaclab/` is not required.

## 6. Functional Requirements

### FR-1. Shadow Hand Support

The system shall provide a single hand backend for the Shadow Hand.

Requirements:

- load the Shadow Hand URDF and mesh assets
- load predefined contact points and penetration points
- expose joint limits, default joint state, and contact candidates
- support batched hand pose parameters

Acceptance:

- a Shadow Hand model can be instantiated on CPU and GPU
- forward kinematics returns valid link poses and contact points

### FR-2. Primitive Object Support

The system shall support a minimal object set for early-stage development.

Required primitive objects:

- sphere
- cylinder
- box
- flat square-like box or plate

The object system shall support either:

- analytic SDF implementations for primitives, or
- mesh loading for a small local primitive asset set

Preference:

- analytic SDFs should be used where practical, since they reduce dependency and preprocessing complexity

Acceptance:

- each primitive can be instantiated with configurable size and pose
- the system can compute signed distance and surface normal for arbitrary query points

### FR-3. Contact Geometry Evaluation

The system shall evaluate the relationship between hand contacts and object surface.

Requirements:

- compute signed distances from hand contact points to object surface
- compute outward surface normals at or near the closest point
- support batched evaluation

Acceptance:

- batched contact queries produce tensors with stable shapes
- gradients with respect to hand pose are available for optimization

### FR-4. Grasp Representation

The system shall represent a grasp as:

- wrist pose in SE(3)
- Shadow Hand joint configuration
- active contact point indices or equivalent contact selection state

Acceptance:

- the representation can be initialized, updated, optimized, and exported

### FR-5. Energy Function

The system shall implement a reduced total grasp energy consistent with the paper and the original code structure:

`E_total = w_fc * E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joint * E_joint`

Required terms:

- `E_fc`: force-closure energy
- `E_dis`: contact distance / contact alignment term
- `E_pen`: hand-object penetration term
- `E_spen`: self-penetration term
- `E_joint`: joint limit penalty

Optional for the first version:

- prior terms
- wall penalty
- manipulability term

Acceptance:

- each term can be computed independently
- total energy is differentiable with respect to the grasp parameters

### FR-6. Force-Closure Metric

The system shall implement the GraspQP force-closure metric based on bounded nonnegative wrench coefficients.

Requirements:

- build contact wrenches from contact points and normals
- support friction-cone approximation with a four-sided pyramid
- solve for bounded coefficients using a QP
- include the singular-value-based scaling term used to discourage rank-deficient wrench matrices

Acceptance:

- the metric returns a scalar batch energy
- the metric returns the optimized coefficient solution when requested
- gradients flow from the metric to contact points and normals

### FR-7. Differentiable QP Solver

The system shall provide a differentiable QP solver layer for the force-closure term.

Requirements:

- solve bounded least-squares / QP problems in batch form
- operate with PyTorch tensors
- support autodiff through the solver output

Implementation note:

- `qpth` is acceptable for the initial version because it matches the original implementation closely

Acceptance:

- the solver runs on a small batch without numerical failure
- backpropagation through the solver produces finite gradients

### FR-8. Optimization Algorithms

The system shall support:

- baseline MALA / annealed gradient optimization
- MALA* with dynamic resetting and adaptive temperature scaling

Requirements:

- update wrist pose and joint parameters
- optionally resample contact indices
- accept or reject proposals based on energy change
- reset poor-performing samples based on batch energy statistics

Acceptance:

- optimization reduces total energy on primitive-object test cases
- MALA* can be enabled or disabled by configuration

### FR-9. Initialization

The system shall provide a simple initialization strategy for grasp candidates.

Minimum requirement:

- initialize the hand around the primitive object with random pose perturbations and valid default joints

Preferred next step:

- convex-hull-inspired initialization similar to the original codebase

Acceptance:

- a batch of initial grasps can be generated without invalid tensor states

### FR-10. Result Export

The system shall export optimization results in a simple machine-readable format.

Required fields:

- object identifier
- object parameters
- hand pose
- joint values
- selected contact indices
- final energy
- per-term energy breakdown
- solver coefficients if enabled

Recommended formats:

- `.pt`
- `.npz`
- `.json` for metadata only

### FR-11. Minimal Visualization for Manual Inspection

The system shall provide a lightweight local visualization capability so a user can visually inspect the current Shadow Hand pose relative to a primitive object.

Requirements:

- render at least one primitive object together with the Shadow Hand
- display the current hand pose produced by initialization or optimization
- support visualization of contact candidate points or selected active contacts
- run without Isaac Lab or any simulator dependency

Preferred implementation:

- a simple Plotly-based local script

Acceptance:

- a user can run one script locally and visually inspect the hand-object configuration
- the visualization works for at least sphere, cylinder, and box primitives
- the visualization can be used before full optimization is implemented

## 7. Non-Functional Requirements

### NFR-1. Simplicity First

The implementation shall prioritize clarity and low dependency count over feature completeness.

### NFR-2. Reproducibility

The system shall expose random seeds for:

- initialization
- contact resampling
- optimizer randomness

### NFR-3. Modularity

The code shall separate:

- hand model
- object model
- energy terms
- force-closure metric
- QP solver
- optimizer
- experiments / scripts

### NFR-4. Numerical Robustness

The implementation shall guard against:

- NaN gradients
- invalid solver outputs
- rank-deficient wrench matrices
- exploding optimizer steps

### NFR-5. CPU/GPU Support

The system should run on CPU for debugging and on CUDA for practical optimization.

GPU support is preferred, but CPU execution shall remain possible for unit tests and small examples.

## 8. Software Architecture Requirements

The repository should evolve toward the following minimal structure:

```text
minimal_graspqp/
├── minimal_graspqp/
│   ├── hands/
│   ├── objects/
│   ├── energy/
│   ├── metrics/
│   ├── solvers/
│   ├── optim/
│   └── utils/
├── scripts/
├── tests/
├── assets/
└── docs/
```

Expected module responsibilities:

- `hands/`: Shadow Hand loading, FK, contact points, self-penetration helpers
- `objects/`: primitive objects and SDF queries
- `energy/`: distance, joint, penetration, and total energy composition
- `metrics/`: force-closure wrench construction and metric computation
- `solvers/`: QP layer
- `optim/`: MALA and MALA*
- `scripts/`: optimize, visualize, and debug entry points

## 9. Dependency Requirements

### Required Dependencies

The first implementation shall depend on:

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- trimesh
- pytorch_kinematics
- qpth

### Optional Dependencies

The following should remain optional:

- plotly
- matplotlib
- wandb
- pytorch3d
- TorchSDF
- warp
- kaolin

Reduction guideline:

- If primitive analytic SDFs are implemented directly, `TorchSDF`, `warp`, and `kaolin` should not be mandatory.
- If primitive surface sampling is implemented without mesh FPS, `pytorch3d` should not be mandatory.

## 10. Data and Asset Requirements

The system shall include or reference only the Shadow Hand assets required for the minimal pipeline:

- URDF
- collision or visual meshes
- contact point definitions
- penetration point definitions

Primitive object support shall not require a large external dataset.

The primitive test set shall be small, local, and version-controlled if possible.

## 11. Testing Requirements

### TR-1. Unit Tests

The project shall include unit tests for:

- primitive SDF correctness
- surface normal correctness
- wrench construction
- friction-cone expansion
- QP solver output shape and boundedness
- gradient propagation through the force-closure metric
- Shadow Hand FK output shape

### TR-2. Integration Tests

The project shall include small integration tests for:

- optimizing a grasp on a sphere
- optimizing a grasp on a cylinder
- optimizing a grasp on a box

Success criteria:

- optimization finishes without NaNs
- final energy is lower than initial energy for most seeds
- output files are generated correctly

### TR-3. No Simulator Validation

The test strategy shall not depend on Isaac Lab or any physics engine.

Instead, validation shall be limited to:

- optimization behavior
- contact geometry consistency
- force-closure energy behavior
- qualitative visualization through the local manual-inspection script

## 12. Deliverables

The first usable milestone of `minimal_graspqp` shall include:

1. A Shadow Hand model loader.
2. Primitive object models for sphere, cylinder, and box.
3. A differentiable force-closure metric with QP backend.
4. A total grasp energy implementation.
5. A MALA* optimizer implementation.
6. A script that runs batch optimization on a primitive object.
7. Basic tests for math and optimization.
8. A README documenting installation and usage.
9. A local visualization script for manual inspection of primitive-object scenes.

## 13. Recommended Development Order

The implementation should proceed in this order:

1. Shadow Hand loading and FK
2. primitive object SDFs
3. contact distance and normal queries
4. wrench construction
5. QP force-closure metric
6. total energy composition
7. MALA baseline
8. MALA*
9. export and visualization
10. tests and cleanup

## 14. Acceptance Summary

`minimal_graspqp` shall be considered to satisfy this SR when:

- it supports only Shadow Hand and no other hand
- it runs without Isaac Lab or simulator validation
- it can optimize grasps on sphere, cylinder, and box-like objects
- it includes a differentiable QP-based force-closure metric
- it provides at least one working offline optimization script
- it includes basic unit and integration tests for the reduced scope
