# Software Requirements for `minimal_graspqp`

## 1. Purpose

This document defines the software requirements for `minimal_graspqp`, a reduced reimplementation of the GraspQP method described in the paper *"GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping"*.

The requirements are derived from the full implementation in `/home/haegu/graspqp`, but intentionally restrict scope to make the first implementation small, testable, and easier to validate.

## 2. Scope

`minimal_graspqp` shall implement only the core offline grasp optimization pipeline needed to reproduce the main analytical method at a reduced scale.

The reduced scope is constrained by the following project decisions:

1. Only **Shadow Hand** shall be supported.
2. No **physics-engine-based validation** shall be included.
3. Initial testing shall be limited to **simple analytic objects and a small local mesh set**, such as:
   - sphere
   - cylinder
   - cube / box
   - square-like box variants
   - a few original-style `coacd/remeshed.obj` mesh objects

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
- interpret Shadow Hand `contact_points.json` consistently with the original repository:
  - entries of the form `[mesh_path, n_points]` shall be treated as surface-sampling directives, not vertex indices
  - contact candidates shall be sampled on the referenced contact mesh and reduced to `n_points` surface representatives
  - candidate surface normals shall be available or derivable for later contact filtering and visualization

Acceptance:

- a Shadow Hand model can be instantiated on CPU and GPU
- forward kinematics returns valid link poses and contact points
- the loaded Shadow Hand contact candidate set is materially larger than one point per link for the provided asset pack

### FR-1.1. Shadow Hand Contact Candidate Editing

The system shall provide a local manual-edit path for Shadow Hand contact candidates.

Requirements:

- expose each contact candidate with a stable global index for inspection and editing
- support per-index local-position overrides without modifying the original sampled contact-candidate definition file
- support disabling individual contact candidate indices without renumbering the remaining indices
- load contact-candidate overrides and disabled-index masks as optional external metadata layers
- ensure disabled contact candidates are excluded consistently from visualization, initialization, and optimization paths when the mask is enabled
- provide a local interactive editor for manual inspection and adjustment of contact candidates
- the editor should support point selection in 3D and direct manipulation of the selected candidate position

Acceptance:

- a user can inspect a specific contact-candidate index and its owning link
- a user can move a selected contact candidate and save the override to a local metadata file
- a user can disable and re-enable a contact-candidate index without changing other global indices
- the resulting override/mask files can be reapplied in a later session

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

### FR-2.1. Optional Mesh / Convex-Hull Object Support

The system shall additionally support a small number of real mesh objects for
initialization, optimization, and visualization experiments, without expanding
to the full original dataset pipeline.

Requirements:

- a mesh object may be loaded directly from a local mesh path
- a mesh object may optionally be resolved from an original-style object code
  layout such as ``<data_root>/<object_code>/coacd/remeshed.obj``
- initialization for mesh objects should prefer convex-hull-based surface
  sampling similar to the original repository
- full differentiable mesh signed distance and surface normal queries should use
  the full object mesh when `TorchSDF` is available
- if full differentiable mesh SDF is unavailable, a convex-hull approximation is
  acceptable only as a fallback for smoke testing

Acceptance:

- at least one local mesh object can be loaded and visualized with the Shadow Hand
- convex-hull-based initialization produces valid grasp batches for that object
- a mesh-object optimization run can complete end-to-end in the local repo

### FR-3. Contact Geometry Evaluation

The system shall evaluate the relationship between hand contacts and object surface.

Requirements:

- compute signed distances from hand contact points to object surface
- compute outward surface normals at or near the closest point
- support batched evaluation
- for mesh objects, use full-mesh signed distance / normal queries rather than
  convex-hull queries whenever the mesh backend supports it

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
- `E_pen` should match the original GraspQP structure more closely by querying
  object surface samples against a hand collision model, rather than relying
  only on a sparse hand-sphere approximation

### FR-6. Force-Closure Metric

The system shall implement the GraspQP force-closure metric based on bounded nonnegative wrench coefficients.

Requirements:

- build contact wrenches from contact points and normals
- support friction-cone approximation with a four-sided pyramid
- solve for bounded coefficients using a QP
- include the singular-value-based scaling term used to discourage rank-deficient wrench matrices
- prefer the original GraspQP overall friction-cone span formulation over a
  simpler `±I` basis surrogate
- support warm-starting the bounded coefficient solve across optimizer steps

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
- expose the original `qpth` tolerance / iteration behavior closely enough to
  compare with the reference implementation

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
- use gradient normalization or equivalent step stabilization to avoid optimizer collapse
- use annealed schedules for both step size and acceptance temperature
- support optimizer-state reset when grasp samples are reinitialized

Acceptance:

- optimization reduces total energy on primitive-object test cases
- MALA* can be enabled or disabled by configuration
- contact-index switching probability should be configurable, since the original
  implementation relies on nonzero contact switching during optimization

### FR-8.1. MALA / MALA* Implementation Constraints

To stay consistent with the original GraspQP implementation, the reduced optimizer shall follow these constraints:

- gradient-based updates shall use an EMA or RMSProp-style normalization scheme
- acceptance temperature shall decay over time instead of remaining fixed
- step size shall decay over time instead of remaining fixed
- MALA* temperature scaling shall depend on the current energy distribution across the batch
- MALA* resetting shall use a distribution-based criterion such as z-score thresholding over current batch energies
- when a sample is reset, the corresponding optimizer internal state shall also be reset

These constraints are required because a naive gradient-descent or fixed-temperature implementation may satisfy the high-level interface but fail to reproduce the intended optimizer behavior.

### FR-9. Initialization

The system shall provide a simple initialization strategy for grasp candidates.

Minimum requirement:

- initialize the hand around the primitive object with random pose perturbations and valid default joints

Preferred next step:

- convex-hull-inspired initialization similar to the original codebase
- object-facing filtering or similarly constrained active-contact selection using the richer candidate set

Optional extension:

- for real mesh objects, initialization should use convex-hull surface samples
  and outward hull normals in the same role that the original repository uses
  for coarse pose seeding

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

- a simple viser-based local script

Acceptance:

- a user can run one script locally and visually inspect the hand-object configuration
- the visualization works for at least sphere, cylinder, and box primitives
- the visualization can be used before full optimization is implemented
- for Shadow Hand contact-candidate editing, the visualization should support
  selecting a displayed candidate point and highlighting the currently edited index

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
- `objects/`: primitive objects, optional mesh objects, and SDF queries
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

- viser
- matplotlib
- wandb
- rtree
- pytorch3d
- TorchSDF
- warp
- kaolin

Reduction guideline:

- If primitive analytic SDFs are implemented directly, `TorchSDF`, `warp`, and `kaolin` should not be mandatory.
- If primitive surface sampling is implemented without mesh FPS, `pytorch3d` should not be mandatory.
- If mesh-object proximity uses `trimesh.nearest` or signed-distance queries,
  `rtree` becomes effectively required for that path.

Object-model dependency tiers:

- Primitive-only path:
  - `torch`
  - `numpy`
  - `scipy`
  - `trimesh`
- Convex-hull mesh initialization path:
  - primitive-only dependencies
  - `rtree` if `trimesh` nearest-surface queries are used directly
- Original-style differentiable mesh SDF path:
  - primitive-only dependencies
  - `pytorch3d`
  - one SDF backend: `TorchSDF`, `warp`, or `kaolin`
  - `rtree` for some proximity and nearest-surface utilities

Current local-environment note:

- the current workspace does not have `rtree`, `pytorch3d`, `TorchSDF`, `warp`,
  or `kaolin` installed
- therefore primitive analytic objects are the only fully supported path today,
  and mesh-object support should be treated as optional or approximate until
  those dependencies are added

## 10. Data and Asset Requirements

The system shall include or reference only the Shadow Hand assets required for the minimal pipeline:

- URDF
- collision or visual meshes
- contact point definitions
- penetration point definitions

Primitive object support shall not require a large external dataset.

The primitive test set shall be small, local, and version-controlled if possible.

Optional mesh-object experiments may reference one or a few local meshes outside
the repository, but they shall not become a hard requirement for the minimal
pipeline.

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
3. Optional local mesh-object smoke support is acceptable, but not required for
   the first milestone.
4. A differentiable force-closure metric with QP backend.
5. A total grasp energy implementation.
6. A MALA* optimizer implementation.
7. A script that runs batch optimization on a primitive object.
8. Basic tests for math and optimization.
9. A README documenting installation and usage.
10. A local visualization script for manual inspection of primitive-object scenes.

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
