Overview
========

``minimal_graspqp`` keeps the core optimization loop from GraspQP while
shrinking the modeling scope to something easy to inspect, test, and modify.

Project Scope
-------------

Included in this repository:

- a single hand model: Shadow Hand
- primitive objects only: sphere, cylinder, and box
- batched forward kinematics and candidate transforms
- primitive-aware initialization
- energy-based grasp scoring
- force-closure evaluation through a differentiable QP
- MALA and MALA* sampling-style optimization
- Plotly and MeshCat visualization

Explicitly out of scope:

- simulator-backed validation
- multi-hand setups
- mesh-object benchmark suites
- full paper-scale hyperparameter reproduction

High-Level Pipeline
-------------------

The main data flow is:

1. Load Shadow Hand metadata and assets.
2. Create a primitive object.
3. Initialize batched grasp states.
4. Transform contact candidates and penetration spheres into world coordinates.
5. Evaluate energy terms and force-closure quality.
6. Optimize grasp states with MALA or MALA*.
7. Export or visualize the resulting grasp.

Repository Layout
-----------------

The repository is organized around the same stages:

- ``minimal_graspqp/hands``: hand metadata, FK, contact and penetration geometry
- ``minimal_graspqp/objects``: primitive SDFs and normals
- ``minimal_graspqp/init``: initial grasp-state generation
- ``minimal_graspqp/metrics``: friction-cone and force-closure utilities
- ``minimal_graspqp/energy``: grasp objective terms
- ``minimal_graspqp/optim``: MALA optimizer and history containers
- ``minimal_graspqp/solvers``: bounded QP backend
- ``minimal_graspqp/visualization``: Plotly and MeshCat scene builders
- ``scripts``: runnable entry points for visualization and optimization
- ``tests``: feature-level regression coverage
