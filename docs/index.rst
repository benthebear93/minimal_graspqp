minimal_graspqp
================

``minimal_graspqp`` is a reduced reproduction of GraspQP focused on a small,
testable pipeline for Shadow Hand grasps on primitive objects.

The package currently covers:

- Shadow Hand asset loading and forward kinematics
- primitive object signed distance and normals
- contact candidate transforms
- penetration sphere transforms
- friction-cone wrench construction
- differentiable force-closure QP evaluation
- grasp energy terms
- MALA and MALA* optimization
- Plotly and MeshCat visualization

.. toctree::
   :maxdepth: 2
   :caption: Overview

   overview
   concepts

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/installation
   guides/quickstart
   guides/optimization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/assets
   api/state
   api/hands
   api/objects
   api/init
   api/metrics
   api/energy
   api/optim
   api/solvers
   api/visualization
