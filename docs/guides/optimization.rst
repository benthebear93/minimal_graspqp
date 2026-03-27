Optimization Guide
==================

Core Inputs
-----------

The optimization loop needs:

- a ``ShadowHandModel``
- a primitive object
- an initialized ``GraspState``
- a ``ForceClosureQP`` metric

The optimizer then repeatedly evaluates the grasp energy and proposes updated
states.

Common Controls
---------------

The most important controls exposed by the example scripts are:

- ``--primitive`` to choose ``sphere``, ``cylinder``, or ``box``
- ``--batch-size`` to control parallel candidate count
- ``--num-steps`` to control optimization horizon
- ``--num-contacts`` to control active contact count
- ``--palm-down`` to bias the initialization
- ``--mala-star`` to enable reset-enhanced MALA*
- ``--seed`` for deterministic smoke runs

Energy Interpretation
---------------------

Useful reading of the term breakdown:

- high ``E_dis`` means the selected contact points are far from the surface
- high ``E_pen`` means the hand geometry is inside the object
- high ``E_spen`` means hand links overlap each other
- high ``E_joint`` means the current pose violates joint limits
- high ``E_fc`` means poor force-closure quality under the current metric

Recommended Smoke Run
---------------------

.. code-block:: bash

   uv run python scripts/optimize_primitive.py \
     --primitive sphere \
     --palm-down \
     --batch-size 1 \
     --num-steps 1 \
     --mala-star \
     --seed 0
