Quickstart
==========

Static Visualization
--------------------

.. code-block:: bash

   uv run python scripts/visualize_shadow_hand_with_primitive.py --primitive sphere --palm-down

Initialization
--------------

Generate a batch of initialized grasps and inspect them:

.. code-block:: bash

   uv run python scripts/visualize_initialization.py --primitive sphere --batch-size 6 --palm-down

Optimization
------------

Run a minimal optimization:

.. code-block:: bash

   uv run python scripts/optimize_primitive.py \
     --primitive sphere \
     --palm-down \
     --batch-size 4 \
     --num-steps 10

The optimizer writes ``outputs/primitive_optimization.pt``.

Inspect A Result
----------------

.. code-block:: bash

   uv run python scripts/visualize_optimization_result.py \
     --input outputs/primitive_optimization.pt \
     --sample-index 0
