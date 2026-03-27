Installation
============

Environment
-----------

Recommended setup:

- Python 3.11
- ``uv``
- optional CUDA-enabled PyTorch for faster optimization

Create and sync the environment:

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate
   uv sync --extra dev --extra viser --extra docs

Asset Resolution
----------------

The Shadow Hand assets are resolved from one of these locations:

1. ``MINIMAL_GRASPQP_SHADOW_ASSETS``
2. ``<repo-root>/assets/shadow_hand``
3. ``/home/haegu/graspqp/graspqp/assets/shadow_hand``

The asset directory must contain:

- ``shadow_hand.urdf``
- ``contact_points.json``
- ``penetration_points.json``
- ``meshes/``
- ``contact_mesh/``

Build The Docs
--------------

Generate the HTML documentation with Sphinx:

.. code-block:: bash

   uv run sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser after the build completes.
