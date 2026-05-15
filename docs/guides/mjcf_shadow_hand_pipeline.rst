MJCF Shadow Hand Pipeline
=========================

This project has two Shadow Hand paths:

- legacy URDF path, based on ``assets/shadow_hand/shadow_hand.urdf``
- current MJCF path, based on ``assets/mujoco_shadow_hand_in_hand_rolling/right_hand.xml``

For the current Shadow Hand grasp experiments, use the MJCF path for both
GraspQP optimization and MuJoCo validation.

Current Assets
--------------

The hand asset used by GraspQP is:

.. code-block:: text

   assets/mujoco_shadow_hand_in_hand_rolling/right_hand.xml

``MJCFShadowHandModel`` reads this MJCF directly and derives:

- forward kinematics
- joint limits
- contact candidates
- penetration/collision meshes

The MuJoCo validation scripts load the same MJCF:

.. code-block:: text

   assets/mujoco_shadow_hand_in_hand_rolling/right_hand.xml

This means the final intended pipeline is:

.. code-block:: text

   GraspQP optimization:
     assets/mujoco_shadow_hand_in_hand_rolling/right_hand.xml
     -> minimal_graspqp.hands.MJCFShadowHandModel

   MuJoCo validation:
     assets/mujoco_shadow_hand_in_hand_rolling/right_hand.xml
     -> mujoco.MjModel.from_xml_path(...)

Why This Exists
---------------

Earlier experiments optimized grasps with a URDF Shadow Hand asset and then
validated with a different MuJoCo Shadow Hand MJCF. That made the result hard
to interpret: a pose could look closed or in contact in the GraspQP visualizer
but appear open, shifted, or weak in MuJoCo.

The current MJCF path removes that asset mismatch. GraspQP and MuJoCo now share
the same kinematic tree, joint names after explicit mapping, mesh geometry, and
contact candidate source. If a grasp fails in MuJoCo now, the likely causes are
physical ones such as actuator strength, collision approximation, friction, or
force-closure quality, not a URDF/MJCF asset mismatch.

Object Mesh
-----------

The current object experiments use:

.. code-block:: text

   assets/objects/test_object.stl

GraspQP stores the object mesh path and scale in the optimization payload. The
MuJoCo validation scripts read the same mesh and export per-run generated files
such as ``object_visual.obj`` and ``object_collision.obj`` under the output
directory.

By default, ``mujoco_vendored_hand_eval.py`` uses the full object mesh for
collision. It only uses a convex hull when ``--convex-object`` is passed.

Optimization Command
--------------------

Example MJCF GraspQP optimization:

.. code-block:: bash

   uv run python -u scripts/optimize_primitive.py \
     --hand-model mjcf \
     --mesh-path assets/objects/test_object.stl \
     --mesh-scale 0.001 \
     --batch-size 64 \
     --num-steps 600 \
     --num-contacts 20 \
     --seed 8 \
     --allowed-contact-links ff,mf,rf,lf,th \
     --equalize-contacts-across-links \
     --w-close 8 \
     --close-target 1.05 \
     --w-pen 1000 \
     --w-spen 100 \
     --fc-qpth-max-iter 30 \
     --output outputs/graspqp_mjcf_20c_b64_s600.pt \
     --log-every 25

Viser Visualization
-------------------

Visualize a saved MJCF optimization result:

.. code-block:: bash

   uv run python scripts/visualize_mjcf_optimization_result.py \
     --input outputs/graspqp_mjcf_20c_b64_s600.pt \
     --sample-index 3 \
     --host 0.0.0.0 \
     --port 8094

MuJoCo Validation
-----------------

Evaluate a sample dynamically in MuJoCo with the same MJCF hand:

.. code-block:: bash

   uv run python scripts/mujoco_vendored_hand_eval.py \
     --input outputs/graspqp_mjcf_20c_b64_s600.pt \
     --samples 3 \
     --out-dir outputs/graspqp_mjcf_s3_eval \
     --mujoco-hand-xml assets/mujoco_shadow_hand_in_hand_rolling/right_hand.xml \
     --steps 3000 \
     --preload-steps 200 \
     --condim 4 \
     --friction 3.0 0.1 0.0 \
     --object-mass 0.05 \
     --no-floor \
     --contact-margin 0 \
     --show-collision-geoms \
     --show-contact-points \
     --viewer

Useful validation options:

- ``--contact-margin 0`` avoids phantom contacts caused by positive MuJoCo
  contact margin.
- ``--show-collision-geoms`` renders hand collision geometry as translucent
  blue geometry.
- ``--show-contact-points`` renders MuJoCo object contact points as red markers.
- ``--require-thumb-contact`` requires nonzero final thumb contact force.
- ``--fail-on-contact-loss`` fails a sample if it loses all object contacts
  during the simulated horizon.
- ``--condim 4 --friction 3.0 0.1 0.0`` is the current preferred validation
  setting. ``condim=6`` is often too sticky for this task.

Asset Directories
-----------------

Keep for the current MJCF pipeline:

.. code-block:: text

   assets/mujoco_shadow_hand_in_hand_rolling/
   assets/objects/test_object.stl

Keep only for legacy URDF workflows:

.. code-block:: text

   assets/shadow_hand/

Intermediate experimental assets that are not part of the final MJCF pipeline
and should not be required in a clean checkout:

.. code-block:: text

   assets/shadow_hand_mujoco_exact/
   assets/shadow_hand_mujoco_vendored_contacts/
   assets/shadow_hand_mujoco_vendored_pad_contacts/

These directories were useful while comparing URDF and MJCF hand variants. They
may be regenerated by helper scripts for debugging, but the current intended
pipeline does not depend on them.
