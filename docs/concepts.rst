Core Concepts
=============

Grasp State
-----------

The central state container is ``GraspState``. A grasp state stores:

- joint values
- wrist translation
- wrist rotation
- active contact candidate indices

Every stage in the pipeline either creates a ``GraspState``, transforms data
derived from it, or updates it during optimization.

Contact Candidates
------------------

``ShadowHandModel.contact_candidates_world`` transforms precomputed contact
candidate points from link-local coordinates into world coordinates. These
points represent plausible hand surface locations for contact with the object.

The resulting points are used to compute:

- signed-distance contact error
- surface normals on the primitive
- force-closure quality

Penetration Spheres
-------------------

``ShadowHandModel.penetration_spheres_world`` transforms a set of small spheres
attached to hand links into world coordinates. Each sphere comes from
``penetration_points.json`` and is encoded as ``[x, y, z, r]`` in a link-local
frame.

These spheres support two penalties:

- object penetration penalty, by querying the primitive SDF at sphere centers
- self-penetration penalty, by checking sphere-sphere overlaps across links

Primitive Objects
-----------------

Primitive objects implement two key geometric queries:

- ``signed_distance(points)``
- ``normals(points)``

This keeps the optimization code independent of the specific primitive shape as
long as the object supplies those operations.

Energy Terms
------------

``compute_grasp_energy`` combines several terms:

- ``E_dis``: contact-point distance to the object surface
- ``E_pen``: hand penetration into the object
- ``E_spen``: hand self-penetration
- ``E_joint``: joint limit violation
- ``E_fc``: force-closure quality

The weighted sum ``E_total`` is the optimization target.

Optimization
------------

The optimizer uses MALA-style stochastic proposals over the grasp state. In
this repository, the optimizer is intentionally small and exposes the parts that
matter for experimentation:

- gradient normalization
- annealed step size
- annealed temperature
- optional MALA* reset logic
