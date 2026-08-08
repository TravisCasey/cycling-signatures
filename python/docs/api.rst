API reference
=============

The compiled bindings exposed by the ``cycling_signatures`` package. See
:doc:`concepts` for the vocabulary these classes and methods use.

Metrics
-------

A metric measures point spacing for resampling, downsampling, and embedding.
`Euclidean` is the standard distance; `SphereBundle` is calibrated for
position-and-direction trajectories built with `SphereBundleInterpolator`.

.. autoclass:: cycling_signatures.Euclidean
   :members:

.. autoclass:: cycling_signatures.SphereBundle
   :members:

Interpolation
-------------

An interpolator supplies the continuous curve fitted through a raw
trajectory, the same curve a dense trajectory then resamples from.
`CubicSpline` fits spatial positions while `SphereBundleInterpolator` wraps a
spatial interpolator to also carry a direction half.

.. autoclass:: cycling_signatures.CubicSpline
   :members:

.. autoclass:: cycling_signatures.SphereBundleInterpolator
   :members:

Trajectories and covers
-----------------------

`Trajectory` is the point-and-parameterization type behind the dense and
detection trajectories; `CubicalCover` is built from a dense trajectory and
computes its cover generators; `EmbeddedTrajectory` pairs a detection trajectory
with the cover it was embedded in. Building these in the wrong order silently
perforates the cover; see :doc:`concepts` for the required stage ordering.

.. autoclass:: cycling_signatures.Trajectory
   :members:

.. autoclass:: cycling_signatures.CubicalCover
   :members:

.. autoclass:: cycling_signatures.EmbeddedTrajectory
   :members:

Homology values
---------------

`HomologyClass` is a single cycle's class, expressed as coordinates in the
cover's generator basis; `Subspace` is a basis-canonicalized span of classes
used for comparing cycling results. `CyclingSignature` is the filtered subspace
a trajectory window visits, with per-generator births attached. Class vectors,
subspace equality, and containment are meaningful only with a fixed generating
basis; see :doc:`concepts` for what that restriction means and when it is safe
to compare across runs.

.. autoclass:: cycling_signatures.HomologyClass
   :members:

.. autoclass:: cycling_signatures.Subspace
   :members:

.. autoclass:: cycling_signatures.CyclingSignature
   :members:

Cycle storage
-------------

`CycleStorage` holds every detected cycle over a trajectory window, grouped
into `Component` instances that each carry one homology class. `Cycle` is a
single detected recurrent segment with its endpoint range and birth.

.. autoclass:: cycling_signatures.CycleStorage
   :members:

.. autoclass:: cycling_signatures.Component
   :members:

.. autoclass:: cycling_signatures.Cycle
   :members:

Errors
------

Most failures surface as standard Python exceptions: a file input/output failure
raises `OSError`, an out-of-range index raises `IndexError`, and every other
invalid-input or malformed-data failure raises `ValueError`.
`FormatVersionMismatchError` is the one exception this package defines itself,
raised when loading a saved cover, embedded trajectory, or cycle storage written
by an incompatible format version; regenerate the file with the current library
version rather than attempting to load it.

.. autoexception:: cycling_signatures.FormatVersionMismatchError
