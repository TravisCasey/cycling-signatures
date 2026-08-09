# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Algebraic topological descriptions of recurrent motions in high-dimensional
dynamical systems.

Given a sampled trajectory, this package finds the stretches that nearly return
to where they started and classifies each cycle by the hole it encloses, so
that two cycles winding the same way around the same obstruction share a class.
The classes a trajectory visits, filtered by adjacency threshold, form its
cycling signature.

The pipeline runs in four stages::

    import cycling_signatures as cs

    dense = cs.Trajectory.resample(interpolator, metric, RESAMPLE_SPACING)
    cover = cs.CubicalCover(dense)
    detection = dense.downsample(metric, DOWNSAMPLE_SPACING)
    embedded = cs.EmbeddedTrajectory(detection, cover, metric)
    storage = cs.CycleStorage.build(embedded, window, MAX_LENGTH, THRESHOLD)

Build the cover from the densest trajectory available. Building it from an
already-thinned trajectory validates successfully but perforates the cover,
reporting classes the curve does not have.
"""

from importlib.metadata import version
from typing import TypeAlias

from ._core import (
    Component,
    CubicalCover,
    CubicSpline,
    Cycle,
    CycleStorage,
    CyclingSignature,
    EmbeddedTrajectory,
    Euclidean,
    FormatVersionMismatchError,
    HomologyClass,
    SphereBundle,
    SphereBundleInterpolator,
    Subspace,
    Trajectory,
)

__version__ = version("cycling-signatures")

Segment: TypeAlias = range | tuple[int, int]
"""An index range: a Python range, or a half-open (start, stop) pair."""

Metric: TypeAlias = Euclidean | SphereBundle
"""A distance function for resampling, downsampling, and embedding."""

Interpolator: TypeAlias = CubicSpline | SphereBundleInterpolator
"""A continuous curve usable as the source for trajectory resampling."""

__all__ = [
    "Component",
    "CubicSpline",
    "CubicalCover",
    "Cycle",
    "CycleStorage",
    "CyclingSignature",
    "EmbeddedTrajectory",
    "Euclidean",
    "FormatVersionMismatchError",
    "HomologyClass",
    "Interpolator",
    "Metric",
    "Segment",
    "SphereBundle",
    "SphereBundleInterpolator",
    "Subspace",
    "Trajectory",
]
