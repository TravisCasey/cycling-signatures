"""Python bindings to the cycling-signatures Rust crate.

Cycling signatures are algebraic topological descriptions of recurrent motions
in dynamical systems.
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
