"""Python bindings to the cycling-signatures Rust crate.

Cycling signatures are algebraic topological descriptions of recurrent motions
in dynamical systems.
"""

from importlib.metadata import version

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
    "SphereBundle",
    "SphereBundleInterpolator",
    "Subspace",
    "Trajectory",
]
