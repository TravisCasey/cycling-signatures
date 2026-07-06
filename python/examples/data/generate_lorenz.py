# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Lorenz cycle storage from the raw trajectory.

Reads the full Lorenz position trajectory from ``lorenz/data/lorenz_raw.npy``
(or the path in ``CYCLING_SIGNATURES_LORENZ_RAW``), discards a leading possibly
off-attractor transient, embeds a swath through the sphere-bundle pipeline, and
writes ``lorenz/data/lorenz_storage.cyc``: the detected cycles the gallery
queries.

To reproduce from the published dataset, download ``lorenz_raw.npy`` into the
gallery's ``lorenz/data`` directory (or point the
``CYCLING_SIGNATURES_LORENZ_RAW`` environment variable at it) before running.

Usage::

    uv run --group examples examples/data/generate_lorenz.py
"""

import os
from pathlib import Path

import numpy as np

import cycling_signatures as cs

# Sphere-bundle parameters are interdependent; see SphereBundleMetric for the
# rationale. direction_weight equals the cover radius so the metric matches the
# radius-scaled direction cubes; the threshold stays under 1/sqrt(3) (~0.577)
# so cycle endpoints stay cube-adjacent; the resample bound equals the
# threshold; boxsize is large enough that recurrences are frequent while the
# cover still resolves both holes.
BOXSIZE = 5.0
SPHERE_HALFSPAN = 3
DIRECTION_WEIGHT = SPHERE_HALFSPAN + 0.5
RESAMPLE_BOUND = 0.55
THRESHOLD = 0.55
MAX_LENGTH = 500

# Leading samples discarded as off-attractor transient, and the number of
# samples after it that are embedded into the storage.
TRANSIENT = 10_000
SWATH = 400_000

CACHE = Path(__file__).resolve().parent.parent / "lorenz" / "data"


def _raw_trajectory() -> np.ndarray:
    """Return the raw Lorenz trajectory from the cache or an override path."""
    override = os.environ.get("CYCLING_SIGNATURES_LORENZ_RAW")
    source = Path(override) if override else CACHE / "lorenz_raw.npy"
    if not source.exists():
        raise FileNotFoundError(
            "raw Lorenz trajectory not found; place lorenz_raw.npy from the "
            "published dataset in the gallery's lorenz/data directory"
        )
    return np.load(source)


def build() -> Path:
    """Read the raw trajectory, build the storage, write and return its path."""
    points = _raw_trajectory()[TRANSIENT : TRANSIENT + SWATH]
    sample_count = len(points)

    spline = cs.CubicSpline(np.arange(sample_count, dtype=np.float64), points / BOXSIZE)
    interpolator = cs.ChebyshevSphereBundleInterpolator(spline, SPHERE_HALFSPAN)
    metric = cs.SphereBundle(DIRECTION_WEIGHT)
    trajectory = cs.Trajectory.resample(interpolator, metric, RESAMPLE_BOUND)

    embedded = cs.EmbeddedTrajectory(trajectory, metric)
    storage = cs.CycleStorage.build(embedded, range(0, sample_count), THRESHOLD, MAX_LENGTH)
    target = CACHE / "lorenz_storage.cyc"
    storage.save(target)
    return target


def report(storage_path: Path) -> None:
    """Print a storage summary: size, generators, classes, components."""
    storage = cs.CycleStorage.load(storage_path)
    print(f"lorenz_storage.cyc  {storage_path.stat().st_size / 1e6:.1f} MB")
    print(
        f"generators {storage.num_generators()}, "
        f"classes {len(storage.classes())}, components {len(storage.components())}"
    )


if __name__ == "__main__":
    report(build())
