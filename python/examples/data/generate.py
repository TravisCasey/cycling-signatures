# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Lorenz storage fixture (storage.cyc) from the raw.csv data file.

raw.csv holds the Lorenz position samples the gallery uses for spatial display.
This script embeds that trajectory through the sphere-bundle pipeline and writes
storage.cyc, the detected cycles that the examples query.

Usage::

    uv run --group examples examples/data/generate.py
"""

from pathlib import Path

import numpy as np

import cycling_signatures as cs

DATA_DIRECTORY = Path(__file__).resolve().parent / "lorenz"

# These sphere-bundle parameters are interdependent; see SphereBundleMetric
# for the rationale. In brief: direction_weight equals the cover radius so the
# metric matches the radius-scaled direction cubes; the threshold stays under
# 1 / sqrt(3) ~ 0.577 so cycle endpoints stay cube-adjacent; the resample bound
# equals the threshold; and the boxsize is large enough that recurrences are
# frequent (rank-2 structure) while the cover still resolves both holes.
BOXSIZE = 5.0
SPHERE_HALFSPAN = 3
DIRECTION_WEIGHT = SPHERE_HALFSPAN + 0.5
RESAMPLE_BOUND = 0.55
THRESHOLD = 0.55
MAX_LENGTH = 500


def build() -> Path:
    """Reads raw.csv, builds storage, writes storage.cyc, returns its path."""
    raw = np.loadtxt(DATA_DIRECTORY / "raw.csv", delimiter=",")
    sample_count = len(raw)

    spline = cs.CubicSpline(np.arange(sample_count, dtype=np.float64), raw / BOXSIZE)
    interpolator = cs.ChebyshevSphereBundleInterpolator(spline, SPHERE_HALFSPAN)
    metric = cs.SphereBundle(DIRECTION_WEIGHT)
    trajectory = cs.Trajectory.resample(interpolator, metric, RESAMPLE_BOUND)

    embedded = cs.EmbeddedTrajectory(trajectory, metric)
    storage = cs.CycleStorage.build(embedded, range(0, sample_count), THRESHOLD, MAX_LENGTH)
    storage_path = DATA_DIRECTORY / "storage.cyc"
    storage.save(storage_path)
    return storage_path


def report(storage_path: Path) -> None:
    """Prints a storage summary: size, generator count, and rank by window."""
    storage = cs.CycleStorage.load(storage_path)
    start, stop = storage.extent()
    generators = storage.num_generators()
    ranks = {
        length: max(
            storage.signature(range(time, time + length)).rank()
            for time in range(start, stop - length, 500)
        )
        for length in (160, 210, 300)
    }
    classes = len(storage.classes())
    components = len(storage.components())
    print(f"storage.cyc  {storage_path.stat().st_size / 1e6:.1f} MB")
    print(f"generators {generators}, classes {classes}, components {components}")
    print(f"max rank by window {ranks}")


if __name__ == "__main__":
    report(build())
