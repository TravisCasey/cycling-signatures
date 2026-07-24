# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Lorenz cycle storage from the raw trajectory.

Reads the Lorenz position trajectory, embeds it in full through the
sphere-bundle pipeline, and writes ``lorenz/data/lorenz_storage.cyc``: the
detected cycles the gallery queries. The trajectory is fetched from the Zenodo
published dataset and cached under ``lorenz/data`` on first use; a file already
present there is used as-is, and running ``integrate_lorenz.py`` regenerates
the same file in place.
"""

import sys
from pathlib import Path

import numpy as np

import cycling_signatures as cs

# The shared helper lives at the examples root, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support

# Sphere-bundle parameters are interdependent; see the SphereBundle metric docs
# for the rationale. The metric derives its direction weight from
# SPHERE_RADIUS_FLOOR, matching the interpolator's direction cubes. The resample
# bound is the resolution chosen from below: the build sweeps the trajectory's
# empirical adjacency bound and detects just under it, so the stored band runs
# from the resample bound up to that top. The bound is tuned against the raw
# trajectory's sampling interval (integrate_lorenz.py) so bisection inserts
# under two percent of extra points, and MAX_LENGTH caps cycles at several
# time units of samples. Boxsize is large enough that recurrences are frequent
# while the cover still resolves both holes.
BOXSIZE = 5.0
SPHERE_RADIUS_FLOOR = 3
RESAMPLE_BOUND = 0.45
MAX_LENGTH = 800


def build() -> tuple[Path, float, float]:
    """Build the storage; return its path, inserted fraction, and bound.

    The fraction is the share of bisection-inserted rows the resample
    added, relative to the original sample count. The bound is the
    trajectory's achieved resolution, the recorded band's lower end.
    """
    raw_path = _support.lorenz_raw()
    points = np.load(raw_path)
    sample_count = len(points)

    spline = cs.CubicSpline(np.arange(sample_count, dtype=np.float64), points / BOXSIZE)
    interpolator = cs.ChebyshevSphereBundleInterpolator(spline, SPHERE_RADIUS_FLOOR)
    metric = cs.SphereBundle(SPHERE_RADIUS_FLOOR)
    trajectory = cs.Trajectory.resample(interpolator, metric, RESAMPLE_BOUND)
    inserted_fraction = (len(trajectory.points()) - sample_count) / sample_count

    embedded = cs.EmbeddedTrajectory(trajectory, metric)
    storage = cs.CycleStorage.build(embedded, range(0, sample_count), MAX_LENGTH)
    target = raw_path.parent / "lorenz_storage.cyc"
    storage.save(target)
    return target, inserted_fraction, embedded.bound()


def report(storage_path: Path, inserted_fraction: float, achieved_bound: float) -> None:
    """Print a storage summary: size, contents, band, and resample cost."""
    storage = cs.CycleStorage.load(storage_path)
    print(f"lorenz_storage.cyc  {storage_path.stat().st_size / 1e6:.1f} MB")
    print(
        f"generators {storage.num_generators()}, "
        f"classes {len(storage.classes())}, components {len(storage.components())}"
    )
    print(
        f"band [{achieved_bound:.6f}, {storage.threshold():.6f}], "
        f"adjacency bound {storage.adjacency_bound():.6f}"
    )
    print(f"inserted points {inserted_fraction:.2%} of original samples")


if __name__ == "__main__":
    storage_path, inserted_fraction, achieved_bound = build()
    report(storage_path, inserted_fraction, achieved_bound)
