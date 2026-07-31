# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Dadras cycle storage from the raw trajectory.

Reads the Dadras position trajectory, embeds it in full through the
sphere-bundle pipeline, and writes ``dadras/data/dadras_storage.cyc``: the
detected cycles the gallery queries. The trajectory is fetched from the Zenodo
published dataset and cached under ``dadras/data`` on first use; a file already
present there is used as-is, and running ``integrate_dadras.py`` regenerates
the same file in place.
"""

import sys
from pathlib import Path

import numpy as np

import cycling_signatures as cs

# The shared helper lives at the examples root, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support

# Sphere-bundle parameters are interdependent; see the SphereBundle metric
# docs for the rationale. SPHERE_RADIUS sets the interpolator's direction
# normalization radius, which the metric measures against directly. The
# resample bound is the resolution chosen from below: the build sweeps the
# trajectory's empirical adjacency bound and detects just under it, so the
# stored band runs from the resample bound up to that top. The bound is tuned
# against the raw trajectory's sample spacing (integrate_dadras.py) so
# bisection inserts about five percent of extra points. The raw samples are
# spaced by distance rather than time, so MAX_LENGTH caps cycles by their
# length through state space. Boxsize is large enough that recurrences are
# frequent while the cover still resolves the attractor's holes.
BOXSIZE = 12.0
SPHERE_RADIUS = 3.5
RESAMPLE_BOUND = 0.45
MAX_LENGTH = 1600


def build() -> tuple[Path, float, float]:
    """Build the storage; return its path, inserted fraction, and bound.

    The fraction is the share of bisection-inserted rows the resample
    added, relative to the original sample count. The bound is the
    trajectory's achieved resolution, the recorded band's lower end.
    """
    raw_path = _support.dadras_raw()
    points = np.load(raw_path)
    sample_count = len(points)

    spline = cs.CubicSpline(np.arange(sample_count, dtype=np.float64), points / BOXSIZE)
    interpolator = cs.SphereBundleInterpolator(spline, SPHERE_RADIUS)
    metric = cs.SphereBundle()
    trajectory = cs.Trajectory.resample(interpolator, metric, RESAMPLE_BOUND)
    inserted_fraction = (trajectory.point_count() - sample_count) / sample_count

    embedded = cs.EmbeddedTrajectory(trajectory, metric)
    del points, spline, interpolator, trajectory

    storage = cs.CycleStorage.build(embedded, range(0, sample_count), MAX_LENGTH)
    target = raw_path.parent / "dadras_storage.cyc"
    storage.save(target)
    return target, inserted_fraction, embedded.bound()


def report(storage_path: Path, inserted_fraction: float, achieved_bound: float) -> None:
    """Print a storage summary: size, contents, band, and resample cost."""
    storage = cs.CycleStorage.load(storage_path)
    print(f"dadras_storage.cyc  {storage_path.stat().st_size / 1e6:.1f} MB")
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
