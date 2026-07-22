# This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
# See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

"""Build the Dadras cycle storage from the raw trajectory.

Reads the Dadras position trajectory, embeds it in full through the
sphere-bundle pipeline, and writes ``dadras/data/dadras_storage.cyc``: the
detected cycles the gallery queries. The trajectory is fetched from the Zenodo
published dataset and cached under ``dadras/data`` on first use; a file already
present there is used as-is.
"""

import sys
from pathlib import Path

import numpy as np

import cycling_signatures as cs

# The shared helper lives at the examples root, one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support

# Sphere-bundle parameters are interdependent; see the SphereBundle metric
# docs for the rationale. The metric derives its direction weight from
# SPHERE_RADIUS_FLOOR, matching the interpolator's direction cubes; the
# resample bound equals the threshold; boxsize is large enough that
# recurrences are frequent while the cover still resolves the attractor's
# holes. The threshold must stay below the trajectory's empirical adjacency
# bound so cycle endpoints land in adjacent cubes.
BOXSIZE = 12.0
SPHERE_RADIUS_FLOOR = 3
RESAMPLE_BOUND = 0.45
THRESHOLD = 0.45
MAX_LENGTH = 600


def build() -> Path:
    """Fetch the raw trajectory, build the storage, and return its path."""
    raw_path = _support.dadras_raw()
    points = np.load(raw_path)
    sample_count = len(points)

    spline = cs.CubicSpline(np.arange(sample_count, dtype=np.float64), points / BOXSIZE)
    interpolator = cs.ChebyshevSphereBundleInterpolator(spline, SPHERE_RADIUS_FLOOR)
    metric = cs.SphereBundle(SPHERE_RADIUS_FLOOR)
    trajectory = cs.Trajectory.resample(interpolator, metric, RESAMPLE_BOUND)

    embedded = cs.EmbeddedTrajectory(trajectory, metric)
    storage = cs.CycleStorage.build(
        embedded, range(0, sample_count), MAX_LENGTH, threshold=THRESHOLD
    )
    target = raw_path.parent / "dadras_storage.cyc"
    storage.save(target)
    return target


def report(storage_path: Path) -> None:
    """Print a storage summary: size, generators, classes, components."""
    storage = cs.CycleStorage.load(storage_path)
    print(f"dadras_storage.cyc  {storage_path.stat().st_size / 1e6:.1f} MB")
    print(
        f"generators {storage.num_generators()}, "
        f"classes {len(storage.classes())}, components {len(storage.components())}"
    )


if __name__ == "__main__":
    report(build())
