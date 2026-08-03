from pathlib import Path

import numpy as np

import cycling_signatures as cs

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load(metric):
    return cs.EmbeddedTrajectory.load(
        str(FIXTURES / "trajectory.cyc"), str(FIXTURES / "cover.cyc"), metric
    )


def test_fixture_carries_the_index_parameterization():
    # The fixture is written by the Rust example under the default
    # parameterization, so the parameters must survive the round trip as the
    # point indices they were assigned.
    trajectory = cs.Trajectory.load(str(FIXTURES / "trajectory.cyc"))
    assert len(trajectory) == 201
    assert np.array_equal(trajectory.parameters(), np.arange(201, dtype=np.float64))


def test_load_rust_fixture_and_query():
    assert _load(cs.Euclidean()).signature(range(0, 201), 0.5).rank() == 1


def test_provenance_fingerprint_comparison():
    euclidean = _load(cs.Euclidean())
    storage = cs.CycleStorage.build(euclidean, range(0, 201), 201, threshold=0.5)
    assert storage.fingerprint() == euclidean.fingerprint()


def test_fingerprint_distinguishes_metric():
    # 4D sphere-bundle-valid square loop: positions step through adjacent
    # cubes; the direction half is constant.
    points = np.array(
        [
            [0.5, 0.5, 1.0, 0.5],
            [1.5, 0.5, 1.0, 0.5],
            [1.5, 1.5, 1.0, 0.5],
            [0.5, 1.5, 1.0, 0.5],
        ]
    )
    trajectory = cs.Trajectory(points)
    # Cube membership depends only on point coordinates, not the metric, so
    # one cover serves both embeddings.
    cover = cs.CubicalCover(trajectory)
    euclidean = cs.EmbeddedTrajectory(trajectory, cover, cs.Euclidean())
    sphere = cs.EmbeddedTrajectory(trajectory, cover, cs.SphereBundle())
    assert euclidean.fingerprint() != sphere.fingerprint()
