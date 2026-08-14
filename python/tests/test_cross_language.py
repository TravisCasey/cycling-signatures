from pathlib import Path

import numpy as np

import cycling_signatures as cs

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

# The point count the Rust example writes into the fixture.
FIXTURE_POINTS = 201


def _load():
    return cs.EmbeddedTrajectory.load(
        str(FIXTURES / "embedded.cyc"),
        str(FIXTURES / "trajectory.cyc"),
        str(FIXTURES / "cover.cyc"),
    )


def test_fixture_carries_the_index_parameterization():
    # The fixture is written by the Rust example under the default
    # parameterization, so the parameters must survive the round trip as the
    # point indices they were assigned.
    trajectory = cs.Trajectory.load(str(FIXTURES / "trajectory.cyc"))
    assert len(trajectory) == FIXTURE_POINTS
    assert np.array_equal(trajectory.parameters(), np.arange(len(trajectory), dtype=np.float64))


def test_load_rust_fixture_and_query():
    embedded = _load()
    assert embedded.signature(range(len(embedded))).rank() == 1


def test_fixture_fingerprints_match_the_values_rust_wrote():
    # Both fingerprints are computed by the Rust crate and read back here, so
    # these literals pin the fixture's identity across the language boundary:
    # a fixture regenerated from different points, a different
    # parameterization or a different metric reports different numbers.
    trajectory = cs.Trajectory.load(str(FIXTURES / "trajectory.cyc"))
    assert trajectory.fingerprint() == 0x6B18D642CAD11553
    assert _load().fingerprint() == 0x8848B44015C73B7C


def test_fingerprint_distinguishes_metric():
    # 4D sphere-bundle-valid square loop: positions step through adjacent
    # cubes; the direction half is constant.
    points = np.array(
        [
            [0.9, 0.9, 1.0, 0.5],
            [1.1, 0.9, 1.0, 0.5],
            [1.1, 1.1, 1.0, 0.5],
            [0.9, 1.1, 1.0, 0.5],
        ]
    )
    trajectory = cs.Trajectory(points)
    # Cube membership depends only on point coordinates, not the metric, so
    # one cover serves both embeddings.
    cover = cs.CubicalCover(trajectory)
    euclidean = cs.EmbeddedTrajectory(trajectory, cover, cs.Euclidean())
    sphere = cs.EmbeddedTrajectory(trajectory, cover, cs.SphereBundle())
    assert euclidean.fingerprint() != sphere.fingerprint()
