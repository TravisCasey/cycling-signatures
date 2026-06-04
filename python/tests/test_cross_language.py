from pathlib import Path

import cycling_signatures as cs

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load(metric):
    return cs.EmbeddedTrajectory.load(
        str(FIXTURES / "trajectory.cyc"), str(FIXTURES / "cover.cyc"), metric
    )


def test_load_rust_fixture_and_query():
    assert _load(cs.Euclidean()).signature(range(0, 201), 1.0).rank() == 1


def test_provenance_fingerprint_comparison():
    euclidean = _load(cs.Euclidean())
    storage = cs.CycleStorage.build(euclidean, range(0, 201), 1.0, 201)
    assert storage.fingerprint() == euclidean.fingerprint()
    assert storage.fingerprint() != _load(cs.Chebyshev()).fingerprint()
