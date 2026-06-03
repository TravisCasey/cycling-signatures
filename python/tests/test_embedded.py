import pytest

import cycling_signatures as cs


def test_signature_of_square_loop(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    signature = embedded.signature(range(0, square_loop_points.shape[0]), 1.0)
    assert signature.rank() == 1
    component_class = signature.components()[0].homology_class()
    assert signature.span().contains(component_class)


def test_threshold_below_bound_raises(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    with pytest.raises(ValueError):
        embedded.signature((0, square_loop_points.shape[0]), embedded.bound() / 2.0)


def test_save_load_roundtrip(tmp_path, square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    trajectory_path = str(tmp_path / "trajectory.cyc")
    cover_path = str(tmp_path / "cover.cyc")
    embedded.save(trajectory_path, cover_path)
    reloaded = cs.EmbeddedTrajectory.load(trajectory_path, cover_path, cs.Euclidean())
    assert reloaded.fingerprint() == embedded.fingerprint()
