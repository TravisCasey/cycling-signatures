from typing import Any, cast

import pytest

import cycling_signatures as cs


def test_signature_of_square_loop(square_loop_embedded, square_loop_points):
    signature = square_loop_embedded.signature(range(0, square_loop_points.shape[0]), 0.5)
    assert signature.rank() == 1
    generator_class = signature.classes()[0]
    assert signature.span().contains(generator_class)


def test_threshold_below_resolution_raises(square_loop_embedded, square_loop_points):
    with pytest.raises(ValueError):
        square_loop_embedded.signature(
            (0, square_loop_points.shape[0]), square_loop_embedded.resolution() / 2.0
        )


def test_save_load_roundtrip(tmp_path, square_loop_embedded):
    trajectory_path = str(tmp_path / "trajectory.cyc")
    cover_path = str(tmp_path / "cover.cyc")
    square_loop_embedded.save(trajectory_path, cover_path)
    reloaded = cs.EmbeddedTrajectory.load(trajectory_path, cover_path, cs.Euclidean())
    assert reloaded.fingerprint() == square_loop_embedded.fingerprint()


def test_signature_rejects_invalid_segment(square_loop_embedded):
    with pytest.raises(ValueError):
        square_loop_embedded.signature(cast(Any, "not a segment"), 0.5)
    with pytest.raises(ValueError):
        square_loop_embedded.signature((5, 2), 0.5)


def test_cycle_class_rejects_short_segment(square_loop_embedded):
    with pytest.raises(ValueError):
        square_loop_embedded.cycle_class((0, 1))


def test_signature_rejects_threshold_at_cube_side(square_loop_embedded, square_loop_points):
    with pytest.raises(ValueError):
        square_loop_embedded.signature(range(0, square_loop_points.shape[0]), 1.0)


def test_cover_missing_the_trajectory_cubes_is_rejected(square_loop_points):
    # Pairing a trajectory with a cover of somewhere else entirely: the
    # cover holds none of the cubes the trajectory needs.
    trajectory = cs.Trajectory(square_loop_points)
    elsewhere = cs.Trajectory(square_loop_points + 100.0)
    with pytest.raises(ValueError):
        cs.EmbeddedTrajectory(trajectory, cs.CubicalCover(elsewhere), cs.Euclidean())


def test_cover_fingerprint_depends_only_on_the_cube_set(square_loop_points):
    trajectory = cs.Trajectory(square_loop_points)
    elsewhere = cs.Trajectory(square_loop_points + 100.0)
    assert cs.CubicalCover(trajectory).fingerprint() == cs.CubicalCover(trajectory).fingerprint()
    assert cs.CubicalCover(trajectory).fingerprint() != cs.CubicalCover(elsewhere).fingerprint()
