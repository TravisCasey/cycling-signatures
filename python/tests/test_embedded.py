from typing import Any, cast

import numpy as np
import pytest

import cycling_signatures as cs


def test_signature_of_square_loop(square_loop_embedded, square_loop_points):
    signature = square_loop_embedded.signature(range(0, square_loop_points.shape[0]), 0.5)
    assert signature.rank() == 1
    generator_class = signature.classes()[0]
    assert signature.span().contains(generator_class)
    # The cover accessor shares the same cover the trajectory was embedded
    # against, which also has exactly one generator around the loop's hole.
    assert square_loop_embedded.cover().num_generators() == 1


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


def test_cover_len_and_num_generators_match_visited_cubes():
    # Boundary of a 4x4 grid of unit cubes, leaving a 2x2 hole in the middle:
    # 12 distinct visited cubes and exactly one first-homology generator
    # around the hole.
    cube_order = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (2, 3),
        (1, 3),
        (0, 3),
        (0, 2),
        (0, 1),
    ]
    centers = np.array([[x + 0.5, y + 0.5] for x, y in cube_order])
    points = np.concatenate([centers, centers[:1]])
    cover = cs.CubicalCover(cs.Trajectory(points))
    assert len(cover) == 12
    assert cover.num_generators() == 1


def test_cover_save_load_roundtrip(tmp_path, square_loop_points):
    cover = cs.CubicalCover(cs.Trajectory(square_loop_points))
    path = str(tmp_path / "cover.cyc")
    cover.save(path)
    reloaded = cs.CubicalCover.load(path)
    assert reloaded.fingerprint() == cover.fingerprint()
    assert len(reloaded) == len(cover)
    assert reloaded.num_generators() == cover.num_generators()
