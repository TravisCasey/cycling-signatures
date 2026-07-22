from typing import Any, cast

import numpy as np
import pytest

import cycling_signatures as cs


def test_signature_of_square_loop(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    signature = embedded.signature(range(0, square_loop_points.shape[0]), 1.0)
    assert signature.rank() == 1
    generator_class = signature.generators()[0]
    assert signature.span().contains(generator_class)


def test_signature_with_omitted_threshold(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    signature = embedded.signature(range(0, square_loop_points.shape[0]))
    assert signature.rank_at(signature.threshold_max()) == signature.rank()
    assert signature.births() == sorted(signature.births())
    assert len(signature.births()) == signature.rank()


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


def test_signature_rejects_invalid_segment(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    with pytest.raises(ValueError):
        embedded.signature(cast(Any, "not a segment"), 1.0)
    with pytest.raises(ValueError):
        embedded.signature((5, 2), 1.0)


def test_cycle_class_rejects_short_segment(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    with pytest.raises(ValueError):
        embedded.cycle_class((0, 1))


def test_adjacency_bound_reports_band_minimum():
    points = np.array([[0.5], [1.5], [2.5], [3.5]])
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(points), cs.Euclidean())
    assert embedded.adjacency_bound(range(0, 4), max_length=4) == pytest.approx(2.0, abs=1e-12)
