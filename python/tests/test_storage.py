import pytest

import cycling_signatures as cs


def test_build_and_query(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    count = square_loop_points.shape[0]
    storage = cs.CycleStorage.build(embedded, range(0, count), count, threshold=0.5)
    assert storage.fingerprint() == embedded.fingerprint()
    assert storage.extent() == (0, count)
    assert storage.signature((0, count)).rank() == 1


def test_save_load_roundtrip(tmp_path, square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    count = square_loop_points.shape[0]
    storage = cs.CycleStorage.build(embedded, (0, count), count, threshold=0.5)
    path = tmp_path / "storage.cyc"  # a pathlib.Path, exercising os.PathLike acceptance
    storage.save(path)
    assert cs.CycleStorage.load(path).fingerprint() == storage.fingerprint()


def test_num_generators_matches_signature(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    count = square_loop_points.shape[0]
    storage = cs.CycleStorage.build(embedded, range(0, count), count, threshold=0.5)
    assert storage.num_generators() > 0
    assert storage.num_generators() == storage.signature((0, count)).num_generators()


def test_component_index_out_of_bounds_raises(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    count = square_loop_points.shape[0]
    storage = cs.CycleStorage.build(embedded, range(0, count), count, threshold=0.5)
    with pytest.raises(IndexError):
        storage.component(9999)


def test_build_records_the_threshold_passed(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    count = square_loop_points.shape[0]
    storage = cs.CycleStorage.build(embedded, range(0, count), count, threshold=0.5)
    assert storage.threshold() == pytest.approx(0.5)
    assert storage.signature((0, count)).rank() == 1


def test_span_at_matches_rank_at_and_rejects_above_threshold_max(square_loop_storage):
    signature = square_loop_storage.signature(square_loop_storage.extent())
    interior_threshold = signature.births()[0]

    assert signature.span_at(interior_threshold).rank() == signature.rank_at(interior_threshold)

    threshold_max = signature.threshold_max()
    with pytest.raises(ValueError):
        signature.rank_at(threshold_max + 1.0)
    with pytest.raises(ValueError):
        signature.span_at(threshold_max + 1.0)
