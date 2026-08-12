import pytest

import cycling_signatures as cs


def test_build_and_query(square_loop_storage, square_loop_embedded):
    count = len(square_loop_embedded)
    assert square_loop_storage.fingerprint() == square_loop_embedded.fingerprint()
    assert square_loop_storage.extent() == (0, count)
    assert square_loop_storage.signature((0, count)).rank() == 1


def test_save_load_roundtrip(tmp_path, square_loop_storage):
    path = tmp_path / "storage.cyc"
    square_loop_storage.save(path)
    assert cs.CycleStorage.load(path).fingerprint() == square_loop_storage.fingerprint()


def test_num_generators_matches_signature(square_loop_storage, square_loop_embedded):
    count = len(square_loop_embedded)
    assert square_loop_storage.num_generators() > 0
    assert (
        square_loop_storage.num_generators()
        == square_loop_storage.signature((0, count)).num_generators()
    )


def test_component_index_out_of_bounds_raises(square_loop_storage):
    with pytest.raises(IndexError):
        square_loop_storage.component(9999)


def test_component_negative_index_raises_but_getitem_wraps(square_loop_storage):
    with pytest.raises(IndexError):
        square_loop_storage.component(-1)
    assert (
        square_loop_storage[-1].class_id()
        == square_loop_storage.component(len(square_loop_storage) - 1).class_id()
    )


def test_components_covering_negative_point_returns_empty(square_loop_storage):
    assert square_loop_storage.components_covering(-1) == []


def test_components_covering_reports_the_component_holding_the_point(square_loop_storage):
    # The loop's cycles all belong to one component spanning the whole extent,
    # so every point in it reports that component.
    _, extent_stop = square_loop_storage.extent()
    for point in (0, extent_stop // 2, extent_stop - 1):
        assert square_loop_storage.components_covering(point) == [0]


def test_segment_range_with_a_step_other_than_one_raises(square_loop_storage):
    with pytest.raises(ValueError):
        square_loop_storage.signature(range(0, 10, 2))


def test_segment_with_a_negative_bound_raises(square_loop_storage):
    with pytest.raises(ValueError, match="must be non-negative"):
        square_loop_storage.signature((-1, 10))


def test_signature_out_of_range_raises_index_error_not_value_error(square_loop_storage):
    with pytest.raises(IndexError):
        square_loop_storage.signature((0, 9999))
    # A malformed segment (start past stop) raises ValueError.
    with pytest.raises(ValueError):
        square_loop_storage.signature((5, 2))


def test_build_records_the_threshold_passed(square_loop_storage):
    assert square_loop_storage.threshold() == pytest.approx(0.5)


def test_span_at_matches_rank_at_and_rejects_above_threshold_max(square_loop_storage):
    signature = square_loop_storage.signature(square_loop_storage.extent())
    interior_threshold = signature.births()[0]

    assert signature.span_at(interior_threshold).rank() == signature.rank_at(interior_threshold)

    threshold_max = signature.threshold_max()
    with pytest.raises(ValueError):
        signature.rank_at(threshold_max + 1.0)
    with pytest.raises(ValueError):
        signature.span_at(threshold_max + 1.0)
