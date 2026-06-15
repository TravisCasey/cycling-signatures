import pytest

import cycling_signatures as cs


def test_metric_reprs_round_trip_construction():
    assert repr(cs.Euclidean()) == "Euclidean()"
    assert repr(cs.Chebyshev()) == "Chebyshev()"
    # A whole-number weight must render as a float, not as "1".
    assert repr(cs.SphereBundle(1.0)) == "SphereBundle(direction_weight=1.0)"
    assert repr(cs.SphereBundle(2.5)) == "SphereBundle(direction_weight=2.5)"


def _square_loop_storage(points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(points), cs.Euclidean())
    count = points.shape[0]
    return cs.CycleStorage.build(embedded, range(count), 1.0, count)


def test_storage_repr_reports_construction_parameters(square_loop_points):
    storage = _square_loop_storage(square_loop_points)
    count = square_loop_points.shape[0]
    # Threshold renders as a float, and the extent and counts are reported.
    assert repr(storage) == (
        f"CycleStorage(extent=(0, {count}), components=1, classes=1, "
        f"threshold=1.0, max_length={count})"
    )


def test_storage_length_and_indexing(square_loop_points):
    storage = _square_loop_storage(square_loop_points)
    assert len(storage) == len(storage.components())
    assert storage[0].class_id() == storage.components()[0].class_id()
    assert storage[-1].class_id() == storage[len(storage) - 1].class_id()
    with pytest.raises(IndexError):
        storage[len(storage)]
    with pytest.raises(IndexError):
        storage[-len(storage) - 1]


def test_component_length_and_indexing(square_loop_points):
    component = _square_loop_storage(square_loop_points)[0]
    assert len(component) == component.cycle_count()
    assert component[0].range() == component.cycles()[0].range()
    assert component[-1].range() == component[len(component) - 1].range()
    with pytest.raises(IndexError):
        component[len(component)]


def test_homology_class_indexing_matches_dense_array(square_loop_points):
    homology_class = _square_loop_storage(square_loop_points).classes()[0]
    dense = homology_class.to_array()
    assert [homology_class[index] for index in range(len(homology_class))] == list(dense)
    assert homology_class[-1] == homology_class[len(homology_class) - 1]
    with pytest.raises(IndexError):
        homology_class[len(homology_class)]


def test_value_type_reprs(square_loop_points):
    storage = _square_loop_storage(square_loop_points)
    count = square_loop_points.shape[0]
    signature = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean()).signature(
        range(count), 1.0
    )

    cycle = storage[0][0]
    cycle_start, cycle_stop = cycle.range()
    assert repr(cycle) == (
        f"Cycle(start={cycle_start}, stop={cycle_stop}, "
        f"birth={cycle.birth()!r}, length={cycle.length()})"
    )
    assert repr(storage.classes()[0]) == "HomologyClass(generators=1, set={0})"
    assert repr(storage.signature((0, count))) == "Subspace(rank=1, generators=1)"
    assert (
        repr(storage[0])
        == f"Component(class_id=0, coverage=(0, {count}), cycles={len(storage[0])})"
    )
    assert repr(signature) == "CyclingSignature(rank=1, components=1)"
    assert (
        repr(signature.components()[0])
        == f"CycleComponent(cycles={len(signature.components()[0].cycles())})"
    )


def test_equatable_types_hash_consistently_with_equality(square_loop_points):
    storage = _square_loop_storage(square_loop_points)
    count = square_loop_points.shape[0]

    def fresh_signature():
        embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
        return embedded.signature(range(count), 1.0)

    # Distinct instances that compare equal must hash equal and deduplicate in a
    # set. A signature hashes by its spanned subspace, matching its equality.
    pairs = [
        (storage.classes()[0], storage.homology_class(0)),
        (storage.signature((0, count)), storage.signature((0, count))),
        (fresh_signature(), fresh_signature()),
    ]
    for left, right in pairs:
        assert left == right
        assert hash(left) == hash(right)
        assert len({left, right}) == 1
