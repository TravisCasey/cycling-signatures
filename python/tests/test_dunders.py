import pytest


def test_storage_repr_reports_construction_parameters(square_loop_storage, square_loop_embedded):
    count = len(square_loop_embedded)
    # Threshold renders as a float, and the extent and counts are reported.
    assert repr(square_loop_storage) == (
        f"CycleStorage(extent=(0, {count}), components=1, classes=1, "
        f"threshold=0.5, max_length={count})"
    )


def test_storage_length_and_indexing(square_loop_storage):
    storage = square_loop_storage
    assert storage[-1].class_id() == storage[len(storage) - 1].class_id()
    with pytest.raises(IndexError):
        storage[len(storage)]
    with pytest.raises(IndexError):
        storage[-len(storage) - 1]


def test_component_length_and_indexing(square_loop_storage):
    component = square_loop_storage[0]
    assert component[-1].range() == component[len(component) - 1].range()
    with pytest.raises(IndexError):
        component[len(component)]


def test_homology_class_indexing_matches_dense_array(square_loop_storage):
    homology_class = square_loop_storage.classes()[0]
    dense = homology_class.to_array()
    assert [homology_class[index] for index in range(len(homology_class))] == list(dense)
    assert homology_class[-1] == homology_class[len(homology_class) - 1]
    with pytest.raises(IndexError):
        homology_class[len(homology_class)]


def test_value_type_reprs(square_loop_storage, square_loop_embedded):
    storage = square_loop_storage
    count = len(square_loop_embedded)
    signature = square_loop_embedded.signature(range(count), 0.5)

    assert repr(storage.classes()[0]) == "HomologyClass(generators=1, set={0})"
    assert repr(storage.signature((0, count))) == "CyclingSignature(rank=1, threshold_max=0.5)"
    assert (
        repr(storage[0])
        == f"Component(class_id=0, coverage=(0, {count}), cycles={len(storage[0])})"
    )
    assert repr(signature) == "CyclingSignature(rank=1, threshold_max=0.5)"


def test_equatable_types_hash_consistently_with_equality(square_loop_storage, square_loop_embedded):
    storage = square_loop_storage
    count = len(square_loop_embedded)

    # Distinct instances that compare equal must hash equal and deduplicate in a
    # set.
    pairs = [
        (storage.classes()[0], storage.homology_class(0)),
        (storage.signature((0, count)).span(), storage.signature((0, count)).span()),
    ]
    for left, right in pairs:
        assert left == right
        assert hash(left) == hash(right)
        assert len({left, right}) == 1

    # CyclingSignature has no equality of its own; two signatures built the
    # same way still compare equal through their spanned subspace.
    first = square_loop_embedded.signature(range(count), 0.5)
    second = square_loop_embedded.signature(range(count), 0.5)
    assert first.span() == second.span()


def test_cycle_equality_without_hashability(square_loop_storage):
    component = square_loop_storage[0]
    assert component[0] == component[0]
    assert component[0] != component[-1]
    with pytest.raises(TypeError):
        hash(component[0])
