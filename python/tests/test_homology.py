def test_signature_span_basis_round_trips_to_homology_classes(square_loop_storage):
    start, stop = square_loop_storage.extent()
    signature = square_loop_storage.signature((start, stop))
    # The loop spans one class, so its basis is exactly that stored class.
    assert signature.rank() == 1
    assert signature.span().basis() == [square_loop_storage.classes()[0]]


def test_trivial_signature_has_empty_basis(square_loop_storage):
    # A single-sample window encloses no cycle, so its signature is trivial.
    trivial = square_loop_storage.signature((0, 1))
    assert trivial.rank() == 0
    assert trivial.span().basis() == []


def test_homology_class_is_zero(square_loop_storage, square_loop_embedded):
    # The loop encircles the hole, so its class is non-trivial.
    assert not square_loop_storage.classes()[0].is_zero()
    # A short arc that closes without enclosing the hole is the trivial class.
    assert square_loop_embedded.cycle_class((0, 10)).is_zero()
