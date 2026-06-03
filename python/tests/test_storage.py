import cycling_signatures as cs


def test_build_and_query(square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    count = square_loop_points.shape[0]
    storage = cs.CycleStorage.build(embedded, range(0, count), 1.0, count)
    assert storage.fingerprint() == embedded.fingerprint()
    assert storage.extent() == (0, count)
    assert storage.signature((0, count)).rank() >= 1


def test_save_load_roundtrip(tmp_path, square_loop_points):
    embedded = cs.EmbeddedTrajectory(cs.Trajectory(square_loop_points), cs.Euclidean())
    count = square_loop_points.shape[0]
    storage = cs.CycleStorage.build(embedded, (0, count), 1.0, count)
    path = str(tmp_path / "storage.cyc")
    storage.save(path)
    assert cs.CycleStorage.load(path).fingerprint() == storage.fingerprint()
