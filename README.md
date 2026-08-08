# cycling-signatures

Algebraic topological descriptions of recurrent motions in high-dimensional dynamical systems.

Given a sampled trajectory from a dynamical system, this crate finds the
stretches that nearly return to where they started, then classifies each cycle
by the hole it encloses: two cycles winding the same way around the same
obstruction share a class. The classes a trajectory visits, filtered by an
adjacency threshold, form its cycling signature: a record of which loops the
orbit encompasses.

The signature was introduced by Bauer, Hien, Junge, and Mischaikow (2023),
[arXiv:2312.04734](https://arxiv.org/abs/2312.04734). This crate computes it
by embedding a trajectory into a cubical complex, a grid of unit cubes, and
finding classes via discrete Morse theory, built on
[CHomP3-rs](https://github.com/TravisCasey/CHomP3-rs).

## Installation

Add the dependency from its repository:

```toml
[dependencies]
cycling_signatures = { git = "https://github.com/TravisCasey/cycling-signatures" }
```

### Feature flags

- `serde` enables saving and loading a `CubicalCover` or `CycleStorage` to
  disk.
- `rayon` adds a shared-memory parallel execution backend for cover
  construction and cycle detection.
- `mpi` adds a distributed execution backend and implies `serde`, since
  coordinating work across processes serializes values between them.

## Getting started

Build the crate documentation locally and open it in a browser:

```bash
cargo doc --open
```

The example gallery renders worked pipelines end to end, including plots.
Build it from the `python` directory:

```bash
cd python
uv run --group docs --group examples sphinx-build -b html docs docs/_build/html
```

then open `docs/_build/html/index.html`. The gallery fetches its example
trajectories from [Zenodo record 21794612](https://zenodo.org/records/21794612)
on first use and caches them locally after that.

## Python bindings

The same pipeline is available from Python as `import cycling_signatures as cs`.
Build the bindings from source with [maturin](https://www.maturin.rs/):

```bash
cd python
uv run maturin develop --release
```

The example gallery above is the fastest way to see the Python API in use.

## License

GPL-3.0-or-later. See `LICENSE`.
