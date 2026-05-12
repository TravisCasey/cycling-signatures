# cycling-signatures

Algebraic topological descriptions of recurrent motions in high-dimensional dynamical systems.

This crate computes the cycling signature invariants introduced in [Bauer, Hien, Junge, and Mischaikow (2023)](https://arxiv.org/abs/2312.04734). Given a sampled trajectory from a dynamical system, it identifies elementary recurrent motions by embedding the trajectory into a cubical complex and computing homological signatures of the resulting cycles. These signatures provide coarse-grained information about the structure of recurrent behavior, even in high-dimensional or noisy nonlinear systems.

Built on [`CHomP3-rs`](https://github.com/TravisCasey/CHomP3-rs) for cubical complex homology computation via discrete Morse theory.

## License

GPL-3.0-or-later. See `LICENSE`.
