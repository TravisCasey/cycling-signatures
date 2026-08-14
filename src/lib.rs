// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Algebraic topological descriptions of recurrent motions in high-dimensional
//! dynamical systems.
//!
//! Given a sampled trajectory, this crate finds the stretches that nearly
//! return to where they started, then classifies each cycle by the hole it
//! encloses: two cycles winding the same way around the same obstruction
//! share a class. The classes a trajectory visits, filtered by adjacency
//! threshold, form its cycling signature ([`CyclingSignature`]).
//!
//! # Pipeline
//!
//! The pipeline runs in four stages: resample to a dense trajectory, build
//! the cover from it, downsample to the sparser detection trajectory, then
//! embed and store its cycles.
//!
//! ```no_run
//! use cycling_signatures::prelude::*;
//! use ndarray::array;
//!
//! const RESAMPLE_SPACING: f64 = 0.1;
//! const DOWNSAMPLE_SPACING: f64 = 0.3;
//!
//! let knots = array![0.0, 1.0, 2.0, 3.0, 4.0];
//! let values =
//!     array![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]];
//! let spline = CubicSpline::new(knots, values.view()).unwrap();
//! let metric = Metric::Euclidean;
//! let backend = ExecutionBackend::default();
//!
//! let dense =
//!     Trajectory::resample(&spline, metric, RESAMPLE_SPACING).unwrap();
//! let cover = CubicalCover::build(&dense, &backend).unwrap();
//! let detection = dense.downsample(metric, DOWNSAMPLE_SPACING).unwrap();
//! let embedded = EmbeddedTrajectory::new(detection, cover, metric).unwrap();
//! let storage = CycleStorage::build(&embedded, .., 100, &backend).unwrap();
//! assert!(!storage.components().is_empty());
//! ```
//!
//! # Build the cover from the densest trajectory
//!
//! Build the cover from the densest trajectory available. Building it from
//! an already-thinned trajectory type-checks and validates, but perforates
//! the cover, leaving spurious holes that report first-homology classes the
//! curve does not have.
//!
//! # Resolution and the cube side
//!
//! Cubes have unit side length. Callers scale a trajectory's coordinates into
//! cube units before constructing it; the crate does not scale or center them
//! itself. The resample spacing sets cover fidelity; the downsample spacing
//! sets detection resolution and is the primary cost-affecting parameter.
//!
//! Detected cycles have endpoints strictly closer than 1 in the designated
//! metrics. A signature admits queries over the filtration band `[0, 1]`,
//! which closes at the same value. An embedded trajectory's own resolution
//! ([`EmbeddedTrajectory::resolution`]) must stay below the cube side as well;
//! see [`Error::ResolutionNotBelowCubeSide`] for details.
//!
//! # Comparing output between runs
//!
//! Cover generators (a cover's `F_2` cohomology generators) are not
//! computed in the same basis from one run to the next. Class vectors,
//! subspaces, and containment checks ([`F2Subspace::contains`]) are
//! meaningful only within one basis: one run, or every process loading the
//! same saved [`CubicalCover`] file.
//!
//! # Feature flags
//!
//! `serde` enables saving and loading a [`CubicalCover`] or [`CycleStorage`]
//! to disk. `rayon` adds a shared-memory parallel [`ExecutionBackend`] variant
//! for cover construction and cycle detection. `mpi` adds a distributed
//! [`ExecutionBackend`] variant and implies `serde`, since coordinating work
//! across processes serializes values between them.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod cover;
pub(crate) mod distance;
pub mod embedded;
pub mod error;
pub mod f2_subspace;
pub mod f2_vector;
pub mod interpolation;
pub mod metric;
pub mod prelude;
#[cfg(feature = "serde")]
pub(crate) mod serialization;
pub mod signature;
pub mod storage;
pub mod trajectory;
pub(crate) mod util;

pub use chomp3rs::{Chain, Cube, ExecutionBackend, F2};

pub use crate::{
    cover::CubicalCover,
    embedded::EmbeddedTrajectory,
    error::{Error, Result},
    f2_subspace::F2Subspace,
    f2_vector::F2Vector,
    interpolation::{CubicSpline, DerivativeInterpolator, Interpolator, SphereBundleInterpolator},
    metric::Metric,
    signature::{CyclingSignature, SignatureGenerator},
    storage::{Component, Cycle, CycleStorage},
    trajectory::Trajectory,
};
