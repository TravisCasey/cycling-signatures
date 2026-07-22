// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Algebraic topological descriptions of recurrent motions in high-dimensional
//! dynamical systems.

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
    interpolation::{
        ChebyshevSphereBundleInterpolator, CubicSpline, DerivativeInterpolator, Interpolator,
    },
    metric::Metric,
    signature::{CyclingSignature, SignatureGenerator},
    storage::cycle_storage::{Component, Cycle, CycleStorage},
    trajectory::Trajectory,
};
