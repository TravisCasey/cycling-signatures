// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Algebraic topological descriptions of recurrent motions in high-dimensional
//! dynamical systems.

pub mod error;
pub mod f2_subspace;
pub mod f2_vector;
pub mod interpolation;
pub mod metric;
pub mod prelude;

pub use chomp3rs::{Cyclic, F2};

pub use crate::{
    error::{Error, Result},
    f2_subspace::F2Subspace,
    f2_vector::F2Vector,
    interpolation::{
        ChebyshevSphereBundleInterpolator, CubicSpline, DerivativeInterpolator, Interpolator,
    },
    metric::{Chebyshev, Euclidean, Metric, SphereBundleMetric},
};
