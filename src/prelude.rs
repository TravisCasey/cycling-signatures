// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Ergonomic re-exports for common types and traits.
//!
//! Glob-import for the core types needed in a typical workflow. Mirrors the
//! crate root re-exports. Specialized internals are not included.
//!
//! ```
//! use cycling_signatures::prelude::*;
//! ```

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
