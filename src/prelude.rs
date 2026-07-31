// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Ergonomic re-exports for common types and traits.
//!
//! Glob-import for the core types needed in a typical workflow.
//!
//! ```
//! use cycling_signatures::prelude::*;
//! ```

pub use crate::{
    Component, Cube, CubicSpline, CubicalCover, Cycle, CycleStorage, CyclingSignature,
    EmbeddedTrajectory, Error, ExecutionBackend, F2Subspace, F2Vector, Interpolator, Metric,
    Result, SignatureGenerator, SphereBundleInterpolator, Trajectory,
};
