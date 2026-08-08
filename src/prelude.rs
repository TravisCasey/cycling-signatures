// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Ergonomic glob-import of the core types needed for a typical workflow.
//!
//! ```
//! use cycling_signatures::prelude::*;
//! use ndarray::array;
//!
//! let values = array![[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]];
//! let spline = CubicSpline::with_integer_knots(values.view()).unwrap();
//! let _ = spline.derivative(1.0);
//! ```

pub use crate::{
    Chain, Component, Cube, CubicSpline, CubicalCover, Cycle, CycleStorage, CyclingSignature,
    DerivativeInterpolator, EmbeddedTrajectory, Error, ExecutionBackend, F2, F2Subspace, F2Vector,
    Interpolator, Metric, Result, SignatureGenerator, SphereBundleInterpolator, Trajectory,
};
