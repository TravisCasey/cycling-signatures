// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Curve interpolation traits and reference implementations.
//!
//! An [`Interpolator`] is fit at construction over a set of strictly increasing
//! parameter values (knots) and a corresponding matrix of sample values. After
//! construction, the interpolator is query-only.

use ndarray::Array1;

pub mod cubic_spline;
pub mod sphere_bundle;

pub use cubic_spline::CubicSpline;
pub use sphere_bundle::ChebyshevSphereBundleInterpolator;

/// A query-only interpolator fit at construction over a set of knots.
pub trait Interpolator {
    /// The value of the interpolated curve at `parameter`.
    ///
    /// # Panics
    ///
    /// Implementations panic if `parameter` is outside the fitted domain
    /// `[knots[0], knots[last]]`.
    #[must_use]
    fn sample(&self, parameter: f64) -> Array1<f64>;

    /// The parameter values at which the curve was fit, in strictly
    /// increasing order.
    #[must_use]
    fn knots(&self) -> &[f64];

    /// The embedding dimension of the interpolating curve. The size of arrays
    /// output by [`Interpolator::sample`] must equal this value.
    #[must_use]
    fn dimension(&self) -> usize;
}

/// An interpolator that also exposes its first derivative.
pub trait DerivativeInterpolator: Interpolator {
    /// The first derivative of the interpolated curve at `parameter`.
    ///
    /// # Panics
    ///
    /// Implementations panic if `parameter` is outside the fitted domain.
    #[must_use]
    fn derivative(&self, parameter: f64) -> Array1<f64>;
}
