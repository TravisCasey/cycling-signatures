// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Curve interpolation traits and reference implementations.
//!
//! An [`Interpolator`] is fit at construction over a set of strictly increasing
//! parameter values (knots) and a corresponding matrix of sample values. After
//! construction, the interpolator is query-only.

use std::sync::Arc;

use ndarray::Array1;

pub mod cubic_spline;
pub mod sphere_bundle;

pub use cubic_spline::CubicSpline;
pub use sphere_bundle::SphereBundleInterpolator;

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
    ///
    /// Implementations must uphold that ordering: it is the order in which
    /// the curve is traversed, and a [`Trajectory`](crate::Trajectory) built
    /// by sampling the curve inherits it as its own parameterization, which
    /// is required to be strictly increasing.
    #[must_use]
    fn knots(&self) -> &[f64];
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

impl<T: Interpolator + ?Sized> Interpolator for Arc<T> {
    fn sample(&self, parameter: f64) -> Array1<f64> {
        (**self).sample(parameter)
    }

    fn knots(&self) -> &[f64] {
        (**self).knots()
    }
}

impl<T: DerivativeInterpolator + ?Sized> DerivativeInterpolator for Arc<T> {
    fn derivative(&self, parameter: f64) -> Array1<f64> {
        (**self).derivative(parameter)
    }
}

impl<T: Interpolator + ?Sized> Interpolator for &T {
    fn sample(&self, parameter: f64) -> Array1<f64> {
        (**self).sample(parameter)
    }

    fn knots(&self) -> &[f64] {
        (**self).knots()
    }
}

impl<T: DerivativeInterpolator + ?Sized> DerivativeInterpolator for &T {
    fn derivative(&self, parameter: f64) -> Array1<f64> {
        (**self).derivative(parameter)
    }
}

impl<T: Interpolator + ?Sized> Interpolator for Box<T> {
    fn sample(&self, parameter: f64) -> Array1<f64> {
        (**self).sample(parameter)
    }

    fn knots(&self) -> &[f64] {
        (**self).knots()
    }
}

impl<T: DerivativeInterpolator + ?Sized> DerivativeInterpolator for Box<T> {
    fn derivative(&self, parameter: f64) -> Array1<f64> {
        (**self).derivative(parameter)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::array;

    use super::{CubicSpline, DerivativeInterpolator, Interpolator};
    use crate::{metric::Metric, trajectory::Trajectory};

    #[test]
    fn arc_delegates_to_the_inner_interpolator() {
        let spline = CubicSpline::new(
            array![0.0, 1.0, 2.0, 3.0],
            array![[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 3.0]].view(),
        )
        .unwrap();
        let shared = Arc::new(spline.clone());

        // Generic dispatch is what needs the shared-pointer impl: `resample`
        // unifies its `&I` parameter with `&Arc<CubicSpline>`, which does not
        // deref away the way a direct method call would.
        let shared_resample = Trajectory::resample(&shared, Metric::Euclidean, 0.5).unwrap();
        let direct_resample = Trajectory::resample(&spline, Metric::Euclidean, 0.5).unwrap();
        assert_eq!(shared_resample.fingerprint(), direct_resample.fingerprint());

        // Delegation is exact, not merely close: a shared fit and the fit it
        // wraps are the same function.
        assert_eq!(shared.knots(), spline.knots());
        assert_eq!(shared.sample(1.5), spline.sample(1.5));
        assert_eq!(shared.derivative(1.5), spline.derivative(1.5));
    }

    #[test]
    fn boxed_dyn_interpolator_satisfies_the_bound() {
        let spline = CubicSpline::new(
            array![0.0, 1.0, 2.0, 3.0],
            array![[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 3.0]].view(),
        )
        .unwrap();
        let boxed: Box<dyn Interpolator> = Box::new(spline.clone());

        // `Box<dyn Interpolator>` unifies with the same generic `&I`
        // parameter that a concrete interpolator does, confirming the
        // forwarding impl is what makes a runtime-chosen curve usable here.
        let boxed_resample = Trajectory::resample(&boxed, Metric::Euclidean, 0.5).unwrap();
        let direct_resample = Trajectory::resample(&spline, Metric::Euclidean, 0.5).unwrap();
        assert_eq!(boxed_resample.fingerprint(), direct_resample.fingerprint());
    }
}
