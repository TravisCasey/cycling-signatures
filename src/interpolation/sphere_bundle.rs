// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Sphere-bundle interpolator wrapper.

use ndarray::Array1;

use crate::interpolation::{DerivativeInterpolator, Interpolator};

/// Wraps a [`DerivativeInterpolator`] and produces samples that concatenate
/// the inner position with a scaled direction.
///
/// At each parameter, the scaled direction is the inner derivative normalized
/// by its Chebyshev norm, then multiplied by the configured radius. Each
/// sample has length twice that of the inner interpolator: the first half is
/// the position, the second half is the scaled direction.
///
/// The radius is set indirectly through a `cube_halfspan: u32` and is fixed at
/// `cube_halfspan + 0.5`. The half-integer offset keeps every direction
/// coordinate strictly between two integers, so a coordinate of an extremal
/// component never lands exactly where the floor function is sign-asymmetric.
#[derive(Debug, Clone)]
pub struct ChebyshevSphereBundleInterpolator<Inner> {
    inner: Inner,
    radius: f64,
}

impl<Inner: DerivativeInterpolator> ChebyshevSphereBundleInterpolator<Inner> {
    /// Wraps `inner` with the given cube halfspan.
    ///
    /// The actual normalization radius is `cube_halfspan + 0.5`; see the
    /// type-level documentation for the cubical-embedding reason. The radius
    /// is recoverable via [`radius`](Self::radius).
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::interpolation::{
    ///     ChebyshevSphereBundleInterpolator, CubicSpline, Interpolator,
    /// };
    /// use ndarray::array;
    ///
    /// let inner = CubicSpline::new(
    ///     array![0.0, 1.0, 2.0],
    ///     array![[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]].view(),
    /// )
    /// .unwrap();
    /// let bundle = ChebyshevSphereBundleInterpolator::new(inner, 1);
    /// assert_eq!(bundle.radius(), 1.5);
    /// let sample = bundle.sample(0.5);
    /// // Output is the inner sample concatenated with the scaled direction.
    /// assert_eq!(sample.len(), 4);
    /// ```
    #[must_use]
    pub fn new(inner: Inner, cube_halfspan: u32) -> Self {
        let radius = f64::from(cube_halfspan) + 0.5;
        Self { inner, radius }
    }

    /// The normalization radius (`cube_halfspan + 0.5` from construction).
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.radius
    }
}

impl<Inner: DerivativeInterpolator> Interpolator for ChebyshevSphereBundleInterpolator<Inner> {
    /// # Panics
    ///
    /// Panics if `parameter` is outside the fitted domain, or if the inner
    /// derivative is zero at the supplied parameter.
    fn sample(&self, parameter: f64) -> Array1<f64> {
        let position = self.inner.sample(parameter);
        let derivative = self.inner.derivative(parameter);

        let linf_norm = derivative
            .iter()
            .map(|component| component.abs())
            .fold(0.0_f64, f64::max);
        assert!(linf_norm > 0.0, "zero derivative at parameter");

        let scaled: Array1<f64> = derivative.mapv(|component| component / linf_norm * self.radius);

        let dimension = position.len();
        let mut result = Array1::<f64>::zeros(2 * dimension);
        result.slice_mut(ndarray::s![..dimension]).assign(&position);
        result.slice_mut(ndarray::s![dimension..]).assign(&scaled);
        result
    }

    fn knots(&self) -> &[f64] {
        self.inner.knots()
    }

    fn dimension(&self) -> usize {
        2 * self.inner.dimension()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::ChebyshevSphereBundleInterpolator;
    use crate::interpolation::{CubicSpline, Interpolator};

    #[test]
    fn sample_concatenates_position_and_scaled_direction() {
        // Three properties asserted together: output length is 2 * inner dim,
        // the spatial half matches an independently constructed inner sample,
        // and the direction half has L-infinity norm equal to the radius.
        let knots = array![0.0, 1.0, 2.0, 3.0];
        let values = array![[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 3.0]];
        let inner = CubicSpline::new(knots.clone(), values.view()).unwrap();
        let bundle = ChebyshevSphereBundleInterpolator::new(inner, 2);
        let radius = bundle.radius();
        let plain = CubicSpline::new(knots, values.view()).unwrap();

        for parameter in [0.5, 1.0, 1.5, 2.0, 2.5] {
            let sample = bundle.sample(parameter);
            assert_eq!(sample.len(), 4);
            let plain_sample = plain.sample(parameter);
            assert!((sample[0] - plain_sample[0]).abs() < 1e-12);
            assert!((sample[1] - plain_sample[1]).abs() < 1e-12);
            let direction_linf = sample
                .iter()
                .skip(2)
                .map(|component| component.abs())
                .fold(0.0_f64, f64::max);
            assert!(
                (direction_linf - radius).abs() < 1e-10,
                "L-infinity norm of direction at parameter {parameter}: got {direction_linf}, \
                 expected {radius}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "zero derivative")]
    fn sample_zero_inner_derivative_panics() {
        // Constant trajectory: derivative is zero everywhere.
        let inner = CubicSpline::new(
            array![0.0, 1.0, 2.0],
            array![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]].view(),
        )
        .unwrap();
        let bundle = ChebyshevSphereBundleInterpolator::new(inner, 0);
        let _ = bundle.sample(0.5);
    }
}
