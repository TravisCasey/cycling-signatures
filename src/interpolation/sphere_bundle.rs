// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Sphere-bundle interpolator wrapper.

use ndarray::Array1;

use crate::interpolation::{DerivativeInterpolator, Interpolator};

/// Wraps a [`DerivativeInterpolator`] and produces samples that concatenate
/// the inner position with a scaled direction.
///
/// At each parameter, the scaled direction is the inner derivative
/// normalized to unit L2 length, then multiplied by the configured
/// `direction_radius`, placing it on the L2 sphere of that radius. Each
/// sample has length twice that of the inner interpolator: the first half is
/// the position, the second half is the scaled direction.
///
/// `direction_radius` is the L2 norm of every stored direction, and it sets
/// the angular resolution of the embedding. Two directions separated by an
/// angle `theta` are stored `2 * direction_radius * sin(theta / 2)` apart, so
/// a cycle-detection threshold `t` admits directions strictly within
/// `2 * arcsin(t / (2 * direction_radius))` of each other: a larger radius
/// distinguishes directions more finely at the same threshold.
#[derive(Debug, Clone)]
pub struct SphereBundleInterpolator<Inner> {
    inner: Inner,
    direction_radius: f64,
}

impl<Inner: DerivativeInterpolator> SphereBundleInterpolator<Inner> {
    /// Wraps `inner` with the given direction radius.
    ///
    /// # Panics
    ///
    /// Panics if `direction_radius` is not positive and finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::interpolation::{
    ///     CubicSpline, Interpolator, SphereBundleInterpolator,
    /// };
    /// use ndarray::array;
    ///
    /// let inner = CubicSpline::new(
    ///     array![0.0, 1.0, 2.0],
    ///     array![[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]].view(),
    /// )
    /// .unwrap();
    /// let bundle = SphereBundleInterpolator::new(inner, 1.5);
    /// assert_eq!(bundle.direction_radius(), 1.5);
    ///
    /// let sample = bundle.sample(0.5);
    /// // Output is the inner sample concatenated with the scaled direction.
    /// assert_eq!(sample.len(), 4);
    /// ```
    #[must_use]
    pub fn new(inner: Inner, direction_radius: f64) -> Self {
        assert!(
            direction_radius.is_finite() && direction_radius > 0.0,
            "direction radius must be positive and finite, got {direction_radius}"
        );
        Self {
            inner,
            direction_radius,
        }
    }

    /// The direction normalization radius given at construction.
    #[must_use]
    pub fn direction_radius(&self) -> f64 {
        self.direction_radius
    }

    /// The wrapped interpolator supplying positions and derivatives.
    #[must_use]
    pub fn inner(&self) -> &Inner {
        &self.inner
    }
}

impl<Inner: DerivativeInterpolator> Interpolator for SphereBundleInterpolator<Inner> {
    /// # Panics
    ///
    /// Panics if `parameter` is outside the fitted domain, or if the inner
    /// derivative is zero at the supplied parameter.
    fn sample(&self, parameter: f64) -> Array1<f64> {
        let position = self.inner.sample(parameter);
        let derivative = self.inner.derivative(parameter);

        let l2_norm = derivative.dot(&derivative).sqrt();
        assert!(l2_norm > 0.0, "zero derivative at parameter {parameter}");

        let scaled: Array1<f64> =
            derivative.mapv(|component| component / l2_norm * self.direction_radius);

        let dimension = position.len();
        let mut result = Array1::<f64>::zeros(2 * dimension);
        result.slice_mut(ndarray::s![..dimension]).assign(&position);
        result.slice_mut(ndarray::s![dimension..]).assign(&scaled);

        result
    }

    fn knots(&self) -> &[f64] {
        self.inner.knots()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::SphereBundleInterpolator;
    use crate::interpolation::{CubicSpline, Interpolator};

    #[test]
    fn sample_concatenates_position_and_scaled_direction() {
        // Three properties asserted together:
        //   - output length is 2 * inner dimension,
        //   - the spatial half matches the inner spline's own sample, and
        //   - the direction half has L2 norm equal to the radius.
        let knots = array![0.0, 1.0, 2.0, 3.0];
        let values = array![[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 3.0]];
        let inner = CubicSpline::new(knots.clone(), values.view()).unwrap();
        let bundle = SphereBundleInterpolator::new(inner.clone(), 2.5);
        let radius = bundle.direction_radius();

        for parameter in [0.5, 1.0, 1.5, 2.0, 2.5] {
            let sample = bundle.sample(parameter);
            assert_eq!(sample.len(), 4);

            let inner_sample = inner.sample(parameter);
            assert!((sample[0] - inner_sample[0]).abs() < 1e-12);
            assert!((sample[1] - inner_sample[1]).abs() < 1e-12);

            let direction_l2 = sample
                .iter()
                .skip(2)
                .map(|component| component * component)
                .sum::<f64>()
                .sqrt();
            assert!(
                (direction_l2 - radius).abs() < 1e-10,
                "L2 norm of direction at parameter {parameter}: got {direction_l2}, expected \
                 {radius}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "direction radius must be positive and finite")]
    fn new_rejects_non_positive_radius() {
        let inner = CubicSpline::new(
            array![0.0, 1.0, 2.0],
            array![[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]].view(),
        )
        .unwrap();
        let _ = SphereBundleInterpolator::new(inner, 0.0);
    }

    #[test]
    #[should_panic(expected = "zero derivative at parameter 0.5")]
    fn sample_zero_inner_derivative_panics() {
        // Constant trajectory: derivative is zero everywhere.
        let inner = CubicSpline::new(
            array![0.0, 1.0, 2.0],
            array![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]].view(),
        )
        .unwrap();
        let bundle = SphereBundleInterpolator::new(inner, 0.5);
        let _ = bundle.sample(0.5);
    }
}
