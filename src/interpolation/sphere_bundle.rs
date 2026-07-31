// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Sphere-bundle interpolator wrapper.

use ndarray::Array1;

use crate::interpolation::{DerivativeInterpolator, Interpolator};

/// Wraps a [`DerivativeInterpolator`] and produces samples that concatenate
/// the inner position with a scaled direction.
///
/// At each parameter, the scaled direction is the inner derivative
/// normalized to unit L2 length, then multiplied by the configured radius,
/// placing it on the L2 sphere of that radius. Each sample has length twice
/// that of the inner interpolator: the first half is the position, the
/// second half is the scaled direction.
///
/// The radius is set indirectly through a `radius_floor: u32` and is fixed at
/// `radius_floor + 0.5`. That radius matches how the cubical cover counts
/// direction cubes: it is both the direction shell's cube resolution and,
/// via [`Metric::SphereBundle`](crate::metric::Metric::SphereBundle), the
/// distance scale a recurrence threshold is measured against, so the same
/// radius must govern both. The half-integer offset also keeps every
/// extremal direction coordinate at `+-(radius_floor + 0.5)`, never an
/// integer, so a direction coordinate at its largest magnitude never lands
/// exactly on a cube boundary.
#[derive(Debug, Clone)]
pub struct SphereBundleInterpolator<Inner> {
    inner: Inner,
    radius: f64,
}

impl<Inner: DerivativeInterpolator> SphereBundleInterpolator<Inner> {
    /// Wraps `inner` with the given radius floor.
    ///
    /// The actual normalization radius is `radius_floor + 0.5`; see the
    /// type-level documentation for the cubical-embedding reason. The radius
    /// is recoverable via [`radius`](Self::radius).
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
    /// let bundle = SphereBundleInterpolator::new(inner, 1);
    /// assert_eq!(bundle.radius(), 1.5);
    ///
    /// let sample = bundle.sample(0.5);
    /// // Output is the inner sample concatenated with the scaled direction.
    /// assert_eq!(sample.len(), 4);
    /// ```
    #[must_use]
    pub fn new(inner: Inner, radius_floor: u32) -> Self {
        let radius = f64::from(radius_floor) + 0.5;
        Self { inner, radius }
    }

    /// The normalization radius (`radius_floor + 0.5` from construction).
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.radius
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

        let scaled: Array1<f64> = derivative.mapv(|component| component / l2_norm * self.radius);

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
        let bundle = SphereBundleInterpolator::new(inner.clone(), 2);
        let radius = bundle.radius();

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
    #[should_panic(expected = "zero derivative at parameter 0.5")]
    fn sample_zero_inner_derivative_panics() {
        // Constant trajectory: derivative is zero everywhere.
        let inner = CubicSpline::new(
            array![0.0, 1.0, 2.0],
            array![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]].view(),
        )
        .unwrap();
        let bundle = SphereBundleInterpolator::new(inner, 0);
        let _ = bundle.sample(0.5);
    }
}
