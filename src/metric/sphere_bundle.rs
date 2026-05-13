// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Metric on the L2 sphere bundle.

use ndarray::{ArrayView1, s};

use crate::{
    error::{Error, Result},
    metric::{Metric, euclidean_distance},
};

/// A distance metric on the L2 sphere bundle.
///
/// Operates on any sphere-bundle-like input: a vector of even length
/// `2 * dimension` whose first half is a spatial position and whose second half
/// is some nonzero scaling of a direction (velocity) vector. The direction half
/// is L2-normalized to a unit vector before the position and direction terms
/// are combined, which lifts the input into the sphere bundle. Following
/// normalization the metric is
///
/// ```text
/// max(
///     euclidean(position_left, position_right),
///     direction_weight * euclidean(direction_left_unit, direction_right_unit),
/// )
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SphereBundleMetric {
    direction_weight: f64,
}

impl SphereBundleMetric {
    /// Constructs a sphere-bundle metric with the given direction weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use cycling_signatures::prelude::*;
    /// use ndarray::array;
    ///
    /// let metric = SphereBundleMetric::new(0.5).unwrap();
    /// let x = array![0.0, 0.0, 1.0, 0.0]; // pos (0, 0), dir (1, 0)
    /// let y = array![3.0, 4.0, 0.0, 1.0]; // pos (3, 4), dir (0, 1)
    /// // Position euclidean = 5; direction (after L2 normalization) = sqrt(2).
    /// // max(5, 0.5 * sqrt(2)) = 5.
    /// assert!((metric.distance(x.view(), y.view()) - 5.0).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::SphereBundleMetricWeight`] if `direction_weight` is
    /// not finite or not strictly positive.
    pub fn new(direction_weight: f64) -> Result<Self> {
        if !direction_weight.is_finite() || direction_weight <= 0.0 {
            return Err(Error::SphereBundleMetricWeight {
                value: direction_weight,
            });
        }
        Ok(Self { direction_weight })
    }

    /// The configured direction weight.
    #[must_use]
    pub fn direction_weight(&self) -> f64 {
        self.direction_weight
    }
}

impl Metric for SphereBundleMetric {
    fn name(&self) -> String {
        format!("SphereBundle[weight={}]", self.direction_weight)
    }

    /// # Panics
    ///
    /// Panics if `point.len() != other.len()`, if the common length is not
    /// even, or if either direction half has zero L2 norm.
    fn distance(&self, point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(point.len(), other.len(), "dimension mismatch");
        assert!(
            point.len().is_multiple_of(2),
            "sphere bundle metric requires even dimension, got {}",
            point.len(),
        );
        let half = point.len() / 2;
        let position_distance =
            euclidean_distance(point.slice(s![..half]), other.slice(s![..half]));

        let point_direction = point.slice(s![half..]);
        let other_direction = other.slice(s![half..]);
        let point_norm = point_direction.dot(&point_direction).sqrt();
        let other_norm = other_direction.dot(&other_direction).sqrt();
        assert!(
            point_norm > 0.0 && other_norm > 0.0,
            "zero direction L2 norm",
        );

        let point_unit = &point_direction / point_norm;
        let other_unit = &other_direction / other_norm;
        let direction_distance = euclidean_distance(point_unit.view(), other_unit.view());

        position_distance.max(self.direction_weight * direction_distance)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::SphereBundleMetric;
    use crate::{error::Error, metric::Metric};

    #[test]
    fn distance_max_of_position_and_weighted_direction() {
        // 4D points: 2 position + 2 direction. Position diff (3, 4) -> 5.
        // Direction halves (1, 0) and (0, 1) are already unit L2, so the
        // L2-normalized direction euclidean is sqrt(2).
        let x = array![0.0, 0.0, 1.0, 0.0];
        let y = array![3.0, 4.0, 0.0, 1.0];

        // Position dominates under a small weight.
        let metric = SphereBundleMetric::new(0.5).unwrap();
        let distance = metric.distance(x.view(), y.view());
        assert!((distance - 5.0).abs() < 1e-12);

        // Direction dominates under a large weight: 10 * sqrt(2) > 5.
        let metric = SphereBundleMetric::new(10.0).unwrap();
        let distance = metric.distance(x.view(), y.view());
        assert!((distance - 10.0 * 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn distance_is_invariant_to_direction_scale() {
        // L2 normalization absorbs any positive rescaling of the direction
        // half, so the cube_halfspan choice on the interpolator side does
        // not change the metric value.
        let metric = SphereBundleMetric::new(1.5).unwrap();

        let baseline_x = array![0.0, 0.0, 1.0, 0.0];
        let baseline_y = array![0.0, 0.0, 0.0, 1.0];
        let baseline = metric.distance(baseline_x.view(), baseline_y.view());

        // Same directions, multiplied by 7.5 (a different cube_halfspan + 0.5).
        let scaled_x = array![0.0, 0.0, 7.5, 0.0];
        let scaled_y = array![0.0, 0.0, 0.0, 7.5];
        let scaled = metric.distance(scaled_x.view(), scaled_y.view());

        assert!((scaled - baseline).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn distance_dimension_mismatch_panics() {
        let metric = SphereBundleMetric::new(1.0).unwrap();
        let x = array![0.0, 0.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let _ = metric.distance(x.view(), y.view());
    }

    #[test]
    #[should_panic(expected = "even dimension")]
    fn distance_odd_length_panics() {
        let metric = SphereBundleMetric::new(1.0).unwrap();
        let x = array![0.0, 0.0, 0.0];
        let y = array![1.0, 2.0, 3.0];
        let _ = metric.distance(x.view(), y.view());
    }

    #[test]
    #[should_panic(expected = "zero direction")]
    fn distance_zero_direction_norm_panics() {
        // Direction half of x is the zero vector; L2 normalization is undefined.
        let metric = SphereBundleMetric::new(1.0).unwrap();
        let x = array![0.0, 0.0, 0.0, 0.0];
        let y = array![0.0, 0.0, 1.0, 0.0];
        let _ = metric.distance(x.view(), y.view());
    }

    #[test]
    fn new_rejects_non_positive_weight() {
        for weight in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let outcome = SphereBundleMetric::new(weight);
            assert!(matches!(
                outcome.unwrap_err(),
                Error::SphereBundleMetricWeight { value }
                    if value.to_bits() == weight.to_bits(),
            ));
        }
    }
}
