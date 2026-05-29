// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Metric on the L2 sphere bundle.

use ndarray::{Array1, ArrayView1, s};

use crate::{
    error::{Error, Result},
    metric::{Metric, euclidean_covers_triple, euclidean_distance},
};

/// Splits a sphere-bundle point into its position half and L2-normalized
/// direction half.
///
/// A valid sphere-bundle point has even-length coordinates (the first half
/// is position, the second half is direction) with a nonzero direction half
/// so its L2-normalization is well-defined. Both conditions are asserted.
///
/// # Panics
///
/// Panics if `point.len()` is odd or if the direction half has zero L2 norm.
#[allow(clippy::needless_pass_by_value)]
fn split_and_normalize(point: ArrayView1<'_, f64>) -> (Array1<f64>, Array1<f64>) {
    assert!(
        point.len().is_multiple_of(2),
        "sphere bundle metric requires even dimension, got {}",
        point.len(),
    );
    let half = point.len() / 2;
    let position = point.slice(s![..half]).to_owned();

    let direction = point.slice(s![half..]);
    let norm = direction.dot(&direction).sqrt();
    assert!(norm > 0.0, "zero direction L2 norm");
    let unit = &direction / norm;
    (position, unit)
}

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        let (position_point, direction_point) = split_and_normalize(point);
        let (position_other, direction_other) = split_and_normalize(other);
        let position_distance = euclidean_distance(position_point.view(), position_other.view());
        let direction_distance = euclidean_distance(direction_point.view(), direction_other.view());
        position_distance.max(self.direction_weight * direction_distance)
    }

    /// # Panics
    ///
    /// Panics if `p1`, `p2`, and `p3` do not all have the same length, if that
    /// length is not even, or if any direction half has zero L2 norm.
    fn covers_triple(
        &self,
        p1: ArrayView1<'_, f64>,
        p2: ArrayView1<'_, f64>,
        p3: ArrayView1<'_, f64>,
        radius: f64,
    ) -> bool {
        assert_eq!(p1.len(), p2.len(), "dimension mismatch between p1 and p2");
        assert_eq!(p1.len(), p3.len(), "dimension mismatch between p1 and p3");

        let (position1, direction1) = split_and_normalize(p1);
        let (position2, direction2) = split_and_normalize(p2);
        let (position3, direction3) = split_and_normalize(p3);

        euclidean_covers_triple(position1.view(), position2.view(), position3.view(), radius)
            && euclidean_covers_triple(
                direction1.view(),
                direction2.view(),
                direction3.view(),
                radius / self.direction_weight,
            )
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

    #[test]
    fn covers_triple_rejects_where_default_would_accept() {
        use ndarray::array;

        // Three sphere-bundle points whose position halves form an equilateral
        // triangle of side 1.0 in 2D, with identical direction halves.
        //
        // Full pairwise sphere-bundle distance per pair: max(position_l2 = 1.0,
        // direction_weight * 0) = 1.0. With radius = 0.5, the default
        // `covers_triple` body checks every pairwise distance against
        // 2 * radius = 1.0 and accepts (all distances are exactly 1.0).
        //
        // The override's position-side Euclidean check sees an equilateral
        // triangle of side 1.0; its smallest enclosing l_2 ball has radius
        // 1.0 / sqrt(3) ~= 0.577 > 0.5, so the override rejects.
        let metric = SphereBundleMetric::new(1.0).unwrap();
        let p1 = array![0.0, 0.0, 1.0, 0.0];
        let p2 = array![1.0, 0.0, 1.0, 0.0];
        let p3 = array![0.5, 3.0_f64.sqrt() / 2.0, 1.0, 0.0];

        // Override rejects; the default body on this fixture would accept.
        assert!(!metric.covers_triple(p1.view(), p2.view(), p3.view(), 0.5));

        // Sanity-check: the same triple at a slightly larger radius (above the
        // 1/sqrt(3) circumradius) is accepted, confirming the smallest-
        // enclosing-ball cutoff is what's controlling the outcome.
        assert!(metric.covers_triple(p1.view(), p2.view(), p3.view(), 0.6));
    }
}
