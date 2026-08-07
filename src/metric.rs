// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! The [`Metric`] enum of distance modes over rows of an
//! [`Array2<f64>`](ndarray::Array2).

use ndarray::{ArrayView1, ArrayView2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod sphere_bundle;

use sphere_bundle::{assert_even_dimension, sphere_bundle_distance, sphere_bundle_distance_slices};

/// A distance mode over rows of a trajectory.
///
/// The crate supports exactly two modes. [`Metric::Euclidean`] measures
/// position coordinates directly. [`Metric::SphereBundle`] measures
/// even-length points whose first half is a spatial position and whose second
/// half is a direction vector; see the variant documentation for the distance
/// formula and its calibration with the cubical cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
pub enum Metric {
    /// The standard Euclidean metric.
    ///
    /// Distance is the square root of the sum of squared coordinate
    /// differences.
    Euclidean = 0, // fingerprint, do not renumber.

    /// A distance on the L2 sphere bundle.
    ///
    /// Operates on any sphere-bundle-like input: a vector of even length
    /// `2 * dimension` whose first half is a spatial position and whose
    /// second half is a direction vector. The distance is the maximum of the
    /// two halves' Euclidean distances:
    ///
    /// ```text
    /// max(
    ///     euclidean(position_left, position_right),
    ///     euclidean(direction_left, direction_right),
    /// )
    /// ```
    ///
    /// The maximum, rather than a Euclidean combination of the two halves,
    /// is what gives `distance <= t` its reading: within `t` in position
    /// *and* within `t` in direction. A combination would let the two trade,
    /// admitting a pair far apart in space merely because it is well aligned,
    /// which is not a recurrence.
    ///
    /// This metric is calibrated against
    /// [`SphereBundleInterpolator`](crate::interpolation::SphereBundleInterpolator),
    /// which stores each direction half as the unit tangent scaled to its
    /// configured direction radius. That radius does double duty: it is the
    /// resolution of the direction half and the exchange rate between
    /// position and direction.
    ///
    /// ```
    /// use cycling_signatures::{
    ///     interpolation::{CubicSpline, SphereBundleInterpolator},
    ///     metric::Metric,
    /// };
    /// use ndarray::array;
    ///
    /// let spline = CubicSpline::new(
    ///     array![0.0, 1.0, 2.0],
    ///     array![[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]].view(),
    /// )
    /// .unwrap();
    /// let interpolator = SphereBundleInterpolator::new(spline, 3.5);
    /// assert_eq!(interpolator.direction_radius(), 3.5);
    ///
    /// let metric = Metric::SphereBundle;
    ///
    /// // Identical direction halves, each at the interpolator's radius:
    /// // the direction term is zero, so the position term dominates.
    /// let left = array![0.0, 0.0, 3.5, 0.0];
    /// let right = array![3.0, 4.0, 3.5, 0.0];
    /// assert!((metric.distance(left.view(), right.view()) - 5.0).abs() < 1e-12);
    ///
    /// // Identical positions, orthogonal direction halves each at magnitude
    /// // 3.5: the direction term is 3.5 * sqrt(2) and dominates, while the
    /// // position term contributes nothing.
    /// let left = array![0.0, 0.0, 3.5, 0.0];
    /// let right = array![0.0, 0.0, 0.0, 3.5];
    /// let expected = 3.5 * 2.0_f64.sqrt();
    /// assert!(
    ///     (metric.distance(left.view(), right.view()) - expected).abs() < 1e-12
    /// );
    /// ```
    SphereBundle = 1, // fingerprint, do not renumber.
}

impl Metric {
    /// The distance from `point` to `other` under this metric.
    ///
    /// # Panics
    ///
    /// Panics if `point.len() != other.len()`. The sphere-bundle mode also
    /// panics if the common length is not even.
    #[must_use]
    pub fn distance(self, point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
        match self {
            Metric::Euclidean => euclidean_distance(point, other),
            Metric::SphereBundle => sphere_bundle_distance(point, other),
        }
    }

    /// Fills `out[k]` with the distance between `points.row(pairs[k].0)` and
    /// `points.row(pairs[k].1)` for each `k`.
    ///
    /// `out` and `pairs` must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != pairs.len()`, or if any index in `pairs` is out
    /// of bounds for `points`, or for any reason the [`Metric::distance`]
    /// method panics.
    pub fn fill_distances(
        self,
        points: ArrayView2<'_, f64>,
        pairs: &[(usize, usize)],
        out: &mut [f64],
    ) {
        assert_eq!(
            out.len(),
            pairs.len(),
            "length mismatch: {} pairs but the output slice has length {}",
            pairs.len(),
            out.len()
        );
        for (slot, &(left, right)) in out.iter_mut().zip(pairs) {
            *slot = self.distance(points.row(left), points.row(right));
        }
    }

    /// Pairs this metric with the contiguous point array `points` for repeated
    /// indexed distance queries.
    ///
    /// # Panics
    ///
    /// Panics if `points` is not laid out row-major and contiguously. Under
    /// [`Metric::SphereBundle`], also panics if `points.ncols()` is odd.
    #[must_use]
    pub(crate) fn over(self, points: ArrayView2<'_, f64>) -> MetricPoints<'_> {
        if let Metric::SphereBundle = self {
            assert_even_dimension(points.ncols());
        }
        let count = points.nrows();
        let dimension = points.ncols();
        let coordinates = points
            .to_slice()
            .expect("point rows must be contiguous to be measured by index");
        MetricPoints {
            coordinates,
            count,
            dimension,
            metric: self,
        }
    }
}

/// A point array viewed through a fixed [`Metric`], built by [`Metric::over`].
///
/// Holds the points as one contiguous run of coordinates, so that
/// [`MetricPoints::distance`] evaluates each pair by slicing that run, without
/// allocating or re-checking layout.
pub(crate) struct MetricPoints<'points> {
    coordinates: &'points [f64],
    count: usize,
    dimension: usize,
    metric: Metric,
}

impl MetricPoints<'_> {
    /// The number of points in view.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.count
    }

    /// The contiguous coordinates of row `index`.
    fn row(&self, index: usize) -> &[f64] {
        &self.coordinates[index * self.dimension..][..self.dimension]
    }

    /// The distance between rows `left_row` and `right_row`, equal to
    /// [`Metric::distance`] evaluated on the corresponding original rows.
    ///
    /// # Panics
    ///
    /// Panics if `left_row` or `right_row` is out of bounds for the points in
    /// view.
    #[must_use]
    pub(crate) fn distance(&self, left_row: usize, right_row: usize) -> f64 {
        let left = self.row(left_row);
        let right = self.row(right_row);

        match self.metric {
            Metric::Euclidean => euclidean_distance_slices(left, right),
            Metric::SphereBundle => sphere_bundle_distance_slices(left, right),
        }
    }
}

/// Accumulates squared coordinate differences over `pairs` in index order
/// and takes the square root last.
fn euclidean_norm_of_differences(pairs: impl Iterator<Item = (f64, f64)>) -> f64 {
    pairs
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The Euclidean distance between two equal-length slices.
fn euclidean_distance_slices(left: &[f64], right: &[f64]) -> f64 {
    euclidean_norm_of_differences(left.iter().copied().zip(right.iter().copied()))
}

/// The Euclidean distance between two equal-length slices.
///
/// # Panics
///
/// Panics if `point.len() != other.len()`.
pub(crate) fn euclidean_distance(point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
    assert_eq!(
        point.len(),
        other.len(),
        "dimension mismatch: first {}, second {}",
        point.len(),
        other.len()
    );
    euclidean_norm_of_differences(point.iter().copied().zip(other.iter().copied()))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::Metric;

    #[test]
    fn euclidean_known_distance() {
        let origin = array![0.0, 0.0, 0.0];
        let target = array![3.0, 4.0, 0.0];
        let distance = Metric::Euclidean.distance(origin.view(), target.view());
        assert!((distance - 5.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch: first 2, second 3")]
    fn distance_dimension_mismatch_panics() {
        let short = array![0.0, 0.0];
        let longer = array![1.0, 2.0, 3.0];
        let _ = Metric::Euclidean.distance(short.view(), longer.view());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn indexed_distance_matches_metric_distance_bit_for_bit() {
        let euclidean_points = array![[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]];
        let euclidean_view = Metric::Euclidean.over(euclidean_points.view());
        for left in 0..euclidean_points.nrows() {
            for right in 0..euclidean_points.nrows() {
                let expected = Metric::Euclidean
                    .distance(euclidean_points.row(left), euclidean_points.row(right));
                assert_eq!(euclidean_view.distance(left, right), expected);
            }
        }

        let sphere_points = array![
            [0.0, 0.0, 1.0, 0.0],
            [3.0, 4.0, 0.0, 2.0],
            [1.0, 1.0, 0.0, 0.0],
        ];
        let metric = Metric::SphereBundle;
        let sphere_view = metric.over(sphere_points.view());
        for left in 0..sphere_points.nrows() {
            for right in 0..sphere_points.nrows() {
                let expected = metric.distance(sphere_points.row(left), sphere_points.row(right));
                assert_eq!(sphere_view.distance(left, right), expected);
            }
        }
    }

    #[test]
    #[should_panic(expected = "sphere bundle metric requires even dimension, got 3")]
    fn over_odd_length_panics() {
        let points = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let _ = Metric::SphereBundle.over(points.view());
    }

    #[test]
    fn fill_distances_matches_per_pair() {
        let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0], [1.0, 1.0]];
        let pairs = [(0, 1), (1, 2), (0, 3)];

        let mut out = vec![0.0; pairs.len()];
        Metric::Euclidean.fill_distances(points.view(), &pairs, &mut out);
        for (slot, &(left_index, right_index)) in out.iter().zip(&pairs) {
            assert!(
                (slot
                    - Metric::Euclidean.distance(points.row(left_index), points.row(right_index)))
                .abs()
                    < 1e-12
            );
        }
    }
}
