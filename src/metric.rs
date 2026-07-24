// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! The [`Metric`] enum of distance modes over rows of an
//! [`Array2<f64>`](ndarray::Array2).

use ndarray::{Array2, ArrayView1, ArrayView2};

mod sphere_bundle;

use sphere_bundle::{
    direction_weight, normalize_directions_in_place, sphere_bundle_covers_triple,
    sphere_bundle_distance,
};

/// A distance mode over rows of a trajectory.
///
/// The crate supports exactly two modes. [`Metric::Euclidean`] measures
/// position coordinates directly. [`Metric::SphereBundle`] measures
/// even-length points whose first half is a spatial position and whose second
/// half is a nonzero scaling of a direction vector; see the variant
/// documentation for the distance formula and its calibration with the
/// cubical cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    /// The standard Euclidean metric.
    ///
    /// Distance is the square root of the sum of squared coordinate
    /// differences.
    Euclidean,

    /// A distance on the L2 sphere bundle.
    ///
    /// Operates on any sphere-bundle-like input: a vector of even length
    /// `2 * dimension` whose first half is a spatial position and whose
    /// second half is some nonzero scaling of a direction (velocity) vector.
    /// The direction half is L2-normalized to a unit vector before the
    /// position and direction terms are combined, which lifts the input into
    /// the sphere bundle. Following normalization the distance is
    ///
    /// ```text
    /// max(
    ///     euclidean(position_left, position_right),
    ///     weight * euclidean(direction_left_unit, direction_right_unit),
    /// )
    /// ```
    ///
    /// where `weight` is the cover radius `radius_floor + 0.5`, derived from
    /// the same `radius_floor` that
    /// [`ChebyshevSphereBundleInterpolator`](crate::interpolation::ChebyshevSphereBundleInterpolator)
    /// takes. The shared integer keeps the direction term measured on the
    /// same scale as the radius-scaled direction cubes, so recurrence
    /// thresholds stay compatible with cube adjacency.
    ///
    /// ```
    /// use cycling_signatures::{
    ///     interpolation::{ChebyshevSphereBundleInterpolator, CubicSpline},
    ///     metric::Metric,
    /// };
    /// use ndarray::array;
    ///
    /// let spline = CubicSpline::new(
    ///     array![0.0, 1.0, 2.0],
    ///     array![[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]].view(),
    /// )
    /// .unwrap();
    /// // One integer drives both the interpolator and the metric.
    /// let interpolator = ChebyshevSphereBundleInterpolator::new(spline, 3);
    /// let metric = Metric::SphereBundle { radius_floor: 3 };
    /// assert_eq!(interpolator.radius(), 3.5);
    ///
    /// // Identical directions: the position term dominates.
    /// let left = array![0.0, 0.0, 1.0, 0.0];
    /// let right = array![3.0, 4.0, 1.0, 0.0];
    /// assert!((metric.distance(left.view(), right.view()) - 5.0).abs() < 1e-12);
    /// ```
    SphereBundle {
        /// Integer floor of the direction-normalization radius. The cover
        /// radius, and the direction weight derived from it, is
        /// `radius_floor + 0.5`.
        radius_floor: u32,
    },
}

impl Metric {
    /// The distance from `point` to `other` under this metric.
    ///
    /// # Panics
    ///
    /// Panics if `point.len() != other.len()`. The sphere-bundle mode also
    /// panics if the common length is not even or if either direction half
    /// has zero L2 norm.
    #[must_use]
    pub fn distance(self, point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
        match self {
            Metric::Euclidean => euclidean_distance(point, other),
            Metric::SphereBundle { radius_floor } => {
                sphere_bundle_distance(radius_floor, point, other)
            },
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

    /// Returns `true` if balls with `radius` around `first`, `second`, and
    /// `third` share a common point.
    ///
    /// Both modes use closed-form smallest-enclosing-ball tests: in terms of
    /// named complexes, this admits the Cech simplex, strictly more selective
    /// than the Vietoris-Rips condition that all pairwise distances are at
    /// most `2 * radius`.
    ///
    /// # Panics
    ///
    /// Panics if `first`, `second`, and `third` do not all have the same
    /// length, with the sphere-bundle additions listed on
    /// [`Metric::distance`].
    #[must_use]
    pub fn covers_triple(
        self,
        first: ArrayView1<'_, f64>,
        second: ArrayView1<'_, f64>,
        third: ArrayView1<'_, f64>,
        radius: f64,
    ) -> bool {
        match self {
            Metric::Euclidean => euclidean_covers_triple(first, second, third, radius),
            Metric::SphereBundle { radius_floor } => {
                sphere_bundle_covers_triple(radius_floor, first, second, third, radius)
            },
        }
    }

    /// Prepares `points` for repeated [`PreparedPoints::distance`] queries
    /// under this metric.
    ///
    /// The result holds a contiguous copy of `points`. Under
    /// [`Metric::SphereBundle`], each row's direction half is replaced by its
    /// L2-normalized unit vector (computed identically to how
    /// [`Metric::distance`] normalizes it), so that later distance queries
    /// need no further normalization or allocation. Under
    /// [`Metric::Euclidean`], rows are copied unchanged.
    ///
    /// # Panics
    ///
    /// Under [`Metric::SphereBundle`], panics if any row's length is odd or
    /// if any row's direction half has zero L2 norm.
    #[must_use]
    pub(crate) fn prepare(self, points: ArrayView2<'_, f64>) -> PreparedPoints {
        let mut rows = points.to_owned();
        if let Metric::SphereBundle { .. } = self {
            normalize_directions_in_place(&mut rows);
        }
        PreparedPoints { rows, metric: self }
    }
}

/// Points prepared for repeated distance evaluation under a fixed
/// [`Metric`], built by [`Metric::prepare`].
///
/// Preparation folds the per-pair normalization work that
/// [`Metric::SphereBundle`] would otherwise repeat on every call into a
/// single pass over the input, so that [`PreparedPoints::distance`]
/// evaluates each pair by reading plain contiguous slices, without
/// allocating.
pub(crate) struct PreparedPoints {
    rows: Array2<f64>,
    metric: Metric,
}

impl PreparedPoints {
    /// The distance between prepared rows `left_row` and `right_row`, equal
    /// to [`Metric::distance`] evaluated on the corresponding original rows.
    ///
    /// # Panics
    ///
    /// Panics if `left_row` or `right_row` is out of bounds for the prepared
    /// points.
    #[must_use]
    pub(crate) fn distance(&self, left_row: usize, right_row: usize) -> f64 {
        let left = self.rows.row(left_row);
        let right = self.rows.row(right_row);
        let left = left
            .as_slice()
            .expect("prepared points are stored contiguously");
        let right = right
            .as_slice()
            .expect("prepared points are stored contiguously");

        match self.metric {
            Metric::Euclidean => euclidean_distance_slices(left, right),
            Metric::SphereBundle { radius_floor } => {
                let half = left.len() / 2;
                let position_distance = euclidean_distance_slices(&left[..half], &right[..half]);
                let direction_distance = euclidean_distance_slices(&left[half..], &right[half..]);
                position_distance.max(direction_weight(radius_floor) * direction_distance)
            },
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

/// Returns `true` if the smallest enclosing `l_2` ball of `first`, `second`,
/// `third` has radius at most `radius`. Closed form in terms of pairwise
/// distances:
///
/// - If the longest pairwise distance `c` satisfies `c^2 >= a^2 + b^2` (obtuse
///   or right triangle at the vertex opposite `c`, or collinear), the smallest
///   enclosing ball has `c` as diameter.
/// - Otherwise the smallest enclosing ball is the circumscribed circle, with
///   radius `(a * b * c) / (4 * area)` and `area` from Heron's formula.
pub(crate) fn euclidean_covers_triple(
    first: ArrayView1<'_, f64>,
    second: ArrayView1<'_, f64>,
    third: ArrayView1<'_, f64>,
    radius: f64,
) -> bool {
    let first_second = euclidean_distance(first, second);
    let first_third = euclidean_distance(first, third);
    let second_third = euclidean_distance(second, third);

    let mut sides = [first_second, first_third, second_third];
    sides.sort_by(f64::total_cmp);
    let [shorter, mid, longest] = sides;

    if longest > 2.0 * radius {
        return false;
    }

    if longest * longest >= shorter * shorter + mid * mid {
        // Obtuse, right, or collinear: enclosing ball has the longest side as
        // diameter.
        return longest <= 2.0 * radius;
    }

    // Acute triangle: circumscribed circle. Heron's formula for area.
    let semi = (first_second + first_third + second_third) / 2.0;
    let area = (semi * (semi - first_second) * (semi - first_third) * (semi - second_third)).sqrt();
    let circumradius = (first_second * first_third * second_third) / (4.0 * area);
    circumradius <= radius
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
    fn euclidean_covers_triple_uses_smallest_enclosing_ball() {
        // Equilateral triangle of side 2.0. All pairwise distances equal
        // 2 * radius at radius 1.0, so a pairwise (Vietoris-Rips) test would
        // accept; the smallest enclosing ball has radius 2 / sqrt(3) > 1.0,
        // so the Cech test rejects there and accepts just above the
        // circumradius.
        let first = array![0.0, 0.0];
        let second = array![2.0, 0.0];
        let third = array![1.0, 3.0_f64.sqrt()];
        assert!(!Metric::Euclidean.covers_triple(first.view(), second.view(), third.view(), 1.0));
        assert!(Metric::Euclidean.covers_triple(
            first.view(),
            second.view(),
            third.view(),
            2.0 / 3.0_f64.sqrt() + 1e-9,
        ));
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
    fn prepared_distance_matches_metric_distance_bit_for_bit() {
        let euclidean_points = array![[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]];
        let euclidean_prepared = Metric::Euclidean.prepare(euclidean_points.view());
        for left in 0..euclidean_points.nrows() {
            for right in 0..euclidean_points.nrows() {
                let expected = Metric::Euclidean
                    .distance(euclidean_points.row(left), euclidean_points.row(right));
                assert_eq!(euclidean_prepared.distance(left, right), expected);
            }
        }

        // Direction halves scaled to different norms per row so
        // normalization is exercised at more than one scale.
        let sphere_points = array![
            [0.0, 0.0, 1.0, 0.0],
            [3.0, 4.0, 0.0, 2.0],
            [1.0, 1.0, 3.0, 4.0],
        ];
        let metric = Metric::SphereBundle { radius_floor: 2 };
        let sphere_prepared = metric.prepare(sphere_points.view());
        for left in 0..sphere_points.nrows() {
            for right in 0..sphere_points.nrows() {
                let expected = metric.distance(sphere_points.row(left), sphere_points.row(right));
                assert_eq!(sphere_prepared.distance(left, right), expected);
            }
        }
    }

    #[test]
    #[should_panic(expected = "zero direction L2 norm")]
    fn prepare_zero_direction_norm_panics() {
        // The first row's direction half is the zero vector; L2
        // normalization during preparation is undefined.
        let points = array![[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]];
        let _ = Metric::SphereBundle { radius_floor: 1 }.prepare(points.view());
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
