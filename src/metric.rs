// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! The [`Metric`] trait and reference implementations.
//!
//! A metric is a distance function over rows of an `ndarray::Array2<f64>`.

use std::fmt::Debug;

use ndarray::{ArrayView1, ArrayView2};

pub mod sphere_bundle;

pub use sphere_bundle::SphereBundleMetric;

/// A distance function over rows of a trajectory.
pub trait Metric: Send + Sync + Debug + 'static {
    /// A unique identifier for this metric.
    #[must_use]
    fn name(&self) -> String;

    /// The distance from `point` to `other` under this metric.
    #[must_use]
    fn distance(&self, point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64;

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
    fn fill_distances(
        &self,
        points: ArrayView2<'_, f64>,
        pairs: &[(usize, usize)],
        out: &mut [f64],
    ) {
        assert_eq!(out.len(), pairs.len(), "out and pairs length mismatch");
        for (slot, &(left, right)) in out.iter_mut().zip(pairs) {
            *slot = self.distance(points.row(left), points.row(right));
        }
    }

    /// Returns `true` if balls with `radius` around `first`, `second`, and
    /// `third` share a common point.
    ///
    /// The default body returns `true` if and only if all three pairwise
    /// distances are at most `2 * radius` (the necessary condition that
    /// pairwise balls touch). Greater precision, such as metrics with
    /// closed-form three-ball-intersection tests, may override.
    ///
    /// In terms of named complexes: the default admits the Vietoris-Rips
    /// simplex; the override admits the Cech simplex, which is strictly more
    /// selective.
    #[must_use]
    fn covers_triple(
        &self,
        first: ArrayView1<'_, f64>,
        second: ArrayView1<'_, f64>,
        third: ArrayView1<'_, f64>,
        radius: f64,
    ) -> bool {
        let diameter = 2.0 * radius;
        self.distance(first, second) <= diameter
            && self.distance(first, third) <= diameter
            && self.distance(second, third) <= diameter
    }
}

/// The standard Euclidean metric.
///
/// Distance is the square root of the sum of squared coordinate differences.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Euclidean;

impl Metric for Euclidean {
    fn name(&self) -> String {
        "Euclidean".to_owned()
    }

    /// # Panics
    ///
    /// Panics if `point.len() != other.len()`.
    fn distance(&self, point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
        euclidean_distance(point, other)
    }

    /// # Panics
    ///
    /// Panics if `out.len() != pairs.len()`, or if any index in `pairs` is out
    /// of bounds for `points`, or if any two rows have mismatched lengths.
    fn fill_distances(
        &self,
        points: ArrayView2<'_, f64>,
        pairs: &[(usize, usize)],
        out: &mut [f64],
    ) {
        assert_eq!(out.len(), pairs.len(), "out and pairs length mismatch");
        for (slot, &(left, right)) in out.iter_mut().zip(pairs) {
            *slot = euclidean_distance(points.row(left), points.row(right));
        }
    }

    fn covers_triple(
        &self,
        first: ArrayView1<'_, f64>,
        second: ArrayView1<'_, f64>,
        third: ArrayView1<'_, f64>,
        radius: f64,
    ) -> bool {
        euclidean_covers_triple(first, second, third, radius)
    }
}

/// The Chebyshev metric.
///
/// Distance is the largest absolute coordinate difference.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Chebyshev;

impl Metric for Chebyshev {
    fn name(&self) -> String {
        "Chebyshev".to_owned()
    }

    /// # Panics
    ///
    /// Panics if `point.len() != other.len()`.
    fn distance(&self, point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
        chebyshev_distance(point, other)
    }

    /// # Panics
    ///
    /// Panics if `out.len() != pairs.len()`, or if any index in `pairs` is out
    /// of bounds for `points`, or if any two rows have mismatched lengths.
    fn fill_distances(
        &self,
        points: ArrayView2<'_, f64>,
        pairs: &[(usize, usize)],
        out: &mut [f64],
    ) {
        assert_eq!(out.len(), pairs.len(), "out and pairs length mismatch");
        for (slot, &(left, right)) in out.iter_mut().zip(pairs) {
            *slot = chebyshev_distance(points.row(left), points.row(right));
        }
    }

    /// # Panics
    ///
    /// Panics if `first`, `second`, and `third` do not all have the same
    /// length.
    fn covers_triple(
        &self,
        first: ArrayView1<'_, f64>,
        second: ArrayView1<'_, f64>,
        third: ArrayView1<'_, f64>,
        radius: f64,
    ) -> bool {
        assert_eq!(first.len(), second.len(), "dimension mismatch");
        assert_eq!(first.len(), third.len(), "dimension mismatch");
        for axis in 0..first.len() {
            let max_coord = first[axis].max(second[axis]).max(third[axis]);
            let min_coord = first[axis].min(second[axis]).min(third[axis]);
            let half_extent = (max_coord - min_coord) / 2.0;
            if half_extent > radius {
                return false;
            }
        }
        true
    }
}

/// The Chebyshev distance between two equal-length slices.
///
/// # Panics
///
/// Panics if `point.len() != other.len()`.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn chebyshev_distance(point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
    assert_eq!(point.len(), other.len(), "dimension mismatch");
    point
        .iter()
        .zip(other.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

/// The Euclidean distance between two equal-length slices.
///
/// # Panics
///
/// Panics if `point.len() != other.len()`.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn euclidean_distance(point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
    assert_eq!(point.len(), other.len(), "dimension mismatch");
    point
        .iter()
        .zip(other.iter())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
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
#[allow(clippy::needless_pass_by_value)]
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

    use super::{Chebyshev, Euclidean, Metric, euclidean_distance};

    #[test]
    fn covers_triple_distinguishes_default_from_euclidean_override() {
        // Stub metric with the trait's default covers_triple body.
        #[derive(Clone, Debug)]
        struct Stub;
        impl Metric for Stub {
            fn name(&self) -> String {
                "stub".into()
            }

            fn distance(
                &self,
                point: ndarray::ArrayView1<'_, f64>,
                other: ndarray::ArrayView1<'_, f64>,
            ) -> f64 {
                euclidean_distance(point, other)
            }
        }

        // Equilateral triangle of side 2.0; ball radius 1.0.
        // Pairwise distances all equal 2.0 == 2 * radius (default accepts).
        // Smallest enclosing ball has radius 2.0 / sqrt(3) > 1.0
        // (Euclidean override rejects). This is the test that justifies
        // having the override at all.
        let p1 = array![0.0, 0.0];
        let p2 = array![2.0, 0.0];
        let p3 = array![1.0, 3.0_f64.sqrt()];

        assert!(Stub.covers_triple(p1.view(), p2.view(), p3.view(), 1.0));
        assert!(!Euclidean.covers_triple(p1.view(), p2.view(), p3.view(), 1.0));
    }

    #[test]
    fn chebyshev_covers_triple_uses_per_axis_extent() {
        // Three points whose per-axis half-extent is 1.0 on axis 0 and
        // 0.5 on axis 1. Smallest enclosing l_inf ball radius = 1.0.
        let p1 = array![0.0, 0.0];
        let p2 = array![2.0, 0.0];
        let p3 = array![1.0, 1.0];
        assert!(Chebyshev.covers_triple(p1.view(), p2.view(), p3.view(), 1.0));
        assert!(!Chebyshev.covers_triple(p1.view(), p2.view(), p3.view(), 0.99));
    }

    #[test]
    fn euclidean_known_distance() {
        let metric = Euclidean;
        let origin = array![0.0, 0.0, 0.0];
        let target = array![3.0, 4.0, 0.0];
        let distance = metric.distance(origin.view(), target.view());
        assert!((distance - 5.0).abs() < 1e-12);
    }

    #[test]
    fn chebyshev_known_distance() {
        let metric = Chebyshev;
        let origin = array![0.0, 0.0, 0.0];
        let target = array![3.0, -4.0, 2.0];
        let distance = metric.distance(origin.view(), target.view());
        assert!((distance - 4.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn distance_dimension_mismatch_panics() {
        let metric = Euclidean;
        let short = array![0.0, 0.0];
        let longer = array![1.0, 2.0, 3.0];
        let _ = metric.distance(short.view(), longer.view());
    }

    #[test]
    fn fill_distances_matches_per_pair() {
        let points = array![[0.0, 0.0], [3.0, 0.0], [6.0, 4.0], [1.0, 1.0]];
        let pairs = [(0, 1), (1, 2), (0, 3)];

        let mut out = vec![0.0; pairs.len()];
        Euclidean.fill_distances(points.view(), &pairs, &mut out);
        for (slot, &(left_index, right_index)) in out.iter().zip(&pairs) {
            assert!(
                (slot - Euclidean.distance(points.row(left_index), points.row(right_index))).abs()
                    < 1e-12
            );
        }

        let mut out_chebyshev = vec![0.0; pairs.len()];
        Chebyshev.fill_distances(points.view(), &pairs, &mut out_chebyshev);
        for (slot, &(left_index, right_index)) in out_chebyshev.iter().zip(&pairs) {
            assert!(
                (slot - Chebyshev.distance(points.row(left_index), points.row(right_index))).abs()
                    < 1e-12
            );
        }
    }
}
