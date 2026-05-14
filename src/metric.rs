// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! The [`Metric`] trait and reference implementations.
//!
//! A metric is a distance function over rows of an `ndarray::Array2<f64>`.

use std::fmt;

use ndarray::ArrayView1;

pub mod sphere_bundle;

pub use sphere_bundle::SphereBundleMetric;

/// A distance function over rows of a trajectory.
pub trait Metric: Clone + Send + Sync + fmt::Debug + 'static {
    /// A unique identifier for this metric.
    #[must_use]
    fn name(&self) -> String;

    /// The distance from `point` to `other` under this metric.
    #[must_use]
    fn distance(&self, point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64;

    /// Returns `true` if balls with `radius` around `p1`, `p2`, and `p3` share
    /// a common point.
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
        p1: ArrayView1<'_, f64>,
        p2: ArrayView1<'_, f64>,
        p3: ArrayView1<'_, f64>,
        radius: f64,
    ) -> bool {
        let diameter = 2.0 * radius;
        self.distance(p1, p2) <= diameter
            && self.distance(p1, p3) <= diameter
            && self.distance(p2, p3) <= diameter
    }
}

/// The standard Euclidean metric.
///
/// Distance is the square root of the sum of squared coordinate differences.
#[derive(Clone, Copy, Debug, Default)]
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

    fn covers_triple(
        &self,
        p1: ArrayView1<'_, f64>,
        p2: ArrayView1<'_, f64>,
        p3: ArrayView1<'_, f64>,
        radius: f64,
    ) -> bool {
        euclidean_covers_triple(p1, p2, p3, radius)
    }
}

/// The Chebyshev metric.
///
/// Distance is the largest absolute coordinate difference.
#[derive(Clone, Copy, Debug, Default)]
pub struct Chebyshev;

impl Metric for Chebyshev {
    fn name(&self) -> String {
        "Chebyshev".to_owned()
    }

    /// # Panics
    ///
    /// Panics if `point.len() != other.len()`.
    fn distance(&self, point: ArrayView1<'_, f64>, other: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(point.len(), other.len(), "dimension mismatch");
        point
            .iter()
            .zip(other.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    /// # Panics
    ///
    /// Panics if `p1`, `p2`, and `p3` do not all have the same length.
    fn covers_triple(
        &self,
        p1: ArrayView1<'_, f64>,
        p2: ArrayView1<'_, f64>,
        p3: ArrayView1<'_, f64>,
        radius: f64,
    ) -> bool {
        assert_eq!(p1.len(), p2.len(), "dimension mismatch");
        assert_eq!(p1.len(), p3.len(), "dimension mismatch");
        for axis in 0..p1.len() {
            let max_coord = p1[axis].max(p2[axis]).max(p3[axis]);
            let min_coord = p1[axis].min(p2[axis]).min(p3[axis]);
            let half_extent = (max_coord - min_coord) / 2.0;
            if half_extent > radius {
                return false;
            }
        }
        true
    }
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

/// Returns `true` if the smallest enclosing `l_2` ball of `p1`, `p2`, `p3` has
/// radius at most `radius`. Closed form in terms of pairwise distances:
///
/// - If the longest pairwise distance `c` satisfies `c^2 >= a^2 + b^2` (obtuse
///   or right triangle at the vertex opposite `c`, or collinear), the smallest
///   enclosing ball has `c` as diameter.
/// - Otherwise the smallest enclosing ball is the circumscribed circle, with
///   radius `(a * b * c) / (4 * area)` and `area` from Heron's formula.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn euclidean_covers_triple(
    p1: ArrayView1<'_, f64>,
    p2: ArrayView1<'_, f64>,
    p3: ArrayView1<'_, f64>,
    radius: f64,
) -> bool {
    let d12 = euclidean_distance(p1, p2);
    let d13 = euclidean_distance(p1, p3);
    let d23 = euclidean_distance(p2, p3);

    let mut sides = [d12, d13, d23];
    sides.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
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
    let semi = (d12 + d13 + d23) / 2.0;
    let area = (semi * (semi - d12) * (semi - d13) * (semi - d23)).sqrt();
    let circumradius = (d12 * d13 * d23) / (4.0 * area);
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
}
