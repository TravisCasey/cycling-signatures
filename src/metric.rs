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

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{Chebyshev, Euclidean, Metric};

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
