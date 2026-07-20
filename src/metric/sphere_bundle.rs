// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Distance functions on the L2 sphere bundle, backing
//! [`Metric::SphereBundle`](crate::metric::Metric::SphereBundle).

use ndarray::{Array1, ArrayView1, s};

use crate::metric::{euclidean_covers_triple, euclidean_distance};

/// The direction weight derived from a radius floor: the cover radius
/// `radius_floor + 0.5`.
pub(crate) fn direction_weight(radius_floor: u32) -> f64 {
    f64::from(radius_floor) + 0.5
}

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

/// The sphere-bundle distance between two even-length points: the maximum of
/// the position-half Euclidean distance and the weighted Euclidean distance
/// between the L2-normalized direction halves, with the weight derived from
/// `radius_floor`.
///
/// # Panics
///
/// Panics if `point.len() != other.len()`, if the common length is not even,
/// or if either direction half has zero L2 norm.
pub(crate) fn sphere_bundle_distance(
    radius_floor: u32,
    point: ArrayView1<'_, f64>,
    other: ArrayView1<'_, f64>,
) -> f64 {
    assert_eq!(
        point.len(),
        other.len(),
        "dimension mismatch: first {}, second {}",
        point.len(),
        other.len()
    );
    let (position_point, direction_point) = split_and_normalize(point);
    let (position_other, direction_other) = split_and_normalize(other);
    let position_distance = euclidean_distance(position_point.view(), position_other.view());
    let direction_distance = euclidean_distance(direction_point.view(), direction_other.view());
    position_distance.max(direction_weight(radius_floor) * direction_distance)
}

/// Returns `true` if sphere-bundle balls with `radius` around the three
/// points share a common point: their position halves must admit a common
/// `l_2` ball of radius `radius`, and their L2-normalized direction halves a
/// common ball of the weight-rescaled radius.
///
/// # Panics
///
/// Panics if `first`, `second`, and `third` do not all have the same length,
/// if that length is not even, or if any direction half has zero L2 norm.
pub(crate) fn sphere_bundle_covers_triple(
    radius_floor: u32,
    first: ArrayView1<'_, f64>,
    second: ArrayView1<'_, f64>,
    third: ArrayView1<'_, f64>,
    radius: f64,
) -> bool {
    assert!(
        first.len() == second.len() && first.len() == third.len(),
        "dimension mismatch: first {}, second {}, third {}",
        first.len(),
        second.len(),
        third.len()
    );

    let (position_first, direction_first) = split_and_normalize(first);
    let (position_second, direction_second) = split_and_normalize(second);
    let (position_third, direction_third) = split_and_normalize(third);

    euclidean_covers_triple(
        position_first.view(),
        position_second.view(),
        position_third.view(),
        radius,
    ) && euclidean_covers_triple(
        direction_first.view(),
        direction_second.view(),
        direction_third.view(),
        radius / direction_weight(radius_floor),
    )
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::metric::Metric;

    #[test]
    fn distance_max_of_position_and_weighted_direction() {
        // 4D points: 2 position + 2 direction. Position diff (3, 4) -> 5.
        // Direction halves (1, 0) and (0, 1) are already unit L2, so the
        // L2-normalized direction euclidean is sqrt(2).
        let left = array![0.0, 0.0, 1.0, 0.0];
        let right = array![3.0, 4.0, 0.0, 1.0];

        // Position dominates under the smallest weight (radius_floor 0 gives
        // weight 0.5).
        let metric = Metric::SphereBundle { radius_floor: 0 };
        let distance = metric.distance(left.view(), right.view());
        assert!((distance - 5.0).abs() < 1e-12);

        // Direction dominates under a large weight: 10.5 * sqrt(2) > 5.
        let metric = Metric::SphereBundle { radius_floor: 10 };
        let distance = metric.distance(left.view(), right.view());
        assert!((distance - 10.5 * 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn distance_is_invariant_to_direction_scale() {
        // L2 normalization absorbs any positive rescaling of the direction
        // half, so the radius_floor choice on the interpolator side does
        // not change the metric value.
        let metric = Metric::SphereBundle { radius_floor: 1 };

        let baseline_left = array![0.0, 0.0, 1.0, 0.0];
        let baseline_right = array![0.0, 0.0, 0.0, 1.0];
        let baseline = metric.distance(baseline_left.view(), baseline_right.view());

        // Same directions, multiplied by 7.5 (a different radius_floor + 0.5).
        let scaled_left = array![0.0, 0.0, 7.5, 0.0];
        let scaled_right = array![0.0, 0.0, 0.0, 7.5];
        let scaled = metric.distance(scaled_left.view(), scaled_right.view());

        assert!((scaled - baseline).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch: first 2, second 4")]
    fn distance_dimension_mismatch_panics() {
        let metric = Metric::SphereBundle { radius_floor: 1 };
        let left = array![0.0, 0.0];
        let right = array![1.0, 2.0, 3.0, 4.0];
        let _ = metric.distance(left.view(), right.view());
    }

    #[test]
    #[should_panic(expected = "sphere bundle metric requires even dimension, got 3")]
    fn distance_odd_length_panics() {
        let metric = Metric::SphereBundle { radius_floor: 1 };
        let left = array![0.0, 0.0, 0.0];
        let right = array![1.0, 2.0, 3.0];
        let _ = metric.distance(left.view(), right.view());
    }

    #[test]
    #[should_panic(expected = "zero direction L2 norm")]
    fn distance_zero_direction_norm_panics() {
        // Direction half of the left point is the zero vector; L2
        // normalization is undefined.
        let metric = Metric::SphereBundle { radius_floor: 1 };
        let left = array![0.0, 0.0, 0.0, 0.0];
        let right = array![0.0, 0.0, 1.0, 0.0];
        let _ = metric.distance(left.view(), right.view());
    }

    #[test]
    fn covers_triple_uses_smallest_enclosing_ball_per_half() {
        // Three sphere-bundle points whose position halves form an
        // equilateral triangle of side 1.0 in 2D, with identical direction
        // halves (so the direction check passes at any radius).
        //
        // Full pairwise sphere-bundle distance per pair is 1.0. A pairwise
        // check against 2 * radius = 1.0 would accept at radius 0.5; the
        // position-side smallest enclosing l_2 ball has radius
        // 1.0 / sqrt(3) ~= 0.577 > 0.5, so the Cech test rejects there and
        // accepts above the circumradius.
        let metric = Metric::SphereBundle { radius_floor: 0 };
        let first = array![0.0, 0.0, 1.0, 0.0];
        let second = array![1.0, 0.0, 1.0, 0.0];
        let third = array![0.5, 3.0_f64.sqrt() / 2.0, 1.0, 0.0];

        assert!(!metric.covers_triple(first.view(), second.view(), third.view(), 0.5));
        assert!(metric.covers_triple(first.view(), second.view(), third.view(), 0.6));
    }

    #[test]
    fn fill_distances_matches_per_pair() {
        // Three 4D sphere-bundle points; verifies the batched path agrees
        // with the per-pair distance on a concrete fixture.
        let points = array![
            [0.0, 0.0, 1.0, 0.0],
            [3.0, 4.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let pairs = [(0, 1), (1, 2), (0, 2)];
        let metric = Metric::SphereBundle { radius_floor: 0 };

        let mut out = vec![0.0; pairs.len()];
        metric.fill_distances(points.view(), &pairs, &mut out);
        for (slot, &(left_index, right_index)) in out.iter().zip(&pairs) {
            assert!(
                (slot - metric.distance(points.row(left_index), points.row(right_index))).abs()
                    < 1e-12
            );
        }
    }
}
