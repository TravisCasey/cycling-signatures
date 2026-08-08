// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Distance functions on the L2 sphere bundle, backing
//! [`Metric::SphereBundle`](crate::metric::Metric::SphereBundle).

use ndarray::{ArrayView1, s};

use crate::metric::{euclidean_distance, euclidean_distance_slices};

/// Checks that `dimension` coordinates can be read as a sphere-bundle point.
///
/// A valid sphere-bundle point has even-length coordinates: the first half is
/// position, the second half is direction. An odd length is not a valid
/// sphere-bundle point.
///
/// # Panics
///
/// Panics if `dimension` is odd.
pub(crate) fn assert_even_dimension(dimension: usize) {
    assert!(
        dimension.is_multiple_of(2),
        "sphere bundle metric requires even dimension, got {dimension}",
    );
}

/// Splits a sphere-bundle point into its position half and direction half.
///
/// # Panics
///
/// Panics if `point.len()` is odd.
fn split(point: ArrayView1<'_, f64>) -> (ArrayView1<'_, f64>, ArrayView1<'_, f64>) {
    assert_even_dimension(point.len());
    let half = point.len() / 2;
    (point.slice_move(s![..half]), point.slice_move(s![half..]))
}

/// The sphere-bundle distance between two even-length points: the maximum of
/// the position-half and direction-half Euclidean distances, measured on the
/// stored coordinates with no normalization.
///
/// # Panics
///
/// Panics if `point.len() != other.len()`, or if the common length is not
/// even.
pub(crate) fn sphere_bundle_distance(
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
    let (position_point, direction_point) = split(point);
    let (position_other, direction_other) = split(other);
    euclidean_distance(position_point, position_other)
        .max(euclidean_distance(direction_point, direction_other))
}

/// The sphere-bundle distance between two slices of the same even length: the
/// maximum of the position-half and direction-half Euclidean distances.
///
/// The lengths are the caller's responsibility, which is what distinguishes
/// this from [`sphere_bundle_distance`]: it serves indexed distance queries
/// over a point array whose coordinate count was checked once when the array
/// was paired with the metric.
#[must_use]
pub(crate) fn sphere_bundle_distance_slices(point: &[f64], other: &[f64]) -> f64 {
    let half = point.len() / 2;
    euclidean_distance_slices(&point[..half], &other[..half])
        .max(euclidean_distance_slices(&point[half..], &other[half..]))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::metric::Metric;

    #[test]
    fn distance_is_max_of_position_and_direction() {
        // 4D points: 2 position + 2 direction. Position diff (3, 4) -> 5,
        // direction diff (1, -1) -> sqrt(2). Neither half is normalized, so
        // both distances are read directly off the stored coordinates.
        let left = array![0.0, 0.0, 1.0, 0.0];
        let right = array![3.0, 4.0, 0.0, 1.0];

        // Position dominates: 5.0 > sqrt(2).
        let distance = Metric::SphereBundle.distance(left.view(), right.view());
        assert!((distance - 5.0).abs() < 1e-12);

        // Identical position halves (distance 0), so each of these two
        // distances is exactly its direction difference's L2 norm: (1, -10)
        // -> sqrt(101), (1, -20) -> sqrt(401). The two right-hand points
        // differ only in direction magnitude, and the metric distinguishes
        // them, since the direction half is not normalized away.
        let right_small_direction = array![0.0, 0.0, 0.0, 10.0];
        let distance_small =
            Metric::SphereBundle.distance(left.view(), right_small_direction.view());
        assert!((distance_small - 101.0_f64.sqrt()).abs() < 1e-12);

        let right_large_direction = array![0.0, 0.0, 0.0, 20.0];
        let distance_large =
            Metric::SphereBundle.distance(left.view(), right_large_direction.view());
        assert!((distance_large - 401.0_f64.sqrt()).abs() < 1e-12);

        assert!((distance_small - distance_large).abs() > 1e-12);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch: first 2, second 4")]
    fn distance_dimension_mismatch_panics() {
        let left = array![0.0, 0.0];
        let right = array![1.0, 2.0, 3.0, 4.0];
        let _ = Metric::SphereBundle.distance(left.view(), right.view());
    }

    #[test]
    #[should_panic(expected = "sphere bundle metric requires even dimension, got 3")]
    fn distance_odd_length_panics() {
        let left = array![0.0, 0.0, 0.0];
        let right = array![1.0, 2.0, 3.0];
        let _ = Metric::SphereBundle.distance(left.view(), right.view());
    }

    #[test]
    fn distance_allows_zero_direction_half() {
        // A zero direction half is a legal point: the direction factor is
        // measured on stored coordinates directly, with no normalization.
        let left = array![0.0, 0.0, 0.0, 0.0];
        let right = array![0.0, 0.0, 1.0, 0.0];
        let distance = Metric::SphereBundle.distance(left.view(), right.view());
        assert!((distance - 1.0).abs() < 1e-12);
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
        let metric = Metric::SphereBundle;

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
