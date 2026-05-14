// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Distance matrices over trajectory segments.

use std::ops::{Range, RangeBounds};

use crate::{
    error::Result, metric::Metric, trajectory::Trajectory, util::range::normalize_segment,
};

/// Pairwise metric distances over a contiguous segment of a trajectory.
///
/// Built from a [`Trajectory<M>`](crate::Trajectory) and a segment of
/// trajectory-point indices. Entries are indexed by `(row, col)`: `row` is the
/// segment-local offset of the first trajectory point and `col` is the gap to
/// the second, so [`get`](Self::get) returns the metric distance between
/// trajectory points at absolute indices `start + row` and `start + row + col`,
/// where `start` is the segment's first absolute index. Self-comparisons
/// (`col == 0`) are zero.
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    data: Vec<f64>,
    range: Range<usize>,
}

impl DistanceMatrix {
    /// Computes the matrix for the segment of `trajectory.points()`
    /// named by `segment`.
    ///
    /// `segment` is any `RangeBounds<usize>` (`a..b`, `a..=b`, `..b`, `..`, and
    /// so on). The resulting matrix covers exactly the trajectory points the
    /// segment describes, with self-comparisons set to zero.
    ///
    /// # Errors
    ///
    /// [`Error::WindowOutOfBounds`](crate::error::Error::WindowOutOfBounds)
    /// if `segment` falls outside `trajectory.points()`.
    pub fn from_trajectory<M: Metric>(
        trajectory: &Trajectory<M>,
        segment: impl RangeBounds<usize>,
    ) -> Result<Self> {
        let range = normalize_segment(segment, trajectory.len())?;
        let size = range.end - range.start;
        let points = trajectory.points();
        let metric = trajectory.metric();

        let mut data: Vec<f64> = Vec::with_capacity(size * (size + 1) / 2);
        for row in 0..size {
            for col in 0..(size - row) {
                let start_index = range.start + row;
                let end_index = start_index + col;
                let distance = if col == 0 {
                    0.0
                } else {
                    metric.distance(points.row(start_index), points.row(end_index))
                };
                data.push(distance);
            }
        }

        Ok(Self { data, range })
    }

    /// Iterates the matrix's `(row, col)` pairs in anti-diagonal order:
    /// ascending `row + col`, ties broken by ascending `row`. The returned
    /// iterator implements `Clone`, so callers can take two independent passes
    /// from the same starting position.
    #[allow(dead_code)]
    pub(crate) fn iter_anti_diagonal(&self) -> impl Iterator<Item = (usize, usize)> + Clone {
        let size = self.size();
        (0..size).flat_map(move |diagonal| (0..=diagonal).map(move |row| (row, diagonal - row)))
    }

    /// The half-open range of trajectory-point indices the matrix covers.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Number of trajectory points in the segment.
    #[must_use]
    pub fn size(&self) -> usize {
        self.range.end - self.range.start
    }

    /// The metric distance at local `(row, col)`. See the type-level
    /// documentation for the indexing convention.
    ///
    /// # Panics
    ///
    /// Panics if `row + col >= self.size()`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let size = self.size();
        assert!(
            row + col < size,
            "distance matrix index ({row}, {col}) out of bounds for size {size}",
        );
        let offset = row * size - row.saturating_sub(1) * row / 2 + col;
        self.data[offset]
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::DistanceMatrix;
    use crate::{Trajectory, metric::Euclidean};

    fn small_trajectory() -> Trajectory<Euclidean> {
        // Five points along a 2D path, all pairwise distances finite.
        let points = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]];
        Trajectory::new(points.view(), Euclidean).unwrap()
    }

    #[test]
    fn from_trajectory_validates_segment_bounds() {
        let trajectory = small_trajectory();
        let err = DistanceMatrix::from_trajectory(&trajectory, 0..10).unwrap_err();
        assert!(matches!(err, crate::error::Error::WindowOutOfBounds { .. }));
    }

    #[test]
    fn get_diagonal_is_zero_and_off_diagonal_matches_metric() {
        let trajectory = small_trajectory();
        let matrix = DistanceMatrix::from_trajectory(&trajectory, 0..5).unwrap();

        // Self-comparisons are hard-coded to 0.0 in the constructor; exact
        // equality is the correct test, not an epsilon tolerance.
        #[allow(clippy::float_cmp)]
        for row in 0..matrix.size() {
            assert_eq!(matrix.get(row, 0), 0.0);
        }

        // Off-diagonal: (row=0, col=2) is points[0] to points[2], distance 1.0.
        assert!((matrix.get(0, 2) - 1.0).abs() < 1e-12);
    }
}
