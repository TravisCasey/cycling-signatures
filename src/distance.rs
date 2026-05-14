// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Distance matrices over trajectory segments.

use std::ops::{Range, RangeBounds};

use rustc_hash::FxHashMap;

use crate::{
    error::{Error, Result},
    metric::Metric,
    trajectory::Trajectory,
    util::{disjoint::DisjointSet, range::normalize_segment},
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
    /// [`Error::WindowOutOfBounds`] if `segment` falls outside
    /// `trajectory.points()`.
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

    /// Connected components of below-threshold matrix entries.
    ///
    /// Each entry represents a cycle segment that walks the trajectory
    /// between its two endpoint indices and closes directly. Two entries
    /// merge into the same component when both:
    ///
    /// - their cycle segments share exactly one endpoint, with the other two
    ///   endpoints adjacent in trajectory-index space, and
    /// - the three trajectory points involved (the shared endpoint and the two
    ///   distinct endpoints) satisfy [`Metric::covers_triple`] with `radius =
    ///   threshold / 2`.
    ///
    /// The transitive closure of this relation partitions the entries.
    /// Connectivity preserves cycling signature: all cycles within one
    /// component share a homology class.
    ///
    /// When `threshold` exceeds `trajectory.bound()`, every sub-threshold
    /// entry's cycle segment is a closed loop in the cubical cover with a
    /// well-defined signature, and components group entries by signature
    /// equivalence. Components reachable from a matrix-diagonal entry (a
    /// self-comparison, `col == 0`, carrying the trivial cycle) inherit the
    /// trivial signature and are filtered at emission.
    ///
    /// Each component is returned as the list of its cycle segments
    /// (`Range<usize>`, in trajectory-index space).
    ///
    /// `trajectory` must be the same trajectory the matrix was constructed
    /// from; the matrix carries no provenance for this and trusts the
    /// caller.
    ///
    /// # Errors
    ///
    /// [`Error::ThresholdBelowTrajectoryBound`] if
    /// `threshold < trajectory.bound()`.
    pub fn detect_components<M: Metric>(
        &self,
        trajectory: &Trajectory<M>,
        threshold: f64,
    ) -> Result<Vec<Vec<Range<usize>>>> {
        let trajectory_bound = trajectory.bound();
        if threshold < trajectory_bound {
            return Err(Error::ThresholdBelowTrajectoryBound {
                given: threshold,
                trajectory_bound,
            });
        }

        let points = trajectory.points();
        let metric = trajectory.metric();
        let ball_radius = threshold / 2.0;
        let base = self.range.start;
        let size = self.size();
        let anti_diagonal = self.iter_anti_diagonal();

        let mut disjoint = DisjointSet::new(0);
        let mut entry_ids: FxHashMap<(usize, usize), usize> = FxHashMap::default();

        for (row, col) in anti_diagonal.clone() {
            if self.get(row, col) > threshold {
                continue;
            }
            let id = disjoint.insert();
            entry_ids.insert((row, col), id);

            // Left neighbor (row, col - 1): shared endpoint x[base + row].
            // Triple: (x[base + row], x[base + row + col - 1], x[base + row + col]).
            if col > 0
                && let Some(&left_id) = entry_ids.get(&(row, col - 1))
                && metric.covers_triple(
                    points.row(base + row),
                    points.row(base + row + col - 1),
                    points.row(base + row + col),
                    ball_radius,
                )
            {
                disjoint.union(id, left_id);
            }

            // Up-right neighbor (row - 1, col + 1): shared endpoint x[base + row + col].
            // Triple: (x[base + row], x[base + row - 1], x[base + row + col]).
            if row > 0
                && col + 1 < size
                && let Some(&up_id) = entry_ids.get(&(row - 1, col + 1))
                && metric.covers_triple(
                    points.row(base + row),
                    points.row(base + row - 1),
                    points.row(base + row + col),
                    ball_radius,
                )
            {
                disjoint.union(id, up_id);
            }
        }

        let mut bucket_index: FxHashMap<usize, usize> = FxHashMap::default();
        let mut components: Vec<Vec<Range<usize>>> = Vec::new();
        for (row, col) in anti_diagonal {
            let Some(&id) = entry_ids.get(&(row, col)) else {
                continue;
            };
            let representative = disjoint.find(id);
            let bucket_id = *bucket_index.entry(representative).or_insert_with(|| {
                components.push(Vec::new());
                components.len() - 1
            });
            let start = base + row;
            let end = base + row + col + 1;
            components[bucket_id].push(start..end);
        }

        components.retain(|cycles| !cycles.iter().any(|cycle| cycle.end <= cycle.start + 1));
        Ok(components)
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

    #[test]
    fn detect_components_rejects_threshold_below_min() {
        let trajectory = small_trajectory();
        let matrix = DistanceMatrix::from_trajectory(&trajectory, 0..5).unwrap();
        // small_trajectory has consecutive distance 0.5, so trajectory.bound() == 0.5.
        let err = matrix.detect_components(&trajectory, 0.1).unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::ThresholdBelowTrajectoryBound { given, trajectory_bound }
                if (given - 0.1).abs() < 1e-12 && (trajectory_bound - 0.5).abs() < 1e-12
        ));
    }

    #[test]
    fn detect_components_filters_trivial_diagonal_and_emits_no_real_recurrence() {
        // A straight-line trajectory of 5 points has no genuine recurrence.
        // All cycles in the distance matrix are either zero-length
        // (col=0) or chain into the trivial component via the col=0 spine.
        // detect_components filters the trivial component and returns empty.
        let trajectory = small_trajectory();
        let matrix = DistanceMatrix::from_trajectory(&trajectory, 0..5).unwrap();
        let components = matrix.detect_components(&trajectory, 0.5).unwrap();
        assert!(
            components.is_empty(),
            "expected no non-trivial components for a straight-line trajectory, got {components:?}",
        );
    }

    #[test]
    fn detect_components_finds_a_known_recurrence() {
        // Trajectory that comes back near its start: a small square loop.
        let points = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [0.5, 1.0],
            [0.0, 1.0],
            [0.0, 0.5],
            [0.0, 0.0],
        ];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let matrix = DistanceMatrix::from_trajectory(&trajectory, ..).unwrap();
        let components = matrix.detect_components(&trajectory, 0.6).unwrap();

        // The loop-closure pair (0, 8) is below threshold (distance 0).
        // Should appear in at least one non-trivial component containing
        // the cycle segment 0..9.
        let found = components.iter().any(|component| {
            component
                .iter()
                .any(|cycle| cycle.start == 0 && cycle.end == 9)
        });
        assert!(
            found,
            "expected to find the loop-closing cycle 0..9 in some component; got {components:?}",
        );
    }
}
