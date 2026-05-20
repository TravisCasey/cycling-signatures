// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Connected components of below-threshold pair-edges over a trajectory
//! segment.

use std::ops::{Range, RangeBounds};

use rustc_hash::FxHashMap;

use crate::{
    error::{Error, Result},
    metric::Metric,
    trajectory::Trajectory,
    util::{disjoint::DisjointSet, range::normalize_segment},
};

/// Connected components of below-threshold pair-edges over a trajectory
/// segment.
///
/// Each entry of the pairwise distance matrix over
/// `trajectory.points()[segment]` represents a cycle segment that walks the
/// trajectory between its two endpoint indices and closes directly. Two such
/// entries merge into the same component when both:
///
/// - their cycle segments share exactly one endpoint, with the other two
///   endpoints adjacent in trajectory-index space, and
/// - the three trajectory points involved (the shared endpoint and the two
///   distinct endpoints) satisfy [`Metric::covers_triple`] with `radius =
///   threshold / 2`.
///
/// The transitive closure of this relation partitions the below-threshold
/// entries. Adjacency preserves cycling signature: every cycle in a given
/// component has the same homology class.
///
/// When `threshold >= trajectory.bound()`, every below-threshold entry's
/// cycle segment is a closed loop in the cubical cover with a well-defined
/// signature, and components group entries by signature equivalence.
/// Components reachable from a matrix-diagonal entry (a self-comparison,
/// `col == 0`, carrying the trivial cycle) inherit the trivial signature
/// and are filtered before return.
///
/// Each component is returned as the list of its cycle segments
/// (`Range<usize>`, in trajectory-index space).
///
/// # Errors
///
/// - [`Error::WindowOutOfBounds`] if `segment` does not normalize to a valid
///   sub-range of `trajectory.points()`.
/// - [`Error::ThresholdBelowTrajectoryBound`] if `threshold <
///   trajectory.bound()`.
#[allow(dead_code)]
pub(crate) fn detect_components<M: Metric>(
    trajectory: &Trajectory<M>,
    segment: impl RangeBounds<usize>,
    threshold: f64,
) -> Result<Vec<Vec<Range<usize>>>> {
    let range = normalize_segment(segment, trajectory.len())?;
    let trajectory_bound = trajectory.bound();
    if threshold < trajectory_bound {
        return Err(Error::ThresholdBelowTrajectoryBound {
            given: threshold,
            trajectory_bound,
        });
    }
    let matrix = DistanceMatrix::new(trajectory, range);
    Ok(matrix.detect_components(threshold))
}

struct DistanceMatrix<'a, M: Metric> {
    data: Vec<f64>,
    range: Range<usize>,
    trajectory: &'a Trajectory<M>,
}

impl<'a, M: Metric> DistanceMatrix<'a, M> {
    fn new(trajectory: &'a Trajectory<M>, range: Range<usize>) -> Self {
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

        Self {
            data,
            range,
            trajectory,
        }
    }

    fn size(&self) -> usize {
        self.range.end - self.range.start
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        let size = self.size();
        assert!(
            row + col < size,
            "distance matrix index ({row}, {col}) out of bounds for size {size}",
        );
        let offset = row * size - row.saturating_sub(1) * row / 2 + col;
        self.data[offset]
    }

    fn iter_anti_diagonal(&self) -> impl Iterator<Item = (usize, usize)> {
        (0..self.size())
            .flat_map(move |diagonal| (0..=diagonal).map(move |row| (row, diagonal - row)))
    }

    fn detect_components(&self, threshold: f64) -> Vec<Vec<Range<usize>>> {
        let points = self.trajectory.points();
        let metric = self.trajectory.metric();
        let base = self.range.start;

        let mut disjoint = DisjointSet::new(0);
        let mut entry_ids: FxHashMap<(usize, usize), usize> = FxHashMap::default();

        for (row, col) in self.iter_anti_diagonal() {
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
                    threshold / 2.0,
                )
            {
                disjoint.union(id, left_id);
            }

            // Up-right neighbor (row - 1, col + 1): shared endpoint x[base + row + col].
            // Triple: (x[base + row], x[base + row - 1], x[base + row + col]).
            if row > 0
                && col + 1 < self.size()
                && let Some(&up_id) = entry_ids.get(&(row - 1, col + 1))
                && metric.covers_triple(
                    points.row(base + row),
                    points.row(base + row - 1),
                    points.row(base + row + col),
                    threshold / 2.0,
                )
            {
                disjoint.union(id, up_id);
            }
        }

        let mut bucket_index: FxHashMap<usize, usize> = FxHashMap::default();
        let mut components: Vec<Vec<Range<usize>>> = Vec::new();
        for (row, col) in self.iter_anti_diagonal() {
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
        components
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::detect_components;
    use crate::{Trajectory, metric::Euclidean};

    fn small_trajectory() -> Trajectory<Euclidean> {
        let points = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]];
        Trajectory::new(points.view(), Euclidean).unwrap()
    }

    #[test]
    fn rejects_segment_out_of_bounds() {
        let trajectory = small_trajectory();
        let err = detect_components(&trajectory, 0..10, 0.5).unwrap_err();
        assert!(matches!(err, crate::error::Error::WindowOutOfBounds { .. }));
    }

    #[test]
    fn rejects_threshold_below_trajectory_bound() {
        let trajectory = small_trajectory();
        let err = detect_components(&trajectory, 0..5, 0.1).unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::ThresholdBelowTrajectoryBound { given, trajectory_bound }
                if (given - 0.1).abs() < 1e-12 && (trajectory_bound - 0.5).abs() < 1e-12
        ));
    }

    #[test]
    fn straight_line_trajectory_emits_no_real_recurrence() {
        let trajectory = small_trajectory();
        let components = detect_components(&trajectory, 0..5, 0.5).unwrap();
        assert!(
            components.is_empty(),
            "expected no non-trivial components for a straight-line trajectory, got {components:?}",
        );
    }

    #[test]
    fn detects_a_known_loop_closure() {
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
        let components = detect_components(&trajectory, .., 0.6).unwrap();

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
