// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Connected components of below-threshold pair-edges over a trajectory
//! segment.

use std::ops::{Range, RangeBounds};

use ndarray::{Array2, ArrayView2};
use rustc_hash::FxHashMap;

use crate::{
    error::{Error, Result},
    metric::Metric,
    trajectory::Trajectory,
    util::{disjoint::DisjointSet, range::normalize_segment},
};

/// Connected components of below-threshold pair-edges over a trajectory
/// segment, with a cycle-length cap.
///
/// Each pair of original-index positions `(start, end)` with
/// `end - start <= max_length` and metric distance at or below `threshold`
/// represents a candidate cycle: the trajectory walks from `start` to `end - 1`
/// and closes directly. Two candidate cycles merge into the same component
/// when:
///
/// - they share exactly one endpoint, with the other two endpoint positions
///   adjacent in original-index space, and
/// - the three trajectory points involved satisfy [`Metric::covers_triple`]
///   with `radius = threshold / 2`.
///
/// The transitive closure of this relation partitions the below-threshold
/// candidates. Components that contain a length-1 entry (a self-comparison,
/// carrying the trivial cycle) are filtered before return; all remaining
/// components group cycles by signature equivalence.
///
/// Segments and returned cycle ranges are in original-index space
/// (`0..trajectory.original_count()`).
///
/// # Errors
///
/// - [`Error::WindowOutOfBounds`] if `segment` does not normalize to a valid
///   sub-range of `0..trajectory.original_count()`.
/// - [`Error::ThresholdBelowTrajectoryBound`] if `threshold <
///   trajectory.bound()`.
pub(crate) fn detect_components<M: Metric>(
    trajectory: &Trajectory<M>,
    segment: impl RangeBounds<usize>,
    threshold: f64,
    max_length: usize,
) -> Result<Vec<Vec<Range<usize>>>> {
    let range = normalize_segment(segment, trajectory.original_count())?;
    let trajectory_bound = trajectory.bound();
    if threshold < trajectory_bound {
        return Err(Error::ThresholdBelowTrajectoryBound {
            given: threshold,
            trajectory_bound,
        });
    }
    let base = range.start;
    let tile = build_distance_tile(trajectory, range, max_length)?;
    Ok(detect_components_in_tile(
        tile.view(),
        trajectory,
        base,
        threshold,
    ))
}

/// Connected components of below-threshold pair-edges over a pre-built
/// distance tile.
///
/// `tile` has shape `(max_length, width)`; `base` is the original-index
/// of the tile's first column (i.e., `tile[(row, col)]` is the distance
/// between `points[original_indices[base + col]]` and
/// `points[original_indices[base + col + row]]`).
///
/// Two adjacent entries are merged into the same component if and only if their
/// three involved trajectory points satisfy `metric.covers_triple` at radius
/// `threshold / 2`. Each emitted component is a list of cycle segments in
/// original-index space. Any component that has merged with a length-1 entry
/// (a self-comparison at `row = 0`) is filtered before return.
#[allow(clippy::needless_pass_by_value)]
fn detect_components_in_tile<M: Metric>(
    tile: ArrayView2<'_, f64>,
    trajectory: &Trajectory<M>,
    base: usize,
    threshold: f64,
) -> Vec<Vec<Range<usize>>> {
    let width = tile.ncols();
    let metric = trajectory.metric();
    let points = trajectory.points();
    let original_indices = trajectory.original_indices();
    let ball_radius = threshold / 2.0;

    let mut disjoint = DisjointSet::new();
    let mut entry_ids: FxHashMap<(usize, usize), usize> = FxHashMap::default();

    // Row-major pass: `Array2::indexed_iter` yields `((row, col), &value)`
    // in row-major order, which suits the predecessor convention (both
    // required predecessors are at `row - 1`).
    for ((row, col), &distance) in tile.indexed_iter() {
        if distance > threshold {
            continue;
        }
        let id = disjoint.insert();
        entry_ids.insert((row, col), id);

        // Shorter-by-one neighbor at (row - 1, col): same start
        // original_indices[base + col], cycle ends one step earlier.
        // Triple of trajectory points involved in the merge:
        //   shared left:        original_indices[base + col]
        //   right of shorter:   original_indices[base + col + row - 1]
        //   right of current:   original_indices[base + col + row]
        if row > 0
            && let Some(&shorter_id) = entry_ids.get(&(row - 1, col))
            && metric.covers_triple(
                points.row(original_indices[base + col]),
                points.row(original_indices[base + col + row - 1]),
                points.row(original_indices[base + col + row]),
                ball_radius,
            )
        {
            disjoint.union(id, shorter_id);
        }

        // Later-start-same-end neighbor at (row - 1, col + 1): same
        // right endpoint original_indices[base + col + row], start
        // shifts one later. Triple:
        //   shared right:       original_indices[base + col + row]
        //   left of later:      original_indices[base + col + 1]
        //   left of current:    original_indices[base + col]
        if row > 0
            && col + 1 < width
            && let Some(&later_id) = entry_ids.get(&(row - 1, col + 1))
            && metric.covers_triple(
                points.row(original_indices[base + col + row]),
                points.row(original_indices[base + col + 1]),
                points.row(original_indices[base + col]),
                ball_radius,
            )
        {
            disjoint.union(id, later_id);
        }
    }

    // Bucketing pass: iterate the tile again to preserve row-major ordering of
    // cycles within each component (short cycles first, ties broken by start
    // position).
    let mut component_index: FxHashMap<usize, usize> = FxHashMap::default();
    let mut components: Vec<Vec<Range<usize>>> = Vec::new();
    for ((row, col), _) in tile.indexed_iter() {
        let Some(&id) = entry_ids.get(&(row, col)) else {
            continue;
        };
        let component_id = *component_index.entry(disjoint.find(id)).or_insert_with(|| {
            components.push(Vec::new());
            components.len() - 1
        });
        // Cycle segment in original-index space: length = row + 1.
        let start = base + col;
        let end = base + col + row + 1;
        components[component_id].push(start..end);
    }

    // Trivial filter: drop components that contain a length-1 cycle (a
    // self-comparison at row 0). Such entries chain into a single trivial
    // component under the connectivity relation.
    components.retain(|cycles| !cycles.iter().any(|cycle| cycle.end <= cycle.start + 1));
    components
}

/// Builds a rectangular distance tile of shape `(max_length, width)` for the
/// segment of original indices `range`. Entry `tile[(row, col)]` holds the
/// metric distance between `points[original_indices[base + col]]` and
/// `points[original_indices[base + col + row]]`, where `base = range.start`
/// and `width = range.end - range.start`. Entries that would refer to past-end
/// original indices are populated with `f64::INFINITY`. Row 0 is the
/// self-comparison (`0.0` for valid columns).
///
/// A cycle of length `L` registers at `row = L - 1` of the returned tile.
/// `max_length = 0` produces an empty tile; `max_length = 1` produces a tile of
/// self-comparisons that detection treats as trivial.
///
/// # Errors
///
/// - [`Error::WindowOutOfBounds`] if `range` is not a valid sub-range of
///   `0..trajectory.original_count()`.
fn build_distance_tile<M: Metric>(
    trajectory: &Trajectory<M>,
    range: Range<usize>,
    max_length: usize,
) -> Result<Array2<f64>> {
    let original_count = trajectory.original_count();
    if range.start > range.end || range.end > original_count {
        return Err(Error::WindowOutOfBounds {
            start: range.start,
            end: range.end,
            trajectory_length: original_count,
        });
    }

    let width = range.end - range.start;
    let height = max_length;
    let base = range.start;
    let original_indices = trajectory.original_indices();
    let metric = trajectory.metric();
    let points = trajectory.points();

    let mut tile = Array2::<f64>::from_elem((height, width), f64::INFINITY);
    for col in 0..width {
        let start_index = original_indices[base + col];
        // Row 0: self-comparison.
        tile[(0, col)] = 0.0;
        // Rows 1..height: valid only while base + col + row < range.end.
        for row in 1..height {
            let original_offset = base + col + row;
            if original_offset >= range.end {
                break;
            }
            let end_index = original_indices[original_offset];
            tile[(row, col)] = metric.distance(points.row(start_index), points.row(end_index));
        }
    }

    Ok(tile)
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
        let err = detect_components(&trajectory, 0..10, 0.5, 5).unwrap_err();
        assert!(matches!(err, crate::error::Error::WindowOutOfBounds { .. }));
    }

    #[test]
    fn rejects_threshold_below_trajectory_bound() {
        let trajectory = small_trajectory();
        let err = detect_components(&trajectory, 0..5, 0.1, 5).unwrap_err();
        assert!(matches!(
            err,
            crate::error::Error::ThresholdBelowTrajectoryBound { given, trajectory_bound }
                if (given - 0.1).abs() < 1e-12 && (trajectory_bound - 0.5).abs() < 1e-12
        ));
    }

    #[test]
    fn straight_line_trajectory_emits_no_real_recurrence() {
        let trajectory = small_trajectory();
        let components = detect_components(&trajectory, 0..5, 0.5, 5).unwrap();
        assert!(
            components.is_empty(),
            "expected no non-trivial components for a straight-line trajectory, got {components:?}",
        );
    }

    #[test]
    fn padded_tile_entries_do_not_leak_into_components() {
        // 9-point square-trace fixture. With max_length=15 (> original_count=9),
        // the underlying tile has padding both above the diagonal of valid rows
        // and along its right edge. The non-trivial cycle 0..9 should still be
        // detected; padded entries must not produce any emitted segment.
        let points = ndarray::array![
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
        let components = detect_components(&trajectory, .., 0.6, 15).unwrap();

        let found = components.iter().any(|component| {
            component
                .iter()
                .any(|cycle| cycle.start == 0 && cycle.end == 9)
        });
        assert!(
            found,
            "expected cycle 0..9 in some component; got {components:?}"
        );

        for component in &components {
            for cycle in component {
                assert!(
                    cycle.end <= trajectory.original_count(),
                    "leaked padded entry as segment {cycle:?}",
                );
            }
        }
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
        let components = detect_components(&trajectory, .., 0.6, 9).unwrap();

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
