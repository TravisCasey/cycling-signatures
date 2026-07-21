// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Connected components of below-threshold pair-edges over a trajectory
//! segment.

use std::{
    collections::hash_map::Entry,
    ops::{Range, RangeBounds},
};

use chomp3rs::{ExecutionBackend, parallel::map::ParallelMap};
use ndarray::{Array2, ArrayView2};
use rustc_hash::FxHashMap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    metric::Metric,
    trajectory::Trajectory,
    util::{disjoint::DisjointSet, range::normalize_segment},
};

/// A single tile's detection outcome.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
enum TileOutcome {
    /// The tile's component partition and its smallest metric distance over
    /// non-adjacent candidate pairs (positive infinity if every candidate
    /// pair is adjacent).
    Components {
        components: Vec<Vec<Range<usize>>>,
        non_adjacent_minimum: f64,
    },
    /// Detection aborted: a candidate pair at or below the threshold landed
    /// in non-adjacent cubes. The distance certifies that the threshold
    /// exceeds the tile's adjacency bound.
    ThresholdExceeded { distance: f64 },
}

/// Default tile width for streaming banded-distance passes.
pub(crate) const DEFAULT_TILE_WIDTH: usize = 1024;

/// Lays out tile column ranges across `range` with stride `tile_width -
/// (max_length - 1)`. The last tile is right-clipped to the extent; all
/// preceding tiles have full width.
fn enumerate_tile_column_ranges(
    range: Range<usize>,
    tile_width: usize,
    max_length: usize,
) -> Vec<Range<usize>> {
    if max_length == 0 {
        return Vec::new();
    }

    let overlap = max_length - 1;
    let stride = tile_width - overlap;
    let mut column_ranges = Vec::new();
    let mut base = range.start;
    while base < range.end {
        let column_end = (base + tile_width).min(range.end);
        column_ranges.push(base..column_end);
        if column_end >= range.end {
            break;
        }
        base += stride;
    }
    column_ranges
}

/// Merges per-tile component partitions into a single global partition.
///
/// Input is the per-tile result vector in tile-index order; per-tile cycles
/// are deduplicated across tiles via their original-index ranges.
///
/// When a tile-component contains cycles previously assigned to different
/// global ids, those ids are merged via the global union-find; the final
/// compaction renumbers representatives to a contiguous range.
fn stitch_per_tile_results(per_tile: Vec<Vec<Vec<Range<usize>>>>) -> Vec<Vec<Range<usize>>> {
    let mut global_id_of_cycle: FxHashMap<(usize, usize), u32> = FxHashMap::default();
    let mut union_find = DisjointSet::new();
    // Per-global-id cycle accumulator. Outer index is the global id;
    // inner is the cycles registered under that id (before union compaction).
    let mut tile_cycle_lists: Vec<Vec<Range<usize>>> = Vec::new();

    for tile_components in per_tile {
        for tile_component in tile_components {
            // Collect existing global ids already assigned to cycles in this
            // tile-component (from earlier overlapping tiles).
            let mut existing_ids: Vec<u32> = tile_component
                .iter()
                .filter_map(|cycle| global_id_of_cycle.get(&(cycle.start, cycle.end)).copied())
                .collect();
            existing_ids.sort_unstable();
            existing_ids.dedup();

            // Pick the chosen id: reuse the smallest existing, or allocate.
            let chosen_id = if let Some(&first) = existing_ids.first() {
                first
            } else {
                let id = u32::try_from(union_find.insert())
                    .expect("global component id exceeds u32::MAX");
                tile_cycle_lists.push(Vec::new());
                id
            };

            // Union any other distinct ids into chosen_id.
            for &other_id in existing_ids.iter().skip(1) {
                union_find.union(chosen_id as usize, other_id as usize);
            }

            // Register every previously-unseen cycle under chosen_id.
            for cycle in tile_component {
                let key = (cycle.start, cycle.end);
                if let Entry::Vacant(entry) = global_id_of_cycle.entry(key) {
                    tile_cycle_lists[chosen_id as usize].push(cycle.clone());
                    entry.insert(chosen_id);
                }
            }
        }
    }

    // Compaction: collapse each cycle's global id through union-find, then
    // renumber representatives to a contiguous range.
    let mut representative_to_compact: FxHashMap<usize, usize> = FxHashMap::default();
    let mut result: Vec<Vec<Range<usize>>> = Vec::new();

    for (global_id, cycles) in tile_cycle_lists.into_iter().enumerate() {
        for cycle in cycles {
            let representative = union_find.find(global_id);
            let compact_id = *representative_to_compact
                .entry(representative)
                .or_insert_with(|| {
                    result.push(Vec::new());
                    result.len() - 1
                });
            result[compact_id].push(cycle);
        }
    }

    result
}

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
/// candidates. Components that contain a length 1 entry (a self-comparison,
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
/// - [`Error::ThresholdExceedsAdjacencyBound`] if the threshold admits a
///   candidate endpoint pair in non-adjacent cubes.
pub(crate) fn detect_components(
    trajectory: &Trajectory,
    metric: Metric,
    segment: impl RangeBounds<usize>,
    threshold: f64,
    max_length: usize,
) -> Result<Vec<Vec<Range<usize>>>> {
    let range = normalize_segment(segment, trajectory.original_count())?;
    // A tile at least as wide as the segment holds the whole segment in one
    // tile, so this routes through the streaming path without splitting into
    // multiple tiles.
    let tile_width = range.len().max(max_length);
    let (components, _adjacency_bound) = detect_components_streaming(
        trajectory,
        metric,
        range,
        threshold,
        max_length,
        tile_width,
        &ExecutionBackend::Sequential,
    )?;
    Ok(components)
}

/// Connected components of below-threshold pair-edges over `range`, streamed
/// across tiles of width `tile_width`.
///
/// Per-tile work (distance-tile construction plus `detect_components_in_tile`)
/// is dispatched through `backend`. The stitching pass that merges per-tile
/// partitions into a global partition runs on the dispatching process.
///
/// Alongside the components, returns the smallest distance over non-adjacent
/// candidate pairs in the band, positive infinity if every candidate pair is
/// adjacent.
///
/// # Errors
///
/// - [`Error::WindowOutOfBounds`] if `range` is outside the trajectory's
///   original-index space.
/// - [`Error::InvalidMaxLength`] if `max_length > tile_width`.
/// - [`Error::ThresholdExceedsAdjacencyBound`] if the threshold admits a
///   candidate endpoint pair in non-adjacent cubes.
pub(crate) fn detect_components_streaming(
    trajectory: &Trajectory,
    metric: Metric,
    range: Range<usize>,
    threshold: f64,
    max_length: usize,
    tile_width: usize,
    backend: &ExecutionBackend,
) -> Result<(Vec<Vec<Range<usize>>>, f64)> {
    if range.start > range.end || range.end > trajectory.original_count() {
        return Err(Error::WindowOutOfBounds {
            start: range.start,
            end: range.end,
            trajectory_length: trajectory.original_count(),
        });
    }
    if max_length > tile_width {
        return Err(Error::InvalidMaxLength { max_length });
    }

    let tile_column_ranges = enumerate_tile_column_ranges(range, tile_width, max_length);

    // `build_distance_tile`'s only failure mode is `WindowOutOfBounds`,
    // already validated above against `trajectory.original_count()` before
    // dispatch, so it is unwrapped inside the closure.
    // `detect_components_in_tile` carries its failure case in its
    // `TileOutcome` return.
    let mut per_tile: Vec<(usize, TileOutcome)> = ParallelMap::new(backend).run(
        tile_column_ranges.into_iter(),
        |column_range: Range<usize>| {
            let start = column_range.start;
            let tile = build_distance_tile(trajectory, metric, column_range, max_length)
                .expect("tile column range fits inside the validated window");
            let outcome =
                detect_components_in_tile(tile.view(), trajectory, metric, start, threshold);
            vec![(start, outcome)]
        },
    );

    // Stitching is order-sensitive: it must see tiles in column-range order
    // to produce a deterministic global partition; taking the first error in
    // tile order keeps the reported exceeding distance deterministic under
    // parallel dispatch.
    per_tile.sort_by_key(|&(start, _)| start);
    let mut non_adjacent_minimum = f64::INFINITY;
    let mut tile_components = Vec::new();
    for (_, outcome) in per_tile {
        match outcome {
            TileOutcome::ThresholdExceeded { distance } => {
                return Err(Error::ThresholdExceedsAdjacencyBound {
                    threshold,
                    distance,
                });
            },
            TileOutcome::Components {
                components,
                non_adjacent_minimum: tile_non_adjacent_minimum,
            } => {
                non_adjacent_minimum = non_adjacent_minimum.min(tile_non_adjacent_minimum);
                tile_components.push(components);
            },
        }
    }
    Ok((
        stitch_per_tile_results(tile_components),
        non_adjacent_minimum,
    ))
}

/// The smallest metric distance between two candidate endpoint samples in
/// `range` whose cubes are not adjacent, over the banded pair set
/// `(sample, sample + offset)` with `1 <= offset < max_length`; positive
/// infinity if every candidate pair is adjacent.
///
/// This is the distance-only counterpart of
/// [`detect_components_streaming`]: it streams the same tiles but performs no
/// admission, merging, or component assembly.
///
/// # Errors
///
/// - [`Error::WindowOutOfBounds`] if `range` is outside the trajectory's
///   original-index space.
/// - [`Error::InvalidMaxLength`] if `max_length > tile_width`.
pub(crate) fn adjacency_bound_streaming(
    trajectory: &Trajectory,
    metric: Metric,
    range: Range<usize>,
    max_length: usize,
    tile_width: usize,
    backend: &ExecutionBackend,
) -> Result<f64> {
    if range.start > range.end || range.end > trajectory.original_count() {
        return Err(Error::WindowOutOfBounds {
            start: range.start,
            end: range.end,
            trajectory_length: trajectory.original_count(),
        });
    }
    if max_length > tile_width {
        return Err(Error::InvalidMaxLength { max_length });
    }

    let tile_column_ranges = enumerate_tile_column_ranges(range, tile_width, max_length);
    let points = trajectory.points();
    let original_indices = trajectory.original_indices();

    let per_tile: Vec<f64> = ParallelMap::new(backend).run(
        tile_column_ranges.into_iter(),
        |column_range: Range<usize>| {
            let base = column_range.start;
            let tile = build_distance_tile(trajectory, metric, column_range, max_length)
                .expect("tile column range fits inside the validated window");
            let mut non_adjacent_minimum = f64::INFINITY;
            for ((row, col), &distance) in tile.indexed_iter() {
                // Row 0 is the self-comparison; padded entries hold infinity.
                if row > 0 && distance < non_adjacent_minimum {
                    let left_point = original_indices[base + col];
                    let right_point = original_indices[base + col + row];
                    if !cubes_adjacent(points, left_point, right_point) {
                        non_adjacent_minimum = distance;
                    }
                }
            }
            vec![non_adjacent_minimum]
        },
    );

    Ok(per_tile.into_iter().fold(f64::INFINITY, f64::min))
}

/// Returns `true` if the cubes of the two dense point rows differ by at most
/// 1 on every axis. Cube coordinates are component-wise floors, consistent
/// with the cover's cube mapping.
fn cubes_adjacent(points: ArrayView2<'_, f64>, left_point: usize, right_point: usize) -> bool {
    let left_row = points.row(left_point);
    let right_row = points.row(right_point);
    for axis in 0..left_row.len() {
        if (left_row[axis].floor() - right_row[axis].floor()).abs() > 1.0 {
            return false;
        }
    }
    true
}

/// Connected components of below-threshold pair-edges over a pre-built
/// distance tile, alongside the tile's smallest non-adjacent-pair distance.
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
///
/// Returns [`TileOutcome::ThresholdExceeded`] if any non-adjacent candidate
/// pair lies at or below `threshold`, reporting the first such pair in
/// row-major scan order: such a pair certifies that `threshold` exceeds the
/// tile's adjacency bound.
fn detect_components_in_tile(
    tile: ArrayView2<'_, f64>,
    trajectory: &Trajectory,
    metric: Metric,
    base: usize,
    threshold: f64,
) -> TileOutcome {
    let width = tile.ncols();
    let points = trajectory.points();
    let original_indices = trajectory.original_indices();
    let ball_radius = threshold / 2.0;
    let mut non_adjacent_minimum = f64::INFINITY;

    let mut disjoint = DisjointSet::new();
    let mut entry_ids: FxHashMap<(usize, usize), usize> = FxHashMap::default();

    // Row-major pass: `Array2::indexed_iter` yields `((row, col), &value)`
    // in row-major order, which suits the predecessor convention (both
    // required predecessors are at `row - 1`).
    for ((row, col), &distance) in tile.indexed_iter() {
        // Track the smallest distance over non-adjacent candidate pairs.
        // Row 0 is the self-comparison (trivially adjacent); padded entries
        // hold infinity and never undercut the running minimum.
        // An admitted non-adjacent pair proves the threshold exceeds the
        // window's adjacency bound: abort.
        // The running minimum stays above the threshold whenever
        // `TileOutcome::ThresholdExceeded` has not been returned, so only pairs
        // below the minimum need the cube comparison.
        if row > 0 && distance < non_adjacent_minimum {
            let left_point = original_indices[base + col];
            let right_point = original_indices[base + col + row];
            if !cubes_adjacent(points, left_point, right_point) {
                if distance <= threshold {
                    return TileOutcome::ThresholdExceeded { distance };
                }
                non_adjacent_minimum = distance;
            }
        }

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
    TileOutcome::Components {
        components,
        non_adjacent_minimum,
    }
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
fn build_distance_tile(
    trajectory: &Trajectory,
    metric: Metric,
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
    let points = trajectory.points();

    let mut tile = Array2::<f64>::from_elem((height, width), f64::INFINITY);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut positions: Vec<(usize, usize)> = Vec::new();
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
            pairs.push((start_index, end_index));
            positions.push((row, col));
        }
    }

    let mut buffer = vec![0.0_f64; pairs.len()];
    metric.fill_distances(points, &pairs, &mut buffer);
    for (&(row, col), &distance) in positions.iter().zip(&buffer) {
        tile[(row, col)] = distance;
    }

    Ok(tile)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, ops::Range};

    use chomp3rs::ExecutionBackend;
    use ndarray::{Array2, array};

    use super::detect_components;
    use crate::{Trajectory, error::Error, metric::Metric, trajectory::max_consecutive_distance};

    fn small_trajectory() -> Trajectory {
        let points = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]];
        Trajectory::new(points.view()).unwrap()
    }

    #[test]
    fn rejects_segment_out_of_bounds() {
        let trajectory = small_trajectory();
        let err = detect_components(&trajectory, Metric::Euclidean, 0..10, 0.5, 5).unwrap_err();
        assert!(matches!(err, Error::WindowOutOfBounds { .. }));
    }

    #[test]
    fn straight_line_trajectory_emits_no_real_recurrence() {
        let trajectory = small_trajectory();
        let components = detect_components(&trajectory, Metric::Euclidean, 0..5, 0.5, 5).unwrap();
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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let components = detect_components(&trajectory, Metric::Euclidean, .., 0.6, 15).unwrap();

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

    /// Renders a component partition as sets-of-sets for equality testing
    fn canonicalize(components: Vec<Vec<Range<usize>>>) -> BTreeSet<BTreeSet<(usize, usize)>> {
        components
            .into_iter()
            .map(|cycles| {
                cycles
                    .into_iter()
                    .map(|cycle_range| (cycle_range.start, cycle_range.end))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn streaming_single_tile_matches_multi_tile() {
        // Build a small 1D trajectory that forms a known recurrent loop and
        // exercises multiple tiles when tile_width is small.
        let positions: Vec<f64> = (0..30)
            .map(|index| (index as f64 * 0.4).sin() * 2.0)
            .collect();
        let points = Array2::from_shape_vec((30, 1), positions).unwrap();
        let trajectory = Trajectory::new(points.view()).unwrap();
        let bound = max_consecutive_distance(trajectory.points(), Metric::Euclidean);
        let threshold = bound.max(0.5);
        let max_length = 5;
        let range = 0..30;

        let (single_tile, _) = super::detect_components_streaming(
            &trajectory,
            Metric::Euclidean,
            range.clone(),
            threshold,
            max_length,
            range.len(),
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let (multi_tile, _) = super::detect_components_streaming(
            &trajectory,
            Metric::Euclidean,
            range.clone(),
            threshold,
            max_length,
            8,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        assert_eq!(canonicalize(single_tile), canonicalize(multi_tile));
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
        let trajectory = Trajectory::new(points.view()).unwrap();
        let components = detect_components(&trajectory, Metric::Euclidean, .., 0.6, 9).unwrap();

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

    #[test]
    fn adjacency_bound_matches_brute_force_on_sine_fixture() {
        let positions: Vec<f64> = (0..30)
            .map(|index| (index as f64 * 0.4).sin() * 2.0)
            .collect();
        let points = Array2::from_shape_vec((30, 1), positions).unwrap();
        let trajectory = Trajectory::new(points.view()).unwrap();
        let max_length = 5;

        let mut expected = f64::INFINITY;
        let trajectory_points = trajectory.points();
        for left in 0..30_usize {
            for right in (left + 1)..(left + max_length).min(30) {
                let left_row = trajectory_points.row(left);
                let right_row = trajectory_points.row(right);
                let adjacent = (left_row[0].floor() - right_row[0].floor()).abs() <= 1.0;
                if !adjacent {
                    let distance = Metric::Euclidean.distance(left_row, right_row);
                    expected = expected.min(distance);
                }
            }
        }

        // Multi-tile streaming agrees with the brute force, and the
        // piggybacked minimum from component detection agrees with the
        // standalone sweep.
        let standalone = super::adjacency_bound_streaming(
            &trajectory,
            Metric::Euclidean,
            0..30,
            max_length,
            8,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        assert!((standalone - expected).abs() < 1e-12);

        let bound = max_consecutive_distance(trajectory.points(), Metric::Euclidean);
        let (_, piggybacked) = super::detect_components_streaming(
            &trajectory,
            Metric::Euclidean,
            0..30,
            bound,
            max_length,
            8,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        assert!((piggybacked - expected).abs() < 1e-12);
    }
}
