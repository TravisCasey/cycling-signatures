// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Connected components of below-threshold pair-edges over a trajectory
//! segment.

use std::{collections::hash_map::Entry, ops::Range};

use chomp3rs::{ExecutionBackend, parallel::map::ParallelMap};
use ndarray::{Array2, ArrayView2};
use rustc_hash::FxHashMap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    metric::{Metric, PreparedPoints},
    trajectory::Trajectory,
    util::disjoint::DisjointSet,
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

/// Partitions `range` into tiles of `owned_columns` consecutive columns, each
/// extended by one read-ahead column where the window allows.
///
/// The read-ahead column is owned by the following tile, so cycles starting
/// there are emitted twice. That duplication is deliberate: the merge relation
/// joins a cycle to the one starting one step later, an edge that would
/// otherwise straddle a tile boundary, and the shared cycles give the stitching
/// pass the join key it needs to reunite the two sides.
fn enumerate_tile_column_ranges(range: Range<usize>, owned_columns: usize) -> Vec<Range<usize>> {
    let mut column_ranges = Vec::new();
    let mut base = range.start;
    while base < range.end {
        let owned_end = (base + owned_columns).min(range.end);
        let tile_end = (owned_end + 1).min(range.end);
        column_ranges.push(base..tile_end);
        base = owned_end;
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
///
/// Components whose merged cycle list contains a length-1 cycle (a
/// self-comparison, carrying the trivial cycle) are dropped from the result.
fn stitch_per_tile_results(per_tile: Vec<Vec<Vec<Range<usize>>>>) -> Vec<Vec<Range<usize>>> {
    let mut global_id_of_cycle: FxHashMap<(u32, u32), u32> = FxHashMap::default();
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
                .filter_map(|cycle| {
                    global_id_of_cycle
                        .get(&(cycle.start as u32, cycle.end as u32))
                        .copied()
                })
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
                let key = (cycle.start as u32, cycle.end as u32);
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

    // Placed after compaction so the predicate sees each component's complete
    // cycle list. `retain` preserves the survivors' relative order, which the
    // class table built from this partition depends on.
    result.retain(|cycles| !cycles.iter().any(|cycle| cycle.end <= cycle.start + 1));

    result
}

/// Connected components of below-threshold pair-edges over `range`, computed
/// across tiles that each own `owned_columns` of it.
///
/// Per-tile work (distance-tile construction plus `detect_components_in_tile`)
/// is dispatched through `backend`. The stitching pass that merges per-tile
/// partitions into a global partition runs on the dispatching process.
///
/// Components that contain a length-1 cycle (a self-comparison, carrying the
/// trivial cycle) are dropped after the per-tile partitions have been merged.
/// The returned partition does not depend on how `range` is divided into tiles.
///
/// Alongside the components, returns the smallest distance over non-adjacent
/// candidate pairs in the band, positive infinity if every candidate pair is
/// adjacent.
///
/// # Panics
///
/// Panics if the trajectory holds more than `u32::MAX` samples.
///
/// # Errors
///
/// - [`Error::WindowOutOfBounds`] if `range` is outside the trajectory's
///   original-index space.
/// - [`Error::ThresholdExceedsAdjacencyBound`] if the threshold admits a
///   candidate endpoint pair in non-adjacent cubes.
pub(crate) fn detect_components(
    trajectory: &Trajectory,
    metric: Metric,
    range: Range<usize>,
    threshold: f64,
    max_length: usize,
    owned_columns: usize,
    backend: &ExecutionBackend,
) -> Result<(Vec<Vec<Range<usize>>>, f64)> {
    if range.start > range.end || range.end > trajectory.original_count() {
        return Err(Error::WindowOutOfBounds {
            start: range.start,
            end: range.end,
            trajectory_length: trajectory.original_count(),
        });
    }
    assert!(
        u32::try_from(trajectory.original_count()).is_ok(),
        "trajectory sample count exceeds the supported maximum"
    );

    // A cycle inside the window cannot outrun the window, so rows above its
    // length are infinite throughout and cost only allocation.
    let capped_length = max_length.min(range.len());
    if capped_length == 0 {
        return Ok((Vec::new(), f64::INFINITY));
    }

    let window_end = range.end;
    let tile_column_ranges = enumerate_tile_column_ranges(range, owned_columns);
    let prepared = metric.prepare(trajectory.points());

    // `build_distance_tile`'s only failure mode is `WindowOutOfBounds`,
    // already validated above against `trajectory.original_count()` before
    // dispatch, so it is unwrapped inside the closure.
    // `detect_components_in_tile` carries its failure case in its
    // `TileOutcome` return.
    let mut per_tile: Vec<(usize, TileOutcome)> = ParallelMap::new(backend).run(
        tile_column_ranges.into_iter(),
        |column_range: Range<usize>| {
            let start = column_range.start;
            let tile = build_distance_tile(
                trajectory,
                &prepared,
                column_range,
                window_end,
                capped_length,
            )
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
/// [`detect_components`]: it streams the same tiles but performs no
/// admission, merging, or component assembly.
///
/// # Errors
///
/// - [`Error::WindowOutOfBounds`] if `range` is outside the trajectory's
///   original-index space.
pub(crate) fn adjacency_bound(
    trajectory: &Trajectory,
    metric: Metric,
    range: Range<usize>,
    max_length: usize,
    owned_columns: usize,
    backend: &ExecutionBackend,
) -> Result<f64> {
    if range.start > range.end || range.end > trajectory.original_count() {
        return Err(Error::WindowOutOfBounds {
            start: range.start,
            end: range.end,
            trajectory_length: trajectory.original_count(),
        });
    }
    // A cycle inside the window cannot outrun the window, so rows above its
    // length are infinite throughout and cost only allocation.
    let capped_length = max_length.min(range.len());
    if capped_length == 0 {
        return Ok(f64::INFINITY);
    }

    let window_end = range.end;
    let tile_column_ranges = enumerate_tile_column_ranges(range, owned_columns);
    let points = trajectory.points();
    let original_indices = trajectory.original_indices();
    let prepared = metric.prepare(points);

    let per_tile: Vec<f64> = ParallelMap::new(backend).run(
        tile_column_ranges.into_iter(),
        |column_range: Range<usize>| {
            let base = column_range.start;
            let tile = build_distance_tile(
                trajectory,
                &prepared,
                column_range,
                window_end,
                capped_length,
            )
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
/// original-index space.
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
        // shifts one later.
        //
        // The `col + 1 < width` guard decides tile ownership of this edge. A
        // tile carries one column past the ones it owns, so for an owned
        // column the partner is present and the edge is found here. On that
        // final read-ahead column the guard fails, which is correct: the same
        // column is owned by the next tile, where the partner is present and
        // the edge is found instead. Every edge is therefore discovered
        // exactly once, and neither side is dropped at a boundary.
        //
        // Triple:
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

    TileOutcome::Components {
        components,
        non_adjacent_minimum,
    }
}

/// Builds a rectangular distance tile of shape `(max_length, width)` over the
/// original indices `columns`. Entry `tile[(row, col)]` holds the metric
/// distance between `points[original_indices[base + col]]` and
/// `points[original_indices[base + col + row]]`, where `base = columns.start`
/// and `width = columns.end - columns.start`.
///
/// Rows reach past the tile's own columns and stop at `window_end`, the end of
/// the analysis window: a cycle starting in this tile may finish outside it.
/// Entries that would refer to indices at or past `window_end` are populated
/// with `f64::INFINITY`. Row 0 is the self-comparison (`0.0` for every column).
///
/// A cycle of length `L` registers at `row = L - 1` of the returned tile.
/// `max_length` must be at least 1; `max_length = 1` produces a tile of
/// self-comparisons that detection treats as trivial.
///
/// # Panics
///
/// Panics if `max_length` is 0, which would leave row 0 with nowhere to write
/// its self-comparisons.
///
/// # Errors
///
/// - [`Error::WindowOutOfBounds`] if `columns` and `window_end` are not a valid
///   nested pair of sub-ranges of `0..trajectory.original_count()`.
fn build_distance_tile(
    trajectory: &Trajectory,
    prepared: &PreparedPoints,
    columns: Range<usize>,
    window_end: usize,
    max_length: usize,
) -> Result<Array2<f64>> {
    let original_count = trajectory.original_count();
    if columns.start > columns.end || columns.end > window_end || window_end > original_count {
        return Err(Error::WindowOutOfBounds {
            start: columns.start,
            end: columns.end,
            trajectory_length: original_count,
        });
    }
    assert!(max_length > 0, "distance tile needs at least one row");

    let width = columns.end - columns.start;
    let height = max_length;
    let base = columns.start;
    let original_indices = trajectory.original_indices();

    let mut tile = Array2::<f64>::from_elem((height, width), f64::INFINITY);
    for col in 0..width {
        let start_index = original_indices[base + col];
        // Row 0: self-comparison.
        tile[(0, col)] = 0.0;
        // Rows 1..height: valid only while base + col + row < window_end.
        for row in 1..height {
            let original_offset = base + col + row;
            if original_offset >= window_end {
                break;
            }
            let end_index = original_indices[original_offset];
            tile[(row, col)] = prepared.distance(start_index, end_index);
        }
    }

    Ok(tile)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, ops::Range};

    use chomp3rs::ExecutionBackend;
    use ndarray::{Array2, array};

    use super::detect_components;
    use crate::{
        Trajectory, embedded::DEFAULT_OWNED_COLUMNS, error::Error, metric::Metric,
        trajectory::max_consecutive_distance,
    };

    fn small_trajectory() -> Trajectory {
        let points = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]];
        Trajectory::new(points.view()).unwrap()
    }

    #[test]
    fn rejects_segment_out_of_bounds() {
        let trajectory = small_trajectory();
        let err = detect_components(
            &trajectory,
            Metric::Euclidean,
            0..10,
            0.5,
            5,
            5,
            &ExecutionBackend::Sequential,
        )
        .unwrap_err();
        assert!(matches!(err, Error::WindowOutOfBounds { .. }));
    }

    #[test]
    fn straight_line_trajectory_emits_no_real_recurrence() {
        let trajectory = small_trajectory();
        let (components, _) = detect_components(
            &trajectory,
            Metric::Euclidean,
            0..5,
            0.5,
            5,
            5,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        assert!(
            components.is_empty(),
            "expected no non-trivial components for a straight-line trajectory, got {components:?}",
        );
    }

    #[test]
    fn padded_tile_entries_do_not_leak_into_components() {
        // 9-point square-trace fixture. A cap larger than the segment is
        // clamped to its length, so the underlying tile is square and every
        // entry whose row would reach past the window end stays infinite: the
        // padding fills the triangle above the anti-diagonal. The non-trivial
        // cycle 0..9 should still be detected; padded entries must not produce
        // any emitted segment.
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
        let (components, _) = detect_components(
            &trajectory,
            Metric::Euclidean,
            0..9,
            0.6,
            15,
            9,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

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
    fn boundary_same_end_merge_survives_partition() {
        // Cycles 7..12 and 8..12 share an endpoint and merge, but their starts
        // straddle the boundary between the tile owning columns 0..8 and the
        // one owning 8..14. The read-ahead column ensures they merge.
        let points = array![
            [1.0, 0.5],
            [0.25, 0.75],
            [0.5, 0.0],
            [1.5, 1.25],
            [0.25, 0.5],
            [1.0, 0.75],
            [1.25, 0.75],
            [1.0, 0.75],
            [0.25, 1.25],
            [0.5, 0.0],
            [1.5, 0.5],
            [0.75, 1.5],
            [0.25, 0.0],
            [0.0, 0.5],
        ];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let max_length = 5;
        let threshold = 1.0;
        let range = 0..points.nrows();

        let (single_tile, _) = super::detect_components(
            &trajectory,
            Metric::Euclidean,
            range.clone(),
            threshold,
            max_length,
            points.nrows(),
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let (multi_tile, _) = super::detect_components(
            &trajectory,
            Metric::Euclidean,
            range,
            threshold,
            max_length,
            8,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        assert_eq!(canonicalize(single_tile), canonicalize(multi_tile.clone()));

        let boundary_pair_shares_a_component = multi_tile.iter().any(|component| {
            let holds = |start: usize, end: usize| {
                component
                    .iter()
                    .any(|cycle| cycle.start == start && cycle.end == end)
            };
            holds(7, 12) && holds(8, 12)
        });
        assert!(
            boundary_pair_shares_a_component,
            "expected 7..12 and 8..12 in one component; got {multi_tile:?}",
        );
    }

    #[test]
    fn default_owned_columns_splits_and_matches_single_tile() {
        // Only a segment longer than `DEFAULT_OWNED_COLUMNS` splits.
        let count = DEFAULT_OWNED_COLUMNS + 200;
        let positions: Vec<f64> = (0..count)
            .map(|index| (index as f64 * 0.4).sin() * 2.0)
            .collect();
        let points = Array2::from_shape_vec((count, 1), positions).unwrap();
        let trajectory = Trajectory::new(points.view()).unwrap();
        let bound = max_consecutive_distance(trajectory.points(), Metric::Euclidean);
        let threshold = bound.max(0.5);
        let max_length = 5;

        let (split, _) = detect_components(
            &trajectory,
            Metric::Euclidean,
            0..count,
            threshold,
            max_length,
            DEFAULT_OWNED_COLUMNS,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        let (whole, _) = detect_components(
            &trajectory,
            Metric::Euclidean,
            0..count,
            threshold,
            max_length,
            count,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        assert_eq!(canonicalize(split), canonicalize(whole));
    }

    #[test]
    fn globally_trivial_component_dropped_across_tiles() {
        // Cycle 6..9 closes at exactly the threshold and, inside a tile that
        // ends at column 9, has no admitted merge partner: it looks like an
        // isolated non-trivial component there. A tile reaching past column 9
        // sees it merge into the component carrying the self-comparisons, so
        // the cycle is trivial globally. Deciding triviality per tile would
        // therefore make the partition depend on the tiling.
        let points = array![
            [0.75, 1.5],
            [1.5, 1.0],
            [1.25, 1.5],
            [0.75, 1.0],
            [1.25, 1.25],
            [0.5, 0.0],
            [0.25, 0.25],
            [0.0, 0.5],
            [1.0, 1.25],
            [0.75, 0.75],
        ];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let max_length = 6;
        let threshold = 1.25;
        let range = 0..points.nrows();

        let (single_tile, _) = super::detect_components(
            &trajectory,
            Metric::Euclidean,
            range.clone(),
            threshold,
            max_length,
            points.nrows(),
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let (multi_tile, _) = super::detect_components(
            &trajectory,
            Metric::Euclidean,
            range,
            threshold,
            max_length,
            6,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        assert_eq!(canonicalize(single_tile.clone()), canonicalize(multi_tile));
        for component in &single_tile {
            for cycle in component {
                assert!(
                    !(cycle.start == 6 && cycle.end == 9),
                    "globally trivial cycle 6..9 survived into {single_tile:?}",
                );
            }
        }
    }

    #[test]
    fn single_tile_matches_multi_tile() {
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

        let (single_tile, _) = super::detect_components(
            &trajectory,
            Metric::Euclidean,
            range.clone(),
            threshold,
            max_length,
            range.len(),
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        let (multi_tile, _) = super::detect_components(
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
        let (components, _) = detect_components(
            &trajectory,
            Metric::Euclidean,
            0..9,
            0.6,
            9,
            9,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

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

        // The multi-tile pass agrees with the brute force, and the
        // piggybacked minimum from component detection agrees with the
        // standalone sweep.
        let standalone = super::adjacency_bound(
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
        let (_, piggybacked) = super::detect_components(
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
