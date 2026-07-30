// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Detection over one tile of the banded pair-distance matrix.
//!
//! A tile covers a contiguous run of the analysis window's columns and reads
//! ahead to the window's end, so a cycle starting inside the tile can finish
//! outside it. Everything a pass needs from a tile is read here, and no tile
//! matrix crosses the module boundary.

use std::ops::Range;

use ndarray::{Array2, ArrayView2};
use rustc_hash::{FxHashMap, FxHashSet};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::tile_components::TileComponents;
use crate::{
    error::{Error, Result},
    metric::PreparedPoints,
    trajectory::Trajectory,
    util::disjoint::DisjointSet,
};

/// A single tile's detection outcome.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(super) enum TileOutcome {
    /// The tile's component partition and its smallest metric distance over
    /// non-adjacent candidate pairs (positive infinity if every candidate
    /// pair is adjacent).
    Components {
        components: TileComponents,
        non_adjacent_minimum: f64,
    },
    /// Detection aborted: a candidate pair at or below the threshold landed
    /// in non-adjacent cubes. The distance certifies that the threshold
    /// exceeds the tile's adjacency bound.
    ThresholdExceeded { distance: f64 },
}

/// Connected components of below-threshold pair-edges over the tile spanning
/// `columns`, alongside the tile's smallest non-adjacent-pair distance.
///
/// Rows reach past the tile's own columns and stop at `window_end`. Two
/// adjacent entries are merged into the same component if and only if balls of
/// radius `threshold / 2` around their three involved trajectory points share
/// a common point. Each emitted component is a list of cycle segments in
/// original-index space.
///
/// Returns [`TileOutcome::ThresholdExceeded`] if any non-adjacent candidate
/// pair lies at or below `threshold`, reporting the first such pair in
/// row-major scan order: such a pair certifies that `threshold` exceeds the
/// tile's adjacency bound.
///
/// # Panics
///
/// Panics if `columns` and `window_end` are not a valid nested pair of
/// sub-ranges of `0..trajectory.original_count()`, or if `max_length` is 0.
pub(super) fn detect_tile_components(
    trajectory: &Trajectory,
    prepared: &PreparedPoints,
    columns: Range<usize>,
    window_end: usize,
    max_length: usize,
    threshold: f64,
) -> TileOutcome {
    let base = columns.start;
    let tile = build_distance_tile(trajectory, prepared, columns, window_end, max_length)
        .expect("tile column range fits inside the validated window");
    partition_tile(tile.view(), trajectory, prepared, base, threshold)
}

/// The smallest metric distance between two candidate endpoint samples of the
/// tile spanning `columns` whose cubes are not adjacent; positive infinity if
/// every candidate pair is adjacent.
///
/// The distance-only counterpart of [`detect_tile_components`]: it reads the
/// same tile but performs no admission, merging, or component assembly.
///
/// # Panics
///
/// Panics if `columns` and `window_end` are not a valid nested pair of
/// sub-ranges of `0..trajectory.original_count()`, or if `max_length` is 0.
pub(super) fn tile_non_adjacent_minimum(
    trajectory: &Trajectory,
    prepared: &PreparedPoints,
    columns: Range<usize>,
    window_end: usize,
    max_length: usize,
) -> f64 {
    let base = columns.start;
    let tile = build_distance_tile(trajectory, prepared, columns, window_end, max_length)
        .expect("tile column range fits inside the validated window");
    let points = trajectory.points();
    let original_indices = trajectory.original_indices();

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
    non_adjacent_minimum
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
fn partition_tile(
    tile: ArrayView2<'_, f64>,
    trajectory: &Trajectory,
    prepared: &PreparedPoints,
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
            && prepared.covers_triple(
                original_indices[base + col],
                original_indices[base + col + row - 1],
                original_indices[base + col + row],
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
            && prepared.covers_triple(
                original_indices[base + col + row],
                original_indices[base + col + 1],
                original_indices[base + col],
                ball_radius,
            )
        {
            disjoint.union(id, later_id);
        }
    }

    // Row 0 is the self-comparison at every column and is always admitted, so
    // the components carrying the trivial cycle are exactly those holding a
    // row-0 entry.
    let mut trivial_roots: FxHashSet<usize> = FxHashSet::default();
    for col in 0..width {
        let id = entry_ids[&(0, col)];
        trivial_roots.insert(disjoint.find(id));
    }

    // Bucketing pass: iterate the tile again to preserve row-major ordering of
    // cycles within each component (short cycles first, ties broken by start
    // position).
    let mut component_index: FxHashMap<usize, usize> = FxHashMap::default();
    let mut grouped: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut contains_trivial: Vec<bool> = Vec::new();
    for ((row, col), _) in tile.indexed_iter() {
        let Some(&id) = entry_ids.get(&(row, col)) else {
            continue;
        };
        let root = disjoint.find(id);
        let trivial = trivial_roots.contains(&root);
        // Only the columns a neighboring tile shares are kept for a component
        // carrying the trivial cycle; see [`TileComponents`].
        if trivial && col != 0 && col + 1 != width {
            continue;
        }
        let component_id = *component_index.entry(root).or_insert_with(|| {
            grouped.push(Vec::new());
            contains_trivial.push(trivial);
            grouped.len() - 1
        });
        // Cycle segment in original-index space: length = row + 1.
        let start = base + col;
        let end = base + col + row + 1;
        grouped[component_id].push((start as u32, end as u32));
    }

    TileOutcome::Components {
        components: TileComponents::from_grouped(&grouped, contains_trivial),
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
