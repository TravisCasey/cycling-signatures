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

use super::tile_components::TileComponents;
use crate::{
    error::{Error, Result},
    metric::MetricPoints,
    util::disjoint::DisjointSet,
};

/// Connected components of the admitted endpoint pairs over the tile spanning
/// `columns`.
///
/// Rows reach past the tile's own columns and stop at `window_end`. Two
/// entries adjacent in the tile are merged into the same component whenever
/// both are admitted. Each emitted component is a list of cycle segments.
///
/// # Panics
///
/// Panics if `columns` and `window_end` are not a valid nested pair of
/// sub-ranges of `0..points.len()`, or if `max_length` is 0.
pub(super) fn detect_tile_components(
    points: &MetricPoints<'_>,
    columns: Range<usize>,
    window_end: usize,
    max_length: usize,
    threshold: f64,
) -> TileComponents {
    let base = columns.start;
    let tile = build_distance_tile(points, columns, window_end, max_length)
        .expect("tile column range fits inside the validated window");
    partition_tile(tile.view(), base, threshold)
}

/// Connected components of the admitted endpoint pairs over a pre-built
/// distance tile.
///
/// `tile` has shape `(max_length, width)`; `base` is the point index of the
/// tile's first column (i.e., `tile[(row, col)]` is the distance between
/// `points[base + col]` and `points[base + col + row]`).
fn partition_tile(tile: ArrayView2<'_, f64>, base: usize, threshold: f64) -> TileComponents {
    let width = tile.ncols();

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
        // `base + col`, cycle ends one step earlier.
        if row > 0
            && let Some(&shorter_id) = entry_ids.get(&(row - 1, col))
        {
            disjoint.union(id, shorter_id);
        }

        // Later-start-same-end neighbor at (row - 1, col + 1): same right
        // endpoint `base + col + row`, start shifts one later.
        //
        // The `col + 1 < width` guard decides tile ownership of this edge. A
        // tile carries one column past the ones it owns, so for an owned
        // column the partner is present and the edge is found here. On that
        // final read-ahead column the guard fails, which is correct: the same
        // column is owned by the next tile, where the partner is present and
        // the edge is found instead. Every edge is therefore discovered
        // exactly once, and neither side is dropped at a boundary.
        if row > 0
            && col + 1 < width
            && let Some(&later_id) = entry_ids.get(&(row - 1, col + 1))
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
        // Cycle segment over trajectory points: length = row + 1.
        let start = base + col;
        let end = base + col + row + 1;
        grouped[component_id].push((start as u32, end as u32));
    }

    TileComponents::from_grouped(&grouped, contains_trivial)
}

/// Builds a rectangular distance tile of shape `(max_length, width)` over the
/// point indices `columns`. Entry `tile[(row, col)]` holds the metric distance
/// between `points[base + col]` and `points[base + col + row]`, where
/// `base = columns.start` and `width = columns.end - columns.start`.
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
///   nested pair of sub-ranges of `0..points.len()`.
fn build_distance_tile(
    points: &MetricPoints<'_>,
    columns: Range<usize>,
    window_end: usize,
    max_length: usize,
) -> Result<Array2<f64>> {
    let point_count = points.len();
    if columns.start > columns.end || columns.end > window_end || window_end > point_count {
        return Err(Error::WindowOutOfBounds {
            start: columns.start,
            end: columns.end,
            trajectory_length: point_count,
        });
    }
    assert!(max_length > 0, "distance tile needs at least one row");

    let width = columns.end - columns.start;
    let height = max_length;
    let base = columns.start;

    let mut tile = Array2::<f64>::from_elem((height, width), f64::INFINITY);
    for col in 0..width {
        let start_index = base + col;
        // Row 0: self-comparison.
        tile[(0, col)] = 0.0;
        // Rows 1..height: valid only while base + col + row < window_end.
        for row in 1..height {
            let end_index = base + col + row;
            if end_index >= window_end {
                break;
            }
            tile[(row, col)] = points.distance(start_index, end_index);
        }
    }

    Ok(tile)
}
