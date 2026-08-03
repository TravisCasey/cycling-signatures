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
/// `tile` has shape `(height, width)`; `base` is the point index of the
/// tile's first column (i.e., `tile[(row, col)]` is the distance between
/// `points[base + col]` and `points[base + col + row]`).
///
/// # Panics
///
/// Panics if the tile holds more cells than an entry index can address.
fn partition_tile(tile: ArrayView2<'_, f64>, base: usize, threshold: f64) -> TileComponents {
    let height = tile.nrows();
    let width = tile.ncols();
    assert!(
        height * width < u32::MAX as usize,
        "distance tile holds more cells than an entry index can address"
    );

    let mut disjoint = DisjointSet::new();
    // Each cell's entry index, addressed by `row * width + col`, or `u32::MAX`
    // where the cell was not admitted. Every cell is written once and read at
    // most twice, both at known positions, so the table is indexed directly.
    let mut entry_ids: Vec<u32> = vec![u32::MAX; height * width];

    // Row-major pass, which suits the predecessor convention (both required
    // predecessors are at `row - 1`).
    for ((row, col), &distance) in tile.indexed_iter() {
        if distance > threshold {
            continue;
        }
        let id = disjoint.insert();
        entry_ids[row * width + col] = id as u32;

        // Shorter-by-one neighbor at (row - 1, col): same start
        // `base + col`, cycle ends one step earlier.
        if row > 0 {
            let shorter_id = entry_ids[(row - 1) * width + col];
            if shorter_id != u32::MAX {
                disjoint.union(id, shorter_id as usize);
            }
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
        if row > 0 && col + 1 < width {
            let later_id = entry_ids[(row - 1) * width + col + 1];
            if later_id != u32::MAX {
                disjoint.union(id, later_id as usize);
            }
        }
    }

    // Row 0 is the self-comparison at every column and is always admitted, so
    // the components carrying the trivial cycle are exactly those holding a
    // row-0 entry.
    let mut trivial_roots: FxHashSet<usize> = FxHashSet::default();
    for &id in &entry_ids[..width] {
        assert_ne!(
            id,
            u32::MAX,
            "a self-comparison is admitted at every column"
        );
        trivial_roots.insert(disjoint.find(id as usize));
    }

    // Bucketing pass: sweep the cells again in row-major order to preserve
    // that ordering of cycles within each component (short cycles first, ties
    // broken by start position).
    let mut component_index: FxHashMap<usize, usize> = FxHashMap::default();
    let mut grouped: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut contains_trivial: Vec<bool> = Vec::new();
    for (cell, &id) in entry_ids.iter().enumerate() {
        if id == u32::MAX {
            continue;
        }
        let (row, col) = (cell / width, cell % width);
        let root = disjoint.find(id as usize);
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

/// Builds a rectangular distance tile over the point indices `columns`. Entry
/// `tile[(row, col)]` holds the metric distance between `points[base + col]`
/// and `points[base + col + row]`, where `base = columns.start` and
/// `width = columns.end - columns.start`.
///
/// Rows reach past the tile's own columns and stop at `window_end`, the end of
/// the analysis window: a cycle starting in this tile may finish outside it.
/// Entries that would refer to indices at or past `window_end` are populated
/// with `f64::INFINITY`. Row 0 is the self-comparison (`0.0` for every column).
///
/// The tile has `min(max_length, window_end - base)` rows: past that, every
/// column of a row would reach outside the window, so the row is infinite
/// throughout and carries no admission.
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
    let base = columns.start;
    // A row beyond the window's reach from this tile's first column is
    // infinite in every column, so it admits nothing and is not allocated.
    let height = max_length.min(window_end - base);

    // Filled row by row, which is how the tile is laid out and how it is read
    // back: both endpoints of a row advance one point per column, so the fill
    // streams through the points as well as through the tile.
    let mut values: Vec<f64> = Vec::with_capacity(height * width);
    // Row 0: the self-comparison at every column.
    values.resize(width, 0.0);
    for row in 1..height {
        // A column at or past this bound would reach a point outside the
        // window, and stays at infinity.
        let measured_columns = (window_end - base).saturating_sub(row).min(width);
        for col in 0..measured_columns {
            values.push(points.distance(base + col, base + col + row));
        }
        values.resize(values.len() + (width - measured_columns), f64::INFINITY);
    }

    Ok(Array2::from_shape_vec((height, width), values)
        .expect("the fill writes one value per tile cell"))
}
