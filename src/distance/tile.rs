// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Detection over one tile of the banded pair-distance matrix.
//!
//! A tile covers a contiguous run of the analysis window's columns and reads
//! ahead to the window's end, so a cycle starting inside the tile can finish
//! outside it. Everything a pass needs from a tile is read here, and only the
//! tile's components cross the module boundary.

use std::{mem, ops::Range};

use rustc_hash::{FxHashMap, FxHashSet};

use super::tile_components::TileComponents;
use crate::{metric::MetricPoints, util::disjoint::DisjointSet};

/// Connected components of the admitted endpoint pairs over the tile spanning
/// `columns`.
///
/// Entry `(row, column)` of the tile is the pair of points
/// `base + column` and `base + column + row`, where `base = columns.start`.
/// Rows reach past the tile's own columns and stop at `window_end`, so a cycle
/// starting in this tile may finish outside it; entries that would refer to
/// indices at or past `window_end` are not measured and admit nothing. The
/// tile has `min(max_length, window_end - base)` rows, since past that every
/// column of a row would reach outside the window.
///
/// Row 0 is the self-comparison at every column, at distance `0.0`. A cycle of
/// length `L` registers at row `L - 1`; `max_length = 1` leaves only the
/// self-comparisons, which detection treats as trivial.
///
/// An entry is admitted when its two points lie strictly closer together than
/// 1, the cube side length. Two entries adjacent in the tile are merged into
/// the same component whenever both are admitted. Each emitted component is a
/// list of cycle segments.
///
/// # Panics
///
/// Panics if `columns` and `window_end` are not a valid nested pair of
/// sub-ranges of `0..points.len()`, if `max_length` is 0, or if the detection
/// window and cycle length cap together exceed the supported working set size.
#[must_use]
pub(super) fn detect_tile_components(
    points: &MetricPoints<'_>,
    columns: Range<usize>,
    window_end: usize,
    max_length: usize,
) -> TileComponents {
    assert!(
        columns.start <= columns.end && columns.end <= window_end && window_end <= points.len(),
        "tile columns must nest inside the analysis window, which must lie within the points"
    );
    assert!(max_length > 0, "a tile needs at least one row");

    let base = columns.start;
    let width = columns.end - columns.start;
    // A row beyond the window's reach from this tile's first column is
    // infinite in every column, so it admits nothing and is not swept.
    let height = max_length.min(window_end - base);
    assert!(
        height * width < u32::MAX as usize,
        "detection window and cycle length cap exceed the supported working set size"
    );

    let mut disjoint = DisjointSet::new();
    // The previous and current row's entry ids by column, or `u32::MAX` where
    // the cell was not admitted. Only these two rows are ever read: both merge
    // predecessors live at `row - 1`.
    let mut previous_row: Vec<u32> = vec![u32::MAX; width];
    let mut current_row: Vec<u32> = vec![u32::MAX; width];
    // Each admitted cell's position, in the order it was admitted. Entry ids
    // come from `disjoint.insert()`, which numbers sequentially, and admission
    // proceeds row-major, so a cell's index in this list is its entry id.
    let mut admitted: Vec<(u32, u32)> = Vec::new();

    // Row 0 is the self-comparison at every column, at distance 0, so it is
    // always admitted. It has no predecessors to merge with and is not subject
    // to the read-ahead bound below, since `base + column` is inside the window
    // for every column the tile carries. Sweeping it here keeps its three
    // special cases out of the loop that does the measuring.
    for (column, slot) in current_row.iter_mut().enumerate() {
        let id = disjoint.insert();
        *slot = id as u32;
        admitted.push((0, column as u32));
    }
    mem::swap(&mut previous_row, &mut current_row);

    for row in 1..height {
        // A column at or past this bound would reach a point outside the
        // window.
        let measured_columns = (window_end - base).saturating_sub(row).min(width);
        // The buffer being reused holds row `row - 2`, whose ids are no longer
        // predecessors of anything.
        current_row.fill(u32::MAX);

        for column in 0..measured_columns {
            let distance = points.distance(base + column, base + column + row);
            // Negated, so a NaN distance is admitted rather than skipped.
            if distance >= 1.0 {
                continue;
            }
            let id = disjoint.insert();
            current_row[column] = id as u32;
            admitted.push((row as u32, column as u32));

            // Shorter-by-one neighbor at (row - 1, column): same start
            // `base + column`, cycle ends one step earlier.
            let shorter_id = previous_row[column];
            if shorter_id != u32::MAX {
                disjoint.union(id, shorter_id as usize);
            }

            // Later-start-same-end neighbor at (row - 1, column + 1): same
            // right endpoint `base + column + row`, start shifts one later.
            //
            // The `column + 1 < width` guard decides tile ownership of this
            // edge. A tile carries one column past the ones it owns, so for an
            // owned column the partner is present and the edge is found here.
            // On that final read-ahead column the guard fails: the same column
            // is owned by the next tile, where the partner is present and the
            // edge is found instead.
            if column + 1 < width {
                let later_id = previous_row[column + 1];
                if later_id != u32::MAX {
                    disjoint.union(id, later_id as usize);
                }
            }
        }

        mem::swap(&mut previous_row, &mut current_row);
    }

    // Row 0 is admitted at every column, so ids `0..width` are exactly its
    // cells and the components carrying the trivial cycle are those holding
    // one.
    let mut trivial_roots: FxHashSet<usize> = FxHashSet::default();
    for id in 0..width {
        assert!(
            admitted.get(id).is_some_and(|&(row, _)| row == 0),
            "a self-comparison is admitted at every column"
        );
        trivial_roots.insert(disjoint.find(id));
    }

    // Bucketing pass, after every row has streamed: the merges keep firing
    // until the last row, so a cell's component is only known once the sweep
    // has finished. Walking the admitted cells in the order they were admitted
    // preserves the row-major ordering of cycles within each component (short
    // cycles first, ties broken by start position).
    let mut component_index: FxHashMap<usize, usize> = FxHashMap::default();
    let mut grouped: Vec<Vec<(u32, u32)>> = Vec::new();
    let mut contains_trivial: Vec<bool> = Vec::new();
    for (id, &(row, column)) in admitted.iter().enumerate() {
        let column = column as usize;
        let root = disjoint.find(id);
        let trivial = trivial_roots.contains(&root);
        // Only the columns a neighboring tile shares are kept for a component
        // carrying the trivial cycle; see [`TileComponents`].
        if trivial && column != 0 && column + 1 != width {
            continue;
        }
        let component_id = *component_index.entry(root).or_insert_with(|| {
            grouped.push(Vec::new());
            contains_trivial.push(trivial);
            grouped.len() - 1
        });
        // Cycle segment over trajectory points: length = row + 1.
        let start = base + column;
        let end = base + column + row as usize + 1;
        grouped[component_id].push((start as u32, end as u32));
    }

    TileComponents::from_grouped(&grouped, contains_trivial)
}
