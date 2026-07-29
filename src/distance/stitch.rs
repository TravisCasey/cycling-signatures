// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Joining per-tile component partitions into one partition over the window.
//!
//! Tiles overlap on the columns they share, so a cycle can be reported by more
//! than one tile. This pass discovers no edges of its own: it reunites tile
//! partitions that are each already internally connected, using the cycles they
//! report in common.

use std::{collections::hash_map::Entry, ops::Range};

use rustc_hash::FxHashMap;

use crate::util::disjoint::DisjointSet;

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
pub(super) fn stitch_per_tile_results(
    per_tile: Vec<Vec<Vec<Range<usize>>>>,
) -> Vec<Vec<Range<usize>>> {
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
