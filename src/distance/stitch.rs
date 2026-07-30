// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Joining per-tile component partitions into one partition over the window.
//!
//! Tiles overlap on the columns they share, so a cycle can be reported by more
//! than one tile. This pass discovers no edges of its own: it reunites tile
//! partitions that are each already internally connected, using the cycles they
//! report in common.

use std::{
    mem::{swap, take},
    ops::Range,
};

use rustc_hash::FxHashMap;

use super::tile_components::TileComponents;
use crate::util::disjoint::DisjointSet;

/// The components under construction, addressed by slot.
///
/// Merging is by union-find, so a slot obtained earlier stays usable after the
/// component it named has been merged into another: [`root`](Self::root) maps
/// it to the slot now holding the result.
///
/// A component that has absorbed a self-comparison carries the trivial cycle
/// and is never returned, so it discards its cycles and keeps only that fact.
/// Every method below preserves this, which is what bounds how much a
/// long-lived trivial component can accumulate.
///
/// [`close`](Self::close) leaves its slot empty rather than removing it, and a
/// closed slot must not be reached again. Nothing that survives a tile resolves
/// to one, and closing a component that holds no cycles panics.
struct OpenComponents {
    membership: DisjointSet,
    cycles: Vec<Vec<Range<usize>>>,
    contains_trivial: Vec<bool>,
}

impl OpenComponents {
    #[must_use]
    fn new() -> Self {
        Self {
            membership: DisjointSet::new(),
            cycles: Vec::new(),
            contains_trivial: Vec::new(),
        }
    }

    /// Adds an empty component and returns its slot.
    #[must_use]
    fn insert(&mut self) -> usize {
        let slot = self.membership.insert();
        self.cycles.push(Vec::new());
        self.contains_trivial.push(false);
        slot
    }

    /// The slot now holding the component `slot` named.
    #[must_use]
    fn root(&mut self, slot: usize) -> usize {
        self.membership.find(slot)
    }

    /// The distinct slots now holding the components `slots` named, sorted in
    /// ascending order.
    #[must_use]
    fn distinct_roots(&mut self, slots: impl IntoIterator<Item = usize>) -> Vec<usize> {
        let mut roots = Vec::new();
        for slot in slots {
            roots.push(self.root(slot));
        }
        roots.sort_unstable();
        roots.dedup();
        roots
    }

    /// Merges the components at `left` and `right`, returning the slot that now
    /// holds the result.
    ///
    /// The longer cycle list stays in place and the shorter is appended to it,
    /// so repeated merging stays linear in the total number of cycles.
    fn merge(&mut self, left: usize, right: usize) -> usize {
        let left_root = self.root(left);
        let right_root = self.root(right);
        if left_root == right_root {
            return left_root;
        }
        self.membership.union(left_root, right_root);

        let root = self.root(left_root);
        let vacated = if root == left_root {
            right_root
        } else {
            left_root
        };
        let vacated_trivial = self.contains_trivial[vacated];
        self.contains_trivial[root] |= vacated_trivial;

        let mut moved = take(&mut self.cycles[vacated]);
        if self.contains_trivial[root] {
            self.cycles[root] = Vec::new();
        } else {
            if self.cycles[root].len() < moved.len() {
                swap(&mut self.cycles[root], &mut moved);
            }
            self.cycles[root].append(&mut moved);
        }
        root
    }

    /// Records that the component at `root` carries the trivial cycle, dropping
    /// the cycles it will never return.
    fn record_trivial(&mut self, root: usize) {
        self.contains_trivial[root] = true;
        self.cycles[root] = Vec::new();
    }

    /// Adds `cycle` to the component at `root`, unless that component carries
    /// the trivial cycle and so will never be returned.
    fn push_cycle(&mut self, root: usize, cycle: Range<usize>) {
        if !self.contains_trivial[root] {
            self.cycles[root].push(cycle);
        }
    }

    /// Takes the cycles of the completed component at `root`, leaving its slot
    /// empty. Returns nothing when the component carries the trivial cycle.
    ///
    /// # Panics
    ///
    /// Panics if a component that does not carry the trivial cycle holds no
    /// cycles. Every component receives a cycle before it can be completed, so
    /// an empty one means its slot was reached after being closed.
    fn close(&mut self, root: usize) -> Option<Vec<Range<usize>>> {
        if self.contains_trivial[root] {
            return None;
        }
        let cycles = take(&mut self.cycles[root]);
        assert!(
            !cycles.is_empty(),
            "a completed component holds at least one cycle"
        );
        Some(cycles)
    }
}

/// Merges per-tile component partitions into a single global partition.
///
/// Input is each tile's first column paired with its components, in column
/// order. Input must satisfy the geometry `enumerate_tile_column_ranges`
/// produces: consecutive tiles share exactly one column and non-consecutive
/// tiles share none. So a cycle is emitted twice only when it starts on a
/// shared column, and two tile-components can be joined only when their tiles
/// are adjacent. The pass therefore carries a frontier: the components holding
/// a cycle on the column shared with the next tile. A component the frontier
/// does not carry is complete, since every later tile's columns lie beyond it.
///
/// Components whose merged cycle list contains a length-1 cycle (a
/// self-comparison, carrying the trivial cycle) are dropped.
///
/// Components are returned ordered by their least cycle under `(start, end)`,
/// with each component's cycles in that order. The order is determined by the
/// partition alone, so the result does not depend on how the window was tiled.
///
/// # Panics
///
/// Panics if the input does not satisfy that geometry, as a cycle on a shared
/// column would then have no counterpart to join and a component can be closed
/// before it has been given any cycle.
pub(super) fn stitch_per_tile_results(
    per_tile: Vec<(usize, TileComponents)>,
) -> Vec<Vec<Range<usize>>> {
    if per_tile.is_empty() {
        return Vec::new();
    }
    let next_bases: Vec<Option<usize>> = (1..per_tile.len())
        .map(|index| Some(per_tile[index].0))
        .chain([None])
        .collect();

    let mut components = OpenComponents::new();
    let mut result: Vec<Vec<Range<usize>>> = Vec::new();
    // Cycles on the column shared with the next tile, keyed by end: every such
    // cycle starts on that column, so the end alone identifies it.
    let mut frontier: FxHashMap<usize, usize> = FxHashMap::default();
    let mut carried_slots: Vec<usize> = Vec::new();

    for (tile_index, (base, tile_components)) in per_tile.into_iter().enumerate() {
        let shared_before = (tile_index > 0).then_some(base);
        let shared_after = next_bases[tile_index];
        let mut next_frontier: FxHashMap<usize, usize> = FxHashMap::default();
        let mut touched: Vec<usize> = Vec::new();

        for index in 0..tile_components.len() {
            // The cycles on the column shared with the previous tile name the
            // components this one continues, so they are looked up rather than
            // added: the preceding tile already holds them. Every such cycle
            // was reported there too, since the two tiles compute that column
            // from the same points over the same row range and so admit the
            // same entries, and a tile keeps its first and last columns even
            // when it reports nothing else of a component.
            let continued = components.distinct_roots(
                tile_components
                    .cycles(index)
                    .iter()
                    .filter(|&&(start, _)| Some(start as usize) == shared_before)
                    .map(|&(_, end)| {
                        *frontier
                            .get(&(end as usize))
                            .expect("a shared-column cycle was registered by the preceding tile")
                    })
                    .collect::<Vec<usize>>(),
            );

            let mut root = continued
                .first()
                .copied()
                .unwrap_or_else(|| components.insert());
            for &other in continued.iter().skip(1) {
                root = components.merge(root, other);
            }
            // Reported by the tile rather than re-derived from the cycles: a
            // component carrying the trivial cycle reports only its
            // shared-column cycles, so the self-comparison that made it trivial
            // is not among them.
            if tile_components.contains_trivial(index) {
                components.record_trivial(root);
            }

            for &(start, end) in tile_components.cycles(index) {
                let start = start as usize;
                if Some(start) == shared_before {
                    continue;
                }
                if Some(start) == shared_after {
                    next_frontier.insert(end as usize, root);
                }
                components.push_cycle(root, start..end as usize);
            }
            touched.push(root);
        }

        // A component the next tile cannot reach is complete. Nothing drains
        // the collection afterwards, so this is the only place a component is
        // emitted, and the final tile shares no column with a successor and so
        // closes everything.
        let surviving = components.distinct_roots(next_frontier.values().copied());
        let live = components.distinct_roots(carried_slots.iter().chain(&touched).copied());
        for root in live {
            if surviving.binary_search(&root).is_err() {
                result.extend(components.close(root));
            }
        }

        carried_slots = surviving;
        frontier = next_frontier;
    }

    for cycles in &mut result {
        cycles.sort_unstable_by_key(|cycle| (cycle.start, cycle.end));
    }
    result.sort_unstable_by_key(|cycles| {
        let least = cycles
            .first()
            .expect("a component holds at least one cycle");
        (least.start, least.end)
    });

    result
}

#[cfg(test)]
mod tests {
    use super::{TileComponents, stitch_per_tile_results};

    /// Builds one tile's input: its first column and its components, each given
    /// as its `(start, end)` cycles paired with the triviality it reports.
    ///
    /// The flag is given rather than derived from the cycles, because a tile
    /// reports it separately: a component carrying the trivial cycle lists only
    /// its shared-column cycles, so the self-comparison need not be among them.
    ///
    /// Every cycle starting on the tile's own first column must also appear in
    /// the preceding tile, which is what the real tiling guarantees and what
    /// the sweep relies on.
    fn tile(base: usize, components: &[(&[(usize, usize)], bool)]) -> (usize, TileComponents) {
        let grouped: Vec<Vec<(u32, u32)>> = components
            .iter()
            .map(|(cycles, _)| {
                cycles
                    .iter()
                    .map(|&(start, end)| (start as u32, end as u32))
                    .collect()
            })
            .collect();
        let contains_trivial = components.iter().map(|&(_, trivial)| trivial).collect();
        (
            base,
            TileComponents::from_grouped(&grouped, contains_trivial),
        )
    }

    #[test]
    fn component_spanning_non_adjacent_tiles_is_joined() {
        // Tiles 0 and 2 share no column, so their halves can only be joined
        // through the tile between them.
        let joined = stitch_per_tile_results(vec![
            tile(0, &[(&[(0, 3), (4, 7)], false)]),
            tile(4, &[(&[(4, 7), (8, 11)], false)]),
            tile(8, &[(&[(8, 11), (10, 13)], false)]),
        ]);
        assert_eq!(joined, vec![vec![0..3, 4..7, 8..11, 10..13]]);
    }

    #[test]
    fn two_frontier_slots_are_absorbed_into_one() {
        // One tile-component reaches back to two separate components of the
        // preceding tile, which merges them.
        let joined = stitch_per_tile_results(vec![
            tile(0, &[(&[(0, 2), (4, 6)], false), (&[(1, 3), (4, 9)], false)]),
            tile(4, &[(&[(4, 6), (4, 9)], false)]),
        ]);
        assert_eq!(joined, vec![vec![0..2, 1..3, 4..6, 4..9]]);
    }

    #[test]
    fn trivial_component_is_dropped_after_crossing_tiles() {
        // The self-comparison enters in the first tile, and the component it
        // belongs to is only completed two tiles later. It must still be
        // dropped, while its non-trivial neighbor survives.
        let joined = stitch_per_tile_results(vec![
            tile(0, &[(&[(0, 1), (4, 6)], true), (&[(0, 3)], false)]),
            tile(4, &[(&[(4, 6), (8, 10)], true)]),
            tile(8, &[(&[(8, 10), (9, 12)], true)]),
        ]);
        assert_eq!(joined, vec![vec![0..3]]);
    }

    #[test]
    fn reported_triviality_is_used_rather_than_rederived() {
        // What a pruned tile reports: a component that carries the trivial
        // cycle, but whose retained cycles are all longer than one point
        // because the self-comparison sat on an interior column and was
        // dropped. Re-deriving triviality from the cycles would read this as an
        // ordinary component and emit it.
        let joined = stitch_per_tile_results(vec![
            tile(0, &[(&[(0, 3), (4, 7)], true), (&[(1, 4)], false)]),
            tile(4, &[(&[(4, 7)], true)]),
        ]);
        assert_eq!(joined, vec![vec![1..4]]);
    }

    #[test]
    fn component_with_no_shared_cycle_is_emitted() {
        // No component here reaches a shared column, so each is complete as
        // soon as its own tile is read. Nothing drains them afterwards, so a
        // component closed at the wrong moment is lost rather than reordered.
        let joined = stitch_per_tile_results(vec![
            tile(0, &[(&[(0, 3)], false)]),
            tile(4, &[(&[(5, 8)], false)]),
            tile(8, &[(&[(9, 12)], false)]),
        ]);
        assert_eq!(joined, vec![vec![0..3], vec![5..8], vec![9..12]]);
    }
}
