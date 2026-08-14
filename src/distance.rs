// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Connected components of recurrent cycles over a trajectory segment.
//!
//! Detection reads the grid of endpoint pairs `(start, end)` over the segment
//! and admits the pairs whose two points lie strictly within the adjacency
//! threshold. An admitted pair is the cycle that runs along the trajectory from
//! `start` to `end` and closes back to `start`. A component is a connected
//! region of the admitted set in that grid: two admitted pairs are neighbors
//! when they share one endpoint and their other two endpoints are consecutive
//! trajectory points.
//!
//! Neighboring cycles are homologous in the cover, so a component carries a
//! single homology class. The two cycles differ by a loop that stays inside
//! the block of cubes spanned by their three distinct endpoints. Those three
//! points are pairwise below the cube side (two by admission, the consecutive
//! pair because an embedded trajectory's resolution clears it), which makes the
//! loop confined to their cubes contract.

mod stitch;
mod tile;
mod tile_components;

use std::ops::Range;

use chomp3rs::{ExecutionBackend, parallel::map::ParallelMap};
use stitch::stitch_per_tile_results;
use tile::detect_tile_components;
use tile_components::TileComponents;

use crate::{
    error::{Error, Result},
    metric::MetricPoints,
};

/// The number of columns each tile owns in the parallel distance
/// computations that back cycle detection.
///
/// The count does not affect the result: the emitted partition is the same at
/// every value. What it sets is cache locality and parallel dispatch
/// granularity.
///
/// Lowering tends to improve throughput up to a point, from cache residence
/// and locality. There is a redundant `1 / owned_columns` portion that grows
/// as the column count is reduced, but it stays small next to the throughput
/// gained. Smaller tiles also balance work across workers more evenly.
pub(crate) const DEFAULT_OWNED_COLUMNS: usize = 256;

/// Partitions `range` into tiles of `owned_columns` consecutive columns, each
/// extended by one read-ahead column where the window allows.
///
/// The read-ahead column is owned by the following tile, so cycles starting
/// there are emitted twice. That duplication is deliberate: the merge relation
/// joins a cycle to the one starting one step later, an edge that would
/// otherwise straddle a tile boundary, and the shared cycles give the stitching
/// pass the join key it needs to reunite the two sides.
#[must_use]
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

/// Connected components of the admitted endpoint pairs over `range`, computed
/// across tiles that each own `owned_columns` of it.
///
/// Per-tile work (distance-tile construction and the tile's own component
/// partition) is dispatched through `backend`. The stitching pass that merges
/// per-tile partitions into a global partition runs on the dispatching process.
///
/// Components that contain a length-1 cycle (a self-comparison, carrying the
/// trivial cycle) are dropped after the per-tile partitions have been merged.
///
/// Components are returned ordered by their least cycle under `(start, end)`,
/// with each component's cycles in that order. Since no two components share a
/// cycle, that order is total, which makes the whole returned partition a
/// function of the measured points, range and cap alone: it depends neither on
/// `owned_columns` nor on `backend`.
///
/// # Panics
///
/// Panics if `owned_columns` is 0, or if `points` holds more than
/// `u32::MAX` points.
///
/// # Errors
///
/// - [`Error::SegmentOutOfBounds`] if `range` is outside the point array's
///   index space.
pub(crate) fn detect_components(
    points: &MetricPoints<'_>,
    range: Range<usize>,
    max_length: usize,
    owned_columns: usize,
    backend: &ExecutionBackend,
) -> Result<Vec<Vec<Range<usize>>>> {
    if range.start > range.end || range.end > points.len() {
        return Err(Error::SegmentOutOfBounds {
            start: range.start,
            end: range.end,
            point_count: points.len(),
        });
    }
    assert!(owned_columns > 0, "a tile must own at least one column");
    // Tiles report cycle endpoints as point indices narrowed to `u32`.
    assert!(
        u32::try_from(points.len()).is_ok(),
        "trajectory point count exceeds the supported maximum"
    );

    // A cycle inside the window cannot outrun the window, so rows above its
    // length are infinite throughout and cost only allocation.
    let capped_length = max_length.min(range.len());
    if capped_length == 0 {
        return Ok(Vec::new());
    }

    let window_end = range.end;
    let tile_column_ranges = enumerate_tile_column_ranges(range, owned_columns);

    let mut per_tile: Vec<(usize, TileComponents)> = ParallelMap::new(backend).run(
        tile_column_ranges.into_iter(),
        |column_range: Range<usize>| {
            let start = column_range.start;
            let components =
                detect_tile_components(points, column_range, window_end, capped_length);
            vec![(start, components)]
        },
    );

    // Stitching is order-sensitive: it must see tiles in column-range order
    // to produce a deterministic global partition.
    per_tile.sort_by_key(|&(start, _)| start);
    Ok(stitch_per_tile_results(per_tile))
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use chomp3rs::ExecutionBackend;
    use ndarray::{Array2, array};

    use super::{DEFAULT_OWNED_COLUMNS, detect_components, tile::detect_tile_components};
    use crate::{
        Trajectory,
        error::{Error, Result},
        metric::Metric,
    };

    /// Runs detection over `trajectory` under the Euclidean metric.
    fn detect_euclidean(
        trajectory: &Trajectory,
        range: Range<usize>,
        max_length: usize,
        owned_columns: usize,
    ) -> Result<Vec<Vec<Range<usize>>>> {
        let points = Metric::Euclidean.over(trajectory.points());
        detect_components(
            &points,
            range,
            max_length,
            owned_columns,
            &ExecutionBackend::Sequential,
        )
    }

    fn small_trajectory() -> Trajectory {
        let points = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
        Trajectory::new(points.view()).unwrap()
    }

    #[test]
    fn rejects_segment_out_of_bounds() {
        let trajectory = small_trajectory();
        let err = detect_euclidean(&trajectory, 0..10, 5, 5).unwrap_err();
        assert!(matches!(err, Error::SegmentOutOfBounds { .. }));
    }

    #[test]
    fn a_zero_length_cap_detects_nothing() {
        // A cap of zero admits no cycle at all, so the partition is empty and
        // detection returns before any tile is laid out. Tiles require at
        // least one row and reject a zero cap outright.
        let trajectory = small_trajectory();
        let components = detect_euclidean(&trajectory, 0..5, 0, 5).unwrap();

        assert!(
            components.is_empty(),
            "expected no components under a zero-length cap, got {components:?}",
        );
    }

    #[test]
    fn straight_line_trajectory_emits_no_real_recurrence() {
        let trajectory = small_trajectory();
        let components = detect_euclidean(&trajectory, 0..5, 5, 5).unwrap();
        assert!(
            components.is_empty(),
            "expected no non-trivial components for a straight-line trajectory, got {components:?}",
        );
    }

    #[test]
    fn admission_is_strict_at_an_exact_cube_side_distance() {
        // Points 0 and 2 sit exactly one cube side apart, the only distance at
        // which the admission comparison's direction is visible.
        // The pair is not admitted, so nothing outside the self-comparisons is,
        // and every component carries the trivial cycle and is dropped.
        //
        // Point 1 is far from both, so the cell holding the exact-distance pair
        // has no admitted predecessor: were the pair admitted it would form a
        // component of its own rather than joining a trivial one, which is what
        // makes the drop below able to see the difference.
        let points = array![[0.0, 0.0], [5.0, 0.0], [1.0, 0.0]];
        let trajectory = Trajectory::new(points.view()).unwrap();

        let components = detect_euclidean(&trajectory, 0..3, 3, 3).unwrap();

        assert!(
            components.is_empty(),
            "a pair exactly one cube side apart must not be admitted; got {components:?}",
        );
    }

    #[test]
    fn merges_read_only_the_immediately_preceding_row() {
        // A cell's two merge predecessors both sit one row above it, and no
        // cell may merge with one further up. The fixture makes an older row
        // reachable if that rule is broken: column 0 is admitted at row 1
        // (points 0 and 1, 0.5 apart) and again at row 4 (points 0 and 4, 0.4
        // apart), with nothing admitted at column 0 in between, since points 0
        // and 2 are 5 apart and points 0 and 3 are 10 apart.
        //
        // Row 1's cell touches a self-comparison, so it carries the trivial
        // cycle. Row 4's must not reach it. What it does reach is row 3 column
        // 1, the pair of points 1 and 4 at distance 0.1, which reaches no
        // self-comparison of its own: points 1 and 3 are 9.5 apart and points 2
        // and 4 are 4.6 apart. So the two form one surviving component, and a
        // merge into row 1 would drop it as trivial and empty the partition.
        let points = array![[0.0, 0.0], [0.5, 0.0], [5.0, 0.0], [10.0, 0.0], [0.4, 0.0],];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let expected = vec![vec![0..5, 1..5]];

        for owned_columns in [5, 2] {
            let components = detect_euclidean(&trajectory, 0..5, 5, owned_columns).unwrap();
            assert_eq!(
                components, expected,
                "partition differs at owned_columns {owned_columns}",
            );
        }
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
            [0.8, 0.0],
            [1.6, 0.0],
            [1.6, 0.8],
            [1.6, 1.6],
            [0.8, 1.6],
            [0.0, 1.6],
            [0.0, 0.8],
            [0.0, 0.0],
        ];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let components = detect_euclidean(&trajectory, 0..9, 15, 9).unwrap();

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
                    cycle.end <= trajectory.len(),
                    "leaked padded entry as segment {cycle:?}",
                );
            }
        }
    }

    const SAMPLES_PER_REVOLUTION: usize = 40;

    /// A radius-6 circle traversed `revolutions` times at 40 samples each.
    ///
    /// The radius sets what the sweep admits: consecutive samples sit 0.94
    /// apart, so every consecutive pair is admitted and every wider one on a
    /// revolution is not.
    ///
    /// A cycle spanning whole revolutions closes on itself while its
    /// mid-revolution points stay far apart, so its merge chain does not reach
    /// down to a self-comparison and the component stays non-trivial. That is
    /// what a fixture needs here: recurrences that fold back through nearby
    /// intermediate points instead collapse the whole admitted set into the
    /// trivial component and leave nothing to compare.
    fn circling_trajectory(revolutions: usize) -> Trajectory {
        let count = revolutions * SAMPLES_PER_REVOLUTION;
        let mut coordinates = Vec::with_capacity(count * 2);
        for index in 0..count {
            let angle = index as f64 * std::f64::consts::TAU / SAMPLES_PER_REVOLUTION as f64;
            coordinates.push(6.0 * angle.cos());
            coordinates.push(6.0 * angle.sin());
        }
        let points = Array2::from_shape_vec((count, 2), coordinates).unwrap();
        Trajectory::new(points.view()).unwrap()
    }

    #[test]
    fn partition_is_identical_across_tilings() {
        // The emitted partition, ordering included, is a function of the
        // trajectory and the detection parameters alone: every tiling of the same
        // window produces the same vector, not merely the same set.
        let trajectory = circling_trajectory(7);
        let count = trajectory.len();
        // Reaches four revolutions, so four recurrence families are detected.
        let max_length = 170;
        assert!(count > DEFAULT_OWNED_COLUMNS, "fixture does not split");

        let detect = |owned_columns: usize| {
            detect_euclidean(&trajectory, 0..count, max_length, owned_columns).unwrap()
        };

        let reference = detect(count);
        assert!(
            reference.len() >= 2,
            "fixture detects {} components; every assertion below is vacuous under 2",
            reference.len(),
        );
        for owned_columns in [3, 7, 64, 199, DEFAULT_OWNED_COLUMNS] {
            assert_eq!(
                detect(owned_columns),
                reference,
                "partition differs at owned_columns {owned_columns}",
            );
        }

        // The documented order: components by their least cycle, and each
        // component's cycles by the same key.
        let key = |cycle: &Range<usize>| (cycle.start, cycle.end);
        for component in &reference {
            assert!(
                component
                    .windows(2)
                    .all(|pair| key(&pair[0]) < key(&pair[1])),
                "cycles are not ordered within {component:?}",
            );
        }
        assert!(
            reference
                .windows(2)
                .all(|pair| key(&pair[0][0]) < key(&pair[1][0])),
            "components are not ordered by their least cycle",
        );
    }

    #[test]
    fn boundary_same_end_merge_survives_partition() {
        // Cycles 7..12 and 8..12 share an endpoint and merge, but their starts
        // straddle the boundary between the tile owning columns 0..8 and the
        // one owning 8..14. The read-ahead column ensures they merge.
        let points = array![
            [0.25, 1.5],
            [0.75, 0.5],
            [0.25, 0.25],
            [0.5, 0.25],
            [0.5, 0.0],
            [0.25, 0.25],
            [1.0, 0.5],
            [0.5, 1.25],
            [0.0, 0.5],
            [1.25, 0.5],
            [0.0, 0.0],
            [0.25, 1.25],
            [0.25, 0.0],
            [1.0, 1.5],
        ];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let max_length = 34;
        let range = 0..points.nrows();

        let single_tile =
            detect_euclidean(&trajectory, range.clone(), max_length, points.nrows()).unwrap();

        let multi_tile = detect_euclidean(&trajectory, range, max_length, 8).unwrap();

        assert_eq!(single_tile, multi_tile);

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
    fn grid_adjacent_admitted_entries_share_a_component() {
        // Points 0, 3 and 4 form an equilateral triangle whose side is inside
        // the cube side, while points 1 and 2 sit far enough away that no
        // other pair is admitted. The pairs (0, 3) and (0, 4) are therefore
        // both admitted, share the endpoint at index 0, and are one row apart
        // in the endpoint-pair grid, so they belong to one component.
        let side = 0.85;
        let points = array![
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [side, 0.0],
            [side / 2.0, side * 3.0_f64.sqrt() / 2.0],
        ];
        let trajectory = Trajectory::new(points.view()).unwrap();
        let components = detect_euclidean(
            &trajectory,
            0..points.nrows(),
            points.nrows(),
            points.nrows(),
        )
        .unwrap();

        assert_eq!(
            components,
            vec![vec![0..4, 0..5]],
            "expected the two admitted pairs sharing point 0 in one component",
        );
    }

    #[test]
    fn globally_trivial_component_dropped_across_tiles() {
        // Whether a component carries the trivial cycle is decided over the
        // whole window rather than inside each tile, so the emitted partition
        // does not depend on how the window was tiled.
        //
        // Six owned columns split this window into the tiles 0..7 and 6..10,
        // which share column 6. Two components take the same shape in the first
        // tile: a cycle merged with the one step longer cycle beside it, and no
        // other admitted neighbor. On column 4 they are 4..9 and 4..10, whose
        // remaining neighbors 4..8, 5..9 and 5..10 are not admitted, so the
        // component reaches no self-comparison and survives. On column 6 they
        // are 6..9 and 6..10, where 6..8 is not admitted either and the
        // neighbors that start at point 7 lie on a column the first tile does
        // not carry.
        //
        // The second tile owns that column. Points 7 and 8 lie within the cube
        // side, so 7..9 is admitted there, merges with 6..9, and brings the
        // self-comparisons of both its endpoints into the component. The
        // stitching pass therefore drops it, and neither tiling emits 6..9 or
        // 6..10.
        let points = array![
            [0.5, 0.75],
            [0.75, 0.75],
            [1.0, 0.5],
            [1.25, 0.75],
            [1.0, 1.0],
            [1.25, 0.25],
            [0.5, 1.0],
            [0.0, 0.0],
            [0.25, 0.5],
            [0.25, 1.25],
        ];
        let trajectory = Trajectory::new(points.view()).unwrap();
        // No cap below the window, so every cycle the window holds is reached.
        let max_length = points.nrows();
        let range = 0..points.nrows();

        // The first tile's own view, which the stitching pass overrules.
        let measured = Metric::Euclidean.over(trajectory.points());
        let first_tile = detect_tile_components(&measured, 0..7, points.nrows(), max_length);
        let looks_non_trivial = (0..first_tile.len()).any(|index| {
            !first_tile.contains_trivial(index) && first_tile.cycles(index) == [(6, 9), (6, 10)]
        });
        assert!(
            looks_non_trivial,
            "the first tile must report 6..9 and 6..10 as a component carrying no trivial cycle",
        );

        let single_tile =
            detect_euclidean(&trajectory, range.clone(), max_length, points.nrows()).unwrap();
        let multi_tile = detect_euclidean(&trajectory, range, max_length, 6).unwrap();

        assert_eq!(single_tile, multi_tile);
        assert_eq!(
            multi_tile,
            vec![vec![4..9, 4..10]],
            "expected only the component on column 4 to survive",
        );
    }
}
