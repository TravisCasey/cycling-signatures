// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Connected components of below-threshold pair-edges over a trajectory
//! segment.

mod stitch;
mod tile;
mod tile_components;

use std::ops::Range;

use chomp3rs::{ExecutionBackend, parallel::map::ParallelMap};
use stitch::stitch_per_tile_results;
use tile::{TileOutcome, detect_tile_components, tile_non_adjacent_minimum};

use crate::{
    error::{Error, Result},
    metric::Metric,
    trajectory::Trajectory,
};

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

/// Connected components of below-threshold pair-edges over `range`, computed
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
/// function of the trajectory, metric, range, threshold and cap alone: it
/// depends neither on `owned_columns` nor on `backend`.
///
/// Alongside the components, returns the smallest distance over non-adjacent
/// candidate pairs in the band, positive infinity if every candidate pair is
/// adjacent.
///
/// # Panics
///
/// Panics if `owned_columns` is 0.
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
    assert!(owned_columns > 0, "a tile must own at least one column");
    // Tiles report cycle endpoints as sample indices narrowed to `u32`.
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

    let mut per_tile: Vec<(usize, TileOutcome)> = ParallelMap::new(backend).run(
        tile_column_ranges.into_iter(),
        |column_range: Range<usize>| {
            let start = column_range.start;
            let outcome = detect_tile_components(
                trajectory,
                &prepared,
                metric,
                column_range,
                window_end,
                capped_length,
                threshold,
            );
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
    for (start, outcome) in per_tile {
        match outcome {
            TileOutcome::ThresholdExceeded { distance } => {
                return Err(Error::ThresholdExceedsAdjacencyBound {
                    threshold,
                    distance,
                });
            },
            TileOutcome::Components {
                components,
                non_adjacent_minimum: tile_minimum,
            } => {
                non_adjacent_minimum = non_adjacent_minimum.min(tile_minimum);
                tile_components.push((start, components));
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
/// # Panics
///
/// Panics if `owned_columns` is 0.
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
    assert!(owned_columns > 0, "a tile must own at least one column");

    // A cycle inside the window cannot outrun the window, so rows above its
    // length are infinite throughout and cost only allocation.
    let capped_length = max_length.min(range.len());
    if capped_length == 0 {
        return Ok(f64::INFINITY);
    }

    let window_end = range.end;
    let tile_column_ranges = enumerate_tile_column_ranges(range, owned_columns);
    let prepared = metric.prepare(trajectory.points());

    let per_tile: Vec<f64> = ParallelMap::new(backend).run(
        tile_column_ranges.into_iter(),
        |column_range: Range<usize>| {
            vec![tile_non_adjacent_minimum(
                trajectory,
                &prepared,
                column_range,
                window_end,
                capped_length,
            )]
        },
    );

    Ok(per_tile.into_iter().fold(f64::INFINITY, f64::min))
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

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

    const SAMPLES_PER_REVOLUTION: usize = 40;
    /// Consecutive samples of [`circling_trajectory`] sit this far apart, so
    /// any threshold above it clears the trajectory's own resolution.
    const CIRCLING_THRESHOLD: f64 = 0.5;

    /// A radius-3 circle traversed `revolutions` times at 40 samples each.
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
            coordinates.push(3.0 * angle.cos());
            coordinates.push(3.0 * angle.sin());
        }
        let points = Array2::from_shape_vec((count, 2), coordinates).unwrap();
        Trajectory::new(points.view()).unwrap()
    }

    #[test]
    fn partition_is_identical_across_tilings() {
        // The emitted partition, ordering included, is a function of the
        // trajectory and the detection parameters alone: every tiling of the same
        // window produces the same vector, not merely the same set.
        let trajectory = circling_trajectory(5);
        let count = trajectory.original_count();
        let threshold = CIRCLING_THRESHOLD;
        // Reaches four revolutions, so four recurrence families are detected.
        let max_length = 170;

        let detect = |owned_columns: usize| {
            detect_components(
                &trajectory,
                Metric::Euclidean,
                0..count,
                threshold,
                max_length,
                owned_columns,
                &ExecutionBackend::Sequential,
            )
            .unwrap()
            .0
        };

        let reference = detect(count);
        assert!(
            reference.len() >= 2,
            "fixture detects {} components; every assertion below is vacuous under 2",
            reference.len(),
        );
        for owned_columns in [3, 7, 64, 199] {
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
        let max_length = 34;
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
    fn default_owned_columns_splits_and_matches_single_tile() {
        // Only a segment longer than `DEFAULT_OWNED_COLUMNS` splits, so the
        // fixture is sized past it to exercise the shipped value.
        let trajectory = circling_trajectory(31);
        let count = trajectory.original_count();
        assert!(count > DEFAULT_OWNED_COLUMNS, "fixture does not split");
        // One revolution plus slack, enough to admit the once-per-revolution
        // recurrence without making the tile expensive.
        let max_length = 45;

        let (split, _) = detect_components(
            &trajectory,
            Metric::Euclidean,
            0..count,
            CIRCLING_THRESHOLD,
            max_length,
            DEFAULT_OWNED_COLUMNS,
            &ExecutionBackend::Sequential,
        )
        .unwrap();
        let (whole, _) = detect_components(
            &trajectory,
            Metric::Euclidean,
            0..count,
            CIRCLING_THRESHOLD,
            max_length,
            count,
            &ExecutionBackend::Sequential,
        )
        .unwrap();

        // Without this the comparison can hold vacuously on two empty
        // partitions, which is what a fold-back fixture silently produced here.
        assert!(!whole.is_empty(), "fixture detects no components");
        assert_eq!(split, whole);
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
        let max_length = 34;
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

        assert_eq!(single_tile, multi_tile);
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
        let max_length = 34;

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
