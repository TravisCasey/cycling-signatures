// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cube-path traversal for cycle segments.

use std::ops::Range;

use chomp3rs::{Cube, Orthant};

use super::EmbeddedTrajectory;
use crate::{
    cover::non_adjacent_axis,
    error::{Error, Result},
};

/// Walks the cubical path of the cycle described by `segment`, invoking
/// `visit` once per 1-cube edge traversed.
///
/// `segment` is the already-normalized half-open range of trajectory points.
/// The forward path walks consecutive points `segment.start..segment.end` in
/// order; the closing path connects `points[segment.end - 1]` back to
/// `points[segment.start]`. Each step, closing included, is one staircase
/// between the cubes of its two points.
///
/// Each step validates that the cube delta is in `{-1, 0, 1}` per axis. The
/// closing step, whose two points are the caller's cycle endpoints, returns
/// `CycleEndpointsNonAdjacent` when it exceeds that.
///
/// # Panics
///
/// Panics if a forward step exceeds the per-axis delta. Adjacency of
/// consecutive points holds across the whole trajectory once an embedding
/// exists, so this is a violated invariant rather than caller input error.
pub(super) fn for_each_cycle_edge<F>(
    embedded: &EmbeddedTrajectory,
    segment: Range<usize>,
    mut visit: F,
) -> Result<()>
where
    F: FnMut(&Cube),
{
    let cubes = embedded.cover().cubes();
    let point_to_cube_coords = |point_index: usize| -> Vec<i64> {
        let cube_index = embedded.point_to_cube(point_index);
        (0..cubes.ncols())
            .map(|axis| cubes[(cube_index, axis)])
            .collect()
    };

    let dimension = cubes.ncols();
    let start_cube = point_to_cube_coords(segment.start);
    // Cube coordinates fit in i32: CubicalCover::from_cubes enforces the
    // [i32::MIN, i32::MAX - 1] range for all coordinates.
    let mut base: Orthant = start_cube.iter().map(|&value| value as i32).collect();
    let mut dual: Orthant = start_cube.iter().map(|&value| value as i32 - 1).collect();

    // Forward path: each consecutive pair of points. Every consecutive pair of
    // the trajectory was shown to land in adjacent cubes before the embedding
    // existed,.
    for point_index in segment.start..(segment.end - 1) {
        let from = point_to_cube_coords(point_index);
        let to = point_to_cube_coords(point_index + 1);
        if let Err((axis, delta)) =
            step_between_cubes(&from, &to, dimension, &mut base, &mut dual, &mut visit)
        {
            panic!(
                "consecutive trajectory points {point_index} and {} land in cubes differing by \
                 {delta} in axis {axis}",
                point_index + 1
            );
        }
    }

    // Closing step: from points[end - 1] to points[start].
    let end_cube = point_to_cube_coords(segment.end - 1);
    step_between_cubes(
        &end_cube,
        &start_cube,
        dimension,
        &mut base,
        &mut dual,
        &mut visit,
    )
    .map_err(|(axis, delta)| Error::CycleEndpointsNonAdjacent {
        start: segment.start,
        end: segment.end,
        axis,
        delta,
    })?;

    Ok(())
}

/// Steps from `from` cube to `to` cube one axis-aligned unit at a time.
/// Positive axis diffs are processed first (axis 0..dim), then negative diffs.
/// Each unit step emits one 1-cube edge via `visit`.
///
/// Returns the offending axis and its signed cube delta if any axis difference
/// exceeds one unit in magnitude.
fn step_between_cubes<F>(
    from: &[i64],
    to: &[i64],
    dimension: usize,
    base: &mut Orthant,
    dual: &mut Orthant,
    visit: &mut F,
) -> std::result::Result<(), (usize, i64)>
where
    F: FnMut(&Cube),
{
    if let Some(gap) = non_adjacent_axis(from, to) {
        return Err(gap);
    }

    // Positive diffs first.
    for axis in 0..dimension {
        if to[axis] - from[axis] != 1 {
            continue;
        }
        dual[axis] += 1;
        let edge = Cube::new(base.clone(), dual.clone());
        visit(&edge);
        base[axis] += 1;
    }

    // Negative diffs second.
    for axis in 0..dimension {
        if to[axis] - from[axis] != -1 {
            continue;
        }
        base[axis] -= 1;
        let edge = Cube::new(base.clone(), dual.clone());
        visit(&edge);
        dual[axis] -= 1;
    }

    Ok(())
}
