// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cube-path traversal for cycle segments.

use std::ops::Range;

use chomp3rs::{Cube, Orthant};
use ndarray::Array2;

use super::EmbeddedTrajectory;
use crate::{
    cover::floor_to_cube,
    error::{Error, Result},
    trajectory::Trajectory,
};

/// Walks the cubical path of the cycle described by `segment`, invoking
/// `visit` once per 1-cube edge traversed.
///
/// `segment` is the already-normalized half-open range in **point-index
/// space** (i.e., indices into `trajectory.points()`); the caller has
/// translated from sample-index space if needed and validated endpoint
/// cube adjacency. The forward path walks consecutive points
/// `segment.start..segment.end` in order; the closing path connects
/// `points[segment.end - 1]` back to `points[segment.start]` via direct
/// cube-to-cube steps.
///
/// Each step validates that the cube delta is in `{-1, 0, 1}` per axis,
/// returning `ConsecutiveCubesNonAdjacent` on violation.
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

    // Forward path: each consecutive pair of points.
    for point_index in segment.start..(segment.end - 1) {
        let from = point_to_cube_coords(point_index);
        let to = point_to_cube_coords(point_index + 1);
        step_between_cubes(&from, &to, dimension, &mut base, &mut dual, &mut visit).map_err(
            |step| Error::ConsecutiveCubesNonAdjacent {
                point_index,
                axis: step.axis,
                delta: step.delta,
            },
        )?;
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
    .map_err(|step| Error::ConsecutiveCubesNonAdjacent {
        point_index: segment.end - 1,
        axis: step.axis,
        delta: step.delta,
    })?;

    Ok(())
}

/// A step whose cube delta exceeds one unit on `axis`, with the signed delta.
struct NonAdjacentStep {
    axis: usize,
    delta: i64,
}

/// Steps from `from` cube to `to` cube one axis-aligned unit at a time.
/// Positive axis diffs are processed first (axis 0..dim), then negative diffs.
/// Each unit step emits one 1-cube edge via `visit`.
///
/// Returns a [`NonAdjacentStep`] naming the offending axis and delta if any
/// axis diff exceeds one unit in magnitude. The caller attaches the trajectory
/// point the step leaves from before surfacing the error.
fn step_between_cubes<F>(
    from: &[i64],
    to: &[i64],
    dimension: usize,
    base: &mut Orthant,
    dual: &mut Orthant,
    visit: &mut F,
) -> std::result::Result<(), NonAdjacentStep>
where
    F: FnMut(&Cube),
{
    for axis in 0..dimension {
        let delta = to[axis] - from[axis];
        if delta.abs() > 1 {
            return Err(NonAdjacentStep { axis, delta });
        }
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

/// Walks `trajectory.points()`, floors each row to its `i64` cube, and returns
/// the canonical (sorted, deduplicated) cube array paired with a per-row
/// cube-index vector. The per-row indices reference the returned canonical
/// array.
pub(super) fn walk_and_canonicalize(trajectory: &Trajectory) -> (Array2<i64>, Vec<usize>) {
    let points = trajectory.points();
    let dimension = points.ncols();
    let num_rows = points.nrows();

    let mut cube_buffer: Vec<i64> = Vec::with_capacity(dimension);

    // (cube_vector, point_index) pairs.
    let mut pairs: Vec<(Vec<i64>, usize)> = (0..num_rows)
        .map(|point_index| {
            floor_to_cube(points.row(point_index), &mut cube_buffer);
            (cube_buffer.clone(), point_index)
        })
        .collect();

    // Sort by cube vector lexicographically
    pairs.sort_by(|left, right| left.0.cmp(&right.0));

    // Build the deduplicated cube list and a point-index -> cube-index map.
    let mut unique_cubes: Vec<Vec<i64>> = Vec::new();
    let mut point_to_cube: Vec<usize> = vec![0; num_rows];
    for (cube, point_index) in pairs {
        let cube_index = match unique_cubes.last() {
            Some(last) if *last == cube => unique_cubes.len() - 1,
            _ => {
                unique_cubes.push(cube);
                unique_cubes.len() - 1
            },
        };
        point_to_cube[point_index] = cube_index;
    }

    let mut canonical = Array2::<i64>::zeros((unique_cubes.len(), dimension));
    for (cube_index, cube_vec) in unique_cubes.iter().enumerate() {
        for (column, &value) in cube_vec.iter().enumerate() {
            canonical[(cube_index, column)] = value;
        }
    }

    (canonical, point_to_cube)
}
