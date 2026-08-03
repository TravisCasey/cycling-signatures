// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Building a cover from the cubes a trajectory visits.

use std::mem;

use chomp3rs::ExecutionBackend;
use ndarray::Array2;
use rustc_hash::FxHashSet;

use super::{CubicalCover, floor_to_cube, non_adjacent_axis};
use crate::{
    error::{Error, Result},
    trajectory::Trajectory,
};

impl CubicalCover {
    /// Builds the cover of exactly the integer cubes `trajectory` visits.
    ///
    /// Each point is floored component-wise to its integer cube, and the
    /// resulting cube set is covered as by [`from_cubes`](Self::from_cubes).
    /// Consecutive points must land in intersecting cubes (differing by at
    /// most 1 per axis): that is what makes the cover a connected tube
    /// around the sampled path rather than a scatter of cubes with gaps. A
    /// trajectory resampled at a spacing of at most 1 (the cube side)
    /// satisfies this by construction.
    ///
    /// Build the cover from the densest trajectory available. A cover built
    /// from a thinned trajectory is a coarser model of the same curve: the
    /// cubes the curve crosses between kept points are absent, which creates
    /// spurious holes and reports first-homology classes the curve does not
    /// have. A cover is reusable, so one built once from the dense trajectory
    /// serves every embedding taken over it.
    ///
    /// # Errors
    ///
    /// - [`Error::ConsecutiveCubesNonAdjacent`] if consecutive points land in
    ///   cubes differing by more than 1 in some axis; resample at a smaller
    ///   spacing or rescale coordinates.
    /// - [`Error::CubicalCoverZeroDimension`] if the trajectory's points have
    ///   zero columns.
    /// - [`Error::CubeCoordinateOutOfRange`] if a point's cube coordinate falls
    ///   outside `[i32::MIN, i32::MAX - 1]`, naming the offending trajectory
    ///   point by its row.
    pub fn build(trajectory: &Trajectory, backend: &ExecutionBackend) -> Result<Self> {
        Self::from_cubes(visited_cubes(trajectory)?.view(), backend)
    }
}

/// Floors every point of `trajectory` to its integer cube, returning the
/// deduplicated cube array in no particular order. Rejects a cube coordinate
/// outside the range the cubical-homology backend accepts, and consecutive
/// points whose cubes differ by more than 1 in some axis.
fn visited_cubes(trajectory: &Trajectory) -> Result<Array2<i64>> {
    let points = trajectory.points();
    let dimension = points.ncols();

    let mut previous_cube: Vec<i64> = Vec::with_capacity(dimension);
    let mut cube_buffer: Vec<i64> = Vec::with_capacity(dimension);
    let mut visited: FxHashSet<Vec<i64>> = FxHashSet::default();
    for (row, point) in points.outer_iter().enumerate() {
        floor_to_cube(point, &mut cube_buffer);
        for (axis, &coordinate) in cube_buffer.iter().enumerate() {
            if coordinate < i64::from(i32::MIN) || coordinate > i64::from(i32::MAX) - 1 {
                return Err(Error::CubeCoordinateOutOfRange {
                    row,
                    axis,
                    coordinate,
                });
            }
        }
        if row > 0
            && let Some((axis, delta)) = non_adjacent_axis(&previous_cube, &cube_buffer)
        {
            return Err(Error::ConsecutiveCubesNonAdjacent {
                point_index: row - 1,
                axis,
                delta,
            });
        }
        if !visited.contains(cube_buffer.as_slice()) {
            visited.insert(cube_buffer.clone());
        }
        mem::swap(&mut previous_cube, &mut cube_buffer);
    }

    let mut cubes = Array2::<i64>::zeros((visited.len(), dimension));
    for (row, cube) in visited.into_iter().enumerate() {
        for (column, coordinate) in cube.into_iter().enumerate() {
            cubes[(row, column)] = coordinate;
        }
    }
    Ok(cubes)
}

#[cfg(test)]
mod tests {
    use chomp3rs::ExecutionBackend;
    use ndarray::array;

    use super::CubicalCover;
    use crate::{error::Error, trajectory::Trajectory};

    #[test]
    fn build_covers_the_visited_cubes_once_each() {
        // Five points landing in three distinct cubes; two cubes are visited
        // twice, and the visit order is not the canonical order.
        let points = array![[1.5, 0.5], [0.1, 0.1], [0.9, 0.9], [1.5, 0.5], [2.5, 0.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();

        let cover = CubicalCover::build(&trajectory, &ExecutionBackend::default()).unwrap();

        let expected = array![[0_i64, 0], [1, 0], [2, 0]];
        assert_eq!(cover.cubes(), expected.view());
    }

    #[test]
    fn build_rejects_out_of_range_cube_coordinate() {
        // The largest representable coordinate is i32::MAX - 1; a point
        // flooring to i32::MAX is one past it.
        let far_coordinate = f64::from(i32::MAX) + 0.5;
        let points = array![[0.5, 0.5], [far_coordinate, 0.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();

        let outcome = CubicalCover::build(&trajectory, &ExecutionBackend::default());

        assert!(matches!(
            outcome.unwrap_err(),
            Error::CubeCoordinateOutOfRange {
                row: 1,
                axis: 0,
                coordinate,
            } if coordinate == i64::from(i32::MAX)
        ));
    }

    #[test]
    fn build_rejects_non_adjacent_consecutive_points() {
        // Consecutive points three cubes apart: the tube the cover models
        // would have a gap, so the build refuses before the reduction runs.
        let points = array![[0.5, 0.5], [3.5, 0.5]];
        let trajectory = Trajectory::new(points.view()).unwrap();

        let outcome = CubicalCover::build(&trajectory, &ExecutionBackend::default());

        assert!(matches!(
            outcome.unwrap_err(),
            Error::ConsecutiveCubesNonAdjacent {
                point_index: 0,
                axis: 0,
                delta: 3,
            },
        ));
    }
}
