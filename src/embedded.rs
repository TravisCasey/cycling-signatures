// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Embedded trajectory: a `Trajectory` paired with a `CubicalCover` and the
//! mapping from trajectory point to cube index.

use std::cmp::Ordering;

use chomp3rs::ExecutionBackend;
use ndarray::{Array2, ArrayView2};

use crate::{
    cover::CubicalCover,
    error::{Error, Result},
    metric::Metric,
    trajectory::Trajectory,
};

/// Pairs a [`Trajectory<M>`](Trajectory) with a [`CubicalCover`] and the
/// per-point cube-index map.
///
/// The bridge is resample-agnostic: `point_to_cube(i)` operates in
/// `trajectory.points()`-index space. Callers holding original-input
/// indices translate first via
/// [`Trajectory::original_indices`](Trajectory::original_indices).
#[derive(Debug)]
pub struct EmbeddedTrajectory<M: Metric> {
    trajectory: Trajectory<M>,
    cover: CubicalCover,
    point_to_cube: Vec<usize>,
}

impl<M: Metric> EmbeddedTrajectory<M> {
    /// Pairs `trajectory` with a cover of exactly the integer cubes it visits,
    /// and records each point's cube index in `trajectory.points()` order.
    ///
    /// # Errors
    ///
    /// Returns any error from [`CubicalCover::from_cubes`].
    pub fn new(trajectory: Trajectory<M>, backend: &ExecutionBackend) -> Result<Self> {
        let (canonical_cubes, point_to_cube) = walk_and_canonicalize(&trajectory);
        let cover = CubicalCover::from_cubes(canonical_cubes.view(), backend)?;
        Ok(Self {
            trajectory,
            cover,
            point_to_cube,
        })
    }

    /// The wrapped trajectory.
    #[must_use]
    pub fn trajectory(&self) -> &Trajectory<M> {
        &self.trajectory
    }

    /// The wrapped cover.
    #[must_use]
    pub fn cover(&self) -> &CubicalCover {
        &self.cover
    }

    /// The cube index in [`cover().cubes()`](CubicalCover::cubes) of trajectory
    /// point at `point_index`.
    ///
    /// `point_index` is in `trajectory.points()`-index space (the same units as
    /// `trajectory.len()`). For resampled trajectories this includes
    /// bisection-inserted points; callers holding original input indices
    /// translate first via
    /// [`Trajectory::original_indices`](Trajectory::original_indices).
    ///
    /// # Panics
    ///
    /// Panics if `point_index >= trajectory.len()`.
    #[must_use]
    pub fn point_to_cube(&self, point_index: usize) -> usize {
        self.point_to_cube[point_index]
    }

    /// Attaches a pre-built cover to a trajectory.
    ///
    /// Validates that the cover's dimension matches the trajectory's and that
    /// every point's cube is present in the cover.
    ///
    /// # Errors
    ///
    /// - [`Error::EmbeddedDimensionMismatch`] if dimensions disagree.
    /// - [`Error::EmbeddedCubeNotInCover`] if any point maps to a cube absent
    ///   from the cover.
    pub fn from_parts(trajectory: Trajectory<M>, cover: CubicalCover) -> Result<Self> {
        if trajectory.dimension() != cover.dimension() {
            return Err(Error::EmbeddedDimensionMismatch {
                trajectory: trajectory.dimension(),
                cover: cover.dimension(),
            });
        }

        let points = trajectory.points();
        let cubes = cover.cubes();
        let mut point_to_cube: Vec<usize> = Vec::with_capacity(points.nrows());
        let dimension = points.ncols();
        let mut buffer: Vec<i64> = vec![0; dimension];

        for point_index in 0..points.nrows() {
            for (axis, &value) in points.row(point_index).iter().enumerate() {
                buffer[axis] = value.floor() as i64;
            }
            match binary_search_cube(cubes, &buffer) {
                Some(cube_index) => point_to_cube.push(cube_index),
                None => {
                    return Err(Error::EmbeddedCubeNotInCover { point_index });
                },
            }
        }

        Ok(Self {
            trajectory,
            cover,
            point_to_cube,
        })
    }
}

/// Walks `trajectory.points()`, floors each row to its `i64` cube, and returns
/// the canonical (sorted, deduplicated) cube array paired with a per-row
/// cube-index vector. The per-row indices reference the returned canonical
/// array.
fn walk_and_canonicalize<M: Metric>(trajectory: &Trajectory<M>) -> (Array2<i64>, Vec<usize>) {
    let points = trajectory.points();
    let dimension = points.ncols();
    let num_rows = points.nrows();

    // (cube_vector, original_row_index) pairs.
    let mut pairs: Vec<(Vec<i64>, usize)> = (0..num_rows)
        .map(|row| {
            let cube: Vec<i64> = points
                .row(row)
                .iter()
                .map(|&value| value.floor() as i64)
                .collect();
            (cube, row)
        })
        .collect();

    // Sort by cube vector lexicographically
    pairs.sort_by(|left, right| left.0.cmp(&right.0));

    // Build the deduplicated cube list and a (row -> cube_index) map.
    let mut unique_cubes: Vec<Vec<i64>> = Vec::new();
    let mut point_to_cube: Vec<usize> = vec![0; num_rows];
    for (cube, row) in pairs {
        let index = match unique_cubes.last() {
            Some(last) if *last == cube => unique_cubes.len() - 1,
            _ => {
                unique_cubes.push(cube);
                unique_cubes.len() - 1
            },
        };
        point_to_cube[row] = index;
    }

    let mut canonical = Array2::<i64>::zeros((unique_cubes.len(), dimension));
    for (row, cube_vec) in unique_cubes.iter().enumerate() {
        for (column, &value) in cube_vec.iter().enumerate() {
            canonical[(row, column)] = value;
        }
    }

    (canonical, point_to_cube)
}

/// Binary-searches `cubes` for a row equal to `target`, returning its row
/// index. `cubes` is assumed to be lexicographically sorted (the canonical
/// order [`CubicalCover`] enforces).
#[allow(clippy::needless_pass_by_value)]
fn binary_search_cube(cubes: ArrayView2<'_, i64>, target: &[i64]) -> Option<usize> {
    let mut low = 0_usize;
    let mut high = cubes.nrows();
    while low < high {
        let mid = low + (high - low) / 2;
        let row = cubes.row(mid);

        match row.iter().copied().cmp(target.iter().copied()) {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid,
            Ordering::Equal => return Some(mid),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use chomp3rs::ExecutionBackend;
    use ndarray::array;

    use super::EmbeddedTrajectory;
    use crate::{cover::CubicalCover, error::Error, metric::Euclidean, trajectory::Trajectory};

    #[test]
    fn new_walks_trajectory_with_deduplication() {
        // Four points landing in three distinct cubes; rows 0 and 1
        // share a cube.
        //
        // (0.1, 0.1), (0.9, 0.9) -> cube (0, 0)
        // (1.5, 0.5)             -> cube (1, 0)
        // (2.5, 0.5)             -> cube (2, 0)
        let points = array![[0.1, 0.1], [0.9, 0.9], [1.5, 0.5], [2.5, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded = EmbeddedTrajectory::new(trajectory, &ExecutionBackend::default()).unwrap();

        let expected_cubes = array![[0_i64, 0], [1, 0], [2, 0]];
        assert_eq!(embedded.cover().cubes(), expected_cubes.view());
        assert_eq!(embedded.point_to_cube(0), 0);
        assert_eq!(embedded.point_to_cube(1), 0);
        assert_eq!(embedded.point_to_cube(2), 1);
        assert_eq!(embedded.point_to_cube(3), 2);
    }

    #[test]
    fn from_parts_matches_new() {
        // Build a trajectory, run new(), then re-pair via from_parts and
        // assert the same point_to_cube mapping.
        let points = array![[0.3, 0.7], [1.4, 0.2], [2.9, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let embedded_via_new =
            EmbeddedTrajectory::new(trajectory.clone(), &ExecutionBackend::default()).unwrap();

        // Rebuild the cover from the same cube set, pair via from_parts.
        let cover_cubes = embedded_via_new.cover().cubes().to_owned();
        let fresh_cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();
        let embedded_via_from_parts =
            EmbeddedTrajectory::from_parts(trajectory, fresh_cover).unwrap();

        for index in 0..points.nrows() {
            assert_eq!(
                embedded_via_new.point_to_cube(index),
                embedded_via_from_parts.point_to_cube(index),
            );
        }
    }

    #[test]
    fn from_parts_rejects_missing_cube() {
        // Build a trajectory that visits cubes (0,0), (1,0), (2,0). Build
        // a cover containing only (0,0) and (1,0). from_parts must fail
        // at point_index 2 with EmbeddedCubeNotInCover.
        let points = array![[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let cover_cubes = array![[0_i64, 0], [1, 0]];
        let cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();

        let outcome = EmbeddedTrajectory::from_parts(trajectory, cover);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::EmbeddedCubeNotInCover { point_index: 2 },
        ));
    }

    #[test]
    fn from_parts_rejects_dimension_mismatch() {
        let points = array![[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]];
        let trajectory = Trajectory::new(points.view(), Euclidean).unwrap();
        let cover_cubes = array![[0_i64, 0], [1, 0]];
        let cover =
            CubicalCover::from_cubes(cover_cubes.view(), &ExecutionBackend::default()).unwrap();

        let outcome = EmbeddedTrajectory::from_parts(trajectory, cover);

        assert!(matches!(
            outcome.unwrap_err(),
            Error::EmbeddedDimensionMismatch {
                trajectory: 3,
                cover: 2,
            },
        ));
    }
}
