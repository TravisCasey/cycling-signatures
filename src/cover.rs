// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cubical cover: integer cubes with the `F_2` homology class of every edge a
//! cycle walk can traverse.

mod build;
mod generators;
#[cfg(feature = "serde")]
mod serialization;
pub(crate) mod staircase;

#[cfg(feature = "serde")]
use std::path::Path;
use std::{cmp::Ordering, mem};

use chomp3rs::{Cube, ExecutionBackend};
use generators::compute_edge_map;
use ndarray::{Array2, ArrayView1, ArrayView2};
use rustc_hash::FxHashMap;

use crate::{
    error::{Error, Result},
    f2_vector::F2Vector,
    util::fingerprint::Fingerprint,
};

/// A cubical cover with the `F_2` homology class of every edge a cycle walk
/// can traverse.
///
/// Built either from the cubes a trajectory visits ([`build`](Self::build)) or
/// from an explicit set of integer cube coordinates
/// ([`from_cubes`](Self::from_cubes)). Either way the cube set is
/// canonicalized, so [`cubes`](Self::cubes) returns it lexicographically
/// sorted and deduplicated. Exposes the cohomology generator count and the
/// homology class of any chain of recognized 1-cube edges.
#[derive(Debug)]
pub struct CubicalCover {
    cubes: Array2<i64>,
    num_generators: usize,
    edge_classes: FxHashMap<Cube, F2Vector>,
}

impl CubicalCover {
    /// Builds a cover from an explicit cube set.
    ///
    /// `cubes` has shape `(n, dimension)`; rows are deduplicated and
    /// sorted lexicographically. Coordinates must fit in
    /// `[i32::MIN, i32::MAX - 1]`. The homology of the cubical complex on the
    /// cubes is computed via `chomp3rs` under `backend`, recording the class
    /// of every edge a cycle walk against this cover can traverse.
    ///
    /// # Errors
    ///
    /// - [`Error::CubicalCoverEmptyCubes`] if `cubes` has zero rows.
    /// - [`Error::CubicalCoverZeroDimension`] if `cubes` has zero columns.
    /// - [`Error::CubeCoordinateOutOfRange`] if any cube coordinate is outside
    ///   `[i32::MIN, i32::MAX - 1]`.
    ///
    /// # Panics
    ///
    /// Panics if the homology reduction leaves a nonzero boundary in degree
    /// at most 2, which would violate the invariant the recorded classes
    /// rest on.
    pub fn from_cubes(cubes: ArrayView2<'_, i64>, backend: &ExecutionBackend) -> Result<Self> {
        if cubes.nrows() == 0 {
            return Err(Error::CubicalCoverEmptyCubes);
        }
        if cubes.ncols() == 0 {
            return Err(Error::CubicalCoverZeroDimension);
        }

        for (row, cube) in cubes.outer_iter().enumerate() {
            for (axis, &coordinate) in cube.iter().enumerate() {
                if coordinate < i64::from(i32::MIN) || coordinate > i64::from(i32::MAX) - 1 {
                    return Err(Error::CubeCoordinateOutOfRange {
                        row,
                        axis,
                        coordinate,
                    });
                }
            }
        }

        let canonical = canonicalize_cubes(cubes);
        let (num_generators, edge_classes) = compute_edge_map(&canonical, backend);

        Ok(Self {
            cubes: canonical,
            num_generators,
            edge_classes,
        })
    }

    /// The canonical (sorted, deduplicated) cube list.
    #[must_use]
    pub fn cubes(&self) -> ArrayView2<'_, i64> {
        self.cubes.view()
    }

    /// The spatial dimension of each cube.
    ///
    /// Must equal [`Trajectory::dimension`](crate::Trajectory::dimension) when
    /// paired in an [`EmbeddedTrajectory`](crate::EmbeddedTrajectory).
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.cubes.ncols()
    }

    /// The number of cohomology generators.
    #[must_use]
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Returns the `F_2` homology class of the chain formed by `edges`, or
    /// `None` when some edge is not recognized.
    ///
    /// Recognized edges are exactly those a cycle walk against this cover can
    /// traverse. The answer is all-or-nothing: one unrecognized edge yields
    /// `None` rather than a class computed from the remainder. A returned
    /// vector has length [`num_generators`](Self::num_generators).
    #[must_use]
    pub fn chain_class<'a, E>(&self, edges: E) -> Option<F2Vector>
    where
        E: IntoIterator<Item = &'a Cube>,
    {
        let mut accumulator = F2Vector::zeros(self.num_generators);
        for edge in edges {
            accumulator ^= self.edge_class(edge)?;
        }
        Some(accumulator)
    }

    /// The `F_2` homology class of the single 1-cube `edge`, or `None` when
    /// the edge is not recognized.
    ///
    /// `None` never means the zero class: zero classes are stored explicitly
    /// and answered as `Some`. An unrecognized edge is one no cycle walk
    /// against this cover can traverse.
    #[must_use]
    pub(crate) fn edge_class(&self, edge: &Cube) -> Option<&F2Vector> {
        self.edge_classes.get(edge)
    }

    /// A stable 64-bit fingerprint of this cover's content.
    ///
    /// Derived from the cube set only. The edge classes are excluded: the same
    /// cubes may yield a different generator basis across builds, so hashing
    /// only the cubes keeps the fingerprint a function of the canonical data.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = Fingerprint::new();
        hasher.write(&(self.cubes.nrows() as u64).to_le_bytes());
        hasher.write(&(self.cubes.ncols() as u64).to_le_bytes());
        for &value in &self.cubes {
            hasher.write(&value.to_le_bytes());
        }
        hasher.finish()
    }

    /// The row index in [`cubes`](Self::cubes) of the cube containing each row
    /// of `points`, in order. Every point is floored component-wise to its
    /// integer cube, which is then located in the canonical (sorted) cube
    /// list.
    ///
    /// # Errors
    ///
    /// - [`Error::EmbeddedCubeNotInCover`] naming the first point whose cube
    ///   the cover does not contain.
    pub(crate) fn cube_indices(&self, points: ArrayView2<'_, f64>) -> Result<Vec<usize>> {
        let dimension = self.cubes.ncols();
        let mut indices: Vec<usize> = Vec::with_capacity(points.nrows());
        let mut cube: Vec<i64> = Vec::with_capacity(dimension);
        let mut previous_cube: Vec<i64> = Vec::with_capacity(dimension);
        let mut previous_index = 0_usize;
        for (point_index, point) in points.outer_iter().enumerate() {
            floor_to_cube(point, &mut cube);
            // Neighboring points frequently share a cube, and repeating the
            // last answer is cheaper than searching the cube list again.
            if point_index > 0 && cube == previous_cube {
                indices.push(previous_index);
                continue;
            }
            let Some(cube_index) = find_cube(self.cubes.view(), &cube) else {
                return Err(Error::EmbeddedCubeNotInCover { point_index });
            };
            indices.push(cube_index);
            previous_index = cube_index;
            mem::swap(&mut previous_cube, &mut cube);
        }
        Ok(indices)
    }

    /// Writes this cover to `path` in the crate's binary format, including its
    /// cube set and its edge classes, which fix its exact generator basis.
    ///
    /// A cover loaded back from the result (via [`load`](Self::load)) shares
    /// this generator basis, unlike one independently rebuilt from the same
    /// cubes.
    ///
    /// # Errors
    ///
    /// - [`Error::Io`] on file or serialization failure.
    #[cfg(feature = "serde")]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        crate::serialization::save_to_path(path, self)
    }

    /// Reads a cover written by [`save`](Self::save), reconstructing the exact
    /// cube set and edge classes it was saved with, in the generator basis they
    /// were saved in.
    ///
    /// # Errors
    ///
    /// - [`Error::FormatVersionMismatch`] if the file's format version differs.
    /// - [`Error::Io`] if the file could not be opened.
    /// - [`Error::Deserialize`] if the file contents could not be read and
    ///   decoded.
    #[cfg(feature = "serde")]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        crate::serialization::load_from_path(path)
    }
}

/// The first axis on which the cubes `from` and `to` sit more than one
/// position apart, paired with the signed cube difference along that axis
/// measured from `from` to `to`. `None` when the cubes are adjacent.
///
/// Two cubes are adjacent when they intersect, which for unit cubes means
/// their coordinates differ by at most 1 on every axis.
#[must_use]
pub(crate) fn non_adjacent_axis<'a>(
    from: impl IntoIterator<Item = &'a i64>,
    to: impl IntoIterator<Item = &'a i64>,
) -> Option<(usize, i64)> {
    for (axis, (&start, &end)) in from.into_iter().zip(to).enumerate() {
        let delta = end - start;
        if delta.abs() > 1 {
            return Some((axis, delta));
        }
    }
    None
}

/// Floors each coordinate of `point` to its integer cube index, writing the
/// result into `out` (cleared first). A point's cube is the component-wise
/// floor of its coordinates.
fn floor_to_cube(point: ArrayView1<'_, f64>, out: &mut Vec<i64>) {
    out.clear();
    out.extend(point.iter().map(|&value| value.floor() as i64));
}

/// Returns the row index of the cube equal to `target`, or `None` if no cube
/// matches. `cubes` must be lexicographically sorted (the canonical order
/// [`CubicalCover`] enforces); the lookup is a binary search over that order.
#[must_use]
fn find_cube(cubes: ArrayView2<'_, i64>, target: &[i64]) -> Option<usize> {
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

/// Lexicographically sorts and deduplicates `cubes`, returning a new owned
/// [`Array2<i64>`](Array2).
#[must_use]
fn canonicalize_cubes(cubes: ArrayView2<'_, i64>) -> Array2<i64> {
    let mut rows: Vec<Vec<i64>> = cubes.outer_iter().map(|row| row.to_vec()).collect();
    rows.sort();
    rows.dedup();

    let dimension = cubes.ncols();
    let mut canonical = Array2::<i64>::zeros((rows.len(), dimension));
    for (row_index, row) in rows.iter().enumerate() {
        for (column, &value) in row.iter().enumerate() {
            canonical[(row_index, column)] = value;
        }
    }

    canonical
}

#[cfg(test)]
mod tests {
    use chomp3rs::{Cube, ExecutionBackend, Orthant};
    use ndarray::{Array2, array};
    use rustc_hash::FxHashSet;

    use super::{CubicalCover, generators, staircase};
    #[cfg(feature = "serde")]
    use crate::serialization::{load_from_reader, save_to_writer};
    use crate::{
        error::Error,
        f2_vector::F2Vector,
        trajectory::Trajectory,
        util::fixtures::{densify_path, embed_euclidean, ring_waypoints},
    };

    /// The twelve cubes on the boundary of a `4x4` grid, leaving a `2x2` hole
    /// in the middle. A cover of these cubes and no others has exactly one
    /// 1-dimensional generator.
    fn ring_cubes() -> Array2<i64> {
        array![
            [0_i64, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [2, 3],
            [1, 3],
            [0, 3],
            [0, 2],
            [0, 1],
        ]
    }

    #[test]
    fn from_cubes_sorts_and_deduplicates() {
        // Arbitrary order with a duplicate; expect lex-sorted, unique output.
        let input = array![
            [2_i64, 0],
            [0, 1],
            [0, 0],
            [0, 0], // duplicate
            [1, 0],
        ];
        let cover = CubicalCover::from_cubes(input.view(), &ExecutionBackend::default()).unwrap();

        let expected = array![[0_i64, 0], [0, 1], [1, 0], [2, 0]];
        assert_eq!(cover.cubes(), expected.view());
        assert_eq!(cover.dimension(), 2);
    }

    #[test]
    fn from_cubes_rejects_empty_input() {
        let input = Array2::<i64>::zeros((0, 2));
        let outcome = CubicalCover::from_cubes(input.view(), &ExecutionBackend::default());

        assert!(matches!(
            outcome.unwrap_err(),
            Error::CubicalCoverEmptyCubes
        ));
    }

    #[test]
    fn from_cubes_rejects_zero_dimension() {
        let input = Array2::<i64>::zeros((3, 0));
        let outcome = CubicalCover::from_cubes(input.view(), &ExecutionBackend::default());

        assert!(matches!(
            outcome.unwrap_err(),
            Error::CubicalCoverZeroDimension
        ));
    }

    #[test]
    fn from_cubes_rejects_out_of_range_coordinate() {
        // The largest valid coordinate is i32::MAX - 1; i32::MAX is one past it,
        // reserved as headroom for the half-open bounding orthant.
        // The offending cube is the second row as supplied.
        let input = array![[0_i64, 0], [1, i64::from(i32::MAX)]];
        let outcome = CubicalCover::from_cubes(input.view(), &ExecutionBackend::default());

        assert!(matches!(
            outcome.unwrap_err(),
            Error::CubeCoordinateOutOfRange {
                row: 1,
                axis: 1,
                coordinate,
            } if coordinate == i64::from(i32::MAX)
        ));
    }

    #[test]
    fn generators_for_loop_with_hole() {
        let cubes = ring_cubes();
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();

        assert_eq!(cover.num_generators(), 1);
    }

    /// Walks the rows of `cubes` in row order, closing back to the first row,
    /// and returns the traversed edges.
    fn closed_walk_edges(cubes: &Array2<i64>) -> Vec<Cube> {
        let rows: Vec<Vec<i64>> = cubes.outer_iter().map(|row| row.to_vec()).collect();
        let dimension = cubes.ncols();
        let mut position: Orthant = rows[0].iter().map(|&value| value as i32).collect();
        let mut edges: Vec<Cube> = Vec::new();
        for index in 0..rows.len() {
            let from = &rows[index];
            let to = &rows[(index + 1) % rows.len()];
            staircase::step_between_cubes(from, to, dimension, &mut position, &mut |edge| {
                edges.push(edge.clone());
            })
            .expect("consecutive ring cubes are adjacent");
        }
        edges
    }

    #[test]
    fn chain_class_matches_known_loop() {
        // The ring cubes in row order trace one loop around the hole, so the
        // walked edge chain carries the single generator: with one generator
        // the nonzero class is the unit F2Vector regardless of basis.
        let cubes = ring_cubes();
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();
        assert_eq!(cover.num_generators(), 1);

        let edges = closed_walk_edges(&cubes);
        let class = cover.chain_class(edges.iter());
        assert_eq!(class, Some(F2Vector::from_nonzero(1, [0])));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn edge_map_matches_after_save_load_roundtrip() {
        // A cover loaded from a save carries the exact edge classes it was
        // saved with, which fixes the generator basis.
        let cubes = ring_cubes();
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();
        assert_eq!(cover.num_generators(), 1);

        let mut buffer = Vec::new();
        save_to_writer(&mut buffer, &cover).unwrap();
        let reloaded: CubicalCover = load_from_reader(&buffer[..]).unwrap();

        assert_eq!(cover.num_generators(), reloaded.num_generators());
        assert_eq!(cover.edge_classes, reloaded.edge_classes);
    }

    #[test]
    fn universe_contains_every_walked_edge() {
        // The enumeration promises to hold every edge any cycle walk can emit;
        // check it against the walker itself on real ring geometry, where the
        // walk's closing step crosses cubes the forward path also visits.
        let points = densify_path(&ring_waypoints(), 0.5);
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded = embed_euclidean(trajectory).unwrap();
        let universe: FxHashSet<Cube> =
            generators::enumerate_edge_universe(&embedded.cover().cubes)
                .into_iter()
                .collect();

        let walked = embedded.walk_cycle(..).unwrap();
        assert!(!walked.is_empty());
        for edge in &walked {
            assert!(
                universe.contains(edge),
                "walked edge {edge:?} not in the universe"
            );
        }
    }

    #[test]
    fn chain_class_distinguishes_zero_from_unrecognized() {
        // The four edges of the unit square at cube (0, 0) are all recognized
        // (each lies on a staircase between adjacent cover cubes), and their
        // chain bounds that cube's 2-cell, so its class is zero in every basis:
        // recognized edges answer Some even when the class is zero. An edge far
        // outside the cover is unrecognized and any chain it appears in
        // returns None.
        let cubes = ring_cubes();
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();

        let square = [
            Cube::from_extent(Orthant::from([0_i32, 0]), 0b01),
            Cube::from_extent(Orthant::from([1_i32, 0]), 0b10),
            Cube::from_extent(Orthant::from([0_i32, 1]), 0b01),
            Cube::from_extent(Orthant::from([0_i32, 0]), 0b10),
        ];
        assert_eq!(
            cover.chain_class(square.iter()),
            Some(F2Vector::zeros(cover.num_generators()))
        );

        let off_edge = Cube::from_extent(Orthant::from([100_i32, 100]), 0b01);
        assert_eq!(cover.chain_class([&off_edge]), None);
        assert_eq!(cover.chain_class([&square[0], &off_edge]), None);
    }
}
