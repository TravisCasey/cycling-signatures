// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cubical cover: integer cubes with cohomology generators computed over
//! `F_2`.

mod build;
mod generators;
#[cfg(feature = "serde")]
mod serialization;

#[cfg(feature = "serde")]
use std::path::Path;
use std::{cmp::Ordering, mem};

use chomp3rs::{Chain, Cube, ExecutionBackend, F2};
use generators::{compute_edge_classes, compute_generators};
use ndarray::{Array2, ArrayView1, ArrayView2};
use rustc_hash::FxHashMap;

use crate::{
    error::{Error, Result},
    f2_vector::F2Vector,
    util::fingerprint::Fingerprint,
};

/// A cubical cover with computed cohomology generators over `F_2`.
///
/// Built either from the cubes a trajectory visits ([`build`](Self::build)) or
/// from an explicit set of integer cube coordinates
/// ([`from_cubes`](Self::from_cubes)). Either way the cube set is
/// canonicalized, so [`cubes`](Self::cubes) returns it lexicographically
/// sorted and deduplicated. Exposes the cohomology generators and the homology
/// class of any chain of 1-cube edges.
#[derive(Debug)]
pub struct CubicalCover {
    cubes: Array2<i64>,
    generators: Vec<Chain<Cube, F2>>,
    edge_classes: FxHashMap<Cube, F2Vector>,
}

impl CubicalCover {
    /// Builds a cover from an explicit cube set.
    ///
    /// `cubes` has shape `(n, dimension)`; rows are deduplicated and
    /// sorted lexicographically. Coordinates must fit in
    /// `[i32::MIN, i32::MAX - 1]`. Cohomology generators are computed via
    /// `chomp3rs` under `backend`.
    ///
    /// # Errors
    ///
    /// - [`Error::CubicalCoverEmptyCubes`] if `cubes` has zero rows.
    /// - [`Error::CubicalCoverZeroDimension`] if `cubes` has zero columns.
    /// - [`Error::CubeCoordinateOutOfRange`] if any cube coordinate is outside
    ///   `[i32::MIN, i32::MAX - 1]`.
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
        let generators = compute_generators(&canonical, backend);
        let edge_classes = compute_edge_classes(&generators);

        Ok(Self {
            cubes: canonical,
            generators,
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
        self.generators.len()
    }

    /// The cohomology generators as `F_2` chains in the cubical complex.
    ///
    /// The iteration order is implementation-defined; two builds of the same
    /// cover might differ in the basis returned, though they span the same
    /// cohomology space.
    #[must_use]
    pub fn generators(&self) -> &[Chain<Cube, F2>] {
        &self.generators
    }

    /// Returns the `F_2` homology class of the chain formed by `edges`.
    ///
    /// Edges not present in any generator chain contribute nothing. The
    /// returned vector has length [`num_generators`](Self::num_generators).
    #[must_use]
    pub fn chain_class<'a, Edges>(&self, edges: Edges) -> F2Vector
    where
        Edges: IntoIterator<Item = &'a Cube>,
    {
        let mut accumulator = F2Vector::zeros(self.generators.len());
        for edge in edges {
            if let Some(class) = self.edge_class(edge) {
                accumulator ^= class;
            }
        }
        accumulator
    }

    /// The `F_2` homology class of the single 1-cube `edge`, or `None` when
    /// the edge lies in no generator chain and so contributes nothing.
    #[must_use]
    pub(crate) fn edge_class(&self, edge: &Cube) -> Option<&F2Vector> {
        self.edge_classes.get(edge)
    }

    /// A stable 64-bit fingerprint of this cover's content.
    ///
    /// Derived from the cube set only. The cohomology generators are
    /// deliberately excluded: the same cubes may yield a different generator
    /// basis across builds, so hashing the cubes keeps the fingerprint a
    /// function of the canonical data.
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

    /// Writes this cover to `path` in the crate's binary format.
    ///
    /// # Errors
    ///
    /// - [`Error::Io`] on file or serialization failure.
    #[cfg(feature = "serde")]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        crate::serialization::save_to_path(path, self)
    }

    /// Reads a cover written by [`save`](Self::save).
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
    use chomp3rs::{Cube, ExecutionBackend, F2, Orthant, Ring};
    use ndarray::{Array2, array};

    use super::CubicalCover;
    use crate::{error::Error, f2_vector::F2Vector};

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
        // Walk the boundary of a 4x4 grid, leaving a 2x2 hole in the middle.
        // Expected: exactly one 1-dimensional generator.
        let cubes = array![
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
        ];
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();

        assert_eq!(cover.num_generators(), 1);
    }

    #[test]
    fn chain_class_matches_known_loop() {
        // 4x4 with hole: one generator. Each of its edges, taken individually,
        // is in that generator (and only that generator), so chain_class on
        // a single such edge yields the unit F2Vector at index 0.
        let cubes = array![
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
            [0, 1]
        ];
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();
        assert_eq!(cover.num_generators(), 1);

        let generator = &cover.generators()[0];
        let first_edge: &Cube = generator
            .into_iter()
            .find(|(_, coefficient)| **coefficient != F2::zero())
            .map(|(cube, _)| cube)
            .expect("generator should have at least one non-zero entry");

        let class = cover.chain_class(std::iter::once(first_edge));
        assert_eq!(class, F2Vector::from_nonzero(1, [0]));
    }

    #[test]
    fn chain_class_off_generator_edges_return_zero() {
        // 4x4 with hole, one generator. Construct an edge with coordinates
        // far outside the cover so it cannot be in any generator chain.
        let cubes = array![
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
            [0, 1]
        ];
        let cover = CubicalCover::from_cubes(cubes.view(), &ExecutionBackend::default()).unwrap();

        let base: Orthant = [100_i32, 100_i32].into();
        let dual: Orthant = [99_i32, 100_i32].into();
        let off_edge = Cube::new(base, dual);

        let class = cover.chain_class(std::iter::once(&off_edge));
        assert_eq!(class, F2Vector::zeros(cover.num_generators()));
    }
}
