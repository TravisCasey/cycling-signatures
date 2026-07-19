// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cubical cover: integer cubes with chomp3rs-computed cohomology generators
//! and an edge-to-class lookup table.

use std::cmp::Ordering;
#[cfg(feature = "serde")]
use std::path::Path;

use chomp3rs::{
    Chain, Complex, CoreductionMatching, Cube, CubicalComplex, ExecutionBackend, F2, MorseMatching,
    Orthant, OrthantTrie, Ring, TopCubeGrader, TopCubicalMatching,
};
use ndarray::{Array2, ArrayView1, ArrayView2};
use rustc_hash::FxHashMap;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize, de::Error as DeserializeError};

use crate::{
    error::{Error, Result},
    f2_vector::F2Vector,
    util::fingerprint::Fingerprint,
};

/// A cubical cover with computed cohomology generators over `F_2`.
///
/// Built from an explicit set of integer cube coordinates. Stores cubes in
/// lexicographically-sorted, deduplicated form and exposes the cohomology
/// generators plus a chain-class lookup keyed by 1-cube edges.
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

        for row in cubes.outer_iter() {
            for (axis, &coordinate) in row.iter().enumerate() {
                if coordinate < i64::from(i32::MIN) || coordinate > i64::from(i32::MAX) - 1 {
                    return Err(Error::CubeCoordinateOutOfRange { axis, coordinate });
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
            if let Some(class) = self.edge_classes.get(edge) {
                accumulator ^= class;
            }
        }
        accumulator
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

    /// The row index in [`cubes`](Self::cubes) of the cube containing `point`,
    /// or `None` if that cube is not in the cover. `point` is floored
    /// component-wise to its integer cube, then located in the canonical
    /// (sorted) cube list.
    #[must_use]
    pub(crate) fn cube_index(&self, point: ArrayView1<'_, f64>) -> Option<usize> {
        let mut buffer: Vec<i64> = Vec::with_capacity(self.cubes.ncols());
        floor_to_cube(point, &mut buffer);
        find_cube(self.cubes.view(), &buffer)
    }

    /// Writes this cover to `path` in the crate's binary format.
    ///
    /// # Errors
    ///
    /// - [`crate::error::Error::Storage`] on file or serialization failure.
    #[cfg(feature = "serde")]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        crate::persistence::save_to_path(path, self)
    }

    /// Reads a cover written by [`save`](Self::save).
    ///
    /// # Errors
    ///
    /// - [`crate::error::Error::FormatVersionMismatch`] if the file's format
    ///   version differs.
    /// - [`crate::error::Error::Storage`] on file or deserialization failure.
    #[cfg(feature = "serde")]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        crate::persistence::load_from_path(path)
    }
}

/// Floors each coordinate of `point` to its integer cube index, writing the
/// result into `out` (cleared first). A point's cube is the component-wise
/// floor of its coordinates.
pub(crate) fn floor_to_cube(point: ArrayView1<'_, f64>, out: &mut Vec<i64>) {
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

/// Computes cohomology generators for the cubical complex defined by
/// `canonical_cubes` (sorted, deduplicated, in-range).
fn compute_generators(
    canonical_cubes: &Array2<i64>,
    backend: &ExecutionBackend,
) -> Vec<Chain<Cube, F2>> {
    let dimension = canonical_cubes.ncols();

    // `from_cubes` rejects zero-row input before calling this helper, so
    // every column has at least one entry.
    let minimum_orthant: Orthant = (0..dimension)
        .map(|axis| {
            canonical_cubes
                .column(axis)
                .iter()
                .copied()
                .min()
                .expect("canonical_cubes has at least one row") as i32
        })
        .collect();

    let maximum_orthant: Orthant = (0..dimension)
        .map(|axis| {
            canonical_cubes
                .column(axis)
                .iter()
                .copied()
                .max()
                .expect("canonical_cubes has at least one row") as i32
                + 1
        })
        .collect();

    let orthants: Vec<Orthant> = canonical_cubes
        .outer_iter()
        .map(|row| row.iter().map(|&value| value as i32).collect())
        .collect();

    let grading_function =
        TopCubeGrader::new(OrthantTrie::uniform(orthants.clone(), 0, 1), Some(0));

    let complex = CubicalComplex::<F2, TopCubeGrader<OrthantTrie>>::new(
        minimum_orthant,
        maximum_orthant,
        grading_function,
    );

    let top_matching = TopCubicalMatching::<F2, OrthantTrie>::builder()
        .max_grade(0)
        .max_dimension(2)
        .subgrid_shape(vec![1_i32; dimension])
        .filter_orthants(orthants)
        .backend(backend.clone())
        .build(complex);

    let (lower_matchings, morse_complex) = top_matching.full_reduce(CoreductionMatching::new);

    let generator_cells: Vec<u32> = morse_complex
        .iter()
        .filter(|cell| morse_complex.cell_dimension(cell) == 1)
        .collect();

    let mut lower_reps: Vec<Chain<u32, F2>> = generator_cells
        .iter()
        .map(|&cell| Chain::from(cell))
        .collect();

    for matching in lower_matchings.iter().rev() {
        lower_reps = lower_reps
            .into_iter()
            .map(|chain| matching.colift_capped(chain, 0))
            .collect();
    }

    let mut generators: Vec<Chain<Cube, F2>> = lower_reps
        .into_iter()
        .map(|chain| top_matching.colift_capped(chain, 0))
        .collect();

    // Lex-sort generators
    generators.sort_by(compare_chains);

    generators
}

/// A total ordering on [`Chain<Cube, F2>`](Chain) used for deterministic
/// sorting.
fn compare_chains(left: &Chain<Cube, F2>, right: &Chain<Cube, F2>) -> std::cmp::Ordering {
    let left_entries: Vec<&Cube> = left.into_iter().map(|(cube, _)| cube).collect();
    let right_entries: Vec<&Cube> = right.into_iter().map(|(cube, _)| cube).collect();
    left_entries.cmp(&right_entries)
}

/// Builds the edge-to-class lookup table from the generator chains. Each
/// generator contributes a `1` at its own index for every edge it contains.
fn compute_edge_classes(generators: &[Chain<Cube, F2>]) -> FxHashMap<Cube, F2Vector> {
    let mut edge_classes: FxHashMap<Cube, F2Vector> = FxHashMap::default();
    let num_generators = generators.len();
    for (generator_index, generator) in generators.iter().enumerate() {
        for (cube, coefficient) in generator {
            if *coefficient == F2::zero() {
                continue;
            }
            let entry = edge_classes
                .entry(cube.clone())
                .or_insert_with(|| F2Vector::zeros(num_generators));
            entry.set(generator_index, F2::one());
        }
    }
    edge_classes
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct CoverData {
    #[serde(with = "crate::persistence::npy_field")]
    cubes: Array2<i64>,
    generators: Vec<Vec<Cube>>,
}

#[cfg(feature = "serde")]
impl Serialize for CubicalCover {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let generators: Vec<Vec<Cube>> = self
            .generators
            .iter()
            .map(|generator| {
                let mut cubes: Vec<Cube> = generator
                    .iter()
                    .filter(|(_, coefficient)| **coefficient != F2::zero())
                    .map(|(cube, _)| cube.clone())
                    .collect();
                cubes.sort();
                cubes
            })
            .collect();
        CoverData {
            cubes: self.cubes.clone(),
            generators,
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for CubicalCover {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = CoverData::deserialize(deserializer)?;
        if data.cubes.nrows() == 0 {
            return Err(D::Error::custom("cubical cover requires at least one cube"));
        }
        let generators: Vec<Chain<Cube, F2>> = data
            .generators
            .into_iter()
            .map(|cubes| cubes.into_iter().map(|cube| (cube, F2::one())).collect())
            .collect();
        let edge_classes = compute_edge_classes(&generators);
        Ok(Self {
            cubes: data.cubes,
            generators,
            edge_classes,
        })
    }
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
        let input = array![[0_i64, 0], [1, i64::from(i32::MAX)]];
        let outcome = CubicalCover::from_cubes(input.view(), &ExecutionBackend::default());

        assert!(matches!(
            outcome.unwrap_err(),
            Error::CubeCoordinateOutOfRange {
                axis: 1,
                coordinate,
            } if coordinate == i64::from(i32::MAX),
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
