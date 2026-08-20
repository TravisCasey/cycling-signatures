// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cohomology generators of a cubical complex and the edge-to-class table
//! derived from them.

use chomp3rs::{
    Chain, Complex, CoreductionMatching, Cube, CubicalComplex, ExecutionBackend, Field,
    MorseMatching, Orthant, OrthantTrie, TopCubeGrader, TopCubicalMatching,
    TopCubicalMatchingConfig,
};
use ndarray::Array2;
use rustc_hash::FxHashMap;

use crate::f2_vector::F2Vector;

/// Computes cohomology generators for the cubical complex defined by
/// `canonical_cubes`, which must be sorted, deduplicated, in range for `i32`,
/// and hold at least one row.
#[must_use]
pub(super) fn compute_generators(
    canonical_cubes: &Array2<i64>,
    backend: &ExecutionBackend,
) -> Vec<Chain<Cube>> {
    let dimension = canonical_cubes.ncols();
    let field = Field::new(2);

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

    let grading_function = TopCubeGrader::new(OrthantTrie::uniform(orthants, 0, 1), Some(0));

    let complex = CubicalComplex::new(minimum_orthant, maximum_orthant, grading_function, field);

    let configuration = TopCubicalMatchingConfig {
        maximum_critical_grade: 0,
        maximum_critical_dimension: 2,
        filter_orthants: true,
        ..TopCubicalMatchingConfig::default()
    };
    let top_matching = TopCubicalMatching::from_config(configuration, complex, backend);

    let (lower_matchings, morse_complex) =
        top_matching.full_reduce(CoreductionMatching::new, backend);

    let generator_cells: Vec<u32> = morse_complex
        .iter()
        .filter(|cell| morse_complex.cell_dimension(cell) == 1)
        .collect();

    let mut lower_reps: Vec<Chain<u32>> = generator_cells
        .iter()
        .map(|&cell| Chain::unit(field, cell))
        .collect();

    for matching in lower_matchings.iter().rev() {
        lower_reps = lower_reps
            .into_iter()
            .map(|chain| matching.colift_capped(chain, 0))
            .collect();
    }

    let mut generators: Vec<Chain<Cube>> = lower_reps
        .into_iter()
        .map(|chain| top_matching.colift_capped(chain, 0))
        .collect();

    // Lex-sort generators
    generators.sort_by(|left, right| {
        let mut left_keys: Vec<(&[i32], u32)> = left
            .iter()
            .map(|(cube, _)| (cube.base().as_slice(), cube.extent()))
            .collect();
        left_keys.sort_unstable();

        let mut right_keys: Vec<(&[i32], u32)> = right
            .iter()
            .map(|(cube, _)| (cube.base().as_slice(), cube.extent()))
            .collect();
        right_keys.sort_unstable();

        left_keys.cmp(&right_keys)
    });

    generators
}

/// Builds the edge-to-class lookup table from the generator chains. Each
/// generator contributes a `1` at its own index for every edge it contains.
#[must_use]
pub(super) fn compute_edge_classes(generators: &[Chain<Cube>]) -> FxHashMap<Cube, F2Vector> {
    let mut edge_classes: FxHashMap<Cube, F2Vector> = FxHashMap::default();
    let num_generators = generators.len();
    for (generator_index, generator) in generators.iter().enumerate() {
        for (cube, _) in generator {
            let entry = edge_classes
                .entry(cube.clone())
                .or_insert_with(|| F2Vector::zeros(num_generators));
            entry.set(generator_index, true);
        }
    }
    edge_classes
}
