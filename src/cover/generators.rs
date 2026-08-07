// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Cohomology generators of a cubical complex and the edge-to-class table
//! derived from them.

use std::cmp::Ordering;

use chomp3rs::{
    Chain, Complex, CoreductionMatching, Cube, CubicalComplex, ExecutionBackend, F2, MorseMatching,
    Orthant, OrthantTrie, Ring, TopCubeGrader, TopCubicalMatching,
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
) -> Vec<Chain<Cube, F2>> {
    let dimension = canonical_cubes.ncols();

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
#[must_use]
fn compare_chains(left: &Chain<Cube, F2>, right: &Chain<Cube, F2>) -> Ordering {
    let left_entries: Vec<&Cube> = left.into_iter().map(|(cube, _)| cube).collect();
    let right_entries: Vec<&Cube> = right.into_iter().map(|(cube, _)| cube).collect();
    left_entries.cmp(&right_entries)
}

/// Builds the edge-to-class lookup table from the generator chains. Each
/// generator contributes a `1` at its own index for every edge it contains.
#[must_use]
pub(super) fn compute_edge_classes(generators: &[Chain<Cube, F2>]) -> FxHashMap<Cube, F2Vector> {
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
