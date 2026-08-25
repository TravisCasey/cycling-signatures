// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! The Morse reduction of a cubical complex and the class table the cover's
//! edge classifier reads.

use chomp3rs::{
    CellComplex, Complex, Coreduction, CubicalComplex, ExecutionBackend, Field, MorseReduction,
    Orthant, OrthantTrie, TopCubeGrader, TopCubicalMatching, TopCubicalMatchingConfig,
};
use ndarray::Array2;
use rustc_hash::FxHashMap;

use super::classifier::EdgeClassifier;
use crate::f2_vector::F2Vector;

/// Builds the graded cubical complex on `canonical_cubes`, which must be
/// sorted, deduplicated, in range for `i32`, and hold at least one row.
///
/// The complex spans the minimal bounding orthant of `canonical_cubes` and
/// grades top cubes uniformly: 0 on `canonical_cubes` itself, 1 elsewhere.
#[must_use]
pub(super) fn graded_complex(
    canonical_cubes: &Array2<i64>,
) -> CubicalComplex<TopCubeGrader<OrthantTrie>> {
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

    CubicalComplex::new(minimum_orthant, maximum_orthant, grading_function, field)
}

/// The top cubical matching configuration this crate reduces every cover
/// under: critical cells capped at grade 0 and dimension 2, matched only
/// against orthants a graded cell occupies.
#[must_use]
pub(super) fn matching_configuration() -> TopCubicalMatchingConfig {
    TopCubicalMatchingConfig {
        maximum_critical_grade: 0,
        maximum_critical_dimension: 2,
        filter_orthants: true,
        ..TopCubicalMatchingConfig::default()
    }
}

/// Runs the discrete Morse reduction of the cubical complex on
/// `canonical_cubes`, which must be sorted, deduplicated, in range for `i32`,
/// and hold at least one row.
///
/// Returns the top matching and the coreduction of its Morse complex.
#[must_use]
fn reduce(
    canonical_cubes: &Array2<i64>,
    backend: &ExecutionBackend,
) -> (TopCubicalMatching<OrthantTrie>, Coreduction) {
    let complex = graded_complex(canonical_cubes);
    let top_matching = TopCubicalMatching::from_config(matching_configuration(), complex, backend);
    let reduction = Coreduction::new(top_matching.construct_morse_complex(backend), backend);

    (top_matching, reduction)
}

/// The 1-cells of the final Morse complex, in the complex's iteration order.
///
/// This order is the generator basis order: coordinate `i` of every edge class
/// refers to the `i`-th cell returned here.
#[must_use]
fn generator_cells(morse_complex: &CellComplex) -> Vec<u32> {
    morse_complex
        .iter()
        .filter(|cell| morse_complex.cell_dimension(cell) == 1)
        .collect()
}

/// Asserts that the final Morse complex has zero boundary through degree 2.
///
/// Treating every 1-cell of the final complex as a generator relies on this:
/// with a vanishing boundary the 1-cells form a basis of first homology rather
/// than mere chains.
fn assert_zero_boundary(morse_complex: &CellComplex) {
    for cell in morse_complex.iter() {
        let dimension = morse_complex.cell_dimension(&cell);
        if dimension > 2 {
            continue;
        }
        let boundary_terms = morse_complex.cell_boundary_terms(&cell).count();
        assert!(
            boundary_terms == 0,
            "a dimension-{dimension} cell of the reduced Morse complex has {boundary_terms} \
             boundary terms"
        );
    }
}

/// Lowers each critical 1-cell of the top matching through `reduction` and
/// reads its class on the final complex's 1-cells.
///
/// Returns one entry per critical cell of the top matching, indexed by Morse
/// cell index; cells of other dimensions hold `None`. The coreduction's
/// projection is a linear map, so this table determines the class of any chain
/// the top matching produces: a chain's class is the sum of its cells' entries.
///
/// The entries are identical on every rank of a distributed backend, since
/// every rank holds the same coreduction.
#[must_use]
fn critical_cell_classes(
    top_matching: &TopCubicalMatching<OrthantTrie>,
    reduction: &Coreduction,
    generator_cells: &[u32],
) -> Vec<Option<F2Vector>> {
    let cell_to_index: FxHashMap<u32, usize> = generator_cells
        .iter()
        .enumerate()
        .map(|(index, &cell)| (cell, index))
        .collect();

    let critical = top_matching.critical_cells();
    let mut table: Vec<Option<F2Vector>> = vec![None; critical.len()];
    for (cell, critical_cell) in critical.iter().enumerate() {
        if critical_cell.dimension() != 1 {
            continue;
        }
        let chain = reduction.lower_cell(cell as u32);
        let mut class = F2Vector::zeros(generator_cells.len());
        for (final_cell, _) in &chain {
            let index = cell_to_index
                .get(final_cell)
                .expect("a lowered 1-chain has coefficients on 1-cells of the final complex");
            class.set(*index, true);
        }
        table[cell] = Some(class);
    }
    table
}

/// Builds an on-demand edge classifier for the cubical complex defined by
/// `canonical_cubes`, which must be sorted, deduplicated, in range for `i32`,
/// and hold at least one row.
///
/// The classifier evaluates the `F_2` homology class of any cover edge against
/// the generator basis of the fully reduced Morse complex; see
/// [`EdgeClassifier`] for that basis.
///
/// Every rank of a distributed backend builds an identical classifier, since
/// the underlying table is deterministic and rank-local; no synchronization is
/// performed.
///
/// # Panics
///
/// Panics if the fully reduced Morse complex has a nonzero boundary in degree
/// at most 2, which would make its 1-cells mere chains rather than a basis of
/// first homology.
#[must_use]
pub(super) fn compute_classifier(
    canonical_cubes: &Array2<i64>,
    backend: &ExecutionBackend,
) -> EdgeClassifier {
    let (top_matching, reduction) = reduce(canonical_cubes, backend);
    let morse_complex = reduction.morse_complex();
    assert_zero_boundary(morse_complex);

    let cells = generator_cells(morse_complex);
    let num_generators = cells.len();
    let classes = critical_cell_classes(&top_matching, &reduction, &cells);

    EdgeClassifier::new(top_matching, classes, num_generators)
}
