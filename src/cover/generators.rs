// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! The Morse reduction of a cubical complex, the edge universe every cycle walk
//! draws from, and the homology class each of those edges carries.

use chomp3rs::{
    CellComplex, Chain, Complex, Coreduction, Cube, CubicalComplex, ExecutionBackend, Field,
    MorseReduction, Orthant, OrthantTrie, TopCubeGrader, TopCubicalMatching,
    TopCubicalMatchingConfig,
};
use ndarray::Array2;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::{classifier::EdgeClassifier, non_adjacent_axis, staircase::step_between_cubes};
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

/// Enumerates the edge universe of `canonical_cubes`: every staircase edge of
/// every ordered pair of adjacent cubes (coordinates differing by at most 1 on
/// every axis). Deduplicated and sorted by base coordinates then extent.
///
/// The universe contains every edge any cycle walk against this cover can emit:
/// walked steps connect cube-adjacent cubes, and each ordered pair has one
/// deterministic staircase.
#[must_use]
pub(super) fn enumerate_edge_universe(canonical_cubes: &Array2<i64>) -> Vec<Cube> {
    let dimension = canonical_cubes.ncols();
    let mut edges: FxHashSet<Cube> = FxHashSet::default();
    let mut emit_staircase = |from: &[i64], to: &[i64]| {
        let mut position: Orthant = from.iter().map(|&value| value as i32).collect();
        step_between_cubes(from, to, dimension, &mut position, &mut |edge| {
            edges.insert(edge.clone());
        })
        .expect("cubes were checked adjacent before emitting their staircase");
    };

    for (first_index, first) in canonical_cubes.outer_iter().enumerate() {
        for second in canonical_cubes.outer_iter().skip(first_index + 1) {
            if non_adjacent_axis(first.iter(), second.iter()).is_some() {
                continue;
            }

            let first = first
                .as_slice()
                .expect("canonical cube rows are contiguous");
            let second = second
                .as_slice()
                .expect("canonical cube rows are contiguous");
            emit_staircase(first, second);
            emit_staircase(second, first);
        }
    }

    let mut universe: Vec<Cube> = edges.into_iter().collect();
    universe.sort_unstable_by(|left, right| {
        (left.base().as_slice(), left.extent()).cmp(&(right.base().as_slice(), right.extent()))
    });
    universe
}

/// Projects each chain onto the top matching's Morse complex, collecting the
/// delivered results back into input order.
///
/// Under a distributed backend the results are delivered on the root rank only;
/// other ranks receive placeholder chains and rely on the caller broadcasting
/// the finished map. Every rank must still make the call: the dispatch is
/// collective.
#[must_use]
fn lower_batch(
    matching: &TopCubicalMatching<OrthantTrie>,
    chains: Vec<Chain<Cube>>,
    backend: &ExecutionBackend,
) -> Vec<Chain<u32>> {
    let expected = chains.len();
    let mut deliveries = 0_usize;
    let mut results: Vec<Option<Chain<u32>>> = Vec::new();
    results.resize_with(expected, || None);
    matching.lower_each(chains, backend, |index, chain| {
        results[index] = Some(chain);
        deliveries += 1;
    });

    if !backend.is_root() {
        return results
            .into_iter()
            .map(|chain| chain.unwrap_or_else(|| Chain::new(Field::new(2))))
            .collect();
    }

    assert!(
        deliveries == expected,
        "{deliveries} chains were delivered where {expected} were dispatched"
    );
    results
        .into_iter()
        .map(|chain| chain.expect("every chain is delivered exactly once"))
        .collect()
}

/// The number of universe edges lowered through the top matching at once.
/// Bounds the lowered chains simultaneously realized during the universe pass
/// to one slice's worth.
const LOWERING_SLICE_CHAINS: usize = 1 << 16;

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

/// Lowers every universe edge through the tower and reads each result's
/// coefficients on the final complex's 1-cells into an edge-to-class map.
///
/// The map holds one entry per universe edge, explicit zero classes included.
/// Under a distributed backend the results exist on the root rank only; the
/// caller broadcasts the finished map.
#[must_use]
fn lower_universe(
    universe: Vec<Cube>,
    top_matching: &TopCubicalMatching<OrthantTrie>,
    reduction: &Coreduction,
    generator_cells: &[u32],
    backend: &ExecutionBackend,
) -> FxHashMap<Cube, F2Vector> {
    let field = Field::new(2);
    let classes = critical_cell_classes(top_matching, reduction, generator_cells);
    let mut edge_classes: FxHashMap<Cube, F2Vector> =
        FxHashMap::with_capacity_and_hasher(universe.len(), FxBuildHasher);

    // Each edge is lowered through the top matching alone; the coreduction's
    // projection is applied through the precomposed table. The universe is
    // dispatched in bounded slices so the lowered chains never exist for more
    // than one slice at a time. Every rank iterates the same slices, so the
    // collective dispatches inside `lower_batch` stay aligned.
    let mut edges = universe.into_iter().peekable();
    while edges.peek().is_some() {
        let slice: Vec<Cube> = edges.by_ref().take(LOWERING_SLICE_CHAINS).collect();
        let unit_chains: Vec<Chain<Cube>> = slice
            .iter()
            .map(|edge| Chain::unit(field, edge.clone()))
            .collect();
        let lowered = lower_batch(top_matching, unit_chains, backend);

        for (edge, chain) in slice.into_iter().zip(lowered) {
            let mut class = F2Vector::zeros(generator_cells.len());
            for (cell, _) in &chain {
                let entry = classes[*cell as usize]
                    .as_ref()
                    .expect("a lowered edge has coefficients on critical 1-cells");
                class ^= entry;
            }
            edge_classes.insert(edge, class);
        }
    }

    edge_classes
}

/// Computes the edge-to-class map for the cubical complex defined by
/// `canonical_cubes`, which must be sorted, deduplicated, in range for `i32`,
/// and hold at least one row.
///
/// Returns the generator count (the number of 1-cells of the fully reduced
/// Morse complex, whose iteration order is the generator basis order) and the
/// class of every edge in the cover's edge universe, explicit zero classes
/// included. Edges outside the map cannot be produced by any cycle walk against
/// this cover. The map is complete on every rank of a distributed backend.
///
/// # Panics
///
/// Panics if the fully reduced Morse complex has a nonzero boundary in degree
/// at most 2, which would make its 1-cells mere chains rather than a basis of
/// first homology.
#[must_use]
pub(super) fn compute_edge_map(
    canonical_cubes: &Array2<i64>,
    backend: &ExecutionBackend,
) -> (usize, FxHashMap<Cube, F2Vector>) {
    let (top_matching, reduction) = reduce(canonical_cubes, backend);
    let morse_complex = reduction.morse_complex();
    assert_zero_boundary(morse_complex);

    let cells = generator_cells(morse_complex);
    let universe = enumerate_edge_universe(canonical_cubes);
    let edge_classes = lower_universe(universe, &top_matching, &reduction, &cells, backend);

    let edge_classes = backend.sync(edge_classes);

    (cells.len(), edge_classes)
}

/// Builds an on-demand edge classifier for the cubical complex defined by
/// `canonical_cubes`, which must be sorted, deduplicated, in range for `i32`,
/// and hold at least one row.
///
/// The classifier evaluates the `F_2` homology class of any cover edge against
/// the same generator basis [`compute_edge_map`] uses; see [`EdgeClassifier`]
/// for that basis.
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
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use chomp3rs::{Cube, ExecutionBackend, Orthant};
    use ndarray::{Array2, array};

    use super::{compute_classifier, compute_edge_map, enumerate_edge_universe};
    use crate::{
        trajectory::Trajectory,
        util::fixtures::{densify_path, embed_euclidean, ring_waypoints_3d},
    };

    #[test]
    fn universe_edge_counts_for_known_pairs() {
        // Two cubes sharing a face: the two staircases traverse the same single
        // edge. Two diagonal cubes: two edges per direction, all four distinct.
        // Two non-adjacent cubes: no pair, no edges.
        let face_sharing = array![[0_i64, 0], [1, 0]];
        assert_eq!(enumerate_edge_universe(&face_sharing).len(), 1);

        let diagonal = array![[0_i64, 0], [1, 1]];
        assert_eq!(enumerate_edge_universe(&diagonal).len(), 4);

        let separated = array![[0_i64, 0], [3, 0]];
        assert!(enumerate_edge_universe(&separated).is_empty());
    }

    /// The twelve cubes on the boundary of a `4x4` grid, leaving a `2x2` hole
    /// in the middle, duplicated from `cover.rs`'s ring fixture so this
    /// scaffolding test carries no dependency on the module it exercises.
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

    /// The canonical cube set of a densified, embedded loop through three
    /// dimensions.
    fn three_dimensional_cubes() -> Array2<i64> {
        let points = densify_path(&ring_waypoints_3d(), 0.5);
        let trajectory = Trajectory::new(points.view()).unwrap();
        let embedded = embed_euclidean(trajectory).unwrap();
        embedded.cover().cubes().to_owned()
    }

    /// A 1-cube edge and a 2-cube, both based two units past the maximum
    /// coordinate any cube of `canonical_cubes` reaches on axis 0, and sized
    /// to `canonical_cubes`'s ambient dimension.
    fn edge_and_square_outside(canonical_cubes: &Array2<i64>) -> (Cube, Cube) {
        let dimension = canonical_cubes.ncols();
        let outside = canonical_cubes.column(0).iter().copied().max().unwrap() as i32 + 2;
        let mut base = vec![0_i32; dimension];
        base[0] = outside;
        let orthant = Orthant::from(base.as_slice());
        let edge = Cube::from_extent(orthant.clone(), 0b1);
        let square = Cube::from_extent(orthant, 0b11);
        (edge, square)
    }

    #[test]
    fn classifier_agrees_with_the_eager_map() {
        let backend = ExecutionBackend::default();
        for canonical_cubes in [ring_cubes(), three_dimensional_cubes()] {
            let (num_generators, edge_classes) = compute_edge_map(&canonical_cubes, &backend);
            let classifier = compute_classifier(&canonical_cubes, &backend);

            assert_eq!(num_generators, classifier.num_generators());
            for (edge, class) in &edge_classes {
                assert!(classifier.is_cover_edge(edge));
                assert_eq!(&classifier.class_of(edge), class);
            }

            let (far_edge, far_square) = edge_and_square_outside(&canonical_cubes);
            assert!(!classifier.is_cover_edge(&far_edge));
            assert!(!classifier.is_cover_edge(&far_square));
        }
    }
}
