// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! An on-demand evaluator of the `F_2` homology class of a single cover edge.

use chomp3rs::{Cube, Grader, MorseReduction, OrthantTrie, TopCubicalMatching};

use crate::f2_vector::F2Vector;

/// Evaluates the `F_2` homology class of a single cover edge on demand, against
/// the discrete Morse reduction of a cubical cover.
///
/// Coordinate `i` of every class this classifier returns refers to the `i`-th
/// 1-cell of the fully reduced Morse complex, in that complex's iteration
/// order: this is the generator basis every class from one classifier shares.
#[allow(dead_code)]
#[derive(Debug)]
pub(super) struct EdgeClassifier {
    top_matching: TopCubicalMatching<OrthantTrie>,
    // The class of each critical cell of `top_matching`, indexed by that
    // cell's Morse cell index. `Some` exactly on critical 1-cells; cells of
    // other dimensions hold `None`.
    classes: Vec<Option<F2Vector>>,
    num_generators: usize,
}

#[allow(dead_code)]
impl EdgeClassifier {
    /// Assembles a classifier from a top matching and the class of each of its
    /// critical 1-cells, indexed by Morse cell index, against a basis of
    /// `num_generators` generators.
    #[must_use]
    pub(super) fn new(
        top_matching: TopCubicalMatching<OrthantTrie>,
        classes: Vec<Option<F2Vector>>,
        num_generators: usize,
    ) -> Self {
        Self {
            top_matching,
            classes,
            num_generators,
        }
    }

    /// The `F_2` homology class of `edge`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `edge` is not a cover edge (see
    /// [`is_cover_edge`](Self::is_cover_edge)). Release builds skip this
    /// check, so a walk path that has already established membership does
    /// not pay it again.
    #[must_use]
    pub(super) fn class_of(&self, edge: &Cube) -> F2Vector {
        debug_assert!(self.is_cover_edge(edge));

        let chain = self.top_matching.lower_cell(edge.clone());
        let mut class = F2Vector::zeros(self.num_generators);
        for (cell, _) in &chain {
            let entry = self.classes[*cell as usize]
                .as_ref()
                .expect("a lowered edge has coefficients on critical 1-cells");
            class ^= entry;
        }
        class
    }

    /// Reports whether `edge` is a cover edge: a 1-cube belonging to the
    /// cubical complex the top matching was built on.
    #[must_use]
    pub(super) fn is_cover_edge(&self, edge: &Cube) -> bool {
        edge.dimension() == 1 && self.top_matching.upper_complex().grade(edge) == 0
    }

    /// The number of cohomology generators, the length of every class this
    /// classifier returns.
    #[must_use]
    pub(super) fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// The critical cells of the top matching, in Morse cell index order.
    #[must_use]
    pub(super) fn critical_cells(&self) -> &[Cube] {
        self.top_matching.critical_cells()
    }
}
