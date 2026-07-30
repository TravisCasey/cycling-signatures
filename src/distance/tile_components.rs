// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! The reported results of per-tile component detection.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// One tile's components: cycle endpoint pairs grouped by component, with a
/// flag per component recording whether it carries a self-comparison.
///
/// A component carrying a self-comparison is a subgraph of the window's trivial
/// component, so none of its cycles is ever part of a result. It can still be
/// the link by which a neighboring tile's component is found trivial, and such
/// a link runs through a column two tiles share. Such a component therefore
/// lists only its cycles on the tile's first and last columns, and one with no
/// cycle on either is absent entirely.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(super) struct TileComponents {
    cycles: Vec<(u32, u32)>,
    offsets: Vec<u32>,
    contains_trivial: Vec<bool>,
}

impl TileComponents {
    /// Flattens per-component cycle lists into one buffer with a component
    /// offset table.
    #[must_use]
    pub(super) fn from_grouped(grouped: &[Vec<(u32, u32)>], contains_trivial: Vec<bool>) -> Self {
        let mut cycles = Vec::with_capacity(grouped.iter().map(Vec::len).sum());
        let mut offsets = Vec::with_capacity(grouped.len() + 1);
        offsets.push(0);
        for component in grouped {
            cycles.extend_from_slice(component);
            offsets.push(cycles.len() as u32);
        }
        Self {
            cycles,
            offsets,
            contains_trivial,
        }
    }

    /// The number of components.
    #[must_use]
    pub(super) fn len(&self) -> usize {
        self.contains_trivial.len()
    }

    /// The cycles of the component at `index`, as `(start, end)` pairs in
    /// original-index space.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Self::len).
    #[must_use]
    pub(super) fn cycles(&self, index: usize) -> &[(u32, u32)] {
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        &self.cycles[start..end]
    }

    /// Whether the component at `index` carries a self-comparison, and so the
    /// trivial cycle.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Self::len).
    #[must_use]
    pub(super) fn contains_trivial(&self, index: usize) -> bool {
        self.contains_trivial[index]
    }
}
