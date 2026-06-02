// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Output types for trajectory cycling signatures.
//!
//! A [`CyclingSignature`] is the `F_2` subspace of cubical-cover homology
//! classes spanned by a trajectory's recurrent cycles, packaged together with
//! the per-component decomposition that produced it. The decomposition is
//! exposed as a slice of [`CycleComponent`] for inspection, each pairing a set
//! of cycle segments in sample-index space with their shared homology class.

use std::ops::Range;

use crate::{F2Subspace, F2Vector};

/// A trajectory's cycling signature as the `F_2` subspace spanned by its
/// non-trivial cycle-component homology classes.
///
/// Constructed by
/// [`EmbeddedTrajectory::signature`](crate::EmbeddedTrajectory::signature). The
/// subspace returned by [`span`](Self::span) is the value identity of the
/// signature; equality compares spans, ignoring how many components contributed
/// each generator. The [`components`](Self::components) accessor exposes the
/// underlying `CycleComponent` decomposition for inspection and visualization.
#[derive(Debug, Clone)]
pub struct CyclingSignature {
    span: F2Subspace,
    components: Vec<CycleComponent>,
}

/// One connected component of below-threshold recurrent segments, together with
/// the homology class shared by every cycle in the component.
///
/// When constructed through [`CyclingSignature::components`], the `cycles`
/// field is non-empty: every component carries at least one detected cycle.
#[derive(Debug, Clone)]
pub struct CycleComponent {
    /// The cycle segments grouped into this component, each a half-open range
    /// `start..end` in sample-index space (`0..trajectory.original_count()`).
    /// Each range is ready to pass directly to
    /// [`EmbeddedTrajectory::walk_cycle`](crate::EmbeddedTrajectory::walk_cycle)
    /// or
    /// [`EmbeddedTrajectory::cycle_class`](crate::EmbeddedTrajectory::cycle_class).
    pub cycles: Vec<Range<usize>>,
    /// The homology class in the cover's generator basis for every segment in
    /// this component.
    pub class: F2Vector,
}

impl CyclingSignature {
    /// Constructs the signature from a vector of `CycleComponent`s. Every
    /// component's `class` is taken as a generator of the spanned subspace.
    /// `num_generators` is the dimension of the ambient space (matching the
    /// cover's generator count); every component's class must have this length.
    #[must_use]
    pub(crate) fn from_components(components: Vec<CycleComponent>, num_generators: usize) -> Self {
        let classes: Vec<F2Vector> = components
            .iter()
            .map(|component| component.class.clone())
            .collect();
        let span = F2Subspace::new(&classes, num_generators)
            .expect("class vectors have the expected length by construction");
        Self { span, components }
    }

    /// The `F_2` subspace spanned by the per-component classes. This is the
    /// signature's value identity; equality and rank are decided here.
    #[must_use]
    pub fn span(&self) -> &F2Subspace {
        &self.span
    }

    /// The per-component decomposition. Each entry pairs a list of cycle
    /// segments with the homology class shared across them.
    #[must_use]
    pub fn components(&self) -> &[CycleComponent] {
        &self.components
    }

    /// The dimension of the spanned subspace: the number of independent cycling
    /// classes the signature carries.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.span.rank()
    }
}

impl PartialEq for CyclingSignature {
    fn eq(&self, other: &Self) -> bool {
        self.span == other.span
    }
}

impl Eq for CyclingSignature {}
