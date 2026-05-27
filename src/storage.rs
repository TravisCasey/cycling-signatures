// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Component-first cycle storage.
//!
//! Persistent cache for all near-recurrent cycles over a trajectory extent
//! (the range of original indices the storage was built over). Cycles are
//! grouped by their connected component in the below-threshold distance
//! graph, paired with the homology class shared across the component.

#[allow(dead_code)]
pub(crate) mod interval_subsumption;
