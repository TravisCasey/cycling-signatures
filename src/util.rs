// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Low-level utilities used across the crate.

pub(crate) mod disjoint;
pub(crate) mod fingerprint;
#[cfg(test)]
pub(crate) mod fixtures;
pub(crate) mod range;
