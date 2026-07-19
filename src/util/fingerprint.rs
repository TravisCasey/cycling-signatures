// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Stable content fingerprint used for persisted-artifact provenance.

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

/// Accumulates a stable 64-bit content fingerprint (FNV-1a).
///
/// Deterministic and portable: callers feed explicit little-endian bytes, so
/// the digest does not depend on platform endianness or pointer width.
#[derive(Clone, Debug)]
pub(crate) struct Fingerprint {
    state: u64,
}

impl Fingerprint {
    /// A fresh accumulator seeded with the FNV-1a offset basis.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            state: FNV_OFFSET_BASIS,
        }
    }

    /// Folds `bytes` into the running state.
    pub(crate) fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    /// The canonical fingerprint of everything written thus far.
    #[must_use]
    pub(crate) fn finish(self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::Fingerprint;

    #[test]
    fn matches_published_fnv1a_vectors() {
        // Published FNV-1a-64 reference vectors.
        let mut empty = Fingerprint::new();
        empty.write(b"");
        assert_eq!(empty.finish(), 0xcbf29ce484222325);

        let mut single = Fingerprint::new();
        single.write(b"a");
        assert_eq!(single.finish(), 0xaf63dc4c8601ec8c);
    }
}
