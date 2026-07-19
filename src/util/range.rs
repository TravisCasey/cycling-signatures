// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Normalization of [`RangeBounds<usize>`](RangeBounds)-implementing segments
//! into half-open [`Range<usize>`](Range) with bounds validation.
//!
//! Used for flexibility of trajectory segment specification with
//! [`Range<usize>`](Range) being the canonical target.

use std::ops::{Bound, Range, RangeBounds};

use crate::error::{Error, Result};

/// Normalizes any [`RangeBounds<usize>`](RangeBounds)-implementor into a
/// half-open [`Range<usize>`](Range), validated against `length` as an upper
/// bound. Resolves unbounded start and end to `0` and `length` respectively.
///
/// # Errors
///
/// [`Error::WindowOutOfBounds`] if the normalized range does not satisfy
/// `start <= end <= length`.
pub(crate) fn normalize_segment(
    segment: impl RangeBounds<usize>,
    length: usize,
) -> Result<Range<usize>> {
    let start = match segment.start_bound() {
        Bound::Included(&value) => value,
        Bound::Excluded(&value) => value.saturating_add(1),
        Bound::Unbounded => 0,
    };
    let end = match segment.end_bound() {
        Bound::Included(&value) => value.saturating_add(1),
        Bound::Excluded(&value) => value,
        Bound::Unbounded => length,
    };
    if start > end || end > length {
        return Err(Error::WindowOutOfBounds {
            start,
            end,
            trajectory_length: length,
        });
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::normalize_segment;
    use crate::error::Error;

    #[test]
    fn bounded_range_normalizes() {
        assert_eq!(normalize_segment(2..7, 10).unwrap(), 2..7);
        assert_eq!(normalize_segment(2..=6, 10).unwrap(), 2..7);
    }

    #[test]
    fn unbounded_resolves_against_length() {
        assert_eq!(normalize_segment(..4, 10).unwrap(), 0..4);
        assert_eq!(normalize_segment(3.., 10).unwrap(), 3..10);
        assert_eq!(normalize_segment(.., 10).unwrap(), 0..10);
        assert_eq!(normalize_segment(..=4, 10).unwrap(), 0..5);
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn out_of_range_returns_window_out_of_bounds() {
        let err = normalize_segment(2..15, 10).unwrap_err();
        assert!(matches!(
            err,
            Error::WindowOutOfBounds {
                start: 2,
                end: 15,
                trajectory_length: 10
            }
        ));

        let err = normalize_segment(8..3, 10).unwrap_err();
        assert!(matches!(
            err,
            Error::WindowOutOfBounds {
                start: 8,
                end: 3,
                trajectory_length: 10
            }
        ));
    }
}
