// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Segment-containment index over intervals carrying integer payloads.
//!
//! [`IntervalSubsumptionIndex`] answers, for a given range, which stored
//! intervals are fully contained in it. At construction, per-payload
//! minimal-subsumption deduplication drops any interval that strictly contains
//! another interval with the same payload.

use std::ops::Range;

/// A stored interval with payload, using half-open `[begin, end)` endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StoredInterval {
    pub(crate) begin: u32,
    pub(crate) end: u32,
    pub(crate) payload: u32,
}

/// Index of intervals, deduplicated per payload by minimal subsumption.
#[derive(Clone, Debug)]
pub(crate) struct IntervalSubsumptionIndex {
    intervals: Vec<StoredInterval>,
}

impl IntervalSubsumptionIndex {
    /// Builds a deduplicated index from `(range, payload)` pairs.
    ///
    /// Within each payload group, any interval that strictly contains another
    /// is dropped.
    pub(crate) fn new(ranges: impl IntoIterator<Item = (Range<u32>, u32)>) -> Self {
        let mut intervals: Vec<StoredInterval> = ranges
            .into_iter()
            .map(|(range, payload)| StoredInterval {
                begin: range.start,
                end: range.end,
                payload,
            })
            .collect();

        // Sort by ascending payload, then ascending `begin`, then descending
        // `end`.
        intervals.sort_unstable_by(|left, right| {
            left.payload
                .cmp(&right.payload)
                .then(left.begin.cmp(&right.begin))
                .then(right.end.cmp(&left.end))
        });
        intervals.dedup();

        // Right-to-left sweep within each payload group.
        //
        // After the sort, for any two entries `left < right` with the same
        // payload, we have `left.begin <= right.begin`. `left` contains `right`
        // iff additionally `left.end >= right.end`. Scanning right-to-left and
        // tracking the minimum end seen so far within the current payload
        // group, an entry whose end is >= min_end contains a later same-payload
        // entry and is dropped.
        let mut result: Vec<StoredInterval> = Vec::with_capacity(intervals.len());
        let mut current_payload: Option<u32> = None;
        let mut min_end: u32 = u32::MAX;
        for interval in intervals.iter().rev() {
            if current_payload != Some(interval.payload) {
                current_payload = Some(interval.payload);
                min_end = u32::MAX;
            }
            if interval.end < min_end {
                result.push(*interval);
            }
            min_end = min_end.min(interval.end);
        }

        // Re-sort by (begin, end) for query-time binary search.
        result.sort_unstable_by_key(|interval| (interval.begin, interval.end));

        Self { intervals: result }
    }

    /// All stored intervals fully contained in `range`. Yielded in
    /// `(begin, end)`-ascending order.
    pub(crate) fn contained_in(
        &self,
        range: Range<u32>,
    ) -> impl Iterator<Item = &StoredInterval> + '_ {
        let start_index = self
            .intervals
            .partition_point(|interval| interval.begin < range.start);
        self.intervals[start_index..]
            .iter()
            .take_while(move |interval| interval.begin < range.end)
            .filter(move |interval| interval.end <= range.end)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, ops::Range};

    use super::{IntervalSubsumptionIndex, StoredInterval};

    /// Brute-force deduplication: keep an interval if and only if it is NOT a
    /// strict super-set of any same-payload interval in the input list.
    fn naive_dedup(intervals: &[(Range<u32>, u32)]) -> BTreeSet<(u32, u32, u32)> {
        intervals
            .iter()
            .filter(|(range, payload)| {
                !intervals.iter().any(|(other_range, other_payload)| {
                    other_payload == payload
                        && (other_range.start != range.start || other_range.end != range.end)
                        && range.start <= other_range.start
                        && range.end >= other_range.end
                })
            })
            .map(|(range, payload)| (range.start, range.end, *payload))
            .collect()
    }

    fn index_as_set(index: &IntervalSubsumptionIndex) -> BTreeSet<(u32, u32, u32)> {
        index
            .intervals
            .iter()
            .map(|interval| (interval.begin, interval.end, interval.payload))
            .collect()
    }

    #[test]
    fn dedup_oracle() {
        let cases: &[&[(Range<u32>, u32)]] = &[
            // Fully nested chain, all same payload: only the smallest survives.
            &[(0..10, 0), (2..6, 0), (4..6, 0)],
            // Different payloads: per-payload dedup leaves one survivor each.
            &[(0..10, 0), (2..6, 0), (4..6, 0), (0..10, 1), (2..6, 1)],
            // Overlapping non-containing: both survive.
            &[(0..5, 0), (3..8, 0)],
            // Exact duplicates collapse to one.
            &[(3..7, 9), (3..7, 9), (3..7, 9)],
            // Mixed: containment + overlap + multiple payloads.
            &[
                (0..10, 0),
                (2..6, 0),
                (3..5, 0),
                (0..5, 1),
                (1..4, 1),
                (6..10, 0),
                (7..8, 0),
            ],
        ];
        for &raw in cases {
            let index = IntervalSubsumptionIndex::new(raw.iter().cloned());
            assert_eq!(
                index_as_set(&index),
                naive_dedup(raw),
                "dedup mismatch on input {raw:?}"
            );
        }
    }

    fn naive_contained_in(
        intervals: &[StoredInterval],
        range: Range<u32>,
    ) -> BTreeSet<(u32, u32, u32)> {
        intervals
            .iter()
            .filter(|interval| interval.begin >= range.start && interval.end <= range.end)
            .map(|interval| (interval.begin, interval.end, interval.payload))
            .collect()
    }

    #[test]
    fn containment_oracle() {
        let raw: &[(Range<u32>, u32)] = &[(0..5, 0), (2..5, 1), (5..9, 2), (1..3, 3), (7..9, 4)];
        let index = IntervalSubsumptionIndex::new(raw.iter().cloned());
        let stored: Vec<StoredInterval> = index.intervals.clone();

        let queries: &[Range<u32>] = &[
            0..10, // covers everything
            0..5,  // exact-edge containment
            1..4,  // partial
            5..9,  // exact-edge match for one entry
            0..1,  // empty result
            0..100,
        ];

        for query in queries {
            let got: BTreeSet<(u32, u32, u32)> = index
                .contained_in(query.clone())
                .map(|interval| (interval.begin, interval.end, interval.payload))
                .collect();
            let expected = naive_contained_in(&stored, query.clone());
            assert_eq!(got, expected, "query {query:?} mismatch");
        }
    }
}
