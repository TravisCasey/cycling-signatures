// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Segment-containment index over intervals carrying integer payloads and a
//! birth distance.
//!
//! [`IntervalSubsumptionIndex`] answers, for a given range, which stored
//! intervals are fully contained in it. At construction, per-payload
//! minimal-subsumption deduplication drops any interval that strictly
//! contains another same-payload interval whose birth is no greater than its
//! own: such an interval can never be the unique best answer to a window
//! query, since any window admitting it also admits the interval it contains,
//! at an equal or better birth.

use std::{collections::BTreeMap, ops::Range};

/// A stored interval with payload and birth, using half-open `[begin, end)`
/// endpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StoredInterval {
    pub(crate) begin: u32,
    pub(crate) end: u32,
    pub(crate) payload: u32,
    pub(crate) birth: f64,
}

/// Index of intervals, deduplicated per payload by birth-aware minimal
/// subsumption.
#[derive(Clone, Debug)]
pub(crate) struct IntervalSubsumptionIndex {
    intervals: Vec<StoredInterval>,
}

impl IntervalSubsumptionIndex {
    /// Builds a deduplicated index from `(range, payload, birth)` triples,
    /// applying the minimal-subsumption rule described on the module.
    #[must_use]
    pub(crate) fn new(ranges: impl IntoIterator<Item = (Range<u32>, u32, f64)>) -> Self {
        let mut intervals: Vec<StoredInterval> = ranges
            .into_iter()
            .map(|(range, payload, birth)| StoredInterval {
                begin: range.start,
                end: range.end,
                payload,
                birth,
            })
            .collect();

        // Sort by ascending payload, then ascending `begin`, then descending
        // `end`, then ascending `birth`.
        intervals.sort_unstable_by(|left, right| {
            left.payload
                .cmp(&right.payload)
                .then(left.begin.cmp(&right.begin))
                .then(right.end.cmp(&left.end))
                .then(left.birth.total_cmp(&right.birth))
        });
        // A stored cycle's range determines its birth, so entries equal on
        // every field are genuine duplicates of the same cycle.
        intervals.dedup();

        // Right-to-left sweep within each payload group: `frontier` maps each
        // already-seen survivor's `end` to its `birth`, kept with births
        // strictly descending as `end` ascends. Because processing runs in
        // descending `begin` order, a frontier entry with `end <= current.end`
        // is exactly a survivor strictly contained in the current interval
        // (exact duplicates were removed above).
        let mut result: Vec<StoredInterval> = Vec::with_capacity(intervals.len());
        let mut current_payload: Option<u32> = None;
        let mut frontier: BTreeMap<u32, f64> = BTreeMap::new();
        for interval in intervals.iter().rev() {
            if current_payload != Some(interval.payload) {
                current_payload = Some(interval.payload);
                frontier.clear();
            }

            let smallest_contained_birth = frontier
                .range(..=interval.end)
                .next_back()
                .map(|(_, &birth)| birth);
            let dominated = smallest_contained_birth.is_some_and(|birth| birth <= interval.birth);
            if dominated {
                continue;
            }
            result.push(*interval);

            // Insert the survivor, removing the run it dominates. Survival
            // means every entry at or below `interval.end` has a strictly
            // larger birth, so the descending-birth invariant holds after
            // the insert.
            while let Some((&end, &birth)) = frontier.range(interval.end..).next() {
                if birth < interval.birth {
                    break;
                }
                frontier.remove(&end);
            }
            frontier.insert(interval.end, interval.birth);
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

    /// Brute-force deduplication: keep an interval if and only if no
    /// same-payload interval in the input list is strictly contained in it
    /// with a birth less than or equal to its own.
    fn naive_dedup(intervals: &[(Range<u32>, u32, f64)]) -> BTreeSet<(u32, u32, u32)> {
        intervals
            .iter()
            .filter(|(range, payload, birth)| {
                !intervals
                    .iter()
                    .any(|(other_range, other_payload, other_birth)| {
                        other_payload == payload
                            && (other_range.start != range.start || other_range.end != range.end)
                            && range.start <= other_range.start
                            && range.end >= other_range.end
                            && other_birth <= birth
                    })
            })
            .map(|(range, payload, _birth)| (range.start, range.end, *payload))
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
        let cases: &[&[(Range<u32>, u32, f64)]] = &[
            // Fully nested chain, all same payload, birth strictly
            // decreasing with nesting depth: only the innermost survives.
            &[(0..10, 0, 3.0), (2..6, 0, 2.0), (4..6, 0, 1.0)],
            // Different payloads: per-payload dedup leaves one survivor
            // each.
            &[
                (0..10, 0, 3.0),
                (2..6, 0, 2.0),
                (4..6, 0, 1.0),
                (0..10, 1, 2.0),
                (2..6, 1, 1.0),
            ],
            // Overlapping non-containing: both survive.
            &[(0..5, 0, 1.0), (3..8, 0, 2.0)],
            // Exact duplicates collapse to one.
            &[(3..7, 9, 1.0), (3..7, 9, 1.0), (3..7, 9, 1.0)],
            // Mixed: containment + overlap + multiple payloads.
            &[
                (0..10, 0, 4.0),
                (2..6, 0, 2.0),
                (3..5, 0, 1.0),
                (0..5, 1, 2.0),
                (1..4, 1, 1.0),
                (6..10, 0, 3.0),
                (7..8, 0, 1.0),
            ],
            // Containing interval with strictly smaller birth: coverage
            // dominance alone is not enough to drop it, so both survive.
            &[(0..10, 0, 1.0), (2..6, 0, 5.0)],
            // Containing interval with equal birth: dominated in both
            // coverage and birth, so the container drops.
            &[(0..10, 0, 3.0), (2..6, 0, 3.0)],
        ];
        for &raw in cases {
            let index = IntervalSubsumptionIndex::new(raw.iter().cloned());
            assert_eq!(
                index_as_set(&index),
                naive_dedup(raw),
                "dedup mismatch on input {raw:?}"
            );
        }

        // Sliding-window chain whose births rise against the sweep's
        // processing order, so the frontier grows to the full chain length,
        // salted with containers that must drop or survive against it.
        let mut sliding: Vec<(Range<u32>, u32, f64)> = (0..40)
            .map(|start| (start..start + 5, 0, f64::from(40 - start) * 0.1))
            .collect();
        // Contains smaller-birth windows: dominated in both axes, drops.
        sliding.push((5..20, 0, 10.0));
        // Contains only larger-birth windows: survives.
        sliding.push((25..33, 0, 0.6));
        // Contains everything at the smallest birth of all: survives.
        sliding.push((0..45, 0, 0.05));
        let index = IntervalSubsumptionIndex::new(sliding.iter().cloned());
        assert_eq!(
            index_as_set(&index),
            naive_dedup(&sliding),
            "dedup mismatch on sliding-window chain"
        );
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
        let raw: &[(Range<u32>, u32, f64)] = &[
            (0..5, 0, 1.0),
            (2..5, 1, 2.0),
            (5..9, 2, 3.0),
            (1..3, 3, 4.0),
            (7..9, 4, 5.0),
        ];
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
