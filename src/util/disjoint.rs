// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Union-find with path compression and union by rank.

use std::cmp::Ordering;

/// Union-find data structure with path compression and union by rank.
///
/// Elements are inserted dynamically via [`insert`](Self::insert), which
/// returns the assigned id.
#[derive(Debug, Clone)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    /// Creates an empty set. `capacity` is a hint for the expected element
    /// count and does not bound how many elements may be inserted.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            parent: Vec::with_capacity(capacity),
            rank: Vec::with_capacity(capacity),
        }
    }

    /// Inserts a fresh singleton element and returns its id.
    #[must_use]
    pub fn insert(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    /// Finds the representative of the component containing `id` with path
    /// compression.
    ///
    /// # Panics
    ///
    /// Panics if `id >= self.len()`.
    #[must_use]
    pub fn find(&mut self, id: usize) -> usize {
        let mut cursor = id;
        while self.parent[cursor] != cursor {
            let parent = self.parent[cursor];
            self.parent[cursor] = self.parent[parent];
            cursor = parent;
        }
        cursor
    }

    /// Unions the components containing `a` and `b`.
    ///
    /// # Panics
    ///
    /// Panics if `a >= self.len()` or `b >= self.len()`.
    pub fn union(&mut self, a: usize, b: usize) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        let rank_a = self.rank[root_a];
        let rank_b = self.rank[root_b];
        match rank_a.cmp(&rank_b) {
            Ordering::Less => self.parent[root_a] = root_b,
            Ordering::Greater => self.parent[root_b] = root_a,
            Ordering::Equal => {
                self.parent[root_b] = root_a;
                self.rank[root_a] += 1;
            },
        }
    }

    /// Returns the number of elements in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }
}

#[cfg(test)]
mod tests {
    use super::DisjointSet;

    #[test]
    fn insert_find_union_groups_correctly() {
        let mut set = DisjointSet::new(0);
        let elements: Vec<usize> = (0..5).map(|_| set.insert()).collect();
        // Initially every element is its own representative.
        for &id in &elements {
            assert_eq!(set.find(id), id);
        }

        // Union 0~1, 2~3, then 1~3 -> all of {0,1,2,3} share a rep; 4 is alone.
        set.union(elements[0], elements[1]);
        set.union(elements[2], elements[3]);
        set.union(elements[1], elements[3]);

        let rep_0 = set.find(elements[0]);
        let rep_1 = set.find(elements[1]);
        let rep_2 = set.find(elements[2]);
        let rep_3 = set.find(elements[3]);
        let rep_4 = set.find(elements[4]);

        assert_eq!(rep_0, rep_1);
        assert_eq!(rep_1, rep_2);
        assert_eq!(rep_2, rep_3);
        assert_ne!(rep_3, rep_4);
    }
}
