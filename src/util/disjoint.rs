// This file is part of cycling-signatures, licensed under the GPL-3.0-or-later.
// See LICENSE or <https://www.gnu.org/licenses/gpl-3.0.html>.

//! Union-find with path compression and union by rank optimizations.

use std::cmp::Ordering;

/// Union-find data structure with path compression and union by rank.
///
/// Elements are inserted dynamically via [`insert`](Self::insert), which
/// returns the assigned index.
#[derive(Debug, Clone)]
pub(crate) struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    /// Empty initialization: no sets are created.
    ///
    /// Use [`insert`](Self::insert) to dynamically create sets and
    /// [`union`](Self::union) to merge them.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
        }
    }

    /// Inserts a fresh singleton set and returns its sequential index.
    #[must_use]
    pub(crate) fn insert(&mut self) -> usize {
        let index = self.parent.len();
        self.parent.push(index);
        self.rank.push(0);
        index
    }

    /// Finds the representative of the set containing `index` with path
    /// compression.
    ///
    /// # Panics
    ///
    /// Panics if `index` was not returned by a prior call to
    /// [`insert`](Self::insert).
    #[must_use]
    pub(crate) fn find(&mut self, index: usize) -> usize {
        let mut cursor = index;
        while self.parent[cursor] != cursor {
            let parent = self.parent[cursor];
            self.parent[cursor] = self.parent[parent];
            cursor = parent;
        }
        cursor
    }

    /// Merges the sets containing `left` and `right`.
    ///
    /// # Panics
    ///
    /// Panics if either `left` or `right` was not returned by a prior call to
    /// [`insert`](Self::insert).
    pub(crate) fn union(&mut self, left: usize, right: usize) {
        let root_left = self.find(left);
        let root_right = self.find(right);
        if root_left == root_right {
            return;
        }
        match self.rank[root_left].cmp(&self.rank[root_right]) {
            Ordering::Less => self.parent[root_left] = root_right,
            Ordering::Greater => self.parent[root_right] = root_left,
            Ordering::Equal => {
                self.parent[root_right] = root_left;
                self.rank[root_left] += 1;
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DisjointSet;

    #[test]
    fn insert_find_union_groups_correctly() {
        let mut set = DisjointSet::new();
        let elements: Vec<usize> = (0..5).map(|_| set.insert()).collect();
        // Initially every element is its own representative.
        for &index in &elements {
            assert_eq!(set.find(index), index);
        }

        // Union 0 <-> 1, 2 <-> 3, then 1 <-> 3
        // By transitivity, all of 0, 1, 2, 3 share a root; 4 is alone.
        set.union(elements[0], elements[1]);
        set.union(elements[2], elements[3]);
        set.union(elements[1], elements[3]);

        let roots: Vec<usize> = elements.iter().map(|&index| set.find(index)).collect();

        assert_eq!(roots[0], roots[1]);
        assert_eq!(roots[1], roots[2]);
        assert_eq!(roots[2], roots[3]);
        assert_ne!(roots[3], roots[4]);
    }
}
