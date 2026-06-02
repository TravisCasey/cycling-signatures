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
    /// Creates an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
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
    /// Panics if `id` was not returned by a prior call to
    /// [`insert`](Self::insert).
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

    /// Unions the components containing `left` and `right`.
    ///
    /// # Panics
    ///
    /// Panics if either `left` or `right` was not returned by a prior call to
    /// [`insert`](Self::insert).
    pub fn union(&mut self, left: usize, right: usize) {
        let root_left = self.find(left);
        let root_right = self.find(right);
        if root_left == root_right {
            return;
        }
        let rank_left = self.rank[root_left];
        let rank_right = self.rank[root_right];
        match rank_left.cmp(&rank_right) {
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
