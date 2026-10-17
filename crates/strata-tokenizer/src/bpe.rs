//! Byte-level BPE merging.
//!
//! A word (already mapped into the byte-level alphabet) is merged by LOWEST
//! RANK first, repeatedly - not left to right. Applying merges in list order
//! rather than rank order is the classic BPE bug: it produces a different
//! segmentation and a plausible token count.
//!
//! A word longer than [`HEAP_MIN`] symbols goes through a heap-based variant
//! that applies the same merges in the same order without rescanning every
//! pair after each merge (the rescan is O(n^2) and takes seconds on a long
//! CJK run or a minified blob).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;

/// Words up to this many symbols keep the plain scan.
pub const HEAP_MIN: usize = 64;

/// The merge table: `(left, right) -> rank`.
#[derive(Debug, Clone, Default)]
pub struct Merges {
    ranks: HashMap<(String, String), u32>,
}

impl Merges {
    /// Builds the merge table. A merge naming a token that is not in the
    /// vocabulary would make BPE silently produce different ids than the
    /// model was trained with, so the caller checks the vocabulary first.
    pub fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut ranks = HashMap::new();
        for (i, pair) in pairs.into_iter().enumerate() {
            let pair = pair.as_ref();
            let (left, right) = pair
                .split_once(' ')
                .expect("a merge line is two tokens separated by a space");
            ranks.insert((left.to_string(), right.to_string()), i as u32);
        }
        Self { ranks }
    }

    pub fn len(&self) -> usize {
        self.ranks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    fn rank(&self, left: &str, right: &str) -> Option<u32> {
        self.ranks.get(&(left.to_string(), right.to_string())).copied()
    }

    /// Merges `word` into its BPE pieces (strings in the byte-level
    /// alphabet).
    pub fn apply(&self, word: &str) -> Vec<String> {
        if word.chars().count() > HEAP_MIN {
            self.apply_heap(word)
        } else {
            self.apply_scan(word)
        }
    }

    /// The plain O(n^2) scan (public for tests and short words).
    pub fn apply_scan(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(String::from).collect();
        while parts.len() > 1 {
            let mut best: Option<usize> = None;
            let mut best_rank = u32::MAX;
            for i in 0..parts.len() - 1 {
                if let Some(r) = self.rank(&parts[i], &parts[i + 1]) {
                    if r < best_rank {
                        best = Some(i);
                        best_rank = r;
                    }
                }
            }
            match best {
                Some(i) => {
                    let merged = format!("{}{}", parts[i], parts[i + 1]);
                    parts.splice(i..=i + 1, [merged]);
                }
                None => break,
            }
        }
        parts
    }

    /// The O(n log n) heap variant (public for tests and long words).
    pub fn apply_heap(&self, word: &str) -> Vec<String> {
        let symbols: Vec<String> = word.chars().map(String::from).collect();
        let n = symbols.len();
        if n < 2 {
            return symbols;
        }
        // Linked list over the symbols: `next`/`prev` in the original order,
        // `usize::MAX` the sentinel at both ends.
        let mut next: Vec<usize> = (1..n).chain(std::iter::once(usize::MAX)).collect();
        let mut prev: Vec<usize> = std::iter::once(usize::MAX).chain(0..n - 1).collect();
        let mut parts = symbols;
        // Heap of (rank, left, right); popping the lowest rank then the lowest
        // position is exactly the order the scan picks (its strict `<` keeps
        // the leftmost of equal ranks). An entry whose two symbols are no
        // longer adjacent, or whose pair has another rank now, is stale.
        let mut heap: BinaryHeap<Reverse<(u32, usize, usize)>> = BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(r) = self.rank(&parts[i], &parts[i + 1]) {
                heap.push(Reverse((r, i, i + 1)));
            }
        }
        while let Some(Reverse((r, i, j))) = heap.pop() {
            if next[i] != j {
                continue; // stale: the right neighbour merged away
            }
            if self.rank(&parts[i], &parts[j]) != Some(r) {
                continue; // stale: the pair's rank changed since the push
            }
            // Merge j into i.
            let merged = format!("{}{}", parts[i], parts[j]);
            parts[i] = merged;
            let k = next[j];
            next[i] = k;
            if k != usize::MAX {
                prev[k] = i;
                if let Some(r2) = self.rank(&parts[i], &parts[k]) {
                    heap.push(Reverse((r2, i, k)));
                }
            }
            let p = prev[i];
            if p != usize::MAX {
                if let Some(r2) = self.rank(&parts[p], &parts[i]) {
                    heap.push(Reverse((r2, p, i)));
                }
            }
        }
        // Walk the list in order.
        let mut out = Vec::new();
        let mut i = 0;
        while i != usize::MAX {
            out.push(std::mem::take(&mut parts[i]));
            i = next[i];
        }
        out
    }
}
