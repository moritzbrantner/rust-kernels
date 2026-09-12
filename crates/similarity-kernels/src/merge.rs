use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Merges any number of strictly sorted, duplicate-free slices into one sorted-unique vector.
///
/// The function performs a k-way heap merge in `O(n log k)` comparisons for
/// `n` total input elements and `k` input slices, retaining at most one cursor
/// per non-empty slice. Inputs are assumed to be sorted-unique already; caller
/// normalization and validation policy stays outside this kernel.
#[must_use]
pub fn merge_sorted_unique_many<T: Clone + Ord>(inputs: &[&[T]]) -> Vec<T> {
    let mut heap = BinaryHeap::new();
    let mut total_len = 0_usize;

    for (input_index, input) in inputs.iter().copied().enumerate() {
        total_len = total_len.saturating_add(input.len());
        if let Some(first) = input.first() {
            heap.push(Reverse((first, input_index, 0_usize)));
        }
    }

    let mut output = Vec::with_capacity(total_len);
    while let Some(Reverse((value, input_index, item_index))) = heap.pop() {
        if output.last().is_none_or(|previous| previous != value) {
            output.push(value.clone());
        }

        let next_index = item_index + 1;
        if let Some(next) = inputs[input_index].get(next_index) {
            heap.push(Reverse((next, input_index, next_index)));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::merge_sorted_unique_many;

    #[test]
    fn merges_sorted_posting_lists_and_removes_cross_list_duplicates() {
        let first = [1, 4, 9];
        let second = [2, 4, 8];
        let third = [1, 3, 10];

        assert_eq!(
            merge_sorted_unique_many(&[&first, &second, &third]),
            vec![1, 2, 3, 4, 8, 9, 10]
        );
    }

    #[test]
    fn empty_and_single_input_boundaries_are_explicit() {
        let empty: [u32; 0] = [];
        let values = [2, 5, 8];

        assert!(merge_sorted_unique_many::<u32>(&[]).is_empty());
        assert!(merge_sorted_unique_many(&[&empty, &empty]).is_empty());
        assert_eq!(merge_sorted_unique_many(&[&values]), values);
        assert_eq!(merge_sorted_unique_many(&[&empty, &values, &empty]), values);
    }

    #[test]
    fn exhaustive_small_posting_lists_match_btree_set_union() {
        let sets = (0_u8..16).map(set_from_mask).collect::<Vec<_>>();

        for first in &sets {
            for second in &sets {
                for third in &sets {
                    let expected = first
                        .iter()
                        .chain(second)
                        .chain(third)
                        .copied()
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>();

                    assert_eq!(
                        merge_sorted_unique_many(&[
                            first.as_slice(),
                            second.as_slice(),
                            third.as_slice(),
                        ]),
                        expected,
                        "first={first:?}, second={second:?}, third={third:?}"
                    );
                }
            }
        }
    }

    fn set_from_mask(mask: u8) -> Vec<u8> {
        (0_u8..4).filter(|value| mask & (1 << value) != 0).collect()
    }
}
