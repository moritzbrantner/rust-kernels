use std::cmp::Ordering;

/// Reorders `values` so the element at `nth` is the same value that would
/// appear there after a full sort, returning a reference to that element.
///
/// Elements before the returned position compare less than or equal to it and
/// elements after it compare greater than or equal to it. The implementation
/// uses deterministic in-place three-way partitioning.
pub fn quickselect<T: Ord>(values: &mut [T], nth: usize) -> Option<&T> {
    if nth >= values.len() {
        return None;
    }

    let mut left = 0_usize;
    let mut right = values.len() - 1;

    loop {
        if left == right {
            return Some(&values[nth]);
        }

        let pivot = median_of_three(values, left, left + (right - left) / 2, right);
        let (equal_start, equal_end) = partition_three_way(values, left, right, pivot);

        if nth < equal_start {
            right = equal_start - 1;
        } else if nth > equal_end {
            left = equal_end + 1;
        } else {
            return Some(&values[nth]);
        }
    }
}

/// Returns the `k` smallest values in ascending order without fully sorting the
/// input when `k` is smaller than the slice length.
#[must_use]
pub fn top_k_smallest<T: Clone + Ord>(values: &[T], k: usize) -> Vec<T> {
    let k = k.min(values.len());
    if k == 0 {
        return Vec::new();
    }

    let mut selected = values.to_vec();
    let _ = quickselect(&mut selected, k - 1);
    selected.truncate(k);
    selected.sort_unstable();
    selected
}

/// Returns the best `k` values from a stream according to a sort-style comparator.
///
/// `compare(left, right) == Ordering::Less` means `left` ranks before `right`.
/// The result is returned best-first. Values that compare equal retain their
/// encounter order, providing deterministic tie-breaking without requiring the
/// caller to add a synthetic secondary key.
///
/// Unlike [`top_k_smallest`], this function does not materialize the full input:
/// it retains at most `k` candidate values while consuming the iterator. This is
/// useful for search and ranking pipelines where candidate records may be large
/// or generated incrementally.
#[must_use]
pub fn top_k_by<I, T, Compare>(values: I, k: usize, mut compare: Compare) -> Vec<T>
where
    I: IntoIterator<Item = T>,
    Compare: FnMut(&T, &T) -> Ordering,
{
    if k == 0 {
        return Vec::new();
    }

    let mut heap = Vec::with_capacity(k);
    for (sequence, value) in values.into_iter().enumerate() {
        let candidate = Ranked { value, sequence };
        if heap.len() < k {
            heap.push(candidate);
            let last = heap.len() - 1;
            sift_up_worst(&mut heap, last, &mut compare);
            continue;
        }

        if rank_cmp(&candidate, &heap[0], &mut compare).is_lt() {
            heap[0] = candidate;
            sift_down_worst(&mut heap, 0, &mut compare);
        }
    }

    heap.sort_by(|left, right| rank_cmp(left, right, &mut compare));
    heap.into_iter().map(|entry| entry.value).collect()
}

struct Ranked<T> {
    value: T,
    sequence: usize,
}

fn rank_cmp<T, Compare>(left: &Ranked<T>, right: &Ranked<T>, compare: &mut Compare) -> Ordering
where
    Compare: FnMut(&T, &T) -> Ordering,
{
    compare(&left.value, &right.value).then_with(|| left.sequence.cmp(&right.sequence))
}

fn sift_up_worst<T, Compare>(heap: &mut [Ranked<T>], mut index: usize, compare: &mut Compare)
where
    Compare: FnMut(&T, &T) -> Ordering,
{
    while index > 0 {
        let parent = (index - 1) / 2;
        if !rank_cmp(&heap[index], &heap[parent], compare).is_gt() {
            break;
        }
        heap.swap(index, parent);
        index = parent;
    }
}

fn sift_down_worst<T, Compare>(heap: &mut [Ranked<T>], mut index: usize, compare: &mut Compare)
where
    Compare: FnMut(&T, &T) -> Ordering,
{
    loop {
        let left = index * 2 + 1;
        if left >= heap.len() {
            break;
        }

        let right = left + 1;
        let mut worst_child = left;
        if right < heap.len() && rank_cmp(&heap[right], &heap[left], compare).is_gt() {
            worst_child = right;
        }

        if !rank_cmp(&heap[worst_child], &heap[index], compare).is_gt() {
            break;
        }
        heap.swap(index, worst_child);
        index = worst_child;
    }
}

fn partition_three_way<T: Ord>(
    values: &mut [T],
    left: usize,
    right: usize,
    pivot: usize,
) -> (usize, usize) {
    values.swap(pivot, right);

    let mut less = left;
    let mut scan = left;
    let mut greater = right;

    while scan < greater {
        match values[scan].cmp(&values[right]) {
            Ordering::Less => {
                values.swap(scan, less);
                less += 1;
                scan += 1;
            }
            Ordering::Equal => {
                scan += 1;
            }
            Ordering::Greater => {
                greater -= 1;
                values.swap(scan, greater);
            }
        }
    }

    values.swap(greater, right);
    (less, greater)
}

fn median_of_three<T: Ord>(values: &[T], first: usize, middle: usize, last: usize) -> usize {
    if values[first] <= values[middle] {
        if values[middle] <= values[last] {
            middle
        } else if values[first] <= values[last] {
            last
        } else {
            first
        }
    } else if values[first] <= values[last] {
        first
    } else if values[middle] <= values[last] {
        last
    } else {
        middle
    }
}

#[cfg(test)]
mod tests {
    use super::{quickselect, top_k_by, top_k_smallest};

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct Candidate {
        id: usize,
        score: i8,
    }

    #[test]
    fn quickselect_matches_a_fully_sorted_oracle_for_every_index() {
        let input = [9, 1, 5, 3, 5, 8, 2, 7, 5, 4, 6, 0];
        let mut sorted = input;
        sorted.sort_unstable();

        for nth in 0..input.len() {
            let mut candidate = input;
            assert_eq!(quickselect(&mut candidate, nth), Some(&sorted[nth]));

            let pivot = candidate[nth];
            assert!(candidate[..nth].iter().all(|value| *value <= pivot));
            assert!(candidate[(nth + 1)..].iter().all(|value| *value >= pivot));
        }
    }

    #[test]
    fn quickselect_matches_sort_for_all_small_ternary_inputs() {
        for len in 1_usize..=6 {
            let cases = 3_usize.pow(len as u32);

            for case in 0..cases {
                let mut encoded = case;
                let mut input = Vec::with_capacity(len);
                for _ in 0..len {
                    input.push((encoded % 3) as i8 - 1);
                    encoded /= 3;
                }

                let mut sorted = input.clone();
                sorted.sort_unstable();

                for (nth, expected) in sorted.iter().enumerate() {
                    let mut candidate = input.clone();
                    assert_eq!(
                        quickselect(&mut candidate, nth),
                        Some(expected),
                        "len={len}, case={case}, nth={nth}, input={input:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn quickselect_handles_duplicate_only_input_without_special_cases() {
        let mut values = [4; 32];
        assert_eq!(quickselect(&mut values, 17), Some(&4));
    }

    #[test]
    fn quickselect_returns_none_for_an_out_of_range_index() {
        let mut values = [1, 2, 3];
        assert_eq!(quickselect(&mut values, 3), None);

        let mut empty: [i32; 0] = [];
        assert_eq!(quickselect(&mut empty, 0), None);
    }

    #[test]
    fn top_k_smallest_matches_sort_and_truncate_oracle() {
        let values = [9, 2, 5, 1, 7, 3, 8, 4, 6, 0, 5];
        for k in 0..=values.len() + 2 {
            let mut expected = values.to_vec();
            expected.sort_unstable();
            expected.truncate(k.min(values.len()));
            assert_eq!(top_k_smallest(&values, k), expected, "k={k}");
        }
    }

    #[test]
    fn ranked_top_k_supports_descending_search_scores_and_stable_ties() {
        let candidates = vec![
            Candidate { id: 0, score: 7 },
            Candidate { id: 1, score: 10 },
            Candidate { id: 2, score: 7 },
            Candidate { id: 3, score: 12 },
            Candidate { id: 4, score: 10 },
        ];

        let selected = top_k_by(candidates, 4, |left, right| right.score.cmp(&left.score));
        assert_eq!(
            selected,
            vec![
                Candidate { id: 3, score: 12 },
                Candidate { id: 1, score: 10 },
                Candidate { id: 4, score: 10 },
                Candidate { id: 0, score: 7 },
            ]
        );
    }

    #[test]
    fn ranked_top_k_matches_a_stable_full_sort_for_all_small_score_sequences() {
        for len in 0_usize..=6 {
            let cases = 3_usize.pow(len as u32);
            for case in 0..cases {
                let mut encoded = case;
                let mut input = Vec::with_capacity(len);
                for id in 0..len {
                    input.push(Candidate {
                        id,
                        score: (encoded % 3) as i8 - 1,
                    });
                    encoded /= 3;
                }

                for k in 0..=len + 1 {
                    let mut expected = input.clone();
                    expected.sort_by(|left, right| right.score.cmp(&left.score));
                    expected.truncate(k.min(len));

                    assert_eq!(
                        top_k_by(input.clone(), k, |left, right| right.score.cmp(&left.score)),
                        expected,
                        "len={len}, case={case}, k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn ranked_top_k_accepts_float_scores_through_total_cmp() {
        let selected = top_k_by(
            [("a", 0.5_f64), ("b", f64::NAN), ("c", 2.0), ("d", -1.0)],
            2,
            |left, right| right.1.total_cmp(&left.1),
        );

        assert_eq!(selected[0].0, "b");
        assert_eq!(selected[1], ("c", 2.0));
    }
}
