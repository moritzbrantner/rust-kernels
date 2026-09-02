use core::cmp::Ordering;

/// Returns the union of two sorted-unique slices.
///
/// Inputs must be strictly increasing according to `Ord`. The result preserves
/// sorted-unique order and is produced in `O(a.len() + b.len())` time.
pub fn sorted_union<T: Ord + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    merge_collect(a, b, MergeMode::Union)
}

/// Returns the intersection of two sorted-unique slices.
///
/// Inputs must be strictly increasing according to `Ord`. The result preserves
/// sorted-unique order and is produced in `O(a.len() + b.len())` time.
pub fn sorted_intersection<T: Ord + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    merge_collect(a, b, MergeMode::Intersection)
}

/// Returns values present in `a` but not `b` for sorted-unique inputs.
pub fn sorted_difference<T: Ord + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    merge_collect(a, b, MergeMode::Difference)
}

/// Returns values present in exactly one input for sorted-unique inputs.
pub fn sorted_symmetric_difference<T: Ord + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    merge_collect(a, b, MergeMode::SymmetricDifference)
}

/// Counts the intersection of two sorted-unique slices without allocating.
pub fn sorted_intersection_count<T: Ord>(a: &[T], b: &[T]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut count = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }

    count
}

/// Counts the union of two sorted-unique slices without allocating.
pub fn sorted_union_count<T: Ord>(a: &[T], b: &[T]) -> usize {
    a.len() + b.len() - sorted_intersection_count(a, b)
}

#[derive(Clone, Copy)]
enum MergeMode {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}

fn merge_collect<T: Ord + Clone>(a: &[T], b: &[T], mode: MergeMode) -> Vec<T> {
    let mut result = Vec::with_capacity(match mode {
        MergeMode::Union | MergeMode::SymmetricDifference => a.len() + b.len(),
        MergeMode::Intersection => a.len().min(b.len()),
        MergeMode::Difference => a.len(),
    });
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                if matches!(
                    mode,
                    MergeMode::Union | MergeMode::Difference | MergeMode::SymmetricDifference
                ) {
                    result.push(a[i].clone());
                }
                i += 1;
            }
            Ordering::Greater => {
                if matches!(mode, MergeMode::Union | MergeMode::SymmetricDifference) {
                    result.push(b[j].clone());
                }
                j += 1;
            }
            Ordering::Equal => {
                if matches!(mode, MergeMode::Union | MergeMode::Intersection) {
                    result.push(a[i].clone());
                }
                i += 1;
                j += 1;
            }
        }
    }

    if matches!(
        mode,
        MergeMode::Union | MergeMode::Difference | MergeMode::SymmetricDifference
    ) {
        result.extend_from_slice(&a[i..]);
    }
    if matches!(mode, MergeMode::Union | MergeMode::SymmetricDifference) {
        result.extend_from_slice(&b[j..]);
    }

    result
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        sorted_difference, sorted_intersection, sorted_intersection_count,
        sorted_symmetric_difference, sorted_union, sorted_union_count,
    };

    fn as_set(values: &[i32]) -> BTreeSet<i32> {
        values.iter().copied().collect()
    }

    #[test]
    fn handles_empty_and_disjoint_inputs() {
        assert_eq!(sorted_union::<i32>(&[], &[]), vec![]);
        assert_eq!(sorted_intersection(&[1, 3], &[2, 4]), vec![]);
        assert_eq!(sorted_difference(&[1, 3], &[2, 4]), vec![1, 3]);
        assert_eq!(
            sorted_symmetric_difference(&[1, 3], &[2, 4]),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn computes_expected_operations() {
        let a = [1, 2, 4, 8];
        let b = [2, 3, 4, 9];
        assert_eq!(sorted_union(&a, &b), vec![1, 2, 3, 4, 8, 9]);
        assert_eq!(sorted_intersection(&a, &b), vec![2, 4]);
        assert_eq!(sorted_difference(&a, &b), vec![1, 8]);
        assert_eq!(sorted_symmetric_difference(&a, &b), vec![1, 3, 8, 9]);
        assert_eq!(sorted_intersection_count(&a, &b), 2);
        assert_eq!(sorted_union_count(&a, &b), 6);
    }

    #[test]
    fn matches_btree_set_oracles_on_generated_inputs() {
        for mask_a in 0u16..128 {
            for mask_b in 0u16..128 {
                let a: Vec<i32> = (0..7)
                    .filter(|bit| mask_a & (1 << bit) != 0)
                    .collect();
                let b: Vec<i32> = (0..7)
                    .filter(|bit| mask_b & (1 << bit) != 0)
                    .collect();
                let a_set = as_set(&a);
                let b_set = as_set(&b);

                let union: Vec<_> = a_set.union(&b_set).copied().collect();
                let intersection: Vec<_> = a_set.intersection(&b_set).copied().collect();
                let difference: Vec<_> = a_set.difference(&b_set).copied().collect();
                let symmetric_difference: Vec<_> =
                    a_set.symmetric_difference(&b_set).copied().collect();

                assert_eq!(sorted_union(&a, &b), union);
                assert_eq!(sorted_intersection(&a, &b), intersection);
                assert_eq!(sorted_difference(&a, &b), difference);
                assert_eq!(
                    sorted_symmetric_difference(&a, &b),
                    symmetric_difference
                );
                assert_eq!(sorted_intersection_count(&a, &b), intersection.len());
                assert_eq!(sorted_union_count(&a, &b), union.len());
            }
        }
    }
}
