//! Regression cases also compiled against the exact pre-repair revision.
use search_kernels::top_k_by;

#[test]
fn boundary_empty_top_k_allows_unbounded_limit() {
    assert!(top_k_by(std::iter::empty::<u64>(), usize::MAX, Ord::cmp).is_empty());
}

#[test]
fn boundary_short_top_k_allows_unbounded_limit() {
    assert_eq!(top_k_by([3_u64, 1, 2], usize::MAX, Ord::cmp), [1, 2, 3]);
}

#[test]
fn boundary_top_k_ordinary_control() {
    assert_eq!(top_k_by([3_u64, 1, 2], 2, Ord::cmp), [1, 2]);
}

#[test]
fn unknown_length_stream_preserves_sort_and_tie_semantics_at_every_limit() {
    let input = [(0, 2), (1, 1), (2, 2), (3, -1), (4, 1)];
    for limit in [0, 1, 3, 5, 500, usize::MAX] {
        let mut cursor = 0;
        let stream = std::iter::from_fn(|| {
            let value = input.get(cursor).copied();
            cursor += 1;
            value
        });
        let actual = top_k_by(stream, limit, |a, b| a.1.cmp(&b.1));
        let mut expected = input.to_vec();
        expected.sort_by_key(|v| v.1);
        expected.truncate(limit.min(input.len()));
        assert_eq!(actual, expected);
    }
}

#[test]
fn zero_limit_never_polls_the_stream_or_comparator() {
    let stream = std::iter::from_fn(|| -> Option<u64> { panic!("polled a zero-limit stream") });
    assert!(top_k_by(stream, 0, |_, _| panic!("compared with zero limit")).is_empty());
}

#[test]
fn top_k_comparison_work_stays_bounded_as_input_grows() {
    use std::cell::Cell;
    for n in [128_usize, 1024, 8192] {
        let comparisons = Cell::new(0_usize);
        let result = top_k_by((0..n).rev(), 8, |a, b| {
            comparisons.set(comparisons.get() + 1);
            a.cmp(b)
        });
        assert_eq!(result, (0..8).collect::<Vec<_>>());
        assert!(
            comparisons.get() <= 8 * n,
            "n={n}, comparisons={}",
            comparisons.get()
        );
    }
}
