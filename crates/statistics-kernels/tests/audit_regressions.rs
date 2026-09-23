//! Native regressions for the numerical audit at ddec70d.
use statistics_kernels::RunningStats;

#[test]
fn audit_merge_preserves_finite_representable_variance() {
    let mut left = RunningStats::new();
    left.extend([0.0, 0.0]);
    let mut right = RunningStats::new();
    right.extend([1e154, 1e154]);
    let mut single_pass = RunningStats::new();
    single_pass.extend([0.0, 0.0, 1e154, 1e154]);
    left.merge(&right);

    let actual = left.population_variance().unwrap();
    let expected = single_pass.population_variance().unwrap();
    assert!(expected.is_finite());
    assert!(
        actual.is_finite(),
        "merged={actual}, single_pass={expected}"
    );
    assert!((actual / expected - 1.0).abs() < 1e-14);
}

#[test]
fn audit_control_ordinary_merge_matches_single_pass() {
    let mut left = RunningStats::new();
    left.extend([0.0, 0.0]);
    let mut right = RunningStats::new();
    right.extend([2.0, 2.0]);
    left.merge(&right);
    assert_eq!(left.mean(), Some(1.0));
    assert_eq!(left.population_variance(), Some(1.0));
}

fn stats(values: impl IntoIterator<Item = f64>) -> RunningStats {
    let mut stats = RunningStats::new();
    stats.extend(values);
    stats
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "actual={actual}, expected={expected}");
    assert!(
        (actual / expected - 1.0).abs() <= 1e-13,
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn singleton_merge_scales_before_squaring() {
    // delta² overflows, but delta²/2 and the population variance do not.
    let mut left = stats([0.0]);
    left.merge(&stats([1.5e154]));
    close(left.mean().unwrap(), 7.5e153);
    close(left.population_variance().unwrap(), 5.625e307);
}

#[test]
fn merged_mean_of_opposite_finite_extremes_is_finite() {
    let mut left = stats([-1e308]);
    left.merge(&stats([1e308]));
    assert_eq!(left.mean(), Some(0.0));
    // The variance itself cannot be represented: infinity is correct, not zero.
    assert_eq!(left.population_variance(), Some(f64::INFINITY));
}

#[test]
fn unbalanced_merge_avoids_weighted_mean_overflow() {
    let mut right = stats([1e307]);
    for _ in 0..40 {
        right.merge(&{ right });
    }
    let mut left = stats([0.0]);
    left.merge(&right);
    close(
        left.mean().unwrap(),
        1e307 * (1.0 - 1.0 / ((1_u64 << 40) + 1) as f64),
    );
    assert_eq!(left.count(), (1_u64 << 40) + 1);
}

#[test]
fn merged_chunks_remain_equivalent_for_large_and_small_finite_scales() {
    for magnitude in [1e-150, 1e-30, 1.0, 1e30, 1e153] {
        let values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0].map(|x| x * magnitude);
        let whole = stats(values);
        for split in 1..values.len() {
            let mut left = stats(values[..split].iter().copied());
            let right = stats(values[split..].iter().copied());
            let mut reverse = right;
            reverse.merge(&left);
            left.merge(&right);
            close(left.mean().unwrap(), whole.mean().unwrap());
            close(
                left.population_variance().unwrap(),
                whole.population_variance().unwrap(),
            );
            close(
                reverse.population_variance().unwrap(),
                whole.population_variance().unwrap(),
            );
            assert_eq!(left.count(), values.len() as u64);
        }
    }
}
