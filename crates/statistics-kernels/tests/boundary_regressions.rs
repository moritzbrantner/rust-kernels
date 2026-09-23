//! Finite-observation push must preserve a representable mean.
use statistics_kernels::RunningStats;

#[test]
fn boundary_push_opposite_extremes_preserves_mean() {
    let mut stats = RunningStats::new();
    stats.extend([-1e308, 1e308]);
    assert_eq!(stats.mean(), Some(0.0));
    assert_eq!(stats.population_variance(), Some(f64::INFINITY));
}

#[test]
fn boundary_push_remains_usable_after_extreme_pair() {
    let mut stats = RunningStats::new();
    stats.extend([-1e308, 1e308, 0.0]);
    assert_eq!(stats.mean(), Some(0.0));
    assert_eq!(stats.count(), 3);
    assert_eq!(stats.population_variance(), Some(f64::INFINITY));
}

#[test]
fn boundary_push_ordinary_control() {
    let mut stats = RunningStats::new();
    stats.extend([0.0, 2.0]);
    assert_eq!(stats.mean(), Some(1.0));
    assert_eq!(stats.population_variance(), Some(1.0));
}

#[test]
fn extreme_push_and_merge_agree_and_do_not_poison_later_queries() {
    for magnitude in [1e308, f64::MAX] {
        for sign in [-1.0, 1.0] {
            let mut single = RunningStats::new();
            single.extend([sign * magnitude, -sign * magnitude, 0.0]);
            let mut left = RunningStats::new();
            left.push(sign * magnitude);
            let mut right = RunningStats::new();
            right.push(-sign * magnitude);
            left.merge(&right);
            left.push(0.0);
            assert_eq!(left, single);
            single.push(magnitude);
            assert!(single.mean().unwrap().is_finite());
            assert!((single.mean().unwrap() / (magnitude / 4.0) - 1.0).abs() < 1e-14);
            assert_eq!(single.population_variance(), Some(f64::INFINITY));
            assert_eq!(single.sample_variance(), Some(f64::INFINITY));
        }
    }
}

#[test]
fn ordinary_push_results_keep_the_historical_evaluation_order() {
    for values in [
        vec![2.0, 4.0, 4.0, 5.0, 9.0],
        vec![1e12 + 1.0, 1e12 + 3.0],
        vec![-1e100, 1e100, 1.0],
    ] {
        let mut stats = RunningStats::new();
        let (mut mean, mut m2) = (0.0, 0.0);
        for (i, value) in values.into_iter().enumerate() {
            let delta = value - mean;
            mean += delta / (i + 1) as f64;
            let after = value - mean;
            m2 += delta * after;
            stats.push(value);
            assert_eq!(stats.mean().unwrap().to_bits(), mean.to_bits());
            assert_eq!(
                stats.population_variance().unwrap().to_bits(),
                (m2 / (i + 1) as f64).max(0.0).to_bits()
            );
        }
    }
}

#[test]
fn invalid_observations_leave_the_accumulator_unchanged() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut stats = RunningStats::new();
        stats.extend([2.0, 4.0]);
        let before = stats;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stats.push(value))).is_err()
        );
        assert_eq!(stats, before);
    }
}
