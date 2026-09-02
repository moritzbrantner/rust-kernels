/// Numerically stable streaming mean and variance using Welford updates.
///
/// Observations must be finite. Two accumulators can be merged without
/// replaying their samples using the parallel-variance formula.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    #[must_use]
    pub const fn count(self) -> u64 {
        self.count
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.count == 0
    }

    pub fn push(&mut self, value: f64) {
        assert!(
            value.is_finite(),
            "running statistics require finite values"
        );
        let new_count = self
            .count
            .checked_add(1)
            .expect("running statistics count overflow");
        let delta = value - self.mean;
        self.mean += delta / new_count as f64;
        let delta_after = value - self.mean;
        self.m2 += delta * delta_after;
        self.count = new_count;
    }

    pub fn extend(&mut self, values: impl IntoIterator<Item = f64>) {
        for value in values {
            self.push(value);
        }
    }

    /// Merges another accumulator without replaying observations.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }

        let combined = self
            .count
            .checked_add(other.count)
            .expect("running statistics count overflow");
        let left_count = self.count as f64;
        let right_count = other.count as f64;
        let combined_count = combined as f64;
        let delta = other.mean - self.mean;

        self.mean += delta * right_count / combined_count;
        self.m2 += other.m2 + delta * delta * left_count * right_count / combined_count;
        self.count = combined;
    }

    #[must_use]
    pub fn mean(self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    #[must_use]
    pub fn population_variance(self) -> Option<f64> {
        (self.count > 0).then(|| (self.m2 / self.count as f64).max(0.0))
    }

    #[must_use]
    pub fn sample_variance(self) -> Option<f64> {
        (self.count > 1).then(|| (self.m2 / (self.count - 1) as f64).max(0.0))
    }

    #[must_use]
    pub fn population_stddev(self) -> Option<f64> {
        self.population_variance().map(f64::sqrt)
    }

    #[must_use]
    pub fn sample_stddev(self) -> Option<f64> {
        self.sample_variance().map(f64::sqrt)
    }
}

#[cfg(test)]
mod tests {
    use super::RunningStats;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn batch(values: &[f64]) -> (f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let m2 = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>();
        (
            mean,
            m2 / values.len() as f64,
            m2 / (values.len() - 1) as f64,
        )
    }

    #[test]
    fn canonical_fixture_matches_known_statistics() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut stats = RunningStats::new();
        stats.extend(values);

        assert_eq!(stats.count(), 8);
        assert_close(stats.mean().expect("mean"), 5.0, 1e-12);
        assert_close(stats.population_variance().expect("variance"), 4.0, 1e-12);
        assert_close(
            stats.sample_variance().expect("variance"),
            32.0 / 7.0,
            1e-12,
        );
        assert_close(stats.population_stddev().expect("stddev"), 2.0, 1e-12);
    }

    #[test]
    fn merged_chunks_match_single_pass_and_batch_oracles() {
        let values = [-8.0, -3.5, 0.0, 1.0, 1.5, 2.0, 5.25, 11.0, 20.0, 21.0, 50.0];
        let mut whole = RunningStats::new();
        whole.extend(values);

        let mut left = RunningStats::new();
        left.extend(values[..4].iter().copied());
        let mut middle = RunningStats::new();
        middle.extend(values[4..8].iter().copied());
        let mut right = RunningStats::new();
        right.extend(values[8..].iter().copied());
        left.merge(&middle);
        left.merge(&right);

        let (mean, population, sample) = batch(&values);
        assert_eq!(left.count(), whole.count());
        assert_close(
            left.mean().expect("mean"),
            whole.mean().expect("mean"),
            1e-12,
        );
        assert_close(
            left.population_variance().expect("variance"),
            whole.population_variance().expect("variance"),
            1e-12,
        );
        assert_close(whole.mean().expect("mean"), mean, 1e-12);
        assert_close(
            whole.population_variance().expect("variance"),
            population,
            1e-12,
        );
        assert_close(whole.sample_variance().expect("variance"), sample, 1e-12);
    }

    #[test]
    fn remains_accurate_for_large_offsets() {
        let values = [1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0, 1e12 + 4.0];
        let mut stats = RunningStats::new();
        stats.extend(values);
        assert_close(stats.mean().expect("mean"), 1e12 + 2.5, 1e-6);
        assert_close(stats.population_variance().expect("variance"), 1.25, 1e-9);
    }

    #[test]
    fn empty_and_singleton_semantics_are_explicit() {
        let mut stats = RunningStats::new();
        assert!(stats.is_empty());
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.population_variance(), None);
        assert_eq!(stats.sample_variance(), None);

        stats.push(42.0);
        assert_eq!(stats.mean(), Some(42.0));
        assert_eq!(stats.population_variance(), Some(0.0));
        assert_eq!(stats.sample_variance(), None);
    }

    #[test]
    #[should_panic(expected = "running statistics require finite values")]
    fn rejects_non_finite_observations() {
        RunningStats::new().push(f64::NAN);
    }
}
