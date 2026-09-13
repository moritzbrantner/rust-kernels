use crate::{
    gjk::{GjkConfig, GjkResult, GjkStatus, gjk_intersection_with_config},
    math3::Vec3,
    support::{MinkowskiSupportPoint, SupportMap3, minkowski_support},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GjkTraceStep {
    /// One-based GJK iteration number.
    pub iteration: usize,
    /// Direction used for this iteration's Minkowski support query.
    pub query_direction: Vec3,
    /// Authoritative support point and its left/right witnesses.
    pub support: MinkowskiSupportPoint,
    /// Simplex retained by GJK after this iteration.
    pub simplex: [MinkowskiSupportPoint; 4],
    pub simplex_len: usize,
    /// Direction GJK would query next when the algorithm continues.
    pub next_search_direction: Vec3,
    /// Present only on the terminal step. Intermediate capped runs report
    /// `IterationLimit` internally, but that synthetic status is not exposed.
    pub terminal_status: Option<GjkStatus>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GjkTrace {
    pub result: GjkResult,
    pub steps: Vec<GjkTraceStep>,
}

/// Run GJK and materialize deterministic teaching/debug evidence.
///
/// The normal GJK path remains allocation-free. This trace intentionally reruns
/// deterministic prefixes with progressively larger iteration caps so every
/// retained simplex comes from the authoritative implementation instead of a
/// second tracing copy of the simplex state machine.
#[must_use]
pub fn gjk_intersection_trace<L, R>(left: &L, right: &R) -> GjkTrace
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    gjk_intersection_trace_with_config(left, right, GjkConfig::default())
}

/// Configurable tracing variant of [`gjk_intersection_trace`].
#[must_use]
pub fn gjk_intersection_trace_with_config<L, R>(left: &L, right: &R, config: GjkConfig) -> GjkTrace
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    let result = gjk_intersection_with_config(left, right, config);
    let mut steps = Vec::with_capacity(result.iterations);
    let mut query_direction = [1.0, 0.0, 0.0];

    for iteration in 1..=result.iterations {
        let support = minkowski_support(left, right, query_direction);
        let prefix = gjk_intersection_with_config(
            left,
            right,
            GjkConfig {
                max_iterations: iteration,
                epsilon: config.epsilon,
            },
        );
        let terminal_status = (iteration == result.iterations).then_some(result.status);

        steps.push(GjkTraceStep {
            iteration,
            query_direction,
            support,
            simplex: prefix.simplex,
            simplex_len: prefix.simplex_len,
            next_search_direction: prefix.search_direction,
            terminal_status,
        });

        query_direction = prefix.search_direction;
    }

    GjkTrace { result, steps }
}

#[cfg(test)]
mod tests {
    use super::{gjk_intersection_trace, gjk_intersection_trace_with_config};
    use crate::{
        Sphere,
        gjk::{GjkConfig, GjkStatus, gjk_intersection, gjk_intersection_with_config},
    };

    #[test]
    fn trace_result_matches_the_allocation_free_path() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let right = Sphere::new([1.25, 0.4, 0.0], 0.75);

        let trace = gjk_intersection_trace(&left, &right);
        assert_eq!(trace.result, gjk_intersection(&left, &right));
        assert_eq!(trace.steps.len(), trace.result.iterations);
        assert_eq!(
            trace.steps.last().and_then(|step| step.terminal_status),
            Some(trace.result.status)
        );
    }

    #[test]
    fn every_trace_step_is_an_authoritative_gjk_prefix() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let right = Sphere::new([1.2, 0.35, 0.0], 0.8);
        let config = GjkConfig {
            max_iterations: 16,
            epsilon: 1.0e-12,
        };
        let trace = gjk_intersection_trace_with_config(&left, &right, config);

        for step in &trace.steps {
            let prefix = gjk_intersection_with_config(
                &left,
                &right,
                GjkConfig {
                    max_iterations: step.iteration,
                    epsilon: config.epsilon,
                },
            );
            assert_eq!(step.simplex, prefix.simplex);
            assert_eq!(step.simplex_len, prefix.simplex_len);
            assert_eq!(step.next_search_direction, prefix.search_direction);
        }
    }

    #[test]
    fn separated_trace_keeps_the_rejected_support_query_visible() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let right = Sphere::new([3.0, 0.0, 0.0], 0.5);
        let trace = gjk_intersection_trace(&left, &right);
        let terminal = trace.steps.last().expect("GJK must emit a terminal step");

        assert_eq!(trace.result.status, GjkStatus::Separated);
        assert_eq!(terminal.terminal_status, Some(GjkStatus::Separated));
        assert!(terminal.simplex_len >= 1);
        assert!(terminal.support.point[0] < 0.0);
    }
}
