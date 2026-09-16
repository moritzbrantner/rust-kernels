use crate::{
    gjk::{GjkResult, GjkStatus},
    math3::{Vec3, sub},
    support::{MinkowskiSupportPoint, SupportMap3, minkowski_support},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EpaStatus {
    Converged,
    NotIntersecting,
    DegenerateSimplex,
    NoProgress,
    IterationLimit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpaConfig {
    pub max_iterations: usize,
    /// Maximum remaining support distance before the closest edge is accepted.
    pub tolerance: f64,
    /// Geometric tolerance used for duplicate points and degenerate edges.
    pub epsilon: f64,
}

impl Default for EpaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            tolerance: 1.0e-9,
            epsilon: 1.0e-12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpaPenetration {
    /// Minimum translation distance in the XY plane.
    pub depth: f64,
    /// Unit normal from the left shape toward the right shape.
    pub normal: Vec3,
    /// Closest Minkowski edge retained when EPA converged.
    pub edge: [MinkowskiSupportPoint; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpaResult {
    pub status: EpaStatus,
    pub iterations: usize,
    pub penetration: Option<EpaPenetration>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpaTraceStep {
    pub iteration: usize,
    pub edge: [MinkowskiSupportPoint; 2],
    pub normal: Vec3,
    pub edge_distance: f64,
    pub support: MinkowskiSupportPoint,
    pub support_distance: f64,
    pub gap: f64,
    /// Polytope after this iteration's insertion, or unchanged on termination.
    pub polytope: Vec<MinkowskiSupportPoint>,
    pub terminal_status: Option<EpaStatus>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpaTrace {
    pub result: EpaResult,
    pub steps: Vec<EpaTraceStep>,
}

/// Expand an intersecting planar GJK simplex until the closest Minkowski edge
/// converges to a penetration depth and normal.
#[must_use]
pub fn epa_penetration_planar_xy<L, R>(left: &L, right: &R, gjk: &GjkResult) -> EpaResult
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    epa_penetration_planar_xy_with_config(left, right, gjk, EpaConfig::default())
}

#[must_use]
pub fn epa_penetration_planar_xy_with_config<L, R>(
    left: &L,
    right: &R,
    gjk: &GjkResult,
    config: EpaConfig,
) -> EpaResult
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    run_epa(left, right, gjk, config, None)
}

/// Trace-quality EPA evidence. The normal path above does not retain polytope
/// snapshots or per-iteration evidence.
#[must_use]
pub fn epa_penetration_trace_planar_xy<L, R>(left: &L, right: &R, gjk: &GjkResult) -> EpaTrace
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    epa_penetration_trace_planar_xy_with_config(left, right, gjk, EpaConfig::default())
}

#[must_use]
pub fn epa_penetration_trace_planar_xy_with_config<L, R>(
    left: &L,
    right: &R,
    gjk: &GjkResult,
    config: EpaConfig,
) -> EpaTrace
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    let mut steps = Vec::new();
    let result = run_epa(left, right, gjk, config, Some(&mut steps));
    EpaTrace { result, steps }
}

fn run_epa<L, R>(
    left: &L,
    right: &R,
    gjk: &GjkResult,
    config: EpaConfig,
    mut trace: Option<&mut Vec<EpaTraceStep>>,
) -> EpaResult
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    validate_config(config);
    if gjk.status != GjkStatus::Intersecting {
        return EpaResult {
            status: EpaStatus::NotIntersecting,
            iterations: 0,
            penetration: None,
        };
    }

    let epsilon_squared = config.epsilon * config.epsilon;
    let mut polytope = initial_polytope(left, right, gjk, config.epsilon);
    if polytope.len() < 3 || !contains_origin_xy(&polytope, config.epsilon) {
        return EpaResult {
            status: EpaStatus::DegenerateSimplex,
            iterations: 0,
            penetration: None,
        };
    }

    for iteration in 1..=config.max_iterations {
        let Some(edge) = closest_edge(&polytope, config.epsilon) else {
            return EpaResult {
                status: EpaStatus::DegenerateSimplex,
                iterations: iteration - 1,
                penetration: None,
            };
        };
        let edge_points = [polytope[edge.start], polytope[edge.end]];
        let support = minkowski_support(left, right, edge.normal);
        let support_distance = dot_xy(edge.normal, support.point);
        let gap = support_distance - edge.distance;

        if gap <= config.tolerance {
            let penetration = EpaPenetration {
                depth: edge.distance.max(0.0),
                normal: edge.normal,
                edge: edge_points,
            };
            push_trace(
                &mut trace,
                EpaTraceStep {
                    iteration,
                    edge: edge_points,
                    normal: edge.normal,
                    edge_distance: edge.distance,
                    support,
                    support_distance,
                    gap,
                    polytope: polytope.clone(),
                    terminal_status: Some(EpaStatus::Converged),
                },
            );
            return EpaResult {
                status: EpaStatus::Converged,
                iterations: iteration,
                penetration: Some(penetration),
            };
        }

        if polytope
            .iter()
            .any(|point| distance_squared_xy(point.point, support.point) <= epsilon_squared)
        {
            push_trace(
                &mut trace,
                EpaTraceStep {
                    iteration,
                    edge: edge_points,
                    normal: edge.normal,
                    edge_distance: edge.distance,
                    support,
                    support_distance,
                    gap,
                    polytope: polytope.clone(),
                    terminal_status: Some(EpaStatus::NoProgress),
                },
            );
            return EpaResult {
                status: EpaStatus::NoProgress,
                iterations: iteration,
                penetration: None,
            };
        }

        polytope.insert(edge.end, support);
        push_trace(
            &mut trace,
            EpaTraceStep {
                iteration,
                edge: edge_points,
                normal: edge.normal,
                edge_distance: edge.distance,
                support,
                support_distance,
                gap,
                polytope: polytope.clone(),
                terminal_status: None,
            },
        );
    }

    EpaResult {
        status: EpaStatus::IterationLimit,
        iterations: config.max_iterations,
        penetration: None,
    }
}

fn initial_polytope<L, R>(
    left: &L,
    right: &R,
    gjk: &GjkResult,
    epsilon: f64,
) -> Vec<MinkowskiSupportPoint>
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    let mut seeds = gjk.simplex[..gjk.simplex_len].to_vec();
    let mut hull = convex_hull_xy(&seeds, epsilon);
    if hull.len() >= 3 && contains_origin_xy(&hull, epsilon) {
        return hull;
    }

    if hull.len() == 2 {
        let delta = sub(hull[1].point, hull[0].point);
        let perpendicular = [-delta[1], delta[0], 0.0];
        let length_squared =
            perpendicular[0] * perpendicular[0] + perpendicular[1] * perpendicular[1];
        if length_squared > epsilon * epsilon {
            seeds.push(minkowski_support(left, right, perpendicular));
            seeds.push(minkowski_support(
                left,
                right,
                [-perpendicular[0], -perpendicular[1], 0.0],
            ));
            hull = convex_hull_xy(&seeds, epsilon);
            if hull.len() >= 3 && contains_origin_xy(&hull, epsilon) {
                return hull;
            }
        }
    }

    for direction in [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ] {
        seeds.push(minkowski_support(left, right, direction));
    }
    convex_hull_xy(&seeds, epsilon)
}

fn validate_config(config: EpaConfig) {
    assert!(
        config.max_iterations > 0,
        "EPA max_iterations must be positive"
    );
    assert!(
        config.tolerance.is_finite() && config.tolerance >= 0.0,
        "EPA tolerance must be non-negative and finite"
    );
    assert!(
        config.epsilon.is_finite() && config.epsilon >= 0.0,
        "EPA epsilon must be non-negative and finite"
    );
}

fn push_trace(trace: &mut Option<&mut Vec<EpaTraceStep>>, step: EpaTraceStep) {
    if let Some(steps) = trace.as_deref_mut() {
        steps.push(step);
    }
}

#[derive(Clone, Copy, Debug)]
struct ClosestEdge {
    start: usize,
    end: usize,
    normal: Vec3,
    distance: f64,
}

fn closest_edge(polytope: &[MinkowskiSupportPoint], epsilon: f64) -> Option<ClosestEdge> {
    let mut best: Option<ClosestEdge> = None;
    for start in 0..polytope.len() {
        let end = (start + 1) % polytope.len();
        let a = polytope[start].point;
        let b = polytope[end].point;
        let edge = [b[0] - a[0], b[1] - a[1], 0.0];
        let length_squared = edge[0] * edge[0] + edge[1] * edge[1];
        if length_squared <= epsilon * epsilon {
            continue;
        }
        let inverse_length = length_squared.sqrt().recip();
        let normal = [edge[1] * inverse_length, -edge[0] * inverse_length, 0.0];
        let distance = dot_xy(normal, a);
        if distance < -epsilon {
            return None;
        }
        let candidate = ClosestEdge {
            start,
            end,
            normal,
            distance: distance.max(0.0),
        };
        if best.is_none_or(|current| candidate.distance < current.distance) {
            best = Some(candidate);
        }
    }
    best
}

fn convex_hull_xy(points: &[MinkowskiSupportPoint], epsilon: f64) -> Vec<MinkowskiSupportPoint> {
    let epsilon_squared = epsilon * epsilon;
    let mut unique = Vec::new();
    for &point in points {
        if unique.iter().all(|existing: &MinkowskiSupportPoint| {
            distance_squared_xy(existing.point, point.point) > epsilon_squared
        }) {
            unique.push(point);
        }
    }
    unique.sort_by(|left, right| {
        left.point[0]
            .total_cmp(&right.point[0])
            .then_with(|| left.point[1].total_cmp(&right.point[1]))
    });
    if unique.len() <= 2 {
        return unique;
    }

    let mut lower: Vec<MinkowskiSupportPoint> = Vec::new();
    for &point in &unique {
        while lower.len() >= 2
            && cross_xy(
                lower[lower.len() - 2].point,
                lower[lower.len() - 1].point,
                point.point,
            ) <= epsilon
        {
            lower.pop();
        }
        lower.push(point);
    }

    let mut upper: Vec<MinkowskiSupportPoint> = Vec::new();
    for &point in unique.iter().rev() {
        while upper.len() >= 2
            && cross_xy(
                upper[upper.len() - 2].point,
                upper[upper.len() - 1].point,
                point.point,
            ) <= epsilon
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn contains_origin_xy(polytope: &[MinkowskiSupportPoint], epsilon: f64) -> bool {
    (0..polytope.len()).all(|start| {
        let end = (start + 1) % polytope.len();
        let a = polytope[start].point;
        let b = polytope[end].point;
        let edge = [b[0] - a[0], b[1] - a[1], 0.0];
        let to_origin = [-a[0], -a[1], 0.0];
        edge[0] * to_origin[1] - edge[1] * to_origin[0] >= -epsilon
    })
}

fn cross_xy(origin: Vec3, left: Vec3, right: Vec3) -> f64 {
    (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])
}

fn dot_xy(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1]
}

fn distance_squared_xy(left: Vec3, right: Vec3) -> f64 {
    let delta = sub(left, right);
    delta[0] * delta[0] + delta[1] * delta[1]
}

#[cfg(test)]
mod tests {
    use super::{EpaStatus, epa_penetration_planar_xy, epa_penetration_trace_planar_xy};
    use crate::{gjk::GjkStatus, planar::gjk_intersection_planar_xy, support::ConvexHull3};

    fn square(center_x: f64, center_y: f64, half: f64) -> [[f64; 3]; 4] {
        [
            [center_x - half, center_y - half, 0.0],
            [center_x + half, center_y - half, 0.0],
            [center_x + half, center_y + half, 0.0],
            [center_x - half, center_y + half, 0.0],
        ]
    }

    #[test]
    fn overlapping_squares_bootstrap_a_line_simplex_and_report_penetration() {
        let left_points = square(0.0, 0.0, 1.0);
        let right_points = square(1.5, 0.0, 1.0);
        let left = ConvexHull3::new(&left_points);
        let right = ConvexHull3::new(&right_points);
        let gjk = gjk_intersection_planar_xy(&left, &right);
        assert_eq!(gjk.status, GjkStatus::Intersecting);
        assert_eq!(gjk.simplex_len, 2);

        let result = epa_penetration_planar_xy(&left, &right, &gjk);
        let penetration = result.penetration.expect("overlap must converge");

        assert_eq!(result.status, EpaStatus::Converged);
        assert!((penetration.depth - 0.5).abs() <= 1.0e-9);
        assert!((penetration.normal[0] - 1.0).abs() <= 1.0e-9);
        assert!(penetration.normal[1].abs() <= 1.0e-9);
        assert_eq!(penetration.normal[2], 0.0);
    }

    #[test]
    fn swapping_shapes_flips_normal_and_preserves_depth() {
        let left_points = square(0.0, 0.0, 1.0);
        let right_points = square(1.5, 0.0, 1.0);
        let left = ConvexHull3::new(&left_points);
        let right = ConvexHull3::new(&right_points);
        let forward_gjk = gjk_intersection_planar_xy(&left, &right);
        let reverse_gjk = gjk_intersection_planar_xy(&right, &left);
        let forward = epa_penetration_planar_xy(&left, &right, &forward_gjk)
            .penetration
            .expect("forward overlap must converge");
        let reverse = epa_penetration_planar_xy(&right, &left, &reverse_gjk)
            .penetration
            .expect("reverse overlap must converge");

        assert!((forward.depth - reverse.depth).abs() <= 1.0e-9);
        assert!((forward.normal[0] + reverse.normal[0]).abs() <= 1.0e-9);
        assert!((forward.normal[1] + reverse.normal[1]).abs() <= 1.0e-9);
    }

    #[test]
    fn separated_shapes_fail_closed_before_expansion() {
        let left_points = square(0.0, 0.0, 1.0);
        let right_points = square(4.0, 0.0, 1.0);
        let left = ConvexHull3::new(&left_points);
        let right = ConvexHull3::new(&right_points);
        let gjk = gjk_intersection_planar_xy(&left, &right);
        let result = epa_penetration_planar_xy(&left, &right, &gjk);

        assert_eq!(result.status, EpaStatus::NotIntersecting);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.penetration, None);
    }

    #[test]
    fn trace_is_deterministic_and_matches_normal_result() {
        let left_points = [
            [-1.2, -0.7, 0.0],
            [1.0, -0.9, 0.0],
            [1.25, 0.6, 0.0],
            [-0.8, 1.1, 0.0],
        ];
        let right_points = [
            [0.1, -0.8, 0.0],
            [1.6, -0.3, 0.0],
            [1.2, 1.0, 0.0],
            [-0.1, 0.7, 0.0],
        ];
        let left = ConvexHull3::new(&left_points);
        let right = ConvexHull3::new(&right_points);
        let gjk = gjk_intersection_planar_xy(&left, &right);
        let expected = epa_penetration_planar_xy(&left, &right, &gjk);
        let first = epa_penetration_trace_planar_xy(&left, &right, &gjk);
        let second = epa_penetration_trace_planar_xy(&left, &right, &gjk);

        assert_eq!(first, second);
        assert_eq!(first.result, expected);
        assert_eq!(
            first.steps.last().and_then(|step| step.terminal_status),
            Some(expected.status)
        );
    }
}
