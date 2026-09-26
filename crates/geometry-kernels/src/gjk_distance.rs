use crate::{
    gjk::{GjkConfig, GjkResult, GjkStatus, gjk_intersection_with_config},
    math3::{Vec3, add, dot, length_squared, neg, scale, sub},
    support::{MinkowskiSupportPoint, SupportMap3, minkowski_support},
};

const BARYCENTRIC_EPSILON: f64 = 32.0 * f64::EPSILON;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GjkDistanceStatus {
    Converged,
    Intersecting,
    NoProgress,
    IterationLimit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GjkDistanceConfig {
    pub max_iterations: usize,
    /// Relative tolerance for improvement in squared Minkowski distance.
    pub relative_tolerance: f64,
    /// World-space tolerance for touching and repeated support points.
    pub epsilon: f64,
}

impl Default for GjkDistanceConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            relative_tolerance: 1.0e-12,
            epsilon: 1.0e-12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GjkClosestPoints {
    pub distance: f64,
    /// Unit direction from the left witness toward the right witness.
    pub normal: Vec3,
    pub point_left: Vec3,
    pub point_right: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GjkDistanceResult {
    pub status: GjkDistanceStatus,
    /// Total intersection-GJK plus distance-refinement iterations.
    pub iterations: usize,
    /// Initial intersection query retained for penetration consumers.
    pub intersection: GjkResult,
    pub closest: Option<GjkClosestPoints>,
}

#[derive(Clone, Copy, Debug)]
struct Projection {
    point: Vec3,
    weights: [f64; 4],
    squared_distance: f64,
}

#[must_use]
pub fn gjk_distance<L, R>(left: &L, right: &R) -> GjkDistanceResult
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    gjk_distance_with_config(left, right, GjkDistanceConfig::default())
}

#[must_use]
pub fn gjk_distance_with_config<L, R>(
    left: &L,
    right: &R,
    config: GjkDistanceConfig,
) -> GjkDistanceResult
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    assert!(
        config.max_iterations > 0,
        "GJK distance max_iterations must be positive"
    );
    assert!(
        config.relative_tolerance.is_finite() && config.relative_tolerance >= 0.0,
        "GJK distance relative_tolerance must be non-negative and finite"
    );
    assert!(
        config.epsilon.is_finite() && config.epsilon >= 0.0,
        "GJK distance epsilon must be non-negative and finite"
    );

    let intersection = gjk_intersection_with_config(
        left,
        right,
        GjkConfig {
            max_iterations: config.max_iterations,
            epsilon: config.epsilon,
        },
    );
    match intersection.status {
        GjkStatus::Intersecting => {
            return GjkDistanceResult {
                status: GjkDistanceStatus::Intersecting,
                iterations: intersection.iterations,
                intersection,
                closest: None,
            };
        }
        GjkStatus::Separated | GjkStatus::NoProgress | GjkStatus::IterationLimit => {}
    }

    let mut simplex = intersection.simplex;
    let mut simplex_len = intersection.simplex_len;
    let epsilon_squared = config.epsilon * config.epsilon;

    for refinement in 1..=config.max_iterations {
        let projection = closest_projection(&simplex[..simplex_len], config.epsilon);
        if projection.squared_distance <= epsilon_squared {
            return GjkDistanceResult {
                status: GjkDistanceStatus::NoProgress,
                iterations: intersection.iterations + refinement - 1,
                intersection,
                closest: None,
            };
        }

        let closest = witnesses(&simplex[..simplex_len], projection);
        let direction = neg(projection.point);
        let support = minkowski_support(left, right, direction);
        let projection_on_support = dot(projection.point, support.point);
        let improvement = projection.squared_distance - projection_on_support;
        let scale =
            (projection.squared_distance + projection_on_support.abs()).max(epsilon_squared);
        if improvement <= config.relative_tolerance * scale + epsilon_squared {
            return GjkDistanceResult {
                status: GjkDistanceStatus::Converged,
                iterations: intersection.iterations + refinement,
                intersection,
                closest: Some(closest),
            };
        }

        compact_simplex(&mut simplex, &mut simplex_len, projection.weights);
        if simplex[..simplex_len]
            .iter()
            .any(|existing| length_squared(sub(existing.point, support.point)) <= epsilon_squared)
        {
            return GjkDistanceResult {
                status: GjkDistanceStatus::NoProgress,
                iterations: intersection.iterations + refinement,
                intersection,
                closest: Some(closest),
            };
        }

        debug_assert!(simplex_len < 4);
        simplex[simplex_len] = support;
        simplex_len += 1;
    }

    let projection = closest_projection(&simplex[..simplex_len], config.epsilon);
    GjkDistanceResult {
        status: GjkDistanceStatus::IterationLimit,
        iterations: intersection.iterations + config.max_iterations,
        intersection,
        closest: (projection.squared_distance > epsilon_squared)
            .then(|| witnesses(&simplex[..simplex_len], projection)),
    }
}

fn witnesses(simplex: &[MinkowskiSupportPoint], projection: Projection) -> GjkClosestPoints {
    let mut point_left = [0.0; 3];
    let mut point_right = [0.0; 3];
    for (index, support) in simplex.iter().enumerate() {
        point_left = add(point_left, scale(support.left, projection.weights[index]));
        point_right = add(point_right, scale(support.right, projection.weights[index]));
    }
    let distance = projection.squared_distance.sqrt();
    GjkClosestPoints {
        distance,
        normal: scale(neg(projection.point), distance.recip()),
        point_left,
        point_right,
    }
}

fn compact_simplex(
    simplex: &mut [MinkowskiSupportPoint; 4],
    simplex_len: &mut usize,
    weights: [f64; 4],
) {
    let old = *simplex;
    let mut len = 0;
    for index in 0..*simplex_len {
        if weights[index] > BARYCENTRIC_EPSILON {
            simplex[len] = old[index];
            len += 1;
        }
    }
    if len == 0 {
        let mut best = 0;
        for index in 1..*simplex_len {
            if weights[index] > weights[best] {
                best = index;
            }
        }
        simplex[0] = old[best];
        len = 1;
    }
    *simplex_len = len;
}

fn closest_projection(simplex: &[MinkowskiSupportPoint], epsilon: f64) -> Projection {
    debug_assert!(!simplex.is_empty() && simplex.len() <= 4);
    let mut best: Option<(usize, Projection)> = None;
    for mask in 1..(1_usize << simplex.len()) {
        if mask.count_ones() > 3 {
            // Intersecting tetrahedra have already been handled by the initial GJK query.
            continue;
        }
        let Some(projection) = project_subset(simplex, mask, epsilon) else {
            continue;
        };
        let replace = best.is_none_or(|(best_mask, current)| {
            projection.squared_distance < current.squared_distance
                || (projection.squared_distance == current.squared_distance && mask < best_mask)
        });
        if replace {
            best = Some((mask, projection));
        }
    }
    best.map(|(_, projection)| projection)
        .unwrap_or_else(|| unreachable!("every non-empty simplex has a vertex projection"))
}

fn project_subset(
    simplex: &[MinkowskiSupportPoint],
    mask: usize,
    epsilon: f64,
) -> Option<Projection> {
    let mut indices = [0_usize; 3];
    let mut len = 0;
    for index in 0..simplex.len() {
        if mask & (1 << index) != 0 {
            indices[len] = index;
            len += 1;
        }
    }

    let mut weights = [0.0; 4];
    match len {
        1 => weights[indices[0]] = 1.0,
        2 => {
            let a = simplex[indices[0]].point;
            let b = simplex[indices[1]].point;
            let ab = sub(b, a);
            let denominator = length_squared(ab);
            if denominator <= epsilon * epsilon {
                return None;
            }
            let parameter = -dot(a, ab) / denominator;
            if !(-BARYCENTRIC_EPSILON..=1.0 + BARYCENTRIC_EPSILON).contains(&parameter) {
                return None;
            }
            let parameter = parameter.clamp(0.0, 1.0);
            weights[indices[0]] = 1.0 - parameter;
            weights[indices[1]] = parameter;
        }
        3 => {
            let a = simplex[indices[0]].point;
            let b = simplex[indices[1]].point;
            let c = simplex[indices[2]].point;
            let ab = sub(b, a);
            let ac = sub(c, a);
            let aa = dot(ab, ab);
            let ab_ac = dot(ab, ac);
            let cc = dot(ac, ac);
            let rhs_ab = -dot(a, ab);
            let rhs_ac = -dot(a, ac);
            let determinant = aa * cc - ab_ac * ab_ac;
            let scale_squared = aa.max(cc);
            if determinant <= epsilon * epsilon * scale_squared {
                return None;
            }
            let second = (rhs_ab * cc - rhs_ac * ab_ac) / determinant;
            let third = (aa * rhs_ac - ab_ac * rhs_ab) / determinant;
            let first = 1.0 - second - third;
            if [first, second, third]
                .into_iter()
                .any(|weight| weight < -BARYCENTRIC_EPSILON)
            {
                return None;
            }
            let values = normalize_weights([first, second, third]);
            for (index, weight) in indices.into_iter().zip(values) {
                weights[index] = weight;
            }
        }
        _ => unreachable!("distance projection subsets contain at most three vertices"),
    }

    let mut point = [0.0; 3];
    for (index, support) in simplex.iter().enumerate() {
        point = add(point, scale(support.point, weights[index]));
    }
    Some(Projection {
        point,
        weights,
        squared_distance: length_squared(point),
    })
}

fn normalize_weights(mut weights: [f64; 3]) -> [f64; 3] {
    for weight in &mut weights {
        if *weight < 0.0 {
            *weight = 0.0;
        }
    }
    let total = weights.into_iter().sum::<f64>();
    if total == 0.0 {
        [1.0, 0.0, 0.0]
    } else {
        weights.map(|weight| weight / total)
    }
}

#[cfg(test)]
mod tests {
    use super::{GjkDistanceConfig, GjkDistanceStatus, gjk_distance, gjk_distance_with_config};
    use crate::{
        Sphere,
        math3::{length_squared, scale, sub},
        primitives::Capsule,
    };
    use spatial_kernels::Aabb;

    #[test]
    fn separated_spheres_match_analytic_distance_normal_and_witnesses() {
        let left = Sphere::new([0.0; 3], 1.0);
        let right = Sphere::new([3.0, 0.0, 0.0], 0.5);
        let result = gjk_distance(&left, &right);
        let closest = result.closest.expect("separated spheres must converge");
        assert_eq!(result.status, GjkDistanceStatus::Converged);
        assert!((closest.distance - 1.5).abs() <= 1.0e-12);
        assert_eq!(closest.normal, [1.0, 0.0, 0.0]);
        assert_eq!(closest.point_left, [1.0, 0.0, 0.0]);
        assert_eq!(closest.point_right, [2.5, 0.0, 0.0]);
    }

    #[test]
    fn diagonal_aabb_distance_matches_closest_corners() {
        let left = Aabb::from_center_half_extents([0.0; 3], [1.0; 3]);
        let right = Aabb::from_center_half_extents([3.0, 4.0, 0.0], [1.0; 3]);
        let result = gjk_distance(&left, &right);
        let closest = result.closest.expect("separated boxes must converge");
        assert_eq!(result.status, GjkDistanceStatus::Converged);
        assert!((closest.distance - 5.0_f64.sqrt()).abs() <= 1.0e-12);
        assert!((closest.normal[0] - 1.0 / 5.0_f64.sqrt()).abs() <= 1.0e-12);
        assert!((closest.normal[1] - 2.0 / 5.0_f64.sqrt()).abs() <= 1.0e-12);
        assert!(closest.normal[2].abs() <= 1.0e-12);
    }

    #[test]
    fn capsule_sphere_distance_matches_axis_oracle() {
        let left = Capsule::new([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 0.5);
        let right = Sphere::new([3.0, 0.0, 0.0], 0.5);
        let result = gjk_distance(&left, &right);
        let closest = result.closest.expect("capsule and sphere must converge");
        assert_eq!(result.status, GjkDistanceStatus::Converged);
        assert!((closest.distance - 2.0).abs() <= 1.0e-9);
        assert!((closest.normal[0] - 1.0).abs() <= 1.0e-9);
        assert!(closest.normal[1].abs() <= 1.0e-9);
        assert!(closest.normal[2].abs() <= 1.0e-9);
    }

    #[test]
    fn touching_and_overlapping_shapes_are_reported_before_distance_refinement() {
        for center in [[2.0, 0.0, 0.0], [1.5, 0.0, 0.0]] {
            let left = Sphere::new([0.0; 3], 1.0);
            let right = Sphere::new(center, 1.0);
            let result = gjk_distance(&left, &right);
            assert_eq!(result.status, GjkDistanceStatus::Intersecting);
            assert_eq!(result.closest, None);
            assert_eq!(
                result.intersection.status,
                crate::gjk::GjkStatus::Intersecting
            );
        }
    }

    #[test]
    fn swapping_shapes_flips_normal_and_witnesses_and_preserves_distance() {
        let left = Sphere::new([-0.4, 0.2, 0.3], 0.7);
        let right = Sphere::new([2.3, -0.5, 1.1], 0.9);
        let forward = gjk_distance(&left, &right)
            .closest
            .expect("forward distance must converge");
        let reverse = gjk_distance(&right, &left)
            .closest
            .expect("reverse distance must converge");
        let center_delta = [2.7, -0.7, 0.8];
        let center_distance = length_squared(center_delta).sqrt();
        let expected_distance = center_distance - 1.6;
        let expected_normal = scale(center_delta, center_distance.recip());

        for closest in [forward, reverse] {
            assert!((closest.distance - expected_distance).abs() <= 1.0e-9);
            let witness_delta = sub(closest.point_right, closest.point_left);
            let reconstructed = scale(closest.normal, closest.distance);
            for axis in 0..3 {
                assert!((witness_delta[axis] - reconstructed[axis]).abs() <= 1.0e-9);
            }
        }
        for (axis, expected) in expected_normal.into_iter().enumerate() {
            assert!((forward.normal[axis] - expected).abs() <= 1.0e-6);
            assert!((reverse.normal[axis] + expected).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn inconclusive_intersection_prepass_can_still_converge_distance() {
        let left = Sphere::new([0.0; 3], 1.0);
        let right = Sphere::new([3.0, 0.0, 0.0], 0.5);
        let result = gjk_distance_with_config(
            &left,
            &right,
            GjkDistanceConfig {
                max_iterations: 1,
                ..GjkDistanceConfig::default()
            },
        );

        assert_eq!(
            result.intersection.status,
            crate::gjk::GjkStatus::IterationLimit
        );
        assert_eq!(result.status, GjkDistanceStatus::Converged);
        let closest = result
            .closest
            .expect("distance refinement must prove separation from the retained simplex");
        assert!((closest.distance - 1.5).abs() <= 1.0e-12);
        assert_eq!(closest.normal, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn result_is_deterministic() {
        let left = Capsule::new([-1.0, -0.5, 0.2], [1.5, 0.75, -0.1], 0.4);
        let right = Sphere::new([3.0, -1.0, 0.5], 0.8);
        assert_eq!(gjk_distance(&left, &right), gjk_distance(&left, &right));
    }
}
