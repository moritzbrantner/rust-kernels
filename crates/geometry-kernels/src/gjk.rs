use crate::{
    math3::{Vec3, cross, dot, length_squared, neg, scale, sub, triple_cross},
    support::{MinkowskiSupportPoint, SupportMap3, minkowski_support},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GjkStatus {
    Intersecting,
    Separated,
    NoProgress,
    IterationLimit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GjkConfig {
    pub max_iterations: usize,
    /// Geometric tolerance used only for simplex degeneracy and repeated support points.
    pub epsilon: f64,
}

impl Default for GjkConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            epsilon: 1.0e-12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GjkResult {
    pub status: GjkStatus,
    pub iterations: usize,
    pub simplex: [MinkowskiSupportPoint; 4],
    pub simplex_len: usize,
    pub search_direction: Vec3,
}

impl GjkResult {
    #[must_use]
    pub fn intersection(self) -> Option<bool> {
        match self.status {
            GjkStatus::Intersecting => Some(true),
            GjkStatus::Separated => Some(false),
            GjkStatus::NoProgress | GjkStatus::IterationLimit => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Simplex {
    points: [MinkowskiSupportPoint; 4],
    len: usize,
}

impl Simplex {
    fn new(point: MinkowskiSupportPoint) -> Self {
        let mut points = [MinkowskiSupportPoint::default(); 4];
        points[0] = point;
        Self { points, len: 1 }
    }

    fn push_front(&mut self, point: MinkowskiSupportPoint) {
        let new_len = (self.len + 1).min(4);
        for index in (1..new_len).rev() {
            self.points[index] = self.points[index - 1];
        }
        self.points[0] = point;
        self.len = new_len;
    }

    fn keep1(&mut self, first: usize) {
        self.points[0] = self.points[first];
        self.len = 1;
    }

    fn keep2(&mut self, first: usize, second: usize) {
        let a = self.points[first];
        let b = self.points[second];
        self.points[0] = a;
        self.points[1] = b;
        self.len = 2;
    }

    fn keep3(&mut self, first: usize, second: usize, third: usize) {
        let a = self.points[first];
        let b = self.points[second];
        let c = self.points[third];
        self.points[0] = a;
        self.points[1] = b;
        self.points[2] = c;
        self.len = 3;
    }

    fn contains_point(&self, point: Vec3, epsilon_squared: f64) -> bool {
        self.points[..self.len]
            .iter()
            .any(|existing| length_squared(sub(existing.point, point)) <= epsilon_squared)
    }
}

#[must_use]
pub fn gjk_intersection<L, R>(left: &L, right: &R) -> GjkResult
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    gjk_intersection_with_config(left, right, GjkConfig::default())
}

#[must_use]
pub fn gjk_intersection_with_config<L, R>(
    left: &L,
    right: &R,
    config: GjkConfig,
) -> GjkResult
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    assert!(config.max_iterations > 0, "GJK max_iterations must be positive");
    assert!(
        config.epsilon.is_finite() && config.epsilon >= 0.0,
        "GJK epsilon must be non-negative and finite"
    );
    let epsilon_squared = config.epsilon * config.epsilon;

    let first = minkowski_support(left, right, [1.0, 0.0, 0.0]);
    let mut simplex = Simplex::new(first);
    let mut direction = neg(first.point);
    if length_squared(direction) <= epsilon_squared {
        return result(GjkStatus::Intersecting, 1, simplex, direction);
    }

    if config.max_iterations == 1 {
        return result(GjkStatus::IterationLimit, 1, simplex, direction);
    }

    for iteration in 2..=config.max_iterations {
        let support = minkowski_support(left, right, direction);
        if dot(support.point, direction) < 0.0 {
            return result(GjkStatus::Separated, iteration, simplex, direction);
        }

        let repeated = simplex.contains_point(support.point, epsilon_squared);
        simplex.push_front(support);
        if process_simplex(&mut simplex, &mut direction, epsilon_squared) {
            return result(GjkStatus::Intersecting, iteration, simplex, direction);
        }
        if length_squared(direction) <= epsilon_squared {
            return result(GjkStatus::Intersecting, iteration, simplex, direction);
        }
        if repeated {
            return result(GjkStatus::NoProgress, iteration, simplex, direction);
        }
    }

    result(
        GjkStatus::IterationLimit,
        config.max_iterations,
        simplex,
        direction,
    )
}

fn result(
    status: GjkStatus,
    iterations: usize,
    simplex: Simplex,
    search_direction: Vec3,
) -> GjkResult {
    GjkResult {
        status,
        iterations,
        simplex: simplex.points,
        simplex_len: simplex.len,
        search_direction,
    }
}

fn process_simplex(simplex: &mut Simplex, direction: &mut Vec3, epsilon_squared: f64) -> bool {
    match simplex.len {
        1 => {
            *direction = neg(simplex.points[0].point);
            length_squared(*direction) <= epsilon_squared
        }
        2 => process_line(simplex, direction, epsilon_squared),
        3 => process_triangle(simplex, direction, epsilon_squared),
        4 => process_tetrahedron(simplex, direction, epsilon_squared),
        _ => unreachable!("GJK simplex must contain one to four points"),
    }
}

fn process_line(simplex: &mut Simplex, direction: &mut Vec3, epsilon_squared: f64) -> bool {
    let a = simplex.points[0].point;
    let b = simplex.points[1].point;
    let ao = neg(a);
    let ab = sub(b, a);

    if dot(ab, ao) <= 0.0 {
        simplex.keep1(0);
        *direction = ao;
        return length_squared(ao) <= epsilon_squared;
    }

    let perpendicular = triple_cross(ab, ao, ab);
    if length_squared(perpendicular) > epsilon_squared {
        *direction = perpendicular;
        return false;
    }

    let closest = closest_point_on_segment_to_origin(a, b);
    if length_squared(closest) <= epsilon_squared {
        return true;
    }
    *direction = neg(closest);
    false
}

fn process_triangle(
    simplex: &mut Simplex,
    direction: &mut Vec3,
    epsilon_squared: f64,
) -> bool {
    let a = simplex.points[0].point;
    let b = simplex.points[1].point;
    let c = simplex.points[2].point;
    let ao = neg(a);
    let ab = sub(b, a);
    let ac = sub(c, a);
    let abc = cross(ab, ac);

    if length_squared(abc) <= epsilon_squared {
        return reduce_degenerate_triangle(simplex, direction, epsilon_squared);
    }

    if dot(cross(abc, ac), ao) > 0.0 {
        if dot(ac, ao) > 0.0 {
            simplex.keep2(0, 2);
        } else {
            simplex.keep2(0, 1);
        }
        return process_line(simplex, direction, epsilon_squared);
    }

    if dot(cross(ab, abc), ao) > 0.0 {
        simplex.keep2(0, 1);
        return process_line(simplex, direction, epsilon_squared);
    }

    if dot(abc, ao) > 0.0 {
        *direction = abc;
    } else {
        simplex.keep3(0, 2, 1);
        *direction = neg(abc);
    }
    false
}

fn reduce_degenerate_triangle(
    simplex: &mut Simplex,
    direction: &mut Vec3,
    epsilon_squared: f64,
) -> bool {
    let candidates = [(0, 1), (0, 2), (1, 2)];
    let mut best = candidates[0];
    let mut best_closest = closest_point_on_segment_to_origin(
        simplex.points[best.0].point,
        simplex.points[best.1].point,
    );
    let mut best_distance_squared = length_squared(best_closest);

    for candidate in candidates.into_iter().skip(1) {
        let closest = closest_point_on_segment_to_origin(
            simplex.points[candidate.0].point,
            simplex.points[candidate.1].point,
        );
        let distance_squared = length_squared(closest);
        if distance_squared < best_distance_squared {
            best = candidate;
            best_closest = closest;
            best_distance_squared = distance_squared;
        }
    }

    simplex.keep2(best.0, best.1);
    if best_distance_squared <= epsilon_squared {
        return true;
    }
    *direction = neg(best_closest);
    false
}

fn process_tetrahedron(
    simplex: &mut Simplex,
    direction: &mut Vec3,
    epsilon_squared: f64,
) -> bool {
    let a = simplex.points[0].point;
    let b = simplex.points[1].point;
    let c = simplex.points[2].point;
    let d = simplex.points[3].point;
    let ao = neg(a);

    if origin_outside_face(a, b, c, d, ao, epsilon_squared) {
        simplex.keep3(0, 1, 2);
        return process_triangle(simplex, direction, epsilon_squared);
    }
    if origin_outside_face(a, c, d, b, ao, epsilon_squared) {
        simplex.keep3(0, 2, 3);
        return process_triangle(simplex, direction, epsilon_squared);
    }
    if origin_outside_face(a, d, b, c, ao, epsilon_squared) {
        simplex.keep3(0, 3, 1);
        return process_triangle(simplex, direction, epsilon_squared);
    }

    let volume_six = dot(cross(sub(b, a), sub(c, a)), sub(d, a)).abs();
    if volume_six <= epsilon_squared {
        simplex.keep3(0, 1, 2);
        return process_triangle(simplex, direction, epsilon_squared);
    }

    true
}

fn origin_outside_face(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    opposite: Vec3,
    ao: Vec3,
    epsilon_squared: f64,
) -> bool {
    let mut normal = cross(sub(b, a), sub(c, a));
    if length_squared(normal) <= epsilon_squared {
        return false;
    }
    if dot(normal, sub(opposite, a)) > 0.0 {
        normal = neg(normal);
    }
    dot(normal, ao) > 0.0
}

fn closest_point_on_segment_to_origin(start: Vec3, end: Vec3) -> Vec3 {
    let direction = sub(end, start);
    let denominator = length_squared(direction);
    if denominator == 0.0 {
        return start;
    }
    let parameter = (-dot(start, direction) / denominator).clamp(0.0, 1.0);
    sub(scale(start, 1.0 - parameter), scale(neg(end), parameter))
}

#[cfg(test)]
mod tests {
    use super::{GjkConfig, GjkStatus, gjk_intersection, gjk_intersection_with_config};
    use crate::{
        Sphere, aabb_aabb,
        obb3::{Obb3, obb3_sat},
        primitives::{Capsule, capsule_capsule},
        sphere_sphere,
        support::ConvexHull3,
    };
    use spatial_kernels::Aabb;

    #[test]
    fn sphere_decisions_match_the_analytical_oracle() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        for right in [
            Sphere::new([0.0, 0.0, 0.0], 0.5),
            Sphere::new([1.4, 1.4, 0.0], 1.0),
            Sphere::new([2.0, 0.0, 0.0], 1.0),
            Sphere::new([2.001, 0.0, 0.0], 1.0),
        ] {
            let expected = sphere_sphere(left, right).overlaps;
            assert_eq!(gjk_intersection(&left, &right).intersection(), Some(expected));
        }
    }

    #[test]
    fn aabb_decisions_match_the_analytical_oracle() {
        let left = Aabb::new([-1.0; 3], [1.0; 3]);
        for right in [
            Aabb::new([0.5, -0.5, -0.5], [2.0, 0.5, 0.5]),
            Aabb::new([1.0, -0.5, -0.5], [2.0, 0.5, 0.5]),
            Aabb::new([1.001, -0.5, -0.5], [2.0, 0.5, 0.5]),
        ] {
            let expected = aabb_aabb(left, right).overlaps;
            assert_eq!(gjk_intersection(&left, &right).intersection(), Some(expected));
        }
    }

    #[test]
    fn capsule_decisions_match_closest_segment_oracle() {
        let left = Capsule::new([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5);
        for right in [
            Capsule::new([0.0, 0.8, -1.0], [0.0, 0.8, 1.0], 0.5),
            Capsule::new([0.0, 1.0, -1.0], [0.0, 1.0, 1.0], 0.5),
            Capsule::new([0.0, 1.1, -1.0], [0.0, 1.1, 1.0], 0.5),
        ] {
            let expected = capsule_capsule(left, right).overlaps;
            assert_eq!(gjk_intersection(&left, &right).intersection(), Some(expected));
        }
    }

    #[test]
    fn obb3_decisions_match_sat_for_rotated_boxes() {
        let left = Obb3::new([0.0, 0.0, 0.0], [1.4, 0.8, 0.9], [0.2, 0.4, -0.15]);
        for right in [
            Obb3::new([1.2, 0.35, 0.25], [1.0, 0.7, 0.8], [-0.3, 0.15, 0.55]),
            Obb3::new([4.0, 1.5, 0.8], [1.0, 0.7, 0.8], [-0.3, 0.15, 0.55]),
        ] {
            let expected = obb3_sat(left, right).overlaps;
            assert_eq!(gjk_intersection(&left, &right).intersection(), Some(expected));
        }
    }

    #[test]
    fn convex_hull_support_sets_work_without_shape_specific_code() {
        let left_points = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ];
        let overlapping_points = left_points.map(|point| [point[0] + 1.5, point[1], point[2]]);
        let separated_points = left_points.map(|point| [point[0] + 3.0, point[1], point[2]]);
        let left = ConvexHull3::new(&left_points);
        let overlapping = ConvexHull3::new(&overlapping_points);
        let separated = ConvexHull3::new(&separated_points);
        assert_eq!(gjk_intersection(&left, &overlapping).intersection(), Some(true));
        assert_eq!(gjk_intersection(&left, &separated).intersection(), Some(false));
    }

    #[test]
    fn result_keeps_final_simplex_and_witness_points() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let right = Sphere::new([1.0, 0.0, 0.0], 1.0);
        let result = gjk_intersection(&left, &right);
        assert_eq!(result.status, GjkStatus::Intersecting);
        assert!((1..=4).contains(&result.simplex_len));
        for support in &result.simplex[..result.simplex_len] {
            assert_eq!(
                support.point,
                [
                    support.left[0] - support.right[0],
                    support.left[1] - support.right[1],
                    support.left[2] - support.right[2],
                ]
            );
        }
    }

    #[test]
    fn iteration_limit_is_explicit_instead_of_guessing() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let right = Sphere::new([1.4, 1.4, 0.0], 1.0);
        let result = gjk_intersection_with_config(
            &left,
            &right,
            GjkConfig {
                max_iterations: 1,
                ..GjkConfig::default()
            },
        );
        assert_eq!(result.status, GjkStatus::IterationLimit);
        assert_eq!(result.intersection(), None);
    }

    #[test]
    fn decision_is_symmetric_for_regular_cases() {
        let left = Capsule::new([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.7);
        let right = Capsule::new([0.0, 0.9, -1.0], [0.0, 0.9, 2.0], 0.4);
        assert_eq!(
            gjk_intersection(&left, &right).intersection(),
            gjk_intersection(&right, &left).intersection()
        );
    }
}