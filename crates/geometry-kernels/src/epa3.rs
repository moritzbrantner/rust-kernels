use crate::{
    gjk::{GjkResult, GjkStatus},
    math3::{Vec3, add, cross, dot, length_squared, neg, normalized, scale, sub},
    support::{MinkowskiSupportPoint, SupportMap3, minkowski_support},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epa3Status {
    Converged,
    NotIntersecting,
    DegenerateSimplex,
    NoProgress,
    IterationLimit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Epa3Config {
    pub max_iterations: usize,
    /// Maximum remaining support distance before the closest face is accepted.
    pub tolerance: f64,
    /// Geometric tolerance used for duplicate points and degenerate simplexes.
    pub epsilon: f64,
}

impl Default for Epa3Config {
    fn default() -> Self {
        Self {
            max_iterations: 128,
            tolerance: 1.0e-6,
            epsilon: 1.0e-12,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Epa3Penetration {
    /// Minimum translation distance required to separate the shapes.
    pub depth: f64,
    /// Unit normal from the left shape toward the right shape.
    pub normal: Vec3,
    /// Witness on the left shape for the converged closest Minkowski face.
    pub point_left: Vec3,
    /// Witness on the right shape for the converged closest Minkowski face.
    pub point_right: Vec3,
    /// Closest Minkowski face retained when EPA converged.
    pub face: [MinkowskiSupportPoint; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Epa3Result {
    pub status: Epa3Status,
    pub iterations: usize,
    pub penetration: Option<Epa3Penetration>,
}

#[derive(Clone, Copy, Debug)]
struct Face {
    indices: [usize; 3],
    normal: Vec3,
    distance: f64,
}

#[must_use]
pub fn epa_penetration_3d<L, R>(left: &L, right: &R, gjk: &GjkResult) -> Epa3Result
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    epa_penetration_3d_with_config(left, right, gjk, Epa3Config::default())
}

#[must_use]
pub fn epa_penetration_3d_with_config<L, R>(
    left: &L,
    right: &R,
    gjk: &GjkResult,
    config: Epa3Config,
) -> Epa3Result
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    assert!(
        config.max_iterations > 0,
        "EPA3 max_iterations must be positive"
    );
    assert!(
        config.tolerance.is_finite() && config.tolerance >= 0.0,
        "EPA3 tolerance must be non-negative and finite"
    );
    assert!(
        config.epsilon.is_finite() && config.epsilon > 0.0,
        "EPA3 epsilon must be positive and finite"
    );

    if gjk.status != GjkStatus::Intersecting {
        return Epa3Result {
            status: Epa3Status::NotIntersecting,
            iterations: 0,
            penetration: None,
        };
    }

    if gjk.simplex_len == 0 {
        return Epa3Result {
            status: Epa3Status::DegenerateSimplex,
            iterations: 0,
            penetration: None,
        };
    }

    if gjk.simplex_len == 1
        && length_squared(gjk.simplex[0].point) <= config.epsilon * config.epsilon
    {
        let witness = gjk.simplex[0];
        return Epa3Result {
            status: Epa3Status::Converged,
            iterations: 0,
            penetration: Some(Epa3Penetration {
                depth: 0.0,
                normal: [1.0, 0.0, 0.0],
                point_left: witness.left,
                point_right: witness.right,
                face: [witness; 3],
            }),
        };
    }

    let Some(mut vertices) = bootstrap_vertices(left, right, gjk, config.epsilon) else {
        return Epa3Result {
            status: Epa3Status::DegenerateSimplex,
            iterations: 0,
            penetration: None,
        };
    };
    let Some(indices) = enclosing_tetrahedron(&vertices, config.epsilon) else {
        return Epa3Result {
            status: Epa3Status::DegenerateSimplex,
            iterations: 0,
            penetration: None,
        };
    };
    vertices = indices.map(|index| vertices[index]).to_vec();

    let mut faces = Vec::with_capacity(4 + 2 * config.max_iterations);
    for indices in [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]] {
        let Some(face) = make_face(&vertices, indices, config.epsilon) else {
            return Epa3Result {
                status: Epa3Status::DegenerateSimplex,
                iterations: 0,
                penetration: None,
            };
        };
        faces.push(face);
    }

    for iteration in 1..=config.max_iterations {
        let Some((_, closest)) = faces
            .iter()
            .copied()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.distance.total_cmp(&right.distance))
        else {
            return Epa3Result {
                status: Epa3Status::DegenerateSimplex,
                iterations: iteration - 1,
                penetration: None,
            };
        };

        let support = minkowski_support(left, right, closest.normal);
        let support_distance = dot(support.point, closest.normal);
        let gap = support_distance - closest.distance;
        let scaled_tolerance = config
            .tolerance
            .max(config.epsilon * (1.0 + support_distance.abs()));

        if gap <= scaled_tolerance {
            return converged(iteration, closest, &vertices);
        }

        if vertices.iter().any(|vertex| {
            length_squared(sub(vertex.point, support.point)) <= config.epsilon * config.epsilon
        }) {
            return Epa3Result {
                status: Epa3Status::NoProgress,
                iterations: iteration,
                penetration: None,
            };
        }

        let new_index = vertices.len();
        vertices.push(support);

        let mut horizon = Vec::<(usize, usize)>::new();
        let mut visible = vec![false; faces.len()];
        for (index, face) in faces.iter().enumerate() {
            let origin = vertices[face.indices[0]].point;
            let visibility = dot(face.normal, sub(support.point, origin));
            if visibility > config.epsilon * (1.0 + support_distance.abs()) {
                visible[index] = true;
                for (start, end) in [
                    (face.indices[0], face.indices[1]),
                    (face.indices[1], face.indices[2]),
                    (face.indices[2], face.indices[0]),
                ] {
                    if let Some(reverse) =
                        horizon.iter().position(|&(existing_start, existing_end)| {
                            existing_start == end && existing_end == start
                        })
                    {
                        horizon.remove(reverse);
                    } else {
                        horizon.push((start, end));
                    }
                }
            }
        }

        if !visible.iter().any(|value| *value) || horizon.is_empty() {
            vertices.pop();
            return Epa3Result {
                status: Epa3Status::NoProgress,
                iterations: iteration,
                penetration: None,
            };
        }

        let mut retained = Vec::with_capacity(faces.len() + horizon.len());
        for (index, face) in faces.into_iter().enumerate() {
            if !visible[index] {
                retained.push(face);
            }
        }
        for (start, end) in horizon {
            let Some(face) = make_face(&vertices, [start, end, new_index], config.epsilon) else {
                return Epa3Result {
                    status: Epa3Status::DegenerateSimplex,
                    iterations: iteration,
                    penetration: None,
                };
            };
            retained.push(face);
        }
        faces = retained;

    }

    Epa3Result {
        status: Epa3Status::IterationLimit,
        iterations: config.max_iterations,
        penetration: None,
    }
}

fn converged(iterations: usize, face: Face, vertices: &[MinkowskiSupportPoint]) -> Epa3Result {
    let face_points = face.indices.map(|index| vertices[index]);
    let target = scale(face.normal, face.distance);
    let weights = barycentric(target, face_points.map(|point| point.point));
    let point_left = weighted_point(face_points.map(|point| point.left), weights);
    let point_right = weighted_point(face_points.map(|point| point.right), weights);

    Epa3Result {
        status: Epa3Status::Converged,
        iterations,
        penetration: Some(Epa3Penetration {
            depth: face.distance.max(0.0),
            normal: face.normal,
            point_left,
            point_right,
            face: face_points,
        }),
    }
}

fn bootstrap_vertices<L, R>(
    left: &L,
    right: &R,
    gjk: &GjkResult,
    epsilon: f64,
) -> Option<Vec<MinkowskiSupportPoint>>
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    let mut vertices = Vec::with_capacity(12);
    for &point in &gjk.simplex[..gjk.simplex_len] {
        push_unique(&mut vertices, point, epsilon);
    }

    match gjk.simplex_len {
        4 => {}
        3 => {
            let a = gjk.simplex[0].point;
            let b = gjk.simplex[1].point;
            let c = gjk.simplex[2].point;
            let normal = normalized(cross(sub(b, a), sub(c, a)))?;
            push_support_pair(left, right, normal, &mut vertices, epsilon);
        }
        2 => {
            let a = gjk.simplex[0].point;
            let b = gjk.simplex[1].point;
            let axis = normalized(sub(b, a))?;
            let seed = least_aligned_axis(axis);
            let first = normalized(cross(axis, seed))?;
            let second = normalized(cross(axis, first))?;
            push_support_pair(left, right, first, &mut vertices, epsilon);
            push_support_pair(left, right, second, &mut vertices, epsilon);
        }
        1 => {}
        _ => return None,
    }

    if enclosing_tetrahedron(&vertices, epsilon).is_none() {
        for direction in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ] {
            let direction = normalized(direction)?;
            push_support_pair(left, right, direction, &mut vertices, epsilon);
        }
    }

    Some(vertices)
}

fn least_aligned_axis(direction: Vec3) -> Vec3 {
    let absolute = direction.map(f64::abs);
    if absolute[0] <= absolute[1] && absolute[0] <= absolute[2] {
        [1.0, 0.0, 0.0]
    } else if absolute[1] <= absolute[2] {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn push_support_pair<L, R>(
    left: &L,
    right: &R,
    direction: Vec3,
    vertices: &mut Vec<MinkowskiSupportPoint>,
    epsilon: f64,
) where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    push_unique(vertices, minkowski_support(left, right, direction), epsilon);
    push_unique(
        vertices,
        minkowski_support(left, right, neg(direction)),
        epsilon,
    );
}

fn push_unique(
    vertices: &mut Vec<MinkowskiSupportPoint>,
    candidate: MinkowskiSupportPoint,
    epsilon: f64,
) {
    let epsilon_squared = epsilon * epsilon;
    if vertices
        .iter()
        .all(|existing| length_squared(sub(existing.point, candidate.point)) > epsilon_squared)
    {
        vertices.push(candidate);
    }
}

fn enclosing_tetrahedron(vertices: &[MinkowskiSupportPoint], epsilon: f64) -> Option<[usize; 4]> {
    let mut best: Option<([usize; 4], f64, f64)> = None;
    for a in 0..vertices.len() {
        for b in a + 1..vertices.len() {
            for c in b + 1..vertices.len() {
                for d in c + 1..vertices.len() {
                    let points = [
                        vertices[a].point,
                        vertices[b].point,
                        vertices[c].point,
                        vertices[d].point,
                    ];
                    let Some(weights) = origin_barycentric_tetrahedron(points, epsilon) else {
                        continue;
                    };
                    let minimum_weight = weights.into_iter().fold(f64::INFINITY, f64::min);
                    // EPA requires the origin to be strictly inside the seed polytope.
                    // A boundary simplex can immediately select a zero-distance face whose
                    // support point is already present, falsely reporting no progress.
                    if minimum_weight <= epsilon {
                        continue;
                    }
                    let volume = tetrahedron_volume_six(points).abs();
                    let candidate = ([a, b, c, d], minimum_weight, volume);
                    let replace = best.is_none_or(|(_, best_weight, best_volume)| {
                        minimum_weight > best_weight + epsilon
                            || ((minimum_weight - best_weight).abs() <= epsilon
                                && volume > best_volume)
                    });
                    if replace {
                        best = Some(candidate);
                    }
                }
            }
        }
    }
    best.map(|(indices, _, _)| indices)
}

fn origin_barycentric_tetrahedron(points: [Vec3; 4], epsilon: f64) -> Option<[f64; 4]> {
    let [a, b, c, d] = points;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ad = sub(d, a);
    let determinant = dot(ab, cross(ac, ad));
    let scale_squared = length_squared(ab)
        .max(length_squared(ac))
        .max(length_squared(ad))
        .max(1.0);
    if determinant.abs() <= epsilon * scale_squared.sqrt() * scale_squared {
        return None;
    }
    let rhs = neg(a);
    let wb = dot(rhs, cross(ac, ad)) / determinant;
    let wc = dot(ab, cross(rhs, ad)) / determinant;
    let wd = dot(ab, cross(ac, rhs)) / determinant;
    let wa = 1.0 - wb - wc - wd;
    Some([wa, wb, wc, wd])
}

fn tetrahedron_volume_six(points: [Vec3; 4]) -> f64 {
    let [a, b, c, d] = points;
    dot(sub(b, a), cross(sub(c, a), sub(d, a)))
}

fn make_face(
    vertices: &[MinkowskiSupportPoint],
    mut indices: [usize; 3],
    epsilon: f64,
) -> Option<Face> {
    let a = vertices[indices[0]].point;
    let b = vertices[indices[1]].point;
    let c = vertices[indices[2]].point;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let raw = cross(ab, ac);
    let longest_edge_squared = length_squared(ab)
        .max(length_squared(ac))
        .max(length_squared(sub(c, b)))
        .max(1.0);
    if length_squared(raw) <= epsilon * epsilon * longest_edge_squared {
        return None;
    }
    let mut normal = normalized(raw)?;
    let mut distance = dot(normal, a);
    if distance < 0.0 {
        indices.swap(1, 2);
        normal = neg(normal);
        distance = -distance;
    }
    Some(Face {
        indices,
        normal,
        distance,
    })
}

fn barycentric(point: Vec3, triangle: [Vec3; 3]) -> [f64; 3] {
    let [a, b, c] = triangle;
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(point, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denominator = d00 * d11 - d01 * d01;
    if denominator == 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let second = (d11 * d20 - d01 * d21) / denominator;
    let third = (d00 * d21 - d01 * d20) / denominator;
    let first = 1.0 - second - third;
    normalize_weights([first, second, third])
}

fn normalize_weights(mut weights: [f64; 3]) -> [f64; 3] {
    for weight in &mut weights {
        if *weight < 0.0 && *weight > -1.0e-10 {
            *weight = 0.0;
        }
    }
    let sum = weights.into_iter().sum::<f64>();
    if sum == 0.0 {
        [1.0, 0.0, 0.0]
    } else {
        weights.map(|weight| weight / sum)
    }
}

fn weighted_point(points: [Vec3; 3], weights: [f64; 3]) -> Vec3 {
    add(
        scale(points[0], weights[0]),
        add(scale(points[1], weights[1]), scale(points[2], weights[2])),
    )
}

#[cfg(test)]
mod tests {
    use super::{Epa3Config, Epa3Status, epa_penetration_3d, epa_penetration_3d_with_config};
    use crate::{
        Sphere,
        gjk::gjk_intersection,
        obb3::{Obb3, obb3_sat},
    };

    #[test]
    fn sphere_penetration_matches_analytic_depth_normal_and_witnesses() {
        let left = Sphere::new([0.0, 0.0, 0.0], 1.0);
        let right = Sphere::new([1.5, 0.0, 0.0], 1.0);
        let gjk = gjk_intersection(&left, &right);
        let result = epa_penetration_3d(&left, &right, &gjk);
        let penetration = result.penetration.expect("overlap must converge");

        assert_eq!(result.status, Epa3Status::Converged);
        assert!((penetration.depth - 0.5).abs() <= 2.0e-6);
        assert!((penetration.normal[0] - 1.0).abs() <= 2.0e-3);
        assert!(penetration.normal[1].abs() <= 2.0e-3);
        assert!(penetration.normal[2].abs() <= 2.0e-3);
        assert!((penetration.point_left[0] - 1.0).abs() <= 2.0e-6);
        assert!((penetration.point_right[0] - 0.5).abs() <= 2.0e-6);
    }

    #[test]
    fn concentric_sphere_line_simplex_bootstraps_without_claiming_false_convergence() {
        let left = Sphere::new([0.0; 3], 2.0);
        let right = Sphere::new([0.0; 3], 1.0);
        let gjk = gjk_intersection(&left, &right);
        assert_eq!(gjk.simplex_len, 2);

        // A smooth, concentric Minkowski sphere is deliberately adversarial for polytope EPA:
        // every direction is an equally valid penetration normal. Prove that the line simplex is
        // expanded into a valid 3D polytope and that the bounded kernel reports its budget instead
        // of mistaking a boundary tetrahedron for convergence.
        let result = epa_penetration_3d_with_config(
            &left,
            &right,
            &gjk,
            Epa3Config {
                max_iterations: 8,
                tolerance: 0.0,
                ..Epa3Config::default()
            },
        );
        assert_eq!(result.status, Epa3Status::IterationLimit);
        assert_eq!(result.iterations, 8);
        assert_eq!(result.penetration, None);
    }

    #[test]
    fn touching_spheres_report_zero_penetration() {
        let left = Sphere::new([0.0; 3], 1.0);
        let right = Sphere::new([2.0, 0.0, 0.0], 1.0);
        let gjk = gjk_intersection(&left, &right);
        assert_eq!(gjk.simplex_len, 1);
        let result = epa_penetration_3d(&left, &right, &gjk);
        assert_eq!(result.status, Epa3Status::Converged);
        assert_eq!(result.penetration.expect("touch must converge").depth, 0.0);
    }

    #[test]
    fn axis_aligned_boxes_match_sat_penetration_depth() {
        let left = Obb3::new([0.0; 3], [1.0, 2.0, 1.5], [0.0; 3]);
        let right = Obb3::new([1.25, 0.25, 0.1], [1.0, 0.5, 0.75], [0.0; 3]);
        let relation = obb3_sat(left, right);
        assert!(relation.overlaps);
        let expected = relation.axes[relation.critical_axis].signed_overlap;

        let gjk = gjk_intersection(&left, &right);
        let result = epa_penetration_3d(&left, &right, &gjk);
        let penetration = result.penetration.expect("box overlap must converge");
        assert_eq!(result.status, Epa3Status::Converged);
        assert!((penetration.depth - expected).abs() <= 1.0e-8);
    }

    #[test]
    fn swapping_shapes_flips_normal_and_witnesses_and_preserves_depth() {
        let left = Sphere::new([-0.25, 0.4, -0.1], 1.25);
        let right = Sphere::new([1.1, 0.2, 0.3], 0.9);
        let forward_gjk = gjk_intersection(&left, &right);
        let reverse_gjk = gjk_intersection(&right, &left);
        let forward = epa_penetration_3d(&left, &right, &forward_gjk)
            .penetration
            .expect("forward overlap must converge");
        let reverse = epa_penetration_3d(&right, &left, &reverse_gjk)
            .penetration
            .expect("reverse overlap must converge");

        assert!((forward.depth - reverse.depth).abs() <= 2.0e-6);
        for axis in 0..3 {
            assert!((forward.normal[axis] + reverse.normal[axis]).abs() <= 2.0e-3);
            assert!((forward.point_left[axis] - reverse.point_right[axis]).abs() <= 2.0e-6);
            assert!((forward.point_right[axis] - reverse.point_left[axis]).abs() <= 2.0e-6);
        }
    }

    #[test]
    fn separated_shapes_are_rejected_before_expansion() {
        let left = Sphere::new([0.0; 3], 1.0);
        let right = Sphere::new([4.0, 0.0, 0.0], 1.0);
        let gjk = gjk_intersection(&left, &right);
        let result = epa_penetration_3d(&left, &right, &gjk);
        assert_eq!(result.status, Epa3Status::NotIntersecting);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.penetration, None);
    }

    #[test]
    fn iteration_limit_is_explicit() {
        let left = Sphere::new([0.0; 3], 1.0);
        let right = Sphere::new([1.4, 0.2, 0.3], 1.0);
        let gjk = gjk_intersection(&left, &right);
        let result = epa_penetration_3d_with_config(
            &left,
            &right,
            &gjk,
            Epa3Config {
                max_iterations: 1,
                tolerance: 0.0,
                ..Epa3Config::default()
            },
        );
        assert!(matches!(
            result.status,
            Epa3Status::IterationLimit | Epa3Status::NoProgress
        ));
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn result_is_deterministic() {
        let left = Obb3::new([0.0; 3], [1.4, 0.8, 0.9], [0.2, 0.4, -0.15]);
        let right = Obb3::new([1.2, 0.35, 0.25], [1.0, 0.7, 0.8], [-0.3, 0.15, 0.55]);
        let gjk = gjk_intersection(&left, &right);
        let first = epa_penetration_3d(&left, &right, &gjk);
        let second = epa_penetration_3d(&left, &right, &gjk);
        assert_eq!(first, second);
    }
}
