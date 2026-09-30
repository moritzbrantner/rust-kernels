//! Degenerate skeleton contacts must provide an actual separating direction.
use geometry_kernels::{
    math3::{Vec3, dot, scale, sub},
    primitive3::{PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, query},
};

// Independent support formulas: no specialized closest-point/contact helpers.
fn support(body: PrimitiveBody3, direction: Vec3) -> Vec3 {
    let length = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    let unit = direction.map(|v| v / length);
    let (half, radius) = match body.shape.kind() {
        geometry_kernels::primitive3::PrimitiveKind3::Sphere => (0.0, body.shape.radius()),
        geometry_kernels::primitive3::PrimitiveKind3::Capsule => {
            let extents = body.shape.half_extents();
            (extents[1] - extents[0], extents[0])
        }
        _ => panic!("fixture supports spheres/capsules only"),
    };
    let endpoint = if dot(direction, body.axes[1]) >= 0.0 {
        half
    } else {
        -half
    };
    std::array::from_fn(|axis| {
        body.position[axis] + body.axes[1][axis] * endpoint + unit[axis] * radius
    })
}

fn plane_gap(a: PrimitiveBody3, b: PrimitiveBody3, normal: Vec3) -> f64 {
    dot(
        sub(support(b, scale(normal, -1.0)), support(a, normal)),
        normal,
    )
}

fn check_clearance(a: PrimitiveBody3, b: PrimitiveBody3) {
    let contact = query(a, b, &mut PrimitiveWork3::default());
    assert!(contact.separation < 0.0);
    let scale = a.shape.radius().max(b.shape.radius());
    let margin = scale * 1e-5;
    let mut moved = b;
    for axis in 0..3 {
        moved.position[axis] += contact.normal[axis] * (-contact.separation + margin);
    }
    let gap = plane_gap(a, moved, contact.normal);
    assert!(
        gap >= margin * 0.99,
        "normal {:?} depth {} leaves independent support-plane gap {gap}",
        contact.normal,
        -contact.separation
    );
}

#[test]
fn sphere_center_on_skeleton_can_clear_along_the_reported_normal() {
    for scale in [1e-6, 1.0, 1e6] {
        let a = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::capsule(5.0 * scale, 0.5 * scale),
            [0.0; 3],
            [0.0; 3],
        );
        let b = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::sphere(0.25 * scale),
            [0.0, 2.0 * scale, 0.0],
            [0.0; 3],
        );
        check_clearance(a, b);
    }
}

#[test]
fn crossing_skeletons_can_clear_along_the_reported_normal() {
    for scale in [1e-6, 1.0, 1e6] {
        let a = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::capsule(5.0 * scale, 0.5 * scale),
            [0.0; 3],
            [0.0; 3],
        );
        let mut b = a;
        b.axes = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
        check_clearance(a, b);
    }
}

fn unit(vector: Vec3) -> Vec3 {
    let length = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
    vector.map(|v| v / length)
}

fn axes(y: Vec3) -> [Vec3; 3] {
    let y = unit(y);
    let helper = if y[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let x = unit(geometry_kernels::math3::cross(y, helper));
    [x, y, geometry_kernels::math3::cross(x, y)]
}

fn capsule(half: f64, radius: f64, position: Vec3, y: Vec3) -> PrimitiveBody3 {
    let mut body =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::capsule(half, radius), position, [0.0; 3]);
    body.axes = axes(y);
    body
}

#[test]
fn rotated_and_translated_skeleton_intersections_have_certified_clearance() {
    let directions = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        [-4.0, 1.0, 2.0],
    ];
    for scale in [1e-6, 1.0, 1e6] {
        let position = [-13.0 * scale, 7.0 * scale, -2.0 * scale];
        for a_axis in directions {
            let a = capsule(5.0 * scale, 0.5 * scale, position, a_axis);
            for t in [-5.0, -2.0, 0.0, 3.0, 5.0] {
                let center = std::array::from_fn(|i| position[i] + a.axes[1][i] * t * scale);
                let sphere = PrimitiveBody3::axis_aligned(
                    PrimitiveShape3::sphere(0.25 * scale),
                    center,
                    [0.0; 3],
                );
                check_clearance(a, sphere);
                let point_capsule = capsule(0.0, 0.25 * scale, center, [1.0, 0.0, 0.0]);
                check_clearance(a, point_capsule);
                check_clearance(point_capsule, a);
            }
            for b_axis in directions {
                let b = capsule(4.0 * scale, 0.75 * scale, position, b_axis);
                check_clearance(a, b);
                check_clearance(b, a);
            }
        }
    }
}

#[test]
fn nearly_parallel_crossings_preserve_a_perpendicular_clearance_axis() {
    for scale in [1e-6, 1.0, 1e6] {
        for angle in [1e-12, 1e-9, 1e-6] {
            let a = capsule(5.0 * scale, 0.5 * scale, [0.0; 3], [1.0, 2.0, 3.0]);
            let y = std::array::from_fn(|i| a.axes[1][i] + angle * a.axes[0][i]);
            for sign in [-1.0, 1.0] {
                let b = capsule(4.0 * scale, 0.75 * scale, [0.0; 3], scale_vector(y, sign));
                check_clearance(a, b);
                check_clearance(b, a);
            }
        }
    }
}

fn scale_vector(vector: Vec3, factor: f64) -> Vec3 {
    vector.map(|v| v * factor)
}

#[test]
fn roundoff_sized_core_offsets_do_not_understate_the_selected_clearance() {
    for scale in [1e-6, 1.0, 1e6] {
        let a = capsule(5.0 * scale, 0.5 * scale, [0.0; 3], [0.0, 1.0, 0.0]);
        for offset in [-1e-15, 0.0, 1e-15] {
            let b = PrimitiveBody3::axis_aligned(
                PrimitiveShape3::sphere(0.25 * scale),
                [offset * scale, 2.0 * scale, 0.0],
                [0.0; 3],
            );
            check_clearance(a, b);
            let contact = query(a, b, &mut PrimitiveWork3::default());
            let projected = dot(sub(contact.point_b, contact.point_a), contact.normal);
            assert!((projected - contact.separation).abs() <= 64.0 * f64::EPSILON * scale);
        }
    }
}

#[test]
fn collapsed_capsules_retain_sphere_distance_and_contact_direction() {
    for scale in [1e-6, 1.0, 1e6] {
        let a = capsule(0.0, 0.5 * scale, [0.0; 3], [1.0, 2.0, 3.0]);
        let b = capsule(0.0, 0.75 * scale, [0.0; 3], [-4.0, 1.0, 2.0]);
        check_clearance(a, b);
        let b = PrimitiveBody3 {
            position: [0.0, 1e-15 * scale, 0.0],
            ..b
        };
        let contact = query(a, b, &mut PrimitiveWork3::default());
        assert_eq!(contact.normal, [0.0, 1.0, 0.0]);
        check_clearance(a, b);
    }
}

#[test]
fn seeded_intersections_include_small_shapes_at_large_coordinate_offsets() {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / ((1_u64 << 53) as f64)
    };
    for _ in 0..1024 {
        let center = std::array::from_fn(|_| (random() - 0.5) * 200.0);
        let y_a = std::array::from_fn(|_| random() - 0.5);
        let y_b = std::array::from_fn(|_| random() - 0.5);
        let a_half = 0.01 + random() * 0.1;
        let b_half = 0.01 + random() * 0.1;
        let mut a = capsule(a_half, 0.001 + random() * 0.01, center, y_a);
        let mut b = capsule(b_half, 0.001 + random() * 0.01, center, y_b);
        let a_offset = (random() - 0.5) * a_half;
        let b_offset = (random() - 0.5) * b_half;
        for i in 0..3 {
            a.position[i] += a.axes[1][i] * a_offset;
            b.position[i] += b.axes[1][i] * b_offset;
        }
        check_clearance(a, b);
    }
}

#[test]
fn very_short_nonzero_skeletons_do_not_collapse_to_their_first_endpoint() {
    let a = capsule(1e-13, 1e-6, [0.0; 3], [0.0, 1.0, 0.0]);
    let b =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(1e-6), [1.5e-6, 0.0, 0.0], [0.0; 3]);
    let contact = query(a, b, &mut PrimitiveWork3::default());
    assert!(contact.normal[1].abs() < 1e-14);
    assert_eq!(contact.normal[0], 1.0);
    check_clearance(a, b);
}

#[test]
fn pose_roundoff_ties_cannot_turn_resolved_misses_into_overlaps() {
    let a = capsule(5.0, 1e-6, [1e12, 1e12, 1e12], [0.0, 1.0, 0.0]);
    let b = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(1e-6),
        [1e12 + 0.005, 1e12 + 2.0, 1e12],
        [0.0; 3],
    );
    let contact = query(a, b, &mut PrimitiveWork3::default());
    assert!(contact.separation > 0.004);
    assert_eq!(contact.normal, [1.0, 0.0, 0.0]);
}

#[test]
fn resolved_touching_and_separated_pairs_keep_the_distance_normal() {
    for scale in [1e-6, 1.0, 1e6] {
        let a = capsule(5.0 * scale, 0.5 * scale, [0.0; 3], [0.0, 1.0, 0.0]);
        for gap in [-0.125, 0.0, 0.125] {
            let position = [(0.75 + gap) * scale, 2.0 * scale, 0.0];
            let sphere = PrimitiveBody3::axis_aligned(
                PrimitiveShape3::sphere(0.25 * scale),
                position,
                [0.0; 3],
            );
            let parallel = capsule(4.0 * scale, 0.25 * scale, position, [0.0, 1.0, 0.0]);
            for b in [sphere, parallel] {
                let contact = query(a, b, &mut PrimitiveWork3::default());
                assert!((contact.normal[0] - 1.0).abs() <= 16.0 * f64::EPSILON);
                assert!(contact.normal[1].abs() <= 16.0 * f64::EPSILON);
                assert!(contact.normal[2].abs() <= 16.0 * f64::EPSILON);
                assert!((contact.separation - gap * scale).abs() <= 16.0 * f64::EPSILON * scale);
                let reverse = query(b, a, &mut PrimitiveWork3::default());
                for axis in 0..3 {
                    assert!(
                        (reverse.normal[axis] + contact.normal[axis]).abs() <= 16.0 * f64::EPSILON
                    );
                }
                assert!(
                    (reverse.separation - contact.separation).abs() <= 16.0 * f64::EPSILON * scale
                );
                let certified = plane_gap(a, b, contact.normal);
                assert!((certified - contact.separation).abs() <= 64.0 * f64::EPSILON * scale);
            }
        }
    }
}
