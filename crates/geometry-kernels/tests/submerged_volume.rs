#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration tests are not production code; a panic is the intended failure report"
)]

//! Submerged volume/centroid against closed forms, independent f64 voxel
//! integration and an independent adaptive slice integral for capsules.

use geometry_kernels::{
    hull::{Hull3, HullDimension, QuickHullWork, quickhull3},
    math3::{Vec3, add, cross, dot, length, normalized, scale, sub},
    primitive3::{
        PrimitiveKind3, PrimitiveShape3, SubmergedError3, SubmergedProperties3,
        try_hull_submerged_properties, try_submerged_properties, try_volume_properties,
    },
};
use std::f64::consts::PI;

const IDENTITY: [Vec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Orthonormal axes from a rotation about a unit axis.
fn rotation(axis: Vec3, angle: f64) -> [Vec3; 3] {
    let axis = normalized(axis).unwrap();
    let (s, c) = angle.sin_cos();
    IDENTITY.map(|v| {
        add(
            add(scale(v, c), scale(cross(axis, v), s)),
            scale(axis, dot(axis, v) * (1.0 - c)),
        )
    })
}

fn shapes() -> [PrimitiveShape3; 5] {
    [
        PrimitiveShape3::sphere(0.8),
        PrimitiveShape3::cuboid([0.5, 1.0, 1.5]),
        PrimitiveShape3::capsule(1.0, 0.4),
        PrimitiveShape3::capsule(0.0, 0.7),
        PrimitiveShape3::wedge([1.0, 0.6, 0.8]),
    ]
}

fn contains(shape: PrimitiveShape3, p: Vec3) -> bool {
    let [hx, hy, hz] = shape.half_extents();
    match shape.kind() {
        PrimitiveKind3::Sphere => dot(p, p) <= hx * hx,
        PrimitiveKind3::Box => p[0].abs() <= hx && p[1].abs() <= hy && p[2].abs() <= hz,
        PrimitiveKind3::Capsule => {
            let r = hx;
            let segment = hy - r;
            let y = p[1].clamp(-segment, segment);
            let d = sub(p, [0.0, y, 0.0]);
            dot(d, d) <= r * r
        }
        PrimitiveKind3::Wedge => {
            p[0].abs() <= hx
                && p[1].abs() <= hy
                && p[2].abs() <= hz
                // Slope through (-hx, hy) and (hx, -hy).
                && hy * p[0] + hx * p[1] <= 0.0
        }
    }
}

/// Midpoint voxel integration of the submerged part in the local bounding box.
fn voxel_reference(
    inside: impl Fn(Vec3) -> bool,
    half: Vec3,
    axes: [Vec3; 3],
    position: Vec3,
    plane_point: Vec3,
    normal: Vec3,
    cells: usize,
) -> (f64, Vec3) {
    let step = half.map(|h| 2.0 * h / cells as f64);
    let cell_volume = step[0] * step[1] * step[2];
    let mut volume = 0.0;
    let mut first = [0.0; 3];
    for i in 0..cells {
        for j in 0..cells {
            for k in 0..cells {
                let local = [
                    -half[0] + (i as f64 + 0.5) * step[0],
                    -half[1] + (j as f64 + 0.5) * step[1],
                    -half[2] + (k as f64 + 0.5) * step[2],
                ];
                if !inside(local) {
                    continue;
                }
                let world = add(
                    position,
                    add(
                        add(scale(axes[0], local[0]), scale(axes[1], local[1])),
                        scale(axes[2], local[2]),
                    ),
                );
                if dot(sub(world, plane_point), normal) <= 0.0 {
                    volume += cell_volume;
                    first = add(first, scale(world, cell_volume));
                }
            }
        }
    }
    let centroid = if volume > 0.0 {
        scale(first, 1.0 / volume)
    } else {
        position
    };
    (volume, centroid)
}

fn submerged(
    shape: PrimitiveShape3,
    axes: [Vec3; 3],
    position: Vec3,
    plane_point: Vec3,
    normal: Vec3,
) -> SubmergedProperties3 {
    try_submerged_properties(shape, position, axes, plane_point, normal).unwrap()
}

fn poses() -> [([Vec3; 3], Vec3); 4] {
    [
        (IDENTITY, [0.0, 0.0, 0.0]),
        (rotation([1.0, 0.0, 0.0], 0.7), [1.0, -2.0, 0.5]),
        (rotation([0.3, 0.2, 1.0], 1.9), [-3.0, 0.25, 4.0]),
        (rotation([1.0, 1.0, -0.5], 2.6), [0.1, 0.2, 0.3]),
    ]
}

#[test]
fn empty_and_full_results_match_volume_properties() {
    for shape in shapes() {
        let properties = try_volume_properties(shape).unwrap();
        let reach = shape.radius();
        for (axes, position) in poses() {
            for normal in [[0.0, 1.0, 0.0], [0.3, -0.8, 0.5]] {
                let unit = normalized(normal).unwrap();
                let surface_above = add(position, scale(unit, reach * 1.001));
                let surface_below = add(position, scale(unit, -reach * 1.001));
                let full = submerged(shape, axes, position, surface_above, normal);
                let expected = add(
                    position,
                    add(
                        add(
                            scale(axes[0], properties.centroid[0]),
                            scale(axes[1], properties.centroid[1]),
                        ),
                        scale(axes[2], properties.centroid[2]),
                    ),
                );
                assert_eq!(full.volume, properties.volume, "{shape:?}");
                assert_eq!(full.centroid, expected, "{shape:?}");
                let empty = submerged(shape, axes, position, surface_below, normal);
                assert_eq!(empty.volume, 0.0, "{shape:?}");
                assert_eq!(empty.centroid, position);
            }
        }
    }
}

#[test]
fn volume_is_continuous_and_monotone_in_depth() {
    for shape in shapes() {
        let total = try_volume_properties(shape).unwrap().volume;
        let reach = shape.radius() * 1.01;
        // Any plane section has area at most pi * reach^2.
        let section = PI * reach * reach;
        for (axes, position) in poses() {
            for normal in [[0.0, 1.0, 0.0], [0.6, 0.64, -0.48], [1.0, 1e-9, 0.0]] {
                let unit = normalized(normal).unwrap();
                let steps = 2000;
                let dz = 2.0 * reach / f64::from(steps);
                let mut previous = 0.0;
                for step in 0..=steps {
                    let level = -reach + dz * f64::from(step);
                    let plane_point = add(position, scale(unit, level));
                    let result = submerged(shape, axes, position, plane_point, normal);
                    let slack = 1e-12 * total;
                    assert!(
                        result.volume + slack >= previous,
                        "{shape:?} level={level} {} < {previous}",
                        result.volume
                    );
                    assert!(
                        result.volume - previous <= section * dz + slack,
                        "{shape:?} jump at level={level}"
                    );
                    assert!(result.volume <= total * (1.0 + 1e-12));
                    previous = result.volume;
                }
                assert_eq!(previous, total);
            }
        }
    }
}

#[test]
fn primitives_match_voxel_integration() {
    let planes = [
        ([0.1, 0.2, -0.1], [0.0, 1.0, 0.0]),
        ([0.0, -0.3, 0.2], [0.4, 0.9, -0.2]),
        ([0.2, 0.35, 0.0], [-0.7, 0.3, 0.6]),
    ];
    let cells = 72;
    for shape in shapes() {
        let total = try_volume_properties(shape).unwrap().volume;
        let reach = shape.radius();
        for (axes, position) in poses() {
            for (offset, normal) in planes {
                let plane_point = add(position, offset);
                let actual = submerged(shape, axes, position, plane_point, normal);
                let (volume, centroid) = voxel_reference(
                    |p| contains(shape, p),
                    shape.half_extents(),
                    axes,
                    position,
                    plane_point,
                    normal,
                    cells,
                );
                assert!(
                    (actual.volume - volume).abs() <= 1e-2 * total,
                    "{shape:?} {} vs voxel {volume}",
                    actual.volume
                );
                assert!(
                    length(sub(actual.centroid, centroid)) <= 1e-2 * reach,
                    "{shape:?} {:?} vs voxel {centroid:?}",
                    actual.centroid
                );
            }
        }
    }
}

fn hull_of(points: &[Vec3]) -> Hull3 {
    let mut work = QuickHullWork::default();
    let mut hull = Hull3::default();
    quickhull3(points, &mut work, &mut hull);
    assert_eq!(hull.dimension, HullDimension::Polytope);
    hull
}

fn hull_points() -> Vec<Vec3> {
    // Deterministic irregular point cloud.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    (0..40)
        .map(|_| [1.2 * next(), 0.8 * next(), next() + 0.3])
        .collect()
}

#[test]
fn hulls_match_voxel_integration_and_box_closed_form() {
    let hull = hull_of(&hull_points());
    let planes_by_hull: Vec<_> = hull
        .triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| hull.vertices[i as usize]);
            let n = cross(sub(b, a), sub(c, a));
            (n, dot(n, a))
        })
        .collect();
    let inside = |p: Vec3| planes_by_hull.iter().all(|&(n, d)| dot(n, p) <= d);
    for (axes, position) in poses() {
        for (offset, normal) in [
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.3, -0.2, 0.5], [0.5, 0.5, 0.7]),
        ] {
            let plane_point = add(position, offset);
            let actual =
                try_hull_submerged_properties(&hull, position, axes, plane_point, normal).unwrap();
            // The voxel box must contain the hull: use the vertex bounds about the origin.
            let half = hull.vertices.iter().fold([0.0; 3], |acc, v| {
                [
                    f64::max(acc[0], v[0].abs()),
                    f64::max(acc[1], v[1].abs()),
                    f64::max(acc[2], v[2].abs()),
                ]
            });
            let (volume, centroid) =
                voxel_reference(inside, half, axes, position, plane_point, normal, 80);
            assert!(
                (actual.volume - volume).abs() <= 4e-3 * 8.0 * half[0] * half[1] * half[2],
                "{} vs voxel {volume}",
                actual.volume
            );
            assert!(length(sub(actual.centroid, centroid)) <= 1e-2);
        }
    }

    // A box hull agrees with the box primitive to rounding.
    let corners: Vec<Vec3> = (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.5 } else { 1.5 },
            ]
        })
        .collect();
    let box_hull = hull_of(&corners);
    let shape = PrimitiveShape3::cuboid([0.5, 1.0, 1.5]);
    for (axes, position) in poses() {
        for level in [-1.4, -0.3, 0.0, 0.9, 1.7] {
            let normal = [0.2, 1.0, -0.3];
            let plane_point = add(position, scale(normalized(normal).unwrap(), level));
            let hull_result =
                try_hull_submerged_properties(&box_hull, position, axes, plane_point, normal)
                    .unwrap();
            let box_result = submerged(shape, axes, position, plane_point, normal);
            assert!((hull_result.volume - box_result.volume).abs() <= 1e-12 * 6.0);
            assert!(length(sub(hull_result.centroid, box_result.centroid)) <= 1e-12 * 4.0);
        }
    }
    // Fully submerged hull volume equals the box volume.
    let full = try_hull_submerged_properties(
        &box_hull,
        [0.0; 3],
        IDENTITY,
        [0.0, 10.0, 0.0],
        [0.0, 1.0, 0.0],
    )
    .unwrap();
    assert!((full.volume - 6.0).abs() <= 1e-12);
    assert!(length(full.centroid) <= 1e-12);
}

/// `integral phi(u) du` with `phi(u) = acos(u) - u sqrt(1-u^2)` (pi below -1, 0 above 1).
fn segment_area(u: f64) -> f64 {
    let u = u.clamp(-1.0, 1.0);
    u.acos() - u * ((1.0 - u) * (1.0 + u)).sqrt()
}

/// Independent adaptive Simpson integration of capsule slices along local Y.
fn capsule_reference(segment: f64, r: f64, n: Vec3, level: f64) -> (f64, Vec3) {
    let cos = n[1];
    let tangent = [n[0], 0.0, n[2]];
    let sin = length(tangent);
    let lateral_dir = scale(tangent, 1.0 / sin);
    // Returns [area, area*y, lateral first moment].
    let slice = |y: f64| -> [f64; 3] {
        let rho2 = if y.abs() <= segment {
            r * r
        } else {
            let d = y.abs() - segment;
            (r * r - d * d).max(0.0)
        };
        let rho = rho2.sqrt();
        if rho == 0.0 {
            return [0.0; 3];
        }
        let u = ((cos * y - level) / (rho * sin)).clamp(-1.0, 1.0);
        let area = rho2 * segment_area(u);
        let chord = rho2 * (1.0 - u * u);
        [area, area * y, -2.0 * chord * chord.sqrt() / 3.0]
    };
    #[allow(clippy::too_many_arguments, reason = "recursive Simpson state")]
    fn simpson(
        f: &dyn Fn(f64) -> [f64; 3],
        a: f64,
        b: f64,
        fa: [f64; 3],
        fm: [f64; 3],
        fb: [f64; 3],
        whole: [f64; 3],
        tol: f64,
        depth: u32,
    ) -> [f64; 3] {
        let m = 0.5 * (a + b);
        let (lm, rm) = (0.5 * (a + m), 0.5 * (m + b));
        let (flm, frm) = (f(lm), f(rm));
        let h = (b - a) / 12.0;
        let left: [f64; 3] = std::array::from_fn(|i| h * (fa[i] + 4.0 * flm[i] + fm[i]));
        let right: [f64; 3] = std::array::from_fn(|i| h * (fm[i] + 4.0 * frm[i] + fb[i]));
        let error = (0..3)
            .map(|i| (left[i] + right[i] - whole[i]).abs())
            .fold(0.0, f64::max);
        if depth == 0 || error <= 15.0 * tol {
            return std::array::from_fn(|i| left[i] + right[i]);
        }
        let l = simpson(f, a, m, fa, flm, fm, left, tol / 2.0, depth - 1);
        let r = simpson(f, m, b, fm, frm, fb, right, tol / 2.0, depth - 1);
        std::array::from_fn(|i| l[i] + r[i])
    }
    let mut total = [0.0; 3];
    // Split at the cylinder's slice-type breakpoints (u = +-1) as well.
    let mut bounds = vec![-segment - r, -segment, segment, segment + r];
    if cos != 0.0 {
        for edge in [-r * sin, r * sin] {
            let t = (edge + level) / cos;
            if t.abs() < segment {
                bounds.push(t);
            }
        }
    }
    bounds.sort_by(f64::total_cmp);
    for window in bounds.windows(2) {
        let (lo, hi) = (window[0], window[1]);
        if hi <= lo {
            continue;
        }
        // Uniform panels first, so narrow submerged slivers cannot hide from
        // the adaptive error estimate.
        let panels = 64;
        for panel in 0..panels {
            let a = lo + (hi - lo) * f64::from(panel) / f64::from(panels);
            let b = lo + (hi - lo) * f64::from(panel + 1) / f64::from(panels);
            let (fa, fm, fb) = (slice(a), slice(0.5 * (a + b)), slice(b));
            let whole: [f64; 3] =
                std::array::from_fn(|i| (b - a) / 6.0 * (fa[i] + 4.0 * fm[i] + fb[i]));
            let part = simpson(&slice, a, b, fa, fm, fb, whole, 1e-13, 30);
            total = add(total, part);
        }
    }
    let volume = total[0];
    let first = add([0.0, total[1], 0.0], scale(lateral_dir, total[2]));
    (volume, scale(first, 1.0 / volume))
}

#[test]
fn capsules_match_adaptive_slice_integration() {
    for (segment, r) in [(1.0, 0.4), (10.0, 1.0), (0.05, 2.0)] {
        let total = try_volume_properties(PrimitiveShape3::capsule(segment, r))
            .unwrap()
            .volume;
        for normal in [
            [0.0, 1.0, 0.0001],
            [0.3, 0.9, 0.1],
            [1.0, 0.2, -0.4],
            [0.7, -0.7, 0.0],
            [1.0, 1e-6, 0.0],
        ] {
            let n = normalized(normal).unwrap();
            let reach = n[1].abs() * segment + r;
            for fraction in [-0.95, -0.6, -0.2, 0.0, 0.33, 0.8, 0.99] {
                let level = fraction * reach;
                let actual = submerged(
                    PrimitiveShape3::capsule(segment, r),
                    IDENTITY,
                    [0.0; 3],
                    scale(n, level),
                    normal,
                );
                let (volume, centroid) = capsule_reference(segment, r, n, level);
                assert!(
                    (actual.volume - volume).abs() <= 1e-9 * total,
                    "segment={segment} r={r} n={normal:?} level={level}: {} vs {volume}",
                    actual.volume
                );
                assert!(
                    length(sub(actual.centroid, centroid)) <= 1e-8 * (segment + r),
                    "segment={segment} r={r} n={normal:?} level={level}: {:?} vs {centroid:?}",
                    actual.centroid
                );
            }
        }
    }
}

#[test]
fn zero_segment_capsule_matches_sphere_cap() {
    for (axes, position) in poses() {
        for normal in [[0.0, 1.0, 0.0], [0.3, -0.8, 0.5], [1.0, 0.0, 0.0]] {
            let unit = normalized(normal).unwrap();
            for level in [-0.69, -0.3, 0.0, 0.2, 0.65] {
                let plane_point = add(position, scale(unit, level));
                let sphere = submerged(
                    PrimitiveShape3::sphere(0.7),
                    axes,
                    position,
                    plane_point,
                    normal,
                );
                let capsule = submerged(
                    PrimitiveShape3::capsule(0.0, 0.7),
                    axes,
                    position,
                    plane_point,
                    normal,
                );
                assert!((sphere.volume - capsule.volume).abs() <= 1e-13);
                assert!(length(sub(sphere.centroid, capsule.centroid)) <= 1e-12);
            }
        }
    }
}

#[test]
fn sphere_cap_is_closed_form() {
    let r = 2.0;
    for depth in [0.1, 1.0, 2.0, 3.5] {
        let result = submerged(
            PrimitiveShape3::sphere(r),
            IDENTITY,
            [1.0, 2.0, 3.0],
            [0.0, 2.0 - r + depth, 0.0],
            [0.0, 1.0, 0.0],
        );
        let expected = PI * depth * depth * (3.0 * r - depth) / 3.0;
        assert!((result.volume - expected).abs() <= 1e-13 * expected);
        let z = 3.0 * (2.0 * r - depth).powi(2) / (4.0 * (3.0 * r - depth));
        assert!((result.centroid[1] - (2.0 - z)).abs() <= 1e-13);
        assert_eq!([result.centroid[0], result.centroid[2]], [1.0, 3.0]);
    }
}

#[test]
fn symmetric_cases_give_symmetric_centroids() {
    let position = [2.0, -1.0, 0.5];
    let up = [0.0, 1.0, 0.0];
    let turned = rotation(up, 0.6);
    // Bodies symmetric about the vertical through their position.
    for shape in [
        PrimitiveShape3::sphere(0.8),
        PrimitiveShape3::cuboid([0.5, 1.0, 1.5]),
        PrimitiveShape3::capsule(1.0, 0.4),
    ] {
        for axes in [IDENTITY, turned] {
            for level in [-0.5, 0.0, 0.3] {
                let result = submerged(shape, axes, position, [0.0, level - 1.0, 0.0], up);
                assert!(
                    (result.centroid[0] - position[0]).abs() <= 1e-12,
                    "{shape:?}"
                );
                assert!(
                    (result.centroid[2] - position[2]).abs() <= 1e-12,
                    "{shape:?}"
                );
            }
        }
    }
    // A horizontal capsule's centroid stays at the axial midpoint.
    let lying = rotation([0.0, 0.0, 1.0], PI / 2.0);
    let result = submerged(
        PrimitiveShape3::capsule(2.0, 0.5),
        lying,
        [0.0; 3],
        [0.0, 0.2, 0.0],
        up,
    );
    assert!(result.centroid[0].abs() <= 1e-12 && result.centroid[2].abs() <= 1e-12);
    // Mirror-image tilts give mirrored centroids.
    for shape in shapes() {
        let left = submerged(
            shape,
            rotation([0.0, 0.0, 1.0], 0.4),
            [0.0; 3],
            [0.0, 0.1, 0.0],
            up,
        );
        let right = submerged(
            shape,
            rotation([0.0, 0.0, 1.0], -0.4),
            [0.0; 3],
            [0.0, 0.1, 0.0],
            up,
        );
        if shape.kind() == PrimitiveKind3::Wedge {
            continue; // the wedge itself is not mirror-symmetric in X
        }
        assert!((left.volume - right.volume).abs() <= 1e-12);
        assert!((left.centroid[0] + right.centroid[0]).abs() <= 1e-12);
        assert!((left.centroid[1] - right.centroid[1]).abs() <= 1e-12);
    }
}

#[test]
fn results_are_deterministic() {
    for shape in shapes() {
        for (axes, position) in poses() {
            let first = submerged(shape, axes, position, [0.1, 0.2, 0.3], [0.2, 1.0, 0.1]);
            let second = submerged(shape, axes, position, [0.1, 0.2, 0.3], [0.2, 1.0, 0.1]);
            assert_eq!(first.volume.to_bits(), second.volume.to_bits());
            assert_eq!(
                first.centroid.map(f64::to_bits),
                second.centroid.map(f64::to_bits)
            );
        }
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let shape = PrimitiveShape3::cuboid([1.0; 3]);
    let invalid = Err(SubmergedError3::InvalidInput);
    let call = |position: Vec3, axes: [Vec3; 3], point: Vec3, normal: Vec3| {
        try_submerged_properties(shape, position, axes, point, normal)
    };
    assert_eq!(
        call([f64::NAN, 0.0, 0.0], IDENTITY, [0.0; 3], [0.0, 1.0, 0.0]),
        invalid
    );
    assert_eq!(
        call(
            [0.0; 3],
            IDENTITY,
            [f64::INFINITY, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        invalid
    );
    assert_eq!(call([0.0; 3], IDENTITY, [0.0; 3], [0.0; 3]), invalid);
    assert_eq!(
        call([0.0; 3], IDENTITY, [0.0; 3], [0.0, f64::NAN, 0.0]),
        invalid
    );
    let skewed = [[1.0, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(call([0.0; 3], skewed, [0.0; 3], [0.0, 1.0, 0.0]), invalid);
    let scaled = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(call([0.0; 3], scaled, [0.0; 3], [0.0, 1.0, 0.0]), invalid);
    // Overflowing level is a computation failure, not a silent result.
    assert_eq!(
        call(
            [-f64::MAX, 0.0, 0.0],
            IDENTITY,
            [f64::MAX, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        Err(SubmergedError3::NonFiniteComputation)
    );

    let hull_call = |hull: &Hull3| {
        try_hull_submerged_properties(hull, [0.0; 3], IDENTITY, [0.0; 3], [0.0, 1.0, 0.0])
    };
    let mut flat = Hull3::default();
    quickhull3(
        &[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        &mut QuickHullWork::default(),
        &mut flat,
    );
    assert_eq!(hull_call(&flat), invalid);
    assert_eq!(hull_call(&Hull3::default()), invalid);
    let mut broken = hull_of(&hull_points());
    broken.triangles[0][1] = broken.vertices.len() as u32;
    assert_eq!(hull_call(&broken), invalid);
    let mut nan = hull_of(&hull_points());
    nan.vertices[3][0] = f64::NAN;
    assert_eq!(hull_call(&nan), invalid);
}
