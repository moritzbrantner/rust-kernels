use divan::{Bencher, black_box};
use geometry_kernels::primitive3::{
    PrimitiveBody3, PrimitiveRayWork3, PrimitiveShape3, try_ray_cast,
};
use geometry_kernels::{
    Ray3, Sphere, TriangleCulling, ray_aabb, ray_sphere, ray_triangle, sphere_sphere_time_of_impact,
};
use spatial_kernels::Aabb;

fn main() {
    divan::main();
}

#[divan::bench(args = [0, 1, 2, 3])]
fn f64_primitive_segment_ray(bencher: Bencher, kind: u8) {
    let shape = match kind {
        0 => PrimitiveShape3::sphere(1.0),
        1 => PrimitiveShape3::cuboid([1.0; 3]),
        2 => PrimitiveShape3::capsule(2.0, 1.0),
        _ => PrimitiveShape3::wedge([1.0; 3]),
    };
    let body = PrimitiveBody3::axis_aligned(shape, [0.0; 3], [0.0; 3]);
    bencher.bench_local(|| {
        let mut work = PrimitiveRayWork3::default();
        black_box(try_ray_cast(
            black_box(body),
            [-3.0, -0.5, 0.0],
            [6.0, 0.0, 0.0],
            &mut work,
        ))
    });
}

fn boxes(n: usize) -> Vec<Aabb> {
    (0..n)
        .map(|i| {
            let x = (i % 64) as f32 * 1.5;
            let y = ((i / 64) % 16) as f32 * 1.5;
            let z = (i / 1024) as f32 * 1.5;
            Aabb::from_center_half_extents([x, y, z], [0.45; 3])
        })
        .collect()
}

fn spheres(n: usize) -> Vec<Sphere> {
    (0..n)
        .map(|i| {
            let x = (i % 64) as f32 * 1.5;
            let y = ((i / 64) % 16) as f32 * 1.5;
            let z = (i / 1024) as f32 * 1.5;
            Sphere::new([x, y, z], 0.45)
        })
        .collect()
}

#[divan::bench(args = [64, 1024, 4096])]
fn ray_aabb_batch(bencher: Bencher, n: usize) {
    let values = boxes(n);
    let ray = Ray3::new([-10.0, 0.25, 0.25], [1.0, 0.01, 0.005]);
    bencher.bench_local(|| {
        let checksum = black_box(&values)
            .iter()
            .filter_map(|aabb| ray_aabb(black_box(ray), black_box(*aabb)))
            .map(|hit| hit.enter_distance)
            .sum::<f64>();
        black_box(checksum)
    });
}

#[divan::bench(args = [64, 1024, 4096])]
fn ray_sphere_batch(bencher: Bencher, n: usize) {
    let values = spheres(n);
    let ray = Ray3::new([-10.0, 0.25, 0.25], [1.0, 0.01, 0.005]);
    bencher.bench_local(|| {
        let checksum = black_box(&values)
            .iter()
            .filter_map(|sphere| ray_sphere(black_box(ray), black_box(*sphere)))
            .map(|hit| hit.enter_distance)
            .sum::<f64>();
        black_box(checksum)
    });
}

#[divan::bench(args = [64, 1024, 4096])]
fn sphere_sweep_batch(bencher: Bencher, n: usize) {
    let targets = spheres(n);
    let moving = Sphere::new([-10.0, 0.25, 0.25], 0.3);
    bencher.bench_local(|| {
        let checksum = black_box(&targets)
            .iter()
            .filter_map(|target| {
                sphere_sphere_time_of_impact(
                    black_box(moving),
                    black_box([5.0, 0.05, 0.025]),
                    black_box(*target),
                    black_box(40.0),
                )
            })
            .map(|hit| hit.time)
            .sum::<f64>();
        black_box(checksum)
    });
}

/// Unit right triangles on a grid in the z = 0 plane; `hit_share` of them lie
/// under the ray footprint (the rest are shifted out of the way).
fn triangles(n: usize, hit_share: usize) -> Vec<[[f32; 3]; 3]> {
    (0..n)
        .map(|i| {
            let x = (i % 64) as f32 * 1.5;
            let y = ((i / 64) % 64) as f32 * 1.5;
            // Every fourth slot is a candidate hit for hit_share = 1, all for 4.
            let offset = if i % 4 < hit_share { 0.0 } else { 0.9 };
            [
                [x + offset, y + offset, 0.0],
                [x + 1.0 + offset, y + offset, 0.0],
                [x + offset, y + 1.0 + offset, 0.0],
            ]
        })
        .collect()
}

/// Hit-heavy (4/4), mixed (2/4) and miss-heavy (0/4) batches: one ray per
/// triangle aimed at the point (0.25, 0.25) of its unshifted slot.
#[divan::bench(args = [64, 1024, 4096], consts = [0, 2, 4])]
fn ray_triangle_batch<const HIT_SHARE: usize>(bencher: Bencher, n: usize) {
    let values = triangles(n, HIT_SHARE);
    let rays: Vec<Ray3> = (0..n)
        .map(|i| {
            let x = (i % 64) as f32 * 1.5 + 0.25;
            let y = ((i / 64) % 64) as f32 * 1.5 + 0.25;
            Ray3::new([x, y, 2.0], [0.01, 0.02, -1.0])
        })
        .collect();
    bencher.bench_local(|| {
        let checksum = black_box(&values)
            .iter()
            .zip(black_box(&rays))
            .filter_map(|(triangle, ray)| {
                ray_triangle(*ray, *triangle, f64::INFINITY, TriangleCulling::BackFaces)
            })
            .map(|hit| hit.distance)
            .sum::<f64>();
        black_box(checksum)
    });
}
