use divan::{Bencher, black_box};
use geometry_kernels::{
    Ray3, Sphere, ray_aabb, ray_sphere, sphere_sphere_time_of_impact,
};
use spatial_kernels::Aabb;

fn main() {
    divan::main();
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
