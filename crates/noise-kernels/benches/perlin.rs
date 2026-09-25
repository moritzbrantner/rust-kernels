//! Workload v1. Input generation is excluded from measured sampling.
use divan::{Bencher, black_box, counter::ItemsCount};
use noise_kernels::{Permutation, perlin2, perlin3};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

fn points2(count: usize, coherent: bool) -> Vec<[f64; 2]> {
    (0..count)
        .map(|index| {
            if coherent {
                let side = (count as f64).sqrt().ceil() as usize;
                [
                    (index % side) as f64 * 0.03125,
                    (index / side) as f64 * 0.03125,
                ]
            } else {
                let value = index as u64;
                [
                    ((value.wrapping_mul(73_856_093) & 0xffff) as f64 - 32_768.0) / 64.0,
                    ((value.wrapping_mul(19_349_663) & 0xffff) as f64 - 32_768.0) / 64.0,
                ]
            }
        })
        .collect()
}

fn points3(count: usize, coherent: bool) -> Vec<[f64; 3]> {
    (0..count)
        .map(|index| {
            if coherent {
                let side = (count as f64).cbrt().ceil() as usize;
                let plane = side * side;
                [
                    (index % side) as f64 * 0.0625,
                    ((index / side) % side) as f64 * 0.0625,
                    (index / plane) as f64 * 0.0625,
                ]
            } else {
                let value = index as u64;
                [
                    ((value.wrapping_mul(73_856_093) & 0xffff) as f64 - 32_768.0) / 64.0,
                    ((value.wrapping_mul(19_349_663) & 0xffff) as f64 - 32_768.0) / 64.0,
                    ((value.wrapping_mul(83_492_791) & 0xffff) as f64 - 32_768.0) / 64.0,
                ]
            }
        })
        .collect()
}

#[divan::bench(args = [64, 4096, 65536])]
fn perlin2_scattered(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);
    let points = points2(count, false);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for point in black_box(&points) {
            sum += perlin2(black_box(&permutation), point[0], point[1]);
        }
        black_box(sum)
    });
}

#[divan::bench(args = [64, 4096, 65536])]
fn perlin2_coherent_grid(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);
    let points = points2(count, true);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for point in black_box(&points) {
            sum += perlin2(black_box(&permutation), point[0], point[1]);
        }
        black_box(sum)
    });
}

#[divan::bench(args = [64, 4096, 32768])]
fn perlin3_scattered(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);
    let points = points3(count, false);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for point in black_box(&points) {
            sum += perlin3(black_box(&permutation), point[0], point[1], point[2]);
        }
        black_box(sum)
    });
}

#[divan::bench(args = [64, 4096, 32768])]
fn perlin3_coherent_grid(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);
    let points = points3(count, true);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for point in black_box(&points) {
            sum += perlin3(black_box(&permutation), point[0], point[1], point[2]);
        }
        black_box(sum)
    });
}
