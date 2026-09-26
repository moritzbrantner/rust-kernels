//! Workload v1. Input generation is excluded from measured sampling.
use divan::{Bencher, black_box, counter::ItemsCount};
use noise_kernels::{Permutation, simplex2, simplex3, simplex4};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

fn point(index: usize, axis: u64) -> f64 {
    let value = (index as u64).wrapping_mul(axis);
    ((value & 0xffff) as f64 - 32_768.0) / 64.0
}

#[divan::bench(args = [64, 4096, 65536])]
fn simplex2_scattered(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += simplex2(
                black_box(&permutation),
                point(index, 73_856_093),
                point(index, 19_349_663),
            );
        }
        black_box(sum)
    });
}

#[divan::bench(args = [64, 4096, 65536])]
fn simplex2_coherent_grid(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);
    let side = (count as f64).sqrt().ceil() as usize;

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += simplex2(
                black_box(&permutation),
                (index % side) as f64 * 0.03125,
                (index / side) as f64 * 0.03125,
            );
        }
        black_box(sum)
    });
}

#[divan::bench(args = [64, 4096, 32768])]
fn simplex3_scattered(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += simplex3(
                black_box(&permutation),
                point(index, 73_856_093),
                point(index, 19_349_663),
                point(index, 83_492_791),
            );
        }
        black_box(sum)
    });
}

#[divan::bench(args = [64, 4096, 32768])]
fn simplex3_coherent_grid(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);
    let side = (count as f64).cbrt().ceil() as usize;
    let plane = side * side;

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += simplex3(
                black_box(&permutation),
                (index % side) as f64 * 0.0625,
                ((index / side) % side) as f64 * 0.0625,
                (index / plane) as f64 * 0.0625,
            );
        }
        black_box(sum)
    });
}

#[divan::bench(args = [64, 4096, 16384])]
fn simplex4_scattered(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += simplex4(
                black_box(&permutation),
                point(index, 73_856_093),
                point(index, 19_349_663),
                point(index, 83_492_791),
                point(index, 49_979_687),
            );
        }
        black_box(sum)
    });
}

#[divan::bench(args = [64, 4096, 16384])]
fn simplex4_time_slice(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            let coordinate = index as f64 * 0.03125;
            sum += simplex4(
                black_box(&permutation),
                coordinate,
                coordinate * 0.5,
                coordinate * 0.25,
                3.5,
            );
        }
        black_box(sum)
    });
}
