//! Workload v1. Input generation is excluded from measured sampling.
//!
//! Fractal scenarios sweep octave counts (const generic) and point batches;
//! warp and curl scenarios sweep point batches over Simplex base fields.
use divan::{Bencher, black_box, counter::ItemsCount};
use noise_kernels::{Curl, DomainWarp, Fractal, Permutation, simplex2, simplex3, worley2};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const SEED: u64 = 0x1020_3040_5060_7080;

fn point(index: usize, axis: u64) -> f64 {
    let value = (index as u64).wrapping_mul(axis);
    ((value & 0xffff) as f64 - 32_768.0) / 1024.0
}

fn fractal<const OCTAVES: u32>() -> Fractal {
    Fractal::new(OCTAVES, 2.0, 0.5).expect("valid schedule")
}

#[divan::bench(consts = [1, 4, 8], args = [1024, 16384])]
fn fbm2_simplex<const OCTAVES: u32>(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(SEED);
    let fractal = fractal::<OCTAVES>();
    let base = |x, y| simplex2(&permutation, x, y);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum +=
                black_box(&fractal).fbm2(&base, point(index, 73_856_093), point(index, 19_349_663));
        }
        black_box(sum)
    });
}

#[divan::bench(consts = [1, 4, 8], args = [1024, 16384])]
fn ridged2_simplex_coherent_grid<const OCTAVES: u32>(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(SEED);
    let fractal = fractal::<OCTAVES>();
    let base = |x, y| simplex2(&permutation, x, y);
    let side = (count as f64).sqrt().ceil() as usize;

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += black_box(&fractal).ridged2(
                &base,
                (index % side) as f64 * 0.03125,
                (index / side) as f64 * 0.03125,
            );
        }
        black_box(sum)
    });
}

#[divan::bench(consts = [1, 4, 8], args = [1024, 16384])]
fn billow3_simplex<const OCTAVES: u32>(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(SEED);
    let fractal = fractal::<OCTAVES>();
    let base = |x, y, z| simplex3(&permutation, x, y, z);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += black_box(&fractal).billow3(
                &base,
                point(index, 73_856_093),
                point(index, 19_349_663),
                point(index, 83_492_791),
            );
        }
        black_box(sum)
    });
}

#[divan::bench(consts = [1, 4, 8], args = [1024, 16384])]
fn fbm2_worley_f1<const OCTAVES: u32>(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(SEED);
    let fractal = fractal::<OCTAVES>();
    let base = |x, y| worley2(&permutation, x, y).f1;

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum +=
                black_box(&fractal).fbm2(&base, point(index, 73_856_093), point(index, 19_349_663));
        }
        black_box(sum)
    });
}

#[divan::bench(args = [1024, 16384])]
fn warp2_simplex(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(SEED);
    let warp = DomainWarp::new(1.5).expect("valid amplitude");
    let base = |x, y| simplex2(&permutation, x, y);
    let offset_x = |x, y| simplex2(&permutation, x + 17.0, y);
    let offset_y = |x, y| simplex2(&permutation, x, y - 31.0);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += black_box(&warp).sample2(
                &base,
                &offset_x,
                &offset_y,
                point(index, 73_856_093),
                point(index, 19_349_663),
            );
        }
        black_box(sum)
    });
}

#[divan::bench(args = [1024, 16384])]
fn warp3_simplex(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(SEED);
    let warp = DomainWarp::new(1.5).expect("valid amplitude");
    let base = |x, y, z| simplex3(&permutation, x, y, z);
    let offset_x = |x, y, z| simplex3(&permutation, x + 17.0, y, z);
    let offset_y = |x, y, z| simplex3(&permutation, x, y - 31.0, z);
    let offset_z = |x, y, z| simplex3(&permutation, x, y, z + 7.0);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            sum += black_box(&warp).sample3(
                &base,
                &offset_x,
                &offset_y,
                &offset_z,
                point(index, 73_856_093),
                point(index, 19_349_663),
                point(index, 83_492_791),
            );
        }
        black_box(sum)
    });
}

#[divan::bench(args = [1024, 16384])]
fn curl2_simplex(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(SEED);
    let curl = Curl::new(1e-3).expect("valid epsilon");
    let potential = |x, y| simplex2(&permutation, x, y);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            let [vx, vy] = black_box(&curl).curl2(
                &potential,
                point(index, 73_856_093),
                point(index, 19_349_663),
            );
            sum += vx + vy;
        }
        black_box(sum)
    });
}

#[divan::bench(args = [1024, 16384])]
fn curl3_simplex(bencher: Bencher, count: usize) {
    let permutation = Permutation::from_seed(SEED);
    let curl = Curl::new(1e-3).expect("valid epsilon");
    let potential_x = |x, y, z| simplex3(&permutation, x, y, z);
    let potential_y = |x, y, z| simplex3(&permutation, x + 31.0, y, z);
    let potential_z = |x, y, z| simplex3(&permutation, x, y + 17.0, z);

    bencher.counter(ItemsCount::new(count)).bench_local(|| {
        let mut sum = 0.0;
        for index in 0..black_box(count) {
            let [vx, vy, vz] = black_box(&curl).curl3(
                &potential_x,
                &potential_y,
                &potential_z,
                point(index, 73_856_093),
                point(index, 19_349_663),
                point(index, 83_492_791),
            );
            sum += vx + vy + vz;
        }
        black_box(sum)
    });
}
