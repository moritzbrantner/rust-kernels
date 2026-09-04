use iai_callgrind::{
    Callgrind, EventKind, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main,
};
use search_kernels::{quickselect, radix_sort_u32, top_k_smallest};
use std::hint::black_box;

const SMOKE_LEN: usize = 4_096;

fn generated_values() -> Vec<u32> {
    let mut state = 0x1234_5678_u32;
    (0..SMOKE_LEN)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect()
}

#[library_benchmark]
#[bench::radix_4096(generated_values())]
fn bench_radix_sort(mut values: Vec<u32>) -> u32 {
    radix_sort_u32(&mut values);
    black_box(values[values.len() / 2])
}

#[library_benchmark]
#[bench::middle_4096(generated_values())]
fn bench_quickselect(mut values: Vec<u32>) -> u32 {
    let nth = values.len() / 2;
    black_box(*quickselect(&mut values, nth).expect("middle index is in range"))
}

#[library_benchmark]
#[bench::top_40_of_4096(generated_values())]
fn bench_top_k(values: Vec<u32>) -> Vec<u32> {
    black_box(top_k_smallest(&values, 40))
}

library_benchmark_group!(
    name = search_smoke;
    benchmarks = bench_radix_sort, bench_quickselect, bench_top_k
);

fn benchmark_config() -> LibraryBenchmarkConfig {
    let mut callgrind = Callgrind::default();
    callgrind
        .soft_limits([(EventKind::Ir, 5.0)])
        .fail_fast(true);
    LibraryBenchmarkConfig::default().tool(callgrind)
}

main!(
    config = benchmark_config();
    library_benchmark_groups = search_smoke
);
