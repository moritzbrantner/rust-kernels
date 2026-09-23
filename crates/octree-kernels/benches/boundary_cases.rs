//! Boundary workload v1; input setup and oracle checks are outside timing.
use divan::{Bencher, black_box};
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();
fn emit(case: &str, metric: &str, value: impl std::fmt::Debug) {
    println!("AUDIT\t{case}\t{metric}\t{value:?}");
}
fn main() {
    if std::env::args().any(|arg| arg == "--audit-probe") {
        probe();
    } else {
        divan::main();
    }
}

use octree_kernels::OctreeBroadPhase;
use spatial_kernels::{Aabb, Body, BroadPhase, NaiveBroadPhase};

fn scene(n: usize) -> Vec<Body> {
    (0..n)
        .map(|id| {
            let cell = id / 2;
            let center = [
                (cell % 8) as f32 * 4.0,
                ((cell / 8) % 8) as f32 * 4.0,
                (cell / 64) as f32 * 4.0,
            ];
            Body::new(id as u32, Aabb::from_center_half_extents(center, [0.5; 3]))
        })
        .collect()
}

#[divan::bench(args = [64, 256, 1024])]
fn detect_pairs(bencher: Bencher, n: usize) {
    let bodies = scene(n);
    let tree = OctreeBroadPhase::new(6, 4);
    assert_eq!(
        tree.detect(&bodies).pairs,
        NaiveBroadPhase.detect(&bodies).pairs
    );
    bencher.bench_local(|| black_box(tree.detect(black_box(&bodies))));
}

#[divan::bench(args = [64, 256, 1024])]
fn trace_pairs(bencher: Bencher, n: usize) {
    let bodies = scene(n);
    let tree = OctreeBroadPhase::new(6, 4);
    assert_eq!(tree.trace(&bodies).result, tree.detect(&bodies));
    bencher.bench_local(|| black_box(tree.trace(black_box(&bodies))));
}

fn probe() {
    for n in [64, 256, 1024] {
        let bodies = scene(n);
        let tree = OctreeBroadPhase::new(6, 4);
        let trace = tree.trace(&bodies);
        assert_eq!(trace.result, tree.detect(&bodies));
        assert_eq!(trace.result.pairs, NaiveBroadPhase.detect(&bodies).pairs);
        emit(&format!("octree_{n}"), "pairs", trace.result.pairs.len());
        emit(
            &format!("octree_{n}"),
            "aabb_tests",
            trace.result.stats.aabb_tests,
        );
        emit(&format!("octree_{n}"), "nodes", trace.nodes.len());
        emit(
            &format!("octree_{n}"),
            "checksum",
            trace
                .result
                .pairs
                .iter()
                .map(|p| u64::from(p.a) * 1009 + u64::from(p.b))
                .sum::<u64>(),
        );
    }
    let point = |x| Aabb::new([x, 0.0, 0.0], [x, 0.0, 0.0]);
    let bodies = [
        Body::new(0, point(-3668.735)),
        Body::new(1, point(1.05e-10)),
        Body::new(2, point(1.05e-10)),
    ];
    emit(
        "octree_asymmetric",
        "pairs",
        OctreeBroadPhase::new(6, 1).detect(&bodies).pairs.len(),
    );
}
