use bvh_kernels::DynamicAabbTree;
use divan::{Bencher, black_box};
use spatial_kernels::{Aabb, Body, BroadPhase, NaiveBroadPhase};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    if std::env::args().any(|arg| arg == "--audit-probe") {
        probe();
    } else {
        divan::main();
    }
}

fn clustered_scene(n: usize) -> Vec<Body> {
    (0..n)
        .map(|id| {
            let cluster = id % 8;
            let local = id / 8;
            let base = [
                if cluster & 1 == 0 { -40.0 } else { 40.0 },
                if cluster & 2 == 0 { -40.0 } else { 40.0 },
                if cluster & 4 == 0 { -40.0 } else { 40.0 },
            ];
            let center = [
                base[0] + (local % 8) as f32 * 0.8,
                base[1] + ((local / 8) % 8) as f32 * 0.8,
                base[2] + (local / 64) as f32 * 0.8,
            ];
            Body::new(
                id as u32,
                Aabb::from_center_half_extents(center, [0.5; 3]),
            )
        })
        .collect()
}

fn retained_tree(bodies: &[Body]) -> DynamicAabbTree {
    let mut tree = DynamicAabbTree::new(1.25);
    for body in bodies {
        tree.insert(*body);
    }
    tree
}

#[divan::bench(args = [256, 1024, 3000])]
fn retained_pair_query(bencher: Bencher, n: usize) {
    let bodies = clustered_scene(n);
    let expected = NaiveBroadPhase.detect(&bodies).pairs;
    let tree = retained_tree(&bodies);
    assert_eq!(tree.overlapping_pairs_result().pairs, expected);
    bencher.bench_local(|| black_box(tree.overlapping_pairs_result()));
}

fn probe() {
    for n in [256, 1024, 3000] {
        let bodies = clustered_scene(n);
        let naive = NaiveBroadPhase.detect(&bodies);
        let tree = retained_tree(&bodies);
        let result = tree.overlapping_pairs_result();
        assert_eq!(result.pairs, naive.pairs);
        println!("AUDIT\tdynamic_clustered_{n}\tpairs\t{}", result.pairs.len());
        println!(
            "AUDIT\tdynamic_clustered_{n}\taabb_tests\t{}",
            result.stats.aabb_tests
        );
        println!(
            "AUDIT\tdynamic_clustered_{n}\tnodes\t{}",
            tree.node_count()
        );
        println!("AUDIT\tdynamic_clustered_{n}\theight\t{}", tree.height());
    }
}
