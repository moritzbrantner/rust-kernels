//! First-hit segment queries over a static town of boxes: the binary `DynamicAabbTree` versus
//! the four-wide `WideBvh`, and the cost of building the wide one.
use bvh_kernels::{DynamicAabbTree, WideBvh};
use divan::{Bencher, black_box};
use spatial_kernels::{Aabb, Body};

fn main() {
    divan::main();
}

/// 24 x 24 buildings with pillars and platforms, about 1,150 boxes.
fn town() -> Vec<Body> {
    let mut boxes = vec![([0.0, -0.5, 0.0], [130.0, 0.5, 130.0])];
    let cell = |i: usize| {
        (
            -115.0 + 10.0 * (i % 24) as f32,
            -115.0 + 10.0 * (i / 24) as f32,
        )
    };
    for i in 0..576 {
        let (x, z) = cell(i);
        let h = 1.0 + (i * 7 % 13) as f32 * 0.25;
        boxes.push(([x, h, z], [3.0, h, 3.0]));
    }
    for i in 0..400 {
        let (x, z) = cell(i);
        boxes.push(([x + 4.2, 1.5, z + 4.2], [0.3, 1.5, 0.3]));
    }
    for i in 0..174 {
        let (x, z) = cell(i);
        boxes.push(([x + 5.0, 0.75, z], [1.0, 0.25, 1.0]));
    }
    boxes
        .into_iter()
        .enumerate()
        .map(|(id, (c, h))| Body::new(id as u32, Aabb::from_center_half_extents(c, h)))
        .collect()
}

fn tree() -> DynamicAabbTree {
    let mut tree = DynamicAabbTree::new(0.0);
    for b in town() {
        tree.insert(b);
    }
    tree.rebuild();
    tree
}

fn rays() -> Vec<([f32; 3], [f32; 3])> {
    (0..256)
        .map(|k| {
            let a = k as f32 * 0.613;
            let (x, z) = ((k * 37 % 230) as f32 - 115.0, (k * 91 % 230) as f32 - 115.0);
            if k % 4 == 0 {
                ([x, 12.0, z], [0.0, -14.0, 0.0])
            } else {
                let l = 5.0 + (k % 35) as f32;
                ([x, 1.5, z], [a.cos() * l, 0.0, a.sin() * l])
            }
        })
        .collect()
}

fn first(t: f32, best: &mut f32) -> f32 {
    *best = best.min(t);
    *best
}

#[divan::bench]
fn dynamic_tree(bencher: Bencher) {
    let tree = tree();
    let rays = rays();
    bencher.bench_local(|| {
        let mut nodes = 0;
        for &(o, d) in &rays {
            let mut best = 1.0;
            nodes += tree.visit_segment(o, d, 1.0, 1e-4, |_, t| first(t, &mut best));
        }
        black_box(nodes)
    });
}

#[divan::bench]
fn wide_bvh(bencher: Bencher) {
    let tree = WideBvh::build(&town());
    let rays = rays();
    bencher.bench_local(|| {
        let mut nodes = 0;
        for &(o, d) in &rays {
            let mut best = 1.0;
            nodes += tree.visit_segment(o, d, 1.0, 1e-4, |_, t| first(t, &mut best));
        }
        black_box(nodes)
    });
}

#[divan::bench]
fn wide_build(bencher: Bencher) {
    let town = town();
    bencher.bench_local(|| black_box(WideBvh::build(&town)));
}
