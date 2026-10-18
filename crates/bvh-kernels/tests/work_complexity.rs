//! Multi-scale work evidence for `DynamicAabbTree` queries: nodes visited per small box query
//! and per short segment over a rebuilt grid of unit boxes.
//!
//! The scale series feeds the `bvh-aabb-query-nodes` and `bvh-segment-query-nodes` operations
//! in `.performance/contract.json` through `scripts/collect-work-evidence.sh`.
use bvh_kernels::DynamicAabbTree;
use spatial_kernels::{Aabb, Body};

/// One scale point: `(dimensions, metrics)` as `(name, value)` pairs.
type WorkSample = (Vec<(&'static str, usize)>, Vec<(&'static str, usize)>);

fn work_evidence_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RUST_KERNELS_WORK_EVIDENCE_DIR").map(Into::into)
}

fn write_work_evidence(dir: &std::path::Path, operation: &str, samples: &[WorkSample]) {
    fn fields(pairs: &[(&str, usize)]) -> String {
        pairs
            .iter()
            .map(|(name, value)| format!("\"{name}\":{value}"))
            .collect::<Vec<_>>()
            .join(",")
    }
    let samples = samples
        .iter()
        .map(|(dimensions, metrics)| {
            format!(
                "{{\"dimensions\":{{{}}},\"metrics\":{{{}}}}}",
                fields(dimensions),
                fields(metrics)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(
        dir.join(format!("{operation}.json")),
        format!("{{\"id\":\"{operation}\",\"samples\":[{samples}]}}\n"),
    )
    .expect("work evidence directory must be writable");
}

/// Powers of two from 512 to 65536 leaves.
const LEAVES: [usize; 8] = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];
const QUERIES: usize = 64;

/// A square-ish grid of unit boxes two units apart.
fn grid(leaves: usize) -> (usize, Vec<Body>) {
    let columns = 1 << (leaves.trailing_zeros().div_ceil(2));
    let bodies = (0..leaves)
        .map(|i| {
            let (x, z) = ((i % columns) as f32 * 2.0, (i / columns) as f32 * 2.0);
            Body {
                id: i as u32,
                aabb: Aabb {
                    min: [x - 0.5, 0.0, z - 0.5],
                    max: [x + 0.5, 1.0, z + 0.5],
                },
            }
        })
        .collect();
    (columns, bodies)
}

fn tree(bodies: &[Body]) -> DynamicAabbTree {
    let mut tree = DynamicAabbTree::new(0.0);
    for &body in bodies {
        tree.insert(body);
    }
    tree.rebuild();
    tree
}

/// Query cell `q`, spread over the grid's first 8 × 8 cells (present at every size).
fn cell(q: usize) -> (f32, f32) {
    ((q % 8) as f32 * 2.0, (q / 8) as f32 * 2.0)
}

/// The tolerant `logarithmic` test of coding-tooling's work checker, for the self-check: the
/// least-squares rate per `1 + log2 x` of the upper half of the points must not exceed 1.25×
/// the lower half's (floored at 0) plus one unit over the upper half's span.
fn assert_logarithmic(name: &str, points: &[(usize, usize)]) {
    let g = points
        .iter()
        .map(|&(x, _)| 1.0 + (x as f64).log2())
        .collect::<Vec<_>>();
    let v = points.iter().map(|&(_, v)| v as f64).collect::<Vec<_>>();
    let slope = |g: &[f64], v: &[f64]| {
        let n = g.len() as f64;
        let (gm, vm) = (g.iter().sum::<f64>() / n, v.iter().sum::<f64>() / n);
        let num: f64 = g.iter().zip(v).map(|(g, v)| (g - gm) * (v - vm)).sum();
        let den: f64 = g.iter().map(|g| (g - gm).powi(2)).sum();
        num / den
    };
    let half = points.len().div_ceil(2);
    let lower = slope(&g[..half], &v[..half]);
    let upper_g = &g[points.len() - half..];
    let upper = slope(upper_g, &v[points.len() - half..]);
    let allowed = 1.25 * lower.max(0.0) + 1.0 / (upper_g[half - 1] - upper_g[0]);
    assert!(
        upper <= allowed * (1.0 + 1e-9),
        "{name} grows faster than logarithmically: {points:?} (rates {lower} then {upper})"
    );
}

#[test]
fn box_and_segment_queries_visit_logarithmically_many_nodes() {
    let (mut boxes, mut segments) = (Vec::new(), Vec::new());
    for leaves in LEAVES {
        let (columns, bodies) = grid(leaves);
        let tree = tree(&bodies);
        let (mut box_nodes, mut segment_nodes) = (0, 0);
        for q in 0..QUERIES {
            let (x, z) = cell(q);
            // A small box around one leaf's top face: exactly that leaf overlaps.
            let query = Aabb {
                min: [x - 0.25, 0.75, z - 0.25],
                max: [x + 0.25, 1.25, z + 0.25],
            };
            let mut found = Vec::new();
            box_nodes += tree.visit_aabb(query, |body| found.push(body.id));
            assert_eq!(
                found,
                vec![(x / 2.0) as u32 + (z / 2.0) as u32 * columns as u32]
            );
            // A vertical segment down onto the same leaf.
            let mut hit = Vec::new();
            segment_nodes +=
                tree.visit_segment([x, 5.0, z], [0.0, -10.0, 0.0], 1.0, 0.0, |body, t| {
                    hit.push((body.id, t));
                    1.0
                });
            assert_eq!(hit.len(), 1, "leaves={leaves} q={q}");
            assert!((hit[0].1 - 0.4).abs() < 1e-6);
        }
        boxes.push((
            vec![("leaves", leaves)],
            vec![("nodes-per-query", box_nodes / QUERIES)],
        ));
        segments.push((
            vec![("leaves", leaves)],
            vec![("nodes-per-query", segment_nodes / QUERIES)],
        ));
    }
    if let Some(dir) = work_evidence_dir() {
        write_work_evidence(&dir, "bvh-aabb-query-nodes", &boxes);
        write_work_evidence(&dir, "bvh-segment-query-nodes", &segments);
        return;
    }
    for (name, samples) in [("aabb", &boxes), ("segment", &segments)] {
        let points = samples
            .iter()
            .map(|(d, m)| (d[0].1, m[0].1))
            .collect::<Vec<_>>();
        assert_logarithmic(name, &points);
    }
}
