//! Multi-scale work evidence for bounded edit distance with a fixed limit.
//!
//! The scale series feeds the `bounded-edit-comparisons` and
//! `bounded-edit-warm-allocations` operations in `.performance/contract.json`
//! through `scripts/collect-work-evidence.sh`.
use similarity_kernels::{LevenshteinWorkspace, levenshtein};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

/// One scale point: `(dimensions, metrics)` as `(name, value)` pairs.
type WorkSample = (Vec<(&'static str, usize)>, Vec<(&'static str, usize)>);

/// Directory that `scripts/collect-work-evidence.sh` sets to collect
/// `coding-tooling/work-evidence/v1` samples. When it is set, the declared
/// work bounds are judged by `coding-tooling performance work` against
/// `.performance/contract.json`; otherwise this test asserts them itself.
fn work_evidence_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RUST_KERNELS_WORK_EVIDENCE_DIR").map(Into::into)
}

/// Writes one operation's samples as `<dir>/<operation>.json`.
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

struct CountingAllocator;
thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
    static REALLOCS: Cell<usize> = const { Cell::new(0) };
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static EQUALITY_CALLS: Cell<usize> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ENABLED.with(|enabled| {
            if enabled.get() {
                ALLOCS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ENABLED.with(|enabled| {
            if enabled.get() {
                REALLOCS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocation_counts<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCS.with(|count| count.set(0));
    REALLOCS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    let result = f();
    ENABLED.with(|enabled| enabled.set(false));
    (result, ALLOCS.with(Cell::get), REALLOCS.with(Cell::get))
}

/// Element whose every equality test is counted, so the count is exactly the
/// number of element comparisons the kernel performs.
struct Counted(u8);

impl PartialEq for Counted {
    fn eq(&self, other: &Self) -> bool {
        EQUALITY_CALLS.with(|count| count.set(count.get() + 1));
        self.0 == other.0
    }
}

impl Eq for Counted {}

/// The fixed small limit; the band width is `2 * LIMIT + 1`.
const LIMIT: usize = 2;
/// Powers of two from 128 to 4096 elements per sequence.
const ELEMENTS: [usize; 6] = [128, 256, 512, 1024, 2048, 4096];
/// Limits varied at a fixed 1024 elements, for the O(n · k) product bound.
const LIMITS: [usize; 4] = [1, 4, 8, 16];

/// Two sequences at distance two: `right` is `left` shifted by one element.
fn fixture(elements: usize) -> (Vec<u8>, Vec<u8>) {
    let left = (0..elements).map(|i| (i % 23) as u8).collect();
    let right = (0..elements).map(|i| ((i + 1) % 23) as u8).collect();
    (left, right)
}

/// Linear-shape check that tolerates a fixed term of either sign: the work
/// added per added element must not grow between consecutive scale points.
/// This rejects `n log n` and `n^2` growth. coding-tooling's `linear` bound,
/// `v(x) * x0 <= v(x0) * x`, also rejects the exact `5n - 4` count here
/// because the band corners make the fixed term negative.
fn assert_non_increasing_marginal_work(name: &str, points: &[(usize, usize)]) {
    for window in points.windows(3) {
        let [(x0, v0), (x1, v1), (x2, v2)] =
            [window[0], window[1], window[2]].map(|(x, v)| (x as i128, v as i128));
        assert!(
            (v2 - v1) * (x1 - x0) <= (v1 - v0) * (x2 - x1),
            "{name} grows faster than linearly between {x0}, {x1} and {x2}: {v0}, {v1}, {v2}"
        );
    }
}

/// The tolerant rate test of coding-tooling's work checker for a model `g` (here the product
/// n · k), for the self-check: ordered by `g`, the least-squares rate dv/dg of the upper half of
/// the points must not exceed 1.25× the lower half's (floored at 0) plus one unit over the
/// upper half's span.
fn assert_within_model(name: &str, mut points: Vec<(f64, usize)>) {
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    let slope = |p: &[(f64, usize)]| {
        let n = p.len() as f64;
        let gm = p.iter().map(|q| q.0).sum::<f64>() / n;
        let vm = p.iter().map(|q| q.1 as f64).sum::<f64>() / n;
        let num: f64 = p.iter().map(|q| (q.0 - gm) * (q.1 as f64 - vm)).sum();
        let den: f64 = p.iter().map(|q| (q.0 - gm).powi(2)).sum();
        num / den
    };
    let half = points.len().div_ceil(2);
    let lower = slope(&points[..half]);
    let upper_points = &points[points.len() - half..];
    let upper = slope(upper_points);
    let span = upper_points[half - 1].0 - upper_points[0].0;
    let allowed = 1.25 * lower.max(0.0) + 1.0 / span;
    assert!(
        upper <= allowed * (1.0 + 1e-9),
        "{name} grows faster than its model: {points:?} (rates {lower} then {upper})"
    );
}

#[test]
fn bounded_edit_comparisons_scale_linearly_with_a_fixed_limit() {
    let mut samples = Vec::new();
    let scale = ELEMENTS
        .iter()
        .map(|&elements| (elements, LIMIT))
        .chain(LIMITS.iter().map(|&limit| (1024, limit)));
    for (elements, limit) in scale {
        let (left, right) = fixture(elements);
        // Correctness first: the full quadratic oracle fixes the distance.
        let exact = levenshtein(&left, &right);
        assert_eq!(exact, 2, "elements={elements}");

        let left = left.into_iter().map(Counted).collect::<Vec<_>>();
        let right = right.into_iter().map(Counted).collect::<Vec<_>>();
        let mut workspace = LevenshteinWorkspace::new();
        EQUALITY_CALLS.with(|count| count.set(0));
        let distance = workspace.distance(&left, &right, limit);
        let comparisons = EQUALITY_CALLS.with(Cell::get);
        let expected = (limit >= exact).then_some(exact);
        assert_eq!(distance, expected, "elements={elements} limit={limit}");

        samples.push((
            vec![("elements", elements), ("limit", limit)],
            vec![("equality-calls", comparisons)],
        ));
    }

    if let Some(dir) = work_evidence_dir() {
        write_work_evidence(&dir, "bounded-edit-comparisons", &samples);
        return;
    }
    for (dimensions, metrics) in &samples {
        let (elements, limit, comparisons) = (dimensions[0].1, dimensions[1].1, metrics[0].1);
        if limit < 2 {
            continue; // Rejected early: the distance exceeds the limit.
        }
        // One prefix and one suffix probe, then the width-(2k + 1) band minus its
        // two truncated corners: (2k + 1) n - k (k + 1) + 2 comparisons.
        assert_eq!(
            comparisons,
            (2 * limit + 1) * elements - limit * (limit + 1) + 2,
            "elements={elements} limit={limit}"
        );
    }
    let along = |fixed: &dyn Fn(usize, usize) -> bool, axis: usize| {
        samples
            .iter()
            .filter(|(d, _)| fixed(d[0].1, d[1].1))
            .map(|(d, m)| (d[axis].1, m[0].1))
            .collect::<Vec<_>>()
    };
    let by_elements = along(&|_, limit| limit == LIMIT, 0);
    assert!(by_elements.iter().all(|&(_, v)| v <= 20484));
    assert_non_increasing_marginal_work("equality-calls by elements", &by_elements);
    let mut by_limit = along(&|elements, _| elements == 1024, 1);
    by_limit.sort_unstable();
    by_limit.dedup();
    assert_non_increasing_marginal_work("equality-calls by limit", &by_limit);
    assert_within_model(
        "equality-calls by elements · limit",
        samples
            .iter()
            .map(|(d, m)| ((d[0].1 * d[1].1) as f64, m[0].1))
            .collect(),
    );
}

#[test]
fn warmed_bounded_edit_queries_allocate_nothing_at_any_scale() {
    let mut samples = Vec::new();
    for elements in ELEMENTS {
        let (left, right) = fixture(elements);
        let mut workspace = LevenshteinWorkspace::new();
        // Warm-up query sizes the retained rows; correctness first.
        assert_eq!(workspace.distance(&left, &right, LIMIT), Some(2));
        let (distance, allocations, reallocations) =
            allocation_counts(|| workspace.distance(&left, &right, LIMIT));
        assert_eq!(distance, Some(2), "elements={elements}");

        samples.push((
            vec![("elements", elements), ("limit", LIMIT)],
            vec![
                ("allocator-alloc-calls", allocations),
                ("allocator-realloc-calls", reallocations),
            ],
        ));
    }

    if let Some(dir) = work_evidence_dir() {
        write_work_evidence(&dir, "bounded-edit-warm-allocations", &samples);
        return;
    }
    for (dimensions, metrics) in &samples {
        assert_eq!(
            (metrics[0].1, metrics[1].1),
            (0, 0),
            "elements={}",
            dimensions[0].1
        );
    }
}
