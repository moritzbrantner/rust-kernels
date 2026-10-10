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

#[test]
fn bounded_edit_comparisons_scale_linearly_with_a_fixed_limit() {
    let mut samples = Vec::new();
    for elements in ELEMENTS {
        let (left, right) = fixture(elements);
        // Correctness first: the full quadratic oracle fixes the distance.
        let exact = levenshtein(&left, &right);
        assert_eq!(exact, 2, "elements={elements}");

        let left = left.into_iter().map(Counted).collect::<Vec<_>>();
        let right = right.into_iter().map(Counted).collect::<Vec<_>>();
        let mut workspace = LevenshteinWorkspace::new();
        EQUALITY_CALLS.with(|count| count.set(0));
        let distance = workspace.distance(&left, &right, LIMIT);
        let comparisons = EQUALITY_CALLS.with(Cell::get);
        assert_eq!(distance, Some(exact), "elements={elements}");

        samples.push((
            vec![("elements", elements), ("limit", LIMIT)],
            vec![("equality-calls", comparisons)],
        ));
    }

    if let Some(dir) = work_evidence_dir() {
        write_work_evidence(&dir, "bounded-edit-comparisons", &samples);
        return;
    }
    let points = samples
        .iter()
        .map(|(dimensions, metrics)| (dimensions[0].1, metrics[0].1))
        .collect::<Vec<_>>();
    for &(elements, comparisons) in &points {
        // One prefix and one suffix probe, then the width-five band minus its
        // two truncated corners: (2k + 1) n - k (k + 1) + 2 comparisons.
        assert_eq!(
            comparisons,
            (2 * LIMIT + 1) * elements - LIMIT * (LIMIT + 1) + 2,
            "elements={elements}"
        );
        assert!(comparisons <= 20484, "elements={elements}");
    }
    assert_non_increasing_marginal_work("equality-calls", &points);
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
