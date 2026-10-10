//! Multi-scale work evidence for the regrouped linear Fenwick builder.
//!
//! The scale series feeds the `fenwick-linear-build` operation in
//! `.performance/contract.json` through `scripts/collect-work-evidence.sh`.
use collection_kernels::FenwickTree;
use std::{
    cell::Cell,
    ops::{AddAssign, Sub},
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

thread_local! {
    static ADD_ASSIGN_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Integer value whose every `+=` is counted, so the count is exactly the
/// number of additions the builder performs on caller values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CountingValue(i64);

impl AddAssign for CountingValue {
    fn add_assign(&mut self, rhs: Self) {
        ADD_ASSIGN_COUNT.with(|count| count.set(count.get() + 1));
        self.0 += rhs.0;
    }
}

impl Sub for CountingValue {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

fn counted<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ADD_ASSIGN_COUNT.with(|count| count.set(0));
    let result = f();
    (result, ADD_ASSIGN_COUNT.with(Cell::get))
}

/// Linear-shape check that tolerates a fixed term of either sign: the work
/// added per added item must not grow between consecutive scale points. This
/// rejects `n log n` and `n^2` growth. coding-tooling's `linear` bound,
/// `v(x) * x0 <= v(x0) * x`, also rejects the exact `2n - 1` count here
/// because its fixed term is negative.
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

/// Powers of two from 64 to 4096 items.
const ITEMS: [usize; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

#[test]
fn fenwick_linear_build_work_scales_linearly() {
    let mut samples = Vec::new();
    for items in ITEMS {
        let values = (0..items)
            .map(|index| CountingValue((index as i64 * 17) % 31 - 15))
            .collect::<Vec<_>>();
        let (linear, add_assigns) = counted(|| FenwickTree::from_slice_linear(&values));

        // Correctness first: the regrouped builder must equal ordered point
        // updates for this associative value type, and answer every prefix.
        let (ordered, _) = counted(|| FenwickTree::from_slice(&values));
        assert_eq!(linear, ordered, "items={items}");
        assert_eq!(linear.prefix_sum(0).0, 0, "items={items}");
        let mut expected = 0;
        for (index, value) in values.iter().enumerate() {
            expected += value.0;
            let end = index + 1;
            assert_eq!(
                linear.prefix_sum(end).0,
                expected,
                "items={items}, end={end}"
            );
        }

        samples.push((
            vec![("items", items)],
            vec![("add-assign-operations", add_assigns)],
        ));
    }

    if let Some(dir) = work_evidence_dir() {
        write_work_evidence(&dir, "fenwick-linear-build", &samples);
        return;
    }
    let points = samples
        .iter()
        .map(|(dimensions, metrics)| (dimensions[0].1, metrics[0].1))
        .collect::<Vec<_>>();
    for &(items, add_assigns) in &points {
        // One leaf addition per item plus one parent propagation for every
        // item whose Fenwick parent lies inside the tree.
        assert_eq!(
            add_assigns,
            2 * items - items.count_ones() as usize,
            "items={items}"
        );
        assert!(add_assigns <= 2 * items, "items={items}");
        assert!(add_assigns <= 8191, "items={items}");
    }
    assert_non_increasing_marginal_work("add-assign-operations", &points);
}
