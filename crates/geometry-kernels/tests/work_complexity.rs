//! Multi-scale work evidence for easy GJK sphere queries.
//!
//! The scale series feeds the `gjk-easy-sphere-iterations` operation in
//! `.performance/contract.json` through `scripts/collect-work-evidence.sh`.
use geometry_kernels::{Sphere, gjk::gjk_intersection, sphere_sphere};

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

/// Binary exponents `e` of the coordinate scale `2^e`.
const SCALE_EXPONENTS: [i32; 6] = [-30, -24, -10, 0, 10, 24];
/// Bias that maps the exponents onto the positive integers that contract
/// dimensions require: `2^-30` is 1 and `2^24` is 55.
const EXPONENT_BIAS: i32 = 31;

#[test]
fn easy_gjk_sphere_iterations_do_not_grow_with_coordinate_scale() {
    let mut samples = Vec::new();
    for exponent in SCALE_EXPONENTS {
        let s = 2.0_f64.powi(exponent) as f32;
        let left = Sphere::new([0.0; 3], s);
        let mut total_iterations = 0;
        let mut max_iterations = 0;
        for center in [
            [0.0, 0.0, 0.0],
            [1.4, 1.4, 0.0],
            [2.0, 0.0, 0.0],
            [2.05, 0.0, 0.0],
            [1.5, 1.5, 0.0],
            [3.0, 1.0, 0.0],
        ] {
            let right = Sphere::new(center.map(|v| v * s), s);
            let expected = sphere_sphere(left, right).overlaps;
            for (a, b) in [(left, right), (right, left)] {
                let result = gjk_intersection(&a, &b);
                // Correctness first: the analytic sphere oracle must agree, so
                // an iteration limit or no-progress exit is never a success.
                assert_eq!(
                    result.intersection(),
                    Some(expected),
                    "exponent={exponent} center={center:?}: {result:?}"
                );
                total_iterations += result.iterations;
                max_iterations = max_iterations.max(result.iterations);
            }
        }
        let biased =
            usize::try_from(exponent + EXPONENT_BIAS).expect("bias keeps exponents positive");
        samples.push((
            vec![("biasedScaleExponent", biased)],
            vec![
                ("max-query-iterations", max_iterations),
                ("total-iterations", total_iterations),
            ],
        ));
    }

    if let Some(dir) = work_evidence_dir() {
        write_work_evidence(&dir, "gjk-easy-sphere-iterations", &samples);
        return;
    }
    let smallest_scale = &samples[0].1;
    for (dimensions, metrics) in &samples {
        // No query exhausts the iteration budget; twelve queries per scale.
        assert!(metrics[0].1 <= 8, "scale={}: {metrics:?}", dimensions[0].1);
        assert!(metrics[1].1 <= 96, "scale={}: {metrics:?}", dimensions[0].1);
        // Constant in coordinate scale: no scale may need more iterations
        // than the smallest one.
        assert!(
            metrics[0].1 <= smallest_scale[0].1,
            "scale={}",
            dimensions[0].1
        );
        assert!(
            metrics[1].1 <= smallest_scale[1].1,
            "scale={}",
            dimensions[0].1
        );
    }
}
