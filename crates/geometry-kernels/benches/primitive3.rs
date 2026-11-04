use divan::{Bencher, black_box};
use geometry_kernels::primitive3::{
    PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, query, swept_time,
};

#[allow(dead_code)]
#[path = "../tests/support/platonic.rs"]
mod platonic_fixtures;

fn main() {
    divan::main();
}

fn capsule() -> PrimitiveBody3 {
    PrimitiveBody3::axis_aligned(PrimitiveShape3::capsule(2.0, 0.5), [0.0; 3], [0.0; 3])
}

fn wedge() -> PrimitiveBody3 {
    PrimitiveBody3::axis_aligned(
        PrimitiveShape3::wedge([2.0, 2.0, 2.0]),
        [1.0, 0.5, 0.0],
        [0.0; 3],
    )
}

#[divan::bench]
fn capsule_wedge_contact(bencher: Bencher) {
    let capsule = capsule();
    let wedge = wedge();
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        black_box(query(
            black_box(capsule),
            black_box(wedge),
            black_box(&mut work),
        ))
    });
}

#[divan::bench]
fn fast_capsule_wedge_sweep(bencher: Bencher) {
    let moving = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::capsule(0.5, 0.1),
        [-1.0, -1.0, 10.0],
        [0.0, 0.0, -10_000.0],
    );
    let target =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::wedge([4.0, 4.0, 0.02]), [0.0; 3], [0.0; 3]);
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        black_box(swept_time(
            black_box(moving),
            black_box(target),
            black_box(1.0 / 60.0),
            black_box(0.02),
            black_box(&mut work),
        ))
    });
}

#[divan::bench]
fn primitive_volume_properties(bencher: Bencher) {
    let shapes = [
        PrimitiveShape3::sphere(1.0),
        PrimitiveShape3::cuboid([1.0, 2.0, 3.0]),
        PrimitiveShape3::capsule(1000.0, 0.01),
        PrimitiveShape3::wedge([1.0, 2.0, 3.0]),
    ];
    bencher.bench_local(|| {
        for shape in shapes {
            let _ = black_box(geometry_kernels::primitive3::try_volume_properties(
                black_box(shape),
            ));
        }
    });
}

#[divan::bench]
fn capsule_skeleton_contacts(bencher: Bencher) {
    let capsule =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::capsule(5.0, 0.5), [0.0; 3], [0.0; 3]);
    let sphere =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(0.25), [0.0, 2.0, 0.0], [0.0; 3]);
    let crossing = PrimitiveBody3 {
        axes: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ..capsule
    };
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        black_box(query(
            black_box(capsule),
            black_box(sphere),
            black_box(&mut work),
        ));
        black_box(query(
            black_box(capsule),
            black_box(crossing),
            black_box(&mut work),
        ));
    });
}

#[divan::bench]
fn capsule_row_contacts(bencher: Bencher) {
    let a = PrimitiveBody3::axis_aligned(PrimitiveShape3::capsule(2.0, 0.25), [0.0; 3], [0.0; 3]);
    let shapes = [
        PrimitiveShape3::sphere(0.5),
        PrimitiveShape3::cuboid([0.5, 0.5, 0.5]),
        PrimitiveShape3::capsule(1.0, 0.5),
    ];
    let targets =
        shapes.map(|shape| PrimitiveBody3::axis_aligned(shape, [1.0, 0.0, 0.0], [0.0; 3]));
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        for b in targets {
            black_box(query(black_box(a), black_box(b), black_box(&mut work)));
        }
    });
}

#[divan::bench]
fn rotated_polytope_contacts(bencher: Bencher) {
    let bodies = rotated_polytopes();
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        for [a, b] in bodies {
            black_box(query(black_box(a), black_box(b), black_box(&mut work)));
        }
    });
}

#[divan::bench]
fn rotated_polytope_sweeps(bencher: Bencher) {
    let bodies = rotated_polytopes();
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        for [a, b] in bodies {
            black_box(swept_time(
                black_box(a),
                black_box(b),
                black_box(1.0 / 60.0),
                black_box(0.02),
                black_box(&mut work),
            ));
        }
    });
}

/// Box/box, box/wedge and wedge/wedge poses whose edge cross products all count.
fn rotated_polytopes() -> [[PrimitiveBody3; 2]; 3] {
    let axes = [[0.6, 0.8, 0.0], [-0.48, 0.36, 0.8], [0.64, -0.48, 0.6]];
    let cuboid = PrimitiveShape3::cuboid([1.0, 1.5, 2.0]);
    let wedge = PrimitiveShape3::wedge([2.0, 1.0, 1.5]);
    let left = |shape| PrimitiveBody3::axis_aligned(shape, [0.0; 3], [0.0; 3]);
    let right = |shape| PrimitiveBody3::new(shape, [1.5, 0.5, -0.5], axes, [-30.0, 5.0, 0.0]);
    [
        [left(cuboid), right(cuboid)],
        [left(cuboid), right(wedge)],
        [left(wedge), right(wedge)],
    ]
}

#[divan::bench]
fn prepared_heightfield_surface_query(bencher: Bencher) {
    use geometry_kernels::heightfield::{HeightfieldData3, PreparedHeightfield3};
    let terrain = PreparedHeightfield3::try_new(HeightfieldData3 {
        columns: (0..65).map(f64::from).collect(),
        rows: (0..65).map(f64::from).collect(),
        heights: (0..65 * 65)
            .map(|index| (f64::from(index) * 0.1).sin())
            .collect(),
        active_cells: None,
        diagonal: geometry_kernels::heightfield::HeightfieldDiagonal3::Checkerboard,
    })
    .expect("finite benchmark geometry");
    bencher.bench_local(|| black_box(terrain.sample(black_box(31.2), black_box(20.7))));
}

#[divan::bench]
fn fast_sphere_sphere_sweep(bencher: Bencher) {
    let moving = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(0.25),
        [-10.0, 0.1, 0.0],
        [5_760.0, 0.0, 0.0],
    );
    let target = PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(0.5), [0.0; 3], [0.0; 3]);
    bencher.bench_local(|| {
        let mut work = PrimitiveWork3::default();
        black_box(swept_time(
            black_box(moving),
            black_box(target),
            black_box(1.0 / 60.0),
            black_box(0.0),
            black_box(&mut work),
        ))
    });
}

/// Experimental exact tetrahedron/octahedron swept SAT (issue #188) against the existing
/// `primitive3` box/box swept SAT and the crate's generic GJK conservative-advancement cast.
///
/// `cargo bench -p geometry-kernels --bench primitive3 -- platonic`
mod platonic {
    use super::platonic_fixtures::{DT, ReferenceCast, conservative_advancement, random_pairs};
    use divan::{Bencher, black_box};
    use geometry_kernels::primitive3::platonic::{
        PlatonicBody3, PlatonicShape3, PlatonicWork3, try_platonic_swept_interval,
    };
    use geometry_kernels::primitive3::{
        PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, try_swept_time,
    };

    const BATCH: usize = 64;
    const PAIRS: [&str; 6] = [
        "tetra_tetra",
        "octa_octa",
        "tetra_octa",
        "tetra_box",
        "octa_box",
        "box_box",
    ];
    /// `hits` inspect every candidate axis (SAT worst case); `receding` is low candidate
    /// density with early separating-axis exits; `near_miss` mixes grazing hits and misses.
    const DENSITIES: [&str; 3] = ["hits", "receding", "near_miss"];
    const BOX_A: [f64; 3] = [1.0, 0.75, 1.25];
    const BOX_B: [f64; 3] = [0.8, 0.6, 1.0];

    fn shapes(pair: &str) -> (PlatonicShape3, PlatonicShape3) {
        let tetra = PlatonicShape3::tetrahedron(1.0);
        let octa = PlatonicShape3::octahedron(1.0);
        let tetra_b = PlatonicShape3::tetrahedron(0.8);
        let octa_b = PlatonicShape3::octahedron(0.8);
        let cube = PlatonicShape3::cuboid(BOX_B);
        match pair {
            "tetra_tetra" => (tetra, tetra_b),
            "octa_octa" => (octa, octa_b),
            "tetra_octa" => (tetra, octa_b),
            "tetra_box" => (tetra, cube),
            "octa_box" => (octa, cube),
            "box_box" => (PlatonicShape3::cuboid(BOX_A), cube),
            _ => unreachable!(),
        }
    }

    fn batch(pair: &str, density: &str) -> [[PlatonicBody3; 2]; BATCH] {
        let (left, right) = shapes(pair);
        let bodies = match density {
            "hits" => random_pairs(0x188, left, right, (0.0, 0.2), false),
            "receding" => random_pairs(0x188, left, right, (0.0, 1.0), true),
            "near_miss" => random_pairs(0x188, left, right, (0.4, 0.8), false),
            _ => unreachable!(),
        };
        let hits = bodies
            .iter()
            .filter(|[a, b]| {
                let mut work = PlatonicWork3::default();
                matches!(
                    try_platonic_swept_interval(*a, *b, DT, 0.0, 64, &mut work),
                    Ok(Some(_))
                )
            })
            .count();
        match density {
            "hits" => assert_eq!(hits, BATCH, "{pair}"),
            "receding" => assert_eq!(hits, 0, "{pair}"),
            _ => assert!(hits > 0 && hits < BATCH, "{pair}: {hits}"),
        }
        bodies
    }

    fn swept_sat(bencher: Bencher, pair: &str, density: &str) {
        let bodies = batch(pair, density);
        bencher.bench_local(|| {
            let mut work = PlatonicWork3::default();
            for [a, b] in bodies {
                let _ = black_box(try_platonic_swept_interval(
                    black_box(a),
                    black_box(b),
                    DT,
                    0.0,
                    64,
                    &mut work,
                ));
            }
            work
        });
    }

    fn gjk_cast(bencher: Bencher, pair: &str, density: &str) {
        let bodies = batch(pair, density);
        bencher.bench_local(|| {
            for [a, b] in bodies {
                black_box(conservative_advancement(
                    black_box(a),
                    black_box(b),
                    DT,
                    0.0,
                ));
            }
        });
    }

    /// The true existing baseline: `primitive3::try_swept_time` box/box on the same poses.
    fn primitive_box_baseline(bencher: Bencher, density: &str) {
        let bodies = batch("box_box", density).map(|[a, b]| {
            let convert = |body: PlatonicBody3, half| {
                PrimitiveBody3::new(
                    PrimitiveShape3::cuboid(half),
                    body.position,
                    body.axes,
                    body.velocity,
                )
            };
            [convert(a, BOX_A), convert(b, BOX_B)]
        });
        bencher.bench_local(|| {
            let mut work = PrimitiveWork3::default();
            for [a, b] in bodies {
                let _ = black_box(try_swept_time(
                    black_box(a),
                    black_box(b),
                    DT,
                    0.0,
                    64,
                    &mut work,
                ));
            }
            work
        });
    }

    #[divan::bench(args = PAIRS)]
    fn swept_sat_hits(bencher: Bencher, pair: &str) {
        swept_sat(bencher, pair, "hits");
    }

    #[divan::bench(args = PAIRS)]
    fn swept_sat_receding(bencher: Bencher, pair: &str) {
        swept_sat(bencher, pair, "receding");
    }

    #[divan::bench(args = PAIRS)]
    fn swept_sat_near_miss(bencher: Bencher, pair: &str) {
        swept_sat(bencher, pair, "near_miss");
    }

    #[divan::bench(args = PAIRS)]
    fn gjk_cast_hits(bencher: Bencher, pair: &str) {
        gjk_cast(bencher, pair, "hits");
    }

    #[divan::bench(args = PAIRS)]
    fn gjk_cast_receding(bencher: Bencher, pair: &str) {
        gjk_cast(bencher, pair, "receding");
    }

    #[divan::bench(args = PAIRS)]
    fn gjk_cast_near_miss(bencher: Bencher, pair: &str) {
        gjk_cast(bencher, pair, "near_miss");
    }

    #[divan::bench(args = DENSITIES)]
    fn primitive_box_baseline_sweep(bencher: Bencher, density: &str) {
        primitive_box_baseline(bencher, density);
    }

    /// Edge-edge graze 1e-6 outside contact: only an edge-cross axis separates, and the
    /// generic cast must converge on a sliding edge-edge distance.
    fn graze() -> [PlatonicBody3; 2] {
        let gap = 2.0 / 3.0_f64.sqrt() + 1.0e-6;
        let tetra = PlatonicShape3::tetrahedron(1.0);
        [
            PlatonicBody3::axis_aligned(tetra, [0.0; 3], [0.0; 3]),
            PlatonicBody3::axis_aligned(tetra, [gap, -3.0, 0.0], [0.0, 360.0, 0.0]),
        ]
    }

    #[divan::bench]
    fn edge_graze_swept_sat(bencher: Bencher) {
        let [a, b] = graze();
        bencher.bench_local(|| {
            let mut work = PlatonicWork3::default();
            black_box(try_platonic_swept_interval(
                black_box(a),
                black_box(b),
                DT,
                0.0,
                64,
                &mut work,
            ))
        });
    }

    #[divan::bench]
    fn edge_graze_gjk_cast(bencher: Bencher) {
        let [a, b] = graze();
        assert!(matches!(
            conservative_advancement(a, b, DT, 0.0),
            ReferenceCast::Miss { .. }
        ));
        bencher.bench_local(|| {
            black_box(conservative_advancement(
                black_box(a),
                black_box(b),
                DT,
                0.0,
            ))
        });
    }
}
