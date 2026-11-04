use divan::{Bencher, black_box};
use geometry_kernels::primitive3::{
    PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, query, swept_time,
};

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

mod ellipsoid {
    //! Experimental ellipsoid kernels (#189) against the existing primitive kernels. Fixtures are
    //! built outside the timed closure; each sample runs one batch of `BATCH` complete calls
    //! over varied inputs, so per-call cost is the reported time divided by `BATCH`.

    use divan::{Bencher, black_box};
    use geometry_kernels::halfspace::capsule_halfspace_sweep;
    use geometry_kernels::math3::Vec3;
    use geometry_kernels::primitive3::ellipsoid::{Ellipsoid3, EllipsoidBody3, EllipsoidPose3};
    use geometry_kernels::primitive3::{
        PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, bounds_extents, support_point, swept_time,
    };

    pub(super) const BATCH: usize = 64;
    /// Rotated orthonormal frame shared by every fixture.
    pub(super) const AXES: [Vec3; 3] = [[0.6, 0.8, 0.0], [-0.48, 0.36, 0.8], [0.64, -0.48, 0.6]];
    const UP: Vec3 = [0.0, 1.0, 0.0];

    /// 1:1:1, 1.5:1:1, 4:1:1 and an adverse 1e4:1:1e-3 sliver far from the origin.
    pub(super) const RATIOS: [&str; 4] = ["1:1:1", "1.5:1:1", "4:1:1", "adverse"];
    const KINDS: [&str; 3] = ["sphere", "capsule", "box"];

    pub(super) fn semiaxes(ratio: &str) -> (Vec3, Vec3) {
        match ratio {
            "1:1:1" => ([1.0, 1.0, 1.0], [0.0, 10.0, 0.0]),
            "1.5:1:1" => ([1.5, 1.0, 1.0], [0.0, 10.0, 0.0]),
            "4:1:1" => ([4.0, 1.0, 1.0], [0.0, 10.0, 0.0]),
            _ => ([1.0e4, 1.0, 1.0e-3], [1.0e7, 2.0e4, -1.0e7]),
        }
    }

    /// Deterministic orthonormal frames: rotations of `AXES` about world Y.
    pub(super) fn frames() -> [[Vec3; 3]; BATCH] {
        std::array::from_fn(|index| {
            let (sin, cos) = (index as f64 * 0.37).sin_cos();
            AXES.map(|axis| {
                [
                    cos * axis[0] + sin * axis[2],
                    axis[1],
                    cos * axis[2] - sin * axis[0],
                ]
            })
        })
    }

    pub(super) fn directions() -> [Vec3; BATCH] {
        std::array::from_fn(|index| {
            let t = index as f64 * 0.7;
            [t.cos(), (1.3 * t).sin(), 0.3 + t.sin()]
        })
    }

    fn bodies(ratio: &str) -> [EllipsoidBody3; BATCH] {
        let (semiaxes, center) = semiaxes(ratio);
        let shape = Ellipsoid3::try_new(semiaxes).expect("valid semiaxes");
        frames().map(|axes| {
            EllipsoidBody3::new(
                shape,
                EllipsoidPose3::try_new(center, axes).expect("orthonormal axes"),
            )
        })
    }

    fn primitives(kind: &str) -> [PrimitiveBody3; BATCH] {
        let shape = match kind {
            "sphere" => PrimitiveShape3::sphere(1.0),
            "capsule" => PrimitiveShape3::capsule(1.5, 1.0),
            _ => PrimitiveShape3::cuboid([4.0, 1.0, 1.0]),
        };
        frames().map(|axes| PrimitiveBody3::new(shape, [0.0, 10.0, 0.0], axes, [0.0; 3]))
    }

    fn displacements(ratio: &str) -> [Vec3; BATCH] {
        let reach = if ratio == "adverse" { 4.0e4 } else { 40.0 };
        directions().map(|d| [d[0], -reach * (1.0 + 0.5 * d[1]), d[2]])
    }

    #[divan::bench(args = RATIOS)]
    fn ellipsoid_support_point(bencher: Bencher, ratio: &str) {
        let body = bodies(ratio)[0];
        let directions = directions();
        bencher.bench_local(|| {
            for direction in black_box(&directions) {
                let _ = black_box(body.try_support_point(*direction));
            }
        });
    }

    #[divan::bench(args = KINDS)]
    fn primitive_support_point(bencher: Bencher, kind: &str) {
        let body = primitives(kind)[0];
        let directions = directions();
        bencher.bench_local(|| {
            let mut work = PrimitiveWork3::default();
            for direction in black_box(&directions) {
                black_box(support_point(body, *direction, &mut work));
            }
        });
    }

    #[divan::bench(args = RATIOS)]
    fn ellipsoid_aabb(bencher: Bencher, ratio: &str) {
        let bodies = bodies(ratio);
        bencher.bench_local(|| {
            for body in black_box(&bodies) {
                let _ = black_box(body.try_aabb_half_extents());
            }
        });
    }

    #[divan::bench(args = KINDS)]
    fn primitive_aabb(bencher: Bencher, kind: &str) {
        let bodies = primitives(kind);
        bencher.bench_local(|| {
            for body in black_box(&bodies) {
                black_box(bounds_extents(*body));
            }
        });
    }

    #[divan::bench(args = RATIOS)]
    fn ellipsoid_plane_cast(bencher: Bencher, ratio: &str) {
        let bodies = bodies(ratio);
        let displacements = displacements(ratio);
        bencher.bench_local(|| {
            for (body, displacement) in black_box(&bodies).iter().zip(&displacements) {
                let _ = black_box(body.try_halfspace_sweep(*displacement, [0.0; 3], UP));
            }
        });
    }

    #[divan::bench(args = ["sphere", "capsule"])]
    fn halfspace_plane_cast(bencher: Bencher, kind: &str) {
        let segments = primitives(kind).map(|body| {
            let half = if kind == "sphere" { 0.0 } else { 1.5 };
            let axis = body.axes[1];
            let offset = [axis[0] * half, axis[1] * half, axis[2] * half];
            (
                [
                    body.position[0] - offset[0],
                    body.position[1] - offset[1],
                    body.position[2] - offset[2],
                ],
                [
                    body.position[0] + offset[0],
                    body.position[1] + offset[1],
                    body.position[2] + offset[2],
                ],
            )
        });
        let displacements = displacements(kind);
        bencher.bench_local(|| {
            for ((p, q), displacement) in black_box(&segments).iter().zip(&displacements) {
                black_box(capsule_halfspace_sweep(
                    *p,
                    *q,
                    1.0,
                    *displacement,
                    [0.0; 3],
                    UP,
                ));
            }
        });
    }

    /// The box has no halfspace kernel; its nearest equivalent is a swept-SAT cast against a
    /// wide floor slab.
    #[divan::bench]
    fn box_slab_cast(bencher: Bencher) {
        let movers = primitives("box");
        let displacements = displacements("box");
        let floor = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::cuboid([1.0e3, 1.0, 1.0e3]),
            [0.0, -1.0, 0.0],
            [0.0; 3],
        );
        bencher.bench_local(|| {
            let mut work = PrimitiveWork3::default();
            for (moving, displacement) in black_box(&movers).iter().zip(&displacements) {
                let moving = PrimitiveBody3 {
                    velocity: *displacement,
                    ..*moving
                };
                black_box(swept_time(moving, floor, 1.0, 0.0, &mut work));
            }
        });
    }
}
