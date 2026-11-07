//! Differential fuzz of the heightfield capsule kernels against copies of the kernels they
//! replaced. Pruned contacts must be bit-identical, including the chosen triangle. Ordered
//! sweeps must return every old `Ok` bit-identically and fail only where the old sweep failed;
//! an old failure past the earliest hit may become that hit.

use super::*;

/// The capsule contact before plane pruning: the exact test on every triangle.
fn reference_capsule_contact(
    field: &PreparedHeightfield3,
    p: Vec3,
    q: Vec3,
    radius: f64,
    margin: f64,
    work: &mut HeightfieldWork3,
) -> Option<HeightfieldContact3> {
    work.queries += 1;
    let reach = radius + margin.max(0.0);
    let min = [p[0].min(q[0]) - reach, p[2].min(q[2]) - reach];
    let max = [p[0].max(q[0]) + reach, p[2].max(q[2]) + reach];
    let mut best: Option<HeightfieldContact3> = None;
    field.visit_triangles_counted(min, max, work, |id, triangle, normal| {
        let Some(contact) = segment_triangle_contact(p, q, radius, triangle, normal) else {
            return;
        };
        if contact.separation <= margin && best.is_none_or(|b| contact.separation < b.separation) {
            best = Some(HeightfieldContact3 {
                triangle: id,
                ..contact
            });
        }
    });
    best
}

/// The capsule sweep before ordered triangle visiting: triangles in id order, each iterating
/// until it settles or passes the best hit so far, failing the sweep when it runs out of
/// iterations first.
#[allow(
    clippy::too_many_lines,
    reason = "verbatim copy of the replaced kernel"
)]
fn reference_capsule_sweep(
    field: &PreparedHeightfield3,
    p: Vec3,
    q: Vec3,
    radius: f64,
    displacement: Vec3,
    max_iterations: u32,
    work: &mut HeightfieldWork3,
) -> Result<Option<HeightfieldSweepHit3>, HeightfieldError3> {
    work.queries += 1;
    let scale_bound = p
        .into_iter()
        .chain(q)
        .chain(displacement)
        .map(f64::abs)
        .fold(radius, f64::max);
    // Contact tolerance: rounding of the distance computation at this scale.
    let tolerance = 1e-12 * (1.0 + scale_bound);
    let end_p = add(p, displacement);
    let end_q = add(q, displacement);
    let reach = radius + tolerance;
    let min = [
        p[0].min(q[0]).min(end_p[0]).min(end_q[0]) - reach,
        p[2].min(q[2]).min(end_p[2]).min(end_q[2]) - reach,
    ];
    let max = [
        p[0].max(q[0]).max(end_p[0]).max(end_q[0]) + reach,
        p[2].max(q[2]).max(end_p[2]).max(end_q[2]) + reach,
    ];
    let mut triangles = [(0_u32, [[0.0; 3]; 3], [0.0; 3]); 64];
    let mut count = 0;
    let mut spill = Vec::new();
    field.visit_triangles_counted(min, max, work, |id, t, n| {
        if count < triangles.len() {
            triangles[count] = (id, t, n);
        } else {
            spill.push((id, t, n));
        }
        count += 1;
    });
    // Distance from the translating capsule to one convex triangle is convex in time, so
    // Newton steps from below converge to that triangle's first touch without passing it.
    // Each triangle is solved on its own; the earliest touch wins (ties: most head-on, then
    // triangle id).
    let length = dot(displacement, displacement).sqrt();
    let mut best: Option<HeightfieldSweepHit3> = None;
    for k in 0..count {
        let (id, triangle, normal) = if k < 64 { triangles[k] } else { spill[k - 64] };
        let mut time = 0.0_f64;
        let mut iterations = 0;
        loop {
            if best.is_some_and(|b| time > b.fraction) {
                break;
            }
            if iterations == max_iterations {
                return Err(HeightfieldError3::NonFiniteComputation);
            }
            iterations += 1;
            work.triangles_tested += 1;
            let offset = scale(displacement, time);
            let Some(c) =
                segment_triangle_contact(add(p, offset), add(q, offset), radius, triangle, normal)
            else {
                break;
            };
            if c.separation <= tolerance {
                let rate = dot(displacement, c.normal);
                if time == 0.0 && rate >= 0.0 {
                    // Starts touching but moves away (or along it): no hit.
                    break;
                }
                let hit = HeightfieldSweepHit3 {
                    fraction: time,
                    contact: HeightfieldContact3 { triangle: id, ..c },
                };
                if best.is_none_or(|b| {
                    let earlier = rate.total_cmp(&dot(displacement, b.contact.normal));
                    time < b.fraction
                        || (time == b.fraction && earlier.then(id.cmp(&b.contact.triangle)).is_lt())
                }) {
                    best = Some(hit);
                }
                break;
            }
            // A lower bound on the time to touch, from the separation and full speed.
            if time + (c.separation - tolerance) / length > 1.0 {
                break;
            }
            let rate = dot(displacement, c.normal);
            if rate >= 0.0 {
                // Moving away: by convexity this triangle only gets farther.
                break;
            }
            time += (c.separation - tolerance * 0.5) / -rate;
            if time > 1.0 {
                break;
            }
        }
    }
    if let Some(hit) = best {
        return Ok(Some(hit));
    }
    Ok(None)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
    /// Uniform in `0..1`.
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1_u64 << 53) as f64
    }
    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn direction(&mut self) -> Vec3 {
        loop {
            let v = [
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            ];
            let l = dot(v, v);
            if l > 1e-4 && l <= 1.0 {
                return scale(v, 1.0 / l.sqrt());
            }
        }
    }
}

fn terrain(rng: &mut Rng) -> PreparedHeightfield3 {
    let offset = [0.0, 0.0, 1e3, -3e4][rng.below(4) as usize];
    let axis = |rng: &mut Rng| {
        let n = 2 + rng.below(7) as usize;
        let uniform = rng.below(2) == 0;
        let mut at = offset + rng.range(-2.0, 2.0);
        (0..n)
            .map(|_| {
                let value = at;
                at += if uniform { 1.0 } else { rng.range(0.2, 3.0) };
                value
            })
            .collect::<Vec<_>>()
    };
    let (columns, rows) = (axis(rng), axis(rng));
    let amplitude = [0.0, 0.5, 3.0, 20.0][rng.below(4) as usize];
    // Quantized heights give coplanar neighbours and exact ties between triangles.
    let quantized = rng.below(3) == 0;
    let heights = (0..columns.len() * rows.len())
        .map(|_| {
            let h = rng.range(-amplitude, amplitude);
            if quantized { h.round() } else { h }
        })
        .collect();
    let cells = (columns.len() - 1) * (rows.len() - 1);
    let active_cells = (rng.below(3) == 0).then(|| (0..cells).map(|_| rng.below(4) != 0).collect());
    let diagonal = if rng.below(2) == 0 {
        HeightfieldDiagonal3::Checkerboard
    } else {
        HeightfieldDiagonal3::Fixed
    };
    PreparedHeightfield3::try_new(HeightfieldData3 {
        columns,
        rows,
        heights,
        active_cells,
        diagonal,
    })
    .expect("random terrains are valid")
}

/// A random capsule near `field`: free placements, and placements whose endpoint sits on the
/// pruning boundary of a triangle's face, edge or vertex (tangent and grazing), with that
/// triangle's normal.
fn capsule(rng: &mut Rng, field: &PreparedHeightfield3) -> (Vec3, Vec3, f64, f64, Option<Vec3>) {
    let radius = match rng.below(4) {
        0 => 0.0,
        _ => rng.range(0.05, 1.5),
    };
    let margin = match rng.below(4) {
        0 => 0.0,
        1 => 1e-9,
        2 => rng.range(-0.1, 0.0),
        _ => rng.range(0.0, 0.3),
    };
    let reach = radius + margin.max(0.0);
    let (columns, rows) = (&field.data.columns, &field.data.rows);
    let triangle = loop {
        let id = rng.below(field.normals.len() as u64) as usize;
        if field.active(id / 2) {
            break Some(id);
        }
        if field
            .data
            .active_cells
            .as_ref()
            .is_some_and(|cells| !cells.contains(&true))
        {
            break None;
        }
    };
    let (p, direction, near) = match (rng.below(3), triangle) {
        (0, _) | (_, None) => {
            let x = rng.range(columns[0] - 1.0, columns[columns.len() - 1] + 1.0);
            let z = rng.range(rows[0] - 1.0, rows[rows.len() - 1] + 1.0);
            ([x, rng.range(-25.0, 25.0), z], rng.direction(), None)
        }
        (mode, Some(id)) => {
            let [a, b, c] = field.vertices(id / 2, id % 2);
            let normal = field.normals[id];
            // A face point, an edge point or a vertex.
            let (u, v) = match rng.below(3) {
                0 => {
                    let (u, v) = (rng.unit(), rng.unit());
                    if u + v > 1.0 {
                        (1.0 - u, 1.0 - v)
                    } else {
                        (u, v)
                    }
                }
                1 => [(rng.unit(), 0.0), (0.0, rng.unit()), {
                    let u = rng.unit();
                    (u, 1.0 - u)
                }][rng.below(3) as usize],
                _ => [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)][rng.below(3) as usize],
            };
            let on = add(a, add(scale(vsub(b, a), u), scale(vsub(c, a), v)));
            // Away along the face normal (tangent to the plane), or any direction off an edge or
            // vertex, possibly below the surface.
            let away = if mode == 1 {
                if rng.below(4) == 0 {
                    scale(normal, -1.0)
                } else {
                    normal
                }
            } else {
                rng.direction()
            };
            let magnitude = on.into_iter().map(f64::abs).fold(reach, f64::max);
            let tolerance = 1e-9 * (1.0 + magnitude);
            let offset = match rng.below(6) {
                0 => 0.0,
                1 => rng.range(-4.0, 4.0) * f64::EPSILON * (1.0 + magnitude),
                2 => rng.range(-2.0, 2.0) * 1e-12 * (1.0 + magnitude),
                3 => rng.range(0.5, 2.0) * tolerance,
                4 => tolerance * (1.0 + rng.range(-1e-6, 1e-6)),
                _ => rng.range(-0.2, 0.5),
            };
            (
                add(on, scale(away, reach + offset)),
                {
                    // Along the plane (grazing), straight away, or anywhere.
                    let w = rng.direction();
                    match rng.below(3) {
                        0 => vsub(w, scale(normal, dot(w, normal))),
                        1 => away,
                        _ => w,
                    }
                },
                Some(normal),
            )
        }
    };
    let length = match rng.below(4) {
        0 => 0.0,
        _ => rng.range(0.0, 3.0),
    };
    (p, add(p, scale(direction, length)), radius, margin, near)
}

type ContactBits = (u32, u64, [u64; 3], [u64; 3]);

fn contact_bits(contact: Option<HeightfieldContact3>) -> Option<ContactBits> {
    contact.map(|c| {
        (
            c.triangle,
            c.separation.to_bits(),
            c.normal.map(f64::to_bits),
            c.point.map(f64::to_bits),
        )
    })
}

#[test]
fn pruned_capsule_contact_is_bit_identical() {
    let mut rng = Rng(0x5eed_0527);
    let (mut queries, mut contacts, mut pruned, mut tested) = (0_u64, 0_u64, 0_u64, 0_u64);
    for _ in 0..1500 {
        let field = terrain(&mut rng);
        for _ in 0..40 {
            let (p, q, radius, margin, _) = capsule(&mut rng, &field);
            let (mut work, mut reference_work) =
                (HeightfieldWork3::default(), HeightfieldWork3::default());
            let actual = field.capsule_contact_counted(p, q, radius, margin, &mut work);
            let expected =
                reference_capsule_contact(&field, p, q, radius, margin, &mut reference_work);
            assert_eq!(
                contact_bits(actual),
                contact_bits(expected),
                "p {p:?} q {q:?} radius {radius} margin {margin} terrain {:?}",
                field.data
            );
            assert_eq!(
                HeightfieldWork3 {
                    triangles_pruned: 0,
                    ..work
                },
                reference_work
            );
            queries += 1;
            contacts += u64::from(expected.is_some());
            pruned += work.triangles_pruned;
            tested += work.triangles_tested;
        }
    }
    // The fuzz must exercise both outcomes and both paths.
    assert!(contacts > queries / 5, "{contacts} contacts in {queries}");
    assert!(
        pruned > tested / 10 && pruned < tested,
        "{pruned} of {tested}"
    );
}

fn sweep_bits(
    result: Result<Option<HeightfieldSweepHit3>, HeightfieldError3>,
) -> Result<Option<(u64, Option<ContactBits>)>, HeightfieldError3> {
    result.map(|hit| hit.map(|h| (h.fraction.to_bits(), contact_bits(Some(h.contact)))))
}

/// A sweep displacement for a capsule near a triangle with normal `near`: anywhere, along the
/// plane (grazing), or along it with a near-tangent component of a few ulps to 1e-6 into or out
/// of it.
fn displacement(rng: &mut Rng, near: Option<Vec3>) -> Vec3 {
    let length = match rng.below(5) {
        0 => 0.0,
        1 => rng.range(0.0, 0.05),
        _ => rng.range(0.0, 4.0),
    };
    let direction = rng.direction();
    let Some(normal) = near.filter(|_| rng.below(4) != 0) else {
        return scale(direction, length);
    };
    let along = vsub(direction, scale(normal, dot(direction, normal)));
    let tilt = match rng.below(4) {
        0 => 0.0,
        1 => rng.range(-8.0, 8.0) * f64::EPSILON,
        2 => rng.range(-1e-9, 1e-9),
        _ => rng.range(-1e-6, 1e-6),
    };
    scale(add(along, scale(normal, tilt)), length)
}

#[test]
fn ordered_capsule_sweep_keeps_old_hits_and_fails_only_where_old_failed() {
    let mut rng = Rng(0x5eed_527c);
    let (mut queries, mut hits, mut both_err, mut recovered, mut confirmed) = (0, 0, 0, 0, 0);
    let (mut recovered_at_64, mut recovered_none) = (0, 0);
    let (mut tested, mut reference_tested) = (0_u64, 0_u64);
    for _ in 0..1000 {
        let field = terrain(&mut rng);
        for _ in 0..40 {
            let (p, q, radius, _, near) = capsule(&mut rng, &field);
            let d = displacement(&mut rng, near);
            let max_iterations = [0, 1, 2, 3, 4, 6, 8, 16, 64, 64][rng.below(10) as usize];
            let (mut work, mut reference_work) =
                (HeightfieldWork3::default(), HeightfieldWork3::default());
            let actual = field.capsule_sweep_counted(p, q, radius, d, max_iterations, &mut work);
            let expected = reference_capsule_sweep(
                &field,
                p,
                q,
                radius,
                d,
                max_iterations,
                &mut reference_work,
            );
            let case = || {
                format!(
                    "p {p:?} q {q:?} radius {radius} d {d:?} max {max_iterations} terrain {:?}",
                    field.data
                )
            };
            queries += 1;
            match (&expected, &actual) {
                (Ok(_), _) => {
                    assert_eq!(sweep_bits(actual), sweep_bits(expected), "{}", case());
                    hits += u64::from(matches!(expected, Ok(Some(_))));
                    tested += work.triangles_tested;
                    reference_tested += reference_work.triangles_tested;
                }
                (Err(_), Err(_)) => both_err += 1,
                (Err(_), Ok(hit)) => {
                    // A failure past the earliest hit (or, without one, of a triangle the motion
                    // cannot reach): the result must be the one the old sweep settles to.
                    recovered += 1;
                    recovered_at_64 += u32::from(max_iterations == 64);
                    recovered_none += u32::from(hit.is_none());
                    let settled = reference_capsule_sweep(
                        &field,
                        p,
                        q,
                        radius,
                        d,
                        1 << 20,
                        &mut HeightfieldWork3::default(),
                    );
                    if let Ok(settled) = settled {
                        assert_eq!(sweep_bits(Ok(*hit)), sweep_bits(Ok(settled)), "{}", case());
                        confirmed += 1;
                    }
                }
            }
        }
    }
    println!(
        "{queries} sweeps: {hits} old hits kept, {both_err} failures kept, {recovered} failures \
         became the settled result ({recovered_none} no hit, {recovered_at_64} at 64 iterations, {confirmed} confirmed with 2^20 iterations); triangle \
         iterations {tested} vs {reference_tested} on old successes"
    );
    // The fuzz must exercise hits, kept failures and recovered failures.
    assert!(hits > queries / 10, "{hits} hits in {queries}");
    assert!(both_err > queries / 20, "{both_err} failures in {queries}");
    assert!(
        recovered > 0 && confirmed > 0,
        "{recovered} recovered, {confirmed} confirmed"
    );
    assert!(tested < reference_tested, "{tested} vs {reference_tested}");
}

/// Character capsules (radius 0.35, half height 0.55) over the rolling 65x65 terrain of the
/// heightfield bench: the best-first search must keep every hit and cut the triangle tests
/// (the old kernel: walk 3699, fall 4885, skim 4163, fast diagonal 11305; now 2716, 2788,
/// 3488, 7928).
#[test]
fn ordered_capsule_sweep_iterates_less_on_rolling_terrain() {
    let n = 65;
    let columns: Vec<f64> = (0..n).map(|i| f64::from(i) - 32.0).collect();
    let field = PreparedHeightfield3::try_new(HeightfieldData3 {
        columns: columns.clone(),
        rows: columns.clone(),
        heights: (0..(n * n) as usize)
            .map(|k| {
                let (x, z) = (columns[k % n as usize], columns[k / n as usize]);
                1.2 * (x * 0.21).sin() * (z * 0.17).cos() + 0.3 * (x * 0.9 + z * 0.4).sin()
            })
            .collect(),
        active_cells: None,
        diagonal: HeightfieldDiagonal3::Fixed,
    })
    .expect("valid terrain");
    let (radius, half) = (0.35, 0.55);
    for (motion, gap, d, most) in [
        ("walk", 0.02, [0.12, -0.03, 0.07], 0.75),
        ("fall", 0.5, [0.05, -1.2, 0.0], 0.6),
        ("skim", 0.0, [0.3, 0.0, 0.2], 0.85),
        ("fast diagonal", 0.1, [1.5, -0.8, 1.1], 0.72),
    ] {
        let (mut work, mut reference_work) =
            (HeightfieldWork3::default(), HeightfieldWork3::default());
        for i in 0..256 {
            let (x, z) = (
                f64::from(i % 16) * 3.1 - 24.0,
                f64::from(i / 16) * 2.9 - 23.0,
            );
            let surface = field
                .sample(x, z)
                .expect("inside")
                .surface
                .expect("no holes");
            let bottom = surface.point[1] + radius + gap;
            let (p, q) = ([x, bottom, z], [x, bottom + 2.0 * half, z]);
            let actual = field.capsule_sweep_counted(p, q, radius, d, 64, &mut work);
            let expected =
                reference_capsule_sweep(&field, p, q, radius, d, 64, &mut reference_work);
            assert!(expected.is_ok(), "{motion} {i}");
            assert_eq!(sweep_bits(actual), sweep_bits(expected), "{motion} {i}");
        }
        let (tested, reference) = (work.triangles_tested, reference_work.triangles_tested);
        println!("{motion}: {reference} -> {tested} triangle iterations");
        assert!(
            (tested as f64) < most * reference as f64,
            "{motion}: {tested} vs {reference}"
        );
    }
}
