//! Differential fuzz of the best-first mesh capsule sweep against a copy of the id-order
//! (BVH-order) kernel it replaced. Every old `Ok` must be bit-identical, every new `Err` must be
//! an old `Err` of the same kind, and an old `Err` may become `Ok` only when it is the result
//! the old kernel settles to with unbounded iterations.

use super::*;

/// The capsule sweep before best-first search: triangles in BVH order, each iterating until it
/// settles or passes the best hit so far, failing the sweep when it runs out of iterations
/// first.
#[allow(
    clippy::too_many_lines,
    reason = "verbatim copy of the replaced kernel"
)]
fn reference_capsule_sweep(
    mesh: &PreparedTriangleMesh3,
    p: Vec3,
    q: Vec3,
    radius: f64,
    displacement: Vec3,
    max_iterations: u32,
    work: &mut TriangleMeshWork3,
) -> Result<Option<TriangleMeshSweepHit3>, TriangleMeshError3> {
    validate_capsule(p, q, radius)?;
    if !displacement.into_iter().all(f64::is_finite) {
        return Err(TriangleMeshError3::InvalidQuery);
    }
    work.queries += 1;
    let end_p = add(p, displacement);
    let end_q = add(q, displacement);
    if !end_p.into_iter().chain(end_q).all(f64::is_finite) {
        return Err(TriangleMeshError3::NonFiniteComputation);
    }
    let magnitude = p
        .into_iter()
        .chain(q)
        .chain(end_p)
        .chain(end_q)
        .map(f64::abs)
        .fold(radius, f64::max);
    let tolerance = 64.0 * f64::EPSILON * (1.0 + magnitude);
    let bounds = Bounds::points([p, q, end_p, end_q]).expanded(radius + tolerance)?;
    let speed = length(displacement);
    if !speed.is_finite() {
        return Err(TriangleMeshError3::NonFiniteComputation);
    }
    let mut best: Option<TriangleMeshSweepHit3> = None;
    mesh.visit(bounds, work, |id, work| {
        work.triangles_tested += 1;
        work.normal_reuses += 1;
        let mut time = 0.0;
        for _ in 0..max_iterations {
            if best.is_some_and(|b| time > b.fraction) {
                return Ok(());
            }
            work.sweep_iterations += 1;
            let offset = scale(displacement, time);
            let c = mesh.contact(id, add(p, offset), add(q, offset), radius)?;
            let rate = dot(displacement, c.normal);
            if !rate.is_finite() {
                return Err(TriangleMeshError3::NonFiniteComputation);
            }
            if c.separation <= tolerance {
                if time == 0.0 && rate >= 0.0 {
                    return Ok(());
                }
                let hit = TriangleMeshSweepHit3 {
                    fraction: time,
                    contact: c,
                };
                if best.is_none_or(|b| {
                    time < b.fraction
                        || (time == b.fraction
                            && rate
                                .total_cmp(&dot(displacement, b.contact.normal))
                                .then(id.cmp(&b.contact.triangle))
                                .is_lt())
                }) {
                    best = Some(hit);
                }
                return Ok(());
            }
            if speed == 0.0 || rate >= 0.0 || time + (c.separation - tolerance) / speed > 1.0 {
                return Ok(());
            }
            let next = time + (c.separation - tolerance * 0.5) / -rate;
            if !next.is_finite() {
                return Err(TriangleMeshError3::NonFiniteComputation);
            }
            if next > 1.0 {
                return Ok(());
            }
            if next <= time {
                return Err(TriangleMeshError3::IterationLimit);
            }
            time = next;
        }
        Err(TriangleMeshError3::IterationLimit)
    })?;
    Ok(best)
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

/// A closed tube of `rings` rings of `around` vertices along X, wobbling radius `size`, its
/// floor flattened to coplanar triangles, both ends capped by fans: a cave.
pub(super) fn cave(
    rings: usize,
    around: usize,
    size: f64,
    wobble: f64,
    offset: Vec3,
) -> TriangleMeshData3 {
    let mut vertices = Vec::new();
    let half = (rings - 1) as f64 * 0.5;
    for i in 0..rings {
        let x = i as f64 - half;
        for k in 0..around {
            let angle = std::f64::consts::TAU * k as f64 / around as f64;
            let r = size
                * (1.0
                    + wobble
                        * ((0.7 * x + 3.0 * angle).sin() + 0.5 * (1.3 * x - 2.0 * angle).cos()));
            let y = (r * angle.sin()).max(-0.6 * size + 0.05 * (0.9 * x).sin());
            vertices.push(add(offset, [x, y, r * angle.cos()]));
        }
    }
    let ring = |i: usize, k: usize| (i * around + k % around) as u32;
    let mut triangles = Vec::new();
    for i in 0..rings - 1 {
        for k in 0..around {
            triangles.push([ring(i, k), ring(i + 1, k), ring(i + 1, k + 1)]);
            triangles.push([ring(i, k), ring(i + 1, k + 1), ring(i, k + 1)]);
        }
    }
    for (i, x) in [(0, -half - 0.5), (rings - 1, half + 0.5)] {
        let centre = vertices.len() as u32;
        vertices.push(add(offset, [x, 0.0, 0.0]));
        for k in 0..around {
            triangles.push([centre, ring(i, k + 1), ring(i, k)]);
        }
    }
    TriangleMeshData3 {
        vertices,
        triangles,
    }
}

/// A random mesh: a closed cave, a jittered grid surface (shared edges and vertices,
/// optionally with coplanar or quantized heights), or a soup of free, thin and nearly
/// degenerate triangles, some sharing vertices; at offsets up to 3e4.
fn mesh(rng: &mut Rng) -> PreparedTriangleMesh3 {
    let offset = [0.0, 0.0, 1e3, -3e4][rng.below(4) as usize];
    let offset = [offset, rng.range(-2.0, 2.0), -offset];
    let data = match rng.below(3) {
        0 => cave(
            2 + rng.below(6) as usize,
            3 + rng.below(8) as usize,
            rng.range(0.8, 3.0),
            [0.0, 0.05, 0.2][rng.below(3) as usize],
            offset,
        ),
        1 => {
            let (n, m) = (2 + rng.below(6) as usize, 2 + rng.below(6) as usize);
            let amplitude = [0.0, 0.3, 2.0][rng.below(3) as usize];
            let quantized = rng.below(3) == 0;
            let mut vertices = Vec::new();
            for i in 0..n {
                for k in 0..m {
                    let h = rng.range(-amplitude, amplitude);
                    let h = if quantized { h.round() } else { h };
                    vertices.push(add(
                        offset,
                        [i as f64 + rng.range(-0.3, 0.3) * amplitude, h, k as f64],
                    ));
                }
            }
            let at = |i: usize, k: usize| (i * m + k) as u32;
            let mut triangles = Vec::new();
            for i in 0..n - 1 {
                for k in 0..m - 1 {
                    if rng.below(2) == 0 {
                        triangles.push([at(i, k), at(i, k + 1), at(i + 1, k + 1)]);
                        triangles.push([at(i, k), at(i + 1, k + 1), at(i + 1, k)]);
                    } else {
                        triangles.push([at(i, k), at(i, k + 1), at(i + 1, k)]);
                        triangles.push([at(i + 1, k), at(i, k + 1), at(i + 1, k + 1)]);
                    }
                }
            }
            TriangleMeshData3 {
                vertices,
                triangles,
            }
        }
        _ => {
            let mut vertices: Vec<Vec3> = Vec::new();
            let mut triangles = Vec::new();
            let extent = rng.range(1.0, 6.0);
            for _ in 0..1 + rng.below(24) {
                let shared = |rng: &mut Rng, vertices: &Vec<Vec3>| {
                    (!vertices.is_empty() && rng.below(3) == 0)
                        .then(|| rng.below(vertices.len() as u64) as u32)
                };
                let corner = |rng: &mut Rng, vertices: &mut Vec<Vec3>| {
                    shared(rng, vertices).unwrap_or_else(|| {
                        vertices.push(add(
                            offset,
                            [
                                rng.range(-extent, extent),
                                rng.range(-extent, extent),
                                rng.range(-extent, extent),
                            ],
                        ));
                        (vertices.len() - 1) as u32
                    })
                };
                let a = corner(rng, &mut vertices);
                let b = corner(rng, &mut vertices);
                let c = if rng.below(3) == 0 {
                    // Thin or nearly degenerate: near the a–b edge, off by a sliver.
                    let (va, vb) = (vertices[a as usize], vertices[b as usize]);
                    let t = rng.range(-0.5, 1.5);
                    let sliver = [1e-9, 1e-6, 1e-3, 0.05][rng.below(4) as usize];
                    vertices.push(add(
                        add(va, scale(sub(vb, va), t)),
                        scale(rng.direction(), sliver),
                    ));
                    (vertices.len() - 1) as u32
                } else {
                    corner(rng, &mut vertices)
                };
                triangles.push([a, b, c]);
            }
            TriangleMeshData3 {
                vertices,
                triangles,
            }
        }
    };
    PreparedTriangleMesh3::try_new(data.clone())
        .or_else(|_| {
            // A random soup may repeat a vertex in one triangle: drop the degenerate ones.
            let triangles: Vec<_> = data
                .triangles
                .iter()
                .copied()
                .filter(|t| {
                    PreparedTriangleMesh3::try_new(TriangleMeshData3 {
                        vertices: data.vertices.clone(),
                        triangles: vec![*t],
                    })
                    .is_ok()
                })
                .collect();
            if triangles.is_empty() {
                PreparedTriangleMesh3::try_new(TriangleMeshData3 {
                    vertices: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    triangles: vec![[0, 2, 1]],
                })
            } else {
                PreparedTriangleMesh3::try_new(TriangleMeshData3 {
                    vertices: data.vertices,
                    triangles,
                })
            }
        })
        .expect("filtered meshes are valid")
}

/// A random capsule near `mesh`: free placements, and placements whose endpoint sits just
/// outside (or inside) a triangle's face, edge or vertex at the capsule's radius, with that
/// triangle's normal.
fn capsule(rng: &mut Rng, mesh: &PreparedTriangleMesh3) -> (Vec3, Vec3, f64, Option<Vec3>) {
    let radius = match rng.below(4) {
        0 => 0.0,
        _ => rng.range(0.05, 1.0),
    };
    let (low, high) = mesh.bounds();
    let (p, direction, near) = if rng.below(3) == 0 {
        (
            std::array::from_fn(|i| rng.range(low[i] - 1.0, high[i] + 1.0)),
            rng.direction(),
            None,
        )
    } else {
        let id = rng.below(mesh.normals.len() as u64) as u32;
        let [a, b, c] = mesh.triangle(id);
        let normal = mesh.normals[id as usize];
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
        let on = add(a, add(scale(sub(b, a), u), scale(sub(c, a), v)));
        let away = match rng.below(4) {
            0 => scale(normal, -1.0),
            1 => rng.direction(),
            _ => normal,
        };
        let magnitude = on.into_iter().map(f64::abs).fold(radius, f64::max);
        let tolerance = 64.0 * f64::EPSILON * (1.0 + magnitude);
        let offset = match rng.below(6) {
            0 => 0.0,
            1 => rng.range(-4.0, 4.0) * f64::EPSILON * (1.0 + magnitude),
            2 => rng.range(0.0, 2.0) * tolerance,
            3 => rng.range(-1e-9, 1e-9) * (1.0 + magnitude),
            4 => rng.range(0.0, 1e-6),
            _ => rng.range(-0.2, 0.5),
        };
        (
            add(on, scale(away, radius + offset)),
            {
                let w = rng.direction();
                match rng.below(3) {
                    0 => sub(w, scale(normal, dot(w, normal))),
                    1 => away,
                    _ => w,
                }
            },
            Some(normal),
        )
    };
    let length = match rng.below(4) {
        0 => 0.0,
        _ => rng.range(0.0, 2.0),
    };
    (p, add(p, scale(direction, length)), radius, near)
}

/// A sweep displacement for a capsule near a triangle with normal `near`: anywhere, along the
/// plane (grazing), or along it with a near-tangent component of a few ulps to 1e-6 into or
/// out of it.
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
    let along = sub(direction, scale(normal, dot(direction, normal)));
    let tilt = match rng.below(4) {
        0 => 0.0,
        1 => rng.range(-8.0, 8.0) * f64::EPSILON,
        2 => rng.range(-1e-9, 1e-9),
        _ => rng.range(-1e-6, 1e-6),
    };
    scale(add(along, scale(normal, tilt)), length)
}

type SweepBits = Option<(u64, u32, u64, [u64; 3], [u64; 3])>;

fn sweep_bits(
    result: Result<Option<TriangleMeshSweepHit3>, TriangleMeshError3>,
) -> Result<SweepBits, TriangleMeshError3> {
    result.map(|hit| {
        hit.map(|h| {
            (
                h.fraction.to_bits(),
                h.contact.triangle,
                h.contact.separation.to_bits(),
                h.contact.normal.map(f64::to_bits),
                h.contact.point.map(f64::to_bits),
            )
        })
    })
}

#[test]
fn best_first_mesh_sweep_keeps_old_hits_and_fails_only_where_old_failed() {
    let mut rng = Rng(0x5eed_0527_3e5b);
    let (mut queries, mut hits, mut both_err, mut recovered) = (0, 0, 0, 0);
    let (mut recovered_at_64, mut recovered_none) = (0, 0);
    let (mut iterations, mut reference_iterations) = (0_u64, 0_u64);
    for _ in 0..1000 {
        let mesh = mesh(&mut rng);
        for _ in 0..40 {
            let (p, q, radius, near) = capsule(&mut rng, &mesh);
            let d = displacement(&mut rng, near);
            let max_iterations = if rng.below(2) == 0 {
                [0, 1, 2, 3, 4, 8, 16, 64][rng.below(8) as usize]
            } else {
                rng.below(65) as u32
            };
            let (mut work, mut reference_work) =
                (TriangleMeshWork3::default(), TriangleMeshWork3::default());
            let actual = mesh.capsule_sweep_counted(p, q, radius, d, max_iterations, &mut work);
            let expected = reference_capsule_sweep(
                &mesh,
                p,
                q,
                radius,
                d,
                max_iterations,
                &mut reference_work,
            );
            let case = || {
                format!(
                    "p {p:?} q {q:?} radius {radius} d {d:?} max {max_iterations} mesh {:?}",
                    mesh.data
                )
            };
            queries += 1;
            match (expected, actual) {
                (Ok(_), _) => {
                    assert_eq!(sweep_bits(actual), sweep_bits(expected), "{}", case());
                    hits += u32::from(matches!(expected, Ok(Some(_))));
                    iterations += work.sweep_iterations;
                    reference_iterations += reference_work.sweep_iterations;
                }
                (Err(old), Err(new)) => {
                    assert_eq!(new, old, "{}", case());
                    both_err += 1;
                }
                (Err(_), Ok(hit)) => {
                    // A failure past the earliest hit (or, without one, of a triangle the motion
                    // cannot reach): the result must be the one the old sweep settles to.
                    recovered += 1;
                    recovered_at_64 += u32::from(max_iterations == 64);
                    recovered_none += u32::from(hit.is_none());
                    let settled = reference_capsule_sweep(
                        &mesh,
                        p,
                        q,
                        radius,
                        d,
                        1 << 20,
                        &mut TriangleMeshWork3::default(),
                    );
                    assert_eq!(sweep_bits(Ok(hit)), sweep_bits(settled), "{}", case());
                }
            }
        }
    }
    println!(
        "{queries} sweeps: {hits} old hits kept, {both_err} failures kept, {recovered} failures \
         became the result the old sweep settles to with 2^20 iterations ({recovered_none} no \
         hit, {recovered_at_64} at 64 iterations); sweep iterations {iterations} vs \
         {reference_iterations} on old successes"
    );
    // The fuzz must exercise hits, kept failures and recovered failures.
    assert!(hits > queries / 10, "{hits} hits in {queries}");
    assert!(both_err > queries / 20, "{both_err} failures in {queries}");
    assert!(recovered > 0, "{recovered} recovered");
    assert!(
        iterations < reference_iterations,
        "{iterations} vs {reference_iterations}"
    );
}

/// Character capsules (radius 0.35, half height 0.55) on the floor of the closed cave of the
/// `triangle_mesh` bench (41 rings of 24, 1968 triangles): the best-first search must keep
/// every hit and cut the Newton steps (the old kernel: walk 2376, fall 4160, skim 2098, fast
/// diagonal 8247; now 1755, 1707, 1691, 3976).
#[test]
fn best_first_mesh_sweep_iterates_less_in_a_cave() {
    let mesh =
        PreparedTriangleMesh3::try_new(cave(41, 24, 3.0, 0.1, [0.0; 3])).expect("valid cave");
    let (radius, half) = (0.35, 0.55);
    for (motion, gap, d, most) in [
        ("walk", 0.02, [0.12, -0.03, 0.07], 0.8),
        ("fall", 0.5, [0.05, -1.2, 0.0], 0.5),
        ("skim", 0.0, [0.3, 0.0, 0.2], 0.9),
        ("fast diagonal", 0.1, [1.5, -0.8, 1.1], 0.55),
    ] {
        let (mut work, mut reference_work) =
            (TriangleMeshWork3::default(), TriangleMeshWork3::default());
        let mut hits = 0;
        for i in 0..256 {
            let (x, z) = (
                f64::from(i % 32) * 1.1 - 17.0,
                f64::from(i / 32) * 0.4 - 1.4,
            );
            let floor = mesh
                .segment_hit([x, 0.0, z], [x, -4.0, z])
                .expect("valid ray")
                .expect("over the floor")
                .point[1];
            let bottom = floor + radius + gap;
            let (p, q) = ([x, bottom, z], [x, bottom + 2.0 * half, z]);
            let actual = mesh.capsule_sweep_counted(p, q, radius, d, 64, &mut work);
            let expected = reference_capsule_sweep(&mesh, p, q, radius, d, 64, &mut reference_work);
            assert!(expected.is_ok(), "{motion} {i}");
            assert_eq!(sweep_bits(actual), sweep_bits(expected), "{motion} {i}");
            hits += u32::from(matches!(expected, Ok(Some(_))));
        }
        let (steps, reference) = (work.sweep_iterations, reference_work.sweep_iterations);
        println!(
            "{motion}: {hits} hits; sweep iterations {reference} -> {steps}, triangles tested {} -> {}",
            reference_work.triangles_tested, work.triangles_tested
        );
        assert!(
            (steps as f64) < most * reference as f64,
            "{motion}: {steps} vs {reference}"
        );
    }
}
