//! Differential fuzz of the pruned heightfield capsule kernels against copies of the unpruned
//! kernels they replaced: results must be bit-identical, including the chosen triangle.

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
/// pruning boundary of a triangle's face, edge or vertex (tangent and grazing).
fn capsule(rng: &mut Rng, field: &PreparedHeightfield3) -> (Vec3, Vec3, f64, f64) {
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
    let (p, direction) = match (rng.below(3), triangle) {
        (0, _) | (_, None) => {
            let x = rng.range(columns[0] - 1.0, columns[columns.len() - 1] + 1.0);
            let z = rng.range(rows[0] - 1.0, rows[rows.len() - 1] + 1.0);
            ([x, rng.range(-25.0, 25.0), z], rng.direction())
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
            (add(on, scale(away, reach + offset)), {
                // Along the plane (grazing), straight away, or anywhere.
                let w = rng.direction();
                match rng.below(3) {
                    0 => vsub(w, scale(normal, dot(w, normal))),
                    1 => away,
                    _ => w,
                }
            })
        }
    };
    let length = match rng.below(4) {
        0 => 0.0,
        _ => rng.range(0.0, 3.0),
    };
    (p, add(p, scale(direction, length)), radius, margin)
}

fn contact_bits(contact: Option<HeightfieldContact3>) -> Option<(u32, u64, [u64; 3], [u64; 3])> {
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
            let (p, q, radius, margin) = capsule(&mut rng, &field);
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
