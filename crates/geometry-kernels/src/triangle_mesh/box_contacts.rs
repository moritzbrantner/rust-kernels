//! Box/triangle feature contacts in the box's orthonormal local frame.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TriangleMeshBoxPoint3 {
    /// Signed distance along the manifold normal; negative means penetration.
    pub separation: f64,
    pub point_on_box: Vec3,
    pub point_on_triangle: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleMeshBoxManifold3 {
    pub triangle: u32,
    /// Unit normal from the two-sided mesh toward the box.
    pub normal: Vec3,
    pub points: [TriangleMeshBoxPoint3; 4],
    /// Only `points[..len]` are contacts; always in `1..=4`.
    pub len: usize,
}

impl PreparedTriangleMesh3 {
    /// Emits at most four contacts per candidate triangle without heap allocation.
    ///
    /// `axes` are the box's orthonormal local X/Y/Z directions in mesh space.
    /// Extents must be positive, inputs finite, and margin nonnegative. Callbacks
    /// follow the retained deterministic BVH order, then deterministic feature
    /// order, rather than sorted triangle ids. Rebuilding identical input retains
    /// this order. No callback is made for triangles farther than `margin`.
    /// Earlier callbacks remain observable if a later candidate returns an error.
    #[allow(clippy::too_many_arguments)]
    pub fn box_contacts_counted(
        &self,
        center: Vec3,
        axes: [Vec3; 3],
        half_extents: Vec3,
        margin: f64,
        work: &mut TriangleMeshWork3,
        mut emit: impl FnMut(TriangleMeshBoxManifold3),
    ) -> Result<(), TriangleMeshError3> {
        if !center
            .into_iter()
            .chain(axes.into_iter().flatten())
            .all(f64::is_finite)
            || !half_extents.into_iter().all(|v| v.is_finite() && v > 0.0)
            || !margin.is_finite()
            || margin < 0.0
            || (0..3).any(|i| {
                (dot(axes[i], axes[i]) - 1.0).abs() > 1e-10
                    || (0..i).any(|j| dot(axes[i], axes[j]).abs() > 1e-10)
            })
        {
            return Err(TriangleMeshError3::InvalidQuery);
        }
        let extents = std::array::from_fn(|i| {
            (0..3).fold(0.0_f64, |sum, j| {
                (sum + (axes[j][i].abs() * half_extents[j]).next_up()).next_up()
            })
        });
        let bounds =
            Bounds::points([sub(center, extents), add(center, extents)]).expanded(margin)?;
        work.queries += 1;
        self.visit(bounds, work, |id, work| {
            work.triangles_tested += 1;
            work.normal_reuses += 1;
            let triangle = self.triangle(id).map(|p| {
                let d = sub(p, center);
                axes.map(|axis| dot(d, axis))
            });
            let normal = axes.map(|axis| dot(self.normals[id as usize], axis));
            if !triangle.into_iter().flatten().all(f64::is_finite) {
                return Err(TriangleMeshError3::NonFiniteComputation);
            }
            if let Some(mut manifold) = contacts(triangle, normal, half_extents, margin)? {
                manifold.triangle = id;
                let rotate = |p: Vec3| (0..3).fold([0.0; 3], |v, i| add(v, scale(axes[i], p[i])));
                manifold.normal = rotate(manifold.normal);
                for point in &mut manifold.points[..manifold.len] {
                    point.point_on_box = add(center, rotate(point.point_on_box));
                    point.point_on_triangle = add(center, rotate(point.point_on_triangle));
                    if !point
                        .point_on_box
                        .into_iter()
                        .chain(point.point_on_triangle)
                        .all(f64::is_finite)
                    {
                        return Err(TriangleMeshError3::NonFiniteComputation);
                    }
                }
                emit(manifold);
            }
            Ok(())
        })
    }
}

#[derive(Clone, Copy)]
enum Feature {
    TriangleFace,
    BoxFace(usize),
    Edge(usize, usize),
}
const BASIS: [Vec3; 3] = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];

fn unit(v: Vec3) -> Option<Vec3> {
    let m = v.into_iter().map(f64::abs).fold(0., f64::max);
    if m == 0. || !m.is_finite() {
        return None;
    }
    let v = v.map(|x| x / m);
    Some(scale(v, length(v).recip()))
}

fn contacts(
    t: [Vec3; 3],
    tn: Vec3,
    h: Vec3,
    margin: f64,
) -> Result<Option<TriangleMeshBoxManifold3>, TriangleMeshError3> {
    let tn = unit(tn).ok_or(TriangleMeshError3::NonFiniteComputation)?;
    let edges = [sub(t[1], t[0]), sub(t[2], t[1]), sub(t[0], t[2])];
    if !edges.into_iter().flatten().all(f64::is_finite) {
        return Err(TriangleMeshError3::NonFiniteComputation);
    }
    let mut best = (f64::NEG_INFINITY, tn, Feature::TriangleFace);
    let mut test = |axis: Vec3, feature: Feature| -> Result<bool, TriangleMeshError3> {
        let radius = (0..3).map(|i| h[i] * axis[i].abs()).sum::<f64>();
        let ds = t.map(|p| dot(p, axis));
        if !radius.is_finite() || !ds.into_iter().all(f64::is_finite) {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
        let lo = ds.into_iter().fold(f64::INFINITY, f64::min);
        let hi = ds.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let positive = -radius - hi;
        let negative = lo - radius;
        let (sep, n) = if positive >= negative {
            (positive, axis)
        } else {
            (negative, scale(axis, -1.))
        };
        if !sep.is_finite() {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
        if sep > margin {
            return Ok(false);
        }
        // Face-first strict tie breaking keeps coplanar patches stable.
        let tie_tolerance = 64. * f64::EPSILON * radius.max(lo.abs()).max(hi.abs());
        if sep > best.0 + tie_tolerance {
            best = (sep, n, feature);
        }
        Ok(true)
    };
    if !test(tn, Feature::TriangleFace)? {
        return Ok(None);
    }
    for (i, axis) in BASIS.into_iter().enumerate() {
        if !test(axis, Feature::BoxFace(i))? {
            return Ok(None);
        }
    }
    for (j, edge) in edges.into_iter().enumerate() {
        // Scale first so tiny or huge valid edges retain their direction.
        let e = unit(edge).ok_or(TriangleMeshError3::NonFiniteComputation)?;
        for (i, axis) in BASIS.into_iter().enumerate() {
            let axis = cross(axis, e);
            let axis_length = length(axis);
            if axis_length > 64. * f64::EPSILON
                && !test(scale(axis, axis_length.recip()), Feature::Edge(i, j))?
            {
                return Ok(None);
            }
        }
    }
    let (separation, normal, feature) = best;
    let mut points = [TriangleMeshBoxPoint3::default(); 12];
    let mut count = 0;
    let tolerance = 128.
        * f64::EPSILON
        * h.into_iter()
            .chain(t.into_iter().flatten())
            .map(f64::abs)
            .fold(0., f64::max);
    match feature {
        Feature::TriangleFace => {
            let face = (0..3)
                .max_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))
                .unwrap_or(0);
            let side = if normal[face] >= 0. { -1. } else { 1. };
            let u = (face + 1) % 3;
            let v = (face + 2) % 3;
            let mut polygon = Polygon::empty();
            for (su, sv) in [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)] {
                let mut p = [0.; 3];
                p[face] = side * h[face];
                p[u] = su * h[u];
                p[v] = sv * h[v];
                polygon.push(p)?;
            }
            for i in 0..3 {
                let inward = unit(cross(
                    tn,
                    unit(edges[i]).ok_or(TriangleMeshError3::NonFiniteComputation)?,
                ))
                .ok_or(TriangleMeshError3::NonFiniteComputation)?;
                polygon.clip(inward, t[i], tolerance)?;
            }
            for &p in &polygon.points[..polygon.len] {
                let sep = dot(sub(p, t[0]), normal);
                if sep <= margin {
                    points[count] = TriangleMeshBoxPoint3 {
                        separation: sep,
                        point_on_box: p,
                        point_on_triangle: sub(p, scale(normal, sep)),
                    };
                    count += 1;
                }
            }
        }
        Feature::BoxFace(face) => {
            let mut polygon = Polygon::empty();
            for p in t {
                polygon.push(p)?;
            }
            for i in 0..3 {
                if i != face {
                    let mut p = [0.; 3];
                    p[i] = -h[i];
                    polygon.clip(BASIS[i], p, tolerance)?;
                    p[i] = h[i];
                    polygon.clip(scale(BASIS[i], -1.), p, tolerance)?;
                }
            }
            let box_plane = -h[face]; // projection of the box face onto mesh->box normal
            for &p in &polygon.points[..polygon.len] {
                let sep = box_plane - dot(p, normal);
                if sep <= margin {
                    points[count] = TriangleMeshBoxPoint3 {
                        separation: sep,
                        point_on_box: add(p, scale(normal, sep)),
                        point_on_triangle: p,
                    };
                    count += 1;
                }
            }
        }
        Feature::Edge(i, j) => {
            let mut a = std::array::from_fn(|k| if normal[k] >= 0. { -h[k] } else { h[k] });
            let mut b = a;
            a[i] = -h[i];
            b[i] = h[i];
            let (p, q) = closest_segment_points(a, b, t[j], t[(j + 1) % 3])
                .ok_or(TriangleMeshError3::NonFiniteComputation)?;
            points[0] = TriangleMeshBoxPoint3 {
                separation: dot(sub(p, q), normal),
                point_on_box: p,
                point_on_triangle: q,
            };
            count = 1;
        }
    }
    // Positive SAT gaps are lower bounds, not Euclidean distances. At corners
    // or an empty clipped face use exact closest feature witnesses; do not admit
    // speculative contacts merely because each projected gap fits the margin.
    if count == 0 || (separation > 0. && matches!(feature, Feature::Edge(_, _))) {
        let (p, q, distance) = closest(t, h)?;
        if distance > margin {
            return Ok(None);
        }
        if count == 0 || matches!(feature, Feature::Edge(_, _)) {
            let n = if distance > 0. {
                scale(sub(p, q), distance.recip())
            } else {
                normal
            };
            return Ok(Some(TriangleMeshBoxManifold3 {
                triangle: 0,
                normal: n,
                points: [TriangleMeshBoxPoint3 {
                    separation: distance,
                    point_on_box: p,
                    point_on_triangle: q,
                }; 4],
                len: 1,
            }));
        }
    }
    // Remove clipping duplicates, then retain a spread of at most four witnesses.
    let mut unique = 0;
    for i in 0..count {
        if !(0..unique).any(|j| {
            length(sub(
                points[i].point_on_triangle,
                points[j].point_on_triangle,
            )) <= tolerance
        }) {
            points[unique] = points[i];
            unique += 1;
        }
    }
    let mut result = TriangleMeshBoxManifold3 {
        triangle: 0,
        normal,
        points: [TriangleMeshBoxPoint3::default(); 4],
        len: unique.min(4),
    };
    if unique <= 4 {
        result.points[..unique].copy_from_slice(&points[..unique]);
    } else {
        let mut selected = [usize::MAX; 4];
        selected[0] = (0..unique)
            .min_by(|&a, &b| points[a].separation.total_cmp(&points[b].separation))
            .unwrap_or(0);
        for k in 1..4 {
            selected[k] = (0..unique)
                .filter(|i| !selected[..k].contains(i))
                .max_by(|&a, &b| {
                    let score = |i: usize| {
                        selected[..k]
                            .iter()
                            .map(|&j| {
                                length(sub(
                                    points[i].point_on_triangle,
                                    points[j].point_on_triangle,
                                ))
                            })
                            .fold(f64::INFINITY, f64::min)
                    };
                    score(a).total_cmp(&score(b)).then(b.cmp(&a))
                })
                .unwrap_or(0);
        }
        for (i, j) in selected.into_iter().enumerate() {
            result.points[i] = points[j];
        }
    }
    if result.points[..result.len].iter().any(|p| {
        !p.separation.is_finite()
            || !p
                .point_on_box
                .into_iter()
                .chain(p.point_on_triangle)
                .all(f64::is_finite)
    }) {
        return Err(TriangleMeshError3::NonFiniteComputation);
    }
    Ok(Some(result))
}

struct Polygon {
    points: [Vec3; 12],
    len: usize,
}
impl Polygon {
    fn empty() -> Self {
        Self {
            points: [[0.; 3]; 12],
            len: 0,
        }
    }
    fn push(&mut self, p: Vec3) -> Result<(), TriangleMeshError3> {
        if !p.into_iter().all(f64::is_finite) {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
        if self.len == self.points.len() {
            return Err(TriangleMeshError3::Capacity);
        }
        self.points[self.len] = p;
        self.len += 1;
        Ok(())
    }
    fn clip(&mut self, n: Vec3, origin: Vec3, tolerance: f64) -> Result<(), TriangleMeshError3> {
        let mut out = Self::empty();
        for i in 0..self.len {
            let a = self.points[i];
            let b = self.points[(i + 1) % self.len];
            let da = dot(sub(a, origin), n);
            let db = dot(sub(b, origin), n);
            if !da.is_finite() || !db.is_finite() {
                return Err(TriangleMeshError3::NonFiniteComputation);
            }
            let inside_a = da >= -tolerance;
            let inside_b = db >= -tolerance;
            if inside_a {
                out.push(a)?;
            }
            if inside_a != inside_b {
                let fraction = (da / (da - db)).clamp(0., 1.);
                out.push(add(scale(a, 1. - fraction), scale(b, fraction)))?;
            }
        }
        *self = out;
        Ok(())
    }
}

fn closest(t: [Vec3; 3], h: Vec3) -> Result<(Vec3, Vec3, f64), TriangleMeshError3> {
    let vertices: [Vec3; 8] = std::array::from_fn(|i| {
        std::array::from_fn(|j| if i & (1 << j) == 0 { -h[j] } else { h[j] })
    });
    let mut best = (vertices[0], t[0], f64::INFINITY);
    let mut consider = |p: Vec3, q: Vec3| -> Result<(), TriangleMeshError3> {
        let d = length(sub(p, q));
        if !d.is_finite() {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
        if d < best.2 {
            best = (p, q, d);
        }
        Ok(())
    };
    for p in vertices {
        consider(p, closest_point_triangle(p, t[0], t[1], t[2]))?;
    }
    for q in t {
        consider(std::array::from_fn(|i| q[i].clamp(-h[i], h[i])), q)?;
    }
    for (i, p) in vertices.into_iter().enumerate() {
        for axis in 0..3 {
            if i & (1 << axis) == 0 {
                for j in 0..3 {
                    let (p, q) =
                        closest_segment_points(p, vertices[i | (1 << axis)], t[j], t[(j + 1) % 3])
                            .ok_or(TriangleMeshError3::NonFiniteComputation)?;
                    consider(p, q)?;
                }
            }
        }
    }
    Ok(best)
}
