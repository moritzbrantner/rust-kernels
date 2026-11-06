//! Immutable two-sided triangle surfaces with a deterministic local BVH.
//!
//! Triangle identity is its input index, including across BVH rebuilds. A mesh is
//! a set of surfaces, not a filled solid: open boundaries and holes remain open.
//! Capsule queries include spheres (equal segment endpoints); sweeps translate a
//! fixed segment. Rotation, collision response and chunk lifetime belong to callers.

mod box_contact;
pub use box_contact::{TriangleMeshBoxContact3, TriangleMeshBoxPoint3};

use crate::{
    heightfield::{ConvexPolytope3, polytope_triangle_contact},
    math3::{Vec3, add, cross, dot, length, scale, sub},
    primitive3::{closest_point_triangle, closest_segment_points},
    ray_triangle::segment_triangle_f64,
};
use std::{error::Error, fmt, mem::size_of};

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleMeshData3 {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TriangleMeshError3 {
    InvalidDimensions,
    InvalidCoordinates,
    InvalidIndex,
    DegenerateTriangle,
    InvalidQuery,
    NonFiniteComputation,
    IterationLimit,
    Capacity,
}
impl fmt::Display for TriangleMeshError3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "triangle mesh geometry failed: {self:?}")
    }
}
impl Error for TriangleMeshError3 {}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TriangleMeshPreparation3 {
    pub triangles_prepared: u64,
    pub nodes_prepared: u64,
    /// Vector payload capacities; excludes allocator and struct overhead.
    pub retained_bytes: usize,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TriangleMeshWork3 {
    pub queries: u64,
    pub nodes_visited: u64,
    pub triangles_tested: u64,
    pub normal_reuses: u64,
    pub sweep_iterations: u64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleMeshContact3 {
    pub triangle: u32,
    pub separation: f64,
    /// Unit normal from the two-sided triangle toward the capsule/polytope.
    pub normal: Vec3,
    pub point: Vec3,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleMeshSweepHit3 {
    pub fraction: f64,
    pub contact: TriangleMeshContact3,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleMeshSegmentHit3 {
    pub triangle: u32,
    pub fraction: f64,
    pub point: Vec3,
    /// Opposes segment travel, independent of triangle winding.
    pub normal: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}
impl Bounds {
    fn points(points: impl IntoIterator<Item = Vec3>) -> Self {
        let mut result = Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        };
        for p in points {
            for (axis, value) in p.into_iter().enumerate() {
                result.min[axis] = result.min[axis].min(value);
                result.max[axis] = result.max[axis].max(value);
            }
        }
        result
    }
    fn union(self, other: Self) -> Self {
        Self {
            min: std::array::from_fn(|i| self.min[i].min(other.min[i])),
            max: std::array::from_fn(|i| self.max[i].max(other.max[i])),
        }
    }
    fn expanded(self, reach: f64) -> Result<Self, TriangleMeshError3> {
        let result = Self {
            min: self.min.map(|v| (v - reach).next_down()),
            max: self.max.map(|v| (v + reach).next_up()),
        };
        if result.min.into_iter().chain(result.max).all(f64::is_finite) {
            Ok(result)
        } else {
            Err(TriangleMeshError3::NonFiniteComputation)
        }
    }
    fn overlaps(self, other: Self) -> bool {
        (0..3).all(|i| self.min[i] <= other.max[i] && self.max[i] >= other.min[i])
    }
}
#[derive(Debug)]
struct Node {
    bounds: Bounds,
    kind: NodeKind,
}
#[derive(Debug)]
enum NodeKind {
    Leaf(u32),
    Branch(usize, usize),
}
#[derive(Debug)]
pub struct PreparedTriangleMesh3 {
    data: TriangleMeshData3,
    normals: Vec<Vec3>,
    nodes: Vec<Node>,
    preparation: TriangleMeshPreparation3,
}
impl PreparedTriangleMesh3 {
    pub fn try_new(data: TriangleMeshData3) -> Result<Self, TriangleMeshError3> {
        use TriangleMeshError3::*;
        if data.vertices.is_empty() || data.triangles.is_empty() {
            return Err(InvalidDimensions);
        }
        if data.triangles.len() > u32::MAX as usize {
            return Err(Capacity);
        }
        if !data.vertices.iter().flatten().all(|v| v.is_finite()) {
            return Err(InvalidCoordinates);
        }
        let mut normals = Vec::with_capacity(data.triangles.len());
        let mut leaves = Vec::with_capacity(data.triangles.len());
        for (id, indices) in data.triangles.iter().enumerate() {
            let mut triangle = [[0.0; 3]; 3];
            for i in 0..3 {
                triangle[i] = *data.vertices.get(indices[i] as usize).ok_or(InvalidIndex)?;
            }
            let a = sub(triangle[1], triangle[0]);
            let b = sub(triangle[2], triangle[0]);
            if !a.into_iter().chain(b).all(f64::is_finite) {
                return Err(NonFiniteComputation);
            }
            // Normalize edges before crossing: finite small/large triangles do not
            // become degenerate solely because an unscaled area under/overflows.
            let span = a.into_iter().chain(b).map(f64::abs).fold(0.0, f64::max);
            if span == 0.0 {
                return Err(DegenerateTriangle);
            }
            let n = cross(a.map(|v| v / span), b.map(|v| v / span));
            let n_span = n.into_iter().map(f64::abs).fold(0.0, f64::max);
            if n_span == 0.0 {
                return Err(DegenerateTriangle);
            }
            let scaled = n.map(|v| v / n_span);
            let normal = scale(scaled, 1.0 / length(scaled));
            normals.push(normal);
            leaves.push((
                u32::try_from(id).map_err(|_| Capacity)?,
                Bounds::points(triangle),
            ));
        }
        let mut nodes =
            Vec::with_capacity(data.triangles.len().checked_mul(2).ok_or(Capacity)? - 1);
        fn build(leaves: &mut [(u32, Bounds)], nodes: &mut Vec<Node>) -> usize {
            let bounds = leaves.iter().skip(1).fold(leaves[0].1, |b, p| b.union(p.1));
            let at = nodes.len();
            nodes.push(Node {
                bounds,
                kind: NodeKind::Leaf(leaves[0].0),
            });
            if leaves.len() > 1 {
                let axis = (0..3)
                    .max_by(|&a, &b| {
                        (bounds.max[a] - bounds.min[a]).total_cmp(&(bounds.max[b] - bounds.min[b]))
                    })
                    .unwrap_or(0);
                leaves.sort_unstable_by(|a, b| {
                    let mid = |p: &(u32, Bounds)| p.1.min[axis] * 0.5 + p.1.max[axis] * 0.5;
                    mid(a).total_cmp(&mid(b)).then(a.0.cmp(&b.0))
                });
                let len = leaves.len() / 2;
                let (left, right) = leaves.split_at_mut(len);
                let a = build(left, nodes);
                let b = build(right, nodes);
                nodes[at].kind = NodeKind::Branch(a, b);
            }
            at
        }
        build(&mut leaves, &mut nodes);
        let preparation = TriangleMeshPreparation3 {
            triangles_prepared: data.triangles.len() as u64,
            nodes_prepared: nodes.len() as u64,
            retained_bytes: data.vertices.capacity() * size_of::<Vec3>()
                + data.triangles.capacity() * size_of::<[u32; 3]>()
                + normals.capacity() * size_of::<Vec3>()
                + nodes.capacity() * size_of::<Node>(),
        };
        Ok(Self {
            data,
            normals,
            nodes,
            preparation,
        })
    }
    pub fn data(&self) -> &TriangleMeshData3 {
        &self.data
    }
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let b = self.nodes[0].bounds;
        (b.min, b.max)
    }
    pub fn preparation(&self) -> TriangleMeshPreparation3 {
        self.preparation
    }
    fn triangle(&self, id: u32) -> [Vec3; 3] {
        self.data.triangles[id as usize].map(|i| self.data.vertices[i as usize])
    }
    fn visit(
        &self,
        bounds: Bounds,
        work: &mut TriangleMeshWork3,
        mut f: impl FnMut(u32, &mut TriangleMeshWork3) -> Result<(), TriangleMeshError3>,
    ) -> Result<(), TriangleMeshError3> {
        fn walk(
            mesh: &PreparedTriangleMesh3,
            at: usize,
            bounds: Bounds,
            work: &mut TriangleMeshWork3,
            f: &mut impl FnMut(u32, &mut TriangleMeshWork3) -> Result<(), TriangleMeshError3>,
        ) -> Result<(), TriangleMeshError3> {
            work.nodes_visited += 1;
            let node = &mesh.nodes[at];
            if !node.bounds.overlaps(bounds) {
                return Ok(());
            }
            match node.kind {
                NodeKind::Leaf(id) => f(id, work),
                NodeKind::Branch(a, b) => {
                    walk(mesh, a, bounds, work, f)?;
                    walk(mesh, b, bounds, work, f)
                }
            }
        }
        walk(self, 0, bounds, work, &mut f)
    }
    pub fn capsule_contact_counted(
        &self,
        p: Vec3,
        q: Vec3,
        radius: f64,
        margin: f64,
        work: &mut TriangleMeshWork3,
    ) -> Result<Option<TriangleMeshContact3>, TriangleMeshError3> {
        validate_capsule(p, q, radius)?;
        if !margin.is_finite() || margin < 0.0 {
            return Err(TriangleMeshError3::InvalidQuery);
        }
        work.queries += 1;
        let bounds = Bounds::points([p, q]).expanded(radius + margin)?;
        let mut best = None;
        self.visit(bounds, work, |id, work| {
            work.triangles_tested += 1;
            work.normal_reuses += 1;
            let c = self.contact(id, p, q, radius)?;
            if c.separation <= margin
                && best.is_none_or(|b: TriangleMeshContact3| {
                    c.separation < b.separation || (c.separation == b.separation && id < b.triangle)
                })
            {
                best = Some(c);
            }
            Ok(())
        })?;
        Ok(best)
    }
    pub fn capsule_contact(
        &self,
        p: Vec3,
        q: Vec3,
        radius: f64,
        margin: f64,
    ) -> Result<Option<TriangleMeshContact3>, TriangleMeshError3> {
        self.capsule_contact_counted(p, q, radius, margin, &mut TriangleMeshWork3::default())
    }
    fn contact(
        &self,
        id: u32,
        p: Vec3,
        q: Vec3,
        radius: f64,
    ) -> Result<TriangleMeshContact3, TriangleMeshError3> {
        let triangle = self.triangle(id);
        let (on_segment, on_triangle) =
            if let Some((_, point)) = segment_triangle_f64(p, q, triangle) {
                (point, point)
            } else {
                closest_segment_triangle(p, q, triangle)
                    .ok_or(TriangleMeshError3::NonFiniteComputation)?
            };
        let delta = sub(on_segment, on_triangle);
        let distance = length(delta);
        if !distance.is_finite() {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
        let mut normal = self.normals[id as usize];
        let mut separation = distance - radius;
        if distance > 0.0 {
            normal = scale(delta, 1.0 / distance);
        } else {
            // An intersecting centreline has two possible exit sides; choose
            // its midpoint side, with winding as the exact-on-plane tie.
            let centre = add(scale(p, 0.5), scale(q, 0.5));
            if dot(sub(centre, triangle[0]), normal) < 0.0 {
                normal = scale(normal, -1.0);
            }
            separation =
                dot(sub(p, triangle[0]), normal).min(dot(sub(q, triangle[0]), normal)) - radius;
        }
        if !separation.is_finite() {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
        Ok(TriangleMeshContact3 {
            triangle: id,
            separation,
            normal,
            point: on_triangle,
        })
    }
    pub fn capsule_sweep_counted(
        &self,
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
        self.visit(bounds, work, |id, work| {
            work.triangles_tested += 1;
            work.normal_reuses += 1;
            let mut time = 0.0;
            for _ in 0..max_iterations {
                if best.is_some_and(|b| time > b.fraction) {
                    return Ok(());
                }
                work.sweep_iterations += 1;
                let offset = scale(displacement, time);
                let c = self.contact(id, add(p, offset), add(q, offset), radius)?;
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
    pub fn capsule_sweep(
        &self,
        p: Vec3,
        q: Vec3,
        radius: f64,
        displacement: Vec3,
        max_iterations: u32,
    ) -> Result<Option<TriangleMeshSweepHit3>, TriangleMeshError3> {
        self.capsule_sweep_counted(
            p,
            q,
            radius,
            displacement,
            max_iterations,
            &mut TriangleMeshWork3::default(),
        )
    }
    pub fn segment_hit_counted(
        &self,
        start: Vec3,
        end: Vec3,
        work: &mut TriangleMeshWork3,
    ) -> Result<Option<TriangleMeshSegmentHit3>, TriangleMeshError3> {
        if !start.into_iter().chain(end).all(f64::is_finite) {
            return Err(TriangleMeshError3::InvalidQuery);
        }
        let delta = sub(end, start);
        if !delta.into_iter().all(f64::is_finite) {
            return Err(TriangleMeshError3::NonFiniteComputation);
        }
        work.queries += 1;
        let mut best: Option<TriangleMeshSegmentHit3> = None;
        self.visit(
            Bounds::points([start, end]).expanded(0.0)?,
            work,
            |id, work| {
                work.triangles_tested += 1;
                if let Some((fraction, point)) = segment_triangle_f64(start, end, self.triangle(id))
                    && best.is_none_or(|b| {
                        fraction < b.fraction || (fraction == b.fraction && id < b.triangle)
                    })
                {
                    let mut normal = self.normals[id as usize];
                    work.normal_reuses += 1;
                    if dot(delta, normal) > 0.0 {
                        normal = scale(normal, -1.0);
                    }
                    best = Some(TriangleMeshSegmentHit3 {
                        triangle: id,
                        fraction,
                        point,
                        normal,
                    });
                }
                Ok(())
            },
        )?;
        Ok(best)
    }
    pub fn segment_hit(
        &self,
        start: Vec3,
        end: Vec3,
    ) -> Result<Option<TriangleMeshSegmentHit3>, TriangleMeshError3> {
        self.segment_hit_counted(start, end, &mut TriangleMeshWork3::default())
    }
    pub fn polytope_contact_counted(
        &self,
        polytope: ConvexPolytope3<'_>,
        work: &mut TriangleMeshWork3,
    ) -> Result<Option<TriangleMeshContact3>, TriangleMeshError3> {
        if polytope.vertices.is_empty()
            || polytope.face_normals.is_empty()
            || polytope.edges.is_empty()
            || !polytope
                .vertices
                .iter()
                .chain(polytope.face_normals)
                .chain(polytope.edges)
                .flatten()
                .all(|v| v.is_finite())
        {
            return Err(TriangleMeshError3::InvalidQuery);
        }
        work.queries += 1;
        let mut best = None;
        self.visit(
            Bounds::points(polytope.vertices.iter().copied()).expanded(0.0)?,
            work,
            |id, work| {
                work.triangles_tested += 1;
                work.normal_reuses += 1;
                if let Some(c) = polytope_triangle_contact(
                    polytope,
                    self.triangle(id),
                    self.normals[id as usize],
                ) {
                    if !c.separation.is_finite()
                        || !c.point.into_iter().chain(c.normal).all(f64::is_finite)
                    {
                        return Err(TriangleMeshError3::NonFiniteComputation);
                    }
                    let c = TriangleMeshContact3 {
                        triangle: id,
                        separation: c.separation,
                        normal: c.normal,
                        point: c.point,
                    };
                    if best.is_none_or(|b: TriangleMeshContact3| {
                        c.separation < b.separation
                            || (c.separation == b.separation && id < b.triangle)
                    }) {
                        best = Some(c);
                    }
                }
                Ok(())
            },
        )?;
        Ok(best)
    }
}
fn validate_capsule(p: Vec3, q: Vec3, radius: f64) -> Result<(), TriangleMeshError3> {
    if !p.into_iter().chain(q).all(f64::is_finite) || !radius.is_finite() || radius < 0.0 {
        Err(TriangleMeshError3::InvalidQuery)
    } else {
        Ok(())
    }
}

/// Closest witnesses for a segment not crossing the triangle. Both callers
/// handle intersections first, retaining their distinct crossing policies.
pub(crate) fn closest_segment_triangle(
    p: Vec3,
    q: Vec3,
    triangle: [Vec3; 3],
) -> Option<(Vec3, Vec3)> {
    let [a, b, c] = triangle;
    let mut best = (f64::INFINITY, [0.0; 3], [0.0; 3]);
    let mut consider = |left: Vec3, right: Vec3| {
        let delta = sub(left, right);
        let d2 = dot(delta, delta);
        if d2 < best.0 {
            best = (d2, left, right);
        }
    };
    for end in [p, q] {
        let on_triangle = closest_point_triangle(end, a, b, c);
        if !on_triangle.into_iter().all(f64::is_finite) {
            return None;
        }
        consider(end, on_triangle);
    }
    for (e0, e1) in [(a, b), (b, c), (c, a)] {
        let (left, right) = closest_segment_points(p, q, e0, e1)?;
        consider(left, right);
    }
    best.0.is_finite().then_some((best.1, best.2))
}

#[cfg(test)]
mod tests;
