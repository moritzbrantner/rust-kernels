//! Deterministic QuickHull construction in 2D and 3D.
//!
//! Output is canonical: it depends only on the *set* of input points, never on
//! their order or multiplicity. Every choice the algorithm makes (initial
//! extremes, farthest point per face, conflict assignment) is a maximum with a
//! total lexicographic tie-break, and the emitted vertices and triangles are
//! sorted before they are returned.
//!
//! Points closer than a scale-relative tolerance to a hull feature are treated
//! as lying on it, so duplicate, collinear and coplanar points never become
//! extra vertices. Degenerate inputs report their [`HullDimension`] instead of
//! failing.
//!
//! The 3D result feeds the GJK/EPA support-map path directly through
//! [`Hull3::support_map`].

use std::cmp::Ordering;

use crate::{
    math3::{Vec3, cross, dot, is_finite, length, sub},
    support::ConvexHull3,
};

pub type Vec2 = [f64; 2];

const NONE: u32 = u32::MAX;

/// Affine dimension of a hull, after tolerance-based degeneracy detection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HullDimension {
    /// No input points.
    #[default]
    Empty,
    /// All points coincide within tolerance.
    Point,
    /// All points are collinear within tolerance.
    Segment,
    /// All points are coplanar within tolerance.
    Polygon,
    /// A closed 3D polytope.
    Polytope,
}

/// A canonical 3D convex hull.
///
/// - `Polytope`: `vertices` are sorted lexicographically and `triangles`
///   index them with outward (counter-clockwise seen from outside) winding.
///   Each triangle starts at its smallest index and the list is sorted.
///   Coplanar hull faces are triangulated, not merged.
/// - `Polygon`: `vertices` are the boundary in cyclic order, starting at the
///   lexicographically smallest vertex in the dominant projection plane;
///   `triangles` is empty.
/// - `Segment`/`Point`/`Empty`: the two endpoints sorted lexicographically, the
///   single point, or nothing; `triangles` is empty.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hull3 {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    pub dimension: HullDimension,
}

impl Hull3 {
    /// Support map over the hull vertices for GJK/EPA.
    ///
    /// # Panics
    /// Panics when the hull is empty.
    #[must_use]
    pub fn support_map(&self) -> ConvexHull3<'_> {
        ConvexHull3::new(&self.vertices)
    }
}

#[derive(Clone, Copy, Debug)]
struct Face {
    /// Point indices, counter-clockwise seen from outside.
    vertices: [u32; 3],
    /// `neighbors[i]` shares edge `vertices[i] -> vertices[i + 1]`.
    neighbors: [u32; 3],
    normal: Vec3,
    offset: f64,
    /// Head of the conflict list (linked through `QuickHullWork::next`).
    head: u32,
    /// Farthest conflict point and its distance.
    far: u32,
    far_distance: f64,
    epoch: u32,
    alive: bool,
}

impl Face {
    fn new(points: &[Vec3], vertices: [u32; 3]) -> Self {
        let [a, b, c] = vertices.map(|index| points[index as usize]);
        let normal = cross(sub(b, a), sub(c, a));
        let len = length(normal);
        let normal = if len > 0.0 {
            normal.map(|component| component / len)
        } else {
            [0.0; 3]
        };
        Self {
            vertices,
            neighbors: [NONE; 3],
            normal,
            offset: dot(normal, a),
            head: NONE,
            far: NONE,
            far_distance: 0.0,
            epoch: 0,
            alive: true,
        }
    }

    fn distance(&self, point: Vec3) -> f64 {
        dot(self.normal, point) - self.offset
    }
}

/// Reusable scratch storage for [`quickhull2`] and [`quickhull3`].
///
/// Reusing one work value (and one output) makes repeated construction of
/// similarly sized hulls allocation-free.
#[derive(Clone, Debug, Default)]
pub struct QuickHullWork {
    faces: Vec<Face>,
    next: Vec<u32>,
    stack: Vec<(u32, u8, u8)>,
    visible: Vec<u32>,
    horizon: Vec<[u32; 4]>,
    indices: Vec<u32>,
    tasks: Vec<Task2>,
    remap: Vec<u32>,
    planes: Vec<[u32; 3]>,
    corners: Vec<Vec3>,
    distance_tests: u64,
}

impl QuickHullWork {
    /// Point/feature distance evaluations performed by the last construction.
    #[must_use]
    pub fn distance_tests(&self) -> u64 {
        self.distance_tests
    }
}

#[derive(Clone, Copy, Debug)]
enum Task2 {
    Hull {
        a: u32,
        b: u32,
        start: u32,
        end: u32,
    },
    Emit(u32),
}

fn lex3(left: Vec3, right: Vec3) -> Ordering {
    left[0]
        .total_cmp(&right[0])
        .then(left[1].total_cmp(&right[1]))
        .then(left[2].total_cmp(&right[2]))
}

fn lex2(left: Vec2, right: Vec2) -> Ordering {
    left[0]
        .total_cmp(&right[0])
        .then(left[1].total_cmp(&right[1]))
}

fn tolerance<const N: usize>(points: impl Iterator<Item = [f64; N]>) -> f64 {
    let mut max = [0.0f64; N];
    for point in points {
        for axis in 0..N {
            max[axis] = max[axis].max(point[axis].abs());
        }
    }
    3.0 * f64::EPSILON * max.iter().sum::<f64>()
}

/// `(b - a) x (p - a)`: positive when `p` is left of `a -> b`.
fn orient2(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// 2D QuickHull over `work.indices`, emitting boundary vertices
/// counter-clockwise from the lexicographically smallest one.
fn hull2_core(
    work: &mut QuickHullWork,
    position: impl Fn(u32) -> Vec2,
    tie: impl Fn(u32, u32) -> Ordering,
    eps: f64,
    mut emit: impl FnMut(u32),
) {
    let order = |left: u32, right: u32| {
        lex2(position(left), position(right)).then_with(|| tie(left, right))
    };
    let indices = &mut work.indices;
    let Some(&first) = indices.first() else {
        return;
    };
    let (mut a, mut b) = (first, first);
    for &index in indices.iter() {
        if order(index, a) == Ordering::Less {
            a = index;
        }
        if order(index, b) == Ordering::Greater {
            b = index;
        }
    }
    let (pa, pb) = (position(a), position(b));
    let span = (pb[0] - pa[0]).hypot(pb[1] - pa[1]);
    if span <= eps {
        emit(a);
        return;
    }
    // Lower chain: right of a -> b. Upper chain: right of b -> a.
    let threshold = eps * span;
    // Three-way partition in one pass: [lower | upper | discarded].
    let (mut lower, mut upper, mut end) = (0, 0, indices.len());
    while upper < end {
        let side = orient2(pa, pb, position(indices[upper]));
        if side < -threshold {
            indices.swap(lower, upper);
            lower += 1;
            upper += 1;
        } else if side > threshold {
            upper += 1;
        } else {
            end -= 1;
            indices.swap(upper, end);
        }
    }
    work.distance_tests += indices.len() as u64;
    let tasks = &mut work.tasks;
    tasks.clear();
    tasks.push(Task2::Hull {
        a: b,
        b: a,
        start: lower as u32,
        end: upper as u32,
    });
    tasks.push(Task2::Emit(b));
    tasks.push(Task2::Hull {
        a,
        b,
        start: 0,
        end: lower as u32,
    });
    tasks.push(Task2::Emit(a));
    while let Some(task) = tasks.pop() {
        let (a, b, start, end) = match task {
            Task2::Emit(index) => {
                emit(index);
                continue;
            }
            Task2::Hull { a, b, start, end } => (a, b, start as usize, end as usize),
        };
        if start == end {
            continue;
        }
        let (pa, pb) = (position(a), position(b));
        let mut far = indices[start];
        let mut far_value = orient2(pa, pb, position(far));
        for &index in &indices[start + 1..end] {
            let value = orient2(pa, pb, position(index));
            if value < far_value || (value == far_value && order(index, far) == Ordering::Less) {
                far = index;
                far_value = value;
            }
        }
        let pf = position(far);
        let left_threshold = eps * (pf[0] - pa[0]).hypot(pf[1] - pa[1]);
        let right_threshold = eps * (pb[0] - pf[0]).hypot(pb[1] - pf[1]);
        let mut left = start;
        for slot in start..end {
            if orient2(pa, pf, position(indices[slot])) < -left_threshold {
                indices.swap(left, slot);
                left += 1;
            }
        }
        let mut right = left;
        for slot in left..end {
            if orient2(pf, pb, position(indices[slot])) < -right_threshold {
                indices.swap(right, slot);
                right += 1;
            }
        }
        work.distance_tests += ((end - start) * 3 - (left - start)) as u64;
        tasks.push(Task2::Hull {
            a: far,
            b,
            start: left as u32,
            end: right as u32,
        });
        tasks.push(Task2::Emit(far));
        tasks.push(Task2::Hull {
            a,
            b: far,
            start: start as u32,
            end: left as u32,
        });
    }
}

/// Builds the canonical 2D convex hull of `points` into `out`.
///
/// `out` receives the hull vertices counter-clockwise, starting at the
/// lexicographically smallest vertex. Duplicate and collinear points are
/// dropped; a collinear input yields its two endpoints and a coincident input
/// a single point.
///
/// # Panics
/// Panics when a point is not finite.
pub fn quickhull2(points: &[Vec2], work: &mut QuickHullWork, out: &mut Vec<Vec2>) {
    out.clear();
    work.distance_tests = 0;
    assert!(
        points
            .iter()
            .all(|point| point.iter().all(|c| c.is_finite())),
        "hull points must be finite"
    );
    assert!(points.len() < NONE as usize, "too many hull points");
    let eps = tolerance(points.iter().copied());
    work.indices.clear();
    work.indices.extend(0..points.len() as u32);
    hull2_core(
        work,
        |index| points[index as usize],
        |_, _| Ordering::Equal,
        eps,
        |index| out.push(points[index as usize]),
    );
}

/// Builds the canonical 3D convex hull of `points` into `out`.
///
/// See [`Hull3`] for the output contract.
///
/// # Panics
/// Panics when a point is not finite.
pub fn quickhull3(points: &[Vec3], work: &mut QuickHullWork, out: &mut Hull3) {
    out.vertices.clear();
    out.triangles.clear();
    out.dimension = HullDimension::Empty;
    work.distance_tests = 0;
    assert!(
        points.iter().copied().all(is_finite),
        "hull points must be finite"
    );
    assert!(points.len() < NONE as usize, "too many hull points");
    if points.is_empty() {
        return;
    }
    let point = |index: u32| points[index as usize];
    let eps = tolerance(points.iter().copied());

    // Extremes along each axis, ties broken lexicographically.
    let mut low = [0u32; 3];
    let mut high = [0u32; 3];
    for index in 1..points.len() as u32 {
        let p = point(index);
        for axis in 0..3 {
            let lo = point(low[axis]);
            if p[axis].total_cmp(&lo[axis]).then(lex3(p, lo)) == Ordering::Less {
                low[axis] = index;
            }
            let hi = point(high[axis]);
            if p[axis].total_cmp(&hi[axis]).then(lex3(p, hi)) == Ordering::Greater {
                high[axis] = index;
            }
        }
    }
    let axis = (0..3)
        .max_by(|&l, &r| {
            let spread = |axis: usize| point(high[axis])[axis] - point(low[axis])[axis];
            spread(l).total_cmp(&spread(r)).then(r.cmp(&l))
        })
        .unwrap_or(0);
    let (v0, v1) = (low[axis], high[axis]);
    if point(v1)[axis] - point(v0)[axis] <= eps {
        out.dimension = HullDimension::Point;
        out.vertices
            .push(argmax(points, |_| 0.0, |l, r| lex3(r, l)));
        return;
    }

    let direction = sub(point(v1), point(v0));
    let direction_len = length(direction);
    let line_distance = |p: Vec3| length(cross(sub(p, point(v0)), direction)) / direction_len;
    let v2 = argmax_index(points, line_distance);
    work.distance_tests += points.len() as u64;
    if line_distance(point(v2)) <= eps {
        out.dimension = HullDimension::Segment;
        let start = argmax(points, |p| -dot(p, direction), lex3);
        let end = argmax(points, |p| dot(p, direction), lex3);
        let (start, end) = if lex3(start, end) == Ordering::Greater {
            (end, start)
        } else {
            (start, end)
        };
        out.vertices.extend([start, end]);
        return;
    }

    let base = Face::new(points, [v0, v1, v2]);
    let v3 = argmax_index(points, |p| base.distance(p).abs());
    work.distance_tests += points.len() as u64;
    if base.distance(point(v3)).abs() <= eps {
        out.dimension = HullDimension::Polygon;
        polygon(points, base.normal, eps, work, &mut out.vertices);
        return;
    }

    out.dimension = HullDimension::Polytope;
    build_polytope(points, [v0, v1, v2, v3], eps, work);
    if collect_corners(points, eps, work) {
        // An early eye point ended up inside a facet or on an edge of the
        // final hull. Rebuilding from the true corners alone cannot add such
        // a point again, so the second pass emits only extreme vertices.
        let mut corners = std::mem::take(&mut work.corners);
        let tests = work.distance_tests;
        quickhull3(&corners, work, out);
        work.distance_tests += tests;
        corners.clear();
        work.corners = corners;
        return;
    }
    emit_polytope(points, work, out);
}

/// Classifies every hull vertex by the distinct facet planes around it and
/// stores the corners (three or more planes) in `work.corners`. Returns
/// whether any hull vertex is not a corner.
fn collect_corners(points: &[Vec3], eps: f64, work: &mut QuickHullWork) -> bool {
    work.planes.clear();
    work.planes.resize(points.len(), [NONE; 3]);
    let faces = &work.faces;
    let coplanar = |plane: u32, face: &Face| {
        let plane = &faces[plane as usize];
        dot(plane.normal, face.normal) > 0.0
            && face
                .vertices
                .iter()
                .all(|&v| plane.distance(points[v as usize]).abs() <= eps)
    };
    for (id, face) in faces.iter().enumerate().filter(|(_, face)| face.alive) {
        for vertex in face.vertices {
            let slots = &mut work.planes[vertex as usize];
            if slots[2] != NONE {
                continue;
            }
            if slots[0] == NONE {
                slots[0] = id as u32;
            } else if coplanar(slots[0], face) {
            } else if slots[1] == NONE {
                slots[1] = id as u32;
            } else if !coplanar(slots[1], face) {
                slots[2] = id as u32;
            }
        }
    }
    work.corners.clear();
    let mut flat = false;
    for (index, slots) in work.planes.iter().enumerate() {
        if slots[2] != NONE {
            work.corners.push(points[index]);
        } else if slots[0] != NONE {
            flat = true;
        }
    }
    flat
}

/// Index of the point maximising `key`, ties broken toward the
/// lexicographically smallest point.
fn argmax_index(points: &[Vec3], key: impl Fn(Vec3) -> f64) -> u32 {
    let mut best = 0u32;
    let mut best_key = key(points[0]);
    for (index, &p) in points.iter().enumerate().skip(1) {
        let value = key(p);
        if value > best_key
            || (value == best_key && lex3(p, points[best as usize]) == Ordering::Less)
        {
            best = index as u32;
            best_key = value;
        }
    }
    best
}

fn argmax(
    points: &[Vec3],
    key: impl Fn(Vec3) -> f64,
    tie: impl Fn(Vec3, Vec3) -> Ordering,
) -> Vec3 {
    let mut best = points[0];
    let mut best_key = key(best);
    for &p in &points[1..] {
        let value = key(p);
        if value > best_key || (value == best_key && tie(p, best) == Ordering::Greater) {
            best = p;
            best_key = value;
        }
    }
    best
}

fn polygon(points: &[Vec3], normal: Vec3, eps: f64, work: &mut QuickHullWork, out: &mut Vec<Vec3>) {
    let drop = (0..3)
        .max_by(|&l, &r| normal[l].abs().total_cmp(&normal[r].abs()).then(r.cmp(&l)))
        .unwrap_or(2);
    let (u, v) = ((drop + 1) % 3, (drop + 2) % 3);
    work.indices.clear();
    work.indices.extend(0..points.len() as u32);
    hull2_core(
        work,
        |index| {
            let p = points[index as usize];
            [p[u], p[v]]
        },
        |l, r| lex3(points[l as usize], points[r as usize]),
        eps,
        |index| out.push(points[index as usize]),
    );
}

impl QuickHullWork {
    fn push_conflict(&mut self, face: u32, index: u32, distance: f64, points: &[Vec3]) {
        let f = &mut self.faces[face as usize];
        self.next[index as usize] = f.head;
        f.head = index;
        if f.far == NONE
            || distance > f.far_distance
            || (distance == f.far_distance
                && lex3(points[index as usize], points[f.far as usize]) == Ordering::Less)
        {
            f.far = index;
            f.far_distance = distance;
        }
    }

    /// Assigns `index` to the first face in `faces` it lies outside of.
    fn assign(&mut self, faces: std::ops::Range<u32>, index: u32, eps: f64, points: &[Vec3]) {
        let p = points[index as usize];
        for face in faces {
            self.distance_tests += 1;
            let distance = self.faces[face as usize].distance(p);
            if distance > eps {
                self.push_conflict(face, index, distance, points);
                return;
            }
        }
    }
}

fn build_polytope(points: &[Vec3], simplex: [u32; 4], eps: f64, work: &mut QuickHullWork) {
    work.faces.clear();
    work.next.clear();
    work.next.resize(points.len(), NONE);
    let [a, b, c, d] = simplex;
    for (mut tri, opposite) in [
        ([a, b, c], d),
        ([a, b, d], c),
        ([a, c, d], b),
        ([b, c, d], a),
    ] {
        if Face::new(points, tri).distance(points[opposite as usize]) > 0.0 {
            tri.swap(1, 2);
        }
        work.faces.push(Face::new(points, tri));
    }
    for face in 0..4 {
        for edge in 0..3 {
            let v = work.faces[face].vertices;
            let (from, to) = (v[edge], v[(edge + 1) % 3]);
            let neighbor = (0..4)
                .find(|&other| {
                    let w = work.faces[other].vertices;
                    (0..3).any(|e| w[e] == to && w[(e + 1) % 3] == from)
                })
                .expect("tetrahedron faces share every edge");
            work.faces[face].neighbors[edge] = neighbor as u32;
        }
    }
    for index in 0..points.len() as u32 {
        if !simplex.contains(&index) {
            work.assign(0..4, index, eps, points);
        }
    }

    let mut epoch = 0u32;
    let mut cursor = 0usize;
    while cursor < work.faces.len() {
        let face = &work.faces[cursor];
        if !face.alive || face.head == NONE {
            cursor += 1;
            continue;
        }
        let eye = face.far;
        let eye_point = points[eye as usize];
        epoch += 1;
        collect_horizon(work, cursor as u32, eye_point, eps, epoch);

        let first = work.faces.len() as u32;
        let count = work.horizon.len() as u32;
        for k in 0..count {
            let [from, to, outside, edge] = work.horizon[k as usize];
            let id = first + k;
            let mut face = Face::new(points, [from, to, eye]);
            face.neighbors = [
                outside,
                first + (k + 1) % count,
                first + (k + count - 1) % count,
            ];
            work.faces[outside as usize].neighbors[edge as usize] = id;
            work.faces.push(face);
        }
        for slot in 0..work.visible.len() {
            let visible = work.visible[slot] as usize;
            work.faces[visible].alive = false;
            let mut index = work.faces[visible].head;
            while index != NONE {
                let next = work.next[index as usize];
                if index != eye {
                    work.assign(first..first + count, index, eps, points);
                }
                index = next;
            }
        }
    }
}

/// Collects the faces visible from `eye` (flood fill from `start`) and the
/// ordered horizon loop `[from, to, outside_face, edge_in_outside_face]`.
fn collect_horizon(work: &mut QuickHullWork, start: u32, eye: Vec3, eps: f64, epoch: u32) {
    work.visible.clear();
    work.horizon.clear();
    work.stack.clear();
    work.faces[start as usize].epoch = epoch;
    work.visible.push(start);
    work.stack.push((start, 0, 0));
    while let Some(top) = work.stack.last_mut() {
        let (face, first_edge, done) = *top;
        if done == 3 {
            work.stack.pop();
            continue;
        }
        top.2 += 1;
        let edge = ((first_edge + done) % 3) as usize;
        let current = work.faces[face as usize];
        let neighbor = current.neighbors[edge];
        let other = &work.faces[neighbor as usize];
        if other.epoch == epoch {
            continue;
        }
        let shared = other
            .neighbors
            .iter()
            .position(|&n| n == face)
            .expect("face adjacency is symmetric") as u8;
        work.distance_tests += 1;
        if other.distance(eye) > eps {
            work.faces[neighbor as usize].epoch = epoch;
            work.visible.push(neighbor);
            work.stack.push((neighbor, (shared + 1) % 3, 0));
        } else {
            work.horizon.push([
                current.vertices[edge],
                current.vertices[(edge + 1) % 3],
                neighbor,
                u32::from(shared),
            ]);
        }
    }
}

fn emit_polytope(points: &[Vec3], work: &mut QuickHullWork, out: &mut Hull3) {
    work.indices.clear();
    work.remap.clear();
    work.remap.resize(points.len(), NONE);
    for face in work.faces.iter().filter(|face| face.alive) {
        for vertex in face.vertices {
            if work.remap[vertex as usize] == NONE {
                work.remap[vertex as usize] = 0;
                work.indices.push(vertex);
            }
        }
    }
    work.indices
        .sort_unstable_by(|&l, &r| lex3(points[l as usize], points[r as usize]));
    for (rank, &vertex) in work.indices.iter().enumerate() {
        work.remap[vertex as usize] = rank as u32;
        out.vertices.push(points[vertex as usize]);
    }
    for face in work.faces.iter().filter(|face| face.alive) {
        let t = face.vertices.map(|vertex| work.remap[vertex as usize]);
        let start = (0..3).min_by_key(|&i| t[i]).unwrap_or(0);
        out.triangles
            .push([t[start], t[(start + 1) % 3], t[(start + 2) % 3]]);
    }
    out.triangles.sort_unstable();
}
