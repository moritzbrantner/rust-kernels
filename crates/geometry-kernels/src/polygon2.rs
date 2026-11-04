//! Deterministic simple-polygon triangulation and convex decomposition in 2D.
//!
//! The input is one closed boundary given as a vertex slice; the closing edge
//! from the last vertex back to the first is implicit. Holes and multiple
//! boundaries are outside this contract.
//!
//! # Input contract
//!
//! Every operation first normalizes the boundary, in this order:
//!
//! 1. **Finite values.** A non-finite coordinate is
//!    [`Polygon2Error::NonFiniteVertex`]. Coordinates whose bounding-box span
//!    cannot be squared without overflow are
//!    [`Polygon2Error::UnrepresentableExtent`].
//! 2. **Repeated points.** Consecutive vertices within the scale-relative
//!    tolerance of each other (per axis), including a closing vertex that
//!    repeats the first one, collapse to the first vertex of the run. Repeated
//!    points that are *not* consecutive touch the boundary and are rejected
//!    as [`Polygon2Error::SelfIntersection`] by validation.
//! 3. **Collinear vertices.** A vertex whose distance from the line through
//!    its neighbours is within tolerance and that continues forward is a
//!    straight (180°) vertex. It is removed from the working boundary and is
//!    never referenced by an output triangle or piece. A vertex where the
//!    boundary turns back on itself (a zero-width spike) is
//!    [`Polygon2Error::SelfIntersection`].
//! 4. **Area.** Fewer than three remaining vertices is
//!    [`Polygon2Error::TooFewVertices`]; a remaining signed area within
//!    tolerance of zero is [`Polygon2Error::ZeroArea`].
//! 5. **Winding.** Both counter-clockwise and clockwise inputs are accepted.
//!    The input winding is reported in [`Polygon2Summary::winding`]; every
//!    output triangle and piece is counter-clockwise.
//! 6. **Simplicity.** With [`SimplicityCheck::Validate`] every pair of
//!    non-adjacent boundary edges is tested with closed-segment predicates; any
//!    crossing or touching (including a non-consecutive repeated vertex) is
//!    [`Polygon2Error::SelfIntersection`]. [`SimplicityCheck::AssumeSimple`]
//!    skips this O(n²) test as a caller precondition. A violated precondition
//!    never panics or loops: it yields either an error or triangles that do
//!    not tile the (ill-defined) interior.
//!
//! The tolerance is `4 ε (max|x| + max|y|)`, in coordinate units, matching the
//! scale-relative convention of [`crate::hull`].
//!
//! # Output contract
//!
//! Triangles and pieces store *input* vertex indices as `u32`. The working
//! boundary starts at its lexicographically smallest `(x, y)` vertex and runs
//! counter-clockwise, so for inputs without near-duplicate vertices the output
//! coordinates are identical under cyclic rotation and reversal of the input.
//! Equal inputs always produce bit-identical outputs.
//!
//! - [`triangulate_polygon2`] emits exactly `boundary_vertices - 2` triangles
//!   in ear-clipping order. Each triangle is `[previous, tip, next]` of its
//!   ear and has strictly positive area.
//! - [`convex_decompose_polygon2`] applies Hertel–Mehlhorn to that
//!   triangulation. Pieces are ordered by the first triangle they contain and
//!   each ring starts at the first vertex of that triangle. Every piece is
//!   convex; a piece may contain a straight vertex where a removed diagonal
//!   ended, but never a reflex one.
//!
//! # Complexity and allocation
//!
//! Normalization is O(n); validation is O(n²). Ear clipping keeps a doubly
//! linked boundary, incremental ear flags and a list of non-convex
//! candidates: each ear test is O(r) for `r` reflex vertices, giving O(n·r)
//! overall (O(n) for convex input, O(n²) worst case). A lap without an ear,
//! counted by [`Polygon2Work::stalls`] and never observed on valid test input,
//! triggers an O(n·r) refresh
//! of every flag before falling back to exact predicates. Hertel–Mehlhorn is O(n)
//! over the triangulation half-edges. Every buffer lives in a caller-owned
//! [`Polygon2Work`] and output value; reusing them makes repeated calls on
//! similarly sized polygons allocation-free.
//!
//! Hertel–Mehlhorn keeps only essential diagonals, so it returns at most
//! `2r + 1` pieces and at most four times the minimum number of convex
//! pieces. It is not an optimal decomposition.
//!
//! No circumcircle helper is provided: ear clipping and Hertel–Mehlhorn need
//! only orientation predicates.

use std::cmp::Ordering;

pub use crate::hull::Vec2;

const NONE: u32 = u32::MAX;

/// Orientation of the input boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Winding {
    CounterClockwise,
    Clockwise,
}

/// Whether the boundary simplicity test runs before triangulation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SimplicityCheck {
    /// Reject crossing or touching boundary edges (O(n²)).
    #[default]
    Validate,
    /// Caller guarantees a simple boundary; skips the O(n²) test.
    AssumeSimple,
}

/// Rejected polygon input. Vertex indices refer to the input slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Polygon2Error {
    /// More vertices than `u32` indices can address.
    TooManyVertices { len: usize },
    /// The vertex at `index` has a non-finite coordinate.
    NonFiniteVertex { index: usize },
    /// The coordinate span cannot be squared without overflow.
    UnrepresentableExtent,
    /// Fewer than three vertices remain after removing repeated and straight
    /// vertices.
    TooFewVertices { remaining: usize },
    /// The remaining signed area is zero within tolerance.
    ZeroArea,
    /// The boundary edges starting at input vertices `first` and `second`
    /// cross, touch or overlap. For a zero-width spike these are the two
    /// edges meeting at the spike tip.
    SelfIntersection { first: usize, second: usize },
    /// Ear clipping found no ear even with exact predicates. This only occurs
    /// for inputs that violate [`SimplicityCheck::AssumeSimple`] or sit at
    /// the limit of floating-point resolution.
    NoEar { remaining: usize },
}

impl std::fmt::Display for Polygon2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyVertices { len } => write!(f, "{len} vertices exceed u32 indexing"),
            Self::NonFiniteVertex { index } => write!(f, "vertex {index} is not finite"),
            Self::UnrepresentableExtent => write!(f, "coordinate span is not representable"),
            Self::TooFewVertices { remaining } => {
                write!(f, "only {remaining} distinct non-straight vertices remain")
            }
            Self::ZeroArea => write!(f, "polygon area is zero within tolerance"),
            Self::SelfIntersection { first, second } => {
                write!(
                    f,
                    "boundary edges at vertices {first} and {second} intersect"
                )
            }
            Self::NoEar { remaining } => {
                write!(f, "no ear found with {remaining} vertices remaining")
            }
        }
    }
}

impl std::error::Error for Polygon2Error {}

/// Facts about the normalized boundary of an accepted polygon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Polygon2Summary {
    /// Winding of the input boundary.
    pub winding: Winding,
    /// Vertices kept after removing repeated and straight vertices.
    pub boundary_vertices: usize,
    /// Unsigned area of the normalized boundary.
    pub area: f64,
}

/// Convex pieces in compressed-row form.
///
/// Piece `i` is `indices[offsets[i]..offsets[i + 1]]`: input vertex indices in
/// counter-clockwise order. `offsets` is empty when there are no pieces and
/// otherwise starts at zero.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConvexDecomposition2 {
    pub indices: Vec<u32>,
    pub offsets: Vec<u32>,
}

impl ConvexDecomposition2 {
    /// Number of pieces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Whether there are no pieces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Piece rings in output order.
    pub fn pieces(&self) -> impl Iterator<Item = &[u32]> + '_ {
        self.offsets
            .windows(2)
            .map(|range| &self.indices[range[0] as usize..range[1] as usize])
    }

    fn clear(&mut self) {
        self.indices.clear();
        self.offsets.clear();
    }
}

/// Reusable scratch storage for the polygon operations.
#[derive(Clone, Debug, Default)]
pub struct Polygon2Work {
    pos: Vec<Vec2>,
    orig: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
    order: Vec<u32>,
    convex: Vec<bool>,
    ear: Vec<bool>,
    removed: Vec<bool>,
    reflex: Vec<u32>,
    ring_twin: Vec<u32>,
    triangles: Vec<[u32; 3]>,
    twin: Vec<u32>,
    he_next: Vec<u32>,
    he_prev: Vec<u32>,
    he_alive: Vec<bool>,
    ear_tests: u64,
    stalls: u64,
}

impl Polygon2Work {
    /// Ear tests (including incremental re-tests) performed by the last
    /// triangulation; a deterministic work counter.
    #[must_use]
    pub fn ear_tests(&self) -> u64 {
        self.ear_tests
    }

    /// Laps without an ear (each followed by a full flag refresh) in the last
    /// triangulation. Zero for every valid input in the test suite.
    #[must_use]
    pub fn stalls(&self) -> u64 {
        self.stalls
    }
}

/// `(b - a) x (p - a)`: positive when `p` is left of `a -> b`.
fn orient2(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Euclidean distance; `normalize` rejects spans whose square overflows, so
/// the unscaled square root is exact enough and cheaper than `hypot`.
fn distance(a: Vec2, b: Vec2) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    (dx * dx + dy * dy).sqrt()
}

fn lex2(left: Vec2, right: Vec2) -> Ordering {
    left[0]
        .total_cmp(&right[0])
        .then(left[1].total_cmp(&right[1]))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Turn {
    Convex,
    Reflex,
    Straight,
    Spike,
}

fn turn(a: Vec2, v: Vec2, c: Vec2, tol: f64) -> Turn {
    let e1 = [v[0] - a[0], v[1] - a[1]];
    let e2 = [c[0] - v[0], c[1] - v[1]];
    let cross = e1[0] * e2[1] - e1[1] * e2[0];
    let margin = tol * (distance(a, v) + distance(v, c));
    if cross > margin {
        Turn::Convex
    } else if cross < -margin {
        Turn::Reflex
    } else if e1[0] * e2[0] + e1[1] * e2[1] > 0.0 {
        Turn::Straight
    } else {
        Turn::Spike
    }
}

fn near(a: Vec2, b: Vec2, tol: f64) -> bool {
    (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol
}

/// Normalizes the boundary into a counter-clockwise linked ring in `work`.
/// Returns the canonical start slot, the ring length and the summary.
fn normalize(
    points: &[Vec2],
    work: &mut Polygon2Work,
) -> Result<(u32, f64, Polygon2Summary), Polygon2Error> {
    if points.len() >= NONE as usize {
        return Err(Polygon2Error::TooManyVertices { len: points.len() });
    }
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    let mut magnitude = [0.0f64; 2];
    for (index, point) in points.iter().enumerate() {
        if !(point[0].is_finite() && point[1].is_finite()) {
            return Err(Polygon2Error::NonFiniteVertex { index });
        }
        for axis in 0..2 {
            lo[axis] = lo[axis].min(point[axis]);
            hi[axis] = hi[axis].max(point[axis]);
            magnitude[axis] = magnitude[axis].max(point[axis].abs());
        }
    }
    if !points.is_empty() {
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        if !(span * span * 8.0).is_finite() {
            return Err(Polygon2Error::UnrepresentableExtent);
        }
    }
    let tol = 4.0 * f64::EPSILON * (magnitude[0] + magnitude[1]);

    work.pos.clear();
    work.orig.clear();
    for (index, &point) in points.iter().enumerate() {
        if work.pos.last().is_some_and(|&last| near(last, point, tol)) {
            continue;
        }
        work.pos.push(point);
        work.orig.push(index as u32);
    }
    while work.pos.len() >= 2 && near(work.pos[work.pos.len() - 1], work.pos[0], tol) {
        work.pos.pop();
        work.orig.pop();
    }
    let m = work.pos.len();
    if m < 3 {
        return Err(Polygon2Error::TooFewVertices { remaining: m });
    }
    let m32 = m as u32;
    work.next.clear();
    work.next.extend((1..=m32).map(|i| i % m32));
    work.prev.clear();
    work.prev.extend((0..m32).map(|i| (i + m32 - 1) % m32));

    // Signed area first, so all-collinear input reports `ZeroArea` rather than
    // the spike it necessarily contains.
    let (area2, perimeter) = signed_area2(work, 0, m);
    if area2.abs() <= tol * perimeter {
        return Err(Polygon2Error::ZeroArea);
    }

    // Remove straight vertices; reject spikes.
    let mut count = m;
    let mut v = 0u32;
    let mut steady = 0;
    while steady < count && count >= 3 {
        let (a, c) = (work.prev[v as usize], work.next[v as usize]);
        let (pa, pv, pc) = (
            work.pos[a as usize],
            work.pos[v as usize],
            work.pos[c as usize],
        );
        match turn(pa, pv, pc, tol) {
            Turn::Straight => {
                work.next[a as usize] = c;
                work.prev[c as usize] = a;
                count -= 1;
                v = a;
                steady = 0;
            }
            Turn::Spike => {
                return Err(Polygon2Error::SelfIntersection {
                    first: work.orig[a as usize] as usize,
                    second: work.orig[v as usize] as usize,
                });
            }
            Turn::Convex | Turn::Reflex => {
                v = c;
                steady += 1;
            }
        }
    }
    if count < 3 {
        return Err(Polygon2Error::TooFewVertices { remaining: count });
    }

    let mut start = v;
    let mut cursor = work.next[v as usize];
    while cursor != v {
        if lex2(work.pos[cursor as usize], work.pos[start as usize]) == Ordering::Less {
            start = cursor;
        }
        cursor = work.next[cursor as usize];
    }
    let (area2, perimeter) = signed_area2(work, start, count);
    if area2.abs() <= tol * perimeter {
        return Err(Polygon2Error::ZeroArea);
    }
    let winding = if area2 > 0.0 {
        Winding::CounterClockwise
    } else {
        std::mem::swap(&mut work.next, &mut work.prev);
        Winding::Clockwise
    };
    work.order.clear();
    let mut cursor = start;
    for _ in 0..count {
        work.order.push(cursor);
        cursor = work.next[cursor as usize];
    }
    Ok((
        start,
        tol,
        Polygon2Summary {
            winding,
            boundary_vertices: count,
            area: area2.abs() * 0.5,
        },
    ))
}

/// Twice the signed area and the perimeter of the ring from `start`.
fn signed_area2(work: &Polygon2Work, start: u32, count: usize) -> (f64, f64) {
    let origin = work.pos[start as usize];
    let mut area2 = 0.0;
    let mut perimeter = 0.0;
    let mut cursor = start;
    for _ in 0..count {
        let next = work.next[cursor as usize];
        let (p, q) = (work.pos[cursor as usize], work.pos[next as usize]);
        area2 += (p[0] - origin[0]) * (q[1] - origin[1]) - (q[0] - origin[0]) * (p[1] - origin[1]);
        perimeter += distance(p, q);
        cursor = next;
    }
    (area2, perimeter)
}

fn on_segment(a: Vec2, b: Vec2, p: Vec2) -> bool {
    p[0] >= a[0].min(b[0])
        && p[0] <= a[0].max(b[0])
        && p[1] >= a[1].min(b[1])
        && p[1] <= a[1].max(b[1])
}

/// Closed-segment intersection with exact-sign floating-point predicates.
fn segments_meet(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool {
    if p1[0].max(p2[0]) < q1[0].min(q2[0])
        || q1[0].max(q2[0]) < p1[0].min(p2[0])
        || p1[1].max(p2[1]) < q1[1].min(q2[1])
        || q1[1].max(q2[1]) < p1[1].min(p2[1])
    {
        return false;
    }
    let d1 = orient2(q1, q2, p1);
    let d2 = orient2(q1, q2, p2);
    let d3 = orient2(p1, p2, q1);
    let d4 = orient2(p1, p2, q2);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    (d1 == 0.0 && on_segment(q1, q2, p1))
        || (d2 == 0.0 && on_segment(q1, q2, p2))
        || (d3 == 0.0 && on_segment(p1, p2, q1))
        || (d4 == 0.0 && on_segment(p1, p2, q2))
}

/// Reports the first crossing pair in boundary order from the canonical start.
fn check_simple(work: &Polygon2Work) -> Result<(), Polygon2Error> {
    let order = &work.order;
    let k = order.len();
    let point = |i: usize| work.pos[order[i % k] as usize];
    for i in 0..k {
        let (p1, p2) = (point(i), point(i + 1));
        let last = if i == 0 { k - 1 } else { k };
        for j in i + 2..last {
            if segments_meet(p1, p2, point(j), point(j + 1)) {
                return Err(Polygon2Error::SelfIntersection {
                    first: work.orig[order[i] as usize] as usize,
                    second: work.orig[order[j] as usize] as usize,
                });
            }
        }
    }
    Ok(())
}

fn validate(points: &[Vec2], work: &mut Polygon2Work) -> Result<Polygon2Summary, Polygon2Error> {
    let (_, _, summary) = normalize(points, work)?;
    check_simple(work)?;
    Ok(summary)
}

/// Normalizes and validates a polygon boundary without triangulating it.
///
/// Returns the same summary and errors as [`triangulate_polygon2`] with
/// [`SimplicityCheck::Validate`], except [`Polygon2Error::NoEar`].
pub fn validate_simple_polygon2(
    points: &[Vec2],
    work: &mut Polygon2Work,
) -> Result<Polygon2Summary, Polygon2Error> {
    validate(points, work)
}

fn classify(work: &mut Polygon2Work, v: u32, tol: f64) {
    let (a, c) = (work.prev[v as usize], work.next[v as usize]);
    let convex = turn(
        work.pos[a as usize],
        work.pos[v as usize],
        work.pos[c as usize],
        tol,
    ) == Turn::Convex;
    if work.convex[v as usize] && !convex {
        // Clipping only shrinks interior angles, so this is rare; keep the
        // candidate list complete regardless.
        work.reflex.push(v);
    }
    work.convex[v as usize] = convex;
}

fn is_ear(work: &mut Polygon2Work, v: u32, tol: f64) -> bool {
    work.ear_tests += 1;
    if !work.convex[v as usize] {
        return false;
    }
    let (a, c) = (work.prev[v as usize], work.next[v as usize]);
    let (pa, pv, pc) = (
        work.pos[a as usize],
        work.pos[v as usize],
        work.pos[c as usize],
    );
    let (ta, tv, tc) = (
        -tol * distance(pa, pv),
        -tol * distance(pv, pc),
        -tol * distance(pc, pa),
    );
    // Only non-convex vertices can lie in an ear of a simple polygon. Stale
    // entries (clipped or now convex) are dropped as they are met.
    let mut i = 0;
    while i < work.reflex.len() {
        let p = work.reflex[i];
        if work.removed[p as usize] || work.convex[p as usize] {
            work.reflex.swap_remove(i);
            continue;
        }
        i += 1;
        if p == a || p == c {
            continue;
        }
        let pp = work.pos[p as usize];
        if orient2(pa, pv, pp) >= ta && orient2(pv, pc, pp) >= tv && orient2(pc, pa, pp) >= tc {
            return false;
        }
    }
    true
}

/// Recomputes every convexity and ear flag; returns whether any ear exists.
fn refresh(work: &mut Polygon2Work, start: u32, remaining: usize, tol: f64) -> bool {
    let mut v = start;
    work.reflex.clear();
    for _ in 0..remaining {
        work.convex[v as usize] = true;
        classify(work, v, tol);
        v = work.next[v as usize];
    }
    let mut any = false;
    for _ in 0..remaining {
        let ear = is_ear(work, v, tol);
        work.ear[v as usize] = ear;
        any |= ear;
        v = work.next[v as usize];
    }
    any
}

fn link(twin: &mut [u32], edge: u32, other: u32) {
    if other != NONE {
        twin[edge as usize] = other;
        twin[other as usize] = edge;
    }
}

/// Ear clipping over the normalized ring; fills `work.triangles` (slots) and
/// `work.twin` (half-edge adjacency, `3 t + k` is edge `k -> k + 1`).
fn triangulate_core(
    points: &[Vec2],
    check: SimplicityCheck,
    work: &mut Polygon2Work,
) -> Result<Polygon2Summary, Polygon2Error> {
    work.triangles.clear();
    work.twin.clear();
    work.ear_tests = 0;
    work.stalls = 0;
    let (start, mut tol, summary) = normalize(points, work)?;
    if check == SimplicityCheck::Validate {
        check_simple(work)?;
    }
    let slots = work.pos.len();
    work.convex.clear();
    work.convex.resize(slots, false);
    work.ear.clear();
    work.ear.resize(slots, false);
    work.removed.clear();
    work.removed.resize(slots, false);
    work.ring_twin.clear();
    work.ring_twin.resize(slots, NONE);

    let mut remaining = summary.boundary_vertices;
    refresh(work, start, remaining, tol);
    let mut cursor = start;
    let mut scanned = 0;
    while remaining > 3 {
        if work.ear[cursor as usize] {
            let v = cursor;
            let (a, c) = (work.prev[v as usize], work.next[v as usize]);
            let t = work.triangles.len() as u32;
            work.triangles.push([a, v, c]);
            work.twin.extend([NONE; 3]);
            link(&mut work.twin, 3 * t, work.ring_twin[a as usize]);
            link(&mut work.twin, 3 * t + 1, work.ring_twin[v as usize]);
            work.ring_twin[a as usize] = 3 * t + 2;
            work.next[a as usize] = c;
            work.prev[c as usize] = a;
            work.removed[v as usize] = true;
            remaining -= 1;
            for w in [a, c] {
                classify(work, w, tol);
            }
            for w in [a, c] {
                let ear = is_ear(work, w, tol);
                work.ear[w as usize] = ear;
            }
            cursor = c;
            scanned = 0;
            continue;
        }
        cursor = work.next[cursor as usize];
        scanned += 1;
        if scanned >= remaining {
            work.stalls += 1;
            // A full lap without an ear: refresh stale flags, then fall back
            // to exact predicates before giving up.
            if !refresh(work, cursor, remaining, tol) {
                tol = 0.0;
                if !refresh(work, cursor, remaining, tol) {
                    work.triangles.clear();
                    work.twin.clear();
                    return Err(Polygon2Error::NoEar { remaining });
                }
            }
            scanned = 0;
        }
    }
    let (x, y) = (cursor, work.next[cursor as usize]);
    let z = work.next[y as usize];
    if orient2(
        work.pos[x as usize],
        work.pos[y as usize],
        work.pos[z as usize],
    ) > 0.0
    {
        let t = work.triangles.len() as u32;
        work.triangles.push([x, y, z]);
        work.twin.extend([NONE; 3]);
        link(&mut work.twin, 3 * t, work.ring_twin[x as usize]);
        link(&mut work.twin, 3 * t + 1, work.ring_twin[y as usize]);
        link(&mut work.twin, 3 * t + 2, work.ring_twin[z as usize]);
    }
    if work.triangles.len() + 2 != summary.boundary_vertices {
        // Only reachable when the precondition of `AssumeSimple` is violated.
        work.triangles.clear();
        work.twin.clear();
        return Err(Polygon2Error::NoEar { remaining: 3 });
    }
    Ok(summary)
}

/// Triangulates a simple polygon by ear clipping into `out`.
///
/// `out` is cleared and receives `boundary_vertices - 2` counter-clockwise
/// triangles of input vertex indices; see the [module contract](self). On
/// error `out` is left empty.
pub fn triangulate_polygon2(
    points: &[Vec2],
    check: SimplicityCheck,
    work: &mut Polygon2Work,
    out: &mut Vec<[u32; 3]>,
) -> Result<Polygon2Summary, Polygon2Error> {
    out.clear();
    let summary = triangulate_core(points, check, work)?;
    let orig = &work.orig;
    out.extend(
        work.triangles
            .iter()
            .map(|t| t.map(|slot| orig[slot as usize])),
    );
    Ok(summary)
}

/// Decomposes a simple polygon into convex pieces (Hertel–Mehlhorn).
///
/// Triangulates as [`triangulate_polygon2`], then removes each diagonal, in
/// triangle order, whose removal keeps both endpoints non-reflex. `out` is
/// cleared and receives counter-clockwise pieces; see the
/// [module contract](self). On error `out` is left empty.
pub fn convex_decompose_polygon2(
    points: &[Vec2],
    check: SimplicityCheck,
    work: &mut Polygon2Work,
    out: &mut ConvexDecomposition2,
) -> Result<Polygon2Summary, Polygon2Error> {
    out.clear();
    let summary = triangulate_core(points, check, work)?;
    let edges = work.triangles.len() * 3;
    work.he_next.clear();
    work.he_prev.clear();
    for t in 0..work.triangles.len() as u32 {
        work.he_next.extend([3 * t + 1, 3 * t + 2, 3 * t]);
        work.he_prev.extend([3 * t + 2, 3 * t, 3 * t + 1]);
    }
    work.he_alive.clear();
    work.he_alive.resize(edges, true);
    let triangles = &work.triangles;
    let origin = |edge: u32| triangles[edge as usize / 3][edge as usize % 3];
    for h in 0..edges as u32 {
        let t = work.twin[h as usize];
        if t == NONE || t < h {
            continue;
        }
        let (hp, hn) = (work.he_prev[h as usize], work.he_next[h as usize]);
        let (tp, tn) = (work.he_prev[t as usize], work.he_next[t as usize]);
        let (u, v) = (origin(h), origin(t));
        let pos = |slot: u32| work.pos[slot as usize];
        let at_u = orient2(
            pos(origin(hp)),
            pos(u),
            pos(origin(work.he_next[tn as usize])),
        );
        let at_v = orient2(
            pos(origin(tp)),
            pos(v),
            pos(origin(work.he_next[hn as usize])),
        );
        if at_u >= 0.0 && at_v >= 0.0 {
            work.he_next[hp as usize] = tn;
            work.he_prev[tn as usize] = hp;
            work.he_next[tp as usize] = hn;
            work.he_prev[hn as usize] = tp;
            work.he_alive[h as usize] = false;
            work.he_alive[t as usize] = false;
        }
    }
    out.offsets.push(0);
    for h in 0..edges {
        if !work.he_alive[h] {
            continue;
        }
        let mut edge = h as u32;
        loop {
            work.he_alive[edge as usize] = false;
            out.indices.push(work.orig[origin(edge) as usize]);
            edge = work.he_next[edge as usize];
            if edge == h as u32 {
                break;
            }
        }
        out.offsets.push(out.indices.len() as u32);
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_triangulates_and_merges_to_one_piece() {
        let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut work = Polygon2Work::default();
        let mut tris = Vec::new();
        let summary =
            triangulate_polygon2(&square, SimplicityCheck::Validate, &mut work, &mut tris).unwrap();
        assert_eq!(summary.winding, Winding::CounterClockwise);
        assert_eq!(summary.area, 1.0);
        assert_eq!(tris.len(), 2);
        let mut pieces = ConvexDecomposition2::default();
        convex_decompose_polygon2(&square, SimplicityCheck::Validate, &mut work, &mut pieces)
            .unwrap();
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces.pieces().next().unwrap().len(), 4);
    }

    #[test]
    fn spike_and_bowtie_are_rejected() {
        let mut work = Polygon2Work::default();
        let spike = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [2.0, 1.0],
            [0.0, 2.0],
        ];
        assert!(matches!(
            validate_simple_polygon2(&spike, &mut work),
            Err(Polygon2Error::SelfIntersection { .. })
        ));
        let bowtie = [[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 3.0]];
        assert!(matches!(
            validate_simple_polygon2(&bowtie, &mut work),
            Err(Polygon2Error::SelfIntersection { .. })
        ));
    }
}
