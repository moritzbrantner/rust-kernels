//! Fixed-orientation box/polytope SAT and clipped contact witnesses.
use super::PreparedSegmentHull;
use crate::math3::{Vec3, add, cross, dot, length_squared, neg, normalized, scale, sub};
use crate::primitive3::closest_segment_points;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxHullError {
    InvalidInput,
    NonFiniteComputation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoxHullPoint {
    pub point_box: Vec3,
    pub point_hull: Vec3,
    pub separation: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxHullContact {
    /// Unit normal from the hull toward the box.
    pub normal: Vec3,
    pub separation: f64,
    pub points: [BoxHullPoint; 4],
    pub len: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoxHullWork {
    pub axis_preparations: u64,
    pub axes_tested: u64,
    pub vertex_tests: u64,
    pub clipping_planes: u64,
}

#[derive(Clone, Copy, Debug)]
enum Feature {
    Hull(usize),
    Box(usize),
    Edge(usize, usize),
}

#[derive(Clone, Debug)]
struct PreparedAxis {
    normal: Vec3,
    origin: Vec3,
    offset: f64,
    feature: Feature,
}
#[derive(Clone, Debug)]
struct BoxFrame {
    half: Vec3,
    axes: [Vec3; 3],
    planes: Vec<PreparedAxis>,
    remaining: std::sync::OnceLock<Result<Vec<PreparedAxis>, BoxHullError>>,
}
impl BoxFrame {
    fn new(
        hull: &PreparedSegmentHull,
        half: Vec3,
        axes: [Vec3; 3],
        work: &mut BoxHullWork,
    ) -> Result<Self, BoxHullError> {
        if !half
            .into_iter()
            .chain(axes.into_iter().flatten())
            .all(f64::is_finite)
            || half.into_iter().any(|v| v <= 0.0)
            || axes.iter().any(|&a| (length_squared(a) - 1.0).abs() > 1e-8)
            || dot(axes[0], axes[1]).abs() > 1e-8
            || dot(axes[0], axes[2]).abs() > 1e-8
            || dot(axes[1], axes[2]).abs() > 1e-8
            || dot(cross(axes[0], axes[1]), axes[2]) < 1.0 - 1e-8
        {
            return Err(BoxHullError::InvalidInput);
        }
        let radius = |n| (0..3).map(|i| half[i] * dot(axes[i], n).abs()).sum::<f64>();
        // Hull faces can certify separation/contact before any vertex projections.
        let mut planes = Vec::with_capacity(hull.faces.len());
        let mut push = |normal: Vec3, origin: Vec3, offset: f64, feature: Feature| {
            if !normal
                .into_iter()
                .chain(origin)
                .chain([offset])
                .all(f64::is_finite)
            {
                return Err(BoxHullError::NonFiniteComputation);
            }
            planes.push(PreparedAxis {
                normal,
                origin,
                offset,
                feature,
            });
            Ok(())
        };
        for (face_id, face) in hull.faces.iter().enumerate() {
            push(
                face.unit_normal,
                [0.0; 3],
                face.unit_offset + radius(face.unit_normal),
                Feature::Hull(face_id),
            )?;
        }
        work.axis_preparations += planes.len() as u64;
        Ok(Self {
            half,
            axes,
            planes,
            remaining: std::sync::OnceLock::new(),
        })
    }

    fn remaining<'a>(
        &'a self,
        hull: &PreparedSegmentHull,
        work: &mut BoxHullWork,
    ) -> Result<&'a [PreparedAxis], BoxHullError> {
        self.remaining
            .get_or_init(|| self.prepare_remaining(hull, work))
            .as_ref()
            .map(Vec::as_slice)
            .map_err(|&error| error)
    }

    fn prepare_remaining(
        &self,
        hull: &PreparedSegmentHull,
        work: &mut BoxHullWork,
    ) -> Result<Vec<PreparedAxis>, BoxHullError> {
        let half = self.half;
        let axes = self.axes;
        let radius = |n| (0..3).map(|i| half[i] * dot(axes[i], n).abs()).sum::<f64>();
        let mut planes = Vec::with_capacity(6);
        let mut push = |normal: Vec3, origin: Vec3, offset: f64, feature: Feature| {
            if !normal
                .into_iter()
                .chain(origin)
                .chain([offset])
                .all(f64::is_finite)
            {
                return Err(BoxHullError::NonFiniteComputation);
            }
            planes.push(PreparedAxis {
                normal,
                origin,
                offset,
                feature,
            });
            Ok(())
        };
        for (k, n) in axes.into_iter().enumerate() {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &v in &hull.hull.vertices {
                work.vertex_tests += 1;
                let d = dot(v, n);
                lo = lo.min(d);
                hi = hi.max(d);
            }
            push(n, [0.0; 3], hi + half[k], Feature::Box(k))?;
            push(neg(n), [0.0; 3], half[k] - lo, Feature::Box(k))?;
        }
        // An edge cross axis matters only inside the hull edge's outward normal
        // cone. Its two adjacent triangles certify support locally, avoiding a
        // full hull-vertex projection for every edge/box-axis pair.
        for (e, edge) in hull.edges.iter().enumerate() {
            let [a, b] = edge.ends.map(|i| hull.hull.vertices[i as usize]);
            let [p, q] = edge
                .opposite
                .map(|i| sub(hull.hull.vertices[i as usize], a));
            for (k, axis) in axes.into_iter().enumerate() {
                let mut dp = dot(axis, edge.cone_normals[0]);
                let mut dq = dot(axis, edge.cone_normals[1]);
                let crossed =
                    if dp.abs() <= edge.cone_tolerance[0] || dq.abs() <= edge.cone_tolerance[1] {
                        let crossed = cross(sub(b, a), axis);
                        dp = dot(crossed, p);
                        dq = dot(crossed, q);
                        Some(crossed)
                    } else {
                        None
                    };
                if (dp < 0.0 && dq > 0.0) || (dp > 0.0 && dq < 0.0) {
                    continue;
                }
                let crossed = crossed.unwrap_or_else(|| cross(sub(b, a), axis));
                if length_squared(crossed) <= 1e-24 {
                    continue;
                }
                let n = if dp == 0.0 && dq == 0.0 {
                    if dot(crossed, sub(hull.centre, a)) <= 0.0 {
                        crossed
                    } else {
                        neg(crossed)
                    }
                } else if dp <= 0.0 && dq <= 0.0 {
                    crossed
                } else if dp >= 0.0 && dq >= 0.0 {
                    neg(crossed)
                } else {
                    continue;
                };
                let Some(n) = normalized(n) else {
                    continue;
                };
                push(n, a, radius(n), Feature::Edge(e, k))?;
            }
        }

        work.axis_preparations += planes.len() as u64;
        Ok(planes)
    }
}

/// A fixed box frame against owned immutable hull geometry. Translation and margin may
/// vary per query; changing orientation, extents or hull requires preparing another frame.
#[derive(Clone, Debug)]
pub struct PreparedBoxHull {
    hull: std::sync::Arc<PreparedSegmentHull>,
    frame: BoxFrame,
}
impl PreparedBoxHull {
    pub fn new(
        hull: std::sync::Arc<PreparedSegmentHull>,
        half: Vec3,
        axes: [Vec3; 3],
        work: &mut BoxHullWork,
    ) -> Result<Self, BoxHullError> {
        let frame = BoxFrame::new(&hull, half, axes, work)?;
        Ok(Self { hull, frame })
    }
    pub fn contact(
        &self,
        centre: Vec3,
        margin: f64,
        work: &mut BoxHullWork,
    ) -> Result<Option<BoxHullContact>, BoxHullError> {
        self.hull
            .box_contact_frame(&self.frame, centre, margin, work)
    }
    /// Owned plane payload only; the shared hull is accounted for by its owner.
    pub fn retained_bytes(&self) -> usize {
        let remaining = self
            .frame
            .remaining
            .get()
            .and_then(|result| result.as_ref().ok())
            .map_or(0, Vec::capacity);
        (self.frame.planes.capacity() + remaining) * size_of::<PreparedAxis>()
    }
}

impl PreparedSegmentHull {
    /// Contacts of a box and this closed hull, in the hull's local coordinates.
    /// `axes` must be an orthonormal right-handed box frame and half extents positive.
    /// Returns `Err` for invalid input, `None` beyond the nonnegative prediction margin.
    /// Face and edge-cross axes participate; empty corners of the hull bounds remain empty.
    /// This is a pose query, not a rotational or translation sweep.
    pub fn box_contact(
        &self,
        centre: Vec3,
        half: Vec3,
        axes: [Vec3; 3],
        margin: f64,
        work: &mut BoxHullWork,
    ) -> Result<Option<BoxHullContact>, BoxHullError> {
        let frame = BoxFrame::new(self, half, axes, work)?;
        self.box_contact_frame(&frame, centre, margin, work)
    }
    fn box_contact_frame(
        &self,
        frame: &BoxFrame,
        centre: Vec3,
        margin: f64,
        work: &mut BoxHullWork,
    ) -> Result<Option<BoxHullContact>, BoxHullError> {
        if !centre.into_iter().chain([margin]).all(f64::is_finite) || margin < 0.0 {
            return Err(BoxHullError::InvalidInput);
        }
        let half = frame.half;
        let axes = frame.axes;
        let mut choice = AxisChoice {
            gap: f64::NEG_INFINITY,
            normal: [1.0, 0.0, 0.0],
            feature: Feature::Hull(0),
        };
        for plane in &frame.planes {
            if choice.offer(
                work,
                margin,
                dot(sub(centre, plane.origin), plane.normal) - plane.offset,
                plane.normal,
                plane.feature,
            )? {
                return Ok(None);
            }
        }
        if choice.gap >= 0.0 {
            let Feature::Hull(face_id) = choice.feature else {
                unreachable!()
            };
            let points = self.face_points(centre, half, axes, face_id, margin, work);
            if points
                .iter()
                .any(|p| (p.separation - choice.gap).abs() <= 1e-10 * (1.0 + choice.gap.abs()))
            {
                return finish(choice.normal, choice.gap, points).map(Some);
            }
        }
        for plane in frame.remaining(self, work)? {
            if choice.offer(
                work,
                margin,
                dot(sub(centre, plane.origin), plane.normal) - plane.offset,
                plane.normal,
                plane.feature,
            )? {
                return Ok(None);
            }
        }
        let AxisChoice {
            gap: best,
            normal,
            feature,
        } = choice;
        let support_box = |direction: Vec3| {
            (0..3).fold(centre, |p, i| {
                add(
                    p,
                    scale(
                        axes[i],
                        if dot(axes[i], direction) >= 0.0 {
                            half[i]
                        } else {
                            -half[i]
                        },
                    ),
                )
            })
        };
        let mut points = Vec::new();
        match feature {
            Feature::Hull(face_id) => {
                points = self.face_points(centre, half, axes, face_id, margin, work);
            }
            Feature::Box(k) => {
                let offset = dot(normal, centre) - half[k];
                let high = self
                    .hull
                    .vertices
                    .iter()
                    .map(|&p| dot(normal, p))
                    .fold(f64::NEG_INFINITY, f64::max);
                let tolerance = 1e-9 * (1.0 + high.abs());
                let mut scratch = Vec::new();
                let mut polygon: Vec<_> = self
                    .hull
                    .vertices
                    .iter()
                    .copied()
                    .filter(|&p| high - dot(normal, p) <= tolerance)
                    .collect();
                // Canonical angular order around the supporting polygon; degenerate edge or
                // vertex supports are also valid polygons for clipping.
                let centroid = scale(
                    polygon.iter().copied().fold([0.0; 3], add),
                    1.0 / polygon.len() as f64,
                );
                let u = axes[(k + 1) % 3];
                let v = axes[(k + 2) % 3];
                polygon.sort_by(|a, b| {
                    let a = sub(*a, centroid);
                    let b = sub(*b, centroid);
                    dot(a, v)
                        .atan2(dot(a, u))
                        .total_cmp(&dot(b, v).atan2(dot(b, u)))
                });
                for i in (0..3).filter(|&i| i != k) {
                    for sign in [-1.0, 1.0] {
                        work.clipping_planes += 1;
                        clip(&mut polygon, &mut scratch, |p| {
                            sign * dot(sub(p, centre), axes[i]) - half[i]
                        });
                    }
                }
                for p in polygon {
                    let separation = offset - dot(normal, p);
                    if separation <= margin {
                        points.push(BoxHullPoint {
                            point_box: add(p, scale(normal, separation)),
                            point_hull: p,
                            separation,
                        });
                    }
                }
            }
            Feature::Edge(e, k) => {
                let [a, b] = self.edges[e].ends.map(|i| self.hull.vertices[i as usize]);
                let edge_centre = (0..3).filter(|&i| i != k).fold(centre, |p, i| {
                    add(
                        p,
                        scale(
                            axes[i],
                            if dot(axes[i], normal) > 0.0 {
                                -half[i]
                            } else {
                                half[i]
                            },
                        ),
                    )
                });
                let u = scale(axes[k], half[k]);
                if let Some((point_box, point_hull)) =
                    closest_segment_points(sub(edge_centre, u), add(edge_centre, u), a, b)
                {
                    // A triangulation edge must actually be supporting this normal.
                    let high = self
                        .hull
                        .vertices
                        .iter()
                        .map(|&p| dot(normal, p))
                        .fold(f64::NEG_INFINITY, f64::max);
                    if (dot(normal, a) - high).abs() < 1e-8 && (dot(normal, b) - high).abs() < 1e-8
                    {
                        points.push(BoxHullPoint {
                            point_box,
                            point_hull,
                            separation: best,
                        });
                    }
                }
            }
        }
        if points.is_empty() {
            let point_hull = *self
                .hull
                .vertices
                .iter()
                .max_by(|a, b| dot(**a, normal).total_cmp(&dot(**b, normal)))
                .unwrap_or(&self.hull.vertices[0]);
            points.push(BoxHullPoint {
                point_box: support_box(neg(normal)),
                point_hull,
                separation: best,
            });
        }
        finish(normal, best, points).map(Some)
    }

    fn face_points(
        &self,
        centre: Vec3,
        half: Vec3,
        axes: [Vec3; 3],
        face_id: usize,
        margin: f64,
        work: &mut BoxHullWork,
    ) -> Vec<BoxHullPoint> {
        let face = &self.faces[face_id];
        let normal = face.unit_normal;
        let mut points = Vec::new();
        let offset = face.unit_offset;
        let k = (0..3)
            .max_by(|&i, &j| {
                dot(normal, axes[i])
                    .abs()
                    .total_cmp(&dot(normal, axes[j]).abs())
            })
            .unwrap_or(0);
        let c = sub(
            centre,
            scale(axes[k], half[k] * dot(normal, axes[k]).signum()),
        );
        let u = scale(axes[(k + 1) % 3], half[(k + 1) % 3]);
        let v = scale(axes[(k + 2) % 3], half[(k + 2) % 3]);
        let mut polygon = vec![
            add(add(c, u), v),
            add(sub(c, u), v),
            sub(sub(c, u), v),
            add(sub(c, v), u),
        ];
        let mut scratch = Vec::new();
        for plane in &self.faces {
            work.clipping_planes += 1;
            let projection = dot(plane.unit_normal, normal);
            let clip_normal = sub(plane.unit_normal, scale(normal, projection));
            let clip_offset = plane.unit_offset - offset * projection;
            clip(&mut polygon, &mut scratch, |p| {
                dot(clip_normal, p) - clip_offset
            });
            if polygon.is_empty() {
                break;
            }
        }
        for p in polygon {
            let separation = dot(normal, p) - offset;
            if separation <= margin {
                points.push(BoxHullPoint {
                    point_box: p,
                    point_hull: sub(p, scale(normal, separation)),
                    separation,
                });
            }
        }
        points
    }
}

struct AxisChoice {
    gap: f64,
    normal: Vec3,
    feature: Feature,
}
impl AxisChoice {
    fn offer(
        &mut self,
        work: &mut BoxHullWork,
        margin: f64,
        gap: f64,
        normal: Vec3,
        feature: Feature,
    ) -> Result<bool, BoxHullError> {
        work.axes_tested += 1;
        // Prepared axes were validated when the frame was built. Only the
        // translation-dependent gap can introduce a new non-finite value.
        if !gap.is_finite() {
            return Err(BoxHullError::NonFiniteComputation);
        }
        if gap > self.gap {
            *self = Self {
                gap,
                normal,
                feature,
            };
        }
        Ok(gap > margin)
    }
}
fn finish(
    normal: Vec3,
    best: f64,
    points: Vec<BoxHullPoint>,
) -> Result<BoxHullContact, BoxHullError> {
    if points.iter().any(|p| {
        !p.point_box
            .into_iter()
            .chain(p.point_hull)
            .chain([p.separation])
            .all(f64::is_finite)
    }) {
        return Err(BoxHullError::NonFiniteComputation);
    }
    let len = points.len().min(4);
    let mut selected = [BoxHullPoint::default(); 4];
    for (i, p) in selected[..len].iter_mut().enumerate() {
        *p = points[i * points.len() / len];
    }
    Ok(BoxHullContact {
        normal,
        separation: best,
        points: selected,
        len,
    })
}

fn clip(polygon: &mut Vec<Vec3>, scratch: &mut Vec<Vec3>, distance: impl Fn(Vec3) -> f64) {
    if polygon.is_empty() {
        return;
    }
    if polygon.iter().all(|&p| distance(p) <= 1e-10) {
        return;
    }
    scratch.clear();
    scratch.reserve(polygon.len() + 1);
    let mut prev = polygon[polygon.len() - 1];
    let mut dp = distance(prev);
    for next in polygon.drain(..) {
        let dn = distance(next);
        // Small slack avoids losing a polygon to roundoff on its own face plane.
        if (dp <= 1e-10) != (dn <= 1e-10) {
            scratch.push(add(prev, scale(sub(next, prev), dp / (dp - dn))));
        }
        if dn <= 1e-10 {
            scratch.push(next);
        }
        prev = next;
        dp = dn;
    }
    std::mem::swap(polygon, scratch);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hull::{Hull3, QuickHullWork, quickhull3};
    fn cube() -> PreparedSegmentHull {
        let vertices: Vec<_> = [-1.0, 1.0]
            .into_iter()
            .flat_map(|x| {
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(move |y| [-1.0, 1.0].into_iter().map(move |z| [x, y, z]))
            })
            .collect();
        let mut hull = Hull3::default();
        quickhull3(&vertices, &mut QuickHullWork::default(), &mut hull);
        PreparedSegmentHull::new(hull).unwrap()
    }
    const AXES: [Vec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    #[test]
    fn face_contacts_have_four_exact_witnesses_and_signed_depth() {
        let hull = cube();
        for (y, gap) in [(1.45, -0.05), (1.55, 0.05)] {
            let c = hull
                .box_contact(
                    [0.0, y, 0.0],
                    [0.5; 3],
                    AXES,
                    0.1,
                    &mut BoxHullWork::default(),
                )
                .unwrap()
                .unwrap();
            assert_eq!(c.normal, [0.0, 1.0, 0.0]);
            assert!((c.separation - gap).abs() < 1e-12);
            assert_eq!(c.len, 4);
            for p in &c.points[..c.len] {
                assert!((p.point_box[1] - (y - 0.5)).abs() < 1e-12);
                assert!((p.point_hull[1] - 1.0).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn rotated_box_sat_matches_the_independent_primitive_box_query() {
        use crate::primitive3::{PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, query};
        let hull = cube();
        let fixed =
            PrimitiveBody3::axis_aligned(PrimitiveShape3::cuboid([1.0; 3]), [0.0; 3], [0.0; 3]);
        let mut seed = 0x36b4_129d_u64;
        let mut random = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as f64 / u32::MAX as f64
        };
        for _ in 0..512 {
            let centre = std::array::from_fn(|_| 4.0 * random() - 2.0);
            let half = std::array::from_fn(|_| 0.05 + random());
            let (s, c) = (random() * 6.0).sin_cos();
            let (t, d) = (random() * 6.0).sin_cos();
            let axes = [[c, 0.0, -s], [s * t, d, c * t], [s * d, -t, c * d]];
            let body = PrimitiveBody3::new(PrimitiveShape3::cuboid(half), centre, axes, [0.0; 3]);
            let expected = query(fixed, body, &mut PrimitiveWork3::default());
            let actual = hull
                .box_contact(centre, half, axes, 10.0, &mut BoxHullWork::default())
                .unwrap()
                .unwrap();
            assert!(
                (actual.separation - expected.separation).abs() < 1e-9,
                "{actual:?} vs {expected:?}"
            );
            for point in &actual.points[..actual.len] {
                assert!(
                    (dot(sub(point.point_box, point.point_hull), actual.normal) - point.separation)
                        .abs()
                        < 1e-8
                );
                assert!(point.point_hull.into_iter().all(|v| v.abs() <= 1.0 + 1e-8));
                let relative = sub(point.point_box, centre);
                assert!((0..3).all(|i| dot(relative, axes[i]).abs() <= half[i] + 1e-8));
            }
        }
    }
    #[test]
    fn prepared_frame_matches_fresh_queries_without_reprojecting_vertices() {
        let hull = std::sync::Arc::new(cube());
        let t = 0.7_f64;
        let (s, c) = t.sin_cos();
        let axes = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]];
        let half = [0.3, 0.5, 0.7];
        let mut preparation = BoxHullWork::default();
        let frame = PreparedBoxHull::new(hull.clone(), half, axes, &mut preparation).unwrap();
        assert!(preparation.axis_preparations > 0);
        assert_eq!(preparation.vertex_tests, 0);
        // A penetrating query prepares the deferred axes; subsequent queries reuse them.
        frame.contact([0.0; 3], 0.0, &mut preparation).unwrap();
        assert!(preparation.vertex_tests > 0);
        for centre in [
            [0.0, 1.51, 0.0],
            [1.2, 1.2, 0.0],
            [1.7, 1.7, 1.7],
            [0.0; 3],
            [-2.0, 0.0, 0.0],
        ] {
            for margin in [0.0, 0.02, 0.15] {
                let mut work = BoxHullWork::default();
                let cached = frame.contact(centre, margin, &mut work).unwrap();
                let fresh = hull
                    .box_contact(centre, half, axes, margin, &mut BoxHullWork::default())
                    .unwrap();
                assert_eq!(cached, fresh);
                assert_eq!(work.axis_preparations, 0);
                assert_eq!(work.vertex_tests, 0);
            }
        }
        assert!(frame.retained_bytes() > 0);
    }
    #[test]
    fn face_certificates_defer_full_sat_until_needed_and_retain_it() {
        let hull = std::sync::Arc::new(cube());
        let mut work = BoxHullWork::default();
        let frame = PreparedBoxHull::new(hull.clone(), [0.5; 3], AXES, &mut work).unwrap();
        assert_eq!(work.vertex_tests, 0);
        let before = frame.retained_bytes();
        for centre in [[3.0, 0.0, 0.0], [0.0, 1.51, 0.0]] {
            let mut query = BoxHullWork::default();
            frame.contact(centre, 0.02, &mut query).unwrap();
            assert_eq!(query.axis_preparations, 0);
            assert_eq!(query.vertex_tests, 0);
            assert_eq!(frame.retained_bytes(), before);
        }
        let mut first = BoxHullWork::default();
        let contact = frame.contact([0.0; 3], 0.0, &mut first).unwrap();
        assert!(first.axis_preparations > 0 && first.vertex_tests > 0);
        assert!(frame.retained_bytes() > before);
        let mut second = BoxHullWork::default();
        assert_eq!(frame.contact([0.0; 3], 0.0, &mut second).unwrap(), contact);
        assert_eq!(second.axis_preparations, 0);
        assert_eq!(second.vertex_tests, 0);
        let mut eager_work = BoxHullWork::default();
        let eager = PreparedBoxHull::new(hull.clone(), [0.5; 3], AXES, &mut eager_work).unwrap();
        eager.frame.remaining(&hull, &mut eager_work).unwrap();
        assert_eq!(
            eager.contact([0.0; 3], 0.0, &mut eager_work).unwrap(),
            contact
        );
    }
    #[test]
    fn separated_and_invalid_queries_do_not_make_contacts() {
        let hull = cube();
        assert!(
            hull.box_contact([2.0; 3], [0.5; 3], AXES, 0.01, &mut BoxHullWork::default())
                .unwrap()
                .is_none()
        );
        assert!(
            hull.box_contact(
                [0.0; 3],
                [0.5; 3],
                [[1.0; 3]; 3],
                0.01,
                &mut BoxHullWork::default()
            )
            .is_err()
        );
    }
}
