use crate::{
    gjk::{GjkResult, gjk_intersection},
    gjk_trace::{GjkTrace, gjk_intersection_trace},
    math3::{Vec3, is_finite},
    support::SupportMap3,
};

/// Adapts a support map that lies in the XY plane into a symmetric thin prism.
///
/// Generic GJK works with full 3D simplexes. A strictly coplanar convex set can
/// legitimately reach a triangle that contains the origin while still lacking a
/// fourth point off the plane. This adapter preserves the original XY support
/// choices while supplying deterministic ±Z witnesses when GJK asks for an
/// out-of-plane support point. Intersections of two equally extruded planar
/// shapes are therefore exactly the intersections of their XY projections.
#[derive(Clone, Copy, Debug)]
pub struct ExtrudedPlanarSupport<'a, S: ?Sized> {
    shape: &'a S,
    half_thickness: f64,
}

impl<'a, S: ?Sized> ExtrudedPlanarSupport<'a, S> {
    #[must_use]
    pub fn new(shape: &'a S, half_thickness: f64) -> Self {
        assert!(
            half_thickness.is_finite() && half_thickness > 0.0,
            "planar extrusion half-thickness must be positive and finite"
        );
        Self {
            shape,
            half_thickness,
        }
    }
}

impl<S> SupportMap3 for ExtrudedPlanarSupport<'_, S>
where
    S: SupportMap3 + ?Sized,
{
    fn support_point(&self, direction: Vec3) -> Vec3 {
        assert!(is_finite(direction), "support direction must be finite");
        let mut point = self.shape.support_point([direction[0], direction[1], 0.0]);
        point[2] = if direction[2] >= 0.0 {
            self.half_thickness
        } else {
            -self.half_thickness
        };
        point
    }
}

/// Run the authoritative 3D GJK implementation for two XY-planar support maps.
#[must_use]
pub fn gjk_intersection_planar_xy<L, R>(left: &L, right: &R) -> GjkResult
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    let left = ExtrudedPlanarSupport::new(left, 1.0);
    let right = ExtrudedPlanarSupport::new(right, 1.0);
    gjk_intersection(&left, &right)
}

/// Materialize deterministic GJK teaching/debug evidence for XY-planar shapes.
#[must_use]
pub fn gjk_intersection_trace_planar_xy<L, R>(left: &L, right: &R) -> GjkTrace
where
    L: SupportMap3 + ?Sized,
    R: SupportMap3 + ?Sized,
{
    let left = ExtrudedPlanarSupport::new(left, 1.0);
    let right = ExtrudedPlanarSupport::new(right, 1.0);
    gjk_intersection_trace(&left, &right)
}

#[cfg(test)]
mod tests {
    use super::{gjk_intersection_planar_xy, gjk_intersection_trace_planar_xy};
    use crate::{gjk::GjkStatus, support::ConvexHull3};

    #[test]
    fn overlapping_coplanar_hulls_are_intersecting() {
        let left_points = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let right_points = [
            [0.25, -0.75, 0.0],
            [2.0, -0.75, 0.0],
            [2.0, 0.75, 0.0],
            [0.25, 0.75, 0.0],
        ];
        let left = ConvexHull3::new(&left_points);
        let right = ConvexHull3::new(&right_points);

        let result = gjk_intersection_planar_xy(&left, &right);
        assert_eq!(result.status, GjkStatus::Intersecting);
        assert_eq!(result.intersection(), Some(true));
    }

    #[test]
    fn separated_coplanar_hulls_remain_separated() {
        let left_points = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let right_points = [
            [3.0, -0.75, 0.0],
            [4.0, -0.75, 0.0],
            [4.0, 0.75, 0.0],
            [3.0, 0.75, 0.0],
        ];
        let left = ConvexHull3::new(&left_points);
        let right = ConvexHull3::new(&right_points);

        assert_eq!(
            gjk_intersection_planar_xy(&left, &right).intersection(),
            Some(false)
        );
    }

    #[test]
    fn planar_trace_preserves_authoritative_terminal_result() {
        let left_points = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let right_points = [
            [0.0, -0.5, 0.0],
            [1.5, -0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let left = ConvexHull3::new(&left_points);
        let right = ConvexHull3::new(&right_points);

        let trace = gjk_intersection_trace_planar_xy(&left, &right);
        assert_eq!(trace.result.status, GjkStatus::Intersecting);
        assert_eq!(trace.steps.len(), trace.result.iterations);
        assert_eq!(
            trace.steps.last().and_then(|step| step.terminal_status),
            Some(GjkStatus::Intersecting)
        );
    }
}
