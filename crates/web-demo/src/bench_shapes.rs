//! Batched collision and shape-query exports for the browser benchmark page.
//!
//! Pair queries read `count` packed records (f32, the kernels' input precision)
//! and write one result per pair; they return the number of hits so a caller
//! can sanity-check without reading the output. Hull and triangulation exports
//! reuse a [`ShapeScratch`] so steady-state timing excludes buffer growth.

use geometry_kernels::{
    Ray3, Sphere, TriangleCulling, aabb_aabb,
    hull::{Hull3, QuickHullWork, Vec2, quickhull2, quickhull3},
    math3::Vec3,
    obb3::{Obb3, obb3_sat},
    polygon2::{Polygon2Work, SimplicityCheck, triangulate_polygon2},
    primitives::sphere_aabb,
    ray_aabb, ray_triangle,
};
use spatial_kernels::Aabb;

use crate::bench::{input, output};

/// Reusable work buffers for hull and polygon exports.
#[derive(Default)]
pub struct ShapeScratch {
    hull_work: QuickHullWork,
    hull2: Vec<Vec2>,
    hull3: Hull3,
    polygon_work: Polygon2Work,
    triangles: Vec<[u32; 3]>,
}

/// Creates an owned [`ShapeScratch`] handle.
#[unsafe(no_mangle)]
pub extern "C" fn shape_scratch_new() -> *mut ShapeScratch {
    Box::into_raw(Box::default())
}

/// Releases a handle from [`shape_scratch_new`].
///
/// # Safety
/// `scratch` must come from `shape_scratch_new` and not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shape_scratch_free(scratch: *mut ShapeScratch) {
    if !scratch.is_null() {
        // SAFETY: guaranteed by the caller contract above.
        drop(unsafe { Box::from_raw(scratch) });
    }
}

fn ray(record: &[f32; 6]) -> Ray3 {
    Ray3::new(
        [record[0], record[1], record[2]],
        [record[3], record[4], record[5]],
    )
}

fn aabb(record: &[f32; 6]) -> Aabb {
    Aabb::new(
        [record[0], record[1], record[2]],
        [record[3], record[4], record[5]],
    )
}

fn obb(record: &[f32; 9]) -> Obb3 {
    Obb3::new(
        [record[0], record[1], record[2]],
        [record[3], record[4], record[5]],
        [record[6], record[7], record[8]],
    )
}

fn pairs<A, B>(left: &[A], right: &[B], out: &mut [u8], query: impl Fn(&A, &B) -> bool) -> u32 {
    let mut hits = 0;
    for ((left, right), flag) in left.iter().zip(right).zip(out.iter_mut()) {
        let hit = query(left, right);
        *flag = u8::from(hit);
        hits += u32::from(hit);
    }
    hits
}

macro_rules! pair_export {
    ($(#[$doc:meta])* $name:ident, $left:literal, $right:literal, |$a:ident, $b:ident| $query:expr) => {
        $(#[$doc])*
        ///
        /// # Safety
        #[doc = concat!(
            "`left` must hold `", stringify!($left), " * count` f32 values, `right` `",
            stringify!($right), " * count` f32 values and `out` `count` bytes."
        )]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            left: *const f32,
            right: *const f32,
            count: u32,
            out: *mut u8,
        ) -> u32 {
            let count = count as usize;
            // SAFETY: guaranteed by the caller contract above.
            let (left, right, out) = unsafe {
                (
                    input(left.cast::<[f32; $left]>(), count),
                    input(right.cast::<[f32; $right]>(), count),
                    output(out, count),
                )
            };
            pairs(left, right, out, |$a, $b| $query)
        }
    };
}

pair_export!(
    /// Sphere `(cx, cy, cz, r)` vs AABB `(min, max)` overlap flags.
    shape_sphere_aabb, 4, 6,
    |sphere, bounds| sphere_aabb(Sphere::new([sphere[0], sphere[1], sphere[2]], sphere[3]), aabb(bounds)).overlaps
);
pair_export!(
    /// AABB `(min, max)` vs AABB overlap flags (touching counts).
    shape_aabb_aabb, 6, 6,
    |left, right| aabb_aabb(aabb(left), aabb(right)).overlaps
);
pair_export!(
    /// OBB `(center, half_extents, rotation_xyz)` vs OBB SAT overlap flags.
    shape_obb_obb, 9, 9,
    |left, right| obb3_sat(obb(left), obb(right)).overlaps
);

/// Ray `(origin, direction)` vs AABB `(min, max)` hit flags within `max_distance`.
///
/// # Safety
/// `rays` and `boxes` must hold `6 * count` f32 values and `out` `count` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shape_ray_aabb(
    rays: *const f32,
    boxes: *const f32,
    count: u32,
    max_distance: f64,
    out: *mut u8,
) -> u32 {
    let count = count as usize;
    // SAFETY: guaranteed by the caller contract above.
    let (rays, boxes, out) = unsafe {
        (
            input(rays.cast::<[f32; 6]>(), count),
            input(boxes.cast::<[f32; 6]>(), count),
            output(out, count),
        )
    };
    pairs(rays, boxes, out, |record, bounds| {
        ray_aabb(ray(record), aabb(bounds)).is_some_and(|hit| hit.enter_distance <= max_distance)
    })
}

/// Ray `(origin, direction)` vs triangle `(a, b, c)` first-hit distances, both
/// faces, within `max_distance`; misses write `-1`.
///
/// # Safety
/// `rays` must hold `6 * count` f32 values, `triangles` `9 * count` f32 values
/// and `out` `count` f64 slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shape_ray_triangle(
    rays: *const f32,
    triangles: *const f32,
    count: u32,
    max_distance: f64,
    out: *mut f64,
) -> u32 {
    let count = count as usize;
    // SAFETY: guaranteed by the caller contract above.
    let (rays, triangles, out) = unsafe {
        (
            input(rays.cast::<[f32; 6]>(), count),
            input(triangles.cast::<[f32; 9]>(), count),
            output(out, count),
        )
    };
    let mut hits = 0;
    for ((record, triangle), distance) in rays.iter().zip(triangles).zip(out.iter_mut()) {
        let corners = [
            [triangle[0], triangle[1], triangle[2]],
            [triangle[3], triangle[4], triangle[5]],
            [triangle[6], triangle[7], triangle[8]],
        ];
        *distance = match ray_triangle(ray(record), corners, max_distance, TriangleCulling::None) {
            Some(hit) => {
                hits += 1;
                hit.distance
            }
            None => -1.0,
        };
    }
    hits
}

/// 2D convex hull of `count` `(x, y)` points; returns the hull vertex count.
///
/// # Safety
/// `scratch` must be a live handle and `points` must hold `2 * count` f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shape_quickhull2(
    scratch: *mut ShapeScratch,
    points: *const f64,
    count: u32,
) -> u32 {
    // SAFETY: guaranteed by the caller contract above.
    let (scratch, points) =
        unsafe { (&mut *scratch, input(points.cast::<Vec2>(), count as usize)) };
    quickhull2(points, &mut scratch.hull_work, &mut scratch.hull2);
    scratch.hull2.len() as u32
}

/// 3D convex hull of `count` `(x, y, z)` points; returns the hull vertex count.
/// [`shape_quickhull3_triangles`] reads the triangle count of the same hull.
///
/// # Safety
/// `scratch` must be a live handle and `points` must hold `3 * count` f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shape_quickhull3(
    scratch: *mut ShapeScratch,
    points: *const f64,
    count: u32,
) -> u32 {
    // SAFETY: guaranteed by the caller contract above.
    let (scratch, points) =
        unsafe { (&mut *scratch, input(points.cast::<Vec3>(), count as usize)) };
    quickhull3(points, &mut scratch.hull_work, &mut scratch.hull3);
    scratch.hull3.vertices.len() as u32
}

/// Triangle count of the last [`shape_quickhull3`] hull.
///
/// # Safety
/// `scratch` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shape_quickhull3_triangles(scratch: *const ShapeScratch) -> u32 {
    // SAFETY: guaranteed by the caller contract above.
    unsafe { (*scratch).hull3.triangles.len() as u32 }
}

/// Ear-clips a simple polygon of `count` `(x, y)` vertices without the O(n²)
/// simplicity check (the JavaScript competitor does not validate either).
/// Returns the triangle count, or `-1` when the kernel rejects the input.
///
/// # Safety
/// `scratch` must be a live handle and `points` must hold `2 * count` f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shape_triangulate_polygon2(
    scratch: *mut ShapeScratch,
    points: *const f64,
    count: u32,
) -> i32 {
    // SAFETY: guaranteed by the caller contract above.
    let (scratch, points) =
        unsafe { (&mut *scratch, input(points.cast::<Vec2>(), count as usize)) };
    match triangulate_polygon2(
        points,
        SimplicityCheck::AssumeSimple,
        &mut scratch.polygon_work,
        &mut scratch.triangles,
    ) {
        Ok(_) => scratch.triangles.len() as i32,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        shape_aabb_aabb, shape_obb_obb, shape_quickhull2, shape_quickhull3,
        shape_quickhull3_triangles, shape_ray_aabb, shape_ray_triangle, shape_scratch_free,
        shape_scratch_new, shape_sphere_aabb, shape_triangulate_polygon2,
    };

    #[test]
    fn pair_queries_report_hits_and_flags() {
        let rays = [
            -5.0, 0.0, 0.0, 1.0, 0.0, 0.0, // hits the unit box at distance 4
            -5.0, 3.0, 0.0, 1.0, 0.0, 0.0, // passes above it
        ];
        let boxes = [
            -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
        ];
        let mut flags = [9_u8; 2];
        // SAFETY: buffers hold two records each.
        unsafe {
            assert_eq!(
                shape_ray_aabb(rays.as_ptr(), boxes.as_ptr(), 2, 100.0, flags.as_mut_ptr()),
                1
            );
            assert_eq!(flags, [1, 0]);
            assert_eq!(
                shape_ray_aabb(rays.as_ptr(), boxes.as_ptr(), 2, 3.0, flags.as_mut_ptr()),
                0
            );
            assert_eq!(
                shape_aabb_aabb(boxes.as_ptr(), boxes.as_ptr(), 2, flags.as_mut_ptr()),
                2
            );
        }

        let spheres = [2.5_f32, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0];
        let obbs = [
            0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0,
        ];
        let far = [
            3.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.3,
            0.2, 0.1,
        ];
        // SAFETY: buffers hold two records each.
        unsafe {
            assert_eq!(
                shape_sphere_aabb(spheres.as_ptr(), boxes.as_ptr(), 2, flags.as_mut_ptr()),
                1
            );
            assert_eq!(flags, [0, 1]);
            assert_eq!(
                shape_obb_obb(obbs.as_ptr(), far.as_ptr(), 2, flags.as_mut_ptr()),
                1
            );
            assert_eq!(flags, [0, 1]);
        }
    }

    #[test]
    fn ray_triangle_writes_distances_and_misses() {
        let rays = [
            0.25_f32, 0.25, 5.0, 0.0, 0.0, -1.0, 3.0, 3.0, 5.0, 0.0, 0.0, -1.0,
        ];
        let triangle = [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let triangles = [triangle, triangle].concat();
        let mut distances = [0.0; 2];
        // SAFETY: buffers hold two records each.
        unsafe {
            let hits = shape_ray_triangle(
                rays.as_ptr(),
                triangles.as_ptr(),
                2,
                100.0,
                distances.as_mut_ptr(),
            );
            assert_eq!(hits, 1);
        }
        assert_eq!(distances, [5.0, -1.0]);
    }

    #[test]
    fn hull_and_polygon_exports_reuse_scratch() {
        let scratch = shape_scratch_new();
        let square = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 1.0, 1.0];
        let cube: Vec<f64> = (0..8)
            .flat_map(|corner| {
                [
                    f64::from(corner & 1),
                    f64::from((corner >> 1) & 1),
                    f64::from(corner >> 2),
                ]
            })
            .chain([0.5, 0.5, 0.5])
            .collect();
        let polygon = [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 2.0, 1.0, 0.0, 4.0];
        let bowtie = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        // SAFETY: the handle is live and buffers match the declared counts.
        unsafe {
            assert_eq!(shape_quickhull2(scratch, square.as_ptr(), 5), 4);
            assert_eq!(shape_quickhull3(scratch, cube.as_ptr(), 9), 8);
            assert_eq!(shape_quickhull3_triangles(scratch), 12);
            assert_eq!(shape_triangulate_polygon2(scratch, polygon.as_ptr(), 5), 3);
            assert_eq!(shape_triangulate_polygon2(scratch, bowtie.as_ptr(), 2), -1);
            shape_scratch_free(scratch);
            shape_scratch_free(std::ptr::null_mut());
        }
    }
}
