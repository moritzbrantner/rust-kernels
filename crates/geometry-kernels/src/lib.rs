pub mod continuous;
pub mod epa;
pub mod epa3;
pub mod gjk;
pub mod gjk_distance;
pub mod gjk_trace;
pub mod halfspace;
pub mod heightfield;
pub mod hull;
pub mod math3;
pub mod planar;
pub mod polygon2;
pub mod primitive3;
pub mod primitives;
pub mod ray;
pub mod ray_triangle;
pub mod segment_hull;
pub mod support;
pub mod triangle_mesh;

pub use continuous::{
    SphereSweepError, SphereSweepHit, sphere_sphere_time_of_impact, try_sphere_sphere_fraction_f64,
};
pub use ray::{Ray3, RayIntervalHit, ray_aabb, ray_sphere};
pub use ray_triangle::{
    RayTriangleHit, SegmentTriangleHit, TriangleCulling, TriangleFacing, ray_triangle,
    segment_triangle,
};

include!("analytical.rs");
