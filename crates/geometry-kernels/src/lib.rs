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
pub mod primitive3;
pub mod primitives;
pub mod ray;
pub mod ray_triangle;
pub mod support;

pub use continuous::{SphereSweepHit, sphere_sphere_time_of_impact};
pub use ray::{Ray3, RayIntervalHit, ray_aabb, ray_sphere};
pub use ray_triangle::{
    RayTriangleHit, SegmentTriangleHit, TriangleCulling, TriangleFacing, ray_triangle,
    segment_triangle,
};

include!("analytical.rs");
