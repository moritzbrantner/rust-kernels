//! Reusable f64 primitive contact geometry for fixed-orientation 3D shapes.
//!
//! This module owns shape math only. Simulation policy, contact persistence, response,
//! collision filtering, sleeping, and CCD admission remain consumer responsibilities.

mod closest;
mod frame;
mod pairs;
mod polytope;
mod query;
mod ray;
mod shape;
mod sweep;
mod volume;
mod wedge;

#[cfg(test)]
mod tests;

pub(crate) use closest::closest_point_triangle;
pub use closest::closest_segment_points;
use frame::{inverse_rotate, rotate_local, unit_or_zero};
pub use query::{
    PrimitiveContact3, PrimitiveWork3, bounds_extents, query, query_canonical, support_point,
};
pub use ray::{
    PrimitiveRayError3, PrimitiveRayFeature3, PrimitiveRayHit3, PrimitiveRayWork3, try_ray_cast,
};
pub use shape::{PrimitiveBody3, PrimitiveKind3, PrimitivePair3, PrimitiveShape3};
pub use sweep::{PrimitiveSweepError3, swept_time, try_swept_time};
pub use volume::{PrimitiveVolumeError3, PrimitiveVolumeProperties3, try_volume_properties};
use wedge::wedge_planes;
