//! Translational first-contact sweeps: conservative advancement and swept SAT.

use super::polytope::{poly_axes, projection_interval};
use super::query::{PrimitiveContact3, PrimitiveWork3, query_canonical};
use super::shape::{PrimitiveBody3, PrimitivePair3};
use crate::math3::{add, dot, scale, sub};

/// Compatibility query that cannot distinguish a miss from a failed search.
/// Use [`try_swept_time`] when a collision decision requires explicit failure handling.
#[must_use]
pub fn swept_time(
    a: PrimitiveBody3,
    b: PrimitiveBody3,
    dt: f64,
    margin: f64,
    work: &mut PrimitiveWork3,
) -> Option<f64> {
    try_swept_time(a, b, dt, margin, 128, work).ok().flatten()
}

/// A failed search provides no conclusion about collision absence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimitiveSweepError3 {
    InvalidInput,
    NonFiniteComputation,
    IterationLimit,
}

impl std::fmt::Display for PrimitiveSweepError3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "primitive translation sweep: {self:?}")
    }
}

impl std::error::Error for PrimitiveSweepError3 {}

/// Sweeps fixed-orientation primitives over `dt`, returning normalized first-contact time.
///
/// `Ok(None)` is a separating-plane miss; numerical failure and an exhausted search are
/// errors. `max_iterations` bounds conservative advances or swept-SAT axes, including the
/// query that admits a hit. Stationary curved pairs need one contact query and no iterations;
/// polyhedral pairs always count the swept-SAT axes they inspect.
/// The work counters include unsuccessful search work. Shape constructors validate dimensions;
/// callers supply finite poses/velocities and the shape's applicable orthonormal axes.
/// This does not sweep angular motion or promise support for arbitrary finite magnitudes.
pub fn try_swept_time(
    a: PrimitiveBody3,
    b: PrimitiveBody3,
    dt: f64,
    margin: f64,
    max_iterations: u32,
    work: &mut PrimitiveWork3,
) -> Result<Option<f64>, PrimitiveSweepError3> {
    if !dt.is_finite()
        || dt < 0.0
        || !margin.is_finite()
        || margin < 0.0
        || [a, b].into_iter().any(|body| {
            !crate::math3::is_finite(body.position)
                || !crate::math3::is_finite(body.velocity)
                || !body.axes.into_iter().all(crate::math3::is_finite)
        })
    {
        return Err(PrimitiveSweepError3::InvalidInput);
    }
    let (pair, reversed) = PrimitivePair3::canonical(a.shape.kind, b.shape.kind);
    let (a, b) = if reversed { (b, a) } else { (a, b) };
    if matches!(
        pair,
        PrimitivePair3::BoxBox | PrimitivePair3::BoxWedge | PrimitivePair3::WedgeWedge
    ) {
        return poly_sweep_time(a, b, dt, margin, max_iterations, work);
    }

    let displacement = scale(sub(b.velocity, a.velocity), dt);
    if !crate::math3::is_finite(displacement) {
        return Err(PrimitiveSweepError3::NonFiniteComputation);
    }
    if displacement == [0.0; 3] {
        let contact = query_canonical(pair, a, b, work);
        validate_sweep_contact(contact)?;
        return Ok((contact.separation <= margin).then_some(0.0));
    }
    let target = margin;
    let tolerance = 1.0e-9 * (1.0 + a.shape.radius() + b.shape.radius());
    if !tolerance.is_finite() {
        return Err(PrimitiveSweepError3::NonFiniteComputation);
    }
    // Relative to the pair's coordinates and sizes, so tiny pairs keep their resolution.
    let magnitude = [
        a.position,
        b.position,
        add(a.position, scale(a.velocity, dt)),
        add(b.position, scale(b.velocity, dt)),
    ]
    .into_iter()
    .flatten()
    .map(f64::abs)
    .fold(a.shape.radius().max(b.shape.radius()), f64::max);
    let rounding = 8.0 * f64::EPSILON * magnitude;
    if !rounding.is_finite() {
        return Err(PrimitiveSweepError3::NonFiniteComputation);
    }
    let mut time = 0.0;
    let mut current_a = a;
    let mut current_b = b;

    for _ in 0..max_iterations {
        work.sweep_iterations += 1;
        current_a.position = add(a.position, scale(a.velocity, dt * time));
        current_b.position = add(b.position, scale(b.velocity, dt * time));
        if !crate::math3::is_finite(current_a.position)
            || !crate::math3::is_finite(current_b.position)
        {
            return Err(PrimitiveSweepError3::NonFiniteComputation);
        }
        let contact = query_canonical(pair, current_a, current_b, work);
        validate_sweep_contact(contact)?;
        let gap = contact.separation - target;
        if !gap.is_finite() {
            return Err(PrimitiveSweepError3::NonFiniteComputation);
        }
        // A gap within the separation's rounding is contact: far from the origin rounding can
        // keep it just above zero, and the advance would creep until the budget ran out (#175).
        if gap <= rounding {
            return Ok(Some(time));
        }
        let closing = -dot(displacement, contact.normal);
        if !closing.is_finite() {
            return Err(PrimitiveSweepError3::NonFiniteComputation);
        }
        if closing <= 0.0 {
            return Ok(None);
        }
        let remaining = 1.0 - time;
        let reachable = closing * remaining + tolerance;
        if !reachable.is_finite() {
            return Err(PrimitiveSweepError3::NonFiniteComputation);
        }
        if gap > reachable {
            return Ok(None);
        }
        let next = time + gap / closing;
        if !next.is_finite() {
            return Err(PrimitiveSweepError3::NonFiniteComputation);
        }
        if next > 1.0 {
            return Ok(None);
        }
        if next <= time {
            let bumped = f64::from_bits(time.to_bits() + 1);
            if bumped > 1.0 {
                return Ok(None);
            }
            time = bumped;
        } else {
            time = next;
        }
    }
    Err(PrimitiveSweepError3::IterationLimit)
}

fn validate_sweep_contact(contact: PrimitiveContact3) -> Result<(), PrimitiveSweepError3> {
    if !contact.separation.is_finite()
        || !crate::math3::is_finite(contact.normal)
        || !crate::math3::is_finite(contact.point_a)
        || !crate::math3::is_finite(contact.point_b)
    {
        return Err(PrimitiveSweepError3::NonFiniteComputation);
    }
    Ok(())
}

fn poly_sweep_time(
    a: PrimitiveBody3,
    b: PrimitiveBody3,
    dt: f64,
    margin: f64,
    max_iterations: u32,
    work: &mut PrimitiveWork3,
) -> Result<Option<f64>, PrimitiveSweepError3> {
    let (axes, len) = poly_axes(a, b);
    let displacement = scale(sub(b.velocity, a.velocity), dt);
    if !crate::math3::is_finite(displacement) {
        return Err(PrimitiveSweepError3::NonFiniteComputation);
    }
    let mut enter = 0.0_f64;
    let mut exit = 1.0_f64;

    for (iteration, &axis) in axes[..len].iter().enumerate() {
        if iteration as u64 >= u64::from(max_iterations) {
            return Err(PrimitiveSweepError3::IterationLimit);
        }
        work.axes_tested += 1;
        work.sweep_iterations += 1;
        let (a_min, a_max) = projection_interval(a, axis, work);
        let (b_min, b_max) = projection_interval(b, axis, work);
        let velocity = dot(displacement, axis);
        if ![a_min, a_max, b_min, b_max, velocity]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err(PrimitiveSweepError3::NonFiniteComputation);
        }
        if velocity == 0.0 {
            let a_limit = a_max + margin;
            let b_limit = b_max + margin;
            if !a_limit.is_finite() || !b_limit.is_finite() {
                return Err(PrimitiveSweepError3::NonFiniteComputation);
            }
            if b_min > a_limit || a_min > b_limit {
                return Ok(None);
            }
            continue;
        }
        let first = (a_min - margin - b_max) / velocity;
        let second = (a_max + margin - b_min) / velocity;
        if !first.is_finite() || !second.is_finite() {
            return Err(PrimitiveSweepError3::NonFiniteComputation);
        }
        enter = enter.max(first.min(second));
        exit = exit.min(first.max(second));
        if enter > exit {
            return Ok(None);
        }
    }

    Ok((exit >= 0.0 && enter <= 1.0).then_some(enter.max(0.0)))
}
