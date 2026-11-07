//! Best-first search over the triangles of one translating-capsule sweep, shared by the
//! heightfield and triangle-mesh kernels.
//!
//! Each candidate triangle is solved on its own by Newton steps from below (distance from the
//! translating capsule to one convex triangle is convex in time). Every step's time, raised to a
//! conservative bound from the triangle's plane, is a lower bound on that triangle's touch, so
//! the search always steps the triangle with the lowest bound (then lowest id) and stops once
//! that bound passes the best settled hit or the end of the motion.
//!
//! A triangle fails when it runs out of `max_iterations` steps or its step reports an error.
//! The search returns that error only when the failure happens at a bound at or before the
//! earliest settled hit (or the end of the motion when nothing is hit); a later failure cannot
//! change the hit and is ignored. This does not depend on the order the triangles are searched.

use crate::math3::{Vec3, dot};

/// The query of one capsule sweep, shared by every candidate triangle.
#[derive(Clone, Copy)]
pub(crate) struct CapsuleMotion {
    pub(crate) p: Vec3,
    pub(crate) q: Vec3,
    pub(crate) radius: f64,
    pub(crate) displacement: Vec3,
    /// Separation at or below which the capsule touches a triangle.
    pub(crate) tolerance: f64,
    /// Largest coordinate magnitude of the query.
    pub(crate) scale: f64,
}

impl CapsuleMotion {
    /// A lower bound on the time the capsule comes within `tolerance` of the triangle, from the
    /// gap between their extents along its unit plane normal: 0 when they may already touch,
    /// and infinite when the motion never closes the gap. Conservative by `1e-9 * (1 + scale)`.
    pub(crate) fn plane_bound(&self, triangle: [Vec3; 3], normal: Vec3) -> f64 {
        let corners = triangle.map(|v| dot(v, normal));
        let (low, high) = (
            corners[0].min(corners[1]).min(corners[2]),
            corners[0].max(corners[1]).max(corners[2]),
        );
        let (ep, eq) = (dot(self.p, normal), dot(self.q, normal));
        let scale = triangle
            .into_iter()
            .flatten()
            .map(f64::abs)
            .fold(self.scale, f64::max);
        let reach = self.radius + self.tolerance + 1e-9 * (1.0 + scale);
        let rate = dot(self.displacement, normal);
        // Gaps above and below the plane; NaN compares false and leaves the bound at 0.
        let (above, below) = (ep.min(eq) - high - reach, low - ep.max(eq) - reach);
        let mut bound = 0.0_f64;
        if above > 0.0 {
            bound = if rate < 0.0 {
                above / -rate
            } else {
                f64::INFINITY
            };
        }
        if below > 0.0 {
            bound = bound.max(if rate > 0.0 {
                below / rate
            } else {
                f64::INFINITY
            });
        }
        // Rounding of the quotient.
        bound * (1.0 - 4.0 * f64::EPSILON)
    }
}

/// One triangle's place in the search: its next Newton step from `time`.
#[derive(Clone, Copy, Default)]
pub(crate) struct SweepStep {
    /// Lower bound on the triangle's touch: `time` raised to `plane`.
    bound: f64,
    plane: f64,
    pub(crate) time: f64,
    pub(crate) id: u32,
    iterations: u32,
    /// The caller's index of the triangle, in gathering order.
    pub(crate) index: usize,
}

impl SweepStep {
    /// Lower bound first (NaN, which no hit passes, before everything), then triangle id.
    fn precedes(&self, other: &Self) -> bool {
        failure_key(self.bound)
            .total_cmp(&failure_key(other.bound))
            .then(self.id.cmp(&other.id))
            .is_lt()
    }
}

/// A bound as it orders the search and failures: NaN, which no hit passes, as -inf.
fn failure_key(bound: f64) -> f64 {
    if bound.is_nan() {
        f64::NEG_INFINITY
    } else {
        bound
    }
}

/// Whether a failure at bound `failure` counts against `limit`: a NaN limit, which no
/// iteration passes, counts.
fn not_after(failure: f64, limit: f64) -> bool {
    failure.partial_cmp(&limit) != Some(std::cmp::Ordering::Greater)
}

/// Restores the min-heap order of `heap` below `i`.
fn sift_down(heap: &mut [SweepStep], mut i: usize) {
    loop {
        let (left, right) = (2 * i + 1, 2 * i + 2);
        let mut least = i;
        if left < heap.len() && heap[left].precedes(&heap[least]) {
            least = left;
        }
        if right < heap.len() && heap[right].precedes(&heap[least]) {
            least = right;
        }
        if least == i {
            return;
        }
        heap.swap(i, least);
        i = least;
    }
}

/// Candidate triangles of one sweep: inline up to [`SweepSteps::INLINE`], spilling to a `Vec`
/// above that.
pub(crate) struct SweepSteps {
    inline: [SweepStep; Self::INLINE],
    len: usize,
    spill: Vec<SweepStep>,
}

impl SweepSteps {
    pub(crate) const INLINE: usize = 64;

    pub(crate) fn new() -> Self {
        Self {
            inline: [SweepStep::default(); Self::INLINE],
            len: 0,
            spill: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Adds triangle `id` with plane bound `plane` at the next gathering index.
    pub(crate) fn push(&mut self, id: u32, plane: f64) {
        let step = SweepStep {
            bound: plane,
            plane,
            time: 0.0,
            id,
            iterations: 0,
            index: self.len,
        };
        if self.len < Self::INLINE {
            self.inline[self.len] = step;
        } else {
            if self.len == Self::INLINE {
                self.spill.extend_from_slice(&self.inline);
            }
            self.spill.push(step);
        }
        self.len += 1;
    }

    /// Runs the search. `step` takes one Newton step of a triangle, records a settled hit in
    /// `best`, and returns the next time while the triangle can still be touched within the
    /// motion, or the triangle's error. `limit` is the fraction of the best hit (1 without
    /// one); `exhausted` is the error of a triangle out of steps.
    pub(crate) fn search<B, E: Copy>(
        &mut self,
        max_iterations: u32,
        best: &mut B,
        limit: impl Fn(&B) -> f64,
        mut step: impl FnMut(&SweepStep, &mut B) -> Result<Option<f64>, E>,
        exhausted: E,
    ) -> Result<(), E> {
        let heap = if self.len <= Self::INLINE {
            &mut self.inline[..self.len]
        } else {
            &mut self.spill[..]
        };
        let mut len = heap.len();
        for i in (0..len / 2).rev() {
            sift_down(&mut heap[..len], i);
        }
        // Lowest bound of a failed triangle (NaN as -inf), and its error.
        let mut failure: Option<(f64, E)> = None;
        while len > 0 {
            let current = heap[0];
            if current.bound > limit(best) {
                break;
            }
            // Every later hit is past this failure: the sweep fails.
            if let Some((at, error)) = failure
                && current.bound > at
                && not_after(at, limit(best))
            {
                return Err(error);
            }
            let next = if current.iterations == max_iterations {
                Err(exhausted)
            } else {
                step(&current, best)
            };
            let next = next.unwrap_or_else(|error| {
                let at = failure_key(current.bound);
                if failure.is_none_or(|(f, _)| at < f) {
                    failure = Some((at, error));
                }
                None
            });
            if let Some(time) = next {
                heap[0] = SweepStep {
                    bound: current.plane.max(time),
                    time,
                    iterations: current.iterations + 1,
                    ..current
                };
            } else {
                len -= 1;
                heap[0] = heap[len];
            }
            sift_down(&mut heap[..len], 0);
        }
        match failure {
            Some((at, error)) if not_after(at, limit(best)) => Err(error),
            _ => Ok(()),
        }
    }
}
