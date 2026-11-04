use spatial_kernels::Aabb;

/// A segment `origin + t * delta` prepared for repeated padded slab tests.
pub(crate) struct Segment {
    origin: [f32; 3],
    inverse: [f32; 3],
    parallel: [bool; 3],
    pad: f32,
}

impl Segment {
    pub(crate) fn new(origin: [f32; 3], delta: [f32; 3], pad: f32) -> Self {
        Self {
            origin,
            inverse: delta.map(|d| if d == 0.0 { 0.0 } else { 1.0 / d }),
            parallel: delta.map(|d| d == 0.0),
            pad,
        }
    }

    /// Entry parameter into `bounds` padded by `pad` for `t` in `0..=max_t`, or `None`.
    pub(crate) fn entry(&self, bounds: &Aabb, max_t: f32) -> Option<f32> {
        let (mut enter, mut exit) = (0.0_f32, max_t);
        for axis in 0..3 {
            let (lo, hi) = (bounds.min[axis] - self.pad, bounds.max[axis] + self.pad);
            let o = self.origin[axis];
            if self.parallel[axis] {
                if o < lo || o > hi {
                    return None;
                }
                continue;
            }
            let a = (lo - o) * self.inverse[axis];
            let b = (hi - o) * self.inverse[axis];
            enter = enter.max(a.min(b));
            exit = exit.min(a.max(b));
        }
        (enter <= exit).then_some(enter)
    }
}

#[cfg(test)]
pub(crate) fn segment_entry(
    bounds: Aabb,
    origin: [f32; 3],
    delta: [f32; 3],
    max_t: f32,
    pad: f32,
) -> Option<f32> {
    Segment::new(origin, delta, pad).entry(&bounds, max_t)
}
