//! Retained f64 geometry for a rectilinear, checkerboard-triangulated heightfield.
//! Sampling projects a point vertically; it is not a ray or swept-volume query.

use std::{error::Error, fmt, mem::size_of};

use crate::math3::{Vec3, add, cross, dot, length, normalized, scale, sub as vsub};
use crate::primitive3::{closest_point_triangle, closest_segment_points};
use crate::ray_triangle::segment_triangle_f64;

/// How each cell is split into two triangles.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HeightfieldDiagonal3 {
    /// Alternating diagonals: even cells ACD/ADB, odd cells ACB/BCD.
    #[default]
    Checkerboard,
    /// Every cell ACB/BCD (the B–C diagonal), as parry/Rapier heightfields split cells.
    Fixed,
}

/// Coordinates increase strictly; heights use row-major Z then X order.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightfieldData3 {
    pub columns: Vec<f64>,
    pub rows: Vec<f64>,
    pub heights: Vec<f64>,
    /// Row-major cells. `None` retains every cell; false omits both triangles.
    pub active_cells: Option<Vec<bool>>,
    pub diagonal: HeightfieldDiagonal3,
}

/// A surface contact of a capsule (or sphere: equal segment endpoints) with a heightfield.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldContact3 {
    pub triangle: u32,
    /// Distance between the capsule surface and the terrain; negative when overlapping.
    pub separation: f64,
    /// Unit normal from the terrain toward the capsule.
    pub normal: Vec3,
    /// Closest point on the terrain.
    pub point: Vec3,
}

/// A convex polytope in world space for overlap tests: its vertices, outward face normals and
/// edge directions (one per parallel edge class; length does not matter).
#[derive(Clone, Copy, Debug)]
pub struct ConvexPolytope3<'a> {
    pub vertices: &'a [Vec3],
    pub face_normals: &'a [Vec3],
    pub edges: &'a [Vec3],
}

/// First hit of a segment against a heightfield.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldSegmentHit3 {
    pub triangle: u32,
    /// Fraction along the segment in `0..=1`.
    pub fraction: f64,
    pub point: Vec3,
    /// The triangle's upward unit normal.
    pub normal: Vec3,
}

/// First contact of a translating capsule with a heightfield.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldSweepHit3 {
    /// Fraction of the displacement at first contact, in `0..=1`.
    pub fraction: f64,
    pub contact: HeightfieldContact3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeightfieldError3 {
    InvalidDimensions,
    InvalidCoordinates,
    InvalidHeights,
    InvalidCellMask,
    NonFiniteComputation,
    Capacity,
}
impl fmt::Display for HeightfieldError3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "heightfield geometry failed: {self:?}")
    }
}
impl Error for HeightfieldError3 {}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeightfieldPreparation3 {
    pub triangles_prepared: u64,
    /// Vector payload capacities only; excludes allocator and object overhead.
    pub retained_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeightfieldWork3 {
    pub queries: u64,
    /// Bounds and binary-search comparisons against one stored coordinate.
    pub coordinate_comparisons: u64,
    /// Includes omitted cells inspected at a shared grid boundary.
    pub cells_visited: u64,
    pub triangles_tested: u64,
    pub normal_reuses: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldPoint3 {
    /// `2 * (cell_z * cells_x + cell_x) + triangle_in_cell`, including holes.
    pub triangle: u32,
    pub point: Vec3,
    /// Unit normal with positive Y, independent of query height or motion.
    pub normal: Vec3,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldSample3 {
    pub surface: Option<HeightfieldPoint3>,
    pub work: HeightfieldWork3,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldQueryFailure3 {
    pub reason: HeightfieldError3,
    pub work: HeightfieldWork3,
}
impl fmt::Display for HeightfieldQueryFailure3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "heightfield sample failed: {:?}", self.reason)
    }
}
impl Error for HeightfieldQueryFailure3 {}

/// Immutable coordinate/height data and normals prepared once for enabled cells.
///
/// Every cell has corners A=(-X,-Z), B=(+X,-Z), C=(-X,+Z), D=(+X,+Z).
/// Even checkerboard cells use ACD/ADB; odd cells use ACB/BCD. Both wind +Y.
/// Cells own their lower edges; the outer maximum edges remain included. At an
/// omitted cell's exact shared grid edge/vertex, an enabled neighbor can supply
/// the surface. Preference is the positive-side cell, then -X, -Z, -X/-Z.
/// Outside the closed coordinate extent or inside a hole, sampling returns None.
/// Coordinates are stored explicitly, so representable nonuniform spacing is valid.
/// Extreme finite inputs whose differences/products cannot be represented fail.
#[derive(Debug)]
pub struct PreparedHeightfield3 {
    data: HeightfieldData3,
    normals: Vec<Vec3>,
    preparation: HeightfieldPreparation3,
}

impl PreparedHeightfield3 {
    pub fn try_new(data: HeightfieldData3) -> Result<Self, HeightfieldError3> {
        use HeightfieldError3::*;
        if data.columns.len() < 2 || data.rows.len() < 2 {
            return Err(InvalidDimensions);
        }
        let vertex_count = data
            .columns
            .len()
            .checked_mul(data.rows.len())
            .ok_or(Capacity)?;
        let cell_count = (data.columns.len() - 1)
            .checked_mul(data.rows.len() - 1)
            .ok_or(Capacity)?;
        let triangle_count = cell_count.checked_mul(2).ok_or(Capacity)?;
        if u32::try_from(triangle_count).is_err() {
            return Err(Capacity);
        }
        for axis in [&data.columns, &data.rows] {
            if axis.iter().any(|value| !value.is_finite())
                || axis
                    .windows(2)
                    .any(|pair| pair[1] <= pair[0] || !(pair[1] - pair[0]).is_finite())
                || !(axis[axis.len() - 1] - axis[0]).is_finite()
            {
                return Err(InvalidCoordinates);
            }
        }
        if data.heights.len() != vertex_count
            || data.heights.iter().any(|height| !height.is_finite())
        {
            return Err(InvalidHeights);
        }
        if data
            .active_cells
            .as_ref()
            .is_some_and(|cells| cells.len() != cell_count)
        {
            return Err(InvalidCellMask);
        }
        let mut normals = Vec::new();
        normals
            .try_reserve_exact(triangle_count)
            .map_err(|_| Capacity)?;
        normals.resize(triangle_count, [0.0; 3]);
        let mut prepared = Self {
            data,
            normals,
            preparation: HeightfieldPreparation3::default(),
        };
        for cell in 0..cell_count {
            if !prepared.active(cell) {
                continue;
            }
            for triangle in 0..2 {
                let vertices = prepared.vertices(cell, triangle);
                let a = vertices[0];
                let b = vertices[1];
                let c = vertices[2];
                // Scale differences before taking the cross product. The remaining
                // inability to represent a nonzero area is an explicit range error.
                let first = std::array::from_fn::<_, 3, _>(|axis| b[axis] - a[axis]);
                let second = std::array::from_fn::<_, 3, _>(|axis| c[axis] - a[axis]);
                let magnitude = first
                    .into_iter()
                    .chain(second)
                    .map(f64::abs)
                    .fold(0.0, f64::max);
                if !magnitude.is_finite() || magnitude == 0.0 {
                    return Err(NonFiniteComputation);
                }
                let first = first.map(|value| value / magnitude);
                let second = second.map(|value| value / magnitude);
                let cross = crate::math3::cross(first, second);
                let length = cross[0].hypot(cross[1]).hypot(cross[2]);
                if !length.is_finite() || length == 0.0 || cross[1] <= 0.0 {
                    return Err(NonFiniteComputation);
                }
                let normal = cross.map(|value| value / length);
                if normal.into_iter().any(|value| !value.is_finite()) || normal[1] <= 0.0 {
                    return Err(NonFiniteComputation);
                }
                prepared.normals[cell * 2 + triangle] = normal;
                prepared.preparation.triangles_prepared += 1;
            }
        }
        prepared.preparation.retained_bytes = prepared.payload_bytes().ok_or(Capacity)?;
        Ok(prepared)
    }

    #[must_use]
    pub const fn preparation(&self) -> HeightfieldPreparation3 {
        self.preparation
    }

    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.normals.len()
    }

    /// None includes omitted cells and identifiers outside this geometry.
    #[must_use]
    pub fn triangle_vertices(&self, triangle: u32) -> Option<[Vec3; 3]> {
        let triangle = usize::try_from(triangle).ok()?;
        if triangle >= self.normals.len() || !self.active(triangle / 2) {
            return None;
        }
        Some(self.vertices(triangle / 2, triangle % 2))
    }

    /// Allocation-free vertical projection; preparation remains unchanged.
    pub fn sample(&self, x: f64, z: f64) -> Result<HeightfieldSample3, HeightfieldQueryFailure3> {
        let mut work = HeightfieldWork3 {
            queries: 1,
            ..HeightfieldWork3::default()
        };
        let fail = |reason, work| HeightfieldQueryFailure3 { reason, work };
        if !x.is_finite() || !z.is_finite() {
            return Err(fail(HeightfieldError3::InvalidCoordinates, work));
        }
        let Some(cell_x) = locate(&self.data.columns, x, &mut work) else {
            return Ok(HeightfieldSample3 {
                surface: None,
                work,
            });
        };
        let Some(cell_z) = locate(&self.data.rows, z, &mut work) else {
            return Ok(HeightfieldSample3 {
                surface: None,
                work,
            });
        };
        let left = (cell_x > 0 && x == self.data.columns[cell_x]).then(|| cell_x - 1);
        let below = (cell_z > 0 && z == self.data.rows[cell_z]).then(|| cell_z - 1);
        let cells = [
            Some((cell_x, cell_z)),
            left.map(|cx| (cx, cell_z)),
            below.map(|cz| (cell_x, cz)),
            left.zip(below),
        ];
        for (cx, cz) in cells.into_iter().flatten() {
            work.cells_visited += 1;
            let cell = cz * (self.data.columns.len() - 1) + cx;
            if !self.active(cell) {
                continue;
            }
            work.triangles_tested += 1;
            let u =
                (x - self.data.columns[cx]) / (self.data.columns[cx + 1] - self.data.columns[cx]);
            let v = (z - self.data.rows[cz]) / (self.data.rows[cz + 1] - self.data.rows[cz]);
            let even = match self.data.diagonal {
                HeightfieldDiagonal3::Checkerboard => (cx + cz) % 2 == 0,
                HeightfieldDiagonal3::Fixed => false,
            };
            let (triangle, weights) = if even {
                if v >= u {
                    (0, [1.0 - v, v - u, u])
                } else {
                    (1, [1.0 - u, v, u - v])
                }
            } else if u + v <= 1.0 {
                (0, [1.0 - (u + v), v, u])
            } else {
                (1, [1.0 - v, 1.0 - u, (u + v) - 1.0])
            };
            let vertices = self.vertices(cell, triangle);
            let height = vertices
                .into_iter()
                .zip(weights)
                .map(|(vertex, weight)| vertex[1] * weight)
                .sum::<f64>();
            if !height.is_finite() {
                return Err(fail(HeightfieldError3::NonFiniteComputation, work));
            }
            let id = u32::try_from(cell * 2 + triangle)
                .map_err(|_| fail(HeightfieldError3::Capacity, work))?;
            work.normal_reuses += 1;
            return Ok(HeightfieldSample3 {
                surface: Some(HeightfieldPoint3 {
                    triangle: id,
                    point: [x, height, z],
                    normal: self.normals[cell * 2 + triangle],
                }),
                work,
            });
        }
        Ok(HeightfieldSample3 {
            surface: None,
            work,
        })
    }

    /// Visits the enabled triangles of every cell whose footprint overlaps the closed XZ box
    /// `min..max`, without allocating: triangle id, corners and upward normal.
    pub fn visit_triangles_counted(
        &self,
        min: [f64; 2],
        max: [f64; 2],
        work: &mut HeightfieldWork3,
        mut visit: impl FnMut(u32, [Vec3; 3], Vec3),
    ) {
        let Some((x0, x1)) = cell_span(&self.data.columns, min[0], max[0]) else {
            return;
        };
        let Some((z0, z1)) = cell_span(&self.data.rows, min[1], max[1]) else {
            return;
        };
        let cells_x = self.data.columns.len() - 1;
        for cz in z0..=z1 {
            for cx in x0..=x1 {
                work.cells_visited += 1;
                let cell = cz * cells_x + cx;
                if !self.active(cell) {
                    continue;
                }
                for triangle in 0..2 {
                    work.triangles_tested += 1;
                    let id = cell * 2 + triangle;
                    visit(id as u32, self.vertices(cell, triangle), self.normals[id]);
                }
            }
        }
    }

    pub fn visit_triangles(
        &self,
        min: [f64; 2],
        max: [f64; 2],
        visit: impl FnMut(u32, [Vec3; 3], Vec3),
    ) {
        self.visit_triangles_counted(min, max, &mut HeightfieldWork3::default(), visit);
    }

    /// Deepest contact of the capsule with segment `p`–`q` and `radius` among triangles whose
    /// surface lies within `margin` of it (separation at most `margin`). Ties keep the lower
    /// triangle id. A sphere is a capsule with `p == q`.
    #[must_use]
    pub fn capsule_contact(
        &self,
        p: Vec3,
        q: Vec3,
        radius: f64,
        margin: f64,
    ) -> Option<HeightfieldContact3> {
        self.capsule_contact_counted(p, q, radius, margin, &mut HeightfieldWork3::default())
    }

    /// [`Self::capsule_contact`], adding the cells and triangles visited to `work`.
    pub fn capsule_contact_counted(
        &self,
        p: Vec3,
        q: Vec3,
        radius: f64,
        margin: f64,
        work: &mut HeightfieldWork3,
    ) -> Option<HeightfieldContact3> {
        work.queries += 1;
        let reach = radius + margin.max(0.0);
        let min = [p[0].min(q[0]) - reach, p[2].min(q[2]) - reach];
        let max = [p[0].max(q[0]) + reach, p[2].max(q[2]) + reach];
        let mut best: Option<HeightfieldContact3> = None;
        self.visit_triangles_counted(min, max, work, |id, triangle, normal| {
            let Some(contact) = segment_triangle_contact(p, q, radius, triangle, normal) else {
                return;
            };
            if contact.separation <= margin
                && best.is_none_or(|b| contact.separation < b.separation)
            {
                best = Some(HeightfieldContact3 {
                    triangle: id,
                    ..contact
                });
            }
        });
        best
    }

    /// Deepest overlap of the convex `polytope` (a box or wedge) with the enabled triangles under
    /// its footprint, by the separating-axis test: face normals of both shapes and the cross
    /// products of their edges. A touching or overlapping pair reports the axis of least
    /// penetration (`separation <= 0`, normal from the terrain toward the polytope; exact for
    /// convex polytopes); a separated pair reports nothing. Ties keep the lower triangle id.
    #[must_use]
    pub fn polytope_contact_counted(
        &self,
        polytope: ConvexPolytope3<'_>,
        work: &mut HeightfieldWork3,
    ) -> Option<HeightfieldContact3> {
        work.queries += 1;
        let (mut min, mut max) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for v in polytope.vertices {
            min = [min[0].min(v[0]), min[1].min(v[2])];
            max = [max[0].max(v[0]), max[1].max(v[2])];
        }
        let mut best: Option<HeightfieldContact3> = None;
        self.visit_triangles_counted(min, max, work, |id, triangle, normal| {
            let Some(contact) = polytope_triangle_contact(polytope, triangle, normal) else {
                return;
            };
            if best.is_none_or(|b| contact.separation < b.separation) {
                best = Some(HeightfieldContact3 {
                    triangle: id,
                    ..contact
                });
            }
        });
        best
    }

    /// First contact of the capsule `p`–`q`/`radius` translated by `displacement` within
    /// `0..=1`, by conservative advancement over the triangles under the swept footprint.
    /// Starting in contact (separation <= 0) reports fraction 0. Returns `Err` when
    /// `max_iterations` do not settle the contact.
    pub fn capsule_sweep(
        &self,
        p: Vec3,
        q: Vec3,
        radius: f64,
        displacement: Vec3,
        max_iterations: u32,
    ) -> Result<Option<HeightfieldSweepHit3>, HeightfieldError3> {
        self.capsule_sweep_counted(
            p,
            q,
            radius,
            displacement,
            max_iterations,
            &mut HeightfieldWork3::default(),
        )
    }

    /// [`Self::capsule_sweep`], adding the cells and triangle tests to `work`.
    pub fn capsule_sweep_counted(
        &self,
        p: Vec3,
        q: Vec3,
        radius: f64,
        displacement: Vec3,
        max_iterations: u32,
        work: &mut HeightfieldWork3,
    ) -> Result<Option<HeightfieldSweepHit3>, HeightfieldError3> {
        work.queries += 1;
        let scale_bound = p
            .into_iter()
            .chain(q)
            .chain(displacement)
            .map(f64::abs)
            .fold(radius, f64::max);
        // Contact tolerance: rounding of the distance computation at this scale.
        let tolerance = 1e-12 * (1.0 + scale_bound);
        let end_p = add(p, displacement);
        let end_q = add(q, displacement);
        let reach = radius + tolerance;
        let min = [
            p[0].min(q[0]).min(end_p[0]).min(end_q[0]) - reach,
            p[2].min(q[2]).min(end_p[2]).min(end_q[2]) - reach,
        ];
        let max = [
            p[0].max(q[0]).max(end_p[0]).max(end_q[0]) + reach,
            p[2].max(q[2]).max(end_p[2]).max(end_q[2]) + reach,
        ];
        let mut triangles = [(0_u32, [[0.0; 3]; 3], [0.0; 3]); 64];
        let mut count = 0;
        let mut spill = Vec::new();
        self.visit_triangles_counted(min, max, work, |id, t, n| {
            if count < triangles.len() {
                triangles[count] = (id, t, n);
            } else {
                spill.push((id, t, n));
            }
            count += 1;
        });
        // Distance from the translating capsule to one convex triangle is convex in time, so
        // Newton steps from below converge to that triangle's first touch without passing it.
        // Each triangle is solved on its own; the earliest touch wins (ties by triangle id).
        let length = dot(displacement, displacement).sqrt();
        let mut best: Option<HeightfieldSweepHit3> = None;
        for k in 0..count {
            let (id, triangle, normal) = if k < 64 { triangles[k] } else { spill[k - 64] };
            let mut time = 0.0_f64;
            let mut iterations = 0;
            loop {
                if best.is_some_and(|b| time > b.fraction) {
                    break;
                }
                if iterations == max_iterations {
                    return Err(HeightfieldError3::NonFiniteComputation);
                }
                iterations += 1;
                work.triangles_tested += 1;
                let offset = scale(displacement, time);
                let Some(c) = segment_triangle_contact(
                    add(p, offset),
                    add(q, offset),
                    radius,
                    triangle,
                    normal,
                ) else {
                    break;
                };
                if c.separation <= tolerance {
                    let hit = HeightfieldSweepHit3 {
                        fraction: time,
                        contact: HeightfieldContact3 { triangle: id, ..c },
                    };
                    if best.is_none_or(|b| {
                        time < b.fraction || (time == b.fraction && id < b.contact.triangle)
                    }) {
                        best = Some(hit);
                    }
                    break;
                }
                // A lower bound on the time to touch, from the separation and full speed.
                if time + (c.separation - tolerance) / length > 1.0 {
                    break;
                }
                let rate = dot(displacement, c.normal);
                if rate >= 0.0 {
                    // Moving away: by convexity this triangle only gets farther.
                    break;
                }
                time += (c.separation - tolerance * 0.5) / -rate;
                if time > 1.0 {
                    break;
                }
            }
        }
        if let Some(hit) = best {
            return Ok(Some(hit));
        }
        Ok(None)
    }

    /// First hit of the segment `start`–`end` by a grid walk over the cells it crosses.
    #[must_use]
    pub fn cast_segment(&self, start: Vec3, end: Vec3) -> Option<HeightfieldSegmentHit3> {
        self.cast_segment_counted(start, end, &mut HeightfieldWork3::default())
    }

    /// [`Self::cast_segment`], adding the cells crossed and triangles tested to `work`. Work is
    /// linear in the cells the segment's XZ projection crosses, independent of terrain size.
    pub fn cast_segment_counted(
        &self,
        start: Vec3,
        end: Vec3,
        work: &mut HeightfieldWork3,
    ) -> Option<HeightfieldSegmentHit3> {
        work.queries += 1;
        let (columns, rows) = (&self.data.columns, &self.data.rows);
        let delta = vsub(end, start);
        // Clip the XZ projection to the grid's extent.
        let (mut enter, mut exit) = (0.0_f64, 1.0_f64);
        for (o, d, lo, hi) in [
            (start[0], delta[0], columns[0], columns[columns.len() - 1]),
            (start[2], delta[2], rows[0], rows[rows.len() - 1]),
        ] {
            if d == 0.0 {
                if o < lo || o > hi {
                    return None;
                }
            } else {
                let (a, b) = ((lo - o) / d, (hi - o) / d);
                enter = enter.max(a.min(b));
                exit = exit.min(a.max(b));
            }
        }
        if enter > exit {
            return None;
        }
        let at = |t: f64| (start[0] + delta[0] * t, start[2] + delta[2] * t);
        let cell_of = |coordinates: &[f64], value: f64, d: f64| {
            let last = coordinates.len() - 2;
            let i = coordinates
                .partition_point(|&c| c <= value)
                .saturating_sub(1)
                .min(last);
            // On a boundary, moving down: start in the lower cell.
            if d < 0.0 && i > 0 && value == coordinates[i] {
                i - 1
            } else {
                i
            }
        };
        let (x, z) = at(enter);
        let (mut cx, mut cz) = (cell_of(columns, x, delta[0]), cell_of(rows, z, delta[2]));
        let cells_x = columns.len() - 1;
        let boundary = |coordinates: &[f64], cell: usize, d: f64, o: f64| -> f64 {
            if d > 0.0 {
                (coordinates[cell + 1] - o) / d
            } else if d < 0.0 {
                (coordinates[cell] - o) / d
            } else {
                f64::INFINITY
            }
        };
        let mut best: Option<HeightfieldSegmentHit3> = None;
        loop {
            work.cells_visited += 1;
            let cell = cz * cells_x + cx;
            if self.active(cell) {
                for triangle in 0..2 {
                    work.triangles_tested += 1;
                    let id = cell * 2 + triangle;
                    let Some((fraction, point)) =
                        segment_triangle_f64(start, end, self.vertices(cell, triangle))
                    else {
                        continue;
                    };
                    if best.is_none_or(|b| {
                        fraction < b.fraction
                            || (fraction == b.fraction && (id as u32) < b.triangle)
                    }) {
                        best = Some(HeightfieldSegmentHit3 {
                            triangle: id as u32,
                            fraction,
                            point,
                            normal: self.normals[id],
                        });
                    }
                }
            }
            let tx = boundary(columns, cx, delta[0], start[0]);
            let tz = boundary(rows, cz, delta[2], start[2]);
            let leave = tx.min(tz).min(exit);
            // A hit inside this cell's span is the earliest: later cells start after it.
            if best.is_some_and(|b| b.fraction <= leave) || leave >= exit {
                return best;
            }
            if tx <= tz {
                if delta[0] > 0.0 {
                    if cx + 1 >= cells_x {
                        return best;
                    }
                    cx += 1;
                } else {
                    if cx == 0 {
                        return best;
                    }
                    cx -= 1;
                }
            } else if delta[2] > 0.0 {
                if cz + 2 > rows.len() - 1 {
                    return best;
                }
                cz += 1;
            } else {
                if cz == 0 {
                    return best;
                }
                cz -= 1;
            }
        }
    }

    fn active(&self, cell: usize) -> bool {
        self.data
            .active_cells
            .as_ref()
            .is_none_or(|cells| cells[cell])
    }

    fn vertices(&self, cell: usize, triangle: usize) -> [Vec3; 3] {
        let columns = self.data.columns.len();
        let cx = cell % (columns - 1);
        let cz = cell / (columns - 1);
        let a = cz * columns + cx;
        let b = a + 1;
        let c = a + columns;
        let d = c + 1;
        let parity = match self.data.diagonal {
            HeightfieldDiagonal3::Checkerboard => (cx + cz) % 2,
            HeightfieldDiagonal3::Fixed => 1,
        };
        let indices = match (parity, triangle) {
            (0, 0) => [a, c, d],
            (0, _) => [a, d, b],
            (_, 0) => [a, c, b],
            (_, _) => [b, c, d],
        };
        indices.map(|index| {
            [
                self.data.columns[index % columns],
                self.data.heights[index],
                self.data.rows[index / columns],
            ]
        })
    }

    fn payload_bytes(&self) -> Option<usize> {
        let coordinates = self
            .data
            .columns
            .capacity()
            .checked_add(self.data.rows.capacity())?
            .checked_add(self.data.heights.capacity())?
            .checked_mul(size_of::<f64>())?;
        let normals = self.normals.capacity().checked_mul(size_of::<Vec3>())?;
        let mask = self
            .data
            .active_cells
            .as_ref()
            .map_or(0, Vec::capacity)
            .checked_mul(size_of::<bool>())?;
        coordinates.checked_add(normals)?.checked_add(mask)
    }
}

/// Range of cells whose closed extent overlaps `min..=max` along one axis.
fn cell_span(coordinates: &[f64], min: f64, max: f64) -> Option<(usize, usize)> {
    let last = coordinates.len() - 1;
    if max < coordinates[0] || min > coordinates[last] || min.is_nan() || max.is_nan() || min > max
    {
        return None;
    }
    let lo = coordinates
        .partition_point(|&c| c <= min)
        .saturating_sub(1)
        .min(last - 1);
    let hi = coordinates
        .partition_point(|&c| c < max)
        .saturating_sub(1)
        .min(last - 1);
    Some((lo, hi.max(lo)))
}

/// Closest points between the segment `p`–`q` and a triangle, as a capsule contact.
/// Separating-axis overlap of a convex polytope with one triangle (see
/// [`PreparedHeightfield3::polytope_contact_counted`]).
fn polytope_triangle_contact(
    polytope: ConvexPolytope3<'_>,
    triangle: [Vec3; 3],
    triangle_normal: Vec3,
) -> Option<HeightfieldContact3> {
    let span = |points: &[Vec3], axis: Vec3| {
        points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                let d = dot(*p, axis);
                (lo.min(d), hi.max(d))
            })
    };
    let centre = scale(add(add(triangle[0], triangle[1]), triangle[2]), 1.0 / 3.0);
    let polytope_centre = scale(
        polytope
            .vertices
            .iter()
            .fold([0.0; 3], |sum, v| add(sum, *v)),
        1.0 / polytope.vertices.len() as f64,
    );
    let mut best: Option<(f64, Vec3)> = None;
    let mut test = |axis: Vec3| -> bool {
        let Some(axis) = normalized(axis) else {
            return true; // Parallel edges: no axis.
        };
        let (a_lo, a_hi) = span(polytope.vertices, axis);
        let (b_lo, b_hi) = span(&triangle, axis);
        let overlap = (a_hi - b_lo).min(b_hi - a_lo);
        if overlap < 0.0 {
            return false;
        }
        // Orient the axis from the triangle toward the polytope.
        let axis = if dot(vsub(polytope_centre, centre), axis) < 0.0 {
            scale(axis, -1.0)
        } else {
            axis
        };
        if best.is_none_or(|(depth, _)| overlap < depth) {
            best = Some((overlap, axis));
        }
        true
    };
    if !test(triangle_normal) {
        return None;
    }
    for &n in polytope.face_normals {
        if !test(n) {
            return None;
        }
    }
    let triangle_edges = [
        vsub(triangle[1], triangle[0]),
        vsub(triangle[2], triangle[1]),
        vsub(triangle[0], triangle[2]),
    ];
    for &e in polytope.edges {
        for &f in &triangle_edges {
            if !test(cross(e, f)) {
                return None;
            }
        }
    }
    let (depth, normal) = best?;
    // Witness: the polytope's deepest vertex along the normal, moved out by the depth.
    let deepest = polytope
        .vertices
        .iter()
        .copied()
        .fold((f64::INFINITY, [0.0; 3]), |(lo, at), v| {
            let d = dot(v, normal);
            if d < lo { (d, v) } else { (lo, at) }
        })
        .1;
    Some(HeightfieldContact3 {
        triangle: 0,
        separation: -depth,
        normal,
        point: add(deepest, scale(normal, depth)),
    })
}

fn segment_triangle_contact(
    p: Vec3,
    q: Vec3,
    radius: f64,
    triangle: [Vec3; 3],
    normal: Vec3,
) -> Option<HeightfieldContact3> {
    let [a, b, c] = triangle;
    // The segment crossing the triangle: overlap by the deeper endpoint below its plane.
    if let Some((_, point)) = segment_triangle_f64(p, q, triangle) {
        let depth = dot(vsub(p, a), normal)
            .min(dot(vsub(q, a), normal))
            .min(0.0);
        return Some(HeightfieldContact3 {
            triangle: 0,
            separation: depth - radius,
            normal,
            point,
        });
    }
    let mut best = (f64::INFINITY, [0.0; 3], [0.0; 3]);
    let mut consider = |on_segment: Vec3, on_triangle: Vec3| {
        let d = vsub(on_segment, on_triangle);
        let d2 = dot(d, d);
        if d2 < best.0 {
            best = (d2, on_segment, on_triangle);
        }
    };
    for end in [p, q] {
        consider(end, closest_point_triangle(end, a, b, c));
    }
    for (e0, e1) in [(a, b), (b, c), (c, a)] {
        let (on_segment, on_edge) = closest_segment_points(p, q, e0, e1)?;
        consider(on_segment, on_edge);
    }
    let (d2, on_segment, on_triangle) = best;
    if !d2.is_finite() {
        return None;
    }
    let distance = d2.sqrt();
    let toward = vsub(on_segment, on_triangle);
    let normal = if distance > 1e-12 * (1.0 + length(on_segment)) {
        scale(toward, 1.0 / distance)
    } else {
        normal
    };
    Some(HeightfieldContact3 {
        triangle: 0,
        separation: distance - radius,
        normal,
        point: on_triangle,
    })
}

fn locate(coordinates: &[f64], value: f64, work: &mut HeightfieldWork3) -> Option<usize> {
    work.coordinate_comparisons += 1;
    if value < coordinates[0] {
        return None;
    }
    work.coordinate_comparisons += 1;
    if value > coordinates[coordinates.len() - 1] {
        return None;
    }
    let mut low = 0;
    let mut high = coordinates.len() - 1;
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        work.coordinate_comparisons += 1;
        if value < coordinates[middle] {
            high = middle;
        } else {
            low = middle;
        }
    }
    Some(low)
}
