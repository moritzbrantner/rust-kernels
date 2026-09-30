//! Retained f64 geometry for a rectilinear, checkerboard-triangulated heightfield.
//! Sampling projects a point vertically; it is not a ray or swept-volume query.

use std::{error::Error, fmt, mem::size_of};

use crate::math3::Vec3;

/// Coordinates increase strictly; heights use row-major Z then X order.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightfieldData3 {
    pub columns: Vec<f64>,
    pub rows: Vec<f64>,
    pub heights: Vec<f64>,
    /// Row-major cells. `None` retains every cell; false omits both triangles.
    pub active_cells: Option<Vec<bool>>,
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
            let (triangle, weights) = if (cx + cz) % 2 == 0 {
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
        let indices = match ((cx + cz) % 2, triangle) {
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
