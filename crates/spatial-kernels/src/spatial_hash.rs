use crate::Aabb;

/// Integer coordinate of a cell in a 3D spatial hash grid.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CellCoord3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CellCoord3 {
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub const fn as_array(self) -> [i32; 3] {
        [self.x, self.y, self.z]
    }
}

/// Maps world-space points and AABBs into deterministic fixed-size 3D cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialHash3D {
    cell_size: f32,
}

impl SpatialHash3D {
    #[must_use]
    pub fn new(cell_size: f32) -> Self {
        assert!(
            cell_size.is_finite() && cell_size > 0.0,
            "cell size must be positive and finite"
        );
        Self { cell_size }
    }

    #[must_use]
    pub const fn cell_size(self) -> f32 {
        self.cell_size
    }

    #[must_use]
    pub fn cell_for_point(self, point: [f32; 3]) -> CellCoord3 {
        assert!(
            point.iter().all(|coordinate| coordinate.is_finite()),
            "spatial-hash point coordinates must be finite"
        );
        CellCoord3::new(
            cell_coordinate(point[0], self.cell_size),
            cell_coordinate(point[1], self.cell_size),
            cell_coordinate(point[2], self.cell_size),
        )
    }

    #[must_use]
    pub fn cell_bounds(self, bounds: Aabb) -> (CellCoord3, CellCoord3) {
        (
            self.cell_for_point(bounds.min),
            self.cell_for_point(bounds.max),
        )
    }

    /// Returns a stable 64-bit hash for a cell coordinate.
    ///
    /// The cell coordinate itself remains the collision-safe key; this value is
    /// useful when a consumer needs a reproducible compact bucket hash.
    #[must_use]
    pub fn hash_cell(cell: CellCoord3) -> u64 {
        let x = mix64(u64::from(cell.x as u32) ^ 0x9e37_79b9_7f4a_7c15);
        let y = mix64(u64::from(cell.y as u32) ^ 0xbf58_476d_1ce4_e5b9);
        let z = mix64(u64::from(cell.z as u32) ^ 0x94d0_49bb_1331_11eb);
        x ^ y.rotate_left(21) ^ z.rotate_left(42)
    }

    #[must_use]
    pub fn hash_point(self, point: [f32; 3]) -> u64 {
        Self::hash_cell(self.cell_for_point(point))
    }
}

fn cell_coordinate(value: f32, cell_size: f32) -> i32 {
    let cell = (f64::from(value) / f64::from(cell_size)).floor();
    assert!(
        cell >= f64::from(i32::MIN) && cell <= f64::from(i32::MAX),
        "spatial-hash coordinate exceeds i32 cell range"
    );
    cell as i32
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{CellCoord3, SpatialHash3D};
    use crate::Aabb;

    #[test]
    fn point_mapping_uses_floor_for_negative_coordinates() {
        let hash = SpatialHash3D::new(2.0);
        assert_eq!(
            hash.cell_for_point([-0.1, 0.0, 3.9]),
            CellCoord3::new(-1, 0, 1)
        );
        assert_eq!(
            hash.cell_for_point([-2.0, 2.0, 4.0]),
            CellCoord3::new(-1, 1, 2)
        );
    }

    #[test]
    fn aabb_mapping_returns_inclusive_minimum_and_maximum_cells() {
        let hash = SpatialHash3D::new(1.0);
        let bounds = Aabb::new([-1.2, 0.1, 2.0], [1.0, 2.9, 2.5]);
        assert_eq!(
            hash.cell_bounds(bounds),
            (CellCoord3::new(-2, 0, 2), CellCoord3::new(1, 2, 2))
        );
    }

    #[test]
    fn compact_hash_is_stable_for_cells_and_points() {
        let hash = SpatialHash3D::new(0.5);
        let point = [1.2, -3.4, 5.6];
        let cell = hash.cell_for_point(point);
        assert_eq!(hash.hash_point(point), SpatialHash3D::hash_cell(cell));
        assert_eq!(
            SpatialHash3D::hash_cell(CellCoord3::new(1, 2, 3)),
            0x4b31_0782_3f9d_80fa
        );
    }

    #[test]
    #[should_panic(expected = "spatial-hash point coordinates must be finite")]
    fn rejects_non_finite_points() {
        let _ = SpatialHash3D::new(1.0).cell_for_point([f32::INFINITY, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "spatial-hash coordinate exceeds i32 cell range")]
    fn rejects_points_outside_representable_cell_range() {
        let _ = SpatialHash3D::new(f32::MIN_POSITIVE).cell_for_point([f32::MAX, 0.0, 0.0]);
    }
}
