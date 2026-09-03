const CROSS_AXIS_EPSILON_SQUARED: f64 = 1.0e-18;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb3 {
    pub center: [f32; 3],
    pub half_extents: [f32; 3],
    /// Intrinsic local X/Y/Z rotations in radians, composed as Rz * Ry * Rx.
    pub rotation_radians_xyz: [f32; 3],
}

impl Obb3 {
    #[must_use]
    pub fn new(
        center: [f32; 3],
        half_extents: [f32; 3],
        rotation_radians_xyz: [f32; 3],
    ) -> Self {
        assert!(
            center.iter().all(|coordinate| coordinate.is_finite()),
            "OBB3 center must be finite"
        );
        assert!(
            half_extents
                .iter()
                .all(|extent| extent.is_finite() && *extent >= 0.0),
            "OBB3 half extents must be non-negative and finite"
        );
        assert!(
            rotation_radians_xyz.iter().all(|angle| angle.is_finite()),
            "OBB3 rotation must be finite"
        );
        Self {
            center,
            half_extents,
            rotation_radians_xyz,
        }
    }

    #[must_use]
    pub fn axes(self) -> [[f64; 3]; 3] {
        let [rx, ry, rz] = self.rotation_radians_xyz.map(f64::from);
        let (sin_x, cos_x) = rx.sin_cos();
        let (sin_y, cos_y) = ry.sin_cos();
        let (sin_z, cos_z) = rz.sin_cos();

        [
            [cos_z * cos_y, sin_z * cos_y, -sin_y],
            [
                cos_z * sin_y * sin_x - sin_z * cos_x,
                sin_z * sin_y * sin_x + cos_z * cos_x,
                cos_y * sin_x,
            ],
            [
                cos_z * sin_y * cos_x + sin_z * sin_x,
                sin_z * sin_y * cos_x - cos_z * sin_x,
                cos_y * cos_x,
            ],
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SatAxis3 {
    /// Unit candidate axis used for the projection test. Inactive cross axes are zero.
    pub axis: [f64; 3],
    pub left_radius: f64,
    pub right_radius: f64,
    pub center_distance: f64,
    /// Positive means projected intervals overlap, zero means touching, and
    /// negative means this axis separates the boxes.
    pub signed_overlap: f64,
    pub separating: bool,
    /// Parallel or near-parallel edge cross products do not define an independent axis.
    pub active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb3SatRelation {
    /// SAT axes in deterministic order: A.x/y/z, B.x/y/z, then A.i × B.j.
    pub axes: [SatAxis3; 15],
    pub overlaps: bool,
    /// Index of the active axis with the smallest signed overlap.
    pub critical_axis: usize,
    pub active_axis_count: usize,
}

#[must_use]
pub fn obb3_sat(left: Obb3, right: Obb3) -> Obb3SatRelation {
    let left_axes = left.axes();
    let right_axes = right.axes();
    let mut candidates = [[0.0; 3]; 15];
    candidates[..3].copy_from_slice(&left_axes);
    candidates[3..6].copy_from_slice(&right_axes);

    let mut next = 6;
    for left_axis in left_axes {
        for right_axis in right_axes {
            candidates[next] = cross(left_axis, right_axis);
            next += 1;
        }
    }

    let center_delta = [
        f64::from(right.center[0]) - f64::from(left.center[0]),
        f64::from(right.center[1]) - f64::from(left.center[1]),
        f64::from(right.center[2]) - f64::from(left.center[2]),
    ];

    let axes = candidates.map(|candidate| {
        let Some(axis) = normalized(candidate) else {
            return SatAxis3 {
                axis: [0.0; 3],
                left_radius: 0.0,
                right_radius: 0.0,
                center_distance: 0.0,
                signed_overlap: 0.0,
                separating: false,
                active: false,
            };
        };

        let left_radius = projection_radius(left, axis);
        let right_radius = projection_radius(right, axis);
        let center_distance = dot(center_delta, axis).abs();
        let signed_overlap = left_radius + right_radius - center_distance;

        SatAxis3 {
            axis,
            left_radius,
            right_radius,
            center_distance,
            signed_overlap,
            separating: signed_overlap < 0.0,
            active: true,
        }
    });

    let critical_axis = axes
        .iter()
        .enumerate()
        .filter(|(_, axis)| axis.active)
        .min_by(|(_, left), (_, right)| left.signed_overlap.total_cmp(&right.signed_overlap))
        .map_or(0, |(index, _)| index);
    let active_axis_count = axes.iter().filter(|axis| axis.active).count();

    Obb3SatRelation {
        overlaps: axes.iter().all(|axis| !axis.active || !axis.separating),
        axes,
        critical_axis,
        active_axis_count,
    }
}

fn projection_radius(obb: Obb3, axis: [f64; 3]) -> f64 {
    let local_axes = obb.axes();
    (0..3)
        .map(|index| f64::from(obb.half_extents[index]) * dot(local_axes[index], axis).abs())
        .sum()
}

fn normalized(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length_squared = dot(vector, vector);
    if length_squared <= CROSS_AXIS_EPSILON_SQUARED {
        return None;
    }
    let inverse_length = length_squared.sqrt().recip();
    Some(vector.map(|value| value * inverse_length))
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[cfg(test)]
mod tests {
    use super::{Obb3, obb3_sat};

    fn axis_aligned(center: [f32; 3], half_extents: [f32; 3]) -> Obb3 {
        Obb3::new(center, half_extents, [0.0; 3])
    }

    #[test]
    fn axis_aligned_overlap_and_separation_match_expected_decisions() {
        let base = axis_aligned([0.0, 0.0, 0.0], [1.0, 2.0, 0.75]);
        assert!(obb3_sat(base, axis_aligned([1.5, 0.0, 0.0], [1.0; 3])).overlaps);
        assert!(obb3_sat(base, axis_aligned([2.0, 0.0, 0.0], [1.0; 3])).overlaps);
        assert!(!obb3_sat(base, axis_aligned([2.01, 0.0, 0.0], [1.0; 3])).overlaps);
    }

    #[test]
    fn rotated_overlap_has_no_active_separator() {
        let left = Obb3::new([0.0, 0.0, 0.0], [1.4, 0.8, 0.9], [0.2, 0.4, -0.15]);
        let right = Obb3::new([1.2, 0.35, 0.25], [1.0, 0.7, 0.8], [-0.3, 0.15, 0.55]);
        let relation = obb3_sat(left, right);
        assert!(relation.overlaps);
        assert!(relation.axes.iter().all(|axis| !axis.active || !axis.separating));
    }

    #[test]
    fn rotated_separation_finds_an_active_separator() {
        let left = Obb3::new([0.0, 0.0, 0.0], [1.4, 0.8, 0.9], [0.2, 0.4, -0.15]);
        let right = Obb3::new([4.0, 1.5, 0.8], [1.0, 0.7, 0.8], [-0.3, 0.15, 0.55]);
        let relation = obb3_sat(left, right);
        assert!(!relation.overlaps);
        assert!(relation.axes.iter().any(|axis| axis.active && axis.separating));
        assert!(relation.axes[relation.critical_axis].separating);
    }

    #[test]
    fn decision_is_symmetric() {
        let left = Obb3::new([-0.5, 0.3, 0.2], [1.3, 0.7, 0.9], [0.42, -0.1, 0.2]);
        let right = Obb3::new([1.2, -0.1, 0.4], [0.8, 1.1, 0.6], [-0.73, 0.35, -0.2]);
        assert_eq!(obb3_sat(left, right).overlaps, obb3_sat(right, left).overlaps);
    }

    #[test]
    fn active_candidate_axes_are_unit_length() {
        let relation = obb3_sat(
            Obb3::new([0.0; 3], [1.0; 3], [0.77, -0.2, 0.1]),
            Obb3::new([1.0, 0.0, 0.5], [1.0; 3], [-0.31, 0.4, -0.7]),
        );
        for axis in relation.axes.into_iter().filter(|axis| axis.active) {
            let length = (axis.axis[0] * axis.axis[0]
                + axis.axis[1] * axis.axis[1]
                + axis.axis[2] * axis.axis[2])
                .sqrt();
            assert!((length - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn parallel_boxes_mark_redundant_cross_axes_inactive() {
        let relation = obb3_sat(
            axis_aligned([0.0; 3], [1.0; 3]),
            axis_aligned([1.0, 0.0, 0.0], [1.0; 3]),
        );
        assert_eq!(relation.active_axis_count, 12);
        assert_eq!(relation.axes.iter().filter(|axis| !axis.active).count(), 3);
    }

    #[test]
    fn critical_axis_has_smallest_active_signed_overlap() {
        let relation = obb3_sat(
            Obb3::new([0.0; 3], [1.4, 0.8, 0.9], [0.2, -0.3, 0.1]),
            Obb3::new([2.7, 0.2, 0.4], [1.0, 0.7, 0.8], [-0.5, 0.25, 0.4]),
        );
        let critical = relation.axes[relation.critical_axis].signed_overlap;
        assert!(
            relation
                .axes
                .iter()
                .filter(|axis| axis.active)
                .all(|axis| critical <= axis.signed_overlap)
        );
    }

    #[test]
    fn axes_are_orthonormal() {
        let axes = Obb3::new([0.0; 3], [1.0; 3], [0.5, -0.7, 1.1]).axes();
        for axis in axes {
            let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-12);
        }
        for left in 0..3 {
            for right in left + 1..3 {
                let dot = axes[left][0] * axes[right][0]
                    + axes[left][1] * axes[right][1]
                    + axes[left][2] * axes[right][2];
                assert!(dot.abs() < 1e-12);
            }
        }
    }

    #[test]
    #[should_panic(expected = "OBB3 half extents must be non-negative and finite")]
    fn rejects_invalid_half_extents() {
        let _ = Obb3::new([0.0; 3], [1.0, -1.0, 1.0], [0.0; 3]);
    }

    #[test]
    #[should_panic(expected = "OBB3 rotation must be finite")]
    fn rejects_non_finite_rotation() {
        let _ = Obb3::new([0.0; 3], [1.0; 3], [0.0, f32::NAN, 0.0]);
    }
}
