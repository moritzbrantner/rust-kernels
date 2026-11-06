use geometry_kernels::primitive3::{
    PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, try_swept_time,
};
use geometry_kernels::{SphereSweepError, try_sphere_sphere_fraction_f64};

#[test]
fn public_fraction_reports_boundaries_and_misses() {
    for (position, travel, radius, expected) in [
        ([-4.0, 0.0, 0.0], [8.0, 0.0, 0.0], 2.0, Some(0.25)),
        ([-4.0, 0.0, 0.0], [2.0, 0.0, 0.0], 2.0, Some(1.0)),
        ([-2.0, 0.0, 0.0], [0.0; 3], 2.0, Some(0.0)),
        ([0.0; 3], [0.0; 3], 0.0, Some(0.0)),
        ([-4.0, 0.0, 0.0], [0.0; 3], 2.0, None),
        ([-4.0, 0.0, 0.0], [-8.0, 0.0, 0.0], 2.0, None),
        ([-4.0, 2.0, 0.0], [8.0, 0.0, 0.0], 2.0, Some(0.5)),
    ] {
        assert_eq!(
            try_sphere_sphere_fraction_f64(position, travel, radius),
            Ok(expected)
        );
    }
}

#[test]
fn public_fraction_distinguishes_invalid_input_from_numerical_failure() {
    for (position, travel, radius) in [
        ([f64::NAN, 0.0, 0.0], [1.0; 3], 1.0),
        ([1.0; 3], [f64::INFINITY, 0.0, 0.0], 1.0),
        ([1.0; 3], [1.0; 3], f64::INFINITY),
        ([1.0; 3], [1.0; 3], -1.0),
    ] {
        assert_eq!(
            try_sphere_sphere_fraction_f64(position, travel, radius),
            Err(SphereSweepError::InvalidInput)
        );
    }
    assert_eq!(
        try_sphere_sphere_fraction_f64([1.0, 0.0, 0.0], [-1e-160, 0.0, 0.0], 0.5),
        Err(SphereSweepError::NonFiniteComputation),
    );
}

#[test]
fn public_fraction_matches_generic_sphere_sweep_exactly() {
    for scale in [1e-100, 1.0, 1e100] {
        for y in [0.0, 1.0, 2.0, 3.0] {
            let position = [-4.0 * scale, y * scale, 0.0];
            let travel = [8.0 * scale, 0.0, 0.0];
            let a =
                PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(scale), [0.0; 3], [0.0; 3]);
            let b = PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(scale), position, travel);
            assert_eq!(
                try_sphere_sphere_fraction_f64(position, travel, 2.0 * scale).ok(),
                try_swept_time(a, b, 1.0, 0.0, 1, &mut PrimitiveWork3::default()).ok(),
            );
        }
    }
}
