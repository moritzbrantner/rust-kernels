use geometry_kernels::primitive3::{
    PrimitiveBody3, PrimitiveShape3, PrimitiveSweepError3, PrimitiveWork3, try_swept_time,
};

fn curved_hit() -> (PrimitiveBody3, PrimitiveBody3) {
    (
        PrimitiveBody3::axis_aligned(
            PrimitiveShape3::capsule(0.0, 1.0),
            [-4.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ),
        PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(1.0), [0.0; 3], [0.0; 3]),
    )
}

#[test]
fn insufficient_budget_is_not_a_miss_and_counts_discarded_work() {
    let (a, b) = curved_hit();
    let mut work = PrimitiveWork3::default();
    assert_eq!(
        try_swept_time(a, b, 1.0, 0.0, 0, &mut work),
        Err(PrimitiveSweepError3::IterationLimit)
    );
    assert_eq!(work.sweep_iterations, 0);
    assert_eq!(
        try_swept_time(a, b, 1.0, 0.0, 1, &mut work),
        Err(PrimitiveSweepError3::IterationLimit)
    );
    assert_eq!(work.sweep_iterations, 1);
    let mut completed = PrimitiveWork3::default();
    assert_eq!(
        try_swept_time(a, b, 1.0, 0.0, 2, &mut completed),
        Ok(Some(0.25))
    );
    assert_eq!(completed.sweep_iterations, 2);
}

#[test]
fn overflowing_distance_is_not_a_separating_plane_miss() {
    // Uniformly scaling the analytic hit above does not change its normalized TOI.
    // The inputs remain finite, but the current distance kernel squares overflow.
    let a = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::capsule(0.0, 1e160),
        [-4e160, 0.0, 0.0],
        [8e160, 0.0, 0.0],
    );
    let b = PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(1e160), [0.0; 3], [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    assert_eq!(
        try_swept_time(a, b, 1.0, 0.0, 128, &mut work),
        Err(PrimitiveSweepError3::NonFiniteComputation)
    );
    assert_eq!(work.sweep_iterations, 1);
}

#[test]
fn separating_and_stationary_cases_have_explicit_successful_misses() {
    let (mut a, b) = curved_hit();
    a.velocity = [-8.0, 0.0, 0.0];
    let mut work = PrimitiveWork3::default();
    assert_eq!(try_swept_time(a, b, 1.0, 0.0, 1, &mut work), Ok(None));
    assert_eq!(work.sweep_iterations, 1);
    a.velocity = [0.0; 3];
    let mut work = PrimitiveWork3::default();
    assert_eq!(try_swept_time(a, b, 1.0, 0.0, 0, &mut work), Ok(None));
    a.position = [2.0, 0.0, 0.0];
    assert_eq!(try_swept_time(a, b, 1.0, 0.0, 0, &mut work), Ok(Some(0.0)));
    assert_eq!(work.sweep_iterations, 0);
}

#[test]
fn swept_sat_obeys_the_same_budget_and_detects_nonfinite_projection() {
    let a = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::cuboid([1.0; 3]),
        [-4.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    );
    let b = PrimitiveBody3::axis_aligned(PrimitiveShape3::wedge([1.0; 3]), [0.0; 3], [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    assert_eq!(
        try_swept_time(a, b, 1.0, 0.0, 1, &mut work),
        Err(PrimitiveSweepError3::IterationLimit)
    );
    assert_eq!(work.sweep_iterations, 1);
    assert_eq!(
        try_swept_time(a, b, 1.0, 0.0, 128, &mut work),
        Ok(Some(0.25))
    );
    let mut overflow = a;
    overflow.position = [f64::MAX, f64::MAX, f64::MAX];
    overflow.axes = [[1.0, 1.0, 1.0]; 3];
    assert_eq!(
        try_swept_time(overflow, b, 1.0, 0.0, 128, &mut work),
        Err(PrimitiveSweepError3::NonFiniteComputation)
    );
}

#[test]
fn invalid_inputs_fail_before_any_search_work() {
    let (a, b) = curved_hit();
    let mut bad_pose = a;
    bad_pose.position[0] = f64::NAN;
    let mut bad_axis = a;
    bad_axis.axes[1][2] = f64::INFINITY;
    let mut bad_velocity = a;
    bad_velocity.velocity[1] = f64::NEG_INFINITY;
    for (a, dt, margin) in [
        (a, -1.0, 0.0),
        (a, f64::NAN, 0.0),
        (a, f64::INFINITY, 0.0),
        (a, 1.0, -1.0),
        (a, 1.0, f64::INFINITY),
        (bad_pose, 1.0, 0.0),
        (bad_axis, 1.0, 0.0),
        (bad_velocity, 1.0, 0.0),
    ] {
        let mut work = PrimitiveWork3::default();
        assert_eq!(
            try_swept_time(a, b, dt, margin, 128, &mut work),
            Err(PrimitiveSweepError3::InvalidInput)
        );
        assert_eq!(work, PrimitiveWork3::default());
    }
}

#[test]
fn all_supported_pair_orders_admit_analytic_center_crossings_and_prove_separating_misses() {
    let shapes = [
        PrimitiveShape3::sphere(1.0),
        PrimitiveShape3::cuboid([1.0; 3]),
        PrimitiveShape3::capsule(0.5, 1.0),
        PrimitiveShape3::wedge([1.0; 3]),
    ];
    for a_shape in shapes {
        for b_shape in shapes {
            for velocity in [[8.0, 0.0, 0.0], [-8.0, 0.0, 0.0], [0.0; 3]] {
                let a = PrimitiveBody3::axis_aligned(a_shape, [-4.0, 0.0, 0.0], velocity);
                let b = PrimitiveBody3::axis_aligned(b_shape, [0.0; 3], [0.0; 3]);
                let mut checked = PrimitiveWork3::default();
                let result = try_swept_time(a, b, 1.0, 0.0, 128, &mut checked).unwrap();
                let mut reversed = PrimitiveWork3::default();
                assert_eq!(
                    try_swept_time(b, a, 1.0, 0.0, 128, &mut reversed),
                    Ok(result)
                );
                assert_eq!(checked, reversed);
                if velocity[0] > 0.0 {
                    assert!(result.is_some());
                } else {
                    assert_eq!(result, None);
                }
            }
        }
    }
}
