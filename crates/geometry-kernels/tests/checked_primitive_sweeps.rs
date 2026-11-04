#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration tests are not production code; a panic is the intended failure report"
)]

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

#[test]
fn far_from_origin_contact_within_tolerance_is_a_hit() {
    // rust-kernels#175 (mmorpg#427 vale-hub-walk): an offset-inflated capsule walking into a
    // crate's vertical edge at x ≈ 15853 in world coordinates. Rounding leaves the gap about
    // 7e-15 above contact; requiring gap <= 0 crept one time ulp per iteration until the
    // budget ran out.
    let widen = |v: [f32; 3]| v.map(f64::from);
    let capsule = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::capsule(45.0, 35.25),
        widen([15811.787, 385.7618, 2472.5217]),
        [7.0, 0.0, 1.0],
    );
    let crate_box = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::cuboid([41.0, 250.0, 41.0]),
        widen([15894.0, 527.0, 2519.0]),
        [0.0; 3],
    );
    let mut work = PrimitiveWork3::default();
    let fraction = try_swept_time(capsule, crate_box, 1.0, 0.0, 128, &mut work)
        .unwrap()
        .unwrap();
    assert!((fraction - 0.894_596_6).abs() < 1e-6, "{fraction}");
    assert!(work.sweep_iterations < 8, "{}", work.sweep_iterations);
}

#[test]
fn moving_spheres_use_one_analytic_step_for_first_contact_with_margin() {
    let a = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(1.0),
        [-5.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    );
    let b = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(1.0),
        [5.0, 0.0, 0.0],
        [-6.0, 0.0, 0.0],
    );
    let mut work = PrimitiveWork3::default();
    let toi = try_swept_time(a, b, 1.0, 1.0, 1, &mut work).unwrap();
    assert_eq!(toi, Some(0.7));
    assert_eq!(work.sweep_iterations, 1);
    assert_eq!(work.pair_dispatches, [0; 10]);
    let mut reverse = PrimitiveWork3::default();
    assert_eq!(try_swept_time(b, a, 1.0, 1.0, 1, &mut reverse), Ok(toi));
    assert_eq!(reverse, work);
}

#[test]
fn analytic_sphere_sweep_distinguishes_grazing_miss_and_initial_overlap() {
    let moving = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(1.0),
        [-5.0, 2.0, 0.0],
        [10.0, 0.0, 0.0],
    );
    let target = PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(1.0), [0.0; 3], [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    assert_eq!(
        try_swept_time(moving, target, 1.0, 0.0, 1, &mut work),
        Ok(Some(0.5))
    );
    assert_eq!(work.sweep_iterations, 1);
    let near = PrimitiveBody3 {
        position: [0.0, 2.000_000_5, 0.0],
        ..target
    };
    let straight = PrimitiveBody3 {
        position: [-5.0, 0.0, 0.0],
        ..moving
    };
    assert_eq!(
        try_swept_time(straight, near, 1.0, 0.0, 1, &mut work),
        Ok(None)
    );
    let overlapping = PrimitiveBody3 {
        position: [0.0, 0.0, 0.0],
        ..moving
    };
    assert_eq!(
        try_swept_time(overlapping, target, 1.0, 0.0, 1, &mut work),
        Ok(Some(0.0))
    );
}

#[test]
fn moving_sphere_budget_and_stationary_behavior_are_explicit() {
    let moving = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(1.0),
        [-4.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    );
    let target = PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(1.0), [0.0; 3], [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    assert_eq!(
        try_swept_time(moving, target, 1.0, 0.0, 0, &mut work),
        Err(PrimitiveSweepError3::IterationLimit)
    );
    assert_eq!(work, PrimitiveWork3::default());
    let stationary = PrimitiveBody3 {
        velocity: [0.0; 3],
        ..moving
    };
    assert_eq!(
        try_swept_time(stationary, target, 1.0, 0.0, 0, &mut work),
        Ok(None)
    );
    assert_eq!(work.sweep_iterations, 0);
    assert_eq!(
        try_swept_time(moving, target, 1.0, 0.0, 1, &mut work),
        Ok(Some(0.25))
    );
}

#[test]
fn sphere_analytic_path_preserves_extreme_scale_or_reports_overflow() {
    for scale in [1.0e-150, 1.0e-16, 1.0, 1.0e160] {
        let a = PrimitiveBody3::axis_aligned(
            PrimitiveShape3::sphere(scale),
            [-4.0 * scale, 0.0, 0.0],
            [8.0 * scale, 0.0, 0.0],
        );
        let b = PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(scale), [0.0; 3], [0.0; 3]);
        let mut work = PrimitiveWork3::default();
        let time = try_swept_time(a, b, 1.0, 0.0, 1, &mut work)
            .unwrap()
            .unwrap();
        assert!((time - 0.25).abs() < 1.0e-12, "scale={scale}, time={time}");
        assert_eq!(work.sweep_iterations, 1);
    }

    let a = PrimitiveBody3::axis_aligned(
        PrimitiveShape3::sphere(f64::MAX),
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    let b = PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(f64::MAX), [0.0; 3], [0.0; 3]);
    let mut work = PrimitiveWork3::default();
    assert_eq!(
        try_swept_time(a, b, 1.0, 0.0, 1, &mut work),
        Err(PrimitiveSweepError3::NonFiniteComputation)
    );
    assert_eq!(work.sweep_iterations, 1);
}

#[test]
fn generated_sphere_sweeps_match_independent_quadratic_oracle() {
    let mut state = 0x9e37_79b9_u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f64 / (1_u32 << 24) as f64
    };
    for _ in 0..4096 {
        let a_radius = 0.1 + next() * 2.0;
        let b_radius = 0.1 + next() * 2.0;
        let a_position: [f64; 3] = std::array::from_fn(|_| next() * 20.0 - 10.0);
        let b_position: [f64; 3] = std::array::from_fn(|_| next() * 20.0 - 10.0);
        let a_velocity: [f64; 3] = std::array::from_fn(|_| next() * 8.0 - 4.0);
        let b_velocity: [f64; 3] = std::array::from_fn(|_| next() * 8.0 - 4.0);
        let margin = next() * 0.5;
        let a =
            PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(a_radius), a_position, a_velocity);
        let b =
            PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(b_radius), b_position, b_velocity);
        let delta: [f64; 3] = std::array::from_fn(|i| b_position[i] - a_position[i]);
        let travel: [f64; 3] = std::array::from_fn(|i| b_velocity[i] - a_velocity[i]);
        let radius = a_radius + b_radius + margin;
        let c = delta.iter().map(|x| x * x).sum::<f64>() - radius * radius;
        let expected = if c <= 0.0 {
            Some(0.0)
        } else {
            let aa = travel.iter().map(|x| x * x).sum::<f64>();
            let bb = 2.0 * delta.iter().zip(travel).map(|(x, v)| x * v).sum::<f64>();
            let discriminant = bb * bb - 4.0 * aa * c;
            if aa == 0.0 || discriminant < 0.0 {
                None
            } else {
                let time = (-bb - discriminant.sqrt()) / (2.0 * aa);
                (0.0..=1.0).contains(&time).then_some(time)
            }
        };
        let mut work = PrimitiveWork3::default();
        let actual = try_swept_time(a, b, 1.0, margin, 1, &mut work).unwrap();
        assert_eq!(
            actual.is_some(),
            expected.is_some(),
            "a={a:?}, b={b:?}, margin={margin}"
        );
        if let (Some(actual), Some(expected)) = (actual, expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-9,
                "actual={actual}, expected={expected}"
            );
        }
    }
}
