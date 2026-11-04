//! Straightforward-loop oracles and numerical properties for the combinators.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration tests are not production code; a panic is the intended failure report"
)]

use noise_kernels::{
    Curl, DomainWarp, Fractal, Permutation, perlin2, perlin3, simplex2, simplex3, worley2, worley3,
};

/// Independent octave schedule: `lacunarity^i` / `gain^i` via `powi`, and the
/// shape applied before normalising by the explicit amplitude sum.
fn oracle_sum(
    octaves: u32,
    lacunarity: f64,
    gain: f64,
    shape: impl Fn(f64) -> f64,
    mut sample_at: impl FnMut(f64) -> f64,
) -> f64 {
    let mut weighted = 0.0;
    let mut amplitudes = 0.0;
    for octave in 0..octaves {
        let exponent = i32::try_from(octave).expect("small octave count");
        let amplitude = gain.powi(exponent);
        weighted += amplitude * shape(sample_at(lacunarity.powi(exponent)));
        amplitudes += amplitude;
    }
    weighted / amplitudes
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    let tolerance = 1e-12 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: expected {expected}, got {actual}"
    );
}

const SCHEDULES: [(u32, f64, f64); 6] = [
    (1, 2.0, 0.5),
    (2, 2.0, 0.5),
    (4, 1.9, 0.65),
    (6, 2.0, 0.0),
    (8, 3.0, 0.45),
    (12, 2.0, 1.0),
];

fn coordinates(index: u32) -> (f64, f64, f64) {
    let t = f64::from(index);
    (t * 0.731 - 40.0, t * -0.379 + 13.5, t * 0.211 - 2.25)
}

#[test]
fn fractals_match_loop_oracles_over_perlin_simplex_and_worley() {
    let permutation = Permutation::from_seed(0x0f0f_1234);
    let perlin_2 = |x, y| perlin2(&permutation, x, y);
    let simplex_2 = |x, y| simplex2(&permutation, x, y);
    let worley_2 = |x, y| worley2(&permutation, x, y).f1;
    let perlin_3 = |x, y, z| perlin3(&permutation, x, y, z);
    let simplex_3 = |x, y, z| simplex3(&permutation, x, y, z);
    let worley_3 = |x, y, z| worley3(&permutation, x, y, z).f2;

    let identity = |signal: f64| signal;
    let billow = |signal: f64| 2.0 * signal.abs() - 1.0;
    let ridge = |signal: f64| (1.0 - signal.abs()) * (1.0 - signal.abs());

    for (octaves, lacunarity, gain) in SCHEDULES {
        let fractal = Fractal::new(octaves, lacunarity, gain).expect("valid schedule");
        for index in 0..96 {
            let (x, y, z) = coordinates(index);
            let context = format!("{octaves}/{lacunarity}/{gain} at ({x}, {y}, {z})");
            let oracle2 = |base: &dyn Fn(f64, f64) -> f64, shape: &dyn Fn(f64) -> f64| {
                oracle_sum(octaves, lacunarity, gain, shape, |f| base(x * f, y * f))
            };
            let oracle3 = |base: &dyn Fn(f64, f64, f64) -> f64, shape: &dyn Fn(f64) -> f64| {
                oracle_sum(octaves, lacunarity, gain, shape, |f| {
                    base(x * f, y * f, z * f)
                })
            };

            for (name, base) in [
                ("perlin", &perlin_2 as &dyn Fn(f64, f64) -> f64),
                ("simplex", &simplex_2),
                ("worley", &worley_2),
            ] {
                assert_close(
                    fractal.fbm2(&base, x, y),
                    oracle2(base, &identity),
                    &format!("fbm2 {name} {context}"),
                );
                assert_close(
                    fractal.billow2(&base, x, y),
                    oracle2(base, &billow),
                    &format!("billow2 {name} {context}"),
                );
                assert_close(
                    fractal.ridged2(&base, x, y),
                    oracle2(base, &ridge),
                    &format!("ridged2 {name} {context}"),
                );
            }
            for (name, base) in [
                ("perlin", &perlin_3 as &dyn Fn(f64, f64, f64) -> f64),
                ("simplex", &simplex_3),
                ("worley", &worley_3),
            ] {
                assert_close(
                    fractal.fbm3(&base, x, y, z),
                    oracle3(base, &identity),
                    &format!("fbm3 {name} {context}"),
                );
                assert_close(
                    fractal.billow3(&base, x, y, z),
                    oracle3(base, &billow),
                    &format!("billow3 {name} {context}"),
                );
                assert_close(
                    fractal.ridged3(&base, x, y, z),
                    oracle3(base, &ridge),
                    &format!("ridged3 {name} {context}"),
                );
            }
        }
    }
}

#[test]
fn bounded_bases_stay_in_documented_ranges() {
    let permutation = Permutation::from_seed(77);
    let base = |x, y| simplex2(&permutation, x, y);
    let fractal = Fractal::new(7, 2.0, 0.55).expect("valid");
    for index in 0..512 {
        let (x, y, _) = coordinates(index);
        let fbm = fractal.fbm2(&base, x, y);
        let billow = fractal.billow2(&base, x, y);
        let ridged = fractal.ridged2(&base, x, y);
        assert!((-1.0..=1.0).contains(&fbm), "fbm {fbm}");
        assert!((-1.0..=1.0).contains(&billow), "billow {billow}");
        assert!((0.0..=1.0).contains(&ridged), "ridged {ridged}");
    }
}

#[test]
fn domain_warp_is_deterministic_and_matches_manual_displacement() {
    let permutation = Permutation::from_seed(0xabcdef);
    let other = Permutation::from_seed(0xabcdf0);
    let offset_x = |x, y| simplex2(&permutation, x + 17.0, y);
    let offset_y = |x, y| simplex2(&other, x, y - 31.0);
    let base = |x, y| perlin2(&permutation, x, y);
    let offset_x3 = |x, y, z| simplex3(&permutation, x, y, z + 5.0);
    let offset_y3 = |x, y, z| simplex3(&other, x, y, z);
    let offset_z3 = |x, y, z| perlin3(&other, x - 9.0, y, z);
    let base3 = |x, y, z| worley3(&permutation, x, y, z).f1;

    for amplitude in [0.0, 0.5, -1.25, 4.0] {
        let warp = DomainWarp::new(amplitude).expect("valid");
        for index in 0..128 {
            let (x, y, z) = coordinates(index);
            let expected2 = [
                x + amplitude * offset_x(x, y),
                y + amplitude * offset_y(x, y),
            ];
            let warped2 = warp.warp2(&offset_x, &offset_y, x, y);
            assert_eq!(warped2, expected2);
            assert_eq!(warp.warp2(&offset_x, &offset_y, x, y), warped2);
            assert_eq!(
                warp.sample2(&base, &offset_x, &offset_y, x, y),
                base(expected2[0], expected2[1])
            );

            let expected3 = [
                x + amplitude * offset_x3(x, y, z),
                y + amplitude * offset_y3(x, y, z),
                z + amplitude * offset_z3(x, y, z),
            ];
            let warped3 = warp.warp3(&offset_x3, &offset_y3, &offset_z3, x, y, z);
            assert_eq!(warped3, expected3);
            assert_eq!(
                warp.sample3(&base3, &offset_x3, &offset_y3, &offset_z3, x, y, z),
                base3(expected3[0], expected3[1], expected3[2])
            );

            // Displacement is linear in the amplitude and bounded by it for
            // offset fields bounded by one.
            let unit = DomainWarp::new(1.0)
                .expect("valid")
                .warp2(&offset_x, &offset_y, x, y);
            assert_close(warped2[0] - x, amplitude * (unit[0] - x), "linear x");
            assert_close(warped2[1] - y, amplitude * (unit[1] - y), "linear y");
            assert!((warped2[0] - x).abs() <= amplitude.abs() + 1e-12);
        }
    }
}

#[test]
fn warp_composes_with_fractal_offsets() {
    let permutation = Permutation::from_seed(5);
    let fractal = Fractal::new(4, 2.0, 0.5).expect("valid");
    let offset_x = |x, y| fractal.fbm2(&|u, v| simplex2(&permutation, u, v), x, y);
    let offset_y = |x, y| fractal.fbm2(&|u, v| simplex2(&permutation, u + 5.2, v + 1.3), x, y);
    let warp = DomainWarp::new(2.0).expect("valid");
    let base = |x, y| fractal.ridged2(&|u, v| perlin2(&permutation, u, v), x, y);
    let first: Vec<f64> = (0..64)
        .map(|index| {
            let (x, y, _) = coordinates(index);
            warp.sample2(&base, &offset_x, &offset_y, x, y)
        })
        .collect();
    let second: Vec<f64> = (0..64)
        .map(|index| {
            let (x, y, _) = coordinates(index);
            warp.sample2(&base, &offset_x, &offset_y, x, y)
        })
        .collect();
    assert_eq!(first, second);
}

#[test]
fn curl2_matches_analytic_curl_of_scalar_potential() {
    // psi = sin(x) cos(y)  =>  (d psi/dy, -d psi/dx) = (-sin x sin y, -cos x cos y)
    let potential = |x: f64, y: f64| x.sin() * y.cos();
    for epsilon in [1e-3, 1e-4] {
        let curl = Curl::new(epsilon).expect("valid");
        // Central differences are second-order accurate.
        let tolerance = epsilon * epsilon;
        for index in 0..128 {
            let (x, y, _) = coordinates(index);
            let [vx, vy] = curl.curl2(&potential, x, y);
            assert!((vx + x.sin() * y.sin()).abs() <= tolerance);
            assert!((vy + x.cos() * y.cos()).abs() <= tolerance);
        }
    }
}

#[test]
fn curl3_matches_analytic_curl_of_vector_potential() {
    // A = (sin(y z), cos(x z), sin(x) y)
    // curl A = (dAz/dy - dAy/dz, dAx/dz - dAz/dx, dAy/dx - dAx/dy)
    //        = (sin x + x sin(x z), y cos(y z) - y cos x, -z sin(x z) - z cos(y z))
    let ax = |_: f64, y: f64, z: f64| (y * z).sin();
    let ay = |x: f64, _: f64, z: f64| (x * z).cos();
    let az = |x: f64, y: f64, _: f64| x.sin() * y;
    let curl = Curl::new(1e-4).expect("valid");
    for index in 0..128 {
        let t = f64::from(index);
        let (x, y, z) = (t * 0.031 - 2.0, t * -0.017 + 1.5, t * 0.023 - 1.25);
        let velocity = curl.curl3(&ax, &ay, &az, x, y, z);
        let expected = [
            x.sin() + x * (x * z).sin(),
            y * (y * z).cos() - y * x.cos(),
            -z * (x * z).sin() - z * (y * z).cos(),
        ];
        for axis in 0..3 {
            assert!(
                (velocity[axis] - expected[axis]).abs() <= 1e-7,
                "axis {axis} at ({x}, {y}, {z}): {} vs {}",
                velocity[axis],
                expected[axis]
            );
        }
    }
}

/// Central-difference divergence of a 2D field with half step `h`.
fn divergence2(field: impl Fn(f64, f64) -> [f64; 2], x: f64, y: f64, h: f64) -> f64 {
    (field(x + h, y)[0] - field(x - h, y)[0]) / (2.0 * h)
        + (field(x, y + h)[1] - field(x, y - h)[1]) / (2.0 * h)
}

fn divergence3(field: impl Fn(f64, f64, f64) -> [f64; 3], x: f64, y: f64, z: f64, h: f64) -> f64 {
    (field(x + h, y, z)[0] - field(x - h, y, z)[0]) / (2.0 * h)
        + (field(x, y + h, z)[1] - field(x, y - h, z)[1]) / (2.0 * h)
        + (field(x, y, z + h)[2] - field(x, y, z - h)[2]) / (2.0 * h)
}

#[test]
fn curl_fields_of_noise_potentials_are_numerically_divergence_free() {
    let permutation = Permutation::from_seed(0xc0ffee);
    let other = Permutation::from_seed(0xc0ffef);
    let third = Permutation::from_seed(0xc0fff0);
    // Dyadic epsilon keeps the stencil steps exact at these coordinates.
    let epsilon = 1.0 / 1024.0;
    let curl = Curl::new(epsilon).expect("valid");

    let potential = |x, y| simplex2(&permutation, x, y);
    let gradient = |x: f64, y: f64| {
        let [vx, vy] = curl.curl2(&potential, x, y);
        [-vy, vx]
    };
    let mut gradient_divergence = 0.0_f64;
    for index in 0..128 {
        let t = f64::from(index);
        let (x, y) = (t * 0.125 - 8.0, t * -0.0625 + 3.0);
        let divergence = divergence2(|u, v| curl.curl2(&potential, u, v), x, y, epsilon);
        assert!(divergence.abs() <= 1e-9, "2D divergence {divergence}");
        gradient_divergence = gradient_divergence.max(divergence2(gradient, x, y, epsilon).abs());
    }
    // Control: the gradient of the same potential is far from divergence-free,
    // so the property above is not vacuous.
    assert!(gradient_divergence > 1.0, "control {gradient_divergence}");

    let ax = |x, y, z| perlin3(&permutation, x, y, z);
    let ay = |x, y, z| simplex3(&other, x, y, z);
    let az = |x, y, z| perlin3(&third, x, y, z);
    for index in 0..128 {
        let t = f64::from(index);
        let (x, y, z) = (t * 0.125 - 8.0, t * -0.0625 + 3.0, t * 0.03125);
        let divergence = divergence3(
            |u, v, w| curl.curl3(&ax, &ay, &az, u, v, w),
            x,
            y,
            z,
            epsilon,
        );
        assert!(divergence.abs() <= 1e-9, "3D divergence {divergence}");
    }
}
