//! Fractal octave sums over a caller-supplied base field.
//!
//! Octave `i` (starting at 0) samples the base field at `frequency_i * p` with
//! `frequency_i = lacunarity^i` and weights it by `amplitude_i = gain^i`. The
//! weighted sum is multiplied by [`Fractal::normalization`], the reciprocal of
//! `sum(amplitude_i)`, so a base field bounded by `[-1, 1]` gives fBm in
//! `[-1, 1]`, billow in `[-1, 1]` and ridged in `[0, 1]`. Frequencies and
//! amplitudes advance by repeated multiplication, exactly the schedule that
//! [`Fractal::new`] validated. The caller scales coordinates for a base
//! frequency and owns per-octave decorrelation (for example a different
//! permutation or offset per octave inside its sampler).

use crate::{CombinatorError, Sampler2, Sampler3};

/// Validated fractal octave schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fractal {
    octaves: u32,
    lacunarity: f64,
    gain: f64,
    normalization: f64,
}

impl Fractal {
    /// Largest accepted octave count. It bounds per-sample work.
    pub const MAX_OCTAVES: u32 = 32;

    /// Validates an octave schedule.
    ///
    /// # Errors
    ///
    /// Returns [`CombinatorError::ZeroOctaves`] for `octaves == 0`,
    /// [`CombinatorError::TooManyOctaves`] above [`Self::MAX_OCTAVES`],
    /// [`CombinatorError::InvalidLacunarity`] unless `lacunarity` is finite and
    /// positive, [`CombinatorError::InvalidGain`] unless `gain` is finite and
    /// non-negative, and [`CombinatorError::NonFiniteOctaveSchedule`] when any
    /// octave frequency, amplitude or the amplitude sum overflows.
    pub fn new(octaves: u32, lacunarity: f64, gain: f64) -> Result<Self, CombinatorError> {
        if octaves == 0 {
            return Err(CombinatorError::ZeroOctaves);
        }
        if octaves > Self::MAX_OCTAVES {
            return Err(CombinatorError::TooManyOctaves {
                octaves,
                max: Self::MAX_OCTAVES,
            });
        }
        if !(lacunarity.is_finite() && lacunarity > 0.0) {
            return Err(CombinatorError::InvalidLacunarity);
        }
        if !(gain.is_finite() && gain >= 0.0) {
            return Err(CombinatorError::InvalidGain);
        }

        let mut frequency = 1.0_f64;
        let mut amplitude = 1.0_f64;
        let mut amplitude_sum = 0.0_f64;
        for _ in 0..octaves {
            if !(frequency.is_finite() && amplitude.is_finite()) {
                return Err(CombinatorError::NonFiniteOctaveSchedule);
            }
            amplitude_sum += amplitude;
            frequency *= lacunarity;
            amplitude *= gain;
        }
        // amplitude_0 = 1, so the sum is at least 1 and its reciprocal finite.
        if !amplitude_sum.is_finite() {
            return Err(CombinatorError::NonFiniteOctaveSchedule);
        }

        Ok(Self {
            octaves,
            lacunarity,
            gain,
            normalization: 1.0 / amplitude_sum,
        })
    }

    /// Number of octaves.
    #[must_use]
    pub const fn octaves(&self) -> u32 {
        self.octaves
    }

    /// Frequency multiplier between consecutive octaves.
    #[must_use]
    pub const fn lacunarity(&self) -> f64 {
        self.lacunarity
    }

    /// Amplitude multiplier between consecutive octaves.
    #[must_use]
    pub const fn gain(&self) -> f64 {
        self.gain
    }

    /// Reciprocal of the amplitude sum applied to every fractal output.
    #[must_use]
    pub const fn normalization(&self) -> f64 {
        self.normalization
    }

    /// Fractional Brownian motion: `sum(a_i * n_i)`, normalised.
    ///
    /// Coordinates must be finite.
    #[must_use]
    pub fn fbm2<S: Sampler2>(&self, sampler: &S, x: f64, y: f64) -> f64 {
        assert_finite2(x, y);
        self.accumulate2(sampler, x, y, |signal| signal)
    }

    /// Fractional Brownian motion: `sum(a_i * n_i)`, normalised.
    ///
    /// Coordinates must be finite.
    #[must_use]
    pub fn fbm3<S: Sampler3>(&self, sampler: &S, x: f64, y: f64, z: f64) -> f64 {
        assert_finite3(x, y, z);
        self.accumulate3(sampler, x, y, z, |signal| signal)
    }

    /// Billow: `sum(a_i * (2 |n_i| - 1))`, normalised.
    ///
    /// Coordinates must be finite.
    #[must_use]
    pub fn billow2<S: Sampler2>(&self, sampler: &S, x: f64, y: f64) -> f64 {
        assert_finite2(x, y);
        self.accumulate2(sampler, x, y, billow)
    }

    /// Billow: `sum(a_i * (2 |n_i| - 1))`, normalised.
    ///
    /// Coordinates must be finite.
    #[must_use]
    pub fn billow3<S: Sampler3>(&self, sampler: &S, x: f64, y: f64, z: f64) -> f64 {
        assert_finite3(x, y, z);
        self.accumulate3(sampler, x, y, z, billow)
    }

    /// Ridged sum: `sum(a_i * (1 - |n_i|)^2)`, normalised.
    ///
    /// This is the feedback-free ridged form; octaves are not weighted by the
    /// previous octave's signal. Coordinates must be finite.
    #[must_use]
    pub fn ridged2<S: Sampler2>(&self, sampler: &S, x: f64, y: f64) -> f64 {
        assert_finite2(x, y);
        self.accumulate2(sampler, x, y, ridge)
    }

    /// Ridged sum: `sum(a_i * (1 - |n_i|)^2)`, normalised.
    ///
    /// See [`Self::ridged2`]. Coordinates must be finite.
    #[must_use]
    pub fn ridged3<S: Sampler3>(&self, sampler: &S, x: f64, y: f64, z: f64) -> f64 {
        assert_finite3(x, y, z);
        self.accumulate3(sampler, x, y, z, ridge)
    }

    #[inline(always)]
    fn accumulate2<S: Sampler2>(
        &self,
        sampler: &S,
        x: f64,
        y: f64,
        shape: impl Fn(f64) -> f64,
    ) -> f64 {
        let mut frequency = 1.0;
        let mut amplitude = 1.0;
        let mut sum = 0.0;
        for _ in 0..self.octaves {
            sum += amplitude * shape(sampler.sample2(x * frequency, y * frequency));
            frequency *= self.lacunarity;
            amplitude *= self.gain;
        }
        sum * self.normalization
    }

    #[inline(always)]
    fn accumulate3<S: Sampler3>(
        &self,
        sampler: &S,
        x: f64,
        y: f64,
        z: f64,
        shape: impl Fn(f64) -> f64,
    ) -> f64 {
        let mut frequency = 1.0;
        let mut amplitude = 1.0;
        let mut sum = 0.0;
        for _ in 0..self.octaves {
            sum += amplitude * shape(sampler.sample3(x * frequency, y * frequency, z * frequency));
            frequency *= self.lacunarity;
            amplitude *= self.gain;
        }
        sum * self.normalization
    }
}

#[inline(always)]
fn billow(signal: f64) -> f64 {
    2.0 * signal.abs() - 1.0
}

#[inline(always)]
fn ridge(signal: f64) -> f64 {
    let ridge = 1.0 - signal.abs();
    ridge * ridge
}

#[inline(always)]
fn assert_finite2(x: f64, y: f64) {
    assert!(
        x.is_finite() && y.is_finite(),
        "Fractal coordinates must be finite"
    );
}

#[inline(always)]
fn assert_finite3(x: f64, y: f64, z: f64) {
    assert!(
        x.is_finite() && y.is_finite() && z.is_finite(),
        "Fractal coordinates must be finite"
    );
}

#[cfg(test)]
mod tests {
    use super::Fractal;
    use crate::CombinatorError;

    #[test]
    fn rejects_invalid_schedules_deterministically() {
        assert_eq!(Fractal::new(0, 2.0, 0.5), Err(CombinatorError::ZeroOctaves));
        assert_eq!(
            Fractal::new(33, 2.0, 0.5),
            Err(CombinatorError::TooManyOctaves {
                octaves: 33,
                max: 32
            })
        );
        for lacunarity in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                Fractal::new(4, lacunarity, 0.5),
                Err(CombinatorError::InvalidLacunarity)
            );
        }
        for gain in [-0.5, f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert_eq!(
                Fractal::new(4, 2.0, gain),
                Err(CombinatorError::InvalidGain)
            );
        }
        assert_eq!(
            Fractal::new(32, 1e300, 0.5),
            Err(CombinatorError::NonFiniteOctaveSchedule)
        );
        assert_eq!(
            Fractal::new(32, 2.0, 1e300),
            Err(CombinatorError::NonFiniteOctaveSchedule)
        );
    }

    #[test]
    fn accepts_boundary_schedules() {
        let single = Fractal::new(1, 1e300, 1e300).expect("one octave never advances");
        assert_eq!(single.normalization(), 1.0);
        let silent = Fractal::new(8, 2.0, 0.0).expect("zero gain is valid");
        assert_eq!(silent.normalization(), 1.0);
        let max = Fractal::new(Fractal::MAX_OCTAVES, 2.0, 1.0).expect("max octaves");
        assert_eq!(max.normalization(), 1.0 / 32.0);
    }

    #[test]
    fn single_octave_fbm_is_the_base_field() {
        let fractal = Fractal::new(1, 2.0, 0.5).expect("valid");
        let base = |x: f64, y: f64| (x * 1.3).sin() * (y * 0.7).cos();
        for index in 0..32 {
            let x = f64::from(index) * 0.41 - 5.0;
            let y = f64::from(index) * -0.17 + 2.0;
            assert_eq!(fractal.fbm2(&base, x, y), base(x, y));
        }
    }

    #[test]
    fn constant_fields_map_to_shape_constants() {
        let fractal = Fractal::new(6, 2.1, 0.6).expect("valid");
        let close = |actual: f64, expected: f64| assert!((actual - expected).abs() <= 1e-15);
        let zero = |_: f64, _: f64, _: f64| 0.0;
        let one = |_: f64, _: f64, _: f64| 1.0;
        close(fractal.fbm3(&one, 1.0, 2.0, 3.0), 1.0);
        close(fractal.billow3(&zero, 1.0, 2.0, 3.0), -1.0);
        close(fractal.billow3(&one, 1.0, 2.0, 3.0), 1.0);
        close(fractal.ridged3(&zero, 1.0, 2.0, 3.0), 1.0);
        close(fractal.ridged3(&one, 1.0, 2.0, 3.0), 0.0);
    }

    #[test]
    #[should_panic(expected = "Fractal coordinates must be finite")]
    fn rejects_non_finite_coordinates() {
        let fractal = Fractal::new(2, 2.0, 0.5).expect("valid");
        let _ = fractal.fbm2(&|x: f64, y: f64| x + y, f64::NAN, 0.0);
    }
}
