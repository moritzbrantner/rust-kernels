//! Caller-supplied scalar field interfaces and combinator configuration errors.
//!
//! Combinators are generic over [`Sampler2`] / [`Sampler3`], so a base field is
//! monomorphised into the octave, warp and curl loops instead of being called
//! through dynamic dispatch. Any `Fn(f64, f64) -> f64` or
//! `Fn(f64, f64, f64) -> f64` closure is a sampler, so a base noise is supplied
//! as, for example, `|x, y| simplex2(&permutation, x, y)`.

use core::fmt;

/// A deterministic two-dimensional scalar field.
pub trait Sampler2 {
    /// Samples the field at `(x, y)`.
    fn sample2(&self, x: f64, y: f64) -> f64;
}

/// A deterministic three-dimensional scalar field.
pub trait Sampler3 {
    /// Samples the field at `(x, y, z)`.
    fn sample3(&self, x: f64, y: f64, z: f64) -> f64;
}

impl<F: Fn(f64, f64) -> f64> Sampler2 for F {
    #[inline(always)]
    fn sample2(&self, x: f64, y: f64) -> f64 {
        self(x, y)
    }
}

impl<F: Fn(f64, f64, f64) -> f64> Sampler3 for F {
    #[inline(always)]
    fn sample3(&self, x: f64, y: f64, z: f64) -> f64 {
        self(x, y, z)
    }
}

/// Rejected combinator configuration.
///
/// Configurations are validated once at construction so repeated sampling
/// never meets an undefined octave schedule, warp or derivative step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CombinatorError {
    /// A fractal needs at least one octave; an empty sum has no normalisation.
    ZeroOctaves,
    /// More octaves than [`crate::Fractal::MAX_OCTAVES`] were requested.
    TooManyOctaves {
        /// Requested octave count.
        octaves: u32,
        /// Largest accepted octave count.
        max: u32,
    },
    /// Lacunarity must be finite and strictly positive.
    InvalidLacunarity,
    /// Gain must be finite and non-negative.
    InvalidGain,
    /// An octave frequency or amplitude, or the amplitude sum, is not finite.
    NonFiniteOctaveSchedule,
    /// Domain-warp amplitude must be finite.
    InvalidWarpAmplitude,
    /// Curl finite-difference epsilon must be finite and strictly positive.
    InvalidCurlEpsilon,
}

impl fmt::Display for CombinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroOctaves => formatter.write_str("fractal octave count must be at least 1"),
            Self::TooManyOctaves { octaves, max } => {
                write!(formatter, "fractal octave count {octaves} exceeds {max}")
            }
            Self::InvalidLacunarity => {
                formatter.write_str("fractal lacunarity must be finite and positive")
            }
            Self::InvalidGain => {
                formatter.write_str("fractal gain must be finite and non-negative")
            }
            Self::NonFiniteOctaveSchedule => {
                formatter.write_str("fractal octave frequencies and amplitudes must stay finite")
            }
            Self::InvalidWarpAmplitude => {
                formatter.write_str("domain-warp amplitude must be finite")
            }
            Self::InvalidCurlEpsilon => {
                formatter.write_str("curl epsilon must be finite and positive")
            }
        }
    }
}

impl std::error::Error for CombinatorError {}
