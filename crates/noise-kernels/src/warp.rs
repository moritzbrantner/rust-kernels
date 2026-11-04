//! Domain warping: displace coordinates by caller-supplied offset fields.
//!
//! A warp maps `p` to `p + amplitude * (w_x(p), w_y(p)[, w_z(p)])`, where each
//! axis has its own offset sampler. Decorrelating the axes (different
//! permutations or coordinate offsets) and chaining warps are caller choices.

use crate::{CombinatorError, Sampler2, Sampler3};

/// Validated domain-warp amplitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainWarp {
    amplitude: f64,
}

impl DomainWarp {
    /// Validates a warp amplitude. Zero is the identity warp; negative values
    /// displace against the offset fields.
    ///
    /// # Errors
    ///
    /// Returns [`CombinatorError::InvalidWarpAmplitude`] unless `amplitude` is
    /// finite.
    pub fn new(amplitude: f64) -> Result<Self, CombinatorError> {
        if amplitude.is_finite() {
            Ok(Self { amplitude })
        } else {
            Err(CombinatorError::InvalidWarpAmplitude)
        }
    }

    /// Displacement scale applied to every offset sample.
    #[must_use]
    pub const fn amplitude(&self) -> f64 {
        self.amplitude
    }

    /// Returns the warped 2D position.
    ///
    /// Input and warped coordinates must be finite.
    #[must_use]
    pub fn warp2<X: Sampler2, Y: Sampler2>(
        &self,
        offset_x: &X,
        offset_y: &Y,
        x: f64,
        y: f64,
    ) -> [f64; 2] {
        assert!(
            x.is_finite() && y.is_finite(),
            "Domain-warp coordinates must be finite"
        );
        let warped = [
            x + self.amplitude * offset_x.sample2(x, y),
            y + self.amplitude * offset_y.sample2(x, y),
        ];
        assert!(
            warped.iter().all(|value| value.is_finite()),
            "Domain-warped coordinates must be finite"
        );
        warped
    }

    /// Returns the warped 3D position.
    ///
    /// Input and warped coordinates must be finite.
    #[must_use]
    pub fn warp3<X: Sampler3, Y: Sampler3, Z: Sampler3>(
        &self,
        offset_x: &X,
        offset_y: &Y,
        offset_z: &Z,
        x: f64,
        y: f64,
        z: f64,
    ) -> [f64; 3] {
        assert!(
            x.is_finite() && y.is_finite() && z.is_finite(),
            "Domain-warp coordinates must be finite"
        );
        let warped = [
            x + self.amplitude * offset_x.sample3(x, y, z),
            y + self.amplitude * offset_y.sample3(x, y, z),
            z + self.amplitude * offset_z.sample3(x, y, z),
        ];
        assert!(
            warped.iter().all(|value| value.is_finite()),
            "Domain-warped coordinates must be finite"
        );
        warped
    }

    /// Samples `base` at the warped 2D position.
    #[must_use]
    pub fn sample2<B: Sampler2, X: Sampler2, Y: Sampler2>(
        &self,
        base: &B,
        offset_x: &X,
        offset_y: &Y,
        x: f64,
        y: f64,
    ) -> f64 {
        let [x, y] = self.warp2(offset_x, offset_y, x, y);
        base.sample2(x, y)
    }

    /// Samples `base` at the warped 3D position.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn sample3<B: Sampler3, X: Sampler3, Y: Sampler3, Z: Sampler3>(
        &self,
        base: &B,
        offset_x: &X,
        offset_y: &Y,
        offset_z: &Z,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        let [x, y, z] = self.warp3(offset_x, offset_y, offset_z, x, y, z);
        base.sample3(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::DomainWarp;
    use crate::CombinatorError;

    #[test]
    fn rejects_non_finite_amplitude() {
        for amplitude in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                DomainWarp::new(amplitude),
                Err(CombinatorError::InvalidWarpAmplitude)
            );
        }
    }

    #[test]
    fn zero_amplitude_is_identity_and_constant_offsets_translate() {
        let identity = DomainWarp::new(0.0).expect("valid");
        let wave = |x: f64, y: f64| (x * 3.1).sin() + y;
        assert_eq!(identity.warp2(&wave, &wave, 1.25, -7.5), [1.25, -7.5]);

        let warp = DomainWarp::new(-2.0).expect("valid");
        let half = |_: f64, _: f64, _: f64| 0.5;
        let quarter = |_: f64, _: f64, _: f64| 0.25;
        assert_eq!(
            warp.warp3(&half, &quarter, &half, 1.0, 2.0, 3.0),
            [0.0, 1.5, 2.0]
        );
    }

    #[test]
    #[should_panic(expected = "Domain-warped coordinates must be finite")]
    fn rejects_non_finite_warped_coordinates() {
        let warp = DomainWarp::new(1.0).expect("valid");
        let _ = warp.warp2(
            &|_: f64, _: f64| f64::INFINITY,
            &|_: f64, _: f64| 0.0,
            0.0,
            0.0,
        );
    }
}
