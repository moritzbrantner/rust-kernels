//! Curl-noise helpers using explicit central finite differences.
//!
//! In 2D the velocity is the curl of a scalar potential `psi`,
//! `(d psi/dy, -d psi/dx)`. In 3D it is the curl of a vector potential
//! `(psi_x, psi_y, psi_z)`. Each derivative is a central difference with the
//! configured epsilon, divided by the step actually represented at the sample
//! coordinate. Because the central-difference operators of different axes
//! commute, the discrete divergence of these fields vanishes up to rounding,
//! for any potential.

use crate::{CombinatorError, Sampler2, Sampler3};

/// Validated curl finite-difference configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curl {
    epsilon: f64,
}

impl Curl {
    /// Validates the central-difference half step.
    ///
    /// # Errors
    ///
    /// Returns [`CombinatorError::InvalidCurlEpsilon`] unless `epsilon` is
    /// finite and strictly positive.
    pub fn new(epsilon: f64) -> Result<Self, CombinatorError> {
        if epsilon.is_finite() && epsilon > 0.0 {
            Ok(Self { epsilon })
        } else {
            Err(CombinatorError::InvalidCurlEpsilon)
        }
    }

    /// Central-difference half step.
    #[must_use]
    pub const fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Divergence-free 2D velocity `(d psi/dy, -d psi/dx)` of a scalar
    /// potential, using four potential samples.
    ///
    /// Coordinates must be finite, and `coordinate +/- epsilon` must remain
    /// finite and distinct at the sample magnitude.
    #[must_use]
    pub fn curl2<P: Sampler2>(&self, potential: &P, x: f64, y: f64) -> [f64; 2] {
        assert!(
            x.is_finite() && y.is_finite(),
            "Curl coordinates must be finite"
        );
        let (x_minus, x_plus, x_step) = self.stencil(x);
        let (y_minus, y_plus, y_step) = self.stencil(y);
        let d_dx = (potential.sample2(x_plus, y) - potential.sample2(x_minus, y)) / x_step;
        let d_dy = (potential.sample2(x, y_plus) - potential.sample2(x, y_minus)) / y_step;
        [d_dy, -d_dx]
    }

    /// Divergence-free 3D velocity, the curl of the vector potential
    /// `(psi_x, psi_y, psi_z)`, using twelve potential samples.
    ///
    /// Coordinates must be finite, and `coordinate +/- epsilon` must remain
    /// finite and distinct at the sample magnitude.
    #[must_use]
    pub fn curl3<X: Sampler3, Y: Sampler3, Z: Sampler3>(
        &self,
        potential_x: &X,
        potential_y: &Y,
        potential_z: &Z,
        x: f64,
        y: f64,
        z: f64,
    ) -> [f64; 3] {
        assert!(
            x.is_finite() && y.is_finite() && z.is_finite(),
            "Curl coordinates must be finite"
        );
        let (x_minus, x_plus, x_step) = self.stencil(x);
        let (y_minus, y_plus, y_step) = self.stencil(y);
        let (z_minus, z_plus, z_step) = self.stencil(z);

        let dz_dy =
            (potential_z.sample3(x, y_plus, z) - potential_z.sample3(x, y_minus, z)) / y_step;
        let dy_dz =
            (potential_y.sample3(x, y, z_plus) - potential_y.sample3(x, y, z_minus)) / z_step;
        let dx_dz =
            (potential_x.sample3(x, y, z_plus) - potential_x.sample3(x, y, z_minus)) / z_step;
        let dz_dx =
            (potential_z.sample3(x_plus, y, z) - potential_z.sample3(x_minus, y, z)) / x_step;
        let dy_dx =
            (potential_y.sample3(x_plus, y, z) - potential_y.sample3(x_minus, y, z)) / x_step;
        let dx_dy =
            (potential_x.sample3(x, y_plus, z) - potential_x.sample3(x, y_minus, z)) / y_step;

        [dz_dy - dy_dz, dx_dz - dz_dx, dy_dx - dx_dy]
    }

    #[inline(always)]
    fn stencil(&self, coordinate: f64) -> (f64, f64, f64) {
        let minus = coordinate - self.epsilon;
        let plus = coordinate + self.epsilon;
        let step = plus - minus;
        assert!(
            step.is_finite() && step > 0.0,
            "Curl epsilon must be representable at the sample coordinate"
        );
        (minus, plus, step)
    }
}

#[cfg(test)]
mod tests {
    use super::Curl;
    use crate::CombinatorError;

    #[test]
    fn rejects_invalid_epsilon() {
        for epsilon in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert_eq!(Curl::new(epsilon), Err(CombinatorError::InvalidCurlEpsilon));
        }
    }

    #[test]
    fn linear_potentials_are_differentiated_exactly() {
        let curl = Curl::new(0.25).expect("valid");
        let potential = |x: f64, y: f64| 3.0 * x - 2.0 * y;
        assert_eq!(curl.curl2(&potential, 1.5, -4.0), [-2.0, -3.0]);
    }

    #[test]
    #[should_panic(expected = "Curl epsilon must be representable at the sample coordinate")]
    fn rejects_epsilon_lost_at_large_coordinates() {
        let curl = Curl::new(1e-9).expect("valid");
        let _ = curl.curl2(&|x: f64, y: f64| x + y, 1e12 * 1e9, 0.0);
    }
}
