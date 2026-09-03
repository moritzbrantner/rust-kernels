pub type Vec3 = [f64; 3];

pub const NORMALIZE_EPSILON_SQUARED: f64 = 1.0e-24;

#[must_use]
pub fn is_finite(vector: Vec3) -> bool {
    vector.into_iter().all(f64::is_finite)
}

#[must_use]
pub fn add(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

#[must_use]
pub fn sub(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

#[must_use]
pub fn neg(vector: Vec3) -> Vec3 {
    [-vector[0], -vector[1], -vector[2]]
}

#[must_use]
pub fn scale(vector: Vec3, scalar: f64) -> Vec3 {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

#[must_use]
pub fn dot(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[must_use]
pub fn cross(left: Vec3, right: Vec3) -> Vec3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[must_use]
pub fn triple_cross(left: Vec3, middle: Vec3, right: Vec3) -> Vec3 {
    cross(cross(left, middle), right)
}

#[must_use]
pub fn length_squared(vector: Vec3) -> f64 {
    dot(vector, vector)
}

#[must_use]
pub fn length(vector: Vec3) -> f64 {
    length_squared(vector).sqrt()
}

#[must_use]
pub fn normalized(vector: Vec3) -> Option<Vec3> {
    let length_squared = length_squared(vector);
    if !length_squared.is_finite() || length_squared <= NORMALIZE_EPSILON_SQUARED {
        return None;
    }
    Some(scale(vector, length_squared.sqrt().recip()))
}

#[must_use]
pub fn lerp(start: Vec3, end: Vec3, parameter: f64) -> Vec3 {
    add(start, scale(sub(end, start), parameter))
}

#[must_use]
pub fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{
        NORMALIZE_EPSILON_SQUARED, add, clamp01, cross, dot, length, length_squared, lerp, neg,
        normalized, scale, sub, triple_cross,
    };

    #[test]
    fn vector_arithmetic_is_component_wise() {
        let left = [1.0, -2.0, 3.0];
        let right = [4.0, 5.0, -6.0];
        assert_eq!(add(left, right), [5.0, 3.0, -3.0]);
        assert_eq!(sub(left, right), [-3.0, -7.0, 9.0]);
        assert_eq!(neg(left), [-1.0, 2.0, -3.0]);
        assert_eq!(scale(left, 2.0), [2.0, -4.0, 6.0]);
    }

    #[test]
    fn dot_cross_and_triple_cross_match_known_axes() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        let z = [0.0, 0.0, 1.0];
        assert_eq!(dot(x, y), 0.0);
        assert_eq!(cross(x, y), z);
        assert_eq!(triple_cross(x, y, x), y);
    }

    #[test]
    fn normalization_preserves_direction_and_rejects_degenerate_vectors() {
        let vector = [3.0, 4.0, 0.0];
        assert_eq!(length_squared(vector), 25.0);
        assert_eq!(length(vector), 5.0);
        let unit = normalized(vector).expect("non-zero vector should normalize");
        assert!((unit[0] - 0.6).abs() < 1.0e-12);
        assert!((unit[1] - 0.8).abs() < 1.0e-12);
        assert_eq!(unit[2], 0.0);
        assert!(normalized([0.0; 3]).is_none());
        assert!(normalized([NORMALIZE_EPSILON_SQUARED.sqrt() / 2.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn interpolation_and_clamping_are_deterministic() {
        assert_eq!(lerp([0.0; 3], [2.0, 4.0, 6.0], 0.25), [0.5, 1.0, 1.5]);
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(0.4), 0.4);
        assert_eq!(clamp01(2.0), 1.0);
    }
}
