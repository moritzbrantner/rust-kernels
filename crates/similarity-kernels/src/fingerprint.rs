use std::fmt;

const GOLDEN_RATIO_64: u64 = 0x9e37_79b9_7f4a_7c15;

/// A deterministic MinHash signature over caller-supplied stable feature hashes.
///
/// Signatures are comparable only when both the seed and signature length match.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MinHashSignature {
    seed: u64,
    values: Vec<u64>,
    feature_set_empty: bool,
}

impl MinHashSignature {
    /// Seed used to derive the deterministic hash family.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Number of independent MinHash components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the signature has zero components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns whether the source feature set was empty.
    #[must_use]
    pub const fn feature_set_empty(&self) -> bool {
        self.feature_set_empty
    }

    /// Component values in deterministic component order.
    #[must_use]
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}

/// Errors that prevent construction or comparison of MinHash evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MinHashError {
    /// A MinHash signature must contain at least one component.
    ZeroLength,
    /// Signatures with different component counts are not comparable.
    LengthMismatch { left: usize, right: usize },
    /// Signatures generated from different hash-family seeds are not comparable.
    SeedMismatch { left: u64, right: u64 },
}

impl fmt::Display for MinHashError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLength => write!(formatter, "MinHash signature length must be greater than zero"),
            Self::LengthMismatch { left, right } => write!(
                formatter,
                "MinHash signature lengths differ: left={left}, right={right}"
            ),
            Self::SeedMismatch { left, right } => {
                write!(formatter, "MinHash seeds differ: left={left}, right={right}")
            }
        }
    }
}

impl std::error::Error for MinHashError {}

/// Builds a deterministic MinHash signature from stable integer feature hashes.
///
/// Duplicate feature hashes and input order do not affect the result. `seed`
/// identifies the deterministic hash family and is part of signature identity.
/// The function returns [`MinHashError::ZeroLength`] when `signature_len == 0`.
pub fn minhash_signature<I>(
    features: I,
    signature_len: usize,
    seed: u64,
) -> Result<MinHashSignature, MinHashError>
where
    I: IntoIterator<Item = u64>,
{
    if signature_len == 0 {
        return Err(MinHashError::ZeroLength);
    }

    let component_seeds = (0..signature_len)
        .map(|index| component_seed(seed, index))
        .collect::<Vec<_>>();
    let mut values = vec![u64::MAX; signature_len];
    let mut feature_set_empty = true;

    for feature in features {
        feature_set_empty = false;
        for (value, component_seed) in values.iter_mut().zip(component_seeds.iter().copied()) {
            *value = (*value).min(mix64(feature ^ component_seed));
        }
    }

    Ok(MinHashSignature {
        seed,
        values,
        feature_set_empty,
    })
}

/// Estimates Jaccard similarity from compatible MinHash signatures.
///
/// Two signatures generated from empty feature sets have similarity `1.0`; one
/// empty and one non-empty source have similarity `0.0`. Otherwise the estimate
/// is the fraction of equal components.
pub fn minhash_jaccard_estimate(
    left: &MinHashSignature,
    right: &MinHashSignature,
) -> Result<f64, MinHashError> {
    if left.len() != right.len() {
        return Err(MinHashError::LengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    if left.seed != right.seed {
        return Err(MinHashError::SeedMismatch {
            left: left.seed,
            right: right.seed,
        });
    }
    if left.feature_set_empty && right.feature_set_empty {
        return Ok(1.0);
    }
    if left.feature_set_empty || right.feature_set_empty {
        return Ok(0.0);
    }

    let equal = left
        .values
        .iter()
        .zip(&right.values)
        .filter(|(left, right)| left == right)
        .count();
    Ok(equal as f64 / left.len() as f64)
}

/// Builds a 64-bit SimHash fingerprint from stable feature hashes.
///
/// Repeated feature hashes contribute repeatedly. This is equivalent to calling
/// [`simhash64_weighted`] with weight `1` for every feature.
#[must_use]
pub fn simhash64<I>(features: I) -> u64
where
    I: IntoIterator<Item = u64>,
{
    simhash64_weighted(features.into_iter().map(|feature| (feature, 1_u32)))
}

/// Builds a weighted 64-bit SimHash fingerprint from stable feature hashes.
///
/// Weight zero contributes nothing. Positive weight adds or subtracts that
/// amount from each bit accumulator according to the feature hash bit. Ties
/// resolve to zero, making the empty fingerprint exactly `0`.
#[must_use]
pub fn simhash64_weighted<I>(features: I) -> u64
where
    I: IntoIterator<Item = (u64, u32)>,
{
    let mut accumulators = [0_i128; 64];
    for (feature, weight) in features {
        let weight = i128::from(weight);
        for (bit, accumulator) in accumulators.iter_mut().enumerate() {
            if feature & (1_u64 << bit) == 0 {
                *accumulator -= weight;
            } else {
                *accumulator += weight;
            }
        }
    }

    accumulators
        .into_iter()
        .enumerate()
        .fold(0_u64, |fingerprint, (bit, accumulator)| {
            if accumulator > 0 {
                fingerprint | (1_u64 << bit)
            } else {
                fingerprint
            }
        })
}

/// Hamming distance between two 64-bit fingerprints.
#[must_use]
pub const fn hamming_distance64(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}

fn component_seed(seed: u64, index: usize) -> u64 {
    mix64(seed.wrapping_add((index as u64).wrapping_mul(GOLDEN_RATIO_64)))
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{
        MinHashError, hamming_distance64, minhash_jaccard_estimate, minhash_signature, mix64,
        simhash64, simhash64_weighted,
    };

    #[test]
    fn minhash_is_order_and_duplicate_insensitive() {
        let first = minhash_signature([1, 2, 3, 4, 3, 2], 64, 17).unwrap();
        let second = minhash_signature([4, 3, 2, 1], 64, 17).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn minhash_identity_includes_seed_and_length() {
        assert_eq!(
            minhash_signature([1, 2], 0, 5),
            Err(MinHashError::ZeroLength)
        );

        let left = minhash_signature([1, 2], 16, 5).unwrap();
        let different_length = minhash_signature([1, 2], 32, 5).unwrap();
        let different_seed = minhash_signature([1, 2], 16, 6).unwrap();
        assert_eq!(
            minhash_jaccard_estimate(&left, &different_length),
            Err(MinHashError::LengthMismatch {
                left: 16,
                right: 32
            })
        );
        assert_eq!(
            minhash_jaccard_estimate(&left, &different_seed),
            Err(MinHashError::SeedMismatch { left: 5, right: 6 })
        );
    }

    #[test]
    fn minhash_preserves_explicit_empty_set_semantics() {
        let empty = minhash_signature([], 32, 9).unwrap();
        let also_empty = minhash_signature([], 32, 9).unwrap();
        let non_empty = minhash_signature([1], 32, 9).unwrap();
        assert_eq!(minhash_jaccard_estimate(&empty, &also_empty), Ok(1.0));
        assert_eq!(minhash_jaccard_estimate(&empty, &non_empty), Ok(0.0));
    }

    #[test]
    fn minhash_estimates_exact_jaccard_on_deterministic_fixtures() {
        let left = [1, 2, 3, 4];
        let right = [3, 4, 5, 6];
        let exact = 1.0 / 3.0;
        let estimate_32 = minhash_jaccard_estimate(
            &minhash_signature(left, 32, 0).unwrap(),
            &minhash_signature(right, 32, 0).unwrap(),
        )
        .unwrap();
        let estimate_256 = minhash_jaccard_estimate(
            &minhash_signature(left, 256, 0).unwrap(),
            &minhash_signature(right, 256, 0).unwrap(),
        )
        .unwrap();

        assert!((estimate_32 - exact).abs() < 0.05);
        assert!((estimate_256 - exact).abs() < 0.02);
    }

    #[test]
    fn simhash_identity_and_weight_one_are_equivalent() {
        let features = [mix64(1), mix64(2), mix64(3), mix64(4)];
        assert_eq!(
            simhash64(features),
            simhash64_weighted(features.map(|feature| (feature, 1)))
        );
        assert_eq!(simhash64(std::iter::empty()), 0);
    }

    #[test]
    fn simhash_near_duplicate_is_closer_than_disjoint_fixture() {
        let base = (0_u64..32).map(mix64).collect::<Vec<_>>();
        let mut near = base.clone();
        near[3] = mix64(1003);
        near[19] = mix64(1019);
        let disjoint = (100_u64..132).map(mix64).collect::<Vec<_>>();

        let base_hash = simhash64(base);
        let near_distance = hamming_distance64(base_hash, simhash64(near));
        let disjoint_distance = hamming_distance64(base_hash, simhash64(disjoint));
        assert!(near_distance < disjoint_distance);
    }

    #[test]
    fn weighted_simhash_ignores_zero_weight_and_hamming_is_exact() {
        let feature = mix64(42);
        assert_eq!(
            simhash64_weighted([(feature, 3), (mix64(999), 0)]),
            simhash64_weighted([(feature, 3)])
        );
        assert_eq!(hamming_distance64(0b1010, 0b0011), 2);
    }
}
