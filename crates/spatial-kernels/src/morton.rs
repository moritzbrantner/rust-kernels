pub const MORTON_3D_BITS_PER_AXIS: u32 = 21;
pub const MORTON_3D_MAX_COORD: u32 = (1_u32 << MORTON_3D_BITS_PER_AXIS) - 1;

/// Interleaves all 32 bits of two coordinates into a 64-bit Morton/Z-order key.
#[must_use]
pub fn morton2_encode(x: u32, y: u32) -> u64 {
    let mut code = 0_u64;
    for bit in 0_u32..32 {
        code |= ((u64::from(x) >> bit) & 1) << (bit * 2);
        code |= ((u64::from(y) >> bit) & 1) << (bit * 2 + 1);
    }
    code
}

/// Reverses `morton2_encode`.
#[must_use]
pub fn morton2_decode(code: u64) -> (u32, u32) {
    let mut x = 0_u32;
    let mut y = 0_u32;
    for bit in 0_u32..32 {
        x |= (((code >> (bit * 2)) & 1) as u32) << bit;
        y |= (((code >> (bit * 2 + 1)) & 1) as u32) << bit;
    }
    (x, y)
}

/// Interleaves 21 bits from each axis into a 63-bit Morton/Z-order key.
///
/// Returns `None` when any coordinate exceeds `MORTON_3D_MAX_COORD`.
#[must_use]
pub fn morton3_encode(x: u32, y: u32, z: u32) -> Option<u64> {
    if x > MORTON_3D_MAX_COORD || y > MORTON_3D_MAX_COORD || z > MORTON_3D_MAX_COORD {
        return None;
    }

    let mut code = 0_u64;
    for bit in 0_u32..MORTON_3D_BITS_PER_AXIS {
        code |= ((u64::from(x) >> bit) & 1) << (bit * 3);
        code |= ((u64::from(y) >> bit) & 1) << (bit * 3 + 1);
        code |= ((u64::from(z) >> bit) & 1) << (bit * 3 + 2);
    }
    Some(code)
}

/// Reverses `morton3_encode` for any 63-bit Morton key.
#[must_use]
pub fn morton3_decode(code: u64) -> (u32, u32, u32) {
    let mut x = 0_u32;
    let mut y = 0_u32;
    let mut z = 0_u32;
    for bit in 0_u32..MORTON_3D_BITS_PER_AXIS {
        x |= (((code >> (bit * 3)) & 1) as u32) << bit;
        y |= (((code >> (bit * 3 + 1)) & 1) as u32) << bit;
        z |= (((code >> (bit * 3 + 2)) & 1) as u32) << bit;
    }
    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::{
        MORTON_3D_MAX_COORD, morton2_decode, morton2_encode, morton3_decode, morton3_encode,
    };

    #[test]
    fn two_dimensional_round_trips_cover_full_u32_range() {
        for (x, y) in [
            (0, 0),
            (1, 2),
            (u32::MAX, 0),
            (0, u32::MAX),
            (u32::MAX, u32::MAX),
            (0x1234_5678, 0x9abc_def0),
        ] {
            assert_eq!(morton2_decode(morton2_encode(x, y)), (x, y));
        }
    }

    #[test]
    fn three_dimensional_round_trips_cover_supported_boundaries() {
        for (x, y, z) in [
            (0, 0, 0),
            (1, 2, 3),
            (MORTON_3D_MAX_COORD, 0, 0),
            (0, MORTON_3D_MAX_COORD, 0),
            (0, 0, MORTON_3D_MAX_COORD),
            (
                MORTON_3D_MAX_COORD,
                MORTON_3D_MAX_COORD,
                MORTON_3D_MAX_COORD,
            ),
        ] {
            let Some(code) = morton3_encode(x, y, z) else {
                panic!("supported coordinates must encode");
            };
            assert_eq!(morton3_decode(code), (x, y, z));
        }
    }

    #[test]
    fn three_dimensional_encoding_rejects_out_of_range_coordinates() {
        assert_eq!(morton3_encode(MORTON_3D_MAX_COORD + 1, 0, 0), None);
        assert_eq!(morton3_encode(0, MORTON_3D_MAX_COORD + 1, 0), None);
        assert_eq!(morton3_encode(0, 0, MORTON_3D_MAX_COORD + 1), None);
    }

    #[test]
    fn deterministic_generated_fixtures_round_trip() {
        let mut state = 0x1234_5678_u32;
        for _ in 0..128 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = state & MORTON_3D_MAX_COORD;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let y = state & MORTON_3D_MAX_COORD;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let z = state & MORTON_3D_MAX_COORD;

            let Some(code) = morton3_encode(x, y, z) else {
                panic!("masked coordinates must encode");
            };
            assert_eq!(morton3_decode(code), (x, y, z));
        }
    }
}
