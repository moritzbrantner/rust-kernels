/// Stable LSD radix sort for `u32` values.
///
/// Performs four byte-wise counting passes in O(4n + 4*256) time with O(n)
/// scratch storage. This is intentionally integer-specific rather than a
/// replacement for generic comparison sorting.
pub fn radix_sort_u32(values: &mut [u32]) {
    radix_sort_by_byte(values, 4, |value, pass| {
        ((value >> (pass * 8)) & 0xff) as usize
    });
}

/// Stable LSD radix sort for `u64` values.
///
/// Performs eight byte-wise counting passes in O(8n + 8*256) time with O(n)
/// scratch storage.
pub fn radix_sort_u64(values: &mut [u64]) {
    radix_sort_by_byte(values, 8, |value, pass| {
        ((value >> (pass * 8)) & 0xff) as usize
    });
}

fn radix_sort_by_byte<T, Bucket>(values: &mut [T], passes: usize, mut bucket: Bucket)
where
    T: Copy,
    Bucket: FnMut(&T, usize) -> usize,
{
    if values.len() < 2 {
        return;
    }

    let mut scratch = values.to_vec();
    for pass in 0..passes {
        let mut counts = [0_usize; 256];
        for value in values.iter() {
            counts[bucket(value, pass)] += 1;
        }

        let mut offsets = [0_usize; 256];
        let mut next = 0_usize;
        for (bucket, count) in counts.into_iter().enumerate() {
            offsets[bucket] = next;
            next += count;
        }

        for &value in values.iter() {
            let bucket = bucket(&value, pass);
            scratch[offsets[bucket]] = value;
            offsets[bucket] += 1;
        }
        values.copy_from_slice(&scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::{radix_sort_by_byte, radix_sort_u32, radix_sort_u64};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Tagged {
        key: u32,
        ordinal: u8,
    }

    fn generated_u32(len: usize) -> Vec<u32> {
        let mut state = 0x1234_5678_u32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            })
            .collect()
    }

    fn generated_u64(len: usize) -> Vec<u64> {
        let mut state = 0x1234_5678_9abc_def0_u64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            })
            .collect()
    }

    #[test]
    fn u32_matches_stable_sort_across_representative_fixtures() {
        let fixtures = [
            Vec::new(),
            vec![1],
            vec![0, 1, 2, 3, 4, 5],
            vec![5, 4, 3, 2, 1, 0],
            vec![u32::MAX, 0, 1, u32::MAX, 255, 256, 65_535, 65_536],
            vec![7, 7, 1, 7, 2, 1, 0, 7, 2, 2],
            generated_u32(1024),
        ];

        for mut actual in fixtures {
            let mut expected = actual.clone();
            expected.sort();
            radix_sort_u32(&mut actual);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn u64_matches_stable_sort_across_boundaries_and_generated_data() {
        let fixtures = [
            Vec::new(),
            vec![u64::MAX, 0, 1, 1 << 32, (1 << 32) - 1, u64::MAX - 1],
            generated_u64(1024),
        ];

        for mut actual in fixtures {
            let mut expected = actual.clone();
            expected.sort();
            radix_sort_u64(&mut actual);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn byte_passes_preserve_equal_key_order() {
        let mut values = [
            Tagged { key: 3, ordinal: 0 },
            Tagged { key: 1, ordinal: 0 },
            Tagged { key: 3, ordinal: 1 },
            Tagged { key: 1, ordinal: 1 },
            Tagged { key: 3, ordinal: 2 },
        ];

        radix_sort_by_byte(&mut values, 4, |value, pass| {
            ((value.key >> (pass * 8)) & 0xff) as usize
        });

        assert_eq!(
            values,
            [
                Tagged { key: 1, ordinal: 0 },
                Tagged { key: 1, ordinal: 1 },
                Tagged { key: 3, ordinal: 0 },
                Tagged { key: 3, ordinal: 1 },
                Tagged { key: 3, ordinal: 2 },
            ]
        );
    }
}
