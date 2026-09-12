const ROLLING_BASE: u64 = 257;

/// Zero-copy iterator over overlapping fixed-width windows.
///
/// `width == 0` and `width > values.len()` both produce an empty iterator.
/// Otherwise the iterator yields every overlapping slice of exactly `width`
/// elements in encounter order.
#[derive(Clone, Debug)]
pub struct Shingles<'a, T> {
    values: &'a [T],
    width: usize,
    next: usize,
    remaining: usize,
}

/// Returns overlapping zero-copy shingles over `values`.
#[must_use]
pub fn shingles<T>(values: &[T], width: usize) -> Shingles<'_, T> {
    let remaining = if width == 0 || width > values.len() {
        0
    } else {
        values.len() - width + 1
    };
    Shingles {
        values,
        width,
        next: 0,
        remaining,
    }
}

impl<'a, T> Iterator for Shingles<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let start = self.next;
        self.next += 1;
        self.remaining -= 1;
        Some(&self.values[start..start + self.width])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for Shingles<'_, T> {
    fn len(&self) -> usize {
        self.remaining
    }
}

/// Iterator over deterministic polynomial hashes of overlapping byte windows.
///
/// After the first window is hashed in `O(width)`, each subsequent value is
/// derived in `O(1)` with wrapping `u64` arithmetic. Hash equality is candidate
/// evidence only: callers must compare original bytes when collisions matter.
#[derive(Clone, Debug)]
pub struct RollingHashes<'a> {
    bytes: &'a [u8],
    width: usize,
    next: usize,
    remaining: usize,
    high_power: u64,
    current: u64,
}

/// Returns deterministic rolling hashes for every overlapping byte shingle.
///
/// `width == 0` and `width > bytes.len()` produce an empty iterator. The hash
/// function is intentionally stable and non-cryptographic.
#[must_use]
pub fn rolling_hashes(bytes: &[u8], width: usize) -> RollingHashes<'_> {
    if width == 0 || width > bytes.len() {
        return RollingHashes {
            bytes,
            width,
            next: 0,
            remaining: 0,
            high_power: 1,
            current: 0,
        };
    }

    let high_power = rolling_power(width - 1);
    let current = hash_window(&bytes[..width]);
    RollingHashes {
        bytes,
        width,
        next: 0,
        remaining: bytes.len() - width + 1,
        high_power,
        current,
    }
}

impl Iterator for RollingHashes<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let output = self.current;
        self.remaining -= 1;
        if self.remaining > 0 {
            let outgoing = byte_term(self.bytes[self.next]).wrapping_mul(self.high_power);
            let incoming = byte_term(self.bytes[self.next + self.width]);
            self.current = self
                .current
                .wrapping_sub(outgoing)
                .wrapping_mul(ROLLING_BASE)
                .wrapping_add(incoming);
            self.next += 1;
        }
        Some(output)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for RollingHashes<'_> {
    fn len(&self) -> usize {
        self.remaining
    }
}

fn hash_window(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0_u64, |hash, byte| {
        hash.wrapping_mul(ROLLING_BASE)
            .wrapping_add(byte_term(*byte))
    })
}

fn byte_term(byte: u8) -> u64 {
    u64::from(byte) + 1
}

fn rolling_power(exponent: usize) -> u64 {
    (0..exponent).fold(1_u64, |power, _| power.wrapping_mul(ROLLING_BASE))
}

#[cfg(test)]
mod tests {
    use super::{hash_window, rolling_hashes, shingles};

    #[test]
    fn shingles_are_zero_copy_overlapping_windows_with_explicit_boundaries() {
        let values = [10, 20, 30, 40];
        let windows = shingles(&values, 2).collect::<Vec<_>>();
        assert_eq!(windows, vec![&values[0..2], &values[1..3], &values[2..4]]);
        assert!(std::ptr::eq(windows[0].as_ptr(), values.as_ptr()));

        assert_eq!(shingles(&values, 1).count(), values.len());
        assert_eq!(
            shingles(&values, values.len()).collect::<Vec<_>>(),
            vec![values.as_slice()]
        );
        assert_eq!(shingles(&values, 0).count(), 0);
        assert_eq!(shingles(&values, values.len() + 1).count(), 0);
    }

    #[test]
    fn shingle_iterator_reports_exact_remaining_length() {
        let values = [1, 2, 3, 4, 5];
        let mut windows = shingles(&values, 3);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows.next(), Some(&values[0..3]));
        assert_eq!(windows.len(), 2);
        assert_eq!(
            windows.collect::<Vec<_>>(),
            vec![&values[1..4], &values[2..5]]
        );
    }

    #[test]
    fn rolling_updates_match_recomputing_every_window() {
        let fixtures = [
            Vec::new(),
            vec![0],
            vec![1, 2, 3, 4, 5, 6],
            vec![7; 32],
            generated_bytes(257),
        ];

        for bytes in fixtures {
            for width in 0..=bytes.len() + 1 {
                let expected = if width == 0 || width > bytes.len() {
                    Vec::new()
                } else {
                    bytes.windows(width).map(hash_window).collect::<Vec<_>>()
                };
                assert_eq!(
                    rolling_hashes(&bytes, width).collect::<Vec<_>>(),
                    expected,
                    "len={}, width={width}",
                    bytes.len()
                );
            }
        }
    }

    #[test]
    fn rolling_hashes_are_stable_for_a_known_fixture() {
        assert_eq!(
            rolling_hashes(b"abcd", 3).collect::<Vec<_>>(),
            vec![6_498_345, 6_564_652]
        );
    }

    #[test]
    fn rolling_hash_iterator_reports_exact_remaining_length() {
        let mut hashes = rolling_hashes(b"abcdef", 3);
        assert_eq!(hashes.len(), 4);
        assert!(hashes.next().is_some());
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes.count(), 3);
    }

    fn generated_bytes(len: usize) -> Vec<u8> {
        let mut state = 0x1234_5678_u32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }
}
