use collection_kernels::LruCache;
use std::hint::black_box;

const CAPACITY: usize = 32_768;
const OPERATIONS: u32 = 1_500_000;
const ROUNDS: u32 = 5;

fn main() {
    let mut checksum = 0_u64;

    for round in 0..ROUNDS {
        let mut cache = LruCache::new(CAPACITY);
        for operation in 0..OPERATIONS {
            let key = operation
                .wrapping_mul(2_654_435_761)
                .wrapping_add(round.wrapping_mul(97))
                & 0xffff;

            if operation % 3 == 0 {
                let value = key ^ operation.rotate_left(7);
                black_box(cache.insert(key, value));
            } else if let Some(value) = cache.get(black_box(&key)) {
                checksum = checksum.wrapping_add(u64::from(*value));
            }
        }
        checksum ^= cache.len() as u64;
    }

    println!("lru-checksum={checksum}");
}
