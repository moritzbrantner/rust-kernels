use search_kernels::{quickselect, radix_sort_u32};
use std::hint::black_box;

const VALUES: usize = 262_144;
const ROUNDS: usize = 24;

fn generated_values() -> Vec<u32> {
    let mut state = 0x1234_5678_u32;
    (0..VALUES)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect()
}

fn main() {
    let input = generated_values();
    let mut checksum = 0_u64;

    for round in 0..ROUNDS {
        let mut sorted = input.clone();
        radix_sort_u32(black_box(&mut sorted));
        checksum ^= u64::from(sorted[(round * 7919) % sorted.len()]);

        let mut selected = input.clone();
        let nth = (round * 104_729) % selected.len();
        checksum = checksum.wrapping_add(u64::from(
            *quickselect(black_box(&mut selected), nth).expect("nth is in range"),
        ));
    }

    println!("search-checksum={checksum}");
}
