use divan::{Bencher, counter::ItemsCount};
use similarity_kernels::rolling_hashes;

const WIDTHS: &[usize] = &[256, 4_096, 65_536];

fn main() {
    divan::main();
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

#[divan::bench(args = WIDTHS, skip_ext_time)]
fn initialize_single_window(bencher: Bencher, width: usize) {
    let bytes = generated_bytes(width);
    bencher
        .counter(ItemsCount::new(width))
        .bench_local(|| divan::black_box(rolling_hashes(&bytes, width).next()));
}
