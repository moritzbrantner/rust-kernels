use graph_kernels::{breadth_first, depth_first};
use std::hint::black_box;

const NODES: usize = 120_000;
const ROUNDS: usize = 10;

fn neighbors(node: &usize) -> [usize; 3] {
    let next = (*node + 1) % NODES;
    let doubled = node.wrapping_mul(2).wrapping_add(1) % NODES;
    let mixed = node.wrapping_mul(65_537).wrapping_add(17) % NODES;
    [next, doubled, mixed]
}

fn main() {
    let mut checksum = 0_u64;

    for round in 0..ROUNDS {
        let start = (round * 7_919) % NODES;
        let breadth = breadth_first(black_box(start), neighbors);
        checksum = checksum.wrapping_add(breadth.len() as u64);
        checksum ^= breadth[(round * 101) % breadth.len()] as u64;

        let depth = depth_first(black_box(start), neighbors);
        checksum = checksum.wrapping_add(depth.len() as u64);
        checksum ^= depth[(round * 211) % depth.len()] as u64;
    }

    println!("traversal-checksum={checksum}");
}
