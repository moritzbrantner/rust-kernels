use noise_kernels::{Permutation, simplex2, simplex3, simplex4};

const F2: f64 = 0.366_025_403_784_438_65;
const G2: f64 = 0.211_324_865_405_187_13;
const F3: f64 = 1.0 / 3.0;
const G3: f64 = 1.0 / 6.0;
const F4: f64 = 0.309_016_994_374_947_45;
const G4: f64 = 0.138_196_601_125_010_5;

const GRADIENTS_3D: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

const GRADIENTS_4D: [[f64; 4]; 32] = [
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, -1.0],
    [0.0, 1.0, -1.0, 1.0],
    [0.0, 1.0, -1.0, -1.0],
    [0.0, -1.0, 1.0, 1.0],
    [0.0, -1.0, 1.0, -1.0],
    [0.0, -1.0, -1.0, 1.0],
    [0.0, -1.0, -1.0, -1.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, -1.0],
    [1.0, 0.0, -1.0, 1.0],
    [1.0, 0.0, -1.0, -1.0],
    [-1.0, 0.0, 1.0, 1.0],
    [-1.0, 0.0, 1.0, -1.0],
    [-1.0, 0.0, -1.0, 1.0],
    [-1.0, 0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, -1.0],
    [1.0, -1.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, -1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, -1.0],
    [-1.0, -1.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0, -1.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, -1.0, 0.0],
    [1.0, -1.0, 1.0, 0.0],
    [1.0, -1.0, -1.0, 0.0],
    [-1.0, 1.0, 1.0, 0.0],
    [-1.0, 1.0, -1.0, 0.0],
    [-1.0, -1.0, 1.0, 0.0],
    [-1.0, -1.0, -1.0, 0.0],
];

fn lattice_index(value: f64) -> usize {
    value.rem_euclid(256.0) as usize
}

fn hash2(table: &[u8; 256], x: usize, y: usize) -> u8 {
    table[(x + usize::from(table[y & 255])) & 255]
}

fn hash3(table: &[u8; 256], x: usize, y: usize, z: usize) -> u8 {
    let z_hash = usize::from(table[z & 255]);
    let yz_hash = usize::from(table[(y + z_hash) & 255]);
    table[(x + yz_hash) & 255]
}

fn hash4(table: &[u8; 256], x: usize, y: usize, z: usize, w: usize) -> u8 {
    let w_hash = usize::from(table[w & 255]);
    let zw_hash = usize::from(table[(z + w_hash) & 255]);
    let yzw_hash = usize::from(table[(y + zw_hash) & 255]);
    table[(x + yzw_hash) & 255]
}

fn contribution2(hash: u8, x: f64, y: f64) -> f64 {
    let attenuation = 0.5 - x * x - y * y;
    if attenuation <= 0.0 {
        return 0.0;
    }

    let squared = attenuation * attenuation;
    let gradient = GRADIENTS_3D[usize::from(hash) % GRADIENTS_3D.len()];
    squared * squared * (gradient[0] * x + gradient[1] * y)
}

fn contribution3(hash: u8, x: f64, y: f64, z: f64) -> f64 {
    let attenuation = 0.6 - x * x - y * y - z * z;
    if attenuation <= 0.0 {
        return 0.0;
    }

    let squared = attenuation * attenuation;
    let gradient = GRADIENTS_3D[usize::from(hash) % GRADIENTS_3D.len()];
    squared * squared * (gradient[0] * x + gradient[1] * y + gradient[2] * z)
}

fn contribution4(hash: u8, point: [f64; 4]) -> f64 {
    let attenuation =
        0.6 - point[0] * point[0] - point[1] * point[1] - point[2] * point[2] - point[3] * point[3];
    if attenuation <= 0.0 {
        return 0.0;
    }

    let squared = attenuation * attenuation;
    let gradient = GRADIENTS_4D[usize::from(hash) % GRADIENTS_4D.len()];
    squared
        * squared
        * (gradient[0] * point[0]
            + gradient[1] * point[1]
            + gradient[2] * point[2]
            + gradient[3] * point[3])
}

fn reference2(permutation: &Permutation, x: f64, y: f64) -> f64 {
    let skew = (x + y) * F2;
    let cell = [(x + skew).floor(), (y + skew).floor()];
    let unskew = (cell[0] + cell[1]) * G2;
    let origin = [x - cell[0] + unskew, y - cell[1] + unskew];

    let first_axis = if origin[0] > origin[1] { 0 } else { 1 };
    let mut middle = [0_usize; 2];
    middle[first_axis] = 1;

    let corners = [[0, 0], middle, [1, 1]];
    let table = permutation.table();
    let base = [lattice_index(cell[0]), lattice_index(cell[1])];
    let mut sum = 0.0;

    for (index, corner) in corners.into_iter().enumerate() {
        let offset = [
            origin[0] - corner[0] as f64 + index as f64 * G2,
            origin[1] - corner[1] as f64 + index as f64 * G2,
        ];
        sum += contribution2(
            hash2(table, base[0] + corner[0], base[1] + corner[1]),
            offset[0],
            offset[1],
        );
    }

    70.0 * sum
}

fn reference3(permutation: &Permutation, x: f64, y: f64, z: f64) -> f64 {
    let skew = (x + y + z) * F3;
    let cell = [(x + skew).floor(), (y + skew).floor(), (z + skew).floor()];
    let unskew = (cell[0] + cell[1] + cell[2]) * G3;
    let origin = [
        x - cell[0] + unskew,
        y - cell[1] + unskew,
        z - cell[2] + unskew,
    ];

    let mut axes = [(origin[0], 0_usize), (origin[1], 1), (origin[2], 2)];
    axes.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    let mut first = [0_usize; 3];
    first[axes[0].1] = 1;
    let mut second = first;
    second[axes[1].1] = 1;
    let corners = [[0, 0, 0], first, second, [1, 1, 1]];

    let table = permutation.table();
    let base = [
        lattice_index(cell[0]),
        lattice_index(cell[1]),
        lattice_index(cell[2]),
    ];
    let mut sum = 0.0;

    for (index, corner) in corners.into_iter().enumerate() {
        let offset = [
            origin[0] - corner[0] as f64 + index as f64 * G3,
            origin[1] - corner[1] as f64 + index as f64 * G3,
            origin[2] - corner[2] as f64 + index as f64 * G3,
        ];
        sum += contribution3(
            hash3(
                table,
                base[0] + corner[0],
                base[1] + corner[1],
                base[2] + corner[2],
            ),
            offset[0],
            offset[1],
            offset[2],
        );
    }

    32.0 * sum
}

fn reference4(permutation: &Permutation, x: f64, y: f64, z: f64, w: f64) -> f64 {
    let skew = (x + y + z + w) * F4;
    let cell = [
        (x + skew).floor(),
        (y + skew).floor(),
        (z + skew).floor(),
        (w + skew).floor(),
    ];
    let unskew = (cell[0] + cell[1] + cell[2] + cell[3]) * G4;
    let origin = [
        x - cell[0] + unskew,
        y - cell[1] + unskew,
        z - cell[2] + unskew,
        w - cell[3] + unskew,
    ];

    let mut axes = [
        (origin[0], 0_usize),
        (origin[1], 1),
        (origin[2], 2),
        (origin[3], 3),
    ];
    axes.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| right.1.cmp(&left.1))
    });

    let mut first = [0_usize; 4];
    first[axes[0].1] = 1;
    let mut second = first;
    second[axes[1].1] = 1;
    let mut third = second;
    third[axes[2].1] = 1;
    let corners = [[0, 0, 0, 0], first, second, third, [1, 1, 1, 1]];

    let table = permutation.table();
    let base = [
        lattice_index(cell[0]),
        lattice_index(cell[1]),
        lattice_index(cell[2]),
        lattice_index(cell[3]),
    ];
    let mut sum = 0.0;

    for (index, corner) in corners.into_iter().enumerate() {
        let offset = [
            origin[0] - corner[0] as f64 + index as f64 * G4,
            origin[1] - corner[1] as f64 + index as f64 * G4,
            origin[2] - corner[2] as f64 + index as f64 * G4,
            origin[3] - corner[3] as f64 + index as f64 * G4,
        ];
        sum += contribution4(
            hash4(
                table,
                base[0] + corner[0],
                base[1] + corner[1],
                base[2] + corner[2],
                base[3] + corner[3],
            ),
            offset,
        );
    }

    27.0 * sum
}

fn next_coordinate(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let signed = ((*state >> 16) & 0x00ff_ffff) as i64 - 0x007f_ffff;
    signed as f64 / 8192.0
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn optimized_simplex_matches_generic_corner_order_oracle() {
    for seed in [0, 1, 42, u64::MAX, 0x1234_5678_9abc_def0] {
        let permutation = Permutation::from_seed(seed);
        let mut state = seed ^ 0xa076_1d64_78bd_642f;

        for _ in 0..512 {
            let x = next_coordinate(&mut state);
            let y = next_coordinate(&mut state);
            let z = next_coordinate(&mut state);
            let w = next_coordinate(&mut state);

            assert_close(simplex2(&permutation, x, y), reference2(&permutation, x, y));
            assert_close(
                simplex3(&permutation, x, y, z),
                reference3(&permutation, x, y, z),
            );
            assert_close(
                simplex4(&permutation, x, y, z, w),
                reference4(&permutation, x, y, z, w),
            );
        }
    }
}
