use noise_kernels::{Permutation, perlin2, perlin3};

const GRADIENTS: [[f64; 3]; 12] = [
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

fn fade(value: f64) -> f64 {
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

fn lerp(left: f64, right: f64, amount: f64) -> f64 {
    left + amount * (right - left)
}

fn wrapped_floor(value: f64) -> (usize, f64) {
    let floor = value.floor();
    (floor.rem_euclid(256.0) as usize, value - floor)
}

fn hash2(table: &[u8; 256], x: usize, y: usize) -> usize {
    usize::from(table[(x + usize::from(table[y & 255])) & 255])
}

fn hash3(table: &[u8; 256], x: usize, y: usize, z: usize) -> usize {
    let z_hash = usize::from(table[z & 255]);
    let yz_hash = usize::from(table[(y + z_hash) & 255]);
    usize::from(table[(x + yz_hash) & 255])
}

fn reference2(permutation: &Permutation, x: f64, y: f64) -> f64 {
    let table = permutation.table();
    let (cell_x, local_x) = wrapped_floor(x);
    let (cell_y, local_y) = wrapped_floor(y);
    let mut contributions = [[0.0; 2]; 2];

    for (corner_y, row) in contributions.iter_mut().enumerate() {
        for (corner_x, contribution) in row.iter_mut().enumerate() {
            let gradient =
                GRADIENTS[hash2(table, cell_x + corner_x, cell_y + corner_y) % GRADIENTS.len()];
            let offset_x = local_x - corner_x as f64;
            let offset_y = local_y - corner_y as f64;
            *contribution = gradient[0] * offset_x + gradient[1] * offset_y;
        }
    }

    let x0 = lerp(contributions[0][0], contributions[0][1], fade(local_x));
    let x1 = lerp(contributions[1][0], contributions[1][1], fade(local_x));
    lerp(x0, x1, fade(local_y))
}

fn reference3(permutation: &Permutation, x: f64, y: f64, z: f64) -> f64 {
    let table = permutation.table();
    let (cell_x, local_x) = wrapped_floor(x);
    let (cell_y, local_y) = wrapped_floor(y);
    let (cell_z, local_z) = wrapped_floor(z);
    let mut contributions = [[[0.0; 2]; 2]; 2];

    for (corner_z, plane) in contributions.iter_mut().enumerate() {
        for (corner_y, row) in plane.iter_mut().enumerate() {
            for (corner_x, contribution) in row.iter_mut().enumerate() {
                let gradient = GRADIENTS[hash3(
                    table,
                    cell_x + corner_x,
                    cell_y + corner_y,
                    cell_z + corner_z,
                ) % GRADIENTS.len()];
                let offset_x = local_x - corner_x as f64;
                let offset_y = local_y - corner_y as f64;
                let offset_z = local_z - corner_z as f64;
                *contribution =
                    gradient[0] * offset_x + gradient[1] * offset_y + gradient[2] * offset_z;
            }
        }
    }

    let mut z_values = [0.0; 2];
    for (corner_z, value) in z_values.iter_mut().enumerate() {
        let y0 = lerp(
            contributions[corner_z][0][0],
            contributions[corner_z][0][1],
            fade(local_x),
        );
        let y1 = lerp(
            contributions[corner_z][1][0],
            contributions[corner_z][1][1],
            fade(local_x),
        );
        *value = lerp(y0, y1, fade(local_y));
    }

    lerp(z_values[0], z_values[1], fade(local_z))
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
fn optimized_perlin_matches_direct_corner_oracle() {
    for seed in [0, 1, 42, u64::MAX, 0x1234_5678_9abc_def0] {
        let permutation = Permutation::from_seed(seed);
        let mut state = seed ^ 0xa076_1d64_78bd_642f;

        for _ in 0..512 {
            let x = next_coordinate(&mut state);
            let y = next_coordinate(&mut state);
            let z = next_coordinate(&mut state);

            assert_close(perlin2(&permutation, x, y), reference2(&permutation, x, y));
            assert_close(
                perlin3(&permutation, x, y, z),
                reference3(&permutation, x, y, z),
            );
        }
    }
}
