//! Box (convex polytope) overlap against heightfield surfaces by the separating-axis test,
//! checked against closed forms on flat terrain and a sampled reference on hills.
use geometry_kernels::heightfield::{
    ConvexPolytope3, HeightfieldData3, HeightfieldDiagonal3, HeightfieldWork3, PreparedHeightfield3,
};

fn field(height: impl Fn(f64, f64) -> f64) -> PreparedHeightfield3 {
    let n = 17;
    let columns: Vec<f64> = (0..n).map(|i| i as f64 * 2.0 - 16.0).collect();
    let rows = columns.clone();
    let heights = (0..n * n)
        .map(|k| height(columns[k % n], rows[k / n]))
        .collect();
    PreparedHeightfield3::try_new(HeightfieldData3 {
        columns,
        rows,
        heights,
        active_cells: None,
        diagonal: HeightfieldDiagonal3::Fixed,
    })
    .unwrap()
}

/// An oriented box: centre, unit axes, half extents.
struct Cuboid {
    vertices: [[f64; 3]; 8],
    axes: [[f64; 3]; 3],
}

fn cuboid(centre: [f64; 3], yaw: f64, roll: f64, half: [f64; 3]) -> Cuboid {
    // Roll about Z, then yaw about Y.
    let (sr, cr) = roll.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    let rotate = |v: [f64; 3]| {
        let r = [v[0] * cr - v[1] * sr, v[0] * sr + v[1] * cr, v[2]];
        [r[0] * cy + r[2] * sy, r[1], -r[0] * sy + r[2] * cy]
    };
    let axes = [
        rotate([1.0, 0.0, 0.0]),
        rotate([0.0, 1.0, 0.0]),
        rotate([0.0, 0.0, 1.0]),
    ];
    let mut vertices = [[0.0; 3]; 8];
    for (k, v) in vertices.iter_mut().enumerate() {
        let s = [
            if k & 1 == 0 { -1.0 } else { 1.0 },
            if k & 2 == 0 { -1.0 } else { 1.0 },
            if k & 4 == 0 { -1.0 } else { 1.0 },
        ];
        *v = std::array::from_fn(|i| {
            centre[i] + (0..3).map(|a| axes[a][i] * s[a] * half[a]).sum::<f64>()
        });
    }
    Cuboid { vertices, axes }
}

fn contact(field: &PreparedHeightfield3, c: &Cuboid) -> Option<f64> {
    field
        .polytope_contact_counted(
            ConvexPolytope3 {
                vertices: &c.vertices,
                face_normals: &c.axes,
                edges: &c.axes,
            },
            &mut HeightfieldWork3::default(),
        )
        .map(|hit| hit.separation)
}

#[test]
fn boxes_on_flat_terrain_overlap_by_their_deepest_vertex() {
    let flat = field(|_, _| 0.0);
    // Axis aligned, 0.25 into the ground: depth 0.25.
    let sunk = cuboid([0.3, 0.75, -0.7], 0.0, 0.0, [1.0, 1.0, 1.0]);
    assert!((contact(&flat, &sunk).unwrap() + 0.25).abs() < 1e-12);
    // Touching counts; just above does not.
    assert_eq!(
        contact(&flat, &cuboid([0.3, 1.0, 0.0], 0.0, 0.0, [1.0; 3])),
        Some(0.0)
    );
    assert_eq!(
        contact(&flat, &cuboid([0.3, 1.001, 0.0], 0.0, 0.0, [1.0; 3])),
        None
    );
    // Rolled 45°: the lowest edge is √2 below the centre.
    let rolled = cuboid([0.0, 1.3, 0.0], 0.4, std::f64::consts::FRAC_PI_4, [1.0; 3]);
    let depth = std::f64::consts::SQRT_2 - 1.3;
    assert!((contact(&flat, &rolled).unwrap() + depth).abs() < 1e-9);
}

#[test]
fn boxes_on_hills_agree_with_a_sampled_reference() {
    let height = |x: f64, z: f64| 1.5 * (x * 0.3).sin() * (z * 0.25).cos() + 0.05 * x;
    let hills = field(height);
    let mut seed = 0x1234_5678_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut overlapping, mut separated) = (0, 0);
    for _ in 0..400 {
        let (x, z) = (next() * 12.0, next() * 12.0);
        let half = [
            0.3 + 0.5 * next().abs(),
            0.2 + 0.4 * next().abs(),
            0.3 + 0.5 * next().abs(),
        ];
        let centre = [x, height(x, z) + next() * 1.5, z];
        let c = cuboid(centre, next() * 3.0, next() * 0.8, half);
        // Reference: lowest vertex height above the surface under it, and the highest
        // surface sample inside the box footprint against the box (dense sampling).
        let heights = c.vertices.map(|v| v[1] - surface(&hills, v[0], v[2]));
        let below = heights.iter().copied().fold(f64::INFINITY, f64::min);
        let above = heights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let found = contact(&hills, &c);
        // The terrain is a surface: a box straddling it crosses it; one wholly beneath it does
        // not touch it (as for parry heightfields).
        if below < -0.05 && above > 0.05 {
            overlapping += 1;
            assert!(
                found.is_some(),
                "a vertex is {below} below the surface: {centre:?}"
            );
        } else if below > 0.05 && clear_above(&c, &hills) {
            separated += 1;
            assert!(
                found.is_none(),
                "clear by {below} but overlapping: {centre:?}"
            );
        }
    }
    assert!(
        overlapping > 50 && separated > 50,
        "{overlapping} / {separated}"
    );
}

/// The triangulated surface height at `(x, z)`.
fn surface(hills: &PreparedHeightfield3, x: f64, z: f64) -> f64 {
    hills.sample(x, z).unwrap().surface.unwrap().point[1]
}

/// Whether every surface point under the box footprint lies below the box's bottom face
/// (sampled), so the box and the terrain cannot touch.
fn clear_above(c: &Cuboid, hills: &PreparedHeightfield3) -> bool {
    let (lo, hi) = c.vertices.iter().fold(
        ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
        |(lo, hi), v| {
            (
                [lo[0].min(v[0]), lo[1].min(v[2])],
                [hi[0].max(v[0]), hi[1].max(v[2])],
            )
        },
    );
    let floor = c
        .vertices
        .iter()
        .map(|v| v[1])
        .fold(f64::INFINITY, f64::min);
    (0..=20).all(|i| {
        (0..=20).all(|j| {
            let x = lo[0] + (hi[0] - lo[0]) * i as f64 / 20.0;
            let z = lo[1] + (hi[1] - lo[1]) * j as f64 / 20.0;
            surface(hills, x, z) < floor - 0.05
        })
    })
}
