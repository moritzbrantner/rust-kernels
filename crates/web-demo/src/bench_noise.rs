//! Batched noise exports for the browser benchmark page.
//!
//! Each call samples a whole workload so per-call JS/WASM overhead stays out of
//! steady-state throughput. A permutation is built once per seed through
//! [`noise_permutation_new`] and passed back as an opaque handle. Worley exports
//! write `f1` only, matching the scalar the JavaScript competitor returns; the
//! kernel still computes its full exact sample.

use noise_kernels::{Permutation, perlin2, perlin3, simplex2, simplex3, worley2, worley3};

use crate::bench::{input, output};

/// Builds a seeded permutation and returns an owned handle.
#[unsafe(no_mangle)]
pub extern "C" fn noise_permutation_new(seed: u32) -> *mut Permutation {
    Box::into_raw(Box::new(Permutation::from_seed(u64::from(seed))))
}

/// Releases a handle from [`noise_permutation_new`].
///
/// # Safety
/// `permutation` must come from `noise_permutation_new` and not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn noise_permutation_free(permutation: *mut Permutation) {
    if !permutation.is_null() {
        // SAFETY: guaranteed by the caller contract above.
        drop(unsafe { Box::from_raw(permutation) });
    }
}

fn points2(
    permutation: &Permutation,
    xy: &[[f64; 2]],
    out: &mut [f64],
    sample: impl Fn(&Permutation, f64, f64) -> f64,
) {
    for (point, value) in xy.iter().zip(out.iter_mut()) {
        *value = sample(permutation, point[0], point[1]);
    }
}

fn points3(
    permutation: &Permutation,
    xyz: &[[f64; 3]],
    out: &mut [f64],
    sample: impl Fn(&Permutation, f64, f64, f64) -> f64,
) {
    for (point, value) in xyz.iter().zip(out.iter_mut()) {
        *value = sample(permutation, point[0], point[1], point[2]);
    }
}

/// Row-major `width × height` grid at `origin + index * step`.
fn grid2(
    permutation: &Permutation,
    origin: [f64; 2],
    step: f64,
    width: usize,
    out: &mut [f64],
    sample: impl Fn(&Permutation, f64, f64) -> f64,
) {
    if width == 0 {
        return;
    }
    for (row, values) in out.chunks_exact_mut(width).enumerate() {
        let y = origin[1] + row as f64 * step;
        for (column, value) in values.iter_mut().enumerate() {
            *value = sample(permutation, origin[0] + column as f64 * step, y);
        }
    }
}

/// `width × height × depth` volume, x fastest then y then z.
#[allow(clippy::too_many_arguments)]
fn grid3(
    permutation: &Permutation,
    origin: [f64; 3],
    step: f64,
    width: usize,
    height: usize,
    out: &mut [f64],
    sample: impl Fn(&Permutation, f64, f64, f64) -> f64,
) {
    if width == 0 || height == 0 {
        return;
    }
    for (layer, slab) in out.chunks_exact_mut(width * height).enumerate() {
        let z = origin[2] + layer as f64 * step;
        for (row, values) in slab.chunks_exact_mut(width).enumerate() {
            let y = origin[1] + row as f64 * step;
            for (column, value) in values.iter_mut().enumerate() {
                *value = sample(permutation, origin[0] + column as f64 * step, y, z);
            }
        }
    }
}

fn worley2_f1(permutation: &Permutation, x: f64, y: f64) -> f64 {
    worley2(permutation, x, y).f1
}

fn worley3_f1(permutation: &Permutation, x: f64, y: f64, z: f64) -> f64 {
    worley3(permutation, x, y, z).f1
}

macro_rules! points_export {
    ($name:ident, $helper:ident, $dims:literal, $sample:path) => {
        /// Samples `count` scattered points read from `coordinates` into `out`.
        ///
        /// # Safety
        /// `permutation` must be a live handle; `coordinates` must hold
        #[doc = concat!("`", stringify!($dims), " * count` f64 values and `out` `count` f64 slots.")]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            permutation: *const Permutation,
            coordinates: *const f64,
            count: u32,
            out: *mut f64,
        ) {
            let count = count as usize;
            // SAFETY: guaranteed by the caller contract above.
            let (permutation, coordinates, out) = unsafe {
                (
                    &*permutation,
                    input(coordinates.cast::<[f64; $dims]>(), count),
                    output(out, count),
                )
            };
            $helper(permutation, coordinates, out, $sample);
        }
    };
}

points_export!(noise_perlin2_points, points2, 2, perlin2);
points_export!(noise_simplex2_points, points2, 2, simplex2);
points_export!(noise_worley2_points, points2, 2, worley2_f1);
points_export!(noise_perlin3_points, points3, 3, perlin3);
points_export!(noise_simplex3_points, points3, 3, simplex3);
points_export!(noise_worley3_points, points3, 3, worley3_f1);

macro_rules! grid2_export {
    ($name:ident, $sample:path) => {
        /// Samples a row-major `width × height` grid at `(x0 + i * step, y0 + j * step)`.
        ///
        /// # Safety
        /// `permutation` must be a live handle and `out` must hold `width * height` f64 slots.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            permutation: *const Permutation,
            x0: f64,
            y0: f64,
            step: f64,
            width: u32,
            height: u32,
            out: *mut f64,
        ) {
            let width = width as usize;
            // SAFETY: guaranteed by the caller contract above.
            let (permutation, out) =
                unsafe { (&*permutation, output(out, width * height as usize)) };
            grid2(permutation, [x0, y0], step, width, out, $sample);
        }
    };
}

grid2_export!(noise_perlin2_grid, perlin2);
grid2_export!(noise_simplex2_grid, simplex2);
grid2_export!(noise_worley2_grid, worley2_f1);

macro_rules! grid3_export {
    ($name:ident, $sample:path) => {
        /// Samples a `width × height × depth` volume (x fastest) at `origin + index * step`.
        ///
        /// # Safety
        /// `permutation` must be a live handle and `out` must hold
        /// `width * height * depth` f64 slots.
        #[allow(clippy::too_many_arguments)]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            permutation: *const Permutation,
            x0: f64,
            y0: f64,
            z0: f64,
            step: f64,
            width: u32,
            height: u32,
            depth: u32,
            out: *mut f64,
        ) {
            let (width, height) = (width as usize, height as usize);
            // SAFETY: guaranteed by the caller contract above.
            let (permutation, out) =
                unsafe { (&*permutation, output(out, width * height * depth as usize)) };
            grid3(permutation, [x0, y0, z0], step, width, height, out, $sample);
        }
    };
}

grid3_export!(noise_perlin3_grid, perlin3);
grid3_export!(noise_simplex3_grid, simplex3);
grid3_export!(noise_worley3_grid, worley3_f1);

#[cfg(test)]
mod tests {
    use noise_kernels::{Permutation, perlin2, simplex3, worley2, worley3};

    use super::{
        noise_perlin2_grid, noise_perlin2_points, noise_permutation_free, noise_permutation_new,
        noise_simplex3_grid, noise_simplex3_points, noise_worley2_points, noise_worley3_grid,
    };

    #[test]
    fn point_batches_delegate_to_the_noise_kernels() {
        let handle = noise_permutation_new(7);
        let reference = Permutation::from_seed(7);
        let xy = [0.25, -1.5, 3.75, 9.125, -40.0, 0.5];
        let xyz = [0.25, -1.5, 2.0, 3.75, 9.125, -0.5];
        let mut out2 = [0.0; 3];
        let mut out3 = [0.0; 2];
        // SAFETY: buffers match the declared counts and the handle is live.
        unsafe {
            noise_perlin2_points(handle, xy.as_ptr(), 3, out2.as_mut_ptr());
            for (index, value) in out2.iter().enumerate() {
                assert_eq!(
                    *value,
                    perlin2(&reference, xy[2 * index], xy[2 * index + 1])
                );
            }
            noise_worley2_points(handle, xy.as_ptr(), 3, out2.as_mut_ptr());
            assert_eq!(out2[1], worley2(&reference, 3.75, 9.125).f1);
            noise_simplex3_points(handle, xyz.as_ptr(), 2, out3.as_mut_ptr());
            assert_eq!(out3[1], simplex3(&reference, 3.75, 9.125, -0.5));
            noise_permutation_free(handle);
        }
    }

    #[test]
    fn grids_walk_x_fastest_from_the_origin() {
        let handle = noise_permutation_new(11);
        let reference = Permutation::from_seed(11);
        let mut grid = [0.0; 6];
        let mut volume = [0.0; 8];
        // SAFETY: buffers match the declared dimensions and the handle is live.
        unsafe {
            noise_perlin2_grid(handle, -1.0, 2.0, 0.5, 3, 2, grid.as_mut_ptr());
            assert_eq!(grid[5], perlin2(&reference, 0.0, 2.5));
            noise_simplex3_grid(handle, 0.0, 0.0, 0.0, 0.25, 2, 2, 2, volume.as_mut_ptr());
            assert_eq!(volume[6], simplex3(&reference, 0.0, 0.25, 0.25));
            noise_worley3_grid(handle, 0.0, 0.0, 0.0, 0.25, 2, 2, 2, volume.as_mut_ptr());
            assert_eq!(volume[1], worley3(&reference, 0.25, 0.0, 0.0).f1);
            noise_permutation_free(handle);
            noise_permutation_free(std::ptr::null_mut());
        }
    }
}
