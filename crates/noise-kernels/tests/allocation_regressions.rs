use noise_kernels::{
    Curl, DomainWarp, Fractal, Permutation, simplex2, simplex3, simplex4, worley2, worley3,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct CountingAllocator;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static REALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ENABLED.with(|enabled| {
            if enabled.get() {
                ALLOCS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ENABLED.with(|enabled| {
            if enabled.get() {
                REALLOCS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocation_counts<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCS.with(|count| count.set(0));
    REALLOCS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    let result = f();
    ENABLED.with(|enabled| enabled.set(false));
    (result, ALLOCS.with(Cell::get), REALLOCS.with(Cell::get))
}

#[test]
fn repeated_simplex_sampling_allocates_nothing() {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);

    let (sum, allocations, reallocations) = allocation_counts(|| {
        let mut sum = 0.0;
        for index in 0..1024 {
            let coordinate = f64::from(index) * 0.03125;
            sum += simplex2(&permutation, coordinate, coordinate * -0.5);
            sum += simplex3(
                &permutation,
                coordinate,
                coordinate * -0.5,
                coordinate * 0.25,
            );
            sum += simplex4(
                &permutation,
                coordinate,
                coordinate * -0.5,
                coordinate * 0.25,
                3.5,
            );
        }
        sum
    });

    assert!(sum.is_finite());
    assert_eq!((allocations, reallocations), (0, 0));
}

#[test]
fn repeated_worley_sampling_allocates_nothing() {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);

    let (sum, allocations, reallocations) = allocation_counts(|| {
        let mut sum = 0.0;
        for index in 0..1024 {
            let coordinate = f64::from(index) * 0.03125;
            // Integer-adjacent inputs force the exact ring expansion path too.
            let corner = f64::from(index) + 1e-6;
            let sample = worley2(&permutation, coordinate, coordinate * -0.5);
            sum += sample.f1 + sample.f2;
            sum += worley2(&permutation, corner, -corner).f2;
            let sample = worley3(
                &permutation,
                coordinate,
                coordinate * -0.5,
                coordinate * 0.25,
            );
            sum += sample.f1 + sample.f2;
            sum += worley3(&permutation, corner, -corner, corner).f2;
        }
        sum
    });

    assert!(sum.is_finite());
    assert_eq!((allocations, reallocations), (0, 0));
}

#[test]
fn repeated_combinator_sampling_allocates_nothing() {
    let permutation = Permutation::from_seed(0x1020_3040_5060_7080);
    let fractal = Fractal::new(6, 2.0, 0.5).expect("valid schedule");
    let warp = DomainWarp::new(1.5).expect("valid amplitude");
    let curl = Curl::new(1e-3).expect("valid epsilon");
    let base2 = |x, y| simplex2(&permutation, x, y);
    let base3 = |x, y, z| simplex3(&permutation, x, y, z);
    let cells3 = |x, y, z| worley3(&permutation, x, y, z).f1;

    let (sum, allocations, reallocations) = allocation_counts(|| {
        let mut sum = 0.0;
        for index in 0..1024 {
            let x = f64::from(index) * 0.03125;
            let y = x * -0.5;
            let z = x * 0.25;
            sum += fractal.fbm2(&base2, x, y);
            sum += fractal.billow3(&base3, x, y, z);
            sum += fractal.ridged3(&cells3, x, y, z);
            sum += warp.sample2(&base2, &base2, &|u, v| base2(v, u), x, y);
            sum += warp.sample3(&cells3, &base3, &base3, &base3, x, y, z);
            sum += curl.curl2(&|u, v| fractal.fbm2(&base2, u, v), x, y)[0];
            sum += curl.curl3(&base3, &cells3, &base3, x, y, z)[2];
        }
        sum
    });

    assert!(sum.is_finite());
    assert_eq!((allocations, reallocations), (0, 0));
}
