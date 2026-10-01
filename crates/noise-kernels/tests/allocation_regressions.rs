use noise_kernels::{Permutation, simplex2, simplex3, simplex4};
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
