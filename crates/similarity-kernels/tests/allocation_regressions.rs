use similarity_kernels::LevenshteinWorkspace;
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
fn warmed_bounded_queries_allocate_nothing() {
    let a: Vec<_> = (0..1024).map(|i| (i % 23) as u8).collect();
    let b: Vec<_> = (0..1024).map(|i| ((i + 1) % 23) as u8).collect();
    let mut workspace = LevenshteinWorkspace::new();
    assert_eq!(workspace.distance(&a, &b, 2), Some(2));
    let (distance, allocations, reallocations) =
        allocation_counts(|| workspace.distance(&a, &b, 2));
    assert_eq!(distance, Some(2));
    assert_eq!((allocations, reallocations), (0, 0));
}

#[test]
fn length_rejection_allocates_nothing() {
    let a = vec![0_u8; 1024];
    let b = vec![1_u8; 1032];
    let mut workspace = LevenshteinWorkspace::new();
    let (distance, allocations, reallocations) =
        allocation_counts(|| workspace.distance(&a, &b, 2));
    assert_eq!(distance, None);
    assert_eq!((allocations, reallocations), (0, 0));
}
