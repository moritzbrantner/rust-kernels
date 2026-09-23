use search_kernels::top_k_by;
use std::{alloc::{GlobalAlloc, Layout, System}, cell::Cell};

struct CountingAllocator;
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ENABLED.with(|enabled| {
            if enabled.get() { ALLOCS.with(|count| count.set(count.get() + 1)); }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) { unsafe { System.dealloc(ptr, layout) } }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ENABLED.with(|enabled| {
            if enabled.get() { ALLOCS.with(|count| count.set(count.get() + 1)); }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOCS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    let result = f();
    ENABLED.with(|enabled| enabled.set(false));
    (result, ALLOCS.with(Cell::get))
}

#[test]
fn empty_oversized_top_k_allocates_nothing() {
    let (result, calls) = allocations(|| top_k_by(std::iter::empty::<u64>(), 65_536, Ord::cmp));
    assert!(result.is_empty());
    assert_eq!(calls, 0);
}

#[test]
fn tiny_result_does_not_reserve_the_requested_limit() {
    let result = top_k_by([3_u64, 1, 2, 0], 65_536, Ord::cmp);
    assert_eq!(result, [0, 1, 2, 3]);
    assert!(result.capacity() <= 8, "retained capacity={}", result.capacity());
}
