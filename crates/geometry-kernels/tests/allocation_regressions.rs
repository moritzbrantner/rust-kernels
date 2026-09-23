use geometry_kernels::{
    epa::epa_penetration_planar_xy, planar::gjk_intersection_planar_xy, support::ConvexHull3,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct CountingAllocator;
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
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
                ALLOCS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn no_trace_quad_epa_uses_at_most_four_allocations() {
    let left_points = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let right_points = [
        [0.25, -0.8, 0.0],
        [1.75, -0.8, 0.0],
        [1.75, 0.8, 0.0],
        [0.25, 0.8, 0.0],
    ];
    let left = ConvexHull3::new(&left_points);
    let right = ConvexHull3::new(&right_points);
    let gjk = gjk_intersection_planar_xy(&left, &right);
    ALLOCS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    let result = epa_penetration_planar_xy(&left, &right, &gjk);
    ENABLED.with(|enabled| enabled.set(false));
    let calls = ALLOCS.with(Cell::get);
    assert!(result.penetration.is_some());
    assert!(calls <= 4, "allocation calls={calls}");
}
