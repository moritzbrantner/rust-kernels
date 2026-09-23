use octree_kernels::OctreeBroadPhase;
use spatial_kernels::{Aabb, Body, BroadPhase, NaiveBroadPhase};
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

fn scene(n: usize) -> Vec<Body> {
    (0..n)
        .map(|id| {
            let cell = id / 2;
            let center = [
                (cell % 8) as f32 * 4.0,
                ((cell / 8) % 8) as f32 * 4.0,
                (cell / 64) as f32 * 4.0,
            ];
            Body::new(id as u32, Aabb::from_center_half_extents(center, [0.5; 3]))
        })
        .collect()
}

#[test]
fn paired_256_scene_keeps_the_allocation_budget() {
    let bodies = scene(256);
    let tree = OctreeBroadPhase::new(6, 4);
    let expected = NaiveBroadPhase.detect(&bodies);
    ALLOCS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    let actual = tree.detect(&bodies);
    ENABLED.with(|enabled| enabled.set(false));
    let calls = ALLOCS.with(Cell::get);
    assert_eq!(actual.pairs, expected.pairs);
    assert!(calls <= 239, "allocation calls={calls}");
}
