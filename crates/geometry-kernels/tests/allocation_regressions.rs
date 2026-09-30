use geometry_kernels::{
    epa::epa_penetration_planar_xy, planar::gjk_intersection_planar_xy, support::ConvexHull3,
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
    REALLOCS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    let result = epa_penetration_planar_xy(&left, &right, &gjk);
    ENABLED.with(|enabled| enabled.set(false));
    let allocations = ALLOCS.with(Cell::get);
    assert!(result.penetration.is_some());
    assert!(allocations <= 4, "allocation calls={allocations}");
}

#[test]
fn primitive_volume_queries_allocate_nothing() {
    use geometry_kernels::primitive3::{PrimitiveShape3, try_volume_properties};
    let shapes = [
        PrimitiveShape3::sphere(1.0),
        PrimitiveShape3::cuboid([1.0, 2.0, 3.0]),
        PrimitiveShape3::capsule(1000.0, 0.01),
        PrimitiveShape3::wedge([1.0, 2.0, 3.0]),
    ];
    ALLOCS.with(|count| count.set(0));
    REALLOCS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    let mut valid = true;
    for _ in 0..1024 {
        for shape in shapes {
            valid &=
                std::hint::black_box(try_volume_properties(std::hint::black_box(shape))).is_ok();
        }
    }
    ENABLED.with(|enabled| enabled.set(false));
    assert!(valid);
    assert_eq!(ALLOCS.with(Cell::get), 0);
    assert_eq!(REALLOCS.with(Cell::get), 0);
}

#[test]
fn capsule_skeleton_contact_queries_allocate_nothing() {
    use geometry_kernels::primitive3::{PrimitiveBody3, PrimitiveShape3, PrimitiveWork3, query};
    let capsule =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::capsule(5.0, 0.5), [0.0; 3], [0.0; 3]);
    let sphere =
        PrimitiveBody3::axis_aligned(PrimitiveShape3::sphere(0.25), [0.0, 2.0, 0.0], [0.0; 3]);
    let crossing = PrimitiveBody3 {
        axes: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ..capsule
    };
    let mut work = PrimitiveWork3::default();
    ALLOCS.with(|count| count.set(0));
    REALLOCS.with(|count| count.set(0));
    ENABLED.with(|enabled| enabled.set(true));
    for _ in 0..2048 {
        for target in [sphere, crossing] {
            std::hint::black_box(query(
                std::hint::black_box(capsule),
                std::hint::black_box(target),
                &mut work,
            ));
        }
    }
    ENABLED.with(|enabled| enabled.set(false));
    assert_eq!(ALLOCS.with(Cell::get), 0);
    assert_eq!(REALLOCS.with(Cell::get), 0);
    assert_eq!(work.pair_dispatches.iter().sum::<u64>(), 4096);
    assert_eq!(work.support_evaluations, 0);
    assert_eq!(work.axes_tested, 0);
    assert_eq!(work.vertex_tests, 0);
    assert_eq!(work.sweep_iterations, 0);
}
