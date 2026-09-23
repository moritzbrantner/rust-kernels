//! Conservative enclosure regressions; NaiveBroadPhase is the pair oracle.
use octree_kernels::OctreeBroadPhase;
use spatial_kernels::{Aabb, Body, BroadPhase, NaiveBroadPhase};

#[test]
fn boundary_octree_keeps_tiny_positive_colliders() {
    let point = |x| Aabb::new([x, 0.0, 0.0], [x, 0.0, 0.0]);
    let bodies = [
        Body::new(0, point(-3668.735)),
        Body::new(1, point(1.05e-10)),
        Body::new(2, point(1.05e-10)),
    ];
    let tree = OctreeBroadPhase::new(6, 1);
    let expected = NaiveBroadPhase.detect(&bodies);
    assert_eq!(expected.pairs.len(), 1);
    assert_eq!(tree.detect(&bodies).pairs, expected.pairs);
    let trace = tree.trace(&bodies);
    let root = trace.nodes[trace.root.unwrap()].bounds;
    assert!(bodies.iter().all(|body| root.contains(body.aabb)));
}

#[test]
fn boundary_octree_supports_finite_extreme_bounds() {
    for bounds in [
        Aabb::new([2e38, 0.0, 0.0], [3e38, 1.0, 1.0]),
        Aabb::new([-f32::MAX; 3], [f32::MAX; 3]),
    ] {
        let bodies = [Body::new(0, bounds), Body::new(1, bounds)];
        let actual = OctreeBroadPhase::new(4, 1).trace(&bodies);
        assert_eq!(actual.result.pairs, NaiveBroadPhase.detect(&bodies).pairs);
        let root = actual.nodes[actual.root.unwrap()].bounds;
        assert!(root.contains(bounds));
    }
}

#[test]
fn boundary_octree_ordinary_control() {
    let bodies = [
        Body::new(0, Aabb::new([0.0; 3], [1.0; 3])),
        Body::new(1, Aabb::new([0.5; 3], [1.5; 3])),
        Body::new(2, Aabb::new([3.0; 3], [4.0; 3])),
    ];
    assert_eq!(
        OctreeBroadPhase::new(4, 1).detect(&bodies).pairs,
        NaiveBroadPhase.detect(&bodies).pairs
    );
}

fn assert_conservative_scene(bodies: &[Body]) {
    let tree = OctreeBroadPhase::new(3, 2);
    let trace = tree.trace(bodies);
    assert_eq!(trace.result.pairs, NaiveBroadPhase.detect(bodies).pairs);
    assert_eq!(trace.result, tree.detect(bodies));
    let root = trace.nodes[trace.root.unwrap()].bounds;
    assert!(
        bodies.iter().all(|b| root.contains(b.aabb)),
        "root={root:?} bodies={bodies:?}"
    );
    for node in &trace.nodes {
        assert!(
            node.bounds
                .min
                .iter()
                .chain(&node.bounds.max)
                .all(|v| v.is_finite())
        );
        if node.children.is_empty() {
            for body in bodies {
                assert_eq!(
                    node.members.contains(&body.id),
                    node.bounds.overlaps(body.aabb)
                );
            }
        } else {
            assert_eq!(node.children.len(), 8);
            for &child in &node.children {
                assert!(node.bounds.contains(trace.nodes[child].bounds));
            }
        }
    }
    let mut reversed = bodies.to_vec();
    reversed.reverse();
    assert_eq!(tree.trace(&reversed), trace);
}

#[test]
fn root_and_descendants_enclose_generated_finite_scenes() {
    let mut state = 0x51ced123_u32;
    for _ in 0..96 {
        let mut number = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Preserve arbitrary signs/mantissas, constrain exponent to finite.
            f32::from_bits((state & 0x807fffff) | (((state >> 24) % 255) << 23))
        };
        let mut bodies = Vec::new();
        for id in 0..10 {
            let a: [f32; 3] = std::array::from_fn(|_| number());
            let b: [f32; 3] = std::array::from_fn(|_| number());
            let bounds = Aabb::new(
                std::array::from_fn(|i| a[i].min(b[i])),
                std::array::from_fn(|i| a[i].max(b[i])),
            );
            bodies.push(Body::new(id, bounds));
        }
        assert_conservative_scene(&bodies);
    }
}

#[test]
fn conservative_enclosure_handles_ulp_subnormal_and_sign_boundaries() {
    for x in [
        0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        1.0,
        -1.0,
        2e38,
        -2e38,
    ] {
        let next = f32::from_bits(x.to_bits() + 1);
        let bounds = Aabb::new([x.min(next); 3], [x.max(next); 3]);
        let bodies = [
            Body::new(0, bounds),
            Body::new(1, bounds),
            Body::new(2, Aabb::new([-10.0; 3], [-9.0; 3])),
        ];
        assert_conservative_scene(&bodies);
    }
}
