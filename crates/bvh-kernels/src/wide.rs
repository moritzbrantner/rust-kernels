//! Four-wide bounding-volume hierarchy for leaves that do not move.
//!
//! Each node holds the bounds of up to four children in structure-of-arrays form, so one node
//! visit is one cache-friendly four-lane slab or overlap test, and a query visits about
//! `log4 n` nodes. Leaf lanes hold the leaves' exact bounds (no fat margin), so a segment that
//! reaches a leaf has already been tested against its exact AABB. The hierarchy is built once
//! with a binned surface-area heuristic; [`WideBvh::remove`] drops a leaf in place and never
//! restructures, so callers rebuild after many removals or new leaves.

use spatial_kernels::{Aabb, Body, ColliderId};

const LANES: usize = 4;
/// Child slot with no child.
const EMPTY: u32 = u32::MAX;
/// High bit of a child slot: the rest indexes `leaves`, not `nodes`.
const LEAF: u32 = 1 << 31;
/// Bounds of an empty lane: every segment entry and overlap test rejects them.
const FAR: f32 = f32::MAX;
/// Splits below this depth use the surface-area heuristic; deeper ones split by count, so the
/// depth stays below `SAH_DEPTH + 16` for any `u32` leaf count.
const SAH_DEPTH: usize = 24;
/// Traversal stack: each visited node replaces itself with at most four children.
const STACK: usize = 3 * (SAH_DEPTH + 16) + 1;
const BINS: usize = 16;

/// Bounds as `[min x, min y, min z, max x, max y, max z]`.
type Bounds = [f32; 6];
const NO_BOUNDS: Bounds = [
    f32::INFINITY,
    f32::INFINITY,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
    f32::NEG_INFINITY,
];

fn grow(acc: &mut Bounds, b: &Bounds) {
    for k in 0..3 {
        acc[k] = acc[k].min(b[k]);
        acc[k + 3] = acc[k + 3].max(b[k + 3]);
    }
}

/// Half the surface area; zero for empty bounds.
fn area(b: &Bounds) -> f32 {
    let (x, y, z) = (b[3] - b[0], b[4] - b[1], b[5] - b[2]);
    if x < 0.0 {
        return 0.0;
    }
    x * y + y * z + z * x
}

/// Leaf bounds and doubled centroids in flat arrays, for the build's binning passes.
struct Builder {
    boxes: Vec<Bounds>,
    centroids: Vec<[f32; 3]>,
    /// Per-axis bin counts and bounds, reused by every split (only the bins in use are reset).
    counts: [[usize; BINS]; 3],
    bins: [[Bounds; BINS]; 3],
}

impl Builder {
    fn bounds(&self, items: &[u32]) -> Bounds {
        let mut acc = NO_BOUNDS;
        for &i in items {
            grow(&mut acc, &self.boxes[i as usize]);
        }
        acc
    }

    /// Reorders `items` (at least two) and returns a split point in `1..items.len()`: the
    /// cheapest binned surface-area split over the three axes, or a count split along the
    /// longest centroid axis when none separates the items or the tree is deep.
    fn split(&mut self, items: &mut [u32], depth: usize) -> usize {
        let n = items.len();
        if n == 2 {
            return 1;
        }
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for &i in items.iter() {
            let c = &self.centroids[i as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        // Small groups use fewer bins: the per-bin sweep below would dominate their cost.
        let nb = n.min(BINS);
        let bin = |c: f32, axis: usize, scale: f32| (((c - lo[axis]) * scale) as usize).min(nb - 1);
        let mut best: Option<(f32, usize, usize, f32)> = None;
        let split_axes = std::array::from_fn::<_, 3, _>(|k| depth < SAH_DEPTH && hi[k] > lo[k]);
        let scales = std::array::from_fn::<_, 3, _>(|k| {
            if split_axes[k] {
                nb as f32 / (hi[k] - lo[k])
            } else {
                0.0
            }
        });
        // One pass bins every axis: each item's bounds are read once.
        let (counts, all_bins) = (&mut self.counts, &mut self.bins);
        for axis in 0..3 {
            counts[axis][..nb].fill(0);
            all_bins[axis][..nb].fill(NO_BOUNDS);
        }
        if split_axes.contains(&true) {
            for &i in items.iter() {
                let (c, b) = (&self.centroids[i as usize], &self.boxes[i as usize]);
                for axis in 0..3 {
                    let k = bin(c[axis], axis, scales[axis]);
                    counts[axis][k] += 1;
                    grow(&mut all_bins[axis][k], b);
                }
            }
        }
        for axis in (0..3).filter(|&k| split_axes[k]) {
            let (scale, counts, bins) = (scales[axis], &counts[axis], &all_bins[axis]);
            let mut right = [0.0_f32; BINS];
            let mut acc = NO_BOUNDS;
            for b in (1..nb).rev() {
                grow(&mut acc, &bins[b]);
                right[b] = area(&acc);
            }
            let (mut left, mut left_count) = (NO_BOUNDS, 0);
            for b in 0..nb - 1 {
                grow(&mut left, &bins[b]);
                left_count += counts[b];
                let right_count = n - left_count;
                if left_count == 0 || right_count == 0 {
                    continue;
                }
                let cost = area(&left) * left_count as f32 + right[b + 1] * right_count as f32;
                if best.is_none_or(|(c, ..)| cost < c) {
                    best = Some((cost, axis, b + 1, scale));
                }
            }
        }
        if let Some((_, axis, split, scale)) = best {
            // Partition by the same bin arithmetic as the binning pass.
            let mut mid = 0;
            for i in 0..n {
                if bin(self.centroids[items[i] as usize][axis], axis, scale) < split {
                    items.swap(i, mid);
                    mid += 1;
                }
            }
            if mid > 0 && mid < n {
                return mid;
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap_or(0);
        let mid = n / 2;
        items.select_nth_unstable_by(mid, |&a, &b| {
            self.centroids[a as usize][axis]
                .total_cmp(&self.centroids[b as usize][axis])
                .then(a.cmp(&b))
        });
        mid
    }
}

#[derive(Clone, Copy, Debug)]
struct Node {
    min: [[f32; LANES]; 3],
    max: [[f32; LANES]; 3],
    child: [u32; LANES],
}

impl Node {
    const EMPTY: Self = Self {
        min: [[FAR; LANES]; 3],
        max: [[FAR; LANES]; 3],
        child: [EMPTY; LANES],
    };

    fn set(&mut self, lane: usize, bounds: Aabb, child: u32) {
        for axis in 0..3 {
            self.min[axis][lane] = bounds.min[axis];
            self.max[axis][lane] = bounds.max[axis];
        }
        self.child[lane] = child;
    }
}

/// Immutable-shape four-wide AABB hierarchy over static leaves.
#[derive(Clone, Debug, Default)]
pub struct WideBvh {
    nodes: Vec<Node>,
    leaves: Vec<Body>,
    /// Node and lane of each leaf, for [`Self::remove`].
    lanes: Vec<(u32, u8)>,
    /// Leaf index per collider ID (`EMPTY` when absent). IDs index this table directly, so
    /// callers should keep them compact, for example slot indices.
    slots: Vec<u32>,
    len: usize,
}

impl WideBvh {
    /// Builds the hierarchy over `bodies`, whose IDs must be unique, in O(n log n).
    #[must_use]
    pub fn build(bodies: &[Body]) -> Self {
        let mut tree = Self {
            nodes: Vec::with_capacity(bodies.len().div_ceil(LANES - 1) + 1),
            leaves: bodies.to_vec(),
            lanes: vec![(0, 0); bodies.len()],
            slots: Vec::new(),
            len: bodies.len(),
        };
        for (index, body) in bodies.iter().enumerate() {
            let id = body.id as usize;
            if tree.slots.len() <= id {
                tree.slots.resize(id + 1, EMPTY);
            }
            assert!(tree.slots[id] == EMPTY, "duplicate collider ID {id}");
            tree.slots[id] = index as u32;
        }
        if let [only] = bodies {
            tree.nodes.push(Node::EMPTY);
            tree.nodes[0].set(0, only.aabb, LEAF);
        } else if !bodies.is_empty() {
            let mut builder = Builder {
                boxes: bodies
                    .iter()
                    .map(|b| {
                        let (lo, hi) = (b.aabb.min, b.aabb.max);
                        [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
                    })
                    .collect(),
                centroids: bodies
                    .iter()
                    .map(|b| std::array::from_fn(|k| b.aabb.min[k] + b.aabb.max[k]))
                    .collect(),
                counts: [[0; BINS]; 3],
                bins: [[NO_BOUNDS; BINS]; 3],
            };
            let mut items: Vec<u32> = (0..bodies.len() as u32).collect();
            tree.build_node(&mut builder, &mut items, 0);
        }
        tree
    }

    /// Builds the node for `items` (at least two) and returns its index.
    fn build_node(&mut self, builder: &mut Builder, items: &mut [u32], depth: usize) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(Node::EMPTY);
        // Split in two, then each half in two again: up to four children.
        let mut groups: [&mut [u32]; LANES] = Default::default();
        let mid = builder.split(items, depth);
        let (left, right) = items.split_at_mut(mid);
        let mut count = 0;
        for half in [left, right] {
            if half.len() > 1 {
                let mid = builder.split(half, depth);
                let (a, b) = half.split_at_mut(mid);
                groups[count] = a;
                groups[count + 1] = b;
                count += 2;
            } else {
                groups[count] = half;
                count += 1;
            }
        }
        for (lane, group) in groups.into_iter().take(count).enumerate() {
            let b = builder.bounds(group);
            let bounds = Aabb {
                min: [b[0], b[1], b[2]],
                max: [b[3], b[4], b[5]],
            };
            let child = if let [leaf] = group[..] {
                self.lanes[leaf as usize] = (index, lane as u8);
                LEAF | leaf
            } else {
                self.build_node(builder, group, depth + 1)
            };
            self.nodes[index as usize].set(lane, bounds, child);
        }
        index
    }

    /// Live leaves.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Nodes in the hierarchy (each holds up to four children).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the leaf `id` is live in the tree.
    #[must_use]
    pub fn contains(&self, id: ColliderId) -> bool {
        self.slots.get(id as usize).is_some_and(|&s| s != EMPTY)
    }

    /// Drops leaf `id` without restructuring; ancestors keep their bounds (still conservative).
    /// Returns the removed leaf.
    pub fn remove(&mut self, id: ColliderId) -> Option<Body> {
        let slot = self.slots.get_mut(id as usize)?;
        if *slot == EMPTY {
            return None;
        }
        let leaf = *slot as usize;
        *slot = EMPTY;
        let (node, lane) = self.lanes[leaf];
        self.nodes[node as usize].set(
            lane as usize,
            Aabb {
                min: [FAR; 3],
                max: [FAR; 3],
            },
            EMPTY,
        );
        self.len -= 1;
        Some(self.leaves[leaf])
    }

    /// Visits every leaf whose exact AABB, padded by `pad`, the segment `origin + t * delta`
    /// enters for `t` in `0..=max_t`, nearer entries first. `visit(leaf, t)` gets the padded
    /// entry parameter and returns the new `max_t` (return the current one to keep searching,
    /// a negative one to stop). Returns the number of nodes and leaves visited.
    pub fn visit_segment(
        &self,
        origin: [f32; 3],
        delta: [f32; 3],
        max_t: f32,
        pad: f32,
        mut visit: impl FnMut(Body, f32) -> f32,
    ) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let inverse = delta.map(|d| if d == 0.0 { 0.0 } else { 1.0 / d });
        let parallel = delta.map(|d| d == 0.0);
        let mut max_t = max_t;
        let mut visited = 0;
        let mut stack = [(0_u32, 0.0_f32); STACK];
        let mut len = 1;
        while len > 0 {
            len -= 1;
            visited += 1;
            let (child, t) = stack[len];
            if t > max_t {
                continue;
            }
            if child & LEAF != 0 {
                max_t = visit(self.leaves[(child & !LEAF) as usize], t);
                continue;
            }
            let node = &self.nodes[child as usize];
            let mut enter = [0.0_f32; LANES];
            let mut exit = [max_t; LANES];
            for axis in 0..3 {
                let o = origin[axis];
                let (lo, hi) = (&node.min[axis], &node.max[axis]);
                if parallel[axis] {
                    for l in 0..LANES {
                        if o < lo[l] - pad || o > hi[l] + pad {
                            exit[l] = f32::NEG_INFINITY;
                        }
                    }
                } else {
                    let inv = inverse[axis];
                    for l in 0..LANES {
                        let a = (lo[l] - pad - o) * inv;
                        let b = (hi[l] + pad - o) * inv;
                        let (near, far) = if a < b { (a, b) } else { (b, a) };
                        enter[l] = if near > enter[l] { near } else { enter[l] };
                        exit[l] = if far < exit[l] { far } else { exit[l] };
                    }
                }
            }
            // Push entered children farthest first, so the nearest is visited next.
            let mut hits = [(0_u32, 0.0_f32); LANES];
            let mut count = 0;
            for l in 0..LANES {
                if enter[l] <= exit[l] && node.child[l] != EMPTY {
                    let mut k = count;
                    while k > 0 && hits[k - 1].1 < enter[l] {
                        hits[k] = hits[k - 1];
                        k -= 1;
                    }
                    hits[k] = (node.child[l], enter[l]);
                    count += 1;
                }
            }
            stack[len..len + count].copy_from_slice(&hits[..count]);
            len += count;
        }
        visited
    }

    /// Visits every leaf whose exact AABB overlaps `query`, in traversal order, without
    /// allocating. Returns the number of nodes visited.
    // Lane loops index the per-lane arrays in lockstep, which is what vectorizes.
    #[allow(clippy::needless_range_loop)]
    pub fn visit_aabb(&self, query: Aabb, mut visit: impl FnMut(Body)) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut visited = 0;
        let mut stack = [0_u32; STACK];
        let mut len = 1;
        while len > 0 {
            len -= 1;
            visited += 1;
            let node = &self.nodes[stack[len] as usize];
            let mut overlap = [true; LANES];
            for axis in 0..3 {
                for l in 0..LANES {
                    overlap[l] &= node.min[axis][l] <= query.max[axis]
                        && node.max[axis][l] >= query.min[axis];
                }
            }
            for l in 0..LANES {
                let child = node.child[l];
                if overlap[l] && child != EMPTY {
                    if child & LEAF != 0 {
                        visit(self.leaves[(child & !LEAF) as usize]);
                    } else {
                        stack[len] = child;
                        len += 1;
                    }
                }
            }
        }
        visited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(n: u32, seed: u64) -> Vec<Body> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..n)
            .map(|id| {
                let c = [next() * 100.0 - 50.0, next() * 10.0, next() * 100.0 - 50.0];
                let h = [next() * 3.0 + 0.1, next() * 2.0, next() * 3.0 + 0.1];
                Body::new(id * 2, Aabb::from_center_half_extents(c, h))
            })
            .collect()
    }

    fn entry(b: &Aabb, o: [f32; 3], d: [f32; 3], max_t: f32, pad: f32) -> Option<f32> {
        let (mut enter, mut exit) = (0.0_f32, max_t);
        for k in 0..3 {
            let (lo, hi) = (b.min[k] - pad, b.max[k] + pad);
            if d[k] == 0.0 {
                if o[k] < lo || o[k] > hi {
                    return None;
                }
                continue;
            }
            let (a, c) = ((lo - o[k]) / d[k], (hi - o[k]) / d[k]);
            enter = enter.max(a.min(c));
            exit = exit.min(a.max(c));
        }
        (enter <= exit).then_some(enter)
    }

    fn rays() -> Vec<([f32; 3], [f32; 3])> {
        let mut out = vec![
            ([0.0, 20.0, 0.0], [0.0, -30.0, 0.0]),
            ([-60.0, 1.0, 0.0], [120.0, 0.0, 0.0]),
            ([0.0, 1.0, -60.0], [0.0, 0.0, 120.0]),
        ];
        for k in 0..200 {
            let a = k as f32 * 0.37;
            out.push((
                [a.sin() * 40.0, 1.0 + (k % 7) as f32, a.cos() * 40.0],
                [
                    a.cos() * 50.0,
                    ((k % 5) as f32 - 2.0) * 0.5,
                    -a.sin() * 50.0,
                ],
            ));
        }
        out
    }

    #[test]
    fn segment_visits_exactly_the_entered_leaves_nearest_first() {
        for n in [0, 1, 2, 3, 5, 17, 300] {
            let bodies = scene(n, 0x1234_5678 + u64::from(n));
            let tree = WideBvh::build(&bodies);
            assert_eq!(tree.len(), n as usize);
            for (o, d) in rays() {
                let mut got = Vec::new();
                tree.visit_segment(o, d, 1.0, 1e-4, |b, t| {
                    got.push((b.id, t));
                    1.0
                });
                let mut want: Vec<_> = bodies
                    .iter()
                    .filter_map(|b| entry(&b.aabb, o, d, 1.0, 1e-4).map(|t| (b.id, t)))
                    .collect();
                let mut ids: Vec<_> = got.iter().map(|g| g.0).collect();
                ids.sort_unstable();
                want.sort_by_key(|w| w.0);
                assert_eq!(ids, want.iter().map(|w| w.0).collect::<Vec<_>>());
                for (id, t) in &got {
                    let w = want.iter().find(|w| w.0 == *id).unwrap().1;
                    assert!((t - w).abs() <= 1e-5 * (1.0 + w.abs()), "{t} vs {w}");
                }
            }
        }
    }

    #[test]
    fn first_hit_pruning_finds_the_nearest_entry() {
        let bodies = scene(500, 7);
        let tree = WideBvh::build(&bodies);
        for (o, d) in rays() {
            let mut best = None::<(u32, f32)>;
            tree.visit_segment(o, d, 1.0, 0.0, |b, t| {
                if best.is_none_or(|(_, bt)| t < bt) {
                    best = Some((b.id, t));
                }
                best.map_or(1.0, |b| b.1)
            });
            let want = bodies
                .iter()
                .filter_map(|b| entry(&b.aabb, o, d, 1.0, 0.0))
                .min_by(f32::total_cmp);
            // The tree multiplies by the inverse direction; the reference divides.
            match (best, want) {
                (Some((_, t)), Some(w)) => assert!((t - w).abs() <= 1e-5 * (1.0 + w), "{t} {w}"),
                (None, None) => {}
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn aabb_visits_exactly_the_overlapping_leaves_and_removal_drops_them() {
        let bodies = scene(400, 99);
        let mut tree = WideBvh::build(&bodies);
        for k in (0..400).step_by(3) {
            assert!(tree.remove(k * 2).is_some());
        }
        assert!(tree.remove(1).is_none());
        assert!(!tree.contains(0) && tree.contains(2));
        let live: Vec<_> = bodies.iter().filter(|b| tree.contains(b.id)).collect();
        assert_eq!(tree.len(), live.len());
        for q in [
            Aabb::new([-10.0, 0.0, -10.0], [10.0, 5.0, 10.0]),
            Aabb::new([-60.0, -5.0, -60.0], [60.0, 20.0, 60.0]),
            Aabb::new([30.0, 0.0, 30.0], [31.0, 1.0, 31.0]),
        ] {
            let mut got = Vec::new();
            tree.visit_aabb(q, |b| got.push(b.id));
            got.sort_unstable();
            let want: Vec<_> = live
                .iter()
                .filter(|b| b.aabb.overlaps(q))
                .map(|b| b.id)
                .collect();
            assert_eq!(got, want);
        }
        for (o, d) in rays() {
            tree.visit_segment(o, d, 1.0, 0.0, |b, _| {
                assert!(tree.contains(b.id));
                1.0
            });
        }
    }

    #[test]
    fn identical_centroids_still_build_a_shallow_tree() {
        let bodies: Vec<_> = (0..4096)
            .map(|id| Body::new(id, Aabb::new([0.0; 3], [1.0; 3])))
            .collect();
        let tree = WideBvh::build(&bodies);
        let mut count = 0;
        let visited = tree.visit_aabb(Aabb::new([0.5; 3], [0.6; 3]), |_| count += 1);
        assert_eq!(count, 4096);
        assert!(visited < 4096);
    }
}
