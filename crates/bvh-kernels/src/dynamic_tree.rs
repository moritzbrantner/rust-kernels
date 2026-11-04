use std::collections::{HashMap, HashSet};

use spatial_kernels::{Aabb, Body, BroadPhaseResult, BroadPhaseStats, ColliderId, Pair};

use crate::segment::Segment;

#[derive(Clone, Copy, Debug)]
struct DynamicNode {
    bounds: Aabb,
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    height: i32,
    body: Option<Body>,
}

impl DynamicNode {
    fn leaf(body: Body, bounds: Aabb) -> Self {
        Self {
            bounds,
            parent: None,
            left: None,
            right: None,
            height: 0,
            body: Some(body),
        }
    }

    fn branch(bounds: Aabb, parent: Option<usize>, left: usize, right: usize, height: i32) -> Self {
        Self {
            bounds,
            parent,
            left: Some(left),
            right: Some(right),
            height,
            body: None,
        }
    }

    const fn is_leaf(self) -> bool {
        self.body.is_some()
    }
}

/// Stable-by-snapshot view of one internal dynamic-tree node. Node indices are
/// debug identifiers for one tree instance, not persistent external handles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicAabbNodeSnapshot {
    pub index: usize,
    pub bounds: Aabb,
    pub exact_bounds: Option<Aabb>,
    pub height: i32,
    pub body: Option<ColliderId>,
    pub parent: Option<usize>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub is_root: bool,
}

/// Optional debug trace for one dynamic-tree body update.
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicAabbUpdateTrace {
    pub id: ColliderId,
    pub reinserted: bool,
    pub previous_fat_bounds: Aabb,
    pub current_fat_bounds: Aabb,
    pub height_before: usize,
    pub height_after: usize,
    pub changed_nodes: Vec<usize>,
    pub before_nodes: Vec<DynamicAabbNodeSnapshot>,
    pub after_nodes: Vec<DynamicAabbNodeSnapshot>,
}

/// Incrementally updatable AABB tree. Leaves use a configurable fat margin so
/// small movements can update the exact body without restructuring the tree.
#[derive(Clone, Debug)]
pub struct DynamicAabbTree {
    nodes: Vec<Option<DynamicNode>>,
    free: Vec<usize>,
    root: Option<usize>,
    /// Leaf node per collider ID (`NO_LEAF` when absent). IDs index this table directly, so
    /// callers should keep them compact, for example slot indices.
    leaves: Vec<usize>,
    len: usize,
    fat_margin: f32,
}

const NO_LEAF: usize = usize::MAX;
/// Traversal stack depth. The tree is height balanced, so 64 levels cover any `u32` ID space.
const STACK_DEPTH: usize = 64;

impl Default for DynamicAabbTree {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl DynamicAabbTree {
    #[must_use]
    pub fn new(fat_margin: f32) -> Self {
        assert!(
            fat_margin.is_finite() && fat_margin >= 0.0,
            "fat AABB margin must be non-negative and finite"
        );
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            root: None,
            leaves: Vec::new(),
            len: 0,
            fat_margin,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn leaf_of(&self, id: ColliderId) -> Option<usize> {
        self.leaves
            .get(id as usize)
            .copied()
            .filter(|&leaf| leaf != NO_LEAF)
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    #[must_use]
    pub const fn fat_margin(&self) -> f32 {
        self.fat_margin
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.root.map_or(0, |root| {
            usize::try_from(self.node(root).height + 1).unwrap_or(0)
        })
    }

    /// Returns a deterministic node-index-ordered snapshot for debug and
    /// visualization tooling. Normal broad-phase consumers do not need this.
    #[must_use]
    pub fn debug_nodes(&self) -> Vec<DynamicAabbNodeSnapshot> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                let node = (*node)?;
                Some(DynamicAabbNodeSnapshot {
                    index,
                    bounds: node.bounds,
                    exact_bounds: node.body.map(|body| body.aabb),
                    height: node.height,
                    body: node.body.map(|body| body.id),
                    parent: node.parent,
                    left: node.left,
                    right: node.right,
                    is_root: self.root == Some(index),
                })
            })
            .collect()
    }

    pub fn insert(&mut self, body: Body) {
        assert!(
            self.leaf_of(body.id).is_none(),
            "collider IDs must be unique"
        );
        let leaf = self.allocate(DynamicNode::leaf(body, body.aabb.expanded(self.fat_margin)));
        let slot = body.id as usize;
        if slot >= self.leaves.len() {
            self.leaves.resize(slot + 1, NO_LEAF);
        }
        self.leaves[slot] = leaf;
        self.len += 1;
        self.insert_leaf(leaf);
    }

    pub fn remove(&mut self, id: ColliderId) -> Option<Body> {
        let leaf = self.leaf_of(id)?;
        self.leaves[id as usize] = NO_LEAF;
        self.len -= 1;
        let body = self.node(leaf).body.expect("leaf must contain a body");
        self.remove_leaf(leaf);
        self.release(leaf);
        Some(body)
    }

    /// Updates an existing body. Returns `true` when the leaf had to be removed
    /// and reinserted because the exact AABB left its previous fat AABB.
    pub fn update(&mut self, body: Body) -> bool {
        let leaf = self
            .leaf_of(body.id)
            .expect("updated collider must already exist");
        if self.node(leaf).bounds.contains(body.aabb) {
            self.node_mut(leaf).body = Some(body);
            return false;
        }

        let fat_bounds = body.aabb.expanded(self.fat_margin);
        self.remove_leaf(leaf);
        let node = self.node_mut(leaf);
        node.bounds = fat_bounds;
        node.body = Some(body);
        node.parent = None;
        node.height = 0;
        self.insert_leaf(leaf);
        true
    }

    /// Like [`Self::update`], but a reinserted leaf's fat AABB also stretches by
    /// `displacement` along its direction (predictive fattening, as in Box2D), so a body that
    /// keeps moving that way stays inside its leaf for several updates. Returns `true` when the
    /// leaf was reinserted.
    pub fn update_predicted(&mut self, body: Body, displacement: [f32; 3]) -> bool {
        assert!(
            displacement.iter().all(|d| d.is_finite()),
            "predicted displacement must be finite"
        );
        let leaf = self
            .leaf_of(body.id)
            .expect("updated collider must already exist");
        if self.node(leaf).bounds.contains(body.aabb) {
            self.node_mut(leaf).body = Some(body);
            return false;
        }

        let fat = body.aabb.expanded(self.fat_margin);
        let (mut min, mut max) = (fat.min, fat.max);
        for axis in 0..3 {
            if displacement[axis] < 0.0 {
                min[axis] += displacement[axis];
            } else {
                max[axis] += displacement[axis];
            }
        }
        self.remove_leaf(leaf);
        let node = self.node_mut(leaf);
        node.bounds = Aabb::new(min, max);
        node.body = Some(body);
        node.parent = None;
        node.height = 0;
        self.insert_leaf(leaf);
        true
    }

    /// Runs the same update as [`Self::update`] while capturing before/after
    /// structural snapshots. This path intentionally allocates; normal updates
    /// remain allocation-free with respect to tracing.
    #[must_use]
    pub fn update_with_trace(&mut self, body: Body) -> DynamicAabbUpdateTrace {
        let previous_fat_bounds = self
            .fat_bounds(body.id)
            .expect("updated collider must already exist");
        let height_before = self.height();
        let before_nodes = self.debug_nodes();
        let reinserted = self.update(body);
        let current_fat_bounds = self
            .fat_bounds(body.id)
            .expect("updated collider must still exist");
        let height_after = self.height();
        let after_nodes = self.debug_nodes();
        let changed_nodes = changed_node_indices(&before_nodes, &after_nodes);

        DynamicAabbUpdateTrace {
            id: body.id,
            reinserted,
            previous_fat_bounds,
            current_fat_bounds,
            height_before,
            height_after,
            changed_nodes,
            before_nodes,
            after_nodes,
        }
    }

    #[must_use]
    pub fn fat_bounds(&self, id: ColliderId) -> Option<Aabb> {
        self.leaf_of(id).map(|index| self.node(index).bounds)
    }

    /// Rebuilds the hierarchy top-down with a binned surface-area heuristic over the current
    /// leaves. Leaves keep their fat bounds and IDs; query results are unchanged. Call after
    /// many incremental inserts to restore query performance.
    pub fn rebuild(&mut self) {
        let mut leaves = Vec::with_capacity(self.len);
        for index in 0..self.nodes.len() {
            match self.nodes[index] {
                Some(node) if node.is_leaf() => leaves.push(index),
                Some(_) => self.release(index),
                None => {}
            }
        }
        self.root = (!leaves.is_empty()).then(|| self.build_sah(&mut leaves, None));
    }

    fn build_sah(&mut self, leaves: &mut [usize], parent: Option<usize>) -> usize {
        if let [leaf] = leaves {
            let node = self.node_mut(*leaf);
            node.parent = parent;
            node.height = 0;
            return *leaf;
        }
        const BINS: usize = 16;
        let centroid = |tree: &Self, leaf: usize, axis: usize| {
            let b = tree.node_ref(leaf).bounds;
            b.min[axis] + b.max[axis]
        };
        let mut best: Option<(f64, usize, f32)> = None;
        for axis in 0..3 {
            let (lo, hi) =
                leaves
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &l| {
                        let c = centroid(self, l, axis);
                        (lo.min(c), hi.max(c))
                    });
            if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
            let scale = BINS as f32 / (hi - lo);
            let bin = |c: f32| (((c - lo) * scale) as usize).min(BINS - 1);
            let mut counts = [0_usize; BINS];
            let mut bounds: [Option<Aabb>; BINS] = [None; BINS];
            for &l in leaves.iter() {
                let b = bin(centroid(self, l, axis));
                counts[b] += 1;
                let leaf_bounds = self.node_ref(l).bounds;
                bounds[b] = Some(bounds[b].map_or(leaf_bounds, |x| x.union(leaf_bounds)));
            }
            let mut right_area = [0.0_f64; BINS];
            let mut acc: Option<Aabb> = None;
            for b in (1..BINS).rev() {
                if let Some(x) = bounds[b] {
                    acc = Some(acc.map_or(x, |a| a.union(x)));
                }
                right_area[b] = acc.map_or(0.0, Aabb::surface_area);
            }
            let (mut left, mut left_count, mut right_count) = (None::<Aabb>, 0, leaves.len());
            for b in 0..BINS - 1 {
                if let Some(x) = bounds[b] {
                    left = Some(left.map_or(x, |a| a.union(x)));
                }
                left_count += counts[b];
                right_count -= counts[b];
                if left_count == 0 || right_count == 0 {
                    continue;
                }
                let cost = left.map_or(0.0, Aabb::surface_area) * left_count as f64
                    + right_area[b + 1] * right_count as f64;
                if best.is_none_or(|(c, ..)| cost < c) {
                    best = Some((cost, axis, lo + (b + 1) as f32 / scale));
                }
            }
        }
        let mid = match best {
            Some((_, axis, split)) => {
                let mut mid = 0;
                for i in 0..leaves.len() {
                    if centroid(self, leaves[i], axis) < split {
                        leaves.swap(i, mid);
                        mid += 1;
                    }
                }
                mid
            }
            None => leaves.len() / 2,
        };
        let mid = if mid == 0 || mid == leaves.len() {
            leaves.len() / 2
        } else {
            mid
        };
        let placeholder = Aabb::new([0.0; 3], [0.0; 3]);
        let index = self.allocate(DynamicNode::branch(placeholder, parent, 0, 0, 0));
        let (l, r) = leaves.split_at_mut(mid);
        let left = self.build_sah(l, Some(index));
        let right = self.build_sah(r, Some(index));
        let bounds = self
            .node_ref(left)
            .bounds
            .union(self.node_ref(right).bounds);
        let height = 1 + self.node_ref(left).height.max(self.node_ref(right).height);
        *self.node_mut(index) = DynamicNode::branch(bounds, parent, left, right, height);
        index
    }

    /// Visits every body whose exact AABB overlaps `query`, in traversal order, without
    /// allocating. Sort the visited IDs if the caller needs a canonical order.
    pub fn visit_aabb(&self, query: Aabb, mut visit: impl FnMut(Body)) -> usize {
        let Some(root) = self.root else {
            return 0;
        };
        let mut visited = 0;
        let mut stack = [0_usize; STACK_DEPTH];
        stack[0] = root;
        let mut len = 1;
        while len > 0 {
            len -= 1;
            visited += 1;
            let node = self.node_ref(stack[len]);
            if !node.bounds.overlaps(query) {
                continue;
            }
            if let Some(body) = node.body {
                if body.aabb.overlaps(query) {
                    visit(body);
                }
            } else {
                stack[len] = node.right.expect("branch must have right child");
                stack[len + 1] = node.left.expect("branch must have left child");
                len += 2;
            }
        }
        visited
    }

    pub fn visit_segment(
        &self,
        origin: [f32; 3],
        delta: [f32; 3],
        max_t: f32,
        pad: f32,
        mut visit: impl FnMut(Body, f32) -> f32,
    ) -> usize {
        let Some(root) = self.root else {
            return 0;
        };
        let mut visited = 0;
        let segment = Segment::new(origin, delta, pad);
        let mut max_t = max_t;
        let Some(t) = segment.entry(&self.node_ref(root).bounds, max_t) else {
            return 1;
        };
        let mut stack = [(0_usize, 0.0_f32); STACK_DEPTH];
        stack[0] = (root, t);
        let mut len = 1;
        while len > 0 {
            len -= 1;
            visited += 1;
            let (index, t) = stack[len];
            if t > max_t {
                continue;
            }
            let node = self.node_ref(index);
            if let Some(body) = node.body {
                if let Some(t) = segment.entry(&body.aabb, max_t) {
                    max_t = visit(body, t);
                }
                continue;
            }
            let left = node.left.expect("branch must have left child");
            let right = node.right.expect("branch must have right child");
            let tl = segment.entry(&self.node_ref(left).bounds, max_t);
            let tr = segment.entry(&self.node_ref(right).bounds, max_t);
            // Push the farther child first so the nearer one is visited first.
            let (first, second) = match (tl, tr) {
                (Some(a), Some(b)) if b < a => ((left, tl), (right, tr)),
                _ => ((right, tr), (left, tl)),
            };
            for (index, t) in [first, second] {
                if let Some(t) = t {
                    stack[len] = (index, t);
                    len += 1;
                }
            }
        }
        visited
    }

    /// Returns collider IDs whose exact AABBs overlap `query`, sorted by ID.
    #[must_use]
    pub fn query_aabb(&self, query: Aabb) -> Vec<ColliderId> {
        let Some(root) = self.root else {
            return Vec::new();
        };
        let mut hits = Vec::new();
        let mut stack = vec![root];
        while let Some(index) = stack.pop() {
            let node = self.node(index);
            if !node.bounds.overlaps(query) {
                continue;
            }
            if let Some(body) = node.body {
                if body.aabb.overlaps(query) {
                    hits.push(body.id);
                }
            } else {
                stack.push(node.right.expect("branch must have right child"));
                stack.push(node.left.expect("branch must have left child"));
            }
        }
        hits.sort_unstable();
        hits
    }

    #[must_use]
    pub fn overlapping_pairs(&self) -> Vec<Pair> {
        self.overlapping_pairs_result().pairs
    }

    /// Returns exact overlap pairs plus deterministic broad-phase work counters
    /// for the current retained tree state.
    #[must_use]
    pub fn overlapping_pairs_result(&self) -> BroadPhaseResult {
        let (pairs, aabb_tests) = self.overlapping_pairs_with_tests();
        BroadPhaseResult {
            pairs,
            stats: BroadPhaseStats {
                aabb_tests,
                occupied_cells: None,
            },
        }
    }

    fn overlapping_pairs_with_tests(&self) -> (Vec<Pair>, u64) {
        let Some(root) = self.root else {
            return (Vec::new(), 0);
        };

        let mut pairs = Vec::new();
        let mut aabb_tests = 0_u64;
        self.collect_within_dynamic(root, &mut pairs, &mut aabb_tests);
        pairs.sort_unstable();
        (pairs, aabb_tests)
    }

    fn collect_within_dynamic(
        &self,
        node_index: usize,
        pairs: &mut Vec<Pair>,
        aabb_tests: &mut u64,
    ) {
        let node = self.node(node_index);
        if node.is_leaf() {
            return;
        }

        let left = node.left.expect("branch must have left child");
        let right = node.right.expect("branch must have right child");
        self.collect_within_dynamic(left, pairs, aabb_tests);
        self.collect_within_dynamic(right, pairs, aabb_tests);
        self.collect_cross_dynamic(left, right, pairs, aabb_tests);
    }

    fn collect_cross_dynamic(
        &self,
        left_index: usize,
        right_index: usize,
        pairs: &mut Vec<Pair>,
        aabb_tests: &mut u64,
    ) {
        let left_node = self.node(left_index);
        let right_node = self.node(right_index);
        if !left_node.bounds.overlaps(right_node.bounds) {
            return;
        }

        match (left_node.body, right_node.body) {
            (Some(left), Some(right)) => {
                *aabb_tests += 1;
                if left.aabb.overlaps(right.aabb) {
                    pairs.push(Pair::new(left.id, right.id));
                }
            }
            (None, Some(_)) => {
                self.collect_cross_dynamic(
                    left_node.left.expect("branch must have left child"),
                    right_index,
                    pairs,
                    aabb_tests,
                );
                self.collect_cross_dynamic(
                    left_node.right.expect("branch must have right child"),
                    right_index,
                    pairs,
                    aabb_tests,
                );
            }
            (Some(_), None) => {
                self.collect_cross_dynamic(
                    left_index,
                    right_node.left.expect("branch must have left child"),
                    pairs,
                    aabb_tests,
                );
                self.collect_cross_dynamic(
                    left_index,
                    right_node.right.expect("branch must have right child"),
                    pairs,
                    aabb_tests,
                );
            }
            (None, None) => {
                let left_left = left_node.left.expect("branch must have left child");
                let left_right = left_node.right.expect("branch must have right child");
                let right_left = right_node.left.expect("branch must have left child");
                let right_right = right_node.right.expect("branch must have right child");
                self.collect_cross_dynamic(left_left, right_left, pairs, aabb_tests);
                self.collect_cross_dynamic(left_left, right_right, pairs, aabb_tests);
                self.collect_cross_dynamic(left_right, right_left, pairs, aabb_tests);
                self.collect_cross_dynamic(left_right, right_right, pairs, aabb_tests);
            }
        }
    }

    fn insert_leaf(&mut self, leaf: usize) {
        let Some(mut sibling) = self.root else {
            self.root = Some(leaf);
            self.node_mut(leaf).parent = None;
            return;
        };

        let leaf_bounds = self.node(leaf).bounds;
        while !self.node(sibling).is_leaf() {
            let current = self.node(sibling);
            let left = current.left.expect("branch must have left child");
            let right = current.right.expect("branch must have right child");
            let combined = current.bounds.union(leaf_bounds);
            let combined_area = combined.surface_area();
            let inheritance = 2.0 * (combined_area - current.bounds.surface_area());
            let parent_cost = 2.0 * combined_area;
            let left_cost = descend_cost(self.node(left), leaf_bounds, inheritance);
            let right_cost = descend_cost(self.node(right), leaf_bounds, inheritance);

            if parent_cost < left_cost && parent_cost < right_cost {
                break;
            }
            sibling = match left_cost.total_cmp(&right_cost) {
                std::cmp::Ordering::Less => left,
                std::cmp::Ordering::Greater => right,
                std::cmp::Ordering::Equal => left.min(right),
            };
        }

        let old_parent = self.node(sibling).parent;
        let new_parent = self.allocate(DynamicNode::branch(
            leaf_bounds.union(self.node(sibling).bounds),
            old_parent,
            sibling,
            leaf,
            self.node(sibling).height + 1,
        ));
        self.node_mut(sibling).parent = Some(new_parent);
        self.node_mut(leaf).parent = Some(new_parent);

        if let Some(parent) = old_parent {
            self.replace_child(parent, sibling, new_parent);
        } else {
            self.root = Some(new_parent);
        }
        self.fix_upwards(Some(new_parent));
    }

    fn remove_leaf(&mut self, leaf: usize) {
        if self.root == Some(leaf) {
            self.root = None;
            self.node_mut(leaf).parent = None;
            return;
        }

        let parent = self
            .node(leaf)
            .parent
            .expect("non-root leaf must have parent");
        let parent_node = self.node(parent);
        let sibling = if parent_node.left == Some(leaf) {
            parent_node.right.expect("branch must have right child")
        } else {
            parent_node.left.expect("branch must have left child")
        };
        let grand_parent = parent_node.parent;

        if let Some(grand_parent) = grand_parent {
            self.replace_child(grand_parent, parent, sibling);
            self.node_mut(sibling).parent = Some(grand_parent);
            self.release(parent);
            self.node_mut(leaf).parent = None;
            self.fix_upwards(Some(grand_parent));
        } else {
            self.root = Some(sibling);
            self.node_mut(sibling).parent = None;
            self.release(parent);
            self.node_mut(leaf).parent = None;
        }
    }

    fn replace_child(&mut self, parent: usize, old_child: usize, new_child: usize) {
        let node = self.node_mut(parent);
        if node.left == Some(old_child) {
            node.left = Some(new_child);
        } else {
            assert_eq!(
                node.right,
                Some(old_child),
                "parent must reference old child"
            );
            node.right = Some(new_child);
        }
    }

    fn fix_upwards(&mut self, mut index: Option<usize>) {
        while let Some(current) = index {
            let balanced = self.balance(current);
            if !self.node(balanced).is_leaf() {
                self.sync_branch(balanced);
            }
            index = self.node(balanced).parent;
        }
    }

    fn balance(&mut self, a: usize) -> usize {
        let a_node = self.node(a);
        if a_node.is_leaf() || a_node.height < 2 {
            return a;
        }
        let b = a_node.left.expect("branch must have left child");
        let c = a_node.right.expect("branch must have right child");
        let balance = self.node(c).height - self.node(b).height;

        if balance > 1 {
            let c_node = self.node(c);
            let f = c_node.left.expect("branch must have left child");
            let g = c_node.right.expect("branch must have right child");
            let old_parent = a_node.parent;

            self.node_mut(c).left = Some(a);
            self.node_mut(c).parent = old_parent;
            self.node_mut(a).parent = Some(c);
            if let Some(parent) = old_parent {
                self.replace_child(parent, a, c);
            } else {
                self.root = Some(c);
            }

            if self.node(f).height > self.node(g).height {
                self.node_mut(c).right = Some(f);
                self.node_mut(a).right = Some(g);
                self.node_mut(f).parent = Some(c);
                self.node_mut(g).parent = Some(a);
            } else {
                self.node_mut(c).right = Some(g);
                self.node_mut(a).right = Some(f);
                self.node_mut(g).parent = Some(c);
                self.node_mut(f).parent = Some(a);
            }
            self.sync_branch(a);
            self.sync_branch(c);
            return c;
        }

        if balance < -1 {
            let b_node = self.node(b);
            let d = b_node.left.expect("branch must have left child");
            let e = b_node.right.expect("branch must have right child");
            let old_parent = a_node.parent;

            self.node_mut(b).left = Some(a);
            self.node_mut(b).parent = old_parent;
            self.node_mut(a).parent = Some(b);
            if let Some(parent) = old_parent {
                self.replace_child(parent, a, b);
            } else {
                self.root = Some(b);
            }

            if self.node(d).height > self.node(e).height {
                self.node_mut(b).right = Some(d);
                self.node_mut(a).left = Some(e);
                self.node_mut(d).parent = Some(b);
                self.node_mut(e).parent = Some(a);
            } else {
                self.node_mut(b).right = Some(e);
                self.node_mut(a).left = Some(d);
                self.node_mut(e).parent = Some(b);
                self.node_mut(d).parent = Some(a);
            }
            self.sync_branch(a);
            self.sync_branch(b);
            return b;
        }

        a
    }

    fn sync_branch(&mut self, index: usize) {
        let node = self.node(index);
        let left = node.left.expect("branch must have left child");
        let right = node.right.expect("branch must have right child");
        let left_node = self.node(left);
        let right_node = self.node(right);
        let current = self.node_mut(index);
        current.bounds = left_node.bounds.union(right_node.bounds);
        current.height = 1 + left_node.height.max(right_node.height);
    }

    fn allocate(&mut self, node: DynamicNode) -> usize {
        if let Some(index) = self.free.pop() {
            self.nodes[index] = Some(node);
            index
        } else {
            let index = self.nodes.len();
            self.nodes.push(Some(node));
            index
        }
    }

    fn release(&mut self, index: usize) {
        assert!(
            self.nodes[index].take().is_some(),
            "released node must exist"
        );
        self.free.push(index);
    }

    fn node(&self, index: usize) -> DynamicNode {
        self.nodes[index].expect("tree node must exist")
    }

    fn node_ref(&self, index: usize) -> &DynamicNode {
        self.nodes[index].as_ref().expect("tree node must exist")
    }

    fn node_mut(&mut self, index: usize) -> &mut DynamicNode {
        self.nodes[index].as_mut().expect("tree node must exist")
    }
}

fn changed_node_indices(
    before: &[DynamicAabbNodeSnapshot],
    after: &[DynamicAabbNodeSnapshot],
) -> Vec<usize> {
    let before_by_index: HashMap<_, _> = before.iter().map(|node| (node.index, *node)).collect();
    let after_by_index: HashMap<_, _> = after.iter().map(|node| (node.index, *node)).collect();
    let mut indices: HashSet<_> = before_by_index.keys().copied().collect();
    indices.extend(after_by_index.keys().copied());
    let mut changed: Vec<_> = indices
        .into_iter()
        .filter(|index| before_by_index.get(index) != after_by_index.get(index))
        .collect();
    changed.sort_unstable();
    changed
}

fn descend_cost(node: DynamicNode, leaf_bounds: Aabb, inheritance: f64) -> f64 {
    let combined = node.bounds.union(leaf_bounds).surface_area();
    if node.is_leaf() {
        combined + inheritance
    } else {
        combined - node.bounds.surface_area() + inheritance
    }
}
