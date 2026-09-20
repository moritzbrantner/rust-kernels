use std::sync::OnceLock;

use collection_kernels::{FenwickTree, UnionFind};
use search_kernels::quickselect;
use spatial_kernels::morton2_encode;
use statistics_kernels::RunningStats;

const FENWICK_VALUES: [i64; 8] = [3, -1, 4, 1, 5, 9, 2, 6];
const QUICKSELECT_VALUES: [i32; 12] = [9, 1, 5, 3, 5, 8, 2, 7, 5, 4, 6, 0];
const RUNNING_STATS_VALUES: [f64; 8] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
const UNION_EDGES: [(usize, usize); 7] = [(0, 1), (2, 3), (1, 2), (4, 5), (3, 4), (0, 5), (1, 4)];
const UNION_NODES: usize = 6;

static FENWICK_TREE: OnceLock<FenwickTree<i64>> = OnceLock::new();

fn fenwick_tree() -> &'static FenwickTree<i64> {
    FENWICK_TREE.get_or_init(|| FenwickTree::from_slice(&FENWICK_VALUES))
}

fn quickselect_partition(nth: usize) -> [i32; 12] {
    let mut values = QUICKSELECT_VALUES;
    if nth < values.len() {
        let _ = quickselect(&mut values, nth);
    }
    values
}

fn union_find(edge_mask: u32) -> UnionFind {
    let mut union_find = UnionFind::new(UNION_NODES);
    for (index, &(left, right)) in UNION_EDGES.iter().enumerate() {
        if edge_mask & (1_u32 << index) != 0 {
            let _ = union_find.union(left, right);
        }
    }
    union_find
}

fn running_stats(end: usize) -> RunningStats {
    let end = end.min(RUNNING_STATS_VALUES.len());
    let mut stats = RunningStats::new();
    stats.extend(RUNNING_STATS_VALUES[..end].iter().copied());
    stats
}

#[unsafe(no_mangle)]
pub extern "C" fn fenwick_dataset_len() -> u32 {
    FENWICK_VALUES.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn fenwick_dataset_value(index: u32) -> i64 {
    FENWICK_VALUES
        .get(index as usize)
        .copied()
        .unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "C" fn fenwick_prefix_sum(end: u32) -> i64 {
    let end = (end as usize).min(FENWICK_VALUES.len());
    fenwick_tree().prefix_sum(end)
}

#[unsafe(no_mangle)]
pub extern "C" fn fenwick_range_sum(start: u32, end: u32) -> i64 {
    let end = (end as usize).min(FENWICK_VALUES.len());
    let start = (start as usize).min(end);
    fenwick_tree().range_sum(start..end)
}

#[unsafe(no_mangle)]
pub extern "C" fn union_find_node_count() -> u32 {
    UNION_NODES as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn union_find_edge_count() -> u32 {
    UNION_EDGES.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn union_find_edge_left(index: u32) -> u32 {
    UNION_EDGES
        .get(index as usize)
        .map(|&(left, _)| left as u32)
        .unwrap_or(u32::MAX)
}

#[unsafe(no_mangle)]
pub extern "C" fn union_find_edge_right(index: u32) -> u32 {
    UNION_EDGES
        .get(index as usize)
        .map(|&(_, right)| right as u32)
        .unwrap_or(u32::MAX)
}

#[unsafe(no_mangle)]
pub extern "C" fn union_find_component_count(edge_mask: u32) -> u32 {
    union_find(edge_mask).component_count() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn union_find_component_root(edge_mask: u32, node: u32) -> u32 {
    if node as usize >= UNION_NODES {
        return u32::MAX;
    }

    let mut union_find = union_find(edge_mask);
    union_find.find(node as usize) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn quickselect_dataset_len() -> u32 {
    QUICKSELECT_VALUES.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn quickselect_dataset_value(index: u32) -> i32 {
    QUICKSELECT_VALUES
        .get(index as usize)
        .copied()
        .unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "C" fn quickselect_value(nth: u32) -> i32 {
    let nth = nth as usize;
    let mut values = QUICKSELECT_VALUES;
    quickselect(&mut values, nth).copied().unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "C" fn quickselect_partition_value(nth: u32, index: u32) -> i32 {
    let values = quickselect_partition(nth as usize);
    values.get(index as usize).copied().unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "C" fn morton2_key(x: u32, y: u32) -> u64 {
    morton2_encode(x, y)
}

#[unsafe(no_mangle)]
pub extern "C" fn running_stats_dataset_len() -> u32 {
    RUNNING_STATS_VALUES.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn running_stats_dataset_value(index: u32) -> f64 {
    RUNNING_STATS_VALUES
        .get(index as usize)
        .copied()
        .unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "C" fn running_stats_count(end: u32) -> u32 {
    running_stats(end as usize).count() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn running_stats_mean(end: u32) -> f64 {
    running_stats(end as usize).mean().unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "C" fn running_stats_population_variance(end: u32) -> f64 {
    running_stats(end as usize)
        .population_variance()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{
        fenwick_dataset_len, fenwick_dataset_value, fenwick_prefix_sum, fenwick_range_sum,
        morton2_key, quickselect_dataset_len, quickselect_dataset_value,
        quickselect_partition_value, quickselect_value, running_stats_count,
        running_stats_dataset_len, running_stats_dataset_value, running_stats_mean,
        running_stats_population_variance, union_find_component_count, union_find_component_root,
        union_find_edge_count, union_find_edge_left, union_find_edge_right, union_find_node_count,
    };

    #[test]
    fn fenwick_demo_delegates_to_collection_kernel() {
        assert_eq!(fenwick_dataset_len(), 8);
        assert_eq!(fenwick_dataset_value(2), 4);
        assert_eq!(fenwick_dataset_value(99), 0);
        assert_eq!(fenwick_prefix_sum(4), 7);
        assert_eq!(fenwick_prefix_sum(99), 29);
        assert_eq!(fenwick_range_sum(2, 6), 19);
        assert_eq!(fenwick_range_sum(6, 2), 0);
    }

    #[test]
    fn union_find_demo_rebuilds_selected_edges_deterministically() {
        assert_eq!(union_find_node_count(), 6);
        assert_eq!(union_find_edge_count(), 7);
        assert_eq!((union_find_edge_left(0), union_find_edge_right(0)), (0, 1));
        assert_eq!(union_find_edge_left(99), u32::MAX);

        let mask = 0b000_0111;
        assert_eq!(union_find_component_count(mask), 3);
        for node in 0..4 {
            assert_eq!(union_find_component_root(mask, node), 0);
        }
        assert_eq!(union_find_component_root(mask, 4), 4);
        assert_eq!(union_find_component_root(mask, 99), u32::MAX);
    }

    #[test]
    fn quickselect_demo_exposes_selected_value_and_partition() {
        assert_eq!(quickselect_dataset_len(), 12);
        assert_eq!(quickselect_dataset_value(0), 9);
        assert_eq!(quickselect_value(0), 0);
        assert_eq!(quickselect_value(5), 5);
        assert_eq!(quickselect_value(99), 0);

        let nth = 5;
        let selected = quickselect_value(nth);
        let partition = (0..quickselect_dataset_len())
            .map(|index| quickselect_partition_value(nth, index))
            .collect::<Vec<_>>();
        assert_eq!(partition[nth as usize], selected);
        assert!(partition[..nth as usize].iter().all(|&value| value <= selected));
        assert!(partition[(nth as usize + 1)..]
            .iter()
            .all(|&value| value >= selected));
    }

    #[test]
    fn spatial_and_statistics_demos_delegate_to_their_kernels() {
        assert_eq!(morton2_key(1, 2), 9);
        assert_eq!(running_stats_dataset_len(), 8);
        assert_eq!(running_stats_dataset_value(6), 7.0);
        assert_eq!(running_stats_count(8), 8);
        assert_eq!(running_stats_mean(8), 5.0);
        assert_eq!(running_stats_population_variance(8), 4.0);
    }
}
