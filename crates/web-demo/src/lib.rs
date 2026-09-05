use collection_kernels::FenwickTree;

const VALUES: [i64; 8] = [3, -1, 4, 1, 5, 9, 2, 6];

fn tree() -> FenwickTree<i64> {
    FenwickTree::from_slice(&VALUES)
}

#[unsafe(no_mangle)]
pub extern "C" fn dataset_len() -> u32 {
    VALUES.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn dataset_value(index: u32) -> i64 {
    VALUES.get(index as usize).copied().unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "C" fn fenwick_prefix_sum(end: u32) -> i64 {
    let end = (end as usize).min(VALUES.len());
    tree().prefix_sum(end)
}

#[unsafe(no_mangle)]
pub extern "C" fn fenwick_range_sum(start: u32, end: u32) -> i64 {
    let end = (end as usize).min(VALUES.len());
    let start = (start as usize).min(end);
    tree().range_sum(start..end)
}

#[cfg(test)]
mod tests {
    use super::{dataset_len, dataset_value, fenwick_prefix_sum, fenwick_range_sum};

    #[test]
    fn exported_demo_matches_expected_fixture() {
        assert_eq!(dataset_len(), 8);
        assert_eq!(dataset_value(2), 4);
        assert_eq!(dataset_value(99), 0);
        assert_eq!(fenwick_prefix_sum(4), 7);
        assert_eq!(fenwick_prefix_sum(99), 29);
        assert_eq!(fenwick_range_sum(2, 6), 19);
        assert_eq!(fenwick_range_sum(6, 2), 0);
    }
}
