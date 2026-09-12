use std::collections::BTreeMap;

use collection_kernels::SparseMap;

#[test]
fn exhaustive_short_sparse_map_sequences_match_btree_map() {
    const ACTIONS: usize = 7;
    const STEPS: usize = 5;

    for case in 0_usize..ACTIONS.pow(STEPS as u32) {
        let mut encoded = case;
        let mut sparse = SparseMap::new();
        let mut model = BTreeMap::new();

        for step in 0..STEPS {
            let action = encoded % ACTIONS;
            encoded /= ACTIONS;

            match action {
                0..=2 => {
                    let key = action;
                    let value = case.wrapping_mul(31).wrapping_add(step);
                    assert_eq!(sparse.insert(key, value), model.insert(key, value));
                }
                3..=5 => {
                    let key = action - 3;
                    assert_eq!(sparse.remove(key), model.remove(&key));
                }
                6 => {
                    sparse.clear();
                    model.clear();
                }
                _ => unreachable!(),
            }

            assert_matches_model(&sparse, &model);
        }
    }
}

fn assert_matches_model(map: &SparseMap<usize>, model: &BTreeMap<usize, usize>) {
    assert_eq!(map.len(), model.len());
    assert_eq!(map.is_empty(), model.is_empty());

    let observed = map
        .iter()
        .map(|(key, value)| (key, *value))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(&observed, model);

    for key in 0..=3 {
        assert_eq!(map.contains_key(key), model.contains_key(&key));
        assert_eq!(map.get(key), model.get(&key));
        assert_eq!(map.dense_index(key).is_some(), model.contains_key(&key));
    }

    let dense_keys = map.keys().collect::<Vec<_>>();
    let dense_values = map.values().copied().collect::<Vec<_>>();
    assert_eq!(dense_keys.len(), dense_values.len());
    for (index, key) in dense_keys.into_iter().enumerate() {
        assert_eq!(map.dense_index(key), Some(index));
        assert_eq!(map.get(key), dense_values.get(index));
    }
}
