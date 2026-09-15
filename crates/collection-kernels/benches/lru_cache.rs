use std::hash::{Hash, Hasher};

use collection_kernels::LruCache;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const ITEMS: usize = 8_192;

#[derive(Clone, Eq, PartialEq)]
struct WideKey([u64; 8]);

impl Hash for WideKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

fn generated_keys(len: usize) -> Vec<WideKey> {
    (0..len)
        .map(|index| {
            let base = index as u64;
            WideKey([
                base,
                base.wrapping_mul(3).wrapping_add(1),
                base.rotate_left(7),
                base ^ 0x9e37_79b9_7f4a_7c15,
                base.wrapping_mul(17),
                base.rotate_right(11),
                base.wrapping_add(0xd1b5_4a32_d192_ed03),
                !base,
            ])
        })
        .collect()
}

fn fill_to_capacity(c: &mut Criterion) {
    let keys = generated_keys(ITEMS);
    c.bench_function("lru/fill_to_capacity_8192", |b| {
        b.iter(|| {
            let mut cache = LruCache::new(keys.len());
            for (index, key) in keys.iter().cloned().enumerate() {
                cache.insert(key, index);
            }
            black_box(cache.len())
        });
    });
}

fn update_existing(c: &mut Criterion) {
    let keys = generated_keys(ITEMS);
    c.bench_function("lru/update_existing_8192", |b| {
        b.iter_batched(
            || {
                let mut cache = LruCache::new(keys.len() + 1);
                for (index, key) in keys.iter().cloned().enumerate() {
                    cache.insert(key, index);
                }
                cache
            },
            |mut cache| {
                for (index, key) in keys.iter().cloned().enumerate() {
                    black_box(cache.insert(key, index + 1));
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

fn evict_at_capacity(c: &mut Criterion) {
    let keys = generated_keys(ITEMS * 2);
    c.bench_function("lru/evict_at_capacity_8192", |b| {
        b.iter_batched(
            || {
                let mut cache = LruCache::new(ITEMS);
                for (index, key) in keys[..ITEMS].iter().cloned().enumerate() {
                    cache.insert(key, index);
                }
                cache
            },
            |mut cache| {
                for (index, key) in keys[ITEMS..].iter().cloned().enumerate() {
                    black_box(cache.insert(key, index));
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, fill_to_capacity, update_existing, evict_at_capacity);
criterion_main!(benches);
