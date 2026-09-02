use collection_kernels::AddressablePriorityQueue;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const ITEMS: usize = 10_000;

fn insert_pop(c: &mut Criterion) {
    c.bench_function("priority_queue/insert_pop", |b| {
        b.iter(|| {
            let mut queue = AddressablePriorityQueue::new();
            for index in 0..ITEMS {
                queue.insert((ITEMS - index) as i64, index);
            }
            while let Some(entry) = queue.pop_min() {
                black_box(entry);
            }
        });
    });
}

fn update_heavy(c: &mut Criterion) {
    c.bench_function("priority_queue/update_heavy", |b| {
        b.iter(|| {
            let mut queue = AddressablePriorityQueue::new();
            let handles = (0..ITEMS)
                .map(|index| queue.insert(index as i64, index))
                .collect::<Vec<_>>();

            for round in 0..4usize {
                for (index, handle) in handles.iter().copied().enumerate() {
                    let priority = ((index * 31 + round * 17) % ITEMS) as i64;
                    queue
                        .update_priority(handle, priority)
                        .expect("benchmark handles remain active");
                }
            }

            black_box(queue.peek_min());
        });
    });
}

fn mixed_mutations(c: &mut Criterion) {
    c.bench_function("priority_queue/mixed_mutations", |b| {
        b.iter(|| {
            let mut queue = AddressablePriorityQueue::new();
            let handles = (0..ITEMS)
                .map(|index| queue.insert(((index * 13) % ITEMS) as i64, index))
                .collect::<Vec<_>>();

            for (index, handle) in handles.iter().copied().enumerate() {
                if index % 3 == 0 {
                    black_box(queue.remove(handle).expect("handle is removed once"));
                } else {
                    queue
                        .update_priority(handle, ((ITEMS - index) * 7 % ITEMS) as i64)
                        .expect("remaining handle stays active");
                }
            }

            while let Some(entry) = queue.pop_min() {
                black_box(entry);
            }
        });
    });
}

criterion_group!(benches, insert_pop, update_heavy, mixed_mutations);
criterion_main!(benches);
