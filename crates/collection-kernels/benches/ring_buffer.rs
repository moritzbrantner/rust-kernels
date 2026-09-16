use collection_kernels::RingBuffer;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const WRAP_CAPACITY: usize = 1_024;
const WRAP_OPERATIONS: usize = 8_192;
const CLEAR_CAPACITY: usize = 65_536;
const CLEAR_LEN: usize = 64;

fn wraparound_push_pop(c: &mut Criterion) {
    c.bench_function("ring_buffer/wraparound_push_pop_8192", |b| {
        b.iter_batched(
            || {
                let mut buffer = RingBuffer::new(WRAP_CAPACITY);
                for value in 0..WRAP_CAPACITY {
                    buffer.push(value).expect("fixture has capacity");
                }
                buffer
            },
            |mut buffer| {
                for value in WRAP_CAPACITY..WRAP_CAPACITY + WRAP_OPERATIONS {
                    black_box(buffer.pop());
                    buffer.push(black_box(value)).expect("pop frees one slot");
                }
                black_box(buffer);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn clear_sparse(c: &mut Criterion) {
    c.bench_function("ring_buffer/clear_64_of_65536", |b| {
        b.iter_batched(
            || {
                let mut buffer = RingBuffer::new(CLEAR_CAPACITY);
                for value in 0..CLEAR_LEN {
                    buffer.push(value).expect("fixture has capacity");
                }
                buffer
            },
            |mut buffer| {
                buffer.clear();
                black_box(buffer.len());
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, wraparound_push_pop, clear_sparse);
criterion_main!(benches);
