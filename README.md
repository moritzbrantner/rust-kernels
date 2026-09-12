# Rust kernels

This workspace contains small, reusable Rust kernels with narrow ownership boundaries, deterministic behavior, and focused validation. Application and laboratory repositories should depend on these low-level crates directly when they need shared algorithms or data structures rather than depending on one another.

## Collection kernels

`collection-kernels` provides reusable data structures including sparse sets and sparse maps. `SparseMap<T>` is the value-bearing companion to `SparseSet`: integer keys map to densely stored values with O(1)-shape lookup, insertion, and swap-remove. It is intentionally a collection primitive rather than an ECS framework; consumers can compose typed component stores without forcing application semantics into this repository.

## Development

Run the repository validation workflows for changes to shared kernels. Keep public APIs small, deterministic, and independently testable.
