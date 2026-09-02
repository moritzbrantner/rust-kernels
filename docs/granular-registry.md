# Granular source registry

`rust-kernels` supports two source-integration lanes in the same registry:

1. **crate items** copy a coherent crate/source set; and
2. **standalone modules** copy one independently compilable algorithm or data-structure module into consumer-owned source.

The second lane is the closest analogue to shadcn-style source ownership: consume the implementation, not a library abstraction around it.

## Standalone-module contract

A granular item has a non-crate registry type such as `registry:algorithm` or `registry:data-structure` and declares:

```json
{
  "name": "radix-sort",
  "type": "registry:algorithm",
  "integration": {
    "mode": "standalone-module",
    "module": "radix_sort"
  },
  "dependencies": [],
  "registryDependencies": [],
  "files": [
    {
      "path": "crates/search-kernels/src/radix_sort.rs",
      "target": "src/kernels/radix_sort.rs"
    }
  ]
}
```

For version 1 of this integration mode, the contract is deliberately strict:

- exactly one Rust source file;
- no external package dependencies;
- no registry dependencies;
- a valid Rust module identifier;
- canonical destination `src/kernels/<module>.rs`;
- the exact registered source file must compile independently under the workspace's Rust edition.

The registry validator checks the structural rules. `scripts/test_standalone_registry.py` invokes `rustc` for every standalone item so hidden crate coupling causes CI to fail.

## Installing a single kernel

For example:

```bash
python3 scripts/source_registry.py install radix-sort --root ../consumer
```

This copies:

```text
crates/search-kernels/src/radix_sort.rs
                     |
                     v
../consumer/src/kernels/radix_sort.rs
```

and records the normal immutable source provenance in the consumer's `.rust-kernels.lock.json`.

The consumer owns the copied file immediately. It may specialize the implementation, benchmark it, or later resolve upstream changes through the same provenance/update/conflict-resolution protocol as a copied crate item.

## Module wiring remains consumer-owned

The registry does not edit `lib.rs`, `main.rs`, or another module root automatically. A Rust consumer may choose the shape that fits its application, for example:

```rust
mod kernels {
    pub mod radix_sort;
}
```

or re-export the symbols through its own public API.

That wiring is application architecture rather than kernel source provenance, so the registry does not guess it. The canonical target gives tools a deterministic landing zone while leaving API exposure under consumer control.

## Current standalone items

The initial set is intentionally limited to source files that can stand on their own:

- `morton-code`
- `bit-set`
- `fenwick-tree`
- `generational-arena`
- `lru-cache`
- `ring-buffer`
- `sparse-set`
- `union-find`
- `selection`
- `radix-sort`
- `running-statistics`

These items coexist with their crate-level parents. A consumer can therefore choose between, for example:

```text
search-kernels   -> copy/use the coherent search crate source set
radix-sort       -> own only the radix-sort module
selection        -> own only quickselect/top-k source
```

## Why some kernels are not standalone yet

A registry item is not labeled standalone merely because the algorithm is conceptually small. The source must actually satisfy the integration contract.

Examples deliberately left crate-level for now include:

- `BloomFilter`, because its implementation uses `collection-kernels::BitSet`;
- Kruskal minimum spanning forest, because it uses `collection-kernels::UnionFind`;
- `SpatialHash3D`, because its source currently imports the crate-level `Aabb` type.

Those are good candidates for a future dependency-aware **source-set** integration mode. That mode should explicitly describe multiple files and dependency relationships rather than silently rewriting imports during installation.

## Relationship to provenance

Granularity changes what is copied, not how ancestry works.

Every standalone install still records:

- the exact upstream Git revision;
- the registry hash;
- upstream source path;
- consumer target path;
- exact source SHA-256.

Therefore the existing lifecycle remains unchanged:

```text
install
  -> locally specialize if useful
  -> plan upstream changes
  -> safely apply non-conflicting changes
  -> export base / ours / theirs for real conflicts
  -> human or coding agent resolves semantics
  -> explicitly accept the newer upstream base
```

A standalone module is simply the smallest first-class unit that can participate in that lifecycle without hidden dependencies.
