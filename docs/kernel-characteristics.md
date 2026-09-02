# Kernel characteristics metadata

Registry version 3 adds machine-readable operational characteristics to granular source items. The goal is not to turn asymptotic analysis into a scoring system; it is to let humans and coding agents discover relevant implementation facts before copying a kernel.

Characteristics live on granular items rather than crate items because one crate may expose several algorithms with very different complexity and mutation behavior.

## Shape

A standalone item describes whether its behavior is deterministic and then lists the important public operations:

```json
{
  "characteristics": {
    "deterministic": true,
    "operations": [
      {
        "operation": "quickselect",
        "time": "O(n) average; O(n^2) worst",
        "extraSpace": "O(1)",
        "mutation": "input",
        "allocation": "none"
      },
      {
        "operation": "top-k smallest",
        "time": "O(n + k log k) average; O(n^2 + k log k) worst",
        "extraSpace": "O(n)",
        "mutation": "none",
        "allocation": "input-sized"
      }
    ]
  }
}
```

Complexity values are strings intentionally. Useful bounds often need qualifiers such as `expected`, `amortized`, multiple parameters, or an average/worst-case distinction. Encoding only a small enum such as `constant | linear | logarithmic` would throw away information that matters when choosing an implementation.

## Fields

### `deterministic`

Whether the kernel produces deterministic semantic results for the same inputs and operation order.

This describes observable algorithm behavior, not necessarily identical wall-clock performance. For example, an implementation backed by Rust's `HashMap` can have expected O(1) operations while still exposing deterministic cache semantics.

### `time`

The asymptotic running-time bound for the named operation. The registry records the implementation that actually exists, including important qualifiers.

Examples include:

- `O(1)`
- `O(1) expected`
- `O(1) amortized`
- `O(alpha(n)) amortized`
- `O(n) average; O(n^2) worst`

### `extraSpace`

Additional storage required by the operation or retained data structure, expressed asymptotically. It is separate from allocation behavior because two operations can have the same big-O space while allocating in materially different ways.

### `mutation`

One of:

- `none` — does not mutate caller input or retained kernel state;
- `input` — rearranges or otherwise mutates caller-provided input;
- `internal-state` — changes retained data-structure state, including recency/path-compression state or values exposed through a mutable accessor.

### `allocation`

One of:

- `none` — the operation performs no heap allocation;
- `fixed` — allocation is constant-sized with respect to input;
- `input-sized` — allocation scales directly with the requested input/universe size;
- `capacity-bounded` — allocation may occur but retained storage is bounded by configured capacity;
- `may-grow` — retained storage can grow as the structure receives new values or keys.

These values intentionally describe implementation behavior rather than promising allocator-level details across every Rust target.

## Verification evidence

Granular items also expose:

```json
{
  "verification": {
    "tests": [
      "crates/search-kernels/src/selection.rs"
    ],
    "benchmarks": []
  }
}
```

`tests` and `benchmarks` contain repository-relative evidence paths.

For the initial standalone catalog, tests are inline in the same Rust source file. CI checks that each item cites its registered source and that the source actually contains a `#[cfg(test)]` module.

An empty `benchmarks` array is meaningful: it says no benchmark evidence is currently registered for that item. Tooling should treat that as a visible evidence gap, not infer performance confidence from the complexity metadata alone.

When benchmark harnesses are added, their paths can be attached without changing the source-integration or provenance contract.

## What agents can do with this

A coding agent can now filter the registry before reading implementations. Examples:

```text
find deterministic kernels
find kernels with a no-allocation operation
find algorithms that do not mutate caller input
find an O(log n) prefix/range query structure
find a selection implementation and inspect its worst-case bound
find kernels whose benchmark evidence is still missing
```

The metadata is descriptive evidence for narrowing a search. It is not a substitute for opening the implementation, running consumer-specific benchmarks, or testing realistic workloads.

## Why operation-level metadata matters

A single complexity label for a whole data structure is usually misleading.

Examples in the current catalog:

- `UnionFind::find` is amortized O(alpha(n)) and mutates internal parent links through path compression.
- `LruCache::get` is expected O(1) but mutates recency state, while `peek` does not.
- `SparseSet::insert` is cheap when its sparse slot already exists but can spend time and memory proportional to a large newly materialized key range.
- `top_k_smallest` currently clones the entire input, so its extra space is O(n), not O(k).

The registry records those distinctions instead of compressing them into a crate-level slogan.

## Relationship to source ownership

Characteristics do not affect provenance. After a kernel is copied, the consumer owns its source and may deliberately change the implementation, complexity, allocation behavior, or determinism.

At that point the registry metadata describes the upstream base recorded in `.rust-kernels.lock.json`; it does not automatically describe the locally modified version. Consumer-specific tooling should treat locally divergent source as needing fresh local analysis or benchmarks.

That distinction is useful: provenance says **where the code came from**, while characteristics describe **what the upstream implementation is designed to do**.
