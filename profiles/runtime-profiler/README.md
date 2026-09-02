# Runtime performance evidence

These scenarios are deterministic representative workloads for the kernel families where runtime cost matters most today.

- `search.json` exercises radix sorting and deterministic quickselect over a fixed generated corpus.
- `lru.json` exercises insert, lookup, promotion, and eviction behavior under deterministic cache churn.
- `traversal.json` exercises breadth-first and depth-first traversal over a deterministic synthetic graph.

Capture evidence with:

```sh
bash scripts/runtime-profile.sh search .artifacts/runtime-profiler/search-001
bash scripts/runtime-profile.sh lru .artifacts/runtime-profiler/lru-001
bash scripts/runtime-profile.sh traversal .artifacts/runtime-profiler/traversal-001
```

The workload launcher builds the release example only when the binary is missing or stale, then executes the binary directly. The runtime-profiler warmup therefore prepares the workload while measured iterations avoid compile-time noise.

These captures are evidence, not absolute performance gates. Baseline/candidate comparison belongs to Moonlight; `runtime-profiler` owns collection. Preserve old bundles instead of overwriting them so regressions remain inspectable.
