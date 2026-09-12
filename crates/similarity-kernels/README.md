# similarity-kernels

Dependency-light similarity primitives for reuse across search, retrieval, deduplication, and comparison workloads.

The crate owns mechanism only. Text normalization, tokenization, query interpretation, score weighting, embeddings, and approximate-index policy remain consumer concerns.

## Exact sequence distance

`levenshtein` computes generic edit distance over slices of `T: Eq` using `O(min(m, n))` auxiliary memory.

## Sorted-unique set operations

The `sorted_unique_*` functions operate on caller-owned slices that are already strictly sorted and duplicate-free. They intentionally do not normalize, sort, or deduplicate inputs. This keeps the kernels allocation-light and makes preprocessing policy explicit at the caller boundary.

Available operations include union, intersection, left-minus-right difference, symmetric difference, and count-only intersection/union paths.

## Jaccard

`jaccard_similarity_sorted_unique` and `jaccard_distance_sorted_unique` reuse the count-only sorted-set path. Two empty sets have similarity `1.0` and distance `0.0`.

## Shingling

`shingles` exposes overlapping fixed-width windows as borrowed slices, so callers can define byte, token, or other sequence features without allocation or text-specific policy. Width zero and widths larger than the input deliberately produce no shingles.

## Rolling hashes

`rolling_hashes` hashes byte shingles with a stable wrapping-`u64` polynomial. The first window costs `O(k)` for width `k`; each subsequent overlapping window updates in `O(1)`. The hash is deterministic and non-cryptographic. Equal hashes are only candidate evidence because collisions are possible; consumers that require exact equality must compare the original bytes.

## MinHash

`minhash_signature` accepts caller-supplied stable `u64` feature hashes and produces a deterministic signature. The seed and signature length are part of the evidence identity: `minhash_jaccard_estimate` refuses to compare incompatible signatures instead of silently producing a number. Input order and duplicate hashes do not affect a signature.

MinHash is an approximation of Jaccard similarity, not exact equality evidence. Larger signatures usually reduce estimator variance but do not establish a universal error bound for an individual pair.

## SimHash

`simhash64` and `simhash64_weighted` turn stable feature hashes into a compact 64-bit fingerprint. Repeated features contribute repeatedly; weighted SimHash accepts explicit non-negative integer weights, and zero weight has no effect. `hamming_distance64` exposes exact fingerprint distance.

SimHash Hamming distance is a locality signal. The crate deliberately defines no universal near-duplicate threshold; consumers must calibrate thresholds against their own feature producer and corpus.
