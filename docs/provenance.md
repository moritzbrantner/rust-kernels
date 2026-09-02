# Source provenance contract

`rust-kernels` supports two first-class ways to consume a kernel:

1. depend on a workspace crate through Cargo; or
2. copy registry source into the consumer and let the consumer own that code.

The second mode is intentionally not anonymous vendoring. A consumer records where the copied source came from in `.rust-kernels.lock.json`.

## Why the lock exists

Copied source is expected to diverge. A search project may specialize BM25 for its document representation; a graphics project may fuse a geometry kernel with its own memory layout; an audio project may add SIMD that only makes sense for one target.

That local ownership is legitimate. The provenance lock preserves the upstream base so future tooling can still answer:

- Which registry item did this file start from?
- Which immutable upstream revision was used?
- Were these exact source bytes copied?
- Has the consumer modified or deleted the copied file?
- What old upstream source should be used as the base of a three-way update?

The lock is metadata about ancestry, not a declaration that copied files must remain unchanged.

## Registry source identity

`registry.json` declares the canonical repository and that installations are pinned to full Git commit SHAs:

```json
{
  "source": {
    "repository": "https://github.com/moritzbrantner/rust-kernels",
    "revisionType": "git-commit"
  }
}
```

The registry does not contain its own current commit SHA. Doing so would make every commit change the value it is trying to identify. The immutable revision is resolved when a consumer installs an item.

## Consumer lock format

The normative schema lives in this repository as [`provenance.schema.json`](../provenance.schema.json) and is published at `https://raw.githubusercontent.com/moritzbrantner/rust-kernels/main/provenance.schema.json`. Consumer locks use that published URI, so validation does not depend on copying a schema file into every consumer repository.

A consumer should commit `.rust-kernels.lock.json` alongside the copied source. A lock entry is per registry item, not global, so different items may advance independently:

```json
{
  "$schema": "https://raw.githubusercontent.com/moritzbrantner/rust-kernels/main/provenance.schema.json",
  "version": 1,
  "registry": {
    "name": "rust-kernels",
    "repository": "https://github.com/moritzbrantner/rust-kernels"
  },
  "items": [
    {
      "name": "search-kernels",
      "revision": "0123456789abcdef0123456789abcdef01234567",
      "registrySha256": "<sha256 of registry.json at that revision>",
      "files": [
        {
          "source": "crates/search-kernels/src/selection.rs",
          "target": "crates/search-kernels/src/selection.rs",
          "sourceSha256": "<sha256 of the upstream source bytes>"
        }
      ]
    }
  ]
}
```

The important fields are:

- `revision`: the exact 40-character Git commit used as the upstream source snapshot.
- `registrySha256`: the SHA-256 of `registry.json` at that revision.
- `source`: the path in `rust-kernels`.
- `target`: the path chosen by the registry item for the consumer.
- `sourceSha256`: the SHA-256 of the exact upstream bytes that were copied.

Timestamps are deliberately omitted from the core lock format. They do not help reconstruct ancestry and would make otherwise identical installs produce different lock files.

## Local divergence

For each locked file, a consumer can compare the current target hash with `sourceSha256`:

| State | Meaning |
| --- | --- |
| `clean` | Target bytes still match the upstream snapshot exactly. |
| `modified` | The consumer has intentionally or accidentally changed the copied file. |
| `missing` | The copied file is no longer present. |

`modified` is not an error by itself. It is an explicit, supported state.

The reference helper reports these states:

```bash
python3 scripts/source_registry.py status --root ../consumer
```

Add `--require-clean` only when a particular workflow intentionally requires an unmodified source snapshot.

## Copying an item

The repository is not a CLI product. `registry.json` and the schemas are the integration contract, and other tools or agents may consume them directly.

For testing the contract and for simple local use, `scripts/source_registry.py` is a small standard-library reference helper. From a clean `rust-kernels` checkout it can copy an item and its registry dependencies:

```bash
python3 scripts/source_registry.py install search-kernels --root ../consumer
```

The helper:

1. resolves `registryDependencies`;
2. resolves the current full Git commit SHA;
3. rejects a dirty upstream snapshot;
4. hashes `registry.json` and every copied source file;
5. refuses to overwrite an existing target whose bytes have diverged;
6. copies the source; and
7. writes or updates `.rust-kernels.lock.json`.

This behavior is intentionally conservative. Re-install is not an update mechanism.

## Safe update protocol

An updater can use the provenance contract without hidden state. For each locked file it has three inputs:

- **base**: source at the recorded `revision`;
- **ours**: the current consumer target;
- **theirs**: source at the candidate new upstream revision.

That enables deterministic update behavior:

1. If `ours == base`, replace it with `theirs`.
2. If `theirs == base`, keep the local file unchanged.
3. If both changed, perform or propose a three-way merge.
4. Run the consumer's tests and benchmarks.
5. Update the lock only after the new upstream base has been accepted.

The lock should continue to describe the original upstream base while unresolved local/upstream conflicts exist. Do not rewrite `sourceSha256` to the consumer's modified hash; doing that would destroy the information required for a future three-way merge.

## Relationship to application experimentation

Consumer repositories are expected to be proving grounds. A project may copy a kernel, specialize it, benchmark the result, and keep the specialization local.

If the change becomes broadly useful, the implementation can be proposed back to `rust-kernels`. Once accepted, other consumers can advance their provenance lock to the new upstream revision.

This creates an explicit cycle:

```text
rust-kernels registry
        |
        v
consumer-owned source
        |
        v
specialize + test + benchmark
        |
        v
generalizable improvement
        |
        v
rust-kernels
```

The registry therefore acts both as a distribution catalog and as the stable ancestry boundary for source-level capability internalization.
