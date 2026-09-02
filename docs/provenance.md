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

The normative schema lives in this repository as [`provenance.schema.json`](../provenance.schema.json). `registry.json` exposes the current published schema location for discovery, but a newly created consumer lock points to the schema file at the exact Git revision used for the install. That makes the schema reference immutable as well as self-describing.

For example, an install from revision `0123456789abcdef0123456789abcdef01234567` records:

```json
{
  "$schema": "https://raw.githubusercontent.com/moritzbrantner/rust-kernels/0123456789abcdef0123456789abcdef01234567/provenance.schema.json",
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

A consumer should commit `.rust-kernels.lock.json` alongside the copied source. A lock entry is per registry item, not global, so different items may advance independently. The lock-level schema URI does not need to move when another item advances as long as the lock remains on provenance format version 1.

The important fields are:

- `$schema`: an immutable Git-revision URL for the provenance schema used when the lock was created.
- `revision`: the exact 40-character Git commit used as the upstream source snapshot for an item.
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
4. pins the consumer lock's schema URI to that Git revision;
5. hashes `registry.json` and every copied source file;
6. refuses to overwrite an existing target whose bytes have diverged;
7. copies the source; and
8. writes or updates `.rust-kernels.lock.json`.

This behavior is intentionally conservative. Re-install is not an update mechanism.

## Safe update protocol

An updater uses three real inputs for every copied file:

- **base**: the upstream source reconstructed with `git show` at the recorded item revision;
- **ours**: the current consumer-owned target file;
- **theirs**: the source from the candidate `rust-kernels` checkout.

`scripts/source_update.py` is the reference implementation of that comparison:

```bash
python3 scripts/source_update.py plan --root ../consumer
```

The planner does not infer ancestry from filenames or from the current registry. Before comparing source it verifies the recorded `registrySha256`, reconstructs the historical registry, checks that the lock's source/target mapping matches that historical registry, and verifies each recorded `sourceSha256` against the reconstructed base bytes.

### File states

| State | Base / ours / theirs relationship | Automatic action |
| --- | --- | --- |
| `unchanged` | all three are identical | keep file; provenance may advance |
| `upstream-only` | ours = base, theirs changed | replace ours with theirs |
| `local-only` | theirs = base, ours changed | keep the local specialization |
| `converged` | ours = theirs, both differ from base | keep file and adopt new upstream base |
| `both-changed` | ours and theirs both diverged differently | manual or agent three-way merge |
| `missing` | consumer target is absent | manual decision |
| `base-mismatch` | reconstructed base does not match recorded source hash | stop; provenance is inconsistent |
| `registry-mismatch` | historical registry bytes do not match recorded registry hash | stop; provenance is inconsistent |
| `lock-mismatch` | lock mapping does not match its recorded historical registry | stop; provenance is inconsistent |
| `layout-changed` | candidate source/target mapping changed | manual migration |
| `dependencies-changed` | registry dependency set changed | manual dependency migration |
| `item-removed` | candidate registry no longer contains the item | manual decision |

`plan --require-safe` exits non-zero when any selected item has a state outside the first four.

### Applying a safe plan

```bash
python3 scripts/source_update.py apply --root ../consumer
```

`apply` first computes the complete selected plan. If any file or item needs manual resolution, it refuses the whole selected update before changing consumer files. If the complete plan is safe, it:

1. copies only `upstream-only` files;
2. preserves `local-only` specializations;
3. leaves `unchanged` and `converged` bytes alone;
4. advances the selected item lock entries to the candidate Git revision and candidate source hashes.

This means a local specialization can remain locally modified while its upstream base advances through revisions where upstream did not touch that file. If a later upstream revision changes the same file, the recorded newer base still provides the correct three-way merge ancestor.

The default is to plan or apply all locked items. Supplying item names narrows the operation. Updating the complete lock is preferable when related registry items evolve together; consumer tests and benchmarks remain the final compatibility gate.

### Deliberate conflict boundary

The reference helper does not yet auto-merge `both-changed` files. That is intentional. The provenance layer's job is to establish deterministic evidence first. A later coding-agent integration can receive the exact `base`, `ours`, and `theirs` inputs, propose a merge, run focused tests and benchmarks, and only then advance the lock.

The lock should continue to describe the original upstream base while unresolved local/upstream conflicts exist. Do not rewrite `sourceSha256` to the consumer's modified hash; doing that would destroy the information required for a future three-way merge.

## Relationship to application experimentation

Consumer repositories are expected to be proving grounds. A project may copy a kernel, specialize it, benchmark the result, and keep the specialization local.

If the change becomes broadly useful, the implementation can be proposed back to `rust-kernels`. Once accepted, other consumers can advance their provenance lock to the new upstream revision.

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
