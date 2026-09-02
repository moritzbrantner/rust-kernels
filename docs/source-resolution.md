# Source conflict resolution

The provenance lock and update planner make copied source intentionally editable. When both the consumer and `rust-kernels` change the same copied file, the planner reports `both-changed` instead of overwriting either side.

The conflict-resolution contract turns that state into an explicit handoff for a human or coding agent.

## 1. Export exact merge evidence

From the candidate `rust-kernels` checkout:

```bash
python3 scripts/source_resolution.py export \
  --root ../consumer \
  --out ../consumer/.rust-kernels-merge
```

You can place registry item names before the options to narrow the export.

The output directory must be empty. It receives:

```text
merge-bundle.json
conflicts/
  <registry-item>/
    <consumer-target>.base
    <consumer-target>.ours
    <consumer-target>.theirs
```

`base` is reconstructed from the revision recorded in `.rust-kernels.lock.json`. `ours` is the consumer-owned file at export time. `theirs` is the source from the current candidate checkout.

The bundle is deterministic: it contains no timestamp or machine-specific absolute path.

## Merge bundle contract

[`merge-bundle.schema.json`](../merge-bundle.schema.json) describes the manifest. As with provenance locks, a generated bundle pins `$schema` to the exact candidate Git revision.

Each conflict records:

- registry item, upstream source path, and consumer target path;
- recorded base revision;
- SHA-256 of `base`, `ours`, and `theirs`;
- relative paths to the three materialized artifacts;
- candidate revision and candidate `registry.json` hash at bundle level.

A consumer or agent therefore does not need to rediscover ancestry or guess which files should be compared.

## 2. Resolve the consumer-owned file

A human or coding agent edits the actual consumer target. The three exported artifacts are evidence and inputs; they are not the file that becomes authoritative.

For example, an agent can receive:

```text
base   = conflicts/search-kernels/.../selection.rs.base
ours   = conflicts/search-kernels/.../selection.rs.ours
theirs = conflicts/search-kernels/.../selection.rs.theirs
target = <consumer>/crates/search-kernels/src/selection.rs
```

The agent can then:

1. understand both changes;
2. edit the consumer target;
3. run focused tests or benchmarks;
4. run the consumer's full deterministic gate when appropriate.

`rust-kernels` deliberately does not prescribe the consumer's test command. The consumer repository owns that compatibility decision.

## 3. Explicitly accept the resolution

After the target has been resolved and verified:

```bash
python3 scripts/source_resolution.py accept search-kernels \
  --root ../consumer \
  --bundle ../consumer/.rust-kernels-merge/merge-bundle.json
```

`accept` is not an automatic merge command. It is the explicit statement:

> this consumer source has been reconciled with this candidate upstream revision; preserve the consumer result and use the candidate source as the new future merge base.

Before changing provenance it verifies:

- the bundle belongs to the same registry;
- the bundle candidate revision is still the current checkout;
- the candidate registry hash still matches;
- bundle artifacts still match their recorded hashes;
- `base` still matches Git history;
- `theirs` still matches the candidate checkout;
- registry layout and dependencies have not changed underneath the operation;
- every current `both-changed` file has bundle evidence.

If any of those checks fails, acceptance stops before applying sibling updates or advancing the lock.

## What acceptance changes

For a selected item:

- resolved `both-changed` target files remain exactly as the consumer left them;
- `converged` targets that now equal upstream remain unchanged and become clean;
- `upstream-only` sibling files are copied from the candidate;
- `local-only` sibling specializations remain local;
- the selected provenance entry advances to the candidate revision and candidate source hashes.

A merged target can therefore remain `modified` after acceptance. That is expected: it means “locally owned source based on this newer upstream revision,” not “untracked divergence.”

## Deliberately keeping the local side

If a conflict target is still byte-for-byte identical to the exported `ours`, `accept` rejects it by default. This catches the common failure mode where a conflict was exported but never actually resolved.

If the intended resolution is specifically to ignore the upstream change and carry the local side forward, make that decision explicit:

```bash
python3 scripts/source_resolution.py accept search-kernels \
  --root ../consumer \
  --bundle ../consumer/.rust-kernels-merge/merge-bundle.json \
  --allow-ours
```

This advances the upstream base while preserving the local bytes. A future upstream change will then compare against the newly accepted base rather than the older ancestor.

## Why this is useful for coding agents

The merge bundle is a deterministic agent handoff boundary. Instead of asking an agent to inspect Git history, infer provenance, and rediscover which version is which, tooling can give it exact files and machine-readable metadata.

That keeps the expensive reasoning step focused on the actual semantic merge:

```text
deterministic tooling
        |
        v
classify conflict
        |
        v
materialize base / ours / theirs
        |
        v
human or coding agent resolves semantics
        |
        v
consumer tests / benchmarks
        |
        v
explicit provenance acceptance
```

Automatic semantic merging can be added later, but it should build on this evidence rather than replacing it.
