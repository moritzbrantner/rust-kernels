# Capsule intersection normals

When capsule skeletons intersect, translating along their center delta can leave
both solids overlapping. The sphere/capsule and capsule/capsule analytic paths
choose a direction perpendicular to every nonzero skeleton for these contacts.
Parallel skeletons use a deterministic perpendicular basis; two collapsed
skeletons retain the sphere center-distance direction. Ordinary resolved deltas
retain the closest-point normal.

The tie budget is `64 * f64::EPSILON * characteristic_length`, where the length
includes both shape radii and the largest absolute pose coordinate. This accounts
for subtracting world-space endpoints. A resolved nonnegative separation always
retains its distance normal, regardless of that budget. Within a tie, depth uses
the closest-point delta projected onto the selected normal; the witnesses and
separation describe the same direction. This may choose a conservative clearance
direction rather than a unique minimum-translation direction. Very short nonzero
skeletons use the segment/segment closest-point routine instead of collapsing at
a fixed squared-length cutoff.

The regression oracle independently constructs sphere/capsule support planes.
After moving the second body by the reported penetration plus a scale-dependent
margin, a positive support-plane gap certifies that the entire solids separate.
Fixtures cover rotated and translated intersections, 1024 deterministic seeded
cases, parallel and nearly parallel axes, skeleton endpoints, collapsed skeletons,
small nonzero skeletons, near-zero deltas and resolved misses at large poses.
Dimensional scales include 1e-6, 1 and 1e6. Generic GJK was an additional diagnosis
probe: it identified the old direction as intersecting, but a crossing-capsule
clearance probe exhausted its iteration limit. That unresolved result is not a
reference success or a miss; the support-plane certificate supplies independent
clearance evidence for this repair.

A Cargo-native allocator test executes 4096 warmed queries without allocations or
reallocations. Divan measures two complete degenerate contact calls per sample.
Existing dispatch/support/axis/sweep counters remain unchanged and do not count
internal segment-distance operations. Full contact-row differential coverage and
expanded work accounting remain separate follow-up acceptance requirements.

```sh
cargo test -p geometry-kernels --test capsule_contact_normals
cargo test -p geometry-kernels --test allocation_regressions
cargo bench -p geometry-kernels --bench primitive3 -- capsule_skeleton_contacts
```
