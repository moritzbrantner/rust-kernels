# Browser comparison with pmndrs/math

[`site/benchmarks.html`](../site/benchmarks.html) runs equivalent named workloads through the
rust-kernels WebAssembly boundary (`crates/web-demo`) and through
[pmndrs/math](https://github.com/pmndrs/math) in the same browser tab. It shows where the
JavaScript library or the Rust/WASM kernels are faster for the same inputs. It is **comparative
evidence only**: it is not a CI gate, it does not replace the Cargo-native Divan/Iai-Callgrind
evidence in [the performance contract](performance.md), and pmndrs/math is not a correctness
authority for any kernel.

## Competitor identity

pmndrs/math is published on npm as the unscoped package **`math`** (repository
`github:pmndrs/math`). The comparison pins `math@0.1.0`
(integrity `sha512-hq5KkLblFiR8OO8ZYqPK02mBqrXuxJ6hwvhL3ZM9OFkAR82/20n4H768SWl4yVVrufh8VijEm42D4V9N8hDa2g==`,
npm `gitHead` `c6713e38dd86de6e3e5bf98b94e22c2a29e4a709`).

- It is a development-only dependency of the private, unpublished
  [`benchmarks/browser`](../benchmarks/browser/package.json) package with a committed lockfile.
  No crate depends on it and no source is copied from it.
- The deployed Pages page loads the same pinned version at run time from
  `https://cdn.jsdelivr.net/npm/math@0.1.0`, so the Pages build and CI stay npm-free. Pass
  `?competitor=<base URL>` to load another copy.
- Every result records the competitor `name`, `version` and source URL, the rust-kernels
  revision from `pages-dist/build-info.json` (written by `scripts/build-pages.sh`), the user agent
  and the timing profile. The headless runner adds the lockfile integrity and an environment
  fingerprint.

## Running it

In a browser: run `bash scripts/build-pages.sh`, serve `pages-dist/` with any static server, open
`benchmarks.html`, choose suites and profile, and press **Run**. Nothing runs automatically
unless `?autorun=1` is given.

Headless (prints Markdown tables and writes raw JSON to `benchmarks/browser/results/`):

```sh
npm ci --prefix benchmarks/browser        # installs only the pinned `math` package
node benchmarks/browser/run.mjs --profile full --suites noise,shapes --repeat 3
```

The runner builds the site with `scripts/build-pages.sh` unless `--no-build` is given. It serves
`pages-dist/` and the installed competitor from one local origin with cross-origin isolation
headers (finer `performance.now()` resolution) and drives Chromium through an already installed
Playwright (`require("playwright")`; Playwright is intentionally not a dependency). `--filter
<substring>` limits scenarios and `--repeat N` reports the per-row median over N fresh tabs.

## Method

- **Batching.** Each Rust export processes a whole workload in one call over buffers in linear
  memory (`bench_alloc`/`bench_free`), so per-call boundary overhead is not part of throughput.
- **Startup vs steady state.** WASM fetch/compile/instantiate time and the competitor module
  import time are reported once per run, separately from the scenario rows. One-time setup that
  the APIs expose (seeded noise tables) is its own `setup/*` row.
- **Columns.** *rust kernel* has the inputs already in linear memory; *rust e2e* adds copying the
  inputs from JavaScript typed arrays into linear memory and the outputs back out; *pmndrs*
  calls the library function per item over the same data. *pmndrs/rust* is the pmndrs time
  divided by the rust kernel time, so values above 1 mean rust-kernels is faster.
- **Timing.** After warm-up batches, each sample repeats the batch until it lasts at least
  10 ms (`full` profile) and the row reports the median of up to 11 samples. Inputs are generated
  by a seeded mulberry32 stream outside the measured region.
- **Sanity cross-check.** Rows carry a check that is meaningful for the workload (see each
  section) and confirm the Rust output is identical across two runs. Disagreement is flagged on
  the page; it is never treated as a Rust failure without a Cargo test.

## Noise workloads (#114)

Both engines get seed value `1337`, the same coordinates and the same sizes:

- `*/scattered`: 1,024 / 16,384 / 262,144 uniform points in `[-256, 256)^N`.
- `*2d/grid`: 64², 256², 1024² row-major grids; `*3d/grid`: 16³, 32³, 64³ volumes (x fastest),
  at `origin + index / 16` with origin `(0.03125, -7.5, 3.25)`.

The seed-to-table derivations differ (rust-kernels: splitmix64 Fisher–Yates; pmndrs/math: the
classic table XOR-ed with the seed), so sampled values are not expected to match and are not
compared point by point. Worley semantics also differ: rust-kernels computes the exact `f1`, `f2`,
nearest offset and feature id with conservative ring expansion, while pmndrs/math returns `f1`
from the fixed `3^N` neighbourhood. Both rows report `f1` only, so the Rust column includes work
the competitor does not do. Row checks: both outputs finite, Worley `f1 >= 0`. Shared invariants:
Perlin is zero on integer lattice points and simplex is zero at the origin, for both engines.

### Recorded noise results

Environment: Chromium 153.0.8010.12 (Playwright, headless, cross-origin isolated), Node v22.23.4,
Linux 6.18.44 x86_64 VM, Intel Xeon @ 2.10 GHz × 4, 16 GiB, rustc 1.99.1,
`wasm32-unknown-unknown` release profile (workspace defaults, no LTO). Revision: base `e33d883`
plus the working-tree changes of the commit that adds this page. `full` profile, `--repeat 3`,
median per row. Startup (first run): 48 KiB WASM, compile 1.8 ms, instantiate 0.1 ms; competitor
fetch + import 88 ms.

| scenario | size | rust kernel ns/item | rust e2e ns/item | pmndrs ns/item | pmndrs/rust |
| --- | --- | ---: | ---: | ---: | ---: |
| setup/permutation-create | 1 | 1317.1 | — | 20097.7 | 15.26× |
| perlin2d/scattered | 1024 | 53.7 | 55.0 | 12.8 | 0.24× |
| perlin2d/scattered | 16384 | 67.1 | 57.8 | 17.5 | 0.26× |
| perlin2d/scattered | 262144 | 59.7 | 61.4 | 17.8 | 0.30× |
| perlin2d/grid | 64x64 | 30.5 | 28.9 | 12.6 | 0.41× |
| perlin2d/grid | 256x256 | 30.4 | 29.3 | 15.1 | 0.50× |
| perlin2d/grid | 1024x1024 | 29.7 | 30.6 | 15.5 | 0.52× |
| simplex2d/scattered | 1024 | 26.1 | 25.4 | 31.0 | 1.19× |
| simplex2d/scattered | 16384 | 52.1 | 51.6 | 44.5 | 0.85× |
| simplex2d/scattered | 262144 | 53.5 | 57.2 | 48.5 | 0.91× |
| simplex2d/grid | 64x64 | 26.1 | 26.3 | 29.4 | 1.13× |
| simplex2d/grid | 256x256 | 40.5 | 36.1 | 47.3 | 1.17× |
| simplex2d/grid | 1024x1024 | 26.2 | 26.6 | 30.6 | 1.17× |
| worley2d-f1/scattered | 1024 | 86.5 | 86.5 | 51.0 | 0.59× |
| worley2d-f1/scattered | 16384 | 135.4 | 134.5 | 70.9 | 0.52× |
| worley2d-f1/scattered | 262144 | 139.1 | 143.9 | 79.1 | 0.57× |
| worley2d-f1/grid | 64x64 | 98.2 | 79.3 | 46.1 | 0.47× |
| worley2d-f1/grid | 256x256 | 98.7 | 95.5 | 51.5 | 0.52× |
| worley2d-f1/grid | 1024x1024 | 110.3 | 101.2 | 52.7 | 0.48× |
| perlin3d/scattered | 1024 | 51.8 | 53.8 | 39.8 | 0.77× |
| perlin3d/scattered | 16384 | 66.1 | 60.6 | 52.0 | 0.79× |
| perlin3d/scattered | 262144 | 64.1 | 68.6 | 45.7 | 0.71× |
| perlin3d/grid | 16x16x16 | 48.5 | 59.1 | 40.2 | 0.83× |
| perlin3d/grid | 32x32x32 | 51.1 | 50.7 | 41.6 | 0.81× |
| perlin3d/grid | 64x64x64 | 51.1 | 53.6 | 37.2 | 0.73× |
| simplex3d/scattered | 1024 | 36.2 | 58.9 | 43.8 | 1.21× |
| simplex3d/scattered | 16384 | 81.8 | 81.7 | 67.4 | 0.82× |
| simplex3d/scattered | 262144 | 83.4 | 87.9 | 72.2 | 0.87× |
| simplex3d/grid | 16x16x16 | 38.6 | 37.7 | 41.8 | 1.08× |
| simplex3d/grid | 32x32x32 | 40.5 | 47.0 | 45.5 | 1.13× |
| simplex3d/grid | 64x64x64 | 43.4 | 37.9 | 43.7 | 1.01× |
| worley3d-f1/scattered | 1024 | 252.8 | 241.8 | 173.3 | 0.69× |
| worley3d-f1/scattered | 16384 | 276.3 | 277.5 | 249.1 | 0.90× |
| worley3d-f1/scattered | 262144 | 283.0 | 288.1 | 197.1 | 0.70× |
| worley3d-f1/grid | 16x16x16 | 219.6 | 214.6 | 151.3 | 0.69× |
| worley3d-f1/grid | 32x32x32 | 243.5 | 271.5 | 169.8 | 0.70× |
| worley3d-f1/grid | 64x64x64 | 211.9 | 219.6 | 169.0 | 0.80× |

Every row passed its check and every shared invariant held for both engines.

Reading: pmndrs/math is faster for Perlin (2–4× on scattered 2D points) and Worley (1.1–2×,
partly the exact search and `f2`/id work above); rust-kernels is ahead or level on 2D/3D simplex
grids and small scattered simplex batches; seeded table construction is ~15× faster in Rust
because pmndrs/math also materialises flat gradient tables. A local experiment with
`CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1` reduced the 2D Perlin grid
rows from ~30 to ~18 ns/item, so cross-crate inlining in the WASM build is a visible factor; the
recorded rows use the shipped Pages build. Expect ±30% run-to-run variation on shared machines.

## Collision and shape workloads (#118)

Only operations with matching semantics are compared. Inputs are rounded to f32 (the
rust-kernels input precision for these shapes) and both engines read identical values;
pmndrs/math receives them in its idiomatic tuple/object layout, prepared once outside the
measured region, while Rust reads packed f32 records from linear memory.

| scenario | rust-kernels | pmndrs/math |
| --- | --- | --- |
| `ray-aabb/*` | `ray_aabb(...)` with `enter_distance <= 1000` | `raycast3.intersectsBox3(o, d, 1000, box)` |
| `ray-triangle/*` | `ray_triangle(..., 1000, TriangleCulling::None)` distance | `raycast3.intersectsTriangle(..., 1000, ..., false)`, `fraction * 1000` |
| `sphere-aabb/mixed` | `primitives::sphere_aabb(...).overlaps` | `box3.intersectsSphere` |
| `aabb-aabb/mixed` | `aabb_aabb(...).overlaps` | `box3.intersectsBox3` |
| `obb-obb/mixed` | `obb3::obb3_sat(...).overlaps` | `obb3.intersectsOBB3` |
| `quickhull2d/*` | `hull::quickhull2` | `quickhull2` |
| `quickhull3d/*` | `hull::quickhull3` | `quickhull3` |
| `triangulate-polygon2/star` | `polygon2::triangulate_polygon2(..., SimplicityCheck::AssumeSimple, ...)` | `triangulatePolygon2` |

Workloads (seeded, per pair unless noted):

- Pair queries at 1,024 / 16,384 / 262,144 pairs. Rays start 15–25 units from their target and
  either aim at it with a ±3 (box) or ±0.75 (triangle) offset (`aimed`, ~50% / ~22% hits) or point
  in a uniformly random direction (`scattered`, <1% hits). Boxes have half extents 0.5–3, spheres
  radius 0.5–3 within ±6 of a box centre, OBBs random XYZ rotations in `[-π, π)`; pmndrs/math gets
  the same OBB rotation matrix that `Obb3::axes` derives (Rz·Ry·Rx), precomputed once.
- Hulls of 1,024 / 16,384 / 131,072 uniform points in a disk or ball, plus 1,024 / 16,384 points on
  a circle or sphere (every point is a hull vertex). ns/item is per input point.
- Ear clipping of star-shaped simple polygons with 64 / 512 / 2,048 vertices (radius 0.5–1 at
  equal angles). ns/item is per vertex. Rust skips its O(n²) simplicity validation because the
  competitor does not validate.

API shape matters in these rows: rust-kernels' `obb3_sat` takes Euler angles, recomputes the
rotation with `sin`/`cos` per call and returns the full 15-axis relation (overlap per axis,
critical axis), while pmndrs/math consumes a precomputed matrix and returns a boolean with early
exit; similarly `Ray3::new` renormalises every direction and the ray queries return full hit
records. Rust scratch buffers for hulls and polygons are reused across calls; pmndrs/math
allocates its outputs per call.

Row checks: hit/no-hit agreement per pair (flagged above 0.1% disagreement), the largest
ray-triangle distance difference among agreed hits (flagged above 1e-4), equal hull vertex and
triangle counts, `n - 2` triangles from both engines, and Rust run-to-run determinism. Points
rounded onto a circle are nearly collinear, so which survive in a 2D hull depends on each
engine's collinearity tolerance; that count difference is reported as informational. Shared
invariants: an axis ray hits the unit box and an offset ray misses; touching AABBs overlap.

### Recorded shape results

Same environment as the noise results. Revision: `377279c` (the #114 commit) plus the
working-tree changes of the commit that adds these workloads. `full` profile, `--repeat 3`,
median per row; cross-check details are from the first run (all three runs passed). Startup:
164 KiB WASM, compile 1.3 ms, instantiate 0.1 ms; competitor fetch + import 58 ms.

| scenario | size | rust kernel ns/item | rust e2e ns/item | pmndrs ns/item | pmndrs/rust | cross-check (first run) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ray-aabb/aimed | 1024 | 67.4 | 56.3 | 10.8 | 0.16× | hits rust 50.7% js 50.7%; 0 disagreement(s) |
| ray-aabb/aimed | 16384 | 97.4 | 103.8 | 62.2 | 0.64× | hits rust 50.4% js 50.4%; 0 disagreement(s) |
| ray-aabb/aimed | 262144 | 103.1 | 117.9 | 84.1 | 0.82× | hits rust 50.6% js 50.6%; 0 disagreement(s) |
| ray-aabb/scattered | 1024 | 29.7 | 34.2 | 13.2 | 0.44× | hits rust 0.4% js 0.4%; 0 disagreement(s) |
| ray-aabb/scattered | 16384 | 73.1 | 74.2 | 55.9 | 0.77× | hits rust 0.4% js 0.4%; 0 disagreement(s) |
| ray-aabb/scattered | 262144 | 76.8 | 77.7 | 54.6 | 0.71× | hits rust 0.4% js 0.4%; 0 disagreement(s) |
| ray-triangle/aimed | 1024 | 47.5 | 47.1 | 14.9 | 0.31× | hits rust 23.4% js 23.4%; 0 disagreement(s); max \|Δdistance\| 9.9e-7 |
| ray-triangle/aimed | 16384 | 68.8 | 79.5 | 47.2 | 0.69× | hits rust 21.9% js 21.9%; 0 disagreement(s); max \|Δdistance\| 9.5e-7 |
| ray-triangle/aimed | 262144 | 74.2 | 78.1 | 69.0 | 0.93× | hits rust 22.1% js 22.1%; 0 disagreement(s); max \|Δdistance\| 1.1e-6 |
| ray-triangle/scattered | 1024 | 45.4 | 47.2 | 19.2 | 0.42× | hits rust 0.1% js 0.1%; 0 disagreement(s); max \|Δdistance\| 1.2e-7 |
| ray-triangle/scattered | 16384 | 66.4 | 73.2 | 41.6 | 0.63× | hits rust 0.0% js 0.0%; 0 disagreement(s); max \|Δdistance\| 5.8e-7 |
| ray-triangle/scattered | 262144 | 65.1 | 72.7 | 56.6 | 0.87× | hits rust 0.0% js 0.0%; 0 disagreement(s); max \|Δdistance\| 8.0e-7 |
| sphere-aabb/mixed | 1024 | 18.1 | 20.0 | 12.0 | 0.66× | hits rust 19.2% js 19.2%; 0 disagreement(s) |
| sphere-aabb/mixed | 16384 | 38.5 | 38.1 | 57.5 | 1.49× | hits rust 18.7% js 18.7%; 0 disagreement(s) |
| sphere-aabb/mixed | 262144 | 41.8 | 44.7 | 50.4 | 1.20× | hits rust 18.7% js 18.7%; 0 disagreement(s) |
| aabb-aabb/mixed | 1024 | 19.1 | 22.4 | 11.0 | 0.58× | hits rust 20.7% js 20.7%; 0 disagreement(s) |
| aabb-aabb/mixed | 16384 | 43.5 | 44.6 | 57.4 | 1.32× | hits rust 20.6% js 20.6%; 0 disagreement(s) |
| aabb-aabb/mixed | 262144 | 41.8 | 49.1 | 35.3 | 0.84× | hits rust 21.1% js 21.1%; 0 disagreement(s) |
| obb-obb/mixed | 1024 | 2453.6 | 2531.7 | 33.7 | 0.01× | hits rust 22.7% js 22.7%; 0 disagreement(s) |
| obb-obb/mixed | 16384 | 2541.8 | 2681.3 | 71.3 | 0.03× | hits rust 24.2% js 24.2%; 0 disagreement(s) |
| obb-obb/mixed | 262144 | 2789.3 | 2741.5 | 59.3 | 0.02× | hits rust 24.0% js 24.0%; 0 disagreement(s) |
| quickhull2d/disk | 1024 | 20.9 | 21.4 | 21.2 | 1.01× | hull vertices rust 39 js 39 |
| quickhull2d/disk | 16384 | 38.8 | 35.2 | 37.0 | 0.95× | hull vertices rust 89 js 89 |
| quickhull2d/disk | 131072 | 38.1 | 43.1 | 45.5 | 1.19× | hull vertices rust 162 js 162 |
| quickhull2d/circle | 1024 | 216.8 | 211.9 | 159.6 | 0.74× | hull vertices rust 1021 js 1009 (near-collinear input: tolerance-dependent, informational) |
| quickhull2d/circle | 16384 | 288.5 | 291.4 | 231.6 | 0.80× | hull vertices rust 13236 js 8154 (near-collinear input: tolerance-dependent, informational) |
| quickhull3d/ball | 1024 | 127.0 | 138.3 | 260.6 | 2.05× | hull vertices rust 142 js 142; triangles rust 280 js 280 |
| quickhull3d/ball | 16384 | 135.0 | 141.8 | 246.7 | 1.83× | hull vertices rust 536 js 536; triangles rust 1068 js 1068 |
| quickhull3d/ball | 131072 | 141.9 | 140.0 | 333.4 | 2.35× | hull vertices rust 1661 js 1661; triangles rust 3318 js 3318 |
| quickhull3d/sphere | 1024 | 772.1 | 792.5 | 1134.0 | 1.47× | hull vertices rust 1024 js 1024; triangles rust 2044 js 2044 |
| quickhull3d/sphere | 16384 | 1098.3 | 1022.9 | 2853.7 | 2.60× | hull vertices rust 16384 js 16384; triangles rust 32764 js 32764 |
| triangulate-polygon2/star | 64 | 184.3 | 187.8 | 2403.6 | 13.04× | triangles rust 62 js 62 (expected 62) |
| triangulate-polygon2/star | 512 | 1002.2 | 1055.0 | 17690.4 | 17.65× | triangles rust 510 js 510 (expected 510) |
| triangulate-polygon2/star | 2048 | 3676.8 | 3459.5 | 65349.1 | 17.77× | triangles rust 2046 js 2046 (expected 2046) |

Every pair scenario agreed on every hit/no-hit decision and ray-triangle distances agreed to
about 1e-6. Reading: rust-kernels is 13–18× faster at ear clipping and 1.5–2.6× faster at 3D
hulls, level on 2D disk hulls, and ahead on sphere–AABB from 16,384 pairs (AABB–AABB is mixed).
pmndrs/math is faster on ray queries (1.1–6×; largest at 1,024 pairs, where its prepared objects
stay cache-resident) and far faster on OBB–OBB (30–80×), where the Rust API's per-call
trigonometry and full diagnostic relation dominate; native `cargo bench -p geometry-kernels
--bench obb3_sat` shows ~600 ns per call on this machine, so the gap is the kernel's contract
rather than the WASM boundary. The e2e columns stay within noise of the kernel columns: copying
packed inputs into linear memory is cheap next to the queries.
