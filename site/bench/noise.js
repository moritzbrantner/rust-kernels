// Noise workloads: rust-kernels (noise-kernels via crates/web-demo) vs pmndrs/math (`math` on npm).
//
// Both engines receive the same seed value, coordinates and workload sizes. Their seed-to-table
// derivations differ (splitmix64 shuffle vs XOR-ed fixed table), so values are not expected to
// match; the sanity checks only confirm finite, well-ranged output and shared lattice invariants.
// Worley semantics also differ: rust-kernels returns the exact f1/f2/id with ring expansion,
// pmndrs/math returns f1 from the fixed 3^N neighbourhood. Both report f1 here.

import { WasmBuffers, checksum, measure, range, rng, row, yieldToBrowser } from "./harness.js";

export const NOISE_SEED = 1337;

const POINT_SIZES = [1024, 16384, 262144];
const GRID2_SIZES = [64, 256, 1024];
const GRID3_SIZES = [16, 32, 64];
const GRID_STEP = 1 / 16;
const GRID_ORIGIN = [0.03125, -7.5, 3.25];
const POINT_EXTENT = 256;

const FIELDS = [
  { id: "perlin2d", dims: 2, js: (noise) => noise.perlin2d, wasm: "perlin2" },
  { id: "simplex2d", dims: 2, js: (noise) => noise.simplex2d, wasm: "simplex2" },
  { id: "worley2d-f1", dims: 2, js: (noise) => noise.worley2d, wasm: "worley2" },
  { id: "perlin3d", dims: 3, js: (noise) => noise.perlin3d, wasm: "perlin3" },
  { id: "simplex3d", dims: 3, js: (noise) => noise.simplex3d, wasm: "simplex3" },
  { id: "worley3d-f1", dims: 3, js: (noise) => noise.worley3d, wasm: "worley3" },
];

export const NOISE_EXPORTS = FIELDS.flatMap((field) => [
  `noise_${field.wasm}_points`,
  `noise_${field.wasm}_grid`,
]).concat(["noise_permutation_new", "noise_permutation_free"]);

function scatteredPoints(count, dims, seed) {
  const random = rng(seed);
  const values = new Float64Array(count * dims);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = (random() * 2 - 1) * POINT_EXTENT;
  }
  return values;
}

function describe(rustOut, jsOut, nonNegative) {
  const rust = range(rustOut);
  const js = range(jsOut);
  const finite = rust.finite && js.finite;
  const signOk = !nonNegative || (rust.min >= 0 && js.min >= 0);
  const fmt = (r) => `[${r.min.toFixed(3)}, ${r.max.toFixed(3)}]`;
  return {
    ok: finite && signOk,
    detail: `${finite ? "finite" : "NON-FINITE"}${nonNegative ? (signOk ? "; f1≥0" : "; NEGATIVE f1") : ""}; rust ${fmt(rust)} js ${fmt(js)}`,
  };
}

function pointScenario(wasm, competitor, field, count) {
  const dims = field.dims;
  const input = scatteredPoints(count, dims, 0x5eed + count + dims);
  const jsOut = new Float64Array(count);
  const rustCopy = new Float64Array(count);
  const generator = field.js(competitor.noise).create(NOISE_SEED);
  const sample = field.js(competitor.noise).sample;
  const run = wasm[`noise_${field.wasm}_points`];
  const perm = wasm.noise_permutation_new(NOISE_SEED);
  const buffers = new WasmBuffers(wasm);
  try {
    const inputAlloc = buffers.alloc(Float64Array, input.length);
    const outAlloc = buffers.alloc(Float64Array, count);
    const wasmInput = buffers.view(inputAlloc);
    const wasmOut = buffers.view(outAlloc);
    wasmInput.set(input);

    const rustKernel = () => run(perm, inputAlloc.ptr, count, outAlloc.ptr);
    const rustEndToEnd = () => {
      wasmInput.set(input);
      run(perm, inputAlloc.ptr, count, outAlloc.ptr);
      rustCopy.set(wasmOut);
    };
    const js =
      dims === 2
        ? () => {
            for (let index = 0; index < count; index += 1) {
              jsOut[index] = sample(generator, input[2 * index], input[2 * index + 1]);
            }
          }
        : () => {
            for (let index = 0; index < count; index += 1) {
              const base = 3 * index;
              jsOut[index] = sample(generator, input[base], input[base + 1], input[base + 2]);
            }
          };
    return { rustKernel, rustEndToEnd, js, rustOut: () => wasmOut, jsOut, cleanup: () => buffers.free(), perm };
  } catch (error) {
    buffers.free();
    wasm.noise_permutation_free(perm);
    throw error;
  }
}

function gridScenario(wasm, competitor, field, edge) {
  const dims = field.dims;
  const count = edge ** dims;
  const [x0, y0, z0] = GRID_ORIGIN;
  const step = GRID_STEP;
  const jsOut = new Float64Array(count);
  const rustCopy = new Float64Array(count);
  const generator = field.js(competitor.noise).create(NOISE_SEED);
  const sample = field.js(competitor.noise).sample;
  const run = wasm[`noise_${field.wasm}_grid`];
  const perm = wasm.noise_permutation_new(NOISE_SEED);
  const buffers = new WasmBuffers(wasm);
  const outAlloc = buffers.alloc(Float64Array, count);
  const wasmOut = buffers.view(outAlloc);
  const call =
    dims === 2
      ? () => run(perm, x0, y0, step, edge, edge, outAlloc.ptr)
      : () => run(perm, x0, y0, z0, step, edge, edge, edge, outAlloc.ptr);
  const js =
    dims === 2
      ? () => {
          let index = 0;
          for (let row = 0; row < edge; row += 1) {
            const y = y0 + row * step;
            for (let column = 0; column < edge; column += 1) {
              jsOut[index++] = sample(generator, x0 + column * step, y);
            }
          }
        }
      : () => {
          let index = 0;
          for (let layer = 0; layer < edge; layer += 1) {
            const z = z0 + layer * step;
            for (let row = 0; row < edge; row += 1) {
              const y = y0 + row * step;
              for (let column = 0; column < edge; column += 1) {
                jsOut[index++] = sample(generator, x0 + column * step, y, z);
              }
            }
          }
        };
  return {
    rustKernel: call,
    rustEndToEnd: () => {
      call();
      rustCopy.set(wasmOut);
    },
    js,
    rustOut: () => wasmOut,
    jsOut,
    cleanup: () => buffers.free(),
    perm,
  };
}

function runScenario(wasm, prepared, profile) {
  try {
    const rust = measure(prepared.rustKernel, profile);
    const first = checksum(prepared.rustOut());
    prepared.rustKernel();
    const deterministic = checksum(prepared.rustOut()) === first;
    const rustEndToEnd = measure(prepared.rustEndToEnd, profile);
    const js = measure(prepared.js, profile);
    return { rust, rustEndToEnd, js, deterministic, rustOut: prepared.rustOut().slice(), jsOut: prepared.jsOut };
  } finally {
    prepared.cleanup();
    wasm.noise_permutation_free(prepared.perm);
  }
}

function latticeSanity(wasm, competitor) {
  // Gradient Perlin noise is zero at integer lattice points; simplex is zero at the origin.
  const perm = wasm.noise_permutation_new(NOISE_SEED);
  const buffers = new WasmBuffers(wasm);
  try {
    const points2 = new Float64Array([0, 0, 3, -7, -128, 255, 41, 9]);
    const points3 = new Float64Array([0, 0, 0, 3, -7, 12, -128, 255, 5]);
    const in2 = buffers.alloc(Float64Array, points2.length);
    const in3 = buffers.alloc(Float64Array, points3.length);
    const out = buffers.alloc(Float64Array, 4);
    buffers.view(in2).set(points2);
    buffers.view(in3).set(points3);
    wasm.noise_perlin2_points(perm, in2.ptr, 4, out.ptr);
    const rust2 = Array.from(buffers.view(out));
    wasm.noise_perlin3_points(perm, in3.ptr, 3, out.ptr);
    const rust3 = Array.from(buffers.view(out)).slice(0, 3);
    const { perlin2d, perlin3d, simplex2d, simplex3d } = competitor.noise;
    const g = perlin2d.create(NOISE_SEED);
    const js2 = [0, 1, 2, 3].map((index) => perlin2d.sample(g, points2[2 * index], points2[2 * index + 1]));
    const js3 = [0, 1, 2].map((index) =>
      perlin3d.sample(g, points3[3 * index], points3[3 * index + 1], points3[3 * index + 2]),
    );
    wasm.noise_simplex2_points(perm, in2.ptr, 1, out.ptr);
    const rustSimplex2 = buffers.view(out)[0];
    wasm.noise_simplex3_points(perm, in3.ptr, 1, out.ptr);
    const rustSimplex3 = buffers.view(out)[0];
    const zero = (values) => values.every((value) => Math.abs(value) < 1e-12);
    return [
      { check: "perlin2 is zero on integer lattice points", rust: zero(rust2), js: zero(js2) },
      { check: "perlin3 is zero on integer lattice points", rust: zero(rust3), js: zero(js3) },
      {
        check: "simplex2/3 are zero at the origin",
        rust: zero([rustSimplex2, rustSimplex3]),
        js: zero([simplex2d.sample(g, 0, 0), simplex3d.sample(g, 0, 0, 0)]),
      },
    ];
  } finally {
    buffers.free();
    wasm.noise_permutation_free(perm);
  }
}

function permutationSetup(wasm, competitor, profile) {
  const rust = measure(() => wasm.noise_permutation_free(wasm.noise_permutation_new(NOISE_SEED)), profile);
  let sink = 0;
  const js = measure(() => {
    sink += competitor.noise.perlin2d.create(NOISE_SEED).perm[7];
  }, profile);
  return row({
    suite: "noise",
    scenario: "setup/permutation-create",
    size: "1",
    items: 1,
    rust,
    rustEndToEnd: null,
    js,
    check: { ok: true, detail: `one seeded table per engine (sink ${sink > 0 ? "live" : "zero"})` },
  });
}

export function noiseScenarios() {
  const scenarios = [];
  for (const field of FIELDS) {
    for (const count of POINT_SIZES) {
      scenarios.push({ name: `${field.id}/scattered`, size: String(count), field, kind: "points", count });
    }
    for (const edge of field.dims === 2 ? GRID2_SIZES : GRID3_SIZES) {
      const size = field.dims === 2 ? `${edge}x${edge}` : `${edge}x${edge}x${edge}`;
      scenarios.push({ name: `${field.id}/grid`, size, field, kind: "grid", edge });
    }
  }
  return scenarios;
}

export async function runNoiseSuite({ wasm, competitor, profile, filter, onRow }) {
  const rows = [permutationSetup(wasm, competitor, profile)];
  onRow?.(rows[0]);
  for (const scenario of noiseScenarios()) {
    if (filter && !filter(scenario.name)) {
      continue;
    }
    await yieldToBrowser();
    const prepared =
      scenario.kind === "points"
        ? pointScenario(wasm, competitor, scenario.field, scenario.count)
        : gridScenario(wasm, competitor, scenario.field, scenario.edge);
    const result = runScenario(wasm, prepared, profile);
    const check = describe(result.rustOut, result.jsOut, scenario.field.id.startsWith("worley"));
    if (!result.deterministic) {
      check.ok = false;
      check.detail += "; RUST NOT DETERMINISTIC";
    }
    const entry = row({
      suite: "noise",
      scenario: scenario.name,
      size: scenario.size,
      items: result.rustOut.length,
      rust: result.rust,
      rustEndToEnd: result.rustEndToEnd,
      js: result.js,
      check,
    });
    rows.push(entry);
    onRow?.(entry);
  }
  return { rows, sanity: latticeSanity(wasm, competitor) };
}
