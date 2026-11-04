// Shared timing, WASM-buffer and input helpers for the browser comparison page.
// Timings are comparative evidence only; correctness stays with the Cargo tests.

export const DEFAULT_COMPETITOR_BASE = "https://cdn.jsdelivr.net/npm/math@0.1.0";

export const PROFILES = {
  // Each sample repeats the batch until it lasts at least `minSampleMs`, so coarse
  // `performance.now()` resolution on non-isolated pages does not dominate.
  quick: { minSampleMs: 4, samples: 5, maxScenarioMs: 400, warmups: 2 },
  full: { minSampleMs: 10, samples: 11, maxScenarioMs: 1500, warmups: 3 },
};

export function now() {
  return performance.now();
}

/** Deterministic 32-bit PRNG (mulberry32) used to generate identical inputs for both engines. */
export function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Times `run` (one whole batch) and returns the median and minimum per-batch time in ms.
 * Warm-up batches let both JIT tiers and WASM tier-up settle before measuring.
 */
export function measure(run, profile) {
  for (let index = 0; index < profile.warmups; index += 1) {
    run();
  }
  let reps = 1;
  for (;;) {
    const start = now();
    for (let rep = 0; rep < reps; rep += 1) {
      run();
    }
    if (now() - start >= profile.minSampleMs || reps >= 1 << 20) {
      break;
    }
    reps *= 2;
  }
  const times = [];
  const started = now();
  while (times.length < profile.samples) {
    const start = now();
    for (let rep = 0; rep < reps; rep += 1) {
      run();
    }
    times.push((now() - start) / reps);
    if (times.length >= 3 && now() - started > profile.maxScenarioMs) {
      break;
    }
  }
  return { medianMs: median(times), minMs: Math.min(...times), samples: times.length, reps };
}

/** Owns typed-array views into WASM linear memory for one scenario. */
export class WasmBuffers {
  constructor(exports) {
    this.exports = exports;
    this.allocations = [];
  }

  /** Allocates a buffer; call `view` only after every allocation of the scenario (memory may grow). */
  alloc(ArrayType, length) {
    const bytes = Math.max(1, length * ArrayType.BYTES_PER_ELEMENT);
    const ptr = this.exports.bench_alloc(bytes);
    if (ptr === 0) {
      throw new Error(`WASM allocation of ${bytes} bytes failed.`);
    }
    const allocation = { ArrayType, length, bytes, ptr };
    this.allocations.push(allocation);
    return allocation;
  }

  view(allocation) {
    return new allocation.ArrayType(this.exports.memory.buffer, allocation.ptr, allocation.length);
  }

  free() {
    for (const allocation of this.allocations) {
      this.exports.bench_free(allocation.ptr, allocation.bytes);
    }
    this.allocations = [];
  }
}

export async function instantiateWasm(url) {
  const started = now();
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Unable to load WebAssembly (${response.status}).`);
  }
  const bytes = await response.arrayBuffer();
  const fetched = now();
  const module = await WebAssembly.compile(bytes);
  const compiled = now();
  const instance = await WebAssembly.instantiate(module, {});
  const instantiated = now();
  return {
    exports: instance.exports,
    startup: {
      wasmBytes: bytes.byteLength,
      fetchMs: fetched - started,
      compileMs: compiled - fetched,
      instantiateMs: instantiated - compiled,
    },
  };
}

export async function loadCompetitor(base) {
  const started = now();
  const pkgResponse = await fetch(`${base}/package.json`);
  if (!pkgResponse.ok) {
    throw new Error(`Unable to load competitor package.json from ${base} (${pkgResponse.status}).`);
  }
  const pkg = await pkgResponse.json();
  const [noise, shapes, geometry] = await Promise.all([
    import(`${base}/dist/noise/index.js`),
    import(`${base}/dist/shapes/index.js`),
    import(`${base}/dist/geometry/index.js`),
  ]);
  return {
    modules: { noise, shapes, geometry },
    identity: {
      name: pkg.name,
      version: pkg.version,
      repository: typeof pkg.repository === "string" ? pkg.repository : pkg.repository?.url,
      gitHead: pkg.gitHead ?? null,
      source: base,
    },
    startup: { importMs: now() - started },
  };
}

export function range(values) {
  let min = Infinity;
  let max = -Infinity;
  let finite = true;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      finite = false;
    }
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { min, max, finite };
}

/** FNV-1a over the raw bytes of a typed array; used to check run-to-run determinism. */
export function checksum(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let hash = 0x811c9dc5;
  for (let index = 0; index < bytes.length; index += 1) {
    hash = Math.imul(hash ^ bytes[index], 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

/**
 * Builds one comparison row. `rust` and `js` are `measure` results for the steady-state
 * kernel; `rustEndToEnd` includes copying inputs into and outputs out of linear memory.
 */
export function row({ suite, scenario, size, items, rust, rustEndToEnd, js, check }) {
  const perItem = (result) => (result ? (result.medianMs * 1e6) / items : null);
  return {
    suite,
    scenario,
    size,
    items,
    rustKernelNsPerItem: perItem(rust),
    rustEndToEndNsPerItem: perItem(rustEndToEnd),
    jsNsPerItem: perItem(js),
    jsOverRust: rust && js ? js.medianMs / rust.medianMs : null,
    samples: { rust: rust?.samples, rustEndToEnd: rustEndToEnd?.samples, js: js?.samples },
    check,
  };
}

/** Lets the browser paint progress between scenarios. */
export function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
