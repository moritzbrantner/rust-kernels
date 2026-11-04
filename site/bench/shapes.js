// Collision and shape-query workloads: rust-kernels (geometry-kernels via crates/web-demo) vs
// pmndrs/math (`math` on npm). Only operations with matching semantics are compared:
//
//   ray vs AABB hit      geometry_kernels::ray_aabb (enter <= max)  ~ raycast3.intersectsBox3
//   ray vs triangle      geometry_kernels::ray_triangle (both faces) ~ raycast3.intersectsTriangle
//   sphere vs AABB       primitives::sphere_aabb().overlaps          ~ box3.intersectsSphere
//   AABB vs AABB         aabb_aabb().overlaps                        ~ box3.intersectsBox3
//   OBB vs OBB           obb3::obb3_sat().overlaps                   ~ obb3.intersectsOBB3
//   convex hull 2D/3D    hull::quickhull2 / quickhull3               ~ quickhull2 / quickhull3
//   polygon ear clipping polygon2::triangulate_polygon2 (AssumeSimple) ~ triangulatePolygon2
//
// Inputs are rounded to f32 (the rust-kernels input precision for these shapes) and both
// engines read the identical values. pmndrs/math receives them in its own idiomatic layout
// (tuples / objects), prepared once outside the measured region.

import { WasmBuffers, checksum, measure, rng, row, yieldToBrowser } from "./harness.js";

const PAIR_SIZES = [1024, 16384, 262144];
const HULL_SIZES = [1024, 16384, 131072];
const HULL_SURFACE_SIZES = [1024, 16384];
const POLYGON_SIZES = [64, 512, 2048];
const MAX_DISTANCE = 1000;

export const SHAPE_EXPORTS = [
  "shape_ray_aabb",
  "shape_ray_triangle",
  "shape_sphere_aabb",
  "shape_aabb_aabb",
  "shape_obb_obb",
  "shape_quickhull2",
  "shape_quickhull3",
  "shape_quickhull3_triangles",
  "shape_triangulate_polygon2",
  "shape_scratch_new",
  "shape_scratch_free",
];

const f = Math.fround;

function uniform(random, low, high) {
  return low + (high - low) * random();
}

function unitVector(random) {
  for (;;) {
    const x = random() * 2 - 1;
    const y = random() * 2 - 1;
    const z = random() * 2 - 1;
    const length = Math.hypot(x, y, z);
    if (length > 1e-3 && length <= 1) return [x / length, y / length, z / length];
  }
}

function writeBox(target, offset, center, half) {
  for (let axis = 0; axis < 3; axis += 1) {
    target[offset + axis] = f(center[axis] - half[axis]);
    target[offset + 3 + axis] = f(center[axis] + half[axis]);
  }
}

/**
 * Rays start 15–25 units from their target and point at it with a ±`jitter` aim offset (`aimed`)
 * or in a uniformly random direction (`scattered`).
 */
function writeRay(target, offset, random, aimAt, aimed, jitter = 3) {
  const out = unitVector(random);
  const distance = uniform(random, 15, 25);
  const origin = aimAt.map((value, axis) => value + out[axis] * distance);
  let direction;
  if (aimed) {
    const toward = aimAt.map((value, axis) => value + uniform(random, -jitter, jitter) - origin[axis]);
    const length = Math.hypot(...toward);
    direction = toward.map((value) => value / length);
  } else {
    direction = unitVector(random);
  }
  for (let axis = 0; axis < 3; axis += 1) {
    target[offset + axis] = f(origin[axis]);
    target[offset + 3 + axis] = f(direction[axis]);
  }
}

function randomPoint(random, extent) {
  return [uniform(random, -extent, extent), uniform(random, -extent, extent), uniform(random, -extent, extent)];
}

function randomHalf(random, low, high) {
  return [uniform(random, low, high), uniform(random, low, high), uniform(random, low, high)];
}

/** Rows of the rust-kernels Obb3 rotation (Rz * Ry * Rx), flattened as pmndrs/math OBB3 axis rows. */
function obbRotation(rx, ry, rz) {
  const [sx, cx, sy, cy, sz, cz] = [Math.sin(rx), Math.cos(rx), Math.sin(ry), Math.cos(ry), Math.sin(rz), Math.cos(rz)];
  return [
    cz * cy, sz * cy, -sy,
    cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx,
    cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx,
  ];
}

const tuple3 = (values, offset) => [values[offset], values[offset + 1], values[offset + 2]];

const PAIR_SCENARIOS = [
  {
    name: "ray-aabb/aimed",
    strides: [6, 6],
    generate(random, count, left, right) {
      for (let index = 0; index < count; index += 1) {
        const center = randomPoint(random, 10);
        writeBox(right, 6 * index, center, randomHalf(random, 0.5, 3));
        writeRay(left, 6 * index, random, center, true);
      }
    },
    rust: (wasm, a, b, count, out) => wasm.shape_ray_aabb(a, b, count, MAX_DISTANCE, out),
    prepareJs: (left, right, index) => ({ origin: tuple3(left, 6 * index), direction: tuple3(left, 6 * index + 3), box: Array.from(right.subarray(6 * index, 6 * index + 6)) }),
    js: (math) => (item) => math.shapes.raycast3.intersectsBox3(item.origin, item.direction, MAX_DISTANCE, item.box),
  },
  {
    name: "ray-aabb/scattered",
    strides: [6, 6],
    generate(random, count, left, right) {
      for (let index = 0; index < count; index += 1) {
        const center = randomPoint(random, 10);
        writeBox(right, 6 * index, center, randomHalf(random, 0.5, 3));
        writeRay(left, 6 * index, random, center, false);
      }
    },
    rust: (wasm, a, b, count, out) => wasm.shape_ray_aabb(a, b, count, MAX_DISTANCE, out),
    prepareJs: (left, right, index) => ({ origin: tuple3(left, 6 * index), direction: tuple3(left, 6 * index + 3), box: Array.from(right.subarray(6 * index, 6 * index + 6)) }),
    js: (math) => (item) => math.shapes.raycast3.intersectsBox3(item.origin, item.direction, MAX_DISTANCE, item.box),
  },
  ...["aimed", "scattered"].map((mode) => ({
    name: `ray-triangle/${mode}`,
    strides: [6, 9],
    distances: true,
    generate(random, count, left, right) {
      for (let index = 0; index < count; index += 1) {
        const centroid = randomPoint(random, 10);
        for (let corner = 0; corner < 3; corner += 1) {
          for (let axis = 0; axis < 3; axis += 1) {
            right[9 * index + 3 * corner + axis] = f(centroid[axis] + uniform(random, -2.5, 2.5));
          }
        }
        writeRay(left, 6 * index, random, centroid, mode === "aimed", 0.75);
      }
    },
    rust: (wasm, a, b, count, out) => wasm.shape_ray_triangle(a, b, count, MAX_DISTANCE, out),
    prepareJs: (left, right, index) => ({
      origin: tuple3(left, 6 * index),
      direction: tuple3(left, 6 * index + 3),
      a: tuple3(right, 9 * index),
      b: tuple3(right, 9 * index + 3),
      c: tuple3(right, 9 * index + 6),
    }),
    js: (math) => {
      const result = math.shapes.raycast3.createIntersectsTriangleResult();
      return (item) => {
        math.shapes.raycast3.intersectsTriangle(result, item.origin, item.direction, MAX_DISTANCE, item.a, item.b, item.c, false);
        return result.hit ? result.fraction * MAX_DISTANCE : -1;
      };
    },
  })),
  {
    name: "sphere-aabb/mixed",
    strides: [4, 6],
    generate(random, count, left, right) {
      for (let index = 0; index < count; index += 1) {
        const center = randomPoint(random, 10);
        writeBox(right, 6 * index, center, randomHalf(random, 0.5, 3));
        const offset = randomPoint(random, 6);
        for (let axis = 0; axis < 3; axis += 1) left[4 * index + axis] = f(center[axis] + offset[axis]);
        left[4 * index + 3] = f(uniform(random, 0.5, 3));
      }
    },
    rust: (wasm, a, b, count, out) => wasm.shape_sphere_aabb(a, b, count, out),
    prepareJs: (left, right, index) => ({ sphere: { center: tuple3(left, 4 * index), radius: left[4 * index + 3] }, box: Array.from(right.subarray(6 * index, 6 * index + 6)) }),
    js: (math) => (item) => math.shapes.box3.intersectsSphere(item.box, item.sphere),
  },
  {
    name: "aabb-aabb/mixed",
    strides: [6, 6],
    generate(random, count, left, right) {
      for (let index = 0; index < count; index += 1) {
        writeBox(left, 6 * index, randomPoint(random, 6), randomHalf(random, 0.5, 4));
        writeBox(right, 6 * index, randomPoint(random, 6), randomHalf(random, 0.5, 4));
      }
    },
    rust: (wasm, a, b, count, out) => wasm.shape_aabb_aabb(a, b, count, out),
    prepareJs: (left, right, index) => ({ a: Array.from(left.subarray(6 * index, 6 * index + 6)), b: Array.from(right.subarray(6 * index, 6 * index + 6)) }),
    js: (math) => (item) => math.shapes.box3.intersectsBox3(item.a, item.b),
  },
  {
    name: "obb-obb/mixed",
    strides: [9, 9],
    generate(random, count, left, right) {
      for (const target of [left, right]) {
        for (let index = 0; index < count; index += 1) {
          const values = [...randomPoint(random, 5), ...randomHalf(random, 0.5, 3), ...randomPoint(random, Math.PI)];
          for (let slot = 0; slot < 9; slot += 1) target[9 * index + slot] = f(values[slot]);
        }
      }
    },
    rust: (wasm, a, b, count, out) => wasm.shape_obb_obb(a, b, count, out),
    prepareJs: (left, right, index) => {
      const obb = (values) => ({
        center: tuple3(values, 9 * index),
        halfExtents: tuple3(values, 9 * index + 3),
        rotation: obbRotation(values[9 * index + 6], values[9 * index + 7], values[9 * index + 8]),
      });
      return { a: obb(left), b: obb(right) };
    },
    js: (math) => (item) => math.shapes.obb3.intersectsOBB3(item.a, item.b),
  },
];

function pairScenario(wasm, competitor, scenario, count, profile) {
  const random = rng(0xc011 + count + scenario.name.length);
  const [leftStride, rightStride] = scenario.strides;
  const left = new Float32Array(count * leftStride);
  const right = new Float32Array(count * rightStride);
  scenario.generate(random, count, left, right);
  const items = Array.from({ length: count }, (_, index) => scenario.prepareJs(left, right, index));
  const query = scenario.js(competitor);
  const OutType = scenario.distances ? Float64Array : Uint8Array;
  const jsOut = new OutType(count);
  const rustCopy = new OutType(count);

  const buffers = new WasmBuffers(wasm);
  try {
    const leftAlloc = buffers.alloc(Float32Array, left.length);
    const rightAlloc = buffers.alloc(Float32Array, right.length);
    const outAlloc = buffers.alloc(OutType, count);
    const wasmLeft = buffers.view(leftAlloc);
    const wasmRight = buffers.view(rightAlloc);
    const wasmOut = buffers.view(outAlloc);
    wasmLeft.set(left);
    wasmRight.set(right);

    const rustKernel = () => scenario.rust(wasm, leftAlloc.ptr, rightAlloc.ptr, count, outAlloc.ptr);
    const rustEndToEnd = () => {
      wasmLeft.set(left);
      wasmRight.set(right);
      scenario.rust(wasm, leftAlloc.ptr, rightAlloc.ptr, count, outAlloc.ptr);
      rustCopy.set(wasmOut);
    };
    const js = scenario.distances
      ? () => {
          for (let index = 0; index < count; index += 1) jsOut[index] = query(items[index]);
        }
      : () => {
          for (let index = 0; index < count; index += 1) jsOut[index] = query(items[index]) ? 1 : 0;
        };

    const rust = measure(rustKernel, profile);
    const first = checksum(wasmOut);
    rustKernel();
    const deterministic = checksum(wasmOut) === first;
    const endToEnd = measure(rustEndToEnd, profile);
    const jsResult = measure(js, profile);
    return { rust, rustEndToEnd: endToEnd, js: jsResult, check: comparePairs(wasmOut, jsOut, scenario.distances, deterministic) };
  } finally {
    buffers.free();
  }
}

function comparePairs(rustOut, jsOut, distances, deterministic) {
  let rustHits = 0;
  let jsHits = 0;
  let disagreements = 0;
  let maxDistanceDelta = 0;
  for (let index = 0; index < rustOut.length; index += 1) {
    const rustHit = distances ? rustOut[index] >= 0 : rustOut[index] === 1;
    const jsHit = distances ? jsOut[index] >= 0 : jsOut[index] === 1;
    rustHits += rustHit ? 1 : 0;
    jsHits += jsHit ? 1 : 0;
    if (rustHit !== jsHit) disagreements += 1;
    else if (distances && rustHit) maxDistanceDelta = Math.max(maxDistanceDelta, Math.abs(rustOut[index] - jsOut[index]));
  }
  const share = (hits) => `${((100 * hits) / rustOut.length).toFixed(1)}%`;
  // A handful of grazing contacts may legitimately differ between tolerances; flag above 0.1%.
  const ok = deterministic && disagreements <= Math.max(1, rustOut.length / 1000) && maxDistanceDelta < 1e-4;
  return {
    ok,
    detail:
      `hits rust ${share(rustHits)} js ${share(jsHits)}; ${disagreements} disagreement(s)` +
      (distances ? `; max |Δdistance| ${maxDistanceDelta.toExponential(1)}` : "") +
      (deterministic ? "" : "; RUST NOT DETERMINISTIC"),
    rustHits,
    jsHits,
    disagreements,
    maxDistanceDelta: distances ? maxDistanceDelta : null,
  };
}

function hullPoints(random, count, dims, surface) {
  const points = new Float64Array(count * dims);
  for (let index = 0; index < count; index += 1) {
    let vector;
    if (dims === 2) {
      const angle = random() * 2 * Math.PI;
      const radius = surface ? 1 : Math.sqrt(random());
      vector = [radius * Math.cos(angle), radius * Math.sin(angle)];
    } else {
      const direction = unitVector(random);
      const radius = surface ? 1 : Math.cbrt(random());
      vector = direction.map((value) => value * radius);
    }
    for (let axis = 0; axis < dims; axis += 1) points[dims * index + axis] = f(vector[axis]);
  }
  return points;
}

function starPolygon(random, count) {
  const points = new Float64Array(count * 2);
  for (let index = 0; index < count; index += 1) {
    const angle = (2 * Math.PI * index) / count;
    const radius = uniform(random, 0.5, 1);
    points[2 * index] = f(radius * Math.cos(angle));
    points[2 * index + 1] = f(radius * Math.sin(angle));
  }
  return points;
}

function bulkScenario(wasm, input, rustCall, jsCall, profile, compare) {
  const scratch = wasm.shape_scratch_new();
  const buffers = new WasmBuffers(wasm);
  try {
    const inputAlloc = buffers.alloc(Float64Array, input.length);
    const wasmInput = buffers.view(inputAlloc);
    wasmInput.set(input);
    let rustValue = 0;
    let jsValue = null;
    const rustKernel = () => {
      rustValue = rustCall(scratch, inputAlloc.ptr);
    };
    const rustEndToEnd = () => {
      buffers.view(inputAlloc).set(input);
      rustValue = rustCall(scratch, inputAlloc.ptr);
    };
    const js = () => {
      jsValue = jsCall();
    };
    const rust = measure(rustKernel, profile);
    const first = rustValue;
    const endToEnd = measure(rustEndToEnd, profile);
    const jsResult = measure(js, profile);
    return { rust, rustEndToEnd: endToEnd, js: jsResult, check: compare(rustValue, jsValue, rustValue === first, scratch) };
  } finally {
    buffers.free();
    wasm.shape_scratch_free(scratch);
  }
}

function bulkScenarios() {
  const list = [];
  for (const [dims, sizesBall, sizesSurface] of [
    [2, HULL_SIZES, HULL_SURFACE_SIZES],
    [3, HULL_SIZES, HULL_SURFACE_SIZES],
  ]) {
    for (const [label, sizes, surface] of [
      [dims === 2 ? "disk" : "ball", sizesBall, false],
      [dims === 2 ? "circle" : "sphere", sizesSurface, true],
    ]) {
      for (const count of sizes) {
        list.push({ name: `quickhull${dims}d/${label}`, size: String(count), kind: "hull", dims, count, surface });
      }
    }
  }
  for (const count of POLYGON_SIZES) {
    list.push({ name: "triangulate-polygon2/star", size: String(count), kind: "polygon", count });
  }
  return list;
}

function runBulk(wasm, competitor, scenario, profile) {
  const random = rng(0x4011 + scenario.count + (scenario.dims ?? 0) + (scenario.surface ? 7 : 0));
  if (scenario.kind === "polygon") {
    const points = starPolygon(random, scenario.count);
    const out = new Uint32Array(3 * (scenario.count - 2));
    return bulkScenario(
      wasm,
      points,
      (scratch, ptr) => wasm.shape_triangulate_polygon2(scratch, ptr, scenario.count),
      () => competitor.geometry.triangulatePolygon2(out, points, scenario.count),
      profile,
      (rust, js, deterministic) => ({
        ok: deterministic && rust === scenario.count - 2 && js === rust,
        detail: `triangles rust ${rust} js ${js} (expected ${scenario.count - 2})`,
      }),
    );
  }
  const points = hullPoints(random, scenario.count, scenario.dims, scenario.surface);
  if (scenario.dims === 2) {
    return bulkScenario(
      wasm,
      points,
      (scratch, ptr) => wasm.shape_quickhull2(scratch, ptr, scenario.count),
      () => competitor.geometry.quickhull2(points).length,
      profile,
      (rust, js, deterministic) => {
        // Points rounded onto a circle are nearly collinear; each engine's collinearity tolerance
        // decides which survive, so a count difference there is informational, not a failure.
        const tolerant = scenario.surface && rust !== js;
        return {
          ok: deterministic && (rust === js || tolerant) && rust >= 3 && js >= 3,
          detail: `hull vertices rust ${rust} js ${js}${tolerant ? " (near-collinear input: tolerance-dependent, informational)" : ""}`,
        };
      },
    );
  }
  return bulkScenario(
    wasm,
    points,
    (scratch, ptr) => wasm.shape_quickhull3(scratch, ptr, scenario.count),
    () => competitor.geometry.quickhull3(points),
    profile,
    (rust, js, deterministic, scratch) => {
      const triangles = wasm.shape_quickhull3_triangles(scratch);
      const jsVertices = new Set(js).size;
      const jsTriangles = js.length / 3;
      return {
        ok: deterministic && rust === jsVertices && triangles === jsTriangles,
        detail: `hull vertices rust ${rust} js ${jsVertices}; triangles rust ${triangles} js ${jsTriangles}`,
      };
    },
  );
}

function shapeSanity(wasm, competitor) {
  // Hand-checkable fixtures both engines must agree on.
  const buffers = new WasmBuffers(wasm);
  try {
    const rays = buffers.alloc(Float32Array, 12);
    const boxes = buffers.alloc(Float32Array, 12);
    const flags = buffers.alloc(Uint8Array, 2);
    buffers.view(rays).set([-5, 0, 0, 1, 0, 0, -5, 3, 0, 1, 0, 0]);
    buffers.view(boxes).set([-1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1]);
    wasm.shape_ray_aabb(rays.ptr, boxes.ptr, 2, MAX_DISTANCE, flags.ptr);
    const rustFlags = Array.from(buffers.view(flags));
    const touching = buffers.alloc(Float32Array, 6);
    buffers.view(touching).set([1, -1, -1, 3, 1, 1]);
    const rustTouching = wasm.shape_aabb_aabb(boxes.ptr, touching.ptr, 1, flags.ptr) === 1;
    const { raycast3, box3 } = competitor.shapes;
    const box = [-1, -1, -1, 1, 1, 1];
    const jsFlags = [raycast3.intersectsBox3([-5, 0, 0], [1, 0, 0], MAX_DISTANCE, box), raycast3.intersectsBox3([-5, 3, 0], [1, 0, 0], MAX_DISTANCE, box)];
    return [
      { check: "ray-aabb: axis ray hits the unit box, offset ray misses", rust: rustFlags[0] === 1 && rustFlags[1] === 0, js: jsFlags[0] && !jsFlags[1] },
      { check: "aabb-aabb: touching faces overlap", rust: rustTouching, js: box3.intersectsBox3(box, [1, -1, -1, 3, 1, 1]) },
    ];
  } finally {
    buffers.free();
  }
}

export function shapeScenarios() {
  return [
    ...PAIR_SCENARIOS.flatMap((scenario) => PAIR_SIZES.map((count) => ({ name: scenario.name, size: String(count), kind: "pair", scenario, count }))),
    ...bulkScenarios(),
  ];
}

export async function runShapesSuite({ wasm, competitor, profile, filter, onRow }) {
  const rows = [];
  for (const scenario of shapeScenarios()) {
    if (filter && !filter(scenario.name)) continue;
    await yieldToBrowser();
    const result =
      scenario.kind === "pair"
        ? pairScenario(wasm, competitor, scenario.scenario, scenario.count, profile)
        : runBulk(wasm, competitor, scenario, profile);
    const entry = row({
      suite: "shapes",
      scenario: scenario.name,
      size: scenario.size,
      items: scenario.count,
      rust: result.rust,
      rustEndToEnd: result.rustEndToEnd,
      js: result.js,
      check: result.check,
    });
    rows.push(entry);
    onRow?.(entry);
  }
  return { rows, sanity: shapeSanity(wasm, competitor) };
}
