#!/usr/bin/env node
// Headless runner for site/benchmarks.html (rust-kernels WASM vs pmndrs/math).
//
// Development-only, advisory evidence: never wire this into a blocking CI job.
//
//   npm ci --prefix benchmarks/browser          # installs the pinned `math` package only
//   node benchmarks/browser/run.mjs [--no-build] [--profile quick|full] [--suites noise,shapes]
//                                   [--filter <substring>] [--repeat N] [--out <file.json>]
//
// --repeat runs the page N times in fresh tabs and reports the per-row median across runs,
// which damps shared-machine noise; every run's raw rows stay in the JSON output.
//
// Playwright is not a dependency of this package: the runner resolves an already installed
// `playwright` (global module path or NODE_PATH) and launches its bundled Chromium.

import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const dist = path.join(root, "pages-dist");
const competitorRoot = path.join(here, "node_modules", "math");

function parseArgs(argv) {
  const options = { build: true, profile: "full", suites: "noise", filter: "", out: null, repeat: 1 };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) throw new Error(`${arg} needs a value`);
      return argv[index];
    };
    if (arg === "--no-build") options.build = false;
    else if (arg === "--profile") options.profile = next();
    else if (arg === "--suites") options.suites = next();
    else if (arg === "--filter") options.filter = next();
    else if (arg === "--out") options.out = next();
    else if (arg === "--repeat") options.repeat = Math.max(1, Number.parseInt(next(), 10) || 1);
    else throw new Error(`Unknown argument ${arg}`);
  }
  return options;
}

function command(file, args) {
  try {
    return execFileSync(file, args, { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".map": "application/json; charset=utf-8",
};

function serve() {
  const server = createServer((request, response) => {
    const url = new URL(request.url, "http://localhost");
    let base = dist;
    let relative = decodeURIComponent(url.pathname);
    if (relative.startsWith("/competitor/")) {
      base = competitorRoot;
      relative = relative.slice("/competitor".length);
    }
    const file = path.normalize(path.join(base, relative.endsWith("/") ? `${relative}index.html` : relative));
    if (!file.startsWith(base) || !existsSync(file) || !statSync(file).isFile()) {
      response.writeHead(404).end("not found");
      return;
    }
    // Cross-origin isolation gives performance.now() its finer resolution.
    response.writeHead(200, {
      "content-type": MIME[path.extname(file)] ?? "application/octet-stream",
      "cache-control": "no-store",
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
      "cross-origin-resource-policy": "same-origin",
    });
    createReadStream(file).pipe(response);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

function loadPlaywright() {
  const require = createRequire(import.meta.url);
  const candidates = ["playwright", ...(process.env.NODE_PATH ?? "").split(path.delimiter).filter(Boolean).map((dir) => path.join(dir, "playwright")), "/opt/node-tools/node_modules/playwright"];
  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch {
      // try the next location
    }
  }
  throw new Error("playwright is not installed; install it outside this package (it is intentionally not a dependency).");
}

function pad(text, width, right) {
  const value = String(text);
  return right ? value.padStart(width) : value.padEnd(width);
}

function fmt(value, digits = 1) {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

function table(rows) {
  const header = ["scenario", "size", "rust kernel ns/item", "rust e2e ns/item", "pmndrs ns/item", "js/rust", "check"];
  const body = rows.map((row) => [
    row.scenario,
    row.size,
    fmt(row.rustKernelNsPerItem),
    fmt(row.rustEndToEndNsPerItem),
    fmt(row.jsNsPerItem),
    row.jsOverRust === null ? "—" : `${row.jsOverRust.toFixed(2)}×`,
    row.check.ok ? "ok" : `CHECK: ${row.check.detail}`,
  ]);
  const widths = header.map((title, column) => Math.max(title.length, ...body.map((line) => line[column].length)));
  const numeric = new Set([2, 3, 4, 5]);
  const line = (cells) => `| ${cells.map((cell, column) => pad(cell, widths[column], numeric.has(column))).join(" | ")} |`;
  const separator = `| ${widths.map((width, column) => (numeric.has(column) ? `${"-".repeat(width - 1)}:` : "-".repeat(width))).join(" | ")} |`;
  return [line(header), separator, ...body.map(line)].join("\n");
}

function median(values) {
  const sorted = values.filter((value) => value !== null).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Per-row median across repeated page runs; checks must hold in every run. */
function aggregate(runs) {
  const [first] = runs;
  const rows = first.rows.map((row, index) => {
    const peers = runs.map((run) => run.rows[index]);
    const rust = median(peers.map((peer) => peer.rustKernelNsPerItem));
    const js = median(peers.map((peer) => peer.jsNsPerItem));
    const failed = peers.find((peer) => !peer.check.ok);
    return {
      ...row,
      rustKernelNsPerItem: rust,
      rustEndToEndNsPerItem: median(peers.map((peer) => peer.rustEndToEndNsPerItem)),
      jsNsPerItem: js,
      jsOverRust: rust && js ? js / rust : null,
      check: failed ? failed.check : row.check,
    };
  });
  const sanity = first.sanity.map((item, index) => ({
    ...item,
    rust: runs.every((run) => run.sanity[index].rust),
    js: runs.every((run) => run.sanity[index].js),
  }));
  return { meta: { ...first.meta, repeat: runs.length }, rows, sanity, runs: runs.map((run) => ({ meta: run.meta, rows: run.rows })) };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (!existsSync(path.join(competitorRoot, "package.json"))) {
    throw new Error("pmndrs/math is not installed: run `npm ci --prefix benchmarks/browser` first.");
  }
  if (options.build) {
    execFileSync("bash", ["scripts/build-pages.sh"], { cwd: root, stdio: "inherit" });
  }
  if (!existsSync(path.join(dist, "benchmarks.html"))) {
    throw new Error("pages-dist/benchmarks.html is missing: run `bash scripts/build-pages.sh` first.");
  }

  const { chromium } = loadPlaywright();
  const server = await serve();
  const { port } = server.address();
  const browser = await chromium.launch();
  try {
    const query = new URLSearchParams({
      autorun: "1",
      competitor: `http://127.0.0.1:${port}/competitor`,
      suites: options.suites,
      profile: options.profile,
      filter: options.filter,
    });
    const runs = [];
    for (let run = 0; run < options.repeat; run += 1) {
      const page = await browser.newPage();
      page.on("pageerror", (error) => console.error(`[page error] ${error.message}`));
      await page.goto(`http://127.0.0.1:${port}/benchmarks.html?${query}`);
      await page.waitForFunction(() => window.__benchResult !== undefined, null, { timeout: 30 * 60 * 1000, polling: 500 });
      const pageResult = await page.evaluate(() => window.__benchResult);
      await page.close();
      if (pageResult.error) throw new Error(`benchmark page failed: ${pageResult.error}`);
      runs.push(pageResult);
      console.error(`run ${run + 1}/${options.repeat} finished`);
    }
    const result = aggregate(runs);

    const lock = JSON.parse(readFileSync(path.join(here, "package-lock.json"), "utf8"));
    const lockEntry = lock.packages?.["node_modules/math"] ?? {};
    result.meta.competitor.integrity = lockEntry.integrity ?? null;
    result.meta.competitor.resolved = lockEntry.resolved ?? null;
    result.meta.environment = {
      browser: `chromium ${browser.version()}`,
      node: process.version,
      os: `${os.type()} ${os.release()} ${os.arch()}`,
      cpu: `${os.cpus()[0]?.model ?? "unknown"} × ${os.cpus().length}`,
      memoryGiB: Math.round(os.totalmem() / 2 ** 30),
      rustc: command("rustc", ["--version"]),
      gitHead: command("git", ["rev-parse", "HEAD"]),
      gitDirty: command("git", ["status", "--porcelain"]) ? true : false,
    };

    const meta = result.meta;
    console.log(`rust-kernels ${meta.rustKernels.revision}${meta.rustKernels.dirty ? " (dirty)" : ""}`);
    console.log(`competitor ${meta.competitor.name}@${meta.competitor.version} ${meta.competitor.integrity}`);
    console.log(`environment ${JSON.stringify(meta.environment)}`);
    console.log(`profile ${meta.profileName} × ${meta.repeat} run(s) · cross-origin isolated ${meta.crossOriginIsolated}`);
    console.log(
      `startup: wasm ${(meta.startup.wasmBytes / 1024).toFixed(0)} KiB, compile ${fmt(meta.startup.compileMs, 2)} ms, ` +
        `instantiate ${fmt(meta.startup.instantiateMs, 2)} ms; competitor fetch+import ${fmt(meta.competitorStartup.importMs, 2)} ms`,
    );
    for (const suite of new Set(result.rows.map((row) => row.suite))) {
      console.log(`\n### ${suite}\n`);
      console.log(table(result.rows.filter((row) => row.suite === suite)));
    }
    console.log("\nshared invariants:");
    for (const item of result.sanity) {
      console.log(`- ${item.suite}: ${item.check} — rust ${item.rust ? "yes" : "NO"}, pmndrs ${item.js ? "yes" : "NO"}`);
    }

    const out = options.out ?? path.join(here, "results", `${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
    mkdirSync(path.dirname(out), { recursive: true });
    writeFileSync(out, `${JSON.stringify(result, null, 2)}\n`);
    console.log(`\nraw results: ${path.relative(process.cwd(), out)}`);
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
