// Browser comparison page: rust-kernels through its WASM boundary vs pmndrs/math (npm `math`).
// Query parameters:
//   competitor=<base URL of the `math` package>  (default: pinned jsDelivr URL)
//   suites=noise,shapes   profile=quick|full   filter=<substring>   autorun=1
// The headless runner (benchmarks/browser/run.mjs) reads `window.__benchResult`.

import { DEFAULT_COMPETITOR_BASE, PROFILES, instantiateWasm, loadCompetitor } from "./bench/harness.js";
import { NOISE_EXPORTS, runNoiseSuite } from "./bench/noise.js";

const SUITES = {
  noise: { exports: NOISE_EXPORTS, run: runNoiseSuite },
};

const params = new URLSearchParams(location.search);
const status = document.querySelector("#bench-status");
const metadata = document.querySelector("#bench-metadata");
const tbody = document.querySelector("#bench-rows");
const sanityList = document.querySelector("#bench-sanity");
const rawOutput = document.querySelector("#bench-json");
const runButton = document.querySelector("#bench-run");
const profileSelect = document.querySelector("#bench-profile");
const filterInput = document.querySelector("#bench-filter");

function setStatus(state, text) {
  status.dataset.state = state;
  status.textContent = text;
}

function fmt(value, digits = 1) {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

function appendRow(entry) {
  const tr = document.createElement("tr");
  const cells = [
    entry.suite,
    entry.scenario,
    entry.size,
    fmt(entry.rustKernelNsPerItem),
    fmt(entry.rustEndToEndNsPerItem),
    fmt(entry.jsNsPerItem),
    entry.jsOverRust === null ? "—" : `${entry.jsOverRust.toFixed(2)}×`,
    `${entry.check.ok ? "ok" : "CHECK"} · ${entry.check.detail}`,
  ];
  for (const [index, text] of cells.entries()) {
    const td = document.createElement("td");
    td.textContent = text;
    if (index >= 3 && index <= 6) td.className = "num";
    if (index === 7 && !entry.check.ok) td.className = "warn";
    tr.append(td);
  }
  tbody.append(tr);
}

function renderMetadata(meta) {
  const entries = [
    ["rust-kernels revision", `${meta.rustKernels.revision}${meta.rustKernels.dirty ? " (dirty)" : ""}`],
    ["competitor", `${meta.competitor.name}@${meta.competitor.version} (${meta.competitor.repository ?? "?"})`],
    ["competitor source", meta.competitor.source],
    ["WASM startup", `${(meta.startup.wasmBytes / 1024).toFixed(0)} KiB · fetch ${fmt(meta.startup.fetchMs, 2)} ms · compile ${fmt(meta.startup.compileMs, 2)} ms · instantiate ${fmt(meta.startup.instantiateMs, 2)} ms`],
    ["competitor startup", `fetch + import ${fmt(meta.competitorStartup.importMs, 2)} ms`],
    ["profile", `${meta.profileName} ${JSON.stringify(meta.profile)}`],
    ["user agent", meta.userAgent],
    ["cross-origin isolated", String(meta.crossOriginIsolated)],
  ];
  metadata.replaceChildren(
    ...entries.map(([term, value]) => {
      const wrap = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = value;
      wrap.append(dt, dd);
      return wrap;
    }),
  );
}

async function buildInfo() {
  try {
    const response = await fetch("./build-info.json", { cache: "no-store" });
    if (response.ok) return await response.json();
  } catch {
    // A local copy of site/ without the Pages build step has no build-info.json.
  }
  return { revision: "unknown", dirty: null };
}

export async function runBenchmarks({ suites = Object.keys(SUITES), profileName = "full", filter = "" } = {}) {
  const profile = PROFILES[profileName] ?? PROFILES.full;
  const competitorBase = (params.get("competitor") ?? DEFAULT_COMPETITOR_BASE).replace(/\/$/, "");
  const [{ exports: wasm, startup }, competitor, info] = await Promise.all([
    instantiateWasm("./pkg/rust_kernels_web_demo.wasm"),
    loadCompetitor(competitorBase),
    buildInfo(),
  ]);
  const meta = {
    rustKernels: info,
    competitor: competitor.identity,
    startup,
    competitorStartup: competitor.startup,
    profileName,
    profile,
    userAgent: navigator.userAgent,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    startedAt: new Date().toISOString(),
  };
  renderMetadata(meta);
  tbody.replaceChildren();
  sanityList.replaceChildren();

  const rows = [];
  const sanity = [];
  for (const name of suites) {
    const suite = SUITES[name];
    if (!suite) throw new Error(`Unknown suite ${name}.`);
    for (const exportName of suite.exports) {
      if (typeof wasm[exportName] !== "function") {
        throw new Error(`WebAssembly module does not expose ${exportName}; rebuild with scripts/build-pages.sh.`);
      }
    }
    setStatus("loading", `Running ${name}…`);
    const result = await suite.run({
      wasm,
      competitor: competitor.modules,
      profile,
      filter: filter ? (scenario) => scenario.includes(filter) : null,
      onRow: (entry) => {
        appendRow(entry);
        setStatus("loading", `Running ${name}: ${entry.scenario} ${entry.size}`);
      },
    });
    rows.push(...result.rows);
    sanity.push(...result.sanity.map((item) => ({ suite: name, ...item })));
  }
  for (const item of sanity) {
    const li = document.createElement("li");
    li.textContent = `${item.suite}: ${item.check} — rust ${item.rust ? "yes" : "NO"}, pmndrs ${item.js ? "yes" : "NO"}`;
    sanityList.append(li);
  }
  const output = { meta: { ...meta, finishedAt: new Date().toISOString() }, rows, sanity };
  rawOutput.textContent = JSON.stringify(output, null, 2);
  return output;
}

async function runFromControls() {
  runButton.disabled = true;
  try {
    const suites = [...document.querySelectorAll("[data-suite]:checked")].map((input) => input.value);
    await runBenchmarks({ suites, profileName: profileSelect.value, filter: filterInput.value.trim() });
    setStatus("ready", "Finished. Lower ns/item is faster; ratio > 1 means rust-kernels is faster.");
  } catch (error) {
    setStatus("error", error instanceof Error ? error.message : String(error));
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", runFromControls);
if (params.has("profile")) profileSelect.value = params.get("profile");
if (params.has("filter")) filterInput.value = params.get("filter");
if (params.has("suites")) {
  const wanted = new Set(params.get("suites").split(","));
  for (const input of document.querySelectorAll("[data-suite]")) input.checked = wanted.has(input.value);
}

if (params.get("autorun") === "1") {
  const suites = (params.get("suites") ?? Object.keys(SUITES).join(",")).split(",");
  runBenchmarks({ suites, profileName: params.get("profile") ?? "full", filter: params.get("filter") ?? "" })
    .then((result) => {
      setStatus("ready", "Finished.");
      window.__benchResult = result;
    })
    .catch((error) => {
      setStatus("error", error instanceof Error ? error.message : String(error));
      window.__benchResult = { error: error instanceof Error ? `${error.message}\n${error.stack}` : String(error) };
    });
} else {
  runButton.disabled = false;
  setStatus("ready", "Choose suites and press Run. Nothing runs until you ask.");
}
