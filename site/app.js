const runtimeStatus = requiredElement("#runtime-status");

function requiredElement(selector) {
  const element = document.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

function requiredInput(selector) {
  const element = document.querySelector(selector);
  if (!(element instanceof HTMLInputElement)) {
    throw new Error(`Missing required input: ${selector}`);
  }
  return element;
}

async function instantiateWasm(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Unable to load WebAssembly (${response.status}).`);
  }

  if (WebAssembly.instantiateStreaming) {
    try {
      return await WebAssembly.instantiateStreaming(response.clone(), {});
    } catch {
      // Static hosts do not always send application/wasm. The byte fallback keeps Pages portable.
    }
  }

  const bytes = await response.arrayBuffer();
  return WebAssembly.instantiate(bytes, {});
}

function expectFunctions(exports, names) {
  for (const name of names) {
    if (typeof exports[name] !== "function") {
      throw new Error(`WebAssembly module does not expose ${name}.`);
    }
  }
}

function enableWasmControls() {
  for (const fieldset of document.querySelectorAll("[data-wasm-control]")) {
    if (fieldset instanceof HTMLFieldSetElement) {
      fieldset.disabled = false;
    }
  }
}

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function renderBars(container, values, selectedStart, selectedEnd) {
  const maxMagnitude = Math.max(...values.map((value) => Math.abs(value)), 1);
  container.replaceChildren(
    ...values.map((value, index) => {
      const wrap = document.createElement("div");
      wrap.className = "bar-wrap";

      const stage = document.createElement("div");
      stage.className = "bar-stage";

      const bar = document.createElement("div");
      bar.className = `bar${index >= selectedStart && index < selectedEnd ? " selected" : ""}`;
      bar.style.height = `${Math.max(10, (Math.abs(value) / maxMagnitude) * 88)}%`;
      bar.title = `Index ${index}: ${value}`;
      stage.append(bar);

      const label = document.createElement("span");
      label.className = "bar-value";
      label.textContent = String(value);
      wrap.append(stage, label);
      return wrap;
    }),
  );
}

function initializeFenwick(exports) {
  const startInput = requiredInput("#fenwick-start");
  const endInput = requiredInput("#fenwick-end");
  const startLabel = requiredElement("#fenwick-start-label");
  const endLabel = requiredElement("#fenwick-end-label");
  const resultStart = requiredElement("#fenwick-result-start");
  const resultEnd = requiredElement("#fenwick-result-end");
  const result = requiredElement("#fenwick-result");
  const bars = requiredElement("#fenwick-bars");

  const length = Number(exports.fenwick_dataset_len());
  const values = Array.from({ length }, (_, index) => Number(exports.fenwick_dataset_value(index)));
  startInput.max = String(length);
  endInput.max = String(length);

  const render = (changed) => {
    let start = Number(startInput.value);
    let end = Number(endInput.value);
    if (start > end) {
      if (changed === startInput) {
        end = start;
        endInput.value = String(end);
      } else {
        start = end;
        startInput.value = String(start);
      }
    }

    startLabel.textContent = String(start);
    endLabel.textContent = String(end);
    resultStart.textContent = String(start);
    resultEnd.textContent = String(end);
    result.textContent = String(exports.fenwick_range_sum(start, end));
    renderBars(bars, values, start, end);
  };

  startInput.addEventListener("input", () => render(startInput));
  endInput.addEventListener("input", () => render(endInput));
  render();
}

function initializeUnionFind(exports) {
  const nodesContainer = requiredElement("#union-nodes");
  const edgesContainer = requiredElement("#union-edges");
  const result = requiredElement("#union-result");
  const nodeCount = Number(exports.union_find_node_count());
  const edgeCount = Number(exports.union_find_edge_count());

  const nodes = Array.from({ length: nodeCount }, (_, index) => {
    const node = document.createElement("span");
    node.className = "union-node";
    node.textContent = String(index);
    node.setAttribute("aria-label", `Node ${index}`);
    return node;
  });
  nodesContainer.replaceChildren(...nodes);

  const checkboxes = Array.from({ length: edgeCount }, (_, index) => {
    const left = Number(exports.union_find_edge_left(index));
    const right = Number(exports.union_find_edge_right(index));
    const label = document.createElement("label");
    label.className = "edge-option";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = index < 3;
    input.value = String(index);

    const text = document.createElement("span");
    text.textContent = `${left} — ${right}`;
    label.append(input, text);
    edgesContainer.append(label);
    return input;
  });

  const render = () => {
    let mask = 0;
    for (const checkbox of checkboxes) {
      if (checkbox.checked) {
        mask |= 1 << Number(checkbox.value);
      }
    }

    for (let node = 0; node < nodeCount; node += 1) {
      const root = Number(exports.union_find_component_root(mask, node));
      nodes[node].dataset.root = String(root % 6);
      nodes[node].title = `Node ${node}, representative ${root}`;
    }
    result.textContent = String(exports.union_find_component_count(mask));
  };

  for (const checkbox of checkboxes) {
    checkbox.addEventListener("change", render);
  }
  render();
}

function initializeQuickselect(exports) {
  const rankInput = requiredInput("#quickselect-rank");
  const rankLabel = requiredElement("#quickselect-rank-label");
  const resultRank = requiredElement("#quickselect-result-rank");
  const result = requiredElement("#quickselect-result");
  const valuesContainer = requiredElement("#quickselect-values");
  const length = Number(exports.quickselect_dataset_len());
  rankInput.max = String(Math.max(0, length - 1));

  const render = () => {
    const nth = Number(rankInput.value);
    const selected = Number(exports.quickselect_value(nth));
    const values = Array.from({ length }, (_, index) => Number(exports.quickselect_partition_value(nth, index)));

    rankLabel.textContent = String(nth);
    resultRank.textContent = String(nth);
    result.textContent = String(selected);
    valuesContainer.replaceChildren(
      ...values.map((value, index) => {
        const item = document.createElement("span");
        item.className = `value-chip${index === nth ? " selected" : index < nth ? " before" : " after"}`;
        item.textContent = String(value);
        item.title = index === nth ? `Selected rank ${nth}` : `Partition index ${index}`;
        return item;
      }),
    );
  };

  rankInput.addEventListener("input", render);
  render();
}

function initializeMorton(exports) {
  const xInput = requiredInput("#morton-x");
  const yInput = requiredInput("#morton-y");
  const xLabel = requiredElement("#morton-x-label");
  const yLabel = requiredElement("#morton-y-label");
  const resultX = requiredElement("#morton-result-x");
  const resultY = requiredElement("#morton-result-y");
  const result = requiredElement("#morton-result");
  const grid = requiredElement("#morton-grid");
  const size = 8;
  const cells = [];

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "morton-cell";
      cell.textContent = String(exports.morton2_key(x, y));
      cell.title = `(${x}, ${y}) → ${cell.textContent}`;
      cell.setAttribute("aria-label", `Coordinate ${x}, ${y}, Morton key ${cell.textContent}`);
      cell.addEventListener("click", () => {
        xInput.value = String(x);
        yInput.value = String(y);
        render();
      });
      cells.push({ cell, x, y });
      grid.append(cell);
    }
  }

  const render = () => {
    const x = Number(xInput.value);
    const y = Number(yInput.value);
    xLabel.textContent = String(x);
    yLabel.textContent = String(y);
    resultX.textContent = String(x);
    resultY.textContent = String(y);
    result.textContent = String(exports.morton2_key(x, y));
    for (const entry of cells) {
      entry.cell.classList.toggle("selected", entry.x === x && entry.y === y);
      entry.cell.setAttribute("aria-pressed", entry.x === x && entry.y === y ? "true" : "false");
    }
  };

  xInput.addEventListener("input", render);
  yInput.addEventListener("input", render);
  render();
}

function initializeRunningStats(exports) {
  const countInput = requiredInput("#stats-count");
  const countLabel = requiredElement("#stats-count-label");
  const mean = requiredElement("#stats-mean");
  const variance = requiredElement("#stats-variance");
  const valuesContainer = requiredElement("#stats-values");
  const length = Number(exports.running_stats_dataset_len());
  const values = Array.from({ length }, (_, index) => Number(exports.running_stats_dataset_value(index)));
  countInput.max = String(length);

  const render = () => {
    const count = Number(countInput.value);
    countLabel.textContent = String(count);
    mean.textContent = formatNumber(exports.running_stats_mean(count));
    variance.textContent = formatNumber(exports.running_stats_population_variance(count));
    valuesContainer.replaceChildren(
      ...values.map((value, index) => {
        const item = document.createElement("span");
        item.className = `value-chip${index < count ? " selected" : " muted"}`;
        item.textContent = formatNumber(value);
        return item;
      }),
    );
  };

  countInput.addEventListener("input", render);
  render();
}

try {
  const { instance } = await instantiateWasm("./pkg/rust_kernels_web_demo.wasm");
  const exports = instance.exports;
  expectFunctions(exports, [
    "fenwick_dataset_len",
    "fenwick_dataset_value",
    "fenwick_range_sum",
    "union_find_node_count",
    "union_find_edge_count",
    "union_find_edge_left",
    "union_find_edge_right",
    "union_find_component_count",
    "union_find_component_root",
    "quickselect_dataset_len",
    "quickselect_value",
    "quickselect_partition_value",
    "morton2_key",
    "running_stats_dataset_len",
    "running_stats_dataset_value",
    "running_stats_mean",
    "running_stats_population_variance",
  ]);

  initializeFenwick(exports);
  initializeUnionFind(exports);
  initializeQuickselect(exports);
  initializeMorton(exports);
  initializeRunningStats(exports);
  enableWasmControls();
  runtimeStatus.dataset.state = "ready";
  runtimeStatus.textContent = "Live examples are running the repository's Rust kernels as WebAssembly.";
} catch (error) {
  runtimeStatus.dataset.state = "error";
  runtimeStatus.textContent = error instanceof Error ? error.message : "Unable to start WebAssembly demos.";
}
