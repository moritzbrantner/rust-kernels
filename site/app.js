const prefixInput = document.querySelector("#prefix-end");
const prefixLabel = document.querySelector("#prefix-label");
const resultEnd = document.querySelector("#result-end");
const prefixResult = document.querySelector("#prefix-result");
const runtimeStatus = document.querySelector("#runtime-status");
const valueBars = document.querySelector("#value-bars");

if (
  !(prefixInput instanceof HTMLInputElement) ||
  !(prefixLabel instanceof HTMLElement) ||
  !(resultEnd instanceof HTMLElement) ||
  !(prefixResult instanceof HTMLElement) ||
  !(runtimeStatus instanceof HTMLElement) ||
  !(valueBars instanceof HTMLElement)
) {
  throw new Error("The rust-kernels demo markup is incomplete.");
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
      // Some static hosts do not serve application/wasm. The byte fallback keeps the demo portable.
    }
  }

  const bytes = await response.arrayBuffer();
  return WebAssembly.instantiate(bytes, {});
}

function renderBars(values, end) {
  const maxMagnitude = Math.max(...values.map((value) => Math.abs(value)), 1);
  valueBars.replaceChildren(
    ...values.map((value, index) => {
      const wrap = document.createElement("div");
      wrap.className = "bar-wrap";

      const stage = document.createElement("div");
      stage.className = "bar-stage";

      const bar = document.createElement("div");
      bar.className = `bar${index < end ? " selected" : ""}`;
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

try {
  const { instance } = await instantiateWasm("./pkg/rust_kernels_web_demo.wasm");
  const { dataset_len, dataset_value, fenwick_prefix_sum } = instance.exports;

  if (
    typeof dataset_len !== "function" ||
    typeof dataset_value !== "function" ||
    typeof fenwick_prefix_sum !== "function"
  ) {
    throw new Error("WebAssembly module does not expose the expected demo functions.");
  }

  const length = Number(dataset_len());
  const values = Array.from({ length }, (_, index) => Number(dataset_value(index)));
  prefixInput.max = String(length);

  const render = () => {
    const end = Number(prefixInput.value);
    prefixLabel.textContent = String(end);
    resultEnd.textContent = String(end);
    prefixResult.textContent = String(fenwick_prefix_sum(end));
    renderBars(values, end);
  };

  prefixInput.addEventListener("input", render);
  runtimeStatus.dataset.state = "ready";
  runtimeStatus.textContent = "Running the repository's Rust FenwickTree implementation as WebAssembly.";
  render();
} catch (error) {
  runtimeStatus.dataset.state = "error";
  runtimeStatus.textContent = error instanceof Error ? error.message : "Unable to start WebAssembly demo.";
  prefixResult.textContent = "Unavailable";
}
