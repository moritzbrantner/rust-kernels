#!/usr/bin/env bash
# Collector for the root `performance:work` capability.
#
# Runs the Cargo tests that measure the deterministic work counters for the
# operations declared in .performance/contract.json and prints one
# `coding-tooling/work-evidence/v1` document on stdout. Cargo output goes to
# stderr so stdout carries only the evidence. Each test checks correctness
# against its oracle before it writes `<operation>.json` into
# RUST_KERNELS_WORK_EVIDENCE_DIR; with that variable set, the tests leave the
# work bounds to `coding-tooling performance work`, and without it they
# assert the same bounds themselves.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

contract=".performance/contract.json"
suite="rust-kernels"

evidence_dir="$(mktemp -d)"
trap 'rm -rf "$evidence_dir"' EXIT
export RUST_KERNELS_WORK_EVIDENCE_DIR="$evidence_dir"

run_tests() {
  cargo test --quiet "$@" >&2
}

run_tests -p collection-kernels --test work_complexity -- --exact \
  fenwick_linear_build_work_scales_linearly
run_tests -p graph-kernels --lib -- --exact \
  topology::tests::iterative_tarjan_edge_visits_scale_linearly_in_stored_edges
run_tests -p similarity-kernels --test work_complexity -- --exact \
  bounded_edit_comparisons_scale_linearly_with_a_fixed_limit \
  warmed_bounded_edit_queries_allocate_nothing_at_any_scale
run_tests -p geometry-kernels --lib -- --exact \
  epa::tests::disabled_trace_copies_stay_zero_across_iteration_caps
run_tests -p geometry-kernels --test work_complexity -- --exact \
  easy_gjk_sphere_iterations_do_not_grow_with_coordinate_scale

# Contract order; a missing fragment means a test did not run or failed.
operations=(
  fenwick-linear-build
  scc-cycle-edge-visits
  bounded-edit-comparisons
  bounded-edit-warm-allocations
  epa-disabled-trace-copies
  gjk-easy-sphere-iterations
)
fragments=()
for operation in "${operations[@]}"; do
  fragment="$evidence_dir/$operation.json"
  if [[ ! -s "$fragment" ]]; then
    printf 'missing work evidence for %s\n' "$operation" >&2
    exit 1
  fi
  fragments+=("$(<"$fragment")")
done

# Hash the contract bytes as they are on disk now, after the tests ran.
if command -v sha256sum >/dev/null 2>&1; then
  contract_sha="$(sha256sum "$contract" | cut -d' ' -f1)"
else
  contract_sha="$(shasum -a 256 "$contract" | cut -d' ' -f1)"
fi

joined="$(IFS=,; printf '%s' "${fragments[*]}")"
printf '{"schemaVersion":"coding-tooling/work-evidence/v1","suite":"%s","contractSha256":"%s","operations":[%s]}\n' \
  "$suite" "$contract_sha" "$joined"
