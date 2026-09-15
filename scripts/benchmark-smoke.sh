#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

artifact_dir="$root/.artifacts/performance-smoke"
target_dir="$root/target/performance-smoke"
baseline_target_dir="$target_dir/baseline"
candidate_target_dir="$target_dir/candidate"
mkdir -p "$artifact_dir" "$baseline_target_dir" "$candidate_target_dir"

{
  printf 'repository=%s\n' "$(git remote get-url origin 2>/dev/null || printf unknown)"
  printf 'candidate=%s\n' "$(git rev-parse HEAD)"
  printf 'baseline=%s\n' "${PERF_BASE_SHA:-none}"
  printf 'rustflags=%s\n' "${RUSTFLAGS:-}"
  printf 'baseline_cargo_target_dir=%s\n' "$baseline_target_dir"
  printf 'candidate_cargo_target_dir=%s\n' "$candidate_target_dir"
  if [[ -f Cargo.lock ]]; then
    printf 'cargo_lock_sha256=%s\n' "$(sha256sum Cargo.lock | cut -d' ' -f1)"
  else
    printf 'cargo_lock=absent\n'
  fi
  rustc -vV
  cargo -V
  valgrind --version
  printf 'iai-callgrind-runner=%s\n' '0.16.1'
  uname -srm
} > "$artifact_dir/fingerprint.txt"

run_benchmark() {
  local cargo_target_dir="$1"
  local log="$2"
  shift 2
  CARGO_TARGET_DIR="$cargo_target_dir" \
    cargo bench -p search-kernels --bench performance_smoke -- "$@" \
    2>&1 | tee "$artifact_dir/$log"
}

base_sha="${PERF_BASE_SHA:-}"
bench_path="crates/search-kernels/benches/performance_smoke.rs"
baseline_name="pr_base"

if [[ -n "$base_sha" ]] && git cat-file -e "$base_sha:$bench_path" 2>/dev/null; then
  worktree_parent="$(mktemp -d)"
  baseline_dir="$worktree_parent/base"

  cleanup() {
    git worktree remove --force "$baseline_dir" >/dev/null 2>&1 || true
    rm -rf "$worktree_parent"
  }
  trap cleanup EXIT

  git worktree add --detach "$baseline_dir" "$base_sha" >/dev/null
  (
    cd "$baseline_dir"
    CARGO_TARGET_DIR="$baseline_target_dir" \
      cargo bench -p search-kernels --bench performance_smoke -- --save-baseline="$baseline_name"
  ) 2>&1 | tee "$artifact_dir/baseline.log"

  if [[ ! -d "$baseline_target_dir/iai" ]]; then
    printf 'Expected Iai-Callgrind baseline output under %s/iai\n' "$baseline_target_dir" >&2
    exit 1
  fi

  # Keep compilation artifacts isolated between the detached base worktree and
  # the candidate checkout. Sharing one Cargo target directory can let Cargo
  # reuse the base benchmark executable because both builds have the same
  # package name and target. Only copy Iai's saved baseline data so the
  # candidate is rebuilt from HEAD while still comparing against the base.
  rm -rf "$candidate_target_dir/iai"
  cp -a "$baseline_target_dir/iai" "$candidate_target_dir/iai"
  run_benchmark "$candidate_target_dir" candidate.log --baseline="$baseline_name"
else
  printf '%s\n' \
    'No compatible base benchmark exists; this run seeds the performance-smoke contract.' \
    | tee "$artifact_dir/baseline.log"
  run_benchmark "$candidate_target_dir" candidate.log --save-baseline=seed
fi
