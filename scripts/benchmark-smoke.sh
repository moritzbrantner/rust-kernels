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

# Each package's Iai-Callgrind sentinels: compared against the pull request base
# when the base has the bench (5% soft Ir limit, fail fast), seeded otherwise.
benches=(
  "search-kernels:crates/search-kernels/benches/performance_smoke.rs"
  "geometry-kernels:crates/geometry-kernels/benches/performance_smoke.rs"
)
base_sha="${PERF_BASE_SHA:-}"
baseline_name="pr_base"
baseline_dir=""

cleanup() {
  if [[ -n "$baseline_dir" ]]; then
    git worktree remove --force "$baseline_dir" >/dev/null 2>&1 || true
    rm -rf "$(dirname "$baseline_dir")"
  fi
}
trap cleanup EXIT

for entry in "${benches[@]}"; do
  package="${entry%%:*}"
  bench_path="${entry#*:}"
  if [[ -n "$base_sha" ]] && git cat-file -e "$base_sha:$bench_path" 2>/dev/null; then
    if [[ -z "$baseline_dir" ]]; then
      baseline_dir="$(mktemp -d)/base"
      git worktree add --detach "$baseline_dir" "$base_sha" >/dev/null
    fi
    (
      cd "$baseline_dir"
      CARGO_TARGET_DIR="$baseline_target_dir" \
        cargo bench -p "$package" --bench performance_smoke -- --save-baseline="$baseline_name"
    ) 2>&1 | tee "$artifact_dir/$package-baseline.log"

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
    CARGO_TARGET_DIR="$candidate_target_dir" \
      cargo bench -p "$package" --bench performance_smoke -- --baseline="$baseline_name" \
      2>&1 | tee "$artifact_dir/$package-candidate.log"
  else
    printf '%s: no compatible base benchmark exists; this run seeds its performance-smoke contract.\n' \
      "$package" | tee "$artifact_dir/$package-baseline.log"
    CARGO_TARGET_DIR="$candidate_target_dir" \
      cargo bench -p "$package" --bench performance_smoke -- --save-baseline=seed \
      2>&1 | tee "$artifact_dir/$package-candidate.log"
  fi
done
