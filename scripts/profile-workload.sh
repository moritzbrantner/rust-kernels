#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <search|lru|traversal>\n' "$0" >&2
  exit 2
fi

root="$(git rev-parse --show-toplevel)"
cd "$root"

case "$1" in
  search)
    package="search-kernels"
    crate_dir="crates/search-kernels"
    example="profile_search"
    ;;
  lru)
    package="collection-kernels"
    crate_dir="crates/collection-kernels"
    example="profile_lru"
    ;;
  traversal)
    package="graph-kernels"
    crate_dir="crates/graph-kernels"
    example="profile_traversal"
    ;;
  *)
    printf 'unknown workload: %s\n' "$1" >&2
    exit 2
    ;;
esac

binary="$root/target/release/examples/$example"
needs_build=0
if [[ ! -x "$binary" ]]; then
  needs_build=1
elif [[ "$root/Cargo.toml" -nt "$binary" || "$root/Cargo.lock" -nt "$binary" ]]; then
  needs_build=1
elif find "$root/$crate_dir" -type f \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "$binary" -print -quit | grep -q .; then
  needs_build=1
fi

if [[ "$needs_build" -eq 1 ]]; then
  cargo build --quiet --release -p "$package" --example "$example"
fi

exec "$binary"
