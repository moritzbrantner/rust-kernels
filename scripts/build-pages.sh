#!/usr/bin/env bash
set -euo pipefail

rustup target add wasm32-unknown-unknown
cargo test -p rust-kernels-web-demo
cargo build --release -p rust-kernels-web-demo --target wasm32-unknown-unknown

rm -rf pages-dist
mkdir -p pages-dist/pkg
cp -R site/. pages-dist/
cp \
  target/wasm32-unknown-unknown/release/rust_kernels_web_demo.wasm \
  pages-dist/pkg/rust_kernels_web_demo.wasm

# Revision identity for benchmarks.html output; "dirty" marks uncommitted changes.
revision="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if git diff --quiet HEAD -- 2>/dev/null; then dirty=false; else dirty=true; fi
printf '{"revision":"%s","dirty":%s}\n' "$revision" "$dirty" > pages-dist/build-info.json

printf 'Pages artifact ready: %s\n' "$(du -sh pages-dist | cut -f1)"
