#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# src/capi.mojo is the single entry point; -I src resolves the `ported` module
# that holds the kernels, so build cost stays a single compilation unit.
#
# --fp-mode contract=off matters: Mojo defaults to fusing `a * b - c * d` into
# an FMA, which rounds once instead of twice and so disagrees with CPython in
# the last bits.  Upstream is plain Python arithmetic, so we turn it off.
mojo build --emit shared-lib --fp-mode contract=off -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-sgp4.so"

# The wheel ships the library inside the package, so a copy lives there too;
# mojosgp4/_lib.py prefers it and falls back to dist/ in a checkout.
cp "$repo_dir/dist/libmojo-sgp4.so" "$repo_dir/python/mojosgp4/libmojo-sgp4.so"
