#!/usr/bin/env bash
# This project builds the upstream C files directly with CMake; no separate library is required.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NEB_UPSTREAM_TAG:=v1.0.4}"
if [ -f vendor/neb/pow/ref/nebhash2.c ]; then
  echo '[OK] upstream NEB C reference already present at vendor/neb'
  exit 0
fi
mkdir -p vendor
git clone --depth 1 --branch "$NEB_UPSTREAM_TAG" \
  https://github.com/SilasEigen/neb.git vendor/neb
for f in nebhash2.c shares.c chacha8.c blake3_min.c nebhash2.h shares.h; do
  test -f "vendor/neb/pow/ref/$f" || { echo "Missing $f" >&2; exit 2; }
done
echo "[OK] pinned NEB source: ${NEB_UPSTREAM_TAG} (Apache-2.0)"
