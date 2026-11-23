#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if grep -R -nE '(#include\s*[<"]torch|#include\s*[<"]ATen|import torch|torch_npu|find_package\(Torch|libtorch)' \
   src kernel native CMakeLists.txt; then
  echo 'ERROR: found a forbidden torch dependency' >&2;exit 1
fi
echo 'PASS: no torch / torch-npu / libtorch dependency'
