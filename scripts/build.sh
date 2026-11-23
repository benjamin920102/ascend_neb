#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NPU_ARCH:=dav-2201}"
if [ ! -f vendor/neb/pow/ref/nebhash2.c ]; then
  echo '[ERROR] official C reference missing; run scripts/setup_reference.sh' >&2
  exit 2
fi
# source the actual environment where CANN was installed
if [ -z "${ASCEND_HOME_PATH:-}" ] && [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
  for envfile in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh /usr/local/Ascend/cann-9.0.0/set_env.sh; do
    if [ -f "$envfile" ]; then
      # shellcheck disable=SC1090
      source "$envfile"
      break
    fi
  done
fi
: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
if [ -z "${ASC_MODULES:-}" ]; then
  for candidate in "$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/asc_modules" \
      "$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake/asc_modules" \
      "$ASCEND_HOME_PATH/tikcpp/ascendc_kernel_cmake/asc_modules"; do
    if [ -d "$candidate" ]; then ASC_MODULES="$candidate";break;fi
  done
fi
if [ -z "${ASC_MODULES:-}" ]; then
  echo '[ERROR] CANN ASC cmake modules not found; please set ASC_MODULES' >&2
  exit 2
fi
cmake -S . -B build -DCMAKE_MODULE_PATH="$ASC_MODULES" \
      -DCMAKE_ASC_RUN_MODE=npu -DCMAKE_ASC_ARCHITECTURES="$NPU_ARCH" \
      -DASCEND_HOME="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo '[OK] build/ascend_neb (C++ host + native Ascend C kernel)'
