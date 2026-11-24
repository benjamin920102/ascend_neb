#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$(realpath "$0")")"
if [[ -z ${CANN_ENV:-} ]]; then
  for p in "${ASCEND_HOME_PATH:-/nonexistent}/set_env.sh" "$HOME/Ascend/cann-9.0.0/set_env.sh" /home/developer/Ascend/cann-9.0.0/set_env.sh /usr/local/Ascend/cann-9.0.0/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    if [[ -f $p ]]; then CANN_ENV=$p; break; fi
  done
fi
[[ -f ${CANN_ENV:-} ]] || { echo 'Set CANN_ENV to your CANN set_env.sh'; exit 1; }
source "$CANN_ENV"
cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES="${ASC_ARCH:-dav-2201}"
cmake --build build -j "${JOBS:-8}"
printf '%s\n' 'Built build/libneb_int4.so. Run: python3 verify.py'
