#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build-host-tests
c++ -O2 -Wall -Wextra -std=c++17 -Isrc tests/test_math.cpp -o build-host-tests/math
c++ -O2 -Wall -Wextra -std=c++17 -Isrc tests/test_protocol.cpp src/protocol.cpp -o build-host-tests/protocol
./build-host-tests/math
./build-host-tests/protocol
bash scripts/check_no_torch.sh
