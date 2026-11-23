#!/usr/bin/env bash
# Download from exact tagged release; native scanner verifies the consensus BLAKE3 hash.
set -euo pipefail
cd "$(dirname "$0")/.."
url="https://github.com/SilasEigen/neb/releases/download/v1.0.4/neb-datasets.bin"
echo "Fetching official NEB dataset from $url"
curl --fail --location --retry 2 --output neb-datasets.bin.part "$url"
mv neb-datasets.bin.part neb-datasets.bin
echo 'Download complete. Runtime verifies official pinned BLAKE3 digest.'
