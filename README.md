# Python INT4 → Ascend C (910C)

This replaces the five Python matrix products with native Ascend C Cube GEMMs:

| Operation | Operand types | Output |
| --- | --- | --- |
| X × W1 | signed INT4 × signed INT4 | INT32 |
| A1 × W2 | signed INT4 × signed INT4 | INT32 |
| E2ᵀ × A1 | signed INT4 × signed INT4 | INT32 |
| E2 × W2ᵀ | signed INT4 × signed INT4 | INT32 |
| E1ᵀ × X | signed INT4 × signed INT4 | INT32 |

Operands use two signed 4-bit values per byte, low nibble first. The physical GEMM is 16×64×64; smaller logical matrices are zero padded and outputs cropped. Gradients retain the original orientation. Python still performs arithmetic right shifts, clipping, sampling, hashing, cache aggregation, and network operations. This is not an all-NPU implementation. No Torch, FP16, FP32, or INT8 GEMM fallback is used. Byte buffers carry packed INT4 data; INT32 accumulation/output is intentional.

## Build and validate

Use the CANN 9.0 ASC environment from the supplied working example, Python 3.10+, CMake, and a 910C device. The default architecture is dav-2201.

```bash
python3 -m pip install -r requirements.txt
export CANN_ENV=/home/developer/Ascend/cann-9.0.0/set_env.sh
source "$CANN_ENV"
bash build.sh
NEB_ASC_DEVICE=0 python3 verify.py --cases 64
NEB_ASC_DEVICE=1 python3 verify.py --cases 64
```

Run verification before use. It checks raw GEMMs, randomized and edge-value G1/G2 outputs, every fixed cache row, and final digests/tickets in both execution modes, against Python integer arithmetic. A missing library, tiling failure, ACL error, or numerical mismatch stops execution. There is no silent CPU fallback.

This package was checked locally for Python syntax and bridge routing with a CPU test double. CANN compilation and execution on 910C were unavailable in the preparation environment; hardware validation remains necessary. The CPU test double is not included in the package.

## Use

The original command-line options remain available:

```bash
NEB_ASC_DEVICE=0 python3 neb_asc.py --workers 1
```

Use `NEB_ASC_DEVICE=1` to select the other device. Each process uses one selected device; this package does not combine two chips into one GEMM. Multiple workers create separate contexts on that selected device.

The optimized original computes fixed per-sample contributions once and reuses them. The default preserves this: all five GEMMs run on ASC during the 256-row cache build. Subsequent tickets aggregate those contributions on the CPU. This equality depends on the unchanged fixed dataset and weights. Call `rebuild_fixed_cache()` after deliberately changing them, without concurrent ticket computation.

For all five GEMMs to run on ASC for every ticket, disable caching:

```bash
NEB_ASC_CACHE=0 NEB_ASC_DEVICE=0 python3 neb_asc.py --workers 1
```

Direct mode is expected to be slower for these small matrices because each GEMM synchronizes and transfers buffers. No throughput claim is made. Cache construction also performs 1,280 GEMM launches per process. Contexts and device buffers persist across calls. The multiprocessing client uses spawn, and never shares initialized ACL contexts across fork.

## Files

- `neb_asc.py`: modified original application; `reference_int4_train_step` is used only by verification.
- `asc_backend.py`: packing, shape validation, ctypes bridge, persistent context.
- `matmul_int4.asc`: INT4 Cube kernel and launch/tiling wrappers, adapted from the supplied sample.
- `bridge.cpp`: C ABI, ACL buffers, synchronization and error propagation.
- `verify.py`: exact comparisons to the original integer algorithm.
- `CMakeLists.txt`, `build.sh`: shared library build.

The kernel retains the supplied Huawei copyright/license notice. The supplied archive did not contain the LICENSE file referenced by that notice; obtain the CANN Open Software License Agreement Version 2.0 from your CANN distribution for its complete terms.
