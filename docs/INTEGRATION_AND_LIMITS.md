# Native Ascend C integration and correctness boundaries

## Reference provenance

- Upstream: `SilasEigen/neb`, pinned tag `v1.0.4` by `scripts/setup_reference.sh`.
- KEP1 layer sizes: `[40,640,40,640,24576,64]`, total 26,000 weights.
- Per-sample convolution gradient stored as signed `int16_t`, with explicit out-of-range check in `native/bridge.c`.
- Linear gradient factorization: `d5[32], x5[768], d6[2], x6[32]`: 834 bytes/sample. This matches the exact `neb2_sample_grad` factors and computes each linear weight's gradient as the 16-sample sum of signed-error times signed-input.
- Aggregation: all gradients use 32-bit signed intermediate; clip after 16 samples to signed [-256,256].
- Lane formula: digest word `D[f&7] += (uint32_t)clip(g[f]) XOR (f + 7919*(layer+1))` within each layer.
- 8 lanes from independent blocks, no inter-block atomic accumulation. Each block writes an isolated aligned 64B cache line. Host extracts word 0 and encodes 8 little-endian words.
- Ticket and target comparison are CPU BLAKE3 and LE256: no device-side floating point or unsupported instruction behavior.

## Ascend kernel launch

`kernel/neb_digest.asc` declares an ASC compiled `__global__ __vector__` kernel and a C-compatible host launcher `neb_launch_digest()`. The code uses `AscendC::GlobalTensor<T>::GetValue()` for GM scalar reads and `SetValue()` + explicit `dcci` writeback. Both are intentionally simple primitives used in the user's PRL Ascend C reference project and documented in CANN.

The kernel is not a Cube GEMM. The branch's original INT4 *weights* only appear in CPU construction of per-sample gradient factors; per-attempt work is integer gather + reduction. Optimizing using Cube Matmul is a distinct implementation project because it would need full KEP1 quantization, pooling, Backprop and gradient semantics; don't conflate weight storage type with hardware matrix instruction type.

## Critical tests

1. `tests/run_host_tests.sh`: integer digest and framing; independent of NPU and upstream checkout.
2. `build/ascend_neb selftest --device 0`: builds deterministic synthetic model/dataset from pinned C reference, uploads table, tests several nonces; for every nonce compares device digest+ticket with upstream C reference.
3. `scan --target-hex ff...`: intentionally makes every nonce a hit; each is independently rechecked by the C reference. On any mismatch, throws and refuses submission.
4. Real network `mine`: submitted share or block passes C reference before transmission; on first few nonces and each nonce divisible by 4096 a reference audit is run even without hits.

## Planned optimization increments (not implemented)

- A1: change scalar GM factors to UB-tiled/coalesced reads, 16-sample x 8-lane SIMD, measure NPU utilization and memory bandwidth.
- A2: fuse sample index generation (ChaCha8), key+ticket (BLAKE3) within NPU; avoid per-attempt H2D indices and D2H digest. 910C may not have a native XOR vector instruction; use rigorously checked bitwise emulation if necessary.
- A3: per-model table generation with Ascend C native Conv1D and Backprop, optionally Cube INT4/INT8 for im2col, with intermediate GM boundary if necessary. Every per-sample gradient must match `neb2_sample_grad`.
- A4: model double buffering, async Host table build, work cancellation, sustained node reconnect and jitter/stale-share metrics.

No speed claims before a real CANN compile and benchmark on a stated 910B/910C, firmware/CANN version, dataset and model state.
