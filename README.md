# CPython optimized version

Requires Python 3.10+ and the real BLAKE3 package.

```bash
python -m pip install -r requirements.txt
python neb_cpython_optimized.py --workers 2
```

The default runs a local mock server. Stop with Ctrl+C.
For an endpoint using the exact custom JSON format in the supplied code:

```bash
python neb_cpython_optimized.py --no-mock --host HOST --port 13333 --worker WALLET.WORKER --workers 2
```

## What changed

- Fixed deletion from an immutable bytes buffer in ChaCha8Stream; use an offset and unpack_from instead.
- Keep ChaCha8 quarter-round words in local variables.
- Precompute per-sample raw gradient contributions for the fixed 256-row dataset and fixed weights. Sum all 16 selected samples before applying the reference gradient clamp.
- Use one ChaCha8 block directly for the 16 sample indices.
- Add multiple worker processes to run Python integer work on multiple CPU cores.
- Scan complete batches and return every qualifying result; keep nonce accounting accurate. Drop completed results when their job has been replaced.
- Require genuine BLAKE3 rather than silently substituting SHA-256, which produces different results.

## Correctness and limits

261 complete tickets/digests/integers matched the supplied reference with only its fatal buffer defect repaired. Dataset, weights, 5,140 stream words, nonce wrap, complete-batch results, and two-process local TCP submission were checked. See validation.txt for measured single-process warm-cache throughput; it excludes startup, process communication and network overhead. Performance depends on your CPU and workload.

The cache is valid because the dataset and weights stay fixed; this program does not update weights. If you deliberately edit FIXED_X, FIXED_T, INIT_W1 or INIT_W2, call rebuild_fixed_cache before further work in every process. Concurrent cache edits are unsupported. Inputs and weights must remain integers in [-8, 7].

This is Python arbitrary-precision integer code with INT4-range values; it does not execute native INT4 hardware instructions. Results match the supplied model, which is not evidence of compatibility with any actual chain or external pool. The built-in protocol is the custom format supplied in the attachment, not a universal Stratum adapter.

The source function deterministic_int4_train_step remains available as a generic reference path. Cache building consumes additional memory per worker, and workers do not share a mutable dataset. The default mock target can require thousands of candidates before a result appears.
