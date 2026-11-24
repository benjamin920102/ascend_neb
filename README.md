# Ascend INT4 KEP1 miner

The mining CLI alternates between Kepler at even block heights and TESS at odd
heights. It uses the dataset and current KEP1 weights carried by each job. The
mock pool supplies deterministic synthetic fixtures and keeps each model and
its residuals in memory.

## Build and validate

On an Ascend 910C Linux host with CANN 9.0 ASC configured:

```bash
python3 -m pip install -r requirements.txt
source /home/developer/Ascend/cann-9.0.0/set_env.sh
bash build.sh
NEB_ASC_DEVICE=0 python3 verify.py --cases 64
NEB_ASC_DEVICE=0 python3 -u neb_asc.py --workers 1
```

The default mock pool validates and queues shares. When it accepts a block
solution, it includes the queued shares, updates the corresponding model, and
broadcasts a new job. The client uses a spawned process and a persistent ASC
context for each worker. Use `NEB_ASC_CACHE=0` to recompute gradients for every
ticket. The default caches sample contributions by dataset contents and weights.

## KEP1 parameters implemented here

Each sample contains a 16-element INT4 signal and a 4-element INT4 target.
The two single-filter branches use valid Conv1D with widths 3 and 5 and stride
1. Each branch uses an arithmetic right shift by 4 and INT4 clipping, then a
global max pool with earliest-index tie breaking. The linear layer maps the
two pooled values to four outputs with the same shift and clip. Backprop uses
clipped prediction-minus-target and routes each branch gradient to its pooling
winner. Convolutions, linear projection, and error propagation use ASC INT4
GEMMs. Per-sample outer products and aggregation use integer arithmetic.

The 16 weights and gradient elements are ordered as width-3 kernel, width-5
kernel, then the 2-by-4 linear matrix in row-major order. ChaCha8 starts at
counter 0 with a zero nonce and draws 16 little-endian uint32 words. Each draw
is reduced modulo the dataset size, allowing repeated samples. The sum of the
16 gradients is clipped to [-256, 256] before folding into eight little-endian
uint32 lanes with the existing digest function. The header nonce is uint32
little-endian. The ticket is interpreted as a big-endian uint256 for comparison.

For each included share, the mock pool adds the clipped gradient to the
corresponding model's residual. Once per accepted block, a magnitude of at
least 256 requests one INT4 weight step opposite the residual. The actual
movement is subtracted from the residual in units of 256. Unused residual
carries forward, including at the INT4 bounds.

## Pool interface

Use `--no-mock --host HOST --port PORT --worker WALLET.worker` with a pool that
implements this newline-delimited JSON protocol. A `mining.notify` object has
a `params` object containing:

- `job_id`, integer `height`, and 76-byte `header_template_76` in hex.
- `block_target_hex` and `share_target_hex`, with block target <= share target.
- `dataset` (`Kepler` or `TESS`), `model` (`KEP1`), `samples` (a list of
  [signal, target] pairs), and the ordered INT4 `weights`.
- Optional integer `start_nonce` (default 0) and `nonce_step` (default 1).

The client submits `mining.submit_block` at or below the block target, or
`mining.submit` at or below the share target. Both methods carry
`[worker, job_id, nonce_hex, ticket_hex, digest_hex]`. Configure a real pool
with matching KEP1 parameters, quantized datasets, and current model state
before using these messages outside the mock. The previous fixed workload
remains in the verification code for existing integer and ASC regression
checks.

The kernel retains the supplied Huawei copyright/license notice. The supplied archive did not contain the LICENSE file referenced by that notice; obtain the CANN Open Software License Agreement Version 2.0 from your CANN distribution for its complete terms.
