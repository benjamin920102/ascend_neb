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

## 實際挖礦教學

### 1. 在 Ascend 910C 主機上編譯並驗證

登入裝有 Ascend 910C 的 Linux 主機，切到本專案目錄，依照上面的 [Build and validate](#build-and-validate) 安裝依賴、載入 CANN 環境並編譯。把 `CANN_ENV` 改成主機上實際的 `set_env.sh` 路徑。編譯完成後會產生 `build/libneb_int4.so`，請在準備使用的裝置上跑 `verify.py`，確認最後顯示 `All checked cases passed on the selected backend.`。只有一張卡時，先用 `NEB_ASC_DEVICE=0` 即可。每次開新的終端機都要重新載入 CANN 環境。

### 2. 先用內建模擬礦池試挖

```bash
NEB_ASC_DEVICE=0 python3 -u neb_asc.py \
  --host 127.0.0.1 --port 13333 --worker neb_wallet.test --workers 1
```

這條指令會在同一個程序啟動本機模擬礦池和礦工。收到 `job_id=job-001` 後，礦工會初始化 NPU、建立快取並開始掃描 nonce。`[Hashrate]` 代表正在運算，`[Share]` 代表找到並嘗試提交候選解。等到 `[MockPool]` 印出 `valid=True`，就確認了接收工作、運算、提交與驗證的完整流程。第一筆 share 的等待時間會隨算力與命中情況變動。按 `Ctrl+C` 停止。模擬礦池的 share 僅供本機測試，沒有鏈上收益。

### 3. 連線到正式礦池挖礦

向礦池取得相容的 NEB 挖礦主機、TCP 埠與 worker 登入名稱。礦池須支援本程式使用的逐行 JSON 協定，以物件形式發送 `mining.notify` 的 `params`，其中含 `job_id`、`header_template_76`、`target_hex`。程式使用明文 TCP，並在 `mining.authorize` 中送出固定密碼 `x`。worker 命名與收款地址的設定方式依礦池規則為準。

將以下三個範例值換成礦池提供的資料。`--host` 填主機名稱或 IP，`--port` 填數字埠號。`--no-mock` 會停用內建模擬礦池，直接連到指定端點。

```bash
NEB_ASC_DEVICE=0 python3 -u neb_asc.py --no-mock \
  --host pool.example.com --port 3333 \
  --worker 'YOUR_WALLET.worker1' --workers 1
```

- 連線後等到程式顯示 `job_id=`，才會開始掃描 nonce。若連線成功卻一直收不到工作，先核對 worker 登入格式與礦池的 `mining.notify` 格式。
- `[Share]` 表示嘗試提交。訂閱與登入也可能回傳 `result=True`，請以提交後對應的礦池回應或礦池後台確認 share 是否被接受。收益與付款請以礦池後台為準。
- 若反覆重連，先核對主機和埠號，並確認該端點接受明文 TCP。若出現共享函式庫或 ACL 錯誤，先確認已載入 CANN 環境，再用同一個裝置編號重跑 `verify.py`。

### 4. 選擇裝置與快取模式

先用 `--workers 1`。有第二張已驗證的裝置時，可另開終端機，用相同的正式礦池指令，把 `NEB_ASC_DEVICE` 改成 `1`，並依礦池規則設定第二個 worker 名稱。`--workers` 增加的是同一張卡上的程序數，每個程序各自建立 NPU context。

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
