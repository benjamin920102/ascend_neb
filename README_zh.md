# ascend_neb — Ascend 910B / 910C 原生 Ascend C 礦工（KEP1 v2）

> **版本 0.2.0 / correctness-first prototype**
> **無 Python、PyTorch、torch-npu、NumPy、libtorch 依賴。**
> NPU 端是真正的 Ascend C `.asc` device kernel；Host 為 C++17 + AscendCL + 官方 NEB C 參考。
>
> 已測試：Host 上 48 組獨立梯度摘要比較、work protocol parser／frame tests、C++ 主程式語法檢查（ACL stub）。
> **未測試：此環境沒有 CANN、910B／910C 設備，無法實際編譯 `.asc`、運作 NPU 或量測速度。**
> 因此目前是「原生實作原始碼」而不是已驗證量產型高效礦工。

## 1. 這次和前一版的差異

| 項目 | 前一版 | 此版本 |
|---|---|---|
| NPU API | torch-npu | **原生 Ascend C `__global__ __vector__`** |
| Host | Python | **C++17 / AscendCL** |
| CPU 參考 | 上游 C FFI | **NEB 官方 C Source 直接連結** |
| NPU 核心 | `torch.index_select` / tensor reduce | **GM GetValue + 8-lane AIV scalar/reduction** |
| H2D / D2H | torch tensors | **aclrtMalloc / aclrtMemcpyAsync / aclrtMemcpy** |
| 工作服務 | Python socket | **C++ 原生 TCP Frame** |
| 正確性驗證 | Reference C | **Reference C（每次命中、起始及每 4096 個 nonce）** |

## 2. 現有原生計算路徑

```text
NEB Node / Model packed INT4
           │
      C++ Host (AscendCL)
           │  官方 C 參考實作 (neb2_sample_grad)
           ├──── 4 個 Conv1D 梯度/sample → INT16 table ─────────┐
           └──── 2 個 Dense 層 error/input → INT8 factor table ──┤
                                                               H2D (一次/模型)
                                                                     │
                                                            Ascend NPU GM 常駐
                                                                     │
每批 Nonce: CPU BLAKE3(header) → ChaCha8 → 16×sample indices ──H2D───┤
                                                                     │
                                            Ascend C __vector__ Kernel (AIV)
                                            - 1 個 AIV block/nonce/lane
                                            - 4 層 Conv INT16 → INT32 聚合
                                            - 2 層 Dense INT8 外積 → INT32 聚合
                                            - 合計 16 samples 再 Clip[-256,256]
                                            - Uint32 XOR / modulo-2^32 sum
                                            - 8-lane digest word
                                                                     │
                                                            D2H：8×64B / nonce
                                                                     │
                            C++ Host：digest || key → BLAKE3(ticket) → target
                                                                     │
                                          **官方 neb2_share_eval 再驗證才提交**
                                                                     │
                                                       NEB Node Work Service
```

### 2.1 重要：不是假裝原生 INT4 Cube GEMM

NEB KEP1 使用**有號 INT4 權重**（−8..7），但其共識運算是 INT32 精確累加、量化右移、飽和、反向傳播、梯度裁切。
**此版本實際走「官方 C 建梯度表、Ascend Vector 掃 Nonce」路徑，並沒有每個 Nonce 重做卷積或 Cube GEMM。**

這與上游 `pow/gpu/nebhash2_table_cuda.cu` 的預計算路徑設計相似，避免每個 Nonce 再次運行全套 forward/backprop，但為了先建立可驗證的最小 Ascend C 核心，採 GM Scalar `GetValue`，**不是高吞吐 SIMD 最佳化**。下一階段才適合加入 UB Tile、Cache 重排、device-side BLAKE3/ChaCha8、或以 Cube 加速建表。

### 2.2 數值等價規則

- 函式來源：`neb2_sample_grad`、`neb2_share_eval`（官方 C）
- 各梯度必須**先合計 16 samples，再做一次** `clip(-256, 256)`。
- 對每一層局部索引 `f`，`D[f % 8] += uint32(clipped_grad) XOR uint32(f + 7919 * (layer + 1))`。
- 八個 `D[0..7]` 都是模 2³² 累加，低位元組在前。
- 最後 `ticket = BLAKE3(digest_32 || key_32)`；Target 按 little-endian 256-bit 整數比較。
- 所有候選 Share／Block 使用官方 `neb2_share_eval` 獨立重算，若不一致則**拒絕提交**。

### 2.3 記憶體配置

| 常駐資料 | 單樣本大小 | 13,404 筆 Kepler |
|---|---:|---:|
| Conv gradient (INT16) | 1,360×2 = 2,720 B | ~36.46 MB |
| Dense factors (INT8) | 834 B | ~11.18 MB |
| 合計 | 3,554 B | ~47.64 MB |

權重改變時必須重建、上傳梯度表；若幾種不同模型狀態同時存在，不能重用上一個權重狀態的梯度表。

## 3. 安裝與編譯

**建議平台：** Linux aarch64（搭配 Ascend 910B／910C）、已安裝 CANN 8.5.x／9.x 的 `ASC` CMake 編譯鏈及 AscendCL，GCC、CMake、OpenMP、Git。

```bash
unzip ascend_neb.zip
cd ascend_neb

# 第一次需要網路：取得官方 NEB v1.0.4 源碼，遵循上游 Apache-2.0 授權
bash scripts/setup_reference.sh

# 必須在已安裝 CANN 的機器上，source 正確環境腳本
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 以你上傳的 ascend_prl 設定作基礎，預設 dav-2201
NPU_ARCH=dav-2201 bash scripts/build.sh
```

如果 CANN 裝在 `/usr/local/Ascend/cann-9.0.0`：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASC_MODULES=/usr/local/Ascend/cann-9.0.0/aarch64-linux/tikcpp/ascendc_kernel_cmake/asc_modules
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux
bash scripts/build.sh
```

需視實際 910C、CANN 工具鏈環境調整 `NPU_ARCH`，不能保證所有 910B／910C 韌體都使用相同配置。

## 4. 使用方式

### 4.1 沒有 NPU 也能先做的 Host 測試

```bash
make test-host
```

### 4.2 實機一致性自測（**強制**先執行）

```bash
./build/ascend_neb selftest --device 0 --rows 48 --attempts 3 --threads 4
```

自測建立合成 KEP1 資料與權重，與上游 `neb2_share_eval` 的 Digest、Ticket 逐筆核對；不會挖礦或連線。若程式未顯示 `[PASS]`，**不要提交 Share**。

### 4.3 離線掃 Nonce

```bash
bash scripts/download_dataset.sh

./build/ascend_neb scan \
  --device 0 --threads 4 --batch 64 \
  --dataset neb-datasets.bin --slot 0 \
  --model /path/to/packed_model_13000_bytes.bin \
  --header-hex <exactly_160_hex_characters> \
  --start 0 --count 128
```

`--target-hex` 預設為 32 個 `00`，可用於無命中掃描效能測試。若要強制每個 Nonce 命中並走 C Reference：

```bash
--target-hex ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
```

### 4.4 接本機 NEB Node（Solo Worker）

```bash
./build/ascend_neb mine \
  --node 127.0.0.1:3441 \
  --dataset neb-datasets.bin \
  --device 0 --batch 64 --threads 4 --audit 3
```

例如兩卡分區（分別運行於兩個終端；共享 Node、不同 nonce 分區）：

```bash
./build/ascend_neb mine --device 0 --dataset neb-datasets.bin --partitions 2 --partition-index 0
./build/ascend_neb mine --device 1 --dataset neb-datasets.bin --partitions 2 --partition-index 1
```

支援 Job／Model Frame、Share／Submit Frame、BLAKE3 Hash 驗證、資料集固定雜湊檢查、候選 CPU Reference 審計。尚未加 pool Stratum、RPC、熱更新長期壓測及連線重試。

## 5. 檔案一覽

```text
ascend_neb/
├── CMakeLists.txt                    # 官方 CANN ASC 語言專案
├── kernel/neb_digest.asc            # 真正的 Ascend C Vector AI Core Kernel
├── src/ascend_backend.{hpp,cpp}     # AscendCL 裝置記憶體/啟動/搬運
├── src/main.cpp                     # C++ Selftest、Scan、Node Miner
├── src/native_math.hpp              # CPU reference-lite for isolated tests
├── src/{data,protocol}.{hpp,cpp}    # Dataset、原生 NEB work frame
├── native/bridge.c                  # 官方 C 參考實作 Thin ABI
├── scripts/setup_reference.sh      # 抓取上游 Apache-2.0 C Source
├── scripts/download_dataset.sh     # 主網 dataset
├── scripts/build.sh                 # 在 CANN 環境建置
├── tests/test_math.cpp              # 無 CANN 的算術測試
├── tests/test_protocol.cpp          # 原生 protocol frame 測試
├── tests/run_host_tests.sh
└── docs/INTEGRATION_AND_LIMITS.md
```

## 6. 已知限制（請先閱讀）

1. **未在 910B／910C 實機編譯。** 若你的 CANN 發行版的 `SetValue`、`dcci`、`__vector__`、`aclrtLaunch` 或 ASC CMake API 略有不同，需要針對實際 CANN SDK 微調。
2. **NPU 核心目前主要採 Scalar GetValue，而非高吞吐 SIMD。** 其目的為先建立精確 NPU 共識基線，並不保證比 CPU、CUDA 或 upstream table miner 更快。
3. Hash 與取樣目前跑 CPU C（非 Python），以保留現行 NEB 官方 C 已驗證路徑；不是 100% Device-side hashing。
4. 初次建表為 CPU/OpenMP，權重變動時重新計算；模型準備時間可能比 Nonce 掃描還長。
5. 預設只支援主網 KEP1 的 26,000 權重架構，不支援測試網 RET2、舊版 v1、或 pool Stratum。
6. PRL 雙 Kernel Cube/Vector Producer-Consumer 路徑**未直接移植**。NEB 和 PRL 的共識運算不同，直接移植會生成無效 Ticket。

## 7. 上游與致謝

- NEB 原始碼與協議：https://github.com/SilasEigen/neb （`v1.0.4` / Apache-2.0）。
- 規格：https://github.com/SilasEigen/neb/blob/main/spec/nebhash-v2.md
- Miner 文件：https://github.com/SilasEigen/neb/blob/main/docs/miner-integration.md
- Ascend C 文件：`https://www.hiascend.com/` / CANN 8.5、9.x 原生 Kernel API。
- PRL 參考：用戶提供 `ascend_prl-main (7).zip`；沿用 `find_package(ASC)`/`.asc` 編譯型態及 `dav-2201` 環境設置，不複製 PRL 的算法。
