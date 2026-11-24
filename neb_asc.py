import asyncio
import json
import os
import struct
import time
from math import gcd

import kep1
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple

try:
    import blake3 as _blake3_mod
except ImportError as exc:
    raise ImportError("BLAKE3 is required: python -m pip install blake3") from exc


def blake3_digest(data: bytes, length: int = 32) -> bytes:
    return _blake3_mod.blake3(data).digest(length=length)


# ==============================================================================
# 2. 密碼學種子與 ChaCha8 樣本抽選器
# ==============================================================================

def _rotl32(v: int, c: int) -> int:
    v &= 0xFFFFFFFF
    return ((v << c) & 0xFFFFFFFF) | (v >> (32 - c))


def _quarter_round(x: List[int], a: int, b: int, c: int, d: int) -> None:
    # Keep the round state in locals: fewer list reads and Python calls.
    aa, bb, cc, dd = x[a], x[b], x[c], x[d]
    aa = (aa + bb) & 0xFFFFFFFF
    dd ^= aa; dd = ((dd << 16) & 0xFFFFFFFF) | (dd >> 16)
    cc = (cc + dd) & 0xFFFFFFFF
    bb ^= cc; bb = ((bb << 12) & 0xFFFFFFFF) | (bb >> 20)
    aa = (aa + bb) & 0xFFFFFFFF
    dd ^= aa; dd = ((dd << 8) & 0xFFFFFFFF) | (dd >> 24)
    cc = (cc + dd) & 0xFFFFFFFF
    bb ^= cc; bb = ((bb << 7) & 0xFFFFFFFF) | (bb >> 25)
    x[a], x[b], x[c], x[d] = aa, bb, cc, dd


def chacha8_block(key32: bytes, counter: int, nonce12: bytes = b"\x00" * 12) -> bytes:
    """標準 ChaCha8（8 rounds = 4 double rounds）單區塊 64 bytes 輸出"""
    if len(key32) != 32:
        raise ValueError("ChaCha8 key must be 32 bytes")
    if len(nonce12) != 12:
        raise ValueError("ChaCha8 nonce must be 12 bytes")

    constants = struct.unpack("<4I", b"expand 32-byte k")
    key_words = struct.unpack("<8I", key32)
    nonce_words = struct.unpack("<3I", nonce12)

    state = [
        *constants,
        *key_words,
        counter & 0xFFFFFFFF,
        *nonce_words,
    ]
    working = state.copy()

    for _ in range(4):  # 8 rounds
        # Column rounds
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        # Diagonal rounds
        _quarter_round(working, 0, 5, 10, 15)
        _quarter_round(working, 1, 6, 11, 12)
        _quarter_round(working, 2, 7, 8, 13)
        _quarter_round(working, 3, 4, 9, 14)

    out_words = [(working[i] + state[i]) & 0xFFFFFFFF for i in range(16)]
    return struct.pack("<16I", *out_words)


class ChaCha8Stream:
    def __init__(self, key32: bytes):
        if len(key32) != 32:
            raise ValueError("ChaCha8 key must be 32 bytes")
        self.key32 = key32
        self.counter = 0
        self.buf = b""
        self.offset = 64

    def read_u32(self) -> int:
        if self.offset == 64:
            self.buf = chacha8_block(self.key32, self.counter)
            self.counter = (self.counter + 1) & 0xFFFFFFFF
            self.offset = 0
        val = struct.unpack_from("<I", self.buf, self.offset)[0]
        self.offset += 4
        return val


# ==============================================================================
# 鏈上固定資料集與 INT4 權重（可替換為節點同步之真實鏈上資料）
# ==============================================================================

INPUT_DIM = 16
HIDDEN_DIM = 16
OUTPUT_DIM = 4
DATASET_SIZE = 256


def build_fixed_onchain_dataset(size: int = DATASET_SIZE) -> Tuple[List[List[int]], List[List[int]]]:
    """
    產生確定性固定資料集 (X, T)：
    - X: 每筆長度 INPUT_DIM，元素為 INT4 範圍 [-8, 7]
    - T: 每筆長度 OUTPUT_DIM，元素為 INT4 範圍 [-8, 7]
    實務上應由鏈上狀態/快照載入相同資料。
    """
    seed = blake3_digest(b"NEB_ONCHAIN_FIXED_DATASET_V1", 32)
    stream = ChaCha8Stream(seed)
    xs: List[List[int]] = []
    ts: List[List[int]] = []
    for _ in range(size):
        x_row = [(stream.read_u32() & 0x0F) - 8 for _ in range(INPUT_DIM)]
        t_row = [(stream.read_u32() & 0x0F) - 8 for _ in range(OUTPUT_DIM)]
        xs.append(x_row)
        ts.append(t_row)
    return xs, ts


def build_initial_int4_weights() -> Tuple[List[List[int]], List[List[int]]]:
    """確定性初始 INT4 權重 W1 (INPUT_DIM x HIDDEN_DIM), W2 (HIDDEN_DIM x OUTPUT_DIM)"""
    seed = blake3_digest(b"NEB_INITIAL_INT4_WEIGHTS_V1", 32)
    stream = ChaCha8Stream(seed)
    w1 = [[(stream.read_u32() & 0x0F) - 8 for _ in range(HIDDEN_DIM)] for _ in range(INPUT_DIM)]
    w2 = [[(stream.read_u32() & 0x0F) - 8 for _ in range(OUTPUT_DIM)] for _ in range(HIDDEN_DIM)]
    return w1, w2


FIXED_X, FIXED_T = build_fixed_onchain_dataset()
INIT_W1, INIT_W2 = build_initial_int4_weights()


# ==============================================================================
# 3~6. 確定性 INT4/INT32 訓練、梯度折疊與 Ticket 憑證計算
# ==============================================================================

def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _sar(v: int, shift: int) -> int:
    """算術右移（整數除法向負無窮對齊，確保跨平台確定性）"""
    return v >> shift


def sample_16_from_dataset(key32: bytes) -> Tuple[List[List[int]], List[List[int]]]:
    """步驟 3：ChaCha8(Key) 抽取 16 筆固定資料集樣本 (X_batch, T_batch)"""
    stream = ChaCha8Stream(key32)
    x_batch: List[List[int]] = []
    t_batch: List[List[int]] = []
    for _ in range(16):
        idx = stream.read_u32() % DATASET_SIZE
        x_batch.append(FIXED_X[idx])
        t_batch.append(FIXED_T[idx])
    return x_batch, t_batch


def reference_int4_train_step(
    x_batch: List[List[int]],
    t_batch: List[List[int]],
    w1: List[List[int]],
    w2: List[List[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    步驟 4：確定性神經網路訓練（INT4 權重 [-8, 7] + INT32 精確整數運算）
    Batch size B = 16
    """
    b_size = len(x_batch)

    # --- Forward Pass ---
    # Z1 = X @ W1 (16 x 16)
    a1: List[List[int]] = [[0] * HIDDEN_DIM for _ in range(b_size)]
    for i in range(b_size):
        xi = x_batch[i]
        for j in range(HIDDEN_DIM):
            acc = 0
            for k in range(INPUT_DIM):
                acc += xi[k] * w1[k][j]
            a1[i][j] = _clamp(_sar(acc, 4), -8, 7)

    # Z2 = A1 @ W2 (16 x 4)
    y2: List[List[int]] = [[0] * OUTPUT_DIM for _ in range(b_size)]
    for i in range(b_size):
        a1i = a1[i]
        for j in range(OUTPUT_DIM):
            acc = 0
            for k in range(HIDDEN_DIM):
                acc += a1i[k] * w2[k][j]
            y2[i][j] = _clamp(_sar(acc, 4), -8, 7)

    # --- Backward Pass ---
    # E2 = Clip(Y2 - T, -8, 7) (16 x 4)
    e2: List[List[int]] = [
        [_clamp(y2[i][j] - t_batch[i][j], -8, 7) for j in range(OUTPUT_DIM)]
        for i in range(b_size)
    ]

    # G2 = Clip(E2^T @ A1, -256, 256) (4 x 16)
    g2: List[List[int]] = [[0] * HIDDEN_DIM for _ in range(OUTPUT_DIM)]
    for o in range(OUTPUT_DIM):
        for h in range(HIDDEN_DIM):
            acc = 0
            for i in range(b_size):
                acc += e2[i][o] * a1[i][h]
            g2[o][h] = _clamp(acc, -256, 256)

    # 反推隱藏層誤差 E1 = Clip((E2 @ W2^T) >> 4, -8, 7) (16 x 16)
    e1: List[List[int]] = [[0] * HIDDEN_DIM for _ in range(b_size)]
    for i in range(b_size):
        e2i = e2[i]
        for h in range(HIDDEN_DIM):
            acc = 0
            for o in range(OUTPUT_DIM):
                acc += e2i[o] * w2[h][o]
            e1[i][h] = _clamp(_sar(acc, 4), -8, 7)

    # G1 = Clip(E1^T @ X, -256, 256) (16 x 16)
    g1: List[List[int]] = [[0] * INPUT_DIM for _ in range(HIDDEN_DIM)]
    for h in range(HIDDEN_DIM):
        for inp in range(INPUT_DIM):
            acc = 0
            for i in range(b_size):
                acc += e1[i][h] * x_batch[i][inp]
            g1[h][inp] = _clamp(acc, -256, 256)

    return g1, g2



def deterministic_int4_train_step(x_batch, t_batch, w1, w2):
    """All five GEMMs: packed INT4 Cube -> INT32. Shift/clamp on CPU."""
    from asc_backend import get_backend
    mm = get_backend().matmul
    b_size = len(x_batch)
    if not 1 <= b_size <= 16:
        raise ValueError("ASC training batch must contain 1..16 rows")
    for matrix, height, width in ((x_batch, b_size, INPUT_DIM),
                                  (t_batch, b_size, OUTPUT_DIM),
                                  (w1, INPUT_DIM, HIDDEN_DIM),
                                  (w2, HIDDEN_DIM, OUTPUT_DIM)):
        if len(matrix) != height or any(len(row) != width or
                any(type(v) is not int or not -8 <= v <= 7 for v in row)
                for row in matrix):
            raise ValueError("Invalid INT4 matrix shape or value")
    def activate(matrix):
        return [[_clamp(v >> 4, -8, 7) for v in row] for row in matrix]
    def clip_gradient(matrix):
        return [[_clamp(v, -256, 256) for v in row] for row in matrix]
    def transpose(matrix):
        return [list(row) for row in zip(*matrix)]
    a1 = activate(mm(x_batch, w1))
    y2 = activate(mm(a1, w2))
    e2 = [[_clamp(y2[i][j] - t_batch[i][j], -8, 7)
           for j in range(OUTPUT_DIM)] for i in range(b_size)]
    g2 = clip_gradient(mm(transpose(e2), a1))
    e1 = activate(mm(e2, transpose(w2)))
    g1 = clip_gradient(mm(transpose(e1), x_batch))
    return g1, g2

def fold_gradients_to_digest32(g1: List[List[int]], g2: List[List[int]]) -> bytes:
    """
    步驟 5：將 G1, G2 透過 UINT32 XOR 與模數加法折疊成 32-Byte Digest
    """
    acc_words = [0x9E3779B9 ^ ((i + 1) * 0x85EBCA6B & 0xFFFFFFFF) for i in range(8)]

    all_vals: List[int] = []
    for row in g1:
        all_vals.extend(row)
    for row in g2:
        all_vals.extend(row)

    for idx, val in enumerate(all_vals):
        u = val & 0xFFFFFFFF
        slot = idx & 7
        acc_words[slot] = ((acc_words[slot] + u) ^ _rotl32(u, (idx * 7) % 31 + 1)) & 0xFFFFFFFF

    return struct.pack("<8I", *acc_words)


_CONTRIBUTIONS = None


def rebuild_fixed_cache() -> None:
    """Rebuild after deliberately editing FIXED_X/T or INIT_W1/W2.

    Each sample's forward pass and errors depend only on that sample and
    the fixed weights. A single-sample gradient is at most 64 in magnitude,
    so its +/-256 clamp does not change it. Sum raw contributions across
    the selected batch, THEN clamp, exactly as the reference does.
    """
    global _CONTRIBUTIONS
    if len(FIXED_X) != DATASET_SIZE or len(FIXED_T) != DATASET_SIZE:
        raise ValueError("Fixed dataset length does not match DATASET_SIZE")
    for matrix, width in ((FIXED_X, INPUT_DIM), (FIXED_T, OUTPUT_DIM),
                          (INIT_W1, HIDDEN_DIM), (INIT_W2, OUTPUT_DIM)):
        if any(len(row) != width or any(type(v) is not int or not -8 <= v <= 7 for v in row)
               for row in matrix):
            raise ValueError("Cache requires correctly shaped INT4 matrices")
    if len(INIT_W1) != INPUT_DIM or len(INIT_W2) != HIDDEN_DIM:
        raise ValueError("Weight matrix height does not match dimensions")
    table = []
    for x, t in zip(FIXED_X, FIXED_T):
        g1, g2 = deterministic_int4_train_step([x], [t], INIT_W1, INIT_W2)
        table.append(tuple(v for matrix in (g1, g2) for row in matrix for v in row))
    # Publish only once the full table is built.
    _CONTRIBUTIONS = tuple(table)



def initialize_asc_worker():
    # Initializing only inside spawn children prevents inherited ACL contexts.
    from asc_backend import get_backend
    get_backend()


def compute_ticket(header_template_76: bytes, nonce_u32: int) -> Tuple[bytes, bytes, int]:
    """Exact reference ticket for the fixed dataset/weights, with cached work.

    Cache assumes the documented INT4 range [-8, 7]. Call rebuild_fixed_cache
    after changing dataset/weights; it must not race with ticket computation.
    """
    if len(header_template_76) != 76:
        raise ValueError("Header template must be 76 bytes so (76 + 4) = 80 bytes")
    use_cache = os.environ.get("NEB_ASC_CACHE", "1") != "0"
    if use_cache and _CONTRIBUTIONS is None:
        rebuild_fixed_cache()
    key = blake3_digest(header_template_76 + struct.pack("<I", nonce_u32 & 0xFFFFFFFF))
    if not use_cache:
        x, t = sample_16_from_dataset(key)
        g1, g2 = deterministic_int4_train_step(x, t, INIT_W1, INIT_W2)
    else:
        # Exactly 16 u32 draws consume exactly one ChaCha8 block.
        indices = struct.unpack("<16I", chacha8_block(key, 0))
        rows = [_CONTRIBUTIONS[i % DATASET_SIZE] for i in indices]
        vals = [max(-256, min(256, sum(col))) for col in zip(*rows)]
        split = HIDDEN_DIM * INPUT_DIM
        g1 = [vals[i:i + INPUT_DIM] for i in range(0, split, INPUT_DIM)]
        g2 = [vals[i:i + HIDDEN_DIM] for i in range(split, len(vals), HIDDEN_DIM)]
    digest32 = fold_gradients_to_digest32(g1, g2)
    ticket = blake3_digest(digest32 + key)
    return ticket, digest32, int.from_bytes(ticket, "big")


# ==============================================================================
# 礦池任務結構與非同步 TCP JSON-RPC 連線客戶端
# ==============================================================================

@dataclass
class MiningJob:
    job_id: str
    header_template_76: bytes
    height: int
    block_target: int
    share_target: int
    workload: kep1.Workload
    start_nonce: int = 0
    nonce_step: int = 1


def compute_kep1_ticket(job, nonce, table=None):
    key = blake3_digest(job.header_template_76 + struct.pack("<I", nonce))
    draws = struct.unpack("<16I", chacha8_block(key, 0))
    indices = [draw % len(job.workload.samples) for draw in draws]
    gradient = kep1.batch_gradient(job.workload, indices, table)
    digest = fold_gradients_to_digest32([gradient], [])
    return blake3_digest(digest + key), digest, gradient


def job_contributions(job):
    if os.getenv("NEB_ASC_CACHE", "1") != "0":
        return kep1.contribution_table(job.workload)
    return None


def solution_method(job, ticket):
    value = int.from_bytes(ticket, "big")
    if value <= job.block_target:
        return "mining.submit_block"
    if value <= job.share_target:
        return "mining.submit"
    return None


class AsyncPoolClient:
    """
    通用非同步 JSON-RPC (逐行換行分隔 `\n`) 礦池客戶端。
    若實際礦池使用不同方法名稱（例如 Stratum 的 `mining.notify` / `mining.submit`），
    只需調整 `_handle_server_message` 與 `submit_share` 的 JSON 欄位即可。
    """

    def __init__(self, host: str, port: int, worker: str, password: str = "x", workers: int = 1):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self._executor = None
        self.host = host
        self.port = port
        self.worker = worker
        self.password = password
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._msg_id = 0
        self._current_job: MiningJob | None = None
        self._job_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._submit_lock = asyncio.Lock()

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def send_json(self, payload: dict) -> None:
        if not self.writer or self.writer.is_closing():
            return
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        async with self._submit_lock:
            self.writer.write(line)
            await self.writer.drain()

    async def subscribe_and_authorize(self) -> None:
        await self.send_json({
            "id": self._next_id(),
            "method": "mining.subscribe",
            "params": ["neb-cpython-miner/0.1.0"],
        })
        await self.send_json({
            "id": self._next_id(),
            "method": "mining.authorize",
            "params": [self.worker, self.password],
        })

    async def submit_solution(self, job, method, nonce, ticket, digest):
        print(f"[Solution] {method} job={job.job_id} nonce={nonce:#010x}")
        await self.send_json({
            "id": self._next_id(),
            "method": method,
            "params": [self.worker, job.job_id, f"{nonce:08x}", ticket.hex(), digest.hex()],
        })

    @staticmethod
    def _parse_job(params: dict) -> MiningJob:
        header = bytes.fromhex(params["header_template_76"])
        height = params["height"]
        block_target = int(params["block_target_hex"], 16)
        share_target = int(params["share_target_hex"], 16)
        start = params.get("start_nonce", 0)
        step = params.get("nonce_step", 1)
        if len(header) != 76:
            raise ValueError("Header template must be 76 bytes")
        if type(height) is not int or height < 0:
            raise ValueError("Height must be a nonnegative integer")
        if (type(start) is not int or type(step) is not int
                or not 0 <= start < (1 << 32) or not 0 < step < (1 << 32)):
            raise ValueError("Invalid start_nonce or nonce_step")
        if not 0 <= block_target <= share_target < (1 << 256):
            raise ValueError("Targets must satisfy 0 <= block target <= share target < 2**256")
        return MiningJob(str(params["job_id"]), header, height, block_target, share_target,
                         kep1.Workload.from_job(height, params), start, step)

    async def _handle_server_message(self, msg: dict) -> None:
        method = msg.get("method")
        if method == "mining.notify":
            params = msg.get("params", {})
            if isinstance(params, dict) and "header_template_76" in params:
                self._current_job = self._parse_job(params)
                self._job_event.set()
                print(
                    f"[Pool] 收到新任務 job_id={self._current_job.job_id} "
                    f"height={self._current_job.height} dataset={self._current_job.workload.dataset}"
                )
        elif "id" in msg and ("result" in msg or "error" in msg):
            if msg.get("error"):
                print(f"[Pool] 回應錯誤: {msg['error']}")
            else:
                print(f"[Pool] 回應 id={msg['id']}: result={msg.get('result')}")

    async def _network_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                print(f"[Pool] 正在連線至 {self.host}:{self.port} ...")
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
                print("[Pool] 已連線，送出 subscribe / authorize")
                await self.subscribe_and_authorize()

                while not self._stop_event.is_set():
                    raw = await self.reader.readline()
                    if not raw:
                        raise ConnectionError("礦池主動中斷連線")
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    msg = json.loads(line)
                    await self._handle_server_message(msg)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._current_job = None
                self._job_event.set()
                print(f"[Pool] 連線異常 ({exc})，3 秒後重試...")
                await asyncio.sleep(3.0)
            finally:
                self._current_job = None
                if self.writer and not self.writer.is_closing():
                    self.writer.close()
                    try:
                        await self.writer.wait_closed()
                    except Exception:
                        pass

    async def _mining_loop(self) -> None:
        loop = asyncio.get_running_loop()
        hashes_done = 0
        last_report = time.monotonic()

        while not self._stop_event.is_set():
            if self._current_job is None:
                self._job_event.clear()
                await self._job_event.wait()
                continue

            job = self._current_job
            self._job_event.clear()
            nonce = job.start_nonce
            remaining = (1 << 32) // gcd(job.nonce_step, 1 << 32)
            batch_count = 64

            while not self._stop_event.is_set() and self._current_job is job:
                if remaining == 0:
                    await self._job_event.wait()
                    break
                workers = min(self.workers, (remaining + batch_count - 1) // batch_count)
                results = await asyncio.gather(*(
                    loop.run_in_executor(
                        self._executor, self._scan_nonce_batch, job,
                        (nonce + worker * batch_count * job.nonce_step) & 0xFFFFFFFF,
                        min(batch_count, remaining - worker * batch_count),
                    ) for worker in range(workers)
                ))
                completed = min(remaining, batch_count * workers)
                remaining -= completed
                nonce = (nonce + completed * job.nonce_step) & 0xFFFFFFFF
                hashes_done += completed
                for found in results:
                    for method, f_nonce, ticket, digest in found:
                        if self._current_job is not job:
                            break
                        await self.submit_solution(job, method, f_nonce, ticket, digest)

                now = time.monotonic()
                if now - last_report >= 5.0:
                    rate = hashes_done / (now - last_report)
                    print(f"[Hashrate] ~{rate:.2f} H/s (ASC INT4 cache + BLAKE3 + ChaCha8)")
                    hashes_done = 0
                    last_report = now

                # 讓出事件迴圈控制權以利即時處理新任務
                await asyncio.sleep(0)

    @staticmethod
    def _scan_nonce_batch(job, start_nonce, batch_count):
        found = []
        nonce = start_nonce
        table = job_contributions(job)
        for _ in range(batch_count):
            ticket, digest, _ = compute_kep1_ticket(job, nonce, table)
            method = solution_method(job, ticket)
            if method:
                found.append((method, nonce, ticket, digest))
            nonce = (nonce + job.nonce_step) & 0xFFFFFFFF
        return found

    async def run(self) -> None:
        import multiprocessing
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_asc_worker,
        )
        net_task = asyncio.create_task(self._network_loop())
        mine_task = asyncio.create_task(self._mining_loop())
        try:
            await asyncio.gather(net_task, mine_task)
        finally:
            self._stop_event.set()
            net_task.cancel()
            mine_task.cancel()
            await asyncio.gather(net_task, mine_task, return_exceptions=True)
            self._executor.shutdown(wait=True, cancel_futures=True)


# ==============================================================================
# 內建本機 Mock 礦池伺服器（供立即測試連線、派工與 Ticket 驗證）
# ==============================================================================

async def run_mock_pool(host: str = "127.0.0.1", port: int = 13333) -> None:
    """Validate submissions and advance the chain in the local demo."""
    datasets = {}
    for name in ("Kepler", "TESS"):
        stream = ChaCha8Stream(blake3_digest(f"NEB_MOCK_{name}".encode()))
        datasets[name] = tuple(
            (
                tuple((stream.read_u32() & 15) - 8 for _ in range(kep1.INPUT_LENGTH)),
                tuple((stream.read_u32() & 15) - 8 for _ in range(kep1.OUTPUT_DIM)),
            )
            for _ in range(256)
        )
    models = {
        name: kep1.ModelState(tuple(
            (value & 15) - 8 for value in blake3_digest(b"NEB_MOCK_KEP1", kep1.WEIGHT_COUNT)
        ))
        for name in datasets
    }
    peers = set()
    accepted = {}
    params = {}
    job = None

    def refresh_job(height):
        nonlocal params, job
        dataset = "Kepler" if height % 2 == 0 else "TESS"
        params = {
            "job_id": f"job-{height + 1:03d}",
            "height": height,
            "header_template_76": os.urandom(76).hex(),
            "block_target_hex": f"{(1 << 244) - 1:064x}",
            "share_target_hex": f"{(1 << 250) - 1:064x}",
            "dataset": dataset,
            "model": "KEP1",
            "samples": datasets[dataset],
            "weights": models[dataset].weights,
        }
        job = AsyncPoolClient._parse_job(params)

    def send(writer, payload):
        writer.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())

    def notify(writer):
        send(writer, {"method": "mining.notify", "params": params})

    def validate(method, submission):
        _worker, job_id, nonce_hex, ticket_hex, digest_hex = submission
        nonce = int(nonce_hex, 16)
        if job_id != job.job_id or not 0 <= nonce < (1 << 32) or nonce in accepted:
            return False
        ticket, digest, gradient = compute_kep1_ticket(job, nonce, job_contributions(job))
        if (ticket != bytes.fromhex(ticket_hex) or digest != bytes.fromhex(digest_hex)
                or solution_method(job, ticket) != method):
            return False
        if method == "mining.submit":
            accepted[nonce] = gradient
        else:
            dataset = job.workload.dataset
            models[dataset] = models[dataset].apply(accepted.values())
            accepted.clear()
            refresh_job(job.height + 1)
        return True

    refresh_job(0)

    async def handle_client(reader, writer):
        try:
            while raw := await reader.readline():
                req = json.loads(raw)
                method = req.get("method")
                req_id = req.get("id")
                valid = method in ("mining.subscribe", "mining.authorize")
                if method in ("mining.submit", "mining.submit_block") and writer in peers:
                    try:
                        valid = validate(method, req["params"])
                    except (KeyError, TypeError, ValueError):
                        valid = False
                    print(f"[MockPool] {method} valid={valid}")
                send(writer, {
                    "id": req_id,
                    "result": valid,
                    "error": None if valid else "Invalid request or solution",
                })
                if method == "mining.authorize":
                    peers.add(writer)
                    notify(writer)
                elif method == "mining.submit_block" and valid:
                    for peer in peers:
                        if not peer.is_closing():
                            notify(peer)
                await writer.drain()
        finally:
            peers.discard(writer)
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host, port)
    print(f"[MockPool] Listening on {host}:{port}")
    async with server:
        await server.serve_forever()


async def main(args) -> None:
    # 若要連真實礦池，請將 USE_MOCK_POOL 改為 False，並填入真實 POOL_HOST / POOL_PORT / WORKER
    USE_MOCK_POOL = not args.no_mock
    POOL_HOST = args.host
    POOL_PORT = args.port
    WORKER = args.worker

    if USE_MOCK_POOL:
        asyncio.create_task(run_mock_pool(POOL_HOST, POOL_PORT))
        await asyncio.sleep(0.3)

    client = AsyncPoolClient(host=POOL_HOST, port=POOL_PORT, worker=WORKER, workers=args.workers)
    await client.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ascend INT4 KEP1 miner")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13333)
    parser.add_argument("--worker", default="neb_wallet.worker1")
    parser.add_argument("--no-mock", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
