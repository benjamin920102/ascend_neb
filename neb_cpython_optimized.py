import asyncio
import json
import os
import struct
import time
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


def deterministic_int4_train_step(
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


def compute_ticket(header_template_76: bytes, nonce_u32: int) -> Tuple[bytes, bytes, int]:
    """Exact reference ticket for the fixed dataset/weights, with cached work.

    Cache assumes the documented INT4 range [-8, 7]. Call rebuild_fixed_cache
    after changing dataset/weights; it must not race with ticket computation.
    """
    if len(header_template_76) != 76:
        raise ValueError("Header template must be 76 bytes so (76 + 4) = 80 bytes")
    if _CONTRIBUTIONS is None:
        rebuild_fixed_cache()
    key = blake3_digest(header_template_76 + struct.pack("<I", nonce_u32 & 0xFFFFFFFF))
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
    target_int: int
    start_nonce: int = 0
    nonce_step: int = 1


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

    async def submit_share(self, job: MiningJob, nonce_u32: int, ticket_hex: str, digest_hex: str) -> None:
        msg = {
            "id": self._next_id(),
            "method": "mining.submit",
            "params": [
                self.worker,
                job.job_id,
                f"{nonce_u32:08x}",
                ticket_hex,
                digest_hex,
            ],
        }
        print(f"[Share] 提交候選解 -> job={job.job_id}, nonce={nonce_u32:#010x}, ticket={ticket_hex[:16]}...")
        await self.send_json(msg)

    def _parse_job(self, params: dict) -> MiningJob:
        header_hex = params["header_template_76"]
        header_bytes = bytes.fromhex(header_hex)
        if len(header_bytes) != 76:
            raise ValueError(f"Invalid header_template_76 length: {len(header_bytes)}")
        target_int = int(params["target_hex"], 16)
        step = int(params.get("nonce_step", 1))
        if not 0 < step < (1 << 32):
            raise ValueError("nonce_step must be in [1, 2**32-1]")
        if not 0 <= target_int <= (1 << 256):
            raise ValueError("target must be in [0, 2**256]")
        return MiningJob(
            job_id=str(params["job_id"]),
            header_template_76=header_bytes,
            target_int=target_int,
            start_nonce=int(params.get("start_nonce", 0)),
            nonce_step=int(params.get("nonce_step", 1)),
        )

    async def _handle_server_message(self, msg: dict) -> None:
        method = msg.get("method")
        if method == "mining.notify":
            params = msg.get("params", {})
            if isinstance(params, dict) and "header_template_76" in params:
                self._current_job = self._parse_job(params)
                self._job_event.set()
                print(
                    f"[Pool] 收到新任務 job_id={self._current_job.job_id} "
                    f"target={self._current_job.target_int:#066x}"
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
            nonce = job.start_nonce & 0xFFFFFFFF

            while not self._stop_event.is_set() and self._current_job is job:
                # 將單批 Nonce 運算丟到 ThreadPoolExecutor，避免阻塞網路讀寫協程
                batch_count = 64
                results = await asyncio.gather(*(
                    loop.run_in_executor(
                        self._executor, self._scan_nonce_batch, job,
                        (nonce + worker * batch_count * job.nonce_step) & 0xFFFFFFFF,
                        batch_count,
                    ) for worker in range(self.workers)
                ))
                completed = batch_count * self.workers
                nonce = (nonce + completed * job.nonce_step) & 0xFFFFFFFF
                hashes_done += completed
                if self._current_job is not job:
                    continue
                for found in results:
                    for f_nonce, ticket_bytes, digest_bytes in found:
                        if self._current_job is not job:
                            break
                        await self.submit_share(job, f_nonce, ticket_bytes.hex(), digest_bytes.hex())

                now = time.monotonic()
                if now - last_report >= 5.0:
                    rate = hashes_done / (now - last_report)
                    print(f"[Hashrate] ~{rate:.2f} H/s (CPython 整數模型+BLAKE3+ChaCha8)")
                    hashes_done = 0
                    last_report = now

                # 讓出事件迴圈控制權以利即時處理新任務
                await asyncio.sleep(0)

    @staticmethod
    def _scan_nonce_batch(
        job: MiningJob, start_nonce: int, batch_count: int
    ) -> list[Tuple[int, bytes, bytes]]:
        found = []
        nonce = start_nonce & 0xFFFFFFFF
        step = job.nonce_step & 0xFFFFFFFF
        for _ in range(batch_count):
            ticket_bytes, digest_bytes, ticket_int = compute_ticket(job.header_template_76, nonce)
            if ticket_int < job.target_int:
                found.append((nonce, ticket_bytes, digest_bytes))
            nonce = (nonce + step) & 0xFFFFFFFF
        return found

    async def run(self) -> None:
        import multiprocessing
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=rebuild_fixed_cache,
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
    """
    測試用本機礦池：
    - 故意設定較寬鬆的 target（前 12 bits 為 0 左右），讓 CPython 在幾秒內就能算出合法 Share
    - 收到 mining.submit 後，實際重算一次 compute_ticket 驗證是否真的 < target
    """
    # 256-bit 上限約 2^256；設定 target = 2^244 約 1/4096 機率命中
    demo_target = (1 << 244) - 1
    header_76 = os.urandom(76)

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        print(f"[MockPool] 礦工已連線: {peer}")
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                req = json.loads(raw.decode("utf-8"))
                method = req.get("method")
                req_id = req.get("id")

                if method == "mining.subscribe":
                    writer.write((json.dumps({"id": req_id, "result": True, "error": None}) + "\n").encode())
                    await writer.drain()
                elif method == "mining.authorize":
                    writer.write((json.dumps({"id": req_id, "result": True, "error": None}) + "\n").encode())
                    await writer.drain()
                    notify = {
                        "method": "mining.notify",
                        "params": {
                            "job_id": "job-001",
                            "header_template_76": header_76.hex(),
                            "target_hex": f"{demo_target:064x}",
                            "start_nonce": 0,
                            "nonce_step": 1,
                        },
                    }
                    writer.write((json.dumps(notify) + "\n").encode())
                    await writer.drain()
                elif method == "mining.submit":
                    _worker, job_id, nonce_hex, ticket_hex, digest_hex = req["params"]
                    nonce_val = int(nonce_hex, 16)
                    calc_ticket, calc_digest, calc_int = compute_ticket(header_76, nonce_val)
                    valid = (
                        job_id == "job-001"
                        and calc_ticket.hex() == ticket_hex.lower()
                        and calc_digest.hex() == digest_hex.lower()
                        and calc_int < demo_target
                    )
                    resp = {
                        "id": req_id,
                        "result": valid,
                        "error": None if valid else "invalid share or ticket >= target",
                    }
                    print(f"[MockPool] 驗證 Share nonce={nonce_hex} -> valid={valid}")
                    writer.write((json.dumps(resp) + "\n").encode())
                    await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, host, port)
    print(f"[MockPool] 測試礦池啟動於 {host}:{port}")
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
    parser = argparse.ArgumentParser(description="Fixed-workload CPython ticket calculator")
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