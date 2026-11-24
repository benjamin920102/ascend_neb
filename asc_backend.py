"""Persistent ctypes bridge to native packed signed-INT4 Cube GEMM."""
import atexit
import ctypes as C
import os
from pathlib import Path
import threading


def pack_matrix(matrix, padded_rows, padded_cols):
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if not rows or not cols or rows > padded_rows or cols > padded_cols:
        raise ValueError('Matrix exceeds padded dimensions or is empty')
    packed = bytearray(padded_rows * padded_cols // 2)
    for i, row in enumerate(matrix):
        if len(row) != cols:
            raise ValueError('Ragged matrix')
        for j, value in enumerate(row):
            if type(value) is not int or not -8 <= value <= 7:
                raise ValueError('Signed INT4 operands must be integers in [-8, 7]')
            offset = i * padded_cols + j
            packed[offset // 2] |= (value & 15) << ((offset & 1) * 4)
    return packed


class AscInt4:
    def __init__(self, device=None):
        self.pid = os.getpid()
        self.handle = None
        path = Path(os.environ.get('NEB_ASC_LIB', Path(__file__).parent / 'build' / 'libneb_int4.so'))
        if not path.is_file():
            raise RuntimeError(f'Build the ASC library first with ./build.sh: {path}')
        self.lib = C.CDLL(str(path.resolve()))
        self.lib.neb_create.argtypes = [C.c_int]
        self.lib.neb_create.restype = C.c_void_p
        self.lib.neb_last_error.argtypes = []
        self.lib.neb_last_error.restype = C.c_char_p
        self.lib.neb_matmul.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        self.lib.neb_matmul.restype = C.c_int
        self.lib.neb_destroy.argtypes = [C.c_void_p]
        self.lib.neb_destroy.restype = None
        self.handle = self.lib.neb_create(int(os.environ.get('NEB_ASC_DEVICE', '0')) if device is None else device)
        if not self.handle:
            raise RuntimeError(self.lib.neb_last_error().decode())
        atexit.register(self.close)

    def close(self):
        if self.handle and self.pid == os.getpid():
            self.lib.neb_destroy(self.handle)
            self.handle = None

    def matmul(self, a, b):
        if self.pid != os.getpid() or not self.handle:
            raise RuntimeError('ASC context is closed or inherited across fork; use spawn')
        m, k, n = len(a), len(b), len(b[0]) if b else 0
        if not 1 <= m <= 16 or not 1 <= k <= 64 or not 1 <= n <= 64:
            raise ValueError('Supported logical dimensions: M=1..16, K,N=1..64')
        if any(len(row) != k for row in a):
            raise ValueError('A columns must equal B rows')
        av = pack_matrix(a, 16, 64)
        bv = pack_matrix(b, 64, 64)
        abuf = (C.c_uint8 * len(av)).from_buffer(av)
        bbuf = (C.c_uint8 * len(bv)).from_buffer(bv)
        output = (C.c_int32 * (16 * 64))()
        if self.lib.neb_matmul(self.handle, abuf, bbuf, output):
            raise RuntimeError(self.lib.neb_last_error().decode())
        return [[output[i * 64 + j] for j in range(n)] for i in range(m)]


_backend = None
_backend_lock = threading.Lock()


def get_backend():
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = AscInt4()
        return _backend
