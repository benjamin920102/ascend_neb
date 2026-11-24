"""Run on 910C: exact integer comparisons, never floating point tolerance."""
import argparse
import os
import random
import struct

import neb_asc as model
from asc_backend import get_backend
import kep1


def reference_ticket(header, nonce):
    key = model.blake3_digest(header + struct.pack('<I', nonce & 0xffffffff))
    x, t = model.sample_16_from_dataset(key)
    gradients = model.reference_int4_train_step(x, t, model.INIT_W1, model.INIT_W2)
    digest = model.fold_gradients_to_digest32(*gradients)
    ticket = model.blake3_digest(digest + key)
    return ticket, digest, int.from_bytes(ticket, 'big')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, default=64)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('--cases must be positive')
    rng = random.Random(910)
    mm = get_backend().matmul
    shapes = [(1, 1, 1), (16, 4, 16), (4, 16, 16), (16, 16, 4),
              (16, 16, 16), (16, 64, 64), (7, 13, 9)]
    gemms = 0
    for m, k, n in shapes:
        for fill in (-8, 7, 0, None):
            a = [[rng.randrange(-8, 8) if fill is None else fill for _ in range(k)] for _ in range(m)]
            b = [[rng.randrange(-8, 8) if fill is None else fill for _ in range(n)] for _ in range(k)]
            ref = [[sum(a[i][z] * b[z][j] for z in range(k)) for j in range(n)] for i in range(m)]
            actual = mm(a, b)
            if actual != ref:
                raise AssertionError(f'INT4 GEMM mismatch: shape={(m,k,n)}, fill={fill}')
            gemms += 1
    print(f'GEMM exact: PASS ({gemms} cases)', flush=True)
    for case in range(args.cases):
        b = (1, 16)[case % 2]
        fill = (-8, 7, 0, None)[case % 4]
        def matrix(rows, cols):
            return [[rng.randrange(-8, 8) if fill is None else fill for _ in range(cols)] for _ in range(rows)]
        x, t, w1, w2 = matrix(b, 16), matrix(b, 4), matrix(16, 16), matrix(16, 4)
        actual = model.deterministic_int4_train_step(x, t, w1, w2)
        ref = model.reference_int4_train_step(x, t, w1, w2)
        if actual != ref:
            raise AssertionError(f'G1/G2 mismatch: case={case}')
    print(f'G1/G2 exact: PASS ({args.cases} cases)', flush=True)
    # Validate all 256 fixed contribution rows before checking cached tickets.
    model.rebuild_fixed_cache()
    expected = []
    for x, t in zip(model.FIXED_X, model.FIXED_T):
        gs = model.reference_int4_train_step([x], [t], model.INIT_W1, model.INIT_W2)
        expected.append(tuple(v for g in gs for row in g for v in row))
    if model._CONTRIBUTIONS != tuple(expected):
        raise AssertionError('Fixed contribution cache mismatch')
    print('All 256 contribution rows exact: PASS', flush=True)
    for case in range(args.cases):
        header = bytes(rng.randrange(256) for _ in range(76))
        nonce = [0, 1, 0x7fffffff, 0xffffffff][case % 4] if case < 4 else rng.randrange(2**32)
        ref = reference_ticket(header, nonce)
        for cache in ('0', '1'):
            os.environ['NEB_ASC_CACHE'] = cache
            if model.compute_ticket(header, nonce) != ref:
                raise AssertionError(f'Ticket/digest mismatch: case={case}, cache={cache}')
    print(f'Digest/ticket exact in direct and cache modes: PASS ({args.cases} cases)', flush=True)
    verify_kep1(mm)
    print('All checked cases passed on the selected backend.')


def verify_kep1(mm):
    signal = (4,) * kep1.INPUT_LENGTH
    target = (-6,) * kep1.OUTPUT_DIM
    weights = (4,) * kep1.WEIGHT_COUNT
    expected = (28,) * 8 + (21,) * 4 + (35,) * 4
    if kep1.sample_gradient(signal, target, weights, mm) != expected:
        raise AssertionError("KEP1 forward/backward gradient mismatch")
    workload = kep1.Workload("Kepler", ((signal, target),), weights)
    job = model.MiningJob("test", bytes(76), 0, 0, (1 << 256) - 1, workload)
    table = kep1.contribution_table(workload)
    direct = model.compute_kep1_ticket(job, 0)
    if direct != model.compute_kep1_ticket(job, 0, table) or direct[2] != (256,) * 16:
        raise AssertionError("KEP1 cache or batch clipping mismatch")
    if model.solution_method(job, bytes(32)) != "mining.submit_block":
        raise AssertionError("KEP1 block target equality mismatch")
    if model.solution_method(job, bytes([255]) * 32) != "mining.submit":
        raise AssertionError("KEP1 share target equality mismatch")
    updated = kep1.ModelState(weights).apply([direct[2], direct[2]])
    if updated.weights != (3,) * 16 or updated.residual != (256,) * 16:
        raise AssertionError("KEP1 residual update mismatch")
    print("KEP1 gradient, ticket, targets and model update: PASS", flush=True)


if __name__ == '__main__':
    main()
