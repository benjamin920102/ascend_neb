"""Integer Conv1D, global max pooling and linear training for KEP1 jobs."""

from dataclasses import dataclass
from functools import lru_cache

from asc_backend import get_backend


BRANCH_WIDTHS = (3, 5)
INPUT_LENGTH = 16
OUTPUT_DIM = 4
WEIGHT_COUNT = sum(BRANCH_WIDTHS) + len(BRANCH_WIDTHS) * OUTPUT_DIM


def int4(value):
    return max(-8, min(7, value))


def gradient_clip(value):
    return max(-256, min(256, value))


def int4_row(values, length):
    if not isinstance(values, (list, tuple)) or len(values) != length or any(
        type(value) is not int or not -8 <= value <= 7 for value in values
    ):
        raise ValueError(f"Expected a row of {length} signed INT4 integers")
    return tuple(values)


@dataclass(frozen=True)
class Workload:
    dataset: str
    samples: tuple
    weights: tuple

    @classmethod
    def from_job(cls, height, params):
        dataset = "Kepler" if height % 2 == 0 else "TESS"
        if params["dataset"] != dataset or params["model"] != "KEP1":
            raise ValueError("Dataset or model does not match the block height")
        samples = tuple(
            (int4_row(signal, INPUT_LENGTH), int4_row(target, OUTPUT_DIM))
            for signal, target in params["samples"]
        )
        if not samples:
            raise ValueError("KEP1 dataset must contain at least one sample")
        return cls(dataset, samples, int4_row(params["weights"], WEIGHT_COUNT))


def sample_gradient(signal, target, weights, mm):
    pooled = []
    winning_windows = []
    offset = 0
    for width in BRANCH_WIDTHS:
        windows = [signal[i:i + width] for i in range(len(signal) - width + 1)]
        kernel = [[value] for value in weights[offset:offset + width]]
        activated = [int4(row[0] >> 4) for row in mm(windows, kernel)]
        winner = max(range(len(windows)), key=activated.__getitem__)
        pooled.append(activated[winner])
        winning_windows.append(windows[winner])
        offset += width

    linear = [
        weights[i:i + OUTPUT_DIM] for i in range(offset, len(weights), OUTPUT_DIM)
    ]
    prediction = [int4(value >> 4) for value in mm([pooled], linear)[0]]
    error = [int4(value - expected) for value, expected in zip(prediction, target)]
    branch_error = mm([error], list(zip(*linear)))[0]
    kernel_gradient = [
        value * int4(delta >> 4)
        for window, delta in zip(winning_windows, branch_error)
        for value in window
    ]
    linear_gradient = [activation * delta for activation in pooled for delta in error]
    return tuple(kernel_gradient + linear_gradient)


@lru_cache(maxsize=2)
def contribution_table(workload):
    mm = get_backend().matmul
    return tuple(sample_gradient(x, y, workload.weights, mm) for x, y in workload.samples)


def batch_gradient(workload, indices, table):
    if table is not None:
        rows = [table[i] for i in indices]
    else:
        mm = get_backend().matmul
        rows = [
            sample_gradient(*workload.samples[i], workload.weights, mm)
            for i in indices
        ]
    # With INT4 operands, each sum of 16 products fits in signed INT32.
    return tuple(gradient_clip(sum(column)) for column in zip(*rows))


@dataclass(frozen=True)
class ModelState:
    weights: tuple
    residual: tuple = (0,) * WEIGHT_COUNT

    def apply(self, gradients):
        weights = []
        residual = list(self.residual)
        for gradient in gradients:
            for index, value in enumerate(gradient):
                residual[index] += value
        for weight, value in zip(self.weights, residual):
            # Gradient descent with one INT4 quantization step per block.
            step = int(value >= 256) - int(value <= -256)
            updated = int4(weight - step)
            weights.append(updated)
        return ModelState(
            tuple(weights),
            tuple(value - (old - new) * 256
                  for value, old, new in zip(residual, self.weights, weights)),
        )
