# Benchmark and correctness notes

Everything below comes from actual runs on my machine (Apple Silicon Mac,
arm64, macOS, `clang++ -std=c++17 -O2`, single thread) using the exact
commands in [Reproduce](#reproduce). The dataset is real MNIST, downloaded
from `storage.googleapis.com/cvdf-datasets/mnist/` with SHA-256 verification
(`data/meta.json` records `"source": "real-mnist"`). Perf numbers are
machine-specific and will vary elsewhere; the accuracy and correctness
numbers should not.

## Model and training

| Item | Value |
|---|---|
| Model | MLP 784-256-128-10, ReLU, trained with sklearn `MLPClassifier` (adam, seed 42, 30 iters) |
| Train set | 60,000 real MNIST images; test set 10,000 |
| Python fp32 oracle test accuracy | **98.09%** (9,809/10,000) |
| sklearn's own `score()` | 98.09% (identical) |

## Fusion pass: op counts

| | ops |
|---|---|
| Before fusion (MatMul, BiasAdd, ReLU x2, final MatMul+BiasAdd, Softmax) | **9** |
| After fusion (3x FusedDense + Softmax) | **4** |

## FP32 correctness vs the Python oracle (10,000-image test set)

| Engine | Top-1 prediction match vs oracle | Max abs logit diff |
|---|---|---|
| Naive (op-by-op) | **100.000%** (10,000/10,000) | **0.0** (bit-exact) |
| Fused | **100.000%** (10,000/10,000) | **1.53e-05** |

The fused kernel initializes the accumulator with the bias, which changes
float summation order, so fused logits differ from naive by up to 1.53e-05.
That is well under the 1e-4 target I set and changes zero predictions.

## Accuracy: FP32 vs INT8 (10,000-image test set)

| Engine | Top-1 accuracy | Delta vs FP32 |
|---|---|---|
| FP32 (naive and fused, identical preds) | **98.09%** | baseline |
| INT8 (per-tensor symmetric PTQ, **1,000**-image calibration) | **98.10%** | **+0.01 pp** |

INT8 lost no accuracy here: 9,810 correct vs 9,809, a net +1 correct
prediction. That's a real measured result, but it's specific to this small
MLP and dataset; deeper nets typically lose more under PTQ.

## Throughput and latency (best of 5 full passes over the 10k test set)

Single thread, full graph including softmax; the partial final batch is
dropped at batch 64 (9,984 images).

| Engine | Batch | images/sec | ms/image |
|---|---|---|---|
| naive | 1 | **70,959** | **0.0141** |
| naive | 64 | **73,325** | **0.0136** |
| fused | 1 | **71,285** | **0.0140** |
| fused | 64 | **73,364** | **0.0136** |
| fused+INT8 | 1 | **244,164** | **0.0041** |
| fused+INT8 | 64 | **253,370** | **0.0039** |

INT8 speedup vs fused FP32: **3.43x at batch 1, 3.45x at batch 64**.

Fusion speedup vs naive: about 1.005x at batch 1 and 1.001x at batch 64, in
other words no meaningful wall-clock gain. This surprised me at first, but
the profiler makes it obvious: the model is matmul-bound. BiasAdd, ReLU and
Softmax total only ~0.5 ms of a ~136.5 ms pass (~0.4%), so eliminating those
separate passes can't buy much. The fusion pass still delivers the 9 -> 4
op-count reduction and is verified numerically equivalent (<1e-6 on random
tensors, 1.53e-05 max on real logits).

## Per-layer profile (one full batch-64 pass, 9,984 images; ms accumulated)

| FP32 fused | ms | | INT8 | ms |
|---|---|---|---|---|
| fused_dense_0 (784->256) | 116.1 (**85%**) | | quant_dense_0 | 24.6 |
| fused_dense_1 (256->128) | 17.4 | | quant_dense_1 | 11.0 |
| fused_dense_2 (128->10) | 1.9 | | quant_dense_2 | 4.7 |
| softmax | 0.15 | | softmax | 0.16 |

One wrinkle worth calling out: INT8 is 4.7x faster on the big 784->256 layer
but actually slower than FP32 on the tiny 128->10 layer (4.7 ms vs 1.9 ms).
Per-row input quantization overhead dominates when the output dimension is
small.

## Tests

| Suite | Command | Result |
|---|---|---|
| C++ (custom assert harness) | `./build/run_tests data` (via `make test`) | **25 passed, 0 failed, 0 skipped** |
| Python (pytest) | `.venv/bin/pytest tests -q --color=no` | **12 passed** |
| **Total** | | **37 passed** |

What they cover: per-op hand-computed values, softmax stability, fusion op
counts / structure / ReLU flags, fusion equivalence on random tensors
(<1e-6), quantize/dequantize round-trip bound, scale/rounding/clamping,
calibration scale, int32-accumulate hand check, INT8-vs-FP32 closeness,
end-to-end golden vectors exported from the Python oracle, IDX parsing,
export-format round-trips, oracle forward-pass hand checks, and
synthetic-fallback determinism.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn pytest
make train       # download real MNIST (checksummed) + train + export data/
make all         # clang++ -std=c++17 -O2 -Wall -Wextra -Werror
make test        # C++ harness + pytest        -> 25 + 12 passed
make eval-run    # ./build/eval data results   -> results/eval.json
make bench       # ./build/bench data results  -> results/bench.json, results/profile.json
```
