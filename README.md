# TensorForge: a CPU mini inference engine with graph fusion and INT8 quantization

TensorForge is a small inference engine I wrote in dependency-free C++17. It
loads a real trained MNIST classifier, runs it through a graph optimizer (op
fusion) and INT8 post-training quantization, and checks every step against a
Python (NumPy) reference oracle. I wanted to see how much of a TensorRT-style
pipeline (graph IR, fusion passes, PTQ calibration) I could build from scratch
and actually measure, instead of taking the usual claims on faith.

The short version of what came out of it: INT8 gives a real 3.4x speedup on
this model with no accuracy loss, and op fusion, despite cutting the graph
from 9 ops to 4, gives essentially zero wall-clock gain because the model is
completely matmul-bound. Full numbers are in [RESULTS.md](RESULTS.md) and
`results/*.json`, all from actual runs.

## What the engine does

- **Reference executor**: naive op-by-op float32 execution (separate MatMul,
  BiasAdd, ReLU, Softmax passes). This is the correctness baseline.
- **Fusion pass**: rewrites `MatMul+BiasAdd+ReLU` into a single
  `FusedDense(relu)` kernel, and `MatMul+BiasAdd` into `FusedDense` for the
  final layer. 9 ops become 4.
- **INT8 post-training quantization**: per-tensor symmetric quantization of
  weights and activations. Activation scales are calibrated from abs-max over
  1,000 calibration images; the int8 x int8 matmul accumulates in int32 and
  dequantizes between layers.

## Architecture

```
              train/ (Python 3.9)                          src/ (C++17, no deps)
 ┌───────────────────────────────────┐        ┌─────────────────────────────────────────┐
 │ MNIST download (checksummed)      │        │            Graph IR (linear chain)      │
 │  -> sklearn MLP 784-256-128-10    │ model  │  MatMul -> BiasAdd -> ReLU -> ... ->    │
 │  -> float32 weights export        │ .bin   │  MatMul -> BiasAdd -> Softmax  (9 ops)  │
 │  -> NumPy fp32 oracle:            │ ─────> │                 │                       │
 │     logits/preds on 10k test set  │        │   pass 1: FUSION│(MatMul+BiasAdd[+ReLU] │
 │  -> golden vectors, 1k calib set  │        │                 ▼   -> FusedDense)      │
 └───────────────────────────────────┘        │  FusedDense x3 -> Softmax     (4 ops)   │
                                              │                 │                       │
        eval/  compare vs oracle              │   pass 2: INT8  │ PTQ (per-tensor       │
        bench/ img/s, ms/img, profiler        │                 ▼  symmetric, 1k calib) │
        tests/ 25 C++ + 12 pytest             │  QuantDense x3 (int8 matmul, int32      │
                                              │  accumulate, dequant) -> Softmax        │
                                              └─────────────────────────────────────────┘
```

## Building and running

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn pytest
make train      # downloads MNIST (checksummed), trains, exports data/ (~1 min)
make all        # builds engine, eval, bench, tests (clang++ -O2 -Werror)
make test       # 25 C++ tests + 12 pytest tests
make eval-run   # correctness vs oracle -> results/eval.json
make bench      # best-of-5 benchmarks + per-layer profile -> results/*.json
```

If every MNIST mirror is unreachable, `make train` falls back to a seeded
synthetic dataset (`train/synthetic.py`) and records `"source": "SYNTHETIC"`
in `data/meta.json` so it can't be confused with real data. The committed
results were produced from real MNIST with verified checksums.

## Results at a glance

- The fp32 engine matches the Python oracle exactly: 100% top-1 agreement on
  all 10,000 test images, bit-exact for the naive executor.
- INT8 quantization lost no accuracy on this model (98.10% vs 98.09% fp32,
  a net +1 correct prediction out of 10,000).
- INT8 runs 3.4x faster than fused fp32 at both batch 1 and batch 64.
- Fusion alone bought roughly nothing (about 1.005x). The per-layer profiler
  explains why: BiasAdd, ReLU and Softmax together are ~0.4% of the runtime,
  so eliminating those passes can't matter on a matmul-bound model.

See [RESULTS.md](RESULTS.md) for the full tables.

## Limitations

- CPU only, single thread, MLP only. No Conv/CNN, no GPU or CUDA, no ONNX,
  no serving layer.
- All performance numbers are from my machine, an Apple-silicon laptop
  (arm64, `clang++ -O2`, single thread). They will differ on other hardware;
  the correctness and accuracy results should not.
- The zero accuracy loss under INT8 is a measured result for this specific
  small MLP. Deeper networks typically lose more under per-tensor PTQ.
- The fused kernel initializes its accumulator with the bias, which changes
  float summation order, so fused logits differ from naive by up to 1.53e-05.
  That changes zero predictions but it is not bit-exact.

## Layout

```
src/    tensor.h, graph.{h,cpp} (IR + executors + fusion), quant.cpp, model_io.cpp
train/  mnist.py (download/parse/checksum), model.py (oracle + binary formats),
        train_mlp.py (train + export), synthetic.py (documented fallback)
eval/   eval_main.cpp: engine vs oracle on the 10k test set
bench/  bench_main.cpp: naive/fused/int8 x batch {1,64}, per-layer profiler
tests/  test_main.cpp (C++ harness), test_python.py (pytest)
results/ eval.json, bench.json, profile.json, tests.json (committed)
data/   gitignored; regenerated by `make train`
```
