# Résumé bullets (all values measured; see RESULTS.md for methodology)

Honesty tags: all perf numbers are CPU **single-thread on an Apple-silicon Mac**
(`clang++ -O2`); dataset is **real MNIST** (checksummed download, not synthetic);
accuracy/match numbers are over the full 10,000-image test set.

1.
Built a TensorRT-style C++17 CPU inference engine (graph IR + fusion pass, 9 ops -> 4) for a real-MNIST 784-256-128-10 MLP; FP32
output matched the NumPy oracle on 100% of 10k test predictions (naive path bit-exact; fused max logit diff 1.5e-05).

2.
Quantized to INT8 (per-tensor symmetric PTQ, 1,000-image calibration) at 98.10% top-1 vs 98.09% FP32 (+0.01 pp) with a 3.4x
measured speedup: 0.0140 -> 0.0041 ms/image at batch 1, 253k img/s at batch 64 (single-thread Apple-silicon CPU, clang -O2).

3.
Verified by 37 passing tests (25 C++ assert-harness + 12 pytest): hand-computed op checks, fusion equivalence <1e-6, quant
round-trip, calibration, oracle golden vectors; per-layer profiler showed 85% of FP32 time in the first matmul (make-reproducible).

Note (honesty): the fusion pass reduced op count 9 -> 4 but produced no
meaningful wall-clock speedup (~1.005x at batch 1) because this model is
matmul-bound (BiasAdd+ReLU+Softmax = ~0.4% of runtime); the bullets therefore
claim the op-count reduction and correctness, not a fusion speedup. The 3.4x
speedup claimed in bullet 2 is INT8 vs fused FP32.
