#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "tensor.h"

// Graph IR: a linear chain of ops (sufficient for an MLP; mirrors the
// optimize-then-deploy structure of a TensorRT-style engine on CPU).
enum class OpType { MatMul, BiasAdd, ReLU, Softmax, FusedDense, QuantDense };

const char *op_name(OpType t);

struct Node {
  OpType type;
  std::string name;
  // MatMul / FusedDense / QuantDense: weight is [in, out] row-major fp32.
  Tensor weight;
  // BiasAdd / FusedDense / QuantDense.
  std::vector<float> bias;
  // FusedDense / QuantDense: apply ReLU after the affine transform.
  bool relu = false;
  // QuantDense: per-tensor symmetric INT8 weights + scales.
  std::vector<int8_t> qweight;  // [in, out] row-major
  float w_scale = 0.0f;         // weight scale (fp32 value = q * w_scale)
  float in_scale = 0.0f;        // input activation scale from calibration
};

struct ProfileEntry {
  std::string name;
  double ms = 0.0;
};

struct Graph {
  std::vector<Node> nodes;
  // Run the whole graph on a [batch, features] input. If profile != nullptr,
  // per-node wall time (ms) is accumulated into it (single-thread timing).
  Tensor run(const Tensor &input, std::vector<ProfileEntry> *profile = nullptr) const;
};

struct FusionReport {
  int ops_before = 0;
  int ops_after = 0;
  int fused_dense = 0;
};

// Optimizer pass 1: MatMul+BiasAdd+ReLU -> FusedDense(relu) and
// MatMul+BiasAdd -> FusedDense (final layer). In-place; returns op counts.
FusionReport fuse(Graph &g);

// ---- Quantization (quant.cpp) ----
float abs_max(const float *p, size_t n);
float quant_scale(float amax);  // amax / 127, with a floor for all-zero tensors
int8_t quantize_value(float v, float scale);
std::vector<int8_t> quantize_tensor(const float *p, size_t n, float scale);

// Optimizer pass 2: per-tensor symmetric INT8 PTQ of a fused graph.
// Calibrates input-activation scales by running the fused fp32 graph on
// `calib` and recording per-layer abs-max. FusedDense -> QuantDense.
Graph quantize_graph(const Graph &fused, const Tensor &calib);

// ---- Model / data I/O (model_io.cpp) ----
bool file_exists(const std::string &path);
// Loads model.bin into the naive op-by-op graph:
// per hidden layer MatMul, BiasAdd, ReLU; final MatMul, BiasAdd, Softmax.
Graph load_model(const std::string &path);
Tensor load_matrix(const std::string &path);
std::vector<uint8_t> load_labels(const std::string &path);

// Row-wise argmax of logits/probabilities.
std::vector<uint8_t> argmax_rows(const Tensor &t);
