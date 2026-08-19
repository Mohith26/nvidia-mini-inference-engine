#include "graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

const char *op_name(OpType t) {
  switch (t) {
    case OpType::MatMul: return "MatMul";
    case OpType::BiasAdd: return "BiasAdd";
    case OpType::ReLU: return "ReLU";
    case OpType::Softmax: return "Softmax";
    case OpType::FusedDense: return "FusedDense";
    case OpType::QuantDense: return "QuantDense";
  }
  return "?";
}

namespace {

// out = in @ w   (in: [B, K], w: [K, N]) — i-k-j loop order for cache locality.
void matmul(const Tensor &in, const Tensor &w, Tensor &out) {
  out = Tensor(in.rows, w.cols);
  for (int r = 0; r < in.rows; ++r) {
    const float *ip = in.row(r);
    float *op = out.row(r);
    for (int k = 0; k < in.cols; ++k) {
      const float a = ip[k];
      const float *wp = w.row(k);
      for (int j = 0; j < w.cols; ++j) op[j] += a * wp[j];
    }
  }
}

void bias_add(Tensor &t, const std::vector<float> &bias) {
  for (int r = 0; r < t.rows; ++r) {
    float *p = t.row(r);
    for (int j = 0; j < t.cols; ++j) p[j] += bias[static_cast<size_t>(j)];
  }
}

void relu(Tensor &t) {
  for (float &v : t.data) v = v > 0.0f ? v : 0.0f;
}

void softmax(Tensor &t) {
  for (int r = 0; r < t.rows; ++r) {
    float *p = t.row(r);
    float mx = p[0];
    for (int j = 1; j < t.cols; ++j) mx = std::max(mx, p[j]);
    float sum = 0.0f;
    for (int j = 0; j < t.cols; ++j) {
      p[j] = std::exp(p[j] - mx);
      sum += p[j];
    }
    for (int j = 0; j < t.cols; ++j) p[j] /= sum;
  }
}

// Fused MatMul+BiasAdd(+ReLU): one pass, output initialized with the bias.
void fused_dense(const Tensor &in, const Node &n, Tensor &out) {
  out = Tensor(in.rows, n.weight.cols);
  for (int r = 0; r < in.rows; ++r) {
    const float *ip = in.row(r);
    float *op = out.row(r);
    for (int j = 0; j < out.cols; ++j) op[j] = n.bias[static_cast<size_t>(j)];
    for (int k = 0; k < in.cols; ++k) {
      const float a = ip[k];
      const float *wp = n.weight.row(k);
      for (int j = 0; j < out.cols; ++j) op[j] += a * wp[j];
    }
    if (n.relu) {
      for (int j = 0; j < out.cols; ++j) op[j] = op[j] > 0.0f ? op[j] : 0.0f;
    }
  }
}

// INT8 dense: quantize input row (per-tensor symmetric), int8 x int8 matmul
// accumulated in int32, dequantize with in_scale * w_scale, add fp32 bias.
void quant_dense(const Tensor &in, const Node &n, Tensor &out) {
  const int kIn = in.cols;
  const int kOut = n.weight.cols;
  out = Tensor(in.rows, kOut);
  std::vector<int8_t> qin(static_cast<size_t>(kIn));
  std::vector<int32_t> acc(static_cast<size_t>(kOut));
  const float deq = n.in_scale * n.w_scale;
  for (int r = 0; r < in.rows; ++r) {
    const float *ip = in.row(r);
    for (int k = 0; k < kIn; ++k) qin[static_cast<size_t>(k)] = quantize_value(ip[k], n.in_scale);
    std::fill(acc.begin(), acc.end(), 0);
    for (int k = 0; k < kIn; ++k) {
      const int32_t a = qin[static_cast<size_t>(k)];
      if (a == 0) continue;
      const int8_t *wp = &n.qweight[static_cast<size_t>(k) * kOut];
      for (int j = 0; j < kOut; ++j) acc[static_cast<size_t>(j)] += a * wp[j];
    }
    float *op = out.row(r);
    for (int j = 0; j < kOut; ++j) {
      float v = static_cast<float>(acc[static_cast<size_t>(j)]) * deq + n.bias[static_cast<size_t>(j)];
      op[j] = (n.relu && v < 0.0f) ? 0.0f : v;
    }
  }
}

}  // namespace

Tensor Graph::run(const Tensor &input, std::vector<ProfileEntry> *profile) const {
  using clock = std::chrono::steady_clock;
  if (profile && profile->empty()) {
    for (const Node &n : nodes) profile->push_back({n.name, 0.0});
  }
  Tensor cur = input;
  Tensor next;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const Node &n = nodes[i];
    const auto t0 = clock::now();
    switch (n.type) {
      case OpType::MatMul:
        matmul(cur, n.weight, next);
        cur = std::move(next);
        break;
      case OpType::BiasAdd:
        bias_add(cur, n.bias);
        break;
      case OpType::ReLU:
        relu(cur);
        break;
      case OpType::Softmax:
        softmax(cur);
        break;
      case OpType::FusedDense:
        fused_dense(cur, n, next);
        cur = std::move(next);
        break;
      case OpType::QuantDense:
        quant_dense(cur, n, next);
        cur = std::move(next);
        break;
    }
    if (profile) {
      const std::chrono::duration<double, std::milli> dt = clock::now() - t0;
      (*profile)[i].ms += dt.count();
    }
  }
  return cur;
}

FusionReport fuse(Graph &g) {
  FusionReport rep;
  rep.ops_before = static_cast<int>(g.nodes.size());
  std::vector<Node> out;
  size_t i = 0;
  while (i < g.nodes.size()) {
    if (g.nodes[i].type == OpType::MatMul && i + 1 < g.nodes.size() &&
        g.nodes[i + 1].type == OpType::BiasAdd) {
      Node fused;
      fused.type = OpType::FusedDense;
      fused.weight = std::move(g.nodes[i].weight);
      fused.bias = std::move(g.nodes[i + 1].bias);
      size_t consumed = 2;
      if (i + 2 < g.nodes.size() && g.nodes[i + 2].type == OpType::ReLU) {
        fused.relu = true;
        consumed = 3;
      }
      fused.name = "fused_dense_" + std::to_string(rep.fused_dense);
      ++rep.fused_dense;
      out.push_back(std::move(fused));
      i += consumed;
    } else {
      out.push_back(std::move(g.nodes[i]));
      ++i;
    }
  }
  g.nodes = std::move(out);
  rep.ops_after = static_cast<int>(g.nodes.size());
  return rep;
}
