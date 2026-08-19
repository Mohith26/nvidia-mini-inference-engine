#include <cmath>
#include <stdexcept>

#include "graph.h"

float abs_max(const float *p, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float a = std::fabs(p[i]);
    if (a > m) m = a;
  }
  return m;
}

float quant_scale(float amax) {
  // Symmetric int8: fp32 value = q * scale, q in [-127, 127].
  if (amax <= 0.0f) return 1.0f;  // all-zero tensor: any scale works; avoid /0
  return amax / 127.0f;
}

int8_t quantize_value(float v, float scale) {
  const float q = std::nearbyint(v / scale);
  const float c = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
  return static_cast<int8_t>(c);
}

std::vector<int8_t> quantize_tensor(const float *p, size_t n, float scale) {
  std::vector<int8_t> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = quantize_value(p[i], scale);
  return out;
}

Graph quantize_graph(const Graph &fused, const Tensor &calib) {
  // Calibration: run the fp32 fused graph layer by layer on the calibration
  // batch, recording the abs-max of each FusedDense INPUT activation.
  Graph q;
  Tensor act = calib;
  for (const Node &n : fused.nodes) {
    if (n.type == OpType::FusedDense) {
      Node qn;
      qn.type = OpType::QuantDense;
      qn.name = "quant_dense_" + std::to_string(q.nodes.size());
      qn.weight = n.weight;  // kept for shape metadata
      qn.bias = n.bias;
      qn.relu = n.relu;
      qn.in_scale = quant_scale(abs_max(act.data.data(), act.data.size()));
      qn.w_scale = quant_scale(abs_max(n.weight.data.data(), n.weight.data.size()));
      qn.qweight = quantize_tensor(n.weight.data.data(), n.weight.data.size(), qn.w_scale);
      q.nodes.push_back(std::move(qn));
    } else if (n.type == OpType::MatMul || n.type == OpType::BiasAdd || n.type == OpType::ReLU) {
      throw std::runtime_error("quantize_graph expects a fused graph (run fuse() first)");
    } else {
      q.nodes.push_back(n);  // Softmax passes through unchanged
    }
    // Advance activations in fp32 through the ORIGINAL fused node so the next
    // layer calibrates on unquantized statistics.
    Graph single;
    single.nodes.push_back(n);
    act = single.run(act);
  }
  return q;
}
