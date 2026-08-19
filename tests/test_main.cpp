// Custom assert-based C++ test harness (no external deps): op unit tests,
// fusion equivalence, quantization round-trips, calibration, and end-to-end
// golden vectors exported from the Python oracle.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "graph.h"

namespace {

int g_pass = 0, g_fail = 0, g_skip = 0;

void check(const std::string &name, bool ok) {
  if (ok) {
    ++g_pass;
    std::printf("PASS  %s\n", name.c_str());
  } else {
    ++g_fail;
    std::printf("FAIL  %s\n", name.c_str());
  }
}

void skip(const std::string &name, const std::string &why) {
  ++g_skip;
  std::printf("SKIP  %s (%s)\n", name.c_str(), why.c_str());
}

bool close(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

float max_abs_diff(const Tensor &a, const Tensor &b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.data.size(); ++i) m = std::max(m, std::fabs(a.data[i] - b.data[i]));
  return m;
}

Node make_matmul(const Tensor &w) {
  Node n;
  n.type = OpType::MatMul;
  n.name = "mm";
  n.weight = w;
  return n;
}

Node make_bias(const std::vector<float> &b) {
  Node n;
  n.type = OpType::BiasAdd;
  n.name = "ba";
  n.bias = b;
  return n;
}

Node make_op(OpType t, const char *name) {
  Node n;
  n.type = t;
  n.name = name;
  return n;
}

Tensor random_tensor(int r, int c, std::mt19937 &rng) {
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  Tensor t(r, c);
  for (float &v : t.data) v = d(rng);
  return t;
}

// Random unfused (MatMul,BiasAdd,ReLU)*2 + (MatMul,BiasAdd,Softmax) graph.
Graph random_mlp(std::mt19937 &rng, const std::vector<int> &dims) {
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  Graph g;
  for (size_t l = 0; l + 1 < dims.size(); ++l) {
    Tensor w = random_tensor(dims[l], dims[l + 1], rng);
    g.nodes.push_back(make_matmul(w));
    std::vector<float> b(static_cast<size_t>(dims[l + 1]));
    for (float &v : b) v = d(rng);
    g.nodes.push_back(make_bias(b));
    if (l + 2 < dims.size()) g.nodes.push_back(make_op(OpType::ReLU, "relu"));
  }
  g.nodes.push_back(make_op(OpType::Softmax, "softmax"));
  return g;
}

// ---------- op unit tests ----------

void test_matmul_hand() {
  Tensor x(1, 2);
  x.at(0, 0) = 1.0f;
  x.at(0, 1) = 2.0f;
  Tensor w(2, 2);
  w.at(0, 0) = 3.0f; w.at(0, 1) = 4.0f;
  w.at(1, 0) = 5.0f; w.at(1, 1) = 6.0f;
  Graph g;
  g.nodes.push_back(make_matmul(w));
  Tensor y = g.run(x);
  check("matmul_hand_computed", y.rows == 1 && y.cols == 2 && close(y.at(0, 0), 13.0f, 1e-6f) &&
                                    close(y.at(0, 1), 16.0f, 1e-6f));
}

void test_matmul_identity() {
  std::mt19937 rng(1);
  Tensor x = random_tensor(3, 3, rng);
  Tensor eye(3, 3);
  for (int i = 0; i < 3; ++i) eye.at(i, i) = 1.0f;
  Graph g;
  g.nodes.push_back(make_matmul(eye));
  Tensor y = g.run(x);
  check("matmul_identity", max_abs_diff(x, y) == 0.0f);
}

void test_matmul_batch_shape() {
  std::mt19937 rng(2);
  Tensor x = random_tensor(4, 3, rng);
  Tensor w = random_tensor(3, 2, rng);
  Graph g;
  g.nodes.push_back(make_matmul(w));
  Tensor y = g.run(x);
  float want = 0.0f;
  for (int k = 0; k < 3; ++k) want += x.at(2, k) * w.at(k, 1);
  check("matmul_batch_shape", y.rows == 4 && y.cols == 2 && close(y.at(2, 1), want, 1e-6f));
}

void test_biasadd_hand() {
  Tensor x(2, 2);
  x.at(0, 0) = 1; x.at(0, 1) = 2; x.at(1, 0) = 3; x.at(1, 1) = 4;
  Graph g;
  g.nodes.push_back(make_bias({10.0f, -1.0f}));
  Tensor y = g.run(x);
  check("biasadd_hand_computed", close(y.at(0, 0), 11, 1e-6f) && close(y.at(0, 1), 1, 1e-6f) &&
                                     close(y.at(1, 0), 13, 1e-6f) && close(y.at(1, 1), 3, 1e-6f));
}

void test_relu_hand() {
  Tensor x(1, 4);
  x.at(0, 0) = -2; x.at(0, 1) = 0; x.at(0, 2) = 3; x.at(0, 3) = -0.5f;
  Graph g;
  g.nodes.push_back(make_op(OpType::ReLU, "relu"));
  Tensor y = g.run(x);
  check("relu_hand_computed",
        y.at(0, 0) == 0 && y.at(0, 1) == 0 && close(y.at(0, 2), 3, 1e-6f) && y.at(0, 3) == 0);
}

void test_softmax_sums_to_one() {
  std::mt19937 rng(3);
  Tensor x = random_tensor(5, 10, rng);
  Graph g;
  g.nodes.push_back(make_op(OpType::Softmax, "softmax"));
  Tensor y = g.run(x);
  bool ok = true;
  for (int r = 0; r < y.rows; ++r) {
    float s = 0.0f;
    for (int c = 0; c < y.cols; ++c) {
      s += y.at(r, c);
      ok = ok && y.at(r, c) >= 0.0f;
    }
    ok = ok && close(s, 1.0f, 1e-5f);
  }
  check("softmax_sums_to_one", ok);
}

void test_softmax_hand() {
  Tensor x(1, 2);
  x.at(0, 0) = 0.0f;
  x.at(0, 1) = std::log(2.0f);
  Graph g;
  g.nodes.push_back(make_op(OpType::Softmax, "softmax"));
  Tensor y = g.run(x);
  check("softmax_hand_computed",
        close(y.at(0, 0), 1.0f / 3.0f, 1e-6f) && close(y.at(0, 1), 2.0f / 3.0f, 1e-6f));
}

void test_softmax_stability() {
  Tensor x(1, 2);
  x.at(0, 0) = 1000.0f;
  x.at(0, 1) = 1000.0f;
  Graph g;
  g.nodes.push_back(make_op(OpType::Softmax, "softmax"));
  Tensor y = g.run(x);
  check("softmax_large_input_stable", std::isfinite(y.at(0, 0)) && close(y.at(0, 0), 0.5f, 1e-6f));
}

// ---------- fusion ----------

void test_fusion_op_counts() {
  std::mt19937 rng(4);
  Graph g = random_mlp(rng, {8, 6, 5, 3});
  // 3-layer MLP graph: 3 MatMul + 3 BiasAdd + 2 ReLU + Softmax = 9 ops -> 4.
  FusionReport rep = fuse(g);
  check("fusion_op_counts_9_to_4",
        rep.ops_before == 9 && rep.ops_after == 4 && rep.fused_dense == 3);
}

void test_fusion_structure() {
  std::mt19937 rng(5);
  Graph g = random_mlp(rng, {8, 6, 5, 3});
  fuse(g);
  check("fusion_structure", g.nodes.size() == 4 && g.nodes[0].type == OpType::FusedDense &&
                                g.nodes[1].type == OpType::FusedDense &&
                                g.nodes[2].type == OpType::FusedDense &&
                                g.nodes[3].type == OpType::Softmax);
}

void test_fusion_relu_flags() {
  std::mt19937 rng(6);
  Graph g = random_mlp(rng, {8, 6, 5, 3});
  fuse(g);
  check("fusion_relu_flags", g.nodes[0].relu && g.nodes[1].relu && !g.nodes[2].relu);
}

void test_fusion_equiv_single_layer() {
  std::mt19937 rng(7);
  Graph g;
  g.nodes.push_back(make_matmul(random_tensor(16, 8, rng)));
  std::vector<float> b(8);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  for (float &v : b) v = d(rng);
  g.nodes.push_back(make_bias(b));
  g.nodes.push_back(make_op(OpType::ReLU, "relu"));
  Graph fused = g;
  fuse(fused);
  Tensor x = random_tensor(4, 16, rng);
  check("fusion_equivalence_single_layer", max_abs_diff(g.run(x), fused.run(x)) < 1e-6f);
}

void test_fusion_equiv_mlp() {
  std::mt19937 rng(8);
  Graph g = random_mlp(rng, {32, 24, 16, 10});
  Graph fused = g;
  fuse(fused);
  Tensor x = random_tensor(5, 32, rng);
  check("fusion_equivalence_mlp", max_abs_diff(g.run(x), fused.run(x)) < 1e-6f);
}

// ---------- quantization ----------

void test_quant_scale() {
  check("quant_scale_amax", close(quant_scale(12.7f), 0.1f, 1e-7f));
}

void test_quant_scale_zero_tensor() {
  check("quant_scale_zero_tensor", quant_scale(0.0f) == 1.0f);
}

void test_quantize_value_round_and_clamp() {
  check("quantize_value_round_and_clamp",
        quantize_value(0.26f, 0.1f) == 3 && quantize_value(-0.26f, 0.1f) == -3 &&
            quantize_value(100.0f, 0.1f) == 127 && quantize_value(-100.0f, 0.1f) == -127);
}

void test_quant_zero_preserved() {
  check("quant_zero_preserved", quantize_value(0.0f, 0.01f) == 0);
}

void test_quant_roundtrip_bound() {
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> d(-2.0f, 2.0f);
  std::vector<float> vals(256);
  for (float &v : vals) v = d(rng);
  const float amax = abs_max(vals.data(), vals.size());
  const float scale = quant_scale(amax);
  std::vector<int8_t> q = quantize_tensor(vals.data(), vals.size(), scale);
  bool ok = true;
  for (size_t i = 0; i < vals.size(); ++i) {
    const float deq = static_cast<float>(q[i]) * scale;
    ok = ok && std::fabs(deq - vals[i]) <= scale / 2.0f + 1e-6f;
  }
  check("quant_dequant_roundtrip_bound", ok);
}

void test_calibration_in_scale() {
  Graph g;
  Node fd;
  fd.type = OpType::FusedDense;
  fd.name = "fd";
  fd.weight = Tensor(2, 2);
  fd.weight.at(0, 0) = 1.0f;
  fd.weight.at(1, 1) = 1.0f;
  fd.bias = {0.0f, 0.0f};
  g.nodes.push_back(fd);
  Tensor calib(1, 2);
  calib.at(0, 0) = 3.0f;
  calib.at(0, 1) = -5.0f;  // abs-max = 5
  Graph q = quantize_graph(g, calib);
  check("calibration_in_scale",
        q.nodes[0].type == OpType::QuantDense && close(q.nodes[0].in_scale, 5.0f / 127.0f, 1e-7f));
}

void test_quant_dense_hand() {
  Node qd;
  qd.type = OpType::QuantDense;
  qd.name = "qd";
  qd.weight = Tensor(2, 2);  // shape metadata
  qd.qweight = {1, -1, 2, 0};  // [[1,-1],[2,0]] row-major [in=2, out=2]
  qd.w_scale = 1.0f;
  qd.in_scale = 1.0f;
  qd.bias = {0.5f, 0.5f};
  qd.relu = true;
  Graph g;
  g.nodes.push_back(qd);
  Tensor x(1, 2);
  x.at(0, 0) = 1.0f;
  x.at(0, 1) = 2.0f;
  // acc = [1*1+2*2, 1*-1+2*0] = [5, -1]; +bias = [5.5, -0.5]; relu -> [5.5, 0]
  Tensor y = g.run(x);
  check("quant_dense_int32_hand", close(y.at(0, 0), 5.5f, 1e-6f) && y.at(0, 1) == 0.0f);
}

void test_quant_dense_close_to_fp32() {
  std::mt19937 rng(10);
  Graph g = random_mlp(rng, {32, 24, 16, 10});
  Graph fused = g;
  fuse(fused);
  Tensor calib = random_tensor(64, 32, rng);
  Graph q = quantize_graph(fused, calib);
  Tensor x = random_tensor(8, 32, rng);
  // Strip softmax; compare pre-softmax logits. PTQ error is relative to
  // activation magnitude, so bound |diff| by 5% of the fp32 logit abs-max
  // (measured: ~2.4% for this seed).
  Graph fl = fused, ql = q;
  fl.nodes.pop_back();
  ql.nodes.pop_back();
  const Tensor a = fl.run(x);
  const Tensor b = ql.run(x);
  const float amax = abs_max(a.data.data(), a.data.size());
  check("quant_dense_close_to_fp32", max_abs_diff(a, b) < 0.05f * amax);
}

// ---------- end-to-end golden vectors (need data/ from train_mlp.py) ----------

void run_e2e_tests(const std::string &data_dir) {
  const std::string model_path = data_dir + "/model.bin";
  if (!file_exists(model_path) || !file_exists(data_dir + "/golden_images.bin")) {
    skip("e2e_model_shapes", "data/ missing; run `make train` first");
    skip("e2e_golden_fp32_naive", "data/ missing");
    skip("e2e_golden_fp32_fused", "data/ missing");
    skip("e2e_golden_int8_preds", "data/ missing");
    return;
  }
  Graph naive = load_model(model_path);
  check("e2e_model_shapes", naive.nodes.size() == 9 && naive.nodes[0].weight.rows == 784 &&
                                naive.nodes[0].weight.cols == 256 &&
                                naive.nodes[3].weight.cols == 128 &&
                                naive.nodes[6].weight.cols == 10);

  const Tensor gx = load_matrix(data_dir + "/golden_images.bin");
  const Tensor glogits = load_matrix(data_dir + "/golden_logits.bin");
  Graph naive_l = naive;
  naive_l.nodes.pop_back();  // strip softmax -> logits
  const Tensor nl = naive_l.run(gx);
  const std::vector<uint8_t> np = argmax_rows(nl);
  const std::vector<uint8_t> gp = argmax_rows(glogits);
  bool preds_ok = true;
  for (size_t i = 0; i < np.size(); ++i) preds_ok = preds_ok && np[i] == gp[i];
  check("e2e_golden_fp32_naive", max_abs_diff(nl, glogits) < 5e-4f && preds_ok);

  Graph fused = naive;
  fuse(fused);
  Graph fused_l = fused;
  fused_l.nodes.pop_back();
  const Tensor fl = fused_l.run(gx);
  check("e2e_golden_fp32_fused", max_abs_diff(fl, glogits) < 5e-4f);

  const Tensor calib = load_matrix(data_dir + "/calib_images.bin");
  Graph q = quantize_graph(fused, calib);
  Graph ql = q;
  ql.nodes.pop_back();
  const std::vector<uint8_t> qp = argmax_rows(ql.run(gx));
  size_t same = 0;
  for (size_t i = 0; i < qp.size(); ++i) same += (qp[i] == gp[i]) ? 1 : 0;
  check("e2e_golden_int8_preds", same >= 7);  // INT8 may legitimately flip rare borderline preds
}

}  // namespace

int main(int argc, char **argv) {
  const std::string data_dir = argc > 1 ? argv[1] : "data";
  test_matmul_hand();
  test_matmul_identity();
  test_matmul_batch_shape();
  test_biasadd_hand();
  test_relu_hand();
  test_softmax_sums_to_one();
  test_softmax_hand();
  test_softmax_stability();
  test_fusion_op_counts();
  test_fusion_structure();
  test_fusion_relu_flags();
  test_fusion_equiv_single_layer();
  test_fusion_equiv_mlp();
  test_quant_scale();
  test_quant_scale_zero_tensor();
  test_quantize_value_round_and_clamp();
  test_quant_zero_preserved();
  test_quant_roundtrip_bound();
  test_calibration_in_scale();
  test_quant_dense_hand();
  test_quant_dense_close_to_fp32();
  run_e2e_tests(data_dir);
  std::printf("\nC++ tests: %d passed, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
  return g_fail == 0 ? 0 : 1;
}
