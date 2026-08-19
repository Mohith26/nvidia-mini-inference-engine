// Correctness eval: FP32 engine (naive + fused) vs Python oracle on the 10k
// MNIST test set, plus INT8 accuracy delta. Writes results/eval.json.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "graph.h"

namespace {

Graph strip_softmax(const Graph &g) {
  Graph out = g;
  if (!out.nodes.empty() && out.nodes.back().type == OpType::Softmax) out.nodes.pop_back();
  return out;
}

Tensor run_batched(const Graph &g, const Tensor &x, int batch) {
  Tensor out;
  for (int start = 0; start < x.rows; start += batch) {
    const int n = std::min(batch, x.rows - start);
    Tensor chunk(n, x.cols);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < x.cols; ++c) chunk.at(r, c) = x.at(start + r, c);
    Tensor y = g.run(chunk);
    if (out.rows == 0) out = Tensor(x.rows, y.cols);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < y.cols; ++c) out.at(start + r, c) = y.at(r, c);
  }
  return out;
}

double match_pct(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
  size_t same = 0;
  for (size_t i = 0; i < a.size(); ++i) same += (a[i] == b[i]) ? 1 : 0;
  return 100.0 * static_cast<double>(same) / static_cast<double>(a.size());
}

double accuracy(const std::vector<uint8_t> &pred, const std::vector<uint8_t> &label) {
  return match_pct(pred, label) / 100.0;
}

float max_abs_diff(const Tensor &a, const Tensor &b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.data.size(); ++i) {
    const float d = std::fabs(a.data[i] - b.data[i]);
    if (d > m) m = d;
  }
  return m;
}

}  // namespace

int main(int argc, char **argv) {
  const std::string data_dir = argc > 1 ? argv[1] : "data";
  const std::string results_dir = argc > 2 ? argv[2] : "results";

  Graph naive = load_model(data_dir + "/model.bin");
  Graph fused = naive;
  const FusionReport rep = fuse(fused);
  const Tensor calib = load_matrix(data_dir + "/calib_images.bin");
  const Graph quant = quantize_graph(fused, calib);

  const Tensor x = load_matrix(data_dir + "/test_images.bin");
  const std::vector<uint8_t> labels = load_labels(data_dir + "/test_labels.bin");
  const Tensor oracle_logits = load_matrix(data_dir + "/oracle_logits.bin");
  const std::vector<uint8_t> oracle_preds = load_labels(data_dir + "/oracle_preds.bin");

  // Compare pre-softmax logits (the oracle exports logits).
  const Graph naive_l = strip_softmax(naive);
  const Graph fused_l = strip_softmax(fused);
  const Graph quant_l = strip_softmax(quant);

  const int kBatch = 256;
  const Tensor naive_logits = run_batched(naive_l, x, kBatch);
  const Tensor fused_logits = run_batched(fused_l, x, kBatch);
  const Tensor int8_logits = run_batched(quant_l, x, kBatch);

  const std::vector<uint8_t> naive_preds = argmax_rows(naive_logits);
  const std::vector<uint8_t> fused_preds = argmax_rows(fused_logits);
  const std::vector<uint8_t> int8_preds = argmax_rows(int8_logits);

  const double naive_match = match_pct(naive_preds, oracle_preds);
  const double fused_match = match_pct(fused_preds, oracle_preds);
  const float naive_maxdiff = max_abs_diff(naive_logits, oracle_logits);
  const float fused_maxdiff = max_abs_diff(fused_logits, oracle_logits);
  const float fused_vs_naive = max_abs_diff(fused_logits, naive_logits);

  const double oracle_acc = accuracy(oracle_preds, labels);
  const double naive_acc = accuracy(naive_preds, labels);
  const double fused_acc = accuracy(fused_preds, labels);
  const double int8_acc = accuracy(int8_preds, labels);

  std::printf("fusion: %d ops -> %d ops (%d FusedDense)\n", rep.ops_before, rep.ops_after,
              rep.fused_dense);
  std::printf("naive vs oracle:  match %.3f%%  logit max|diff| %.3e\n", naive_match, naive_maxdiff);
  std::printf("fused vs oracle:  match %.3f%%  logit max|diff| %.3e\n", fused_match, fused_maxdiff);
  std::printf("fused vs naive:   logit max|diff| %.3e\n", fused_vs_naive);
  std::printf("accuracy: oracle %.4f  naive %.4f  fused %.4f  int8 %.4f  (int8 delta %+.4f)\n",
              oracle_acc, naive_acc, fused_acc, int8_acc, int8_acc - fused_acc);

  const std::string out = results_dir + "/eval.json";
  FILE *f = std::fopen(out.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 1;
  }
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"note\": \"CPU single-thread, Apple Silicon (arm64); real MNIST 10k test set unless data/meta.json says SYNTHETIC\",\n");
  std::fprintf(f, "  \"test_examples\": %d,\n", x.rows);
  std::fprintf(f, "  \"fusion\": {\"ops_before\": %d, \"ops_after\": %d, \"fused_dense\": %d},\n",
               rep.ops_before, rep.ops_after, rep.fused_dense);
  std::fprintf(f, "  \"fp32_naive_vs_oracle\": {\"pred_match_pct\": %.4f, \"logit_max_abs_diff\": %.6e},\n",
               naive_match, naive_maxdiff);
  std::fprintf(f, "  \"fp32_fused_vs_oracle\": {\"pred_match_pct\": %.4f, \"logit_max_abs_diff\": %.6e},\n",
               fused_match, fused_maxdiff);
  std::fprintf(f, "  \"fused_vs_naive_logit_max_abs_diff\": %.6e,\n", fused_vs_naive);
  std::fprintf(f, "  \"accuracy\": {\"oracle\": %.4f, \"fp32_naive\": %.4f, \"fp32_fused\": %.4f, \"int8\": %.4f, \"int8_minus_fp32\": %.4f},\n",
               oracle_acc, naive_acc, fused_acc, int8_acc, int8_acc - fused_acc);
  std::fprintf(f, "  \"int8_calibration_images\": %d\n", calib.rows);
  std::fprintf(f, "}\n");
  std::fclose(f);
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
