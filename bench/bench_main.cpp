// Benchmarks: images/sec + ms/image for naive vs fused vs fused+INT8 engines,
// batch sizes 1 and 64, best-of-5 full passes over the 10k test set.
// Also dumps a per-layer profile for one batch-64 pass of each engine.
// All numbers are CPU single-thread on this machine (Apple Silicon, arm64).
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "graph.h"

namespace {

using clock_t_ = std::chrono::steady_clock;

struct BenchResult {
  double best_secs = 0.0;
  int images = 0;
  double images_per_sec() const { return images / best_secs; }
  double ms_per_image() const { return best_secs * 1000.0 / images; }
};

// One full pass over x in `batch`-sized chunks (last partial batch dropped for
// uniform batch shape). Returns wall seconds.
double one_pass(const Graph &g, const Tensor &x, int batch, int *images_out,
                std::vector<ProfileEntry> *profile = nullptr) {
  const int n_batches = x.rows / batch;
  *images_out = n_batches * batch;
  Tensor chunk(batch, x.cols);
  volatile float sink = 0.0f;  // defeat dead-code elimination
  const auto t0 = clock_t_::now();
  for (int b = 0; b < n_batches; ++b) {
    const float *src = x.row(b * batch);
    std::copy(src, src + static_cast<size_t>(batch) * x.cols, chunk.data.begin());
    Tensor y = g.run(chunk, profile);
    sink = sink + y.data[0];
  }
  const std::chrono::duration<double> dt = clock_t_::now() - t0;
  (void)sink;
  return dt.count();
}

BenchResult best_of_5(const Graph &g, const Tensor &x, int batch) {
  BenchResult r;
  for (int i = 0; i < 5; ++i) {
    int images = 0;
    const double s = one_pass(g, x, batch, &images);
    r.images = images;
    if (i == 0 || s < r.best_secs) r.best_secs = s;
  }
  return r;
}

void write_bench_entry(FILE *f, const char *engine, int batch, const BenchResult &r, bool last) {
  std::fprintf(f,
               "    {\"engine\": \"%s\", \"batch\": %d, \"images\": %d, \"best_of_5_secs\": %.6f, "
               "\"images_per_sec\": %.1f, \"ms_per_image\": %.6f}%s\n",
               engine, batch, r.images, r.best_secs, r.images_per_sec(), r.ms_per_image(),
               last ? "" : ",");
}

}  // namespace

int main(int argc, char **argv) {
  const std::string data_dir = argc > 1 ? argv[1] : "data";
  const std::string results_dir = argc > 2 ? argv[2] : "results";

  Graph naive = load_model(data_dir + "/model.bin");
  Graph fused = naive;
  fuse(fused);
  const Tensor calib = load_matrix(data_dir + "/calib_images.bin");
  const Graph quant = quantize_graph(fused, calib);
  const Tensor x = load_matrix(data_dir + "/test_images.bin");

  struct EngineRef {
    const char *name;
    const Graph *g;
  };
  const EngineRef engines[] = {{"naive", &naive}, {"fused", &fused}, {"int8", &quant}};
  const int batches[] = {1, 64};

  const std::string bench_path = results_dir + "/bench.json";
  FILE *f = std::fopen(bench_path.c_str(), "w");
  if (!f) return 1;
  std::fprintf(f, "{\n  \"note\": \"CPU single-thread, Apple Silicon (arm64), clang -O2; full graph incl. softmax; best of 5 full passes over the 10k MNIST test set (partial final batch dropped)\",\n");
  std::fprintf(f, "  \"runs\": [\n");
  for (size_t e = 0; e < 3; ++e) {
    for (size_t b = 0; b < 2; ++b) {
      const BenchResult r = best_of_5(*engines[e].g, x, batches[b]);
      std::printf("%-5s batch=%-2d  %8.1f img/s  %.4f ms/img  (best of 5: %.3fs / %d imgs)\n",
                  engines[e].name, batches[b], r.images_per_sec(), r.ms_per_image(), r.best_secs,
                  r.images);
      write_bench_entry(f, engines[e].name, batches[b], r, e == 2 && b == 1);
    }
  }
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
  std::printf("wrote %s\n", bench_path.c_str());

  // Per-layer profile: one full batch-64 pass per engine.
  const std::string prof_path = results_dir + "/profile.json";
  FILE *pf = std::fopen(prof_path.c_str(), "w");
  if (!pf) return 1;
  std::fprintf(pf, "{\n  \"note\": \"per-node wall time (ms) accumulated over one full batch-64 pass of the 10k test set; CPU single-thread, Apple Silicon\",\n");
  for (size_t e = 0; e < 3; ++e) {
    std::vector<ProfileEntry> prof;
    int images = 0;
    const double total = one_pass(*engines[e].g, x, 64, &images, &prof);
    std::fprintf(pf, "  \"%s\": {\"images\": %d, \"total_secs\": %.6f, \"layers\": [\n",
                 engines[e].name, images, total);
    for (size_t i = 0; i < prof.size(); ++i) {
      std::fprintf(pf, "    {\"name\": \"%s\", \"ms\": %.3f}%s\n", prof[i].name.c_str(),
                   prof[i].ms, i + 1 < prof.size() ? "," : "");
    }
    std::fprintf(pf, "  ]}%s\n", e < 2 ? "," : "");
    std::printf("profiled %s: %d images in %.3fs\n", engines[e].name, images, total);
  }
  std::fprintf(pf, "}\n");
  std::fclose(pf);
  std::printf("wrote %s\n", prof_path.c_str());
  return 0;
}
