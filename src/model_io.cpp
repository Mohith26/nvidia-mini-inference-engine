#include <cstdio>
#include <stdexcept>
#include <sys/stat.h>

#include "graph.h"

namespace {

constexpr uint32_t kMagic = 0x54464F52;  // "TFOR"

struct File {
  FILE *f;
  explicit File(const std::string &path) : f(std::fopen(path.c_str(), "rb")) {
    if (!f) throw std::runtime_error("cannot open " + path);
  }
  ~File() { std::fclose(f); }
  void read(void *dst, size_t bytes) {
    if (std::fread(dst, 1, bytes, f) != bytes) throw std::runtime_error("short read");
  }
  uint32_t u32() {
    uint32_t v = 0;
    read(&v, 4);
    return v;  // little-endian host (arm64/x86_64)
  }
};

}  // namespace

bool file_exists(const std::string &path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

Graph load_model(const std::string &path) {
  File f(path);
  if (f.u32() != kMagic) throw std::runtime_error("bad model magic in " + path);
  const uint32_t layers = f.u32();
  Graph g;
  for (uint32_t l = 0; l < layers; ++l) {
    const uint32_t d_in = f.u32();
    const uint32_t d_out = f.u32();
    Node mm;
    mm.type = OpType::MatMul;
    mm.name = "matmul_" + std::to_string(l);
    mm.weight = Tensor(static_cast<int>(d_in), static_cast<int>(d_out));
    f.read(mm.weight.data.data(), sizeof(float) * d_in * d_out);
    Node ba;
    ba.type = OpType::BiasAdd;
    ba.name = "bias_add_" + std::to_string(l);
    ba.bias.resize(d_out);
    f.read(ba.bias.data(), sizeof(float) * d_out);
    g.nodes.push_back(std::move(mm));
    g.nodes.push_back(std::move(ba));
    if (l + 1 < layers) {
      Node r;
      r.type = OpType::ReLU;
      r.name = "relu_" + std::to_string(l);
      g.nodes.push_back(std::move(r));
    }
  }
  Node sm;
  sm.type = OpType::Softmax;
  sm.name = "softmax";
  g.nodes.push_back(std::move(sm));
  return g;
}

Tensor load_matrix(const std::string &path) {
  File f(path);
  const uint32_t n = f.u32();
  const uint32_t d = f.u32();
  Tensor t(static_cast<int>(n), static_cast<int>(d));
  f.read(t.data.data(), sizeof(float) * n * d);
  return t;
}

std::vector<uint8_t> load_labels(const std::string &path) {
  File f(path);
  const uint32_t n = f.u32();
  std::vector<uint8_t> v(n);
  f.read(v.data(), n);
  return v;
}

std::vector<uint8_t> argmax_rows(const Tensor &t) {
  std::vector<uint8_t> out(static_cast<size_t>(t.rows));
  for (int r = 0; r < t.rows; ++r) {
    const float *p = t.row(r);
    int best = 0;
    for (int j = 1; j < t.cols; ++j) {
      if (p[j] > p[best]) best = j;
    }
    out[static_cast<size_t>(r)] = static_cast<uint8_t>(best);
  }
  return out;
}
