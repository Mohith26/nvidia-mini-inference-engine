#pragma once
#include <cstddef>
#include <vector>

// Row-major float32 2-D tensor [rows, cols].
struct Tensor {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;

  Tensor() = default;
  Tensor(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * static_cast<size_t>(c), 0.0f) {}

  float &at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
  const float &at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
  const float *row(int r) const { return &data[static_cast<size_t>(r) * cols]; }
  float *row(int r) { return &data[static_cast<size_t>(r) * cols]; }
};
