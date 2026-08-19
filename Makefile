CXX := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -Isrc
PY := .venv/bin/python
PYTEST := .venv/bin/pytest

OBJS := build/graph.o build/quant.o build/model_io.o

.PHONY: all test bench train eval-run clean

all: build/eval build/bench build/run_tests

build:
	mkdir -p build

build/%.o: src/%.cpp src/graph.h src/tensor.h | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/eval: eval/eval_main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

build/bench: bench/bench_main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

build/run_tests: tests/test_main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Download MNIST, train the MLP, export weights + eval data to data/.
train:
	$(PY) train/train_mlp.py

test: build/run_tests
	./build/run_tests data
	$(PYTEST) tests -q --color=no -rN 2>&1 | grep -aE "passed|failed|error" | tail -1

eval-run: build/eval
	./build/eval data results

bench: build/bench
	./build/bench data results

clean:
	rm -rf build
