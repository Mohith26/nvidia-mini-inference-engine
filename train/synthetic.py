"""Seeded synthetic MNIST-shaped fallback dataset.

ONLY used if every MNIST mirror fails (see train_mlp.py). Any results produced
from this data MUST be tagged SYNTHETIC in RESULTS.md / results/*.json.

Generator: 10 fixed random class prototypes in [0,1]^784; each sample is its
class prototype plus seeded Gaussian noise, clipped to [0,1]. Deterministic for
a given seed.
"""
import numpy as np


def generate(seed=42, n_train=60000, n_test=10000, noise=0.35):
    rng = np.random.default_rng(seed)
    protos = rng.uniform(0.0, 1.0, size=(10, 784)).astype(np.float32)

    def make(n):
        y = rng.integers(0, 10, size=n).astype(np.uint8)
        x = protos[y] + rng.normal(0.0, noise, size=(n, 784)).astype(np.float32)
        return np.clip(x, 0.0, 1.0).astype(np.float32), y

    x_train, y_train = make(n_train)
    x_test, y_test = make(n_test)
    return x_train, y_train, x_test, y_test
