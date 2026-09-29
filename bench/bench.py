"""Speed check: HA++ vs the same loop in C (clang -O3) vs NumPy, on this computer's CPU.

    python3 bench/bench.py
"""
import ctypes
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from happ.driver import build  # noqa: E402
from happ.targets import Toolchain  # noqa: E402

HA = """
export fn sigmoid_all(x: *f32, n: i64) { for i in 0..n { x[i] = 1.0 / (1.0 + exp(-x[i])); } }
export fn saxpy(y: *f32, x: *f32, a: f32, n: i64) { for i in 0..n { y[i] = a * x[i] + y[i]; } }
export fn dot_all(x: *f32, y: *f32, n: i64) -> f32 { var s = 0.0; for i in 0..n { s += x[i] * y[i]; } return s; }
export fn matmul(a: *f32, b: *f32, c: *f32, m: i64, n: i64, k: i64) {
    for i in 0..m {
        let crow = c + i * n;
        for j in 0..n { crow[j] = 0.0; }
        for p in 0..k {
            let av = a[i * k + p];
            let brow = b + p * n;
            for j in 0..n { crow[j] += av * brow[j]; }
        }
    }
}
"""

C = r"""
#include <math.h>
#include <stdint.h>
void sigmoid_all(float *x, int64_t n) { for (int64_t i = 0; i < n; i++) x[i] = 1.0f / (1.0f + expf(-x[i])); }
void saxpy(float *y, const float *x, float a, int64_t n) { for (int64_t i = 0; i < n; i++) y[i] = a * x[i] + y[i]; }
float dot_all(const float *x, const float *y, int64_t n) { float s = 0; for (int64_t i = 0; i < n; i++) s += x[i] * y[i]; return s; }
void matmul(const float *a, const float *b, float *c, int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; i++) {
        float *crow = c + i * n;
        for (int64_t j = 0; j < n; j++) crow[j] = 0;
        for (int64_t p = 0; p < k; p++) {
            float av = a[i * k + p];
            const float *brow = b + p * n;
            for (int64_t j = 0; j < n; j++) crow[j] += av * brow[j];
        }
    }
}
"""


def best(fn, reps=7):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def main():
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "bench.ha")
    with open(src, "w") as f:
        f.write(HA)
    ha = ctypes.CDLL(build(src, ["linux-x64"], tmp, quiet=True, bridges=False)["linux-x64"])
    csrc = os.path.join(tmp, "bench.c")
    with open(csrc, "w") as f:
        f.write(C)
    clang = Toolchain().find("clang")
    cso = os.path.join(tmp, "libcbench.so")
    subprocess.run([clang, "-O3", "-march=x86-64-v3", "-ffast-math", "-shared", "-fPIC", csrc, "-o", cso, "-lm"],
                   check=True)
    cl = ctypes.CDLL(cso)
    P, I64 = ctypes.c_void_p, ctypes.c_int64
    for lib in (ha, cl):
        lib.sigmoid_all.argtypes = [P, I64]
        lib.saxpy.argtypes = [P, P, ctypes.c_float, I64]
        lib.dot_all.argtypes = [P, P, I64]
        lib.dot_all.restype = ctypes.c_float
        lib.matmul.argtypes = [P, P, P, I64, I64, I64]
    rng = np.random.default_rng(0)
    n = 10_000_000
    x = rng.normal(size=n).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    m = 384
    a = rng.normal(size=(m, m)).astype(np.float32)
    b = rng.normal(size=(m, m)).astype(np.float32)
    c = np.zeros((m, m), np.float32)
    rows = [
        ("sigmoid, 10M floats", lambda L: L.sigmoid_all(x.ctypes.data, n), lambda: 1 / (1 + np.exp(-x))),
        ("saxpy, 10M floats", lambda L: L.saxpy(y.ctypes.data, x.ctypes.data, 0.5, n), lambda: 0.5 * x + y),
        ("dot product, 10M floats", lambda L: L.dot_all(x.ctypes.data, y.ctypes.data, n), lambda: np.dot(x, y)),
        (f"matmul {m}x{m}", lambda L: L.matmul(a.ctypes.data, b.ctypes.data, c.ctypes.data, m, m, m),
         lambda: a @ b),
    ]
    print(f"{'task':28s} {'HA++':>9s} {'C -O3':>9s} {'NumPy':>9s}   (ms, best of 7, this CPU)")
    for name, run, npf in rows:
        t_ha = best(lambda: run(ha))
        t_c = best(lambda: run(cl))
        t_np = best(npf)
        print(f"{name:28s} {t_ha:9.2f} {t_c:9.2f} {t_np:9.2f}")


if __name__ == "__main__":
    main()
