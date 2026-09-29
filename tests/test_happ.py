"""HA++ test suite.   python3 -m unittest discover -s tests -v

Tests skip (with a reason) when an optional tool is missing: qemu-aarch64 (ARM64
runs), glslangValidator + a Vulkan driver (GPU runs), javac/java (JNI), clang++
(Metal emulation).
"""
import contextlib
import ctypes
import io
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
sys.path.insert(0, ROOT)
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(ROOT, "gaussian"))

from happ.checker import check  # noqa: E402
from happ.driver import build, load_program, main as happ_main  # noqa: E402
from happ.errors import HappError  # noqa: E402
from happ.parser import parse  # noqa: E402
from happ.targets import Toolchain, expand_targets  # noqa: E402

TC = Toolchain()
P, U32, I64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64


def have(*tools):
    return all(TC.find(t) or shutil.which(t) for t in tools)


def vulkan_ok():
    if not have("glslangValidator"):
        return False
    try:
        r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=60)
        return r.returncode == 0 and "deviceName" in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


VULKAN = vulkan_ok()


def check_src(src):
    with open(os.path.join(ROOT, "happ", "std", "math.ha")) as fh:
        std_text = fh.read()
    m = parse(src, "test.ha")
    s = parse(std_text, "std/math.ha")
    s.is_std = True
    return check([m, s])


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="happ-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write(text)
        return path


# ====================================================================== front end

class FrontEnd(unittest.TestCase):
    ERRORS = [
        ("fn f() -> i32 { return 1.5; }", "expected an integer"),
        ("fn f(x: f32) -> f32 { return x + 1 as f64; }", "can't combine f32 and f64"),
        ("fn f() { let x = 1; x = 2; }", "declared with 'let'"),
        ("fn f() -> f32 { return y; }", "unknown name 'y'"),
        ("fn f() { let v = vec3(1.0, 2.0); }", "needs 3 components"),
        ("struct S { a: f32 } fn f(s: S) -> f32 { return s.b; }", "has no field 'b'"),
        ("fn f(x: i32) -> i32 { if x > 0 { return 1; } }", "without returning"),
        ("fn f() { break; }", "outside a loop"),
        ("fn f() { let x: u8 = 300; }", "doesn't fit in u8"),
        ("export fn f(v: vec3) {}", "only pass numbers, bools and pointers"),
        ("kernel k(x: *f64) {}", "GPUs can't store f64"),
        ("fn h(p: *f32) -> f32 { return p[0]; }\nkernel k(b: *f32) { b[0] = h(b); }", "takes a pointer"),
        ("fn f() { print(1); }\nkernel k(b: *f32) { f(); }", "print isn't available"),
        ("fn f() { barrier(); }", "only be used directly inside a kernel"),
        ("fn f() -> f32 { return sqr(2.0); }", "unknown function 'sqr'"),
        ("fn f() -> i32 { return 1 < 2 < 3; }", "cannot be chained"),
        ("struct A { b: B } struct B { a: A }", "contains itself"),
        ("fn f() { let x = 1.0u; }", "after number"),
        ("fn f(a: f32) -> f64 { return exp(a as f64); }", "not available for f64"),
    ]

    def test_error_messages(self):
        for src, expect in self.ERRORS:
            with self.subTest(src=src):
                with self.assertRaises(HappError) as cm:
                    check_src(src)
                self.assertIn(expect, str(cm.exception))

    def test_error_points_at_source(self):
        with self.assertRaises(HappError) as cm:
            check_src("fn f() -> f32 {\n    return unknown_thing;\n}")
        text = str(cm.exception)
        self.assertIn("test.ha:2:12", text)
        self.assertIn("return unknown_thing;", text)
        self.assertIn("^", text)

    def test_literals_adapt_to_context(self):
        prog = check_src("fn f(h: f16, u: u8, v: vec3) -> f16 { let a = u + 1; let b = v * 2.0; return h * 2.0; }")
        self.assertIn("f", prog.fns)

    def test_user_names_override_std(self):
        prog = check_src("const PI: f32 = 3.0; fn sigmoid(x: f32) -> f32 { return x; }")
        self.assertEqual(prog.consts["PI"].value, 3.0)
        self.assertFalse(prog.fns["sigmoid"].is_std)


# ====================================================================== CPU

class CPU(TempDirCase):
    def test_run_hello(self):
        path = self.write("hello.ha", """
struct P { pos: vec3, w: f32 }
fn main() -> i32 {
    var p = P(pos: vec3(1.0, 2.0, 3.0), w: 0.5);
    p.pos.xz = vec2(9.0, 8.0);
    print("hi", 42, p.pos, sigmoid(0.0), true);
    return 3;
}
""")
        r = subprocess.run([sys.executable, "-m", "happ", "run", path], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertEqual(r.stdout.strip(), "hi 42 (9, 2, 8) 0.5 true")

    def test_features_native(self):
        import arm64_run
        values = arm64_run.run_native(self.tmp)
        ints = [v for t, v in values if t == "i"]
        self.assertEqual(ints[:17], [-56, 4, -2147483648, -3, -1, 1333333333, 3, 2, -4, 1073741820, 242, 24, 31,
                                     7, 250, 23416728348467685, 9223372030926249001])
        self.assertEqual(ints[17:22], [2147483647, -2147483648, 0, 255, 0])

    def test_math_accuracy(self):
        path = self.write("m.ha", "".join(
            f"export fn t_{f}(x: *f32, o: *f32, n: i64) {{ for i in 0..n {{ o[i] = {f}(x[i]); }} }}\n"
            for f in ("exp", "log", "sin", "cos", "tanh", "atan", "asin", "exp2", "log2")))
        lib = ctypes.CDLL(build(path, ["linux-x64"], self.tmp, quiet=True, bridges=False)["linux-x64"])
        rng = np.random.default_rng(1)
        cases = {"exp": (np.exp, rng.uniform(-80, 80, 200000), 2),
                 "log": (np.log, np.exp(rng.uniform(-80, 80, 200000)), 2),
                 "sin": (np.sin, rng.uniform(-50, 50, 200000), 1e-7),
                 "cos": (np.cos, rng.uniform(-50, 50, 200000), 1e-7),
                 "tanh": (np.tanh, rng.uniform(-10, 10, 200000), 2),
                 "atan": (np.arctan, rng.uniform(-50, 50, 200000), 3),
                 "asin": (np.arcsin, rng.uniform(-1, 1, 200000), 3),
                 "exp2": (np.exp2, rng.uniform(-120, 120, 200000), 2),
                 "log2": (np.log2, np.exp(rng.uniform(-80, 80, 200000)), 2)}
        for name, (ref, x, tol) in cases.items():
            with self.subTest(fn=name):
                x = x.astype(np.float32)
                o = np.empty_like(x)
                f = getattr(lib, f"t_{name}")
                f(P(x.ctypes.data), P(o.ctypes.data), I64(len(x)))
                r = ref(x.astype(np.float64))
                if tol < 1:   # absolute error (sin/cos near zeros)
                    self.assertLess(np.abs(o - r).max(), tol)
                else:         # ulp
                    ulp = np.spacing(np.abs(r.astype(np.float32))).astype(np.float64)
                    self.assertLessEqual((np.abs(o - r) / ulp).max(), tol)

    def test_fast_loops_vectorize(self):
        path = self.write("v.ha", "export fn f(x: *f32, n: i64) { for i in 0..n { x[i] = sigmoid(x[i]) * 2.0; } }")
        out = io.StringIO()
        for target, pattern in (("android-arm64", "fmul\tv"), ("linux-x64", "vmulps")):
            with self.subTest(target=target), contextlib.redirect_stdout(out):
                happ_main(["emit", path, "asm", "-t", target])
            self.assertIn(pattern, out.getvalue())


# ====================================================================== targets

class Targets(TempDirCase):
    SRC = """
struct V { p: vec3, h: f16 }
export fn scale(a: *f32, n: i64, k: f32) { for i in 0..n { a[i] = a[i] * k; } }
export fn norm(v: *V) -> f32 { return length(v.p) + (v.h as f32); }
"""

    def test_every_target_links_without_sdks(self):
        path = self.write("lib.ha", self.SRC)
        out = os.path.join(self.tmp, "build")
        produced = build(path, expand_targets("all"), out, quiet=True)
        readelf, readobj = TC.find("llvm-readelf"), TC.find("llvm-readobj")
        for t in expand_targets("all"):
            with self.subTest(target=t):
                with open(produced[t], "rb") as fh:
                    data = fh.read(8)
                if t.startswith(("android", "linux")):
                    self.assertEqual(data[:4], b"\x7fELF")
                elif t.startswith("windows"):
                    self.assertEqual(data[:2], b"MZ")
                else:
                    self.assertEqual(data[:8], b"!<arch>\n")
        if readelf:
            for abi in ("arm64-v8a", "x86_64"):
                so = os.path.join(out, "android", "jniLibs", abi, "liblib.so")
                seg = subprocess.run([readelf, "-lW", so], capture_output=True, text=True).stdout
                self.assertIn("0x4000", seg, "Android 15+ needs 16 KB aligned segments")
                dyn = subprocess.run([readelf, "-d", so], capture_output=True, text=True).stdout
                self.assertNotIn("NEEDED", dyn, "a plain HA++ library needs no other library")
        if readobj:
            exp = subprocess.run([readobj, "--coff-exports", produced["windows-x64"]], capture_output=True,
                                 text=True).stdout
            for sym in ("scale", "norm", "Java_com_happ_lib_Lib_scale"):
                self.assertIn(sym, exp)
        with open(os.path.join(out, "apple", "Lib", "LibCore.xcframework", "Info.plist"), "rb") as fh:
            plist = plistlib.load(fh)
        ids = sorted(lib["LibraryIdentifier"] for lib in plist["AvailableLibraries"])
        self.assertEqual(ids, ["ios-arm64", "ios-arm64_x86_64-simulator", "macos-arm64_x86_64"])
        with open(os.path.join(out, "include", "lib.h")) as fh:
            header = fh.read()
        self.assertIn("float norm(V *v);", header)
        self.assertIn("HA_ASSERT_SIZE(V, 16);", header)
        # the generated C header compiles and agrees with HA++ layout
        cfile = self.write("use.c", '#include "lib.h"\nint main(void) { V v = {{1,2,3}, 0}; return (int)norm(&v); }\n')
        subprocess.run([TC.find("clang"), "-fsyntax-only", "-I", os.path.join(out, "include"), cfile], check=True)

    @unittest.skipUnless(have("qemu-aarch64", "ld.lld"), "needs qemu-aarch64 and ld.lld")
    def test_arm64_matches_x86_64(self):
        import arm64_run
        native = arm64_run.run_native(self.tmp)
        arm = arm64_run.run_arm64(self.tmp)
        self.assertEqual(len(native), len(arm))
        self.assertEqual(arm64_run.compare(native, arm), [])


# ====================================================================== GPU

@unittest.skipUnless(have("clang") and os.path.exists("/usr/include/vulkan/vulkan.h"), "needs vulkan.h")
class VulkanHeader(unittest.TestCase):
    def test_minimal_header_matches_official(self):
        import vk_layout
        n, diffs = vk_layout.run()
        self.assertGreater(n, 200)
        self.assertEqual(diffs, [])


@unittest.skipUnless(VULKAN, "needs glslangValidator and a Vulkan driver (e.g. lavapipe)")
class GPU(TempDirCase):
    def gpu_lib(self, src, name):
        out = build(src, ["linux-x64"], os.path.join(self.tmp, name), quiet=True)
        lib = ctypes.CDLL(out["linux-x64"])
        for f, r, a in [("ha_gpu_create", P, [ctypes.c_char_p, ctypes.c_size_t]),
                        ("ha_buffer_create", P, [P, ctypes.c_size_t]), ("ha_buffer_data", P, [P]),
                        ("ha_kernel_create", P, [P, P, ctypes.c_size_t, U32, U32]),
                        ("ha_dispatch", ctypes.c_int, [P, P, P, P, U32, U32, U32])]:
            fn = getattr(lib, f)
            fn.restype, fn.argtypes = r, a
        gpu = lib.ha_gpu_create(None, 0)
        self.assertTrue(gpu)
        return lib, gpu

    def run_kernel(self, lib, gpu, libname, kname, arrays, push, groups):
        getter = getattr(lib, f"{libname}_spirv_{kname}")
        getter.restype, getter.argtypes = P, [P]
        size = ctypes.c_uint64()
        blob = getter(ctypes.byref(size))
        k = lib.ha_kernel_create(gpu, blob, size.value, len(arrays), len(push))
        bufs = []
        for a in arrays:
            b = lib.ha_buffer_create(gpu, a.nbytes)
            ctypes.memmove(lib.ha_buffer_data(b), a.ctypes.data, a.nbytes)
            bufs.append(b)
        arr = (P * len(bufs))(*bufs)
        self.assertEqual(lib.ha_dispatch(gpu, k, arr, push or None, *groups), 0)
        return [np.frombuffer(ctypes.string_at(lib.ha_buffer_data(b), a.nbytes), a.dtype).reshape(a.shape).copy()
                for b, a in zip(bufs, arrays)]

    def test_kernels_match_cpu(self):
        src = os.path.join(TESTS, "kernels.ha")
        lib, gpu = self.gpu_lib(src, "k")
        rng = np.random.default_rng(0)
        n = 1000
        splats = rng.normal(size=(n, 14)).astype(np.float32)
        splats[:, 2] = rng.uniform(1, 5, n)
        push = struct.pack("<If", n, 500.0)
        _, gpu_out = self.run_kernel(lib, gpu, "kernels", "project", [splats, np.zeros((n, 7), np.float32)], push,
                                     ((n + 63) // 64, 1, 1))
        cpu_out = np.zeros((n, 7), np.float32)
        lib.project_cpu.argtypes = [P, P, U32, ctypes.c_float, U32, U32, U32]
        lib.project_cpu(splats.ctypes.data, cpu_out.ctypes.data, n, 500.0, (n + 63) // 64, 1, 1)
        self.assertLess((np.abs(gpu_out - cpu_out) / (np.abs(cpu_out) + 1e-3)).max(), 1e-3)
        keys = rng.integers(0, 2 ** 32, 100000, dtype=np.uint32)
        _, counts = self.run_kernel(lib, gpu, "kernels", "hist", [keys, np.zeros(256, np.uint32)],
                                    struct.pack("<I", len(keys)), ((len(keys) + 255) // 256, 1, 1))
        np.testing.assert_array_equal(counts, np.bincount(keys >> 24, minlength=256))

    def test_splat_pipeline_matches_reference(self):
        import render as R
        import splat_reference as ref
        pipe = R.Pipeline(R.build(out_dir=os.path.join(self.tmp, "splat")))
        raw = R.synthetic_scene(1500, seed=3)
        pipe.upload(raw)
        packed = pipe.view("splats", np.uint32, len(raw) * 8).reshape(-1, 8).copy()
        np.testing.assert_array_equal(packed, ref.pack(raw))
        w, h = 200, 136
        cam = R.look_at([0, 2.0, -3.0], [0, 0, 0], w, h)
        img, total = pipe.render(cam, w, h, background=(8, 10, 22))
        rimg, rkeys, rvals = ref.render(packed, cam, w, h, background=(8, 10, 22))
        self.assertEqual(total, len(rkeys))
        np.testing.assert_array_equal(pipe.view("keys0", np.uint32, total), rkeys)
        np.testing.assert_array_equal(pipe.view("vals0", np.uint32, total), rvals)
        diff = np.abs(img.view(np.uint8).astype(int) - rimg.view(np.uint8).astype(int))
        self.assertLessEqual(diff.max(), 2)

    def test_mlp_block_matches_numpy(self):
        lib, gpu = self.gpu_lib(os.path.join(ROOT, "examples", "ai", "nn.ha"), "nn")
        rng = np.random.default_rng(0)
        m, k, n = 48, 96, 80
        a = rng.normal(size=(m, k)).astype(np.float32)
        b = rng.normal(size=(k, n)).astype(np.float32)
        c = np.zeros((m, n), np.float32)
        _, _, c_gpu = self.run_kernel(lib, gpu, "nn", "matmul", [a, b, c], struct.pack("<III", m, n, k),
                                      ((n + 15) // 16, (m + 15) // 16, 1))
        np.testing.assert_allclose(c_gpu, a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4)
        (s_gpu,) = self.run_kernel(lib, gpu, "nn", "softmax", [a.copy()], struct.pack("<II", m, k), (m, 1, 1))
        e = np.exp(a - a.max(1, keepdims=True))
        np.testing.assert_allclose(s_gpu, e / e.sum(1, keepdims=True), rtol=1e-5, atol=1e-6)
        lib.matmul_cpu.argtypes = [P, P, P, I64, I64, I64]
        lib.matmul_cpu(a.ctypes.data, b.ctypes.data, c.ctypes.data, m, n, k)
        np.testing.assert_allclose(c, a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(have("javac", "java"), "needs a JDK")
    def test_jni_from_java(self):
        out = os.path.join(self.tmp, "jk")
        build(os.path.join(TESTS, "kernels.ha"), ["linux-x64"], out, quiet=True)
        classes = os.path.join(self.tmp, "classes")
        subprocess.run(["javac", "-d", classes, os.path.join(out, "android", "java", "com", "happ", "kernels",
                                                             "Kernels.java"),
                        os.path.join(TESTS, "jvm", "KernelsTest.java")], check=True, capture_output=True)
        r = subprocess.run(["java", f"-Djava.library.path={os.path.join(out, 'linux-x64')}", "-cp", classes,
                            "KernelsTest"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count("correct"), 3, r.stdout)


    @unittest.skipUnless(have("javac", "java"), "needs a JDK")
    def test_android_splat_renderer_from_java(self):
        """gaussian/android/SplatRenderer.java must render exactly what the Python host renders."""
        import render as R
        out = os.path.join(self.tmp, "sb")
        lib = R.build(out_dir=out)
        raw = R.synthetic_scene(2000, seed=5)
        raw.tofile(os.path.join(self.tmp, "raw.bin"))
        w, h = 160, 112
        cam = R.look_at([0, 2.4, -3.5], [0, 0, 0], w, h)
        with open(os.path.join(self.tmp, "cam.bin"), "wb") as f:
            f.write(R.camera_bytes(cam, w, h, (w + 15) // 16, (h + 15) // 16))
        pipe = R.Pipeline(lib)
        pipe.upload(raw)
        img, _ = pipe.render(cam, w, h, background=(8, 10, 22))
        classes = os.path.join(self.tmp, "classes")
        subprocess.run(["javac", "-d", classes, os.path.join(out, "android", "java", "com", "happ", "splat", "Splat.java"),
                        os.path.join(ROOT, "gaussian", "android", "SplatRenderer.java"),
                        os.path.join(TESTS, "jvm", "SplatDemo.java")], check=True, capture_output=True)
        r = subprocess.run(["java", f"-Djava.library.path={os.path.join(out, 'linux-x64')}", "-cp", classes,
                            "SplatDemo", self.tmp, str(w), str(h)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        java_img = np.fromfile(os.path.join(self.tmp, "out.bin"), np.uint8)
        np.testing.assert_array_equal(java_img, img.view(np.uint8).ravel())


@unittest.skipUnless(have("clang++"), "needs clang++")
class MetalEmulation(TempDirCase):
    def test_metal_kernels_compile_and_match_cpu(self):
        from happ.gpu import generate_metal
        prog = load_program(os.path.join(TESTS, "kernels.ha"))
        with open(os.path.join(self.tmp, "k.metal"), "w") as f:
            f.write(generate_metal(prog))
        n = 500
        rng = np.random.default_rng(0)
        splats = rng.normal(size=(n, 14)).astype(np.float32)
        splats[:, 2] = rng.uniform(1, 5, n)
        splats.tofile(os.path.join(self.tmp, "in.bin"))
        drv = self.write("drv.cpp", f"""#include "k.metal"
#include <cstdio>
#include <vector>
int main() {{
    const unsigned n = {n};
    std::vector<float> s(n * 14), out(n * 7, 0.0f);
    FILE *f = fopen("in.bin", "rb"); fread(s.data(), 4, n * 14, f); fclose(f);
    ha_params_project p{{n, 500.0f}};
    static_assert(sizeof(h_Splat) == 56 && sizeof(h_Out2D) == 28, "layout");
    for (unsigned x = 0; x < ((n + 63) / 64) * 64; x++)
        project((h_Splat *)s.data(), (h_Out2D *)out.data(), p, metal::uint3{{x, 0, 0}}, metal::uint3{{x % 64, 0, 0}},
                metal::uint3{{x / 64, 0, 0}}, metal::uint3{{(n + 63) / 64, 1, 1}}, metal::uint3{{64, 1, 1}});
    f = fopen("out.bin", "wb"); fwrite(out.data(), 4, n * 7, f); fclose(f);
}}
""")
        exe = os.path.join(self.tmp, "drv")
        subprocess.run(["clang++", "-std=c++17", "-O1", "-I", os.path.join(TESTS, "metal_shim"),
                        "-Wno-unknown-attributes", "-Wno-ignored-attributes", drv, "-o", exe], check=True)
        subprocess.run([exe], check=True, cwd=self.tmp)
        metal_out = np.fromfile(os.path.join(self.tmp, "out.bin"), np.float32).reshape(n, 7)
        lib = ctypes.CDLL(build(os.path.join(TESTS, "kernels.ha"), ["linux-x64"], os.path.join(self.tmp, "b"),
                                quiet=True, bridges=False, gpu_runtime=False)["linux-x64"])
        lib.project_cpu.argtypes = [P, P, U32, ctypes.c_float, U32, U32, U32]
        cpu = np.zeros((n, 7), np.float32)
        lib.project_cpu(splats.ctypes.data, cpu.ctypes.data, n, 500.0, (n + 63) // 64, 1, 1)
        self.assertLess((np.abs(metal_out - cpu) / (np.abs(cpu) + 1e-3)).max(), 1e-3)
        # every example's Metal output at least compiles
        for ex in ("gaussian/splat.ha", "examples/ai/nn.ha"):
            with self.subTest(example=ex):
                src = os.path.join(self.tmp, "ex.metal")
                with open(src, "w") as f:
                    f.write(generate_metal(load_program(os.path.join(ROOT, ex))))
                subprocess.run(["clang++", "-std=c++17", "-fsyntax-only", "-x", "c++", "-I",
                                os.path.join(TESTS, "metal_shim"), "-Wno-unknown-attributes",
                                "-Wno-ignored-attributes", src], check=True)


if __name__ == "__main__":
    unittest.main()
