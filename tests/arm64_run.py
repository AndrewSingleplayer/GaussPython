"""Run tests/features.ha natively and on ARM64 (qemu-aarch64) and compare every value."""
import ctypes
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from happ.driver import build, compile_ir, load_program  # noqa: E402
from happ.llvm import LLVMGen  # noqa: E402
from happ.targets import Toolchain, target_info  # noqa: E402

FEATURES = os.path.join(ROOT, "tests", "features.ha")


def make_input(n=1000, seed=7):
    data = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    return struct.pack("<q", n) + data.tobytes()


def decode(buf):
    out = []
    for i in range(0, len(buf), 16):
        tag, = struct.unpack_from("<q", buf, i)
        if tag == 0:
            out.append(("i", struct.unpack_from("<q", buf, i + 8)[0]))
        else:
            out.append(("f", struct.unpack_from("<d", buf, i + 8)[0]))
    return out


def run_native(work):
    lib = build(FEATURES, ["linux-x64"], work, quiet=True, bridges=False)["linux-x64"]
    dll = ctypes.CDLL(lib)
    dll.test_main.restype = ctypes.c_int64
    dll.test_main.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    out = ctypes.create_string_buffer(1 << 20)
    n = dll.test_main(make_input(), out)
    return decode(out.raw[:n])


def run_arm64(work):
    tc = Toolchain()
    tinfo = target_info("linux-arm64")
    prog = load_program(FEATURES)
    ir = LLVMGen(prog, tinfo, mode="lib").generate(["test_main"])
    obj = compile_ir(tc, ir, tinfo, os.path.join(work, "features-arm64.o"))
    hobj = os.path.join(work, "harness-arm64.o")
    subprocess.run([tc.find("clang"), "--target=aarch64-linux-gnu", "-O2", "-ffreestanding", "-fno-stack-protector",
                    "-c", os.path.join(ROOT, "tests", "arm64_harness.c"), "-o", hobj], check=True)
    exe = os.path.join(work, "features-arm64")
    subprocess.run([tc.find("ld.lld"), "-static", "-e", "_start", hobj, obj, "-o", exe], check=True)
    r = subprocess.run(["qemu-aarch64", exe], input=make_input(), capture_output=True, check=True)
    return decode(r.stdout)


def compare(a, b):
    """Integers must match exactly; floats to within a few ulp (FMA use can differ)."""
    bad = []
    for i, ((ta, va), (tb, vb)) in enumerate(zip(a, b)):
        if ta != tb:
            bad.append((i, va, vb))
        elif ta == "i" and va != vb:
            bad.append((i, va, vb))
        elif ta == "f":
            if va != va and vb != vb:
                continue
            if abs(va - vb) > 1e-6 * max(1.0, abs(va)):
                bad.append((i, va, vb))
    return bad


if __name__ == "__main__":
    work = tempfile.mkdtemp()
    try:
        native = run_native(work)
        arm = run_arm64(work)
        bad = compare(native, arm)
        print(f"{len(native)} values; ARM64 vs x86-64 mismatches: {len(bad)}")
        for m in bad[:10]:
            print("  mismatch", m)
        if "-v" in sys.argv:
            for i, v in enumerate(native):
                print(i, v)
    finally:
        shutil.rmtree(work)
