"""Render Gaussian splats with the HA++ splat pipeline on this PC's GPU (Vulkan).

    python gaussian/render.py --synthetic 20000 --out splats.png
    python gaussian/render.py --ply scene.ply --out scene.png --size 1280x720

This is the same code path an Android app uses (the Kotlin/Java version calls
the same kernels through JNI); on iPhone the Swift wrapper runs the Metal
version of the same kernels.
"""

import argparse
import ctypes
import math
import os
import struct
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

P, U32, U64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
RAW_FLOATS = 14          # GaussianRaw: pos3 log_scale3 rot4 opacity1 sh_dc3
PACKED_BYTES = 32
PROJ_BYTES = 44
CAMERA_BYTES = 104
TILE = 16


def build(out_dir=None, quiet=True):
    """Compile splat.ha for this computer and return the library path."""
    from happ.driver import build as happ_build
    from happ.targets import host_target
    out_dir = out_dir or os.path.join(HERE, "build")
    t = host_target()
    produced = happ_build(os.path.join(HERE, "splat.ha"), [t], out_dir, quiet=quiet)
    return produced[t]


class Pipeline:
    """Owns the GPU context, kernels and buffers; renders frames."""

    KERNELS = {  # name: (buffers, push-constant bytes)
        "preprocess": (5, 4), "scan_blocks": (2, 4), "scan_sums": (1, 4), "scan_add": (2, 4),
        "bin": (5, 8), "radix_hist": (2, 12), "radix_scatter": (5, 12), "clear": (1, 4),
        "ranges": (2, 4), "render": (4, 12),
    }

    def __init__(self, lib_path):
        self.lib = lib = ctypes.CDLL(lib_path)
        sig = {"ha_gpu_create": (P, [ctypes.c_char_p, ctypes.c_size_t]), "ha_gpu_name": (ctypes.c_char_p, [P]),
               "ha_buffer_create": (P, [P, ctypes.c_size_t]), "ha_buffer_data": (P, [P]),
               "ha_buffer_destroy": (None, [P]), "ha_kernel_create": (P, [P, P, ctypes.c_size_t, U32, U32]),
               "ha_batch_begin": (ctypes.c_int, [P]), "ha_batch_submit": (ctypes.c_int, [P]),
               "ha_batch_dispatch": (ctypes.c_int, [P, P, P, P, U32, U32, U32]),
               "pack_splats": (None, [P, P, ctypes.c_int64])}
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        err = ctypes.create_string_buffer(256)
        self.gpu = lib.ha_gpu_create(err, 256)
        if not self.gpu:
            raise RuntimeError(f"no Vulkan GPU: {err.value.decode()}")
        self.name = lib.ha_gpu_name(self.gpu).decode()
        self.kernels = {}
        for k, (nbuf, push) in self.KERNELS.items():
            getter = getattr(lib, f"splat_spirv_{k}")
            getter.restype, getter.argtypes = P, [P]
            size = U64()
            blob = getter(ctypes.byref(size))
            self.kernels[k] = lib.ha_kernel_create(self.gpu, blob, size.value, nbuf, push)
            if not self.kernels[k]:
                raise RuntimeError(f"could not create kernel {k}")
        self.buffers = {}
        self.n = 0

    # ---- buffers
    def buf(self, name, nbytes):
        """(Re)allocate a named GPU buffer of at least nbytes."""
        old = self.buffers.get(name)
        if old is not None and old[1] >= nbytes:
            return old[0]
        if old is not None:
            self.lib.ha_buffer_destroy(old[0])
        b = self.lib.ha_buffer_create(self.gpu, max(nbytes, 16))
        self.buffers[name] = (b, max(nbytes, 16))
        return b

    def view(self, name, dtype, count):
        b, size = self.buffers[name]
        ptr = self.lib.ha_buffer_data(b)
        return np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(ptr)).view(dtype)[:count]

    def dispatch(self, kernel, bufs, push, groups):
        arr = (P * len(bufs))(*[self.buffers[b][0] for b in bufs])
        pb = ctypes.create_string_buffer(push, len(push)) if push else None
        rc = self.lib.ha_batch_dispatch(self.gpu, self.kernels[kernel], arr, pb, *groups)
        if rc:
            raise RuntimeError(f"dispatch {kernel} failed ({rc})")

    def submit(self):
        rc = self.lib.ha_batch_submit(self.gpu)
        if rc:
            raise RuntimeError(f"GPU submit failed ({rc})")

    # ---- scene
    def upload(self, raw):
        """raw: float32 array (n, 14) in GaussianRaw layout. Packs on the CPU (HA++ code)."""
        raw = np.ascontiguousarray(raw, np.float32)
        self.n = n = len(raw)
        self.buf("splats", n * PACKED_BYTES)
        packed = self.view("splats", np.uint8, n * PACKED_BYTES)
        self.lib.pack_splats(raw.ctypes.data, packed.ctypes.data, n)
        for name, nbytes in (("projs", n * PROJ_BYTES), ("counts", n * 4), ("offsets", n * 4),
                             ("sums", 4 * ((n + 255) // 256)), ("camera", CAMERA_BYTES)):
            self.buf(name, nbytes)

    def scan(self, data, n, sums):
        nb = (n + 255) // 256
        self.dispatch("scan_blocks", [data, sums], struct.pack("<I", n), (nb, 1, 1))
        self.dispatch("scan_sums", [sums], struct.pack("<I", nb), (1, 1, 1))
        self.dispatch("scan_add", [data, sums], struct.pack("<I", n), (nb, 1, 1))

    def render(self, cam, width, height, background=(0, 0, 0)):
        n = self.n
        tiles_x, tiles_y = (width + TILE - 1) // TILE, (height + TILE - 1) // TILE
        self.view("camera", np.uint8, CAMERA_BYTES)[:] = np.frombuffer(
            camera_bytes(cam, width, height, tiles_x, tiles_y), np.uint8)
        # phase A: project + prefix sum of tiles touched
        self.lib.ha_batch_begin(self.gpu)
        self.dispatch("preprocess", ["splats", "camera", "projs", "counts", "offsets"],
                      struct.pack("<I", n), ((n + 127) // 128, 1, 1))
        self.scan("offsets", n, "sums")
        self.submit()
        counts = self.view("counts", np.uint32, n)
        offsets = self.view("offsets", np.uint32, n)
        total = int(offsets[-1]) + int(counts[-1]) if n else 0
        # phase B: bin, sort, tile ranges, render
        ntiles = tiles_x * tiles_y
        nb = max(1, (total + 255) // 256)
        for name, nbytes in (("keys0", total * 4), ("vals0", total * 4), ("keys1", total * 4),
                             ("vals1", total * 4), ("hist", 256 * nb * 4), ("hsums", 4 * nb),
                             ("spans", ntiles * 8), ("image", width * height * 4)):
            self.buf(name, nbytes)
        self.lib.ha_batch_begin(self.gpu)
        if total:
            self.dispatch("bin", ["projs", "offsets", "counts", "keys0", "vals0"],
                          struct.pack("<II", n, tiles_x), ((n + 127) // 128, 1, 1))
            src, dst = ("keys0", "vals0"), ("keys1", "vals1")
            for shift in (0, 8, 16, 24):
                push = struct.pack("<III", total, shift, nb)
                self.dispatch("radix_hist", [src[0], "hist"], push, (nb, 1, 1))
                self.scan("hist", 256 * nb, "hsums")
                self.dispatch("radix_scatter", [src[0], src[1], dst[0], dst[1], "hist"], push, (nb, 1, 1))
                src, dst = dst, src
        self.dispatch("clear", ["spans"], struct.pack("<I", 2 * ntiles), ((2 * ntiles + 255) // 256, 1, 1))
        if total:
            self.dispatch("ranges", ["keys0", "spans"], struct.pack("<I", total), ((total + 255) // 256, 1, 1))
        bg = int(background[0]) | int(background[1]) << 8 | int(background[2]) << 16 | 255 << 24
        self.dispatch("render", ["projs", "vals0", "spans", "image"], struct.pack("<III", width, height, bg),
                      (tiles_x, tiles_y, 1))
        self.submit()
        img = self.view("image", np.uint32, width * height).reshape(height, width).copy()
        return img, total


# ---------------------------------------------------------------- scenes & cameras

def camera_bytes(cam, width, height, tiles_x, tiles_y):
    view = np.asarray(cam["view"], np.float32)
    return (view.T.astype(np.float32).tobytes() +          # column-major mat4
            struct.pack("<ffffffffII", cam["fx"], cam["fy"], cam["cx"], cam["cy"], width, height,
                        cam["near"], cam["far"], tiles_x, tiles_y))


def look_at(eye, target, width, height, fov_deg=60.0, near=0.2, far=100.0):
    """Camera looking from eye to target (camera space: x right, y down, z forward)."""
    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    r = np.stack([right, down, fwd])
    view = np.eye(4)
    view[:3, :3] = r
    view[:3, 3] = -r @ eye
    f = 0.5 * height / math.tan(math.radians(fov_deg) / 2)
    return {"view": view, "fx": f, "fy": f, "cx": width / 2, "cy": height / 2, "near": near, "far": far}


def synthetic_scene(n, seed=0):
    """A colorful spiral galaxy of splats, so the demo works without a .ply file."""
    rng = np.random.default_rng(seed)
    raw = np.zeros((n, RAW_FLOATS), np.float32)
    arm = rng.integers(0, 3, n)
    r = rng.gamma(2.0, 0.6, n)
    theta = r * 1.8 + arm * (2 * math.pi / 3) + rng.normal(0, 0.25, n)
    raw[:, 0] = r * np.cos(theta)
    raw[:, 1] = rng.normal(0, 0.08, n) * (1.0 + 0.5 / (r + 0.3))
    raw[:, 2] = r * np.sin(theta)
    raw[:, 3:6] = np.log(rng.uniform(0.02, 0.09, (n, 3)))
    q = rng.normal(size=(n, 4))
    raw[:, 6:10] = q / np.linalg.norm(q, axis=1, keepdims=True)
    raw[:, 10] = rng.normal(1.5, 1.0, n)
    hue = (theta / (2 * math.pi) + r * 0.15) % 1.0
    rgb = np.stack([0.5 + 0.5 * np.cos(2 * math.pi * (hue + k / 3)) for k in range(3)], 1)
    rgb = rgb * 0.8 + 0.2 * np.exp(-r)[:, None]
    raw[:, 11:14] = (rgb - 0.5) / 0.28209479177387814
    return raw


def load_ply(path):
    """Read a standard 3D Gaussian Splatting .ply (binary little endian)."""
    with open(path, "rb") as f:
        props, count = [], 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(count * 4 * len(props)), np.float32).reshape(count, len(props))
    col = {p: i for i, p in enumerate(props)}
    pick = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
            "f_dc_0", "f_dc_1", "f_dc_2"]
    return np.ascontiguousarray(data[:, [col[p] for p in pick]])


def write_png(path, img_u32):
    h, w = img_u32.shape
    rgba = img_u32.view(np.uint8).reshape(h, w, 4)
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", help="3D Gaussian Splatting .ply file")
    ap.add_argument("--synthetic", type=int, default=20000, help="splats in the demo scene")
    ap.add_argument("--size", default="960x640")
    ap.add_argument("--out", default="splats.png")
    ap.add_argument("--eye", default="0,2.2,-3.2", help="camera position x,y,z")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    raw = load_ply(args.ply) if args.ply else synthetic_scene(args.synthetic)
    pipe = Pipeline(build())
    pipe.upload(raw)
    center = raw[:, :3].mean(0) if args.ply else np.zeros(3)
    eye = np.array([float(v) for v in args.eye.split(",")]) + (center if args.ply else 0)
    import time
    t0 = time.perf_counter()
    img, pairs = pipe.render(look_at(eye, center, w, h), w, h, background=(8, 10, 22))
    dt = time.perf_counter() - t0
    write_png(args.out, img)
    print(f"{len(raw)} splats, {pairs} tile pairs, {w}x{h} on '{pipe.name}' in {dt * 1000:.1f} ms -> {args.out}")


if __name__ == "__main__":
    main()
