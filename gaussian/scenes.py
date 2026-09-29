"""Real 3D Gaussian Splatting scenes: download and decode (PLY, SPZ, .splat).

    python3 gaussian/scenes.py                 # list the scenes
    python3 gaussian/scenes.py racoons         # download one (cached in gaussian/data/)

The scenes are trained captures published in the Babylon.js asset repository
(https://github.com/BabylonJS/Assets, splats/ folder), licensed CC BY 4.0.
They are downloaded on demand and not stored in this repository.

Every loader returns a Scene with numpy arrays in the reference 3DGS convention
(the one used by graphdeco-inria/gaussian-splatting .ply files):
    pos (n,3)            position
    log_scale (n,3)      log of the per-axis standard deviation
    rot (n,4)            quaternion (w, x, y, z), not necessarily normalized
    opacity_logit (n,)   inverse sigmoid of the opacity
    sh (n,k,3)           spherical harmonics, k = (degree+1)^2, sh[:,0] = f_dc
"""
import gzip
import os
import struct
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BASE_URL = "https://raw.githubusercontent.com/BabylonJS/Assets/master/splats/"
SH_C0 = 0.28209479177387814

# name: (file, what it is, the scene's up direction)
SCENES = {
    "racoons": ("racoonfamily.spz", "raccoon family figurines, 932k splats, SH degree 3", (0, -1, 0)),
    "lizard": ("hornedlizard.splat", "horned lizard, 786k splats, base color only", (0, 1, 0)),
    "halo": ("Halo_Believe.ply", "Halo 'Believe' statue, 345k splats, base color only", (0, -1, 0)),
    "firepit": ("gs_Fire_Pit.splat", "fire pit, 562k splats, base color only", (0, -1, 0)),
    "unicorn": ("Unicorn_Stuffy.ply", "unicorn plush toy, 50k splats, SH degree 3", (0, -1, 0)),
    "skull": ("gs_Skull.splat", "skull, 163k splats, base color only", (0, -1, 0)),
}
CREDIT = "Scenes: Babylon.js Assets (github.com/BabylonJS/Assets), CC BY 4.0"


class Scene:
    def __init__(self, pos, log_scale, rot, opacity_logit, sh, name=""):
        self.pos = np.ascontiguousarray(pos, np.float32)
        self.log_scale = np.ascontiguousarray(log_scale, np.float32)
        self.rot = np.ascontiguousarray(rot, np.float32)
        self.opacity_logit = np.ascontiguousarray(opacity_logit, np.float32)
        self.sh = np.ascontiguousarray(sh, np.float32)
        self.name = name

    def __len__(self):
        return len(self.pos)

    @property
    def sh_degree(self):
        return int(round(np.sqrt(self.sh.shape[1]))) - 1

    def raw14(self):
        """The 14 floats per splat of gaussian/splat.ha's GaussianRaw (base color only)."""
        return np.ascontiguousarray(np.concatenate(
            [self.pos, self.log_scale, self.rot, self.opacity_logit[:, None], self.sh[:, 0, :]], 1), np.float32)

    def subset(self, idx):
        return Scene(self.pos[idx], self.log_scale[idx], self.rot[idx], self.opacity_logit[idx], self.sh[idx],
                     self.name)

    def with_degree(self, degree):
        k = (degree + 1) ** 2
        sh = self.sh[:, :k] if self.sh.shape[1] >= k else np.concatenate(
            [self.sh, np.zeros((len(self), k - self.sh.shape[1], 3), np.float32)], 1)
        return Scene(self.pos, self.log_scale, self.rot, self.opacity_logit, sh, self.name)


# ------------------------------------------------------------------ formats
PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "char": "i1",
             "int8": "i1", "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2", "uint": "u4",
             "uint32": "u4", "int": "i4", "int32": "i4"}


def load_ply(path):
    """Standard 3DGS .ply (binary little endian): x y z, f_dc_*, f_rest_*, opacity, scale_*, rot_*."""
    with open(path, "rb") as f:
        props, count, in_vertex = [], 0, False
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                raise ValueError(f"{path}: only binary little-endian PLY is supported")
            if line.startswith("element"):
                in_vertex = line.split()[1] == "vertex"
                if in_vertex:
                    count = int(line.split()[2])
                elif count:
                    raise ValueError(f"{path}: compressed/multi-element PLY is not supported")
            elif line.startswith("property") and in_vertex:
                _, typ, name = line.split()
                props.append((name, "<" + PLY_TYPES[typ]))
            elif line == "end_header":
                break
        v = np.frombuffer(f.read(count * np.dtype(props).itemsize), np.dtype(props), count)
    names = v.dtype.names
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    k = 1 + len(rest) // 3
    sh = np.zeros((count, k, 3), np.float32)
    for c in range(3):
        sh[:, 0, c] = v[f"f_dc_{c}"]
        for j in range(k - 1):                  # f_rest is channel-major: R coefficients, then G, then B
            sh[:, 1 + j, c] = v[rest[c * (k - 1) + j]]
    col = lambda *ns: np.stack([v[n].astype(np.float32) for n in ns], 1)
    return Scene(col("x", "y", "z"), col("scale_0", "scale_1", "scale_2"),
                 col("rot_0", "rot_1", "rot_2", "rot_3"), v["opacity"].astype(np.float32), sh,
                 os.path.basename(path))


def load_splat(path):
    """antimatter15 .splat: 32 bytes = pos f32x3, scale f32x3 (linear), RGBA u8, quaternion u8x4 (w,x,y,z)."""
    d = np.fromfile(path, np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4),
                                    ("rot", "u1", 4)]))
    rgba = d["rgba"].astype(np.float32) / 255.0
    a = np.clip(rgba[:, 3], 1e-6, 1 - 1e-6)
    sh = ((rgba[:, :3] - 0.5) / SH_C0)[:, None, :]
    return Scene(d["pos"], np.log(np.maximum(d["scale"], 1e-12)), (d["rot"].astype(np.float32) - 128.0) / 128.0,
                 np.log(a / (1 - a)), sh, os.path.basename(path))


# SH sign flips for (x, y, z) -> (x, -y, -z), i.e. a 180 degree turn about x, per coefficient in the
# reference basis order (degree 1: y z x; degree 2: xy yz zz xz xx-yy; degree 3: 7 terms)
_FLIP_YZ = np.array([1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1], np.float32)


def load_spz(path):
    """Niantic .spz (gzip): 24-bit fixed-point positions, 8-bit alpha/color/scale/rotation/SH.
    SPZ stores the OpenGL convention (x right, y up, z back); converted here to the .ply one
    (y down, z forward), which is a 180 degree turn about the x axis."""
    raw = gzip.open(path).read()
    magic, version, n, deg, frac, flags, _ = struct.unpack_from("<IIIBBBB", raw, 0)
    if magic != 0x5053474E:
        raise ValueError(f"{path}: not an SPZ file")
    if version != 2:
        raise ValueError(f"{path}: SPZ version {version} is not supported (only version 2)")
    k = (deg + 1) ** 2
    o = 16

    def take(nbytes):
        nonlocal o
        b = np.frombuffer(raw, np.uint8, nbytes, o)
        o += nbytes
        return b

    p = take(n * 9).reshape(n, 3, 3).astype(np.int32)
    fixed = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    pos = fixed.astype(np.float32) / float(1 << frac)
    alpha = take(n).astype(np.float32) / 255.0
    color = take(n * 3).reshape(n, 3).astype(np.float32)
    scale = take(n * 3).reshape(n, 3).astype(np.float32) / 16.0 - 10.0
    xyz = take(n * 3).reshape(n, 3).astype(np.float32) / 127.5 - 1.0     # w >= 0 is implied
    w = np.sqrt(np.maximum(0.0, 1.0 - (xyz * xyz).sum(1)))
    rot = np.concatenate([w[:, None], xyz], 1)
    sh = np.zeros((n, k, 3), np.float32)
    sh[:, 0] = (color / 255.0 - 0.5) / 0.15
    if k > 1:
        sh[:, 1:] = (take(n * (k - 1) * 3).reshape(n, k - 1, 3).astype(np.float32) - 128.0) / 128.0
    a = np.clip(alpha, 1e-6, 1 - 1e-6)
    # OpenGL (RUB) -> PLY (RDF)
    pos = pos * np.array([1, -1, -1], np.float32)
    rot = rot * np.array([1, 1, -1, -1], np.float32)
    sh = sh * _FLIP_YZ[:k, None]
    return Scene(pos, scale, rot, np.log(a / (1 - a)), sh, os.path.basename(path))


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return load_ply(path)
    if ext == ".spz":
        return load_spz(path)
    if ext == ".splat":
        return load_splat(path)
    raise ValueError(f"unknown splat format: {path}")


# ------------------------------------------------------------------ download
def path_of(name):
    if name not in SCENES:
        raise KeyError(f"unknown scene '{name}'; choose from {', '.join(SCENES)}")
    return os.path.join(DATA, SCENES[name][0])


def fetch(name, quiet=False):
    """Download a scene once (cached in gaussian/data/) and return the file path."""
    path = path_of(name)
    if os.path.exists(path):
        return path
    os.makedirs(DATA, exist_ok=True)
    url = BASE_URL + SCENES[name][0]
    if not quiet:
        print(f"downloading {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def scene(name):
    return load(fetch(name))


def up_of(name):
    return SCENES[name][2]


def framing(s, name=None):
    """A robust center and viewing distance (ignores the far-away floaters many captures have)."""
    c = np.median(s.pos, 0)
    lo, hi = np.percentile(s.pos, 10, 0), np.percentile(s.pos, 90, 0)
    return c, float(np.linalg.norm(hi - lo)) * 1.1


def main():
    if len(sys.argv) < 2:
        print(CREDIT)
        for name, (file, what, _) in SCENES.items():
            have = "downloaded" if os.path.exists(path_of(name)) else ""
            print(f"  {name:9s} {file:22s} {what}  {have}")
        return
    for name in sys.argv[1:]:
        s = scene(name)
        print(f"{name}: {len(s)} splats, SH degree {s.sh_degree}, bounds {s.pos.min(0)} .. {s.pos.max(0)}")


if __name__ == "__main__":
    main()
