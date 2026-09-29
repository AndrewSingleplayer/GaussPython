"""The HA++ type system.

Memory layout rule (the same on CPU, Vulkan and Metal): every type is laid out
like the matching C type, and vectors/matrices are tightly packed floats
(vec3 = 12 bytes, align 4; mat4 = 16 floats column-major). So one buffer of
splats has the same bytes on Android, iPhone, PC and every GPU.
"""


class Type:
    kind = "?"

    def __eq__(self, other):
        return isinstance(other, Type) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return str(self)

    # classification helpers
    @property
    def is_scalar(self):
        return isinstance(self, Scalar)

    @property
    def is_int(self):
        return isinstance(self, Scalar) and self.cls in ("int", "uint")

    @property
    def is_signed(self):
        return isinstance(self, Scalar) and self.cls == "int"

    @property
    def is_float(self):
        return isinstance(self, Scalar) and self.cls == "float"

    @property
    def is_bool(self):
        return isinstance(self, Scalar) and self.cls == "bool"

    @property
    def is_numeric(self):
        return isinstance(self, Scalar) and self.cls != "bool"

    @property
    def is_vec(self):
        return isinstance(self, Vec)

    @property
    def is_mat(self):
        return isinstance(self, Mat)

    @property
    def is_ptr(self):
        return isinstance(self, Ptr)

    @property
    def is_struct(self):
        return isinstance(self, Struct)

    @property
    def is_array(self):
        return isinstance(self, Array)

    @property
    def is_void(self):
        return isinstance(self, Void)

    @property
    def in_memory(self):
        """Structs and arrays are handled by address, everything else by value."""
        return isinstance(self, (Struct, Array))

    @property
    def elem_scalar(self):
        """Scalar type of a scalar or vector (for arithmetic rules)."""
        if isinstance(self, Scalar):
            return self
        if isinstance(self, Vec):
            return self.elem
        if isinstance(self, Mat):
            return F32
        return None


class Scalar(Type):
    kind = "scalar"

    def __init__(self, name, cls, bits):
        self.name, self.cls, self.bits = name, cls, bits

    def key(self):
        return ("s", self.name)

    def __str__(self):
        return self.name

    def int_range(self):
        if self.cls == "int":
            return -(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1
        return 0, (1 << self.bits) - 1


class Vec(Type):
    kind = "vec"

    def __init__(self, elem, n):
        self.elem, self.n = elem, n

    def key(self):
        return ("v", self.elem.name, self.n)

    def __str__(self):
        return VEC_PREFIX[self.elem.name] + str(self.n)


class Mat(Type):
    """Square float matrix, column-major like GLSL and Metal."""
    kind = "mat"

    def __init__(self, n):
        self.n = n

    def key(self):
        return ("m", self.n)

    def __str__(self):
        return f"mat{self.n}"

    @property
    def col(self):
        return Vec(F32, self.n)


class Ptr(Type):
    kind = "ptr"

    def __init__(self, pointee):
        self.pointee = pointee

    def key(self):
        return ("p", self.pointee.key())

    def __str__(self):
        return f"*{self.pointee}"


class Array(Type):
    kind = "array"

    def __init__(self, elem, n):
        self.elem, self.n = elem, n

    def key(self):
        return ("a", self.elem.key(), self.n)

    def __str__(self):
        return f"[{self.n}]{self.elem}"


class Struct(Type):
    kind = "struct"

    def __init__(self, name):
        self.name = name
        self.fields = []        # [(name, Type)]
        self.field_index = {}
        self.loc = None

    def key(self):
        return ("S", self.name)

    def __str__(self):
        return self.name


class Void(Type):
    kind = "void"

    def key(self):
        return ("void",)

    def __str__(self):
        return "void"


class Str(Type):
    """String literals; only usable as print() arguments."""
    kind = "str"

    def key(self):
        return ("str",)

    def __str__(self):
        return "str"


BOOL = Scalar("bool", "bool", 1)
I8, I16, I32, I64 = (Scalar(f"i{b}", "int", b) for b in (8, 16, 32, 64))
U8, U16, U32, U64 = (Scalar(f"u{b}", "uint", b) for b in (8, 16, 32, 64))
F16, F32, F64 = (Scalar(f"f{b}", "float", b) for b in (16, 32, 64))
VOID = Void()
STR = Str()

VEC_PREFIX = {"f32": "vec", "i32": "ivec", "u32": "uvec", "f16": "hvec"}

NAMED_TYPES = {s.name: s for s in (BOOL, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64)}
for _elem in (F32, I32, U32, F16):
    for _n in (2, 3, 4):
        NAMED_TYPES[VEC_PREFIX[_elem.name] + str(_n)] = Vec(_elem, _n)
for _n in (2, 3, 4):
    NAMED_TYPES[f"mat{_n}"] = Mat(_n)


def size_align(t):
    """C-compatible size and alignment in bytes (identical on all 64-bit targets)."""
    if isinstance(t, Scalar):
        if t.cls == "bool":
            return 1, 1
        b = t.bits // 8
        return b, b
    if isinstance(t, Vec):
        es, ea = size_align(t.elem)
        return es * t.n, ea
    if isinstance(t, Mat):
        return 4 * t.n * t.n, 4
    if isinstance(t, Ptr):
        return 8, 8
    if isinstance(t, Array):
        es, ea = size_align(t.elem)
        return es * t.n, ea
    if isinstance(t, Struct):
        off, align = 0, 1
        for _, ft in t.fields:
            fs, fa = size_align(ft)
            off = (off + fa - 1) // fa * fa
            off += fs
            align = max(align, fa)
        return (off + align - 1) // align * align, align
    raise TypeError(f"no size for {t}")


def field_offsets(t):
    offs, off = [], 0
    for _, ft in t.fields:
        fs, fa = size_align(ft)
        off = (off + fa - 1) // fa * fa
        offs.append(off)
        off += fs
    return offs


SWIZZLE_SETS = ("xyzw", "rgba")


def parse_swizzle(name, n):
    """Return the lane list for a swizzle like 'xyz' on an n-wide vector, or None."""
    if not 1 <= len(name) <= 4:
        return None
    for s in SWIZZLE_SETS:
        if all(c in s for c in name):
            lanes = [s.index(c) for c in name]
            if all(l < n for l in lanes):
                return lanes
            return None
    return None
