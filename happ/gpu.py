"""GPU back end: HA++ kernels -> GLSL (-> SPIR-V for Vulkan) and Metal Shading Language.

Both outputs use the HA++ memory layout (C layout, packed vectors), so the
same buffer bytes work on Vulkan (Android, Windows, Linux) and Metal (iPhone, Mac).
  * GLSL: vectors inside buffers/structs are stored as `ha_p_vec3 { float v[3]; }`
    (std430 then matches C exactly); `ha_ld()` / `ha_st_*()` convert.
  * Metal: vectors inside buffers/structs use packed_float3 & co.
Kernel pointer parameters become buffers (binding / [[buffer(i)]] in parameter
order); scalar parameters become push constants (Vulkan) or one constant
struct at the next buffer index (Metal). The C header describes both.
"""

import os
import struct as pystruct
import subprocess

from . import ast as A
from . import types as T
from .errors import HappError

SWIZ = "xyzw"


def kernel_layout(k):
    """Buffer and push-constant layout of a kernel (shared with the bridges)."""
    buffers = [p for p in k.params if p.ty.is_ptr]
    scalars = [p for p in k.params if not p.ty.is_ptr]
    return buffers, scalars


class Emitter:
    def __init__(self, prog, lang):
        self.prog = prog
        self.lang = lang            # 'glsl' | 'msl'
        self.glsl = lang == "glsl"
        self.packed_used = set()    # vec/mat types that appear in memory form
        self.structs_used = set()
        self.exts = set()
        self.tmp = 0
        self.kernel = None
        self.fn = None
        self.helpers = set()
        self.helper_defs = {}       # name -> definition (generated on demand, per type)
        self.override = {}          # id(expr) -> temp name (expressions hoisted out of an assignment)

    # ------------------------------------------------------------ type names
    def tname(self, t):
        """Value-form type name."""
        if t.is_scalar:
            if self.glsl:
                return {"bool": "bool", "i32": "int", "u32": "uint", "f32": "float", "f16": "float16_t"}[t.name]
            return {"bool": "bool", "i32": "int", "u32": "uint", "f32": "float", "f16": "half"}[t.name]
        if t.is_vec:
            if self.glsl:
                return {"f32": "vec", "i32": "ivec", "u32": "uvec", "f16": "f16vec"}[t.elem.name] + str(t.n)
            return self.tname(t.elem) + str(t.n)
        if t.is_mat:
            return f"mat{t.n}" if self.glsl else f"float{t.n}x{t.n}"
        if t.is_struct:
            self.use_struct(t)
            return f"hs_{t.name}"
        if t.is_array:
            if self.glsl:
                # GLSL writes array types outermost size first: [3][2]f32 is float[3][2]
                dims, inner = [], t
                while inner.is_array:
                    dims.append(inner.n)
                    inner = inner.elem
                return self.mname(inner) + "".join(f"[{d}]" for d in dims)
            return f"array<{self.mname(t.elem)}, {t.n}>"
        raise HappError(f"internal: no GPU type for {t}")

    @staticmethod
    def f32_of(t):
        """The f32 type with the shape of an f16 type (f16 math runs in f32 in GLSL)."""
        if t.is_vec:
            return T.Vec(T.F32, t.n)
        return T.F32

    def utname(self, t):
        """Unsigned type with the same shape (for wrap-around math in Metal)."""
        if t.is_vec:
            return self.tname(T.Vec(T.U32, t.n))
        return "uint"

    def mname(self, t):
        """Memory-form type name (inside buffers, structs and arrays)."""
        if t.is_vec or t.is_mat:
            self.packed_used.add(t)
            if t.is_vec and not self.glsl:
                return "packed_" + self.tname(t)
            return f"ha_p_{self.key(t)}"
        return self.tname(t)

    @staticmethod
    def key(t):
        return str(t)

    def use_struct(self, st):
        if st.name in self.structs_used:
            return
        self.structs_used.add(st.name)
        for _, ft in st.fields:
            self.mname(ft)
            inner = ft
            while inner.is_array:
                inner = inner.elem
            if inner.is_struct:
                self.use_struct(inner)
        if any(ft.elem_scalar == T.F16 for _, ft in st.fields):
            self.exts.add("f16")

    def note_type(self, t):
        if t is None:
            return
        if t.elem_scalar == T.F16:
            self.exts.add("f16")
        if t.is_struct:
            self.use_struct(t)
        if t.is_array:
            self.mname(t.elem)
            self.note_type(t.elem)

    # ------------------------------------------------------------ literals
    def flit(self, v, s):
        v = float(v)
        if v != v:
            return "uintBitsToFloat(0x7fc00000u)" if self.glsl else "NAN"
        if v in (float("inf"), float("-inf")):
            if self.glsl:
                return f"uintBitsToFloat({'0x7f800000u' if v > 0 else '0xff800000u'})"
            return "INFINITY" if v > 0 else "(-INFINITY)"
        v32 = pystruct.unpack("<f", pystruct.pack("<f", v))[0] if abs(v) < 3.5e38 else v
        text = "%.9g" % v32
        if "e" not in text and "." not in text and "inf" not in text:
            text += ".0"
        if s == T.F16:
            if self.glsl:
                return f"float16_t({text})"
            return f"({text}h)" if text.startswith("-") else f"{text}h"
        if not self.glsl:
            text += "f"
        return f"({text})" if text.startswith("-") else text

    def ilit(self, v, s):
        v = int(v)
        if s == T.U32:
            return f"{v & 0xffffffff}u"
        if s == T.BOOL:
            return "true" if v else "false"
        if v == -(1 << 31):
            return "(-2147483647-1)"
        return f"({v})" if v < 0 else str(v)

    def lit(self, v, t):
        if t.is_bool:
            return "true" if v else "false"
        if t.is_float:
            return self.flit(v, t)
        return self.ilit(v, t)

    def zero(self, t):
        if t.is_bool:
            return "false"
        if t.is_scalar:
            return self.lit(0, t)
        if t.is_vec or t.is_mat:
            return f"{self.tname(t)}({self.lit(0, t.elem_scalar)})"
        if t.is_struct:
            if not self.glsl:
                return f"{self.tname(t)}{{}}"
            return f"{self.tname(t)}({', '.join(self.zero_mem(ft) for _, ft in t.fields)})"
        if t.is_array:
            if not self.glsl:
                return f"{self.tname(t)}{{}}"
            return f"{self.tname(t)}({', '.join(self.zero_mem(t.elem) for _ in range(t.n))})"
        raise AssertionError(t)

    def zero_mem(self, t):
        if t.is_vec or t.is_mat:
            return self.pack(t, self.zero(t))
        return self.zero(t)

    def pack(self, t, value):
        """Value form -> memory form."""
        if t.is_vec or t.is_mat:
            self.packed_used.add(t)
            return f"ha_st_{self.key(t)}({value})"
        return value

    def unpack(self, t, text):
        if t.is_vec or t.is_mat:
            self.packed_used.add(t)
            return f"ha_ld({text})"
        return text

    # ------------------------------------------------------------ expressions
    def ex(self, e):
        if id(e) in self.override:
            return self.override[id(e)]
        t = e.ty
        self.note_type(t)
        if isinstance(e, A.IntLit):
            return self.lit(e.value, t)
        if isinstance(e, A.FloatLit):
            return self.lit(e.value, t)
        if isinstance(e, A.BoolLit):
            return "true" if e.value else "false"
        if isinstance(e, A.Name):
            sym = e.ref
            if sym.kind == "const":
                return self.lit(sym.value, t)
            if sym.kind == "kernelvar":
                return f"ha_{sym.name}"
            if sym.kind == "param" and self.fn is self.kernel:
                if sym.ty.is_ptr:
                    raise HappError("a buffer can only be indexed (buf[i]) inside a kernel", e.loc)
                return f"ha_p.h_{sym.name}"
            return f"h_{sym.name}"
        if isinstance(e, A.Unary):
            v = self.ex(e.operand)
            if e.op == "-":
                st = e.ty.elem_scalar
                if not self.glsl and st is not None and st.is_int and st.is_signed:
                    return f"{self.tname(e.ty)}(-{self.utname(e.ty)}({v}))"
                return f"(-{v})"
            if e.op == "!":
                return f"(!{v})"
            if e.op == "~":
                return f"(~{v})"
            raise HappError(f"'{e.op}' isn't available on GPUs", e.loc)
        if isinstance(e, A.Binary):
            return self.binary(e)
        if isinstance(e, A.Cast):
            src = e.expr.ty
            v = self.ex(e.expr)
            if src == e.target:
                return v
            return f"{self.tname(e.target)}({v})"
        if isinstance(e, A.Call):
            return self.call(e)
        if isinstance(e, (A.Index, A.Field)):
            text, packed = self.access(e)
            if packed == "mcol":
                mbase, col, _, mt = text
                return f"{self.unpack(mt, mbase)}[{col}]"
            return self.unpack(t, text) if packed else text
        if isinstance(e, A.ArrayLit):
            items = [self.pack(x.ty, self.ex(x)) for x in e.elems]
            if self.glsl:
                return f"{self.tname(t)}({', '.join(items)})"
            return f"{self.tname(t)}{{{', '.join(items)}}}"
        raise HappError("internal: unsupported GPU expression", e.loc)

    def access(self, e):
        """Text for an element/field access, plus whether it is in packed memory form."""
        if isinstance(e, A.Name):
            return self.ex(e), False
        if isinstance(e, A.Index):
            bt = e.base.ty
            idx = self.ex(e.index)
            if bt.is_ptr:
                name = e.base.id
                base = f"ha_b_{name}.data" if self.glsl else f"h_{name}"
                el = bt.pointee
                return f"{base}[{idx}]", (el.is_vec or el.is_mat)
            if bt.is_array:
                base, _ = self.access(e.base) if isinstance(e.base, (A.Name, A.Index, A.Field)) \
                    else (self.ex(e.base), False)
                return f"{base}[{idx}]", (bt.elem.is_vec or bt.elem.is_mat)
            if bt.is_vec:
                base, packed = self.base_text(e.base)
                if packed == "mcol":         # one float of a matrix stored in memory
                    mbase, col, n, _ = base
                    return f"{mbase}.v[({col}) * {n} + ({idx})]", False
                if packed and self.glsl:
                    return f"{base}.v[{idx}]", False
                return f"{base}[{idx}]", False
            if bt.is_mat:
                base, packed = self.base_text(e.base)
                if packed:
                    # a column of a matrix in memory: read as ha_ld(m)[c], written float by float
                    return (base, idx, bt.n, bt), "mcol"
                return f"{base}[{idx}]", False
        if isinstance(e, A.Field):
            if e.kind == "field":
                base, _ = self.base_text(e.base)
                ft = e.ty
                return f"{base}.h_{e.name}", (ft.is_vec or ft.is_mat)
            if e.kind == "swizzle":
                base, packed = self.base_text(e.base)
                if packed == "mcol":
                    mbase, col, n, mt = base
                    if len(e.info) == 1:
                        return f"{mbase}.v[({col}) * {n} + {e.info[0]}]", False
                    return f"{self.unpack(mt, mbase)}[{col}].{''.join(SWIZ[i] for i in e.info)}", False
                if packed:
                    if len(e.info) == 1:
                        return (f"{base}.v[{e.info[0]}]" if self.glsl else f"{base}[{e.info[0]}]"), False
                    return f"{self.unpack(e.base.ty, base)}.{''.join(SWIZ[i] for i in e.info)}", False
                return f"{base}.{''.join(SWIZ[i] for i in e.info)}", False
            raise HappError("'buf.field' needs an index on the GPU; write buf[0].field", e.loc)
        return self.ex(e), False

    def base_text(self, e):
        if isinstance(e, (A.Name, A.Index, A.Field)):
            return self.access(e)
        return f"({self.ex(e)})", False

    def binary(self, e):
        op = e.op
        a, b = self.ex(e.left), self.ex(e.right)
        lt, rt = e.left.ty, e.right.ty
        if op in ("&&", "||"):
            return f"({a} {op} {b})"
        if op in ("==", "!=", "<", ">", "<=", ">="):
            return f"({a} {op} {b})"
        if lt.is_bool and op in ("&", "|", "^"):
            if op == "^":
                return f"({a} != {b})"
            # unlike && and ||, & and | evaluate both sides
            if self.glsl:
                return f"bool(uint({a}) {op} uint({b}))"
            return f"bool({a} {op} {b})"
        s = (lt if not lt.is_mat else rt).elem_scalar
        wrap_signed = not self.glsl and s.is_int and s.is_signed
        if op in ("<<", ">>"):
            amount = f"({b} & {self.lit(s.bits - 1, rt.elem_scalar)})"
            if op == "<<" and wrap_signed:
                # Metal is C++: shifting a negative int left is undefined, so shift the raw bits
                return f"{self.tname(e.ty)}({self.utname(lt)}({a}) << {amount})"
            return f"({a} {op} {amount})"
        if wrap_signed and op in ("+", "-", "*"):
            # HA++ ints wrap on overflow; in Metal (C++) signed overflow is undefined -> use unsigned math
            ua = f"{self.utname(lt)}({a})"
            ub = f"{self.utname(rt)}({b})"
            return f"{self.tname(e.ty)}({ua} {op} {ub})"
        if op == "%" and s.is_float:
            if not self.glsl:
                return f"fmod({a}, {b})"
            rt_ = e.ty
            if s == T.F16:          # GLSL f16 math runs in f32 (see builtin)
                f = self.helper("fmod", self.f32_of(rt_))
                t32 = self.tname(self.f32_of(rt_))
                return f"{self.tname(rt_)}({f}({t32}({a}), {t32}({b})))"
            return f"{self.helper('fmod', rt_)}({self.tname(rt_)}({a}), {self.tname(rt_)}({b}))"
        if op in ("/", "%") and s.is_int:
            return self.int_divide(op, a, b, e)
        return f"({a} {op} {b})"

    def const_int(self, e):
        """Value of an integer constant expression (literal or const), else None."""
        if isinstance(e, A.IntLit):
            return e.value
        if isinstance(e, A.Name) and e.ref is not None and e.ref.kind == "const" and isinstance(e.ref.value, int):
            return e.ref.value
        return None

    def int_divide(self, op, a, b, e):
        """HA++ integer division never faults: x/0 = 0, x%0 = x, MIN/-1 = MIN, MIN%-1 = 0 (as on the CPU)."""
        t = e.ty
        s = t.elem_scalar
        c = self.const_int(e.right)
        if c is not None and c != 0 and not (s.is_signed and c == -1):
            if op == "/" or not s.is_signed or not self.glsl:
                return f"({a} {op} {b})"     # constant divisor: nothing to guard (GLSL % needs a >= 0)
        tn = self.tname(t)
        return f"{self.helper('div' if op == '/' else 'rem', t)}({tn}({a}), {tn}({b}))"

    # ------------------------------------------------------------ calls
    def call(self, e):
        k = e.kind
        args = [a.value for a in e.args]
        if k == "fn":
            return f"hf_{e.name}({', '.join(self.ex(a) for a in args)})"
        if k == "ctor_vec":
            return f"{self.tname(e.target)}({', '.join(self.ex(a) for a in args)})"
        if k == "ctor_mat":
            return f"{self.tname(e.target)}({', '.join(self.ex(a) for a in args)})"
        if k == "ctor_struct":
            st = e.target
            given = {idx: val for idx, val in e.info}
            vals = []
            for i, (_, ft) in enumerate(st.fields):
                vals.append(self.pack(ft, self.ex(given[i])) if i in given else self.zero_mem(ft))
            if self.glsl:
                return f"{self.tname(st)}({', '.join(vals)})"
            return f"{self.tname(st)}{{{', '.join(vals)}}}"
        return self.builtin(e, args)

    SAME = {"sqrt", "exp", "exp2", "log", "log2", "pow", "sin", "cos", "tan", "asin", "acos", "atan",
            "tanh", "floor", "ceil", "trunc", "fract", "abs", "sign", "min", "max", "clamp", "mix",
            "step", "smoothstep", "fma", "dot", "cross", "length", "distance", "normalize", "transpose"}

    F16_VIA_F32 = SAME | {"rsqrt", "atan2", "round", "mod"}

    def builtin(self, e, args):
        name = e.name
        atomic = name.startswith("atomic_")
        xs = [None if (atomic and i == 0) else self.ex(a) for i, a in enumerate(args)]
        tys = [a.ty for a in args]
        t = e.ty
        if self.glsl and name in self.F16_VIA_F32 and any(x.elem_scalar == T.F16 for x in tys + [t]):
            # glslang can't emit GLSL.std.450 math on f16 without extra capabilities that many phones
            # lack; compute in f32 (exact for min/max/abs/floor..., correctly rounded otherwise)
            xs = [f"{self.tname(self.f32_of(ty))}({x})" if ty.elem_scalar == T.F16 else x for x, ty in zip(xs, tys)]
            tys = [self.f32_of(ty) if ty.elem_scalar == T.F16 else ty for ty in tys]
            t32 = self.f32_of(t) if t.elem_scalar == T.F16 else t
            text = self.builtin_text(e, name, xs, tys, t32)
            return f"{self.tname(t)}({text})" if t.elem_scalar == T.F16 else text
        return self.builtin_text(e, name, xs, tys, t)

    def builtin_text(self, e, name, xs, tys, t):
        g = self.glsl
        args = [a.value for a in e.args]
        if name in ("min", "max", "clamp", "mix") and t.is_vec:
            # scalar bounds/weights broadcast
            for i in range(1, len(args)):
                if tys[i].is_scalar:
                    xs[i] = f"{self.tname(t)}({xs[i]})"
        if name == "smoothstep" and t.is_vec:
            for i in range(2):
                if tys[i].is_scalar:
                    xs[i] = f"{self.tname(t)}({xs[i]})"
        if name == "abs" and t.elem_scalar.is_int and not t.elem_scalar.is_signed:
            return xs[0]                       # GLSL/Metal have no abs(uint); it is the identity
        if name == "abs" and not g and t.elem_scalar.is_int:
            self.helpers.add("iabs")           # abs(INT_MIN) must wrap, not be undefined
            return f"ha_iabs({xs[0]})"
        if name == "sign" and not t.elem_scalar.is_signed and not t.elem_scalar.is_float:
            tn = self.tname(t)                 # sign of an unsigned number: 0 or 1 (no sign(uint) in GLSL)
            if t.is_vec:
                return f"{tn}(notEqual({xs[0]}, {tn}(0u)))" if g else f"{tn}({xs[0]} != 0u)"
            return f"{tn}({xs[0]} != 0u)"
        if name == "sign" and not g and not t.elem_scalar.is_float:
            return f"{self.helper('isign', t)}({xs[0]})"     # Metal's sign() is float-only
        if name in self.SAME:
            return f"{name}({', '.join(xs)})"
        if name == "rsqrt":
            return f"{'inversesqrt' if g else 'rsqrt'}({xs[0]})"
        if name == "atan2":
            return f"{'atan' if g else 'atan2'}({xs[0]}, {xs[1]})"
        if name == "round":
            if g:
                return f"{self.helper('round', t)}({xs[0]})"
            return f"round({xs[0]})"
        if name == "mod":
            y = xs[1]
            if t.is_vec and tys[1].is_scalar:
                y = f"{self.tname(t)}({y})"
            if g:
                return f"mod({xs[0]}, {y})"
            return f"{self.helper('mod', t)}({xs[0]}, {y})"
        if name == "select":
            # evaluates both values, like the CPU (a ?: would skip an atomic in the other one)
            if t.is_scalar or t.is_vec:
                c = xs[0]
                if t.is_vec:
                    c = f"{'bvec' if g else 'bool'}{t.n}({c})"
                if g:
                    return f"mix({xs[2]}, {xs[1]}, {c})"
                return f"select({xs[2]}, {xs[1]}, {c})"
            if self.has_effects(args[1]) or self.has_effects(args[2]):
                raise HappError("on the GPU, select() of this type can't contain atomic_*() calls; "
                                "store their results with 'let' first", e.loc)
            return f"({xs[0]} ? {xs[1]} : {xs[2]})"
        if name == "f32_bits":
            return f"floatBitsToUint({xs[0]})" if g else f"as_type<uint>({xs[0]})"
        if name == "f32_from_bits":
            return f"uintBitsToFloat({xs[0]})" if g else f"as_type<float>({xs[0]})"
        if name == "pack_half2":
            return f"packHalf2x16({xs[0]})" if g else f"as_type<uint>(half2({xs[0]}))"
        if name == "unpack_half2":
            return f"unpackHalf2x16({xs[0]})" if g else f"float2(as_type<half2>({xs[0]}))"
        if name == "popcount":
            return f"{self.tname(t)}(bitCount({xs[0]}))" if g else f"popcount({xs[0]})"
        if name == "clz":
            # findMSB of a negative int finds the highest 0 bit, so count on the raw bits (uint)
            return f"{self.tname(t)}(31 - findMSB({self.utname(t)}({xs[0]})))" if g else f"clz({xs[0]})"
        if name == "ctz":
            if g:
                return f"{self.helper('ctz', t)}({xs[0]})"
            return f"ctz({xs[0]})"
        if name in ("atomic_add", "atomic_min", "atomic_max", "atomic_exchange"):
            buf = args[0]
            idx = xs[1]
            if buf.ty.is_ptr:
                place = f"ha_b_{buf.id}.data[{idx}]" if g else f"h_{buf.id}[{idx}]"
                space = "device"
            else:
                place = f"h_{buf.id}[{idx}]"
                space = "threadgroup"
            op = name.split("_")[1]
            if g:
                return f"atomic{op.capitalize()}({place}, {xs[2]})"
            at = "atomic_int" if t.is_signed else "atomic_uint"
            return f"atomic_fetch_{op}_explicit(({space} {at}*)&{place}, {xs[2]}, memory_order_relaxed)" \
                if op != "exchange" else \
                f"atomic_exchange_explicit(({space} {at}*)&{place}, {xs[2]}, memory_order_relaxed)"
        if name == "barrier":
            # glslang emits a workgroup-memory acquire/release fence with barrier()
            return "barrier()" if g else "threadgroup_barrier(mem_flags::mem_threadgroup)"
        if name == "subgroup_add":
            self.exts.add("subgroup")
            return f"subgroupAdd({xs[0]})" if g else f"simd_sum({xs[0]})"
        if name == "subgroup_exclusive_add":
            self.exts.add("subgroup")
            return f"subgroupExclusiveAdd({xs[0]})" if g else f"simd_prefix_exclusive_sum({xs[0]})"
        if name == "subgroup_lane":
            self.exts.add("subgroup")
            return "gl_SubgroupInvocationID" if g else "ha_lane"
        if name == "subgroup_size":
            self.exts.add("subgroup")
            return "gl_SubgroupSize" if g else "ha_simd_size"
        raise HappError(f"'{name}' isn't available on GPUs", e.loc)

    # ------------------------------------------------------------ helper functions
    def helper(self, kind, t):
        """Name of a small function (one per type) that evaluates its arguments once."""
        tn = self.tname(t)
        name = f"ha_{kind}_{tn.replace(' ', '')}"
        if name in self.helper_defs:
            return name
        g = self.glsl
        st = "" if g else "static inline "
        vec = t.is_vec
        n = t.n if vec else 1
        bt = (f"bvec{n}" if g else f"bool{n}") if vec else "bool"
        ut = self.utname(t)

        def pick(cond, yes, no):         # cond ? yes : no, component-wise for vectors
            if not vec:
                return f"({cond} ? {yes} : {no})"
            return f"mix({no}, {yes}, {cond})" if g else f"select({no}, {yes}, {cond})"

        def is_(x, v):
            if not vec:
                return f"{x} == {v}"
            return f"equal({x}, {tn}({v}))" if g else f"{x} == {tn}({v})"

        body = None
        if kind == "round":         # exact half-away-from-zero in any precision
            body = (f"{tn} r = trunc(x); return r + sign(x) * step({tn}(0.5), abs(x - r));", "x")
        elif kind == "fmod":
            body = ("return a - b * trunc(a / b);", "a, b")
        elif kind == "mod":
            body = ("return x - y * floor(x / y);", "x, y")
        elif kind == "ctz":
            lsb = f"{tn}(findLSB(x))"
            body = (f"return {pick(is_('x', '0'), f'{tn}(32)', lsb)};", "x")
        elif kind == "isign":
            body = (f"return {tn}(x > 0) - {tn}(x < 0);", "x")
        elif kind in ("div", "rem"):
            signed = t.elem_scalar.is_signed
            zero, one = ("0", "1") if signed else ("0u", "1u")
            lines = [f"{bt} z = {is_('b', zero)};"]
            safe = pick("z", f"{tn}({one})", "b")
            if signed:
                lines.append(f"{bt} m = {is_('b', '-1')};")
                safe = pick("m", f"{tn}(1)", safe)
            lines.append(f"{tn} s = {safe};")
            if kind == "div":
                lines.append(f"{tn} q = a / s;")
                if signed:
                    lines.append(f"q = {pick('m', f'{tn}(0u - {ut}(a))', 'q')};")
                lines.append(f"return {pick('z', f'{tn}({zero})', 'q')};")
            else:
                # GLSL's % is undefined for negative operands; a - s * (a / s) is exact
                lines.append(f"{tn} r = a - s * (a / s);" if (g and signed) else f"{tn} r = a % s;")
                if signed:
                    lines.append(f"r = {pick('m', f'{tn}(0)', 'r')};")
                lines.append(f"return {pick('z', 'a', 'r')};")
            body = (" ".join(lines), "a, b")
        if body is None:
            raise AssertionError(kind)
        code, names = body
        params = ", ".join(f"{tn} {x.strip()}" for x in names.split(","))
        self.helper_defs[name] = f"{st}{tn} {name}({params}) {{ {code} }}"
        return name

    @staticmethod
    def has_effects(e):
        """True if evaluating e twice (or not at all) would change the result: atomics."""
        if isinstance(e, A.Call):
            if e.name.startswith("atomic_"):
                return True
            return any(Emitter.has_effects(a.value) for a in e.args)
        for kid in ("operand", "left", "right", "expr", "base", "index"):
            sub = getattr(e, kid, None)
            if isinstance(sub, A.Node) and Emitter.has_effects(sub):
                return True
        if isinstance(e, A.ArrayLit):
            return any(Emitter.has_effects(x) for x in e.elems)
        return False

    # ------------------------------------------------------------ statements
    def block(self, b, ind):
        out = []
        for s in b.stmts:
            out.extend(self.stmt(s, ind))
        return out

    def stmt(self, s, ind):
        p = "    " * ind
        if isinstance(s, A.Let):
            self.note_type(s.ty)
            if s.value is not None:
                return [f"{p}{self.decl(s.ty, 'h_' + s.name)} = {self.ex(s.value)};"]
            return [f"{p}{self.decl(s.ty, 'h_' + s.name)} = {self.zero(s.ty)};"]
        if isinstance(s, A.Shared):
            return []   # declared at kernel scope
        if isinstance(s, A.Assign):
            return [p + x for x in self.assign(s)]
        if isinstance(s, A.If):
            out = [f"{p}if ({self.cond(self.ex(s.cond))}) {{"] + self.block(s.then, ind + 1)
            if s.els is None:
                return out + [f"{p}}}"]
            if isinstance(s.els, A.If):
                sub = self.stmt(s.els, ind)
                return out + [f"{p}}} else {sub[0].lstrip()}"] + sub[1:]
            return out + [f"{p}}} else {{"] + self.block(s.els, ind + 1) + [f"{p}}}"]
        if isinstance(s, A.While):
            return [f"{p}while ({self.cond(self.ex(s.cond))}) {{"] + self.block(s.body, ind + 1) + [f"{p}}}"]
        if isinstance(s, A.For):
            self.tmp += 1
            end = f"ha_end{self.tmp}"
            ty = self.tname(s.ty)
            return ([f"{p}{{", f"{p}    {ty} {end} = {self.ex(s.end)};",
                     f"{p}    for ({ty} h_{s.var} = {self.ex(s.start)}; h_{s.var} < {end}; h_{s.var}++) {{"]
                    + self.block(s.body, ind + 2) + [f"{p}    }}", f"{p}}}"])
        if isinstance(s, A.Return):
            if s.value is None:
                return [f"{p}return;"]
            return [f"{p}return {self.ex(s.value)};"]
        if isinstance(s, A.Break):
            return [f"{p}break;"]
        if isinstance(s, A.Continue):
            return [f"{p}continue;"]
        if isinstance(s, A.ExprStmt):
            return [f"{p}{self.ex(s.expr)};"]
        if isinstance(s, A.Block):
            return [f"{p}{{"] + self.block(s, ind + 1) + [f"{p}}}"]
        raise AssertionError(s)

    @staticmethod
    def cond(text):
        """Drop one redundant pair of parentheses: if ((a == b)) -> if (a == b)."""
        if text.startswith("(") and text.endswith(")"):
            depth = 0
            for i, ch in enumerate(text):
                depth += ch == "("
                depth -= ch == ")"
                if depth == 0 and i < len(text) - 1:
                    return text
            return text[1:-1]
        return text

    def decl(self, t, name):
        return f"{self.tname(t)} {name}"

    def assign(self, s):
        """Assignment lines. The place is evaluated once even when it is written in several steps."""
        tgt = s.target
        pre = []
        if self.has_effects(tgt):
            # e.g. out[atomic_add(count, 0, 1)] += x: compute the index once, into a temporary
            chain, node = [], tgt
            while isinstance(node, (A.Index, A.Field)):
                if isinstance(node, A.Index):
                    chain.append(node)
                node = node.base
            for ix in reversed(chain):
                if self.has_effects(ix.index):
                    self.tmp += 1
                    tmp = f"ha_x{self.tmp}"
                    pre.append(f"{self.tname(ix.index.ty)} {tmp} = {self.ex(ix.index)};")
                    self.override[id(ix.index)] = tmp
        try:
            lines = self.assign_place(s, tgt)
        finally:
            for ix in list(self.override):
                del self.override[ix]
        if pre:
            return ["{"] + ["    " + x for x in pre + lines] + ["}"]
        return lines

    def assign_place(self, s, tgt):
        if s.op == "=":
            value = self.ex(s.value)
        else:
            op = s.op[:-1]
            fake = A.Binary(op, tgt, s.value, s.loc)
            fake.ty = tgt.ty
            value = self.binary(fake)
        t = tgt.ty
        if isinstance(tgt, A.Field) and tgt.kind == "swizzle" and len(tgt.info) > 1:
            base, packed = self.base_text(tgt.base)
            if packed:
                self.tmp += 1
                tv, ti = f"ha_t{self.tmp}", f"ha_i{self.tmp}"
                lines = [f"{{ {self.tname(t)} {tv} = {value};"]
                if packed == "mcol":             # some floats of one column of a matrix in memory
                    mbase, col, n, _ = base
                    lines[0] += f" int {ti} = int({col});"
                    dsts = [f"{mbase}.v[{ti} * {n} + {lane}]" for lane in tgt.info]
                else:
                    dsts = [f"{base}.v[{lane}]" if self.glsl else f"{base}[{lane}]" for lane in tgt.info]
                for k, dst in enumerate(dsts):
                    lines.append(f"  {dst} = {tv}.{SWIZ[k]};")
                return lines + ["}"]
        text, packed = self.access(tgt)
        if packed == "mcol":                     # a whole column of a matrix in memory
            mbase, col, n, _ = text
            self.tmp += 1
            tv, ti = f"ha_t{self.tmp}", f"ha_i{self.tmp}"
            lines = [f"{{ {self.tname(t)} {tv} = {value}; int {ti} = int({col});"]
            for r in range(n):
                lines.append(f"  {mbase}.v[{ti} * {n} + {r}] = {tv}[{r}];")
            return lines + ["}"]
        if packed:
            return [f"{text} = {self.pack(t, value)};"]
        return [f"{text} = {value};"]

    # ------------------------------------------------------------ functions
    def helper_fns(self, kernel):
        """Functions reachable from a kernel, callees first."""
        order, seen = [], set()

        def walk_expr(e):
            if isinstance(e, A.Call):
                if e.kind == "fn" and e.target.name not in seen:
                    seen.add(e.target.name)
                    visit(e.target)
                    order.append(e.target)
                for a in e.args:
                    walk_expr(a.value)
            elif isinstance(e, A.Unary):
                walk_expr(e.operand)
            elif isinstance(e, A.Binary):
                walk_expr(e.left)
                walk_expr(e.right)
            elif isinstance(e, A.Cast):
                walk_expr(e.expr)
            elif isinstance(e, A.Index):
                walk_expr(e.base)
                walk_expr(e.index)
            elif isinstance(e, A.Field):
                walk_expr(e.base)
            elif isinstance(e, A.ArrayLit):
                for x in e.elems:
                    walk_expr(x)

        def walk(s):
            if isinstance(s, A.Block):
                for x in s.stmts:
                    walk(x)
            elif isinstance(s, A.Let):
                if s.value is not None:
                    walk_expr(s.value)
            elif isinstance(s, A.Assign):
                walk_expr(s.target)
                walk_expr(s.value)
            elif isinstance(s, A.If):
                walk_expr(s.cond)
                walk(s.then)
                if s.els is not None:
                    walk(s.els)
            elif isinstance(s, A.While):
                walk_expr(s.cond)
                walk(s.body)
            elif isinstance(s, A.For):
                walk_expr(s.start)
                walk_expr(s.end)
                walk(s.body)
            elif isinstance(s, A.Return):
                if s.value is not None:
                    walk_expr(s.value)
            elif isinstance(s, A.ExprStmt):
                walk_expr(s.expr)

        def visit(fn):
            walk(fn.body)

        visit(kernel)
        return order

    def function(self, fn):
        self.fn = fn
        self.note_type(fn.ret_ty)
        ret = "void" if fn.ret_ty.is_void else self.tname(fn.ret_ty)
        params = []
        for p in fn.params:
            self.note_type(p.ty)
            params.append(self.decl(p.ty, "h_" + p.name))
        body = self.block(fn.body, 1)
        prefix = "" if self.glsl else "static "
        return [f"{prefix}{ret} hf_{fn.name}({', '.join(params)}) {{"] + body + ["}", ""]

    def shared_decls(self, kernel, ind):
        p = "    " * ind
        out = []
        for s in kernel.body.stmts:
            if isinstance(s, A.Shared):
                self.note_type(s.ty)
                q = "shared" if self.glsl else "threadgroup"
                out.append(f"{p}{q} {self.tname(s.ty.elem)} h_{s.name}[{s.ty.n}];")
        return out

    # ------------------------------------------------------------ assembling
    def packed_defs(self):
        out = []
        for t in sorted(self.packed_used, key=str):
            key = self.key(t)
            vt = self.tname(t)
            if t.is_vec:
                et = self.tname(t.elem)
                if self.glsl:
                    comps = ", ".join(f"p.v[{i}]" for i in range(t.n))
                    out += [f"struct ha_p_{key} {{ {et} v[{t.n}]; }};",
                            f"{vt} ha_ld(ha_p_{key} p) {{ return {vt}({comps}); }}",
                            f"ha_p_{key} ha_st_{key}({vt} x) {{ ha_p_{key} p; "
                            + " ".join(f"p.v[{i}] = x.{SWIZ[i]};" for i in range(t.n)) + " return p; }"]
                else:
                    out += [f"static inline {vt} ha_ld(packed_{vt} p) {{ return {vt}(p); }}",
                            f"static inline packed_{vt} ha_st_{key}({vt} x) {{ return packed_{vt}(x); }}"]
            else:
                n = t.n
                cols = ", ".join(("vec" if self.glsl else "float") + f"{n}(" +
                                 ", ".join(f"p.v[{c * n + r}]" for r in range(n)) + ")" for c in range(n))
                st = "" if self.glsl else "static inline "
                out += [f"struct ha_p_{key} {{ float v[{n * n}]; }};",
                        f"{st}{vt} ha_ld(ha_p_{key} p) {{ return {vt}({cols}); }}",
                        f"{st}ha_p_{key} ha_st_{key}({vt} m) {{ ha_p_{key} p; "
                        f"for (int c = 0; c < {n}; c++) {{ for (int r = 0; r < {n}; r++) {{ "
                        f"p.v[c * {n} + r] = m[c][r]; }} }} return p; }}"]
        return out

    def struct_defs(self):
        out = []
        for st in self.prog.structs:
            if st.name not in self.structs_used:
                continue
            out.append(f"struct hs_{st.name} {{")
            for fname, ft in st.fields:
                if ft.is_array:
                    out.append(f"    {self.mname(ft.elem)} h_{fname}[{ft.n}];" if self.glsl else
                               f"    {self.tname(ft)} h_{fname};")
                else:
                    out.append(f"    {self.mname(ft)} h_{fname};")
            out += ["};", ""]
        return out


# ====================================================================== GLSL

def glsl_kernel(prog, k):
    em = Emitter(prog, "glsl")
    em.kernel = k
    buffers, scalars = kernel_layout(k)
    helpers = em.helper_fns(k)
    fn_lines = []
    for fn in helpers:
        fn_lines += em.function(fn)
    em.fn = k
    shared = em.shared_decls(k, 0)
    body = em.block(k.body, 1)
    buf_lines = []
    for i, p in enumerate(buffers):
        el = p.ty.pointee
        if el.is_array:
            raise HappError(f"buffer '{p.name}': pointers to arrays aren't supported on GPUs; "
                            f"use a struct", p.loc)
        em.note_type(el)
        buf_lines.append(f"layout(std430, set = 0, binding = {i}) buffer ha_B_{p.name} "
                         f"{{ {em.mname(el)} data[]; }} ha_b_{p.name};")
    if scalars:
        fields = " ".join(f"{em.tname(p.ty)} h_{p.name};" for p in scalars)
        buf_lines.append(f"layout(push_constant) uniform ha_Params {{ {fields} }} ha_p;")
    wg = k.sig["workgroup"]
    head = ["#version 450",
            f"// HA++ kernel '{k.name}' (generated; edit the .ha source instead)"]
    ext_at = len(head)
    if "subgroup" in em.exts:
        head += ["#extension GL_KHR_shader_subgroup_basic : require",
                 "#extension GL_KHR_shader_subgroup_arithmetic : require"]
    head.append(f"layout(local_size_x = {wg[0]}, local_size_y = {wg[1]}, local_size_z = {wg[2]}) in;")
    packed = em.packed_defs()
    structs = em.struct_defs()
    # struct defs may have added packed types
    packed = em.packed_defs()
    main = ["void main() {",
            "    uvec3 ha_global_id = gl_GlobalInvocationID;",
            "    uvec3 ha_local_id = gl_LocalInvocationID;",
            "    uvec3 ha_group_id = gl_WorkGroupID;",
            "    uvec3 ha_num_groups = gl_NumWorkGroups;",
            "    uvec3 ha_group_size = gl_WorkGroupSize;"] + body + ["}"]
    helpers = list(em.helper_defs.values())
    parts = head + [""] + packed + [""] + structs + buf_lines + shared + [""] + helpers + fn_lines + main
    text = "\n".join(parts) + "\n"
    if "f16" in em.exts or "float16_t" in text or "f16vec" in text:
        # decided from the finished text, so no path that produces a 16-bit type can miss it
        head[ext_at:ext_at] = ["#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require",
                               "#extension GL_EXT_shader_16bit_storage : require"]
        parts = head + [""] + packed + [""] + structs + buf_lines + shared + [""] + helpers + fn_lines + main
        text = "\n".join(parts) + "\n"
    return text


def generate_glsl(prog):
    return {k.name: glsl_kernel(prog, k) for k in prog.kernels}


# ====================================================================== Metal

def generate_metal(prog):
    em = Emitter(prog, "msl")
    fn_lines, kernel_lines, param_structs = [], [], []
    done = set()
    for k in prog.kernels:
        em.kernel = k
        for fn in em.helper_fns(k):
            if fn.name not in done:
                done.add(fn.name)
                fn_lines += em.function(fn)
        em.fn = k
        buffers, scalars = kernel_layout(k)
        params = []
        for i, p in enumerate(buffers):
            el = p.ty.pointee
            if el.is_array:
                raise HappError(f"buffer '{p.name}': pointers to arrays aren't supported on GPUs; "
                                f"use a struct", p.loc)
            em.note_type(el)
            params.append(f"device {em.mname(el)}* h_{p.name} [[buffer({i})]]")
        if scalars:
            fields = " ".join(f"{em.tname(p.ty)} h_{p.name};" for p in scalars)
            param_structs.append(f"struct ha_params_{k.name} {{ {fields} }};")
            params.append(f"constant ha_params_{k.name}& ha_p [[buffer({len(buffers)})]]")
        params += ["uint3 ha_global_id [[thread_position_in_grid]]",
                   "uint3 ha_local_id [[thread_position_in_threadgroup]]",
                   "uint3 ha_group_id [[threadgroup_position_in_grid]]",
                   "uint3 ha_num_groups [[threadgroups_per_grid]]",
                   "uint3 ha_group_size [[threads_per_threadgroup]]"]
        if "subgroup" in k.uses:
            params += ["uint ha_lane [[thread_index_in_simdgroup]]",
                       "uint ha_simd_size [[threads_per_simdgroup]]"]
        shared = em.shared_decls(k, 1)
        body = em.block(k.body, 1)
        wg = k.sig["workgroup"]
        kernel_lines += [f"// workgroup size {wg[0]}x{wg[1]}x{wg[2]}: dispatch with this threadsPerThreadgroup",
                         f"kernel void {k.name}(" + ",\n    ".join(params) + ") {"] + shared + body + ["}", ""]
    em.struct_defs()
    packed = em.packed_defs()
    structs = em.struct_defs()
    packed = em.packed_defs()
    head = ["// HA++ Metal kernels (generated; edit the .ha source instead)",
            "#include <metal_stdlib>", "using namespace metal;", ""]
    helpers = []
    if "iabs" in em.helpers:
        helpers.append("static inline int ha_iabs(int x) { return x < 0 ? int(-uint(x)) : x; }")
        for n in (2, 3, 4):
            helpers.append(f"static inline int{n} ha_iabs(int{n} x) {{ return select(x, int{n}(-uint{n}(x)), x < 0); }}")
    helpers += list(em.helper_defs.values())
    parts = head + helpers + packed + [""] + structs + param_structs + [""] + fn_lines + kernel_lines
    return "\n".join(parts) + "\n"


# ====================================================================== build step

def build_gpu(prog, tc, out_dir, name, log):
    os.makedirs(out_dir, exist_ok=True)
    files, spirv = {}, {}
    glslang = tc.find("glslangValidator")
    spirv_val = tc.find("spirv-val")
    for kname, src in generate_glsl(prog).items():
        comp = os.path.join(out_dir, f"{name}_{kname}.comp")
        with open(comp, "w") as f:
            f.write(src)
        files[f"glsl:{kname}"] = comp
        if glslang is None:
            continue
        spv = os.path.join(out_dir, f"{name}_{kname}.spv")
        r = subprocess.run([glslang, "-V", "--target-env", "vulkan1.1", "-o", spv, comp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise HappError(f"internal: GLSL for kernel '{kname}' did not compile "
                            f"(please report):\n{r.stdout}{r.stderr}")
        if spirv_val:
            r = subprocess.run([spirv_val, "--target-env", "vulkan1.1", spv], capture_output=True, text=True)
            if r.returncode != 0:
                raise HappError(f"internal: invalid SPIR-V for kernel '{kname}':\n{r.stdout}{r.stderr}")
        with open(spv, "rb") as f:
            spirv[kname] = f.read()
        files[f"spirv:{kname}"] = spv
    if glslang is None:
        log("  note: glslangValidator not found; wrote GLSL but no SPIR-V (install the Vulkan SDK "
            "or 'apt install glslang-tools')")
    metal = generate_metal(prog)
    mpath = os.path.join(out_dir, f"{name}.metal")
    with open(mpath, "w") as f:
        f.write(metal)
    files["metal"] = mpath
    log(f"  built gpu            -> {len(spirv)} SPIR-V kernel(s) + {os.path.basename(mpath)}")
    return {"files": files, "spirv": spirv, "metal": metal}
