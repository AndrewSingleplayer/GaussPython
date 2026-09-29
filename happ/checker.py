"""Name resolution and type checking.

Produces a `Program`: every expression gets a `.ty`, every name a `.ref`, and
calls are classified (user function, builtin, constructor). Both back ends
(LLVM for CPUs, GLSL/Metal for GPUs) work only from this checked tree.
"""

import difflib

from . import ast as A
from . import types as T
from .errors import HappError

# ---------------------------------------------------------------- builtins

FLOAT_UNARY = {"sqrt", "rsqrt", "exp", "exp2", "log", "log2", "sin", "cos", "tan",
               "asin", "acos", "atan", "tanh", "floor", "ceil", "round", "trunc", "fract"}
TRANSCENDENTAL = {"exp", "exp2", "log", "log2", "sin", "cos", "tan", "asin", "acos",
                  "atan", "tanh", "pow", "atan2"}
FLOAT_BINARY = {"pow", "atan2", "mod", "step"}
BIT_FUNCS = {"f32_bits": (T.F32, T.U32), "f32_from_bits": (T.U32, T.F32),
             "f16_bits": (T.F16, T.U16), "f16_from_bits": (T.U16, T.F16),
             "f64_bits": (T.F64, T.U64), "f64_from_bits": (T.U64, T.F64)}
INT_UNARY = {"popcount", "clz", "ctz"}
ATOMICS = {"atomic_add", "atomic_min", "atomic_max", "atomic_exchange"}
SUBGROUP = {"subgroup_add", "subgroup_exclusive_add", "subgroup_lane", "subgroup_size"}
BUILTIN_NAMES = (FLOAT_UNARY | FLOAT_BINARY | set(BIT_FUNCS) | INT_UNARY | ATOMICS | SUBGROUP |
                 {"abs", "sign", "min", "max", "clamp", "mix", "smoothstep", "fma", "dot",
                  "cross", "length", "distance", "normalize", "transpose", "select", "print",
                  "barrier", "pack_half2", "unpack_half2"})
KERNEL_VARS = {"global_id", "local_id", "group_id", "num_groups", "group_size"}

# std library function that implements each builtin on CPUs (f32 versions)
STD_IMPL = {"exp": "_std_exp", "exp2": "_std_exp2", "log": "_std_log", "log2": "_std_log2",
            "sin": "_std_sin", "cos": "_std_cos", "tan": "_std_tan", "asin": "_std_asin",
            "acos": "_std_acos", "atan": "_std_atan", "tanh": "_std_tanh", "pow": "_std_pow",
            "atan2": "_std_atan2"}

GPU_SCALARS = {"i32", "u32", "f32", "f16", "bool"}
RESERVED = {"main", "kernel", "device", "constant", "thread", "threadgroup", "void"}


class Symbol:
    __slots__ = ("name", "kind", "ty", "mutable", "node", "id", "value", "untyped")
    _next = 0

    def __init__(self, name, kind, ty, mutable=False, node=None, value=None, untyped=None):
        self.name, self.kind, self.ty, self.mutable, self.node = name, kind, ty, mutable, node
        self.value, self.untyped = value, untyped
        Symbol._next += 1
        self.id = Symbol._next


class Program:
    def __init__(self):
        self.structs = []       # dependency order
        self.consts = {}
        self.fns = {}           # name -> FnDef (incl. std, extern, kernels)
        self.kernels = []
        self.exports = []
        self.externs = []
        self.main = None
        self.cpu_kernels = []

    def reachable(self, roots):
        """Functions reachable from `roots` (names), following calls and builtin impls."""
        seen, order, stack = set(), [], list(roots)
        while stack:
            name = stack.pop()
            if name in seen or name not in self.fns:
                continue
            seen.add(name)
            fn = self.fns[name]
            order.append(fn)
            stack.extend(sorted(fn.sig["calls"]))
        return order


class Checker:
    def __init__(self, modules):
        self.modules = modules
        self.prog = Program()
        self.structs = {}
        self.const_nodes = {}
        self.const_state = {}
        self.scopes = []
        self.fn = None
        self.loops = 0

    # ------------------------------------------------------------ entry
    def run(self):
        # user definitions win over std-library definitions with the same name
        user_names = {it.name for m in self.modules if not m.is_std for it in m.items
                      if isinstance(it, (A.StructDef, A.ConstDef, A.FnDef))}
        items = [(m, it) for m in self.modules for it in m.items
                 if not (m.is_std and getattr(it, "name", None) in user_names)]
        for m, it in items:
            if isinstance(it, A.StructDef):
                self.declare_global(it.name, it.loc)
                st = T.Struct(it.name)
                st.loc = it.loc
                it.ty = st
                self.structs[it.name] = (st, it)
        for m, it in items:
            if isinstance(it, A.ConstDef):
                self.declare_global(it.name, it.loc)
                self.const_nodes[it.name] = it
        for m, it in items:
            if isinstance(it, A.FnDef):
                self.declare_global(it.name, it.loc)
                it.is_std = m.is_std
                self.prog.fns[it.name] = it
        for st, node in self.structs.values():
            self.resolve_struct(st, node)
        self.order_structs()
        for name in self.const_nodes:
            self.eval_const_def(name)
        for fn in self.prog.fns.values():
            self.check_signature(fn)
        for fn in self.prog.fns.values():
            if fn.body is not None:
                self.check_body(fn)
        for fn in self.prog.fns.values():
            if fn.is_kernel:
                self.check_gpu(fn)
        # kernels without barriers/shared memory/subgroups also run on CPUs as <name>_cpu,
        # unless the program defines its own <name>_cpu
        self.prog.cpu_kernels = [k for k in self.prog.kernels
                                 if not k.uses & {"barrier", "shared", "subgroup"}
                                 and f"{k.name}_cpu" not in self.prog.fns]
        return self.prog

    def declare_global(self, name, loc):
        if name in T.NAMED_TYPES or name in BUILTIN_NAMES or name in KERNEL_VARS:
            raise HappError(f"'{name}' is a built-in name and can't be redefined", loc)
        if name in self.structs or name in self.const_nodes or name in self.prog.fns:
            raise HappError(f"'{name}' is defined twice", loc)

    # ------------------------------------------------------------ types
    def resolve_type(self, te):
        if isinstance(te, A.TName):
            if te.name in T.NAMED_TYPES:
                return T.NAMED_TYPES[te.name]
            if te.name in self.structs:
                return self.structs[te.name][0]
            close = difflib.get_close_matches(te.name, list(T.NAMED_TYPES) + list(self.structs), 1)
            raise HappError(f"unknown type '{te.name}'", te.loc,
                            f"did you mean '{close[0]}'?" if close else None)
        if isinstance(te, A.TPtr):
            return T.Ptr(self.resolve_type(te.inner))
        if isinstance(te, A.TArray):
            n, nty = self.const_eval(te.size)
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                raise HappError("array size must be a positive integer constant", te.size.loc)
            return T.Array(self.resolve_type(te.inner), n)
        raise AssertionError(te)

    def resolve_struct(self, st, node):
        seen = set()
        for fname, fte, floc in node.fields:
            if fname in seen:
                raise HappError(f"field '{fname}' appears twice", floc)
            seen.add(fname)
            ft = self.resolve_type(fte)
            if ft.is_bool:
                ft = T.BOOL
            st.field_index[fname] = len(st.fields)
            st.fields.append((fname, ft))
        if not st.fields:
            raise HappError(f"struct '{st.name}' has no fields", node.loc)

    def order_structs(self):
        state = {}

        def visit(st, path):
            s = state.get(st.name)
            if s == 2:
                return
            if s == 1:
                raise HappError(f"struct '{st.name}' contains itself; use a pointer (*{st.name})",
                                st.loc)
            state[st.name] = 1
            for _, ft in st.fields:
                inner = ft
                while isinstance(inner, T.Array):
                    inner = inner.elem
                if isinstance(inner, T.Struct):
                    visit(inner, path + [st.name])
            state[st.name] = 2
            self.prog.structs.append(st)

        for st, _ in self.structs.values():
            visit(st, [])

    # ------------------------------------------------------------ constants
    def eval_const_def(self, name):
        node = self.const_nodes[name]
        st = self.const_state.get(name)
        if st == "done":
            return node
        if st == "busy":
            raise HappError(f"constant '{name}' depends on itself", node.loc)
        self.const_state[name] = "busy"
        val, vty = self.const_eval(node.value)
        if node.type is not None:
            ty = self.resolve_type(node.type)
            val = self.convert_const(val, vty, ty, node.value.loc)
            node.ty, node.val = ty, val
            sym = Symbol(name, "const", ty, node=node, value=val)
        else:
            if vty in ("int", "float"):
                ty = T.I32 if vty == "int" else T.F32
                untyped = vty
            else:
                ty, untyped = vty, None
            node.ty, node.val = ty, val
            sym = Symbol(name, "const", ty, node=node, value=val, untyped=untyped)
        self.prog.consts[name] = sym
        self.const_state[name] = "done"
        return node

    def convert_const(self, val, vty, ty, loc):
        if not ty.is_scalar:
            raise HappError("constants must be numbers or bools", loc)
        if ty.is_bool:
            if not isinstance(val, bool):
                raise HappError(f"expected a bool constant, found {vty}", loc)
            return val
        if isinstance(val, bool):
            raise HappError("expected a number, found bool", loc)
        if ty.is_int:
            if isinstance(val, float):
                raise HappError(f"float constant can't be used as {ty}", loc)
            lo, hi = ty.int_range()
            if not lo <= val <= hi:
                raise HappError(f"{val} doesn't fit in {ty} (range {lo}..{hi})", loc)
            return val
        return float(val)

    def const_eval(self, e):
        """Evaluate a compile-time expression -> (python value, 'int'|'float'|Type)."""
        if isinstance(e, A.IntLit):
            return e.value, "int"
        if isinstance(e, A.FloatLit):
            return e.value, "float"
        if isinstance(e, A.BoolLit):
            return e.value, T.BOOL
        if isinstance(e, A.Name):
            if e.id not in self.const_nodes:
                raise HappError(f"'{e.id}' is not a constant", e.loc)
            node = self.eval_const_def(e.id)
            sym = self.prog.consts[e.id]
            return node.val, (sym.untyped or node.ty)
        if isinstance(e, A.Unary):
            v, t = self.const_eval(e.operand)
            if e.op == "-" and not isinstance(v, bool):
                return -v, t
            if e.op == "!" and isinstance(v, bool):
                return not v, t
            if e.op == "~" and isinstance(v, int) and not isinstance(v, bool):
                return ~v, t
        if isinstance(e, A.Cast):
            v, _ = self.const_eval(e.expr)
            ty = self.resolve_type(e.type)
            if ty.is_int and not isinstance(v, bool):
                return int(v), ty
            if ty.is_float and not isinstance(v, bool):
                return float(v), ty
        if isinstance(e, A.Binary):
            a, at = self.const_eval(e.left)
            b, bt = self.const_eval(e.right)
            op = e.op
            is_float = isinstance(a, float) or isinstance(b, float)
            rt = at if not isinstance(at, str) else bt
            if isinstance(rt, str):
                rt = "float" if is_float else "int"
            try:
                if op in ("+", "-", "*"):
                    return {"+": a + b, "-": a - b, "*": a * b}[op], rt
                if op == "/":
                    return (a / b if is_float else int(a / b)), rt
                if op == "%":
                    return (a - b * int(a / b)), rt
                if op in ("<<", ">>", "&", "|", "^") and not is_float:
                    return {"<<": a << b, ">>": a >> b, "&": a & b, "|": a | b, "^": a ^ b}[op], rt
                if op in ("==", "!=", "<", ">", "<=", ">="):
                    return {"==": a == b, "!=": a != b, "<": a < b, ">": a > b,
                            "<=": a <= b, ">=": a >= b}[op], T.BOOL
                if op in ("&&", "||"):
                    return (a and b) if op == "&&" else (a or b), T.BOOL
            except ZeroDivisionError:
                raise HappError("division by zero in constant", e.loc)
        raise HappError("this expression is not a compile-time constant", e.loc)

    # ------------------------------------------------------------ signatures
    def check_signature(self, fn):
        if fn.name in RESERVED:
            if not (fn.name == "main" and not fn.is_kernel and not fn.is_export):
                raise HappError(f"'{fn.name}' is reserved; pick another name", fn.loc)
        for p in fn.params:
            p.ty = self.resolve_type(p.type)
        names = [p.name for p in fn.params]
        for i, n in enumerate(names):
            if n in names[:i]:
                raise HappError(f"parameter '{n}' appears twice", fn.params[i].loc)
        fn.ret_ty = self.resolve_type(fn.ret) if fn.ret is not None else T.VOID
        fn.sig = {"calls": set(), "builtins": set()}
        for attr, (args, loc) in fn.attrs.items():
            if attr == "workgroup":
                if not fn.is_kernel:
                    raise HappError("@workgroup only applies to kernels", loc)
                if not 1 <= len(args) <= 3 or any(a <= 0 for a in args):
                    raise HappError("@workgroup takes 1 to 3 positive sizes, e.g. @workgroup(64)", loc)
            elif attr in ("inline", "noinline", "strict"):
                if args:
                    raise HappError(f"@{attr} takes no arguments", loc)
            else:
                raise HappError(f"unknown attribute @{attr}", loc,
                                "known: @workgroup(x,y,z), @inline, @noinline, @strict")
        if fn.is_kernel:
            wg = fn.attrs.get("workgroup", ([64], None))[0]
            fn.sig["workgroup"] = (list(wg) + [1, 1, 1])[:3]
            if fn.ret is not None:
                raise HappError("kernels can't return a value; write results into a buffer", fn.loc)
            for p in fn.params:
                if p.ty.is_ptr:
                    self.require_gpu_storable(p.ty.pointee, p.loc, f"buffer '{p.name}'")
                elif p.ty not in (T.I32, T.U32, T.F32):
                    raise HappError(f"kernel parameter '{p.name}' must be a pointer (a GPU buffer) "
                                    f"or i32/u32/f32, not {p.ty}", p.loc)
            self.prog.kernels.append(fn)
        if fn.is_export or fn.is_extern:
            what = "exported" if fn.is_export else "extern"
            for p in fn.params:
                self.require_c_abi(p.ty, p.loc, what)
            if not fn.ret_ty.is_void:
                self.require_c_abi(fn.ret_ty, fn.loc, what)
            (self.prog.exports if fn.is_export else self.prog.externs).append(fn)
        if fn.name == "main":
            if fn.params or fn.ret_ty not in (T.VOID, T.I32):
                raise HappError("main must be 'fn main()' or 'fn main() -> i32'", fn.loc)
            self.prog.main = fn

    def require_c_abi(self, ty, loc, what):
        if ty.is_ptr or (ty.is_scalar and ty != T.F16):
            return
        raise HappError(f"{what} functions can only pass numbers, bools and pointers, not {ty}",
                        loc, f"pass a pointer instead, e.g. '*{ty}'")

    def require_gpu_storable(self, ty, loc, what):
        if ty.is_scalar:
            if ty.name in ("i32", "u32", "f32", "f16"):
                return
        elif ty.is_vec or ty.is_mat:
            return
        elif ty.is_array:
            return self.require_gpu_storable(ty.elem, loc, what)
        elif ty.is_struct:
            for _, ft in ty.fields:
                self.require_gpu_storable(ft, loc, f"{what} (field of {ty.name})")
            return
        raise HappError(f"{what}: GPUs can't store {ty}", loc,
                        "GPU buffers hold i32, u32, f32, f16, vectors, matrices, and structs/arrays of them")

    # ------------------------------------------------------------ scopes
    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def bind(self, name, sym, loc):
        if name in self.scopes[-1]:
            raise HappError(f"'{name}' is already defined in this block", loc)
        if name in T.NAMED_TYPES or name in self.structs:
            raise HappError(f"'{name}' is a type name", loc)
        self.scopes[-1][name] = sym

    def lookup(self, name):
        for sc in reversed(self.scopes):
            if name in sc:
                return sc[name]
        if name in self.prog.consts:
            return self.prog.consts[name]
        return None

    # ------------------------------------------------------------ bodies
    def check_body(self, fn):
        self.fn = fn
        self.scopes = []
        self.loops = 0
        self.push()
        if fn.is_kernel:
            for v in KERNEL_VARS:
                self.scopes[-1][v] = Symbol(v, "kernelvar", T.Vec(T.U32, 3))
        for p in fn.params:
            self.bind(p.name, Symbol(p.name, "param", p.ty, False, p), p.loc)
        self.check_block(fn.body, new_scope=False, top=True)
        if not fn.ret_ty.is_void and not self.always_returns(fn.body):
            raise HappError(f"function '{fn.name}' can reach its end without returning a {fn.ret_ty}",
                            fn.loc)
        self.pop()
        self.fn = None

    def always_returns(self, st):
        if isinstance(st, A.Return):
            return True
        if isinstance(st, A.Block):
            return any(self.always_returns(s) for s in st.stmts)
        if isinstance(st, A.If):
            return st.els is not None and self.always_returns(st.then) and self.always_returns(st.els)
        if isinstance(st, A.While):
            return (isinstance(st.cond, A.BoolLit) and st.cond.value and not self.has_break(st.body))
        return False

    def has_break(self, st):
        if isinstance(st, A.Break):
            return True
        if isinstance(st, A.Block):
            return any(self.has_break(s) for s in st.stmts)
        if isinstance(st, A.If):
            return self.has_break(st.then) or (st.els is not None and self.has_break(st.els))
        return False   # breaks inside nested loops belong to those loops

    def check_block(self, block, new_scope=True, top=False):
        if new_scope:
            self.push()
        for s in block.stmts:
            self.check_stmt(s, top)
        if new_scope:
            self.pop()

    def check_stmt(self, s, top=False):
        if isinstance(s, A.Let):
            declared = self.resolve_type(s.type) if s.type is not None else None
            if s.value is not None:
                vt = self.check(s.value, declared)
                if declared is not None:
                    self.expect_type(s.value, vt, declared)
                    ty = declared
                else:
                    ty = vt
                if ty.is_void or ty == T.STR:
                    raise HappError(f"can't store a {ty} in a variable", s.value.loc)
            else:
                ty = declared
            s.ty = ty
            self.bind(s.name, Symbol(s.name, "local", ty, s.mutable, s), s.loc)
        elif isinstance(s, A.Shared):
            if not (self.fn.is_kernel and top):
                raise HappError("'shared' arrays must be declared at the top level of a kernel", s.loc)
            ty = self.resolve_type(s.type)
            if not (ty.is_array and ty.elem.is_scalar and ty.elem.name in ("i32", "u32", "f32", "f16")):
                raise HappError("shared memory must be an array of i32, u32, f32 or f16, e.g. [256]f32",
                                s.loc)
            s.ty = ty
            self.fn.uses.add("shared")
            self.bind(s.name, Symbol(s.name, "shared", ty, True, s), s.loc)
        elif isinstance(s, A.Assign):
            lt, mutable, why = self.check_lvalue(s.target)
            if not mutable:
                raise HappError(f"can't assign here: {why}", s.target.loc)
            if s.op == "=":
                vt = self.check(s.value, lt)
                self.expect_type(s.value, vt, lt)
            else:
                op = s.op[:-1]
                vt = self.check_operand_pair(s.target, s.value, lt, skip_left=True)
                rt = self.binary_result(op, lt, vt, s.loc)
                if rt != lt:
                    raise HappError(f"'{s.op}' would change the type from {lt} to {rt}", s.loc)
        elif isinstance(s, A.If):
            self.expect_bool(s.cond)
            self.check_block(s.then)
            if isinstance(s.els, A.If):
                self.check_stmt(s.els)
            elif s.els is not None:
                self.check_block(s.els)
        elif isinstance(s, A.While):
            self.expect_bool(s.cond)
            self.loops += 1
            self.check_block(s.body)
            self.loops -= 1
        elif isinstance(s, A.For):
            want = self.resolve_type(s.vtype) if s.vtype is not None else None
            if want is not None and not want.is_int:
                raise HappError(f"a loop variable must be an integer type, not {want}", s.loc)
            ty = self.check_operand_pair(s.start, s.end, want)
            st = s.start.ty
            if want is not None and st != want:
                raise HappError(f"range start is {st} but the loop variable is {want}", s.start.loc)
            if not st.is_int or st != s.end.ty:
                raise HappError(f"range bounds must be the same integer type (got {s.start.ty} "
                                f"and {s.end.ty})", s.loc)
            s.ty = st
            self.push()
            self.bind(s.var, Symbol(s.var, "local", st, False, s), s.loc)
            self.loops += 1
            self.check_block(s.body, new_scope=False)
            self.loops -= 1
            self.pop()
        elif isinstance(s, A.Return):
            rt = self.fn.ret_ty
            if s.value is None:
                if not rt.is_void:
                    raise HappError(f"missing return value of type {rt}", s.loc)
            else:
                if rt.is_void:
                    raise HappError(f"'{self.fn.name}' doesn't return a value", s.value.loc)
                vt = self.check(s.value, rt)
                self.expect_type(s.value, vt, rt)
        elif isinstance(s, (A.Break, A.Continue)):
            if self.loops == 0:
                raise HappError(f"'{'break' if isinstance(s, A.Break) else 'continue'}' outside a loop",
                                s.loc)
        elif isinstance(s, A.ExprStmt):
            if not isinstance(s.expr, A.Call):
                raise HappError("this expression does nothing; did you mean to assign it?", s.loc)
            self.check(s.expr, None)
        elif isinstance(s, A.Block):
            self.check_block(s)
        else:
            raise AssertionError(s)

    def expect_bool(self, e):
        t = self.check(e, T.BOOL)
        if not t.is_bool:
            raise HappError(f"condition must be a bool, found {t}", e.loc,
                            "compare explicitly, e.g. 'x != 0'")

    def expect_type(self, e, got, want):
        if got != want:
            hint = None
            if got.is_numeric and want.is_numeric:
                hint = f"convert with 'as', e.g. '({self.short(e)}) as {want}'"
            raise HappError(f"expected {want}, found {got}", e.loc, hint)

    @staticmethod
    def short(e):
        if isinstance(e, A.Name):
            return e.id
        return "..."

    # ------------------------------------------------------------ lvalues
    def check_lvalue(self, e):
        """Type-check an assignable expression -> (type, mutable, reason_if_not)."""
        if isinstance(e, A.Name):
            t = self.check(e)
            sym = e.ref
            if sym.kind in ("local", "shared"):
                return t, sym.mutable, f"'{e.id}' is declared with 'let'; use 'var' to change it"
            return t, False, f"'{e.id}' is a {'parameter' if sym.kind == 'param' else sym.kind} " \
                             f"and can't be changed"
        if isinstance(e, A.Index):
            t = self.check(e)
            bt = e.base.ty
            if bt.is_ptr:
                return t, True, ""
            _, m, why = self.check_lvalue(e.base)
            return t, m, why
        if isinstance(e, A.Field):
            t = self.check(e)
            if e.kind == "ptrfield":
                return t, True, ""
            if e.kind == "swizzle" and len(set(e.info)) != len(e.info):
                return t, False, "a swizzle with repeated lanes can't be assigned"
            _, m, why = self.check_lvalue(e.base)
            return t, m, why
        if isinstance(e, A.Unary) and e.op == "*":
            return self.check(e), True, ""
        t = self.check(e)
        return t, False, "this is a value, not a variable"

    # ------------------------------------------------------------ expressions
    def untyped(self, e):
        """'int'/'float' if e is a literal (or untyped constant) that adapts to context."""
        if isinstance(e, A.IntLit):
            return "int"
        if isinstance(e, A.FloatLit):
            return "float"
        if isinstance(e, A.Name):
            sym = self.lookup(e.id)
            if sym is not None and sym.kind == "const" and sym.untyped:
                return sym.untyped
        return None

    def literal_type(self, kind, value, expected, loc):
        exp = expected.elem_scalar if expected is not None else None
        if kind == "int":
            if exp is not None and exp.is_int:
                lo, hi = exp.int_range()
                if not lo <= value <= hi:
                    raise HappError(f"{value} doesn't fit in {exp} (range {lo}..{hi})", loc)
                return exp
            if exp is not None and exp.is_float:
                return exp
            if -(1 << 31) <= value < (1 << 31):
                return T.I32
            if -(1 << 63) <= value < (1 << 63):
                return T.I64
            raise HappError(f"{value} is too large", loc)
        if exp is not None and exp.is_float:
            return exp
        if exp is not None and exp.is_int:
            raise HappError(f"expected an integer ({exp}), found a decimal number", loc)
        return T.F32

    def check_operand_pair(self, left, right, expected, skip_left=False):
        """Type two operands so that literals adapt to the other side. Returns right type."""
        lk = None if skip_left else self.untyped(left)
        rk = self.untyped(right)
        hint = expected if expected is not None and (expected.is_numeric or expected.is_vec
                                                     or expected.is_mat) else None
        if lk and rk:
            if hint is not None and hint.elem_scalar is not None:
                target = hint.elem_scalar
                if target.is_int and "float" in (lk, rk):
                    target = T.F32
            else:
                target = T.F32 if "float" in (lk, rk) else T.I32
            self.check(left, target)
            return self.check(right, target)
        if lk:
            rt = self.check(right, hint)
            self.check(left, rt if not rt.is_ptr else T.I64)
            return rt
        lt = left.ty if skip_left else self.check(left, hint)
        if lt.is_ptr:
            return self.check(right, T.I64)
        return self.check(right, lt)

    def check(self, e, expected=None):
        t = self._check(e, expected)
        e.ty = t
        return t

    def _check(self, e, expected):
        if isinstance(e, A.IntLit):
            return self.literal_type("int", e.value, expected, e.loc)
        if isinstance(e, A.FloatLit):
            return self.literal_type("float", e.value, expected, e.loc)
        if isinstance(e, A.BoolLit):
            return T.BOOL
        if isinstance(e, A.StrLit):
            return T.STR
        if isinstance(e, A.Name):
            sym = self.lookup(e.id)
            if sym is None:
                if e.id in self.prog.fns:
                    raise HappError(f"'{e.id}' is a function; call it with '{e.id}(...)'", e.loc)
                if e.id in KERNEL_VARS:
                    raise HappError(f"'{e.id}' only exists inside kernels", e.loc)
                cands = [n for sc in self.scopes for n in sc] + list(self.prog.consts)
                close = difflib.get_close_matches(e.id, cands, 1)
                raise HappError(f"unknown name '{e.id}'", e.loc,
                                f"did you mean '{close[0]}'?" if close else None)
            e.ref = sym
            if sym.kind == "kernelvar":
                self.fn.uses.add("kernelvar")
            if sym.kind == "const" and sym.untyped:
                return self.literal_type(sym.untyped, sym.value, expected, e.loc)
            return sym.ty
        if isinstance(e, A.Unary):
            return self.check_unary(e, expected)
        if isinstance(e, A.Binary):
            if e.op in ("&&", "||"):
                for side in (e.left, e.right):
                    t = self.check(side, T.BOOL)
                    if not t.is_bool:
                        raise HappError(f"'{e.op}' needs bools, found {t}", side.loc)
                return T.BOOL
            if e.op in ("==", "!=", "<", ">", "<=", ">="):
                rt = self.check_operand_pair(e.left, e.right, None)
                lt = e.left.ty
                if lt != rt:
                    raise HappError(f"can't compare {lt} with {rt}", e.loc,
                                    "convert one side with 'as'")
                if not (lt.is_numeric or (lt.is_bool and e.op in ("==", "!="))):
                    raise HappError(f"can't compare values of type {lt} with '{e.op}'", e.loc)
                return T.BOOL
            if e.op in ("<<", ">>"):
                lt = self.check(e.left, expected if expected is not None and expected.is_int else None)
                rt = self.check(e.right, lt.elem_scalar if lt.elem_scalar is not None else None)
                return self.binary_result(e.op, lt, rt, e.loc)
            rt = self.check_operand_pair(e.left, e.right, expected)
            return self.binary_result(e.op, e.left.ty, rt, e.loc)
        if isinstance(e, A.Cast):
            target = self.resolve_type(e.type)
            e.target = target
            k = self.untyped(e.expr)
            if k and target.is_numeric:
                src = self.check(e.expr, target if (k == "int" or target.is_float) else None)
            else:
                src = self.check(e.expr)
            self.check_cast(src, target, e.loc)
            return target
        if isinstance(e, A.Call):
            return self.check_call(e, expected)
        if isinstance(e, A.Index):
            bt = self.check(e.base)
            it = self.check(e.index, None)
            if not it.is_int:
                raise HappError(f"index must be an integer, found {it}", e.index.loc)
            if bt.is_ptr:
                if bt.pointee.is_void:
                    raise HappError("can't index a void pointer", e.loc)
                self.fn.uses.add("ptrindex")
                return bt.pointee
            if bt.is_array:
                if isinstance(e.index, A.IntLit) and not 0 <= e.index.value < bt.n:
                    raise HappError(f"index {e.index.value} is out of bounds for {bt}", e.index.loc)
                return bt.elem
            if bt.is_vec:
                return bt.elem
            if bt.is_mat:
                return bt.col
            raise HappError(f"can't index a value of type {bt}", e.loc)
        if isinstance(e, A.Field):
            bt = self.check(e.base)
            if bt.is_ptr and bt.pointee.is_struct:
                st = bt.pointee
                e.kind = "ptrfield"
            elif bt.is_struct:
                st = bt
                e.kind = "field"
            elif bt.is_vec:
                lanes = T.parse_swizzle(e.name, bt.n)
                if lanes is None:
                    raise HappError(f"{bt} has no component '{e.name}'", e.loc,
                                    f"use x, y{', z' if bt.n > 2 else ''}{', w' if bt.n > 3 else ''}")
                e.kind, e.info = "swizzle", lanes
                return bt.elem if len(lanes) == 1 else T.Vec(bt.elem, len(lanes))
            else:
                raise HappError(f"{bt} has no fields", e.loc)
            if e.name not in st.field_index:
                close = difflib.get_close_matches(e.name, list(st.field_index), 1)
                raise HappError(f"struct {st.name} has no field '{e.name}'", e.loc,
                                f"did you mean '{close[0]}'?" if close else None)
            e.info = st.field_index[e.name]
            return st.fields[e.info][1]
        if isinstance(e, A.ArrayLit):
            if not e.elems:
                raise HappError("empty array literal; use 'var a: [N]T;' for a zeroed array", e.loc)
            et = expected.elem if expected is not None and expected.is_array else None
            first = self.check(e.elems[0], et)
            for x in e.elems[1:]:
                t = self.check(x, first)
                self.expect_type(x, t, first)
            if first.is_void or first == T.STR:
                raise HappError(f"arrays can't hold {first}", e.loc)
            return T.Array(first, len(e.elems))
        raise AssertionError(e)

    def check_unary(self, e, expected):
        op = e.op
        if op == "&":
            t, mutable, why = self.check_lvalue(e.operand)
            if isinstance(e.operand, A.Field) and e.operand.kind == "swizzle" or \
                    isinstance(e.operand, A.Index) and e.operand.base.ty.is_vec:
                raise HappError("can't take the address of a vector component", e.loc)
            if isinstance(e.operand, A.Name) and not mutable:
                raise HappError(f"can't take the address of '{e.operand.id}': {why}", e.loc)
            self.fn.uses.add("addr")
            return T.Ptr(t)
        if op == "*":
            t = self.check(e.operand)
            if not t.is_ptr:
                raise HappError(f"can't dereference a {t}", e.loc)
            self.fn.uses.add("ptrindex")
            return t.pointee
        t = self.check(e.operand, expected)
        if op == "-":
            if t.is_numeric or (t.is_vec and not t.elem.is_bool) or t.is_mat:
                return t
            raise HappError(f"can't negate a {t}", e.loc)
        if op == "!":
            if t.is_bool:
                return t
            raise HappError(f"'!' needs a bool, found {t}", e.loc, "for integers use '~'")
        if op == "~":
            if t.is_int or (t.is_vec and t.elem.is_int):
                return t
            raise HappError(f"'~' needs an integer, found {t}", e.loc)
        raise AssertionError(op)

    def binary_result(self, op, lt, rt, loc):
        if op in ("+", "-", "*", "/", "%"):
            if lt.is_ptr and rt.is_int and op in ("+", "-"):
                self.fn.uses.add("ptrarith")
                return lt
            if lt.is_numeric and lt == rt:
                return lt
            if lt.is_vec and lt == rt:
                return lt
            if lt.is_vec and rt.is_scalar and rt == lt.elem:
                return lt
            if rt.is_vec and lt.is_scalar and lt == rt.elem:
                return rt
            if lt.is_mat and rt == lt and op in ("+", "-", "*"):
                return lt
            if lt.is_mat and rt == lt.col and op == "*":
                return rt
            if lt.is_mat and rt == T.F32 and op in ("*", "/"):
                return lt
            if rt.is_mat and lt == T.F32 and op == "*":
                return rt
            raise HappError(f"'{op}' can't combine {lt} and {rt}", loc,
                            "HA++ never converts types silently; use 'as'" if
                            (lt.is_numeric and rt.is_numeric) else None)
        if op in ("&", "|", "^"):
            if lt == rt and (lt.is_int or lt.is_bool or (lt.is_vec and lt.elem.is_int)):
                return lt
            raise HappError(f"'{op}' needs two integers (or bools) of the same type, found {lt} and {rt}",
                            loc)
        if op in ("<<", ">>"):
            if (lt.is_int and rt == lt) or (lt.is_vec and lt.elem.is_int and rt in (lt, lt.elem)):
                return lt
            raise HappError(f"'{op}' needs integers of the same type, found {lt} and {rt}", loc)
        raise AssertionError(op)

    def check_cast(self, src, dst, loc):
        if src == dst:
            return
        if src.is_numeric and dst.is_numeric:
            return
        if src.is_bool and dst.is_numeric and dst.is_int:
            return
        if src.is_vec and dst.is_vec and src.n == dst.n:
            return
        if src.is_ptr and dst.is_ptr:
            return
        if src.is_ptr and dst in (T.U64, T.I64):
            return
        if src in (T.U64, T.I64) and dst.is_ptr:
            return
        hint = "use 'x != 0' to turn a number into a bool" if dst.is_bool else None
        raise HappError(f"can't convert {src} to {dst}", loc, hint)

    # ------------------------------------------------------------ calls
    def check_call(self, e, expected):
        name = e.name
        if name in self.prog.fns:
            fn = self.prog.fns[name]
            if fn.is_kernel:
                raise HappError(f"'{name}' is a kernel; kernels are launched by the host (GPU) "
                                f"or through '{name}_cpu' from outside HA++", e.loc)
            if any(a.name for a in e.args):
                raise HappError("named arguments are only for struct constructors", e.loc)
            if len(e.args) != len(fn.params):
                raise HappError(f"'{name}' takes {len(fn.params)} argument(s), got {len(e.args)}",
                                e.loc)
            for a, p in zip(e.args, fn.params):
                t = self.check(a.value, p.ty)
                self.expect_type(a.value, t, p.ty)
            e.kind, e.target = "fn", fn
            self.fn.sig["calls"].add(name)
            return fn.ret_ty
        if name in T.NAMED_TYPES:
            ty = T.NAMED_TYPES[name]
            if ty.is_vec:
                return self.check_vec_ctor(e, ty)
            if ty.is_mat:
                return self.check_mat_ctor(e, ty)
            raise HappError(f"use 'x as {name}' to convert to {name}", e.loc)
        if name in self.structs:
            return self.check_struct_ctor(e, self.structs[name][0])
        if name in BUILTIN_NAMES:
            if any(a.name for a in e.args):
                raise HappError("named arguments are only for struct constructors", e.loc)
            e.kind = "builtin"
            self.fn.sig["builtins"].add(name)
            return self.check_builtin(e, expected)
        cands = list(self.prog.fns) + list(BUILTIN_NAMES) + list(self.structs)
        close = difflib.get_close_matches(name, cands, 1)
        raise HappError(f"unknown function '{name}'", e.loc,
                        f"did you mean '{close[0]}'?" if close else None)

    def check_vec_ctor(self, e, ty):
        e.kind, e.target = "ctor_vec", ty
        if any(a.name for a in e.args):
            raise HappError("vector constructors take positional arguments", e.loc)
        lanes = 0
        for a in e.args:
            t = self.check(a.value, ty.elem)
            if t == ty.elem:
                lanes += 1
            elif t.is_vec and t.elem == ty.elem:
                lanes += t.n
            else:
                raise HappError(f"{ty} can't be built from {t}", a.value.loc,
                                f"convert with 'as {ty.elem}'" if t.is_numeric else None)
        if not (lanes == ty.n or (len(e.args) == 1 and lanes == 1)):
            raise HappError(f"{ty} needs {ty.n} components (or 1 to fill all), got {lanes}", e.loc)
        return ty

    def check_mat_ctor(self, e, ty):
        e.kind, e.target = "ctor_mat", ty
        n = ty.n
        ts = [self.check(a.value, T.F32) for a in e.args]
        if len(ts) == 1 and ts[0] == T.F32:
            e.info = "diag"
        elif len(ts) == n and all(t == ty.col for t in ts):
            e.info = "cols"
        elif len(ts) == n * n and all(t == T.F32 for t in ts):
            e.info = "scalars"
        else:
            raise HappError(f"{ty} takes 1 f32 (diagonal), {n} {ty.col} columns, or {n * n} f32 values",
                            e.loc)
        return ty

    def check_struct_ctor(self, e, st):
        e.kind, e.target = "ctor_struct", st
        named = [a.name is not None for a in e.args]
        if any(named) and not all(named):
            raise HappError("use either all named or all positional arguments", e.loc)
        if e.args and not any(named) and len(e.args) != len(st.fields):
            raise HappError(f"{st.name} has {len(st.fields)} fields; positional construction needs all of "
                            f"them (or name them: {st.name}({st.fields[0][0]}: ...))", e.loc)
        mapping = []
        for i, a in enumerate(e.args):
            if a.name is not None:
                if a.name not in st.field_index:
                    raise HappError(f"struct {st.name} has no field '{a.name}'", a.loc)
                idx = st.field_index[a.name]
                if idx in [m for m, _ in mapping]:
                    raise HappError(f"field '{a.name}' given twice", a.loc)
            else:
                idx = i
            ft = st.fields[idx][1]
            t = self.check(a.value, ft)
            self.expect_type(a.value, t, ft)
            mapping.append((idx, a.value))
        e.info = mapping
        return st

    def check_same_float(self, e, args, expected, allow_scalar_after=0):
        """Type args that must share one float scalar/vector type (later ones may be scalars)."""
        # find the first non-literal to anchor literal types
        anchor = None
        for a in args:
            if not self.untyped(a):
                anchor = self.check(a, expected)
                break
        if anchor is None:
            hint = expected if expected is not None and expected.elem_scalar is not None \
                and expected.elem_scalar.is_float else T.F32
            anchor = hint
        ts = []
        for i, a in enumerate(args):
            t = a.ty if a.ty is not None and not self.untyped(a) else \
                self.check(a, anchor.elem_scalar if (i >= allow_scalar_after and allow_scalar_after)
                           else anchor)
            ts.append(t)
        base = ts[0]
        if not (base.elem_scalar is not None and base.elem_scalar.is_float and not base.is_mat):
            raise HappError(f"'{e.name}' needs float values (f16/f32/f64 or float vectors), found {base}",
                            args[0].loc)
        for i, (a, t) in enumerate(zip(args, ts)):
            if t == base:
                continue
            if allow_scalar_after and i >= allow_scalar_after and t == base.elem_scalar:
                continue
            raise HappError(f"'{e.name}': argument {i + 1} is {t} but the first is {base}", a.loc)
        return base

    def check_builtin(self, e, expected):
        name = e.name
        args = [a.value for a in e.args]

        def need(n):
            if len(args) != n:
                raise HappError(f"'{name}' takes {n} argument(s), got {len(args)}", e.loc)

        if name in FLOAT_UNARY:
            need(1)
            t = self.check_same_float(e, args, expected)
            self.note_transcendental(name, t, e)
            return t
        if name in ("pow", "atan2", "mod"):
            need(2)
            t = self.check_same_float(e, args, expected, allow_scalar_after=1 if name == "mod" else 0)
            self.note_transcendental(name, t, e)
            return t
        if name == "step":
            need(2)
            return self.check_same_float(e, args, expected)
        if name in ("abs", "sign"):
            need(1)
            t = self.check(args[0], expected)
            if not (t.is_numeric or (t.is_vec and t.elem.is_numeric)):
                raise HappError(f"'{name}' needs a number, found {t}", args[0].loc)
            return t
        if name in ("min", "max"):
            need(2)
            rt = self.check_operand_pair(args[0], args[1], expected)
            lt = args[0].ty
            if lt == rt and (lt.is_numeric or lt.is_vec):
                return lt
            if lt.is_vec and rt == lt.elem:
                return lt
            raise HappError(f"'{name}' needs two values of the same type, found {lt} and {rt}", e.loc)
        if name == "clamp":
            need(3)
            if not self.untyped(args[0]):
                xt = self.check(args[0], expected)
            else:
                anchor = next((a for a in args[1:] if not self.untyped(a)), None)
                hint = self.check(anchor, expected) if anchor is not None else expected
                if hint is None or hint.elem_scalar is None:
                    kinds = {self.untyped(a) for a in args}
                    hint = T.F32 if "float" in kinds else T.I32
                xt = self.check(args[0], hint.elem_scalar if hint.is_vec else hint)
            for a in args[1:]:
                t = self.check(a, xt.elem_scalar if xt.is_vec else xt)
                if t != xt and not (xt.is_vec and t == xt.elem):
                    raise HappError(f"clamp bounds must be {xt}"
                                    f"{' or ' + str(xt.elem) if xt.is_vec else ''}, found {t}", a.loc)
            if not (xt.is_numeric or xt.is_vec):
                raise HappError(f"can't clamp a {xt}", args[0].loc)
            return xt
        if name == "mix":
            need(3)
            return self.check_same_float(e, args, expected, allow_scalar_after=2)
        if name == "smoothstep":
            need(3)
            t = self.check_same_float(e, [args[2], args[0], args[1]], expected, allow_scalar_after=1)
            return t
        if name == "fma":
            need(3)
            return self.check_same_float(e, args, expected)
        if name in ("dot", "distance"):
            need(2)
            t = self.check_same_float(e, args, expected)
            if not t.is_vec:
                raise HappError(f"'{name}' needs vectors, found {t}", e.loc)
            return t.elem
        if name == "length":
            need(1)
            t = self.check_same_float(e, args, None)
            if not t.is_vec:
                raise HappError(f"'length' needs a vector, found {t}", e.loc, "for numbers use abs()")
            return t.elem
        if name == "normalize":
            need(1)
            t = self.check_same_float(e, args, expected)
            if not t.is_vec:
                raise HappError(f"'normalize' needs a vector, found {t}", e.loc)
            return t
        if name == "cross":
            need(2)
            t = self.check_same_float(e, args, expected)
            if not (t.is_vec and t.n == 3):
                raise HappError(f"'cross' needs 3-component vectors, found {t}", e.loc)
            return t
        if name == "transpose":
            need(1)
            t = self.check(args[0])
            if not t.is_mat:
                raise HappError(f"'transpose' needs a matrix, found {t}", e.loc)
            return t
        if name == "select":
            need(3)
            c = self.check(args[0], T.BOOL)
            if not c.is_bool:
                raise HappError(f"select's first argument must be a bool, found {c}", args[0].loc)
            rt = self.check_operand_pair(args[1], args[2], expected)
            if args[1].ty != rt or rt.in_memory or rt.is_void or rt == T.STR:
                raise HappError(f"select needs two values of the same type, found {args[1].ty} and {rt}",
                                e.loc)
            return rt
        if name in BIT_FUNCS:
            need(1)
            src, dst = BIT_FUNCS[name]
            t = self.check(args[0], src)
            self.expect_type(args[0], t, src)
            return dst
        if name == "pack_half2":
            need(1)
            t = self.check(args[0], T.Vec(T.F32, 2))
            self.expect_type(args[0], t, T.Vec(T.F32, 2))
            return T.U32
        if name == "unpack_half2":
            need(1)
            t = self.check(args[0], T.U32)
            self.expect_type(args[0], t, T.U32)
            return T.Vec(T.F32, 2)
        if name in INT_UNARY:
            need(1)
            t = self.check(args[0], expected)
            if not t.is_int:
                raise HappError(f"'{name}' needs an integer, found {t}", args[0].loc)
            return t
        if name == "print":
            for a in args:
                t = self.check(a)
                if not (t.is_scalar or t.is_vec or t == T.STR or t.is_ptr):
                    raise HappError(f"print can't show a {t} yet; print its fields", a.loc)
            self.fn.uses.add("print")
            return T.VOID
        if name == "barrier":
            need(0)
            self.require_kernel(e)
            self.fn.uses.add("barrier")
            return T.VOID
        if name in SUBGROUP:
            self.require_kernel(e)
            self.fn.uses.add("subgroup")
            if name in ("subgroup_lane", "subgroup_size"):
                need(0)
                return T.U32
            need(1)
            t = self.check(args[0], expected)
            if t not in (T.I32, T.U32, T.F32):
                raise HappError(f"'{name}' works on i32, u32 or f32, found {t}", args[0].loc)
            return t
        if name in ATOMICS:
            need(3)
            bt = self.check(args[0])
            if bt.is_ptr and bt.pointee in (T.I32, T.U32):
                et = bt.pointee
            elif bt.is_array and bt.elem in (T.I32, T.U32) and isinstance(args[0], A.Name) \
                    and args[0].ref.kind == "shared":
                et = bt.elem
            else:
                raise HappError(f"'{name}' needs a *i32/*u32 buffer or a shared i32/u32 array, found {bt}",
                                args[0].loc)
            it = self.check(args[1])
            if not it.is_int:
                raise HappError(f"index must be an integer, found {it}", args[1].loc)
            vt = self.check(args[2], et)
            self.expect_type(args[2], vt, et)
            self.fn.uses.add("atomic")
            return et
        raise AssertionError(name)

    def require_kernel(self, e):
        if not self.fn.is_kernel:
            raise HappError(f"'{e.name}' can only be used directly inside a kernel", e.loc)

    def note_transcendental(self, name, t, e):
        if name in TRANSCENDENTAL:
            if t.elem_scalar == T.F64:
                raise HappError(f"'{name}' is not available for f64 yet", e.loc,
                                "convert to f32 with 'as f32'")
            self.fn.sig["calls"].add(STD_IMPL[name])

    # ------------------------------------------------------------ GPU rules
    def check_gpu(self, kernel):
        """Everything a kernel can reach must be expressible on Vulkan and Metal."""
        visiting, done = [], set()

        def visit_fn(fn, via):
            if fn.name in done:
                return
            if fn in visiting:
                raise HappError(f"kernel '{kernel.name}' reaches recursive function '{fn.name}'; "
                                f"GPUs don't support recursion", fn.loc)
            if fn.is_extern:
                raise HappError(f"kernel '{kernel.name}' calls extern function '{fn.name}', "
                                f"which can't run on a GPU", via)
            if fn is not kernel:
                for p in fn.params:
                    if p.ty.is_ptr:
                        raise HappError(f"'{fn.name}' is called from kernel '{kernel.name}' but takes a "
                                        f"pointer; GPU helper functions take values only", p.loc)
                if fn.ret_ty.is_array:
                    raise HappError(f"'{fn.name}' is called from a kernel and returns an array; "
                                    f"wrap it in a struct", fn.loc)
            visiting.append(fn)
            self.gpu_walk(fn, kernel, visit_fn)
            visiting.pop()
            done.add(fn.name)

        visit_fn(kernel, kernel.loc)

    def gpu_walk(self, fn, kernel, visit_fn):
        def bad(node, why, hint=None):
            where = f"in kernel '{kernel.name}'" if fn is kernel else \
                f"in '{fn.name}' (used by kernel '{kernel.name}')"
            raise HappError(f"{why} {where}", node.loc, hint)

        def ty_ok(t, node):
            s = t.elem_scalar
            if t.is_ptr:
                if not (isinstance(node, A.Name) and node.ref is not None and node.ref.kind == "param"
                        and fn is kernel):
                    bad(node, "pointers can't be used here on the GPU", "index the buffer directly: buf[i]")
            elif t.is_array:
                ty_ok(t.elem, node)
            elif t.is_struct:
                for _, ft in t.fields:
                    ty_ok(ft, node)
            elif t == T.STR:
                bad(node, "strings aren't available")
            elif s is not None and s.name not in GPU_SCALARS:
                bad(node, f"type {s} isn't available on GPUs",
                    "use i32, u32, f32, f16 or bool in kernels")

        def expr(e):
            if e is None:
                return
            if e.ty is not None and not e.ty.is_void:
                ty_ok(e.ty, e)
            if isinstance(e, A.Unary):
                if e.op in ("&", "*"):
                    bad(e, "pointer operations aren't available")
                expr(e.operand)
            elif isinstance(e, A.Binary):
                if e.op in ("+", "-") and (e.left.ty.is_ptr or e.right.ty.is_ptr):
                    bad(e, "pointer arithmetic isn't available")
                expr(e.left)
                expr(e.right)
            elif isinstance(e, A.Cast):
                expr(e.expr)
            elif isinstance(e, A.Call):
                if e.kind == "builtin" and e.name == "print":
                    bad(e, "print isn't available")
                if e.kind == "builtin" and e.name in ATOMICS:
                    # buffer argument is a pointer by design
                    for a in e.args[1:]:
                        expr(a.value)
                    return
                if e.kind == "fn":
                    visit_fn(e.target, e.loc)
                for a in e.args:
                    expr(a.value)
            elif isinstance(e, A.Index):
                if e.base.ty.is_ptr:
                    if not (isinstance(e.base, A.Name) and e.base.ref.kind == "param"):
                        bad(e, "only kernel buffer parameters can be indexed like pointers")
                else:
                    expr(e.base)
                expr(e.index)
            elif isinstance(e, A.Field):
                if e.kind == "ptrfield":
                    bad(e, "'buf.field' needs an index on the GPU", "write buf[0].field")
                expr(e.base)
            elif isinstance(e, A.ArrayLit):
                for x in e.elems:
                    expr(x)

        def stmt(s):
            if isinstance(s, A.Let):
                ty_ok(s.ty, s)
                expr(s.value)
            elif isinstance(s, A.Assign):
                expr(s.target)
                expr(s.value)
            elif isinstance(s, A.If):
                expr(s.cond)
                stmt(s.then)
                if s.els is not None:
                    stmt(s.els)
            elif isinstance(s, A.While):
                expr(s.cond)
                stmt(s.body)
            elif isinstance(s, A.For):
                ty_ok(s.ty, s)
                expr(s.start)
                expr(s.end)
                stmt(s.body)
            elif isinstance(s, A.Return):
                expr(s.value)
            elif isinstance(s, A.ExprStmt):
                expr(s.expr)
            elif isinstance(s, A.Block):
                for x in s.stmts:
                    stmt(x)

        stmt(fn.body)


def check(modules):
    return Checker(modules).run()
