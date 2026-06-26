import math
from types import NoneType
from typing import TYPE_CHECKING

from spy import ast
from spy.backend.c import c_ast as C
from spy.errors import SPyError
from spy.fqn import FQN
from spy.location import Loc
from spy.textbuilder import TextBuilder
from spy.util import magic_dispatch, shortrepr
from spy.vm.b import TYPES, B
from spy.vm.function import W_ASTFunc, W_Func
from spy.vm.irtag import IRTag
from spy.vm.modules.posix import W__FILE
from spy.vm.modules.unsafe.ptr import W_Ptr
from spy.vm.object import W_Type
from spy.vm.struct import W_StructType
from spy_be_glair.context import Context, GLAIR_Ident

if TYPE_CHECKING:
    from spy_be_glair.glairmodwriter import GlairModuleWriter


# Mapping from W_Type to GLAIR integer/float literal suffixes
_SUFFIX_MAP = {
    B.w_i8: "_i8",
    B.w_u8: "_u8",
    B.w_i32: "_i32",
    B.w_u32: "_u32",
    B.w_f32: "_f32",
    B.w_f64: "_f64",
}


def _fmt_float_body(val: float) -> str:
    """
    Format a float value without scientific notation, with a digit
    on both sides of the decimal point.
    """
    s = repr(val)
    if "e" in s or "E" in s or math.isinf(val) or math.isnan(val):
        # Expand to fixed-point with enough precision
        s = format(abs(val), ".17f").rstrip("0")
        if s.endswith("."):
            s += "0"
        if val < 0:
            s = "-" + s
    if "." not in s:
        s += ".0"
    # Ensure a digit on both sides of '.'
    if s.startswith("."):
        s = "0" + s
    elif s.startswith("-."):
        s = "-0." + s[2:]
    return s


def _escape_glair_asm(asm: str) -> str:
    # MLIR asm strings can carry backslash sequences (e.g. `\0A`) and embedded
    # quotes that must survive verbatim through the GLAIR lexer's string parsing.
    return asm.replace("\\", "\\\\").replace('"', '\\"')


def _glair_string_literal(b: bytes) -> str:
    """
    Format bytes as a GLAIR string literal (double-quoted).
    Uses only the GLAIR escape set: \\n, \\t, \\r, \\0, \\\\, \\", \\xNN.
    No C-style "" concatenation tricks needed since \\xNN is exactly 2 hex digits.
    """

    def char_repr(val: int) -> str:
        if val == ord("\\"):
            return r"\\"
        elif val == ord('"'):
            return r"\""
        elif val == ord("\n"):
            return r"\n"
        elif val == ord("\t"):
            return r"\t"
        elif val == ord("\r"):
            return r"\r"
        elif val == 0:
            return r"\0"
        elif 32 <= val < 127:
            return chr(val)
        return rf"\x{val:02x}"

    return '"' + "".join(char_repr(v) for v in b) + '"'


class GlairFuncWriter:
    ctx: Context
    gmodw: "GlairModuleWriter"
    tb: TextBuilder
    fqn: FQN
    w_func: W_ASTFunc
    last_emitted_lineno: int

    def __init__(
        self,
        ctx: Context,
        gmodw: "GlairModuleWriter",
        fqn: FQN,
        w_func: W_ASTFunc,
    ) -> None:
        self.ctx = ctx
        self.gmodw = gmodw
        self.tb = gmodw.tb_content
        self.fqn = fqn
        self.last_emitted_lineno = -1
        self._unused_result_counter = 0

        assert w_func.lowering_stage == "linearize"
        self.w_func = w_func
        # GLAIR has no multi-value type: a SPy multivalues intermediary expands
        # into one independent local per field. Map each multivalues local to
        # its fanned-out fields so getfield reads, copy-assignments, and the
        # producing mlir op all agree on the same names.
        self._multivalues_fanout: dict[str, list[str]] = {}
        self._multivalues_field_types: dict[str, list[W_Type]] = {}
        self._collect_multivalues()

    def emit(self) -> None:
        self.emit_local_vars()
        for stmt in self.w_func.funcdef.body:
            self.emit_stmt(stmt)

        if self.w_func.w_functype.w_restype is not TYPES.w_NoneType:
            # Non-void function: guard against falling off the end; abort() is
            # declared in _prelude which every module imports
            msg = "reached the end of the function without a `return`"
            self.tb.wl(f"abort(); // {msg}")

    def _collect_multivalues(self) -> None:
        assert self.w_func.locals_types_w is not None
        for varname, w_T in self.w_func.locals_types_w.items():
            irtag = self.ctx.vm.get_irtag(w_T.fqn)
            if irtag.tag == "mlir.type" and irtag.data["spelling"] == "multivalues":
                members_w = list(w_T.members_w)  # type: ignore[attr-defined]
                self._multivalues_fanout[varname] = [
                    f"{varname}$f{i}" for i in range(len(members_w))
                ]
                self._multivalues_field_types[varname] = members_w

    def emit_local_vars(self) -> None:
        assert self.w_func.locals_types_w is not None
        param_names = [arg.name for arg in self.w_func.funcdef.args]
        for varname, w_T in self.w_func.locals_types_w.items():
            if w_T is TYPES.w_NoneType:
                # GLAIR §3.7: no void-typed variable declarations
                continue
            if varname in ("@return", "@if", "@and", "@or", "@while", "@assert"):
                continue
            if varname in param_names:
                continue
            if varname in self._multivalues_fanout:
                # GLAIR has no multi-value type: declare one local per field
                # instead of declaring the multivalues intermediary itself.
                fields = self._multivalues_fanout[varname]
                types = self._multivalues_field_types[varname]
                for fname, w_fT in zip(fields, types):
                    c_ftype = self.ctx.w2c(w_fT)
                    self.tb.wl(f"let {GLAIR_Ident(fname)}: {c_ftype};")
                continue
            glair_varname = GLAIR_Ident(varname)
            c_type = self.ctx.w2c(w_T)
            self.tb.wl(f"let {glair_varname}: {c_type};")

    def _is_mlir_asm_call(self, expr: ast.Expr) -> "IRTag | None":
        if not isinstance(expr, ast.Call):
            return None
        if not isinstance(expr.func, ast.FQNConst):
            return None
        irtag = self.ctx.vm.get_irtag(expr.func.fqn)
        if irtag.tag == "mlir.asm" and "asm" in irtag.data:
            return irtag
        return None

    def _fresh_unused_local(self, w_T: W_Type) -> GLAIR_Ident:
        # GLAIR requires a name for every MLIR result, even when SPy drops it.
        name = f"$unused{self._unused_result_counter}"
        self._unused_result_counter += 1
        c_type = self.ctx.w2c(w_T)
        self.tb.wl(f"let {GLAIR_Ident(name)}: {c_type};")
        return GLAIR_Ident(name)

    def _emit_mlir_stmt(
        self,
        loc: Loc,
        target_name: "str | None",
        call: ast.Call,
        irtag: IRTag,
    ) -> None:
        asm = _escape_glair_asm(irtag.data["asm"])
        args_str = ", ".join(str(self.fmt_expr(arg)) for arg in call.args)

        if call.w_T is TYPES.w_NoneType:
            self.emit_lineno_maybe(loc)
            self.tb.wl(f'mlir "{asm}" ({args_str}) -> ();')
            return

        w_restype = call.w_T
        assert w_restype is not None
        res_irtag = self.ctx.vm.get_irtag(w_restype.fqn)
        is_multivalues = (
            res_irtag.tag == "mlir.type"
            and res_irtag.data.get("spelling") == "multivalues"
        )

        self.emit_lineno_maybe(loc)
        if is_multivalues:
            # GLAIR has no multi-value type, so each result binds directly to
            # the per-field local declared in emit_local_vars.
            if target_name is None:
                members_w = w_restype.members_w  # type: ignore[attr-defined]
                results = [self._fresh_unused_local(w_fT) for w_fT in members_w]
            else:
                results = [
                    GLAIR_Ident(f) for f in self._multivalues_fanout[target_name]
                ]
            results_str = ", ".join(str(r) for r in results)
            self.tb.wl(f'mlir "{asm}" ({args_str}) -> ({results_str});')
            return

        if target_name is None:
            target = self._fresh_unused_local(w_restype)
        else:
            target = GLAIR_Ident(target_name)
        self.tb.wl(f'mlir "{asm}" ({args_str}) -> ({target});')

    def emit_lineno_maybe(self, loc: Loc) -> None:
        if loc.line_start != self.last_emitted_lineno:
            self.emit_lineno(loc.line_start)

    def emit_lineno(self, spyline: int) -> None:
        if self.gmodw.glair_mod.spyfile is None:
            return
        spyfile = str(self.gmodw.glair_mod.spyfile)
        self.tb.wl(f'@loc("{spyfile}", {spyline})')
        self.last_emitted_lineno = spyline

    def emit_stmt(self, stmt: ast.Stmt) -> None:
        self.emit_lineno_maybe(stmt.loc)
        magic_dispatch(self, "emit_stmt", stmt)

    def fmt_expr(self, expr: ast.Expr) -> C.Expr:
        return magic_dispatch(self, "fmt_expr", expr)

    def fmt_expr_BlockExpr(self, expr: ast.BlockExpr) -> C.Expr:
        msg = (
            "The GLAIR backend doesn't support ast.BlockExpr.\n"
            + "This probably means that there is a bug in the compilation pipeline\n"
            + "and that `linearize` was not called."
        )
        raise SPyError.simple("W_ValueError", msg, "", expr.loc)

    # ===== statements =====

    def emit_stmt_Pass(self, stmt: ast.Pass) -> None:
        pass

    def emit_stmt_Break(self, stmt: ast.Break) -> None:
        self.tb.wl("break;")

    def emit_stmt_Continue(self, stmt: ast.Continue) -> None:
        self.tb.wl("continue;")

    def emit_stmt_Return(self, ret: ast.Return) -> None:
        v = self.fmt_expr(ret.value)
        if v is C.Void():
            self.tb.wl("return;")
        else:
            self.tb.wl(f"return {v};")

    def emit_stmt_VarDef(self, vardef: ast.VarDef) -> None:
        # Local variable declaration is in emit_local_vars; here we assign the value
        if vardef.value:
            target = vardef.name.value
            v = self.fmt_expr(vardef.value)
            if vardef.value.w_T is TYPES.w_NoneType:
                # void-typed: emit the call for side effects, drop the assignment
                if v is not C.Void():
                    self.tb.wl(f"{v};")
            else:
                self.tb.wl(f"{target} = {v};")

    def emit_stmt_Assign(self, assign: ast.Assign) -> None:
        assert False, "ast.Assign nodes should not survive redshifting"

    def emit_stmt_AssignLocal(self, assign: ast.AssignLocal) -> None:
        target = assign.target.value
        irtag = self._is_mlir_asm_call(assign.value)
        if irtag is not None:
            assert isinstance(assign.value, ast.Call)
            self._emit_mlir_stmt(assign.loc, target, assign.value, irtag)
            return
        if target in self._multivalues_fanout:
            # GLAIR has no multi-value type: a multivalues-typed local
            # assignment must fan out into per-field copies. The RHS must
            # itself be a multivalues local (the only way to produce one
            # outside an mlir.asm call, which is handled above).
            assert isinstance(assign.value, ast.NameLocalDirect)
            src_name = assign.value.sym.name
            src_fields = self._multivalues_fanout[src_name]
            dst_fields = self._multivalues_fanout[target]
            for dst, src in zip(dst_fields, src_fields):
                self.tb.wl(f"{GLAIR_Ident(dst)} = {GLAIR_Ident(src)};")
            return
        v = self.fmt_expr(assign.value)
        glair_varname = GLAIR_Ident(target)
        if assign.value.w_T is TYPES.w_NoneType:
            # void-typed: keep the call, drop the assignment target
            if v is not C.Void():
                self.tb.wl(f"{v};")
        else:
            self.tb.wl(f"{glair_varname} = {v};")

    def emit_stmt_AssignCell(self, assign: ast.AssignCell) -> None:
        v = self.fmt_expr(assign.value)
        target = assign.target_fqn.c_name
        glair_varname = GLAIR_Ident(target)
        self.tb.wl(f"{glair_varname} = {v};")

    def emit_stmt_UnpackAssign(self, unpack: ast.UnpackAssign) -> None:
        if isinstance(unpack.value, ast.Tuple):
            for target, item in zip(unpack.targets, unpack.value.items):
                glair_target = GLAIR_Ident(target.value)
                v = self.fmt_expr(item)
                self.tb.wl(f"{glair_target} = {v};")
        else:
            assert unpack.value.w_T is not None
            c_tuple_type = self.ctx.w2c(unpack.value.w_T)
            v = self.fmt_expr(unpack.value)
            self.tb.wl("{")
            with self.tb.indent():
                self.tb.wl(f"let tmp: {c_tuple_type};")
                self.tb.wl(f"tmp = {v};")
                for i, target in enumerate(unpack.targets):
                    glair_target = GLAIR_Ident(target.value)
                    self.tb.wl(f"{glair_target} = tmp._item{i};")
            self.tb.wl("}")

    def emit_stmt_StmtExpr(self, stmt: ast.StmtExpr) -> None:
        irtag = self._is_mlir_asm_call(stmt.value)
        if irtag is not None:
            assert isinstance(stmt.value, ast.Call)
            self._emit_mlir_stmt(stmt.loc, None, stmt.value, irtag)
            return
        v = self.fmt_expr(stmt.value)
        if v is C.Void():
            pass
        else:
            self.tb.wl(f"{v};")

    def emit_stmt_If(self, if_node: ast.If) -> None:
        test = self.fmt_expr(if_node.test)
        self.tb.wl(f"if ({test})" + " {")
        with self.tb.indent():
            for stmt in if_node.then_body:
                self.emit_stmt(stmt)
        if if_node.else_body:
            self.tb.wl("} else {")
            with self.tb.indent():
                for stmt in if_node.else_body:
                    self.emit_stmt(stmt)
        self.tb.wl("}")

    def emit_stmt_While(self, while_node: ast.While) -> None:
        test = self.fmt_expr(while_node.test)
        self.tb.wl(f"while ({test})" + " {")
        with self.tb.indent():
            for stmt in while_node.body:
                self.emit_stmt(stmt)
        self.tb.wl("}")

    def emit_stmt_Assert(self, assert_node: ast.Assert) -> None:
        test = self.fmt_expr(assert_node.test)
        self.tb.wl(f"if (!({test}))" + " {")
        with self.tb.indent():
            if assert_node.msg is not None:
                msg = self.fmt_expr(assert_node.msg)
                self.tb.wl(
                    f'spy_panic("AssertionError", ({msg})->utf8, '
                    f'"{assert_node.loc.filename}", {assert_node.loc.line_start});'
                )
            else:
                self.tb.wl(
                    f'spy_panic("AssertionError", "assertion failed", '
                    f'"{assert_node.loc.filename}", {assert_node.loc.line_start});'
                )
        self.tb.wl("}")

    # ===== expressions =====

    def fmt_expr_Const(self, const: ast.Const) -> C.Expr:
        # XXX: Hack for migration
        # TODO: this should match on const.w_T instead of the value in w_val.
        import ctypes

        from spy.vm.modules.types import TYPES

        if const.w_T == TYPES.w_NoneType:
            return C.Void()

        const_value = const.w_val.value
        T = type(const_value)
        if T is bool:
            return C.Literal("true" if const_value else "false")
        elif T is float:
            suffix = _SUFFIX_MAP.get(const.w_T, "")  # type: ignore[arg-type]
            body = _fmt_float_body(float(const_value))
            if body.startswith("-"):
                return C.UnaryOp("-", C.Literal(f"{body[1:]}{suffix}"))
            return C.Literal(f"{body}{suffix}")
        elif T is ctypes.c_float:
            suffix = _SUFFIX_MAP.get(const.w_T, "")  # type: ignore[arg-type]
            body = _fmt_float_body(float(const_value.value))
            if body.startswith("-"):
                return C.UnaryOp("-", C.Literal(f"{body[1:]}{suffix}"))
            return C.Literal(f"{body}{suffix}")
        elif T is complex:
            val = complex(const_value)
            re_body = _fmt_float_body(val.real)
            im_body = _fmt_float_body(val.imag)
            # Use named-field compound literal syntax
            if re_body.startswith("-"):
                re_expr = f"-{re_body[1:]}_f64"
            else:
                re_expr = f"{re_body}_f64"
            if im_body.startswith("-"):
                im_expr = f"-{im_body[1:]}_f64"
            else:
                im_expr = f"{im_body}_f64"
            return C.Literal(f"spy_Complex128 {{ real: {re_expr}, imag: {im_expr}, }}")
        else:
            from fixedint.aliases import Int32, UInt8

            match const_value:
                case UInt8(val):
                    suffix = "_u8"
                    val = int(const_value)
                    return C.Literal(f"{val}{suffix}")
                case Int32(val):
                    suffix = "_i32"
                    val = int(const_value)
                    if val < 0:
                        return C.UnaryOp("-", C.Literal(f"{-val}{suffix}"))
                    return C.Literal(f"{val}{suffix}")
                case _:
                    raise TypeError(f"unsupported type: {type(const_value)}")

    def fmt_expr_StrLiteral(self, const: ast.StrLiteral) -> C.Expr:
        # String literals must be initialized as GLAIR globals.
        # Generate:
        #     let _g_str0: spy_Str = spy_Str { length: N_usize, flags: 0_i32, data: "...", };
        s = const.value
        utf8 = s.encode("utf-8")
        v = self.gmodw.new_global_var("str")  # _g_str0
        n = len(utf8)
        lit = _glair_string_literal(utf8)
        comment = shortrepr(utf8.decode("utf-8"), 15)
        self.gmodw.tb_globals.wl(
            f"let {v}: spy_Str = spy_Str"
            f" {{ length: {n}_usize, flags: 0_i32, data: {lit}, }};"
            f"  // {comment}"
        )
        return C.UnaryOp("&", C.Literal(v))

    def fmt_expr_FQNConst(self, const: ast.FQNConst) -> C.Expr:
        w_obj = self.ctx.vm.lookup_global(const.fqn)
        if isinstance(w_obj, W_Ptr):
            assert w_obj.addr == 0, "only NULL ptrs can be constants"
            return C.Literal(const.fqn.c_name)
        elif isinstance(w_obj, W_Func):
            return C.Literal(const.fqn.c_name)
        elif isinstance(w_obj, W__FILE):
            assert w_obj.h == 0, "only NULL _FILE can be a constant"
            # GLAIR doesn't have a NULL keyword; use a zero-init for FILE pointers
            return C.Literal("NULL")
        else:
            w_T = self.ctx.vm.dynamic_type(w_obj)
            t = w_T.fqn.human_name
            raise SPyError.simple(
                "W_WIP",
                f"Prebuilt constant of type `{t}` are not supported by the GLAIR backend",
                f"This is `{t}`",
                const.loc,
            )

    def fmt_expr_Name(self, name: ast.Name) -> C.Expr:
        assert False, "ast.Name nodes should not survive redshifting"

    def fmt_expr_NameLocalDirect(self, name: ast.NameLocalDirect) -> C.Expr:
        varname = GLAIR_Ident(name.sym.name)
        if name.w_T is TYPES.w_NoneType:
            return C.Void()
        else:
            return C.Literal(f"{varname}")

    def fmt_expr_NameOuterCell(self, name: ast.NameOuterCell) -> C.Expr:
        return C.Literal(name.fqn.c_name)

    def fmt_expr_NameOuterDirect(self, name: ast.NameOuterDirect) -> C.Expr:
        assert False, "unexpected NameOuterDirect"

    def fmt_expr_AssignExpr(self, assignexpr: ast.AssignExpr) -> C.Expr:
        return self._fmt_assignexpr(assignexpr.target.value, assignexpr.value)

    def fmt_expr_AssignExprLocal(self, assignexpr: ast.AssignExprLocal) -> C.Expr:
        return self._fmt_assignexpr(assignexpr.target.value, assignexpr.value)

    def fmt_expr_AssignExprCell(self, assignexpr: ast.AssignExprCell) -> C.Expr:
        return self._fmt_assignexpr(assignexpr.target_fqn.c_name, assignexpr.value)

    def _fmt_assignexpr(self, target: str, value_expr: ast.Expr) -> C.Expr:
        target_lit = C.Literal(target)
        value = self.fmt_expr(value_expr)
        return C.BinOp("=", target_lit, value)

    def fmt_expr_BinOp(self, binop: ast.BinOp) -> C.Expr:
        raise NotImplementedError(
            "ast.BinOp not supported. It should have been redshifted away"
        )

    def fmt_expr_And(self, op: ast.And) -> C.Expr:
        l = self.fmt_expr(op.left)
        r = self.fmt_expr(op.right)
        return C.BinOp("&&", l, r)

    def fmt_expr_Or(self, op: ast.Or) -> C.Expr:
        l = self.fmt_expr(op.left)
        r = self.fmt_expr(op.right)
        return C.BinOp("||", l, r)

    FQN2BinOp = {
        FQN("operator::i8_add"): "+",
        FQN("operator::i8_sub"): "-",
        FQN("operator::i8_mul"): "*",
        FQN("operator::i8_lshift"): "<<",
        FQN("operator::i8_rshift"): ">>",
        FQN("operator::i8_and"): "&",
        FQN("operator::i8_or"): "|",
        FQN("operator::i8_xor"): "^",
        FQN("operator::i8_eq"): "==",
        FQN("operator::i8_ne"): "!=",
        FQN("operator::i8_lt"): "<",
        FQN("operator::i8_le"): "<=",
        FQN("operator::i8_gt"): ">",
        FQN("operator::i8_ge"): ">=",
        #
        FQN("operator::u8_add"): "+",
        FQN("operator::u8_sub"): "-",
        FQN("operator::u8_mul"): "*",
        FQN("operator::u8_lshift"): "<<",
        FQN("operator::u8_rshift"): ">>",
        FQN("operator::u8_and"): "&",
        FQN("operator::u8_or"): "|",
        FQN("operator::u8_xor"): "^",
        FQN("operator::u8_eq"): "==",
        FQN("operator::u8_ne"): "!=",
        FQN("operator::u8_lt"): "<",
        FQN("operator::u8_le"): "<=",
        FQN("operator::u8_gt"): ">",
        FQN("operator::u8_ge"): ">=",
        #
        FQN("operator::i32_add"): "+",
        FQN("operator::i32_sub"): "-",
        FQN("operator::i32_mul"): "*",
        FQN("operator::i32_lshift"): "<<",
        FQN("operator::i32_rshift"): ">>",
        FQN("operator::i32_and"): "&",
        FQN("operator::i32_or"): "|",
        FQN("operator::i32_xor"): "^",
        FQN("operator::i32_eq"): "==",
        FQN("operator::i32_ne"): "!=",
        FQN("operator::i32_lt"): "<",
        FQN("operator::i32_le"): "<=",
        FQN("operator::i32_gt"): ">",
        FQN("operator::i32_ge"): ">=",
        #
        FQN("operator::u32_add"): "+",
        FQN("operator::u32_sub"): "-",
        FQN("operator::u32_mul"): "*",
        FQN("operator::u32_lshift"): "<<",
        FQN("operator::u32_rshift"): ">>",
        FQN("operator::u32_and"): "&",
        FQN("operator::u32_or"): "|",
        FQN("operator::u32_xor"): "^",
        FQN("operator::u32_eq"): "==",
        FQN("operator::u32_ne"): "!=",
        FQN("operator::u32_lt"): "<",
        FQN("operator::u32_le"): "<=",
        FQN("operator::u32_gt"): ">",
        FQN("operator::u32_ge"): ">=",
        #
        FQN("operator::f64_add"): "+",
        FQN("operator::f64_sub"): "-",
        FQN("operator::f64_mul"): "*",
        FQN("unsafe::f64_ieee754_div"): "/",
        FQN("operator::f64_eq"): "==",
        FQN("operator::f64_ne"): "!=",
        FQN("operator::f64_lt"): "<",
        FQN("operator::f64_le"): "<=",
        FQN("operator::f64_gt"): ">",
        FQN("operator::f64_ge"): ">=",
    }

    FQN2UnaryOp = {
        FQN("operator::i8_neg"): "-",
        FQN("operator::i32_neg"): "-",
        FQN("operator::f64_neg"): "-",
    }

    def fmt_expr_Call(self, call: ast.Call) -> C.Expr:
        assert isinstance(call.func, ast.FQNConst), (
            "indirect calls are not supported yet"
        )
        fqn = call.func.fqn

        irtag = self.ctx.vm.get_irtag(fqn)

        if op := self.FQN2BinOp.get(fqn):
            assert len(call.args) == 2
            l, r = [self.fmt_expr(arg) for arg in call.args]
            return C.BinOp(op, l, r)

        elif op := self.FQN2UnaryOp.get(fqn):
            assert len(call.args) == 1
            v = self.fmt_expr(call.args[0])
            return C.UnaryOp(op, v)

        elif irtag.tag == "struct.make":
            return self.fmt_struct_make(fqn, call, irtag)

        elif irtag.tag == "struct.getfield":
            return self.fmt_struct_getfield(fqn, call, irtag)

        elif irtag.tag == "ptr.getfield":
            return self.fmt_ptr_getfield(fqn, call, irtag)

        elif irtag.tag == "ptr.setfield":
            return self.fmt_ptr_setfield(fqn, call)

        elif irtag.tag == "ptr.deref":
            # Per GLAIR §6.7, "*expr" is the canonical whole-value load through
            # a raw pointer.  A raw_ref[T] / raw_ptr[T] is a ptr_wrapper around
            # *T whose inner pointer is the `.p` field (§3.3, §6.6), so the
            # full deref spelling is `*ref.p`.  Inlining here avoids emitting
            # an undefined `<Wrapper>$deref` extern.
            assert len(call.args) == 1
            c_ref = self.fmt_expr(call.args[0])
            return C.Literal(f"*{c_ref}.p")

        elif irtag.tag in ("ptr.getitem", "ptr.store"):
            # Remove the trailing W_Loc argument (GLAIR has @loc annotations instead)
            assert isinstance(call.args[-1], ast.Const), call.args[-1]
            call.args.pop()
            # GLAIR §3.3: ptr_wrapper ops are *implicitly* declared as
            # `<wrapper>_load` / `<wrapper>_store` (underscore separator).
            # Rewrite SPy's `$store` / `$getitem_byval` / `$getitem_byref`
            # suffix to the canonical GLAIR form and bypass fmt_generic_call
            # so we don't emit a spurious `@builtin extern fn` decl.
            c_name = fqn.c_name
            if irtag.tag == "ptr.store":
                assert c_name.endswith("$store")
                c_name = c_name[: -len("$store")] + "_store"
            else:
                for suffix in ("$getitem_byval", "$getitem_byref"):
                    if c_name.endswith(suffix):
                        c_name = c_name[: -len(suffix)] + "_load"
                        break
                else:
                    raise AssertionError(f"unexpected ptr.getitem fqn: {fqn.c_name!r}")
            c_args = [self.fmt_expr(arg) for arg in call.args]
            return C.Call(c_name, c_args)

        elif irtag.tag == "mlir.asm" and "asm" in irtag.data:
            raise SPyError.simple(
                "W_ValueError",
                "mlir.asm call reached expression context; "
                "it must appear as a statement",
                "",
                call.loc,
            )

        else:
            return self.fmt_generic_call(fqn, call)

    def fmt_generic_call(self, fqn: FQN, call: ast.Call) -> C.Expr:
        w_mod = self.ctx.vm.modules_w[fqn.modname]
        if w_mod.is_builtin():
            self.gmodw.add_extern_maybe(fqn)
        else:
            self.ctx.add_import_maybe(fqn)
        c_name = fqn.c_name
        c_args = [self.fmt_expr(arg) for arg in call.args]
        return C.Call(c_name, c_args)

    def fmt_struct_make(self, fqn: FQN, call: ast.Call, irtag: IRTag) -> C.Expr:
        w_func = self.ctx.vm.lookup_global(fqn)
        assert isinstance(w_func, W_Func)
        w_restype = w_func.w_functype.w_restype
        assert isinstance(w_restype, W_StructType)
        c_restype = self.ctx.w2c(w_restype)
        fields = list(w_restype.iterfields_w())
        c_args = [self.fmt_expr(arg) for arg in call.args]
        field_inits = ", ".join(
            f"{fields[i].name}: {c_args[i]}" for i in range(len(c_args))
        )
        return C.Literal(f"{c_restype} {{ {field_inits}, }}")

    def fmt_struct_getfield(self, fqn: FQN, call: ast.Call, irtag: IRTag) -> C.Expr:
        assert len(call.args) == 1
        arg = call.args[0]
        name = irtag.data["name"]
        if isinstance(arg, ast.NameLocalDirect):
            fields = self._multivalues_fanout.get(arg.sym.name)
            if fields is not None:
                assert name.startswith("_field")
                idx = int(name[len("_field") :])
                return C.Literal(str(GLAIR_Ident(fields[idx])))
        c_struct = self.fmt_expr(arg)
        return C.Dot(c_struct, name)

    def fmt_ptr_getfield(self, fqn: FQN, call: ast.Call, irtag: IRTag) -> C.Expr:
        assert isinstance(call.args[1], ast.StrLiteral)
        c_ptr = self.fmt_expr(call.args[0])
        attr = call.args[1].value
        offset = call.args[2]  # ignored
        c_field = C.PtrField(c_ptr, attr)
        if irtag.data["by"] == "byref":
            c_restype = self.ctx.c_restype_by_fqn(fqn)
            return C.PtrFieldByRef(c_restype, c_field)
        else:
            return c_field

    def fmt_ptr_setfield(self, fqn: FQN, call: ast.Call) -> C.Expr:
        assert isinstance(call.args[1], ast.StrLiteral)
        c_ptr = self.fmt_expr(call.args[0])
        attr = call.args[1].value
        offset = call.args[2]  # ignored
        c_lval = C.PtrField(c_ptr, attr)
        c_rval = self.fmt_expr(call.args[3])
        return C.BinOp("=", c_lval, c_rval)
