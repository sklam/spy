import itertools
from dataclasses import dataclass

import py.path

from spy.errors import WIP
from spy.fqn import FQN
from spy.textbuilder import TextBuilder
from spy.vm.b import TYPES, B
from spy.vm.cell import W_Cell
from spy.vm.function import W_ASTFunc, W_BuiltinFunc, W_Func
from spy.vm.modules.unsafe.ptr import W_Ptr, W_PtrType
from spy.vm.object import W_Object
from spy.vm.primitive import W_I32
from spy.vm.vm import SPyVM
from spy_be_glair.context import Context, glair_func_decl


@dataclass
class GlairModule:
    modname: str
    spyfile: py.path.local
    glairfile: py.path.local
    content: list[tuple[FQN, W_Object]]

    def __repr__(self) -> str:
        return f"<GlairModule {self.modname}>"


class GlairModuleWriter:
    ctx: Context
    glair_mod: GlairModule
    global_vars: set[str]
    seen_externs: set[str]

    # single TextBuilder for the .glair file
    tb: TextBuilder
    tb_imports: TextBuilder
    tb_externs: TextBuilder
    tb_globals: TextBuilder
    tb_content: TextBuilder

    def __init__(self, vm: SPyVM, glair_mod: GlairModule) -> None:
        self.ctx = Context(vm)
        self.glair_mod = glair_mod
        self.global_vars = set()
        self.seen_externs = set()
        self.tb = TextBuilder(use_colors=False)
        self._init()

    def __repr__(self) -> str:
        return f"<GlairModuleWriter for {self.glair_mod.modname}>"

    def write_glair_source(self) -> None:
        self._emit_content()
        self.glair_mod.glairfile.write(self.tb.build())

    def new_global_var(self, prefix: str) -> str:
        """
        Create a unique name for a global variable whose name starts with 'prefix'.
        GLAIR uses _g_ prefix (no SPY_ prefix unlike the C backend).
        """
        prefix = f"_g_{prefix}"
        for i in itertools.count():
            varname = f"{prefix}{i}"
            if varname not in self.global_vars:
                break
        self.global_vars.add(varname)
        return varname

    def _init(self) -> None:
        basename = self.glair_mod.glairfile.purebasename
        self.tb.wl(f"module {basename};")
        self.tb.wl()
        self.tb.wl("import spy_structdefs;")
        self.tb.wl("import _prelude;")
        self.tb_imports = self.tb.make_nested_builder()
        self.ctx.tb_imports = self.tb_imports
        # Prevent self-imports: this module name is already "available" in its own file
        self.ctx.seen_modules.add(self.glair_mod.modname)
        self.tb.wl()
        self.tb_externs = self.tb.make_nested_builder()
        self.tb.wl()
        self.tb_globals = self.tb.make_nested_builder()
        self.tb.wl()
        self.tb_content = self.tb.make_nested_builder()

    def _emit_content(self) -> None:
        for fqn, w_obj in self.glair_mod.content:
            assert w_obj is not None, "uninitialized global?"
            self._emit_obj(fqn, w_obj)

    def _emit_obj(self, fqn: FQN, w_obj: W_Object) -> None:
        if hasattr(w_obj, "fqn"):
            assert fqn == w_obj.fqn

        if isinstance(w_obj, W_ASTFunc):
            if w_obj.color == "red" and not w_obj.is_force_inline:
                self._emit_func(fqn, w_obj)

        elif isinstance(w_obj, W_BuiltinFunc):
            pass  # ignore builtin functions in module content

        elif isinstance(w_obj, W_Cell):
            w_content = w_obj.get()
            w_T = self.ctx.vm.dynamic_type(w_content)
            if isinstance(w_content, W_I32):
                intval = self.ctx.vm.unwrap(w_content)
                c_type = self.ctx.w2c(w_T)
                self.tb_globals.wl(f"let {fqn.c_name}: {c_type} = {intval}_i32;")
            elif isinstance(w_T, W_PtrType):
                assert isinstance(w_content, W_Ptr)
                assert w_content.addr == 0, (
                    "only NULL pointers can be stored in constants for now"
                )
                c_type = self.ctx.w2c(w_T)
                self.tb_globals.wl(f"let {fqn.c_name}: {c_type} = {c_type}$NULL;")
            else:
                raise WIP(f"var type `{w_T}` not supported")

        else:
            raise NotImplementedError("WIP")

    def _emit_func(self, fqn: FQN, w_func: W_ASTFunc) -> None:
        from spy_be_glair.glairwriter import GlairFuncWriter

        c_func = self.ctx.c_function(fqn.c_name, w_func)

        # Emit @loc annotation before the function declaration
        if self.glair_mod.spyfile is not None:
            spyline = w_func.funcdef.loc.line_start
            spyfile = str(self.glair_mod.spyfile)
            self.tb_content.wl(f'@loc("{spyfile}", {spyline})')

        self.tb_content.wl(glair_func_decl(c_func) + " {")
        with self.tb_content.indent():
            fw = GlairFuncWriter(self.ctx, self, fqn, w_func)
            fw.emit()
        self.tb_content.wl("}")
        self.tb_content.wl()

    def add_extern_maybe(self, fqn: FQN) -> None:
        """
        Emit a @builtin extern fn declaration for a builtin function,
        if it hasn't been emitted yet.
        """
        c_name = fqn.c_name
        if c_name in self.seen_externs:
            return
        self.seen_externs.add(c_name)

        try:
            w_obj = self.ctx.vm.lookup_global(fqn)
        except Exception:
            return

        if not isinstance(w_obj, W_Func):
            return

        w_functype = w_obj.w_functype
        params = []
        for i, param in enumerate(w_functype.params):
            if param.kind == "simple":
                # Skip the spy_types$Loc parameter: it's a SPy runtime detail
                # for bounds checking that isn't part of the GLAIR-level signature
                if param.w_T is TYPES.w_Loc:
                    continue
                c_type = self.ctx.w2c(param.w_T)
                if c_type.name != "void":
                    params.append(f"p{i}: {c_type}")
        c_restype = self.ctx.w2c(w_functype.w_restype)
        s_params = ", ".join(params)

        irtag = self.ctx.vm.get_irtag(fqn)
        if irtag.tag == "mlir.asm":
            return  # inlined as mlir statement, no extern fn needed
        elif irtag.tag == "mlir.op":
            self.tb_externs.wl(f'@mlir_op("{irtag.data["opname"]}")')
        else:
            self.tb_externs.wl("@builtin")
        self.tb_externs.wl(f"extern fn {c_name}({s_params}) -> {c_restype};")
        self.tb_externs.wl()
