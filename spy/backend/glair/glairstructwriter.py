from dataclasses import dataclass

import py.path

from spy.backend.c.context import C_Type
from spy.backend.glair.context import Context
from spy.fqn import FQN
from spy.textbuilder import TextBuilder
from spy.vm.modules.unsafe.ptr import W_PtrType, W_RefType
from spy.vm.object import W_Type
from spy.vm.struct import W_StructType
from spy.vm.vm import SPyVM


@dataclass
class GlairStructDefs:
    glairfile: py.path.local
    content: list[tuple[FQN, W_Type]]


class GlairStructWriter:
    ctx: Context
    glair_structdefs: GlairStructDefs

    tb: TextBuilder
    tb_structs: TextBuilder

    def __init__(self, vm: SPyVM, glair_structdefs: GlairStructDefs) -> None:
        self.ctx = Context(vm)
        self.glair_structdefs = glair_structdefs
        self.tb = TextBuilder(use_colors=False)
        self._init()

    def __repr__(self) -> str:
        return f"<GlairStructWriter for {self.glair_structdefs.glairfile}>"

    def write_glair_source(self) -> None:
        self._emit_content()
        self.glair_structdefs.glairfile.write(self.tb.build())

    def _init(self) -> None:
        self.tb.wl("module spy_structdefs;")
        self.tb.wl()
        # Builtin runtime structs: defined in spy.h on the C side but have no
        # W_StructType in the SPy type system, so they must be hardcoded here.
        self.tb.wl("struct spy_Complex128 {")
        self.tb.wl("    real: f64,")
        self.tb.wl("    imag: f64,")
        self.tb.wl("};")
        self.tb.wl()
        self.tb.wl("struct spy_Str {")
        self.tb.wl("    length: usize,")
        self.tb.wl("    flags: i32,")
        self.tb.wl("    data: *u8,")
        self.tb.wl("};")
        self.tb.wl()
        self.tb_structs = self.tb.make_nested_builder()

    def _emit_content(self) -> None:
        for fqn, w_type in self.glair_structdefs.content:
            assert fqn == w_type.fqn
            if isinstance(w_type, W_StructType):
                self._emit_StructType(fqn, w_type)
            elif isinstance(w_type, W_PtrType):
                self._emit_PtrType(fqn, w_type)
            elif isinstance(w_type, W_RefType):
                self._emit_RefType(fqn, w_type)
            else:
                assert False, f"Unknown type: {w_type}"

    def _emit_StructType(self, fqn: FQN, w_st: W_StructType) -> None:
        if not w_st.is_defined():
            # Undefined structs are spurious fwdecls — skip them.
            # See test_struct::test_fwdecl_is_ignored_by_C_backend for context.
            return

        irtag = self.ctx.vm.get_irtag(fqn)
        if irtag.tag == "struct.builtin":
            return

        c_st = C_Type(w_st.fqn.c_name)
        tb = self.tb_structs
        tb.wl(f"struct {c_st} {{")
        with tb.indent():
            for w_field in w_st.iterfields_w():
                c_fieldtype = self.ctx.w2c(w_field.w_T)
                tb.wl(f"{w_field.name}: {c_fieldtype},")
        tb.wl("};")
        tb.wl()

    def _emit_PtrType(self, fqn: FQN, w_ptrtype: W_PtrType) -> None:
        c_ptrtype = C_Type(w_ptrtype.fqn.c_name)
        w_itemT = w_ptrtype.w_itemT
        c_itemT = self.ctx.w2c(w_itemT)
        self.tb_structs.wl(f"ptr_wrapper {c_ptrtype} -> {c_itemT};")
        self.tb_structs.wl()

    def _emit_RefType(self, fqn: FQN, w_reftype: W_RefType) -> None:
        w_ptrtype = w_reftype.as_ptrtype(self.ctx.vm)
        c_reftype = C_Type(w_reftype.fqn.c_name)
        c_ptrtype = C_Type(w_ptrtype.fqn.c_name)
        self.tb_structs.wl(f"ref_alias {c_reftype} = {c_ptrtype};")
        self.tb_structs.wl()
