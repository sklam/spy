from dataclasses import dataclass

from spy.backend.c.context import C_FuncParam, C_Function, C_Type
from spy.errors import SPyError
from spy.fqn import FQN
from spy.textbuilder import TextBuilder
from spy.vm.b import TYPES, B
from spy.vm.function import W_ASTFunc, W_Func
from spy.vm.modules.jsffi import JSFFI
from spy.vm.modules.posix import POSIX
from spy.vm.modules.rawbuffer import RB
from spy.vm.object import W_Type
from spy.vm.vm import SPyVM


@dataclass
class GLAIR_Ident:
    GLAIR_KEYWORDS = (
        "module",
        "import",
        "extern",
        "fn",
        "let",
        "struct",
        "ptr_wrapper",
        "ref_alias",
        "mlir_type",
        "return",
        "if",
        "else",
        "while",
        "break",
        "continue",
        "true",
        "false",
        "void",
    )

    variable_name: str

    def check_glair_keyword(self) -> str:
        if self.variable_name in self.GLAIR_KEYWORDS:
            return f"{self.variable_name}$"
        return self.variable_name

    def __str__(self) -> str:
        return self.check_glair_keyword()


def glair_func_decl(c_func: C_Function) -> str:
    """Format a C_Function as a GLAIR function signature."""
    if not c_func.params:
        s_params = ""
    else:
        paramlist = [
            f"{p.name}: {p.c_type}" for p in c_func.params if p.c_type.name != "void"
        ]
        s_params = ", ".join(paramlist)
    return f"fn {c_func.name}({s_params}) -> {c_func.c_restype}"


class Context:
    """
    Global context of the GLAIR writer.

    Keep track of things like the mapping from W_* types to GLAIR type names.
    """

    vm: SPyVM
    tb_imports: TextBuilder
    seen_modules: set[str]
    _d: dict[W_Type, C_Type]

    def __init__(self, vm: SPyVM) -> None:
        self.vm = vm
        self.seen_modules = set()
        # set by GlairModuleWriter
        self.tb_imports = None  # type: ignore
        self._d = {}
        self._d[TYPES.w_NoneType] = C_Type("void")
        self._d[B.w_i8] = C_Type("i8")
        self._d[B.w_u8] = C_Type("u8")
        self._d[B.w_i32] = C_Type("i32")
        self._d[B.w_u32] = C_Type("u32")
        self._d[B.w_f64] = C_Type("f64")
        self._d[B.w_f32] = C_Type("f32")
        self._d[B.w_complex128] = C_Type("spy_Complex128")
        self._d[B.w_bool] = C_Type("bool")
        self._d[B.w_str] = C_Type("*spy_Str")
        self._d[RB.w_RawBuffer] = C_Type("*spy_RawBuffer")
        self._d[JSFFI.w_JsRef] = C_Type("JsRef")
        self._d[POSIX.w__FILE] = C_Type("*FILE")

    def w2c(self, w_T: W_Type) -> C_Type:
        if w_T in self._d:
            return self._d[w_T]

        elif isinstance(w_T, W_Type):
            c_type = C_Type(w_T.fqn.c_name)
            self._d[w_T] = c_type
            return c_type

        raise NotImplementedError(f"Cannot translate type {w_T} to GLAIR")

    def c_restype_by_fqn(self, fqn: FQN) -> C_Type:
        w_func = self.vm.lookup_global(fqn)
        assert isinstance(w_func, W_Func)
        w_restype = w_func.w_functype.w_restype
        return self.w2c(w_restype)

    def c_function(self, name: str, w_func: W_ASTFunc) -> C_Function:
        w_functype = w_func.w_functype
        funcdef = w_func.funcdef

        c_params = []
        for i, param in enumerate(w_functype.params):
            c_type = self.w2c(param.w_T)
            if param.kind == "simple":
                c_param_name = GLAIR_Ident(funcdef.args[i].name)
                # GLAIR_Ident satisfies the C_Ident protocol (has __str__)
                c_params.append(C_FuncParam(c_param_name, c_type))  # type: ignore[arg-type]
            elif param.kind == "var_positional":
                assert i == len(funcdef.args) - 1
                raise SPyError.simple(
                    "W_WIP",
                    "*args not yet supported by the GLAIR backend",
                    "*args declared here",
                    funcdef.args[i].loc,
                )
            else:
                assert False

        c_restype = self.w2c(w_functype.w_restype)
        return C_Function(name, c_params, c_restype)

    def add_import_maybe(self, fqn: FQN) -> None:
        modname = fqn.modname
        if modname in self.seen_modules:
            return

        self.seen_modules.add(modname)
        w_mod = self.vm.modules_w[modname]
        if not w_mod.is_builtin():
            self.tb_imports.wl(f"import {modname};")
