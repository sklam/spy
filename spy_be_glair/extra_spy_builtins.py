from typing import TYPE_CHECKING, Annotated, Any, cast

from spy.fqn import FQN
from spy.vm.b import TYPES, B
from spy.vm.field import W_Field
from spy.vm.function import (
    FuncParam,
    W_ASTFunc,
    W_BuiltinFunc,
    W_Func,
    W_FuncType,
)

# from spy.vm.builtin import IRTag
from spy.vm.irtag import IRTag

# from spy.vm.tuple import W_Tuple
from spy.vm.modules.__spy__.interp_tuple import W_InterpTuple

# from spy.vm.member import Member
from spy.vm.object import W_Object, W_Type, builtin_method
from spy.vm.opspec import W_MetaArg, W_OpSpec
from spy.vm.registry import ModuleRegistry
from spy.vm.str import W_Str
from spy.vm.struct import W_Struct, W_StructField, W_StructType, calc_layout

from .mlir_utils import (
    create_mlir_type_fqn,
    encode_asm_operation,
    parse_composite_type,
)

if TYPE_CHECKING:
    from spy.vm.vm import SPyVM

MLIR = ModuleRegistry("mlir")


class W_MLIR_Value(W_Object):
    __spy_storage_category__ = "value"

    def spy_key(self, vm: "SPyVM") -> object:
        # MLIR values are red-only SSA values; identity suffices for blue-time caching
        return id(self)


_type_caches: dict[str, "W_MLIR_Type"] = {}


@MLIR.builtin_type("MLIR_Type")
class W_MLIR_Type(W_StructType):
    original_name: str
    size: int
    members_w: "tuple[W_Type, ...]"

    @builtin_method("__new__", color="blue")
    @staticmethod
    def w_new(vm: "SPyVM", w_name: W_Str, *w_argtypes: "W_MLIR_Type") -> "W_MLIR_Type":
        name = vm.unwrap_str(w_name)

        def fmt(t: "W_MLIR_Type"):
            fn = getattr(t, "w_str", None)
            if fn is not None:
                out = vm.unwrap_str(fn(vm, t))
                return out
            else:
                raise TypeError

        formatted_name = name.format(*map(fmt, w_argtypes))
        fqn = vm.get_unique_FQN(create_mlir_type_fqn(formatted_name))

        if fqn in _type_caches:
            return _type_caches[fqn]
        w_type = W_MLIR_Type.from_pyclass(fqn, W_MLIR_Value)
        w_type.original_name = formatted_name
        w_type.size = 0  # Fake sizeof for SPy
        w_type.members_w = ()
        _type_caches[fqn] = w_type
        vm.add_global(fqn, w_type, irtag=IRTag("mlir.type", spelling=formatted_name))
        return w_type

    @builtin_method("__str__")
    @staticmethod
    def w_str(vm: "SPyVM", w_self: "W_MLIR_Type") -> "W_Str":
        return vm.wrap(str(w_self.original_name))


def _make_multivalue_type(vm: "SPyVM", results_w: "list[W_Type]") -> "W_MLIR_Type":
    """
    Build an ephemeral struct type with _field0, _field1, ... for multi-result ops.
    """
    from spy.location import Loc

    loc = Loc.here()
    field_names = [f"_field{i}" for i in range(len(results_w))]
    fields_w = {
        name: W_Field(name, w_T, loc) for name, w_T in zip(field_names, results_w)
    }
    struct_fields_w, size = calc_layout(fields_w)

    fqn = vm.get_unique_FQN(
        create_mlir_type_fqn(
            "multivalues$" + "|".join(f.fqn.fullname for f in results_w)
        )
    )
    if fqn in _type_caches:
        return _type_caches[fqn]  # type: ignore[return-value]

    w_type = W_MLIR_Type.declare(fqn)
    w_type.original_name = "multivalues"
    w_type.size = size
    w_type.members_w = tuple(results_w)

    dict_w: dict[str, W_Object] = {}
    for w_sf in struct_fields_w:
        dict_w[w_sf.name] = w_sf
    W_StructType.define(w_type, W_Struct, dict_w)

    _type_caches[fqn] = w_type  # type: ignore[assignment]
    vm.add_global(fqn, w_type, irtag=IRTag("mlir.type", spelling="multivalues"))
    return w_type


def _finalize_op(
    vm: "SPyVM",
    asm: str,
    args_w: "list[W_Type]",
    results_w: "list[W_Type]",
) -> W_BuiltinFunc:
    """Build the W_BuiltinFunc for a finalized MLIR op builder."""
    if len(results_w) == 0:
        fn_retty: W_Type = TYPES.w_NoneType
        resname = "void"
    elif len(results_w) == 1:
        fn_retty = results_w[0]
        resname = fn_retty.fqn.fullname
    else:
        fn_retty = _make_multivalue_type(vm, results_w)
        resname = fn_retty.fqn.fullname

    RESTYPE = Annotated[W_Object, fn_retty]
    opname = encode_asm_operation([asm, resname])
    params = [FuncParam(cast(W_Type, w_T), "simple") for w_T in args_w]
    w_functype = W_FuncType.new(params, w_restype=fn_retty)

    def w_opimpl(vm: "SPyVM", *args_w: W_Object) -> RESTYPE:
        raise NotImplementedError("MLIR ops are not supposed to be called")

    fqn = vm.get_unique_FQN(FQN(["mlir", "asm", opname]))
    w_op = W_BuiltinFunc(w_functype, fqn, w_opimpl)
    vm.add_global(fqn, w_op, irtag=IRTag("mlir.asm", asm=asm))
    return w_op


@MLIR.builtin_type("_MLIROp")
class W_MLIROp(W_Object):
    """
    Builder returned by MLIR_op("asm"). Supports chained .args(...).result(...).
    """

    __spy_storage_category__ = "reference"

    asm: str
    args_w: "list[W_Type]"

    def __init__(self, asm: str) -> None:
        self.asm = asm
        self.args_w = []

    @builtin_method("__new__", color="blue")
    @staticmethod
    def w_new(vm: "SPyVM", w_asm: W_Str) -> "W_MLIROp":
        asm = vm.unwrap_str(w_asm)
        obj = W_MLIROp.__new__(W_MLIROp)
        obj.asm = asm
        obj.args_w = []
        return obj

    @builtin_method("__call_method__", color="blue", kind="metafunc")
    @staticmethod
    def w_CALL_METHOD(
        vm: "SPyVM", wam_self: W_MetaArg, wam_name: W_MetaArg, *args_wam: W_MetaArg
    ) -> W_OpSpec:
        w_self = wam_self.w_val
        assert isinstance(w_self, W_MLIROp)
        name = wam_name.blue_unwrap_str(vm)

        if name == "args":
            w_self.args_w = [cast(W_Type, wam.w_val) for wam in args_wam]
            return W_OpSpec.const(w_self)

        elif name == "result":
            results_w = [cast(W_Type, wam.w_val) for wam in args_wam]
            w_func = _finalize_op(vm, w_self.asm, w_self.args_w, results_w)
            return W_OpSpec.const(w_func)

        return W_OpSpec.NULL


@MLIR.builtin_func("MLIR_op_builder", color="blue")
def w_MLIR_op_builder(vm: "SPyVM", w_asm: W_Str) -> W_MLIROp:
    asm = vm.unwrap_str(w_asm)
    obj = W_MLIROp.__new__(W_MLIROp)
    obj.asm = asm
    obj.args_w = []
    return obj


@MLIR.builtin_func("MLIR_op")
def w_MLIR_op(
    vm: "SPyVM", w_opname: W_Str, w_restype: W_Type, w_argtypes: W_InterpTuple
) -> W_BuiltinFunc:
    RESTYPE = Annotated[W_Object, w_restype]
    opname = vm.unwrap_str(w_opname)
    argtypes_w = w_argtypes.items_w

    # functype - cast W_Object to W_Type (they should be types)
    params = [FuncParam(cast(W_Type, w_T), "simple") for w_T in argtypes_w]
    w_functype = W_FuncType.new(params, w_restype=w_restype)

    def w_opimpl(vm: "SPyVM", *args_w: W_Object) -> W_Object:
        raise NotImplementedError("MLIR ops are not supposed to be called")

    fqn = FQN(["mlir", "op", opname])
    w_op = W_BuiltinFunc(w_functype, fqn, w_opimpl)
    irtag = IRTag("mlir.op", opname=opname)
    vm.add_global(fqn, w_op, irtag=irtag)
    return w_op


@MLIR.builtin_func("MLIR_unpack")
def w_MLIR_unpack(vm: "SPyVM", w_fn: W_Func, w_idx: W_Object) -> W_BuiltinFunc:
    restype = cast(W_MLIR_Type, w_fn.w_functype.w_restype)

    assert restype.original_name.startswith("multivalues")
    types = restype.members_w

    idx = vm.unwrap_i32(w_idx)
    retty = types[idx]

    params = [FuncParam(B.w_object, "simple")]
    w_functype = W_FuncType.new(params, w_restype=retty)

    def w_opimpl(vm: "SPyVM", fn: W_Object) -> W_Object:
        raise NotImplementedError("MLIR ops are not supposed to be called")

    fqn = (
        FQN(["mlir", "unpack"])
        .with_suffix(str(idx))
        .with_qualifiers([restype.fqn.fullname])
    )
    w_op = W_BuiltinFunc(w_functype, fqn, w_opimpl)
    irtag = IRTag("mlir.asm", idx=idx)  # we can add any extra metadata we want here
    vm.add_global(fqn, w_op, irtag=irtag)
    return w_op


@MLIR.builtin_func("MLIR_asm")
def w_MLIR_asm(
    vm: "SPyVM", w_asm: W_Str, w_restype: W_Object, w_argtypes: W_Object
) -> W_BuiltinFunc:
    RESTYPE: Any
    if isinstance(w_restype, W_InterpTuple):
        fn_retty = _make_multivalue_type(vm, list(w_restype.items_w))
        RESTYPE = Annotated[W_Object, fn_retty]
        resname = fn_retty.fqn.fullname
    elif isinstance(w_restype, W_Type):
        RESTYPE = Annotated[W_Object, w_restype]
        fn_retty = w_restype
        resname = fn_retty.fqn.fullname
    else:
        raise AssertionError(f"unexpected restype: {w_restype!r}")

    asm = vm.unwrap_str(w_asm)
    if isinstance(w_argtypes, W_InterpTuple):
        argtypes_w = tuple(w_argtypes.items_w)
    else:
        argtypes_w = tuple(w_argtypes.values_w.values())

    # Ensure all argument types have fqn attribute
    for at in argtypes_w:
        assert hasattr(at, "fqn"), f"Argument type {at} missing fqn attribute"

    # Cast to W_Type after assertion for type safety
    argtypes_typed = cast(list[W_Type], list(argtypes_w))

    fqn_parts = [
        asm,
        resname,
        # *(at.fqn.fullname for at in argtypes_typed),
    ]
    opname = encode_asm_operation(fqn_parts)

    # functype - use the typed argtypes after assertion
    params = [FuncParam(w_T, "simple") for w_T in argtypes_typed]
    w_functype = W_FuncType.new(params, w_restype=fn_retty)

    def w_opimpl(vm: "SPyVM", *args_w: W_Object) -> RESTYPE:
        raise NotImplementedError("MLIR ops are not supposed to be called")

    fqn = vm.get_unique_FQN(FQN(["mlir", "asm", opname]))

    w_op = W_BuiltinFunc(w_functype, fqn, w_opimpl)
    irtag = IRTag("mlir.asm", asm=asm)  # we can add any extra metadata we want here
    vm.add_global(fqn, w_op, irtag=irtag)
    return w_op


@MLIR.builtin_func("MLIR_transform", color="blue")
def w_MLIR_transform(
    vm: "SPyVM",
    fn: W_ASTFunc,
    passes: W_InterpTuple,
) -> W_ASTFunc:
    newfn = W_ASTFunc(
        w_functype=fn.w_functype,
        fqn=fn.fqn.with_suffix("transformed"),
        funcdef=fn.funcdef,
        closure=fn.closure,
        locals_types_w=fn.locals_types_w,
    )
    passes_list = [vm.unwrap_str(ps) for ps in passes.items_w]
    vm.add_global(
        newfn.fqn,
        newfn,
        irtag=IRTag("mlir.transforms", transforms=" ".join(passes_list)),
    )
    return newfn



@MLIR.builtin_func("export_ffi_c", color="blue")
def w_export_ffi_c(
    vm: "SPyVM",
    fn: W_ASTFunc,
) -> W_ASTFunc:
    newfn = W_ASTFunc(
        w_functype=fn.w_functype,
        fqn=fn.fqn.with_suffix("cffi"),
        funcdef=fn.funcdef,
        closure=fn.closure,
        locals_types_w=fn.locals_types_w,
        defaults_w=fn.defaults_w,
        lowering_stage=fn.lowering_stage,
    )
    irtag = IRTag("glair", **{"export_ffi_c": True})
    vm.add_global(newfn.fqn, newfn, irtag=irtag)
    return newfn
