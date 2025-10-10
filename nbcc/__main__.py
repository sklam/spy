import sys
from pprint import pprint

from spy.fqn import FQN
from spy.interop import redshift
from spy.vm.function import W_ASTFunc

from .restructure import restructure
from .spy_ast import Node, convert_to_node


def main(argv: list[str]) -> None:
    [filename] = argv

    vm, w_mod = redshift(filename)

    symtab: dict[FQN, Node] = {}
    for fqn, w_obj in vm.fqns_by_modname(w_mod.name):
        print(fqn, w_obj)
        if isinstance(w_obj, W_ASTFunc):
            print("functype:", w_obj.w_functype)
            print("locals:")
            if w_obj.locals_types_w is not None:
                for varname, w_T in w_obj.locals_types_w.items():
                    print("   ", varname, w_T)
                node = convert_to_node(w_obj.funcdef, vm=vm)
                pprint(node)
                symtab[fqn] = node
                print()

    # restructure

    for fqn, func_node in symtab.items():
        restructure(fqn.fullname, func_node)


if __name__ == "__main__":
    main(sys.argv[1:])
