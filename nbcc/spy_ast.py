import ast as py_ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import spy.ast
from spy.analyze.symtable import Color, Symbol

if TYPE_CHECKING:
    from spy.vm.vm import SPyVM


@dataclass
class Node:
    """Simple node representation with opname and attributes dictionary."""

    opname: str
    attrdict: dict[str, Any]


def convert_to_node(
    node: Any,
    vm: "SPyVM",
    *,
    fields_to_ignore: Any = (),
) -> Node:
    dumper = Dumper(fields_to_ignore=fields_to_ignore, vm=vm)
    return dumper.dump_anything(node)


class Dumper:
    """Convert AST nodes to simple Node representation"""

    fields_to_ignore: tuple[str, ...]
    vm: "SPyVM"

    def __init__(
        self,
        vm: "SPyVM",
        *,
        fields_to_ignore: Any = (),
    ) -> None:
        self.fields_to_ignore = tuple(fields_to_ignore)
        self.vm = vm

    def dump_anything(self, obj: Any) -> Node:
        if isinstance(obj, spy.ast.Node):
            return self.dump_spy_node(obj)
        elif isinstance(obj, py_ast.AST):
            return self.dump_py_node(obj)
        elif type(obj) is list:
            return self.dump_list(obj)
        elif type(obj) is Symbol:
            return self.dump_Symbol(obj)
        elif type(obj) is str:
            return Node("str", {"value": obj})
        else:
            return Node("literal", {"value": obj})

    def dump_spy_node(self, node: spy.ast.Node) -> Node:
        name = node.__class__.__name__
        fields = list(node.__class__.__dataclass_fields__)
        fields = [f for f in fields if f not in self.fields_to_ignore]
        return self._dump_node(node, name, fields)

    def dump_py_node(self, node: py_ast.AST) -> Node:
        name = "py:" + node.__class__.__name__
        fields = list(node.__class__._fields)
        fields = [f for f in fields if f not in self.fields_to_ignore]
        if isinstance(node, py_ast.Name):
            fields.append("is_var")
        return self._dump_node(node, name, fields)

    def dump_Symbol(self, sym: Symbol) -> Node:
        return Node(
            "Symbol",
            {
                "name": sym.name,
                "color": sym.color,
                "varkind": sym.varkind,
                "storage": sym.storage,
            },
        )

    def _dump_node(self, node: Any, name: str, fields: list[str]) -> Node:
        attrdict = {}
        # Add color information from VM if available
        if self.vm and self.vm.expr_color_map:
            color: Optional[Color] = self.vm.expr_color_map.get(node, None)
            if color:
                attrdict["_color"] = color

        # Process each field
        for field in fields:
            value = getattr(node, field)
            attrdict[field] = self.dump_anything(value)

        return Node(name, attrdict)

    def dump_list(self, lst: list[Any]) -> Node:
        items = [self.dump_anything(item) for item in lst]
        return items
