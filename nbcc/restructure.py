from dataclasses import dataclass, field
from pprint import pformat, pprint
from textwrap import indent
from typing import Any

from numba_scfg.core.datastructures.ast_transforms import WritableASTBlock
from numba_scfg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    SyntheticAssignment,
    SyntheticBranch,
)
from numba_scfg.core.datastructures.scfg import SCFG, RegionBlock

from .spy_ast import Node


def _format_stmt(stmt: Any) -> str:
    """Format a statement for display"""
    match stmt:
        case Node():
            args = [f"{k}={_format_stmt(v)}" for k, v in stmt.attrdict.items()]
            return f"{stmt.opname}({', '.join(args)})"
        case list():
            return str([_format_stmt(v) for v in stmt])
        case str():
            return repr(stmt)
        case _:
            return str(stmt)


def _show_region(region: RegionBlock, level: int = 0) -> str:
    """Show region structure for debugging"""
    buf: list[str] = []
    buf.append(f"+++REGION {region.name}")

    for k, blk in region.subregion.graph.items():
        buf.append(f"   BLOCK  {type(blk)} : {k}")
        if hasattr(blk, "body"):
            for stmt in blk.body:
                buf.append(indent(_format_stmt(stmt), " " * 4))
        else:
            buf.append(str(blk))

    if region.kind == "head":
        # Process siblings
        branches = []
        tails = []
        for reg in region.parent_region.subregion.graph.values():
            match reg.kind:
                case "head":
                    pass
                case "branch":
                    branches.append(reg)
                case "tail":
                    tails.append(reg)
                case _:
                    raise RuntimeError("unreachable")

        for br in branches:
            buf.append(_show_region(br, level=level + 1))

        assert len(tails) == 1
        [tail] = tails
        buf.append(_show_region(tail, level=level))

    return indent("\n".join(buf), prefix=" " * (level * 4))


def _create_basic_blocks(node: Node) -> dict[str, BasicBlock]:
    """Create basic blocks from AST node"""
    state = BasicBlockBuilderState.create_with_entry_block()
    _build_basic_blocks(state, node.attrdict["body"])
    return state.block_map


def _print_basic_blocks(block_map: dict[str, BasicBlock]) -> None:
    """Print basic blocks for debugging"""
    for name, block in block_map.items():
        print(f"BB {name!r}:  # {type(block)}")
        for stmt in block.body:
            print("    ", stmt)


def _create_and_process_scfg(block_map: dict[str, BasicBlock]) -> RegionBlock:
    """Create SCFG and process it, returning the head region"""
    scfg = SCFG(graph=block_map)
    scfg.restructure()
    # scfg.view()
    return scfg.graph[scfg.find_head()]


@dataclass(frozen=True)
class SpyBasicBlock(BasicBlock):
    body: list[Node] = field(default_factory=list)


def restructure(name: str, node: Node) -> None:
    """Restructure AST node into structured control flow graph"""
    print("---", name)

    # Create basic blocks from AST
    block_map = _create_basic_blocks(node)

    # Print basic blocks for debugging
    _print_basic_blocks(block_map)

    # Create and process SCFG
    region = _create_and_process_scfg(block_map)

    # Show final region structure
    print(_show_region(region))


@dataclass
class BasicBlockBuilderState:
    """Mutable state for building basic blocks"""

    block_map: dict[str, BasicBlock]
    current_block: BasicBlock
    end_block: str | None = None
    _counter: int = field(default=0, init=False)

    @classmethod
    def create_with_entry_block(cls) -> "BasicBlockBuilderState":
        """Create a new state with an entry block"""
        block_map: dict[str, BasicBlock] = {}
        state = cls(block_map, None)  # type: ignore
        entry_block = state.make_entry_block()
        state.current_block = entry_block
        state.add_basic_block(entry_block)
        return state

    def make_entry_block(self) -> SpyBasicBlock:
        """Create an entry basic block"""
        block = SpyBasicBlock(name="entry")
        self._counter += 1
        return block

    def make_block(self, prefix: str = "bb") -> SpyBasicBlock:
        """Create a new basic block with auto-generated name"""
        block = SpyBasicBlock(name=f"{prefix}.{self._counter}")
        self._counter += 1
        return block

    def replace_jump_targets_and_update(self, targets: tuple[str, ...]) -> None:
        """Replace jump targets for current basic block and update block map"""
        self.current_block = self.block_map[self.current_block.name] = (
            self.current_block.replace_jump_targets(targets)
        )

    def add_basic_block(self, block: BasicBlock) -> None:
        """Add a basic block to the block map"""
        self.block_map[block.name] = block

    def set_current_block(self, block: BasicBlock) -> None:
        """Set the current basic block"""
        self.current_block = block

    def create_and_register_blocks(
        self, count: int, prefix: str = "bb"
    ) -> list[BasicBlock]:
        """Create multiple basic blocks and register them in block map"""
        blocks = [self.make_block(prefix) for _ in range(count)]
        for block in blocks:
            self.add_basic_block(block)
        return blocks

    def create_child_state(
        self, current_block: BasicBlock, end_block: str | None = None
    ) -> "BasicBlockBuilderState":
        """Create a child state for recursive calls"""
        child_state = BasicBlockBuilderState(
            self.block_map, current_block, end_block or self.end_block
        )
        child_state._counter = self._counter
        return child_state

    def append_to_current_block(self, stmt: Node) -> None:
        """Append a statement to the current basic block"""
        self.current_block.body.append(stmt)

    def finalize_with_end_block(self) -> BasicBlock:
        """Finalize the current block by setting jump target to end block if present"""
        if self.end_block is not None:
            self.replace_jump_targets_and_update((self.end_block,))
        return self.current_block


def _handle_if_statement(state: BasicBlockBuilderState, stmt: Node) -> None:
    """Handle If statement by creating then/else/endif blocks and processing recursively"""
    state.append_to_current_block(stmt.attrdict["test"])
    then_block, else_block, endif_block = state.create_and_register_blocks(3)
    state.replace_jump_targets_and_update((then_block.name, else_block.name))

    # Process then branch
    then_state = state.create_child_state(then_block, endif_block.name)
    _build_basic_blocks(then_state, stmt.attrdict["then_body"])

    # Process else branch
    else_state = state.create_child_state(else_block, endif_block.name)
    _build_basic_blocks(else_state, stmt.attrdict["else_body"])

    state.set_current_block(endif_block)


def _build_basic_blocks(
    state: BasicBlockBuilderState,
    body: list[Node],
) -> BasicBlock:
    for stmt in body:
        match stmt.opname:
            case "If":
                _handle_if_statement(state, stmt)
            case "Return":
                state.append_to_current_block(stmt)
                return state.current_block
            case _:
                state.append_to_current_block(stmt)

    return state.finalize_with_end_block()
