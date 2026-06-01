"""
MLIR Utilities - Common functionality for type name encoding/decoding and MLIR operations.

This module provides utilities to avoid code duplication between different backends
and the frontend type system.
"""

import hashlib
import re


def encode_type_name(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:8]


def encode_asm_operation(fqn_parts: list[str]) -> str:
    joined = "$".join(fqn_parts)
    human = re.sub(r"[^a-zA-Z0-9_]", "_", joined)
    digest = hashlib.sha256(joined.encode()).hexdigest()[:8]
    return f"{human}_{digest}"


def parse_composite_type(tyname: str) -> list[str] | None:
    """
    Parse a composite type name that contains multiple type components.

    Composite types are encoded as "multivalues$type1|type2|type3|..."

    Args:
        tyname: The type name to parse

    Returns:
        List of individual type component names if it's a composite type,
        None if it's not a composite type
    """
    if not tyname.startswith("multivalues$"):
        return None

    _, _, raw_items = tyname.partition("$")
    items = raw_items.split("|")
    return items


def create_mlir_type_fqn(formatted_name: str):
    """
    Create an FQN for MLIR types with proper encoding.

    Args:
        formatted_name: The formatted type name

    Returns:
        FQN object with encoded qualifiers
    """
    from spy.fqn import FQN

    if formatted_name == "()":
        return FQN(["mlir", "type", "void"])
    else:
        humane_name = "_" + re.sub(r"[^a-zA-Z0-9_]", "", formatted_name)
        assert humane_name, formatted_name
        full_name = encode_type_name(formatted_name)

        return FQN(["mlir", "type", humane_name]).with_qualifiers([full_name])
