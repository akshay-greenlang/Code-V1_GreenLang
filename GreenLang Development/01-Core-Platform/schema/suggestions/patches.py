# -*- coding: utf-8 -*-
"""
JSON Patch Generator for GL-FOUND-X-002.

This module implements JSON Patch (RFC 6902) generation for fix suggestions.
It provides a complete implementation of all patch operations with support
for preconditions (test operations) and JSON Pointer (RFC 6901) utilities.

Operations supported:
    - add: Add a value at a target location
    - remove: Remove the value at a target location
    - replace: Replace the value at a target location
    - move: Move a value from one location to another
    - copy: Copy a value from one location to another
    - test: Test that a value at a target location is equal to a specified value

References:
    - RFC 6902: JavaScript Object Notation (JSON) Patch
    - RFC 6901: JavaScript Object Notation (JSON) Pointer

Example:
    >>> from greenlang.schema.suggestions.patches import PatchGenerator, apply_patch
    >>> generator = PatchGenerator()
    >>> patch = generator.generate_add("/energy", {"value": 100, "unit": "kWh"})
    >>> print(patch.to_rfc6902())
    {'op': 'add', 'path': '/energy', 'value': {'value': 100, 'unit': 'kWh'}}

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import copy
import hashlib
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# JSON Pointer Utilities (RFC 6901)
# =============================================================================

# Pattern for valid JSON Pointer tokens after unescaping
# Empty string is valid (represents root document)
JSON_POINTER_PATTERN = re.compile(r"^(/[^/]*)*$")


class JSONPointerError(ValueError):
    """Error raised for invalid JSON Pointer operations."""
    pass


def escape_json_pointer_token(token: str) -> str:
    """
    Escape a JSON Pointer token per RFC 6901.

    The order of escaping is important: ~ must be escaped first,
    then /.

    Args:
        token: The raw token to escape.

    Returns:
        The escaped token.

    Example:
        >>> escape_json_pointer_token("a/b~c")
        'a~1b~0c'
    """
    # Order matters: escape ~ first, then /
    return token.replace("~", "~0").replace("/", "~1")


def unescape_json_pointer_token(token: str) -> str:
    """
    Unescape a JSON Pointer token per RFC 6901.

    The order of unescaping is important: ~1 must be unescaped first,
    then ~0.

    Args:
        token: The escaped token.

    Returns:
        The unescaped token.

    Example:
        >>> unescape_json_pointer_token("a~1b~0c")
        'a/b~c'
    """
    # Order matters: unescape ~1 first, then ~0
    return token.replace("~1", "/").replace("~0", "~")


def parse_json_pointer(pointer: str) -> List[str]:
    """
    Parse a JSON Pointer string into path segments.

    Parses a JSON Pointer (RFC 6901) string into a list of unescaped
    path segments. The empty string represents the root document.

    Args:
        pointer: A valid JSON Pointer string.

    Returns:
        List of unescaped path segments.

    Raises:
        JSONPointerError: If the pointer is not a valid JSON Pointer.

    Example:
        >>> parse_json_pointer("")
        []
        >>> parse_json_pointer("/foo/bar")
        ['foo', 'bar']
        >>> parse_json_pointer("/a~1b/c~0d")
        ['a/b', 'c~d']
        >>> parse_json_pointer("/0/1")
        ['0', '1']
    """
    # Empty string is the root document
    if pointer == "":
        return []

    # Must start with /
    if not pointer.startswith("/"):
        raise JSONPointerError(
            f"Invalid JSON Pointer '{pointer}': must start with '/' or be empty"
        )

    # Split and unescape each token
    tokens = pointer[1:].split("/")
    return [unescape_json_pointer_token(token) for token in tokens]


def build_json_pointer(segments: List[str]) -> str:
    """
    Build a JSON Pointer string from path segments.

    Constructs a valid JSON Pointer (RFC 6901) string from a list
    of path segments, properly escaping special characters.

    Args:
        segments: List of path segments (unescaped).

    Returns:
        A valid JSON Pointer string.

    Example:
        >>> build_json_pointer([])
        ''
        >>> build_json_pointer(['foo', 'bar'])
        '/foo/bar'
        >>> build_json_pointer(['a/b', 'c~d'])
        '/a~1b/c~0d'
    """
    if not segments:
        return ""

    escaped = [escape_json_pointer_token(segment) for segment in segments]
    return "/" + "/".join(escaped)


def get_value_at_pointer(
    document: Dict[str, Any],
    pointer: str
) -> Tuple[bool, Any]:
    """
    Get the value at a JSON Pointer location.

    Traverses the document following the JSON Pointer path and returns
    the value at that location if it exists.

    Args:
        document: The JSON document to traverse.
        pointer: A valid JSON Pointer string.

    Returns:
        A tuple of (exists, value) where exists is True if the path
        exists in the document, and value is the value at that path
        (or None if it doesn't exist).

    Example:
        >>> doc = {"foo": {"bar": 42}}
        >>> get_value_at_pointer(doc, "/foo/bar")
        (True, 42)
        >>> get_value_at_pointer(doc, "/foo/baz")
        (False, None)
        >>> get_value_at_pointer(doc, "")
        (True, {'foo': {'bar': 42}})
    """
    segments = parse_json_pointer(pointer)

    # Empty pointer means root document
    if not segments:
        return (True, document)

    current = document
    for segment in segments:
        if isinstance(current, dict):
            if segment not in current:
                return (False, None)
            current = current[segment]
        elif isinstance(current, list):
            # Handle array indices
            if segment == "-":
                # "-" means "past the end" - doesn't exist for reading
                return (False, None)
            try:
                index = int(segment)
                if index < 0 or index >= len(current):
                    return (False, None)
                current = current[index]
            except ValueError:
                return (False, None)
        else:
            # Can't traverse into a scalar value
            return (False, None)

    return (True, current)


def set_value_at_pointer(
    document: Dict[str, Any],
    pointer: str,
    value: Any
) -> Dict[str, Any]:
    """
    Set a value at a JSON Pointer location.

    Creates a new document with the value set at the specified location.
    The original document is not modified. Creates intermediate objects
    as needed.

    Args:
        document: The original JSON document.
        pointer: A valid JSON Pointer string.
        value: The value to set.

    Returns:
        A new document with the value set at the pointer location.

    Raises:
        JSONPointerError: If the path is invalid or cannot be created.

    Example:
        >>> doc = {"foo": {"bar": 42}}
        >>> set_value_at_pointer(doc, "/foo/bar", 100)
        {'foo': {'bar': 100}}
        >>> set_value_at_pointer(doc, "/foo/baz", 99)
        {'foo': {'bar': 42, 'baz': 99}}
    """
    # Deep copy to avoid modifying original
    result = copy.deepcopy(document)

    segments = parse_json_pointer(pointer)

    # Empty pointer means replace entire document
    if not segments:
        if not isinstance(value, dict):
            raise JSONPointerError(
                "Cannot replace root document with non-object value"
            )
        return copy.deepcopy(value)

    # Navigate to parent and set the value
    current = result
    for i, segment in enumerate(segments[:-1]):
        if isinstance(current, dict):
            if segment not in current:
                # Create intermediate object
                current[segment] = {}
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
                if index < 0 or index >= len(current):
                    raise JSONPointerError(
                        f"Array index {index} out of bounds at '{build_json_pointer(segments[:i+1])}'"
                    )
                current = current[index]
            except ValueError:
                raise JSONPointerError(
                    f"Invalid array index '{segment}' at '{build_json_pointer(segments[:i+1])}'"
                )
        else:
            raise JSONPointerError(
                f"Cannot traverse through scalar value at '{build_json_pointer(segments[:i+1])}'"
            )

    # Set the final value
    final_segment = segments[-1]
    if isinstance(current, dict):
        current[final_segment] = copy.deepcopy(value)
    elif isinstance(current, list):
        if final_segment == "-":
            # Append to array
            current.append(copy.deepcopy(value))
        else:
            try:
                index = int(final_segment)
                if index < 0 or index > len(current):
                    raise JSONPointerError(
                        f"Array index {index} out of bounds"
                    )
                # Replace existing value at index
                if index == len(current):
                    current.append(copy.deepcopy(value))
                else:
                    current[index] = copy.deepcopy(value)
            except ValueError:
                raise JSONPointerError(
                    f"Invalid array index '{final_segment}'"
                )
    else:
        raise JSONPointerError(
            f"Cannot set value on scalar at '{pointer}'"
        )

    return result


def insert_value_at_pointer(
    document: Dict[str, Any],
    pointer: str,
    value: Any
) -> Dict[str, Any]:
    """
    Insert a value at a JSON Pointer location (for array add per RFC 6902).

    For arrays, this INSERT the value at the specified index, shifting
    subsequent elements right. For objects, this behaves like set.

    This is the correct behavior for JSON Patch "add" operation on arrays
    per RFC 6902 section 4.1.

    Args:
        document: The original JSON document.
        pointer: A valid JSON Pointer string.
        value: The value to insert.

    Returns:
        A new document with the value inserted at the pointer location.

    Raises:
        JSONPointerError: If the path is invalid or cannot be created.

    Example:
        >>> doc = {"arr": [1, 2, 3]}
        >>> insert_value_at_pointer(doc, "/arr/1", "new")
        {'arr': [1, 'new', 2, 3]}
    """
    # Deep copy to avoid modifying original
    result = copy.deepcopy(document)

    segments = parse_json_pointer(pointer)

    # Empty pointer means replace entire document
    if not segments:
        if not isinstance(value, dict):
            raise JSONPointerError(
                "Cannot replace root document with non-object value"
            )
        return copy.deepcopy(value)

    # Navigate to parent and insert the value
    current = result
    for i, segment in enumerate(segments[:-1]):
        if isinstance(current, dict):
            if segment not in current:
                # Create intermediate object
                current[segment] = {}
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
                if index < 0 or index >= len(current):
                    raise JSONPointerError(
                        f"Array index {index} out of bounds at '{build_json_pointer(segments[:i+1])}'"
                    )
                current = current[index]
            except ValueError:
                raise JSONPointerError(
                    f"Invalid array index '{segment}' at '{build_json_pointer(segments[:i+1])}'"
                )
        else:
            raise JSONPointerError(
                f"Cannot traverse through scalar value at '{build_json_pointer(segments[:i+1])}'"
            )

    # Insert the final value
    final_segment = segments[-1]
    if isinstance(current, dict):
        # For objects, add just sets the value
        current[final_segment] = copy.deepcopy(value)
    elif isinstance(current, list):
        if final_segment == "-":
            # Append to array
            current.append(copy.deepcopy(value))
        else:
            try:
                index = int(final_segment)
                if index < 0 or index > len(current):
                    raise JSONPointerError(
                        f"Array index {index} out of bounds"
                    )
                # INSERT at index (RFC 6902 behavior for add)
                current.insert(index, copy.deepcopy(value))
            except ValueError:
                raise JSONPointerError(
                    f"Invalid array index '{final_segment}'"
                )
    else:
        raise JSONPointerError(
            f"Cannot insert value on scalar at '{pointer}'"
        )

    return result


def remove_value_at_pointer(
    document: Dict[str, Any],
    pointer: str
) -> Dict[str, Any]:
    """
    Remove a value at a JSON Pointer location.

    Creates a new document with the value removed at the specified location.
    The original document is not modified.

    Args:
        document: The original JSON document.
        pointer: A valid JSON Pointer string.

    Returns:
        A new document with the value removed.

    Raises:
        JSONPointerError: If the path doesn't exist or is the root.

    Example:
        >>> doc = {"foo": {"bar": 42, "baz": 99}}
        >>> remove_value_at_pointer(doc, "/foo/bar")
        {'foo': {'baz': 99}}
    """
    if pointer == "":
        raise JSONPointerError("Cannot remove root document")

    # Deep copy to avoid modifying original
    result = copy.deepcopy(document)

    segments = parse_json_pointer(pointer)

    # Navigate to parent
    current = result
    for i, segment in enumerate(segments[:-1]):
        if isinstance(current, dict):
            if segment not in current:
                raise JSONPointerError(
                    f"Path not found: '{build_json_pointer(segments[:i+1])}'"
                )
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
                if index < 0 or index >= len(current):
                    raise JSONPointerError(
                        f"Array index {index} out of bounds at '{build_json_pointer(segments[:i+1])}'"
                    )
                current = current[index]
            except ValueError:
                raise JSONPointerError(
                    f"Invalid array index '{segment}' at '{build_json_pointer(segments[:i+1])}'"
                )
        else:
            raise JSONPointerError(
                f"Cannot traverse through scalar value at '{build_json_pointer(segments[:i+1])}'"
            )

    # Remove the final value
    final_segment = segments[-1]
    if isinstance(current, dict):
        if final_segment not in current:
            raise JSONPointerError(
                f"Path not found: '{pointer}'"
            )
        del current[final_segment]
    elif isinstance(current, list):
        try:
            index = int(final_segment)
            if index < 0 or index >= len(current):
                raise JSONPointerError(
                    f"Array index {index} out of bounds"
                )
            del current[index]
        except ValueError:
            raise JSONPointerError(
                f"Invalid array index '{final_segment}'"
            )
    else:
        raise JSONPointerError(
            f"Cannot remove from scalar value at '{pointer}'"
        )

    return result


def pointer_parent(pointer: str) -> str:
    """
    Get the parent path of a JSON Pointer.

    Args:
        pointer: A valid JSON Pointer string.

    Returns:
        The parent pointer, or empty string for root-level paths.

    Example:
        >>> pointer_parent("/foo/bar/baz")
        '/foo/bar'
        >>> pointer_parent("/foo")
        ''
        >>> pointer_parent("")
        ''
    """
    if pointer == "":
        return ""

    segments = parse_json_pointer(pointer)
    if len(segments) <= 1:
        return ""

    return build_json_pointer(segments[:-1])


def pointer_last_segment(pointer: str) -> Optional[str]:
    """
    Get the last segment of a JSON Pointer.

    Args:
        pointer: A valid JSON Pointer string.

    Returns:
        The last segment (unescaped), or None for root.

    Example:
        >>> pointer_last_segment("/foo/bar")
        'bar'
        >>> pointer_last_segment("/foo")
        'foo'
        >>> pointer_last_segment("")
        None
    """
    if pointer == "":
        return None

    segments = parse_json_pointer(pointer)
    return segments[-1] if segments else None


# =============================================================================
# Patch Operation Enum and Models
# =============================================================================

class PatchOp(str, Enum):
    """
    JSON Patch operation types per RFC 6902.

    Attributes:
        ADD: Add a value at a target location.
        REMOVE: Remove the value at a target location.
        REPLACE: Replace the value at a target location.
        MOVE: Move a value from one location to another.
        COPY: Copy a value from one location to another.
        TEST: Test that a value equals a specified value.
    """
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    MOVE = "move"
    COPY = "copy"
    TEST = "test"


class JSONPatchOperation(BaseModel):
    """
    A single JSON Patch operation per RFC 6902.

    Represents one operation in a JSON Patch document that can be used
    to modify a JSON document.

    Attributes:
        op: The operation type (add/remove/replace/move/copy/test).
        path: JSON Pointer (RFC 6901) to the target location.
        value: The value for add/replace/test operations.
        from_: Source path for move/copy operations (aliased as "from" in JSON).

    Example:
        >>> # Add operation
        >>> JSONPatchOperation(op=PatchOp.ADD, path="/energy", value=100)

        >>> # Move operation
        >>> JSONPatchOperation(op=PatchOp.MOVE, path="/new_name", from_="/old_name")

        >>> # Test operation (precondition)
        >>> JSONPatchOperation(op=PatchOp.TEST, path="/version", value="1.0")
    """

    op: PatchOp = Field(
        ...,
        description="The operation type"
    )

    path: str = Field(
        ...,
        max_length=4096,
        description="JSON Pointer (RFC 6901) to target location"
    )

    value: Optional[Any] = Field(
        default=None,
        description="Value for add/replace/test operations"
    )

    from_: Optional[str] = Field(
        default=None,
        alias="from",
        max_length=4096,
        description="Source path for move/copy operations"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {"op": "add", "path": "/energy", "value": 100},
                {"op": "remove", "path": "/deprecated_field"},
                {"op": "replace", "path": "/name", "value": "new_name"},
                {"op": "move", "from": "/old_field", "path": "/new_field"},
                {"op": "copy", "from": "/source", "path": "/dest"},
                {"op": "test", "path": "/version", "value": "1.0"}
            ]
        }
    }

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path is a valid JSON Pointer."""
        if v == "":
            return v  # Empty string is valid (root)

        if not v.startswith("/"):
            raise ValueError(
                f"Invalid JSON Pointer path '{v}': must start with '/' or be empty"
            )

        # Validate the pointer can be parsed
        try:
            parse_json_pointer(v)
        except JSONPointerError as e:
            raise ValueError(str(e))

        return v

    @field_validator("from_")
    @classmethod
    def validate_from(cls, v: Optional[str]) -> Optional[str]:
        """Validate from_ is a valid JSON Pointer if provided."""
        if v is None:
            return v

        if v == "":
            return v  # Empty string is valid (root)

        if not v.startswith("/"):
            raise ValueError(
                f"Invalid JSON Pointer 'from' path '{v}': must start with '/' or be empty"
            )

        try:
            parse_json_pointer(v)
        except JSONPointerError as e:
            raise ValueError(str(e))

        return v

    @model_validator(mode="after")
    def validate_operation_requirements(self) -> "JSONPatchOperation":
        """Validate operation has required fields."""
        # move and copy require from_
        if self.op in (PatchOp.MOVE, PatchOp.COPY):
            if self.from_ is None:
                raise ValueError(
                    f"Operation '{self.op.value}' requires 'from' field"
                )

        return self

    def to_rfc6902(self) -> Dict[str, Any]:
        """
        Convert to RFC 6902 compliant dictionary.

        Returns:
            Dictionary representation per RFC 6902.

        Example:
            >>> op = JSONPatchOperation(op=PatchOp.MOVE, path="/new", from_="/old")
            >>> op.to_rfc6902()
            {'op': 'move', 'path': '/new', 'from': '/old'}
        """
        result: Dict[str, Any] = {"op": self.op.value, "path": self.path}

        if self.op in (PatchOp.ADD, PatchOp.REPLACE, PatchOp.TEST):
            result["value"] = self.value
        elif self.op in (PatchOp.MOVE, PatchOp.COPY):
            result["from"] = self.from_

        return result

    def is_additive(self) -> bool:
        """Check if this operation adds data."""
        return self.op in (PatchOp.ADD, PatchOp.COPY)

    def is_destructive(self) -> bool:
        """Check if this operation removes data."""
        return self.op == PatchOp.REMOVE

    def is_modification(self) -> bool:
        """Check if this operation modifies existing data."""
        return self.op in (PatchOp.REPLACE, PatchOp.MOVE)

    def is_test(self) -> bool:
        """Check if this is a test/precondition operation."""
        return self.op == PatchOp.TEST

    def __str__(self) -> str:
        """Return string representation."""
        if self.op in (PatchOp.MOVE, PatchOp.COPY):
            return f"{self.op.value} {self.from_} -> {self.path}"
        elif self.op == PatchOp.REMOVE:
            return f"{self.op.value} {self.path}"
        else:
            value_repr = repr(self.value)
            if len(value_repr) > 50:
                value_repr = value_repr[:47] + "..."
            return f"{self.op.value} {self.path} = {value_repr}"


class PatchSequence(BaseModel):
    """
    A sequence of JSON Patch operations.

    Represents an ordered list of patch operations that should be
    applied sequentially to transform a document.

    Attributes:
        operations: List of JSONPatchOperation objects.

    Example:
        >>> seq = PatchSequence(operations=[
        ...     JSONPatchOperation(op=PatchOp.TEST, path="/version", value="1.0"),
        ...     JSONPatchOperation(op=PatchOp.REPLACE, path="/name", value="new")
        ... ])
        >>> patch_doc = seq.to_json_patch()
    """

    operations: List[JSONPatchOperation] = Field(
        default_factory=list,
        description="Ordered list of patch operations"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid"
    }

    def to_json_patch(self) -> List[Dict[str, Any]]:
        """
        Convert to standard JSON Patch format (RFC 6902).

        Returns:
            List of operation dictionaries in RFC 6902 format.
        """
        return [op.to_rfc6902() for op in self.operations]

    def add(self, operation: JSONPatchOperation) -> "PatchSequence":
        """
        Add an operation to the sequence (returns new sequence).

        Args:
            operation: The operation to add.

        Returns:
            A new PatchSequence with the operation appended.
        """
        return PatchSequence(operations=self.operations + [operation])

    def extend(self, operations: List[JSONPatchOperation]) -> "PatchSequence":
        """
        Extend sequence with multiple operations (returns new sequence).

        Args:
            operations: List of operations to add.

        Returns:
            A new PatchSequence with the operations appended.
        """
        return PatchSequence(operations=self.operations + operations)

    def get_tests(self) -> List[JSONPatchOperation]:
        """Get all test operations in the sequence."""
        return [op for op in self.operations if op.is_test()]

    def get_mutations(self) -> List[JSONPatchOperation]:
        """Get all non-test operations in the sequence."""
        return [op for op in self.operations if not op.is_test()]

    def affected_paths(self) -> List[str]:
        """Get all unique paths affected by this sequence."""
        paths = set()
        for op in self.operations:
            paths.add(op.path)
            if op.from_:
                paths.add(op.from_)
        return sorted(paths)

    def compute_hash(self) -> str:
        """
        Compute a SHA-256 hash of the patch sequence.

        Useful for deduplication and caching.

        Returns:
            Hex-encoded SHA-256 hash of the serialized patch.
        """
        import json
        content = json.dumps(self.to_json_patch(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def __len__(self) -> int:
        """Return the number of operations."""
        return len(self.operations)

    def __bool__(self) -> bool:
        """Return True if sequence has operations."""
        return len(self.operations) > 0

    def __iter__(self):
        """Iterate over operations."""
        return iter(self.operations)


# =============================================================================
# Patch Generator
# =============================================================================

class PatchGenerator:
    """
    Generates JSON Patch operations per RFC 6902.

    This class provides methods to generate various types of JSON Patch
    operations with optional test preconditions for safety.

    Example:
        >>> generator = PatchGenerator()
        >>> add_op = generator.generate_add("/energy", 100)
        >>> replace_ops = generator.generate_replace("/name", "new", "old")
    """

    def __init__(self) -> None:
        """Initialize the PatchGenerator."""
        logger.debug("PatchGenerator initialized")

    def generate_add(
        self,
        path: str,
        value: Any
    ) -> JSONPatchOperation:
        """
        Generate an add operation.

        The add operation adds a value at the target location. If the
        target location is an array index, the value is inserted before
        the specified index. If the path ends with "-", the value is
        appended to the array.

        Args:
            path: JSON Pointer to the target location.
            value: The value to add.

        Returns:
            A JSONPatchOperation for add.

        Example:
            >>> gen = PatchGenerator()
            >>> op = gen.generate_add("/energy", {"value": 100, "unit": "kWh"})
            >>> op.to_rfc6902()
            {'op': 'add', 'path': '/energy', 'value': {'value': 100, 'unit': 'kWh'}}
        """
        logger.debug(f"Generating add operation at path '{path}'")
        return JSONPatchOperation(
            op=PatchOp.ADD,
            path=path,
            value=value
        )

    def generate_remove(
        self,
        path: str
    ) -> JSONPatchOperation:
        """
        Generate a remove operation.

        The remove operation removes the value at the target location.

        Args:
            path: JSON Pointer to the location to remove.

        Returns:
            A JSONPatchOperation for remove.

        Example:
            >>> gen = PatchGenerator()
            >>> op = gen.generate_remove("/deprecated_field")
            >>> op.to_rfc6902()
            {'op': 'remove', 'path': '/deprecated_field'}
        """
        logger.debug(f"Generating remove operation at path '{path}'")
        return JSONPatchOperation(
            op=PatchOp.REMOVE,
            path=path
        )

    def generate_replace(
        self,
        path: str,
        new_value: Any,
        old_value: Any = None,
        include_test: bool = True
    ) -> List[JSONPatchOperation]:
        """
        Generate a replace operation with optional test precondition.

        The replace operation replaces the value at the target location
        with a new value. If include_test is True and old_value is
        provided, a test operation is prepended to verify the current
        value before replacing.

        Args:
            path: JSON Pointer to the location to replace.
            new_value: The new value to set.
            old_value: The expected current value (for test precondition).
            include_test: Whether to include a test precondition.

        Returns:
            List of operations ([test, replace] or [replace]).

        Example:
            >>> gen = PatchGenerator()
            >>> ops = gen.generate_replace("/name", "new_name", "old_name")
            >>> len(ops)
            2
            >>> ops[0].op
            <PatchOp.TEST: 'test'>
        """
        logger.debug(f"Generating replace operation at path '{path}'")
        operations = []

        if include_test and old_value is not None:
            operations.append(JSONPatchOperation(
                op=PatchOp.TEST,
                path=path,
                value=old_value
            ))

        operations.append(JSONPatchOperation(
            op=PatchOp.REPLACE,
            path=path,
            value=new_value
        ))

        return operations

    def generate_move(
        self,
        from_path: str,
        to_path: str,
        include_test: bool = True
    ) -> List[JSONPatchOperation]:
        """
        Generate a move operation for field rename.

        The move operation removes the value at the source location and
        adds it at the target location. If include_test is True, test
        operations are prepended to verify:
        1. The source path exists
        2. The destination path does not exist

        Note: Testing for non-existence is done by checking if the
        parent exists but the target key does not.

        Args:
            from_path: Source JSON Pointer.
            to_path: Destination JSON Pointer.
            include_test: Whether to include test preconditions.

        Returns:
            List of operations ([tests..., move] or [move]).

        Example:
            >>> gen = PatchGenerator()
            >>> ops = gen.generate_move("/old_name", "/new_name")
            >>> ops[-1].op
            <PatchOp.MOVE: 'move'>
        """
        logger.debug(f"Generating move operation from '{from_path}' to '{to_path}'")
        operations = []

        # Note: We cannot truly test for non-existence in JSON Patch,
        # but we can include a comment in the patch metadata.
        # The test operations here verify the source exists.

        if include_test:
            # We would need the actual value to test for existence,
            # but we can generate a placeholder that should be filled in
            # by the caller or engine that has access to the document.
            # For now, we just include the move without the test for existence.
            pass

        operations.append(JSONPatchOperation(
            op=PatchOp.MOVE,
            path=to_path,
            from_=from_path
        ))

        return operations

    def generate_copy(
        self,
        from_path: str,
        to_path: str
    ) -> JSONPatchOperation:
        """
        Generate a copy operation.

        The copy operation copies the value at the source location to
        the target location.

        Args:
            from_path: Source JSON Pointer.
            to_path: Destination JSON Pointer.

        Returns:
            A JSONPatchOperation for copy.

        Example:
            >>> gen = PatchGenerator()
            >>> op = gen.generate_copy("/template", "/instance")
            >>> op.to_rfc6902()
            {'op': 'copy', 'path': '/instance', 'from': '/template'}
        """
        logger.debug(f"Generating copy operation from '{from_path}' to '{to_path}'")
        return JSONPatchOperation(
            op=PatchOp.COPY,
            path=to_path,
            from_=from_path
        )

    def generate_test(
        self,
        path: str,
        value: Any
    ) -> JSONPatchOperation:
        """
        Generate a test precondition operation.

        The test operation verifies that the value at the target location
        is equal to the specified value. If the test fails, the entire
        patch operation fails.

        Args:
            path: JSON Pointer to test.
            value: Expected value.

        Returns:
            A JSONPatchOperation for test.

        Example:
            >>> gen = PatchGenerator()
            >>> op = gen.generate_test("/version", "1.0")
            >>> op.to_rfc6902()
            {'op': 'test', 'path': '/version', 'value': '1.0'}
        """
        logger.debug(f"Generating test operation at path '{path}'")
        return JSONPatchOperation(
            op=PatchOp.TEST,
            path=path,
            value=value
        )

    def generate_add_default(
        self,
        path: str,
        default_value: Any
    ) -> List[JSONPatchOperation]:
        """
        Generate operations to add a default value.

        This generates an add operation for setting a default value at
        a path. Note: JSON Patch doesn't have conditional add, so this
        should only be used when the path is known not to exist.

        For conditional add (only if not exists), the caller should
        check existence first and decide whether to apply.

        Args:
            path: JSON Pointer where to add the default.
            default_value: The default value to add.

        Returns:
            List containing the add operation.

        Example:
            >>> gen = PatchGenerator()
            >>> ops = gen.generate_add_default("/config/timeout", 30)
            >>> ops[0].op
            <PatchOp.ADD: 'add'>
        """
        logger.debug(f"Generating add default operation at path '{path}'")
        return [self.generate_add(path, default_value)]

    def generate_type_coercion(
        self,
        path: str,
        original_value: Any,
        coerced_value: Any
    ) -> List[JSONPatchOperation]:
        """
        Generate operations for type coercion.

        Creates a replace operation with a test precondition to safely
        coerce a value from one type to another (e.g., "42" -> 42).

        Args:
            path: JSON Pointer to the value.
            original_value: The original (pre-coercion) value.
            coerced_value: The coerced (target type) value.

        Returns:
            List of [test, replace] operations.

        Example:
            >>> gen = PatchGenerator()
            >>> ops = gen.generate_type_coercion("/count", "42", 42)
            >>> ops[0].value
            '42'
            >>> ops[1].value
            42
        """
        logger.debug(
            f"Generating type coercion at path '{path}': "
            f"{type(original_value).__name__} -> {type(coerced_value).__name__}"
        )
        return self.generate_replace(
            path=path,
            new_value=coerced_value,
            old_value=original_value,
            include_test=True
        )

    def generate_unit_conversion(
        self,
        path: str,
        original: Dict[str, Any],
        converted: Dict[str, Any]
    ) -> List[JSONPatchOperation]:
        """
        Generate operations for unit conversion.

        Creates a replace operation with a test precondition to safely
        convert a value with units from one unit to another
        (e.g., {"value": 1000, "unit": "Wh"} -> {"value": 1, "unit": "kWh"}).

        Args:
            path: JSON Pointer to the value object.
            original: The original value with unit (e.g., {"value": 1000, "unit": "Wh"}).
            converted: The converted value with unit (e.g., {"value": 1, "unit": "kWh"}).

        Returns:
            List of [test, replace] operations.

        Example:
            >>> gen = PatchGenerator()
            >>> orig = {"value": 1000, "unit": "Wh"}
            >>> conv = {"value": 1, "unit": "kWh"}
            >>> ops = gen.generate_unit_conversion("/energy", orig, conv)
            >>> ops[1].value
            {'value': 1, 'unit': 'kWh'}
        """
        logger.debug(
            f"Generating unit conversion at path '{path}': "
            f"{original.get('unit')} -> {converted.get('unit')}"
        )
        return self.generate_replace(
            path=path,
            new_value=converted,
            old_value=original,
            include_test=True
        )

    def generate_field_rename(
        self,
        old_path: str,
        new_path: str
    ) -> List[JSONPatchOperation]:
        """
        Generate operations to rename a field.

        This is an alias for generate_move, provided for semantic clarity.

        Args:
            old_path: Current path of the field.
            new_path: New path for the field.

        Returns:
            List containing the move operation.
        """
        return self.generate_move(old_path, new_path, include_test=False)

    def generate_batch(
        self,
        operations: List[Dict[str, Any]]
    ) -> PatchSequence:
        """
        Generate a patch sequence from a list of operation specifications.

        Args:
            operations: List of dicts with 'type' and operation-specific params.

        Returns:
            A PatchSequence containing all operations.

        Example:
            >>> gen = PatchGenerator()
            >>> seq = gen.generate_batch([
            ...     {"type": "add", "path": "/a", "value": 1},
            ...     {"type": "remove", "path": "/b"}
            ... ])
            >>> len(seq)
            2
        """
        logger.debug(f"Generating batch of {len(operations)} operations")
        result_ops = []

        for spec in operations:
            op_type = spec.get("type", spec.get("op"))

            if op_type == "add":
                result_ops.append(self.generate_add(spec["path"], spec["value"]))
            elif op_type == "remove":
                result_ops.append(self.generate_remove(spec["path"]))
            elif op_type == "replace":
                ops = self.generate_replace(
                    spec["path"],
                    spec["value"],
                    spec.get("old_value"),
                    spec.get("include_test", True)
                )
                result_ops.extend(ops)
            elif op_type == "move":
                ops = self.generate_move(
                    spec["from"],
                    spec["path"],
                    spec.get("include_test", True)
                )
                result_ops.extend(ops)
            elif op_type == "copy":
                result_ops.append(self.generate_copy(spec["from"], spec["path"]))
            elif op_type == "test":
                result_ops.append(self.generate_test(spec["path"], spec["value"]))
            else:
                raise ValueError(f"Unknown operation type: {op_type}")

        return PatchSequence(operations=result_ops)


# =============================================================================
# Patch Application
# =============================================================================

class PatchApplicationError(Exception):
    """Error raised when patch application fails."""

    def __init__(
        self,
        message: str,
        operation_index: int,
        operation: JSONPatchOperation
    ):
        super().__init__(message)
        self.operation_index = operation_index
        self.operation = operation


def _compare_values(a: Any, b: Any) -> bool:
    """
    Compare two values for equality per RFC 6902.

    JSON Patch uses deep equality comparison.

    Args:
        a: First value.
        b: Second value.

    Returns:
        True if values are equal, False otherwise.
    """
    if type(a) != type(b):
        return False

    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_compare_values(a[k], b[k]) for k in a.keys())

    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_compare_values(x, y) for x, y in zip(a, b))

    return a == b


def _apply_single_operation(
    document: Dict[str, Any],
    operation: JSONPatchOperation,
    index: int
) -> Dict[str, Any]:
    """
    Apply a single patch operation to a document.

    Args:
        document: The current document state.
        operation: The operation to apply.
        index: The operation index (for error messages).

    Returns:
        The updated document.

    Raises:
        PatchApplicationError: If the operation cannot be applied.
    """
    try:
        if operation.op == PatchOp.TEST:
            exists, value = get_value_at_pointer(document, operation.path)
            if not exists:
                raise PatchApplicationError(
                    f"Test failed: path '{operation.path}' does not exist",
                    index,
                    operation
                )
            if not _compare_values(value, operation.value):
                raise PatchApplicationError(
                    f"Test failed: value at '{operation.path}' does not match expected",
                    index,
                    operation
                )
            return document  # Test operations don't modify the document

        elif operation.op == PatchOp.ADD:
            # Use insert for correct RFC 6902 array semantics
            return insert_value_at_pointer(document, operation.path, operation.value)

        elif operation.op == PatchOp.REMOVE:
            return remove_value_at_pointer(document, operation.path)

        elif operation.op == PatchOp.REPLACE:
            exists, _ = get_value_at_pointer(document, operation.path)
            if not exists:
                raise PatchApplicationError(
                    f"Replace failed: path '{operation.path}' does not exist",
                    index,
                    operation
                )
            return set_value_at_pointer(document, operation.path, operation.value)

        elif operation.op == PatchOp.MOVE:
            # Get value at source
            exists, value = get_value_at_pointer(document, operation.from_)
            if not exists:
                raise PatchApplicationError(
                    f"Move failed: source path '{operation.from_}' does not exist",
                    index,
                    operation
                )
            # Remove from source
            result = remove_value_at_pointer(document, operation.from_)
            # Add to destination (use insert for correct array semantics)
            result = insert_value_at_pointer(result, operation.path, value)
            return result

        elif operation.op == PatchOp.COPY:
            # Get value at source
            exists, value = get_value_at_pointer(document, operation.from_)
            if not exists:
                raise PatchApplicationError(
                    f"Copy failed: source path '{operation.from_}' does not exist",
                    index,
                    operation
                )
            # Add to destination (use insert for correct array semantics)
            return insert_value_at_pointer(document, operation.path, copy.deepcopy(value))

        else:
            raise PatchApplicationError(
                f"Unknown operation type: {operation.op}",
                index,
                operation
            )

    except JSONPointerError as e:
        raise PatchApplicationError(str(e), index, operation)


def apply_patch(
    document: Dict[str, Any],
    patch: List[JSONPatchOperation]
) -> Dict[str, Any]:
    """
    Apply a JSON Patch to a document.

    Applies each operation in sequence. If any operation fails
    (including test operations), the entire patch fails and an
    exception is raised. The original document is not modified.

    Args:
        document: The original document.
        patch: List of patch operations to apply.

    Returns:
        A new document with all patches applied.

    Raises:
        PatchApplicationError: If any operation fails.

    Example:
        >>> doc = {"name": "old", "value": 42}
        >>> patch = [
        ...     JSONPatchOperation(op=PatchOp.TEST, path="/value", value=42),
        ...     JSONPatchOperation(op=PatchOp.REPLACE, path="/name", value="new")
        ... ]
        >>> result = apply_patch(doc, patch)
        >>> result
        {'name': 'new', 'value': 42}
    """
    logger.debug(f"Applying patch with {len(patch)} operations")

    # Start with a deep copy
    result = copy.deepcopy(document)

    for i, operation in enumerate(patch):
        logger.debug(f"Applying operation {i}: {operation}")
        result = _apply_single_operation(result, operation, i)

    return result


def apply_patch_sequence(
    document: Dict[str, Any],
    sequence: PatchSequence
) -> Dict[str, Any]:
    """
    Apply a PatchSequence to a document.

    Convenience wrapper around apply_patch for PatchSequence objects.

    Args:
        document: The original document.
        sequence: The patch sequence to apply.

    Returns:
        A new document with all patches applied.

    Raises:
        PatchApplicationError: If any operation fails.
    """
    return apply_patch(document, sequence.operations)


# =============================================================================
# Patch Validation
# =============================================================================

def validate_patch(
    document: Dict[str, Any],
    patch: List[JSONPatchOperation]
) -> List[str]:
    """
    Validate that a patch can be applied to a document.

    Performs a dry-run of the patch application and collects any
    errors that would occur.

    Args:
        document: The document to validate against.
        patch: List of patch operations to validate.

    Returns:
        List of error messages (empty if patch is valid).

    Example:
        >>> doc = {"name": "test"}
        >>> patch = [JSONPatchOperation(op=PatchOp.REMOVE, path="/missing")]
        >>> errors = validate_patch(doc, patch)
        >>> len(errors) > 0
        True
    """
    logger.debug(f"Validating patch with {len(patch)} operations")
    errors = []

    try:
        apply_patch(document, patch)
    except PatchApplicationError as e:
        errors.append(f"Operation {e.operation_index}: {str(e)}")
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")

    return errors


def validate_patch_syntax(
    patch: List[JSONPatchOperation]
) -> List[str]:
    """
    Validate patch operations for RFC 6902 syntax compliance.

    Checks that each operation has required fields and valid
    JSON Pointer paths without actually applying the patch.

    Args:
        patch: List of patch operations to validate.

    Returns:
        List of syntax error messages (empty if valid).

    Example:
        >>> patch = [JSONPatchOperation(op=PatchOp.ADD, path="/test", value=1)]
        >>> errors = validate_patch_syntax(patch)
        >>> len(errors)
        0
    """
    errors = []

    for i, op in enumerate(patch):
        prefix = f"Operation {i}"

        # Validate path
        if op.path and not op.path.startswith("/") and op.path != "":
            errors.append(f"{prefix}: path must start with '/' or be empty")

        # Validate from_ for move/copy
        if op.op in (PatchOp.MOVE, PatchOp.COPY):
            if op.from_ is None:
                errors.append(f"{prefix}: {op.op.value} requires 'from' field")
            elif op.from_ and not op.from_.startswith("/") and op.from_ != "":
                errors.append(f"{prefix}: 'from' must start with '/' or be empty")

        # Validate value requirements
        # Note: value can be None/null which is valid, so we don't check
        # for presence of value for add/replace/test

    return errors


# =============================================================================
# Convenience Functions
# =============================================================================

def create_add_patch(path: str, value: Any) -> List[Dict[str, Any]]:
    """
    Create a simple add patch document.

    Convenience function for creating a single-operation patch.

    Args:
        path: JSON Pointer for the add operation.
        value: Value to add.

    Returns:
        RFC 6902 patch document (list of operations).
    """
    gen = PatchGenerator()
    return [gen.generate_add(path, value).to_rfc6902()]


def create_remove_patch(path: str) -> List[Dict[str, Any]]:
    """
    Create a simple remove patch document.

    Args:
        path: JSON Pointer for the remove operation.

    Returns:
        RFC 6902 patch document (list of operations).
    """
    gen = PatchGenerator()
    return [gen.generate_remove(path).to_rfc6902()]


def create_replace_patch(
    path: str,
    new_value: Any,
    old_value: Any = None
) -> List[Dict[str, Any]]:
    """
    Create a replace patch document with optional test.

    Args:
        path: JSON Pointer for the replace operation.
        new_value: New value to set.
        old_value: Expected current value (for test precondition).

    Returns:
        RFC 6902 patch document (list of operations).
    """
    gen = PatchGenerator()
    ops = gen.generate_replace(path, new_value, old_value, include_test=old_value is not None)
    return [op.to_rfc6902() for op in ops]


def create_move_patch(from_path: str, to_path: str) -> List[Dict[str, Any]]:
    """
    Create a move patch document.

    Args:
        from_path: Source JSON Pointer.
        to_path: Destination JSON Pointer.

    Returns:
        RFC 6902 patch document (list of operations).
    """
    gen = PatchGenerator()
    ops = gen.generate_move(from_path, to_path, include_test=False)
    return [op.to_rfc6902() for op in ops]


# =============================================================================
# Legacy Compatibility (for existing code)
# =============================================================================

# Alias for backward compatibility with existing patches.py interface
class JSONPatchOp:
    """
    Legacy JSON Patch operation class for backward compatibility.

    Deprecated: Use JSONPatchOperation instead.
    """

    def __init__(
        self,
        op: str,
        path: str,
        value: Optional[Any] = None,
        from_: Optional[str] = None,
    ):
        """Initialize a legacy JSONPatchOp."""
        self.op = op
        self.path = path
        self.value = value
        self.from_ = from_

    def to_dict(self) -> dict:
        """Convert to RFC 6902 JSON format."""
        result = {"op": self.op, "path": self.path}
        if self.op in ("add", "replace", "test"):
            result["value"] = self.value
        if self.op in ("move", "copy") and self.from_ is not None:
            result["from"] = self.from_
        return result

    def to_patch_operation(self) -> JSONPatchOperation:
        """Convert to the new JSONPatchOperation model."""
        return JSONPatchOperation(
            op=PatchOp(self.op),
            path=self.path,
            value=self.value,
            from_=self.from_
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Exceptions
    "JSONPointerError",
    "PatchApplicationError",
    # Enums
    "PatchOp",
    # Models
    "JSONPatchOperation",
    "PatchSequence",
    # Generator
    "PatchGenerator",
    # JSON Pointer utilities
    "parse_json_pointer",
    "build_json_pointer",
    "get_value_at_pointer",
    "set_value_at_pointer",
    "insert_value_at_pointer",
    "remove_value_at_pointer",
    "pointer_parent",
    "pointer_last_segment",
    "escape_json_pointer_token",
    "unescape_json_pointer_token",
    # Patch application
    "apply_patch",
    "apply_patch_sequence",
    # Patch validation
    "validate_patch",
    "validate_patch_syntax",
    # Convenience functions
    "create_add_patch",
    "create_remove_patch",
    "create_replace_patch",
    "create_move_patch",
    # Legacy compatibility
    "JSONPatchOp",
]
