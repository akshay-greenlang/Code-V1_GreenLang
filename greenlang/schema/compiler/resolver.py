# -*- coding: utf-8 -*-
"""
Schema Reference Resolver for GL-FOUND-X-002.

This module implements $ref resolution with support for:
    - Local refs: #/definitions/Foo, #/$defs/Bar
    - External refs: gl://schemas/emissions/activity@1.3.0
    - Circular reference detection with clear cycle traces
    - Reference expansion limiting
    - Session-scoped caching for performance

The resolver follows JSON Pointer (RFC 6901) for navigating within documents
and supports the GreenLang gl:// URI scheme for external schema references.

Example:
    >>> from greenlang.schema.compiler.resolver import RefResolver, LocalFileRegistry
    >>> registry = LocalFileRegistry("./schemas")
    >>> resolver = RefResolver(schema_registry=registry)
    >>> resolved = resolver.resolve("#/definitions/Activity", context_document)

Author: GreenLang Team
Date: 2026-01-29
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Attempt Pydantic import for BaseModel, fall back to dataclass
try:
    from pydantic import BaseModel, Field, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore
    ConfigDict = None  # type: ignore
    def Field(*args, **kwargs):  # type: ignore
        return kwargs.get('default', None)

from greenlang.schema.constants import (
    MAX_REF_EXPANSIONS,
    GREENLANG_SCHEMA_PREFIX,
)
from greenlang.schema.errors import ErrorCode, format_error_message

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================


class CircularRefError(Exception):
    """
    Raised when a circular reference is detected during resolution.

    This exception provides the complete cycle trace for debugging,
    showing the path of references that form the cycle.

    Attributes:
        cycle: List of $ref values forming the cycle
        message: Human-readable error message

    Example:
        >>> try:
        ...     resolver.resolve("#/definitions/A", doc)
        ... except CircularRefError as e:
        ...     print(f"Cycle: {' -> '.join(e.cycle)}")
        Cycle: #/definitions/A -> #/definitions/B -> #/definitions/A
    """

    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        self.message = f"Circular reference detected: {' -> '.join(cycle)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"CircularRefError(cycle={self.cycle!r})"


class RefResolutionError(Exception):
    """
    Raised when a $ref cannot be resolved.

    This exception provides details about why resolution failed,
    including the problematic reference and a reason string.

    Attributes:
        ref: The $ref value that could not be resolved
        reason: Human-readable explanation of why resolution failed
        error_code: Associated GLSCHEMA error code

    Example:
        >>> try:
        ...     resolver.resolve("#/definitions/Missing", doc)
        ... except RefResolutionError as e:
        ...     print(f"Cannot resolve '{e.ref}': {e.reason}")
    """

    def __init__(
        self,
        ref: str,
        reason: str,
        error_code: Optional[ErrorCode] = None,
    ):
        self.ref = ref
        self.reason = reason
        self.error_code = error_code or ErrorCode.REF_RESOLUTION_FAILED
        self.message = f"Cannot resolve $ref '{ref}': {reason}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"RefResolutionError(ref={self.ref!r}, reason={self.reason!r})"


class MaxExpansionsExceededError(Exception):
    """
    Raised when the maximum number of $ref expansions is exceeded.

    This exception is raised to prevent potential denial-of-service
    attacks through deeply nested or recursive schema references.

    Attributes:
        count: The number of expansions attempted
        max_expansions: The configured maximum

    Example:
        >>> try:
        ...     resolver.resolve(ref, doc)
        ... except MaxExpansionsExceededError as e:
        ...     print(f"Too many expansions: {e.count}/{e.max_expansions}")
    """

    def __init__(self, count: int, max_expansions: int):
        self.count = count
        self.max_expansions = max_expansions
        self.message = (
            f"Maximum $ref expansions exceeded: {count} >= {max_expansions}. "
            f"This may indicate a very deep reference chain or potential abuse."
        )
        super().__init__(self.message)


# ============================================================================
# DATA MODELS
# ============================================================================


if PYDANTIC_AVAILABLE:
    class SchemaSource(BaseModel):
        """
        Schema source retrieved from a registry.

        This model represents a schema document fetched from a schema
        registry, including metadata about the source.

        Attributes:
            content: The raw schema content (YAML or JSON string)
            content_type: MIME type ("application/json" or "application/yaml")
            schema_id: Unique identifier for the schema
            version: Schema version string
            etag: Optional ETag for caching and conditional requests
        """
        model_config = ConfigDict(frozen=True)

        content: str = Field(..., description="Raw schema content")
        content_type: str = Field(
            ...,
            description="MIME type: application/json or application/yaml"
        )
        schema_id: str = Field(..., description="Unique schema identifier")
        version: str = Field(..., description="Schema version")
        etag: Optional[str] = Field(
            None,
            description="ETag for caching"
        )
else:
    @dataclass(frozen=True)
    class SchemaSource:  # type: ignore
        """Schema source retrieved from a registry."""
        content: str
        content_type: str
        schema_id: str
        version: str
        etag: Optional[str] = None


class RefType(str, Enum):
    """Types of $ref references."""
    LOCAL = "local"              # #/definitions/Foo
    EXTERNAL_GL = "external_gl"  # gl://schemas/...
    EXTERNAL_HTTP = "external_http"  # http(s)://...
    RELATIVE = "relative"        # ./other-schema.json


@dataclass
class ParsedRef:
    """
    Parsed components of a $ref value.

    Attributes:
        ref_type: The type of reference
        original: The original $ref string
        schema_id: Schema identifier (for external refs)
        version: Schema version (for external refs)
        fragment: JSON Pointer fragment (path within schema)
        uri: Full URI for external refs
    """
    ref_type: RefType
    original: str
    schema_id: Optional[str] = None
    version: Optional[str] = None
    fragment: Optional[str] = None
    uri: Optional[str] = None


# ============================================================================
# PROTOCOLS
# ============================================================================


class SchemaRegistry(Protocol):
    """
    Protocol for schema registry backends.

    This protocol defines the interface that schema registries must
    implement to be used with the RefResolver.

    Implementations may include:
        - LocalFileRegistry: File-based registry for development
        - GitSchemaRegistry: Git-backed registry
        - HttpSchemaRegistry: HTTP-based registry with caching
    """

    def resolve(self, schema_id: str, version: str) -> SchemaSource:
        """
        Resolve a schema by ID and version.

        Args:
            schema_id: The schema identifier (e.g., "emissions/activity")
            version: The schema version (e.g., "1.3.0")

        Returns:
            SchemaSource containing the schema content

        Raises:
            RefResolutionError: If schema cannot be found
        """
        ...

    def list_versions(self, schema_id: str) -> List[str]:
        """
        List available versions for a schema.

        Args:
            schema_id: The schema identifier

        Returns:
            List of available version strings, sorted newest first
        """
        ...


# ============================================================================
# JSON POINTER IMPLEMENTATION (RFC 6901)
# ============================================================================


def _escape_json_pointer_token(token: str) -> str:
    """
    Escape a token for use in a JSON Pointer.

    Per RFC 6901:
    - ~ is escaped as ~0
    - / is escaped as ~1

    Args:
        token: The unescaped token

    Returns:
        The escaped token
    """
    return token.replace("~", "~0").replace("/", "~1")


def _unescape_json_pointer_token(token: str) -> str:
    """
    Unescape a JSON Pointer token.

    Per RFC 6901:
    - ~1 becomes /
    - ~0 becomes ~

    Order matters: ~1 must be processed before ~0.

    Args:
        token: The escaped token

    Returns:
        The unescaped token
    """
    return token.replace("~1", "/").replace("~0", "~")


def parse_json_pointer(pointer: str) -> List[str]:
    """
    Parse a JSON Pointer (RFC 6901) into path segments.

    JSON Pointers start with "/" and use "/" as a delimiter.
    Special characters are escaped:
    - ~0 represents ~
    - ~1 represents /

    Args:
        pointer: The JSON Pointer string (e.g., "/definitions/Foo")

    Returns:
        List of unescaped path segments

    Raises:
        ValueError: If pointer is malformed

    Example:
        >>> parse_json_pointer("/definitions/Foo")
        ['definitions', 'Foo']
        >>> parse_json_pointer("/a~1b/c~0d")
        ['a/b', 'c~d']
        >>> parse_json_pointer("")
        []
    """
    # Empty pointer refers to the whole document
    if pointer == "":
        return []

    # Pointer must start with /
    if not pointer.startswith("/"):
        raise ValueError(
            f"Invalid JSON Pointer: must start with '/' but got '{pointer}'"
        )

    # Split by / and unescape each segment
    # First character is /, so skip it and split the rest
    segments = pointer[1:].split("/")

    # Unescape each segment
    return [_unescape_json_pointer_token(seg) for seg in segments]


def navigate_json_pointer(
    document: Dict[str, Any],
    pointer: str,
) -> Any:
    """
    Navigate to a location in a document using a JSON Pointer.

    Args:
        document: The document to navigate
        pointer: JSON Pointer path (e.g., "/definitions/Foo")

    Returns:
        The value at the pointer location

    Raises:
        RefResolutionError: If path segment not found

    Example:
        >>> doc = {"definitions": {"Foo": {"type": "string"}}}
        >>> navigate_json_pointer(doc, "/definitions/Foo")
        {'type': 'string'}
    """
    if pointer == "" or pointer == "/":
        return document

    segments = parse_json_pointer(pointer)
    current = document

    for i, segment in enumerate(segments):
        if isinstance(current, dict):
            if segment not in current:
                path_so_far = "/" + "/".join(segments[:i+1])
                raise RefResolutionError(
                    ref=pointer,
                    reason=f"Path segment '{segment}' not found at '{path_so_far}'",
                    error_code=ErrorCode.SCHEMA_DEFINITION_NOT_FOUND,
                )
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
                if index < 0 or index >= len(current):
                    raise RefResolutionError(
                        ref=pointer,
                        reason=f"Array index {index} out of bounds (length: {len(current)})",
                        error_code=ErrorCode.SCHEMA_DEFINITION_NOT_FOUND,
                    )
                current = current[index]
            except ValueError:
                raise RefResolutionError(
                    ref=pointer,
                    reason=f"Invalid array index '{segment}' at path",
                    error_code=ErrorCode.SCHEMA_DEFINITION_NOT_FOUND,
                )
        else:
            path_so_far = "/" + "/".join(segments[:i])
            raise RefResolutionError(
                ref=pointer,
                reason=f"Cannot navigate into primitive value at '{path_so_far}'",
                error_code=ErrorCode.SCHEMA_DEFINITION_NOT_FOUND,
            )

    return current


# ============================================================================
# REF PARSER
# ============================================================================


# Regex for parsing gl:// URIs
# Format: gl://schemas/{schema_id}@{version}[#/fragment]
GL_URI_PATTERN = re.compile(
    r"^gl://schemas/(?P<schema_id>[^@#]+)@(?P<version>[^#]+)(?:#(?P<fragment>.*))?$"
)

# Regex for parsing version-less gl:// URIs (latest version)
GL_URI_LATEST_PATTERN = re.compile(
    r"^gl://schemas/(?P<schema_id>[^@#]+)(?:#(?P<fragment>.*))?$"
)


def parse_ref(ref: str) -> ParsedRef:
    """
    Parse a $ref value into its components.

    Supports:
    - Local refs: #/definitions/Foo, #/$defs/Bar
    - GreenLang refs: gl://schemas/emissions/activity@1.3.0
    - HTTP refs: http(s)://example.com/schema.json
    - Relative refs: ./other-schema.json

    Args:
        ref: The $ref value to parse

    Returns:
        ParsedRef with all extracted components

    Example:
        >>> parsed = parse_ref("#/definitions/Foo")
        >>> parsed.ref_type
        <RefType.LOCAL: 'local'>
        >>> parsed.fragment
        '/definitions/Foo'

        >>> parsed = parse_ref("gl://schemas/emissions/activity@1.3.0")
        >>> parsed.schema_id
        'emissions/activity'
        >>> parsed.version
        '1.3.0'
    """
    # Local reference
    if ref.startswith("#"):
        fragment = ref[1:] if len(ref) > 1 else ""
        return ParsedRef(
            ref_type=RefType.LOCAL,
            original=ref,
            fragment=fragment,
        )

    # GreenLang URI with version
    match = GL_URI_PATTERN.match(ref)
    if match:
        return ParsedRef(
            ref_type=RefType.EXTERNAL_GL,
            original=ref,
            schema_id=match.group("schema_id"),
            version=match.group("version"),
            fragment=match.group("fragment"),
            uri=ref,
        )

    # GreenLang URI without version (latest)
    match = GL_URI_LATEST_PATTERN.match(ref)
    if match and not ref.startswith("gl://schemas/") or match:
        # Re-match to check for version-less URI
        if ref.startswith("gl://schemas/") and "@" not in ref.split("#")[0]:
            match = GL_URI_LATEST_PATTERN.match(ref)
            if match:
                return ParsedRef(
                    ref_type=RefType.EXTERNAL_GL,
                    original=ref,
                    schema_id=match.group("schema_id"),
                    version="latest",
                    fragment=match.group("fragment"),
                    uri=ref,
                )

    # HTTP/HTTPS URI
    if ref.startswith("http://") or ref.startswith("https://"):
        # Split URI and fragment
        if "#" in ref:
            uri, fragment = ref.rsplit("#", 1)
        else:
            uri = ref
            fragment = None
        return ParsedRef(
            ref_type=RefType.EXTERNAL_HTTP,
            original=ref,
            fragment=fragment,
            uri=uri,
        )

    # Relative reference
    if ref.startswith("./") or ref.startswith("../"):
        if "#" in ref:
            uri, fragment = ref.rsplit("#", 1)
        else:
            uri = ref
            fragment = None
        return ParsedRef(
            ref_type=RefType.RELATIVE,
            original=ref,
            fragment=fragment,
            uri=uri,
        )

    # Unknown format - treat as local ref without #
    logger.warning(f"Unknown $ref format: '{ref}', treating as local ref")
    return ParsedRef(
        ref_type=RefType.LOCAL,
        original=ref,
        fragment=ref if ref.startswith("/") else f"/{ref}",
    )


# ============================================================================
# REF RESOLVER
# ============================================================================


class RefResolver:
    """
    Resolves $ref references in JSON schemas.

    This class handles resolution of both local references (within the
    same document) and external references (to other schemas in a registry).

    Features:
        - Full JSON Pointer (RFC 6901) support
        - Local ref resolution (#/definitions/Foo, #/$defs/Bar)
        - External ref resolution via SchemaRegistry
        - Circular reference detection with clear cycle traces
        - Expansion limit enforcement for DoS protection
        - Session-scoped caching for performance

    Thread Safety:
        RefResolver instances are NOT thread-safe. Create a new instance
        per thread or use proper synchronization.

    Example:
        >>> registry = LocalFileRegistry("./schemas")
        >>> resolver = RefResolver(schema_registry=registry)
        >>>
        >>> # Resolve a local reference
        >>> schema_def = resolver.resolve(
        ...     ref="#/definitions/Activity",
        ...     context_document=schema_dict,
        ...     context_path=""
        ... )
        >>>
        >>> # Resolve an external reference
        >>> external_def = resolver.resolve(
        ...     ref="gl://schemas/emissions/activity@1.3.0",
        ...     context_document=schema_dict,
        ...     context_path=""
        ... )
        >>>
        >>> # Reset for new compilation session
        >>> resolver.reset()
    """

    def __init__(
        self,
        schema_registry: Optional[SchemaRegistry] = None,
        max_expansions: Optional[int] = None,
    ):
        """
        Initialize the RefResolver.

        Args:
            schema_registry: Optional registry for external schema lookups.
                If None, external refs will raise RefResolutionError.
            max_expansions: Maximum number of ref expansions allowed.
                Defaults to MAX_REF_EXPANSIONS from constants.
        """
        self.registry = schema_registry
        self.max_expansions = (
            max_expansions if max_expansions is not None
            else MAX_REF_EXPANSIONS
        )

        # Session state
        self._cache: Dict[str, Any] = {}
        self._resolution_stack: List[str] = []
        self._expansion_count: int = 0

        # External document cache (for external refs)
        self._external_documents: Dict[str, Dict[str, Any]] = {}

        logger.debug(
            f"RefResolver initialized with max_expansions={self.max_expansions}, "
            f"registry={'present' if schema_registry else 'None'}"
        )

    def resolve(
        self,
        ref: str,
        context_document: Dict[str, Any],
        context_path: str = "",
    ) -> Dict[str, Any]:
        """
        Resolve a $ref to its target schema definition.

        This method handles both local and external references, with
        support for caching and cycle detection.

        Args:
            ref: The $ref value (e.g., "#/definitions/Foo" or "gl://...")
            context_document: The root document containing the ref
            context_path: JSON Pointer path where ref appears (for error messages)

        Returns:
            The resolved schema definition as a dictionary

        Raises:
            CircularRefError: If circular reference detected
            RefResolutionError: If ref cannot be resolved
            MaxExpansionsExceededError: If max expansions exceeded (GLSCHEMA-E803)

        Example:
            >>> resolved = resolver.resolve(
            ...     ref="#/definitions/Activity",
            ...     context_document={"definitions": {"Activity": {"type": "object"}}},
            ...     context_path="/properties/activity/$ref"
            ... )
            >>> resolved
            {'type': 'object'}
        """
        logger.debug(f"Resolving $ref: '{ref}' at path: '{context_path}'")

        # Check expansion limit
        self._check_expansion_limit()

        # Check for cycles
        cycle = self._detect_cycle(ref)
        if cycle is not None:
            logger.error(f"Circular reference detected: {' -> '.join(cycle)}")
            raise CircularRefError(cycle)

        # Check cache
        cache_key = self._compute_cache_key(ref, context_document)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for ref: '{ref}'")
            return self._cache[cache_key]

        # Push onto resolution stack
        self._resolution_stack.append(ref)
        self._expansion_count += 1

        try:
            # Parse the ref
            parsed = parse_ref(ref)

            # Resolve based on type
            if parsed.ref_type == RefType.LOCAL:
                result = self._resolve_local_ref(ref, context_document)
            elif parsed.ref_type == RefType.EXTERNAL_GL:
                result = self._resolve_external_ref(ref)
            elif parsed.ref_type == RefType.EXTERNAL_HTTP:
                result = self._resolve_http_ref(ref)
            elif parsed.ref_type == RefType.RELATIVE:
                result = self._resolve_relative_ref(ref, context_path)
            else:
                raise RefResolutionError(
                    ref=ref,
                    reason=f"Unknown ref type: {parsed.ref_type}",
                )

            # Cache the result
            self._cache[cache_key] = result

            logger.debug(f"Successfully resolved $ref: '{ref}'")
            return result

        finally:
            # Pop from resolution stack
            self._resolution_stack.pop()

    def _resolve_local_ref(
        self,
        ref: str,
        document: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve a local reference (#/path/to/def).

        Local references use JSON Pointer syntax to navigate within
        the same document. Common patterns include:
        - #/definitions/Foo (JSON Schema draft-07 style)
        - #/$defs/Foo (JSON Schema draft 2019-09+ style)
        - #/properties/name

        Args:
            ref: The local ref (starting with #)
            document: The document to navigate

        Returns:
            The resolved schema definition

        Raises:
            RefResolutionError: If target not found
        """
        # Extract the pointer (everything after #)
        if not ref.startswith("#"):
            raise RefResolutionError(
                ref=ref,
                reason="Local ref must start with '#'",
            )

        pointer = ref[1:] if len(ref) > 1 else ""

        # Handle empty fragment (refers to root)
        if pointer == "":
            return document

        # Navigate to the target
        try:
            result = navigate_json_pointer(document, pointer)
        except RefResolutionError:
            raise
        except Exception as e:
            raise RefResolutionError(
                ref=ref,
                reason=str(e),
            )

        # Ensure result is a dict (schema definition)
        if not isinstance(result, dict):
            raise RefResolutionError(
                ref=ref,
                reason=f"Target is not an object (found {type(result).__name__})",
            )

        # Recursively resolve any $ref in the result
        if "$ref" in result:
            nested_ref = result["$ref"]
            logger.debug(f"Found nested $ref in result: '{nested_ref}'")
            return self.resolve(
                ref=nested_ref,
                context_document=document,
                context_path=pointer,
            )

        return result

    def _resolve_external_ref(self, ref: str) -> Dict[str, Any]:
        """
        Resolve an external reference (gl://...).

        External references use the GreenLang URI scheme to reference
        schemas in the registry. The format is:
            gl://schemas/{schema_id}@{version}[#/fragment]

        Args:
            ref: The external ref URI

        Returns:
            The resolved schema definition

        Raises:
            RefResolutionError: If schema not found or registry not available
        """
        if self.registry is None:
            raise RefResolutionError(
                ref=ref,
                reason="No schema registry configured for external ref resolution",
                error_code=ErrorCode.SCHEMA_REGISTRY_ERROR,
            )

        parsed = parse_ref(ref)

        if parsed.schema_id is None or parsed.version is None:
            raise RefResolutionError(
                ref=ref,
                reason="Invalid gl:// URI format",
            )

        # Create a cache key for the external document
        doc_cache_key = f"{parsed.schema_id}@{parsed.version}"

        # Check if we already have the document
        if doc_cache_key not in self._external_documents:
            try:
                # Fetch from registry
                logger.info(
                    f"Fetching external schema: {parsed.schema_id}@{parsed.version}"
                )
                source = self.registry.resolve(parsed.schema_id, parsed.version)

                # Parse the content
                document = self._parse_schema_content(source)
                self._external_documents[doc_cache_key] = document

            except RefResolutionError:
                raise
            except Exception as e:
                raise RefResolutionError(
                    ref=ref,
                    reason=f"Failed to fetch schema from registry: {e}",
                    error_code=ErrorCode.SCHEMA_REGISTRY_ERROR,
                )
        else:
            document = self._external_documents[doc_cache_key]

        # If there's a fragment, navigate to it
        if parsed.fragment:
            return navigate_json_pointer(document, f"/{parsed.fragment}" if not parsed.fragment.startswith("/") else parsed.fragment)

        return document

    def _resolve_http_ref(self, ref: str) -> Dict[str, Any]:
        """
        Resolve an HTTP/HTTPS reference.

        Currently not implemented - raises RefResolutionError.

        Args:
            ref: The HTTP ref URI

        Raises:
            RefResolutionError: HTTP refs not supported
        """
        raise RefResolutionError(
            ref=ref,
            reason="HTTP/HTTPS references are not supported. "
                   "Use gl:// URIs for external schemas.",
        )

    def _resolve_relative_ref(
        self,
        ref: str,
        context_path: str,
    ) -> Dict[str, Any]:
        """
        Resolve a relative reference (./other-schema.json).

        Currently not implemented - raises RefResolutionError.

        Args:
            ref: The relative ref path
            context_path: The path where the ref appears

        Raises:
            RefResolutionError: Relative refs not supported
        """
        raise RefResolutionError(
            ref=ref,
            reason="Relative file references are not supported. "
                   "Use gl:// URIs for external schemas.",
        )

    def _detect_cycle(self, ref: str) -> Optional[List[str]]:
        """
        Check for circular reference, return cycle trace if found.

        Examines the resolution stack to detect if resolving this
        ref would create a cycle.

        Args:
            ref: The ref being resolved

        Returns:
            List of refs forming the cycle if detected, None otherwise

        Example:
            >>> resolver._resolution_stack = ["#/definitions/A", "#/definitions/B"]
            >>> resolver._detect_cycle("#/definitions/A")
            ['#/definitions/A', '#/definitions/B', '#/definitions/A']
        """
        if ref in self._resolution_stack:
            # Find where the cycle starts
            start_idx = self._resolution_stack.index(ref)
            # Return the cycle path including the duplicate ref at the end
            cycle = self._resolution_stack[start_idx:] + [ref]
            return cycle
        return None

    def _check_expansion_limit(self) -> None:
        """
        Check if expansion limit has been exceeded.

        Raises:
            MaxExpansionsExceededError: If limit exceeded (maps to GLSCHEMA-E803)
        """
        if self._expansion_count >= self.max_expansions:
            logger.error(
                f"Maximum ref expansions exceeded: {self._expansion_count} >= {self.max_expansions}"
            )
            raise MaxExpansionsExceededError(
                count=self._expansion_count,
                max_expansions=self.max_expansions,
            )

    def _parse_json_pointer(self, pointer: str) -> List[str]:
        """
        Parse JSON Pointer (RFC 6901) to path segments.

        This is a convenience wrapper around the module-level function.

        Args:
            pointer: The JSON Pointer string

        Returns:
            List of unescaped path segments
        """
        return parse_json_pointer(pointer)

    def _navigate_to_path(
        self,
        document: Dict[str, Any],
        path: List[str],
    ) -> Dict[str, Any]:
        """
        Navigate to a path in a document.

        Args:
            document: The document to navigate
            path: List of path segments

        Returns:
            The value at the path

        Raises:
            RefResolutionError: If path not found
        """
        # Convert path list to JSON Pointer
        pointer = "/" + "/".join(_escape_json_pointer_token(seg) for seg in path)
        result = navigate_json_pointer(document, pointer)

        if not isinstance(result, dict):
            raise RefResolutionError(
                ref=pointer,
                reason=f"Target is not an object (found {type(result).__name__})",
            )
        return result

    def _compute_cache_key(
        self,
        ref: str,
        context_document: Dict[str, Any],
    ) -> str:
        """
        Compute a cache key for a reference.

        For local refs, the key includes a hash of the context document
        to handle the case where the same ref points to different
        definitions in different documents.

        Args:
            ref: The $ref value
            context_document: The document containing the ref

        Returns:
            A unique cache key string
        """
        parsed = parse_ref(ref)

        if parsed.ref_type == RefType.LOCAL:
            # For local refs, include document identity
            doc_id = id(context_document)
            return f"local:{doc_id}:{ref}"
        elif parsed.ref_type == RefType.EXTERNAL_GL:
            # For external refs, the URI itself is unique
            return f"external:{ref}"
        else:
            # For other types, use full ref
            return f"other:{ref}"

    def _parse_schema_content(self, source: SchemaSource) -> Dict[str, Any]:
        """
        Parse schema content from a SchemaSource.

        Args:
            source: The schema source to parse

        Returns:
            Parsed schema as a dictionary

        Raises:
            RefResolutionError: If parsing fails
        """
        content = source.content

        try:
            if source.content_type == "application/json":
                return json.loads(content)
            elif source.content_type in ("application/yaml", "application/x-yaml", "text/yaml"):
                # Try to import yaml
                try:
                    import yaml
                    return yaml.safe_load(content)
                except ImportError:
                    raise RefResolutionError(
                        ref=f"{source.schema_id}@{source.version}",
                        reason="YAML parsing requires PyYAML package",
                    )
            else:
                # Try JSON first, then YAML
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    try:
                        import yaml
                        return yaml.safe_load(content)
                    except ImportError:
                        raise RefResolutionError(
                            ref=f"{source.schema_id}@{source.version}",
                            reason=f"Unknown content type '{source.content_type}' and YAML not available",
                        )
        except json.JSONDecodeError as e:
            raise RefResolutionError(
                ref=f"{source.schema_id}@{source.version}",
                reason=f"JSON parse error: {e}",
                error_code=ErrorCode.SCHEMA_PARSE_ERROR,
            )
        except Exception as e:
            raise RefResolutionError(
                ref=f"{source.schema_id}@{source.version}",
                reason=f"Parse error: {e}",
                error_code=ErrorCode.SCHEMA_PARSE_ERROR,
            )

    def reset(self) -> None:
        """
        Reset resolver state for a new compilation session.

        This clears all caches and resets the expansion counter.
        Call this before compiling a new schema to ensure clean state.

        Example:
            >>> resolver.reset()
            >>> resolver._expansion_count
            0
            >>> resolver._cache
            {}
        """
        self._cache.clear()
        self._resolution_stack.clear()
        self._expansion_count = 0
        self._external_documents.clear()
        logger.debug("RefResolver state reset")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get resolver statistics for monitoring.

        Returns:
            Dictionary with resolver statistics

        Example:
            >>> stats = resolver.get_stats()
            >>> print(f"Expansions: {stats['expansion_count']}")
        """
        return {
            "expansion_count": self._expansion_count,
            "max_expansions": self.max_expansions,
            "cache_size": len(self._cache),
            "external_documents_cached": len(self._external_documents),
            "current_stack_depth": len(self._resolution_stack),
        }


# ============================================================================
# LOCAL FILE REGISTRY (Development)
# ============================================================================


class LocalFileRegistry:
    """
    Simple file-based schema registry for development and testing.

    This registry loads schemas from the local filesystem, with schemas
    organized by ID and version in a directory structure:
        {base_path}/{schema_id}@{version}.json
        {base_path}/{schema_id}@{version}.yaml

    Or in a nested structure:
        {base_path}/{schema_id}/{version}.json
        {base_path}/{schema_id}/{version}.yaml

    Example:
        >>> registry = LocalFileRegistry("./schemas")
        >>> source = registry.resolve("emissions/activity", "1.3.0")
        >>> print(source.content_type)
        'application/json'
    """

    def __init__(self, base_path: str):
        """
        Initialize the LocalFileRegistry.

        Args:
            base_path: Root directory containing schema files
        """
        self.base_path = Path(base_path)

        if not self.base_path.exists():
            logger.warning(f"Schema base path does not exist: {base_path}")

    def resolve(self, schema_id: str, version: str) -> SchemaSource:
        """
        Load a schema from the local filesystem.

        Searches for the schema in multiple locations:
        1. {base_path}/{schema_id}@{version}.json
        2. {base_path}/{schema_id}@{version}.yaml
        3. {base_path}/{schema_id}/{version}.json
        4. {base_path}/{schema_id}/{version}.yaml

        Args:
            schema_id: The schema identifier (e.g., "emissions/activity")
            version: The schema version (e.g., "1.3.0")

        Returns:
            SchemaSource with the loaded content

        Raises:
            RefResolutionError: If schema file not found
        """
        # Possible file locations
        candidates = [
            self.base_path / f"{schema_id}@{version}.json",
            self.base_path / f"{schema_id}@{version}.yaml",
            self.base_path / f"{schema_id}@{version}.yml",
            self.base_path / schema_id / f"{version}.json",
            self.base_path / schema_id / f"{version}.yaml",
            self.base_path / schema_id / f"{version}.yml",
            self.base_path / schema_id / f"v{version}.json",
            self.base_path / schema_id / f"v{version}.yaml",
        ]

        for candidate in candidates:
            if candidate.exists():
                logger.debug(f"Found schema file: {candidate}")

                # Determine content type
                suffix = candidate.suffix.lower()
                if suffix == ".json":
                    content_type = "application/json"
                elif suffix in (".yaml", ".yml"):
                    content_type = "application/yaml"
                else:
                    content_type = "application/octet-stream"

                # Read content
                try:
                    content = candidate.read_text(encoding="utf-8")
                except Exception as e:
                    raise RefResolutionError(
                        ref=f"{schema_id}@{version}",
                        reason=f"Failed to read schema file: {e}",
                        error_code=ErrorCode.SCHEMA_REGISTRY_ERROR,
                    )

                # Compute ETag from content hash
                etag = hashlib.sha256(content.encode()).hexdigest()[:16]

                return SchemaSource(
                    content=content,
                    content_type=content_type,
                    schema_id=schema_id,
                    version=version,
                    etag=etag,
                )

        # Not found
        raise RefResolutionError(
            ref=f"{schema_id}@{version}",
            reason=f"Schema not found in {self.base_path}. "
                   f"Searched: {[str(c) for c in candidates[:4]]}...",
            error_code=ErrorCode.SCHEMA_DEFINITION_NOT_FOUND,
        )

    def list_versions(self, schema_id: str) -> List[str]:
        """
        List available versions for a schema.

        Scans the filesystem for schema files matching the schema_id
        and extracts version numbers from filenames.

        Args:
            schema_id: The schema identifier

        Returns:
            List of version strings, sorted in descending order
        """
        versions = set()

        # Pattern for flat structure: {schema_id}@{version}.{ext}
        flat_pattern = f"{schema_id}@*.json"
        for path in self.base_path.glob(flat_pattern):
            # Extract version from filename
            name = path.stem  # e.g., "emissions/activity@1.3.0"
            if "@" in name:
                version = name.split("@")[-1]
                versions.add(version)

        # Also check YAML files
        flat_pattern_yaml = f"{schema_id}@*.yaml"
        for path in self.base_path.glob(flat_pattern_yaml):
            name = path.stem
            if "@" in name:
                version = name.split("@")[-1]
                versions.add(version)

        # Pattern for nested structure: {schema_id}/{version}.{ext}
        nested_dir = self.base_path / schema_id
        if nested_dir.exists() and nested_dir.is_dir():
            for path in nested_dir.iterdir():
                if path.suffix in (".json", ".yaml", ".yml"):
                    version = path.stem
                    # Remove 'v' prefix if present
                    if version.startswith("v"):
                        version = version[1:]
                    versions.add(version)

        # Sort versions (simple string sort - for proper semver, use packaging.version)
        sorted_versions = sorted(versions, reverse=True)

        logger.debug(f"Found versions for {schema_id}: {sorted_versions}")
        return sorted_versions


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def resolve_all_refs(
    document: Dict[str, Any],
    resolver: RefResolver,
    current_path: str = "",
    _root_document: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Recursively resolve all $ref nodes in a document.

    This function walks the entire document tree and replaces all
    $ref nodes with their resolved definitions. The result is a
    fully dereferenced schema.

    Note: This creates a new document; the original is not modified.

    Args:
        document: The document to process
        resolver: RefResolver instance to use
        current_path: Current JSON Pointer path (for recursion)
        _root_document: Internal parameter - the root document for ref resolution

    Returns:
        New document with all refs resolved

    Example:
        >>> doc = {
        ...     "type": "object",
        ...     "properties": {
        ...         "activity": {"$ref": "#/definitions/Activity"}
        ...     },
        ...     "definitions": {
        ...         "Activity": {"type": "string"}
        ...     }
        ... }
        >>> resolved = resolve_all_refs(doc, resolver)
        >>> resolved["properties"]["activity"]
        {'type': 'string'}
    """
    if not isinstance(document, dict):
        return document

    # Track the root document for resolving local refs
    root = _root_document if _root_document is not None else document

    # Check if this node is a $ref
    if "$ref" in document:
        ref = document["$ref"]
        resolved = resolver.resolve(
            ref=ref,
            context_document=root,
            context_path=current_path,
        )
        # Recursively resolve any refs in the resolved document
        # Pass the root document for continued resolution
        return resolve_all_refs(resolved, resolver, current_path, root)

    # Recursively process all properties
    result = {}
    for key, value in document.items():
        new_path = f"{current_path}/{key}"

        if isinstance(value, dict):
            result[key] = resolve_all_refs(value, resolver, new_path, root)
        elif isinstance(value, list):
            result[key] = [
                resolve_all_refs(item, resolver, f"{new_path}/{i}", root)
                if isinstance(item, dict) else item
                for i, item in enumerate(value)
            ]
        else:
            result[key] = value

    return result


def validate_ref_format(ref: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a $ref value has a valid format.

    Args:
        ref: The $ref value to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> validate_ref_format("#/definitions/Foo")
        (True, None)
        >>> validate_ref_format("invalid ref")
        (False, "Unknown reference format")
    """
    if not ref:
        return False, "Empty $ref value"

    parsed = parse_ref(ref)

    if parsed.ref_type == RefType.LOCAL:
        # Validate JSON Pointer format
        if parsed.fragment and not parsed.fragment.startswith("/") and parsed.fragment != "":
            return False, f"Local ref fragment must start with '/': {ref}"
        return True, None

    if parsed.ref_type == RefType.EXTERNAL_GL:
        if not parsed.schema_id:
            return False, f"Missing schema_id in gl:// URI: {ref}"
        if not parsed.version:
            return False, f"Missing version in gl:// URI: {ref}"
        return True, None

    if parsed.ref_type == RefType.EXTERNAL_HTTP:
        return True, None  # HTTP URIs are valid but not supported

    if parsed.ref_type == RefType.RELATIVE:
        return True, None  # Relative refs are valid but not supported

    return False, f"Unknown reference format: {ref}"


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    # Exceptions
    "CircularRefError",
    "RefResolutionError",
    "MaxExpansionsExceededError",
    # Data models
    "SchemaSource",
    "ParsedRef",
    "RefType",
    # Protocols
    "SchemaRegistry",
    # Main classes
    "RefResolver",
    "LocalFileRegistry",
    # Functions
    "parse_json_pointer",
    "navigate_json_pointer",
    "parse_ref",
    "resolve_all_refs",
    "validate_ref_format",
]
