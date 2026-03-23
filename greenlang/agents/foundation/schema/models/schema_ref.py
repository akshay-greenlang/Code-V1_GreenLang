# -*- coding: utf-8 -*-
"""
Schema Reference Model
======================

Pydantic model for schema identifiers and version references.

A SchemaRef uniquely identifies a schema in the GreenLang registry.
It consists of:
- schema_id: The schema identifier (e.g., "emissions/activity")
- version: The schema version using semver (e.g., "1.3.0")
- variant: Optional variant identifier (e.g., "strict", "legacy")

Example:
    >>> ref = SchemaRef(schema_id="emissions/activity", version="1.3.0")
    >>> print(ref.to_uri())
    gl://schemas/emissions/activity@1.3.0

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# Schema ID pattern: domain/name or just name
SCHEMA_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)*$")

# Semantic version pattern: major.minor.patch with optional prerelease
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-((0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(\.(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?$"
)

# Variant pattern: lowercase alphanumeric with underscores
VARIANT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class SchemaRef(BaseModel):
    """
    Reference to a schema in the GreenLang registry.

    A SchemaRef uniquely identifies a schema by its ID, version, and optional variant.
    This is the primary mechanism for specifying which schema to use for validation.

    Attributes:
        schema_id: The schema identifier (e.g., "emissions/activity").
            Must be lowercase alphanumeric with underscores, using "/" for namespacing.
        version: The schema version using semantic versioning (e.g., "1.3.0").
            Must follow semver format: major.minor.patch[-prerelease][+build].
        variant: Optional variant identifier (e.g., "strict", "legacy").
            Used for schema variants that share the same base but have different rules.

    Example:
        >>> ref = SchemaRef(schema_id="emissions/activity", version="1.3.0")
        >>> print(ref.to_uri())
        gl://schemas/emissions/activity@1.3.0

        >>> ref_with_variant = SchemaRef(
        ...     schema_id="emissions/activity",
        ...     version="1.3.0",
        ...     variant="strict"
        ... )
        >>> print(ref_with_variant.to_uri())
        gl://schemas/emissions/activity@1.3.0#strict
    """

    schema_id: str = Field(
        ...,
        min_length=1,
        max_length=253,
        description="Schema identifier (e.g., 'emissions/activity')"
    )

    version: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Schema version using semantic versioning (e.g., '1.3.0')"
    )

    variant: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=63,
        description="Optional variant identifier (e.g., 'strict', 'legacy')"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "schema_id": "emissions/activity",
                    "version": "1.3.0"
                },
                {
                    "schema_id": "emissions/activity",
                    "version": "1.3.0",
                    "variant": "strict"
                }
            ]
        }
    }

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, v: str) -> str:
        """
        Validate schema_id format.

        Args:
            v: The schema_id value to validate.

        Returns:
            The validated schema_id.

        Raises:
            ValueError: If schema_id doesn't match the required pattern.
        """
        if not SCHEMA_ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid schema_id '{v}'. Must be lowercase alphanumeric with "
                "underscores, using '/' for namespacing (e.g., 'emissions/activity')."
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """
        Validate version follows semantic versioning.

        Args:
            v: The version string to validate.

        Returns:
            The validated version string.

        Raises:
            ValueError: If version doesn't follow semver format.
        """
        if not SEMVER_PATTERN.match(v):
            raise ValueError(
                f"Invalid version '{v}'. Must follow semantic versioning "
                "(e.g., '1.3.0', '2.0.0-beta.1', '1.0.0+build.123')."
            )
        return v

    @field_validator("variant")
    @classmethod
    def validate_variant(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate variant format if provided.

        Args:
            v: The variant string to validate (may be None).

        Returns:
            The validated variant string or None.

        Raises:
            ValueError: If variant doesn't match the required pattern.
        """
        if v is not None and not VARIANT_PATTERN.match(v):
            raise ValueError(
                f"Invalid variant '{v}'. Must be lowercase alphanumeric "
                "with underscores (e.g., 'strict', 'legacy_v1')."
            )
        return v

    def to_uri(self) -> str:
        """
        Convert to GreenLang schema URI format.

        Returns:
            Schema URI in format: gl://schemas/{schema_id}@{version}[#{variant}]

        Example:
            >>> ref = SchemaRef(schema_id="emissions/activity", version="1.3.0")
            >>> ref.to_uri()
            'gl://schemas/emissions/activity@1.3.0'
        """
        uri = f"gl://schemas/{self.schema_id}@{self.version}"
        if self.variant:
            uri = f"{uri}#{self.variant}"
        return uri

    def to_cache_key(self) -> str:
        """
        Generate a cache key for this schema reference.

        Returns:
            A unique, filesystem-safe cache key.

        Example:
            >>> ref = SchemaRef(schema_id="emissions/activity", version="1.3.0")
            >>> ref.to_cache_key()
            'emissions__activity__1.3.0'
        """
        key = f"{self.schema_id.replace('/', '__')}__{self.version}"
        if self.variant:
            key = f"{key}__{self.variant}"
        return key

    def major_version(self) -> int:
        """
        Extract the major version number.

        Returns:
            The major version as an integer.

        Example:
            >>> ref = SchemaRef(schema_id="test", version="2.3.1")
            >>> ref.major_version()
            2
        """
        return int(self.version.split(".")[0])

    def minor_version(self) -> int:
        """
        Extract the minor version number.

        Returns:
            The minor version as an integer.

        Example:
            >>> ref = SchemaRef(schema_id="test", version="2.3.1")
            >>> ref.minor_version()
            3
        """
        return int(self.version.split(".")[1])

    def patch_version(self) -> int:
        """
        Extract the patch version number.

        Returns:
            The patch version as an integer.

        Example:
            >>> ref = SchemaRef(schema_id="test", version="2.3.1")
            >>> ref.patch_version()
            1
        """
        # Handle prerelease and build metadata
        patch_str = self.version.split(".")[2]
        # Remove prerelease and build metadata
        patch_str = patch_str.split("-")[0].split("+")[0]
        return int(patch_str)

    def is_compatible_with(self, other: "SchemaRef") -> bool:
        """
        Check if this schema is compatible with another (same major version).

        Two schemas are considered compatible if they have the same schema_id
        and the same major version number.

        Args:
            other: Another SchemaRef to compare with.

        Returns:
            True if schemas are compatible, False otherwise.

        Example:
            >>> ref1 = SchemaRef(schema_id="test", version="1.0.0")
            >>> ref2 = SchemaRef(schema_id="test", version="1.5.0")
            >>> ref1.is_compatible_with(ref2)
            True
        """
        return (
            self.schema_id == other.schema_id
            and self.major_version() == other.major_version()
        )

    @classmethod
    def from_uri(cls, uri: str) -> "SchemaRef":
        """
        Parse a GreenLang schema URI into a SchemaRef.

        Args:
            uri: Schema URI in format: gl://schemas/{schema_id}@{version}[#{variant}]

        Returns:
            SchemaRef parsed from the URI.

        Raises:
            ValueError: If URI format is invalid.

        Example:
            >>> ref = SchemaRef.from_uri("gl://schemas/emissions/activity@1.3.0")
            >>> ref.schema_id
            'emissions/activity'
            >>> ref.version
            '1.3.0'
        """
        # Handle variant fragment
        variant = None
        if "#" in uri:
            uri, variant = uri.rsplit("#", 1)

        # Parse gl://schemas/{schema_id}@{version}
        if not uri.startswith("gl://schemas/"):
            raise ValueError(
                f"Invalid schema URI '{uri}'. Must start with 'gl://schemas/'."
            )

        remainder = uri[len("gl://schemas/"):]

        if "@" not in remainder:
            raise ValueError(
                f"Invalid schema URI '{uri}'. Must contain version after '@'."
            )

        schema_id, version = remainder.rsplit("@", 1)

        return cls(schema_id=schema_id, version=version, variant=variant)

    def __str__(self) -> str:
        """Return string representation."""
        return self.to_uri()

    def __repr__(self) -> str:
        """Return detailed representation."""
        if self.variant:
            return f"SchemaRef(schema_id='{self.schema_id}', version='{self.version}', variant='{self.variant}')"
        return f"SchemaRef(schema_id='{self.schema_id}', version='{self.version}')"

    def __hash__(self) -> int:
        """Return hash for use in sets and dicts."""
        return hash((self.schema_id, self.version, self.variant))

    def __eq__(self, other: object) -> bool:
        """Check equality with another SchemaRef."""
        if not isinstance(other, SchemaRef):
            return NotImplemented
        return (
            self.schema_id == other.schema_id
            and self.version == other.version
            and self.variant == other.variant
        )
