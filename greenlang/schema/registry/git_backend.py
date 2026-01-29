# -*- coding: utf-8 -*-
"""
Git-Backed Schema Registry for GL-FOUND-X-002.

This module implements a Git-backed schema registry that reads schemas
from a Git repository (local or remote) with version resolution via
file naming conventions.

Schema Path Convention:
    {schema_dir}/{domain}/{name}@{version}.yaml
    Example: schemas/emissions/activity@1.3.0.yaml

Key Features:
    - Read schemas from local or remote Git repositories
    - Version resolution via semver-named files
    - Support for latest version lookup with semver constraints
    - Caching of fetched schemas for performance
    - Git pull support for remote synchronization
    - YAML and JSON schema file support with automatic parsing

Example:
    >>> registry = GitSchemaRegistry("./schemas")
    >>> schema = registry.resolve("emissions/activity", "1.3.0")
    >>> print(schema.content)

    >>> versions = registry.list_versions("emissions/activity")
    >>> print(versions)  # ['2.0.0', '1.3.0', '1.2.0', '1.0.0']

    >>> latest = registry.get_latest("emissions/activity", "^1.0.0")
    >>> print(latest)  # '1.3.0'

    >>> if registry.exists("emissions/activity", "1.3.0"):
    ...     print("Schema exists!")

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from greenlang.schema.constants import SCHEMA_CACHE_TTL_SECONDS
from greenlang.schema.registry.resolver import SchemaRegistry, SchemaSource

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class SchemaNotFoundError(Exception):
    """
    Schema not found in registry.

    Raised when a requested schema ID and version combination
    cannot be found in the registry.

    Attributes:
        schema_id: The requested schema identifier.
        version: The requested version.

    Example:
        >>> raise SchemaNotFoundError("emissions/activity", "9.9.9")
        SchemaNotFoundError: Schema not found: emissions/activity@9.9.9
    """

    def __init__(self, schema_id: str, version: str):
        """
        Initialize SchemaNotFoundError.

        Args:
            schema_id: The schema identifier that was not found.
            version: The version that was not found.
        """
        self.schema_id = schema_id
        self.version = version
        super().__init__(f"Schema not found: {schema_id}@{version}")


class InvalidSchemaIdError(Exception):
    """
    Invalid schema ID format.

    Raised when a schema ID does not follow the expected format.

    Attributes:
        schema_id: The invalid schema identifier.
        reason: Explanation of why the ID is invalid.
    """

    def __init__(self, schema_id: str, reason: str):
        """
        Initialize InvalidSchemaIdError.

        Args:
            schema_id: The invalid schema identifier.
            reason: Explanation of why the ID is invalid.
        """
        self.schema_id = schema_id
        self.reason = reason
        super().__init__(f"Invalid schema ID '{schema_id}': {reason}")


class VersionConstraintError(Exception):
    """
    Invalid version constraint.

    Raised when a version constraint string cannot be parsed.

    Attributes:
        constraint: The invalid constraint string.
        reason: Explanation of why the constraint is invalid.
    """

    def __init__(self, constraint: str, reason: str):
        """
        Initialize VersionConstraintError.

        Args:
            constraint: The invalid constraint string.
            reason: Explanation of why the constraint is invalid.
        """
        self.constraint = constraint
        self.reason = reason
        super().__init__(f"Invalid version constraint '{constraint}': {reason}")


class GitOperationError(Exception):
    """
    Git operation failed.

    Raised when a Git command fails (e.g., pull, fetch).

    Attributes:
        operation: The Git operation that failed.
        message: Error message from Git.
    """

    def __init__(self, operation: str, message: str):
        """
        Initialize GitOperationError.

        Args:
            operation: The Git operation that failed.
            message: Error message from Git.
        """
        self.operation = operation
        self.message = message
        super().__init__(f"Git {operation} failed: {message}")


class SchemaParseError(Exception):
    """
    Schema parsing failed.

    Raised when a schema file cannot be parsed as YAML or JSON.

    Attributes:
        path: The path to the schema file.
        reason: Explanation of why parsing failed.
    """

    def __init__(self, path: str, reason: str):
        """
        Initialize SchemaParseError.

        Args:
            path: The path to the schema file.
            reason: Explanation of why parsing failed.
        """
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse schema at '{path}': {reason}")


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SchemaSourceModel(BaseModel):
    """
    Schema source from registry (Pydantic model version).

    This model contains the schema content along with metadata
    about where it came from and how it was resolved.

    Attributes:
        content: The raw schema content (YAML or JSON string).
        content_type: MIME type ("application/json" or "application/yaml").
        schema_id: The schema identifier.
        version: The resolved version string.
        path: The file path within the repository.
        etag: Optional entity tag for caching.
        commit_hash: Optional Git commit hash for provenance.
        parsed_content: Optional parsed schema content as dictionary.
    """

    content: str = Field(
        ...,
        description="Raw schema content (YAML or JSON string)"
    )
    content_type: str = Field(
        ...,
        description="MIME type: 'application/json' or 'application/yaml'"
    )
    schema_id: str = Field(
        ...,
        description="Schema identifier (e.g., 'emissions/activity')"
    )
    version: str = Field(
        ...,
        description="Resolved version string (e.g., '1.3.0')"
    )
    path: str = Field(
        ...,
        description="File path within the repository"
    )
    etag: Optional[str] = Field(
        default=None,
        description="Entity tag for caching (SHA-256 of content)"
    )
    commit_hash: Optional[str] = Field(
        default=None,
        description="Git commit hash for provenance tracking"
    )
    parsed_content: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parsed schema content as dictionary"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate content type is supported."""
        valid_types = {"application/yaml", "application/json"}
        if v not in valid_types:
            raise ValueError(f"Invalid content_type: {v}. Must be one of {valid_types}")
        return v


# =============================================================================
# CACHE ENTRY
# =============================================================================


@dataclass
class CachedSchema:
    """
    Cached schema entry with TTL support.

    Attributes:
        source: The cached SchemaSourceModel.
        cached_at: Timestamp when the schema was cached.
        ttl_seconds: Time-to-live in seconds.
    """

    source: SchemaSourceModel
    cached_at: float = field(default_factory=time.time)
    ttl_seconds: int = SCHEMA_CACHE_TTL_SECONDS

    def is_expired(self) -> bool:
        """
        Check if the cache entry has expired.

        Returns:
            True if the entry has expired, False otherwise.
        """
        return (time.time() - self.cached_at) > self.ttl_seconds


# =============================================================================
# SEMVER UTILITIES
# =============================================================================


# Semver regex pattern for parsing version strings
SEMVER_REGEX = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


@dataclass
class SemVer:
    """
    Semantic version representation.

    Provides comparison and parsing for semver strings.

    Attributes:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
        prerelease: Optional prerelease identifier.
        build: Optional build metadata.

    Example:
        >>> v = SemVer.parse("1.2.3-beta.1+build.123")
        >>> print(v.major, v.minor, v.patch)
        1 2 3
        >>> v1 = SemVer.parse("1.0.0")
        >>> v2 = SemVer.parse("2.0.0")
        >>> print(v1 < v2)
        True
    """

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version: str) -> "SemVer":
        """
        Parse a semver string into a SemVer object.

        Args:
            version: The version string to parse.

        Returns:
            SemVer object representing the version.

        Raises:
            ValueError: If the version string is not valid semver.

        Example:
            >>> v = SemVer.parse("1.2.3-beta.1+build.123")
            >>> v.major, v.minor, v.patch
            (1, 2, 3)
        """
        match = SEMVER_REGEX.match(version)
        if not match:
            raise ValueError(f"Invalid semver: {version}")

        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
            build=match.group("build"),
        )

    @classmethod
    def is_valid(cls, version: str) -> bool:
        """
        Check if a string is a valid semver version.

        Args:
            version: The version string to check.

        Returns:
            True if valid semver, False otherwise.

        Example:
            >>> SemVer.is_valid("1.2.3")
            True
            >>> SemVer.is_valid("invalid")
            False
        """
        return SEMVER_REGEX.match(version) is not None

    def __str__(self) -> str:
        """Convert to string representation."""
        result = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            result = f"{result}-{self.prerelease}"
        if self.build:
            result = f"{result}+{self.build}"
        return result

    def _prerelease_tuple(self) -> Tuple[bool, Tuple[Any, ...]]:
        """
        Convert prerelease to comparable tuple.

        Returns:
            Tuple of (has_prerelease, parts) for comparison.
            Versions without prerelease are considered greater.
        """
        if self.prerelease is None:
            # No prerelease = release version, sorts after prereleases
            return (True, ())

        # Split prerelease into parts
        parts = []
        for part in self.prerelease.split("."):
            if part.isdigit():
                parts.append((0, int(part)))  # Numeric parts sort first
            else:
                parts.append((1, part))  # String parts sort after
        return (False, tuple(parts))

    def __lt__(self, other: "SemVer") -> bool:
        """Compare less than."""
        if not isinstance(other, SemVer):
            return NotImplemented

        # Compare major.minor.patch first
        self_tuple = (self.major, self.minor, self.patch)
        other_tuple = (other.major, other.minor, other.patch)

        if self_tuple != other_tuple:
            return self_tuple < other_tuple

        # Then compare prerelease
        return self._prerelease_tuple() < other._prerelease_tuple()

    def __le__(self, other: "SemVer") -> bool:
        """Compare less than or equal."""
        return self < other or self == other

    def __gt__(self, other: "SemVer") -> bool:
        """Compare greater than."""
        if not isinstance(other, SemVer):
            return NotImplemented
        return other < self

    def __ge__(self, other: "SemVer") -> bool:
        """Compare greater than or equal."""
        return self > other or self == other

    def __eq__(self, other: object) -> bool:
        """Compare equality."""
        if not isinstance(other, SemVer):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def is_compatible_with(self, other: "SemVer") -> bool:
        """
        Check if this version is compatible with another (same major version).

        Args:
            other: The version to compare against.

        Returns:
            True if compatible (same major version), False otherwise.

        Example:
            >>> v1 = SemVer.parse("1.2.3")
            >>> v2 = SemVer.parse("1.5.0")
            >>> v1.is_compatible_with(v2)
            True
        """
        return self.major == other.major

    def bump_major(self) -> "SemVer":
        """Return a new SemVer with major version incremented."""
        return SemVer(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemVer":
        """Return a new SemVer with minor version incremented."""
        return SemVer(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemVer":
        """Return a new SemVer with patch version incremented."""
        return SemVer(self.major, self.minor, self.patch + 1)


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two semver version strings.

    Args:
        v1: First version string.
        v2: Second version string.

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2.

    Example:
        >>> compare_versions("1.0.0", "2.0.0")
        -1
        >>> compare_versions("2.0.0", "1.0.0")
        1
        >>> compare_versions("1.0.0", "1.0.0")
        0
    """
    try:
        sv1 = SemVer.parse(v1)
        sv2 = SemVer.parse(v2)
    except ValueError:
        # Fall back to string comparison if not valid semver
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        return 0

    if sv1 < sv2:
        return -1
    elif sv1 > sv2:
        return 1
    return 0


def sort_versions(versions: List[str], reverse: bool = True) -> List[str]:
    """
    Sort version strings by semver.

    Args:
        versions: List of version strings to sort.
        reverse: If True, sort newest first (default). If False, oldest first.

    Returns:
        Sorted list of version strings.

    Example:
        >>> sort_versions(["1.0.0", "2.0.0", "1.5.0"])
        ['2.0.0', '1.5.0', '1.0.0']
        >>> sort_versions(["1.0.0", "2.0.0", "1.5.0"], reverse=False)
        ['1.0.0', '1.5.0', '2.0.0']
    """
    def sort_key(v: str) -> Tuple[Any, ...]:
        try:
            sv = SemVer.parse(v)
            # Use tuple for comparison, with prerelease handling
            return (sv.major, sv.minor, sv.patch, sv._prerelease_tuple())
        except ValueError:
            # Fall back to string for invalid semver
            return (0, 0, 0, (False, (v,)))

    return sorted(versions, key=sort_key, reverse=reverse)


def filter_versions(
    versions: List[str],
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
    include_prerelease: bool = False
) -> List[str]:
    """
    Filter version list by criteria.

    Args:
        versions: List of version strings to filter.
        min_version: Minimum version (inclusive).
        max_version: Maximum version (inclusive).
        include_prerelease: Whether to include prerelease versions.

    Returns:
        Filtered list of version strings.

    Example:
        >>> filter_versions(["1.0.0", "1.5.0", "2.0.0"], min_version="1.2.0")
        ['1.5.0', '2.0.0']
    """
    result = []
    min_sv = SemVer.parse(min_version) if min_version else None
    max_sv = SemVer.parse(max_version) if max_version else None

    for v in versions:
        try:
            sv = SemVer.parse(v)

            # Filter prereleases if not requested
            if not include_prerelease and sv.prerelease:
                continue

            # Check min version
            if min_sv and sv < min_sv:
                continue

            # Check max version
            if max_sv and sv > max_sv:
                continue

            result.append(v)
        except ValueError:
            # Skip invalid versions
            continue

    return result


# =============================================================================
# VERSION CONSTRAINT
# =============================================================================


class VersionConstraint:
    """
    Semver constraint matching.

    Supports the following constraint formats:
        - "1.2.3" - Exact version match
        - "^1.2.3" - Compatible with 1.x.x (major version match)
        - "~1.2.3" - Approximately 1.2.x (minor version match)
        - ">=1.2.3" - Greater than or equal
        - ">1.2.3" - Greater than
        - "<=1.2.3" - Less than or equal
        - "<1.2.3" - Less than
        - "*" - Any version
        - "latest" - Latest available version

    Example:
        >>> constraint = VersionConstraint("^1.0.0")
        >>> constraint.matches("1.5.0")
        True
        >>> constraint.matches("2.0.0")
        False
    """

    # Constraint operators pattern
    CONSTRAINT_PATTERN = re.compile(
        r"^(?P<op>[\^~>=<]*)(?P<version>.+)$"
    )

    def __init__(self, constraint: str):
        """
        Initialize VersionConstraint.

        Args:
            constraint: The constraint string to parse.

        Raises:
            VersionConstraintError: If the constraint is invalid.
        """
        self.constraint = constraint.strip()
        self._op: str = ""
        self._version: Optional[SemVer] = None
        self._parse()

    def _parse(self) -> None:
        """
        Parse the constraint string.

        Raises:
            VersionConstraintError: If parsing fails.
        """
        if not self.constraint:
            raise VersionConstraintError(self.constraint, "Empty constraint")

        # Handle wildcard and latest
        if self.constraint in ("*", "latest"):
            self._op = "*"
            self._version = None
            return

        match = self.CONSTRAINT_PATTERN.match(self.constraint)
        if not match:
            raise VersionConstraintError(
                self.constraint,
                "Does not match expected format"
            )

        self._op = match.group("op") or "="
        version_str = match.group("version")

        # Validate operator
        valid_ops = {"^", "~", ">=", ">", "<=", "<", "=", ""}
        if self._op not in valid_ops:
            raise VersionConstraintError(
                self.constraint,
                f"Invalid operator '{self._op}'"
            )

        # Parse version
        try:
            self._version = SemVer.parse(version_str)
        except ValueError as e:
            raise VersionConstraintError(
                self.constraint,
                f"Invalid version: {e}"
            )

    def matches(self, version: str) -> bool:
        """
        Check if a version matches this constraint.

        Args:
            version: The version string to check.

        Returns:
            True if the version matches the constraint.

        Example:
            >>> c = VersionConstraint(">=1.0.0")
            >>> c.matches("1.5.0")
            True
            >>> c.matches("0.9.0")
            False
        """
        # Wildcard matches everything
        if self._op == "*":
            return True

        try:
            sv = SemVer.parse(version)
        except ValueError:
            return False

        if self._version is None:
            return False

        # Exact match
        if self._op in ("=", ""):
            return sv == self._version

        # Caret: ^1.2.3 matches >=1.2.3 and <2.0.0 (same major)
        if self._op == "^":
            if sv < self._version:
                return False
            return sv.major == self._version.major

        # Tilde: ~1.2.3 matches >=1.2.3 and <1.3.0 (same major.minor)
        if self._op == "~":
            if sv < self._version:
                return False
            return (
                sv.major == self._version.major
                and sv.minor == self._version.minor
            )

        # Comparison operators
        if self._op == ">=":
            return sv >= self._version
        if self._op == ">":
            return sv > self._version
        if self._op == "<=":
            return sv <= self._version
        if self._op == "<":
            return sv < self._version

        return False

    def __str__(self) -> str:
        """String representation."""
        return self.constraint

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"VersionConstraint({self.constraint!r})"


# =============================================================================
# SCHEMA CACHE
# =============================================================================


class SchemaCache:
    """
    Thread-safe schema cache with TTL support.

    This class provides a thread-safe cache for schema content with
    configurable TTL and optional size limits.

    Attributes:
        ttl_seconds: Time-to-live for cache entries in seconds.
        max_size: Maximum number of entries in cache (0 for unlimited).

    Example:
        >>> cache = SchemaCache(ttl_seconds=3600, max_size=1000)
        >>> cache.set("key", schema_model)
        >>> cached = cache.get("key")
    """

    def __init__(
        self,
        ttl_seconds: int = SCHEMA_CACHE_TTL_SECONDS,
        max_size: int = 0
    ):
        """
        Initialize SchemaCache.

        Args:
            ttl_seconds: Time-to-live for entries in seconds.
            max_size: Maximum entries (0 for unlimited).
        """
        self._cache: Dict[str, CachedSchema] = {}
        self._lock = threading.RLock()
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

    def get(self, key: str) -> Optional[SchemaSourceModel]:
        """
        Get a schema from cache.

        Args:
            key: Cache key (typically "schema_id@version").

        Returns:
            Cached SchemaSourceModel or None if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._cache[key]
                return None
            return entry.source

    def set(self, key: str, source: SchemaSourceModel) -> None:
        """
        Store a schema in cache.

        Args:
            key: Cache key.
            source: Schema source to cache.
        """
        with self._lock:
            # Enforce max_size if set
            if self.max_size > 0 and len(self._cache) >= self.max_size:
                self._evict_oldest()

            self._cache[key] = CachedSchema(
                source=source,
                ttl_seconds=self.ttl_seconds,
            )

    def delete(self, key: str) -> bool:
        """
        Delete a schema from cache.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidate all entries with matching prefix.

        Args:
            prefix: Key prefix to match.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for key in keys:
                del self._cache[key]
            return len(keys)

    def _evict_oldest(self) -> None:
        """Evict the oldest entry to make room."""
        if not self._cache:
            return

        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].cached_at
        )
        del self._cache[oldest_key]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        with self._lock:
            now = time.time()
            expired_count = sum(
                1 for entry in self._cache.values() if entry.is_expired()
            )
            return {
                "total_entries": len(self._cache),
                "expired_entries": expired_count,
                "valid_entries": len(self._cache) - expired_count,
                "ttl_seconds": self.ttl_seconds,
                "max_size": self.max_size,
            }


# =============================================================================
# GIT SCHEMA REGISTRY
# =============================================================================


class GitSchemaRegistry(SchemaRegistry):
    """
    Git-backed schema registry.

    Reads schemas from a Git repository with path convention:
        {schema_dir}/{domain}/{name}@{version}.yaml

    Supports both YAML and JSON schema files with automatic content type detection
    and parsing.

    Attributes:
        repo_path: Path to the Git repository.
        remote_url: Optional remote URL for pulling updates.
        branch: Git branch to use (default: "main").
        schema_dir: Directory within the repo containing schemas (default: "schemas").

    Example:
        >>> registry = GitSchemaRegistry("./my-schemas")
        >>> schema = registry.resolve("emissions/activity", "1.3.0")
        >>> print(schema.content)

        >>> versions = registry.list_versions("emissions/activity")
        >>> print(versions)  # ['2.0.0', '1.3.0', '1.2.0']

        >>> latest = registry.get_latest("emissions/activity", "^1.0.0")
        >>> print(latest)  # '1.3.0'

        >>> if registry.exists("emissions/activity", "1.3.0"):
        ...     print("Schema exists!")
    """

    # Supported schema file extensions
    SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}

    # Content type mapping
    CONTENT_TYPES = {
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".json": "application/json",
    }

    # Pattern for extracting version from filename
    # Matches: name@version.ext (e.g., activity@1.3.0.yaml)
    # Version can contain digits, dots, hyphens, alphanumerics (for semver with prerelease)
    VERSION_FILENAME_PATTERN = re.compile(
        r"^(?P<name>[a-z][a-z0-9_]*)@(?P<version>[0-9][0-9a-zA-Z._+-]*)(?P<ext>\.(?:yaml|yml|json))$",
        re.IGNORECASE
    )

    def __init__(
        self,
        repo_path: str,
        remote_url: Optional[str] = None,
        branch: str = "main",
        schema_dir: str = "schemas",
        cache_ttl_seconds: int = SCHEMA_CACHE_TTL_SECONDS,
        cache_max_size: int = 1000,
        auto_parse: bool = True,
    ):
        """
        Initialize GitSchemaRegistry.

        Args:
            repo_path: Path to the Git repository (local directory).
            remote_url: Optional remote URL for pulling updates.
            branch: Git branch to use (default: "main").
            schema_dir: Directory containing schemas within the repo.
            cache_ttl_seconds: Cache time-to-live in seconds.
            cache_max_size: Maximum number of cached schemas.
            auto_parse: Whether to automatically parse YAML/JSON content.

        Example:
            >>> registry = GitSchemaRegistry(
            ...     repo_path="./schemas",
            ...     remote_url="https://github.com/org/schemas.git",
            ...     branch="main",
            ...     schema_dir="schemas"
            ... )
        """
        self.repo_path = Path(repo_path).resolve()
        self.remote_url = remote_url
        self.branch = branch
        self.schema_dir = schema_dir
        self.auto_parse = auto_parse

        # Internal cache
        self._cache = SchemaCache(
            ttl_seconds=cache_ttl_seconds,
            max_size=cache_max_size
        )

        # Lazy-loaded commit hash
        self._current_commit_hash: Optional[str] = None

        # Thread lock for git operations
        self._git_lock = threading.Lock()

        logger.debug(
            f"Initialized GitSchemaRegistry: repo_path={self.repo_path}, "
            f"schema_dir={self.schema_dir}, branch={self.branch}"
        )

    def resolve(
        self,
        schema_id: str,
        version: str
    ) -> SchemaSource:
        """
        Resolve schema by ID and version.

        Schema path: {schema_dir}/{domain}/{name}@{version}.yaml
        Example: schemas/emissions/activity@1.3.0.yaml

        Args:
            schema_id: Schema identifier (e.g., "emissions/activity").
            version: Version string (e.g., "1.3.0").

        Returns:
            SchemaSource containing the schema content and metadata.

        Raises:
            SchemaNotFoundError: If the schema is not found.
            InvalidSchemaIdError: If the schema_id format is invalid.
            SchemaParseError: If the schema cannot be parsed.

        Example:
            >>> source = registry.resolve("emissions/activity", "1.3.0")
            >>> print(source.content_type)
            'application/yaml'
        """
        start_time = time.time()

        # Check cache first
        cache_key = f"{schema_id}@{version}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {cache_key}")
            # Convert to SchemaSource for interface compatibility
            return self._source_model_to_schema_source(cached)

        # Parse schema_id
        domain, name = self._parse_schema_id(schema_id)

        # Try to find the schema file
        schema_path = self._find_schema_file(domain, name, version)
        if schema_path is None:
            raise SchemaNotFoundError(schema_id, version)

        # Read the file
        content = self._read_file(schema_path)
        content_type = self._detect_content_type(schema_path)

        # Parse content if auto_parse is enabled
        parsed_content = None
        if self.auto_parse:
            parsed_content = self._parse_content(content, content_type, str(schema_path))

        # Compute etag
        etag = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Get commit hash
        commit_hash = self._get_commit_hash()

        # Create source model for caching
        source_model = SchemaSourceModel(
            content=content,
            content_type=content_type,
            schema_id=schema_id,
            version=version,
            path=str(schema_path.relative_to(self.repo_path)),
            etag=etag,
            commit_hash=commit_hash,
            parsed_content=parsed_content,
        )

        # Cache it
        self._cache.set(cache_key, source_model)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Resolved {cache_key} in {elapsed_ms:.2f}ms "
            f"(path={schema_path})"
        )

        # Return SchemaSource for interface compatibility
        return self._source_model_to_schema_source(source_model)

    def resolve_parsed(
        self,
        schema_id: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Resolve schema and return parsed content.

        This is a convenience method that resolves the schema and returns
        the parsed YAML/JSON content directly.

        Args:
            schema_id: Schema identifier.
            version: Version string.

        Returns:
            Parsed schema content as dictionary.

        Raises:
            SchemaNotFoundError: If the schema is not found.
            SchemaParseError: If the schema cannot be parsed.

        Example:
            >>> schema = registry.resolve_parsed("emissions/activity", "1.3.0")
            >>> print(schema.get("$schema"))
        """
        source = self.resolve(schema_id, version)
        content = source.content.get("_raw", "")
        content_type = source.content.get("_content_type", "application/yaml")

        if "_parsed" in source.content:
            return source.content["_parsed"]

        return self._parse_content(content, content_type, source.source_uri)

    def list_versions(self, schema_id: str) -> List[str]:
        """
        List available versions for a schema.

        Returns versions sorted by semver (newest first).

        Args:
            schema_id: Schema identifier (e.g., "emissions/activity").

        Returns:
            List of version strings sorted by semver (newest first).

        Raises:
            InvalidSchemaIdError: If the schema_id format is invalid.

        Example:
            >>> versions = registry.list_versions("emissions/activity")
            >>> print(versions)
            ['2.0.0', '1.3.0', '1.2.0', '1.0.0']
        """
        domain, name = self._parse_schema_id(schema_id)

        # Find all schema files for this schema
        schema_files = self._list_schema_files(domain, name)

        # Extract versions from filenames
        versions = []
        for file_path in schema_files:
            version = self._parse_version_from_filename(file_path.name)
            if version:
                versions.append(version)

        # Sort by semver (newest first)
        return sort_versions(versions, reverse=True)

    def get_latest(
        self,
        schema_id: str,
        constraint: Optional[str] = None
    ) -> str:
        """
        Get latest version matching constraint.

        Args:
            schema_id: Schema identifier.
            constraint: Optional semver constraint (e.g., "^1.0.0", ">=1.2.0").
                If None, returns the absolute latest version.

        Returns:
            Latest matching version string.

        Raises:
            SchemaNotFoundError: If no matching version is found.
            InvalidSchemaIdError: If the schema_id format is invalid.
            VersionConstraintError: If the constraint is invalid.

        Example:
            >>> latest = registry.get_latest("emissions/activity")
            >>> print(latest)  # '2.0.0'

            >>> latest = registry.get_latest("emissions/activity", "^1.0.0")
            >>> print(latest)  # '1.3.0'
        """
        versions = self.list_versions(schema_id)

        if not versions:
            raise SchemaNotFoundError(schema_id, constraint or "latest")

        # If no constraint, return the latest
        if constraint is None:
            return versions[0]

        # Parse constraint
        version_constraint = VersionConstraint(constraint)

        # Find the first (newest) version that matches
        for version in versions:
            if version_constraint.matches(version):
                return version

        # No matching version found
        raise SchemaNotFoundError(
            schema_id,
            f"matching constraint {constraint}"
        )

    def exists(self, schema_id: str, version: str) -> bool:
        """
        Check if a schema version exists.

        Args:
            schema_id: Schema identifier.
            version: Version string.

        Returns:
            True if the schema exists, False otherwise.

        Example:
            >>> registry.exists("emissions/activity", "1.3.0")
            True
            >>> registry.exists("emissions/activity", "9.9.9")
            False
        """
        try:
            domain, name = self._parse_schema_id(schema_id)
            schema_path = self._find_schema_file(domain, name, version)
            return schema_path is not None
        except InvalidSchemaIdError:
            return False

    def list_schemas(self, domain: Optional[str] = None) -> List[str]:
        """
        List all available schema IDs.

        Args:
            domain: Optional domain to filter by.

        Returns:
            List of schema IDs.

        Example:
            >>> schemas = registry.list_schemas()
            >>> print(schemas)
            ['emissions/activity', 'emissions/factors', 'energy/consumption']
        """
        schemas = set()
        base_path = self.repo_path / self.schema_dir

        if not base_path.exists():
            return []

        if domain:
            # List schemas in specific domain
            domain_path = base_path / domain
            if domain_path.exists():
                for file_path in domain_path.iterdir():
                    if file_path.is_file():
                        match = self.VERSION_FILENAME_PATTERN.match(file_path.name)
                        if match:
                            name = match.group("name")
                            schemas.add(f"{domain}/{name}")
        else:
            # List all schemas across all domains
            for domain_path in base_path.iterdir():
                if domain_path.is_dir():
                    domain_name = domain_path.name
                    for file_path in domain_path.iterdir():
                        if file_path.is_file():
                            match = self.VERSION_FILENAME_PATTERN.match(file_path.name)
                            if match:
                                name = match.group("name")
                                schemas.add(f"{domain_name}/{name}")

        return sorted(schemas)

    def _source_model_to_schema_source(
        self,
        model: SchemaSourceModel
    ) -> SchemaSource:
        """
        Convert SchemaSourceModel to SchemaSource for interface compatibility.

        Args:
            model: The SchemaSourceModel to convert.

        Returns:
            SchemaSource compatible with the resolver interface.
        """
        content: Dict[str, Any] = {
            "_raw": model.content,
            "_content_type": model.content_type,
        }
        if model.parsed_content:
            content["_parsed"] = model.parsed_content
            # Merge parsed content for easier access
            content.update(model.parsed_content)

        return SchemaSource(
            schema_id=model.schema_id,
            version=model.version,
            content=content,
            source_uri=model.path,
        )

    def _parse_content(
        self,
        content: str,
        content_type: str,
        path: str
    ) -> Dict[str, Any]:
        """
        Parse schema content as YAML or JSON.

        Args:
            content: Raw content string.
            content_type: MIME type.
            path: File path for error messages.

        Returns:
            Parsed content as dictionary.

        Raises:
            SchemaParseError: If parsing fails.
        """
        try:
            if content_type == "application/json":
                import json
                return json.loads(content)
            else:
                # YAML can parse both YAML and JSON
                return yaml.safe_load(content) or {}
        except yaml.YAMLError as e:
            raise SchemaParseError(path, f"YAML parse error: {e}")
        except Exception as e:
            raise SchemaParseError(path, str(e))

    def _get_schema_path(
        self,
        schema_id: str,
        version: str
    ) -> Path:
        """
        Construct path to schema file.

        Args:
            schema_id: Schema identifier.
            version: Version string.

        Returns:
            Path to the schema file (may not exist).

        Example:
            >>> path = registry._get_schema_path("emissions/activity", "1.3.0")
            >>> print(path)
            PosixPath('/repo/schemas/emissions/activity@1.3.0.yaml')
        """
        domain, name = self._parse_schema_id(schema_id)

        # Build base path
        base_path = self.repo_path / self.schema_dir
        if domain:
            base_path = base_path / domain

        # Return path with .yaml extension (primary)
        return base_path / f"{name}@{version}.yaml"

    def _parse_schema_id(self, schema_id: str) -> Tuple[str, str]:
        """
        Parse schema_id into domain and name.

        Args:
            schema_id: Schema identifier (e.g., "emissions/activity" or "simple").

        Returns:
            Tuple of (domain, name). Domain is empty string if not specified.

        Raises:
            InvalidSchemaIdError: If the schema_id is invalid.

        Example:
            >>> registry._parse_schema_id("emissions/activity")
            ('emissions', 'activity')
            >>> registry._parse_schema_id("simple")
            ('', 'simple')
        """
        if not schema_id:
            raise InvalidSchemaIdError(schema_id, "Empty schema ID")

        parts = schema_id.split("/")

        if len(parts) == 1:
            # No domain, just name
            name = parts[0]
            if not re.match(r"^[a-z][a-z0-9_]*$", name):
                raise InvalidSchemaIdError(
                    schema_id,
                    "Name must be lowercase alphanumeric with underscores"
                )
            return ("", name)

        if len(parts) == 2:
            # domain/name
            domain, name = parts
            if not re.match(r"^[a-z][a-z0-9_]*$", domain):
                raise InvalidSchemaIdError(
                    schema_id,
                    "Domain must be lowercase alphanumeric with underscores"
                )
            if not re.match(r"^[a-z][a-z0-9_]*$", name):
                raise InvalidSchemaIdError(
                    schema_id,
                    "Name must be lowercase alphanumeric with underscores"
                )
            return (domain, name)

        # Nested domains (e.g., domain/subdomain/name)
        domain = "/".join(parts[:-1])
        name = parts[-1]

        for part in parts:
            if not re.match(r"^[a-z][a-z0-9_]*$", part):
                raise InvalidSchemaIdError(
                    schema_id,
                    f"Part '{part}' must be lowercase alphanumeric with underscores"
                )

        return (domain, name)

    def _read_file(self, path: Path) -> str:
        """
        Read file content.

        Args:
            path: Path to the file.

        Returns:
            File content as string.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read.
        """
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise IOError(f"Failed to read {path}: {e}")

    def _detect_content_type(self, path: Path) -> str:
        """
        Detect content type from file extension.

        Args:
            path: Path to the file.

        Returns:
            MIME type string.

        Example:
            >>> registry._detect_content_type(Path("schema.yaml"))
            'application/yaml'
        """
        ext = path.suffix.lower()
        return self.CONTENT_TYPES.get(ext, "application/yaml")

    def _get_commit_hash(self) -> str:
        """
        Get current Git commit hash.

        Returns:
            The current HEAD commit hash, or empty string if not a git repo.
        """
        if self._current_commit_hash is not None:
            return self._current_commit_hash

        try:
            with self._git_lock:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    self._current_commit_hash = result.stdout.strip()
                else:
                    self._current_commit_hash = ""
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            self._current_commit_hash = ""

        return self._current_commit_hash

    def _find_schema_file(
        self,
        domain: str,
        name: str,
        version: str
    ) -> Optional[Path]:
        """
        Find the schema file, trying multiple extensions.

        Args:
            domain: Schema domain (may be empty).
            name: Schema name.
            version: Version string.

        Returns:
            Path to the schema file if found, None otherwise.
        """
        # Build base path
        base_path = self.repo_path / self.schema_dir
        if domain:
            base_path = base_path / domain

        # Try each supported extension
        for ext in [".yaml", ".yml", ".json"]:
            file_path = base_path / f"{name}@{version}{ext}"
            if file_path.exists():
                return file_path

        return None

    def _list_schema_files(self, domain: str, name: str) -> List[Path]:
        """
        List all version files for a schema.

        Args:
            domain: Schema domain (may be empty).
            name: Schema name.

        Returns:
            List of paths to all version files for this schema.
        """
        # Build base path
        base_path = self.repo_path / self.schema_dir
        if domain:
            base_path = base_path / domain

        if not base_path.exists():
            return []

        # Find all files matching name@version.ext pattern
        files = []
        pattern = re.compile(
            rf"^{re.escape(name)}@[^@]+\.(yaml|yml|json)$",
            re.IGNORECASE
        )

        for file_path in base_path.iterdir():
            if file_path.is_file() and pattern.match(file_path.name):
                files.append(file_path)

        return files

    def _parse_version_from_filename(
        self,
        filename: str
    ) -> Optional[str]:
        """
        Extract version from filename like 'activity@1.3.0.yaml'.

        Args:
            filename: The filename to parse.

        Returns:
            Version string if found, None otherwise.

        Example:
            >>> registry._parse_version_from_filename("activity@1.3.0.yaml")
            '1.3.0'
        """
        match = self.VERSION_FILENAME_PATTERN.match(filename)
        if match:
            return match.group("version")
        return None

    def pull(self) -> bool:
        """
        Pull latest from remote.

        Returns:
            True if the repository was updated, False if already up-to-date.

        Raises:
            GitOperationError: If the pull operation fails.

        Example:
            >>> if registry.pull():
            ...     print("Updated!")
            ... else:
            ...     print("Already up-to-date")
        """
        if self.remote_url is None:
            logger.warning("Cannot pull: no remote_url configured")
            return False

        old_hash = self._get_commit_hash()

        try:
            with self._git_lock:
                # Fetch from remote
                result = subprocess.run(
                    ["git", "fetch", "origin", self.branch],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    raise GitOperationError("fetch", result.stderr.strip())

                # Reset to remote branch
                result = subprocess.run(
                    ["git", "reset", "--hard", f"origin/{self.branch}"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    raise GitOperationError("reset", result.stderr.strip())

                # Clear cached commit hash to force refresh
                self._current_commit_hash = None
                new_hash = self._get_commit_hash()

            # Check if we got new commits
            updated = old_hash != new_hash

            if updated:
                logger.info(
                    f"Pulled updates from remote: {old_hash[:8]}..{new_hash[:8]}"
                )
                # Invalidate cache on updates
                self.invalidate_cache()
            else:
                logger.debug("Already up-to-date")

            return updated

        except subprocess.TimeoutExpired:
            raise GitOperationError("pull", "Operation timed out")
        except FileNotFoundError:
            raise GitOperationError("pull", "Git command not found")

    def invalidate_cache(self, schema_id: Optional[str] = None) -> None:
        """
        Invalidate cached schemas.

        Args:
            schema_id: Optional schema ID to invalidate. If None, invalidates all.

        Example:
            >>> registry.invalidate_cache()  # Clear all
            >>> registry.invalidate_cache("emissions/activity")  # Clear specific
        """
        if schema_id is None:
            # Clear all
            count = self._cache.clear()
            self._current_commit_hash = None
            logger.debug(f"Invalidated all {count} cached schemas")
        else:
            # Clear specific schema (all versions)
            count = self._cache.invalidate_prefix(f"{schema_id}@")
            logger.debug(
                f"Invalidated {count} cached versions of {schema_id}"
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics.

        Example:
            >>> stats = registry.get_cache_stats()
            >>> print(stats)
            {'total_entries': 10, 'expired_entries': 2, ...}
        """
        return self._cache.get_stats()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_git_registry(
    repo_path: str,
    remote_url: Optional[str] = None,
    branch: str = "main",
    schema_dir: str = "schemas",
    cache_ttl_seconds: int = SCHEMA_CACHE_TTL_SECONDS,
    auto_pull: bool = False,
) -> GitSchemaRegistry:
    """
    Factory function to create a GitSchemaRegistry.

    This is a convenience function that creates a registry and optionally
    pulls the latest changes from the remote.

    Args:
        repo_path: Path to the Git repository.
        remote_url: Optional remote URL for pulling updates.
        branch: Git branch to use.
        schema_dir: Directory containing schemas.
        cache_ttl_seconds: Cache TTL in seconds.
        auto_pull: Whether to pull latest on creation.

    Returns:
        Configured GitSchemaRegistry instance.

    Example:
        >>> registry = create_git_registry(
        ...     "./schemas",
        ...     remote_url="https://github.com/org/schemas.git",
        ...     auto_pull=True
        ... )
    """
    registry = GitSchemaRegistry(
        repo_path=repo_path,
        remote_url=remote_url,
        branch=branch,
        schema_dir=schema_dir,
        cache_ttl_seconds=cache_ttl_seconds,
    )

    if auto_pull and remote_url:
        try:
            registry.pull()
        except GitOperationError as e:
            logger.warning(f"Auto-pull failed: {e}")

    return registry


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Main class
    "GitSchemaRegistry",
    # Factory
    "create_git_registry",
    # Exceptions
    "SchemaNotFoundError",
    "InvalidSchemaIdError",
    "VersionConstraintError",
    "GitOperationError",
    "SchemaParseError",
    # Models
    "SchemaSourceModel",
    "CachedSchema",
    "SemVer",
    "VersionConstraint",
    # Cache
    "SchemaCache",
    # Helper functions
    "compare_versions",
    "sort_versions",
    "filter_versions",
]
