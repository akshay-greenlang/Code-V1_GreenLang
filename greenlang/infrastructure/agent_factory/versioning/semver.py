# -*- coding: utf-8 -*-
"""
Semantic Versioning - Parse, compare, and match semver strings.

Implements the Semantic Versioning 2.0.0 specification with support for
pre-release identifiers, build metadata, range matching (^, ~, >=, <),
comparison operators, and version bumping.

Example:
    >>> v1 = SemanticVersion.parse("1.2.3-beta.1+build.42")
    >>> v2 = v1.bump_minor()
    >>> assert v2 == SemanticVersion.parse("1.3.0")
    >>> r = VersionRange.parse("^1.0.0")
    >>> assert r.satisfies(v1)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import total_ordering
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(
    r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


# ---------------------------------------------------------------------------
# SemanticVersion
# ---------------------------------------------------------------------------


@total_ordering
@dataclass(frozen=True)
class SemanticVersion:
    """Semantic version with full comparison support.

    Attributes:
        major: Major version (breaking changes).
        minor: Minor version (backward-compatible features).
        patch: Patch version (backward-compatible fixes).
        prerelease: Pre-release identifiers (e.g. 'beta.1').
        build: Build metadata (e.g. 'build.42'). Ignored in comparisons.
    """

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_str: str) -> SemanticVersion:
        """Parse a semantic version string.

        Args:
            version_str: Version string (e.g. '1.2.3', 'v1.2.3-beta.1+build.42').

        Returns:
            Parsed SemanticVersion.

        Raises:
            ValueError: If the string is not a valid semver.
        """
        match = _SEMVER_RE.match(version_str.strip())
        if not match:
            raise ValueError(f"Invalid semantic version: '{version_str}'")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    @property
    def core(self) -> Tuple[int, int, int]:
        """Return the (major, minor, patch) tuple."""
        return (self.major, self.minor, self.patch)

    def __str__(self) -> str:
        """Return the canonical string representation."""
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            s += f"-{self.prerelease}"
        if self.build:
            s += f"+{self.build}"
        return s

    def __repr__(self) -> str:
        return f"SemanticVersion({self!s})"

    def _prerelease_key(self) -> Tuple:
        """Create a sortable key for pre-release comparison.

        Per semver spec: pre-release versions have lower precedence.
        Numeric identifiers are compared as integers; alpha as strings.
        """
        if self.prerelease is None:
            # No prerelease = higher precedence than any prerelease
            return (1,)
        parts: List = []
        for ident in self.prerelease.split("."):
            if ident.isdigit():
                parts.append((0, int(ident)))
            else:
                parts.append((1, ident))
        return (0, tuple(parts))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self.core == other.core and self.prerelease == other.prerelease

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        if self.core != other.core:
            return self.core < other.core
        return self._prerelease_key() < other._prerelease_key()

    def __hash__(self) -> int:
        return hash((self.core, self.prerelease))

    # ------------------------------------------------------------------
    # Bump helpers
    # ------------------------------------------------------------------

    def bump_major(self) -> SemanticVersion:
        """Return a new version with major incremented, minor/patch reset."""
        return SemanticVersion(major=self.major + 1, minor=0, patch=0)

    def bump_minor(self) -> SemanticVersion:
        """Return a new version with minor incremented, patch reset."""
        return SemanticVersion(major=self.major, minor=self.minor + 1, patch=0)

    def bump_patch(self) -> SemanticVersion:
        """Return a new version with patch incremented."""
        return SemanticVersion(major=self.major, minor=self.minor, patch=self.patch + 1)

    # ------------------------------------------------------------------
    # Comparison utilities
    # ------------------------------------------------------------------

    @staticmethod
    def is_breaking_change(old: SemanticVersion, new: SemanticVersion) -> bool:
        """Check if the version change is a breaking change.

        A breaking change is indicated by a major version bump.

        Args:
            old: Previous version.
            new: New version.

        Returns:
            True if new.major > old.major.
        """
        return new.major > old.major

    @staticmethod
    def is_compatible(old: SemanticVersion, new: SemanticVersion) -> bool:
        """Check if two versions are compatible.

        Compatible means same major version and new >= old.

        Args:
            old: Base version.
            new: Candidate version.

        Returns:
            True if same major and new >= old.
        """
        return new.major == old.major and new >= old


# ---------------------------------------------------------------------------
# VersionRange
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionRange:
    """Semantic version range matcher.

    Supports the following constraint formats:
      *           - any version
      1.2.3       - exact match
      ^1.2.3      - caret: >=1.2.3, <2.0.0
      ~1.2.3      - tilde: >=1.2.3, <1.3.0
      >=1.0.0     - greater than or equal
      >=1.0.0,<2.0.0 - compound range

    Attributes:
        raw: Original constraint string.
    """

    raw: str

    @classmethod
    def parse(cls, constraint: str) -> VersionRange:
        """Parse a version range string.

        Args:
            constraint: Version range constraint.

        Returns:
            VersionRange instance.
        """
        return cls(raw=constraint.strip())

    def satisfies(self, version: SemanticVersion | str) -> bool:
        """Check if a version satisfies this range.

        Args:
            version: SemanticVersion instance or version string.

        Returns:
            True if the version is within the range.
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return self._check(version, self.raw)

    def _check(self, ver: SemanticVersion, constraint: str) -> bool:
        """Internal range check logic."""
        constraint = constraint.strip()
        if constraint in ("*", ""):
            return True

        # Compound range
        if "," in constraint:
            parts = [c.strip() for c in constraint.split(",")]
            return all(self._check(ver, p) for p in parts)

        # Caret
        if constraint.startswith("^"):
            base = SemanticVersion.parse(constraint[1:])
            upper = SemanticVersion(major=base.major + 1, minor=0, patch=0)
            return base <= ver < upper

        # Tilde
        if constraint.startswith("~"):
            base = SemanticVersion.parse(constraint[1:])
            upper = SemanticVersion(major=base.major, minor=base.minor + 1, patch=0)
            return base <= ver < upper

        # Comparison operators
        if constraint.startswith(">="):
            return ver >= SemanticVersion.parse(constraint[2:].strip())
        if constraint.startswith("<="):
            return ver <= SemanticVersion.parse(constraint[2:].strip())
        if constraint.startswith("!="):
            return ver != SemanticVersion.parse(constraint[2:].strip())
        if constraint.startswith("=="):
            return ver == SemanticVersion.parse(constraint[2:].strip())
        if constraint.startswith(">") and not constraint.startswith(">="):
            return ver > SemanticVersion.parse(constraint[1:].strip())
        if constraint.startswith("<") and not constraint.startswith("<="):
            return ver < SemanticVersion.parse(constraint[1:].strip())

        # Exact match
        return ver == SemanticVersion.parse(constraint)

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        return f"VersionRange('{self.raw}')"
