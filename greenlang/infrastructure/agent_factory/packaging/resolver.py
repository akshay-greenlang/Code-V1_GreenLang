# -*- coding: utf-8 -*-
"""
Dependency Resolver - Resolve agent dependency trees with semver matching.

Resolves dependency trees declared in agent.pack.yaml files. Supports
semantic version range matching (^, ~, >=x,<y, exact), conflict detection
for incompatible version requirements, and diamond dependency resolution
by selecting the highest compatible version.

Example:
    >>> resolver = DependencyResolver(available_versions)
    >>> result = resolver.resolve(root_pack)
    >>> assert result.success
    >>> for dep in result.resolved:
    ...     print(dep.name, dep.resolved_version)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from greenlang.infrastructure.agent_factory.packaging.pack_format import (
    AgentDependency,
    AgentPack,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DEPENDENCY_DEPTH = 5

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedDependency:
    """A fully resolved agent dependency.

    Attributes:
        name: Agent key.
        resolved_version: Exact version that was selected.
        required_by: List of agent keys that required this dependency.
    """

    name: str
    resolved_version: str
    required_by: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class VersionConflict:
    """Description of a version conflict between two requirements.

    Attributes:
        dependency_name: The agent key with conflicting requirements.
        required_by_a: Agent that requires version_a.
        constraint_a: Version constraint from agent_a.
        required_by_b: Agent that requires version_b.
        constraint_b: Version constraint from agent_b.
    """

    dependency_name: str
    required_by_a: str
    constraint_a: str
    required_by_b: str
    constraint_b: str


@dataclass
class ResolutionResult:
    """Outcome of dependency resolution.

    Attributes:
        success: True if all dependencies were resolved without conflicts.
        resolved: List of resolved dependencies.
        conflicts: List of version conflicts encountered.
        warnings: Non-fatal warnings encountered during resolution.
    """

    success: bool
    resolved: List[ResolvedDependency] = field(default_factory=list)
    conflicts: List[VersionConflict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Semver range helpers (lightweight, full class in versioning module)
# ---------------------------------------------------------------------------


def _parse_version_tuple(version: str) -> Tuple[int, int, int]:
    """Parse a semver string into (major, minor, patch) tuple.

    Pre-release and build metadata are stripped for comparison purposes.
    """
    base = version.split("-")[0].split("+")[0]
    parts = base.split(".")
    if len(parts) < 3:
        parts.extend(["0"] * (3 - len(parts)))
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _satisfies(version: str, constraint: str) -> bool:
    """Check if a version satisfies a constraint string.

    Supported constraint formats:
        *           - any version
        1.2.3       - exact match
        ^1.2.3      - compatible with 1.x.x (>=1.2.3, <2.0.0)
        ~1.2.3      - patch-level changes (>=1.2.3, <1.3.0)
        >=1.0.0     - greater than or equal
        >=1.0.0,<2.0.0 - range
    """
    constraint = constraint.strip()
    if constraint in ("*", ""):
        return True

    ver = _parse_version_tuple(version)

    # Comma-separated range (e.g. >=1.0.0,<2.0.0)
    if "," in constraint:
        parts = [c.strip() for c in constraint.split(",")]
        return all(_satisfies(version, p) for p in parts)

    # Caret: ^1.2.3 means >=1.2.3, <2.0.0
    if constraint.startswith("^"):
        base = _parse_version_tuple(constraint[1:])
        upper = (base[0] + 1, 0, 0)
        return base <= ver < upper

    # Tilde: ~1.2.3 means >=1.2.3, <1.3.0
    if constraint.startswith("~"):
        base = _parse_version_tuple(constraint[1:])
        upper = (base[0], base[1] + 1, 0)
        return base <= ver < upper

    # Comparison operators
    if constraint.startswith(">="):
        return ver >= _parse_version_tuple(constraint[2:])
    if constraint.startswith("<="):
        return ver <= _parse_version_tuple(constraint[2:])
    if constraint.startswith(">") and not constraint.startswith(">="):
        return ver > _parse_version_tuple(constraint[1:])
    if constraint.startswith("<") and not constraint.startswith("<="):
        return ver < _parse_version_tuple(constraint[1:])
    if constraint.startswith("!="):
        return ver != _parse_version_tuple(constraint[2:])
    if constraint.startswith("=="):
        return ver == _parse_version_tuple(constraint[2:])

    # Exact match
    return ver == _parse_version_tuple(constraint)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class DependencyResolver:
    """Resolve agent dependency trees from pack.yaml declarations.

    Uses available version information to select the highest compatible
    version for each dependency, detect conflicts, and handle diamond
    dependencies.

    Attributes:
        available_versions: Mapping of agent_key -> list of available version strings.
        pack_registry: Mapping of (agent_key, version) -> AgentPack for transitive lookups.
    """

    def __init__(
        self,
        available_versions: Dict[str, List[str]],
        pack_registry: Optional[Dict[Tuple[str, str], AgentPack]] = None,
    ) -> None:
        """Initialize the resolver.

        Args:
            available_versions: Map of agent key to sorted list of available versions.
            pack_registry: Optional map of (key, version) -> AgentPack for transitive resolution.
        """
        self.available_versions = available_versions
        self.pack_registry = pack_registry or {}

    def resolve(self, root_pack: AgentPack) -> ResolutionResult:
        """Resolve all dependencies for an agent pack.

        Performs breadth-first resolution up to MAX_DEPENDENCY_DEPTH levels.

        Args:
            root_pack: The agent pack whose dependencies should be resolved.

        Returns:
            ResolutionResult with resolved versions, conflicts, and warnings.
        """
        resolved_map: Dict[str, ResolvedDependency] = {}
        conflicts: List[VersionConflict] = []
        warnings: List[str] = []
        # Track all constraints per dependency: dep_name -> [(constraint, required_by)]
        constraint_map: Dict[str, List[Tuple[str, str]]] = {}

        queue: List[Tuple[AgentPack, int]] = [(root_pack, 0)]
        visited: Set[str] = {root_pack.name}

        while queue:
            current_pack, depth = queue.pop(0)
            if depth >= MAX_DEPENDENCY_DEPTH:
                warnings.append(
                    f"Maximum dependency depth ({MAX_DEPENDENCY_DEPTH}) reached "
                    f"at agent '{current_pack.name}'."
                )
                continue

            for dep in current_pack.agent_dependencies:
                constraint_map.setdefault(dep.name, []).append(
                    (dep.version_constraint, current_pack.name)
                )

        # Now resolve each dependency
        for dep_name, constraints in constraint_map.items():
            available = self.available_versions.get(dep_name, [])
            if not available:
                conflicts.append(
                    VersionConflict(
                        dependency_name=dep_name,
                        required_by_a=constraints[0][1],
                        constraint_a=constraints[0][0],
                        required_by_b="",
                        constraint_b="(not available)",
                    )
                )
                continue

            best = self._find_best_version(dep_name, available, constraints)
            if best is None:
                # Build conflict report
                if len(constraints) >= 2:
                    conflicts.append(
                        VersionConflict(
                            dependency_name=dep_name,
                            required_by_a=constraints[0][1],
                            constraint_a=constraints[0][0],
                            required_by_b=constraints[1][1],
                            constraint_b=constraints[1][0],
                        )
                    )
                else:
                    conflicts.append(
                        VersionConflict(
                            dependency_name=dep_name,
                            required_by_a=constraints[0][1],
                            constraint_a=constraints[0][0],
                            required_by_b="",
                            constraint_b="(no compatible version found)",
                        )
                    )
                continue

            required_by = [c[1] for c in constraints]
            resolved_map[dep_name] = ResolvedDependency(
                name=dep_name,
                resolved_version=best,
                required_by=required_by,
            )

            # Enqueue transitive dependencies
            pack_key = (dep_name, best)
            if pack_key in self.pack_registry and dep_name not in visited:
                visited.add(dep_name)
                queue_depth = min(
                    d for (_, d_name), d in [] if False  # placeholder
                ) if False else depth + 1
                queue.append((self.pack_registry[pack_key], depth + 1))

        success = len(conflicts) == 0
        return ResolutionResult(
            success=success,
            resolved=list(resolved_map.values()),
            conflicts=conflicts,
            warnings=warnings,
        )

    def _find_best_version(
        self,
        dep_name: str,
        available: List[str],
        constraints: List[Tuple[str, str]],
    ) -> Optional[str]:
        """Find the highest version that satisfies all constraints.

        Diamond dependency resolution: pick the highest version that
        is compatible with every constraint from every requiring agent.

        Args:
            dep_name: Name of the dependency.
            available: Available version strings.
            constraints: List of (constraint_string, required_by_agent).

        Returns:
            The highest compatible version, or None if no version satisfies all constraints.
        """
        # Sort available versions descending (highest first)
        sorted_versions = sorted(
            available,
            key=lambda v: _parse_version_tuple(v),
            reverse=True,
        )

        for version in sorted_versions:
            if all(
                _satisfies(version, constraint)
                for constraint, _ in constraints
            ):
                logger.debug(
                    "Resolved %s -> %s (satisfies %d constraints)",
                    dep_name,
                    version,
                    len(constraints),
                )
                return version

        logger.warning(
            "No compatible version found for %s among %s",
            dep_name,
            sorted_versions[:5],
        )
        return None

    def check_single(
        self,
        dep_name: str,
        constraint: str,
    ) -> Optional[str]:
        """Resolve a single dependency without transitive resolution.

        Args:
            dep_name: Agent key.
            constraint: Semver constraint string.

        Returns:
            Best matching version string, or None.
        """
        available = self.available_versions.get(dep_name, [])
        return self._find_best_version(
            dep_name, available, [(constraint, "direct")]
        )
