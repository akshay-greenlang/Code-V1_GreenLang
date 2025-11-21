# -*- coding: utf-8 -*-
"""
Advanced Dependency Resolver for GreenLang Packs

This module provides comprehensive dependency resolution with graph-based
algorithms, conflict detection, and version constraint satisfaction.
"""

import json
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime
import networkx as nx
from packaging import version, specifiers
from packaging.requirements import Requirement
from greenlang.determinism import DeterministicClock


class ResolutionStrategy(Enum):
    """Dependency resolution strategies"""

    LATEST = "latest"  # Always prefer latest compatible version
    MINIMAL = "minimal"  # Prefer minimal compatible version
    LOCKED = "locked"  # Use locked versions from lock file
    CONSERVATIVE = "conservative"  # Minimize version changes


@dataclass
class PackageInfo:
    """Information about a package/pack"""

    name: str
    version: str
    dependencies: List["DependencySpec"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def __hash__(self):
        return hash((self.name, self.version))

    def __eq__(self, other):
        if not isinstance(other, PackageInfo):
            return False
        return self.name == other.name and self.version == other.version


@dataclass
class DependencySpec:
    """Dependency specification"""

    name: str
    version_spec: str  # e.g., ">=1.0.0,<2.0.0"
    optional: bool = False
    group: Optional[str] = None  # e.g., "dev", "test"
    source: Optional[str] = None  # e.g., "pypi", "hub", "github"

    def matches(self, version_str: str) -> bool:
        """Check if a version satisfies this dependency spec"""
        try:
            spec_set = specifiers.SpecifierSet(self.version_spec)
            return version.parse(version_str) in spec_set
        except Exception:
            # Fallback to simple string comparison
            if self.version_spec.startswith("=="):
                return version_str == self.version_spec[2:].strip()
            elif self.version_spec.startswith(">="):
                return version.parse(version_str) >= version.parse(
                    self.version_spec[2:].strip()
                )
            return True

    def __str__(self):
        return f"{self.name}{self.version_spec}"


@dataclass
class ResolutionConflict:
    """Information about a dependency conflict"""

    package: str
    requesters: List[Tuple[str, str]]  # [(package, version_spec), ...]
    available_versions: List[str]
    reason: str

    def __str__(self):
        requesters_str = ", ".join(f"{p} requires {v}" for p, v in self.requesters)
        return f"Conflict for {self.package}: {requesters_str}. Reason: {self.reason}"


@dataclass
class ResolutionResult:
    """Result of dependency resolution"""

    resolved: Dict[str, str]  # {package_name: version}
    graph: nx.DiGraph
    conflicts: List[ResolutionConflict]
    warnings: List[str]
    install_order: List[str]

    @property
    def success(self) -> bool:
        return len(self.conflicts) == 0


class DependencyResolver:
    """
    Advanced dependency resolver for GreenLang packs.

    Features:
    - Graph-based dependency resolution
    - Conflict detection and reporting
    - Circular dependency detection
    - Version constraint satisfaction
    - Multiple resolution strategies
    - Lock file support
    """

    def __init__(
        self, registry=None, strategy: ResolutionStrategy = ResolutionStrategy.LATEST
    ):
        """
        Initialize the dependency resolver.

        Args:
            registry: Pack registry for fetching package information
            strategy: Resolution strategy to use
        """
        self.registry = registry
        self.strategy = strategy
        self.graph = nx.DiGraph()
        self.package_pool: Dict[str, List[PackageInfo]] = defaultdict(list)
        self.resolved: Dict[str, str] = {}
        self.conflicts: List[ResolutionConflict] = []
        self.warnings: List[str] = []

    def add_package_to_pool(self, package: PackageInfo) -> None:
        """Add a package to the available package pool"""
        self.package_pool[package.name].append(package)
        # Sort by version (latest first)
        self.package_pool[package.name].sort(
            key=lambda p: version.parse(p.version), reverse=True
        )

    def load_package_info(self, name: str, version_spec: str = "") -> List[PackageInfo]:
        """Load package information from registry"""
        if self.registry:
            # Get available versions from registry
            versions = self.registry.get_versions(name)
            packages = []

            for ver in versions:
                if not version_spec or DependencySpec(name, version_spec).matches(ver):
                    # Get package metadata
                    metadata = self.registry.get_metadata(name, ver)
                    deps = []

                    # Parse dependencies
                    for dep in metadata.get("dependencies", []):
                        if isinstance(dep, str):
                            # Parse requirement string
                            try:
                                req = Requirement(dep)
                                deps.append(
                                    DependencySpec(
                                        name=req.name,
                                        version_spec=(
                                            str(req.specifier) if req.specifier else ""
                                        ),
                                    )
                                )
                            except:
                                # Simple format: "package>=1.0"
                                parts = dep.split(">=")
                                if len(parts) == 2:
                                    deps.append(
                                        DependencySpec(
                                            name=parts[0].strip(),
                                            version_spec=f">={parts[1].strip()}",
                                        )
                                    )
                                else:
                                    deps.append(
                                        DependencySpec(name=dep, version_spec="")
                                    )
                        elif isinstance(dep, dict):
                            deps.append(
                                DependencySpec(
                                    name=dep.get("name", ""),
                                    version_spec=dep.get("version", ""),
                                    optional=dep.get("optional", False),
                                    group=dep.get("group"),
                                    source=dep.get("source"),
                                )
                            )

                    packages.append(
                        PackageInfo(
                            name=name, version=ver, dependencies=deps, metadata=metadata
                        )
                    )

            return packages

        # Return from pool if no registry
        return self.package_pool.get(name, [])

    def resolve(
        self,
        requirements: List[DependencySpec],
        locked_versions: Optional[Dict[str, str]] = None,
        include_optional: bool = False,
        include_groups: Optional[List[str]] = None,
    ) -> ResolutionResult:
        """
        Resolve dependencies for given requirements.

        Args:
            requirements: List of top-level requirements
            locked_versions: Previously locked versions to prefer
            include_optional: Include optional dependencies
            include_groups: Include dependencies from specific groups

        Returns:
            ResolutionResult with resolved versions or conflicts
        """
        self.resolved = {}
        self.conflicts = []
        self.warnings = []
        self.graph = nx.DiGraph()

        # Use locked versions if strategy is LOCKED
        if self.strategy == ResolutionStrategy.LOCKED and locked_versions:
            self.resolved = copy.deepcopy(locked_versions)

        # Build dependency graph
        queue = deque([(None, req) for req in requirements])
        visited = set()

        while queue:
            parent, dep_spec = queue.popleft()

            # Skip if already processed
            if dep_spec.name in visited:
                continue
            visited.add(dep_spec.name)

            # Skip optional dependencies if not requested
            if dep_spec.optional and not include_optional:
                continue

            # Skip group dependencies if not requested
            if (
                dep_spec.group
                and include_groups
                and dep_spec.group not in include_groups
            ):
                continue

            # Find compatible version
            compatible_version = self._find_compatible_version(
                dep_spec,
                locked_versions.get(dep_spec.name) if locked_versions else None,
            )

            if not compatible_version:
                # Record conflict
                self.conflicts.append(
                    ResolutionConflict(
                        package=dep_spec.name,
                        requesters=[(parent, dep_spec.version_spec)] if parent else [],
                        available_versions=[
                            p.version for p in self.package_pool.get(dep_spec.name, [])
                        ],
                        reason="No compatible version found",
                    )
                )
                continue

            # Add to resolved
            self.resolved[dep_spec.name] = compatible_version.version

            # Add to graph
            if parent:
                self.graph.add_edge(parent, dep_spec.name)
            else:
                self.graph.add_node(dep_spec.name)

            # Add sub-dependencies to queue
            for sub_dep in compatible_version.dependencies:
                if sub_dep.name not in visited:
                    queue.append((dep_spec.name, sub_dep))

        # Check for conflicts
        self._check_conflicts()

        # Check for circular dependencies
        cycles = self._detect_cycles()
        if cycles:
            for cycle in cycles:
                self.warnings.append(
                    f"Circular dependency detected: {' -> '.join(cycle)}"
                )

        # Calculate installation order
        install_order = []
        if not self.conflicts:
            try:
                install_order = list(nx.topological_sort(self.graph))
            except nx.NetworkXError:
                # Graph has cycles, use partial order
                install_order = list(self.resolved.keys())

        return ResolutionResult(
            resolved=self.resolved,
            graph=self.graph,
            conflicts=self.conflicts,
            warnings=self.warnings,
            install_order=install_order,
        )

    def _find_compatible_version(
        self, dep_spec: DependencySpec, locked_version: Optional[str] = None
    ) -> Optional[PackageInfo]:
        """Find a compatible version for a dependency spec"""
        # Load available packages
        packages = self.load_package_info(dep_spec.name, dep_spec.version_spec)

        if not packages:
            # Try to fetch from pool
            packages = self.package_pool.get(dep_spec.name, [])

        # Filter compatible versions
        compatible = [p for p in packages if dep_spec.matches(p.version)]

        if not compatible:
            return None

        # Apply resolution strategy
        if self.strategy == ResolutionStrategy.LOCKED and locked_version:
            # Try to use locked version
            for pkg in compatible:
                if pkg.version == locked_version:
                    return pkg

        if self.strategy == ResolutionStrategy.MINIMAL:
            # Return minimum compatible version
            return min(compatible, key=lambda p: version.parse(p.version))

        if self.strategy == ResolutionStrategy.CONSERVATIVE:
            # Prefer currently resolved version if compatible
            if dep_spec.name in self.resolved:
                current = self.resolved[dep_spec.name]
                for pkg in compatible:
                    if pkg.version == current:
                        return pkg

        # Default: return latest compatible version
        return max(compatible, key=lambda p: version.parse(p.version))

    def _check_conflicts(self) -> None:
        """Check for version conflicts in resolved dependencies"""
        # Group constraints by package
        constraints: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for node in self.graph.nodes():
            if node in self.resolved:
                # Get all edges pointing to this node
                for parent in self.graph.predecessors(node):
                    # Get the constraint from parent
                    parent_info = self._get_package_info(
                        parent, self.resolved.get(parent)
                    )
                    if parent_info:
                        for dep in parent_info.dependencies:
                            if dep.name == node:
                                constraints[node].append((parent, dep.version_spec))

        # Check if all constraints are satisfied
        for package, version_constraints in constraints.items():
            resolved_version = self.resolved.get(package)
            if not resolved_version:
                continue

            unsatisfied = []
            for requester, constraint in version_constraints:
                if constraint and not DependencySpec(package, constraint).matches(
                    resolved_version
                ):
                    unsatisfied.append((requester, constraint))

            if unsatisfied:
                self.conflicts.append(
                    ResolutionConflict(
                        package=package,
                        requesters=unsatisfied,
                        available_versions=[
                            p.version for p in self.package_pool.get(package, [])
                        ],
                        reason=f"Version {resolved_version} does not satisfy all constraints",
                    )
                )

    def _detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies in the graph"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except:
            return []

    def _get_package_info(self, name: str, version: str) -> Optional[PackageInfo]:
        """Get package info for a specific version"""
        for pkg in self.package_pool.get(name, []):
            if pkg.version == version:
                return pkg

        # Try to load from registry
        packages = self.load_package_info(name, f"=={version}")
        return packages[0] if packages else None

    def generate_lock_file(self, result: ResolutionResult) -> Dict[str, Any]:
        """Generate a lock file from resolution result"""
        lock_data = {
            "version": "1.0",
            "resolved_at": DeterministicClock.now().isoformat(),
            "strategy": self.strategy.value,
            "packages": {},
        }

        for name, version in result.resolved.items():
            pkg_info = self._get_package_info(name, version)
            if pkg_info:
                lock_data["packages"][name] = {
                    "version": version,
                    "dependencies": [str(dep) for dep in pkg_info.dependencies],
                    "source": pkg_info.source,
                    "metadata": pkg_info.metadata,
                }

        return lock_data

    def load_lock_file(self, lock_file: Path) -> Dict[str, str]:
        """Load locked versions from a lock file"""
        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        locked_versions = {}
        for name, info in lock_data.get("packages", {}).items():
            locked_versions[name] = info["version"]

        return locked_versions


def resolve_dependencies(
    requirements: List[Union[str, DependencySpec]],
    registry=None,
    strategy: ResolutionStrategy = ResolutionStrategy.LATEST,
    lock_file: Optional[Path] = None,
) -> ResolutionResult:
    """
    Convenience function to resolve dependencies.

    Args:
        requirements: List of requirements (strings or DependencySpec objects)
        registry: Pack registry
        strategy: Resolution strategy
        lock_file: Optional lock file path

    Returns:
        ResolutionResult
    """
    resolver = DependencyResolver(registry=registry, strategy=strategy)

    # Parse requirements
    dep_specs = []
    for req in requirements:
        if isinstance(req, str):
            # Parse requirement string
            try:
                parsed = Requirement(req)
                dep_specs.append(
                    DependencySpec(
                        name=parsed.name,
                        version_spec=str(parsed.specifier) if parsed.specifier else "",
                    )
                )
            except:
                # Simple format
                dep_specs.append(DependencySpec(name=req, version_spec=""))
        else:
            dep_specs.append(req)

    # Load lock file if provided
    locked_versions = None
    if lock_file and lock_file.exists():
        locked_versions = resolver.load_lock_file(lock_file)

    return resolver.resolve(dep_specs, locked_versions=locked_versions)
