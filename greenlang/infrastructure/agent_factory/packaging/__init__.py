"""
Agent Factory Packaging - INFRA-010 Phase 3

Production-grade packaging system for GreenLang agent packages. Provides
pack format parsing, package building, dependency resolution, installation,
and manifest generation with SHA-256 integrity verification.

Public API:
    - PackFormat: Parse and validate agent.pack.yaml files.
    - PackageBuilder: Build .glpack archives from agent source directories.
    - DependencyResolver: Resolve agent dependency trees with semver matching.
    - PackageInstaller: Install/uninstall agent packages from .glpack archives.
    - ManifestGenerator: Generate and verify SHA-256 file manifests.

Example:
    >>> from greenlang.infrastructure.agent_factory.packaging import (
    ...     PackFormat, PackageBuilder, DependencyResolver,
    ...     PackageInstaller, ManifestGenerator,
    ... )
    >>> pack = PackFormat.load("my_agent/agent.pack.yaml")
    >>> result = await PackageBuilder().build("my_agent/", "dist/")
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.packaging.pack_format import (
    AgentDependency,
    AgentPack,
    AgentType,
    PackFormat,
    PythonDependency,
    ResourceSpec,
)
from greenlang.infrastructure.agent_factory.packaging.builder import (
    BuildResult,
    PackageBuilder,
)
from greenlang.infrastructure.agent_factory.packaging.resolver import (
    DependencyResolver,
    ResolvedDependency,
    ResolutionResult,
    VersionConflict,
)
from greenlang.infrastructure.agent_factory.packaging.installer import (
    InstallResult,
    PackageInstaller,
)
from greenlang.infrastructure.agent_factory.packaging.manifest import (
    ManifestGenerator,
    PackageManifest,
)

__all__ = [
    "AgentDependency",
    "AgentPack",
    "AgentType",
    "BuildResult",
    "DependencyResolver",
    "InstallResult",
    "ManifestGenerator",
    "PackFormat",
    "PackageBuilder",
    "PackageInstaller",
    "PackageManifest",
    "PythonDependency",
    "ResolvedDependency",
    "ResolutionResult",
    "ResourceSpec",
    "VersionConflict",
]
