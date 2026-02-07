"""
Agent Factory Versioning - INFRA-010 Phase 3

Production-grade semantic versioning, compatibility tracking, migration
framework, canary deployments, and automated rollback for GreenLang agents.

Public API:
    - SemanticVersion: Parse and compare semver strings.
    - VersionRange: Semver range matching with ^, ~, >=, < operators.
    - VersionCompatibilityMatrix: Track agent-to-agent compatibility.
    - VersionMigrationFramework: Register and execute version migration scripts.
    - CanaryController: Progressive canary deployment with auto-promote/rollback.
    - RollbackController: Automated rollback on error rate or latency thresholds.

Example:
    >>> from greenlang.infrastructure.agent_factory.versioning import (
    ...     SemanticVersion, VersionRange, CanaryController,
    ... )
    >>> v = SemanticVersion.parse("1.2.3")
    >>> r = VersionRange.parse("^1.0.0")
    >>> assert r.satisfies(v)
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.versioning.semver import (
    SemanticVersion,
    VersionRange,
)
from greenlang.infrastructure.agent_factory.versioning.compatibility import (
    CompatibilityEntry,
    CompatibilityStatus,
    VersionCompatibilityMatrix,
)
from greenlang.infrastructure.agent_factory.versioning.migration import (
    MigrationResult,
    MigrationScript,
    VersionMigrationFramework,
)
from greenlang.infrastructure.agent_factory.versioning.canary import (
    CanaryConfig,
    CanaryController,
    CanaryDeployment,
    CanaryStatus,
)
from greenlang.infrastructure.agent_factory.versioning.rollback import (
    RollbackConfig,
    RollbackController,
    RollbackResult,
)

__all__ = [
    "CanaryConfig",
    "CanaryController",
    "CanaryDeployment",
    "CanaryStatus",
    "CompatibilityEntry",
    "CompatibilityStatus",
    "MigrationResult",
    "MigrationScript",
    "RollbackConfig",
    "RollbackController",
    "RollbackResult",
    "SemanticVersion",
    "VersionCompatibilityMatrix",
    "VersionMigrationFramework",
    "VersionRange",
]
