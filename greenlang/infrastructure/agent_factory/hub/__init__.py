"""
Agent Factory Hub - INFRA-010 Phase 3

Central registry and distribution hub for GreenLang agent packages. Provides
package publishing, search, download, local indexing, and pre-publish validation.

Public API:
    - AgentHubRegistry: Central package registry with S3/PostgreSQL backend.
    - HubClient: Async REST client for remote hub access.
    - LocalIndex: Local agent package index with file locking.
    - HubValidator: Pre-publish validation for agent packages.

Example:
    >>> from greenlang.infrastructure.agent_factory.hub import (
    ...     AgentHubRegistry, HubClient, LocalIndex, HubValidator,
    ... )
    >>> registry = AgentHubRegistry(storage_path="./hub")
    >>> record = await registry.publish("dist/my-agent-1.0.0.glpack")
    >>> packages = await registry.search("emissions")
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.hub.registry import (
    AgentHubRegistry,
    PackageRecord,
)
from greenlang.infrastructure.agent_factory.hub.client import (
    HubClient,
    HubClientConfig,
)
from greenlang.infrastructure.agent_factory.hub.index import (
    IndexEntry,
    LocalIndex,
)
from greenlang.infrastructure.agent_factory.hub.validator import (
    HubValidator,
    ValidationCheck,
    ValidationResult,
)

__all__ = [
    "AgentHubRegistry",
    "HubClient",
    "HubClientConfig",
    "HubValidator",
    "IndexEntry",
    "LocalIndex",
    "PackageRecord",
    "ValidationCheck",
    "ValidationResult",
]
