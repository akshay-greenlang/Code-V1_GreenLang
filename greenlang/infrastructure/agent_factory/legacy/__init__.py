"""
Legacy Agent Migration Support - Agent Factory (INFRA-010)

Provides tools for discovering, registering, and generating pack manifests
for the 119 existing GreenLang agents so they can be managed by the Agent
Factory without requiring an immediate rewrite.

Public API:
    - LegacyAgentDiscovery: Scans the codebase for legacy agents.
    - DiscoveredAgent: Value object representing a found agent.
    - LegacyRegistrar: Registers discovered agents in the factory registry.
    - RegistrationReport: Summary of a batch registration operation.
    - LegacyPackGenerator: Generates agent.pack.yaml from agent code.

Example:
    >>> from greenlang.infrastructure.agent_factory.legacy import (
    ...     LegacyAgentDiscovery,
    ...     LegacyRegistrar,
    ...     LegacyPackGenerator,
    ... )
    >>> discovery = LegacyAgentDiscovery()
    >>> agents = discovery.discover_all()
    >>> registrar = LegacyRegistrar(db_pool, redis_client)
    >>> report = await registrar.register_all(agents)
    >>> generator = LegacyPackGenerator()
    >>> for agent in agents:
    ...     generator.write_pack_yaml(agent)
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.legacy.discovery import (
    DiscoveredAgent,
    LegacyAgentDiscovery,
)
from greenlang.infrastructure.agent_factory.legacy.registrar import (
    LegacyRegistrar,
    RegistrationReport,
)
from greenlang.infrastructure.agent_factory.legacy.pack_generator import (
    LegacyPackGenerator,
)

__all__ = [
    "DiscoveredAgent",
    "LegacyAgentDiscovery",
    "LegacyPackGenerator",
    "LegacyRegistrar",
    "RegistrationReport",
]
