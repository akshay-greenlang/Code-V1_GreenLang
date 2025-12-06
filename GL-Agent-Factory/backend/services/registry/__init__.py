"""
Registry Service Module

Provides agent lifecycle management including:
- Agent registration and discovery
- Version management
- Capability indexing
- State machine transitions
"""

from services.registry.agent_registry_service import (
    AgentRegistryService,
    AgentSpec,
    AgentState,
    AgentRegistration,
)
from services.registry.version_manager import VersionManager

__all__ = [
    "AgentRegistryService",
    "AgentSpec",
    "AgentState",
    "AgentRegistration",
    "VersionManager",
]
