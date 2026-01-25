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

# VersionManager is optional - import if available
try:
    from services.registry.version_manager import VersionManager
    _HAS_VERSION_MANAGER = True
except ImportError:
    VersionManager = None  # type: ignore
    _HAS_VERSION_MANAGER = False

__all__ = [
    "AgentRegistryService",
    "AgentSpec",
    "AgentState",
    "AgentRegistration",
]

if _HAS_VERSION_MANAGER:
    __all__.append("VersionManager")
