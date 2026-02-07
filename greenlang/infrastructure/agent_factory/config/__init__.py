"""
Agent Factory Config Module - INFRA-010

Provides configuration management for agents in the GreenLang Climate OS
platform. Implements two-layer storage (Redis + PostgreSQL), hot reload
via Redis pub/sub, structured diff with breaking change detection, and
Pydantic v2 schema validation with environment-specific defaults.

Public API:
    - ConfigHotReload: Live config reload via Redis pub/sub.
    - ConfigChangeEvent: Change notification dataclass.
    - AgentConfigSchema: Pydantic v2 config model with validation.
    - ConfigStore: Two-layer config persistence (Redis + PostgreSQL).
    - ConfigDiff: Structured diff with breaking change detection.
    - ConfigChange: Single field-level change dataclass.
    - ChangeType: Change type enumeration.

Example:
    >>> from greenlang.infrastructure.agent_factory.config import (
    ...     ConfigStore, AgentConfigSchema, ConfigHotReload, ConfigDiff,
    ... )
    >>> config = AgentConfigSchema(agent_key="intake-agent", version=1)
    >>> await store.set_config("intake-agent", config)
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.config.diff import (
    ChangeType,
    ConfigChange,
    ConfigDiff,
)
from greenlang.infrastructure.agent_factory.config.hot_reload import (
    ConfigChangeEvent,
    ConfigHotReload,
)
from greenlang.infrastructure.agent_factory.config.schema import (
    AgentConfigSchema,
    CircuitBreakerConfigSchema,
    CURRENT_SCHEMA_VERSION,
    ResourceLimitsSchema,
    RetryConfigSchema,
)
from greenlang.infrastructure.agent_factory.config.store import (
    ConfigStore,
    ConfigVersionConflictError,
)

__all__ = [
    # Hot Reload
    "ConfigChangeEvent",
    "ConfigHotReload",
    # Schema
    "AgentConfigSchema",
    "CircuitBreakerConfigSchema",
    "CURRENT_SCHEMA_VERSION",
    "ResourceLimitsSchema",
    "RetryConfigSchema",
    # Store
    "ConfigStore",
    "ConfigVersionConflictError",
    # Diff
    "ChangeType",
    "ConfigChange",
    "ConfigDiff",
]
