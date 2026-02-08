# -*- coding: utf-8 -*-
"""
GL-FOUND-X-007: GreenLang Agent Registry & Service Catalog SDK
===============================================================

This package provides the agent registry, service catalog, health
checking, dependency resolution, capability matching, and provenance
tracking SDK for the GreenLang framework. It supports:

- Thread-safe agent registration with full metadata, versioning, and capabilities
- Agent discovery by layer, sector, capability, tag, and health status
- Health checking with TTL-based refresh and history tracking
- Dependency resolution with topological sort and cycle detection
- Capability matching for finding agents by required capabilities
- SHA-256 provenance tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_AGENT_REGISTRY_ env prefix

Key Components:
    - registry: AgentRegistry for agent CRUD and indexed discovery
    - health_checker: HealthChecker for probe-based health monitoring
    - dependency_resolver: DependencyResolver for topological sorting
    - capability_matcher: CapabilityMatcher for capability-based discovery
    - provenance: ProvenanceTracker for SHA-256 audit trails
    - config: AgentRegistryConfig with GL_AGENT_REGISTRY_ env prefix
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: AgentRegistryService facade

Example:
    >>> from greenlang.agent_registry import AgentRegistry, AgentMetadataEntry
    >>> r = AgentRegistry()
    >>> # Register, query, health-check, resolve dependencies...

    >>> from greenlang.agent_registry import AgentRegistryService
    >>> service = AgentRegistryService()
    >>> service.startup()

Agent ID: GL-FOUND-X-007
Agent Name: Agent Registry & Service Catalog
"""

__version__ = "1.0.0"
__agent_id__ = "GL-FOUND-X-007"
__agent_name__ = "Agent Registry & Service Catalog"

# SDK availability flag
AGENT_REGISTRY_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.agent_registry.config import (
    AgentRegistryConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, results, provenance)
# ---------------------------------------------------------------------------
from greenlang.agent_registry.models import (
    # Enumerations
    AgentLayer,
    SectorClassification,
    AgentHealthStatus,
    ExecutionMode,
    IdempotencySupport,
    CapabilityCategory,
    RegistryChangeType,
    # Core models
    ResourceProfile,
    ContainerSpec,
    LegacyHttpConfig,
    SemanticVersion,
    AgentCapability,
    AgentVariant,
    AgentDependency,
    AgentMetadataEntry,
    RegistryQueryInput,
    RegistryQueryOutput,
    DependencyResolutionInput,
    DependencyResolutionOutput,
    # SDK additions
    HealthCheckResult,
    ServiceCatalogEntry,
    RegistryAuditEntry,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.agent_registry.registry import AgentRegistry
from greenlang.agent_registry.health_checker import HealthChecker
from greenlang.agent_registry.dependency_resolver import DependencyResolver
from greenlang.agent_registry.capability_matcher import CapabilityMatcher
from greenlang.agent_registry.provenance import ProvenanceTracker, ProvenanceEntry

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.agent_registry.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    agent_registry_operations_total,
    agent_registry_operation_duration_seconds,
    agent_registry_registrations_total,
    agent_registry_unregistrations_total,
    agent_registry_queries_total,
    agent_registry_query_results_total,
    agent_registry_health_checks_total,
    agent_registry_agents_total,
    agent_registry_agents_by_health,
    agent_registry_hot_reloads_total,
    agent_registry_dependency_resolutions_total,
    agent_registry_dependency_depth,
    # Helper functions
    record_operation,
    record_registration,
    record_unregistration,
    record_query,
    record_query_results,
    record_health_check,
    update_agents_count,
    update_agents_by_health,
    record_hot_reload,
    record_dependency_resolution,
    record_dependency_depth,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.agent_registry.setup import (
    AgentRegistryService,
    configure_agent_registry,
    get_agent_registry,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "AGENT_REGISTRY_SDK_AVAILABLE",
    # Configuration
    "AgentRegistryConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations
    "AgentLayer",
    "SectorClassification",
    "AgentHealthStatus",
    "ExecutionMode",
    "IdempotencySupport",
    "CapabilityCategory",
    "RegistryChangeType",
    # Core models
    "ResourceProfile",
    "ContainerSpec",
    "LegacyHttpConfig",
    "SemanticVersion",
    "AgentCapability",
    "AgentVariant",
    "AgentDependency",
    "AgentMetadataEntry",
    "RegistryQueryInput",
    "RegistryQueryOutput",
    "DependencyResolutionInput",
    "DependencyResolutionOutput",
    # SDK additions
    "HealthCheckResult",
    "ServiceCatalogEntry",
    "RegistryAuditEntry",
    # Core engines
    "AgentRegistry",
    "HealthChecker",
    "DependencyResolver",
    "CapabilityMatcher",
    "ProvenanceTracker",
    "ProvenanceEntry",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "agent_registry_operations_total",
    "agent_registry_operation_duration_seconds",
    "agent_registry_registrations_total",
    "agent_registry_unregistrations_total",
    "agent_registry_queries_total",
    "agent_registry_query_results_total",
    "agent_registry_health_checks_total",
    "agent_registry_agents_total",
    "agent_registry_agents_by_health",
    "agent_registry_hot_reloads_total",
    "agent_registry_dependency_resolutions_total",
    "agent_registry_dependency_depth",
    # Metric helper functions
    "record_operation",
    "record_registration",
    "record_unregistration",
    "record_query",
    "record_query_results",
    "record_health_check",
    "update_agents_count",
    "update_agents_by_health",
    "record_hot_reload",
    "record_dependency_resolution",
    "record_dependency_depth",
    # Service setup facade
    "AgentRegistryService",
    "configure_agent_registry",
    "get_agent_registry",
    "get_router",
]
