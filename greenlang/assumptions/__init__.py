# -*- coding: utf-8 -*-
"""
GL-FOUND-X-004: GreenLang Assumptions Registry SDK
===================================================

This package provides the assumptions management, scenario analysis,
and provenance tracking SDK for the GreenLang framework. It supports:

- Version-controlled assumption registry with full CRUD
- Scenario management (baseline, conservative, optimistic, custom)
- Value validation with custom rules and data type checking
- SHA-256 provenance tracking for complete audit trails
- Dependency graph tracking between assumptions and calculations
- Sensitivity analysis across scenarios
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_ASSUMPTIONS_ env prefix

Key Components:
    - registry: AssumptionRegistry for assumption CRUD and versioning
    - scenarios: ScenarioManager for what-if analysis
    - validator: AssumptionValidator for rule-based validation
    - provenance: ProvenanceTracker for SHA-256 audit trails
    - dependencies: DependencyTracker for graph analysis
    - config: AssumptionsConfig with GL_ASSUMPTIONS_ env prefix
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: AssumptionsService facade

Example:
    >>> from greenlang.assumptions import AssumptionRegistry
    >>> r = AssumptionRegistry()
    >>> a = r.create("ef.electricity", "Grid EF", "Emission factor",
    ...     "emission_factor", "float", 0.42,
    ...     user_id="analyst", change_reason="Initial",
    ...     metadata_source="EPA")
    >>> print(r.get_value("ef.electricity"))  # 0.42

    >>> from greenlang.assumptions import ScenarioManager
    >>> s = ScenarioManager()
    >>> scenarios = s.list()
    >>> print(len(scenarios))  # 3 (baseline, conservative, optimistic)

Agent ID: GL-FOUND-X-004
Agent Name: Assumptions Registry
"""

__version__ = "1.0.0"
__agent_id__ = "GL-FOUND-X-004"
__agent_name__ = "Assumptions Registry"

# SDK availability flag
ASSUMPTIONS_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.assumptions.config import (
    AssumptionsConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, results, provenance)
# ---------------------------------------------------------------------------
from greenlang.assumptions.models import (
    # Enumerations
    AssumptionDataType,
    AssumptionCategory,
    ScenarioType,
    ChangeType,
    ValidationSeverity,
    # Core models
    ValidationRule,
    ValidationResult,
    AssumptionMetadata,
    AssumptionVersion,
    Assumption,
    Scenario,
    # Audit models
    ChangeLogEntry,
    # Graph models
    DependencyNode,
    # Analysis models
    SensitivityResult,
    # Value models
    AssumptionValue,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.assumptions.registry import AssumptionRegistry
from greenlang.assumptions.scenarios import ScenarioManager
from greenlang.assumptions.validator import AssumptionValidator
from greenlang.assumptions.provenance import ProvenanceTracker
from greenlang.assumptions.dependencies import DependencyTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.assumptions.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    assumptions_operations_total,
    assumptions_operation_duration_seconds,
    assumptions_validations_total,
    assumptions_validation_failures_total,
    assumptions_scenario_accesses_total,
    assumptions_version_creates_total,
    assumptions_change_log_entries,
    assumptions_total,
    assumptions_scenarios_total,
    assumptions_cache_hits_total,
    assumptions_cache_misses_total,
    assumptions_dependency_depth,
    # Helper functions
    record_operation,
    record_validation,
    record_validation_failure,
    record_scenario_access,
    record_version_create,
    update_change_log_count,
    update_assumptions_count,
    update_scenarios_count,
    record_cache_hit,
    record_cache_miss,
    record_dependency_depth,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.assumptions.setup import (
    AssumptionsService,
    configure_assumptions_service,
    get_assumptions_service,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "ASSUMPTIONS_SDK_AVAILABLE",
    # Configuration
    "AssumptionsConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations
    "AssumptionDataType",
    "AssumptionCategory",
    "ScenarioType",
    "ChangeType",
    "ValidationSeverity",
    # Core models
    "ValidationRule",
    "ValidationResult",
    "AssumptionMetadata",
    "AssumptionVersion",
    "Assumption",
    "Scenario",
    # Audit models
    "ChangeLogEntry",
    # Graph models
    "DependencyNode",
    # Analysis models
    "SensitivityResult",
    # Value models
    "AssumptionValue",
    # Core engines
    "AssumptionRegistry",
    "ScenarioManager",
    "AssumptionValidator",
    "ProvenanceTracker",
    "DependencyTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "assumptions_operations_total",
    "assumptions_operation_duration_seconds",
    "assumptions_validations_total",
    "assumptions_validation_failures_total",
    "assumptions_scenario_accesses_total",
    "assumptions_version_creates_total",
    "assumptions_change_log_entries",
    "assumptions_total",
    "assumptions_scenarios_total",
    "assumptions_cache_hits_total",
    "assumptions_cache_misses_total",
    "assumptions_dependency_depth",
    # Metric helper functions
    "record_operation",
    "record_validation",
    "record_validation_failure",
    "record_scenario_access",
    "record_version_create",
    "update_change_log_count",
    "update_assumptions_count",
    "update_scenarios_count",
    "record_cache_hit",
    "record_cache_miss",
    "record_dependency_depth",
    # Service setup facade
    "AssumptionsService",
    "configure_assumptions_service",
    "get_assumptions_service",
    "get_router",
]
