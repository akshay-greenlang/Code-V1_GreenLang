"""
GreenLang Infrastructure Components
====================================

Core infrastructure for validation, caching, telemetry, and provenance tracking.

This module provides the foundational infrastructure components used throughout
the GreenLang framework for ensuring data integrity, performance optimization,
observability, and audit trails.

Components:
- ValidationFramework: Schema validation and data integrity
- CacheManager: Multi-tier caching with TTL support
- TelemetryCollector: Metrics collection and aggregation
- ProvenanceTracker: Data lineage and audit trails

Example:
    >>> from greenlang.infrastructure import ValidationFramework, CacheManager
    >>> validator = ValidationFramework()
    >>> cache = CacheManager()
"""

from greenlang.execution.infrastructure.base import (
    BaseInfrastructureComponent,
    InfrastructureConfig,
    ComponentStatus,
)
from greenlang.execution.infrastructure.validation import (
    ValidationFramework,
    ValidationResult,
    ValidationRule,
)
from greenlang.execution.infrastructure.cache import (
    CacheManager,
    CacheConfig,
    CacheEntry,
)
from greenlang.execution.infrastructure.telemetry import (
    TelemetryCollector,
    Metric,
    MetricType,
)
from greenlang.execution.infrastructure.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    DataLineage,
)

__all__ = [
    # Base classes
    'BaseInfrastructureComponent',
    'InfrastructureConfig',
    'ComponentStatus',
    # Validation
    'ValidationFramework',
    'ValidationResult',
    'ValidationRule',
    # Cache
    'CacheManager',
    'CacheConfig',
    'CacheEntry',
    # Telemetry
    'TelemetryCollector',
    'Metric',
    'MetricType',
    # Provenance
    'ProvenanceTracker',
    'ProvenanceRecord',
    'DataLineage',
]

__version__ = '1.0.0'