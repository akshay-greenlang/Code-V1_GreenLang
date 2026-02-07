# -*- coding: utf-8 -*-
"""
Centralized Audit Logging Service - SEC-005

A unified audit logging service that consolidates audit events from all
GreenLang security components (auth, RBAC, encryption) into a single,
consistent audit trail with multi-destination routing.

**Architecture:**
    +------------------+
    |  AuditService    |  <-- Main entry point
    +--------+---------+
             |
    +--------v---------+
    | AuditEventCollector | <-- Async queue with backpressure
    +--------+---------+
             |
    +--------v---------+
    | AuditEventEnricher  | <-- GeoIP, user agent, context
    +--------+---------+
             |
    +--------v---------+
    |  AuditEventRouter   | <-- Multi-destination routing
    +--------+---------+
             |
     +-------+-------+-------+
     |       |       |       |
     v       v       v       v
   Postgres  Loki   Redis   (future)

**Features:**
- 70+ event types across 8 categories
- Async, non-blocking event collection
- Backpressure handling (10,000 event queue)
- GeoIP and user agent enrichment
- Multi-destination routing (PostgreSQL, Loki, Redis)
- Event deduplication with 5-minute TTL
- PII redaction for compliance
- REST API for querying and searching
- Real-time event streaming via WebSocket
- Compliance reporting (SOC2, ISO27001, GDPR)
- Multi-format export (CSV, JSON, Parquet)

**Quick Start:**
    >>> from greenlang.infrastructure.audit_service import (
    ...     AuditService,
    ...     UnifiedAuditEventType,
    ...     AuditResult,
    ... )
    >>> service = AuditService(db_pool=pool, redis_client=redis)
    >>> await service.start()
    >>> await service.log_auth_event(
    ...     event_type=UnifiedAuditEventType.AUTH_LOGIN_SUCCESS,
    ...     user_id="u-123",
    ...     tenant_id="t-corp",
    ...     client_ip="10.0.0.1",
    ... )
    >>> await service.stop()

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Module Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Core Service Components (Phase 1)
# ---------------------------------------------------------------------------

# Event Types
from greenlang.infrastructure.audit_service.event_types import (
    AuditEventCategory,
    AuditSeverity,
    AuditAction,
    AuditResult,
    UnifiedAuditEventType,
    AUTH_EVENT_TYPE_MAP,
    RBAC_EVENT_TYPE_MAP,
    ENCRYPTION_EVENT_TYPE_MAP,
)

# Event Model
from greenlang.infrastructure.audit_service.event_model import (
    UnifiedAuditEvent,
    EventBuilder,
)

# Collector
from greenlang.infrastructure.audit_service.event_collector import (
    AuditEventCollector,
    CollectorConfig,
    CollectorMetrics,
)

# Enricher
from greenlang.infrastructure.audit_service.event_enricher import (
    AuditEventEnricher,
    EnricherConfig,
    set_request_context,
    set_user_context,
    get_request_context,
    get_user_context,
    clear_context,
)

# Router
from greenlang.infrastructure.audit_service.event_router import (
    AuditEventRouter,
    RouterConfig,
    RouterMetrics,
)

# Cache
from greenlang.infrastructure.audit_service.audit_cache import (
    AuditCache,
    AuditCacheConfig,
    CacheMetrics,
)

# Service
from greenlang.infrastructure.audit_service.audit_service import (
    AuditService,
    AuditServiceConfig,
    AuditServiceMetrics,
    get_audit_service,
    configure_audit_service,
    shutdown_audit_service,
)


# ---------------------------------------------------------------------------
# API exports (Phase 2+)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.audit_service.api import audit_router
except ImportError:
    audit_router = None

# ---------------------------------------------------------------------------
# Reporting exports (Phase 2+)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.audit_service.reporting import (
        ComplianceReportService,
        SOC2ReportGenerator,
        ISO27001ReportGenerator,
        GDPRReportGenerator,
    )
except ImportError:
    ComplianceReportService = None
    SOC2ReportGenerator = None
    ISO27001ReportGenerator = None
    GDPRReportGenerator = None

# ---------------------------------------------------------------------------
# Export service exports (Phase 2+)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.audit_service.export import (
        AuditExportService,
        CSVExporter,
        JSONExporter,
        ParquetExporter,
    )
except ImportError:
    AuditExportService = None
    CSVExporter = None
    JSONExporter = None
    ParquetExporter = None


# ---------------------------------------------------------------------------
# Middleware exports (Phase 5)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.audit_service.middleware import (
    AuditMiddleware,
    create_audit_middleware,
    AuditRequestContext,
    AuditResponseContext,
    AuditUserContext,
    AuditEvent,
)

from greenlang.infrastructure.audit_service.exclusions import (
    AuditExclusionRules,
    EXCLUDED_PATHS,
    EXCLUDED_PREFIXES,
    SENSITIVITY_MAP,
    SENSITIVITY_LEVELS,
    get_exclusion_rules,
    should_exclude,
    get_sensitivity,
)


# ---------------------------------------------------------------------------
# Metrics exports (Phase 6)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.audit_service.audit_metrics import (
    AuditMetrics,
    get_audit_metrics,
)


# ---------------------------------------------------------------------------
# Retention exports (Phase 7)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.audit_service.retention import (
        RetentionPolicyService,
        RetentionTier,
        RETENTION_TIERS,
        COMPLIANCE_RETENTION,
        DataClassification,
        ArchivalService,
        ArchivalStatus,
        ArchiveMetadata,
    )
except ImportError:
    RetentionPolicyService = None
    RetentionTier = None
    RETENTION_TIERS = None
    COMPLIANCE_RETENTION = None
    DataClassification = None
    ArchivalService = None
    ArchivalStatus = None
    ArchiveMetadata = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Event Types
    "AuditEventCategory",
    "AuditSeverity",
    "AuditAction",
    "AuditResult",
    "UnifiedAuditEventType",
    "AUTH_EVENT_TYPE_MAP",
    "RBAC_EVENT_TYPE_MAP",
    "ENCRYPTION_EVENT_TYPE_MAP",
    # Event Model
    "UnifiedAuditEvent",
    "EventBuilder",
    # Collector
    "AuditEventCollector",
    "CollectorConfig",
    "CollectorMetrics",
    # Enricher
    "AuditEventEnricher",
    "EnricherConfig",
    "set_request_context",
    "set_user_context",
    "get_request_context",
    "get_user_context",
    "clear_context",
    # Router
    "AuditEventRouter",
    "RouterConfig",
    "RouterMetrics",
    # Cache
    "AuditCache",
    "AuditCacheConfig",
    "CacheMetrics",
    # Service
    "AuditService",
    "AuditServiceConfig",
    "AuditServiceMetrics",
    "get_audit_service",
    "configure_audit_service",
    "shutdown_audit_service",
    # API (Phase 2+)
    "audit_router",
    # Reporting (Phase 2+)
    "ComplianceReportService",
    "SOC2ReportGenerator",
    "ISO27001ReportGenerator",
    "GDPRReportGenerator",
    # Export (Phase 2+)
    "AuditExportService",
    "CSVExporter",
    "JSONExporter",
    "ParquetExporter",
    # Middleware (Phase 5)
    "AuditMiddleware",
    "create_audit_middleware",
    "AuditRequestContext",
    "AuditResponseContext",
    "AuditUserContext",
    "AuditEvent",
    # Exclusions (Phase 5)
    "AuditExclusionRules",
    "EXCLUDED_PATHS",
    "EXCLUDED_PREFIXES",
    "SENSITIVITY_MAP",
    "SENSITIVITY_LEVELS",
    "get_exclusion_rules",
    "should_exclude",
    "get_sensitivity",
    # Metrics (Phase 6)
    "AuditMetrics",
    "get_audit_metrics",
    # Retention (Phase 7)
    "RetentionPolicyService",
    "RetentionTier",
    "RETENTION_TIERS",
    "COMPLIANCE_RETENTION",
    "DataClassification",
    "ArchivalService",
    "ArchivalStatus",
    "ArchiveMetadata",
]
