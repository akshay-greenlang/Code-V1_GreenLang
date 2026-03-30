# -*- coding: utf-8 -*-
"""
GreenLang Shared Schema Infrastructure
========================================

Centralized base models, mixins, enums, and field factories that replace
duplicated patterns across 1,100+ agent model files.

Quick Start::

    # Base classes for new models
    from greenlang.schemas import (
        GreenLangBase,        # Minimal base (model_config only)
        GreenLangRecord,      # + timestamps + tenant + provenance
        GreenLangAuditRecord, # + audit actor tracking
        GreenLangRequest,     # For API requests
        GreenLangResponse,    # For API responses
        GreenLangConfig,      # For configuration models
        GreenLangResult,      # For calculation results
    )

    # Mixins for composition
    from greenlang.schemas import (
        TimestampMixin,       # created_at, updated_at
        TenantMixin,          # tenant_id
        ProvenanceMixin,      # provenance_hash
        AuditMixin,           # timestamps + created_by, updated_by
        MetadataMixin,        # metadata dict
    )

    # Utilities (replace 91+ _utcnow() definitions)
    from greenlang.schemas import utcnow, new_uuid, prefixed_uuid, compute_provenance_hash

    # Shared enums (replace 1,200+ duplicated enum definitions)
    from greenlang.schemas.enums import (
        CalculationStatus, JobStatus, Severity,
        Environment, LogLevel, HealthStatus, ReportFormat,
        AlertSeverity, AlertStatus, NotificationChannel,
    )

    # Field factories (replace 200+ duplicated field definitions)
    from greenlang.schemas.fields import id_field, tenant_field, provenance_field

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

# -- Base classes -----------------------------------------------------------
from greenlang.schemas.base import (
    GreenLangBase,
    GreenLangRecord,
    GreenLangAuditRecord,
    GreenLangRequest,
    GreenLangResponse,
    GreenLangConfig,
    GreenLangResult,
)

# -- Mixins -----------------------------------------------------------------
from greenlang.schemas.base import (
    TimestampMixin,
    TenantMixin,
    ProvenanceMixin,
    AuditMixin,
    MetadataMixin,
)

# -- Utility functions ------------------------------------------------------
from greenlang.schemas.base import (
    utcnow,
    new_uuid,
    prefixed_uuid,
    compute_provenance_hash,
)

__all__ = [
    # Base classes
    "GreenLangBase",
    "GreenLangRecord",
    "GreenLangAuditRecord",
    "GreenLangRequest",
    "GreenLangResponse",
    "GreenLangConfig",
    "GreenLangResult",
    # Mixins
    "TimestampMixin",
    "TenantMixin",
    "ProvenanceMixin",
    "AuditMixin",
    "MetadataMixin",
    # Utilities
    "utcnow",
    "new_uuid",
    "prefixed_uuid",
    "compute_provenance_hash",
]
