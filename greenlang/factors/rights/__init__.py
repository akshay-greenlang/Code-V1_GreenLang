# -*- coding: utf-8 -*-
"""GreenLang Factors — source-rights enforcement (Phase 1).

Public surface:

    from greenlang.factors.rights import (
        SourceRightsService,
        EntitlementRecord,
        EntitlementStore,
        RightsDenied,
        IngestionBlocked,
        check_ingestion_allowed,
        check_factor_read_allowed,
        check_pack_download_allowed,
        audit_licensed_access,
    )

Phase 1 wires this module into the alpha API + ingestion + audit
paths. See `docs/factors/source-rights/SOURCE_RIGHTS_MATRIX.md`.
"""
from __future__ import annotations

from .errors import (
    IngestionBlocked,
    LicenceMismatch,
    RightsDenied,
    SourceRightsError,
)
from .audit import (
    AuditDecision,
    AuditEvent,
    audit_licensed_access,
    get_audit_log,
    set_audit_sink,
)
from .entitlements import (
    EntitlementRecord,
    EntitlementStatus,
    EntitlementStore,
    EntitlementType,
    load_alpha_entitlements,
)
from .service import (
    Decision,
    SourceRightsService,
    check_factor_read_allowed,
    check_ingestion_allowed,
    check_pack_download_allowed,
    default_service,
)

__all__ = [
    "AuditDecision",
    "AuditEvent",
    "audit_licensed_access",
    "check_factor_read_allowed",
    "check_ingestion_allowed",
    "check_pack_download_allowed",
    "Decision",
    "default_service",
    "EntitlementRecord",
    "EntitlementStatus",
    "EntitlementStore",
    "EntitlementType",
    "get_audit_log",
    "IngestionBlocked",
    "LicenceMismatch",
    "load_alpha_entitlements",
    "RightsDenied",
    "set_audit_sink",
    "SourceRightsError",
    "SourceRightsService",
]
