# -*- coding: utf-8 -*-
"""
Auditor Portal - SEC-009 Phase 5

Secure portal for external auditors to access SOC 2 audit materials, submit
information requests, and review evidence. Provides comprehensive access
management, request handling, and activity logging for audit transparency.

Components:
    - AuditorAccessManager: Provision and manage auditor access credentials
    - AuditorRequestHandler: Handle information requests with SLA tracking
    - AuditorActivityLogger: Log all auditor actions for security audit

Example:
    >>> from greenlang.infrastructure.soc2_preparation.auditor_portal import (
    ...     AuditorAccessManager,
    ...     AuditorRequestHandler,
    ... )
    >>> access_manager = AuditorAccessManager()
    >>> await access_manager.provision_access(
    ...     auditor_id=uuid.uuid4(),
    ...     firm="Big Four Audit Firm",
    ...     permissions=["read:evidence", "read:reports"],
    ... )
"""

from greenlang.infrastructure.soc2_preparation.auditor_portal.access_manager import (
    AuditorAccessManager,
    AuditorCredentials,
    AuditorSession,
    Permission,
)
from greenlang.infrastructure.soc2_preparation.auditor_portal.request_handler import (
    AuditorRequestHandler,
    AuditorRequest,
    RequestCreate,
    RequestPriority,
    RequestStatus,
)
from greenlang.infrastructure.soc2_preparation.auditor_portal.activity_logger import (
    AuditorActivityLogger,
    ActivityLog,
    ActivityType,
    Anomaly,
    AnomalyType,
)

__all__ = [
    # Access Manager
    "AuditorAccessManager",
    "AuditorCredentials",
    "AuditorSession",
    "Permission",
    # Request Handler
    "AuditorRequestHandler",
    "AuditorRequest",
    "RequestCreate",
    "RequestPriority",
    "RequestStatus",
    # Activity Logger
    "AuditorActivityLogger",
    "ActivityLog",
    "ActivityType",
    "Anomaly",
    "AnomalyType",
]
