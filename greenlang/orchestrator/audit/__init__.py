# -*- coding: utf-8 -*-
"""
Orchestrator Audit Trail
=========================

Provides hash-chained audit events for tamper-evidence.

Components:
    - EventStore: Append-only event storage
    - RunEvent: Individual audit event
    - AuditPackage: Exportable audit bundle

Author: GreenLang Team
"""

try:
    from greenlang.orchestrator.audit.event_store import (
        EventStore,
        RunEvent,
        EventType,
        AuditPackage,
    )
    AUDIT_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Audit event store not available: {e}")
    EventStore = None
    RunEvent = None
    EventType = None
    AuditPackage = None
    AUDIT_AVAILABLE = False

__all__ = ["EventStore", "RunEvent", "EventType", "AuditPackage", "AUDIT_AVAILABLE"]
