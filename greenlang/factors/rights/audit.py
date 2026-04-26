# -*- coding: utf-8 -*-
"""Phase 1 audit logging for source-rights decisions.

Every read of a non-`community_open` source emits one audit event.
Reads of `community_open` sources do NOT emit audit events (volume
would dominate the log; provenance is captured by the existing
metering layer).

In v0.1 alpha the sink is an in-memory list (also drained to the
module logger at INFO). v0.5+ swaps the sink for the centralised
audit pipeline (SEC-005).
"""
from __future__ import annotations

import enum
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

logger = logging.getLogger("greenlang.factors.rights.audit")


class AuditDecision(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"
    METADATA_ONLY = "metadata_only"


@dataclass(frozen=True)
class AuditEvent:
    tenant_id: Optional[str]
    api_key_id: Optional[str]
    source_urn: Optional[str]
    factor_urn: Optional[str]
    pack_urn: Optional[str]
    licence_class: Optional[str]
    decision: AuditDecision
    reason: Optional[str]
    request_id: Optional[str]
    action: str  # "read", "list", "ingest", "pack_download"
    occurred_at: str  # ISO-8601 UTC


# In-memory sink (v0.1 alpha).
_AUDIT_LOG: List[AuditEvent] = []
_AUDIT_LOCK = threading.Lock()


# Pluggable sink. v0.5+ replaces this with the SEC-005 audit pipeline.
_SINK: Optional[Callable[[AuditEvent], None]] = None


def set_audit_sink(sink: Optional[Callable[[AuditEvent], None]]) -> None:
    """Register an external audit sink (e.g. SEC-005 pipeline).

    Pass ``None`` to revert to the default in-memory sink.
    """
    global _SINK
    _SINK = sink


def get_audit_log(clear: bool = False) -> List[AuditEvent]:
    """Snapshot of the in-memory audit log (testing helper).

    For production sinks (`set_audit_sink`), this returns the local
    log only.
    """
    with _AUDIT_LOCK:
        snap = list(_AUDIT_LOG)
        if clear:
            _AUDIT_LOG.clear()
    return snap


def audit_licensed_access(
    *,
    tenant_id: Optional[str],
    source_urn: Optional[str],
    factor_urn: Optional[str] = None,
    pack_urn: Optional[str] = None,
    licence_class: Optional[str] = None,
    decision: AuditDecision = AuditDecision.ALLOW,
    reason: Optional[str] = None,
    request_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    action: str = "read",
) -> AuditEvent:
    """Emit one audit event for a licensed-source access.

    Always emits; callers decide whether to call this (community_open
    reads typically skip it). The function returns the emitted event
    so tests can assert on it directly.
    """
    event = AuditEvent(
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        source_urn=source_urn,
        factor_urn=factor_urn,
        pack_urn=pack_urn,
        licence_class=licence_class,
        decision=decision,
        reason=reason,
        request_id=request_id,
        action=action,
        occurred_at=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    )
    with _AUDIT_LOCK:
        _AUDIT_LOG.append(event)
    if _SINK is not None:
        try:
            _SINK(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit sink raised: %s (event=%r)", exc, event)
    logger.info(
        "rights_audit tenant=%s source=%s factor=%s pack=%s decision=%s reason=%s",
        tenant_id, source_urn, factor_urn, pack_urn, decision.value, reason,
    )
    return event
