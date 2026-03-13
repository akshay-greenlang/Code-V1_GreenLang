# -*- coding: utf-8 -*-
"""
Audit Recorder Engine - AGENT-EUDR-036: EU Information System Interface

Engine 7: Records Article 31 audit trail events for all EU Information
System interface operations. Maintains complete, tamper-evident audit
records with SHA-256 provenance hashes for regulatory compliance.

Responsibilities:
    - Record all DDS lifecycle events (creation, validation, submission, etc.)
    - Record operator registration and update events
    - Record API call and error events
    - Record status check and status change events
    - Calculate retention dates per Article 31 (minimum 5 years)
    - Generate audit reports for competent authority requests
    - Maintain hash chain integrity across audit records
    - Support audit record querying by entity, event type, date range

EUDR Article 31 Requirements:
    - Operators shall keep DDS information for 5 years
    - Information available to competent authorities on request
    - Records shall demonstrate due diligence was exercised
    - Records must include supporting documentation references

Zero-Hallucination Guarantees:
    - All timestamps are UTC with second precision
    - Retention dates calculated from deterministic date arithmetic
    - SHA-256 hashes computed from canonical JSON
    - No LLM involvement in audit recording

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-036 (GL-EUDR-EUIS-036)
Regulation: EU 2023/1115 (EUDR) Article 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import EUInformationSystemInterfaceConfig, get_config
from .models import AuditEventType, AuditRecord
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class AuditRecorder:
    """Records Article 31 audit trail events for EU IS operations.

    Maintains a complete, tamper-evident audit trail with SHA-256
    provenance hashes for all EU Information System interface
    operations per EUDR Article 31 record-keeping requirements.

    Attributes:
        _config: Agent configuration instance.
        _provenance: Provenance tracker for audit trail.
        _records: In-memory audit record storage.
        _record_count: Total records created.

    Example:
        >>> recorder = AuditRecorder()
        >>> record = await recorder.record_event(
        ...     event_type="dds_submitted",
        ...     entity_type="dds",
        ...     entity_id="dds-abc123",
        ...     actor="system",
        ...     action="submit",
        ...     details={"commodity": "cocoa"},
        ... )
        >>> assert record.retention_until is not None
    """

    def __init__(
        self,
        config: Optional[EUInformationSystemInterfaceConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize AuditRecorder.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._records: Dict[str, AuditRecord] = {}
        self._records_by_entity: Dict[str, List[str]] = {}
        self._record_count = 0

        logger.info(
            "AuditRecorder initialized: retention=%d years, "
            "detail_level=%s, include_request=%s, include_response=%s",
            self._config.audit_retention_years,
            self._config.audit_detail_level,
            self._config.audit_include_request_body,
            self._config.audit_include_response_body,
        )

    async def record_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        actor: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        request_summary: Optional[str] = None,
        response_summary: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditRecord:
        """Record a single audit event.

        Creates an audit record with provenance hash, retention date,
        and full event context for Article 31 compliance.

        Args:
            event_type: Type of audit event (see AuditEventType).
            entity_type: Entity type (dds/operator/submission/package).
            entity_id: Entity identifier.
            actor: User or system performing the action.
            action: Action performed.
            details: Optional event details and context.
            request_summary: Optional API request summary.
            response_summary: Optional API response summary.
            ip_address: Optional client IP address.
            user_agent: Optional client user agent.

        Returns:
            AuditRecord with provenance hash and retention date.
        """
        audit_id = f"aud-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Calculate retention date (Article 31: minimum 5 years)
        retention_years = max(
            self._config.audit_retention_years, 5
        )
        retention_until = now + timedelta(days=retention_years * 365)

        # Validate event type
        try:
            evt_type = AuditEventType(event_type)
        except ValueError:
            evt_type = AuditEventType.API_CALL_MADE
            logger.warning(
                "Unknown event type '%s', defaulting to API_CALL_MADE",
                event_type,
            )

        # Filter details based on detail level
        filtered_details = self._filter_details(details or {})

        # Handle request/response inclusion based on config
        req_summary = None
        resp_summary = None
        if self._config.audit_include_request_body:
            req_summary = request_summary
        if self._config.audit_include_response_body:
            resp_summary = response_summary

        record = AuditRecord(
            audit_id=audit_id,
            event_type=evt_type,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=actor,
            action=action,
            details=filtered_details,
            request_summary=req_summary,
            response_summary=resp_summary,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=now,
            retention_until=retention_until,
        )

        # Compute provenance hash
        record.provenance_hash = self._provenance.compute_hash({
            "audit_id": audit_id,
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "actor": actor,
            "action": action,
            "timestamp": now.isoformat(),
        })

        # Store record
        self._records[audit_id] = record
        self._record_count += 1

        # Index by entity
        entity_key = f"{entity_type}:{entity_id}"
        if entity_key not in self._records_by_entity:
            self._records_by_entity[entity_key] = []
        self._records_by_entity[entity_key].append(audit_id)

        # Record provenance entry
        self._provenance.create_entry(
            step="record_audit",
            source=f"{entity_type}:{entity_id}",
            input_hash=self._provenance.compute_hash(
                {"entity_id": entity_id, "action": action}
            ),
            output_hash=record.provenance_hash,
        )

        logger.debug(
            "Audit record %s: %s %s:%s by %s (retention until %s)",
            audit_id, action, entity_type, entity_id,
            actor, retention_until.isoformat(),
        )

        return record

    async def record_dds_event(
        self,
        dds_id: str,
        event_type: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Convenience method to record a DDS-related audit event.

        Args:
            dds_id: DDS identifier.
            event_type: DDS event type.
            actor: Actor performing the action.
            details: Optional event details.

        Returns:
            AuditRecord for the DDS event.
        """
        return await self.record_event(
            event_type=event_type,
            entity_type="dds",
            entity_id=dds_id,
            actor=actor,
            action=event_type,
            details=details,
        )

    async def record_api_call_event(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
    ) -> AuditRecord:
        """Record an API call audit event.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            status_code: Response status code.
            duration_ms: Call duration in milliseconds.
            success: Whether call succeeded.
            error_message: Optional error message.

        Returns:
            AuditRecord for the API call.
        """
        event_type = (
            AuditEventType.API_CALL_MADE.value
            if success
            else AuditEventType.API_CALL_FAILED.value
        )

        return await self.record_event(
            event_type=event_type,
            entity_type="api_call",
            entity_id=f"{method}:{endpoint}",
            actor="AGENT-EUDR-036",
            action=f"{method} {endpoint}",
            details={
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "error": error_message,
            },
        )

    async def get_records_for_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[AuditRecord]:
        """Retrieve all audit records for a specific entity.

        Args:
            entity_type: Entity type (dds/operator/submission).
            entity_id: Entity identifier.

        Returns:
            Chronological list of audit records.
        """
        entity_key = f"{entity_type}:{entity_id}"
        audit_ids = self._records_by_entity.get(entity_key, [])

        records = []
        for audit_id in audit_ids:
            record = self._records.get(audit_id)
            if record is not None:
                records.append(record)

        return sorted(records, key=lambda r: r.timestamp)

    async def get_records_by_event_type(
        self,
        event_type: str,
        limit: int = 100,
    ) -> List[AuditRecord]:
        """Retrieve audit records filtered by event type.

        Args:
            event_type: Event type to filter by.
            limit: Maximum number of records to return.

        Returns:
            List of matching audit records.
        """
        matching = [
            r for r in self._records.values()
            if r.event_type.value == event_type
        ]

        # Sort by timestamp descending (most recent first)
        matching.sort(key=lambda r: r.timestamp, reverse=True)

        return matching[:limit]

    async def get_records_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        entity_type: Optional[str] = None,
    ) -> List[AuditRecord]:
        """Retrieve audit records within a date range.

        Args:
            start_date: Range start (inclusive).
            end_date: Range end (inclusive).
            entity_type: Optional entity type filter.

        Returns:
            List of matching audit records.
        """
        # Ensure timezone awareness
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        matching = [
            r for r in self._records.values()
            if start_date <= r.timestamp <= end_date
            and (entity_type is None or r.entity_type == entity_type)
        ]

        return sorted(matching, key=lambda r: r.timestamp)

    async def generate_audit_report(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Dict[str, Any]:
        """Generate an audit report for competent authority requests.

        Per EUDR Article 31, operators must make records available
        to competent authorities on request. This method generates
        a comprehensive report for a specific entity.

        Args:
            entity_type: Entity type.
            entity_id: Entity identifier.

        Returns:
            Audit report dictionary.
        """
        records = await self.get_records_for_entity(entity_type, entity_id)

        report_id = f"rpt-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Summarize events
        event_summary: Dict[str, int] = {}
        for record in records:
            evt = record.event_type.value
            event_summary[evt] = event_summary.get(evt, 0) + 1

        report = {
            "report_id": report_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "total_events": len(records),
            "event_summary": event_summary,
            "first_event": (
                records[0].timestamp.isoformat() if records else None
            ),
            "last_event": (
                records[-1].timestamp.isoformat() if records else None
            ),
            "records": [
                {
                    "audit_id": r.audit_id,
                    "event_type": r.event_type.value,
                    "action": r.action,
                    "actor": r.actor,
                    "timestamp": r.timestamp.isoformat(),
                    "details": r.details,
                    "provenance_hash": r.provenance_hash,
                }
                for r in records
            ],
            "generated_at": now.isoformat(),
            "regulation_reference": "EU 2023/1115 Article 31",
            "retention_years": self._config.audit_retention_years,
            "provenance_hash": self._provenance.compute_hash({
                "report_id": report_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "event_count": len(records),
                "generated_at": now.isoformat(),
            }),
        }

        logger.info(
            "Audit report %s generated for %s:%s (%d events)",
            report_id, entity_type, entity_id, len(records),
        )

        return report

    async def purge_expired_records(self) -> Dict[str, Any]:
        """Purge audit records that have exceeded their retention period.

        Only removes records where retention_until is in the past.
        This ensures Article 31 compliance while managing storage.

        Returns:
            Purge result with count of removed records.
        """
        now = datetime.now(timezone.utc)
        expired_ids: List[str] = []

        for audit_id, record in self._records.items():
            if record.retention_until and record.retention_until < now:
                expired_ids.append(audit_id)

        for audit_id in expired_ids:
            record = self._records.pop(audit_id, None)
            if record:
                entity_key = f"{record.entity_type}:{record.entity_id}"
                if entity_key in self._records_by_entity:
                    ids = self._records_by_entity[entity_key]
                    if audit_id in ids:
                        ids.remove(audit_id)

        result = {
            "purged_count": len(expired_ids),
            "remaining_count": len(self._records),
            "purged_at": now.isoformat(),
        }

        if expired_ids:
            logger.info(
                "Purged %d expired audit records", len(expired_ids)
            )

        return result

    def _filter_details(
        self,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Filter detail level based on configuration.

        Args:
            details: Raw details dictionary.

        Returns:
            Filtered details based on configured detail level.
        """
        level = self._config.audit_detail_level

        if level == "full":
            return details
        elif level == "summary":
            # Keep only top-level string/number values
            return {
                k: v for k, v in details.items()
                if isinstance(v, (str, int, float, bool))
            }
        elif level == "minimal":
            # Keep only key identifiers
            return {
                k: v for k, v in details.items()
                if k.endswith("_id") or k == "status"
            }

        return details

    @property
    def record_count(self) -> int:
        """Return total audit records stored."""
        return len(self._records)

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and record statistics.
        """
        return {
            "engine": "AuditRecorder",
            "status": "available",
            "record_count": len(self._records),
            "total_created": self._record_count,
            "entity_count": len(self._records_by_entity),
            "config": {
                "retention_years": self._config.audit_retention_years,
                "detail_level": self._config.audit_detail_level,
            },
        }
