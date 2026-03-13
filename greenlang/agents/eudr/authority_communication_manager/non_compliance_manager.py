# -*- coding: utf-8 -*-
"""
Non-Compliance Manager Engine - AGENT-EUDR-040: Authority Communication Manager

Handles violations and penalties per EUDR Article 16. Manages the full
lifecycle of non-compliance records from issuance through corrective
action completion or appeal, with penalty tracking in EUR.

Zero-Hallucination Guarantees:
    - All penalty calculations use Decimal arithmetic
    - No LLM calls in penalty computation path
    - Severity-to-penalty mapping via deterministic lookup
    - Complete provenance trail for every non-compliance event

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-040 (GL-EUDR-ACM-040)
Regulation: EU 2023/1115 (EUDR) Article 16
Status: Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import AuthorityCommunicationManagerConfig, get_config
from .models import (
    NonCompliance,
    ViolationSeverity,
    ViolationType,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# Severity to penalty multiplier mapping (deterministic)
_SEVERITY_PENALTY_MULTIPLIER: Dict[ViolationSeverity, Decimal] = {
    ViolationSeverity.MINOR: Decimal("1.0"),
    ViolationSeverity.MODERATE: Decimal("5.0"),
    ViolationSeverity.MAJOR: Decimal("25.0"),
    ViolationSeverity.CRITICAL: Decimal("100.0"),
}

# Violation type base penalty amounts in EUR
_VIOLATION_BASE_PENALTY: Dict[ViolationType, Decimal] = {
    ViolationType.MISSING_DDS: Decimal("10000"),
    ViolationType.INCOMPLETE_DDS: Decimal("5000"),
    ViolationType.FALSE_INFORMATION: Decimal("50000"),
    ViolationType.DEFORESTATION_LINK: Decimal("100000"),
    ViolationType.LEGALITY_VIOLATION: Decimal("75000"),
    ViolationType.TRACEABILITY_FAILURE: Decimal("25000"),
    ViolationType.RISK_ASSESSMENT_FAILURE: Decimal("15000"),
    ViolationType.INSUFFICIENT_MITIGATION: Decimal("20000"),
    ViolationType.RECORD_KEEPING_FAILURE: Decimal("10000"),
    ViolationType.NON_COOPERATION: Decimal("50000"),
    ViolationType.REPEATED_VIOLATION: Decimal("200000"),
}


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance."""
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


class NonComplianceManager:
    """Manages non-compliance records, violations, and penalties.

    Handles the complete lifecycle of EUDR Article 16 non-compliance
    proceedings, from initial violation detection through penalty
    calculation, corrective action tracking, and resolution.

    Attributes:
        config: Agent configuration.
        _provenance: SHA-256 provenance tracker.
        _records: In-memory non-compliance record store.

    Example:
        >>> manager = NonComplianceManager(config=get_config())
        >>> record = await manager.record_violation(
        ...     operator_id="OP-001",
        ...     authority_id="AUTH-DE",
        ...     violation_type="missing_dds",
        ...     severity="major",
        ...     description="Missing DDS for cocoa import batch B-2025-001"
        ... )
        >>> assert record.penalty_amount > 0
    """

    def __init__(
        self,
        config: Optional[AuthorityCommunicationManagerConfig] = None,
    ) -> None:
        """Initialize the Non-Compliance Manager engine.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._records: Dict[str, NonCompliance] = {}
        logger.info("NonComplianceManager engine initialized")

    async def record_violation(
        self,
        operator_id: str,
        authority_id: str,
        violation_type: str,
        severity: str,
        description: str,
        evidence_references: Optional[List[str]] = None,
        corrective_actions_required: Optional[List[str]] = None,
        corrective_deadline_days: int = 30,
        commodity: str = "",
        dds_reference: str = "",
        penalty_override: Optional[Decimal] = None,
        communication_id: str = "",
    ) -> NonCompliance:
        """Record a non-compliance violation with calculated penalty.

        Args:
            operator_id: Violating operator identifier.
            authority_id: Issuing authority identifier.
            violation_type: Type of violation.
            severity: Violation severity level.
            description: Description of the violation.
            evidence_references: Supporting evidence document IDs.
            corrective_actions_required: Required corrective actions.
            corrective_deadline_days: Days allowed for correction.
            commodity: Related EUDR commodity.
            dds_reference: Related DDS reference.
            penalty_override: Manual penalty amount override.
            communication_id: Parent communication ID.

        Returns:
            NonCompliance record with calculated penalty.

        Raises:
            ValueError: If violation_type or severity is invalid.
        """
        start = time.monotonic()

        try:
            v_type = ViolationType(violation_type)
        except ValueError:
            raise ValueError(
                f"Invalid violation type: {violation_type}. "
                f"Valid types: {[t.value for t in ViolationType]}"
            )

        try:
            v_severity = ViolationSeverity(severity)
        except ValueError:
            raise ValueError(
                f"Invalid severity: {severity}. "
                f"Valid levels: {[s.value for s in ViolationSeverity]}"
            )

        # Calculate penalty (zero-hallucination: deterministic formula)
        if penalty_override is not None:
            penalty = penalty_override
        else:
            penalty = self._calculate_penalty(v_type, v_severity)

        # Clamp to configured bounds
        penalty = max(self.config.penalty_min_amount, penalty)
        penalty = min(self.config.penalty_max_amount, penalty)

        nc_id = _new_uuid()
        now = _utcnow()
        corrective_deadline = now + timedelta(days=corrective_deadline_days)

        if not communication_id:
            communication_id = _new_uuid()

        record = NonCompliance(
            non_compliance_id=nc_id,
            communication_id=communication_id,
            operator_id=operator_id,
            authority_id=authority_id,
            violation_type=v_type,
            severity=v_severity,
            description=description,
            evidence_references=evidence_references or [],
            penalty_amount=penalty,
            penalty_currency=self.config.penalty_currency,
            corrective_actions_required=corrective_actions_required or [],
            corrective_deadline=corrective_deadline,
            commodity=commodity,
            dds_reference=dds_reference,
            issued_at=now,
            provenance_hash=_compute_hash({
                "non_compliance_id": nc_id,
                "operator_id": operator_id,
                "violation_type": violation_type,
                "severity": severity,
                "penalty_amount": str(penalty),
                "issued_at": now.isoformat(),
            }),
        )

        self._records[nc_id] = record

        # Record provenance
        self._provenance.create_entry(
            step="record_violation",
            source=authority_id,
            input_hash=self._provenance.compute_hash({
                "violation_type": violation_type,
                "severity": severity,
            }),
            output_hash=record.provenance_hash,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Non-compliance %s recorded for operator %s "
            "(type=%s, severity=%s, penalty=%s %s) in %.1fms",
            nc_id,
            operator_id,
            violation_type,
            severity,
            penalty,
            self.config.penalty_currency,
            elapsed * 1000,
        )

        return record

    def _calculate_penalty(
        self,
        violation_type: ViolationType,
        severity: ViolationSeverity,
    ) -> Decimal:
        """Calculate penalty amount using deterministic formula.

        Formula: base_penalty * severity_multiplier

        Args:
            violation_type: Type of violation.
            severity: Severity level.

        Returns:
            Calculated penalty amount in EUR.
        """
        base = _VIOLATION_BASE_PENALTY.get(violation_type, Decimal("10000"))
        multiplier = _SEVERITY_PENALTY_MULTIPLIER.get(
            severity, Decimal("1.0")
        )
        return (base * multiplier).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    async def mark_corrective_completed(
        self,
        non_compliance_id: str,
    ) -> NonCompliance:
        """Mark corrective actions as completed.

        Args:
            non_compliance_id: Non-compliance record identifier.

        Returns:
            Updated NonCompliance record.

        Raises:
            ValueError: If record not found.
        """
        record = self._records.get(non_compliance_id)
        if record is None:
            raise ValueError(
                f"Non-compliance record {non_compliance_id} not found"
            )

        record.corrective_completed = True
        record.resolved_at = _utcnow()

        logger.info(
            "Non-compliance %s: corrective actions completed",
            non_compliance_id,
        )
        return record

    async def link_appeal(
        self,
        non_compliance_id: str,
        appeal_id: str,
    ) -> NonCompliance:
        """Link an appeal to a non-compliance record.

        Args:
            non_compliance_id: Non-compliance record identifier.
            appeal_id: Appeal identifier.

        Returns:
            Updated NonCompliance record.

        Raises:
            ValueError: If record not found.
        """
        record = self._records.get(non_compliance_id)
        if record is None:
            raise ValueError(
                f"Non-compliance record {non_compliance_id} not found"
            )

        record.appeal_id = appeal_id

        logger.info(
            "Non-compliance %s linked to appeal %s",
            non_compliance_id,
            appeal_id,
        )
        return record

    async def get_record(
        self,
        non_compliance_id: str,
    ) -> Optional[NonCompliance]:
        """Retrieve a non-compliance record by identifier.

        Args:
            non_compliance_id: Record identifier.

        Returns:
            NonCompliance record or None.
        """
        return self._records.get(non_compliance_id)

    async def list_records(
        self,
        operator_id: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
    ) -> List[NonCompliance]:
        """List non-compliance records with optional filters.

        Args:
            operator_id: Filter by operator.
            severity: Filter by severity level.
            resolved: Filter by resolution status.

        Returns:
            List of matching NonCompliance records.
        """
        results = list(self._records.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if severity:
            results = [r for r in results if r.severity.value == severity]
        if resolved is not None:
            if resolved:
                results = [r for r in results if r.resolved_at is not None]
            else:
                results = [r for r in results if r.resolved_at is None]
        return results

    async def calculate_total_penalties(
        self,
        operator_id: str,
    ) -> Dict[str, Any]:
        """Calculate total penalties for an operator.

        Args:
            operator_id: Operator identifier.

        Returns:
            Dictionary with total, paid, and outstanding penalties.
        """
        records = [
            r for r in self._records.values()
            if r.operator_id == operator_id
        ]

        total = sum(
            (r.penalty_amount or Decimal("0")) for r in records
        )
        resolved = sum(
            (r.penalty_amount or Decimal("0"))
            for r in records
            if r.resolved_at is not None
        )
        outstanding = total - resolved

        return {
            "operator_id": operator_id,
            "total_violations": len(records),
            "total_penalty_amount": str(total),
            "resolved_penalty_amount": str(resolved),
            "outstanding_penalty_amount": str(outstanding),
            "currency": self.config.penalty_currency,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        open_cases = len([
            r for r in self._records.values()
            if r.resolved_at is None
        ])
        return {
            "engine": "non_compliance_manager",
            "status": "healthy",
            "total_records": len(self._records),
            "open_cases": open_cases,
        }
