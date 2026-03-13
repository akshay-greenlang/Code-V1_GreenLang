# -*- coding: utf-8 -*-
"""
Overdraft Detector - AGENT-EUDR-011 Engine 4

Real-time overdraft detection for mass balance ledgers:
- Balance check on every output entry
- Configurable modes: zero_tolerance, percentage (5% ISCC), absolute
- Severity classification: warning, violation, critical
- Alert generation with batch_ids, quantities, recommended actions
- Overdraft history per facility per commodity
- Resolution tracking: matching input within configurable timeframe
- Auto-reject on critical overdraft
- Trend analysis for recurring overdraft patterns
- Pre-output balance forecast
- Exemption management with approval and expiry

Zero-Hallucination Guarantees:
    - All balance checks are deterministic Python Decimal arithmetic
    - Tolerance calculations use exact arithmetic (no floating-point drift)
    - Severity classification uses deterministic threshold comparison
    - SHA-256 provenance hashes on every detection and resolution
    - No ML/LLM used for any overdraft calculation or decision logic

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Mass balance chain of custody
    - EU 2023/1115 (EUDR) Article 10(2)(f): Mass balance verification
    - ISO 22095:2020 Section 6.3: Balance integrity requirements
    - ISCC 203: 5% overdraft tolerance for mass balance
    - RSPO SCC 2020: Zero overdraft tolerance

Performance Targets:
    - Single overdraft check: <2ms
    - Alert generation: <3ms
    - Forecast calculation: <5ms
    - History retrieval: <10ms

PRD Feature References:
    - PRD-AGENT-EUDR-011 Feature 4: Overdraft Detection and Enforcement
    - F4.1: Balance check on every output entry
    - F4.2: Configurable enforcement modes (zero_tolerance, percentage, absolute)
    - F4.3: Severity classification (warning, violation, critical)
    - F4.4: Alert generation with batch details and recommended actions
    - F4.5: Overdraft history per facility per commodity
    - F4.6: Resolution tracking within configurable timeframe
    - F4.7: Auto-reject on critical overdraft
    - F4.8: Trend analysis for recurring overdraft patterns
    - F4.9: Pre-output balance forecast
    - F4.10: Exemption management with approval and expiry

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011
Agent ID: GL-EUDR-MBC-011
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.mass_balance_calculator.config import get_config
from greenlang.agents.eudr.mass_balance_calculator.metrics import (
    observe_overdraft_check_duration,
    record_api_error,
    record_overdraft_critical,
    record_overdraft_detected,
)
from greenlang.agents.eudr.mass_balance_calculator.models import (
    OverdraftMode,
    OverdraftSeverity,
)
from greenlang.agents.eudr.mass_balance_calculator.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a new UUID4 string identifier.

    Returns:
        UUID4 string.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Severity thresholds for multi-level classification
# ---------------------------------------------------------------------------

#: Severity levels by relative magnitude of overdraft.
#: If overdraft exceeds balance by these factors, severity escalates.
_SEVERITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "warning_factor": {
        "description": "Overdraft within 1x tolerance",
        "min_factor": 0.0,
        "max_factor": 1.0,
    },
    "violation_factor": {
        "description": "Overdraft within 2x tolerance",
        "min_factor": 1.0,
        "max_factor": 2.0,
    },
    "critical_factor": {
        "description": "Overdraft exceeds 2x tolerance",
        "min_factor": 2.0,
        "max_factor": float("inf"),
    },
}


# ---------------------------------------------------------------------------
# OverdraftDetector
# ---------------------------------------------------------------------------


class OverdraftDetector:
    """Real-time overdraft detection engine for EUDR mass balance ledgers.

    Detects when output entries would cause the mass balance ledger to
    go into a negative balance (or exceed the configured tolerance
    threshold). Supports multiple enforcement modes, severity
    classification, alert generation, resolution tracking, exemption
    management, and trend analysis.

    All operations follow the zero-hallucination principle: balance
    checks and tolerance calculations use deterministic Python Decimal
    arithmetic. No ML/LLM is used for any overdraft decision logic.

    Enforcement Modes:
        - **zero_tolerance**: Any output exceeding the current balance
          is an immediate violation. Strictest mode.
        - **percentage**: Allows overdraft up to a configured percentage
          of total period inputs before triggering a violation.
        - **absolute**: Allows overdraft up to a configured absolute
          quantity (in kg) before triggering a violation.

    Severity Levels:
        - **warning**: Overdraft within tolerance. Logged, not blocked.
        - **violation**: Overdraft exceeds tolerance. Must be resolved
          within ``overdraft_resolution_hours``.
        - **critical**: Overdraft significantly exceeds tolerance or
          multiple unresolved violations. Blocks further outputs.

    Attributes:
        _config: Mass balance calculator configuration.
        _provenance: ProvenanceTracker for audit trail.
        _overdraft_events: In-memory event store keyed by event_id.
        _facility_events: Index of event_ids by facility_id.
        _active_alerts: Active alert store keyed by alert_id.
        _exemptions: Active exemptions keyed by exemption_id.
        _facility_exemptions: Index of exemption_ids by facility_id.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> from greenlang.agents.eudr.mass_balance_calculator.overdraft_detector import (
        ...     OverdraftDetector,
        ... )
        >>> detector = OverdraftDetector()
        >>> result = detector.check_overdraft(
        ...     ledger_id="ledger-001",
        ...     current_balance=Decimal("1000"),
        ...     proposed_output_qty=Decimal("1200"),
        ...     facility_id="facility-001",
        ...     commodity="cocoa",
        ... )
        >>> assert result["is_overdraft"] is True
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize OverdraftDetector with configuration and provenance tracker.

        Args:
            config: Optional MassBalanceCalculatorConfig override. If None,
                uses the singleton from get_config().
            provenance: Optional ProvenanceTracker override. If None,
                uses the singleton from get_provenance_tracker().
        """
        self._config = config or get_config()
        self._provenance = provenance or get_provenance_tracker()

        # -- In-memory storage -------------------------------------------------
        self._overdraft_events: Dict[str, Dict[str, Any]] = {}
        self._facility_events: Dict[str, List[str]] = {}
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        self._facility_alerts: Dict[str, List[str]] = {}
        self._exemptions: Dict[str, Dict[str, Any]] = {}
        self._facility_exemptions: Dict[str, List[str]] = {}

        # -- Thread safety -----------------------------------------------------
        self._lock = threading.RLock()

        logger.info(
            "OverdraftDetector initialized: module_version=%s, "
            "mode=%s, tolerance_pct=%.1f%%, tolerance_kg=%.1f, "
            "resolution_hours=%d, provenance_enabled=%s",
            _MODULE_VERSION,
            self._config.overdraft_mode,
            self._config.overdraft_tolerance_percent,
            self._config.overdraft_tolerance_kg,
            self._config.overdraft_resolution_hours,
            self._config.enable_provenance,
        )

    # ------------------------------------------------------------------
    # Public API: Overdraft Check
    # ------------------------------------------------------------------

    def check_overdraft(
        self,
        ledger_id: str,
        current_balance: Decimal,
        proposed_output_qty: Decimal,
        facility_id: str,
        commodity: str,
        total_period_inputs: Optional[Decimal] = None,
        batch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check whether a proposed output would cause an overdraft.

        This is the primary method called by the LedgerManager before
        recording an output entry. It determines if the proposed
        quantity exceeds the available balance (considering the
        configured tolerance mode).

        PRD Reference: F4.1 - Balance check on every output entry.

        Args:
            ledger_id: Ledger identifier being checked.
            current_balance: Current ledger balance in kg (Decimal).
            proposed_output_qty: Proposed output quantity in kg (Decimal).
            facility_id: Facility identifier.
            commodity: EUDR commodity type.
            total_period_inputs: Total inputs in the current period (for
                percentage mode). If None, defaults to current_balance.
            batch_id: Optional batch identifier for the output.
            metadata: Optional additional context.

        Returns:
            Dictionary containing:
                - check_id: Unique check identifier
                - is_overdraft: Whether the output would cause overdraft
                - severity: Overdraft severity (None if no overdraft)
                - current_balance: Current balance
                - proposed_output: Proposed output quantity
                - resulting_balance: Balance after proposed output
                - tolerance: Computed tolerance amount
                - overdraft_amount: Amount exceeding tolerance
                - auto_reject: Whether output should be auto-rejected
                - event_id: Overdraft event ID (if created)
                - alert_id: Alert ID (if generated)
                - exemption_applied: Whether an exemption was applied
                - provenance_hash: SHA-256 provenance hash

        Raises:
            ValueError: If required inputs are invalid.
        """
        start_time = time.monotonic()

        # Validate inputs
        self._validate_string("ledger_id", ledger_id)
        self._validate_string("facility_id", facility_id)
        self._validate_string("commodity", commodity)

        current_balance = Decimal(str(current_balance))
        proposed_output_qty = Decimal(str(proposed_output_qty))
        if proposed_output_qty <= 0:
            raise ValueError(
                f"proposed_output_qty must be > 0, got {proposed_output_qty}"
            )

        if total_period_inputs is not None:
            total_period_inputs = Decimal(str(total_period_inputs))
        else:
            total_period_inputs = current_balance

        # Calculate resulting balance
        resulting_balance = current_balance - proposed_output_qty

        # Calculate tolerance based on mode
        tolerance = self._calculate_tolerance(
            mode=self._config.overdraft_mode,
            tolerance_pct=Decimal(str(self._config.overdraft_tolerance_percent)),
            tolerance_kg=Decimal(str(self._config.overdraft_tolerance_kg)),
            total_inputs=total_period_inputs,
        )

        # Check for exemption
        exemption_applied = False
        exemption_id = None
        if resulting_balance < Decimal("0"):
            overdraft_amount = abs(resulting_balance)
            is_exempted, exempt_id = self._is_exempted(
                facility_id, commodity, overdraft_amount
            )
            if is_exempted:
                exemption_applied = True
                exemption_id = exempt_id

        # Determine if overdraft occurs
        is_overdraft = resulting_balance < -tolerance

        # If exempted, override overdraft to False
        if exemption_applied and is_overdraft:
            is_overdraft = False

        check_id = _generate_id()
        now = _utcnow()
        severity = None
        event_id = None
        alert_id = None
        auto_reject = False

        if is_overdraft:
            overdraft_amount = abs(resulting_balance) - tolerance
            if overdraft_amount < Decimal("0"):
                overdraft_amount = Decimal("0")

            # Classify severity
            severity = self.classify_severity(
                current_balance=current_balance,
                tolerance=tolerance,
                proposed_qty=proposed_output_qty,
                total_period_inputs=total_period_inputs,
            )

            # Record overdraft event
            event_data = self._record_overdraft_event(
                check_id=check_id,
                ledger_id=ledger_id,
                facility_id=facility_id,
                commodity=commodity,
                severity=severity,
                current_balance=current_balance,
                overdraft_amount=overdraft_amount,
                proposed_output_qty=proposed_output_qty,
                resulting_balance=resulting_balance,
                tolerance=tolerance,
                batch_id=batch_id,
                metadata=metadata,
            )
            event_id = event_data["event_id"]

            # Generate alert
            alert = self.generate_alert(event_data)
            alert_id = alert["alert_id"]

            # Auto-reject on critical
            if severity == OverdraftSeverity.CRITICAL.value:
                auto_reject = True

            # Record metrics
            record_overdraft_detected(severity)
            if severity == OverdraftSeverity.CRITICAL.value:
                record_overdraft_critical()
        else:
            overdraft_amount = Decimal("0")

        # Compute provenance hash
        provenance_hash = _compute_hash({
            "check_id": check_id,
            "ledger_id": ledger_id,
            "is_overdraft": is_overdraft,
            "severity": severity,
            "current_balance": str(current_balance),
            "proposed_output": str(proposed_output_qty),
            "resulting_balance": str(resulting_balance),
            "action": "check_overdraft",
        })

        # Record provenance
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="overdraft_event",
                action="detect" if is_overdraft else "validate",
                entity_id=check_id,
                data={
                    "is_overdraft": is_overdraft,
                    "severity": severity,
                    "overdraft_amount": str(overdraft_amount),
                },
                metadata={
                    "facility_id": facility_id,
                    "commodity": commodity,
                    "ledger_id": ledger_id,
                },
            )

        elapsed_s = time.monotonic() - start_time
        elapsed_ms = elapsed_s * 1000
        observe_overdraft_check_duration(elapsed_s)

        log_level = logging.WARNING if is_overdraft else logging.DEBUG
        logger.log(
            log_level,
            "Overdraft check: ledger=%s balance=%s output=%s "
            "resulting=%s overdraft=%s severity=%s auto_reject=%s "
            "elapsed=%.1fms",
            ledger_id[:12],
            str(current_balance),
            str(proposed_output_qty),
            str(resulting_balance),
            is_overdraft,
            severity,
            auto_reject,
            elapsed_ms,
        )

        return {
            "check_id": check_id,
            "ledger_id": ledger_id,
            "facility_id": facility_id,
            "commodity": commodity,
            "is_overdraft": is_overdraft,
            "severity": severity,
            "current_balance": str(current_balance),
            "proposed_output": str(proposed_output_qty),
            "resulting_balance": str(resulting_balance),
            "tolerance": str(tolerance),
            "tolerance_mode": self._config.overdraft_mode,
            "overdraft_amount": str(overdraft_amount),
            "auto_reject": auto_reject,
            "event_id": event_id,
            "alert_id": alert_id,
            "exemption_applied": exemption_applied,
            "exemption_id": exemption_id,
            "batch_id": batch_id,
            "provenance_hash": provenance_hash,
            "checked_at": now.isoformat(),
            "elapsed_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Public API: Severity Classification
    # ------------------------------------------------------------------

    def classify_severity(
        self,
        current_balance: Decimal,
        tolerance: Decimal,
        proposed_qty: Decimal,
        total_period_inputs: Decimal,
    ) -> str:
        """Classify the severity of an overdraft event.

        Determines severity based on the magnitude of the overdraft
        relative to the tolerance and total period inputs.

        PRD Reference: F4.3 - Severity classification.

        Args:
            current_balance: Current ledger balance.
            tolerance: Computed tolerance amount.
            proposed_qty: Proposed output quantity.
            total_period_inputs: Total inputs in the current period.

        Returns:
            Severity string: "warning", "violation", or "critical".
        """
        current_balance = Decimal(str(current_balance))
        tolerance = Decimal(str(tolerance))
        proposed_qty = Decimal(str(proposed_qty))
        total_period_inputs = Decimal(str(total_period_inputs))

        overdraft_raw = proposed_qty - current_balance
        if overdraft_raw <= Decimal("0"):
            return OverdraftSeverity.WARNING.value

        # Calculate overdraft as fraction of total inputs
        if total_period_inputs > 0:
            overdraft_fraction = overdraft_raw / total_period_inputs
        else:
            overdraft_fraction = Decimal("1")

        # Critical: overdraft > 10% of total inputs or > 2x tolerance
        if tolerance > 0:
            tolerance_multiple = overdraft_raw / tolerance
        else:
            tolerance_multiple = overdraft_raw  # Treat as very large

        if overdraft_fraction > Decimal("0.10") or tolerance_multiple > 2:
            return OverdraftSeverity.CRITICAL.value

        # Violation: overdraft > tolerance
        if overdraft_raw > tolerance:
            return OverdraftSeverity.VIOLATION.value

        # Warning: overdraft within tolerance
        return OverdraftSeverity.WARNING.value

    # ------------------------------------------------------------------
    # Public API: Alert Generation
    # ------------------------------------------------------------------

    def generate_alert(
        self,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate an alert for an overdraft event.

        Creates a structured alert with details about the overdraft,
        affected batches, recommended actions, and resolution deadline.

        PRD Reference: F4.4 - Alert generation.

        Args:
            event_data: Overdraft event dictionary (from _record_overdraft_event).

        Returns:
            Dictionary containing:
                - alert_id: Unique alert identifier
                - event_id: Associated overdraft event ID
                - severity: Alert severity
                - facility_id: Affected facility
                - commodity: Affected commodity
                - overdraft_amount: Amount of overdraft
                - resolution_deadline: Deadline for resolution
                - recommended_actions: List of recommended actions
                - created_at: Alert creation timestamp
        """
        alert_id = _generate_id()
        now = _utcnow()
        severity = event_data.get("severity", OverdraftSeverity.WARNING.value)
        facility_id = event_data.get("facility_id", "")
        commodity = event_data.get("commodity", "")

        resolution_deadline = now + timedelta(
            hours=self._config.overdraft_resolution_hours
        )

        # Build recommended actions based on severity
        actions = self._build_recommended_actions(
            severity, event_data
        )

        alert: Dict[str, Any] = {
            "alert_id": alert_id,
            "event_id": event_data.get("event_id", ""),
            "ledger_id": event_data.get("ledger_id", ""),
            "severity": severity,
            "facility_id": facility_id,
            "commodity": commodity,
            "current_balance": event_data.get("current_balance", "0"),
            "overdraft_amount": event_data.get("overdraft_amount", "0"),
            "proposed_output": event_data.get("proposed_output_qty", "0"),
            "batch_id": event_data.get("batch_id"),
            "resolution_deadline": resolution_deadline.isoformat(),
            "resolution_hours": self._config.overdraft_resolution_hours,
            "recommended_actions": actions,
            "acknowledged": False,
            "acknowledged_at": None,
            "acknowledged_by": None,
            "resolved": False,
            "resolved_at": None,
            "created_at": now.isoformat(),
        }

        # Store alert
        with self._lock:
            self._active_alerts[alert_id] = alert
            if facility_id not in self._facility_alerts:
                self._facility_alerts[facility_id] = []
            self._facility_alerts[facility_id].append(alert_id)

        logger.warning(
            "Overdraft alert generated: alert_id=%s severity=%s "
            "facility=%s commodity=%s deadline=%s",
            alert_id[:12],
            severity,
            facility_id,
            commodity,
            resolution_deadline.isoformat(),
        )

        return alert

    # ------------------------------------------------------------------
    # Public API: Get Active Alerts
    # ------------------------------------------------------------------

    def get_active_alerts(
        self,
        facility_id: str,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts for a facility.

        Args:
            facility_id: Facility identifier.
            severity: Optional severity filter.

        Returns:
            List of active alert dictionaries, sorted by creation
            time descending.
        """
        self._validate_string("facility_id", facility_id)

        results: List[Dict[str, Any]] = []
        with self._lock:
            alert_ids = self._facility_alerts.get(facility_id, [])
            for aid in alert_ids:
                alert = self._active_alerts.get(aid)
                if alert and not alert.get("resolved", False):
                    if severity and alert.get("severity") != severity:
                        continue
                    results.append(dict(alert))

        results.sort(
            key=lambda a: a.get("created_at", ""),
            reverse=True,
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Resolve Overdraft
    # ------------------------------------------------------------------

    def resolve_overdraft(
        self,
        event_id: str,
        resolution_entry_id: str,
        resolved_by: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark an overdraft event as resolved.

        Records the resolution in the audit trail and marks associated
        alerts as resolved.

        PRD Reference: F4.6 - Resolution tracking.

        Args:
            event_id: Overdraft event identifier to resolve.
            resolution_entry_id: Identifier of the ledger entry that
                resolved the overdraft (e.g., a matching input entry).
            resolved_by: Optional operator identifier.
            resolution_notes: Optional notes on how the overdraft was
                resolved.

        Returns:
            Dictionary containing:
                - event_id: Resolved event identifier
                - resolution_entry_id: Resolving entry identifier
                - resolved_at: UTC timestamp of resolution
                - operation_status: "resolved"

        Raises:
            ValueError: If event not found or already resolved.
        """
        start_time = time.monotonic()

        self._validate_string("event_id", event_id)
        self._validate_string("resolution_entry_id", resolution_entry_id)

        now = _utcnow()

        with self._lock:
            event = self._overdraft_events.get(event_id)
            if event is None:
                raise ValueError(f"Overdraft event not found: {event_id}")

            if event.get("resolved", False):
                raise ValueError(
                    f"Overdraft event already resolved: {event_id}"
                )

            # Resolve the event
            event["resolved"] = True
            event["resolved_at"] = now
            event["resolved_by"] = resolved_by
            event["resolution_entry_id"] = resolution_entry_id
            event["resolution_notes"] = resolution_notes
            event["updated_at"] = now

            # Resolve associated alerts
            for alert in self._active_alerts.values():
                if alert.get("event_id") == event_id:
                    alert["resolved"] = True
                    alert["resolved_at"] = now.isoformat()

        # Compute provenance hash
        provenance_hash = _compute_hash({
            "event_id": event_id,
            "resolution_entry_id": resolution_entry_id,
            "resolved_by": resolved_by,
            "action": "resolve",
        })

        # Record provenance
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="overdraft_event",
                action="update",
                entity_id=event_id,
                data={
                    "resolved": True,
                    "resolution_entry_id": resolution_entry_id,
                    "resolved_by": resolved_by,
                },
                metadata={
                    "facility_id": event.get("facility_id", ""),
                    "commodity": event.get("commodity", ""),
                },
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Overdraft resolved: event_id=%s resolution_entry=%s "
            "resolved_by=%s elapsed=%.1fms",
            event_id[:12],
            resolution_entry_id[:12],
            resolved_by,
            elapsed_ms,
        )

        return {
            "event_id": event_id,
            "resolution_entry_id": resolution_entry_id,
            "resolved_by": resolved_by,
            "resolution_notes": resolution_notes,
            "resolved_at": now.isoformat(),
            "provenance_hash": provenance_hash,
            "operation_status": "resolved",
            "elapsed_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Public API: Pre-Output Forecast
    # ------------------------------------------------------------------

    def forecast_output(
        self,
        ledger_id: str,
        current_balance: Decimal,
        proposed_qty: Decimal,
        facility_id: str,
        total_period_inputs: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Forecast the result of a proposed output before recording.

        Provides a preview of what would happen if the output were
        recorded, without actually creating any events or alerts.

        PRD Reference: F4.9 - Pre-output balance forecast.

        Args:
            ledger_id: Ledger identifier.
            current_balance: Current balance in kg.
            proposed_qty: Proposed output quantity in kg.
            facility_id: Facility identifier.
            total_period_inputs: Optional total period inputs for
                percentage-mode tolerance calculation.

        Returns:
            Dictionary containing:
                - would_overdraft: Whether the output would cause overdraft
                - resulting_balance: Projected balance after output
                - tolerance: Computed tolerance
                - available_for_output: Maximum output without overdraft
                - utilization_after: Balance utilization after output
                - severity_if_overdraft: Projected severity
                - recommendation: Action recommendation
        """
        current_balance = Decimal(str(current_balance))
        proposed_qty = Decimal(str(proposed_qty))
        if total_period_inputs is not None:
            total_period_inputs = Decimal(str(total_period_inputs))
        else:
            total_period_inputs = current_balance

        resulting_balance = current_balance - proposed_qty

        tolerance = self._calculate_tolerance(
            mode=self._config.overdraft_mode,
            tolerance_pct=Decimal(str(self._config.overdraft_tolerance_percent)),
            tolerance_kg=Decimal(str(self._config.overdraft_tolerance_kg)),
            total_inputs=total_period_inputs,
        )

        would_overdraft = resulting_balance < -tolerance
        available_for_output = current_balance + tolerance

        # Calculate utilization
        utilization_after = Decimal("0")
        if total_period_inputs > 0:
            total_outputs_projected = proposed_qty
            utilization_after = total_outputs_projected / total_period_inputs

        # Determine severity if overdraft would occur
        severity_if_overdraft = None
        if would_overdraft:
            severity_if_overdraft = self.classify_severity(
                current_balance=current_balance,
                tolerance=tolerance,
                proposed_qty=proposed_qty,
                total_period_inputs=total_period_inputs,
            )

        # Build recommendation
        if not would_overdraft:
            recommendation = "Output can proceed without overdraft."
        elif severity_if_overdraft == OverdraftSeverity.WARNING.value:
            recommendation = (
                "Output would cause a warning-level overdraft. "
                "Proceed with caution."
            )
        elif severity_if_overdraft == OverdraftSeverity.VIOLATION.value:
            recommendation = (
                f"Output would cause a violation. "
                f"Reduce output to {available_for_output} kg or add "
                f"matching input within "
                f"{self._config.overdraft_resolution_hours}h."
            )
        else:
            recommendation = (
                f"Output would cause a CRITICAL overdraft and will "
                f"be AUTO-REJECTED. Maximum available output: "
                f"{available_for_output} kg."
            )

        # Check for exemption
        exemption_available = False
        if would_overdraft:
            overdraft_amount = abs(resulting_balance) - tolerance
            is_exempt, _ = self._is_exempted(
                facility_id,
                "",  # commodity not needed for exemption check
                max(Decimal("0"), overdraft_amount),
            )
            exemption_available = is_exempt

        return {
            "ledger_id": ledger_id,
            "facility_id": facility_id,
            "current_balance": str(current_balance),
            "proposed_qty": str(proposed_qty),
            "resulting_balance": str(resulting_balance),
            "tolerance": str(tolerance),
            "tolerance_mode": self._config.overdraft_mode,
            "available_for_output": str(available_for_output),
            "would_overdraft": would_overdraft,
            "severity_if_overdraft": severity_if_overdraft,
            "utilization_after": str(
                round(float(utilization_after) * 100, 2)
            ) + "%",
            "exemption_available": exemption_available,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Public API: Exemption Management
    # ------------------------------------------------------------------

    def request_exemption(
        self,
        facility_id: str,
        commodity: str,
        amount: Decimal,
        reason: str,
        requested_by: str,
        expiry_date: datetime,
        approved_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request an overdraft exemption for a facility.

        Exemptions allow a temporary overdraft up to the specified
        amount without triggering alerts or auto-rejection.

        PRD Reference: F4.10 - Exemption management.

        Args:
            facility_id: Facility identifier.
            commodity: EUDR commodity.
            amount: Maximum overdraft amount allowed (kg).
            reason: Justification for the exemption.
            requested_by: Identifier of the requesting operator.
            expiry_date: Date when the exemption expires.
            approved_by: Optional identifier of the approving authority.
                If provided, the exemption is immediately active.

        Returns:
            Dictionary containing:
                - exemption_id: Unique exemption identifier
                - facility_id, commodity, amount
                - status: "approved" or "pending"
                - expiry_date: Expiry date
                - provenance_hash: SHA-256 provenance hash

        Raises:
            ValueError: If inputs are invalid.
        """
        start_time = time.monotonic()

        self._validate_string("facility_id", facility_id)
        self._validate_string("commodity", commodity)
        self._validate_string("reason", reason)
        self._validate_string("requested_by", requested_by)

        amount = Decimal(str(amount))
        if amount <= 0:
            raise ValueError(f"amount must be > 0, got {amount}")

        if expiry_date.tzinfo is None:
            expiry_date = expiry_date.replace(tzinfo=timezone.utc)

        now = _utcnow()
        if expiry_date <= now:
            raise ValueError("expiry_date must be in the future")

        exemption_id = _generate_id()
        is_approved = approved_by is not None and approved_by.strip() != ""

        exemption_data: Dict[str, Any] = {
            "exemption_id": exemption_id,
            "facility_id": facility_id,
            "commodity": commodity,
            "amount_kg": str(amount),
            "reason": reason.strip(),
            "requested_by": requested_by.strip(),
            "requested_at": now.isoformat(),
            "approved_by": approved_by.strip() if approved_by else None,
            "approved_at": now.isoformat() if is_approved else None,
            "status": "approved" if is_approved else "pending",
            "expiry_date": expiry_date.isoformat(),
            "utilized_amount": "0",
            "active": is_approved,
            "created_at": now.isoformat(),
        }

        # Compute provenance hash
        provenance_hash = _compute_hash({
            "exemption_id": exemption_id,
            "facility_id": facility_id,
            "commodity": commodity,
            "amount": str(amount),
            "action": "request_exemption",
        })
        exemption_data["provenance_hash"] = provenance_hash

        # Store exemption
        with self._lock:
            self._exemptions[exemption_id] = exemption_data
            if facility_id not in self._facility_exemptions:
                self._facility_exemptions[facility_id] = []
            self._facility_exemptions[facility_id].append(exemption_id)

        # Record provenance
        if self._config.enable_provenance:
            self._provenance.record(
                entity_type="overdraft_event",
                action="create",
                entity_id=exemption_id,
                data=exemption_data,
                metadata={
                    "facility_id": facility_id,
                    "commodity": commodity,
                    "type": "exemption",
                },
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Exemption %s: exemption_id=%s facility=%s commodity=%s "
            "amount=%s expiry=%s elapsed=%.1fms",
            "approved" if is_approved else "requested",
            exemption_id[:12],
            facility_id,
            commodity,
            str(amount),
            expiry_date.isoformat(),
            elapsed_ms,
        )

        return {
            **exemption_data,
            "operation_status": "approved" if is_approved else "pending",
            "elapsed_ms": round(elapsed_ms, 2),
        }

    # ------------------------------------------------------------------
    # Public API: Overdraft History
    # ------------------------------------------------------------------

    def get_overdraft_history(
        self,
        facility_id: str,
        commodity: Optional[str] = None,
        include_resolved: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get overdraft event history for a facility.

        PRD Reference: F4.5 - Overdraft history.

        Args:
            facility_id: Facility identifier.
            commodity: Optional commodity filter.
            include_resolved: Whether to include resolved events.
            limit: Maximum number of events to return.

        Returns:
            List of overdraft event dictionaries, most recent first.
        """
        self._validate_string("facility_id", facility_id)

        results: List[Dict[str, Any]] = []
        with self._lock:
            event_ids = self._facility_events.get(facility_id, [])
            for eid in event_ids:
                event = self._overdraft_events.get(eid)
                if event is None:
                    continue
                if commodity and event.get("commodity") != commodity:
                    continue
                if not include_resolved and event.get("resolved", False):
                    continue
                results.append(self._serialize_event(event))

        results.sort(
            key=lambda e: e.get("created_at", ""),
            reverse=True,
        )
        return results[:limit]

    # ------------------------------------------------------------------
    # Public API: Pattern Detection
    # ------------------------------------------------------------------

    def detect_patterns(
        self,
        facility_id: str,
        commodity: Optional[str] = None,
        lookback_days: int = 90,
    ) -> Dict[str, Any]:
        """Analyze overdraft history for recurring patterns.

        Examines frequency, severity distribution, resolution times,
        and trend direction for overdraft events at a facility.

        PRD Reference: F4.8 - Trend analysis for recurring patterns.

        Args:
            facility_id: Facility identifier.
            commodity: Optional commodity filter.
            lookback_days: Number of days to look back for analysis.

        Returns:
            Dictionary containing:
                - total_events: Total overdraft events in window
                - events_per_month: Average events per month
                - severity_distribution: Counts by severity
                - average_resolution_hours: Mean resolution time
                - unresolved_count: Number of unresolved events
                - trend_direction: "improving", "stable", or "worsening"
                - repeat_offender: Whether facility has recurring pattern
                - recommendations: List of recommendations
        """
        self._validate_string("facility_id", facility_id)

        cutoff = _utcnow() - timedelta(days=lookback_days)
        events = self.get_overdraft_history(
            facility_id=facility_id,
            commodity=commodity,
            include_resolved=True,
            limit=1000,
        )

        # Filter to lookback window
        window_events = [
            e for e in events
            if e.get("created_at", "") >= cutoff.isoformat()
        ]

        total_events = len(window_events)

        # Severity distribution
        severity_dist: Dict[str, int] = {
            OverdraftSeverity.WARNING.value: 0,
            OverdraftSeverity.VIOLATION.value: 0,
            OverdraftSeverity.CRITICAL.value: 0,
        }
        resolution_hours: List[float] = []
        unresolved_count = 0
        monthly_counts: Dict[str, int] = {}

        for e in window_events:
            sev = e.get("severity", OverdraftSeverity.WARNING.value)
            severity_dist[sev] = severity_dist.get(sev, 0) + 1

            if e.get("resolved", False):
                created = e.get("created_at", "")
                resolved = e.get("resolved_at", "")
                if created and resolved:
                    try:
                        c_dt = datetime.fromisoformat(created)
                        r_dt = datetime.fromisoformat(resolved)
                        hours = (r_dt - c_dt).total_seconds() / 3600
                        resolution_hours.append(hours)
                    except (ValueError, TypeError):
                        pass
            else:
                unresolved_count += 1

            # Monthly aggregation
            month_key = e.get("created_at", "")[:7]
            if month_key:
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1

        # Calculate metrics
        months_in_window = max(1, lookback_days / 30)
        events_per_month = total_events / months_in_window

        avg_resolution = (
            statistics.mean(resolution_hours) if resolution_hours else 0.0
        )

        # Determine trend direction
        trend = self._analyze_trend_direction(monthly_counts)

        # Determine repeat offender status
        repeat_offender = total_events >= 5 and events_per_month >= 2.0

        # Build recommendations
        recommendations = self._build_pattern_recommendations(
            total_events, severity_dist, avg_resolution,
            unresolved_count, trend, repeat_offender,
        )

        return {
            "facility_id": facility_id,
            "commodity_filter": commodity,
            "lookback_days": lookback_days,
            "total_events": total_events,
            "events_per_month": round(events_per_month, 2),
            "severity_distribution": severity_dist,
            "average_resolution_hours": round(avg_resolution, 2),
            "unresolved_count": unresolved_count,
            "trend_direction": trend,
            "repeat_offender": repeat_offender,
            "monthly_counts": monthly_counts,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Internal: Tolerance Calculation
    # ------------------------------------------------------------------

    def _calculate_tolerance(
        self,
        mode: str,
        tolerance_pct: Decimal,
        tolerance_kg: Decimal,
        total_inputs: Decimal,
    ) -> Decimal:
        """Calculate the overdraft tolerance based on enforcement mode.

        PRD Reference: F4.2 - Configurable enforcement modes.

        Args:
            mode: Overdraft enforcement mode.
            tolerance_pct: Tolerance as percentage (0-100).
            tolerance_kg: Tolerance in kilograms.
            total_inputs: Total period inputs for percentage calculation.

        Returns:
            Tolerance amount in kilograms (Decimal).
        """
        if mode == OverdraftMode.ZERO_TOLERANCE.value:
            return Decimal("0")
        elif mode == OverdraftMode.PERCENTAGE.value:
            if total_inputs <= 0:
                return Decimal("0")
            return (total_inputs * tolerance_pct / Decimal("100")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        elif mode == OverdraftMode.ABSOLUTE.value:
            return tolerance_kg
        else:
            logger.warning(
                "Unknown overdraft mode '%s', defaulting to zero_tolerance",
                mode,
            )
            return Decimal("0")

    # ------------------------------------------------------------------
    # Internal: Exemption Check
    # ------------------------------------------------------------------

    def _is_exempted(
        self,
        facility_id: str,
        commodity: str,
        amount: Decimal,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a facility has an active exemption covering the amount.

        Args:
            facility_id: Facility identifier.
            commodity: Commodity type.
            amount: Overdraft amount to check.

        Returns:
            Tuple of (is_exempted, exemption_id or None).
        """
        now = _utcnow()
        amount = Decimal(str(amount))

        with self._lock:
            exemption_ids = self._facility_exemptions.get(facility_id, [])
            for eid in exemption_ids:
                exemption = self._exemptions.get(eid)
                if exemption is None:
                    continue
                if not exemption.get("active", False):
                    continue

                # Check expiry
                try:
                    expiry = datetime.fromisoformat(exemption["expiry_date"])
                    if expiry <= now:
                        exemption["active"] = False
                        continue
                except (ValueError, KeyError):
                    continue

                # Check commodity (if specified in exemption)
                exempt_commodity = exemption.get("commodity", "")
                if exempt_commodity and commodity and exempt_commodity != commodity:
                    continue

                # Check amount
                exempt_amount = Decimal(exemption.get("amount_kg", "0"))
                utilized = Decimal(exemption.get("utilized_amount", "0"))
                remaining = exempt_amount - utilized
                if remaining >= amount:
                    return True, eid

        return False, None

    # ------------------------------------------------------------------
    # Internal: Record Overdraft Event
    # ------------------------------------------------------------------

    def _record_overdraft_event(
        self,
        check_id: str,
        ledger_id: str,
        facility_id: str,
        commodity: str,
        severity: str,
        current_balance: Decimal,
        overdraft_amount: Decimal,
        proposed_output_qty: Decimal,
        resulting_balance: Decimal,
        tolerance: Decimal,
        batch_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Record an overdraft event in the internal store.

        Args:
            check_id: Check identifier that triggered this event.
            ledger_id: Affected ledger.
            facility_id: Affected facility.
            commodity: Affected commodity.
            severity: Severity classification.
            current_balance: Balance at time of detection.
            overdraft_amount: Amount exceeding tolerance.
            proposed_output_qty: Proposed output that triggered overdraft.
            resulting_balance: Projected balance after output.
            tolerance: Computed tolerance.
            batch_id: Optional batch identifier.
            metadata: Optional additional context.

        Returns:
            Overdraft event dictionary.
        """
        event_id = _generate_id()
        now = _utcnow()
        resolution_deadline = now + timedelta(
            hours=self._config.overdraft_resolution_hours
        )

        event_data: Dict[str, Any] = {
            "event_id": event_id,
            "check_id": check_id,
            "ledger_id": ledger_id,
            "facility_id": facility_id,
            "commodity": commodity,
            "severity": severity,
            "current_balance": str(current_balance),
            "overdraft_amount": str(overdraft_amount),
            "proposed_output_qty": str(proposed_output_qty),
            "resulting_balance": str(resulting_balance),
            "tolerance": str(tolerance),
            "tolerance_mode": self._config.overdraft_mode,
            "batch_id": batch_id,
            "resolution_deadline": resolution_deadline.isoformat(),
            "resolved": False,
            "resolved_at": None,
            "resolved_by": None,
            "resolution_entry_id": None,
            "resolution_notes": None,
            "metadata": metadata or {},
            "provenance_hash": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        # Compute provenance hash
        provenance_hash = _compute_hash({
            "event_id": event_id,
            "ledger_id": ledger_id,
            "severity": severity,
            "overdraft_amount": str(overdraft_amount),
            "action": "detect",
        })
        event_data["provenance_hash"] = provenance_hash

        # Store event
        with self._lock:
            self._overdraft_events[event_id] = event_data
            if facility_id not in self._facility_events:
                self._facility_events[facility_id] = []
            self._facility_events[facility_id].append(event_id)

        return event_data

    # ------------------------------------------------------------------
    # Internal: Recommended Actions Builder
    # ------------------------------------------------------------------

    def _build_recommended_actions(
        self,
        severity: str,
        event_data: Dict[str, Any],
    ) -> List[str]:
        """Build a list of recommended actions based on severity.

        Args:
            severity: Overdraft severity.
            event_data: Overdraft event data.

        Returns:
            List of recommended action strings.
        """
        actions: List[str] = []
        overdraft_amount = event_data.get("overdraft_amount", "0")
        deadline_hours = self._config.overdraft_resolution_hours

        if severity == OverdraftSeverity.WARNING.value:
            actions = [
                "Monitor balance levels for this facility",
                f"Consider adding input of {overdraft_amount} kg "
                f"within {deadline_hours} hours",
                "Review output scheduling to prevent future overdrafts",
            ]
        elif severity == OverdraftSeverity.VIOLATION.value:
            actions = [
                f"REQUIRED: Add matching input of at least "
                f"{overdraft_amount} kg within {deadline_hours} hours",
                "Investigate root cause of overdraft",
                "Review and adjust output scheduling",
                "Consider requesting temporary exemption if justified",
            ]
        else:  # critical
            actions = [
                "URGENT: Output has been AUTO-REJECTED",
                f"Add matching input of at least {overdraft_amount} kg "
                f"immediately",
                "Escalate to compliance team",
                "Review all pending outputs for this facility",
                "Suspend further outputs until balance is restored",
                "Document root cause analysis",
            ]

        return actions

    # ------------------------------------------------------------------
    # Internal: Trend Direction Analysis
    # ------------------------------------------------------------------

    def _analyze_trend_direction(
        self,
        monthly_counts: Dict[str, int],
    ) -> str:
        """Analyze monthly overdraft counts to determine trend direction.

        Args:
            monthly_counts: Dictionary mapping month keys (YYYY-MM) to
                event counts.

        Returns:
            "improving", "stable", or "worsening".
        """
        if len(monthly_counts) < 2:
            return "stable"

        sorted_months = sorted(monthly_counts.keys())
        counts = [monthly_counts[m] for m in sorted_months]

        # Compare first half vs second half
        mid = len(counts) // 2
        if mid == 0:
            return "stable"

        first_half_avg = statistics.mean(counts[:mid])
        second_half_avg = statistics.mean(counts[mid:])

        if second_half_avg > first_half_avg * 1.2:
            return "worsening"
        elif second_half_avg < first_half_avg * 0.8:
            return "improving"
        else:
            return "stable"

    # ------------------------------------------------------------------
    # Internal: Pattern Recommendations Builder
    # ------------------------------------------------------------------

    def _build_pattern_recommendations(
        self,
        total_events: int,
        severity_dist: Dict[str, int],
        avg_resolution: float,
        unresolved: int,
        trend: str,
        repeat_offender: bool,
    ) -> List[str]:
        """Build recommendations based on overdraft pattern analysis.

        Args:
            total_events: Total events in analysis window.
            severity_dist: Severity distribution.
            avg_resolution: Average resolution time in hours.
            unresolved: Number of unresolved events.
            trend: Trend direction.
            repeat_offender: Whether facility is a repeat offender.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if total_events == 0:
            recs.append("No overdraft events detected. Facility is compliant.")
            return recs

        if repeat_offender:
            recs.append(
                "ATTENTION: Facility shows recurring overdraft pattern. "
                "Recommend enhanced monitoring and process review."
            )

        if trend == "worsening":
            recs.append(
                "Overdraft frequency is WORSENING. Investigate root causes "
                "and consider stricter output controls."
            )
        elif trend == "improving":
            recs.append(
                "Overdraft frequency is improving. Continue current "
                "mitigation measures."
            )

        critical_count = severity_dist.get(OverdraftSeverity.CRITICAL.value, 0)
        if critical_count > 0:
            recs.append(
                f"{critical_count} critical overdraft(s) detected. "
                f"Immediate compliance review required."
            )

        if unresolved > 0:
            recs.append(
                f"{unresolved} unresolved overdraft event(s). "
                f"Prioritize resolution."
            )

        deadline = self._config.overdraft_resolution_hours
        if avg_resolution > deadline:
            recs.append(
                f"Average resolution time ({avg_resolution:.1f}h) exceeds "
                f"deadline ({deadline}h). Improve response procedures."
            )

        if not recs:
            recs.append("Overdraft patterns are within acceptable limits.")

        return recs

    # ------------------------------------------------------------------
    # Internal: Event Serialization
    # ------------------------------------------------------------------

    def _serialize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize an overdraft event for API response.

        Args:
            event: Raw event dictionary.

        Returns:
            Serialized event dictionary with string representations.
        """
        return {
            "event_id": event.get("event_id", ""),
            "check_id": event.get("check_id", ""),
            "ledger_id": event.get("ledger_id", ""),
            "facility_id": event.get("facility_id", ""),
            "commodity": event.get("commodity", ""),
            "severity": event.get("severity", ""),
            "current_balance": event.get("current_balance", "0"),
            "overdraft_amount": event.get("overdraft_amount", "0"),
            "proposed_output_qty": event.get("proposed_output_qty", "0"),
            "resulting_balance": event.get("resulting_balance", "0"),
            "tolerance": event.get("tolerance", "0"),
            "tolerance_mode": event.get("tolerance_mode", ""),
            "batch_id": event.get("batch_id"),
            "resolution_deadline": event.get("resolution_deadline", ""),
            "resolved": event.get("resolved", False),
            "resolved_at": (
                event["resolved_at"].isoformat()
                if isinstance(event.get("resolved_at"), datetime)
                else event.get("resolved_at")
            ),
            "resolved_by": event.get("resolved_by"),
            "resolution_entry_id": event.get("resolution_entry_id"),
            "resolution_notes": event.get("resolution_notes"),
            "provenance_hash": event.get("provenance_hash"),
            "created_at": event.get("created_at", ""),
        }

    # ------------------------------------------------------------------
    # Internal: Input Validation
    # ------------------------------------------------------------------

    def _validate_string(self, field_name: str, value: str) -> None:
        """Validate that a string field is non-empty.

        Args:
            field_name: Name of the field for error messages.
            value: String value to validate.

        Raises:
            ValueError: If value is empty or None.
        """
        if not value or not str(value).strip():
            raise ValueError(f"{field_name} must not be empty")

    # ------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of overdraft events tracked."""
        with self._lock:
            return len(self._overdraft_events)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            total = len(self._overdraft_events)
            unresolved = sum(
                1
                for e in self._overdraft_events.values()
                if not e.get("resolved", False)
            )
            active_alerts = sum(
                1
                for a in self._active_alerts.values()
                if not a.get("resolved", False)
            )
        return (
            f"OverdraftDetector(events={total}, unresolved={unresolved}, "
            f"active_alerts={active_alerts}, mode={self._config.overdraft_mode})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "OverdraftDetector",
]
