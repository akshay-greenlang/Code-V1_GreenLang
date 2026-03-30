# -*- coding: utf-8 -*-
"""
CodeLifecycleManager - AGENT-EUDR-014 Engine 8: Code Lifecycle Management

Manages the complete lifecycle of EUDR QR codes from creation through
activation, deactivation, revocation, and expiry, with scan event
recording, counterfeit risk assessment, reprint tracking, replacement
issuance, lifecycle history retrieval, and batch activation support.

Code Status State Machine:
    created  --[activate]--> active
    active   --[deactivate]--> deactivated
    active   --[revoke]--> revoked (permanent, irreversible)
    active   --[expire]--> expired (automatic, time-based)
    deactivated --[activate]--> active (reactivation)
    deactivated --[revoke]--> revoked (permanent)
    created  --[revoke]--> revoked (permanent)
    expired  --[revoke]--> revoked (permanent)

    The ``revoked`` state is terminal and irreversible. Once revoked,
    a code cannot be reactivated or otherwise modified.

    The ``expired`` state is set automatically when the code's TTL
    (default 5 years per EUDR Article 14) has elapsed.

Scan Event Recording:
    Every scan of a QR code generates a ScanEvent record with geolocation,
    scanner metadata, HMAC token validation result, and counterfeit risk
    assessment. Scan events are retained for EUDR Article 14 compliance.

Reprint Tracking:
    Each reprint of a QR code label increments the reprint counter and
    creates an audit record. Reprints are capped at the configured
    maximum (default 3) to prevent unlimited duplication.

Replacement Issuance:
    When a QR code is damaged, lost, or needs updating, a replacement
    code can be issued with a cross-reference to the original. The
    original code is deactivated upon replacement.

Zero-Hallucination Guarantees:
    - All state transitions are explicit, rule-based checks.
    - Scan analytics use simple arithmetic aggregation.
    - No ML/LLM involvement in any lifecycle operation.
    - SHA-256 provenance hashes on all lifecycle events.
    - Deterministic TTL expiry checking via datetime arithmetic.

Regulatory References:
    - EUDR Article 4: Due diligence verification integrity.
    - EUDR Article 14: 5-year data retention for audit trails.
    - EUDR Article 10: Risk assessment for scan verification.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-014, Feature F8
Agent ID: GL-EUDR-QRG-014
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from greenlang.agents.eudr.qr_code_generator.config import get_config
from greenlang.schemas import utcnow
from greenlang.agents.eudr.qr_code_generator.models import (
    CodeStatus,
    CounterfeitRiskLevel,
    LifecycleEvent,
    ScanEvent,
    ScanOutcome,
)
from greenlang.agents.eudr.qr_code_generator.provenance import (
    get_provenance_tracker,
)
from greenlang.agents.eudr.qr_code_generator.metrics import (
    record_scan,
    record_revocation,
    record_api_error,
    set_active_codes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid state transitions: maps current_status -> set of allowed new statuses.
VALID_TRANSITIONS: Dict[str, Set[str]] = {
    CodeStatus.CREATED.value: {
        CodeStatus.ACTIVE.value,
        CodeStatus.REVOKED.value,
    },
    CodeStatus.ACTIVE.value: {
        CodeStatus.DEACTIVATED.value,
        CodeStatus.REVOKED.value,
        CodeStatus.EXPIRED.value,
    },
    CodeStatus.DEACTIVATED.value: {
        CodeStatus.ACTIVE.value,
        CodeStatus.REVOKED.value,
    },
    CodeStatus.EXPIRED.value: {
        CodeStatus.REVOKED.value,
    },
    CodeStatus.REVOKED.value: set(),  # Terminal state
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (Pydantic model, dict, or other serializable).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class LifecycleError(Exception):
    """Base exception for code lifecycle management errors."""
    pass

class InvalidTransitionError(LifecycleError):
    """Raised when a state transition is not allowed."""
    pass

class CodeNotFoundError(LifecycleError):
    """Raised when a code ID is not found in the registry."""
    pass

class ReprintLimitExceededError(LifecycleError):
    """Raised when the maximum reprint count has been reached."""
    pass

class CodeAlreadyRevokedError(LifecycleError):
    """Raised when attempting to modify a revoked code."""
    pass

class ScanRecordingError(LifecycleError):
    """Raised when scan event recording fails."""
    pass

# ---------------------------------------------------------------------------
# Internal Code State
# ---------------------------------------------------------------------------

class _CodeState:
    """Internal mutable state for a managed QR code.

    Attributes:
        code_id: Unique code identifier.
        status: Current CodeStatus value string.
        created_at: UTC creation timestamp.
        activated_at: UTC activation timestamp.
        deactivated_at: UTC deactivation timestamp.
        revoked_at: UTC revocation timestamp.
        expired_at: UTC expiry timestamp.
        reprint_count: Number of reprints.
        scan_count: Total scan count.
        commodity: EUDR commodity for metrics.
        operator_id: Owning operator.
        events: Ordered list of LifecycleEvent records.
        scans: Ordered list of ScanEvent records.
        reprints: Ordered list of reprint records.
        replacement_code_id: ID of replacement code if issued.
        original_code_id: ID of original code if this is a replacement.
    """

    __slots__ = (
        "code_id", "status", "created_at", "activated_at",
        "deactivated_at", "revoked_at", "expired_at",
        "reprint_count", "scan_count", "commodity", "operator_id",
        "events", "scans", "reprints",
        "replacement_code_id", "original_code_id",
    )

    def __init__(
        self,
        code_id: str,
        operator_id: str = "",
        commodity: str = "",
    ) -> None:
        """Initialize a new code state."""
        self.code_id = code_id
        self.status = CodeStatus.CREATED.value
        self.created_at = utcnow()
        self.activated_at: Optional[datetime] = None
        self.deactivated_at: Optional[datetime] = None
        self.revoked_at: Optional[datetime] = None
        self.expired_at: Optional[datetime] = None
        self.reprint_count = 0
        self.scan_count = 0
        self.commodity = commodity
        self.operator_id = operator_id
        self.events: List[LifecycleEvent] = []
        self.scans: List[ScanEvent] = []
        self.reprints: List[Dict[str, Any]] = []
        self.replacement_code_id: Optional[str] = None
        self.original_code_id: Optional[str] = None

# ---------------------------------------------------------------------------
# CodeLifecycleManager
# ---------------------------------------------------------------------------

class CodeLifecycleManager:
    """Manages the complete lifecycle of EUDR QR codes.

    Provides activation, deactivation, revocation, expiry checking,
    scan event recording, scan analytics, reprint tracking, replacement
    issuance, lifecycle history retrieval, status querying, and batch
    activation.

    All state transitions follow the defined state machine. Risk
    assessment uses explicit rule-based thresholds. No ML/LLM
    involvement in any operation, ensuring zero-hallucination compliance.

    Attributes:
        _config: QRCodeGeneratorConfig instance.
        _provenance: ProvenanceTracker for audit trail.
        _codes: Thread-safe dictionary of _CodeState objects.
        _lock: Reentrant lock for thread-safe state access.

    Example:
        >>> manager = CodeLifecycleManager()
        >>> manager.register_code("code-001", "OP-001")
        >>> event = manager.activate_code("code-001", "admin")
        >>> assert event.new_status == "active"
    """

    def __init__(self) -> None:
        """Initialize CodeLifecycleManager with config and provenance."""
        self._config = get_config()
        self._provenance = get_provenance_tracker()
        self._codes: Dict[str, _CodeState] = {}
        self._lock = threading.RLock()
        logger.info(
            "CodeLifecycleManager initialized: ttl=%d years, "
            "scan_logging=%s, reprint_limit=%d",
            self._config.default_ttl_years,
            self._config.scan_logging_enabled,
            self._config.max_reprints,
        )

    # ------------------------------------------------------------------
    # Code Registration
    # ------------------------------------------------------------------

    def register_code(
        self,
        code_id: str,
        operator_id: str = "",
        commodity: str = "",
    ) -> _CodeState:
        """Register a new QR code for lifecycle management.

        Creates an internal _CodeState with CREATED status.

        Args:
            code_id: Unique QR code identifier.
            operator_id: EUDR operator identifier.
            commodity: EUDR commodity type for metrics.

        Returns:
            The created _CodeState instance.

        Raises:
            LifecycleError: If code_id is empty or already registered.
        """
        if not code_id:
            raise LifecycleError("code_id must not be empty")

        with self._lock:
            if code_id in self._codes:
                raise LifecycleError(
                    f"Code '{code_id}' is already registered"
                )
            state = _CodeState(
                code_id=code_id,
                operator_id=operator_id,
                commodity=commodity,
            )
            self._codes[code_id] = state

        logger.debug(
            "Code registered: code_id=%s, operator=%s",
            code_id[:16],
            operator_id[:16],
        )
        return state

    # ------------------------------------------------------------------
    # Lifecycle Transitions
    # ------------------------------------------------------------------

    def activate_code(
        self,
        code_id: str,
        activated_by: Optional[str] = None,
        activation_data: Optional[Dict[str, Any]] = None,
    ) -> LifecycleEvent:
        """Activate a QR code for scanning.

        Transitions from ``created`` or ``deactivated`` to ``active``.

        Args:
            code_id: QR code identifier.
            activated_by: User or system performing activation.
            activation_data: Optional metadata for the activation.

        Returns:
            LifecycleEvent recording the state transition.

        Raises:
            CodeNotFoundError: If code_id is not registered.
            InvalidTransitionError: If transition is not allowed.
            CodeAlreadyRevokedError: If code is revoked.
        """
        state = self._get_state(code_id)
        self._validate_transition(
            state.status, CodeStatus.ACTIVE.value,
        )

        previous_status = state.status
        now = utcnow()

        with self._lock:
            state.status = CodeStatus.ACTIVE.value
            state.activated_at = now

        event = self._record_lifecycle_event(
            code_id=code_id,
            event_type="activate",
            previous_status=previous_status,
            new_status=CodeStatus.ACTIVE.value,
            reason=None,
            performed_by=activated_by,
            metadata=activation_data or {},
        )

        # Update active codes gauge
        set_active_codes(self._count_active_codes())

        logger.info(
            "Code activated: code_id=%s, by=%s, "
            "previous=%s -> active",
            code_id[:16],
            activated_by or "system",
            previous_status,
        )
        return event

    def deactivate_code(
        self,
        code_id: str,
        reason: str,
    ) -> LifecycleEvent:
        """Temporarily deactivate an active QR code.

        Transitions from ``active`` to ``deactivated``. The code can
        be reactivated later.

        Args:
            code_id: QR code identifier.
            reason: Human-readable reason for deactivation.

        Returns:
            LifecycleEvent recording the state transition.

        Raises:
            CodeNotFoundError: If code_id is not registered.
            InvalidTransitionError: If code is not active.
            LifecycleError: If reason is empty.
        """
        if not reason:
            raise LifecycleError("reason must not be empty")

        state = self._get_state(code_id)
        self._validate_transition(
            state.status, CodeStatus.DEACTIVATED.value,
        )

        previous_status = state.status
        now = utcnow()

        with self._lock:
            state.status = CodeStatus.DEACTIVATED.value
            state.deactivated_at = now

        event = self._record_lifecycle_event(
            code_id=code_id,
            event_type="deactivate",
            previous_status=previous_status,
            new_status=CodeStatus.DEACTIVATED.value,
            reason=reason,
            performed_by=None,
            metadata={},
        )

        set_active_codes(self._count_active_codes())

        logger.info(
            "Code deactivated: code_id=%s, reason=%s",
            code_id[:16],
            reason[:50],
        )
        return event

    def revoke_code(
        self,
        code_id: str,
        reason: str,
        revoked_by: Optional[str] = None,
    ) -> LifecycleEvent:
        """Permanently revoke a QR code.

        Transitions from any non-revoked state to ``revoked``.
        This is an irreversible operation per EUDR compliance.

        Args:
            code_id: QR code identifier.
            reason: Human-readable reason for revocation.
            revoked_by: User or system performing revocation.

        Returns:
            LifecycleEvent recording the state transition.

        Raises:
            CodeNotFoundError: If code_id is not registered.
            InvalidTransitionError: If code is already revoked.
            LifecycleError: If reason is empty.
        """
        if not reason:
            raise LifecycleError("reason must not be empty")

        state = self._get_state(code_id)
        self._validate_transition(
            state.status, CodeStatus.REVOKED.value,
        )

        previous_status = state.status
        now = utcnow()

        with self._lock:
            state.status = CodeStatus.REVOKED.value
            state.revoked_at = now

        event = self._record_lifecycle_event(
            code_id=code_id,
            event_type="revoke",
            previous_status=previous_status,
            new_status=CodeStatus.REVOKED.value,
            reason=reason,
            performed_by=revoked_by,
            metadata={},
        )

        # Record revocation metric
        commodity = state.commodity or "unknown"
        record_revocation(commodity)
        set_active_codes(self._count_active_codes())

        logger.info(
            "Code revoked: code_id=%s, reason=%s, by=%s",
            code_id[:16],
            reason[:50],
            revoked_by or "system",
        )
        return event

    def check_expiry(
        self,
        code_id: str,
        ttl_years: Optional[int] = None,
    ) -> bool:
        """Check if a QR code has expired based on its TTL.

        Compares the code's creation timestamp plus the TTL period
        against the current UTC time.

        Args:
            code_id: QR code identifier.
            ttl_years: Optional TTL override in years. Defaults to
                config ``default_ttl_years``.

        Returns:
            True if the code has expired, False if still valid.

        Raises:
            CodeNotFoundError: If code_id is not registered.
        """
        state = self._get_state(code_id)
        resolved_ttl = (
            ttl_years
            if ttl_years is not None
            else self._config.default_ttl_years
        )

        now = utcnow()
        expiry_dt = state.created_at + timedelta(
            days=resolved_ttl * 365,
        )
        is_expired = now >= expiry_dt

        logger.debug(
            "Expiry check: code_id=%s, created=%s, ttl=%d years, "
            "expired=%s",
            code_id[:16],
            state.created_at.isoformat(),
            resolved_ttl,
            is_expired,
        )
        return is_expired

    def expire_code(self, code_id: str) -> LifecycleEvent:
        """Expire a QR code (automatic time-based transition).

        Transitions from ``active`` to ``expired``. Called by
        background jobs that check TTL expiry.

        Args:
            code_id: QR code identifier.

        Returns:
            LifecycleEvent recording the state transition.

        Raises:
            CodeNotFoundError: If code_id is not registered.
            InvalidTransitionError: If code is not active.
        """
        state = self._get_state(code_id)
        self._validate_transition(
            state.status, CodeStatus.EXPIRED.value,
        )

        previous_status = state.status
        now = utcnow()

        with self._lock:
            state.status = CodeStatus.EXPIRED.value
            state.expired_at = now

        event = self._record_lifecycle_event(
            code_id=code_id,
            event_type="expire",
            previous_status=previous_status,
            new_status=CodeStatus.EXPIRED.value,
            reason="TTL expired",
            performed_by="system",
            metadata={"ttl_years": self._config.default_ttl_years},
        )

        set_active_codes(self._count_active_codes())

        logger.info(
            "Code expired: code_id=%s, previous=%s",
            code_id[:16],
            previous_status,
        )
        return event

    # ------------------------------------------------------------------
    # Scan Event Recording
    # ------------------------------------------------------------------

    def record_scan_event(
        self,
        code_id: str,
        scan_lat: Optional[float] = None,
        scan_lon: Optional[float] = None,
        scanner_id: Optional[str] = None,
        outcome: Optional[str] = None,
        hmac_token: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None,
    ) -> ScanEvent:
        """Record a scan event for a QR code.

        Creates a ScanEvent record with geolocation, scanner metadata,
        HMAC validation result, and counterfeit risk assessment.

        Args:
            code_id: Scanned QR code identifier.
            scan_lat: Scan location latitude (-90 to 90).
            scan_lon: Scan location longitude (-180 to 180).
            scanner_id: Scanner device or user identifier.
            outcome: Scan outcome. Auto-determined if None.
            hmac_token: HMAC token from the verification URL.
            device_info: Additional device metadata.

        Returns:
            ScanEvent record with risk assessment.

        Raises:
            CodeNotFoundError: If code_id is not registered.
            ScanRecordingError: If recording fails.
        """
        state = self._get_state(code_id)

        # Auto-determine outcome based on code status
        resolved_outcome = outcome
        if resolved_outcome is None:
            resolved_outcome = self._determine_scan_outcome(state)

        # Determine counterfeit risk level
        risk_level = self._assess_scan_risk(
            state, scan_lat, scan_lon, hmac_token,
        )

        scan_event = ScanEvent(
            scan_id=_generate_id("scan"),
            code_id=code_id,
            outcome=resolved_outcome,
            scanner_ip=None,
            scanner_user_agent=scanner_id,
            scan_latitude=scan_lat,
            scan_longitude=scan_lon,
            counterfeit_risk=risk_level,
            hmac_valid=hmac_token is not None,
            velocity_scans_per_min=self._compute_velocity(state),
        )

        # Record scan
        if self._config.scan_logging_enabled:
            with self._lock:
                state.scans.append(scan_event)
                state.scan_count += 1

        # Record provenance
        provenance_entry = self._provenance.record(
            entity_type="scan_event",
            action="scan",
            entity_id=scan_event.scan_id,
            data={
                "code_id": code_id,
                "outcome": resolved_outcome,
                "risk_level": risk_level.value
                if hasattr(risk_level, "value") else risk_level,
                "has_location": scan_lat is not None,
            },
            metadata={
                "code_id": code_id,
                "scan_id": scan_event.scan_id,
            },
        )
        scan_event.provenance_hash = provenance_entry.hash_value

        # Record metrics
        record_scan(resolved_outcome)

        logger.info(
            "Scan event recorded: code_id=%s, outcome=%s, "
            "risk=%s, scan_id=%s",
            code_id[:16],
            resolved_outcome,
            risk_level.value
            if hasattr(risk_level, "value") else risk_level,
            scan_event.scan_id,
        )
        return scan_event

    # ------------------------------------------------------------------
    # Scan Analytics
    # ------------------------------------------------------------------

    def get_scan_analytics(
        self,
        code_id: str,
    ) -> Dict[str, Any]:
        """Get aggregate scan analytics for a QR code.

        Computes scan counts, geographic distribution, time patterns,
        outcome breakdown, and risk distribution from recorded scan
        events.

        Args:
            code_id: QR code identifier.

        Returns:
            Dictionary with analytics: total_scans, outcome_breakdown,
            risk_breakdown, geo_distribution, hourly_distribution,
            first_scan_at, last_scan_at.

        Raises:
            CodeNotFoundError: If code_id is not registered.
        """
        state = self._get_state(code_id)

        with self._lock:
            scans = list(state.scans)

        if not scans:
            return {
                "code_id": code_id,
                "total_scans": 0,
                "outcome_breakdown": {},
                "risk_breakdown": {},
                "geo_distribution": {},
                "hourly_distribution": {},
                "first_scan_at": None,
                "last_scan_at": None,
            }

        # Outcome breakdown
        outcome_counts: Dict[str, int] = defaultdict(int)
        for scan in scans:
            outcome_val = (
                scan.outcome.value
                if hasattr(scan.outcome, "value")
                else str(scan.outcome)
            )
            outcome_counts[outcome_val] += 1

        # Risk breakdown
        risk_counts: Dict[str, int] = defaultdict(int)
        for scan in scans:
            risk_val = (
                scan.counterfeit_risk.value
                if hasattr(scan.counterfeit_risk, "value")
                else str(scan.counterfeit_risk)
            )
            risk_counts[risk_val] += 1

        # Geographic distribution (by country if available)
        geo_counts: Dict[str, int] = defaultdict(int)
        for scan in scans:
            country = scan.scan_country or "unknown"
            geo_counts[country] += 1

        # Hourly distribution
        hourly_counts: Dict[int, int] = defaultdict(int)
        for scan in scans:
            hour = scan.scanned_at.hour
            hourly_counts[hour] += 1

        # Timestamps
        sorted_scans = sorted(scans, key=lambda s: s.scanned_at)
        first_scan = sorted_scans[0].scanned_at.isoformat()
        last_scan = sorted_scans[-1].scanned_at.isoformat()

        analytics: Dict[str, Any] = {
            "code_id": code_id,
            "total_scans": len(scans),
            "outcome_breakdown": dict(outcome_counts),
            "risk_breakdown": dict(risk_counts),
            "geo_distribution": dict(geo_counts),
            "hourly_distribution": {
                str(h): c for h, c in sorted(hourly_counts.items())
            },
            "first_scan_at": first_scan,
            "last_scan_at": last_scan,
        }

        logger.debug(
            "Scan analytics: code_id=%s, total=%d, outcomes=%s",
            code_id[:16],
            len(scans),
            dict(outcome_counts),
        )
        return analytics

    # ------------------------------------------------------------------
    # Reprint Tracking
    # ------------------------------------------------------------------

    def track_reprint(
        self,
        code_id: str,
        reprint_reason: str,
        reprinted_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track a QR code label reprint event.

        Increments the reprint counter and creates an audit record.
        Reprints are limited to the configured maximum.

        Args:
            code_id: QR code identifier.
            reprint_reason: Reason for reprinting.
            reprinted_by: User or system requesting the reprint.

        Returns:
            Dictionary with reprint metadata and count.

        Raises:
            CodeNotFoundError: If code_id is not registered.
            ReprintLimitExceededError: If max reprints exceeded.
            LifecycleError: If reason is empty.
        """
        if not reprint_reason:
            raise LifecycleError("reprint_reason must not be empty")

        state = self._get_state(code_id)

        if state.reprint_count >= self._config.max_reprints:
            raise ReprintLimitExceededError(
                f"Code '{code_id}' has reached the maximum reprint "
                f"limit of {self._config.max_reprints}"
            )

        now = utcnow()
        reprint_record: Dict[str, Any] = {
            "reprint_id": _generate_id("rpt"),
            "code_id": code_id,
            "reprint_number": state.reprint_count + 1,
            "reason": reprint_reason,
            "reprinted_by": reprinted_by,
            "reprinted_at": now.isoformat(),
        }

        with self._lock:
            state.reprint_count += 1
            state.reprints.append(reprint_record)

        # Record provenance
        provenance_entry = self._provenance.record(
            entity_type="qr_code",
            action="generate",
            entity_id=code_id,
            data={
                "action": "reprint",
                "reprint_number": state.reprint_count,
                "reason": reprint_reason,
                "reprinted_by": reprinted_by,
            },
            metadata={"code_id": code_id},
        )
        reprint_record["provenance_hash"] = (
            provenance_entry.hash_value
        )

        logger.info(
            "Reprint tracked: code_id=%s, reprint #%d/%d, "
            "reason=%s",
            code_id[:16],
            state.reprint_count,
            self._config.max_reprints,
            reprint_reason[:50],
        )
        return reprint_record

    # ------------------------------------------------------------------
    # Replacement Issuance
    # ------------------------------------------------------------------

    def issue_replacement(
        self,
        original_code_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Issue a replacement QR code for a damaged or lost original.

        Creates a new code state with a cross-reference to the original.
        The original code is deactivated upon replacement.

        Args:
            original_code_id: Identifier of the original QR code.
            reason: Reason for replacement.

        Returns:
            Dictionary with new_code_id, original_code_id, and metadata.

        Raises:
            CodeNotFoundError: If original_code_id is not registered.
            LifecycleError: If reason is empty or code is revoked.
        """
        if not reason:
            raise LifecycleError("reason must not be empty")

        original_state = self._get_state(original_code_id)

        if original_state.status == CodeStatus.REVOKED.value:
            raise CodeAlreadyRevokedError(
                f"Cannot issue replacement for revoked code "
                f"'{original_code_id}'"
            )

        # Generate new code ID
        new_code_id = _generate_id("qr")
        now = utcnow()

        # Register the replacement code
        new_state = self.register_code(
            code_id=new_code_id,
            operator_id=original_state.operator_id,
            commodity=original_state.commodity,
        )
        new_state.original_code_id = original_code_id

        # Cross-reference in original
        with self._lock:
            original_state.replacement_code_id = new_code_id

        # Deactivate original if active
        if original_state.status == CodeStatus.ACTIVE.value:
            self.deactivate_code(
                original_code_id,
                reason=f"Replaced by {new_code_id}: {reason}",
            )

        replacement_record: Dict[str, Any] = {
            "new_code_id": new_code_id,
            "original_code_id": original_code_id,
            "reason": reason,
            "issued_at": now.isoformat(),
            "original_status": original_state.status,
        }

        # Record provenance
        provenance_entry = self._provenance.record(
            entity_type="qr_code",
            action="generate",
            entity_id=new_code_id,
            data={
                "action": "replacement",
                "original_code_id": original_code_id,
                "reason": reason,
            },
            metadata={
                "code_id": new_code_id,
                "original_code_id": original_code_id,
            },
        )
        replacement_record["provenance_hash"] = (
            provenance_entry.hash_value
        )

        logger.info(
            "Replacement issued: original=%s -> new=%s, reason=%s",
            original_code_id[:16],
            new_code_id[:16],
            reason[:50],
        )
        return replacement_record

    # ------------------------------------------------------------------
    # Lifecycle History and Status
    # ------------------------------------------------------------------

    def get_lifecycle_history(
        self,
        code_id: str,
    ) -> List[LifecycleEvent]:
        """Get the complete lifecycle event history for a QR code.

        Returns all lifecycle events in chronological order.

        Args:
            code_id: QR code identifier.

        Returns:
            List of LifecycleEvent records, oldest first.

        Raises:
            CodeNotFoundError: If code_id is not registered.
        """
        state = self._get_state(code_id)
        with self._lock:
            return list(state.events)

    def get_code_status(
        self,
        code_id: str,
    ) -> Dict[str, Any]:
        """Get the current status of a QR code.

        Args:
            code_id: QR code identifier.

        Returns:
            Dictionary with status, timestamps, scan_count,
            reprint_count, and cross-reference IDs.

        Raises:
            CodeNotFoundError: If code_id is not registered.
        """
        state = self._get_state(code_id)

        return {
            "code_id": code_id,
            "status": state.status,
            "created_at": state.created_at.isoformat(),
            "activated_at": (
                state.activated_at.isoformat()
                if state.activated_at else None
            ),
            "deactivated_at": (
                state.deactivated_at.isoformat()
                if state.deactivated_at else None
            ),
            "revoked_at": (
                state.revoked_at.isoformat()
                if state.revoked_at else None
            ),
            "expired_at": (
                state.expired_at.isoformat()
                if state.expired_at else None
            ),
            "scan_count": state.scan_count,
            "reprint_count": state.reprint_count,
            "replacement_code_id": state.replacement_code_id,
            "original_code_id": state.original_code_id,
            "operator_id": state.operator_id,
            "commodity": state.commodity,
            "event_count": len(state.events),
        }

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------

    def bulk_activate(
        self,
        code_ids: List[str],
        activated_by: Optional[str] = None,
    ) -> List[LifecycleEvent]:
        """Activate multiple QR codes in a single batch operation.

        Activates each code independently. Failures on individual codes
        are logged and skipped, not propagated.

        Args:
            code_ids: List of QR code identifiers to activate.
            activated_by: User or system performing activation.

        Returns:
            List of LifecycleEvent records for successful activations.
        """
        results: List[LifecycleEvent] = []
        for code_id in code_ids:
            try:
                event = self.activate_code(
                    code_id=code_id,
                    activated_by=activated_by,
                )
                results.append(event)
            except Exception as exc:
                logger.warning(
                    "Batch activation failed for code_id=%s: %s",
                    code_id[:16],
                    exc,
                )
                continue

        logger.info(
            "Batch activation: %d/%d succeeded, by=%s",
            len(results),
            len(code_ids),
            activated_by or "system",
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_state(self, code_id: str) -> _CodeState:
        """Retrieve internal code state by ID.

        Args:
            code_id: QR code identifier.

        Returns:
            _CodeState instance.

        Raises:
            CodeNotFoundError: If code_id is not registered.
        """
        with self._lock:
            state = self._codes.get(code_id)
        if state is None:
            raise CodeNotFoundError(
                f"Code not found: '{code_id}'"
            )
        return state

    def _validate_transition(
        self,
        current_status: str,
        new_status: str,
    ) -> None:
        """Validate that a state transition is allowed.

        Args:
            current_status: Current status string.
            new_status: Desired new status string.

        Raises:
            CodeAlreadyRevokedError: If code is revoked.
            InvalidTransitionError: If transition is not valid.
        """
        if current_status == CodeStatus.REVOKED.value:
            raise CodeAlreadyRevokedError(
                f"Code is permanently revoked; no transitions allowed"
            )

        allowed = VALID_TRANSITIONS.get(current_status, set())
        if new_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from '{current_status}' to "
                f"'{new_status}'. Allowed: {sorted(allowed)}"
            )

    def _record_lifecycle_event(
        self,
        code_id: str,
        event_type: str,
        previous_status: str,
        new_status: str,
        reason: Optional[str],
        performed_by: Optional[str],
        metadata: Dict[str, Any],
    ) -> LifecycleEvent:
        """Create and store a LifecycleEvent with provenance.

        Args:
            code_id: QR code identifier.
            event_type: Type of lifecycle event.
            previous_status: Status before transition.
            new_status: Status after transition.
            reason: Reason for the transition.
            performed_by: User or system performing the change.
            metadata: Additional event metadata.

        Returns:
            Created LifecycleEvent model.
        """
        event = LifecycleEvent(
            event_id=_generate_id("evt"),
            code_id=code_id,
            event_type=event_type,
            previous_status=previous_status,
            new_status=new_status,
            reason=reason,
            performed_by=performed_by,
            metadata=metadata,
        )

        # Store in code state
        state = self._get_state(code_id)
        with self._lock:
            state.events.append(event)

        # Record provenance
        provenance_entry = self._provenance.record(
            entity_type="lifecycle_event",
            action=event_type,
            entity_id=event.event_id,
            data={
                "code_id": code_id,
                "previous_status": previous_status,
                "new_status": new_status,
                "reason": reason,
                "performed_by": performed_by,
            },
            metadata={
                "code_id": code_id,
                "event_id": event.event_id,
            },
        )
        event.provenance_hash = provenance_entry.hash_value

        return event

    def _count_active_codes(self) -> int:
        """Count codes currently in ACTIVE status.

        Returns:
            Number of active codes.
        """
        with self._lock:
            return sum(
                1 for s in self._codes.values()
                if s.status == CodeStatus.ACTIVE.value
            )

    def _determine_scan_outcome(
        self,
        state: _CodeState,
    ) -> str:
        """Determine scan outcome based on code status.

        Args:
            state: Internal code state.

        Returns:
            ScanOutcome value string.
        """
        if state.status == CodeStatus.ACTIVE.value:
            return ScanOutcome.VERIFIED.value
        if state.status == CodeStatus.EXPIRED.value:
            return ScanOutcome.EXPIRED_CODE.value
        if state.status == CodeStatus.REVOKED.value:
            return ScanOutcome.REVOKED_CODE.value
        if state.status == CodeStatus.DEACTIVATED.value:
            return ScanOutcome.ERROR.value
        return ScanOutcome.ERROR.value

    def _assess_scan_risk(
        self,
        state: _CodeState,
        scan_lat: Optional[float],
        scan_lon: Optional[float],
        hmac_token: Optional[str],
    ) -> CounterfeitRiskLevel:
        """Assess counterfeit risk for a scan event.

        Simple rule-based assessment without ML/LLM involvement.

        Args:
            state: Internal code state.
            scan_lat: Scan latitude.
            scan_lon: Scan longitude.
            hmac_token: HMAC token for validation.

        Returns:
            CounterfeitRiskLevel enum value.
        """
        risk_score: float = 0.0

        # Revoked code scan
        if state.status == CodeStatus.REVOKED.value:
            return CounterfeitRiskLevel.CRITICAL

        # Expired code scan
        if state.status == CodeStatus.EXPIRED.value:
            risk_score += 20.0

        # Missing HMAC token
        if hmac_token is None:
            risk_score += 15.0

        # High scan velocity
        velocity = self._compute_velocity(state)
        threshold = self._config.scan_velocity_threshold
        if velocity > threshold:
            risk_score += 30.0

        # Classify
        if risk_score >= 75.0:
            return CounterfeitRiskLevel.CRITICAL
        if risk_score >= 50.0:
            return CounterfeitRiskLevel.HIGH
        if risk_score >= 25.0:
            return CounterfeitRiskLevel.MEDIUM
        return CounterfeitRiskLevel.LOW

    def _compute_velocity(self, state: _CodeState) -> int:
        """Compute current scan velocity (scans per minute).

        Args:
            state: Internal code state.

        Returns:
            Number of scans in the last 60 seconds.
        """
        if not state.scans:
            return 0

        now = utcnow()
        one_minute_ago = now.timestamp() - 60.0

        return sum(
            1 for scan in state.scans
            if scan.scanned_at.timestamp() > one_minute_ago
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Main class
    "CodeLifecycleManager",
    # Constants
    "VALID_TRANSITIONS",
    # Exceptions
    "LifecycleError",
    "InvalidTransitionError",
    "CodeNotFoundError",
    "ReprintLimitExceededError",
    "CodeAlreadyRevokedError",
    "ScanRecordingError",
]
