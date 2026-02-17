# -*- coding: utf-8 -*-
"""
SLA Definition Engine - AGENT-DATA-016 Data Freshness Monitor

Pure-Python engine for managing Service Level Agreement (SLA) rules for
monitored datasets. Provides SLA creation, evaluation, template
management, business-hours-aware threshold calculation, and breach
severity classification. All state is stored in-memory with thread-safe
access via a threading.Lock.

Zero-Hallucination: All SLA evaluations use deterministic Python
arithmetic against explicit threshold values. No LLM calls for
numeric computations. SLA status and breach severity are derived
from simple, auditable comparisons against configured warning and
critical hour thresholds.

Engine 2 of 7 in the Data Freshness Monitor pipeline.

Example:
    >>> from greenlang.data_freshness_monitor.sla_definition import (
    ...     SLADefinitionEngine,
    ... )
    >>> engine = SLADefinitionEngine()
    >>> sla = engine.create_sla(
    ...     dataset_id="ds-001",
    ...     warning_hours=24.0,
    ...     critical_hours=72.0,
    ... )
    >>> status = engine.evaluate_sla(sla.id, age_hours=36.0)
    >>> print(status)  # SLAStatus.WARNING

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.models import (
        SLAStatus,
        BreachSeverity,
        EscalationPolicy,
        SLADefinition,
        SLATemplate,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.info(
        "data_freshness_monitor.models not available; "
        "using inline model definitions for SLA definition engine"
    )

    class SLAStatus(Enum):  # type: ignore[no-redef]
        """SLA compliance status for a dataset freshness check."""

        COMPLIANT = "compliant"
        WARNING = "warning"
        BREACHED = "breached"
        CRITICAL = "critical"

    class BreachSeverity(Enum):  # type: ignore[no-redef]
        """Severity level of an SLA breach."""

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    class EscalationPolicy(Enum):  # type: ignore[no-redef]
        """Escalation policy applied when an SLA is breached."""

        NONE = "none"
        NOTIFY = "notify"
        ESCALATE = "escalate"
        PAGE = "page"

    class SLADefinition:  # type: ignore[no-redef]
        """In-memory representation of a dataset SLA rule.

        Attributes:
            id: Unique SLA identifier (UUID4 hex).
            dataset_id: Dataset this SLA applies to.
            warning_hours: Hours before warning status triggers.
            critical_hours: Hours before breach status triggers.
            breach_severity: Default severity when breached.
            escalation_policy: Escalation policy on breach.
            business_hours_only: Whether to count only business hours.
            created_at: UTC creation timestamp.
            updated_at: UTC last-update timestamp.
            provenance_hash: SHA-256 provenance chain hash.
        """

        __slots__ = (
            "id", "dataset_id", "warning_hours", "critical_hours",
            "breach_severity", "escalation_policy", "business_hours_only",
            "created_at", "updated_at", "provenance_hash",
        )

        def __init__(
            self,
            *,
            id: str,
            dataset_id: str,
            warning_hours: float,
            critical_hours: float,
            breach_severity: str = "medium",
            escalation_policy: str = "notify",
            business_hours_only: bool = False,
            created_at: Optional[datetime] = None,
            updated_at: Optional[datetime] = None,
            provenance_hash: str = "",
        ) -> None:
            """Initialize SLADefinition."""
            self.id = id
            self.dataset_id = dataset_id
            self.warning_hours = warning_hours
            self.critical_hours = critical_hours
            self.breach_severity = breach_severity
            self.escalation_policy = escalation_policy
            self.business_hours_only = business_hours_only
            self.created_at = created_at or _utcnow()
            self.updated_at = updated_at or _utcnow()
            self.provenance_hash = provenance_hash

        def to_dict(self) -> Dict[str, Any]:
            """Serialize to dictionary."""
            return {
                "id": self.id,
                "dataset_id": self.dataset_id,
                "warning_hours": self.warning_hours,
                "critical_hours": self.critical_hours,
                "breach_severity": self.breach_severity,
                "escalation_policy": self.escalation_policy,
                "business_hours_only": self.business_hours_only,
                "created_at": self.created_at.isoformat()
                if self.created_at else None,
                "updated_at": self.updated_at.isoformat()
                if self.updated_at else None,
                "provenance_hash": self.provenance_hash,
            }

    class SLATemplate:  # type: ignore[no-redef]
        """Reusable SLA template that can be applied to multiple datasets.

        Attributes:
            id: Unique template identifier (UUID4 hex).
            name: Human-readable template name.
            warning_hours: Default warning threshold hours.
            critical_hours: Default critical threshold hours.
            breach_severity: Default breach severity.
            escalation_policy: Default escalation policy.
            created_at: UTC creation timestamp.
        """

        __slots__ = (
            "id", "name", "warning_hours", "critical_hours",
            "breach_severity", "escalation_policy", "created_at",
        )

        def __init__(
            self,
            *,
            id: str,
            name: str,
            warning_hours: float,
            critical_hours: float,
            breach_severity: str = "medium",
            escalation_policy: str = "notify",
            created_at: Optional[datetime] = None,
        ) -> None:
            """Initialize SLATemplate."""
            self.id = id
            self.name = name
            self.warning_hours = warning_hours
            self.critical_hours = critical_hours
            self.breach_severity = breach_severity
            self.escalation_policy = escalation_policy
            self.created_at = created_at or _utcnow()

        def to_dict(self) -> Dict[str, Any]:
            """Serialize to dictionary."""
            return {
                "id": self.id,
                "name": self.name,
                "warning_hours": self.warning_hours,
                "critical_hours": self.critical_hours,
                "breach_severity": self.breach_severity,
                "escalation_policy": self.escalation_policy,
                "created_at": self.created_at.isoformat()
                if self.created_at else None,
            }


# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.metrics import (
        inc_jobs_processed as _inc_jobs_processed,
        observe_duration as _observe_duration_raw,
        inc_errors as _inc_errors,
    )
    _METRICS_AVAILABLE = True

    def inc_jobs_processed(status: str) -> None:
        """Delegate to metrics module inc_jobs_processed."""
        _inc_jobs_processed(status)

    def observe_duration(operation: str, duration: float) -> None:
        """Delegate to metrics module observe_duration (duration only)."""
        _observe_duration_raw(duration)

    def inc_errors(error_type: str) -> None:
        """Delegate to metrics module inc_errors."""
        _inc_errors(error_type)

except ImportError:
    _METRICS_AVAILABLE = False

    def inc_jobs_processed(status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    def observe_duration(operation: str, duration: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    def inc_errors(error_type: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    logger.info(
        "data_freshness_monitor.metrics not available; "
        "SLA definition engine metrics disabled"
    )

# ---------------------------------------------------------------------------
# Provenance import (graceful fallback with inline tracker)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_freshness_monitor.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    logger.info(
        "data_freshness_monitor.provenance not available; "
        "using inline ProvenanceTracker"
    )

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal inline provenance tracker for standalone operation.

        Provides SHA-256 chain hashing without external dependencies.
        """

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-data-freshness-monitor-genesis"
        ).hexdigest()

        def __init__(self) -> None:
            """Initialize with genesis hash."""
            self._last_chain_hash: str = self.GENESIS_HASH
            self._chain: List[Dict[str, Any]] = []
            self._lock = threading.Lock()

        def hash_record(self, data: Dict[str, Any]) -> str:
            """Compute deterministic SHA-256 hash of a data record.

            Args:
                data: Dictionary to hash.

            Returns:
                Hex-encoded SHA-256 hash string.
            """
            serialized = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        def add_to_chain(
            self,
            operation: str,
            input_hash: str,
            output_hash: str,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> str:
            """Add a chain link and return the new chain hash.

            Args:
                operation: Name of the operation performed.
                input_hash: SHA-256 hash of the operation input.
                output_hash: SHA-256 hash of the operation output.
                metadata: Optional additional metadata.

            Returns:
                New chain hash linking this entry to the previous.
            """
            timestamp = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
            with self._lock:
                combined = json.dumps({
                    "previous": self._last_chain_hash,
                    "input": input_hash,
                    "output": output_hash,
                    "operation": operation,
                    "timestamp": timestamp,
                }, sort_keys=True)
                chain_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()

                self._chain.append({
                    "operation": operation,
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "chain_hash": chain_hash,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                })
                self._last_chain_hash = chain_hash

            return chain_hash

        def get_chain(self) -> List[Dict[str, Any]]:
            """Return the full provenance chain.

            Returns:
                List of provenance entries, oldest first.
            """
            with self._lock:
                return list(self._chain)

        @property
        def entry_count(self) -> int:
            """Return the total number of provenance entries."""
            with self._lock:
                return len(self._chain)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id() -> str:
    """Generate a unique identifier (UUID4 hex string).

    Returns:
        32-character lowercase hexadecimal string.
    """
    return uuid4().hex


def _resolve_breach_severity(value: str) -> str:
    """Resolve a breach severity string to a canonical lowercase value.

    Args:
        value: Breach severity string (case-insensitive).

    Returns:
        Canonical lowercase severity string.

    Raises:
        ValueError: If the value is not a recognised breach severity.
    """
    canonical = value.lower().strip()
    valid_values = {e.value for e in BreachSeverity}
    if canonical not in valid_values:
        raise ValueError(
            f"Invalid breach severity: '{value}'. "
            f"Valid values: {sorted(valid_values)}"
        )
    return canonical


def _resolve_escalation_policy(value: str) -> str:
    """Resolve an escalation policy string to a canonical lowercase value.

    Args:
        value: Escalation policy string (case-insensitive).

    Returns:
        Canonical lowercase policy string.

    Raises:
        ValueError: If the value is not a recognised escalation policy.
    """
    canonical = value.lower().strip()
    valid_values = {e.value for e in EscalationPolicy}
    if canonical not in valid_values:
        raise ValueError(
            f"Invalid escalation policy: '{value}'. "
            f"Valid values: {sorted(valid_values)}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Business hours constants
# ---------------------------------------------------------------------------

#: Business hours start (inclusive), 09:00 UTC.
_BUSINESS_HOUR_START: int = 9

#: Business hours end (exclusive), 18:00 UTC (i.e., last business hour is 17).
_BUSINESS_HOUR_END: int = 18

#: Business hours per day.
_BUSINESS_HOURS_PER_DAY: float = float(_BUSINESS_HOUR_END - _BUSINESS_HOUR_START)

#: Business days per week (Monday=0 through Friday=4).
_BUSINESS_DAYS: set = {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# SLADefinitionEngine
# ---------------------------------------------------------------------------


class SLADefinitionEngine:
    """Pure-Python engine for managing SLA rules for monitored datasets.

    Manages the lifecycle of SLA definitions including creation,
    evaluation, template management, business-hours-aware threshold
    calculation, and breach severity classification. Each mutation is
    tracked with SHA-256 provenance hashing for complete audit trails.

    The engine stores all state in-memory with thread-safe access for
    high-performance SLA evaluation during freshness monitoring runs.
    For persistent storage, the service layer serialises state to
    PostgreSQL.

    SLA Evaluation Logic:
        - age_hours <= warning_hours: COMPLIANT
        - warning_hours < age_hours <= critical_hours: WARNING
        - critical_hours < age_hours <= critical_hours * 2: BREACHED
        - age_hours > critical_hours * 2: CRITICAL

    Breach Severity Mapping:
        - age_hours <= warning_hours * 1.5: LOW
        - age_hours <= critical_hours: MEDIUM
        - age_hours <= critical_hours * 1.5: HIGH
        - age_hours > critical_hours * 1.5: CRITICAL

    Business Hours:
        - Monday-Friday, 09:00-18:00 UTC
        - If business_hours_only=True, only business hours count

    Attributes:
        _slas: SLA definitions keyed by SLA ID.
        _dataset_sla_map: Dataset ID to SLA ID mapping.
        _templates: SLA templates keyed by template ID.
        _provenance: SHA-256 provenance tracker.
        _lock: Thread-safety lock for concurrent access.

    Example:
        >>> engine = SLADefinitionEngine()
        >>> sla = engine.create_sla(
        ...     dataset_id="ds-001",
        ...     warning_hours=24.0,
        ...     critical_hours=72.0,
        ... )
        >>> status = engine.evaluate_sla(sla.id, age_hours=36.0)
        >>> assert status == SLAStatus.WARNING
    """

    def __init__(self) -> None:
        """Initialize SLADefinitionEngine with empty state."""
        self._slas: Dict[str, SLADefinition] = {}
        self._dataset_sla_map: Dict[str, str] = {}
        self._templates: Dict[str, SLATemplate] = {}
        self._provenance = ProvenanceTracker()
        self._lock = threading.Lock()

        logger.info("SLADefinitionEngine initialized")

    # ------------------------------------------------------------------
    # 1. create_sla
    # ------------------------------------------------------------------

    def create_sla(
        self,
        dataset_id: str,
        warning_hours: float = 24.0,
        critical_hours: float = 72.0,
        breach_severity: str = "medium",
        escalation_policy: str = "notify",
        business_hours_only: bool = False,
    ) -> SLADefinition:
        """Create a new SLA definition for a dataset.

        Each dataset can have at most one active SLA. If the dataset
        already has an SLA, a ValueError is raised. The warning_hours
        threshold must be strictly less than critical_hours.

        Args:
            dataset_id: Unique identifier of the dataset to monitor.
            warning_hours: Hours after last update before a warning-level
                SLA alert triggers. Must be > 0. Defaults to 24.0.
            critical_hours: Hours after last update before a breach-level
                SLA alert triggers. Must be > warning_hours. Defaults
                to 72.0.
            breach_severity: Default breach severity classification.
                One of: low, medium, high, critical. Defaults to
                "medium".
            escalation_policy: Escalation policy when SLA is breached.
                One of: none, notify, escalate, page. Defaults to
                "notify".
            business_hours_only: Whether to count only business hours
                (Mon-Fri, 09:00-18:00 UTC) for SLA evaluation.
                Defaults to False.

        Returns:
            SLADefinition with populated id, timestamps, and provenance
            hash.

        Raises:
            ValueError: If dataset_id is empty, warning_hours <= 0,
                critical_hours <= warning_hours, dataset already has an
                SLA, or breach_severity/escalation_policy are invalid.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> sla = engine.create_sla(
            ...     dataset_id="ds-001",
            ...     warning_hours=24.0,
            ...     critical_hours=72.0,
            ...     breach_severity="high",
            ...     escalation_policy="escalate",
            ... )
            >>> assert sla.warning_hours == 24.0
        """
        start = time.time()

        # Validate dataset_id
        if not dataset_id or not dataset_id.strip():
            raise ValueError("dataset_id must not be empty")

        dataset_id = dataset_id.strip()

        # Validate thresholds
        if warning_hours <= 0.0:
            raise ValueError(
                f"warning_hours must be > 0, got {warning_hours}"
            )
        if critical_hours <= warning_hours:
            raise ValueError(
                f"critical_hours ({critical_hours}) must be > "
                f"warning_hours ({warning_hours})"
            )

        # Validate enums
        breach_severity = _resolve_breach_severity(breach_severity)
        escalation_policy = _resolve_escalation_policy(escalation_policy)

        # Check for existing SLA on this dataset
        with self._lock:
            if dataset_id in self._dataset_sla_map:
                existing_sla_id = self._dataset_sla_map[dataset_id]
                raise ValueError(
                    f"Dataset '{dataset_id}' already has an SLA: "
                    f"{existing_sla_id}"
                )

        # Build SLADefinition
        sla_id = _generate_id()
        now = _utcnow()

        sla = SLADefinition(
            id=sla_id,
            dataset_id=dataset_id,
            warning_hours=warning_hours,
            critical_hours=critical_hours,
            breach_severity=breach_severity,
            escalation_policy=escalation_policy,
            business_hours_only=business_hours_only,
            created_at=now,
            updated_at=now,
        )

        # Compute provenance hash
        input_hash = self._provenance.hash_record({
            "dataset_id": dataset_id,
            "warning_hours": warning_hours,
            "critical_hours": critical_hours,
            "breach_severity": breach_severity,
            "escalation_policy": escalation_policy,
            "business_hours_only": business_hours_only,
        })
        output_hash = self._provenance.hash_record({
            "sla_id": sla_id,
            "dataset_id": dataset_id,
            "created_at": now.isoformat(),
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="create_sla",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "sla_id": sla_id,
                "dataset_id": dataset_id,
                "warning_hours": warning_hours,
                "critical_hours": critical_hours,
            },
        )
        sla.provenance_hash = provenance_hash

        # Store under lock
        with self._lock:
            # Double-check in case of race condition
            if dataset_id in self._dataset_sla_map:
                existing_sla_id = self._dataset_sla_map[dataset_id]
                raise ValueError(
                    f"Dataset '{dataset_id}' already has an SLA: "
                    f"{existing_sla_id}"
                )
            self._slas[sla_id] = sla
            self._dataset_sla_map[dataset_id] = sla_id

        elapsed = time.time() - start
        inc_jobs_processed("sla_created")
        observe_duration("create_sla", elapsed)

        logger.info(
            "SLA created: id=%s, dataset=%s, warning=%.1fh, "
            "critical=%.1fh, severity=%s, policy=%s, biz_hours=%s, "
            "%.3fms",
            sla_id[:12], dataset_id, warning_hours, critical_hours,
            breach_severity, escalation_policy, business_hours_only,
            elapsed * 1000,
        )
        return sla

    # ------------------------------------------------------------------
    # 2. get_sla
    # ------------------------------------------------------------------

    def get_sla(self, sla_id: str) -> Optional[SLADefinition]:
        """Retrieve an SLA definition by its ID.

        Args:
            sla_id: Unique identifier of the SLA to retrieve.

        Returns:
            SLADefinition if found, None otherwise.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> sla = engine.create_sla(dataset_id="ds-001")
            >>> retrieved = engine.get_sla(sla.id)
            >>> assert retrieved is not None
            >>> assert retrieved.dataset_id == "ds-001"
        """
        with self._lock:
            sla = self._slas.get(sla_id)

        if sla is None:
            logger.debug("SLA not found: id=%s", sla_id[:12])
            return None

        logger.debug(
            "SLA retrieved: id=%s, dataset=%s",
            sla_id[:12], sla.dataset_id,
        )
        return sla

    # ------------------------------------------------------------------
    # 3. get_sla_for_dataset
    # ------------------------------------------------------------------

    def get_sla_for_dataset(
        self,
        dataset_id: str,
    ) -> Optional[SLADefinition]:
        """Retrieve the SLA definition for a specific dataset.

        Args:
            dataset_id: Unique identifier of the dataset.

        Returns:
            SLADefinition if the dataset has an SLA, None otherwise.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> engine.create_sla(dataset_id="ds-001", warning_hours=12.0)
            >>> sla = engine.get_sla_for_dataset("ds-001")
            >>> assert sla is not None
            >>> assert sla.warning_hours == 12.0
        """
        with self._lock:
            sla_id = self._dataset_sla_map.get(dataset_id)
            if sla_id is None:
                logger.debug(
                    "No SLA for dataset: %s", dataset_id,
                )
                return None
            sla = self._slas.get(sla_id)

        if sla is None:
            logger.warning(
                "SLA map references missing SLA: dataset=%s, sla_id=%s",
                dataset_id, sla_id[:12],
            )
            return None

        logger.debug(
            "SLA for dataset retrieved: dataset=%s, sla_id=%s",
            dataset_id, sla.id[:12],
        )
        return sla

    # ------------------------------------------------------------------
    # 4. list_slas
    # ------------------------------------------------------------------

    def list_slas(
        self,
        dataset_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SLADefinition]:
        """List SLA definitions with optional filtering and pagination.

        Args:
            dataset_id: If provided, filter to SLAs for this dataset
                only. None returns all SLAs.
            limit: Maximum number of SLAs to return. Defaults to 100.
            offset: Number of SLAs to skip before returning results.
                Defaults to 0.

        Returns:
            List of SLADefinition objects sorted by creation time
            descending (newest first), applying offset and limit.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> engine.create_sla(dataset_id="ds-001")
            >>> engine.create_sla(dataset_id="ds-002")
            >>> all_slas = engine.list_slas()
            >>> assert len(all_slas) == 2
        """
        with self._lock:
            slas = list(self._slas.values())

        # Filter by dataset_id if provided
        if dataset_id is not None:
            dataset_id = dataset_id.strip()
            slas = [s for s in slas if s.dataset_id == dataset_id]

        # Sort by created_at descending (newest first)
        slas.sort(
            key=lambda s: s.created_at if s.created_at else _utcnow(),
            reverse=True,
        )

        # Apply pagination
        slas = slas[offset:offset + limit]

        logger.debug(
            "Listed SLAs: total=%d, dataset_filter=%s, limit=%d, "
            "offset=%d",
            len(slas), dataset_id, limit, offset,
        )
        return slas

    # ------------------------------------------------------------------
    # 5. update_sla
    # ------------------------------------------------------------------

    def update_sla(self, sla_id: str, **updates: Any) -> SLADefinition:
        """Update specified fields of an existing SLA definition.

        Only the fields provided in ``updates`` are modified. All other
        fields retain their current values. A new provenance hash is
        computed for the update operation.

        Supported fields:
            warning_hours, critical_hours, breach_severity,
            escalation_policy, business_hours_only.

        Args:
            sla_id: ID of the SLA to update.
            **updates: Field-name/value pairs to update.

        Returns:
            Updated SLADefinition with new provenance hash.

        Raises:
            KeyError: If sla_id is not found.
            ValueError: If updated values fail validation or if an
                unsupported field is provided.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> sla = engine.create_sla(
            ...     dataset_id="ds-001", warning_hours=24.0,
            ... )
            >>> updated = engine.update_sla(
            ...     sla.id, warning_hours=12.0,
            ... )
            >>> assert updated.warning_hours == 12.0
        """
        start = time.time()

        with self._lock:
            if sla_id not in self._slas:
                raise KeyError(f"SLA not found: {sla_id}")
            sla = self._slas[sla_id]

        # Validate updatable fields
        updatable = {
            "warning_hours", "critical_hours", "breach_severity",
            "escalation_policy", "business_hours_only",
        }
        invalid_keys = set(updates.keys()) - updatable
        if invalid_keys:
            raise ValueError(
                f"Cannot update fields: {sorted(invalid_keys)}. "
                f"Updatable: {sorted(updatable)}"
            )

        # Track changes
        changes: Dict[str, Any] = {}

        # Apply updates
        for key, value in updates.items():
            old_val = getattr(sla, key)

            if key == "breach_severity":
                value = _resolve_breach_severity(value)
            elif key == "escalation_policy":
                value = _resolve_escalation_policy(value)
            elif key == "warning_hours":
                if value <= 0.0:
                    raise ValueError(
                        f"warning_hours must be > 0, got {value}"
                    )
            elif key == "critical_hours":
                if value <= 0.0:
                    raise ValueError(
                        f"critical_hours must be > 0, got {value}"
                    )

            if old_val != value:
                setattr(sla, key, value)
                changes[key] = {"old": str(old_val), "new": str(value)}

        # Cross-validate warning < critical after all updates
        if sla.critical_hours <= sla.warning_hours:
            # Revert changes and raise
            for key, change in changes.items():
                setattr(sla, key, type(getattr(sla, key))(change["old"])
                        if key in ("warning_hours", "critical_hours")
                        else change["old"])
            raise ValueError(
                f"critical_hours ({sla.critical_hours}) must be > "
                f"warning_hours ({sla.warning_hours}) after update"
            )

        # Update timestamp
        sla.updated_at = _utcnow()

        # Compute provenance
        input_hash = self._provenance.hash_record({
            "sla_id": sla_id,
            "changes": {k: v["new"] for k, v in changes.items()},
        })
        output_hash = self._provenance.hash_record({
            "sla_id": sla_id,
            "warning_hours": sla.warning_hours,
            "critical_hours": sla.critical_hours,
            "updated_at": sla.updated_at.isoformat(),
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="update_sla",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "sla_id": sla_id,
                "changes": list(changes.keys()),
            },
        )
        sla.provenance_hash = provenance_hash

        with self._lock:
            self._slas[sla_id] = sla

        elapsed = time.time() - start
        inc_jobs_processed("sla_updated")
        observe_duration("update_sla", elapsed)

        logger.info(
            "SLA updated: id=%s, fields=%s, %.3fms",
            sla_id[:12], list(changes.keys()), elapsed * 1000,
        )
        return sla

    # ------------------------------------------------------------------
    # 6. delete_sla
    # ------------------------------------------------------------------

    def delete_sla(self, sla_id: str) -> bool:
        """Delete an SLA definition by its ID.

        Removes the SLA from both the SLA store and the dataset-to-SLA
        mapping. Records a provenance entry for the deletion.

        Args:
            sla_id: ID of the SLA to delete.

        Returns:
            True if the SLA was found and deleted, False if not found.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> sla = engine.create_sla(dataset_id="ds-001")
            >>> assert engine.delete_sla(sla.id) is True
            >>> assert engine.get_sla(sla.id) is None
        """
        start = time.time()

        with self._lock:
            if sla_id not in self._slas:
                logger.debug("Cannot delete: SLA not found id=%s", sla_id[:12])
                return False

            sla = self._slas.pop(sla_id)
            dataset_id = sla.dataset_id

            # Remove dataset mapping
            if self._dataset_sla_map.get(dataset_id) == sla_id:
                del self._dataset_sla_map[dataset_id]

        # Record provenance
        input_hash = self._provenance.hash_record({
            "sla_id": sla_id,
            "dataset_id": dataset_id,
        })
        output_hash = self._provenance.hash_record({
            "sla_id": sla_id,
            "deleted": True,
        })
        self._provenance.add_to_chain(
            operation="delete_sla",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "sla_id": sla_id,
                "dataset_id": dataset_id,
            },
        )

        elapsed = time.time() - start
        inc_jobs_processed("sla_deleted")
        observe_duration("delete_sla", elapsed)

        logger.info(
            "SLA deleted: id=%s, dataset=%s, %.3fms",
            sla_id[:12], dataset_id, elapsed * 1000,
        )
        return True

    # ------------------------------------------------------------------
    # 7. evaluate_sla
    # ------------------------------------------------------------------

    def evaluate_sla(
        self,
        sla_id: str,
        age_hours: float,
    ) -> SLAStatus:
        """Evaluate an SLA against the current data age.

        Determines the SLA compliance status by comparing the data age
        (in hours) against the configured warning and critical thresholds.

        Evaluation logic (deterministic, zero-hallucination):
            - age_hours <= warning_hours: COMPLIANT
            - warning_hours < age_hours <= critical_hours: WARNING
            - critical_hours < age_hours <= critical_hours * 2: BREACHED
            - age_hours > critical_hours * 2: CRITICAL

        Args:
            sla_id: ID of the SLA to evaluate.
            age_hours: Current age of the dataset in hours (must be >= 0).

        Returns:
            SLAStatus enum value indicating compliance level.

        Raises:
            KeyError: If sla_id is not found.
            ValueError: If age_hours is negative.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> sla = engine.create_sla(
            ...     dataset_id="ds-001",
            ...     warning_hours=24.0,
            ...     critical_hours=72.0,
            ... )
            >>> engine.evaluate_sla(sla.id, 12.0)
            <SLAStatus.COMPLIANT: 'compliant'>
            >>> engine.evaluate_sla(sla.id, 48.0)
            <SLAStatus.WARNING: 'warning'>
        """
        start = time.time()

        if age_hours < 0.0:
            raise ValueError(
                f"age_hours must be >= 0, got {age_hours}"
            )

        with self._lock:
            if sla_id not in self._slas:
                raise KeyError(f"SLA not found: {sla_id}")
            sla = self._slas[sla_id]

        status = self._compute_sla_status(
            age_hours=age_hours,
            warning_hours=sla.warning_hours,
            critical_hours=sla.critical_hours,
        )

        elapsed = time.time() - start
        inc_jobs_processed("sla_evaluated")
        observe_duration("evaluate_sla", elapsed)

        logger.debug(
            "SLA evaluated: id=%s, age=%.1fh, warning=%.1fh, "
            "critical=%.1fh, status=%s, %.3fms",
            sla_id[:12], age_hours, sla.warning_hours,
            sla.critical_hours, status.value, elapsed * 1000,
        )
        return status

    # ------------------------------------------------------------------
    # 8. evaluate_dataset_sla
    # ------------------------------------------------------------------

    def evaluate_dataset_sla(
        self,
        dataset_id: str,
        age_hours: float,
    ) -> SLAStatus:
        """Evaluate the SLA for a dataset by its dataset ID.

        Convenience method that looks up the SLA for the given dataset
        and delegates to ``evaluate_sla``.

        Args:
            dataset_id: Unique identifier of the dataset.
            age_hours: Current age of the dataset in hours.

        Returns:
            SLAStatus enum value indicating compliance level.

        Raises:
            KeyError: If no SLA exists for the dataset.
            ValueError: If age_hours is negative.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> engine.create_sla(
            ...     dataset_id="ds-001",
            ...     warning_hours=24.0,
            ...     critical_hours=72.0,
            ... )
            >>> engine.evaluate_dataset_sla("ds-001", 36.0)
            <SLAStatus.WARNING: 'warning'>
        """
        with self._lock:
            sla_id = self._dataset_sla_map.get(dataset_id)

        if sla_id is None:
            raise KeyError(
                f"No SLA defined for dataset: {dataset_id}"
            )

        return self.evaluate_sla(sla_id, age_hours)

    # ------------------------------------------------------------------
    # 9. create_template
    # ------------------------------------------------------------------

    def create_template(
        self,
        name: str,
        warning_hours: float = 24.0,
        critical_hours: float = 72.0,
        breach_severity: str = "medium",
        escalation_policy: str = "notify",
    ) -> SLATemplate:
        """Create a reusable SLA template.

        Templates define default SLA parameters that can be applied to
        multiple datasets via ``apply_template``. Template names must
        be unique.

        Args:
            name: Human-readable name for the template (must be
                non-empty and unique).
            warning_hours: Default warning threshold hours. Must be > 0.
                Defaults to 24.0.
            critical_hours: Default critical threshold hours. Must be >
                warning_hours. Defaults to 72.0.
            breach_severity: Default breach severity. One of: low,
                medium, high, critical. Defaults to "medium".
            escalation_policy: Default escalation policy. One of: none,
                notify, escalate, page. Defaults to "notify".

        Returns:
            SLATemplate with populated id and creation timestamp.

        Raises:
            ValueError: If name is empty, thresholds are invalid, or
                a template with the same name already exists.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> tmpl = engine.create_template(
            ...     name="Daily Dataset SLA",
            ...     warning_hours=24.0,
            ...     critical_hours=48.0,
            ... )
            >>> assert tmpl.name == "Daily Dataset SLA"
        """
        start = time.time()

        # Validate name
        if not name or not name.strip():
            raise ValueError("Template name must not be empty")

        name = name.strip()

        # Validate thresholds
        if warning_hours <= 0.0:
            raise ValueError(
                f"warning_hours must be > 0, got {warning_hours}"
            )
        if critical_hours <= warning_hours:
            raise ValueError(
                f"critical_hours ({critical_hours}) must be > "
                f"warning_hours ({warning_hours})"
            )

        # Validate enums
        breach_severity = _resolve_breach_severity(breach_severity)
        escalation_policy = _resolve_escalation_policy(escalation_policy)

        # Check for duplicate name
        with self._lock:
            for existing in self._templates.values():
                if existing.name == name:
                    raise ValueError(
                        f"Template with name '{name}' already exists: "
                        f"{existing.id}"
                    )

        template_id = _generate_id()

        template = SLATemplate(
            id=template_id,
            name=name,
            warning_hours=warning_hours,
            critical_hours=critical_hours,
            breach_severity=breach_severity,
            escalation_policy=escalation_policy,
            created_at=_utcnow(),
        )

        # Provenance
        input_hash = self._provenance.hash_record({
            "name": name,
            "warning_hours": warning_hours,
            "critical_hours": critical_hours,
        })
        output_hash = self._provenance.hash_record({
            "template_id": template_id,
            "created_at": template.created_at.isoformat(),
        })
        self._provenance.add_to_chain(
            operation="create_template",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "template_id": template_id,
                "name": name,
            },
        )

        with self._lock:
            self._templates[template_id] = template

        elapsed = time.time() - start
        inc_jobs_processed("template_created")
        observe_duration("create_template", elapsed)

        logger.info(
            "SLA template created: id=%s, name=%s, warning=%.1fh, "
            "critical=%.1fh, %.3fms",
            template_id[:12], name, warning_hours, critical_hours,
            elapsed * 1000,
        )
        return template

    # ------------------------------------------------------------------
    # 10. apply_template
    # ------------------------------------------------------------------

    def apply_template(
        self,
        template_id: str,
        dataset_ids: List[str],
    ) -> List[SLADefinition]:
        """Apply an SLA template to one or more datasets.

        Creates individual SLA definitions for each dataset using the
        template's default values. Datasets that already have an SLA
        are skipped with a warning.

        Args:
            template_id: ID of the template to apply.
            dataset_ids: List of dataset IDs to create SLAs for.

        Returns:
            List of newly created SLADefinition objects (one per
            dataset that did not already have an SLA).

        Raises:
            KeyError: If template_id is not found.
            ValueError: If dataset_ids is empty.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> tmpl = engine.create_template(
            ...     name="Standard",
            ...     warning_hours=24.0,
            ...     critical_hours=72.0,
            ... )
            >>> slas = engine.apply_template(
            ...     tmpl.id, ["ds-001", "ds-002", "ds-003"],
            ... )
            >>> assert len(slas) == 3
        """
        start = time.time()

        with self._lock:
            if template_id not in self._templates:
                raise KeyError(f"Template not found: {template_id}")
            template = self._templates[template_id]

        if not dataset_ids:
            raise ValueError("dataset_ids must not be empty")

        created_slas: List[SLADefinition] = []

        for ds_id in dataset_ids:
            try:
                sla = self.create_sla(
                    dataset_id=ds_id,
                    warning_hours=template.warning_hours,
                    critical_hours=template.critical_hours,
                    breach_severity=template.breach_severity,
                    escalation_policy=template.escalation_policy,
                )
                created_slas.append(sla)
            except ValueError as exc:
                logger.warning(
                    "Skipping dataset '%s' during template apply: %s",
                    ds_id, str(exc),
                )

        # Record provenance for the batch operation
        input_hash = self._provenance.hash_record({
            "template_id": template_id,
            "dataset_ids": dataset_ids,
        })
        output_hash = self._provenance.hash_record({
            "created_count": len(created_slas),
            "sla_ids": [s.id for s in created_slas],
        })
        self._provenance.add_to_chain(
            operation="apply_template",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "template_id": template_id,
                "requested": len(dataset_ids),
                "created": len(created_slas),
            },
        )

        elapsed = time.time() - start
        inc_jobs_processed("template_applied")
        observe_duration("apply_template", elapsed)

        logger.info(
            "Template applied: template=%s, requested=%d, created=%d, "
            "%.3fms",
            template_id[:12], len(dataset_ids), len(created_slas),
            elapsed * 1000,
        )
        return created_slas

    # ------------------------------------------------------------------
    # 11. list_templates
    # ------------------------------------------------------------------

    def list_templates(self) -> List[SLATemplate]:
        """List all registered SLA templates.

        Returns:
            List of SLATemplate objects sorted by name ascending.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> engine.create_template(name="Daily", warning_hours=24.0)
            >>> engine.create_template(name="Weekly", warning_hours=168.0)
            >>> templates = engine.list_templates()
            >>> assert len(templates) == 2
            >>> assert templates[0].name == "Daily"
        """
        with self._lock:
            templates = list(self._templates.values())

        templates.sort(key=lambda t: t.name)

        logger.debug("Listed templates: total=%d", len(templates))
        return templates

    # ------------------------------------------------------------------
    # 12. get_breach_severity_for_age
    # ------------------------------------------------------------------

    def get_breach_severity_for_age(
        self,
        sla_id: str,
        age_hours: float,
    ) -> BreachSeverity:
        """Determine breach severity for a given data age.

        Maps the data age to a breach severity level using the SLA's
        warning and critical thresholds. This is independent of the
        SLA's default breach_severity field -- it computes severity
        purely from the age relative to thresholds.

        Severity mapping (deterministic, zero-hallucination):
            - age_hours <= warning_hours * 1.5: LOW
            - age_hours <= critical_hours: MEDIUM
            - age_hours <= critical_hours * 1.5: HIGH
            - age_hours > critical_hours * 1.5: CRITICAL

        Args:
            sla_id: ID of the SLA to use for threshold lookup.
            age_hours: Current age of the dataset in hours.

        Returns:
            BreachSeverity enum value.

        Raises:
            KeyError: If sla_id is not found.
            ValueError: If age_hours is negative.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> sla = engine.create_sla(
            ...     dataset_id="ds-001",
            ...     warning_hours=24.0,
            ...     critical_hours=72.0,
            ... )
            >>> engine.get_breach_severity_for_age(sla.id, 30.0)
            <BreachSeverity.LOW: 'low'>
            >>> engine.get_breach_severity_for_age(sla.id, 60.0)
            <BreachSeverity.MEDIUM: 'medium'>
        """
        if age_hours < 0.0:
            raise ValueError(
                f"age_hours must be >= 0, got {age_hours}"
            )

        with self._lock:
            if sla_id not in self._slas:
                raise KeyError(f"SLA not found: {sla_id}")
            sla = self._slas[sla_id]

        return self._compute_breach_severity(
            age_hours=age_hours,
            warning_hours=sla.warning_hours,
            critical_hours=sla.critical_hours,
        )

    # ------------------------------------------------------------------
    # 13. is_business_hours
    # ------------------------------------------------------------------

    def is_business_hours(
        self,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Check whether a given timestamp falls within business hours.

        Business hours are defined as Monday through Friday (weekday
        0-4), 09:00-18:00 UTC. The check is inclusive of 09:00 and
        exclusive of 18:00 (i.e., hour 17 is the last business hour).

        Args:
            timestamp: UTC datetime to check. If None, uses current
                UTC time.

        Returns:
            True if the timestamp is within business hours, False
            otherwise.

        Example:
            >>> from datetime import datetime, timezone
            >>> engine = SLADefinitionEngine()
            >>> # Monday 10:00 UTC
            >>> dt = datetime(2026, 2, 16, 10, 0, tzinfo=timezone.utc)
            >>> engine.is_business_hours(dt)
            True
            >>> # Saturday 10:00 UTC
            >>> dt = datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc)
            >>> engine.is_business_hours(dt)
            False
        """
        if timestamp is None:
            timestamp = _utcnow()

        weekday = timestamp.weekday()
        hour = timestamp.hour

        is_biz = (
            weekday in _BUSINESS_DAYS
            and _BUSINESS_HOUR_START <= hour < _BUSINESS_HOUR_END
        )

        logger.debug(
            "Business hours check: ts=%s, weekday=%d, hour=%d, "
            "result=%s",
            timestamp.isoformat(), weekday, hour, is_biz,
        )
        return is_biz

    # ------------------------------------------------------------------
    # 14. get_effective_sla_hours
    # ------------------------------------------------------------------

    def get_effective_sla_hours(
        self,
        sla_id: str,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Get the effective SLA hours adjusted for business hours.

        If the SLA has business_hours_only=True, this method returns
        the critical_hours threshold scaled by the ratio of business
        hours to total hours (9 business hours per 24-hour day,
        5 business days per 7-day week). This expands the calendar
        deadline to account for non-business hours.

        If business_hours_only=False, returns the critical_hours
        unchanged.

        The scaling formula is:
            effective = critical_hours * (24 / 9) * (7 / 5)

        This converts business-hour thresholds into calendar-hour
        equivalents.

        Args:
            sla_id: ID of the SLA to look up.
            current_time: Optional current UTC datetime (for future
                use and logging). If None, uses current UTC time.

        Returns:
            Effective SLA hours (calendar hours).

        Raises:
            KeyError: If sla_id is not found.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> sla = engine.create_sla(
            ...     dataset_id="ds-001",
            ...     critical_hours=9.0,
            ...     business_hours_only=True,
            ... )
            >>> effective = engine.get_effective_sla_hours(sla.id)
            >>> # 9.0 * (24/9) * (7/5) = 33.6
            >>> assert abs(effective - 33.6) < 0.01
        """
        if current_time is None:
            current_time = _utcnow()

        with self._lock:
            if sla_id not in self._slas:
                raise KeyError(f"SLA not found: {sla_id}")
            sla = self._slas[sla_id]

        if not sla.business_hours_only:
            logger.debug(
                "Effective SLA hours (24/7): id=%s, hours=%.1f",
                sla_id[:12], sla.critical_hours,
            )
            return sla.critical_hours

        # Scale business hours to calendar hours
        # Business day = 9 hours (09:00-18:00), calendar day = 24 hours
        # Business week = 5 days, calendar week = 7 days
        calendar_factor = (24.0 / _BUSINESS_HOURS_PER_DAY) * (7.0 / 5.0)
        effective = sla.critical_hours * calendar_factor

        logger.debug(
            "Effective SLA hours (biz): id=%s, critical=%.1f, "
            "factor=%.3f, effective=%.1f",
            sla_id[:12], sla.critical_hours, calendar_factor, effective,
        )
        return effective

    # ------------------------------------------------------------------
    # 15. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the SLA definition engine.

        Returns a snapshot of the engine state including total counts,
        threshold distributions, business hours adoption, and
        provenance chain length.

        Returns:
            Dictionary with keys:
                - total_slas: Total number of SLA definitions.
                - total_templates: Total number of SLA templates.
                - total_datasets_covered: Number of datasets with SLAs.
                - business_hours_slas: Count of SLAs with
                    business_hours_only=True.
                - avg_warning_hours: Average warning threshold (0.0 if
                    no SLAs).
                - avg_critical_hours: Average critical threshold (0.0
                    if no SLAs).
                - min_warning_hours: Minimum warning threshold (0.0 if
                    no SLAs).
                - max_critical_hours: Maximum critical threshold (0.0
                    if no SLAs).
                - escalation_policies: Distribution of escalation
                    policies as {policy: count}.
                - breach_severities: Distribution of breach severities
                    as {severity: count}.
                - provenance_entries: Number of provenance chain
                    entries.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> engine.create_sla(dataset_id="ds-001", warning_hours=12.0)
            >>> engine.create_sla(dataset_id="ds-002", warning_hours=24.0)
            >>> stats = engine.get_statistics()
            >>> assert stats["total_slas"] == 2
        """
        with self._lock:
            slas = list(self._slas.values())
            total_templates = len(self._templates)
            total_mapped = len(self._dataset_sla_map)

        total_slas = len(slas)

        if total_slas == 0:
            return {
                "total_slas": 0,
                "total_templates": total_templates,
                "total_datasets_covered": total_mapped,
                "business_hours_slas": 0,
                "avg_warning_hours": 0.0,
                "avg_critical_hours": 0.0,
                "min_warning_hours": 0.0,
                "max_critical_hours": 0.0,
                "escalation_policies": {},
                "breach_severities": {},
                "provenance_entries": self._provenance.entry_count,
            }

        warning_hours_list = [s.warning_hours for s in slas]
        critical_hours_list = [s.critical_hours for s in slas]
        biz_hours_count = sum(1 for s in slas if s.business_hours_only)

        # Escalation policy distribution
        policy_dist: Dict[str, int] = {}
        for s in slas:
            policy = (
                s.escalation_policy.value
                if isinstance(s.escalation_policy, Enum)
                else str(s.escalation_policy)
            )
            policy_dist[policy] = policy_dist.get(policy, 0) + 1

        # Breach severity distribution
        severity_dist: Dict[str, int] = {}
        for s in slas:
            severity = (
                s.breach_severity.value
                if isinstance(s.breach_severity, Enum)
                else str(s.breach_severity)
            )
            severity_dist[severity] = severity_dist.get(severity, 0) + 1

        stats = {
            "total_slas": total_slas,
            "total_templates": total_templates,
            "total_datasets_covered": total_mapped,
            "business_hours_slas": biz_hours_count,
            "avg_warning_hours": sum(warning_hours_list) / total_slas,
            "avg_critical_hours": sum(critical_hours_list) / total_slas,
            "min_warning_hours": min(warning_hours_list),
            "max_critical_hours": max(critical_hours_list),
            "escalation_policies": policy_dist,
            "breach_severities": severity_dist,
            "provenance_entries": self._provenance.entry_count,
        }

        logger.debug(
            "Statistics: slas=%d, templates=%d, datasets=%d, "
            "biz_hours=%d",
            total_slas, total_templates, total_mapped, biz_hours_count,
        )
        return stats

    # ------------------------------------------------------------------
    # 16. reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all engine state to empty.

        Clears all SLA definitions, dataset mappings, templates, and
        reinitializes the provenance tracker. Primarily used for
        testing.

        Example:
            >>> engine = SLADefinitionEngine()
            >>> engine.create_sla(dataset_id="ds-001")
            >>> engine.reset()
            >>> assert engine.get_statistics()["total_slas"] == 0
        """
        with self._lock:
            self._slas.clear()
            self._dataset_sla_map.clear()
            self._templates.clear()
            self._provenance = ProvenanceTracker()

        logger.info("SLADefinitionEngine state reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sla_status(
        age_hours: float,
        warning_hours: float,
        critical_hours: float,
    ) -> SLAStatus:
        """Compute SLA status from age and thresholds.

        Deterministic, pure-function evaluation with no external
        dependencies. Implements the four-tier status model:

            - age <= warning: COMPLIANT
            - warning < age <= critical: WARNING
            - critical < age <= critical * 2: BREACHED
            - age > critical * 2: CRITICAL

        Args:
            age_hours: Dataset age in hours.
            warning_hours: Warning threshold in hours.
            critical_hours: Critical threshold in hours.

        Returns:
            SLAStatus enum value.
        """
        if age_hours <= warning_hours:
            return SLAStatus.COMPLIANT

        if age_hours <= critical_hours:
            return SLAStatus.WARNING

        if age_hours <= critical_hours * 2.0:
            return SLAStatus.BREACHED

        return SLAStatus.CRITICAL

    @staticmethod
    def _compute_breach_severity(
        age_hours: float,
        warning_hours: float,
        critical_hours: float,
    ) -> BreachSeverity:
        """Compute breach severity from age and thresholds.

        Deterministic, pure-function evaluation with no external
        dependencies. Implements the four-tier severity model:

            - age <= warning * 1.5: LOW
            - age <= critical: MEDIUM
            - age <= critical * 1.5: HIGH
            - age > critical * 1.5: CRITICAL

        Args:
            age_hours: Dataset age in hours.
            warning_hours: Warning threshold in hours.
            critical_hours: Critical threshold in hours.

        Returns:
            BreachSeverity enum value.
        """
        if age_hours <= warning_hours * 1.5:
            return BreachSeverity.LOW

        if age_hours <= critical_hours:
            return BreachSeverity.MEDIUM

        if age_hours <= critical_hours * 1.5:
            return BreachSeverity.HIGH

        return BreachSeverity.CRITICAL


# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = [
    "SLADefinitionEngine",
    "SLAStatus",
    "BreachSeverity",
    "EscalationPolicy",
    "SLADefinition",
    "SLATemplate",
]
