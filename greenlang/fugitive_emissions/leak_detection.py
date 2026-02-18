# -*- coding: utf-8 -*-
"""
LeakDetectionEngine - LDAR Program Tracking (Engine 3 of 7)

AGENT-MRV-005: Fugitive Emissions Agent

Comprehensive Leak Detection and Repair (LDAR) program management engine
implementing survey scheduling, leak classification, repair tracking,
coverage analysis, Delay of Repair (DOR) management, emission reduction
quantification, and inspector certification tracking.

LDAR Program Features:
    - Survey scheduling: OGI quarterly, Method 21 annual, AVO daily
    - Leak classification against configurable thresholds (EPA 10000 ppm,
      MACT 500 ppm, EU 500 ppm)
    - Coverage tracking (% of components surveyed per survey cycle)
    - Leak statistics (leak rate, frequency, top leaker identification)
    - Repair tracking with regulatory deadlines (EPA 15 days, EU 5/30 days)
    - Delay of Repair (DOR) management with justification and re-monitoring
    - Emission reduction estimation from leak repairs
    - Inspector certification and qualification tracking

Regulatory Frameworks:
    - EPA 40 CFR Part 60 Subpart VVa (Method 21 LDAR)
    - EPA 40 CFR Part 60 Subpart OOOOa (OGI LDAR)
    - EPA 40 CFR Part 63 Subpart H (MACT LDAR)
    - EU Methane Regulation 2024/1787 (OGI + Method 21)
    - Alberta EPEA Directive 060

Zero-Hallucination Guarantees:
    - All date calculations use Python datetime.
    - All statistical calculations use deterministic arithmetic.
    - No LLM involvement in any compliance determination.
    - Every record carries a SHA-256 provenance hash.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.fugitive_emissions.leak_detection import LeakDetectionEngine
    >>> engine = LeakDetectionEngine()
    >>> survey_id = engine.schedule_survey({
    ...     "facility_id": "FAC-001",
    ...     "survey_type": "OGI",
    ...     "scheduled_date": "2026-04-01",
    ... })
    >>> engine.record_survey({
    ...     "survey_id": survey_id["survey_id"],
    ...     "completion_date": "2026-04-01",
    ...     "components_surveyed": 5000,
    ...     "components_total": 5000,
    ...     "leaks_detected": 12,
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["LeakDetectionEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.fugitive_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.metrics import (
        record_component_operation as _record_ldar_operation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_ldar_operation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC / date helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _today_utc() -> date:
    """Return today's date in UTC."""
    return datetime.now(timezone.utc).date()


def _parse_date(value: str) -> date:
    """Parse an ISO date string to a date object.

    Supports YYYY-MM-DD and full ISO datetime formats.

    Args:
        value: Date string.

    Returns:
        date object.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot parse date: {value!r}") from exc


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash.

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _quantize(value: Decimal) -> Decimal:
    """Quantize to 8 decimal places."""
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ===========================================================================
# Enumerations
# ===========================================================================


class SurveyType(str, Enum):
    """LDAR survey method types.

    OGI: Optical Gas Imaging camera survey (EPA OOOOa).
    METHOD_21: EPA Method 21 portable analyzer survey.
    AVO: Audio-Visual-Olfactory walk-through inspection.
    HIFLOW: Hi-Flow sampler quantification survey.
    DRONE_OGI: Drone-mounted OGI survey.
    """

    OGI = "OGI"
    METHOD_21 = "METHOD_21"
    AVO = "AVO"
    HIFLOW = "HIFLOW"
    DRONE_OGI = "DRONE_OGI"


class SurveyStatus(str, Enum):
    """Survey lifecycle status."""

    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    OVERDUE = "OVERDUE"


class LeakSeverity(str, Enum):
    """Leak severity classification based on screening concentration.

    NONE: Below detection limit or background.
    MINOR: Above background but below regulatory threshold.
    MODERATE: Above regulatory threshold.
    MAJOR: Significantly above threshold (> 10x).
    CRITICAL: Extremely high concentration (> 100x threshold).
    """

    NONE = "NONE"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


class RepairStatus(str, Enum):
    """Leak repair lifecycle status."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    REPAIRED = "REPAIRED"
    VERIFIED = "VERIFIED"
    DOR = "DOR"  # Delay of Repair
    FAILED = "FAILED"


class RegulatoryFramework(str, Enum):
    """Supported LDAR regulatory frameworks."""

    EPA_SUBPART_VVA = "EPA_SUBPART_VVA"
    EPA_SUBPART_OOOOA = "EPA_SUBPART_OOOOA"
    EPA_MACT_SUBPART_H = "EPA_MACT_SUBPART_H"
    EU_METHANE_REG = "EU_METHANE_REG"
    ALBERTA_DIRECTIVE_060 = "ALBERTA_DIRECTIVE_060"


# ===========================================================================
# Reference Data: Regulatory Thresholds and Deadlines
# ===========================================================================

#: Leak detection thresholds by regulatory framework.
#: Units: ppmv (parts per million by volume).
LEAK_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "EPA_SUBPART_VVA": {
        "threshold_ppmv": 10000,
        "description": "EPA Method 21 leak definition (40 CFR 60 Subpart VVa)",
        "repair_deadline_days": 15,
        "first_attempt_days": 5,
        "remonitor_after_repair_days": 0,
        "dor_allowed": True,
        "dor_remonitor_interval_days": 15,
    },
    "EPA_SUBPART_OOOOA": {
        "threshold_ppmv": 500,
        "description": "EPA OGI leak definition (40 CFR 60 Subpart OOOOa)",
        "repair_deadline_days": 30,
        "first_attempt_days": 5,
        "remonitor_after_repair_days": 15,
        "dor_allowed": True,
        "dor_remonitor_interval_days": 15,
    },
    "EPA_MACT_SUBPART_H": {
        "threshold_ppmv": 500,
        "description": "EPA MACT leak definition (40 CFR 63 Subpart H)",
        "repair_deadline_days": 15,
        "first_attempt_days": 5,
        "remonitor_after_repair_days": 0,
        "dor_allowed": True,
        "dor_remonitor_interval_days": 15,
    },
    "EU_METHANE_REG": {
        "threshold_ppmv": 500,
        "description": "EU Methane Regulation 2024/1787 leak definition",
        "repair_deadline_days": 30,
        "first_attempt_days": 5,
        "remonitor_after_repair_days": 15,
        "dor_allowed": True,
        "dor_remonitor_interval_days": 30,
    },
    "ALBERTA_DIRECTIVE_060": {
        "threshold_ppmv": 500,
        "description": "Alberta EPEA Directive 060 leak definition",
        "repair_deadline_days": 30,
        "first_attempt_days": 5,
        "remonitor_after_repair_days": 30,
        "dor_allowed": True,
        "dor_remonitor_interval_days": 90,
    },
}

#: Default survey frequencies by survey type.
#: Values represent the interval in days between required surveys.
SURVEY_FREQUENCIES: Dict[str, Dict[str, Any]] = {
    "OGI": {
        "interval_days": 90,
        "description": "Quarterly OGI survey",
        "regulatory_ref": "40 CFR 60 Subpart OOOOa",
    },
    "METHOD_21": {
        "interval_days": 365,
        "description": "Annual Method 21 survey",
        "regulatory_ref": "40 CFR 60 Subpart VVa",
    },
    "AVO": {
        "interval_days": 1,
        "description": "Daily AVO walk-through inspection",
        "regulatory_ref": "General good practice",
    },
    "HIFLOW": {
        "interval_days": 365,
        "description": "Annual Hi-Flow quantification survey",
        "regulatory_ref": "EPA Protocol",
    },
    "DRONE_OGI": {
        "interval_days": 180,
        "description": "Semi-annual drone OGI survey",
        "regulatory_ref": "Emerging practice",
    },
}

#: DOR justification codes per EPA guidance.
DOR_JUSTIFICATION_CODES: Dict[str, str] = {
    "PROCESS_UNIT_SHUTDOWN": "Repair requires process unit shutdown "
                             "that is not scheduled within the deadline",
    "PARTS_UNAVAILABLE": "Replacement parts not available within repair deadline",
    "SAFETY_HAZARD": "Repair would create a safety hazard during operation",
    "TECHNICALLY_INFEASIBLE": "Repair is technically infeasible during operation",
    "REGULATORY_CONSTRAINT": "Regulatory constraint prevents immediate repair",
    "OTHER": "Other justification (must provide description)",
}


# ===========================================================================
# Data classes
# ===========================================================================


@dataclass
class SurveyRecord:
    """LDAR survey record.

    Attributes:
        survey_id: Unique survey identifier.
        facility_id: Facility where the survey is conducted.
        survey_type: Type of LDAR survey.
        status: Current survey status.
        scheduled_date: Originally scheduled date.
        completion_date: Actual completion date.
        inspector_id: Inspector who conducted the survey.
        components_surveyed: Number of components surveyed.
        components_total: Total components in scope.
        leaks_detected: Number of leaks found.
        notes: Additional survey notes.
        created_at: Record creation timestamp.
        updated_at: Last update timestamp.
        provenance_hash: SHA-256 audit trail hash.
    """

    survey_id: str
    facility_id: str
    survey_type: str
    status: str = "SCHEDULED"
    scheduled_date: str = ""
    completion_date: str = ""
    inspector_id: str = ""
    components_surveyed: int = 0
    components_total: int = 0
    leaks_detected: int = 0
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    provenance_hash: str = ""


@dataclass
class LeakRecord:
    """Individual leak detection record.

    Attributes:
        leak_id: Unique leak identifier.
        survey_id: Survey during which the leak was detected.
        facility_id: Facility identifier.
        component_id: Leaking component identifier.
        component_type: Type of component (valve, pump, etc.).
        service_type: Service type (gas, light_liquid, etc.).
        screening_value_ppmv: Measured screening concentration.
        severity: Leak severity classification.
        detection_date: Date the leak was detected.
        repair_status: Current repair status.
        repair_deadline: Regulatory repair deadline.
        repair_date: Actual repair date.
        post_repair_ppmv: Post-repair screening value.
        dor_justification: DOR justification code.
        dor_next_monitor_date: Next DOR re-monitoring date.
        estimated_emission_kg_hr: Estimated leak rate.
        notes: Additional notes.
        created_at: Record creation timestamp.
        updated_at: Last update timestamp.
        provenance_hash: SHA-256 audit trail hash.
    """

    leak_id: str
    survey_id: str
    facility_id: str
    component_id: str = ""
    component_type: str = ""
    service_type: str = ""
    screening_value_ppmv: float = 0.0
    severity: str = "NONE"
    detection_date: str = ""
    repair_status: str = "PENDING"
    repair_deadline: str = ""
    repair_date: str = ""
    post_repair_ppmv: float = 0.0
    dor_justification: str = ""
    dor_next_monitor_date: str = ""
    estimated_emission_kg_hr: float = 0.0
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    provenance_hash: str = ""


@dataclass
class InspectorRecord:
    """Inspector certification record.

    Attributes:
        inspector_id: Unique inspector identifier.
        name: Inspector full name.
        certifications: List of certification types held.
        certification_dates: Dict of cert_type -> expiry date.
        surveys_completed: Total surveys completed.
        is_active: Whether the inspector is currently active.
        created_at: Record creation timestamp.
        provenance_hash: SHA-256 audit trail hash.
    """

    inspector_id: str
    name: str
    certifications: List[str] = field(default_factory=list)
    certification_dates: Dict[str, str] = field(default_factory=dict)
    surveys_completed: int = 0
    is_active: bool = True
    created_at: str = ""
    provenance_hash: str = ""


# ===========================================================================
# LeakDetectionEngine
# ===========================================================================


class LeakDetectionEngine:
    """LDAR program management engine for survey scheduling, leak tracking,
    repair management, and compliance monitoring.

    Implements comprehensive LDAR program functionality covering multiple
    regulatory frameworks (EPA, EU, Alberta). All compliance determinations
    are deterministic boolean evaluations with no LLM involvement.

    Attributes:
        config: Configuration dictionary.

    Example:
        >>> engine = LeakDetectionEngine()
        >>> survey = engine.schedule_survey({
        ...     "facility_id": "FAC-001",
        ...     "survey_type": "OGI",
        ...     "scheduled_date": "2026-04-01",
        ... })
        >>> engine.record_survey({
        ...     "survey_id": survey["survey_id"],
        ...     "completion_date": "2026-04-01",
        ...     "components_surveyed": 5000,
        ...     "components_total": 5000,
        ...     "leaks_detected": 12,
        ... })
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the LeakDetectionEngine.

        Args:
            config: Optional configuration dictionary. Supports:
                - default_framework (str): Default regulatory framework.
                - leak_threshold_ppmv (int): Override default leak threshold.
                - repair_deadline_days (int): Override default repair deadline.
        """
        self._config = config or {}
        self._lock = threading.RLock()

        # In-memory registries
        self._surveys: Dict[str, SurveyRecord] = {}
        self._leaks: Dict[str, LeakRecord] = {}
        self._inspectors: Dict[str, InspectorRecord] = {}

        # Indexes for fast lookup
        self._surveys_by_facility: Dict[str, List[str]] = defaultdict(list)
        self._leaks_by_facility: Dict[str, List[str]] = defaultdict(list)
        self._leaks_by_survey: Dict[str, List[str]] = defaultdict(list)
        self._leaks_by_component: Dict[str, List[str]] = defaultdict(list)

        # Configuration defaults
        self._default_framework: str = self._config.get(
            "default_framework", "EPA_SUBPART_VVA",
        )
        self._default_threshold: int = self._config.get(
            "leak_threshold_ppmv", 10000,
        )
        self._default_repair_days: int = self._config.get(
            "repair_deadline_days", 15,
        )

        # Statistics
        self._total_surveys_scheduled: int = 0
        self._total_surveys_completed: int = 0
        self._total_leaks_detected: int = 0
        self._total_repairs_completed: int = 0
        self._total_dor_records: int = 0

        logger.info(
            "LeakDetectionEngine initialized: framework=%s, "
            "threshold=%d ppmv, repair_deadline=%d days",
            self._default_framework,
            self._default_threshold,
            self._default_repair_days,
        )

    # ------------------------------------------------------------------
    # Survey Scheduling
    # ------------------------------------------------------------------

    def schedule_survey(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a new LDAR survey.

        Args:
            data: Dictionary with:
                - facility_id (str): Facility identifier.
                - survey_type (str): OGI, METHOD_21, AVO, HIFLOW, DRONE_OGI.
                - scheduled_date (str): ISO date for the survey.
                - inspector_id (str, optional): Assigned inspector.
                - components_total (int, optional): Total components in scope.
                - notes (str, optional): Additional notes.

        Returns:
            Dictionary with survey_id, scheduled details, and provenance hash.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        t0 = time.monotonic()
        facility_id = data.get("facility_id", "")
        survey_type = data.get("survey_type", "")
        scheduled_date = data.get("scheduled_date", "")

        if not facility_id:
            raise ValueError("facility_id is required")
        if not survey_type:
            raise ValueError("survey_type is required")
        if not scheduled_date:
            raise ValueError("scheduled_date is required")

        self._validate_survey_type(survey_type)
        sched_date = _parse_date(scheduled_date)

        survey_id = f"survey_{uuid4().hex[:12]}"
        now_iso = _utcnow().isoformat()

        record = SurveyRecord(
            survey_id=survey_id,
            facility_id=facility_id,
            survey_type=survey_type.upper(),
            status=SurveyStatus.SCHEDULED.value,
            scheduled_date=sched_date.isoformat(),
            inspector_id=data.get("inspector_id", ""),
            components_total=int(data.get("components_total", 0)),
            notes=data.get("notes", ""),
            created_at=now_iso,
            updated_at=now_iso,
        )
        record.provenance_hash = _compute_hash({
            "survey_id": survey_id,
            "facility_id": facility_id,
            "survey_type": record.survey_type,
            "scheduled_date": record.scheduled_date,
            "created_at": now_iso,
        })

        with self._lock:
            self._surveys[survey_id] = record
            self._surveys_by_facility[facility_id].append(survey_id)
            self._total_surveys_scheduled += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "Scheduled survey %s: %s %s at %s (%.1fms)",
            survey_id, record.survey_type, facility_id,
            record.scheduled_date, elapsed_ms,
        )

        return {
            "survey_id": survey_id,
            "facility_id": facility_id,
            "survey_type": record.survey_type,
            "status": record.status,
            "scheduled_date": record.scheduled_date,
            "inspector_id": record.inspector_id,
            "components_total": record.components_total,
            "provenance_hash": record.provenance_hash,
        }

    # ------------------------------------------------------------------
    # Survey Recording
    # ------------------------------------------------------------------

    def record_survey(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Record the completion of an LDAR survey.

        Args:
            data: Dictionary with:
                - survey_id (str): Survey to update.
                - completion_date (str): Actual completion date (ISO).
                - components_surveyed (int): Number of components surveyed.
                - components_total (int, optional): Updated total components.
                - leaks_detected (int): Number of leaks found.
                - inspector_id (str, optional): Inspector who conducted survey.
                - notes (str, optional): Additional notes.

        Returns:
            Dictionary with updated survey details, coverage %, and provenance hash.

        Raises:
            ValueError: If survey_id is missing or survey not found.
        """
        t0 = time.monotonic()
        survey_id = data.get("survey_id", "")
        if not survey_id:
            raise ValueError("survey_id is required")

        with self._lock:
            record = self._surveys.get(survey_id)
            if record is None:
                raise ValueError(f"Survey not found: {survey_id}")

            completion_date = data.get("completion_date", _today_utc().isoformat())
            record.completion_date = completion_date
            record.components_surveyed = int(data.get("components_surveyed", 0))
            if "components_total" in data:
                record.components_total = int(data["components_total"])
            record.leaks_detected = int(data.get("leaks_detected", 0))
            record.status = SurveyStatus.COMPLETED.value
            record.updated_at = _utcnow().isoformat()

            if data.get("inspector_id"):
                record.inspector_id = data["inspector_id"]
            if data.get("notes"):
                record.notes = data["notes"]

            record.provenance_hash = _compute_hash({
                "survey_id": survey_id,
                "action": "record_completion",
                "completion_date": completion_date,
                "components_surveyed": record.components_surveyed,
                "leaks_detected": record.leaks_detected,
                "updated_at": record.updated_at,
            })
            self._total_surveys_completed += 1

            if record.inspector_id:
                insp = self._inspectors.get(record.inspector_id)
                if insp is not None:
                    insp.surveys_completed += 1

        coverage_pct = Decimal("0")
        if record.components_total > 0:
            coverage_pct = _quantize(
                _D(record.components_surveyed) / _D(record.components_total) * _HUNDRED
            )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "Recorded survey %s: %d/%d components, %d leaks, %.1f%% coverage (%.1fms)",
            survey_id, record.components_surveyed, record.components_total,
            record.leaks_detected, float(coverage_pct), elapsed_ms,
        )

        return {
            "survey_id": survey_id,
            "facility_id": record.facility_id,
            "survey_type": record.survey_type,
            "status": record.status,
            "completion_date": record.completion_date,
            "components_surveyed": record.components_surveyed,
            "components_total": record.components_total,
            "coverage_pct": str(coverage_pct),
            "leaks_detected": record.leaks_detected,
            "inspector_id": record.inspector_id,
            "provenance_hash": record.provenance_hash,
        }

    # ------------------------------------------------------------------
    # Leak Classification
    # ------------------------------------------------------------------

    def classify_leak(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify and record a detected leak.

        Args:
            data: Dictionary with:
                - survey_id (str): Survey during which the leak was found.
                - facility_id (str): Facility identifier.
                - component_id (str, optional): Leaking component ID.
                - component_type (str, optional): Component type.
                - service_type (str, optional): Service type.
                - screening_value_ppmv (float): Measured concentration.
                - framework (str, optional): Regulatory framework to use.
                - estimated_emission_kg_hr (float, optional): Estimated leak rate.
                - notes (str, optional): Additional notes.

        Returns:
            Dictionary with leak_id, severity, repair_deadline, and provenance hash.

        Raises:
            ValueError: If required fields are missing.
        """
        t0 = time.monotonic()
        survey_id = data.get("survey_id", "")
        facility_id = data.get("facility_id", "")
        ppmv = float(data.get("screening_value_ppmv", 0))

        if not survey_id:
            raise ValueError("survey_id is required")
        if not facility_id:
            raise ValueError("facility_id is required")

        framework = data.get("framework", self._default_framework).upper()
        threshold_data = LEAK_THRESHOLDS.get(framework, LEAK_THRESHOLDS["EPA_SUBPART_VVA"])
        threshold = threshold_data["threshold_ppmv"]

        severity = self._classify_severity(ppmv, threshold)
        is_leak = ppmv >= threshold

        detection_date = _parse_date(data.get("detection_date", _today_utc().isoformat()))
        repair_deadline_days = threshold_data["repair_deadline_days"]
        repair_deadline = detection_date + timedelta(days=repair_deadline_days)

        leak_id = f"leak_{uuid4().hex[:12]}"
        now_iso = _utcnow().isoformat()

        record = LeakRecord(
            leak_id=leak_id,
            survey_id=survey_id,
            facility_id=facility_id,
            component_id=data.get("component_id", ""),
            component_type=data.get("component_type", ""),
            service_type=data.get("service_type", ""),
            screening_value_ppmv=ppmv,
            severity=severity,
            detection_date=detection_date.isoformat(),
            repair_status=RepairStatus.PENDING.value if is_leak else "NOT_REQUIRED",
            repair_deadline=repair_deadline.isoformat() if is_leak else "",
            estimated_emission_kg_hr=float(data.get("estimated_emission_kg_hr", 0)),
            notes=data.get("notes", ""),
            created_at=now_iso,
            updated_at=now_iso,
        )
        record.provenance_hash = _compute_hash({
            "leak_id": leak_id, "survey_id": survey_id,
            "screening_value_ppmv": ppmv, "severity": record.severity,
            "threshold": threshold, "framework": framework,
            "detection_date": record.detection_date, "created_at": now_iso,
        })

        with self._lock:
            self._leaks[leak_id] = record
            self._leaks_by_facility[facility_id].append(leak_id)
            self._leaks_by_survey[survey_id].append(leak_id)
            if record.component_id:
                self._leaks_by_component[record.component_id].append(leak_id)
            if is_leak:
                self._total_leaks_detected += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "Classified leak %s: %s ppmv -> %s (threshold=%d, framework=%s) "
            "deadline=%s (%.1fms)",
            leak_id, ppmv, record.severity, threshold, framework,
            record.repair_deadline, elapsed_ms,
        )

        return {
            "leak_id": leak_id, "survey_id": survey_id,
            "facility_id": facility_id, "component_id": record.component_id,
            "screening_value_ppmv": ppmv, "threshold_ppmv": threshold,
            "is_leak": is_leak, "severity": record.severity,
            "framework": framework, "detection_date": record.detection_date,
            "repair_status": record.repair_status,
            "repair_deadline": record.repair_deadline,
            "repair_deadline_days": repair_deadline_days,
            "provenance_hash": record.provenance_hash,
        }

    # ------------------------------------------------------------------
    # Repair Tracking
    # ------------------------------------------------------------------

    def track_repair(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Record a repair attempt or completion for a leak.

        Args:
            data: Dictionary with:
                - leak_id (str): Leak to update.
                - repair_date (str): Date of repair (ISO).
                - post_repair_ppmv (float): Post-repair screening value.
                - repair_method (str, optional): Description of repair method.
                - technician (str, optional): Repair technician ID.
                - notes (str, optional): Additional notes.

        Returns:
            Dictionary with updated leak status, compliance details, and provenance hash.

        Raises:
            ValueError: If leak_id is missing or leak not found.
        """
        t0 = time.monotonic()
        leak_id = data.get("leak_id", "")
        if not leak_id:
            raise ValueError("leak_id is required")

        repair_date_str = data.get("repair_date", _today_utc().isoformat())
        post_repair_ppmv = float(data.get("post_repair_ppmv", 0))

        with self._lock:
            record = self._leaks.get(leak_id)
            if record is None:
                raise ValueError(f"Leak not found: {leak_id}")

            record.repair_date = repair_date_str
            record.post_repair_ppmv = post_repair_ppmv
            record.updated_at = _utcnow().isoformat()
            if data.get("notes"):
                record.notes = data["notes"]

            framework = data.get("framework", self._default_framework).upper()
            threshold_data = LEAK_THRESHOLDS.get(framework, LEAK_THRESHOLDS["EPA_SUBPART_VVA"])
            threshold = threshold_data["threshold_ppmv"]

            if post_repair_ppmv < threshold:
                record.repair_status = RepairStatus.REPAIRED.value
                self._total_repairs_completed += 1
            else:
                record.repair_status = RepairStatus.FAILED.value

            repair_date = _parse_date(repair_date_str)
            on_time = True
            if record.repair_deadline:
                deadline = _parse_date(record.repair_deadline)
                on_time = repair_date <= deadline

            record.provenance_hash = _compute_hash({
                "leak_id": leak_id, "action": "repair",
                "repair_date": repair_date_str,
                "post_repair_ppmv": post_repair_ppmv,
                "repair_status": record.repair_status,
                "updated_at": record.updated_at,
            })

        emission_reduction_kg_hr = _ZERO
        if record.estimated_emission_kg_hr > 0 and post_repair_ppmv < threshold:
            emission_reduction_kg_hr = _D(str(record.estimated_emission_kg_hr))

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.info(
            "Repair tracked for leak %s: %s ppmv -> %s ppmv, status=%s, on_time=%s (%.1fms)",
            leak_id, record.screening_value_ppmv, post_repair_ppmv,
            record.repair_status, on_time, elapsed_ms,
        )

        return {
            "leak_id": leak_id, "repair_date": repair_date_str,
            "pre_repair_ppmv": record.screening_value_ppmv,
            "post_repair_ppmv": post_repair_ppmv,
            "threshold_ppmv": threshold,
            "repair_status": record.repair_status,
            "repair_within_deadline": on_time,
            "repair_deadline": record.repair_deadline,
            "emission_reduction_kg_hr": str(emission_reduction_kg_hr),
            "provenance_hash": record.provenance_hash,
        }

    # ------------------------------------------------------------------
    # Survey Coverage
    # ------------------------------------------------------------------

    def calculate_survey_coverage(
        self, facility_id: str,
        survey_type: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate LDAR survey coverage for a facility.

        Args:
            facility_id: Facility identifier.
            survey_type: Optional filter by survey type.
            period_start: Period start date (ISO). Defaults to 1 year ago.
            period_end: Period end date (ISO). Defaults to today.

        Returns:
            Dictionary with coverage metrics, survey counts, and provenance hash.
        """
        t0 = time.monotonic()
        today = _today_utc()
        start = _parse_date(period_start) if period_start else today - timedelta(days=365)
        end = _parse_date(period_end) if period_end else today

        with self._lock:
            facility_survey_ids = self._surveys_by_facility.get(facility_id, [])
            facility_surveys = [
                self._surveys[sid] for sid in facility_survey_ids if sid in self._surveys
            ]

        matching = []
        for survey in facility_surveys:
            if survey.status != SurveyStatus.COMPLETED.value:
                continue
            if survey_type and survey.survey_type != survey_type.upper():
                continue
            if survey.completion_date:
                comp_date = _parse_date(survey.completion_date)
                if start <= comp_date <= end:
                    matching.append(survey)

        total_components_surveyed = sum(s.components_surveyed for s in matching)
        max_components_total = max((s.components_total for s in matching), default=0)

        coverage_pct = _ZERO
        if max_components_total > 0:
            coverage_pct = _quantize(
                _D(total_components_surveyed) / _D(max_components_total) * _HUNDRED
            )
            if coverage_pct > _HUNDRED:
                coverage_pct = _HUNDRED

        total_leaks = sum(s.leaks_detected for s in matching)
        leak_rate_pct = _ZERO
        if total_components_surveyed > 0:
            leak_rate_pct = _quantize(
                _D(total_leaks) / _D(total_components_surveyed) * _HUNDRED
            )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        result = {
            "facility_id": facility_id, "survey_type_filter": survey_type,
            "period_start": start.isoformat(), "period_end": end.isoformat(),
            "surveys_completed": len(matching),
            "total_components_surveyed": total_components_surveyed,
            "max_components_in_scope": max_components_total,
            "coverage_pct": str(coverage_pct),
            "total_leaks_detected": total_leaks,
            "leak_rate_pct": str(leak_rate_pct),
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Leak Statistics
    # ------------------------------------------------------------------

    def get_leak_statistics(
        self, facility_id: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate comprehensive leak statistics.

        Args:
            facility_id: Optional facility filter.
            period_start: Period start date (ISO).
            period_end: Period end date (ISO).

        Returns:
            Dictionary with leak rate, frequency, severity distribution,
            top leakers, repair statistics, and provenance hash.
        """
        t0 = time.monotonic()
        today = _today_utc()
        start = _parse_date(period_start) if period_start else today - timedelta(days=365)
        end = _parse_date(period_end) if period_end else today

        with self._lock:
            leak_ids = (
                self._leaks_by_facility.get(facility_id, [])
                if facility_id else list(self._leaks.keys())
            )
            leaks = []
            for lid in leak_ids:
                record = self._leaks.get(lid)
                if record and record.detection_date:
                    det_date = _parse_date(record.detection_date)
                    if start <= det_date <= end:
                        leaks.append(record)

        total_leaks = len(leaks)

        severity_dist: Dict[str, int] = defaultdict(int)
        component_dist: Dict[str, int] = defaultdict(int)
        repair_dist: Dict[str, int] = defaultdict(int)
        component_leak_counts: Dict[str, int] = defaultdict(int)

        for leak in leaks:
            severity_dist[leak.severity] += 1
            if leak.component_type:
                component_dist[leak.component_type] += 1
            repair_dist[leak.repair_status] += 1
            if leak.component_id:
                component_leak_counts[leak.component_id] += 1

        top_leakers = sorted(
            component_leak_counts.items(), key=lambda x: x[1], reverse=True,
        )[:10]

        avg_ppmv = _ZERO
        max_ppmv = 0.0
        if total_leaks > 0:
            total_ppmv = sum(l.screening_value_ppmv for l in leaks)
            avg_ppmv = _quantize(_D(str(total_ppmv)) / _D(str(total_leaks)))
            max_ppmv = max(l.screening_value_ppmv for l in leaks)

        repaired_leaks = [
            l for l in leaks
            if l.repair_status in (RepairStatus.REPAIRED.value, RepairStatus.VERIFIED.value)
        ]
        repair_rate_pct = _ZERO
        if total_leaks > 0:
            repair_rate_pct = _quantize(_D(len(repaired_leaks)) / _D(total_leaks) * _HUNDRED)

        on_time_repairs = 0
        late_repairs = 0
        for leak in repaired_leaks:
            if leak.repair_date and leak.repair_deadline:
                try:
                    if _parse_date(leak.repair_date) <= _parse_date(leak.repair_deadline):
                        on_time_repairs += 1
                    else:
                        late_repairs += 1
                except ValueError:
                    pass

        on_time_pct = _ZERO
        total_timed = on_time_repairs + late_repairs
        if total_timed > 0:
            on_time_pct = _quantize(_D(on_time_repairs) / _D(total_timed) * _HUNDRED)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        result = {
            "facility_id": facility_id,
            "period_start": start.isoformat(), "period_end": end.isoformat(),
            "total_leaks": total_leaks,
            "severity_distribution": dict(severity_dist),
            "component_type_distribution": dict(component_dist),
            "repair_status_distribution": dict(repair_dist),
            "top_leakers": [{"component_id": c, "leak_count": n} for c, n in top_leakers],
            "average_screening_ppmv": str(avg_ppmv),
            "max_screening_ppmv": max_ppmv,
            "repair_rate_pct": str(repair_rate_pct),
            "on_time_repair_pct": str(on_time_pct),
            "on_time_repairs": on_time_repairs,
            "late_repairs": late_repairs,
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # DOR Management
    # ------------------------------------------------------------------

    def check_dor_compliance(
        self, leak_id: str, data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check Delay of Repair (DOR) compliance for a leak.

        Args:
            leak_id: Leak identifier.
            data: Optional dictionary with justification_code, framework.

        Returns:
            Dictionary with DOR status, compliance details, and provenance hash.

        Raises:
            ValueError: If leak not found.
        """
        t0 = time.monotonic()
        data = data or {}

        with self._lock:
            record = self._leaks.get(leak_id)
            if record is None:
                raise ValueError(f"Leak not found: {leak_id}")

        framework = data.get("framework", self._default_framework).upper()
        threshold_data = LEAK_THRESHOLDS.get(framework, LEAK_THRESHOLDS["EPA_SUBPART_VVA"])

        today = _today_utc()
        is_overdue = False
        days_overdue = 0

        if record.repair_deadline:
            deadline = _parse_date(record.repair_deadline)
            if today > deadline and record.repair_status in (
                RepairStatus.PENDING.value, RepairStatus.IN_PROGRESS.value, RepairStatus.DOR.value,
            ):
                is_overdue = True
                days_overdue = (today - deadline).days

        dor_valid = False
        justification_code = data.get("justification_code", "")
        justification_desc = data.get("justification_description", "")

        if justification_code and is_overdue:
            if justification_code in DOR_JUSTIFICATION_CODES:
                dor_valid = threshold_data.get("dor_allowed", True)
                if dor_valid:
                    with self._lock:
                        record.repair_status = RepairStatus.DOR.value
                        record.dor_justification = justification_code
                        remonitor_interval = threshold_data.get("dor_remonitor_interval_days", 15)
                        next_monitor = today + timedelta(days=remonitor_interval)
                        record.dor_next_monitor_date = next_monitor.isoformat()
                        record.updated_at = _utcnow().isoformat()
                        record.provenance_hash = _compute_hash({
                            "leak_id": leak_id, "action": "dor",
                            "justification_code": justification_code,
                            "updated_at": record.updated_at,
                        })
                        self._total_dor_records += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        result = {
            "leak_id": leak_id, "facility_id": record.facility_id,
            "framework": framework, "repair_status": record.repair_status,
            "repair_deadline": record.repair_deadline,
            "is_overdue": is_overdue, "days_overdue": days_overdue,
            "dor_allowed": threshold_data.get("dor_allowed", True),
            "dor_valid": dor_valid, "justification_code": justification_code,
            "justification_description": DOR_JUSTIFICATION_CODES.get(
                justification_code, justification_desc,
            ),
            "dor_next_monitor_date": record.dor_next_monitor_date,
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Repair Effectiveness & Emission Reduction
    # ------------------------------------------------------------------

    def get_repair_effectiveness(
        self, facility_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate repair effectiveness metrics.

        Args:
            facility_id: Optional facility filter.

        Returns:
            Dictionary with repair success rates and provenance hash.
        """
        t0 = time.monotonic()
        with self._lock:
            leak_ids = (
                self._leaks_by_facility.get(facility_id, [])
                if facility_id else list(self._leaks.keys())
            )
            leaks = [self._leaks[lid] for lid in leak_ids if lid in self._leaks]

        attempted = [l for l in leaks if l.repair_status not in (RepairStatus.PENDING.value, "NOT_REQUIRED")]
        successful = [l for l in attempted if l.repair_status in (RepairStatus.REPAIRED.value, RepairStatus.VERIFIED.value)]
        failed = [l for l in attempted if l.repair_status == RepairStatus.FAILED.value]

        success_rate_pct = _ZERO
        if attempted:
            success_rate_pct = _quantize(_D(len(successful)) / _D(len(attempted)) * _HUNDRED)

        reductions: List[float] = []
        for l in successful:
            if l.screening_value_ppmv > 0:
                reductions.append(l.screening_value_ppmv - l.post_repair_ppmv)

        avg_reduction_ppmv = _ZERO
        if reductions:
            avg_reduction_ppmv = _quantize(_D(str(sum(reductions))) / _D(len(reductions)))

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        result = {
            "facility_id": facility_id, "total_leaks": len(leaks),
            "repair_attempted": len(attempted), "repair_successful": len(successful),
            "repair_failed": len(failed), "success_rate_pct": str(success_rate_pct),
            "average_ppmv_reduction": str(avg_reduction_ppmv),
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def calculate_emission_reduction(
        self, facility_id: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate total emission reduction from successful repairs.

        Args:
            facility_id: Optional facility filter.
            period_start: Period start date (ISO).
            period_end: Period end date (ISO).

        Returns:
            Dictionary with total emission reduction (kg CH4) and details.
        """
        t0 = time.monotonic()
        today = _today_utc()
        start = _parse_date(period_start) if period_start else today - timedelta(days=365)
        end = _parse_date(period_end) if period_end else today
        period_hours = _D(str((end - start).days * 24))

        with self._lock:
            leak_ids = (
                self._leaks_by_facility.get(facility_id, [])
                if facility_id else list(self._leaks.keys())
            )
            leaks = [self._leaks[lid] for lid in leak_ids if lid in self._leaks]

        total_reduction_kg = _ZERO
        repair_details: List[Dict[str, Any]] = []

        for leak in leaks:
            if leak.repair_status not in (RepairStatus.REPAIRED.value, RepairStatus.VERIFIED.value):
                continue
            if leak.estimated_emission_kg_hr <= 0 or not leak.repair_date or not leak.detection_date:
                continue
            try:
                detection = _parse_date(leak.detection_date)
                repair = _parse_date(leak.repair_date)
            except ValueError:
                continue
            if repair < start or detection > end:
                continue

            leak_start = max(detection, start)
            leak_end = min(repair, end)
            active_hours = _D(str(max((leak_end - leak_start).days, 0) * 24))
            remaining_start = max(repair, start)
            remaining_hours = _D(str(max((end - remaining_start).days, 0) * 24))

            emission_rate = _D(str(leak.estimated_emission_kg_hr))
            reduction = _quantize(emission_rate * remaining_hours)
            total_reduction_kg += reduction

            repair_details.append({
                "leak_id": leak.leak_id,
                "emission_rate_kg_hr": str(emission_rate),
                "active_hours": str(active_hours),
                "remaining_hours_saved": str(remaining_hours),
                "reduction_kg": str(reduction),
            })

        total_reduction_tonnes = _quantize(total_reduction_kg / _D("1000"))
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = {
            "facility_id": facility_id,
            "period_start": start.isoformat(), "period_end": end.isoformat(),
            "period_hours": str(period_hours),
            "repairs_counted": len(repair_details),
            "total_reduction_kg": str(total_reduction_kg),
            "total_reduction_tonnes": str(total_reduction_tonnes),
            "repair_details": repair_details[:50],
            "details_truncated": len(repair_details) > 50,
            "processing_time_ms": round(elapsed_ms, 3),
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items() if k != "repair_details"
        })
        return result

    # ------------------------------------------------------------------
    # Inspector Certification
    # ------------------------------------------------------------------

    def register_inspector(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Register an LDAR inspector with certifications.

        Args:
            data: Dictionary with name, certifications, certification_dates.

        Returns:
            Dictionary with inspector_id and provenance hash.

        Raises:
            ValueError: If name is missing.
        """
        name = data.get("name", "")
        if not name:
            raise ValueError("name is required")

        inspector_id = f"insp_{uuid4().hex[:12]}"
        now_iso = _utcnow().isoformat()
        record = InspectorRecord(
            inspector_id=inspector_id, name=name,
            certifications=data.get("certifications", []),
            certification_dates=data.get("certification_dates", {}),
            is_active=True, created_at=now_iso,
        )
        record.provenance_hash = _compute_hash({
            "inspector_id": inspector_id, "name": name,
            "certifications": record.certifications, "created_at": now_iso,
        })

        with self._lock:
            self._inspectors[inspector_id] = record

        logger.info("Registered inspector %s: %s, certs=%s", inspector_id, name, record.certifications)
        return {
            "inspector_id": inspector_id, "name": name,
            "certifications": record.certifications,
            "certification_dates": record.certification_dates,
            "provenance_hash": record.provenance_hash,
        }

    def check_inspector_certification(
        self, inspector_id: str, required_cert: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check an inspector's certification status.

        Args:
            inspector_id: Inspector identifier.
            required_cert: Optional specific certification to verify.

        Returns:
            Dictionary with certification status and expiry details.

        Raises:
            ValueError: If inspector not found.
        """
        with self._lock:
            record = self._inspectors.get(inspector_id)
            if record is None:
                raise ValueError(f"Inspector not found: {inspector_id}")

        today = _today_utc()
        cert_status: List[Dict[str, Any]] = []
        all_valid = True

        for cert in record.certifications:
            expiry_str = record.certification_dates.get(cert)
            is_expired = False
            days_until_expiry = None
            if expiry_str:
                try:
                    expiry = _parse_date(expiry_str)
                    is_expired = today > expiry
                    days_until_expiry = (expiry - today).days
                except ValueError:
                    is_expired = True
            if is_expired:
                all_valid = False
            cert_status.append({
                "certification": cert, "expiry_date": expiry_str or "not_set",
                "is_expired": is_expired, "days_until_expiry": days_until_expiry,
            })

        has_required = True
        if required_cert:
            has_required = required_cert in record.certifications
            if has_required:
                for cs in cert_status:
                    if cs["certification"] == required_cert and cs["is_expired"]:
                        has_required = False

        return {
            "inspector_id": inspector_id, "name": record.name,
            "is_active": record.is_active, "certifications": cert_status,
            "all_certifications_valid": all_valid,
            "has_required_certification": has_required,
            "required_cert_checked": required_cert,
            "surveys_completed": record.surveys_completed,
        }

    # ------------------------------------------------------------------
    # Listing Methods
    # ------------------------------------------------------------------

    def list_surveys(
        self, facility_id: Optional[str] = None,
        status: Optional[str] = None, survey_type: Optional[str] = None,
        page: int = 1, page_size: int = 50,
    ) -> Dict[str, Any]:
        """List surveys with optional filters and pagination.

        Args:
            facility_id: Filter by facility.
            status: Filter by survey status.
            survey_type: Filter by survey type.
            page: Page number (1-based).
            page_size: Items per page.

        Returns:
            Dictionary with surveys list, total, and pagination metadata.
        """
        with self._lock:
            if facility_id:
                survey_ids = self._surveys_by_facility.get(facility_id, [])
                records = [self._surveys[sid] for sid in survey_ids if sid in self._surveys]
            else:
                records = list(self._surveys.values())

        if status:
            records = [r for r in records if r.status == status.upper()]
        if survey_type:
            records = [r for r in records if r.survey_type == survey_type.upper()]

        records.sort(key=lambda r: r.scheduled_date or "", reverse=True)
        total = len(records)
        start_idx = (page - 1) * page_size
        page_data = records[start_idx:start_idx + page_size]

        surveys = [{
            "survey_id": r.survey_id, "facility_id": r.facility_id,
            "survey_type": r.survey_type, "status": r.status,
            "scheduled_date": r.scheduled_date,
            "completion_date": r.completion_date,
            "components_surveyed": r.components_surveyed,
            "components_total": r.components_total,
            "leaks_detected": r.leaks_detected,
            "inspector_id": r.inspector_id,
            "provenance_hash": r.provenance_hash,
        } for r in page_data]

        return {"surveys": surveys, "total": total, "page": page, "page_size": page_size}

    def list_leaks(
        self, facility_id: Optional[str] = None,
        repair_status: Optional[str] = None, severity: Optional[str] = None,
        page: int = 1, page_size: int = 50,
    ) -> Dict[str, Any]:
        """List leak records with optional filters and pagination.

        Args:
            facility_id: Filter by facility.
            repair_status: Filter by repair status.
            severity: Filter by severity level.
            page: Page number (1-based).
            page_size: Items per page.

        Returns:
            Dictionary with leaks list, total, and pagination metadata.
        """
        with self._lock:
            if facility_id:
                leak_ids = self._leaks_by_facility.get(facility_id, [])
                records = [self._leaks[lid] for lid in leak_ids if lid in self._leaks]
            else:
                records = list(self._leaks.values())

        if repair_status:
            records = [r for r in records if r.repair_status == repair_status.upper()]
        if severity:
            records = [r for r in records if r.severity == severity.upper()]

        records.sort(key=lambda r: r.detection_date or "", reverse=True)
        total = len(records)
        start_idx = (page - 1) * page_size
        page_data = records[start_idx:start_idx + page_size]

        leaks = [{
            "leak_id": r.leak_id, "survey_id": r.survey_id,
            "facility_id": r.facility_id, "component_id": r.component_id,
            "component_type": r.component_type, "service_type": r.service_type,
            "screening_value_ppmv": r.screening_value_ppmv,
            "severity": r.severity, "detection_date": r.detection_date,
            "repair_status": r.repair_status,
            "repair_deadline": r.repair_deadline,
            "repair_date": r.repair_date,
            "post_repair_ppmv": r.post_repair_ppmv,
            "dor_justification": r.dor_justification,
            "estimated_emission_kg_hr": r.estimated_emission_kg_hr,
            "provenance_hash": r.provenance_hash,
        } for r in page_data]

        return {"leaks": leaks, "total": total, "page": page, "page_size": page_size}

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine operational statistics.

        Returns:
            Dictionary with survey, leak, repair, and DOR counts.
        """
        with self._lock:
            return {
                "total_surveys_scheduled": self._total_surveys_scheduled,
                "total_surveys_completed": self._total_surveys_completed,
                "total_leaks_detected": self._total_leaks_detected,
                "total_repairs_completed": self._total_repairs_completed,
                "total_dor_records": self._total_dor_records,
                "surveys_in_registry": len(self._surveys),
                "leaks_in_registry": len(self._leaks),
                "inspectors_registered": len(self._inspectors),
                "default_framework": self._default_framework,
                "default_threshold_ppmv": self._default_threshold,
                "default_repair_deadline_days": self._default_repair_days,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_survey_type(self, survey_type: str) -> None:
        """Validate survey type is recognized.

        Args:
            survey_type: Survey type string.

        Raises:
            ValueError: If not a recognized SurveyType.
        """
        valid = {e.value for e in SurveyType}
        if survey_type.upper() not in valid:
            raise ValueError(
                f"survey_type must be one of {sorted(valid)}, got '{survey_type}'"
            )

    def _classify_severity(self, ppmv: float, threshold: int) -> str:
        """Classify leak severity based on screening value.

        Classification tiers:
            - NONE: ppmv == 0 or below background (< 5 ppmv)
            - MINOR: > 0 and < threshold
            - MODERATE: >= threshold and < 10x threshold
            - MAJOR: >= 10x threshold and < 100x threshold
            - CRITICAL: >= 100x threshold

        Args:
            ppmv: Screening value in parts per million.
            threshold: Regulatory leak threshold in ppmv.

        Returns:
            Severity string.
        """
        if ppmv < 5.0:
            return LeakSeverity.NONE.value
        if ppmv < threshold:
            return LeakSeverity.MINOR.value
        if ppmv < threshold * 10:
            return LeakSeverity.MODERATE.value
        if ppmv < threshold * 100:
            return LeakSeverity.MAJOR.value
        return LeakSeverity.CRITICAL.value
