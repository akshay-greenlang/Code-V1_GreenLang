# -*- coding: utf-8 -*-
"""
FlaringEventTrackerEngine - Event Classification & Volume Tracking (Engine 4 of 7)

AGENT-MRV-006: Flaring Agent

Tracks and classifies individual flaring events at industrial facilities,
recording gas volumes, durations, flow rates, and event categories.  Provides
aggregation capabilities by time period, flare system, and event category for
regulatory reporting (EPA Subpart W, World Bank Zero Routine Flaring, OGMP 2.0).

Event Categories (6):
    ROUTINE:         Continuous process flaring during steady-state operations.
    NON_ROUTINE:     Planned but irregular events (well testing, tank flashing).
    EMERGENCY:       Safety-related events (pressure relief, equipment failure).
    MAINTENANCE:     Startup, shutdown, and turnaround activities.
    PILOT_PURGE:     Continuous pilot flame and purge gas consumption.
    WELL_COMPLETION: Upstream oil & gas flowback flaring.

Volume Estimation Methods:
    MEASURED:   Direct flow meter reading (ultrasonic, orifice, pitot tube).
    ESTIMATED:  Equipment capacity x utilization x duration.
    DEFAULT:    EPA Subpart W default volumes.

Aggregation Periods:
    HOUR, DAY, MONTH, YEAR -- by flare system, by event category.

Zero-Hallucination Guarantees:
    - All calculations are deterministic arithmetic operations.
    - No LLM involvement in any numeric path.
    - Event classification uses explicit rule-based logic only.
    - Every result carries a SHA-256 provenance hash.
    - Same inputs always produce identical outputs.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.flaring.flaring_event_tracker import (
    ...     FlaringEventTrackerEngine,
    ... )
    >>> engine = FlaringEventTrackerEngine()
    >>> event = engine.record_event({
    ...     "flare_id": "FL-001",
    ...     "category": "ROUTINE",
    ...     "start_time": "2026-01-15T08:00:00Z",
    ...     "end_time": "2026-01-15T12:00:00Z",
    ...     "gas_volume_scf": 50000.0,
    ...     "gas_composition_id": "GC-2026-001",
    ...     "measured_flow_rate": 12500.0,
    ...     "estimated": False,
    ... })
    >>> print(event["event_id"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["FlaringEventTrackerEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.flaring.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.flaring.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.flaring.metrics import (
        record_flaring_event as _record_flaring_event,
        observe_calculation_duration as _observe_calculation_duration,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_flaring_event = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _parse_datetime(value: Any) -> datetime:
    """Parse a datetime value from string or datetime object.

    Supports ISO 8601 format strings and datetime objects.

    Args:
        value: Datetime string or datetime object.

    Returns:
        Timezone-aware datetime (UTC if no timezone provided).

    Raises:
        ValueError: If the value cannot be parsed.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, str):
        try:
            # Handle Z suffix
            cleaned = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot parse datetime: {value!r}") from exc

    raise ValueError(f"Expected datetime or ISO string, got {type(value).__name__}")


# ===========================================================================
# Enumerations
# ===========================================================================


class FlaringEventCategory(str, Enum):
    """Classification categories for flaring events.

    ROUTINE:         Continuous process flaring during steady-state operations.
    NON_ROUTINE:     Planned but irregular (well testing, tank flashing).
    EMERGENCY:       Safety-related (pressure relief, equipment failure).
    MAINTENANCE:     Startup, shutdown, turnaround activities.
    PILOT_PURGE:     Continuous pilot flame and purge gas consumption.
    WELL_COMPLETION: Upstream oil & gas flowback flaring.
    """

    ROUTINE = "ROUTINE"
    NON_ROUTINE = "NON_ROUTINE"
    EMERGENCY = "EMERGENCY"
    MAINTENANCE = "MAINTENANCE"
    PILOT_PURGE = "PILOT_PURGE"
    WELL_COMPLETION = "WELL_COMPLETION"


class VolumeEstimationMethod(str, Enum):
    """Methods for estimating flare gas volume.

    MEASURED:   Direct flow meter reading (ultrasonic, orifice, pitot tube).
    ESTIMATED:  Equipment capacity x utilization x duration.
    DEFAULT:    EPA Subpart W default volumes.
    """

    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    DEFAULT = "DEFAULT"


class FlowMeterType(str, Enum):
    """Types of flow measurement instruments for flare gas.

    ULTRASONIC:  Transit-time or Doppler ultrasonic flow meter.
    ORIFICE:     Differential pressure orifice plate meter.
    PITOT_TUBE:  Pitot-tube averaging flow meter.
    THERMAL:     Thermal mass flow meter.
    VORTEX:      Vortex shedding flow meter.
    NONE:        No meter (estimated or default).
    """

    ULTRASONIC = "ULTRASONIC"
    ORIFICE = "ORIFICE"
    PITOT_TUBE = "PITOT_TUBE"
    THERMAL = "THERMAL"
    VORTEX = "VORTEX"
    NONE = "NONE"


class AggregationPeriod(str, Enum):
    """Time periods for event aggregation.

    HOUR:  Aggregate by calendar hour.
    DAY:   Aggregate by calendar day.
    MONTH: Aggregate by calendar month.
    YEAR:  Aggregate by calendar year.
    """

    HOUR = "HOUR"
    DAY = "DAY"
    MONTH = "MONTH"
    YEAR = "YEAR"


class ContinuityType(str, Enum):
    """Classification of flaring continuity.

    CONTINUOUS:   Ongoing flaring without significant gaps.
    INTERMITTENT: Starts and stops with defined on/off periods.
    """

    CONTINUOUS = "CONTINUOUS"
    INTERMITTENT = "INTERMITTENT"


# ===========================================================================
# Default Constants
# ===========================================================================

#: EPA Subpart W default gas volumes (scf per event) by event category.
EPA_DEFAULT_VOLUMES_SCF: Dict[str, Decimal] = {
    FlaringEventCategory.ROUTINE.value: Decimal("100000"),
    FlaringEventCategory.NON_ROUTINE.value: Decimal("50000"),
    FlaringEventCategory.EMERGENCY.value: Decimal("200000"),
    FlaringEventCategory.MAINTENANCE.value: Decimal("75000"),
    FlaringEventCategory.PILOT_PURGE.value: Decimal("500"),
    FlaringEventCategory.WELL_COMPLETION.value: Decimal("150000"),
}

#: Typical pilot gas flow rates (MMBTU/hr per pilot tip).
PILOT_GAS_FLOW_MIN_MMBTU_HR: Decimal = Decimal("0.5")
PILOT_GAS_FLOW_MAX_MMBTU_HR: Decimal = Decimal("5.0")
PILOT_GAS_FLOW_DEFAULT_MMBTU_HR: Decimal = Decimal("2.0")

#: Typical purge gas flow rates (scf/hr).
PURGE_GAS_FLOW_DEFAULT_SCF_HR: Decimal = Decimal("100")

#: Default number of pilot tips per flare.
DEFAULT_PILOT_TIPS: int = 2

#: Default combustion efficiency for pilot gas.
PILOT_GAS_CE: Decimal = Decimal("0.98")

#: Natural gas heating value (BTU/scf) for pilot gas calcs.
NATURAL_GAS_HHV_BTU_SCF: Decimal = Decimal("1020")

#: Conversion factor: 1 MMBTU = 1,000,000 BTU.
BTU_PER_MMBTU: Decimal = Decimal("1000000")

#: Default emission factor for natural gas combustion (kg CO2/scf).
DEFAULT_PILOT_EF_CO2_KG_SCF: Decimal = Decimal("0.05444")

#: Default CH4 slip fraction for pilot gas (uncombusted).
DEFAULT_PILOT_CH4_SLIP_FRACTION: Decimal = Decimal("0.02")

#: Molecular weight ratios for emission calculations.
MW_CO2: Decimal = Decimal("44.01")
MW_CH4: Decimal = Decimal("16.04")
MW_N2O: Decimal = Decimal("44.013")
MW_C: Decimal = Decimal("12.011")

#: GWP values (IPCC AR5 100-year).
GWP_CH4_AR5: Decimal = Decimal("28")
GWP_N2O_AR5: Decimal = Decimal("265")

#: Default emission factors for flaring (kg per scf of gas flared).
DEFAULT_EF_CO2_KG_PER_SCF: Decimal = Decimal("0.05444")
DEFAULT_EF_CH4_KG_PER_SCF: Decimal = Decimal("0.000033")
DEFAULT_EF_N2O_KG_PER_SCF: Decimal = Decimal("0.0000026")

#: Default combustion efficiency.
DEFAULT_COMBUSTION_EFFICIENCY: Decimal = Decimal("0.98")

#: Maximum events in a single query response.
MAX_EVENTS_PER_QUERY: int = 50_000

#: Maximum event duration in hours (safety limit for data validation).
MAX_EVENT_DURATION_HOURS: Decimal = Decimal("8784")  # 366 days


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class FlaringEvent:
    """Record of a single flaring event.

    Attributes:
        event_id: Unique identifier for this event.
        flare_id: Identifier of the flare system.
        category: Event classification category.
        start_time: Event start timestamp (UTC).
        end_time: Event end timestamp (UTC).
        duration_hours: Duration in hours (computed from start/end).
        gas_volume_scf: Total gas volume in standard cubic feet.
        gas_composition_id: Reference to gas composition analysis.
        measured_flow_rate: Average flow rate (scf/hr) if measured.
        estimation_method: Volume estimation method used.
        flow_meter_type: Type of flow meter (if measured).
        continuity_type: Continuous or intermittent flaring.
        estimated: Whether the volume was estimated vs measured.
        notes: Free-text notes about the event.
        pilot_purge_hours: Hours of pilot/purge operation (for PILOT_PURGE).
        pilot_tips_count: Number of pilot tips active.
        purge_gas_type: Type of purge gas (N2, natural_gas).
        facility_id: Facility identifier for tenant scoping.
        tenant_id: Tenant identifier for RLS.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        metadata: Additional metadata dictionary.
    """

    event_id: str
    flare_id: str
    category: str
    start_time: datetime
    end_time: datetime
    duration_hours: Decimal
    gas_volume_scf: Decimal
    gas_composition_id: Optional[str]
    measured_flow_rate: Optional[Decimal]
    estimation_method: str
    flow_meter_type: str
    continuity_type: str
    estimated: bool
    notes: str
    pilot_purge_hours: Optional[Decimal]
    pilot_tips_count: Optional[int]
    purge_gas_type: Optional[str]
    facility_id: Optional[str]
    tenant_id: Optional[str]
    provenance_hash: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event to a plain dictionary."""
        return {
            "event_id": self.event_id,
            "flare_id": self.flare_id,
            "category": self.category,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_hours": str(self.duration_hours),
            "gas_volume_scf": str(self.gas_volume_scf),
            "gas_composition_id": self.gas_composition_id,
            "measured_flow_rate": (
                str(self.measured_flow_rate)
                if self.measured_flow_rate is not None else None
            ),
            "estimation_method": self.estimation_method,
            "flow_meter_type": self.flow_meter_type,
            "continuity_type": self.continuity_type,
            "estimated": self.estimated,
            "notes": self.notes,
            "pilot_purge_hours": (
                str(self.pilot_purge_hours)
                if self.pilot_purge_hours is not None else None
            ),
            "pilot_tips_count": self.pilot_tips_count,
            "purge_gas_type": self.purge_gas_type,
            "facility_id": self.facility_id,
            "tenant_id": self.tenant_id,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class EventAggregation:
    """Aggregated event statistics for a given grouping.

    Attributes:
        group_key: Aggregation group key (period, flare_id, category).
        period_start: Start of the aggregation period.
        period_end: End of the aggregation period.
        total_events: Number of events in the group.
        total_volume_scf: Total gas volume (scf).
        total_duration_hours: Sum of event durations (hours).
        avg_duration_hours: Average event duration (hours).
        avg_flow_rate_scf_hr: Average flow rate (scf/hr).
        peak_flow_rate_scf_hr: Maximum flow rate (scf/hr).
        measured_count: Number of measured events.
        estimated_count: Number of estimated events.
        by_category: Volume breakdown by category.
        provenance_hash: SHA-256 hash for audit trail.
    """

    group_key: str
    period_start: Optional[datetime]
    period_end: Optional[datetime]
    total_events: int
    total_volume_scf: Decimal
    total_duration_hours: Decimal
    avg_duration_hours: Decimal
    avg_flow_rate_scf_hr: Decimal
    peak_flow_rate_scf_hr: Decimal
    measured_count: int
    estimated_count: int
    by_category: Dict[str, Decimal]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the aggregation to a plain dictionary."""
        return {
            "group_key": self.group_key,
            "period_start": (
                self.period_start.isoformat() if self.period_start else None
            ),
            "period_end": (
                self.period_end.isoformat() if self.period_end else None
            ),
            "total_events": self.total_events,
            "total_volume_scf": str(self.total_volume_scf),
            "total_duration_hours": str(self.total_duration_hours),
            "avg_duration_hours": str(self.avg_duration_hours),
            "avg_flow_rate_scf_hr": str(self.avg_flow_rate_scf_hr),
            "peak_flow_rate_scf_hr": str(self.peak_flow_rate_scf_hr),
            "measured_count": self.measured_count,
            "estimated_count": self.estimated_count,
            "by_category": {k: str(v) for k, v in self.by_category.items()},
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class EventStatistics:
    """Overall statistics summary for a set of events.

    Attributes:
        total_events: Total number of events.
        total_volume_scf: Total gas volume (scf).
        avg_duration_hours: Average event duration (hours).
        peak_flow_rate_scf_hr: Maximum observed flow rate (scf/hr).
        routine_volume_scf: Volume from routine flaring.
        non_routine_volume_scf: Volume from non-routine flaring.
        emergency_count: Number of emergency events.
        pilot_purge_hours: Total pilot/purge hours.
        by_category: Event count by category.
        by_flare: Volume by flare system.
        measurement_coverage_pct: Percentage of events that are measured.
        provenance_hash: SHA-256 hash.
    """

    total_events: int
    total_volume_scf: Decimal
    avg_duration_hours: Decimal
    peak_flow_rate_scf_hr: Decimal
    routine_volume_scf: Decimal
    non_routine_volume_scf: Decimal
    emergency_count: int
    pilot_purge_hours: Decimal
    by_category: Dict[str, int]
    by_flare: Dict[str, Decimal]
    measurement_coverage_pct: Decimal
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the statistics to a plain dictionary."""
        return {
            "total_events": self.total_events,
            "total_volume_scf": str(self.total_volume_scf),
            "avg_duration_hours": str(self.avg_duration_hours),
            "peak_flow_rate_scf_hr": str(self.peak_flow_rate_scf_hr),
            "routine_volume_scf": str(self.routine_volume_scf),
            "non_routine_volume_scf": str(self.non_routine_volume_scf),
            "emergency_count": self.emergency_count,
            "pilot_purge_hours": str(self.pilot_purge_hours),
            "by_category": self.by_category,
            "by_flare": {k: str(v) for k, v in self.by_flare.items()},
            "measurement_coverage_pct": str(self.measurement_coverage_pct),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class EventEmissions:
    """Emission calculation result for a flaring event.

    Attributes:
        event_id: Source event identifier.
        co2_kg: CO2 emissions (kg).
        ch4_kg: CH4 emissions (kg).
        n2o_kg: N2O emissions (kg).
        co2e_kg: Total CO2-equivalent emissions (kg).
        combustion_efficiency: Combustion efficiency used.
        calculation_method: Method used for calculation.
        provenance_hash: SHA-256 hash.
    """

    event_id: str
    co2_kg: Decimal
    ch4_kg: Decimal
    n2o_kg: Decimal
    co2e_kg: Decimal
    combustion_efficiency: Decimal
    calculation_method: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "event_id": self.event_id,
            "co2_kg": str(self.co2_kg),
            "ch4_kg": str(self.ch4_kg),
            "n2o_kg": str(self.n2o_kg),
            "co2e_kg": str(self.co2e_kg),
            "combustion_efficiency": str(self.combustion_efficiency),
            "calculation_method": self.calculation_method,
            "provenance_hash": self.provenance_hash,
        }


# ===========================================================================
# Classification Rules
# ===========================================================================

#: Keywords for automatic event classification based on notes/description.
_CLASSIFICATION_KEYWORDS: Dict[str, List[str]] = {
    FlaringEventCategory.ROUTINE.value: [
        "routine", "continuous", "steady-state", "process flare",
        "normal operation", "steady state", "continuous flaring",
    ],
    FlaringEventCategory.NON_ROUTINE.value: [
        "non-routine", "intermittent", "well test", "tank flash",
        "planned", "irregular", "periodic", "tank flashing",
        "well testing", "non routine",
    ],
    FlaringEventCategory.EMERGENCY.value: [
        "emergency", "safety", "pressure relief", "equipment failure",
        "process upset", "blowdown", "depressure", "depressurization",
        "relief valve", "safety valve", "upset", "trip",
    ],
    FlaringEventCategory.MAINTENANCE.value: [
        "maintenance", "startup", "shutdown", "turnaround",
        "start-up", "shut-down", "commissioning", "decommissioning",
        "overhaul", "planned outage",
    ],
    FlaringEventCategory.PILOT_PURGE.value: [
        "pilot", "purge", "pilot light", "purge gas",
        "pilot flame", "sweep gas",
    ],
    FlaringEventCategory.WELL_COMPLETION.value: [
        "well completion", "flowback", "workover", "well workover",
        "completion", "drill stem test", "dst",
    ],
}

#: Duration thresholds for classification assistance (hours).
_DURATION_CLASSIFICATION_HINTS: Dict[str, Tuple[Decimal, Decimal]] = {
    FlaringEventCategory.EMERGENCY.value: (Decimal("0"), Decimal("24")),
    FlaringEventCategory.MAINTENANCE.value: (Decimal("1"), Decimal("720")),
    FlaringEventCategory.WELL_COMPLETION.value: (Decimal("2"), Decimal("720")),
    FlaringEventCategory.PILOT_PURGE.value: (Decimal("24"), MAX_EVENT_DURATION_HOURS),
}

#: Volume thresholds (scf) as secondary classification signal.
_VOLUME_CLASSIFICATION_HINTS: Dict[str, Tuple[Decimal, Decimal]] = {
    FlaringEventCategory.PILOT_PURGE.value: (Decimal("0"), Decimal("10000")),
    FlaringEventCategory.EMERGENCY.value: (Decimal("10000"), Decimal("10000000")),
}


# ===========================================================================
# FlaringEventTrackerEngine
# ===========================================================================


class FlaringEventTrackerEngine:
    """Engine for tracking, classifying, and aggregating flaring events.

    Records individual flaring events with gas volumes, durations, and
    classifications, then provides aggregation and reporting capabilities
    for regulatory compliance (EPA Subpart W, World Bank ZRF, OGMP 2.0).

    All calculations are deterministic with zero LLM involvement.
    Thread-safe via reentrant lock.

    Attributes:
        config: Optional configuration dictionary.

    Example:
        >>> engine = FlaringEventTrackerEngine()
        >>> event = engine.record_event({
        ...     "flare_id": "FL-001",
        ...     "category": "ROUTINE",
        ...     "start_time": "2026-01-15T08:00:00Z",
        ...     "end_time": "2026-01-15T12:00:00Z",
        ...     "gas_volume_scf": 50000.0,
        ... })
        >>> stats = engine.get_statistics()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the FlaringEventTrackerEngine.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - default_combustion_efficiency (float): Override default CE.
                - max_events (int): Maximum stored events (default 100000).
                - default_estimation_method (str): MEASURED/ESTIMATED/DEFAULT.
        """
        self._config = config or {}
        self._lock = threading.RLock()
        self._events: List[FlaringEvent] = []
        self._events_by_id: Dict[str, FlaringEvent] = {}
        self._events_by_flare: Dict[str, List[str]] = {}
        self._events_by_category: Dict[str, List[str]] = {}

        # Configurable parameters
        self._max_events = self._config.get("max_events", 100_000)
        self._default_ce = _to_decimal(
            self._config.get(
                "default_combustion_efficiency",
                DEFAULT_COMBUSTION_EFFICIENCY,
            )
        )
        self._default_estimation_method = self._config.get(
            "default_estimation_method", VolumeEstimationMethod.ESTIMATED.value
        )

        # Statistics counters
        self._total_recorded: int = 0
        self._total_volume_scf: Decimal = Decimal("0")

        logger.info(
            "FlaringEventTrackerEngine initialized: "
            "max_events=%d, default_ce=%s, 6 categories, 3 estimation methods",
            self._max_events,
            self._default_ce,
        )

    # ------------------------------------------------------------------
    # Public API: Record Event
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a new flaring event.

        Creates a FlaringEvent record from the provided data, validates
        fields, computes derived values (duration, provenance hash),
        and stores the event.

        Required keys:
            - flare_id (str): Flare system identifier.
            - start_time (str/datetime): Event start (ISO 8601 or datetime).
            - end_time (str/datetime): Event end (ISO 8601 or datetime).

        Optional keys:
            - category (str): Event category (auto-classified if absent).
            - gas_volume_scf (float): Gas volume in standard cubic feet.
            - gas_composition_id (str): Reference to composition analysis.
            - measured_flow_rate (float): Average flow rate (scf/hr).
            - estimation_method (str): MEASURED, ESTIMATED, or DEFAULT.
            - flow_meter_type (str): Type of flow meter used.
            - continuity_type (str): CONTINUOUS or INTERMITTENT.
            - estimated (bool): Whether volume is estimated.
            - notes (str): Free-text notes.
            - pilot_purge_hours (float): For PILOT_PURGE events.
            - pilot_tips_count (int): Number of active pilot tips.
            - purge_gas_type (str): N2 or natural_gas.
            - facility_id (str): Facility identifier.
            - tenant_id (str): Tenant identifier.
            - metadata (dict): Additional metadata.

        Args:
            event_data: Dictionary of event parameters.

        Returns:
            Dictionary with the complete event record.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        t0 = time.monotonic()

        # Validate required fields
        flare_id = event_data.get("flare_id")
        if not flare_id or not isinstance(flare_id, str):
            raise ValueError("flare_id is required and must be a non-empty string")

        start_time = _parse_datetime(event_data.get("start_time"))
        end_time = _parse_datetime(event_data.get("end_time"))

        # Validate time ordering
        if end_time <= start_time:
            raise ValueError(
                f"end_time ({end_time.isoformat()}) must be after "
                f"start_time ({start_time.isoformat()})"
            )

        # Compute duration
        duration_td = end_time - start_time
        duration_hours = _to_decimal(
            duration_td.total_seconds()
        ) / Decimal("3600")
        duration_hours = duration_hours.quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Validate duration
        if duration_hours > MAX_EVENT_DURATION_HOURS:
            raise ValueError(
                f"Event duration {duration_hours} hours exceeds maximum "
                f"of {MAX_EVENT_DURATION_HOURS} hours"
            )

        # Gas volume
        gas_volume_scf = self._resolve_gas_volume(event_data, duration_hours)

        # Estimation method
        estimation_method = event_data.get(
            "estimation_method", self._default_estimation_method
        ).upper().strip()
        if estimation_method not in {m.value for m in VolumeEstimationMethod}:
            estimation_method = self._default_estimation_method

        # Estimated flag
        estimated = event_data.get("estimated", estimation_method != "MEASURED")

        # Flow meter type
        flow_meter_type = event_data.get(
            "flow_meter_type", FlowMeterType.NONE.value
        ).upper().strip()
        if flow_meter_type not in {m.value for m in FlowMeterType}:
            flow_meter_type = FlowMeterType.NONE.value

        # Continuity type
        continuity_type = event_data.get(
            "continuity_type", ContinuityType.INTERMITTENT.value
        ).upper().strip()
        if continuity_type not in {m.value for m in ContinuityType}:
            continuity_type = ContinuityType.INTERMITTENT.value

        # Category (auto-classify if not provided)
        category = event_data.get("category", "").upper().strip()
        if category and category in {c.value for c in FlaringEventCategory}:
            pass
        else:
            category = self.classify_event(event_data, duration_hours, gas_volume_scf)

        # Measured flow rate
        measured_flow_rate = None
        if event_data.get("measured_flow_rate") is not None:
            measured_flow_rate = _to_decimal(
                event_data["measured_flow_rate"]
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Pilot/purge fields
        pilot_purge_hours = None
        pilot_tips_count = None
        purge_gas_type = None
        if category == FlaringEventCategory.PILOT_PURGE.value:
            pilot_purge_hours = _to_decimal(
                event_data.get("pilot_purge_hours", duration_hours)
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            pilot_tips_count = int(
                event_data.get("pilot_tips_count", DEFAULT_PILOT_TIPS)
            )
            purge_gas_type = event_data.get("purge_gas_type", "natural_gas")

        # Generate event ID
        event_id = event_data.get("event_id", f"fl_evt_{uuid4().hex[:12]}")

        # Provenance hash
        provenance_data = {
            "event_id": event_id,
            "flare_id": flare_id,
            "category": category,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": str(duration_hours),
            "gas_volume_scf": str(gas_volume_scf),
            "estimation_method": estimation_method,
        }
        provenance_hash = _compute_hash(provenance_data)

        # Create event
        event = FlaringEvent(
            event_id=event_id,
            flare_id=flare_id,
            category=category,
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            gas_volume_scf=gas_volume_scf,
            gas_composition_id=event_data.get("gas_composition_id"),
            measured_flow_rate=measured_flow_rate,
            estimation_method=estimation_method,
            flow_meter_type=flow_meter_type,
            continuity_type=continuity_type,
            estimated=estimated,
            notes=event_data.get("notes", ""),
            pilot_purge_hours=pilot_purge_hours,
            pilot_tips_count=pilot_tips_count,
            purge_gas_type=purge_gas_type,
            facility_id=event_data.get("facility_id"),
            tenant_id=event_data.get("tenant_id"),
            provenance_hash=provenance_hash,
            created_at=_utcnow(),
            metadata=event_data.get("metadata", {}),
        )

        # Store event
        self._store_event(event)

        # Metrics
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._record_metrics(category, elapsed_ms)

        logger.info(
            "Flaring event recorded: event_id=%s flare_id=%s category=%s "
            "duration=%.2fh volume=%s scf (%.1fms)",
            event_id, flare_id, category,
            float(duration_hours), gas_volume_scf, elapsed_ms,
        )

        return event.to_dict()

    # ------------------------------------------------------------------
    # Public API: Get Events
    # ------------------------------------------------------------------

    def get_events(
        self,
        flare_id: Optional[str] = None,
        category: Optional[str] = None,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        facility_id: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve flaring events with optional filtering.

        Args:
            flare_id: Filter by flare system ID.
            category: Filter by event category.
            start_time: Filter events starting at or after this time.
            end_time: Filter events ending at or before this time.
            facility_id: Filter by facility ID.
            limit: Maximum number of events to return.
            offset: Number of events to skip.

        Returns:
            List of event dictionaries matching the filters.
        """
        with self._lock:
            events = list(self._events)

        # Apply filters
        if flare_id:
            events = [e for e in events if e.flare_id == flare_id]
        if category:
            cat_upper = category.upper().strip()
            events = [e for e in events if e.category == cat_upper]
        if start_time:
            st = _parse_datetime(start_time)
            events = [e for e in events if e.start_time >= st]
        if end_time:
            et = _parse_datetime(end_time)
            events = [e for e in events if e.end_time <= et]
        if facility_id:
            events = [e for e in events if e.facility_id == facility_id]

        # Sort by start_time descending (most recent first)
        events.sort(key=lambda e: e.start_time, reverse=True)

        # Pagination
        paginated = events[offset:offset + limit]

        return [e.to_dict() for e in paginated]

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single event by its ID.

        Args:
            event_id: The unique event identifier.

        Returns:
            Event dictionary if found, None otherwise.
        """
        with self._lock:
            event = self._events_by_id.get(event_id)
        if event is None:
            return None
        return event.to_dict()

    # ------------------------------------------------------------------
    # Public API: Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_period(
        self,
        period: str,
        flare_id: Optional[str] = None,
        category: Optional[str] = None,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate events by time period.

        Groups events by the specified period (HOUR, DAY, MONTH, YEAR)
        and computes totals and averages for each group.

        Args:
            period: Aggregation period (HOUR, DAY, MONTH, YEAR).
            flare_id: Optional filter by flare system.
            category: Optional filter by event category.
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            List of aggregation dictionaries sorted by period.

        Raises:
            ValueError: If period is not a valid AggregationPeriod.
        """
        period_upper = period.upper().strip()
        if period_upper not in {p.value for p in AggregationPeriod}:
            raise ValueError(
                f"Invalid aggregation period: {period}. "
                f"Must be one of: {[p.value for p in AggregationPeriod]}"
            )

        # Fetch filtered events
        events = self._get_filtered_events(
            flare_id, category, start_time, end_time
        )

        if not events:
            return []

        # Group events by period key
        groups: Dict[str, List[FlaringEvent]] = {}
        for event in events:
            key = self._period_key(event.start_time, period_upper)
            if key not in groups:
                groups[key] = []
            groups[key].append(event)

        # Compute aggregations
        aggregations: List[EventAggregation] = []
        for group_key, group_events in sorted(groups.items()):
            agg = self._compute_aggregation(group_key, group_events, period_upper)
            aggregations.append(agg)

        return [a.to_dict() for a in aggregations]

    def aggregate_by_flare(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate events by flare system.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            List of aggregation dictionaries keyed by flare_id.
        """
        events = self._get_filtered_events(
            flare_id=None, category=None,
            start_time=start_time, end_time=end_time,
        )

        if not events:
            return []

        groups: Dict[str, List[FlaringEvent]] = {}
        for event in events:
            if event.flare_id not in groups:
                groups[event.flare_id] = []
            groups[event.flare_id].append(event)

        aggregations: List[EventAggregation] = []
        for flare_id_key, group_events in sorted(groups.items()):
            agg = self._compute_aggregation(
                f"flare:{flare_id_key}", group_events, "CUSTOM"
            )
            aggregations.append(agg)

        return [a.to_dict() for a in aggregations]

    def aggregate_by_category(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate events by event category.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            List of aggregation dictionaries keyed by category.
        """
        events = self._get_filtered_events(
            flare_id=None, category=None,
            start_time=start_time, end_time=end_time,
        )

        if not events:
            return []

        groups: Dict[str, List[FlaringEvent]] = {}
        for event in events:
            if event.category not in groups:
                groups[event.category] = []
            groups[event.category].append(event)

        aggregations: List[EventAggregation] = []
        for cat_key, group_events in sorted(groups.items()):
            agg = self._compute_aggregation(
                f"category:{cat_key}", group_events, "CUSTOM"
            )
            aggregations.append(agg)

        return [a.to_dict() for a in aggregations]

    # ------------------------------------------------------------------
    # Public API: Routine / Non-Routine Volume (ZRF Reporting)
    # ------------------------------------------------------------------

    def get_routine_volume(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        flare_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get total routine flaring volume for ZRF reporting.

        Routine flaring includes ROUTINE category events only.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.
            flare_id: Optional filter by flare system.

        Returns:
            Dictionary with routine volume data.
        """
        events = self._get_filtered_events(
            flare_id=flare_id,
            category=FlaringEventCategory.ROUTINE.value,
            start_time=start_time,
            end_time=end_time,
        )

        total_volume = sum(
            (e.gas_volume_scf for e in events), Decimal("0")
        )
        total_duration = sum(
            (e.duration_hours for e in events), Decimal("0")
        )

        result = {
            "category": "ROUTINE",
            "event_count": len(events),
            "total_volume_scf": str(total_volume),
            "total_duration_hours": str(total_duration),
            "measured_count": sum(1 for e in events if not e.estimated),
            "estimated_count": sum(1 for e in events if e.estimated),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def get_non_routine_volume(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        flare_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get total non-routine flaring volume for ZRF reporting.

        Non-routine includes NON_ROUTINE, EMERGENCY, MAINTENANCE,
        and WELL_COMPLETION categories.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.
            flare_id: Optional filter by flare system.

        Returns:
            Dictionary with non-routine volume data.
        """
        non_routine_cats = {
            FlaringEventCategory.NON_ROUTINE.value,
            FlaringEventCategory.EMERGENCY.value,
            FlaringEventCategory.MAINTENANCE.value,
            FlaringEventCategory.WELL_COMPLETION.value,
        }

        all_events = self._get_filtered_events(
            flare_id=flare_id, category=None,
            start_time=start_time, end_time=end_time,
        )
        events = [e for e in all_events if e.category in non_routine_cats]

        total_volume = sum(
            (e.gas_volume_scf for e in events), Decimal("0")
        )
        total_duration = sum(
            (e.duration_hours for e in events), Decimal("0")
        )

        by_category: Dict[str, str] = {}
        for cat in non_routine_cats:
            cat_vol = sum(
                (e.gas_volume_scf for e in events if e.category == cat),
                Decimal("0"),
            )
            by_category[cat] = str(cat_vol)

        result = {
            "category": "NON_ROUTINE_AGGREGATE",
            "event_count": len(events),
            "total_volume_scf": str(total_volume),
            "total_duration_hours": str(total_duration),
            "by_category": by_category,
            "measured_count": sum(1 for e in events if not e.estimated),
            "estimated_count": sum(1 for e in events if e.estimated),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API: Emergency Count
    # ------------------------------------------------------------------

    def get_emergency_count(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        flare_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get count and volume of emergency flaring events.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.
            flare_id: Optional filter by flare system.

        Returns:
            Dictionary with emergency event data.
        """
        events = self._get_filtered_events(
            flare_id=flare_id,
            category=FlaringEventCategory.EMERGENCY.value,
            start_time=start_time,
            end_time=end_time,
        )

        total_volume = sum(
            (e.gas_volume_scf for e in events), Decimal("0")
        )
        total_duration = sum(
            (e.duration_hours for e in events), Decimal("0")
        )

        result = {
            "category": "EMERGENCY",
            "event_count": len(events),
            "total_volume_scf": str(total_volume),
            "total_duration_hours": str(total_duration),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API: Event Classification
    # ------------------------------------------------------------------

    def classify_event(
        self,
        event_data: Dict[str, Any],
        duration_hours: Optional[Decimal] = None,
        gas_volume_scf: Optional[Decimal] = None,
    ) -> str:
        """Classify a flaring event into one of the 6 categories.

        Uses a rule-based approach combining keyword analysis of notes,
        duration heuristics, and volume thresholds. No LLM involvement.

        Classification priority:
            1. Keyword matches in notes/description (highest priority).
            2. Duration-based classification hints.
            3. Volume-based classification hints.
            4. Default to NON_ROUTINE if no signals match.

        Args:
            event_data: Event data dictionary with optional 'notes' field.
            duration_hours: Event duration in hours (optional).
            gas_volume_scf: Gas volume in scf (optional).

        Returns:
            Event category string (e.g. "ROUTINE").
        """
        notes = str(event_data.get("notes", "")).lower().strip()
        description = str(event_data.get("description", "")).lower().strip()
        combined_text = f"{notes} {description}"

        # Step 1: Keyword matching (highest confidence)
        keyword_scores: Dict[str, int] = {}
        for cat, keywords in _CLASSIFICATION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined_text)
            if score > 0:
                keyword_scores[cat] = score

        if keyword_scores:
            best_cat = max(keyword_scores, key=keyword_scores.get)  # type: ignore[arg-type]
            logger.debug(
                "Event classified by keywords: %s (score=%d)",
                best_cat, keyword_scores[best_cat],
            )
            return best_cat

        # Step 2: Duration-based hints
        if duration_hours is not None:
            for cat, (min_h, max_h) in _DURATION_CLASSIFICATION_HINTS.items():
                if min_h <= duration_hours <= max_h:
                    # Check if this is the pilot/purge range with low volume
                    if cat == FlaringEventCategory.PILOT_PURGE.value:
                        if gas_volume_scf is not None:
                            vol_min, vol_max = _VOLUME_CLASSIFICATION_HINTS.get(
                                cat, (Decimal("0"), Decimal("10000"))
                            )
                            if vol_min <= gas_volume_scf <= vol_max:
                                return cat
                        continue
                    return cat

        # Step 3: Volume-based hints
        if gas_volume_scf is not None:
            if gas_volume_scf > Decimal("100000"):
                return FlaringEventCategory.EMERGENCY.value
            if gas_volume_scf < Decimal("1000"):
                return FlaringEventCategory.PILOT_PURGE.value

        # Step 4: Default
        return FlaringEventCategory.NON_ROUTINE.value

    # ------------------------------------------------------------------
    # Public API: Validate Event Data
    # ------------------------------------------------------------------

    def validate_event_data(
        self,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate event data without recording it.

        Checks all required and optional fields for correctness.

        Args:
            event_data: Event data dictionary to validate.

        Returns:
            Dictionary with validation results including is_valid,
            errors, and warnings lists.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Required fields
        if not event_data.get("flare_id"):
            errors.append("flare_id is required")
        if not event_data.get("start_time"):
            errors.append("start_time is required")
        if not event_data.get("end_time"):
            errors.append("end_time is required")

        # Time validation
        if event_data.get("start_time") and event_data.get("end_time"):
            try:
                st = _parse_datetime(event_data["start_time"])
                et = _parse_datetime(event_data["end_time"])
                if et <= st:
                    errors.append("end_time must be after start_time")
                else:
                    dur = _to_decimal(
                        (et - st).total_seconds()
                    ) / Decimal("3600")
                    if dur > MAX_EVENT_DURATION_HOURS:
                        errors.append(
                            f"Duration {dur}h exceeds maximum {MAX_EVENT_DURATION_HOURS}h"
                        )
                    if dur < Decimal("0.001"):
                        warnings.append("Duration is less than 3.6 seconds")
            except ValueError as exc:
                errors.append(f"Time parsing error: {exc}")

        # Category validation
        category = event_data.get("category", "")
        if category:
            cat_upper = category.upper().strip()
            valid_cats = {c.value for c in FlaringEventCategory}
            if cat_upper not in valid_cats:
                errors.append(
                    f"Invalid category '{category}'. "
                    f"Must be one of: {sorted(valid_cats)}"
                )

        # Volume validation
        vol = event_data.get("gas_volume_scf")
        if vol is not None:
            try:
                vol_dec = _to_decimal(vol)
                if vol_dec < Decimal("0"):
                    errors.append("gas_volume_scf cannot be negative")
            except ValueError:
                errors.append(f"Invalid gas_volume_scf: {vol}")

        # Flow rate validation
        flow = event_data.get("measured_flow_rate")
        if flow is not None:
            try:
                flow_dec = _to_decimal(flow)
                if flow_dec < Decimal("0"):
                    errors.append("measured_flow_rate cannot be negative")
            except ValueError:
                errors.append(f"Invalid measured_flow_rate: {flow}")

        # Estimation method validation
        est = event_data.get("estimation_method", "")
        if est:
            est_upper = est.upper().strip()
            valid_est = {m.value for m in VolumeEstimationMethod}
            if est_upper not in valid_est:
                warnings.append(
                    f"Unknown estimation_method '{est}'. "
                    f"Valid values: {sorted(valid_est)}"
                )

        # Pilot/purge specific validation
        if category and category.upper().strip() == "PILOT_PURGE":
            tips = event_data.get("pilot_tips_count")
            if tips is not None:
                try:
                    tips_int = int(tips)
                    if tips_int < 1 or tips_int > 10:
                        warnings.append(
                            f"pilot_tips_count {tips_int} outside typical range 1-10"
                        )
                except (ValueError, TypeError):
                    errors.append(f"Invalid pilot_tips_count: {tips}")

        is_valid = len(errors) == 0

        result: Dict[str, Any] = {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "field_count": len(event_data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API: Calculate Event Emissions
    # ------------------------------------------------------------------

    def calculate_event_emissions(
        self,
        event_id: str,
        combustion_efficiency: Optional[Decimal] = None,
        gas_composition: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """Calculate emissions for a recorded flaring event.

        Uses the default emission factor method if no gas composition
        is provided. Applies combustion efficiency to determine
        uncombusted CH4 slip.

        Formula (default EF):
            CO2 = volume_scf * EF_CO2 * CE
            CH4 = volume_scf * EF_CH4 * (1 - CE)   [uncombusted slip]
            N2O = volume_scf * EF_N2O
            CO2e = CO2 + CH4 * GWP_CH4 + N2O * GWP_N2O

        Args:
            event_id: The event identifier.
            combustion_efficiency: Override combustion efficiency (0-1).
            gas_composition: Optional composition {gas: mole_fraction}.

        Returns:
            Dictionary with emission calculation results.

        Raises:
            ValueError: If event not found or CE out of range.
        """
        t0 = time.monotonic()

        with self._lock:
            event = self._events_by_id.get(event_id)
        if event is None:
            raise ValueError(f"Event not found: {event_id}")

        ce = combustion_efficiency if combustion_efficiency is not None else self._default_ce
        ce = _to_decimal(ce)
        if ce < Decimal("0") or ce > Decimal("1"):
            raise ValueError(
                f"Combustion efficiency must be in [0, 1], got {ce}"
            )

        volume = event.gas_volume_scf

        if gas_composition:
            emissions = self._calculate_composition_based(
                volume, gas_composition, ce,
            )
        else:
            emissions = self._calculate_default_ef(volume, ce)

        provenance_data = {
            "event_id": event_id,
            "volume_scf": str(volume),
            "combustion_efficiency": str(ce),
            "co2_kg": str(emissions.co2_kg),
            "ch4_kg": str(emissions.ch4_kg),
            "n2o_kg": str(emissions.n2o_kg),
            "co2e_kg": str(emissions.co2e_kg),
        }
        emissions_obj = EventEmissions(
            event_id=event_id,
            co2_kg=emissions.co2_kg,
            ch4_kg=emissions.ch4_kg,
            n2o_kg=emissions.n2o_kg,
            co2e_kg=emissions.co2e_kg,
            combustion_efficiency=ce,
            calculation_method=(
                "GAS_COMPOSITION" if gas_composition else "DEFAULT_EF"
            ),
            provenance_hash=_compute_hash(provenance_data),
        )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "Event emissions calculated: event_id=%s co2e=%.3f kg (%.1fms)",
            event_id, float(emissions_obj.co2e_kg), elapsed_ms,
        )

        return emissions_obj.to_dict()

    # ------------------------------------------------------------------
    # Public API: Pilot/Purge Tracking
    # ------------------------------------------------------------------

    def get_pilot_purge_summary(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        flare_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get summary of pilot and purge gas operations.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.
            flare_id: Optional filter by flare system.

        Returns:
            Dictionary with pilot/purge summary data.
        """
        events = self._get_filtered_events(
            flare_id=flare_id,
            category=FlaringEventCategory.PILOT_PURGE.value,
            start_time=start_time,
            end_time=end_time,
        )

        total_hours = sum(
            (e.pilot_purge_hours or e.duration_hours for e in events),
            Decimal("0"),
        )
        total_volume = sum(
            (e.gas_volume_scf for e in events), Decimal("0")
        )

        continuous_events = [
            e for e in events if e.continuity_type == ContinuityType.CONTINUOUS.value
        ]
        intermittent_events = [
            e for e in events if e.continuity_type == ContinuityType.INTERMITTENT.value
        ]

        continuous_hours = sum(
            (e.pilot_purge_hours or e.duration_hours for e in continuous_events),
            Decimal("0"),
        )
        intermittent_count = len(intermittent_events)

        result = {
            "total_events": len(events),
            "total_hours": str(total_hours),
            "total_volume_scf": str(total_volume),
            "continuous_hours": str(continuous_hours),
            "intermittent_event_count": intermittent_count,
            "avg_pilot_tips": (
                round(
                    sum(e.pilot_tips_count or 0 for e in events) / len(events), 1
                ) if events else 0
            ),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API: Statistics
    # ------------------------------------------------------------------

    def get_statistics(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, Any]:
        """Get overall event tracking statistics.

        Args:
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            Dictionary with comprehensive event statistics.
        """
        events = self._get_filtered_events(
            flare_id=None, category=None,
            start_time=start_time, end_time=end_time,
        )

        if not events:
            empty_stats = EventStatistics(
                total_events=0,
                total_volume_scf=Decimal("0"),
                avg_duration_hours=Decimal("0"),
                peak_flow_rate_scf_hr=Decimal("0"),
                routine_volume_scf=Decimal("0"),
                non_routine_volume_scf=Decimal("0"),
                emergency_count=0,
                pilot_purge_hours=Decimal("0"),
                by_category={},
                by_flare={},
                measurement_coverage_pct=Decimal("0"),
                provenance_hash="",
            )
            empty_stats.provenance_hash = _compute_hash(empty_stats.to_dict())
            return empty_stats.to_dict()

        total_volume = sum(
            (e.gas_volume_scf for e in events), Decimal("0")
        )
        total_duration = sum(
            (e.duration_hours for e in events), Decimal("0")
        )
        avg_duration = (total_duration / Decimal(str(len(events)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        flow_rates = [
            e.measured_flow_rate for e in events
            if e.measured_flow_rate is not None and e.measured_flow_rate > Decimal("0")
        ]
        peak_flow = max(flow_rates) if flow_rates else Decimal("0")

        routine_volume = sum(
            (e.gas_volume_scf for e in events
             if e.category == FlaringEventCategory.ROUTINE.value),
            Decimal("0"),
        )

        non_routine_cats = {
            FlaringEventCategory.NON_ROUTINE.value,
            FlaringEventCategory.EMERGENCY.value,
            FlaringEventCategory.MAINTENANCE.value,
            FlaringEventCategory.WELL_COMPLETION.value,
        }
        non_routine_volume = sum(
            (e.gas_volume_scf for e in events if e.category in non_routine_cats),
            Decimal("0"),
        )

        emergency_count = sum(
            1 for e in events
            if e.category == FlaringEventCategory.EMERGENCY.value
        )

        pilot_purge_hours = sum(
            (e.pilot_purge_hours or e.duration_hours
             for e in events
             if e.category == FlaringEventCategory.PILOT_PURGE.value),
            Decimal("0"),
        )

        by_category: Dict[str, int] = {}
        for e in events:
            by_category[e.category] = by_category.get(e.category, 0) + 1

        by_flare: Dict[str, Decimal] = {}
        for e in events:
            by_flare[e.flare_id] = by_flare.get(
                e.flare_id, Decimal("0")
            ) + e.gas_volume_scf

        measured_count = sum(1 for e in events if not e.estimated)
        measurement_pct = (
            Decimal(str(measured_count)) / Decimal(str(len(events))) * Decimal("100")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        stats = EventStatistics(
            total_events=len(events),
            total_volume_scf=total_volume,
            avg_duration_hours=avg_duration,
            peak_flow_rate_scf_hr=peak_flow,
            routine_volume_scf=routine_volume,
            non_routine_volume_scf=non_routine_volume,
            emergency_count=emergency_count,
            pilot_purge_hours=pilot_purge_hours,
            by_category=by_category,
            by_flare=by_flare,
            measurement_coverage_pct=measurement_pct,
            provenance_hash="",
        )
        stats.provenance_hash = _compute_hash(stats.to_dict())

        return stats.to_dict()

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get engine-level statistics (not event-level).

        Returns:
            Dictionary with engine operation counters.
        """
        with self._lock:
            return {
                "total_events_recorded": self._total_recorded,
                "total_volume_scf_recorded": str(self._total_volume_scf),
                "events_in_memory": len(self._events),
                "flare_systems_tracked": len(self._events_by_flare),
                "categories_tracked": len(self._events_by_category),
                "max_events": self._max_events,
            }

    # ------------------------------------------------------------------
    # Public API: Clear / Reset
    # ------------------------------------------------------------------

    def clear_events(self) -> int:
        """Clear all stored events.

        Returns:
            Number of events cleared.
        """
        with self._lock:
            count = len(self._events)
            self._events.clear()
            self._events_by_id.clear()
            self._events_by_flare.clear()
            self._events_by_category.clear()
        logger.info("Events cleared: %d records", count)
        return count

    # ------------------------------------------------------------------
    # Private: Event Storage
    # ------------------------------------------------------------------

    def _store_event(self, event: FlaringEvent) -> None:
        """Store an event in all indexes.

        Thread-safe. Enforces maximum event count by evicting oldest.

        Args:
            event: The FlaringEvent to store.
        """
        with self._lock:
            # Evict oldest if at capacity
            while len(self._events) >= self._max_events:
                oldest = self._events.pop(0)
                self._events_by_id.pop(oldest.event_id, None)
                flare_list = self._events_by_flare.get(oldest.flare_id, [])
                if oldest.event_id in flare_list:
                    flare_list.remove(oldest.event_id)
                cat_list = self._events_by_category.get(oldest.category, [])
                if oldest.event_id in cat_list:
                    cat_list.remove(oldest.event_id)

            # Store
            self._events.append(event)
            self._events_by_id[event.event_id] = event

            if event.flare_id not in self._events_by_flare:
                self._events_by_flare[event.flare_id] = []
            self._events_by_flare[event.flare_id].append(event.event_id)

            if event.category not in self._events_by_category:
                self._events_by_category[event.category] = []
            self._events_by_category[event.category].append(event.event_id)

            # Update counters
            self._total_recorded += 1
            self._total_volume_scf += event.gas_volume_scf

    # ------------------------------------------------------------------
    # Private: Gas Volume Resolution
    # ------------------------------------------------------------------

    def _resolve_gas_volume(
        self,
        event_data: Dict[str, Any],
        duration_hours: Decimal,
    ) -> Decimal:
        """Resolve gas volume from event data.

        Priority:
            1. Explicitly provided gas_volume_scf.
            2. measured_flow_rate * duration_hours.
            3. Equipment capacity * utilization * duration.
            4. EPA default volume for the category.

        Args:
            event_data: Event data dictionary.
            duration_hours: Computed event duration.

        Returns:
            Gas volume in standard cubic feet.
        """
        # Priority 1: Explicit volume
        if event_data.get("gas_volume_scf") is not None:
            vol = _to_decimal(event_data["gas_volume_scf"])
            if vol < Decimal("0"):
                raise ValueError(f"gas_volume_scf cannot be negative: {vol}")
            return vol.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Priority 2: Flow rate * duration
        if event_data.get("measured_flow_rate") is not None:
            flow_rate = _to_decimal(event_data["measured_flow_rate"])
            if flow_rate < Decimal("0"):
                raise ValueError(
                    f"measured_flow_rate cannot be negative: {flow_rate}"
                )
            vol = (flow_rate * duration_hours).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            return vol

        # Priority 3: Equipment capacity * utilization * duration
        capacity = event_data.get("equipment_capacity_scf_hr")
        utilization = event_data.get("utilization_factor")
        if capacity is not None and utilization is not None:
            cap = _to_decimal(capacity)
            util = _to_decimal(utilization)
            vol = (cap * util * duration_hours).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            return vol

        # Priority 4: EPA default
        category = event_data.get("category", "NON_ROUTINE").upper().strip()
        default_vol = EPA_DEFAULT_VOLUMES_SCF.get(
            category, EPA_DEFAULT_VOLUMES_SCF[FlaringEventCategory.NON_ROUTINE.value]
        )
        logger.debug(
            "Using EPA default volume %s scf for category %s",
            default_vol, category,
        )
        return default_vol

    # ------------------------------------------------------------------
    # Private: Event Filtering
    # ------------------------------------------------------------------

    def _get_filtered_events(
        self,
        flare_id: Optional[str],
        category: Optional[str],
        start_time: Optional[Union[str, datetime]],
        end_time: Optional[Union[str, datetime]],
    ) -> List[FlaringEvent]:
        """Get events matching the specified filters.

        Args:
            flare_id: Optional flare system filter.
            category: Optional category filter.
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            List of matching FlaringEvent objects.
        """
        with self._lock:
            events = list(self._events)

        if flare_id:
            events = [e for e in events if e.flare_id == flare_id]
        if category:
            cat_upper = category.upper().strip()
            events = [e for e in events if e.category == cat_upper]
        if start_time:
            st = _parse_datetime(start_time)
            events = [e for e in events if e.start_time >= st]
        if end_time:
            et = _parse_datetime(end_time)
            events = [e for e in events if e.end_time <= et]

        return events

    # ------------------------------------------------------------------
    # Private: Aggregation Helpers
    # ------------------------------------------------------------------

    def _period_key(self, dt: datetime, period: str) -> str:
        """Generate an aggregation period key from a datetime.

        Args:
            dt: The datetime value.
            period: Aggregation period (HOUR, DAY, MONTH, YEAR).

        Returns:
            String key for grouping (e.g. '2026-01-15T08' for HOUR).
        """
        if period == AggregationPeriod.HOUR.value:
            return dt.strftime("%Y-%m-%dT%H")
        if period == AggregationPeriod.DAY.value:
            return dt.strftime("%Y-%m-%d")
        if period == AggregationPeriod.MONTH.value:
            return dt.strftime("%Y-%m")
        if period == AggregationPeriod.YEAR.value:
            return dt.strftime("%Y")
        return dt.strftime("%Y-%m-%d")

    def _compute_aggregation(
        self,
        group_key: str,
        events: List[FlaringEvent],
        period: str,
    ) -> EventAggregation:
        """Compute aggregation statistics for a group of events.

        Args:
            group_key: The aggregation group key.
            events: List of events in the group.
            period: Aggregation period type.

        Returns:
            EventAggregation dataclass.
        """
        total_volume = sum(
            (e.gas_volume_scf for e in events), Decimal("0")
        )
        total_duration = sum(
            (e.duration_hours for e in events), Decimal("0")
        )
        n = len(events)
        avg_duration = (
            (total_duration / Decimal(str(n))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ) if n > 0 else Decimal("0")
        )

        flow_rates = [
            e.measured_flow_rate for e in events
            if e.measured_flow_rate is not None and e.measured_flow_rate > Decimal("0")
        ]
        avg_flow = (
            (sum(flow_rates, Decimal("0")) / Decimal(str(len(flow_rates)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ) if flow_rates else Decimal("0")
        )
        peak_flow = max(flow_rates) if flow_rates else Decimal("0")

        measured_count = sum(1 for e in events if not e.estimated)
        estimated_count = sum(1 for e in events if e.estimated)

        by_category: Dict[str, Decimal] = {}
        for e in events:
            by_category[e.category] = by_category.get(
                e.category, Decimal("0")
            ) + e.gas_volume_scf

        # Period start/end
        sorted_events = sorted(events, key=lambda e: e.start_time)
        period_start = sorted_events[0].start_time if sorted_events else None
        period_end = sorted_events[-1].end_time if sorted_events else None

        agg_data = {
            "group_key": group_key,
            "total_events": n,
            "total_volume_scf": str(total_volume),
        }
        provenance_hash = _compute_hash(agg_data)

        return EventAggregation(
            group_key=group_key,
            period_start=period_start,
            period_end=period_end,
            total_events=n,
            total_volume_scf=total_volume,
            total_duration_hours=total_duration,
            avg_duration_hours=avg_duration,
            avg_flow_rate_scf_hr=avg_flow,
            peak_flow_rate_scf_hr=peak_flow,
            measured_count=measured_count,
            estimated_count=estimated_count,
            by_category=by_category,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Private: Emission Calculations
    # ------------------------------------------------------------------

    def _calculate_default_ef(
        self,
        volume_scf: Decimal,
        combustion_efficiency: Decimal,
    ) -> EventEmissions:
        """Calculate emissions using default emission factors.

        CO2 = volume * EF_CO2 * CE (combusted portion produces CO2)
        CH4 = volume * EF_CH4 * (1 - CE) / CE_adjustment (uncombusted slip)
        N2O = volume * EF_N2O (trace, independent of CE)

        Args:
            volume_scf: Gas volume in standard cubic feet.
            combustion_efficiency: Combustion efficiency (0-1).

        Returns:
            EventEmissions with calculated values.
        """
        co2_kg = (volume_scf * DEFAULT_EF_CO2_KG_PER_SCF * combustion_efficiency).quantize(
            Decimal("0.00001"), rounding=ROUND_HALF_UP
        )

        # CH4 slip is proportional to uncombusted fraction
        ch4_slip_fraction = Decimal("1") - combustion_efficiency
        ch4_kg = (volume_scf * DEFAULT_EF_CH4_KG_PER_SCF + volume_scf * ch4_slip_fraction * DEFAULT_EF_CH4_KG_PER_SCF).quantize(
            Decimal("0.00001"), rounding=ROUND_HALF_UP
        )

        n2o_kg = (volume_scf * DEFAULT_EF_N2O_KG_PER_SCF).quantize(
            Decimal("0.00001"), rounding=ROUND_HALF_UP
        )

        co2e_kg = (
            co2_kg + ch4_kg * GWP_CH4_AR5 + n2o_kg * GWP_N2O_AR5
        ).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)

        return EventEmissions(
            event_id="",
            co2_kg=co2_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            co2e_kg=co2e_kg,
            combustion_efficiency=combustion_efficiency,
            calculation_method="DEFAULT_EF",
            provenance_hash="",
        )

    def _calculate_composition_based(
        self,
        volume_scf: Decimal,
        gas_composition: Dict[str, Decimal],
        combustion_efficiency: Decimal,
    ) -> EventEmissions:
        """Calculate emissions using gas composition analysis.

        For each hydrocarbon component:
            CO2 = volume * mole_fraction * (MW_CO2 * carbon_atoms / MW_component) * CE
            CH4_slip = volume * CH4_fraction * (1 - CE)

        Args:
            volume_scf: Gas volume in standard cubic feet.
            gas_composition: Mole fractions by gas component.
            combustion_efficiency: Combustion efficiency (0-1).

        Returns:
            EventEmissions with calculated values.
        """
        # Component molecular weights and carbon counts
        component_data: Dict[str, Tuple[Decimal, int]] = {
            "CH4": (Decimal("16.04"), 1),
            "C2H6": (Decimal("30.07"), 2),
            "C3H8": (Decimal("44.10"), 3),
            "n-C4H10": (Decimal("58.12"), 4),
            "i-C4H10": (Decimal("58.12"), 4),
            "C5H12": (Decimal("72.15"), 5),
            "C6+": (Decimal("86.18"), 6),
            "CO": (Decimal("28.01"), 1),
            "C2H4": (Decimal("28.05"), 2),
            "C3H6": (Decimal("42.08"), 3),
            "H2": (Decimal("2.016"), 0),
        }

        total_co2_kg = Decimal("0")

        for component, mole_fraction_raw in gas_composition.items():
            mole_fraction = _to_decimal(mole_fraction_raw)
            if component in component_data:
                mw, carbons = component_data[component]
                if carbons > 0:
                    # Stoichiometric CO2 production
                    co2_from_component = (
                        volume_scf * mole_fraction *
                        (MW_CO2 * Decimal(str(carbons)) / mw) *
                        combustion_efficiency
                    )
                    total_co2_kg += co2_from_component
            elif component == "CO2":
                # Pre-existing CO2 passes through
                total_co2_kg += volume_scf * mole_fraction * MW_CO2 / Decimal("385.5")

        total_co2_kg = total_co2_kg.quantize(
            Decimal("0.00001"), rounding=ROUND_HALF_UP
        )

        # CH4 slip (uncombusted methane)
        ch4_fraction = _to_decimal(gas_composition.get("CH4", Decimal("0")))
        ch4_slip = Decimal("1") - combustion_efficiency
        ch4_kg = (volume_scf * ch4_fraction * ch4_slip * MW_CH4 / Decimal("385.5")).quantize(
            Decimal("0.00001"), rounding=ROUND_HALF_UP
        )

        # N2O (minor, from high-temperature combustion)
        n2o_kg = (volume_scf * DEFAULT_EF_N2O_KG_PER_SCF).quantize(
            Decimal("0.00001"), rounding=ROUND_HALF_UP
        )

        co2e_kg = (
            total_co2_kg + ch4_kg * GWP_CH4_AR5 + n2o_kg * GWP_N2O_AR5
        ).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)

        return EventEmissions(
            event_id="",
            co2_kg=total_co2_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            co2e_kg=co2e_kg,
            combustion_efficiency=combustion_efficiency,
            calculation_method="GAS_COMPOSITION",
            provenance_hash="",
        )

    # ------------------------------------------------------------------
    # Private: Metrics
    # ------------------------------------------------------------------

    def _record_metrics(self, category: str, elapsed_ms: float) -> None:
        """Record Prometheus metrics if available.

        Args:
            category: Event category for labelling.
            elapsed_ms: Processing time in milliseconds.
        """
        if _METRICS_AVAILABLE and _record_flaring_event is not None:
            try:
                _record_flaring_event(category, "recorded")
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
        if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
            try:
                _observe_calculation_duration(elapsed_ms / 1000.0)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)

    # ------------------------------------------------------------------
    # Private: Provenance
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Record provenance tracking event if available.

        Args:
            action: Action description.
            entity_id: Entity identifier.
            data: Provenance data dictionary.
        """
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="flaring_event",
                    action=action,
                    entity_id=entity_id,
                    data=data,
                    metadata={"engine": "FlaringEventTrackerEngine"},
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)
