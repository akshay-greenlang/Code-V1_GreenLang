# -*- coding: utf-8 -*-
"""
LoadFlexibilityEngine - PACK-037 Demand Response Engine 1
==========================================================

Comprehensive load flexibility assessment engine for demand response
programs.  Categorises facility loads by curtailment priority, scores
each load across seven flexibility factors, calculates curtailment
capacity matrices by notification time and event duration, and generates
a facility-wide flexibility register for DR program enrollment.

Calculation Methodology:
    Flexibility Score (0-100):
        score = sum(factor_weight_i * factor_score_i)
        where factors (weights summing to 1.0):
            capacity_weight     = 0.30  (curtailable MW fraction)
            speed_weight        = 0.15  (ramp-down speed)
            duration_weight     = 0.15  (sustainable curtailment hours)
            rebound_weight      = 0.10  (post-event demand spike)
            comfort_weight      = 0.10  (occupant comfort impact)
            operational_weight  = 0.10  (production / process impact)
            automation_weight   = 0.10  (automation readiness)

    Curtailment Capacity:
        For each (notification_time, duration) pair:
            available_kw = sum(load_kw * curtailment_pct
                               for load in eligible_loads
                               if load.ramp_time <= notification_time
                               and load.max_duration >= duration)

    Load Categories:
        CRITICAL (0)   - Never curtailed (life safety, data centres)
        ESSENTIAL (1)  - Curtailed only in emergencies
        DEFERRABLE (2) - Can be shifted to off-peak
        SHEDDABLE (3)  - Can be dropped entirely during events
        FLEXIBLE (4)   - Fully modular, continuous DR asset

Regulatory References:
    - FERC Order 2222 - Distributed Energy Resource Aggregation
    - FERC Order 745 - Demand Response Compensation
    - OpenADR 2.0 - Automated Demand Response standard
    - IEEE 2030.5 - Smart Energy Profile
    - EU Clean Energy Package - Demand Side Flexibility
    - NAESB REQ.18 - DR Communication Standards
    - ISO 50001:2018 - Energy management systems

Zero-Hallucination:
    - All flexibility scores use deterministic weighted-sum formulas
    - Curtailment matrices built from explicit load parameters only
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-037 Demand Response
Engine:  1 of 8
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LoadCategory(int, Enum):
    """Load curtailment priority category.

    CRITICAL (0):   Never curtailed -- life safety, data integrity.
    ESSENTIAL (1):  Curtailed only during grid emergencies.
    DEFERRABLE (2): Can be shifted to off-peak periods.
    SHEDDABLE (3):  Can be dropped entirely during DR events.
    FLEXIBLE (4):   Fully modular, continuous DR asset.
    """
    CRITICAL = 0
    ESSENTIAL = 1
    DEFERRABLE = 2
    SHEDDABLE = 3
    FLEXIBLE = 4


class LoadType(str, Enum):
    """Type of electrical load within a facility.

    HVAC:          Heating, ventilation, and air conditioning.
    LIGHTING:      Interior and exterior lighting.
    PLUG_LOAD:     Office equipment, appliances, outlets.
    MOTOR:         Industrial motors and drives.
    PROCESS:       Manufacturing / process loads.
    REFRIGERATION: Commercial / industrial refrigeration.
    EV_CHARGING:   Electric vehicle charging infrastructure.
    WATER_HEATING: Domestic / process water heating.
    DER:           Distributed energy resources (batteries, solar).
    """
    HVAC = "hvac"
    LIGHTING = "lighting"
    PLUG_LOAD = "plug_load"
    MOTOR = "motor"
    PROCESS = "process"
    REFRIGERATION = "refrigeration"
    EV_CHARGING = "ev_charging"
    WATER_HEATING = "water_heating"
    DER = "der"


class NotificationTime(str, Enum):
    """Lead time before a DR event begins.

    IMMEDIATE:   0 minutes (automated signal).
    TEN_MIN:     10-minute advance notice.
    THIRTY_MIN:  30-minute advance notice.
    TWO_HOUR:    2-hour advance notice.
    DAY_AHEAD:   Day-ahead (12-24 hour) notice.
    """
    IMMEDIATE = "immediate"
    TEN_MIN = "10_min"
    THIRTY_MIN = "30_min"
    TWO_HOUR = "2_hour"
    DAY_AHEAD = "day_ahead"


class AutomationLevel(str, Enum):
    """Automation readiness for demand response.

    MANUAL:       Requires on-site personnel to curtail.
    SEMI_AUTO:    Automated signal, manual confirmation required.
    FULLY_AUTO:   Fully automated curtailment via OpenADR / BMS.
    AUTONOMOUS:   AI-driven autonomous dispatch capability.
    """
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    FULLY_AUTO = "fully_auto"
    AUTONOMOUS = "autonomous"


class FlexibilityGrade(str, Enum):
    """Overall flexibility grade assigned to a load or facility.

    A_EXCELLENT:  Score 80-100, fully DR-ready.
    B_GOOD:       Score 60-79, good DR candidate.
    C_MODERATE:   Score 40-59, limited DR potential.
    D_LOW:        Score 20-39, marginal DR candidate.
    F_MINIMAL:    Score 0-19, not suitable for DR.
    """
    A_EXCELLENT = "a_excellent"
    B_GOOD = "b_good"
    C_MODERATE = "c_moderate"
    D_LOW = "d_low"
    F_MINIMAL = "f_minimal"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Flexibility scoring weights (must sum to 1.0).
FLEXIBILITY_WEIGHTS: Dict[str, Decimal] = {
    "capacity": Decimal("0.30"),
    "speed": Decimal("0.15"),
    "duration": Decimal("0.15"),
    "rebound": Decimal("0.10"),
    "comfort": Decimal("0.10"),
    "operational": Decimal("0.10"),
    "automation": Decimal("0.10"),
}

# Notification time in minutes for matching.
NOTIFICATION_MINUTES: Dict[str, int] = {
    NotificationTime.IMMEDIATE.value: 0,
    NotificationTime.TEN_MIN.value: 10,
    NotificationTime.THIRTY_MIN.value: 30,
    NotificationTime.TWO_HOUR.value: 120,
    NotificationTime.DAY_AHEAD.value: 1440,
}

# Standard event durations (hours) for curtailment matrix.
EVENT_DURATIONS_HOURS: List[int] = [1, 2, 4, 6, 8]

# Automation level scores (0-100).
AUTOMATION_SCORES: Dict[str, Decimal] = {
    AutomationLevel.MANUAL.value: Decimal("20"),
    AutomationLevel.SEMI_AUTO.value: Decimal("50"),
    AutomationLevel.FULLY_AUTO.value: Decimal("85"),
    AutomationLevel.AUTONOMOUS.value: Decimal("100"),
}

# Default curtailment percentages by load category.
DEFAULT_CURTAILMENT_PCT: Dict[int, Decimal] = {
    LoadCategory.CRITICAL.value: Decimal("0"),
    LoadCategory.ESSENTIAL.value: Decimal("0.15"),
    LoadCategory.DEFERRABLE.value: Decimal("0.50"),
    LoadCategory.SHEDDABLE.value: Decimal("0.90"),
    LoadCategory.FLEXIBLE.value: Decimal("1.00"),
}

# Default ramp-down time (minutes) by load type.
DEFAULT_RAMP_MINUTES: Dict[str, int] = {
    LoadType.HVAC.value: 5,
    LoadType.LIGHTING.value: 1,
    LoadType.PLUG_LOAD.value: 2,
    LoadType.MOTOR.value: 10,
    LoadType.PROCESS.value: 30,
    LoadType.REFRIGERATION.value: 15,
    LoadType.EV_CHARGING.value: 1,
    LoadType.WATER_HEATING.value: 5,
    LoadType.DER.value: 1,
}

# Default maximum sustainable curtailment duration (hours) by load type.
DEFAULT_MAX_DURATION_HOURS: Dict[str, int] = {
    LoadType.HVAC.value: 4,
    LoadType.LIGHTING.value: 8,
    LoadType.PLUG_LOAD.value: 8,
    LoadType.MOTOR.value: 2,
    LoadType.PROCESS.value: 1,
    LoadType.REFRIGERATION.value: 2,
    LoadType.EV_CHARGING.value: 8,
    LoadType.WATER_HEATING.value: 6,
    LoadType.DER.value: 4,
}

# Rebound multiplier by load type (post-event demand spike as fraction
# of curtailed load).
DEFAULT_REBOUND_FACTOR: Dict[str, Decimal] = {
    LoadType.HVAC.value: Decimal("0.30"),
    LoadType.LIGHTING.value: Decimal("0.05"),
    LoadType.PLUG_LOAD.value: Decimal("0.10"),
    LoadType.MOTOR.value: Decimal("0.15"),
    LoadType.PROCESS.value: Decimal("0.20"),
    LoadType.REFRIGERATION.value: Decimal("0.35"),
    LoadType.EV_CHARGING.value: Decimal("0.25"),
    LoadType.WATER_HEATING.value: Decimal("0.20"),
    LoadType.DER.value: Decimal("0.05"),
}

# Grade thresholds.
GRADE_THRESHOLDS: List[Tuple[Decimal, FlexibilityGrade]] = [
    (Decimal("80"), FlexibilityGrade.A_EXCELLENT),
    (Decimal("60"), FlexibilityGrade.B_GOOD),
    (Decimal("40"), FlexibilityGrade.C_MODERATE),
    (Decimal("20"), FlexibilityGrade.D_LOW),
    (Decimal("0"), FlexibilityGrade.F_MINIMAL),
]


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class LoadProfile(BaseModel):
    """Profile of a single controllable load within a facility.

    Attributes:
        load_id: Unique load identifier.
        name: Human-readable load name.
        load_type: Type of load (HVAC, Lighting, etc.).
        category: Curtailment priority category.
        rated_kw: Rated / nameplate power (kW).
        typical_kw: Typical operating power (kW).
        curtailment_pct: Maximum curtailment fraction (0.0 to 1.0).
        ramp_down_minutes: Time to reach curtailed state (minutes).
        max_curtailment_hours: Maximum sustainable curtailment (hours).
        rebound_factor: Post-event demand spike as fraction of curtailed.
        automation_level: Automation readiness.
        comfort_impact_score: Occupant comfort impact (0-100, 0=none).
        operational_impact_score: Production impact (0-100, 0=none).
        is_seasonal: Whether load is seasonal.
        notes: Additional notes.
    """
    load_id: str = Field(
        default_factory=_new_uuid, description="Unique load identifier"
    )
    name: str = Field(
        default="", max_length=500, description="Load name"
    )
    load_type: LoadType = Field(
        default=LoadType.PLUG_LOAD, description="Type of load"
    )
    category: LoadCategory = Field(
        default=LoadCategory.DEFERRABLE, description="Curtailment priority"
    )
    rated_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Rated power (kW)"
    )
    typical_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Typical operating power (kW)"
    )
    curtailment_pct: Decimal = Field(
        default=Decimal("0.50"), ge=0, le=Decimal("1"),
        description="Maximum curtailment fraction (0-1)"
    )
    ramp_down_minutes: int = Field(
        default=5, ge=0, le=1440, description="Ramp-down time (minutes)"
    )
    max_curtailment_hours: int = Field(
        default=4, ge=0, le=24, description="Max sustained curtailment (hours)"
    )
    rebound_factor: Decimal = Field(
        default=Decimal("0.20"), ge=0, le=Decimal("1"),
        description="Post-event rebound factor (0-1)"
    )
    automation_level: AutomationLevel = Field(
        default=AutomationLevel.SEMI_AUTO, description="Automation readiness"
    )
    comfort_impact_score: Decimal = Field(
        default=Decimal("30"), ge=0, le=Decimal("100"),
        description="Comfort impact (0=none, 100=severe)"
    )
    operational_impact_score: Decimal = Field(
        default=Decimal("20"), ge=0, le=Decimal("100"),
        description="Operational impact (0=none, 100=severe)"
    )
    is_seasonal: bool = Field(
        default=False, description="Whether load is seasonal"
    )
    notes: str = Field(default="", max_length=2000, description="Notes")

    @field_validator("load_id")
    @classmethod
    def validate_load_id(cls, v: str) -> str:
        """Ensure load_id is non-empty."""
        if not v or not v.strip():
            return _new_uuid()
        return v

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v: Any) -> Any:
        """Accept integer values for LoadCategory."""
        if isinstance(v, int):
            valid = {c.value for c in LoadCategory}
            if v not in valid:
                raise ValueError(
                    f"Invalid load category {v}. Must be one of: {sorted(valid)}"
                )
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class FlexibilityAssessment(BaseModel):
    """Flexibility assessment result for a single load.

    Attributes:
        load_id: Load identifier.
        name: Load name.
        load_type: Type of load.
        category: Load curtailment category.
        flexibility_score: Weighted flexibility score (0-100).
        grade: Flexibility grade (A-F).
        capacity_score: Capacity factor sub-score (0-100).
        speed_score: Ramp speed sub-score (0-100).
        duration_score: Duration sub-score (0-100).
        rebound_score: Rebound sub-score (0-100).
        comfort_score: Comfort sub-score (0-100).
        operational_score: Operational sub-score (0-100).
        automation_score: Automation sub-score (0-100).
        curtailable_kw: Maximum curtailable power (kW).
        recommended_actions: List of improvement recommendations.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    load_id: str = Field(default="", description="Load ID")
    name: str = Field(default="", description="Load name")
    load_type: LoadType = Field(default=LoadType.PLUG_LOAD)
    category: LoadCategory = Field(default=LoadCategory.DEFERRABLE)
    flexibility_score: Decimal = Field(
        default=Decimal("0"), description="Weighted flexibility score (0-100)"
    )
    grade: FlexibilityGrade = Field(
        default=FlexibilityGrade.F_MINIMAL, description="Flexibility grade"
    )
    capacity_score: Decimal = Field(default=Decimal("0"))
    speed_score: Decimal = Field(default=Decimal("0"))
    duration_score: Decimal = Field(default=Decimal("0"))
    rebound_score: Decimal = Field(default=Decimal("0"))
    comfort_score: Decimal = Field(default=Decimal("0"))
    operational_score: Decimal = Field(default=Decimal("0"))
    automation_score: Decimal = Field(default=Decimal("0"))
    curtailable_kw: Decimal = Field(
        default=Decimal("0"), description="Maximum curtailable power (kW)"
    )
    recommended_actions: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class CurtailmentCapacity(BaseModel):
    """Curtailment capacity matrix for a facility.

    Attributes:
        facility_id: Facility identifier.
        matrix: Dict keyed by notification_time -> duration_hours -> kW.
        total_load_kw: Total facility load (kW).
        max_curtailable_kw: Maximum curtailable across all scenarios.
        max_curtailment_pct: Maximum curtailment as fraction of total load.
        fastest_response_minutes: Fastest achievable response time.
        longest_duration_hours: Longest sustainable event duration.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    facility_id: str = Field(default="", description="Facility ID")
    matrix: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description="Notification time -> duration -> curtailable kW"
    )
    total_load_kw: Decimal = Field(default=Decimal("0"))
    max_curtailable_kw: Decimal = Field(default=Decimal("0"))
    max_curtailment_pct: Decimal = Field(default=Decimal("0"))
    fastest_response_minutes: int = Field(default=0)
    longest_duration_hours: int = Field(default=0)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class FlexibilityRegister(BaseModel):
    """Complete flexibility register for a facility.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        assessments: Individual load assessments.
        curtailment_capacity: Curtailment capacity matrix.
        total_load_kw: Total facility load (kW).
        total_curtailable_kw: Total curtailable capacity (kW).
        overall_flexibility_score: Facility-level weighted score (0-100).
        overall_grade: Facility-level flexibility grade.
        load_count: Total number of loads assessed.
        dr_ready_count: Number of loads scoring >= 60.
        category_summary: Loads by curtailment category.
        type_summary: Loads by load type.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    facility_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="", max_length=500)
    assessments: List[FlexibilityAssessment] = Field(default_factory=list)
    curtailment_capacity: CurtailmentCapacity = Field(
        default_factory=CurtailmentCapacity
    )
    total_load_kw: Decimal = Field(default=Decimal("0"))
    total_curtailable_kw: Decimal = Field(default=Decimal("0"))
    overall_flexibility_score: Decimal = Field(default=Decimal("0"))
    overall_grade: FlexibilityGrade = Field(
        default=FlexibilityGrade.F_MINIMAL
    )
    load_count: int = Field(default=0)
    dr_ready_count: int = Field(default=0)
    category_summary: Dict[str, int] = Field(default_factory=dict)
    type_summary: Dict[str, int] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LoadFlexibilityEngine:
    """Load flexibility assessment engine for demand response programs.

    Assesses individual loads and entire facilities for DR participation
    readiness.  Produces flexibility scores, curtailment capacity matrices,
    and a comprehensive flexibility register.  All calculations use
    deterministic Decimal arithmetic with SHA-256 provenance hashing.

    Usage::

        engine = LoadFlexibilityEngine()
        load = LoadProfile(
            name="Rooftop AHU-1",
            load_type=LoadType.HVAC,
            category=LoadCategory.DEFERRABLE,
            rated_kw=Decimal("150"),
            typical_kw=Decimal("120"),
        )
        result = engine.assess_load(load)
        print(f"Flexibility score: {result.flexibility_score}")
        print(f"Grade: {result.grade.value}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise LoadFlexibilityEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - flexibility_weights (dict): override factor weights
                - curtailment_pct (dict): override default curtailment %
                - rebound_factors (dict): override default rebound factors
        """
        self.config = config or {}
        self._weights = dict(FLEXIBILITY_WEIGHTS)
        if "flexibility_weights" in self.config:
            self._weights.update(self.config["flexibility_weights"])
        self._curtailment_pct = dict(DEFAULT_CURTAILMENT_PCT)
        if "curtailment_pct" in self.config:
            self._curtailment_pct.update(self.config["curtailment_pct"])
        self._rebound_factors = dict(DEFAULT_REBOUND_FACTOR)
        if "rebound_factors" in self.config:
            self._rebound_factors.update(self.config["rebound_factors"])
        logger.info(
            "LoadFlexibilityEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def assess_load(self, load: LoadProfile) -> FlexibilityAssessment:
        """Assess a single load for DR flexibility.

        Calculates seven factor sub-scores, computes the weighted
        flexibility score, assigns a grade, and generates improvement
        recommendations.

        Args:
            load: Load profile to assess.

        Returns:
            FlexibilityAssessment with scores and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Assessing load: %s (type=%s, category=%s, kW=%s)",
            load.name, load.load_type.value,
            load.category.name, load.typical_kw,
        )

        # Critical loads score zero by definition
        if load.category == LoadCategory.CRITICAL:
            result = FlexibilityAssessment(
                load_id=load.load_id,
                name=load.name,
                load_type=load.load_type,
                category=load.category,
                flexibility_score=Decimal("0"),
                grade=FlexibilityGrade.F_MINIMAL,
                curtailable_kw=Decimal("0"),
                recommended_actions=[
                    "Critical load - not eligible for demand response."
                ],
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Sub-scores
        capacity_score = self._score_capacity(load)
        speed_score = self._score_speed(load)
        duration_score = self._score_duration(load)
        rebound_score = self._score_rebound(load)
        comfort_score = self._score_comfort(load)
        operational_score = self._score_operational(load)
        automation_score = self._score_automation(load)

        # Weighted total
        weighted = (
            self._weights["capacity"] * capacity_score
            + self._weights["speed"] * speed_score
            + self._weights["duration"] * duration_score
            + self._weights["rebound"] * rebound_score
            + self._weights["comfort"] * comfort_score
            + self._weights["operational"] * operational_score
            + self._weights["automation"] * automation_score
        )

        grade = self._assign_grade(weighted)
        curtailable_kw = load.typical_kw * load.curtailment_pct
        recommendations = self._generate_recommendations(
            load, capacity_score, speed_score, duration_score,
            rebound_score, comfort_score, operational_score,
            automation_score,
        )

        result = FlexibilityAssessment(
            load_id=load.load_id,
            name=load.name,
            load_type=load.load_type,
            category=load.category,
            flexibility_score=_round_val(weighted, 2),
            grade=grade,
            capacity_score=_round_val(capacity_score, 2),
            speed_score=_round_val(speed_score, 2),
            duration_score=_round_val(duration_score, 2),
            rebound_score=_round_val(rebound_score, 2),
            comfort_score=_round_val(comfort_score, 2),
            operational_score=_round_val(operational_score, 2),
            automation_score=_round_val(automation_score, 2),
            curtailable_kw=_round_val(curtailable_kw, 2),
            recommended_actions=recommendations,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Load assessed: %s, score=%.1f, grade=%s, curtailable=%.1f kW, "
            "hash=%s (%.1f ms)",
            load.name, float(weighted), grade.value,
            float(curtailable_kw), result.provenance_hash[:16], elapsed,
        )
        return result

    def assess_facility(
        self,
        facility_id: str,
        facility_name: str,
        loads: List[LoadProfile],
    ) -> FlexibilityRegister:
        """Assess all loads in a facility and produce a flexibility register.

        Args:
            facility_id: Unique facility identifier.
            facility_name: Facility name.
            loads: List of load profiles.

        Returns:
            FlexibilityRegister with all assessments and summaries.
        """
        t0 = time.perf_counter()
        logger.info(
            "Assessing facility: %s (%d loads)", facility_name, len(loads)
        )

        assessments: List[FlexibilityAssessment] = []
        for load in loads:
            assessment = self.assess_load(load)
            assessments.append(assessment)

        # Summaries
        total_kw = sum((_decimal(l.typical_kw) for l in loads), Decimal("0"))
        total_curtailable = sum(
            (a.curtailable_kw for a in assessments), Decimal("0")
        )
        dr_ready = sum(1 for a in assessments if a.flexibility_score >= Decimal("60"))

        # Category summary
        cat_summary: Dict[str, int] = {}
        for load in loads:
            cat_name = load.category.name
            cat_summary[cat_name] = cat_summary.get(cat_name, 0) + 1

        # Type summary
        type_summary: Dict[str, int] = {}
        for load in loads:
            type_name = load.load_type.value
            type_summary[type_name] = type_summary.get(type_name, 0) + 1

        # Overall score (kW-weighted average of non-critical loads)
        overall_score = self._calculate_overall_score(loads, assessments)
        overall_grade = self._assign_grade(overall_score)

        # Curtailment capacity matrix
        curtailment = self.calculate_curtailment_capacity(
            facility_id, loads
        )

        result = FlexibilityRegister(
            facility_id=facility_id,
            facility_name=facility_name,
            assessments=assessments,
            curtailment_capacity=curtailment,
            total_load_kw=_round_val(total_kw, 2),
            total_curtailable_kw=_round_val(total_curtailable, 2),
            overall_flexibility_score=_round_val(overall_score, 2),
            overall_grade=overall_grade,
            load_count=len(loads),
            dr_ready_count=dr_ready,
            category_summary=cat_summary,
            type_summary=type_summary,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Facility assessed: %s, %d loads, overall=%.1f (%s), "
            "curtailable=%.1f kW, DR-ready=%d, hash=%s (%.1f ms)",
            facility_name, len(loads), float(overall_score),
            overall_grade.value, float(total_curtailable), dr_ready,
            result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_curtailment_capacity(
        self,
        facility_id: str,
        loads: List[LoadProfile],
    ) -> CurtailmentCapacity:
        """Build the curtailment capacity matrix for a facility.

        For each combination of notification time and event duration,
        calculates the total curtailable kW from eligible loads.

        Args:
            facility_id: Facility identifier.
            loads: List of load profiles.

        Returns:
            CurtailmentCapacity with the complete matrix.
        """
        t0 = time.perf_counter()

        total_kw = sum((_decimal(l.typical_kw) for l in loads), Decimal("0"))
        max_curtailable = Decimal("0")
        fastest_response = 1440
        longest_duration = 0

        matrix: Dict[str, Dict[str, Decimal]] = {}

        for notif_key, notif_minutes in NOTIFICATION_MINUTES.items():
            duration_dict: Dict[str, Decimal] = {}

            for dur_hours in EVENT_DURATIONS_HOURS:
                available = Decimal("0")

                for load in loads:
                    # Critical loads are never curtailed
                    if load.category == LoadCategory.CRITICAL:
                        continue

                    # Essential loads only with day-ahead notice
                    if (load.category == LoadCategory.ESSENTIAL
                            and notif_key != NotificationTime.DAY_AHEAD.value):
                        continue

                    # Check ramp time fits within notification
                    if load.ramp_down_minutes > notif_minutes:
                        # For immediate, allow loads with ramp <= 2 min
                        if notif_key == NotificationTime.IMMEDIATE.value:
                            if load.ramp_down_minutes > 2:
                                continue
                        else:
                            continue

                    # Check duration is sustainable
                    if load.max_curtailment_hours < dur_hours:
                        continue

                    # Add curtailable capacity
                    load_curtailable = load.typical_kw * load.curtailment_pct
                    available += load_curtailable

                    # Track fastest / longest
                    if load_curtailable > Decimal("0"):
                        fastest_response = min(
                            fastest_response, load.ramp_down_minutes
                        )
                        longest_duration = max(
                            longest_duration, load.max_curtailment_hours
                        )

                duration_dict[str(dur_hours)] = _round_val(available, 2)
                max_curtailable = max(max_curtailable, available)

            matrix[notif_key] = duration_dict

        max_pct = _safe_pct(max_curtailable, total_kw)

        result = CurtailmentCapacity(
            facility_id=facility_id,
            matrix=matrix,
            total_load_kw=_round_val(total_kw, 2),
            max_curtailable_kw=_round_val(max_curtailable, 2),
            max_curtailment_pct=_round_val(max_pct, 2),
            fastest_response_minutes=fastest_response if fastest_response < 1440 else 0,
            longest_duration_hours=longest_duration,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Curtailment matrix: facility=%s, max=%.1f kW (%.1f%%), "
            "hash=%s (%.1f ms)",
            facility_id, float(max_curtailable), float(max_pct),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def generate_flexibility_register(
        self,
        facility_id: str,
        facility_name: str,
        loads: List[LoadProfile],
    ) -> FlexibilityRegister:
        """Generate a complete flexibility register for DR enrollment.

        Alias for assess_facility that emphasises the enrollment use case.

        Args:
            facility_id: Unique facility identifier.
            facility_name: Facility name.
            loads: List of load profiles.

        Returns:
            FlexibilityRegister suitable for DR program enrollment.
        """
        return self.assess_facility(facility_id, facility_name, loads)

    # ------------------------------------------------------------------ #
    # Factor Scoring Methods                                               #
    # ------------------------------------------------------------------ #

    def _score_capacity(self, load: LoadProfile) -> Decimal:
        """Score the capacity factor (0-100).

        Higher scores for larger curtailable fractions and larger loads.
        Formula: curtailment_pct * 80 + min(typical_kw / 500, 1) * 20

        Args:
            load: Load profile.

        Returns:
            Capacity sub-score (0-100).
        """
        pct_component = load.curtailment_pct * Decimal("80")
        size_component = min(
            _safe_divide(load.typical_kw, Decimal("500")), Decimal("1")
        ) * Decimal("20")
        return min(pct_component + size_component, Decimal("100"))

    def _score_speed(self, load: LoadProfile) -> Decimal:
        """Score the ramp-down speed factor (0-100).

        Faster ramp = higher score.
        Formula: max(0, 100 - ramp_minutes * 2)

        Args:
            load: Load profile.

        Returns:
            Speed sub-score (0-100).
        """
        score = Decimal("100") - _decimal(load.ramp_down_minutes) * Decimal("2")
        return max(score, Decimal("0"))

    def _score_duration(self, load: LoadProfile) -> Decimal:
        """Score the sustainable duration factor (0-100).

        Longer sustainable curtailment = higher score.
        Formula: min(max_hours / 8 * 100, 100)

        Args:
            load: Load profile.

        Returns:
            Duration sub-score (0-100).
        """
        score = _safe_divide(
            _decimal(load.max_curtailment_hours) * Decimal("100"),
            Decimal("8"),
        )
        return min(score, Decimal("100"))

    def _score_rebound(self, load: LoadProfile) -> Decimal:
        """Score the rebound factor (0-100).

        Lower rebound = higher score (inverted).
        Formula: (1 - rebound_factor) * 100

        Args:
            load: Load profile.

        Returns:
            Rebound sub-score (0-100).
        """
        return (Decimal("1") - load.rebound_factor) * Decimal("100")

    def _score_comfort(self, load: LoadProfile) -> Decimal:
        """Score the comfort impact factor (0-100).

        Lower comfort impact = higher score (inverted).
        Formula: 100 - comfort_impact_score

        Args:
            load: Load profile.

        Returns:
            Comfort sub-score (0-100).
        """
        return Decimal("100") - load.comfort_impact_score

    def _score_operational(self, load: LoadProfile) -> Decimal:
        """Score the operational impact factor (0-100).

        Lower operational impact = higher score (inverted).
        Formula: 100 - operational_impact_score

        Args:
            load: Load profile.

        Returns:
            Operational sub-score (0-100).
        """
        return Decimal("100") - load.operational_impact_score

    def _score_automation(self, load: LoadProfile) -> Decimal:
        """Score the automation readiness factor (0-100).

        Uses lookup table from AutomationLevel.

        Args:
            load: Load profile.

        Returns:
            Automation sub-score (0-100).
        """
        return AUTOMATION_SCORES.get(
            load.automation_level.value, Decimal("20")
        )

    # ------------------------------------------------------------------ #
    # Grade Assignment                                                     #
    # ------------------------------------------------------------------ #

    def _assign_grade(self, score: Decimal) -> FlexibilityGrade:
        """Assign a flexibility grade based on score thresholds.

        Args:
            score: Flexibility score (0-100).

        Returns:
            Corresponding FlexibilityGrade.
        """
        for threshold, grade in GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return FlexibilityGrade.F_MINIMAL

    # ------------------------------------------------------------------ #
    # Overall Facility Score                                               #
    # ------------------------------------------------------------------ #

    def _calculate_overall_score(
        self,
        loads: List[LoadProfile],
        assessments: List[FlexibilityAssessment],
    ) -> Decimal:
        """Calculate kW-weighted average flexibility score for the facility.

        Excludes critical loads from the weighting.

        Args:
            loads: Load profiles.
            assessments: Corresponding assessments.

        Returns:
            Overall flexibility score (0-100).
        """
        weighted_sum = Decimal("0")
        weight_total = Decimal("0")

        for load, assessment in zip(loads, assessments):
            if load.category == LoadCategory.CRITICAL:
                continue
            kw = _decimal(load.typical_kw)
            weighted_sum += assessment.flexibility_score * kw
            weight_total += kw

        return _safe_divide(weighted_sum, weight_total, Decimal("0"))

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        load: LoadProfile,
        capacity: Decimal,
        speed: Decimal,
        duration: Decimal,
        rebound: Decimal,
        comfort: Decimal,
        operational: Decimal,
        automation: Decimal,
    ) -> List[str]:
        """Generate improvement recommendations based on sub-scores.

        Args:
            load: Load profile.
            capacity: Capacity sub-score.
            speed: Speed sub-score.
            duration: Duration sub-score.
            rebound: Rebound sub-score.
            comfort: Comfort sub-score.
            operational: Operational sub-score.
            automation: Automation sub-score.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []
        threshold = Decimal("50")

        if capacity < threshold:
            recs.append(
                "Increase curtailment fraction by installing variable-speed "
                "drives or staged control."
            )

        if speed < threshold:
            recs.append(
                "Reduce ramp-down time by pre-staging equipment or adding "
                "fast-acting controls."
            )

        if duration < threshold:
            recs.append(
                "Extend sustainable curtailment duration with thermal "
                "storage or load rotation strategies."
            )

        if rebound < threshold:
            recs.append(
                "Mitigate post-event rebound with graduated ramp-up "
                "sequences or thermal pre-conditioning."
            )

        if comfort < threshold:
            recs.append(
                "Reduce occupant comfort impact by widening temperature "
                "deadbands gradually or using task lighting."
            )

        if operational < threshold:
            recs.append(
                "Reduce operational impact by scheduling curtailment "
                "during low-production periods or adding buffer capacity."
            )

        if automation < threshold:
            recs.append(
                "Upgrade to OpenADR 2.0 or BACnet-based automated "
                "curtailment to improve response time and reliability."
            )

        if not recs:
            recs.append(
                "Load is well-positioned for DR participation. "
                "Consider enrolling in multiple DR programs."
            )

        return recs
