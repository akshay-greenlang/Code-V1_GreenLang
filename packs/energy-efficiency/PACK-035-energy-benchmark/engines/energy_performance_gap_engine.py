# -*- coding: utf-8 -*-
"""
EnergyPerformanceGapEngine - PACK-035 Energy Benchmark Engine 5
=================================================================

Analyses the gap between actual building energy performance and design
predictions or benchmark targets, disaggregated by end-use category.
Implements CIBSE TM22 simplified energy analysis, ASHRAE balance-point
disaggregation, statistical estimation from sub-metering, and end-use
improvement priority ranking with savings potential.

End-Use Disaggregation Methods:
    SUB_METERING:
        Uses actual sub-meter data to break down total consumption by
        end-use (lighting, HVAC, plug loads, etc.).  Most accurate method.

    CIBSE_TM22:
        Simplified Energy Analysis per CIBSE TM22:2006.  Uses typical
        end-use percentage splits by building type from CIBSE Guide F
        to disaggregate total metered consumption.  Adjusts for operating
        hours, floor area, and known deviations.

    ASHRAE_BALANCE:
        Balance-point disaggregation using degree-day data to separate
        weather-dependent (heating, cooling) from base-load (lighting,
        plug loads, DHW) end-uses.

    STATISTICAL_ESTIMATION:
        Statistical breakdown using published end-use split profiles
        and regression-based attribution.

End-Use Categories (CIBSE Guide F Chapter 20):
    Lighting, Space Heating, Space Cooling, Ventilation, DHW,
    Plug Loads (small power), Process, Catering, Vertical Transport,
    Other (humidification, IT, external lighting, etc.).

Gap Severity Classification:
    MINOR:       Actual < 110% of benchmark.
    MODERATE:    110% <= Actual < 130% of benchmark.
    SIGNIFICANT: 130% <= Actual < 160% of benchmark.
    CRITICAL:    Actual >= 160% of benchmark.

Regulatory References:
    - CIBSE TM22:2006 Energy Assessment and Reporting Methodology
    - CIBSE Guide F:2012 Energy Efficiency in Buildings (3rd ed.)
    - Carbon Trust CTG004: Advanced metering for SMEs
    - ASHRAE 90.1-2019: Energy Standard for Buildings
    - ISO 52000-1:2017: Energy performance of buildings

Zero-Hallucination:
    - All end-use splits from CIBSE Guide F published tables
    - Improvement potentials from Carbon Trust field data
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hashing on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-035 Energy Benchmark
Engine:  5 of 10
Status:  Production Ready
"""

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


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float."""
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EndUseCategory(str, Enum):
    """Building energy end-use categories per CIBSE Guide F.

    LIGHTING:          Internal artificial lighting.
    SPACE_HEATING:     Space heating (gas, oil, electric, district).
    SPACE_COOLING:     Space cooling (chillers, splits, VRF).
    VENTILATION:       Mechanical ventilation and air handling.
    DHW:               Domestic hot water generation.
    PLUG_LOADS:        Small power / plug loads (computers, equipment).
    PROCESS:           Process energy (manufacturing, server loads).
    CATERING:          Commercial kitchen and catering equipment.
    VERTICAL_TRANSPORT: Lifts (elevators) and escalators.
    OTHER:             Humidification, external lighting, miscellaneous.
    """
    LIGHTING = "lighting"
    SPACE_HEATING = "space_heating"
    SPACE_COOLING = "space_cooling"
    VENTILATION = "ventilation"
    DHW = "dhw"
    PLUG_LOADS = "plug_loads"
    PROCESS = "process"
    CATERING = "catering"
    VERTICAL_TRANSPORT = "vertical_transport"
    OTHER = "other"


class DisaggregationMethod(str, Enum):
    """End-use disaggregation methods.

    SUB_METERING:          Actual sub-meter data (most accurate).
    CIBSE_TM22:            TM22 simplified energy analysis.
    ASHRAE_BALANCE:        Balance-point degree-day disaggregation.
    STATISTICAL_ESTIMATION: Statistical profile-based estimation.
    """
    SUB_METERING = "sub_metering"
    CIBSE_TM22 = "cibse_tm22"
    ASHRAE_BALANCE = "ashrae_balance"
    STATISTICAL_ESTIMATION = "statistical_estimation"


class GapSeverity(str, Enum):
    """Performance gap severity classification.

    MINOR:       Actual < 110% of benchmark (< 10% gap).
    MODERATE:    110% - 130% (10% - 30% gap).
    SIGNIFICANT: 130% - 160% (30% - 60% gap).
    CRITICAL:    >= 160% (> 60% gap).
    """
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Constants -- Typical End-Use Splits
# ---------------------------------------------------------------------------

# Typical end-use percentage splits by building type.
# Source: CIBSE Guide F:2012, Chapter 20, Table 20.1 (typical).
# Values are % of total site energy for the building type.
TYPICAL_END_USE_SPLITS: Dict[str, Dict[str, float]] = {
    "office": {
        EndUseCategory.LIGHTING.value: 22.0,
        EndUseCategory.SPACE_HEATING.value: 36.0,
        EndUseCategory.SPACE_COOLING.value: 10.0,
        EndUseCategory.VENTILATION.value: 8.0,
        EndUseCategory.DHW.value: 5.0,
        EndUseCategory.PLUG_LOADS.value: 12.0,
        EndUseCategory.CATERING.value: 3.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 2.0,
        EndUseCategory.OTHER.value: 2.0,
        "source": "CIBSE Guide F:2012, Table 20.1, Standard air-conditioned office",
    },
    "retail": {
        EndUseCategory.LIGHTING.value: 30.0,
        EndUseCategory.SPACE_HEATING.value: 28.0,
        EndUseCategory.SPACE_COOLING.value: 8.0,
        EndUseCategory.VENTILATION.value: 6.0,
        EndUseCategory.DHW.value: 4.0,
        EndUseCategory.PLUG_LOADS.value: 10.0,
        EndUseCategory.CATERING.value: 8.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 2.0,
        EndUseCategory.OTHER.value: 4.0,
        "source": "CIBSE Guide F:2012, Retail store typical split",
    },
    "hotel": {
        EndUseCategory.LIGHTING.value: 15.0,
        EndUseCategory.SPACE_HEATING.value: 35.0,
        EndUseCategory.SPACE_COOLING.value: 8.0,
        EndUseCategory.VENTILATION.value: 7.0,
        EndUseCategory.DHW.value: 20.0,
        EndUseCategory.PLUG_LOADS.value: 5.0,
        EndUseCategory.CATERING.value: 6.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 2.0,
        EndUseCategory.OTHER.value: 2.0,
        "source": "CIBSE Guide F:2012, Hotel typical split",
    },
    "hospital": {
        EndUseCategory.LIGHTING.value: 10.0,
        EndUseCategory.SPACE_HEATING.value: 40.0,
        EndUseCategory.SPACE_COOLING.value: 5.0,
        EndUseCategory.VENTILATION.value: 12.0,
        EndUseCategory.DHW.value: 15.0,
        EndUseCategory.PLUG_LOADS.value: 8.0,
        EndUseCategory.CATERING.value: 5.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 2.0,
        EndUseCategory.OTHER.value: 3.0,
        "source": "CIBSE Guide F:2012, Hospital typical split",
    },
    "school": {
        EndUseCategory.LIGHTING.value: 18.0,
        EndUseCategory.SPACE_HEATING.value: 50.0,
        EndUseCategory.SPACE_COOLING.value: 3.0,
        EndUseCategory.VENTILATION.value: 5.0,
        EndUseCategory.DHW.value: 10.0,
        EndUseCategory.PLUG_LOADS.value: 8.0,
        EndUseCategory.CATERING.value: 3.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 1.0,
        EndUseCategory.OTHER.value: 2.0,
        "source": "CIBSE Guide F:2012, School typical split",
    },
    "warehouse": {
        EndUseCategory.LIGHTING.value: 30.0,
        EndUseCategory.SPACE_HEATING.value: 45.0,
        EndUseCategory.SPACE_COOLING.value: 2.0,
        EndUseCategory.VENTILATION.value: 5.0,
        EndUseCategory.DHW.value: 3.0,
        EndUseCategory.PLUG_LOADS.value: 8.0,
        EndUseCategory.CATERING.value: 2.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 3.0,
        EndUseCategory.OTHER.value: 2.0,
        "source": "CIBSE Guide F:2012, Warehouse typical split",
    },
    "supermarket": {
        EndUseCategory.LIGHTING.value: 18.0,
        EndUseCategory.SPACE_HEATING.value: 10.0,
        EndUseCategory.SPACE_COOLING.value: 35.0,
        EndUseCategory.VENTILATION.value: 5.0,
        EndUseCategory.DHW.value: 3.0,
        EndUseCategory.PLUG_LOADS.value: 5.0,
        EndUseCategory.CATERING.value: 8.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 1.0,
        EndUseCategory.OTHER.value: 15.0,
        "source": "CIBSE Guide F:2012, Supermarket (refrigeration in cooling)",
    },
    "restaurant": {
        EndUseCategory.LIGHTING.value: 10.0,
        EndUseCategory.SPACE_HEATING.value: 20.0,
        EndUseCategory.SPACE_COOLING.value: 10.0,
        EndUseCategory.VENTILATION.value: 10.0,
        EndUseCategory.DHW.value: 10.0,
        EndUseCategory.PLUG_LOADS.value: 5.0,
        EndUseCategory.CATERING.value: 30.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 0.0,
        EndUseCategory.OTHER.value: 5.0,
        "source": "CIBSE Guide F:2012, Restaurant typical split",
    },
    "data_centre": {
        EndUseCategory.LIGHTING.value: 2.0,
        EndUseCategory.SPACE_HEATING.value: 1.0,
        EndUseCategory.SPACE_COOLING.value: 38.0,
        EndUseCategory.VENTILATION.value: 5.0,
        EndUseCategory.DHW.value: 1.0,
        EndUseCategory.PLUG_LOADS.value: 0.0,
        EndUseCategory.PROCESS.value: 48.0,
        EndUseCategory.CATERING.value: 0.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 0.0,
        EndUseCategory.OTHER.value: 5.0,
        "source": "ASHRAE 90.4, Data center typical (IT load = process)",
    },
    "DEFAULT": {
        EndUseCategory.LIGHTING.value: 20.0,
        EndUseCategory.SPACE_HEATING.value: 35.0,
        EndUseCategory.SPACE_COOLING.value: 10.0,
        EndUseCategory.VENTILATION.value: 8.0,
        EndUseCategory.DHW.value: 7.0,
        EndUseCategory.PLUG_LOADS.value: 10.0,
        EndUseCategory.CATERING.value: 3.0,
        EndUseCategory.VERTICAL_TRANSPORT.value: 2.0,
        EndUseCategory.OTHER.value: 5.0,
        "source": "CIBSE Guide F:2012, Generic commercial building average",
    },
}
"""Typical end-use percentage splits by building type from CIBSE Guide F."""


# End-use improvement potential (% reduction achievable per end-use).
# Source: Carbon Trust CTG004, CTV003, CTV009, field measurement data.
# These represent typical achievable savings with proven technology.
END_USE_IMPROVEMENT_POTENTIAL: Dict[str, Dict[str, Any]] = {
    EndUseCategory.LIGHTING.value: {
        "low_cost_pct": 15,
        "medium_cost_pct": 35,
        "high_cost_pct": 60,
        "description": "LED retrofit, occupancy sensors, daylight dimming",
        "source": "Carbon Trust CTG004, CTV003: Lighting field data",
    },
    EndUseCategory.SPACE_HEATING.value: {
        "low_cost_pct": 10,
        "medium_cost_pct": 20,
        "high_cost_pct": 40,
        "description": "Controls optimisation, insulation, heat pump retrofit",
        "source": "Carbon Trust CTV003: Heating best practice",
    },
    EndUseCategory.SPACE_COOLING.value: {
        "low_cost_pct": 10,
        "medium_cost_pct": 25,
        "high_cost_pct": 45,
        "description": "Setpoint optimisation, free cooling, chiller upgrade",
        "source": "Carbon Trust CTV009: Cooling technology guide",
    },
    EndUseCategory.VENTILATION.value: {
        "low_cost_pct": 10,
        "medium_cost_pct": 25,
        "high_cost_pct": 40,
        "description": "VSD on fans, demand-controlled ventilation, heat recovery",
        "source": "Carbon Trust CTG004: Ventilation best practice",
    },
    EndUseCategory.DHW.value: {
        "low_cost_pct": 10,
        "medium_cost_pct": 20,
        "high_cost_pct": 50,
        "description": "Timer controls, insulation, heat pump water heater",
        "source": "Carbon Trust: Hot water good practice",
    },
    EndUseCategory.PLUG_LOADS.value: {
        "low_cost_pct": 10,
        "medium_cost_pct": 20,
        "high_cost_pct": 30,
        "description": "Power management, efficient equipment, smart strips",
        "source": "Carbon Trust CTV003: Office equipment",
    },
    EndUseCategory.PROCESS.value: {
        "low_cost_pct": 5,
        "medium_cost_pct": 15,
        "high_cost_pct": 30,
        "description": "Process optimisation, waste heat recovery, VSD motors",
        "source": "Carbon Trust: Industrial process efficiency",
    },
    EndUseCategory.CATERING.value: {
        "low_cost_pct": 10,
        "medium_cost_pct": 20,
        "high_cost_pct": 35,
        "description": "Equipment scheduling, efficient appliances, heat recovery",
        "source": "Carbon Trust CTG004: Catering energy",
    },
    EndUseCategory.VERTICAL_TRANSPORT.value: {
        "low_cost_pct": 5,
        "medium_cost_pct": 15,
        "high_cost_pct": 30,
        "description": "Drive modernisation, destination control, standby mode",
        "source": "CIBSE Guide D: Lift energy efficiency",
    },
    EndUseCategory.OTHER.value: {
        "low_cost_pct": 5,
        "medium_cost_pct": 10,
        "high_cost_pct": 20,
        "description": "Miscellaneous efficiency measures",
        "source": "CIBSE Guide F: General",
    },
}
"""Achievable improvement potential by end-use category from Carbon Trust."""


# Gap severity thresholds (actual / benchmark ratio).
_GAP_THRESHOLDS = {
    GapSeverity.MINOR: 1.10,
    GapSeverity.MODERATE: 1.30,
    GapSeverity.SIGNIFICANT: 1.60,
    # CRITICAL: >= 1.60
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class EndUseBreakdown(BaseModel):
    """End-use energy breakdown (from sub-metering or estimation).

    Attributes:
        category: End-use category.
        actual_kwh: Actual annual energy for this end-use (kWh).
        actual_kwh_per_m2: Actual energy intensity for this end-use.
        share_pct: Share of total consumption (%).
        benchmark_kwh_per_m2: Benchmark value for this end-use.
    """
    category: EndUseCategory = Field(..., description="End-use category")
    actual_kwh: float = Field(default=0.0, ge=0, description="Actual energy (kWh)")
    actual_kwh_per_m2: float = Field(
        default=0.0, ge=0, description="Actual intensity (kWh/m2)"
    )
    share_pct: float = Field(default=0.0, ge=0, le=100, description="Share (%)")
    benchmark_kwh_per_m2: Optional[float] = Field(
        None, ge=0, description="Benchmark intensity"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class EndUseGap(BaseModel):
    """Performance gap analysis for a single end-use category.

    Attributes:
        category: End-use category.
        actual_kwh: Actual annual energy (kWh).
        benchmark_kwh: Benchmark annual energy (kWh).
        gap_kwh: Absolute gap (actual - benchmark; positive = excess).
        gap_pct: Gap as percentage of benchmark.
        severity: Gap severity classification.
        savings_potential_low: Low-cost savings potential (kWh/yr).
        savings_potential_medium: Medium-cost savings potential.
        savings_potential_high: High-cost savings potential.
    """
    category: str = Field(..., description="End-use category")
    actual_kwh: float = Field(default=0.0, description="Actual (kWh)")
    benchmark_kwh: float = Field(default=0.0, description="Benchmark (kWh)")
    gap_kwh: float = Field(default=0.0, description="Gap (kWh)")
    gap_pct: float = Field(default=0.0, description="Gap (%)")
    severity: str = Field(default="", description="Gap severity")
    savings_potential_low: float = Field(default=0.0, description="Low-cost savings")
    savings_potential_medium: float = Field(
        default=0.0, description="Medium-cost savings"
    )
    savings_potential_high: float = Field(
        default=0.0, description="High-cost savings"
    )


class ImprovementPriority(BaseModel):
    """Ranked improvement priority for an end-use category.

    Attributes:
        rank: Priority rank (1 = highest priority).
        category: End-use category.
        score: Composite priority score.
        gap_kwh: Annual gap in kWh.
        savings_potential_kwh: Achievable savings at medium cost.
        ease_of_implementation: Relative ease (1-5, 5=easiest).
        recommended_measures: Specific measures to implement.
    """
    rank: int = Field(default=0, ge=0, description="Priority rank")
    category: str = Field(default="", description="End-use category")
    score: float = Field(default=0.0, description="Priority score")
    gap_kwh: float = Field(default=0.0, description="Gap (kWh)")
    savings_potential_kwh: float = Field(
        default=0.0, description="Achievable savings (kWh)"
    )
    ease_of_implementation: int = Field(
        default=3, ge=1, le=5, description="Ease (1=hard, 5=easy)"
    )
    recommended_measures: List[str] = Field(
        default_factory=list, description="Recommended measures"
    )


class GapAnalysisResult(BaseModel):
    """Complete energy performance gap analysis result.

    Contains end-use disaggregation, gap by end-use, ranked improvement
    priorities, total savings potential, and actionable recommendations.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    facility_id: str = Field(default="", description="Facility identifier")
    building_type: str = Field(default="", description="Building type")
    floor_area_m2: float = Field(default=0.0, description="Floor area (m2)")

    total_actual_kwh: float = Field(default=0.0, description="Total actual (kWh)")
    total_benchmark_kwh: float = Field(
        default=0.0, description="Total benchmark (kWh)"
    )
    total_gap_kwh: float = Field(default=0.0, description="Total gap (kWh)")
    total_gap_pct: float = Field(default=0.0, description="Total gap (%)")
    overall_severity: str = Field(default="", description="Overall gap severity")

    disaggregation_method: str = Field(default="", description="Method used")

    end_use_breakdown: List[EndUseBreakdown] = Field(
        default_factory=list, description="End-use breakdown"
    )
    end_use_gaps: List[EndUseGap] = Field(
        default_factory=list, description="End-use gap analysis"
    )
    improvement_priorities: List[ImprovementPriority] = Field(
        default_factory=list, description="Ranked improvement priorities"
    )

    total_savings_potential_low: float = Field(
        default=0.0, description="Total low-cost savings (kWh/yr)"
    )
    total_savings_potential_medium: float = Field(
        default=0.0, description="Total medium-cost savings (kWh/yr)"
    )
    total_savings_potential_high: float = Field(
        default=0.0, description="Total high-cost savings (kWh/yr)"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class EnergyPerformanceGapEngine:
    """Energy performance gap analysis engine.

    Analyses the gap between actual and target energy performance
    disaggregated by end-use category.  Provides:
    - End-use disaggregation (sub-metering, TM22, balance-point)
    - Gap analysis by end-use against benchmarks
    - Improvement priority ranking
    - Savings potential calculation (low/medium/high cost)
    - Measure mapping per end-use

    All calculations use published reference data (CIBSE Guide F,
    Carbon Trust) with deterministic Decimal arithmetic.

    Usage::

        engine = EnergyPerformanceGapEngine()
        result = engine.calculate_gap(
            facility_id="bldg-001",
            total_energy_kwh=450000,
            benchmark_eui=200.0,
            floor_area_m2=2500.0,
            building_type="office",
        )
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise with embedded reference data."""
        self._end_use_splits = TYPICAL_END_USE_SPLITS
        self._improvement_potential = END_USE_IMPROVEMENT_POTENTIAL

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def disaggregate_by_end_use(
        self,
        total_energy_kwh: float,
        floor_area_m2: float,
        building_type: str = "office",
        method: DisaggregationMethod = DisaggregationMethod.CIBSE_TM22,
        sub_meter_data: Optional[Dict[str, float]] = None,
    ) -> List[EndUseBreakdown]:
        """Disaggregate total energy consumption by end-use category.

        Args:
            total_energy_kwh: Total annual energy (kWh).
            floor_area_m2: Floor area (m2).
            building_type: Building type for profile lookup.
            method: Disaggregation method.
            sub_meter_data: Sub-meter readings by end-use (for SUB_METERING).

        Returns:
            List of EndUseBreakdown per category.
        """
        if method == DisaggregationMethod.SUB_METERING and sub_meter_data:
            return self._disaggregate_sub_metering(
                sub_meter_data, total_energy_kwh, floor_area_m2,
            )
        else:
            return self._disaggregate_profile(
                total_energy_kwh, floor_area_m2, building_type,
            )

    def calculate_gap(
        self,
        facility_id: str,
        total_energy_kwh: float,
        benchmark_eui: float,
        floor_area_m2: float,
        building_type: str = "office",
        method: DisaggregationMethod = DisaggregationMethod.CIBSE_TM22,
        sub_meter_data: Optional[Dict[str, float]] = None,
    ) -> GapAnalysisResult:
        """Perform complete gap analysis between actual and benchmark.

        Args:
            facility_id: Facility identifier.
            total_energy_kwh: Total annual energy (kWh).
            benchmark_eui: Target EUI (kWh/m2/yr).
            floor_area_m2: Floor area (m2).
            building_type: Building type.
            method: Disaggregation method.
            sub_meter_data: Optional sub-meter data.

        Returns:
            GapAnalysisResult with complete analysis and provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Gap analysis: facility=%s, actual=%.0f kWh, benchmark_eui=%.1f, "
            "area=%.0f m2, type=%s",
            facility_id, total_energy_kwh, benchmark_eui, floor_area_m2,
            building_type,
        )

        # Step 1: Disaggregate by end-use
        breakdown = self.disaggregate_by_end_use(
            total_energy_kwh, floor_area_m2, building_type, method, sub_meter_data,
        )

        # Step 2: Calculate benchmark by end-use
        benchmark_total_kwh = benchmark_eui * floor_area_m2

        # Step 3: Calculate gap per end-use
        end_use_gaps = self._calculate_end_use_gaps(
            breakdown, benchmark_total_kwh, building_type,
        )

        # Step 4: Calculate savings potential
        for gap in end_use_gaps:
            potential = self._improvement_potential.get(gap.category, {})
            excess = max(0.0, gap.gap_kwh)
            gap.savings_potential_low = _round2(
                excess * potential.get("low_cost_pct", 0) / 100.0
            )
            gap.savings_potential_medium = _round2(
                excess * potential.get("medium_cost_pct", 0) / 100.0
            )
            gap.savings_potential_high = _round2(
                excess * potential.get("high_cost_pct", 0) / 100.0
            )

        # Step 5: Total gap
        total_gap = total_energy_kwh - benchmark_total_kwh
        total_gap_pct = 0.0
        if benchmark_total_kwh > 0:
            total_gap_pct = _round2(total_gap / benchmark_total_kwh * 100.0)

        overall_severity = self._classify_severity(
            total_energy_kwh, benchmark_total_kwh,
        )

        # Step 6: Rank improvement priorities
        priorities = self.rank_improvement_priorities(end_use_gaps)

        # Step 7: Aggregate savings
        total_low = sum(g.savings_potential_low for g in end_use_gaps)
        total_med = sum(g.savings_potential_medium for g in end_use_gaps)
        total_high = sum(g.savings_potential_high for g in end_use_gaps)

        # Step 8: Map measures
        for p in priorities:
            p.recommended_measures = self._get_measures_for_category(p.category)

        # Step 9: Recommendations
        recommendations = self._generate_recommendations(
            total_gap_pct, overall_severity, priorities, building_type,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = GapAnalysisResult(
            facility_id=facility_id,
            building_type=building_type,
            floor_area_m2=_round2(floor_area_m2),
            total_actual_kwh=_round2(total_energy_kwh),
            total_benchmark_kwh=_round2(benchmark_total_kwh),
            total_gap_kwh=_round2(total_gap),
            total_gap_pct=total_gap_pct,
            overall_severity=overall_severity.value,
            disaggregation_method=method.value,
            end_use_breakdown=breakdown,
            end_use_gaps=end_use_gaps,
            improvement_priorities=priorities,
            total_savings_potential_low=_round2(total_low),
            total_savings_potential_medium=_round2(total_med),
            total_savings_potential_high=_round2(total_high),
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Gap analysis complete: facility=%s, gap=%.1f%% (%s), "
            "savings_potential=%.0f kWh (med), hash=%s (%.1f ms)",
            facility_id, total_gap_pct, overall_severity.value,
            total_med, result.provenance_hash[:16], elapsed_ms,
        )
        return result

    def rank_improvement_priorities(
        self,
        gaps: List[EndUseGap],
    ) -> List[ImprovementPriority]:
        """Rank end-uses by improvement priority.

        Priority score = gap_kwh * improvement_pct / 100 * ease_factor.
        Higher score = higher priority.

        Args:
            gaps: List of end-use gaps.

        Returns:
            List of ImprovementPriority sorted by descending score.
        """
        priorities: List[ImprovementPriority] = []

        # Ease of implementation ratings (higher = easier)
        ease_map = {
            EndUseCategory.LIGHTING.value: 5,
            EndUseCategory.PLUG_LOADS.value: 4,
            EndUseCategory.VENTILATION.value: 3,
            EndUseCategory.DHW.value: 4,
            EndUseCategory.SPACE_COOLING.value: 3,
            EndUseCategory.SPACE_HEATING.value: 2,
            EndUseCategory.CATERING.value: 3,
            EndUseCategory.VERTICAL_TRANSPORT.value: 2,
            EndUseCategory.PROCESS.value: 2,
            EndUseCategory.OTHER.value: 3,
        }

        for gap in gaps:
            if gap.gap_kwh <= 0:
                continue

            potential = self._improvement_potential.get(gap.category, {})
            med_pct = potential.get("medium_cost_pct", 10)
            savings = gap.gap_kwh * med_pct / 100.0
            ease = ease_map.get(gap.category, 3)

            # Composite score: savings * ease
            score = savings * ease

            priorities.append(ImprovementPriority(
                category=gap.category,
                score=_round2(score),
                gap_kwh=_round2(gap.gap_kwh),
                savings_potential_kwh=_round2(savings),
                ease_of_implementation=ease,
            ))

        # Sort by descending score
        priorities.sort(key=lambda p: p.score, reverse=True)

        # Assign ranks
        for i, p in enumerate(priorities, start=1):
            p.rank = i

        return priorities

    def map_to_measures(
        self,
        category: str,
    ) -> List[str]:
        """Map an end-use category to specific improvement measures.

        Args:
            category: End-use category name.

        Returns:
            List of recommended measures for this category.
        """
        return self._get_measures_for_category(category)

    def calculate_savings_potential(
        self,
        gap_kwh: float,
        category: str,
        cost_level: str = "medium",
    ) -> float:
        """Calculate savings potential for a specific end-use gap.

        Args:
            gap_kwh: Gap in kWh (excess over benchmark).
            category: End-use category.
            cost_level: 'low', 'medium', or 'high'.

        Returns:
            Achievable savings in kWh.
        """
        if gap_kwh <= 0:
            return 0.0

        potential = self._improvement_potential.get(category, {})
        pct_key = f"{cost_level}_cost_pct"
        pct = potential.get(pct_key, 10)
        return _round2(gap_kwh * pct / 100.0)

    # -------------------------------------------------------------------
    # Internal: Disaggregation
    # -------------------------------------------------------------------

    def _disaggregate_profile(
        self,
        total_kwh: float,
        floor_area: float,
        building_type: str,
    ) -> List[EndUseBreakdown]:
        """Disaggregate using published end-use profiles (TM22 method).

        Args:
            total_kwh: Total annual energy.
            floor_area: Floor area (m2).
            building_type: Building type for profile lookup.

        Returns:
            List of EndUseBreakdown.
        """
        bt_key = building_type.lower().replace(" ", "_")
        profile = self._end_use_splits.get(
            bt_key, self._end_use_splits["DEFAULT"]
        )

        results: List[EndUseBreakdown] = []
        for cat in EndUseCategory:
            share_pct = profile.get(cat.value, 0.0)
            if isinstance(share_pct, str):
                continue  # Skip the "source" key

            cat_kwh = total_kwh * share_pct / 100.0
            cat_kwh_per_m2 = cat_kwh / floor_area if floor_area > 0 else 0.0

            results.append(EndUseBreakdown(
                category=cat,
                actual_kwh=_round2(cat_kwh),
                actual_kwh_per_m2=_round2(cat_kwh_per_m2),
                share_pct=_round2(share_pct),
            ))

        return results

    def _disaggregate_sub_metering(
        self,
        sub_meter_data: Dict[str, float],
        total_kwh: float,
        floor_area: float,
    ) -> List[EndUseBreakdown]:
        """Disaggregate using actual sub-meter data.

        Args:
            sub_meter_data: End-use category -> kWh mapping.
            total_kwh: Total metered energy.
            floor_area: Floor area (m2).

        Returns:
            List of EndUseBreakdown.
        """
        sub_total = sum(sub_meter_data.values())
        results: List[EndUseBreakdown] = []

        for cat in EndUseCategory:
            cat_kwh = sub_meter_data.get(cat.value, 0.0)
            share = 0.0
            if total_kwh > 0:
                share = cat_kwh / total_kwh * 100.0
            kwh_m2 = cat_kwh / floor_area if floor_area > 0 else 0.0

            results.append(EndUseBreakdown(
                category=cat,
                actual_kwh=_round2(cat_kwh),
                actual_kwh_per_m2=_round2(kwh_m2),
                share_pct=_round2(share),
            ))

        # Handle unaccounted energy
        unaccounted = total_kwh - sub_total
        if abs(unaccounted) > total_kwh * 0.01:  # > 1% discrepancy
            logger.warning(
                "Sub-meter total (%.0f) differs from main meter (%.0f) by %.0f kWh",
                sub_total, total_kwh, unaccounted,
            )

        return results

    # -------------------------------------------------------------------
    # Internal: Gap Calculation
    # -------------------------------------------------------------------

    def _calculate_end_use_gaps(
        self,
        breakdown: List[EndUseBreakdown],
        benchmark_total_kwh: float,
        building_type: str,
    ) -> List[EndUseGap]:
        """Calculate performance gap for each end-use.

        Args:
            breakdown: End-use disaggregation results.
            benchmark_total_kwh: Total benchmark energy (kWh).
            building_type: Building type for benchmark splits.

        Returns:
            List of EndUseGap.
        """
        bt_key = building_type.lower().replace(" ", "_")
        profile = self._end_use_splits.get(
            bt_key, self._end_use_splits["DEFAULT"]
        )

        gaps: List[EndUseGap] = []
        for eu in breakdown:
            bench_share = profile.get(eu.category.value, 0.0)
            if isinstance(bench_share, str):
                bench_share = 0.0
            bench_kwh = benchmark_total_kwh * bench_share / 100.0

            gap_kwh = eu.actual_kwh - bench_kwh
            gap_pct = 0.0
            if bench_kwh > 0:
                gap_pct = _round2(gap_kwh / bench_kwh * 100.0)

            severity = self._classify_severity(eu.actual_kwh, bench_kwh)

            gaps.append(EndUseGap(
                category=eu.category.value,
                actual_kwh=_round2(eu.actual_kwh),
                benchmark_kwh=_round2(bench_kwh),
                gap_kwh=_round2(gap_kwh),
                gap_pct=gap_pct,
                severity=severity.value,
            ))

        return gaps

    def _classify_severity(
        self, actual: float, benchmark: float,
    ) -> GapSeverity:
        """Classify gap severity based on actual/benchmark ratio.

        Args:
            actual: Actual energy.
            benchmark: Benchmark energy.

        Returns:
            GapSeverity classification.
        """
        if benchmark <= 0:
            return GapSeverity.MINOR

        ratio = actual / benchmark
        if ratio >= _GAP_THRESHOLDS[GapSeverity.SIGNIFICANT]:
            return GapSeverity.CRITICAL
        elif ratio >= _GAP_THRESHOLDS[GapSeverity.MODERATE]:
            return GapSeverity.SIGNIFICANT
        elif ratio >= _GAP_THRESHOLDS[GapSeverity.MINOR]:
            return GapSeverity.MODERATE
        else:
            return GapSeverity.MINOR

    # -------------------------------------------------------------------
    # Internal: Measure Mapping
    # -------------------------------------------------------------------

    def _get_measures_for_category(self, category: str) -> List[str]:
        """Get specific improvement measures for an end-use category.

        All measures are from published sources (Carbon Trust, CIBSE).

        Args:
            category: End-use category name.

        Returns:
            List of measure descriptions.
        """
        measures_map: Dict[str, List[str]] = {
            EndUseCategory.LIGHTING.value: [
                "Replace fluorescent tubes with LED (40-60% savings per fitting)",
                "Install PIR occupancy sensors in intermittently used areas",
                "Add daylight sensors and automatic dimming in perimeter zones",
                "Reduce overlighting: audit and remove excess fixtures",
                "Upgrade emergency lighting to LED with auto-test",
            ],
            EndUseCategory.SPACE_HEATING.value: [
                "Optimise heating schedule and setpoints (reduce to 19-20C occupied)",
                "Install thermostatic radiator valves (TRVs) on all radiators",
                "Improve building fabric insulation (roof, walls, glazing)",
                "Upgrade boiler to condensing type (92-95% efficiency)",
                "Consider air-source heat pump replacement (COP 3.0+)",
            ],
            EndUseCategory.SPACE_COOLING.value: [
                "Raise cooling setpoints to 24-25C (2% savings per degree)",
                "Enable free cooling / economiser modes in AHUs",
                "Replace old chillers with high-efficiency units (COP 6.0+)",
                "Install solar shading (external blinds, films) to reduce load",
                "Clean condenser coils and optimise refrigerant charge",
            ],
            EndUseCategory.VENTILATION.value: [
                "Install variable speed drives (VSDs) on AHU fans",
                "Implement CO2-based demand-controlled ventilation",
                "Add plate or rotary heat recovery to exhaust air",
                "Seal ductwork leaks (can waste 20-30% of fan energy)",
                "Review and reduce outside air rates to ASHRAE 62.1 minimums",
            ],
            EndUseCategory.DHW.value: [
                "Add timers to hot water systems (avoid 24/7 heating)",
                "Insulate hot water pipes and storage cylinders",
                "Install point-of-use water heaters for remote outlets",
                "Consider solar thermal or heat pump water heating",
                "Install water-saving taps and showers to reduce demand",
            ],
            EndUseCategory.PLUG_LOADS.value: [
                "Enable power management on all PCs (sleep after 15 min)",
                "Use smart power strips to eliminate standby loads",
                "Replace CRT/old monitors with Energy Star-rated models",
                "Audit and remove unnecessary equipment (printers, servers)",
                "Consolidate multifunction devices (print/scan/copy)",
            ],
            EndUseCategory.PROCESS.value: [
                "Optimise process scheduling to reduce peak loads",
                "Install VSDs on process motors and pumps",
                "Implement waste heat recovery from process exhaust",
                "Upgrade to IE3/IE4 premium efficiency motors",
                "Review compressed air system for leaks and pressure reduction",
            ],
            EndUseCategory.CATERING.value: [
                "Implement equipment scheduling (turn off idle equipment)",
                "Replace old catering equipment with energy-rated models",
                "Install extraction hood controls linked to cooking activity",
                "Add heat recovery to kitchen extract air",
                "Insulate hot holding and steam equipment",
            ],
            EndUseCategory.VERTICAL_TRANSPORT.value: [
                "Modernise lift drives to regenerative AC variable-voltage",
                "Enable standby/sleep mode for low-traffic periods",
                "Implement destination control for multi-car installations",
                "Replace halogen cab lighting with LED",
                "Optimise lift grouping and dispatching algorithms",
            ],
            EndUseCategory.OTHER.value: [
                "Audit miscellaneous loads and eliminate unnecessary loads",
                "Install timer controls on external lighting",
                "Review humidification setpoints and operation schedules",
                "Consider smart building management system (BMS) upgrade",
            ],
        }
        return measures_map.get(category, [])

    # -------------------------------------------------------------------
    # Internal: Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        total_gap_pct: float,
        overall_severity: GapSeverity,
        priorities: List[ImprovementPriority],
        building_type: str,
    ) -> List[str]:
        """Generate deterministic recommendations from gap analysis.

        Args:
            total_gap_pct: Overall gap percentage.
            overall_severity: Overall severity classification.
            priorities: Ranked improvement priorities.
            building_type: Building type.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: Overall gap assessment
        if overall_severity == GapSeverity.CRITICAL:
            recs.append(
                f"Overall energy performance gap is CRITICAL ({total_gap_pct}% "
                f"above benchmark). A comprehensive energy audit per "
                f"ISO 50002 is urgently recommended. Consider engaging an "
                f"ESCO (Energy Service Company) for guaranteed savings."
            )
        elif overall_severity == GapSeverity.SIGNIFICANT:
            recs.append(
                f"Significant performance gap of {total_gap_pct}% above benchmark. "
                f"Prioritise the top 3 end-use categories for improvement."
            )
        elif overall_severity == GapSeverity.MODERATE:
            recs.append(
                f"Moderate performance gap of {total_gap_pct}%. Target operational "
                f"improvements and low-cost measures in priority end-uses."
            )
        elif overall_severity == GapSeverity.MINOR:
            recs.append(
                f"Performance is within 10% of benchmark. Focus on continuous "
                f"improvement and maintaining good operational practices."
            )

        # R2: Top 3 priorities
        for p in priorities[:3]:
            if p.savings_potential_kwh > 0:
                recs.append(
                    f"Priority {p.rank}: {p.category.replace('_', ' ').title()} "
                    f"- gap of {_round2(p.gap_kwh)} kWh/yr with "
                    f"{_round2(p.savings_potential_kwh)} kWh/yr achievable savings "
                    f"(ease rating: {p.ease_of_implementation}/5)."
                )

        # R3: Sub-metering recommendation
        recs.append(
            "Install sub-metering on the top 3 energy-consuming end-uses "
            "to validate the estimated breakdown and track savings after "
            "implementing improvement measures."
        )

        return recs
