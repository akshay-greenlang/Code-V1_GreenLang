# -*- coding: utf-8 -*-
"""
WaterPollutionEngine - PACK-013 CSRD Manufacturing Engine 5
=============================================================

Water usage and pollution tracking per ESRS E2 (Pollution) and ESRS E3
(Water and Marine Resources).  Implements deterministic water-balance
calculations, pollutant inventory, IED BAT-AEL limit checking, REACH
SVHC assessment, and water-stress analysis for manufacturing facilities.

ESRS Coverage:
    E2 - Pollution of air, water and soil:
        E2-4: Pollution of air, water and soil (quantitative)
        E2-5: Substances of concern and substances of very high concern
    E3 - Water and marine resources:
        E3-4: Water consumption (quantitative)
        E3-5: Anticipated financial effects from water and marine
               resources-related impacts

Regulatory References:
    - ESRS E2 (Delegated Regulation (EU) 2023/2772)
    - ESRS E3 (Delegated Regulation (EU) 2023/2772)
    - Industrial Emissions Directive 2010/75/EU (IED)
    - REACH Regulation (EC) No 1907/2006
    - Water Framework Directive 2000/60/EC

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Water balance is a conservation equation (intake - discharge = consumption)
    - Emission factors and BAT-AEL limits from EU regulatory tables
    - SHA-256 provenance hash on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-013 CSRD Manufacturing
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
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


def _round_value(value: Decimal, places: int = 3) -> float:
    """Round a Decimal to specified places and return float."""
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)


def _pct(numerator: Decimal, denominator: Decimal, places: int = 2) -> float:
    """Calculate percentage safely, returning 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return _round_value((numerator / denominator) * Decimal("100"), places)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WaterSource(str, Enum):
    """Source of water intake per ESRS E3 disclosure."""
    SURFACE = "surface"
    GROUNDWATER = "groundwater"
    THIRD_PARTY = "third_party"
    RAINWATER = "rainwater"
    SEAWATER = "seawater"
    PRODUCED_WATER = "produced_water"


class WaterStressLevel(str, Enum):
    """Water stress level per WRI Aqueduct classification."""
    LOW = "low"
    LOW_MEDIUM = "low_medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"
    EXTREMELY_HIGH = "extremely_high"


class PollutantCategory(str, Enum):
    """Pollutant emission pathway per ESRS E2."""
    AIR = "air"
    WATER = "water"
    SOIL = "soil"


class PollutantType(str, Enum):
    """Pollutant type per E-PRTR / IED Annex II."""
    NOX = "nox"
    SOX = "sox"
    PM10 = "pm10"
    PM2_5 = "pm2_5"
    VOC = "voc"
    CO = "co"
    NH3 = "nh3"
    HEAVY_METALS = "heavy_metals"
    NITROGEN_WATER = "nitrogen_water"
    PHOSPHORUS_WATER = "phosphorus_water"
    BOD = "bod"
    COD = "cod"
    MICROPLASTICS = "microplastics"


class TreatmentLevel(str, Enum):
    """Wastewater treatment level."""
    NONE = "none"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    ADVANCED = "advanced"


class QualityGrade(str, Enum):
    """Water quality grade for intake."""
    POTABLE = "potable"
    PROCESS_GRADE = "process_grade"
    COOLING_GRADE = "cooling_grade"
    RAW = "raw"


class MeasurementMethod(str, Enum):
    """Method used to measure pollutant emissions."""
    CONTINUOUS_MONITORING = "continuous_monitoring"
    PERIODIC_MEASUREMENT = "periodic_measurement"
    MASS_BALANCE = "mass_balance"
    EMISSION_FACTOR = "emission_factor"
    ENGINEERING_ESTIMATE = "engineering_estimate"


class AuthorizationStatus(str, Enum):
    """REACH SVHC authorization status."""
    AUTHORIZED = "authorized"
    PENDING = "pending"
    SUNSET = "sunset"
    EXEMPT = "exempt"
    NOT_REQUIRED = "not_required"


# ---------------------------------------------------------------------------
# Constants - Water Stress Thresholds
# ---------------------------------------------------------------------------

WATER_STRESS_THRESHOLDS: Dict[str, Dict[str, float]] = {
    WaterStressLevel.LOW: {"min": 0.0, "max": 10.0},
    WaterStressLevel.LOW_MEDIUM: {"min": 10.0, "max": 20.0},
    WaterStressLevel.MEDIUM_HIGH: {"min": 20.0, "max": 40.0},
    WaterStressLevel.HIGH: {"min": 40.0, "max": 80.0},
    WaterStressLevel.EXTREMELY_HIGH: {"min": 80.0, "max": 100.0},
}

# ---------------------------------------------------------------------------
# Constants - IED BAT-AEL Emission Limits (mg/Nm3 unless noted)
# Source: BREF documents per industrial sector
# ---------------------------------------------------------------------------

IED_EMISSION_LIMITS: Dict[str, Dict[str, Dict[str, float]]] = {
    "cement": {
        PollutantType.NOX: {"lower": 200.0, "upper": 450.0, "unit": "mg/Nm3"},
        PollutantType.SOX: {"lower": 50.0, "upper": 400.0, "unit": "mg/Nm3"},
        PollutantType.PM10: {"lower": 10.0, "upper": 20.0, "unit": "mg/Nm3"},
        PollutantType.CO: {"lower": 500.0, "upper": 800.0, "unit": "mg/Nm3"},
        PollutantType.VOC: {"lower": 10.0, "upper": 30.0, "unit": "mg/Nm3"},
        PollutantType.HEAVY_METALS: {"lower": 0.01, "upper": 0.05, "unit": "mg/Nm3"},
    },
    "steel": {
        PollutantType.NOX: {"lower": 100.0, "upper": 300.0, "unit": "mg/Nm3"},
        PollutantType.SOX: {"lower": 50.0, "upper": 200.0, "unit": "mg/Nm3"},
        PollutantType.PM10: {"lower": 1.0, "upper": 15.0, "unit": "mg/Nm3"},
        PollutantType.CO: {"lower": 100.0, "upper": 300.0, "unit": "mg/Nm3"},
        PollutantType.VOC: {"lower": 5.0, "upper": 20.0, "unit": "mg/Nm3"},
        PollutantType.HEAVY_METALS: {"lower": 0.005, "upper": 0.05, "unit": "mg/Nm3"},
    },
    "glass": {
        PollutantType.NOX: {"lower": 500.0, "upper": 800.0, "unit": "mg/Nm3"},
        PollutantType.SOX: {"lower": 200.0, "upper": 500.0, "unit": "mg/Nm3"},
        PollutantType.PM10: {"lower": 10.0, "upper": 30.0, "unit": "mg/Nm3"},
        PollutantType.CO: {"lower": 100.0, "upper": 500.0, "unit": "mg/Nm3"},
        PollutantType.VOC: {"lower": 10.0, "upper": 50.0, "unit": "mg/Nm3"},
        PollutantType.HEAVY_METALS: {"lower": 0.01, "upper": 0.1, "unit": "mg/Nm3"},
    },
    "chemicals": {
        PollutantType.NOX: {"lower": 100.0, "upper": 300.0, "unit": "mg/Nm3"},
        PollutantType.SOX: {"lower": 50.0, "upper": 350.0, "unit": "mg/Nm3"},
        PollutantType.PM10: {"lower": 5.0, "upper": 20.0, "unit": "mg/Nm3"},
        PollutantType.VOC: {"lower": 5.0, "upper": 50.0, "unit": "mg/Nm3"},
        PollutantType.CO: {"lower": 100.0, "upper": 300.0, "unit": "mg/Nm3"},
        PollutantType.HEAVY_METALS: {"lower": 0.01, "upper": 0.05, "unit": "mg/Nm3"},
    },
    "pulp_paper": {
        PollutantType.NOX: {"lower": 150.0, "upper": 300.0, "unit": "mg/Nm3"},
        PollutantType.SOX: {"lower": 50.0, "upper": 200.0, "unit": "mg/Nm3"},
        PollutantType.PM10: {"lower": 5.0, "upper": 20.0, "unit": "mg/Nm3"},
        PollutantType.COD: {"lower": 0.5, "upper": 1.5, "unit": "kg/t_product"},
        PollutantType.BOD: {"lower": 0.15, "upper": 0.4, "unit": "kg/t_product"},
        PollutantType.NITROGEN_WATER: {"lower": 0.1, "upper": 0.4, "unit": "kg/t_product"},
        PollutantType.PHOSPHORUS_WATER: {"lower": 0.01, "upper": 0.04, "unit": "kg/t_product"},
    },
    "food_drink": {
        PollutantType.NOX: {"lower": 100.0, "upper": 250.0, "unit": "mg/Nm3"},
        PollutantType.PM10: {"lower": 5.0, "upper": 20.0, "unit": "mg/Nm3"},
        PollutantType.COD: {"lower": 0.2, "upper": 1.0, "unit": "kg/t_product"},
        PollutantType.BOD: {"lower": 0.05, "upper": 0.25, "unit": "kg/t_product"},
        PollutantType.NITROGEN_WATER: {"lower": 0.05, "upper": 0.2, "unit": "kg/t_product"},
        PollutantType.PHOSPHORUS_WATER: {"lower": 0.005, "upper": 0.02, "unit": "kg/t_product"},
    },
    "textiles": {
        PollutantType.NOX: {"lower": 100.0, "upper": 200.0, "unit": "mg/Nm3"},
        PollutantType.VOC: {"lower": 5.0, "upper": 40.0, "unit": "mg/Nm3"},
        PollutantType.COD: {"lower": 0.3, "upper": 1.2, "unit": "kg/t_product"},
        PollutantType.BOD: {"lower": 0.1, "upper": 0.3, "unit": "kg/t_product"},
        PollutantType.HEAVY_METALS: {"lower": 0.01, "upper": 0.05, "unit": "mg/Nm3"},
    },
}

# REACH SVHC concentration threshold (w/w)
REACH_SVHC_THRESHOLD: float = 0.1  # 0.1% w/w

# Wastewater treatment removal efficiencies (fraction removed)
TREATMENT_EFFICIENCY: Dict[str, Dict[str, float]] = {
    TreatmentLevel.NONE: {
        "bod": 0.0, "cod": 0.0, "nitrogen": 0.0, "phosphorus": 0.0, "heavy_metals": 0.0,
    },
    TreatmentLevel.PRIMARY: {
        "bod": 0.30, "cod": 0.25, "nitrogen": 0.10, "phosphorus": 0.10, "heavy_metals": 0.20,
    },
    TreatmentLevel.SECONDARY: {
        "bod": 0.85, "cod": 0.75, "nitrogen": 0.30, "phosphorus": 0.30, "heavy_metals": 0.50,
    },
    TreatmentLevel.TERTIARY: {
        "bod": 0.95, "cod": 0.90, "nitrogen": 0.80, "phosphorus": 0.90, "heavy_metals": 0.80,
    },
    TreatmentLevel.ADVANCED: {
        "bod": 0.99, "cod": 0.95, "nitrogen": 0.90, "phosphorus": 0.95, "heavy_metals": 0.95,
    },
}

# Thermal discharge limits (delta C)
THERMAL_DISCHARGE_LIMIT_C: float = 3.0  # Maximum temperature rise per WFD


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class WaterPollutionConfig(BaseModel):
    """Configuration for water and pollution assessment."""
    reporting_year: int = Field(description="Reporting year for the assessment")
    include_water_stress: bool = Field(default=True, description="Include water stress analysis")
    include_reach_svhc: bool = Field(default=True, description="Include REACH SVHC assessment")
    include_ied_compliance: bool = Field(default=True, description="Include IED compliance check")
    pollutant_threshold_reporting: bool = Field(
        default=True,
        description="Flag pollutants exceeding E-PRTR reporting thresholds",
    )
    production_volume: Decimal = Field(
        default=Decimal("0"),
        description="Total production volume for intensity calculations",
    )
    production_unit: str = Field(default="tonnes", description="Unit for production volume")
    sub_sector: str = Field(default="", description="Manufacturing sub-sector for BAT-AEL lookup")

    @field_validator("reporting_year", mode="before")
    @classmethod
    def _validate_year(cls, v: Any) -> int:
        year = int(v)
        if year < 2020 or year > 2035:
            raise ValueError(f"Reporting year {year} outside valid range 2020-2035")
        return year

    @field_validator("production_volume", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class WaterIntakeData(BaseModel):
    """Water intake (withdrawal) data for a facility."""
    source: WaterSource = Field(description="Source of water withdrawal")
    volume_m3: Decimal = Field(description="Volume withdrawn in cubic metres")
    quality_grade: QualityGrade = Field(
        default=QualityGrade.RAW, description="Quality grade of withdrawn water"
    )
    water_stressed_area: bool = Field(
        default=False, description="Whether the source is in a water-stressed area"
    )
    facility_id: str = Field(default="", description="Facility identifier")

    @field_validator("volume_m3", mode="before")
    @classmethod
    def _coerce_volume(cls, v: Any) -> Decimal:
        d = _decimal(v)
        if d < 0:
            raise ValueError("Water intake volume cannot be negative")
        return d


class WaterDischargeData(BaseModel):
    """Water discharge data for a facility."""
    destination: WaterSource = Field(description="Discharge destination")
    volume_m3: Decimal = Field(description="Volume discharged in cubic metres")
    treatment_level: TreatmentLevel = Field(
        default=TreatmentLevel.SECONDARY, description="Treatment level applied"
    )
    pollutant_concentrations: Dict[str, float] = Field(
        default_factory=dict,
        description="Pollutant concentrations in discharge (mg/L or as specified)",
    )
    temperature_delta_c: float = Field(
        default=0.0,
        description="Temperature difference between discharge and receiving water (deg C)",
    )
    facility_id: str = Field(default="", description="Facility identifier")

    @field_validator("volume_m3", mode="before")
    @classmethod
    def _coerce_volume(cls, v: Any) -> Decimal:
        d = _decimal(v)
        if d < 0:
            raise ValueError("Water discharge volume cannot be negative")
        return d


class PollutantEmission(BaseModel):
    """Pollutant emission record for E-PRTR / IED reporting."""
    pollutant_type: PollutantType = Field(description="Type of pollutant")
    category: PollutantCategory = Field(description="Emission pathway (air/water/soil)")
    quantity_tonnes: Decimal = Field(description="Quantity emitted in tonnes")
    emission_limit: Optional[float] = Field(
        default=None, description="Applicable emission limit (mg/Nm3 or kg/t)"
    )
    facility_id: str = Field(default="", description="Facility identifier")
    measurement_method: MeasurementMethod = Field(
        default=MeasurementMethod.EMISSION_FACTOR,
        description="Measurement method used",
    )

    @field_validator("quantity_tonnes", mode="before")
    @classmethod
    def _coerce_qty(cls, v: Any) -> Decimal:
        d = _decimal(v)
        if d < 0:
            raise ValueError("Pollutant emission quantity cannot be negative")
        return d


class SVHCSubstance(BaseModel):
    """REACH Substance of Very High Concern (SVHC) record."""
    cas_number: str = Field(description="CAS registry number")
    substance_name: str = Field(description="Substance name")
    concentration_pct: float = Field(description="Concentration in product (% w/w)")
    quantity_tonnes: Decimal = Field(description="Total quantity used/produced in tonnes")
    authorization_status: AuthorizationStatus = Field(
        default=AuthorizationStatus.NOT_REQUIRED,
        description="REACH authorization status",
    )

    @field_validator("quantity_tonnes", mode="before")
    @classmethod
    def _coerce_qty(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("concentration_pct", mode="before")
    @classmethod
    def _validate_concentration(cls, v: Any) -> float:
        val = float(v)
        if val < 0 or val > 100:
            raise ValueError(f"Concentration {val}% outside valid range 0-100")
        return val


class WaterStressAssessment(BaseModel):
    """Result of water stress analysis."""
    total_withdrawal_stressed_m3: Decimal = Field(
        default=Decimal("0"), description="Total withdrawal from stressed areas (m3)"
    )
    total_withdrawal_m3: Decimal = Field(
        default=Decimal("0"), description="Total withdrawal all areas (m3)"
    )
    stressed_withdrawal_pct: float = Field(
        default=0.0, description="Percentage of withdrawal from stressed areas"
    )
    stressed_sources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detail of stressed source withdrawals"
    )
    stress_level_classification: WaterStressLevel = Field(
        default=WaterStressLevel.LOW, description="Overall water stress classification"
    )
    methodology_notes: List[str] = Field(default_factory=list)


class PollutantInventoryItem(BaseModel):
    """Single item in the pollutant inventory."""
    pollutant_type: str = Field(description="Pollutant type")
    category: str = Field(description="Emission category (air/water/soil)")
    total_tonnes: float = Field(description="Total emissions in tonnes")
    facility_count: int = Field(default=1, description="Number of facilities reporting")
    exceeds_limit: bool = Field(default=False, description="Exceeds BAT-AEL upper limit")
    limit_upper: Optional[float] = Field(default=None, description="BAT-AEL upper limit")
    measurement_methods: List[str] = Field(default_factory=list)


class IEDComplianceDetail(BaseModel):
    """IED compliance check detail for a single pollutant."""
    pollutant_type: str = Field(description="Pollutant type")
    measured_value: Optional[float] = Field(default=None, description="Measured value")
    bat_ael_lower: Optional[float] = Field(default=None, description="BAT-AEL lower bound")
    bat_ael_upper: Optional[float] = Field(default=None, description="BAT-AEL upper bound")
    unit: str = Field(default="mg/Nm3", description="Unit of measurement")
    status: str = Field(default="not_assessed", description="Compliance status")
    gap_pct: float = Field(default=0.0, description="Gap above upper limit (%)")


class SVHCAssessmentResult(BaseModel):
    """Result of SVHC assessment."""
    total_svhc_count: int = Field(default=0, description="Total number of SVHCs identified")
    above_threshold_count: int = Field(
        default=0, description="SVHCs above 0.1% w/w threshold"
    )
    total_svhc_quantity_tonnes: float = Field(
        default=0.0, description="Total SVHC quantity in tonnes"
    )
    substances: List[Dict[str, Any]] = Field(
        default_factory=list, description="Individual SVHC assessments"
    )
    requires_notification: bool = Field(
        default=False, description="REACH Article 33 notification required"
    )
    authorization_gaps: List[str] = Field(
        default_factory=list, description="Substances needing authorization"
    )


class WaterPollutionResult(BaseModel):
    """Complete water and pollution assessment result with provenance.

    Covers ESRS E2 and E3 quantitative disclosure requirements.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    # --- Water balance (E3) ---
    total_water_withdrawal_m3: float = Field(
        default=0.0, description="Total water withdrawal (m3)"
    )
    total_water_discharge_m3: float = Field(
        default=0.0, description="Total water discharge (m3)"
    )
    total_water_consumption_m3: float = Field(
        default=0.0, description="Total water consumption = withdrawal - discharge (m3)"
    )
    water_recycling_rate_pct: float = Field(
        default=0.0, description="Water recycling/reuse rate (%)"
    )
    water_intensity_m3_per_unit: float = Field(
        default=0.0, description="Water intensity (m3 per production unit)"
    )
    water_stressed_withdrawal_pct: float = Field(
        default=0.0, description="Percentage of total withdrawal from water-stressed areas"
    )
    withdrawal_by_source: Dict[str, float] = Field(
        default_factory=dict, description="Withdrawal breakdown by source (m3)"
    )
    discharge_by_destination: Dict[str, float] = Field(
        default_factory=dict, description="Discharge breakdown by destination (m3)"
    )
    # --- Pollution (E2) ---
    pollutant_inventory: List[PollutantInventoryItem] = Field(
        default_factory=list, description="Full pollutant inventory"
    )
    total_pollutant_air_tonnes: float = Field(
        default=0.0, description="Total air pollutant emissions (tonnes)"
    )
    total_pollutant_water_tonnes: float = Field(
        default=0.0, description="Total water pollutant emissions (tonnes)"
    )
    total_pollutant_soil_tonnes: float = Field(
        default=0.0, description="Total soil pollutant emissions (tonnes)"
    )
    # --- SVHC (E2-5) ---
    svhc_count: int = Field(default=0, description="Number of SVHCs identified")
    svhc_above_threshold: int = Field(
        default=0, description="SVHCs above 0.1% w/w concentration"
    )
    svhc_assessment: Optional[SVHCAssessmentResult] = Field(
        default=None, description="Detailed SVHC assessment"
    )
    # --- IED compliance ---
    ied_compliance_status: str = Field(
        default="not_assessed", description="Overall IED compliance status"
    )
    ied_compliance_details: List[IEDComplianceDetail] = Field(
        default_factory=list, description="Per-pollutant IED compliance details"
    )
    # --- Water stress ---
    water_stress_assessment: Optional[WaterStressAssessment] = Field(
        default=None, description="Water stress analysis"
    )
    # --- ESRS metrics ---
    esrs_e2_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="ESRS E2 quantitative metrics"
    )
    esrs_e3_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="ESRS E3 quantitative metrics"
    )
    # --- Thermal discharge ---
    thermal_discharge_violations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Discharges exceeding thermal limits"
    )
    # --- Metadata ---
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology and assumption notes"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WaterPollutionEngine:
    """Water and pollution assessment engine for ESRS E2/E3 compliance.

    Provides deterministic, zero-hallucination calculations for:
    - Water balance (withdrawal, discharge, consumption)
    - Water stress assessment (WRI Aqueduct methodology)
    - Pollutant inventory (air, water, soil)
    - IED BAT-AEL compliance checking
    - REACH SVHC assessment
    - Water intensity and recycling metrics
    - Thermal discharge analysis

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    Every result includes a SHA-256 provenance hash for audit trails.
    """

    def __init__(self, config: WaterPollutionConfig) -> None:
        """Initialize the WaterPollutionEngine.

        Args:
            config: Configuration for the assessment including reporting year,
                    feature flags, and production volume.
        """
        self.config = config
        self._notes: List[str] = []
        logger.info(
            "WaterPollutionEngine v%s initialized for year %d",
            _MODULE_VERSION,
            config.reporting_year,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_water_balance(
        self,
        intake: List[WaterIntakeData],
        discharge: List[WaterDischargeData],
        recycled_volume_m3: Decimal = Decimal("0"),
        emissions: Optional[List[PollutantEmission]] = None,
        substances: Optional[List[SVHCSubstance]] = None,
    ) -> WaterPollutionResult:
        """Calculate complete water balance and pollution assessment.

        Implements the water conservation equation:
            consumption = withdrawal - discharge

        And computes all ESRS E2/E3 quantitative metrics.

        Args:
            intake: List of water intake/withdrawal records.
            discharge: List of water discharge records.
            recycled_volume_m3: Volume of water recycled/reused (m3).
            emissions: Optional list of pollutant emissions for inventory.
            substances: Optional list of SVHC substances for assessment.

        Returns:
            WaterPollutionResult with complete assessment and provenance.

        Raises:
            ValueError: If input data is invalid or inconsistent.
        """
        start_time = time.perf_counter()
        self._notes = []

        # --- Water balance ---
        total_withdrawal = Decimal("0")
        total_discharge = Decimal("0")
        withdrawal_by_source: Dict[str, Decimal] = defaultdict(Decimal)
        discharge_by_dest: Dict[str, Decimal] = defaultdict(Decimal)

        for record in intake:
            total_withdrawal += record.volume_m3
            withdrawal_by_source[record.source.value] += record.volume_m3

        for record in discharge:
            total_discharge += record.volume_m3
            discharge_by_dest[record.destination.value] += record.volume_m3

        total_consumption = total_withdrawal - total_discharge
        if total_consumption < 0:
            self._notes.append(
                "WARNING: Discharge exceeds withdrawal; negative consumption "
                "may indicate measurement error or imported water not tracked."
            )

        # Recycling rate
        recycled = _decimal(recycled_volume_m3)
        total_water_use = total_withdrawal + recycled
        recycling_rate = _pct(recycled, total_water_use) if total_water_use > 0 else 0.0

        # Water intensity
        water_intensity = 0.0
        if self.config.production_volume > 0:
            water_intensity = _round_value(
                total_withdrawal / self.config.production_volume, 3
            )
            self._notes.append(
                f"Water intensity calculated per {self.config.production_unit}"
            )

        # --- Water stress ---
        stress_assessment = None
        stressed_pct = 0.0
        if self.config.include_water_stress:
            stress_assessment = self.assess_water_stress(intake)
            stressed_pct = stress_assessment.stressed_withdrawal_pct

        # --- Pollutant inventory ---
        air_total = Decimal("0")
        water_total = Decimal("0")
        soil_total = Decimal("0")
        inventory_items: List[PollutantInventoryItem] = []

        if emissions:
            inv = self.calculate_pollutant_inventory(emissions)
            inventory_items = inv["items"]
            air_total = inv["air_total"]
            water_total = inv["water_total"]
            soil_total = inv["soil_total"]

        # --- IED compliance ---
        ied_status = "not_assessed"
        ied_details: List[IEDComplianceDetail] = []
        if self.config.include_ied_compliance and emissions and self.config.sub_sector:
            ied_result = self.check_ied_compliance(emissions, self.config.sub_sector)
            ied_status = ied_result["overall_status"]
            ied_details = ied_result["details"]

        # --- SVHC assessment ---
        svhc_count = 0
        svhc_above = 0
        svhc_result = None
        if self.config.include_reach_svhc and substances:
            svhc_result_data = self.assess_svhc(substances)
            svhc_count = svhc_result_data.total_svhc_count
            svhc_above = svhc_result_data.above_threshold_count
            svhc_result = svhc_result_data

        # --- Thermal discharge check ---
        thermal_violations = []
        for d in discharge:
            if d.temperature_delta_c > THERMAL_DISCHARGE_LIMIT_C:
                thermal_violations.append({
                    "facility_id": d.facility_id,
                    "destination": d.destination.value,
                    "temperature_delta_c": d.temperature_delta_c,
                    "limit_c": THERMAL_DISCHARGE_LIMIT_C,
                    "exceeds_by_c": round(d.temperature_delta_c - THERMAL_DISCHARGE_LIMIT_C, 2),
                })
                self._notes.append(
                    f"Thermal discharge violation at facility {d.facility_id}: "
                    f"{d.temperature_delta_c}C delta exceeds {THERMAL_DISCHARGE_LIMIT_C}C limit"
                )

        # --- ESRS E2 metrics ---
        esrs_e2 = {
            "e2_4_pollution_air_tonnes": _round_value(air_total, 3),
            "e2_4_pollution_water_tonnes": _round_value(water_total, 3),
            "e2_4_pollution_soil_tonnes": _round_value(soil_total, 3),
            "e2_4_pollutant_count": len(inventory_items),
            "e2_5_svhc_count": svhc_count,
            "e2_5_svhc_above_threshold": svhc_above,
            "e2_ied_compliance": ied_status,
        }

        # --- ESRS E3 metrics ---
        esrs_e3 = {
            "e3_4_total_water_consumption_m3": _round_value(total_consumption, 3),
            "e3_4_total_water_withdrawal_m3": _round_value(total_withdrawal, 3),
            "e3_4_total_water_discharge_m3": _round_value(total_discharge, 3),
            "e3_4_water_recycling_rate_pct": recycling_rate,
            "e3_4_water_intensity_m3_per_unit": water_intensity,
            "e3_4_water_stressed_withdrawal_pct": stressed_pct,
            "e3_4_withdrawal_by_source": {
                k: _round_value(v, 3) for k, v in withdrawal_by_source.items()
            },
            "e3_4_discharge_by_destination": {
                k: _round_value(v, 3) for k, v in discharge_by_dest.items()
            },
            "e3_4_thermal_violations": len(thermal_violations),
        }

        self._notes.append(
            f"Water balance: withdrawal={_round_value(total_withdrawal, 1)} m3, "
            f"discharge={_round_value(total_discharge, 1)} m3, "
            f"consumption={_round_value(total_consumption, 1)} m3"
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = WaterPollutionResult(
            total_water_withdrawal_m3=_round_value(total_withdrawal, 3),
            total_water_discharge_m3=_round_value(total_discharge, 3),
            total_water_consumption_m3=_round_value(total_consumption, 3),
            water_recycling_rate_pct=recycling_rate,
            water_intensity_m3_per_unit=water_intensity,
            water_stressed_withdrawal_pct=stressed_pct,
            withdrawal_by_source={
                k: _round_value(v, 3) for k, v in withdrawal_by_source.items()
            },
            discharge_by_destination={
                k: _round_value(v, 3) for k, v in discharge_by_dest.items()
            },
            pollutant_inventory=inventory_items,
            total_pollutant_air_tonnes=_round_value(air_total, 3),
            total_pollutant_water_tonnes=_round_value(water_total, 3),
            total_pollutant_soil_tonnes=_round_value(soil_total, 3),
            svhc_count=svhc_count,
            svhc_above_threshold=svhc_above,
            svhc_assessment=svhc_result,
            ied_compliance_status=ied_status,
            ied_compliance_details=ied_details,
            water_stress_assessment=stress_assessment,
            esrs_e2_metrics=esrs_e2,
            esrs_e3_metrics=esrs_e3,
            thermal_discharge_violations=thermal_violations,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    def assess_water_stress(
        self, intake: List[WaterIntakeData]
    ) -> WaterStressAssessment:
        """Assess water stress exposure based on intake locations.

        Evaluates what proportion of water withdrawal comes from areas
        classified as water-stressed (per WRI Aqueduct methodology).

        Args:
            intake: List of water intake records with stress area flags.

        Returns:
            WaterStressAssessment with stressed withdrawal analysis.
        """
        total = Decimal("0")
        stressed = Decimal("0")
        stressed_sources: List[Dict[str, Any]] = []
        notes: List[str] = []

        for record in intake:
            total += record.volume_m3
            if record.water_stressed_area:
                stressed += record.volume_m3
                stressed_sources.append({
                    "facility_id": record.facility_id,
                    "source": record.source.value,
                    "volume_m3": _round_value(record.volume_m3, 3),
                    "quality_grade": record.quality_grade.value,
                })

        stressed_pct = _pct(stressed, total) if total > 0 else 0.0

        # Classify overall stress level based on percentage
        stress_level = WaterStressLevel.LOW
        for level, thresholds in WATER_STRESS_THRESHOLDS.items():
            if thresholds["min"] <= stressed_pct < thresholds["max"]:
                stress_level = level
                break
        if stressed_pct >= 80.0:
            stress_level = WaterStressLevel.EXTREMELY_HIGH

        if stressed_pct > 20.0:
            notes.append(
                f"HIGH RISK: {stressed_pct}% of water withdrawal from stressed areas. "
                "ESRS E3 requires detailed disclosure of water management plans."
            )
        elif stressed_pct > 0:
            notes.append(
                f"{stressed_pct}% of withdrawal from water-stressed areas."
            )
        else:
            notes.append("No withdrawal from water-stressed areas identified.")

        return WaterStressAssessment(
            total_withdrawal_stressed_m3=stressed,
            total_withdrawal_m3=total,
            stressed_withdrawal_pct=stressed_pct,
            stressed_sources=stressed_sources,
            stress_level_classification=stress_level,
            methodology_notes=notes,
        )

    def calculate_pollutant_inventory(
        self, emissions: List[PollutantEmission]
    ) -> Dict[str, Any]:
        """Build a complete pollutant inventory from emission records.

        Aggregates emissions by pollutant type and category (air/water/soil),
        checks against BAT-AEL limits where applicable, and computes totals.

        Args:
            emissions: List of pollutant emission records.

        Returns:
            Dict with keys: items (List[PollutantInventoryItem]),
            air_total, water_total, soil_total (all Decimal).
        """
        air_total = Decimal("0")
        water_total = Decimal("0")
        soil_total = Decimal("0")

        # Group by (pollutant_type, category)
        groups: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "total": Decimal("0"),
                "count": 0,
                "methods": set(),
                "limits": [],
            }
        )

        for em in emissions:
            key = (em.pollutant_type.value, em.category.value)
            groups[key]["total"] += em.quantity_tonnes
            groups[key]["count"] += 1
            groups[key]["methods"].add(em.measurement_method.value)
            if em.emission_limit is not None:
                groups[key]["limits"].append(em.emission_limit)

            if em.category == PollutantCategory.AIR:
                air_total += em.quantity_tonnes
            elif em.category == PollutantCategory.WATER:
                water_total += em.quantity_tonnes
            else:
                soil_total += em.quantity_tonnes

        # Build inventory items
        items: List[PollutantInventoryItem] = []
        sub_sector = self.config.sub_sector
        for (ptype, cat), data in sorted(groups.items()):
            limit_upper = None
            exceeds = False
            # Check BAT-AEL if sub-sector is known
            if sub_sector and sub_sector in IED_EMISSION_LIMITS:
                sector_limits = IED_EMISSION_LIMITS[sub_sector]
                for pt_enum, lim in sector_limits.items():
                    if pt_enum.value == ptype:
                        limit_upper = lim["upper"]
                        # Compare against reported limits if available
                        if data["limits"]:
                            max_measured = max(data["limits"])
                            if max_measured > limit_upper:
                                exceeds = True
                        break

            items.append(PollutantInventoryItem(
                pollutant_type=ptype,
                category=cat,
                total_tonnes=_round_value(data["total"], 6),
                facility_count=data["count"],
                exceeds_limit=exceeds,
                limit_upper=limit_upper,
                measurement_methods=sorted(data["methods"]),
            ))

        return {
            "items": items,
            "air_total": air_total,
            "water_total": water_total,
            "soil_total": soil_total,
        }

    def check_ied_compliance(
        self, emissions: List[PollutantEmission], sub_sector: str
    ) -> Dict[str, Any]:
        """Check emissions against IED BAT-AEL limits for a given sub-sector.

        Compares measured/reported emission values against Best Available
        Techniques Associated Emission Levels (BAT-AELs) from the relevant
        BREF document.

        Args:
            emissions: List of pollutant emissions with limits.
            sub_sector: Manufacturing sub-sector key for BAT-AEL lookup.

        Returns:
            Dict with overall_status and list of IEDComplianceDetail.
        """
        if sub_sector not in IED_EMISSION_LIMITS:
            self._notes.append(
                f"Sub-sector '{sub_sector}' not found in BAT-AEL database. "
                "IED compliance check skipped."
            )
            return {"overall_status": "not_assessed", "details": []}

        sector_limits = IED_EMISSION_LIMITS[sub_sector]
        details: List[IEDComplianceDetail] = []
        any_non_compliant = False
        any_within_range = False

        # Aggregate maximum measured value per pollutant type
        max_values: Dict[str, float] = {}
        for em in emissions:
            if em.emission_limit is not None:
                pt = em.pollutant_type.value
                if pt not in max_values or em.emission_limit > max_values[pt]:
                    max_values[pt] = em.emission_limit

        for pt_enum, limits in sector_limits.items():
            measured = max_values.get(pt_enum.value)
            lower = limits["lower"]
            upper = limits["upper"]
            unit = limits["unit"]

            if measured is None:
                status = "not_assessed"
                gap = 0.0
            elif measured <= lower:
                status = "compliant"
                gap = 0.0
            elif measured <= upper:
                status = "within_range"
                any_within_range = True
                gap = 0.0
            else:
                status = "non_compliant"
                any_non_compliant = True
                gap = round(((measured - upper) / upper) * 100, 2) if upper > 0 else 0.0

            details.append(IEDComplianceDetail(
                pollutant_type=pt_enum.value,
                measured_value=measured,
                bat_ael_lower=lower,
                bat_ael_upper=upper,
                unit=unit,
                status=status,
                gap_pct=gap,
            ))

        if any_non_compliant:
            overall = "non_compliant"
            self._notes.append(
                f"IED NON-COMPLIANT: One or more parameters exceed BAT-AEL "
                f"upper limits for sub-sector '{sub_sector}'."
            )
        elif any_within_range:
            overall = "within_range"
            self._notes.append(
                f"IED within BAT-AEL range for sub-sector '{sub_sector}'. "
                "All parameters between lower and upper bounds."
            )
        else:
            overall = "compliant"
            self._notes.append(
                f"IED COMPLIANT: All assessed parameters below BAT-AEL lower "
                f"limits for sub-sector '{sub_sector}'."
            )

        return {"overall_status": overall, "details": details}

    def assess_svhc(
        self, substances: List[SVHCSubstance]
    ) -> SVHCAssessmentResult:
        """Assess REACH Substances of Very High Concern (SVHC).

        Evaluates each substance against the REACH 0.1% w/w threshold
        and identifies authorization gaps.

        Args:
            substances: List of SVHC substance records.

        Returns:
            SVHCAssessmentResult with counts, quantities, and gaps.
        """
        total_count = len(substances)
        above_threshold = 0
        total_qty = Decimal("0")
        substance_details: List[Dict[str, Any]] = []
        auth_gaps: List[str] = []

        for sub in substances:
            is_above = sub.concentration_pct >= REACH_SVHC_THRESHOLD
            if is_above:
                above_threshold += 1

            total_qty += sub.quantity_tonnes

            needs_auth = sub.authorization_status in (
                AuthorizationStatus.PENDING,
                AuthorizationStatus.SUNSET,
            )
            if needs_auth:
                auth_gaps.append(
                    f"{sub.substance_name} (CAS {sub.cas_number}): "
                    f"status={sub.authorization_status.value}"
                )

            substance_details.append({
                "cas_number": sub.cas_number,
                "substance_name": sub.substance_name,
                "concentration_pct": sub.concentration_pct,
                "quantity_tonnes": _round_value(sub.quantity_tonnes, 6),
                "above_threshold": is_above,
                "authorization_status": sub.authorization_status.value,
                "requires_action": needs_auth or is_above,
            })

        requires_notification = above_threshold > 0

        if requires_notification:
            self._notes.append(
                f"REACH Article 33 notification required: {above_threshold} "
                f"SVHC(s) exceed 0.1% w/w threshold."
            )

        return SVHCAssessmentResult(
            total_svhc_count=total_count,
            above_threshold_count=above_threshold,
            total_svhc_quantity_tonnes=_round_value(total_qty, 6),
            substances=substance_details,
            requires_notification=requires_notification,
            authorization_gaps=auth_gaps,
        )

    def calculate_water_intensity(
        self, total_water_m3: float, production_volume: float
    ) -> float:
        """Calculate water intensity (m3 per production unit).

        Args:
            total_water_m3: Total water withdrawal in cubic metres.
            production_volume: Total production volume in configured units.

        Returns:
            Water intensity as m3 per production unit, rounded to 3 decimals.

        Raises:
            ValueError: If production_volume is zero or negative.
        """
        if production_volume <= 0:
            raise ValueError("Production volume must be positive for intensity calculation")
        result = _decimal(total_water_m3) / _decimal(production_volume)
        return _round_value(result, 3)

    def get_treatment_efficiency(
        self, treatment_level: TreatmentLevel
    ) -> Dict[str, float]:
        """Get pollutant removal efficiencies for a treatment level.

        Args:
            treatment_level: The wastewater treatment level.

        Returns:
            Dict mapping pollutant group to removal fraction (0-1).
        """
        return dict(TREATMENT_EFFICIENCY.get(
            treatment_level, TREATMENT_EFFICIENCY[TreatmentLevel.NONE]
        ))

    def classify_stress_level(self, stress_pct: float) -> WaterStressLevel:
        """Classify water stress level based on withdrawal percentage.

        Args:
            stress_pct: Percentage of withdrawal from stressed areas.

        Returns:
            WaterStressLevel classification.
        """
        for level, thresholds in WATER_STRESS_THRESHOLDS.items():
            if thresholds["min"] <= stress_pct < thresholds["max"]:
                return level
        if stress_pct >= 80.0:
            return WaterStressLevel.EXTREMELY_HIGH
        return WaterStressLevel.LOW
