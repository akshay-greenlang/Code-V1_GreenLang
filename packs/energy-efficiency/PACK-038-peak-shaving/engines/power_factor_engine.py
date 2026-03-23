# -*- coding: utf-8 -*-
"""
PowerFactorEngine - PACK-038 Peak Shaving Engine 8
====================================================

Power factor and reactive power analysis engine.  Analyses power
factor quality, sizes capacitor bank and active filter corrections,
assesses harmonic distortion profiles, calculates utility power factor
penalties, and estimates savings from correction investments.

Calculation Methodology:
    Power Factor:
        PF = kW / kVA  (apparent power triangle)
        PF_angle = arccos(PF)
        reactive_kvar = sqrt(kVA^2 - kW^2)
        reactive_kvar = kW * tan(arccos(PF))

    Capacitor Sizing:
        required_kvar = kW * (tan(arccos(PF_current)) - tan(arccos(PF_target)))
        bank_stages = ceil(required_kvar / stage_kvar)
        total_kvar = bank_stages * stage_kvar

    Harmonic THD:
        THD_v = sqrt(sum(V_h^2 for h=2..50)) / V_1 * 100
        THD_i = sqrt(sum(I_h^2 for h=2..50)) / I_1 * 100
        resonance_freq = sqrt(kVA_sc / kvar_cap) * f_fundamental

    Power Factor Penalty:
        kVA billing: billed = kW / min_PF (e.g. 0.90)
        PF penalty: surcharge = base_charge * (min_PF / actual_PF - 1)
        kVAR charge: charge = excess_kvar * kvar_rate

    Capacity Recovery:
        recovered_kva = kw / pf_new - kw / pf_old
        pct_capacity = recovered_kva / transformer_kva * 100

Regulatory References:
    - IEEE 519-2022 - Standard for Harmonic Control
    - IEEE 1459-2010 - Power Quality Measurement
    - IEC 61000-3-2 - Limits for Harmonic Current Emissions
    - IEC 61642 - Industrial AC Networks: Capacitors and Filters
    - NEMA MG 1-2016 - Motors and Generators (derating for harmonics)
    - NEC Article 460 - Capacitors
    - EN 50160 - Voltage Characteristics of Public Networks

Zero-Hallucination:
    - Power triangle calculations use deterministic trigonometry
    - Capacitor sizing from IEEE/IEC standard formulas
    - THD calculations from harmonic spectrum summation
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-038 Peak Shaving
Engine:  8 of 10
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


class PFStatus(str, Enum):
    """Power factor quality classification.

    EXCELLENT:  PF >= 0.98 - no correction needed.
    GOOD:       PF 0.95-0.97 - acceptable for most utilities.
    FAIR:       PF 0.90-0.94 - correction recommended.
    POOR:       PF 0.80-0.89 - correction strongly recommended.
    CRITICAL:   PF < 0.80 - immediate correction required.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class CorrectionType(str, Enum):
    """Power factor correction equipment type.

    FIXED_CAPACITOR:    Fixed capacitor bank (single stage).
    SWITCHED_CAPACITOR: Automatically switched capacitor bank.
    ACTIVE_FILTER:      Active harmonic filter / active PF corrector.
    VSD_TUNING:         Variable speed drive PF optimisation.
    COMBINED:           Combined capacitor + active filter solution.
    """
    FIXED_CAPACITOR = "fixed_capacitor"
    SWITCHED_CAPACITOR = "switched_capacitor"
    ACTIVE_FILTER = "active_filter"
    VSD_TUNING = "vsd_tuning"
    COMBINED = "combined"


class LoadCategory(str, Enum):
    """Electrical load harmonic category.

    LINEAR:      Linear loads (resistive, inductive motors).
    NON_LINEAR:  Non-linear loads (VFDs, rectifiers, LED drivers).
    MIXED:       Mixed linear and non-linear loads.
    """
    LINEAR = "linear"
    NON_LINEAR = "non_linear"
    MIXED = "mixed"


class HarmonicOrder(str, Enum):
    """Harmonic order for THD analysis.

    Odd harmonics are most common in three-phase systems.
    H5 and H7 are dominant for 6-pulse rectifier loads.

    H3:  3rd harmonic (triplen, zero-sequence).
    H5:  5th harmonic (most common).
    H7:  7th harmonic.
    H9:  9th harmonic (triplen).
    H11: 11th harmonic.
    H13: 13th harmonic.
    """
    H3 = "h3"
    H5 = "h5"
    H7 = "h7"
    H9 = "h9"
    H11 = "h11"
    H13 = "h13"


class BillingMethod(str, Enum):
    """Utility power factor billing method.

    KW_ONLY:     Billed on kW only (no PF penalty).
    KVA_BILLING: Billed on kVA (inherent PF incentive).
    PF_PENALTY:  Surcharge when PF below minimum.
    KVAR_CHARGE: Separate charge for reactive power (kVAR).
    """
    KW_ONLY = "kw_only"
    KVA_BILLING = "kva_billing"
    PF_PENALTY = "pf_penalty"
    KVAR_CHARGE = "kvar_charge"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Power factor status thresholds.
PF_THRESHOLDS: Dict[str, Decimal] = {
    PFStatus.EXCELLENT.value: Decimal("0.98"),
    PFStatus.GOOD.value: Decimal("0.95"),
    PFStatus.FAIR.value: Decimal("0.90"),
    PFStatus.POOR.value: Decimal("0.80"),
    PFStatus.CRITICAL.value: Decimal("0.00"),
}

# Correction equipment costs (USD per kVAR installed).
CORRECTION_COSTS: Dict[str, Dict[str, Decimal]] = {
    CorrectionType.FIXED_CAPACITOR.value: {
        "cost_per_kvar": Decimal("12"),
        "installation_factor": Decimal("1.50"),
        "annual_maintenance_pct": Decimal("2.0"),
        "lifespan_years": Decimal("15"),
    },
    CorrectionType.SWITCHED_CAPACITOR.value: {
        "cost_per_kvar": Decimal("25"),
        "installation_factor": Decimal("1.40"),
        "annual_maintenance_pct": Decimal("3.0"),
        "lifespan_years": Decimal("12"),
    },
    CorrectionType.ACTIVE_FILTER.value: {
        "cost_per_kvar": Decimal("80"),
        "installation_factor": Decimal("1.30"),
        "annual_maintenance_pct": Decimal("4.0"),
        "lifespan_years": Decimal("10"),
    },
    CorrectionType.VSD_TUNING.value: {
        "cost_per_kvar": Decimal("5"),
        "installation_factor": Decimal("1.10"),
        "annual_maintenance_pct": Decimal("1.0"),
        "lifespan_years": Decimal("20"),
    },
    CorrectionType.COMBINED.value: {
        "cost_per_kvar": Decimal("45"),
        "installation_factor": Decimal("1.35"),
        "annual_maintenance_pct": Decimal("3.5"),
        "lifespan_years": Decimal("12"),
    },
}

# IEEE 519-2022 voltage THD limits at PCC.
IEEE_519_THD_LIMITS: Dict[str, Decimal] = {
    "thd_v_max_pct": Decimal("5.0"),
    "individual_v_max_pct": Decimal("3.0"),
    "thd_i_max_pct_isc_il_lt20": Decimal("5.0"),
    "thd_i_max_pct_isc_il_20_50": Decimal("8.0"),
    "thd_i_max_pct_isc_il_50_100": Decimal("12.0"),
    "thd_i_max_pct_isc_il_100_1000": Decimal("15.0"),
    "thd_i_max_pct_isc_il_gt1000": Decimal("20.0"),
}

# Common harmonic magnitudes for 6-pulse rectifier (% of fundamental).
TYPICAL_6PULSE_HARMONICS: Dict[str, Decimal] = {
    HarmonicOrder.H5.value: Decimal("20.0"),
    HarmonicOrder.H7.value: Decimal("14.3"),
    HarmonicOrder.H11.value: Decimal("9.1"),
    HarmonicOrder.H13.value: Decimal("7.7"),
}

# Default minimum power factor for penalty calculation.
DEFAULT_MIN_PF: Decimal = Decimal("0.90")

# Standard frequency.
FUNDAMENTAL_FREQ_HZ: Decimal = Decimal("60")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class PowerFactorReading(BaseModel):
    """Power factor measurement reading.

    Attributes:
        reading_id: Reading identifier.
        timestamp: Measurement timestamp.
        kw: Real power (kW).
        kva: Apparent power (kVA).
        kvar: Reactive power (kVAR).
        power_factor: Measured power factor (0-1).
        pf_leading_lagging: Leading or lagging PF.
        voltage_v: RMS voltage (V).
        current_a: RMS current (A).
        frequency_hz: System frequency (Hz).
        load_category: Load type classification.
        notes: Additional notes.
    """
    reading_id: str = Field(
        default_factory=_new_uuid, description="Reading ID"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Measurement timestamp"
    )
    kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Real power (kW)"
    )
    kva: Decimal = Field(
        default=Decimal("0"), ge=0, description="Apparent power (kVA)"
    )
    kvar: Decimal = Field(
        default=Decimal("0"), ge=0, description="Reactive power (kVAR)"
    )
    power_factor: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("1.0"),
        description="Power factor (0-1)"
    )
    pf_leading_lagging: str = Field(
        default="lagging", description="Leading or lagging"
    )
    voltage_v: Decimal = Field(
        default=Decimal("480"), ge=0, description="RMS voltage (V)"
    )
    current_a: Decimal = Field(
        default=Decimal("0"), ge=0, description="RMS current (A)"
    )
    frequency_hz: Decimal = Field(
        default=FUNDAMENTAL_FREQ_HZ, description="System frequency (Hz)"
    )
    load_category: LoadCategory = Field(
        default=LoadCategory.MIXED, description="Load category"
    )
    notes: str = Field(
        default="", max_length=1000, description="Notes"
    )

    @field_validator("power_factor", mode="before")
    @classmethod
    def compute_pf_if_zero(cls, v: Any, info: Any) -> Any:
        """Allow zero PF to be computed later from kW/kVA."""
        return v


class ReactiveAnalysis(BaseModel):
    """Reactive power analysis input for correction sizing.

    Attributes:
        analysis_id: Analysis identifier.
        avg_kw: Average real power (kW).
        avg_kva: Average apparent power (kVA).
        avg_kvar: Average reactive power (kVAR).
        current_pf: Current average power factor.
        target_pf: Target power factor after correction.
        max_kw: Maximum real power (kW).
        transformer_kva: Transformer/service capacity (kVA).
        short_circuit_kva: Short circuit capacity at PCC (kVA).
        load_category: Load harmonic category.
    """
    analysis_id: str = Field(
        default_factory=_new_uuid, description="Analysis ID"
    )
    avg_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Average kW"
    )
    avg_kva: Decimal = Field(
        default=Decimal("0"), ge=0, description="Average kVA"
    )
    avg_kvar: Decimal = Field(
        default=Decimal("0"), ge=0, description="Average kVAR"
    )
    current_pf: Decimal = Field(
        default=Decimal("0.85"), ge=Decimal("0.10"), le=Decimal("1.0"),
        description="Current PF"
    )
    target_pf: Decimal = Field(
        default=Decimal("0.95"), ge=Decimal("0.80"), le=Decimal("1.0"),
        description="Target PF"
    )
    max_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Maximum kW"
    )
    transformer_kva: Decimal = Field(
        default=Decimal("0"), ge=0, description="Transformer kVA"
    )
    short_circuit_kva: Decimal = Field(
        default=Decimal("0"), ge=0, description="Short circuit kVA"
    )
    load_category: LoadCategory = Field(
        default=LoadCategory.MIXED, description="Load category"
    )

    @field_validator("target_pf")
    @classmethod
    def validate_target(cls, v: Decimal) -> Decimal:
        """Ensure target PF is reasonable."""
        if v < Decimal("0.80"):
            raise ValueError("Target PF must be >= 0.80")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class CorrectionSizing(BaseModel):
    """Power factor correction equipment sizing result.

    Attributes:
        sizing_id: Sizing result identifier.
        correction_type: Recommended correction type.
        required_kvar: Required reactive power correction (kVAR).
        bank_stages: Number of switched stages.
        stage_kvar: kVAR per stage.
        total_installed_kvar: Total installed kVAR.
        pf_before: Power factor before correction.
        pf_after: Estimated power factor after correction.
        kva_reduction: kVA reduction from correction.
        current_reduction_a: Current reduction (A).
        capacity_recovered_pct: Transformer capacity recovered (%).
        equipment_cost_usd: Equipment cost (USD).
        installation_cost_usd: Installation cost (USD).
        total_cost_usd: Total installed cost (USD).
        annual_maintenance_usd: Annual maintenance (USD).
        resonance_check: Resonance frequency check result.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    sizing_id: str = Field(
        default_factory=_new_uuid, description="Sizing ID"
    )
    correction_type: CorrectionType = Field(
        default=CorrectionType.SWITCHED_CAPACITOR, description="Correction type"
    )
    required_kvar: Decimal = Field(
        default=Decimal("0"), description="Required kVAR"
    )
    bank_stages: int = Field(
        default=1, ge=1, description="Bank stages"
    )
    stage_kvar: Decimal = Field(
        default=Decimal("25"), description="kVAR per stage"
    )
    total_installed_kvar: Decimal = Field(
        default=Decimal("0"), description="Total installed kVAR"
    )
    pf_before: Decimal = Field(
        default=Decimal("0"), description="PF before"
    )
    pf_after: Decimal = Field(
        default=Decimal("0"), description="PF after"
    )
    kva_reduction: Decimal = Field(
        default=Decimal("0"), description="kVA reduction"
    )
    current_reduction_a: Decimal = Field(
        default=Decimal("0"), description="Current reduction (A)"
    )
    capacity_recovered_pct: Decimal = Field(
        default=Decimal("0"), description="Capacity recovered (%)"
    )
    equipment_cost_usd: Decimal = Field(
        default=Decimal("0"), description="Equipment cost (USD)"
    )
    installation_cost_usd: Decimal = Field(
        default=Decimal("0"), description="Installation cost (USD)"
    )
    total_cost_usd: Decimal = Field(
        default=Decimal("0"), description="Total cost (USD)"
    )
    annual_maintenance_usd: Decimal = Field(
        default=Decimal("0"), description="Annual maintenance (USD)"
    )
    resonance_check: Dict[str, Any] = Field(
        default_factory=dict, description="Resonance check"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class HarmonicProfile(BaseModel):
    """Harmonic distortion analysis profile.

    Attributes:
        profile_id: Profile identifier.
        thd_voltage_pct: Total voltage harmonic distortion (%).
        thd_current_pct: Total current harmonic distortion (%).
        individual_harmonics: Individual harmonic magnitudes.
        ieee_519_compliant: Whether THD is within IEEE 519 limits.
        resonance_risk: Resonance risk assessment.
        recommended_filter: Recommended filtering approach.
        derating_required_pct: Motor/transformer derating required (%).
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    profile_id: str = Field(
        default_factory=_new_uuid, description="Profile ID"
    )
    thd_voltage_pct: Decimal = Field(
        default=Decimal("0"), description="THD voltage (%)"
    )
    thd_current_pct: Decimal = Field(
        default=Decimal("0"), description="THD current (%)"
    )
    individual_harmonics: Dict[str, Decimal] = Field(
        default_factory=dict, description="Individual harmonics"
    )
    ieee_519_compliant: bool = Field(
        default=True, description="IEEE 519 compliant"
    )
    resonance_risk: str = Field(
        default="low", description="Resonance risk"
    )
    recommended_filter: str = Field(
        default="", description="Recommended filter"
    )
    derating_required_pct: Decimal = Field(
        default=Decimal("0"), description="Derating required (%)"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class PowerFactorResult(BaseModel):
    """Comprehensive power factor analysis result.

    Attributes:
        result_id: Result identifier.
        pf_status: Power factor quality status.
        avg_power_factor: Average measured PF.
        min_power_factor: Minimum measured PF.
        max_power_factor: Maximum measured PF.
        avg_kw: Average real power (kW).
        avg_kva: Average apparent power (kVA).
        avg_kvar: Average reactive power (kVAR).
        correction_sizing: Correction equipment sizing.
        harmonic_profile: Harmonic distortion profile.
        penalty_assessment: Penalty calculation.
        savings_estimate: Savings from correction.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID"
    )
    pf_status: PFStatus = Field(
        default=PFStatus.FAIR, description="PF status"
    )
    avg_power_factor: Decimal = Field(
        default=Decimal("0"), description="Average PF"
    )
    min_power_factor: Decimal = Field(
        default=Decimal("0"), description="Minimum PF"
    )
    max_power_factor: Decimal = Field(
        default=Decimal("0"), description="Maximum PF"
    )
    avg_kw: Decimal = Field(
        default=Decimal("0"), description="Average kW"
    )
    avg_kva: Decimal = Field(
        default=Decimal("0"), description="Average kVA"
    )
    avg_kvar: Decimal = Field(
        default=Decimal("0"), description="Average kVAR"
    )
    correction_sizing: Optional[CorrectionSizing] = Field(
        default=None, description="Correction sizing"
    )
    harmonic_profile: Optional[HarmonicProfile] = Field(
        default=None, description="Harmonic profile"
    )
    penalty_assessment: Optional[Dict[str, Any]] = Field(
        default=None, description="Penalty assessment"
    )
    savings_estimate: Optional[Dict[str, Any]] = Field(
        default=None, description="Savings estimate"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PowerFactorEngine:
    """Power factor and reactive power analysis engine.

    Analyses power factor quality, sizes correction equipment,
    assesses harmonic distortion, calculates utility PF penalties,
    and estimates savings from correction investments.

    Usage::

        engine = PowerFactorEngine()
        result = engine.analyze_power_factor(readings)
        sizing = engine.size_correction(analysis)
        harmonics = engine.assess_harmonics(harmonic_data)
        penalties = engine.calculate_penalties(readings, billing_method)
        savings = engine.estimate_savings(analysis, correction_type)

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise PowerFactorEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - min_pf (Decimal): minimum acceptable PF
                - target_pf (Decimal): correction target PF
                - stage_kvar (Decimal): capacitor bank stage size
                - discount_rate (Decimal): financial discount rate
        """
        self.config = config or {}
        self._min_pf = _decimal(
            self.config.get("min_pf", DEFAULT_MIN_PF)
        )
        self._target_pf = _decimal(
            self.config.get("target_pf", Decimal("0.95"))
        )
        self._stage_kvar = _decimal(
            self.config.get("stage_kvar", Decimal("25"))
        )
        self._discount_rate = _decimal(
            self.config.get("discount_rate", Decimal("0.08"))
        )
        logger.info(
            "PowerFactorEngine v%s initialised (min_pf=%.2f, target=%.2f, stage=%.0f kVAR)",
            self.engine_version,
            float(self._min_pf),
            float(self._target_pf),
            float(self._stage_kvar),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze_power_factor(
        self,
        readings: List[PowerFactorReading],
    ) -> PowerFactorResult:
        """Analyse power factor from measurement readings.

        Computes average, minimum, and maximum PF, classifies PF status,
        calculates reactive power consumption, and provides initial
        correction recommendations.

        Args:
            readings: List of PF measurement readings.

        Returns:
            PowerFactorResult with comprehensive PF analysis.

        Raises:
            ValueError: If no readings provided.
        """
        t0 = time.perf_counter()
        logger.info("Analysing power factor: %d readings", len(readings))

        if not readings:
            raise ValueError("No power factor readings provided.")

        # Compute PF for readings where it is zero
        processed_readings = self._preprocess_readings(readings)

        # Statistics
        pf_values = [r.power_factor for r in processed_readings if r.power_factor > Decimal("0")]
        if not pf_values:
            raise ValueError("No valid PF measurements in readings.")

        avg_pf = sum(pf_values, Decimal("0")) / _decimal(len(pf_values))
        min_pf = min(pf_values)
        max_pf = max(pf_values)

        # Average power components
        avg_kw = sum(
            (r.kw for r in processed_readings), Decimal("0")
        ) / _decimal(len(processed_readings))
        avg_kva = sum(
            (r.kva for r in processed_readings), Decimal("0")
        ) / _decimal(len(processed_readings))
        avg_kvar = sum(
            (r.kvar for r in processed_readings), Decimal("0")
        ) / _decimal(len(processed_readings))

        # PF Status
        status = self._classify_pf(avg_pf)

        elapsed = (time.perf_counter() - t0) * 1000.0

        result = PowerFactorResult(
            pf_status=status,
            avg_power_factor=_round_val(avg_pf, 4),
            min_power_factor=_round_val(min_pf, 4),
            max_power_factor=_round_val(max_pf, 4),
            avg_kw=_round_val(avg_kw, 2),
            avg_kva=_round_val(avg_kva, 2),
            avg_kvar=_round_val(avg_kvar, 2),
            processing_time_ms=round(elapsed, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "PF analysis: avg=%.4f (%s), min=%.4f, max=%.4f, "
            "kW=%.0f, kVA=%.0f, kVAR=%.0f, hash=%s (%.1f ms)",
            float(avg_pf), status.value, float(min_pf), float(max_pf),
            float(avg_kw), float(avg_kva), float(avg_kvar),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def size_correction(
        self,
        analysis: ReactiveAnalysis,
        correction_type: CorrectionType = CorrectionType.SWITCHED_CAPACITOR,
    ) -> CorrectionSizing:
        """Size power factor correction equipment.

        Calculates required kVAR, determines bank stages, checks for
        resonance, and computes equipment costs.

        kVAR_required = kW * (tan(arccos(PF_current)) - tan(arccos(PF_target)))

        Args:
            analysis: Reactive power analysis input.
            correction_type: Type of correction equipment.

        Returns:
            CorrectionSizing with equipment specifications and costs.
        """
        t0 = time.perf_counter()
        logger.info(
            "Sizing correction: kW=%.0f, PF %.3f -> %.3f, type=%s",
            float(analysis.avg_kw), float(analysis.current_pf),
            float(analysis.target_pf), correction_type.value,
        )

        kw = analysis.avg_kw
        pf_current = analysis.current_pf
        pf_target = analysis.target_pf

        # Calculate required kVAR
        angle_current = _decimal(math.acos(min(float(pf_current), 1.0)))
        angle_target = _decimal(math.acos(min(float(pf_target), 1.0)))

        tan_current = _decimal(math.tan(float(angle_current)))
        tan_target = _decimal(math.tan(float(angle_target)))

        required_kvar = kw * (tan_current - tan_target)
        required_kvar = max(required_kvar, Decimal("0"))

        # Bank stages
        stage_kvar = self._stage_kvar
        if stage_kvar <= Decimal("0"):
            stage_kvar = Decimal("25")

        bank_stages = 1
        if required_kvar > Decimal("0"):
            # Ceiling division
            bank_stages = int(
                (required_kvar / stage_kvar).to_integral_value(rounding=ROUND_HALF_UP)
            )
            bank_stages = max(bank_stages, 1)

        total_kvar = _decimal(bank_stages) * stage_kvar

        # PF after correction
        kvar_after = max(analysis.avg_kvar - total_kvar, Decimal("0"))
        kva_after = _decimal(math.sqrt(float(kw ** 2 + kvar_after ** 2)))
        pf_after = _safe_divide(kw, kva_after, Decimal("1.0"))
        pf_after = min(pf_after, Decimal("1.0"))

        # kVA reduction
        kva_before = analysis.avg_kva
        if kva_before <= Decimal("0"):
            kva_before = _safe_divide(kw, pf_current, kw)
        kva_reduction = kva_before - kva_after

        # Current reduction
        voltage = Decimal("480")
        sqrt3 = _decimal(math.sqrt(3))
        current_before = _safe_divide(kva_before * Decimal("1000"), voltage * sqrt3)
        current_after = _safe_divide(kva_after * Decimal("1000"), voltage * sqrt3)
        current_reduction = current_before - current_after

        # Capacity recovered
        capacity_pct = Decimal("0")
        if analysis.transformer_kva > Decimal("0"):
            capacity_pct = _safe_pct(kva_reduction, analysis.transformer_kva)

        # Costs
        cost_data = CORRECTION_COSTS.get(
            correction_type.value,
            CORRECTION_COSTS[CorrectionType.SWITCHED_CAPACITOR.value],
        )
        equip_cost = total_kvar * cost_data["cost_per_kvar"]
        install_cost = equip_cost * (cost_data["installation_factor"] - Decimal("1"))
        total_cost = equip_cost + install_cost
        annual_maint = equip_cost * cost_data["annual_maintenance_pct"] / Decimal("100")

        # Resonance check
        resonance = self._check_resonance(
            total_kvar, analysis.short_circuit_kva, analysis.avg_kw,
        )

        sizing = CorrectionSizing(
            correction_type=correction_type,
            required_kvar=_round_val(required_kvar, 2),
            bank_stages=bank_stages,
            stage_kvar=_round_val(stage_kvar, 2),
            total_installed_kvar=_round_val(total_kvar, 2),
            pf_before=_round_val(pf_current, 4),
            pf_after=_round_val(pf_after, 4),
            kva_reduction=_round_val(kva_reduction, 2),
            current_reduction_a=_round_val(current_reduction, 2),
            capacity_recovered_pct=_round_val(capacity_pct, 2),
            equipment_cost_usd=_round_val(equip_cost, 2),
            installation_cost_usd=_round_val(install_cost, 2),
            total_cost_usd=_round_val(total_cost, 2),
            annual_maintenance_usd=_round_val(annual_maint, 2),
            resonance_check=resonance,
        )
        sizing.provenance_hash = _compute_hash(sizing)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Correction sizing: %.0f kVAR (%d stages x %.0f), "
            "PF %.3f->%.3f, cost=$%.2f, hash=%s (%.1f ms)",
            float(total_kvar), bank_stages, float(stage_kvar),
            float(pf_current), float(pf_after), float(total_cost),
            sizing.provenance_hash[:16], elapsed,
        )
        return sizing

    def assess_harmonics(
        self,
        harmonic_magnitudes: Dict[str, Decimal],
        fundamental_current_a: Decimal = Decimal("100"),
        fundamental_voltage_v: Decimal = Decimal("480"),
        short_circuit_ratio: Optional[Decimal] = None,
    ) -> HarmonicProfile:
        """Assess harmonic distortion profile.

        Calculates total harmonic distortion (THD) for voltage and
        current, checks IEEE 519-2022 compliance, and recommends
        filtering if needed.

        Args:
            harmonic_magnitudes: Harmonic order -> magnitude (% of fundamental).
            fundamental_current_a: Fundamental current magnitude (A).
            fundamental_voltage_v: Fundamental voltage magnitude (V).
            short_circuit_ratio: Isc/IL ratio at PCC.

        Returns:
            HarmonicProfile with THD and compliance assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Assessing harmonics: %d orders, I1=%.0fA, V1=%.0fV",
            len(harmonic_magnitudes), float(fundamental_current_a),
            float(fundamental_voltage_v),
        )

        # Calculate THD_i (current)
        sum_sq_i = Decimal("0")
        for order, magnitude_pct in harmonic_magnitudes.items():
            sum_sq_i += (magnitude_pct / Decimal("100")) ** 2

        thd_i = _decimal(math.sqrt(float(sum_sq_i))) * Decimal("100")

        # THD_v (estimate from current harmonics, simplified)
        # For a given system impedance, V_h approx proportional to I_h
        # Use simplified ratio based on short circuit ratio
        if short_circuit_ratio and short_circuit_ratio > Decimal("0"):
            impedance_factor = _safe_divide(Decimal("1"), short_circuit_ratio)
        else:
            impedance_factor = Decimal("0.05")

        thd_v = thd_i * impedance_factor

        # IEEE 519 compliance
        thd_v_limit = IEEE_519_THD_LIMITS["thd_v_max_pct"]
        if short_circuit_ratio is None or short_circuit_ratio <= Decimal("0"):
            thd_i_limit = IEEE_519_THD_LIMITS["thd_i_max_pct_isc_il_20_50"]
        elif short_circuit_ratio < Decimal("20"):
            thd_i_limit = IEEE_519_THD_LIMITS["thd_i_max_pct_isc_il_lt20"]
        elif short_circuit_ratio < Decimal("50"):
            thd_i_limit = IEEE_519_THD_LIMITS["thd_i_max_pct_isc_il_20_50"]
        elif short_circuit_ratio < Decimal("100"):
            thd_i_limit = IEEE_519_THD_LIMITS["thd_i_max_pct_isc_il_50_100"]
        elif short_circuit_ratio < Decimal("1000"):
            thd_i_limit = IEEE_519_THD_LIMITS["thd_i_max_pct_isc_il_100_1000"]
        else:
            thd_i_limit = IEEE_519_THD_LIMITS["thd_i_max_pct_isc_il_gt1000"]

        compliant = thd_v <= thd_v_limit and thd_i <= thd_i_limit

        # Resonance risk
        resonance_risk = "low"
        dominant = max(harmonic_magnitudes.values(), default=Decimal("0"))
        if dominant > Decimal("15"):
            resonance_risk = "high"
        elif dominant > Decimal("8"):
            resonance_risk = "medium"

        # Recommended filter
        if not compliant:
            if thd_i > Decimal("15"):
                recommended = "Active harmonic filter recommended (high THD-I)"
            elif thd_i > Decimal("8"):
                recommended = "Passive tuned filter or active filter recommended"
            else:
                recommended = "Detuned capacitor bank with reactor recommended"
        else:
            recommended = "No filtering required (IEEE 519 compliant)"

        # Motor/transformer derating
        derating = self._calculate_derating(thd_i, thd_v)

        profile = HarmonicProfile(
            thd_voltage_pct=_round_val(thd_v, 2),
            thd_current_pct=_round_val(thd_i, 2),
            individual_harmonics={k: _round_val(v, 2) for k, v in harmonic_magnitudes.items()},
            ieee_519_compliant=compliant,
            resonance_risk=resonance_risk,
            recommended_filter=recommended,
            derating_required_pct=_round_val(derating, 2),
        )
        profile.provenance_hash = _compute_hash(profile)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Harmonics: THD-V=%.2f%%, THD-I=%.2f%%, IEEE 519=%s, "
            "risk=%s, hash=%s (%.1f ms)",
            float(thd_v), float(thd_i),
            "PASS" if compliant else "FAIL",
            resonance_risk, profile.provenance_hash[:16], elapsed,
        )
        return profile

    def calculate_penalties(
        self,
        readings: List[PowerFactorReading],
        billing_method: BillingMethod = BillingMethod.PF_PENALTY,
        demand_rate_per_kw: Decimal = Decimal("15"),
        kvar_rate: Decimal = Decimal("0.50"),
        min_pf: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Calculate utility power factor penalties.

        Computes PF-related charges under different billing methods
        (kVA billing, PF penalty surcharge, or kVAR charge).

        Args:
            readings: PF measurement readings.
            billing_method: Utility billing method.
            demand_rate_per_kw: Demand charge rate (USD/kW).
            kvar_rate: kVAR charge rate (USD/kVAR).
            min_pf: Minimum PF threshold for penalties.

        Returns:
            Dictionary with penalty calculations.
        """
        t0 = time.perf_counter()
        target_pf = min_pf or self._min_pf
        logger.info(
            "Calculating PF penalties: %d readings, method=%s, min_pf=%.2f",
            len(readings), billing_method.value, float(target_pf),
        )

        if not readings:
            raise ValueError("No readings provided for penalty calculation.")

        processed = self._preprocess_readings(readings)
        monthly_details: List[Dict[str, Any]] = []
        total_penalty = Decimal("0")
        total_base_charge = Decimal("0")

        for reading in processed:
            kw = reading.kw
            pf = reading.power_factor
            kva = reading.kva if reading.kva > Decimal("0") else _safe_divide(kw, pf)
            kvar = reading.kvar

            if billing_method == BillingMethod.KW_ONLY:
                base_charge = kw * demand_rate_per_kw
                penalty = Decimal("0")

            elif billing_method == BillingMethod.KVA_BILLING:
                base_charge = kw * demand_rate_per_kw
                kva_charge = kva * demand_rate_per_kw
                penalty = max(kva_charge - base_charge, Decimal("0"))

            elif billing_method == BillingMethod.PF_PENALTY:
                base_charge = kw * demand_rate_per_kw
                if pf < target_pf and pf > Decimal("0"):
                    surcharge_factor = _safe_divide(target_pf, pf) - Decimal("1")
                    penalty = base_charge * surcharge_factor
                else:
                    penalty = Decimal("0")

            elif billing_method == BillingMethod.KVAR_CHARGE:
                base_charge = kw * demand_rate_per_kw
                # Calculate excess kVAR above what target PF would produce
                target_kvar = kw * _decimal(math.tan(math.acos(float(target_pf))))
                excess_kvar = max(kvar - target_kvar, Decimal("0"))
                penalty = excess_kvar * kvar_rate

            else:
                base_charge = kw * demand_rate_per_kw
                penalty = Decimal("0")

            total_penalty += penalty
            total_base_charge += base_charge

            monthly_details.append({
                "reading_id": reading.reading_id,
                "kw": str(_round_val(kw, 2)),
                "kva": str(_round_val(kva, 2)),
                "kvar": str(_round_val(kvar, 2)),
                "pf": str(_round_val(pf, 4)),
                "base_charge_usd": str(_round_val(base_charge, 2)),
                "penalty_usd": str(_round_val(penalty, 2)),
                "total_charge_usd": str(_round_val(base_charge + penalty, 2)),
            })

        penalty_pct = _safe_pct(total_penalty, total_base_charge)
        annual_penalty = total_penalty * _safe_divide(
            Decimal("12"), _decimal(len(processed))
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "billing_method": billing_method.value,
            "min_pf_threshold": str(_round_val(target_pf, 2)),
            "total_readings": len(processed),
            "total_base_charge_usd": str(_round_val(total_base_charge, 2)),
            "total_penalty_usd": str(_round_val(total_penalty, 2)),
            "penalty_pct_of_base": str(_round_val(penalty_pct, 2)),
            "annualised_penalty_usd": str(_round_val(annual_penalty, 2)),
            "monthly_details": monthly_details,
            "calculated_at": _utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "PF penalties: method=%s, total=$%.2f (%.1f%% of base), "
            "annual=$%.2f, hash=%s (%.1f ms)",
            billing_method.value, float(total_penalty), float(penalty_pct),
            float(annual_penalty), result["provenance_hash"][:16], elapsed,
        )
        return result

    def estimate_savings(
        self,
        analysis: ReactiveAnalysis,
        correction_type: CorrectionType = CorrectionType.SWITCHED_CAPACITOR,
        billing_method: BillingMethod = BillingMethod.PF_PENALTY,
        demand_rate_per_kw: Decimal = Decimal("15"),
        kvar_rate: Decimal = Decimal("0.50"),
    ) -> Dict[str, Any]:
        """Estimate savings from power factor correction.

        Calculates annual savings, payback period, ROI, and lifetime
        value of PF correction investment.

        Args:
            analysis: Reactive power analysis.
            correction_type: Correction equipment type.
            billing_method: Utility billing method.
            demand_rate_per_kw: Demand rate (USD/kW).
            kvar_rate: kVAR rate (USD/kVAR).

        Returns:
            Dictionary with savings analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Estimating savings: PF %.3f->%.3f, type=%s, method=%s",
            float(analysis.current_pf), float(analysis.target_pf),
            correction_type.value, billing_method.value,
        )

        # Size correction
        sizing = self.size_correction(analysis, correction_type)

        # Monthly penalty before correction
        kw = analysis.avg_kw
        pf_before = analysis.current_pf
        pf_after = min(sizing.pf_after, Decimal("1.0"))

        # Calculate savings based on billing method
        if billing_method == BillingMethod.KVA_BILLING:
            kva_before = _safe_divide(kw, pf_before)
            kva_after = _safe_divide(kw, pf_after)
            monthly_before = kva_before * demand_rate_per_kw
            monthly_after = kva_after * demand_rate_per_kw

        elif billing_method == BillingMethod.PF_PENALTY:
            base_charge = kw * demand_rate_per_kw
            if pf_before < self._min_pf and pf_before > Decimal("0"):
                surcharge_before = base_charge * (
                    _safe_divide(self._min_pf, pf_before) - Decimal("1")
                )
            else:
                surcharge_before = Decimal("0")

            if pf_after < self._min_pf and pf_after > Decimal("0"):
                surcharge_after = base_charge * (
                    _safe_divide(self._min_pf, pf_after) - Decimal("1")
                )
            else:
                surcharge_after = Decimal("0")

            monthly_before = base_charge + surcharge_before
            monthly_after = base_charge + surcharge_after

        elif billing_method == BillingMethod.KVAR_CHARGE:
            kvar_before = analysis.avg_kvar
            kvar_after = max(
                kvar_before - sizing.total_installed_kvar, Decimal("0")
            )
            target_kvar = kw * _decimal(math.tan(math.acos(float(self._min_pf))))
            excess_before = max(kvar_before - target_kvar, Decimal("0"))
            excess_after = max(kvar_after - target_kvar, Decimal("0"))
            monthly_before = kw * demand_rate_per_kw + excess_before * kvar_rate
            monthly_after = kw * demand_rate_per_kw + excess_after * kvar_rate

        else:
            monthly_before = kw * demand_rate_per_kw
            monthly_after = kw * demand_rate_per_kw

        monthly_savings = max(monthly_before - monthly_after, Decimal("0"))
        annual_savings = monthly_savings * Decimal("12")
        net_annual = annual_savings - sizing.annual_maintenance_usd

        # Payback
        monthly_net = _safe_divide(net_annual, Decimal("12"))
        payback_months = _safe_divide(sizing.total_cost_usd, monthly_net)

        # ROI
        roi = _safe_pct(net_annual, sizing.total_cost_usd)

        # NPV (equipment lifespan)
        cost_data = CORRECTION_COSTS.get(
            correction_type.value,
            CORRECTION_COSTS[CorrectionType.SWITCHED_CAPACITOR.value],
        )
        lifespan = cost_data["lifespan_years"]
        npv = -sizing.total_cost_usd
        for yr in range(1, int(lifespan) + 1):
            factor = (Decimal("1") + self._discount_rate) ** _decimal(yr)
            npv += _safe_divide(net_annual, factor)

        # Lifetime savings
        lifetime_savings = net_annual * lifespan

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "pf_before": str(_round_val(pf_before, 4)),
            "pf_after": str(_round_val(pf_after, 4)),
            "correction_type": correction_type.value,
            "billing_method": billing_method.value,
            "monthly_before_usd": str(_round_val(monthly_before, 2)),
            "monthly_after_usd": str(_round_val(monthly_after, 2)),
            "monthly_savings_usd": str(_round_val(monthly_savings, 2)),
            "annual_savings_usd": str(_round_val(annual_savings, 2)),
            "annual_maintenance_usd": str(_round_val(sizing.annual_maintenance_usd, 2)),
            "net_annual_savings_usd": str(_round_val(net_annual, 2)),
            "total_investment_usd": str(_round_val(sizing.total_cost_usd, 2)),
            "payback_months": str(_round_val(payback_months, 1)),
            "roi_pct": str(_round_val(roi, 2)),
            "npv_usd": str(_round_val(npv, 2)),
            "lifetime_savings_usd": str(_round_val(lifetime_savings, 2)),
            "lifespan_years": str(lifespan),
            "kvar_installed": str(_round_val(sizing.total_installed_kvar, 2)),
            "kva_reduction": str(_round_val(sizing.kva_reduction, 2)),
            "capacity_recovered_pct": str(_round_val(sizing.capacity_recovered_pct, 2)),
            "calculated_at": _utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Savings estimate: PF %.3f->%.3f, annual=$%.2f, "
            "payback=%.0f mo, ROI=%.1f%%, NPV=$%.2f, hash=%s (%.1f ms)",
            float(pf_before), float(pf_after), float(annual_savings),
            float(payback_months), float(roi), float(npv),
            result["provenance_hash"][:16], elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal: Preprocessing and Classification                          #
    # ------------------------------------------------------------------ #

    def _preprocess_readings(
        self,
        readings: List[PowerFactorReading],
    ) -> List[PowerFactorReading]:
        """Preprocess readings, computing derived values if missing.

        Args:
            readings: Raw PF readings.

        Returns:
            Preprocessed readings with all values computed.
        """
        processed: List[PowerFactorReading] = []
        for r in readings:
            kw = r.kw
            kva = r.kva
            kvar = r.kvar
            pf = r.power_factor

            # Compute PF from kW/kVA if missing
            if pf <= Decimal("0") and kw > Decimal("0") and kva > Decimal("0"):
                pf = min(_safe_divide(kw, kva), Decimal("1.0"))

            # Compute kVA from kW/PF if missing
            if kva <= Decimal("0") and kw > Decimal("0") and pf > Decimal("0"):
                kva = _safe_divide(kw, pf)

            # Compute kVAR if missing
            if kvar <= Decimal("0") and kva > Decimal("0") and kw > Decimal("0"):
                kvar_sq = kva ** 2 - kw ** 2
                if kvar_sq > Decimal("0"):
                    kvar = _decimal(math.sqrt(float(kvar_sq)))
                else:
                    kvar = Decimal("0")

            processed.append(PowerFactorReading(
                reading_id=r.reading_id,
                timestamp=r.timestamp,
                kw=kw,
                kva=kva,
                kvar=kvar,
                power_factor=pf,
                pf_leading_lagging=r.pf_leading_lagging,
                voltage_v=r.voltage_v,
                current_a=r.current_a,
                frequency_hz=r.frequency_hz,
                load_category=r.load_category,
                notes=r.notes,
            ))
        return processed

    def _classify_pf(self, pf: Decimal) -> PFStatus:
        """Classify power factor status.

        Args:
            pf: Power factor value (0-1).

        Returns:
            PFStatus classification.
        """
        if pf >= PF_THRESHOLDS[PFStatus.EXCELLENT.value]:
            return PFStatus.EXCELLENT
        elif pf >= PF_THRESHOLDS[PFStatus.GOOD.value]:
            return PFStatus.GOOD
        elif pf >= PF_THRESHOLDS[PFStatus.FAIR.value]:
            return PFStatus.FAIR
        elif pf >= PF_THRESHOLDS[PFStatus.POOR.value]:
            return PFStatus.POOR
        else:
            return PFStatus.CRITICAL

    # ------------------------------------------------------------------ #
    # Internal: Resonance and Derating                                    #
    # ------------------------------------------------------------------ #

    def _check_resonance(
        self,
        cap_kvar: Decimal,
        short_circuit_kva: Decimal,
        load_kw: Decimal,
    ) -> Dict[str, Any]:
        """Check for potential resonance with capacitor installation.

        resonance_harmonic = sqrt(Ssc / Qcap)
        If resonance_harmonic is near 5, 7, 11, or 13 -> risk.

        Args:
            cap_kvar: Capacitor bank kVAR.
            short_circuit_kva: Short circuit capacity (kVA).
            load_kw: Load power (kW).

        Returns:
            Resonance check result dictionary.
        """
        if cap_kvar <= Decimal("0") or short_circuit_kva <= Decimal("0"):
            return {
                "resonance_harmonic": "N/A",
                "risk": "unknown",
                "recommendation": "Provide short circuit data for resonance check",
            }

        # Resonance harmonic order
        h_res = _decimal(math.sqrt(float(_safe_divide(short_circuit_kva, cap_kvar))))

        # Check proximity to common harmonic orders
        risk_orders = [Decimal("3"), Decimal("5"), Decimal("7"),
                       Decimal("11"), Decimal("13")]
        risk = "low"
        nearest = Decimal("999")
        for h in risk_orders:
            dist = abs(h_res - h)
            if dist < nearest:
                nearest = dist
            if dist < Decimal("0.5"):
                risk = "high"
                break
            elif dist < Decimal("1.0"):
                risk = "medium"

        if risk == "high":
            rec = (
                f"Resonance risk at h={_round_val(h_res, 1)}. "
                f"Use detuned reactor (7% or 14%) with capacitor bank."
            )
        elif risk == "medium":
            rec = (
                f"Moderate resonance risk at h={_round_val(h_res, 1)}. "
                f"Consider detuned reactor."
            )
        else:
            rec = f"Low resonance risk (h={_round_val(h_res, 1)})."

        return {
            "resonance_harmonic": str(_round_val(h_res, 2)),
            "risk": risk,
            "nearest_harmonic_distance": str(_round_val(nearest, 2)),
            "recommendation": rec,
        }

    def _calculate_derating(
        self,
        thd_i_pct: Decimal,
        thd_v_pct: Decimal,
    ) -> Decimal:
        """Calculate motor/transformer derating due to harmonics.

        Per NEMA MG 1, motors must be derated when THD exceeds limits.
        Simplified derating: 1% derating per 2% THD-I above 5%.

        Args:
            thd_i_pct: Current THD (%).
            thd_v_pct: Voltage THD (%).

        Returns:
            Required derating percentage.
        """
        threshold = Decimal("5.0")
        if thd_i_pct <= threshold:
            return Decimal("0")

        excess = thd_i_pct - threshold
        derating = excess * Decimal("0.5")
        return min(derating, Decimal("25"))
