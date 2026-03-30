# -*- coding: utf-8 -*-
"""
UncertaintyEngine - PACK-040 M&V Engine 4
============================================

Comprehensive uncertainty quantification engine for Measurement &
Verification per ASHRAE Guideline 14-2014 and IPMVP Core Concepts 2022.
Computes measurement uncertainty, model (regression) uncertainty, sampling
uncertainty, combined uncertainty via root-sum-square, and fractional
savings uncertainty (FSU) at 68% and 90% confidence levels.

Calculation Methodology:
    Measurement Uncertainty:
        u_measurement = sqrt(u_meter^2 + u_ct_pt^2 + u_calibration^2)

    Model (Regression) Uncertainty:
        u_model = t * sqrt((n+2)/n) * CVRMSE/100 * E_baseline
        where t = t-statistic for (n-p) degrees of freedom at confidence level

    Sampling Uncertainty (Option A):
        u_sampling = t * CV / sqrt(n_sample)
        where CV = coefficient of variation of sampled parameter

    Combined Uncertainty:
        u_combined = sqrt(u_measurement^2 + u_model^2 + u_sampling^2)

    Fractional Savings Uncertainty (FSU):
        FSU = u_combined / savings * 100%
        FSU = t * sqrt((n+2)/n) * (CVRMSE/100) * E_baseline / (E_baseline - E_actual) * 100%

    Minimum Detectable Savings:
        MDS = 2 * u_combined (at 68% confidence)

    Required Sample Size (Option A):
        n = (t * CV / precision)^2

    Prediction Interval:
        y_hat +/- t * se * sqrt(1 + 1/n + (x-x_mean)^2 / Sxx)

    Expanded Uncertainty (k=2, 95%):
        U = k * u_combined

Regulatory References:
    - ASHRAE Guideline 14-2014, Section 5.2.11 (Uncertainty)
    - IPMVP Core Concepts 2022, Chapter 7 (Assessing Uncertainty)
    - ISO 50015:2014, Clause 8.5 (Uncertainty Assessment)
    - FEMP M&V Guidelines 4.0, Appendix E (Uncertainty)
    - GUM (Guide to Expression of Uncertainty in Measurement)
    - JCGM 100:2008 (Evaluation of Measurement Data)

Zero-Hallucination:
    - All uncertainty computed via deterministic formulas
    - t-statistics from lookup table (no statistical library required)
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
Engine:  4 of 10
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class UncertaintyComponent(str, Enum):
    """Component of total uncertainty.

    MEASUREMENT: Instrument/metering uncertainty.
    MODEL:       Regression model uncertainty.
    SAMPLING:    Sampling uncertainty (Option A).
    COMBINED:    Combined uncertainty (RSS).
    """
    MEASUREMENT = "measurement"
    MODEL = "model"
    SAMPLING = "sampling"
    COMBINED = "combined"

class ConfidenceLevel(str, Enum):
    """Statistical confidence level for uncertainty bounds.

    CL_68:  68% confidence (1-sigma, standard ASHRAE 14).
    CL_80:  80% confidence.
    CL_90:  90% confidence (commonly reported).
    CL_95:  95% confidence (2-sigma, expanded uncertainty).
    CL_99:  99% confidence.
    """
    CL_68 = "68"
    CL_80 = "80"
    CL_90 = "90"
    CL_95 = "95"
    CL_99 = "99"

class MeterAccuracyClass(str, Enum):
    """Meter accuracy class per ANSI C12.20 / IEC 62053.

    CLASS_02:  +/- 0.2% (revenue-grade).
    CLASS_05:  +/- 0.5% (utility-grade).
    CLASS_10:  +/- 1.0% (building sub-meter).
    CLASS_15:  +/- 1.5% (portable meter).
    CLASS_20:  +/- 2.0% (check meter).
    CLASS_50:  +/- 5.0% (estimate-grade).
    """
    CLASS_02 = "0.2"
    CLASS_05 = "0.5"
    CLASS_10 = "1.0"
    CLASS_15 = "1.5"
    CLASS_20 = "2.0"
    CLASS_50 = "5.0"

class UncertaintyGrade(str, Enum):
    """Qualitative grade for overall uncertainty assessment.

    EXCELLENT: FSU < 25% at 68% confidence.
    GOOD:      FSU 25-50% at 68% confidence.
    MARGINAL:  FSU 50-75% at 68% confidence.
    POOR:      FSU > 75% at 68% confidence.
    INVALID:   Cannot compute uncertainty (insufficient data).
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"
    INVALID = "invalid"

class SamplingMethod(str, Enum):
    """Sampling method for Option A uncertainty.

    SIMPLE_RANDOM:   Simple random sampling.
    STRATIFIED:      Stratified random sampling.
    SYSTEMATIC:      Systematic sampling (every nth unit).
    CENSUS:          Full census (all units measured).
    """
    SIMPLE_RANDOM = "simple_random"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"
    CENSUS = "census"

class PropagationMethod(str, Enum):
    """Uncertainty propagation method.

    RSS:            Root-sum-square (independent errors).
    LINEAR:         Linear sum (conservative, correlated errors).
    MONTE_CARLO:    Monte Carlo simulation.
    """
    RSS = "rss"
    LINEAR = "linear"
    MONTE_CARLO = "monte_carlo"

# ---------------------------------------------------------------------------
# Constants -- t-distribution lookup table
# ---------------------------------------------------------------------------

# Two-tailed t-values for common confidence levels and degrees of freedom.
# Key: (degrees_of_freedom, confidence_level_pct) -> t-value.
# For large df (>120), use z-values.
T_TABLE: Dict[Tuple[int, int], Decimal] = {}

# Populate standard t-values for key degrees of freedom.
_T_VALUES_68: Dict[int, str] = {
    1: "1.841", 2: "1.320", 3: "1.189", 4: "1.134",
    5: "1.104", 6: "1.085", 7: "1.071", 8: "1.060",
    9: "1.052", 10: "1.046", 11: "1.041", 12: "1.037",
    15: "1.029", 20: "1.021", 25: "1.017", 30: "1.014",
    40: "1.011", 50: "1.009", 60: "1.007", 80: "1.006",
    100: "1.005", 120: "1.004", 200: "1.003", 500: "1.001",
}
_T_VALUES_90: Dict[int, str] = {
    1: "6.314", 2: "2.920", 3: "2.353", 4: "2.132",
    5: "2.015", 6: "1.943", 7: "1.895", 8: "1.860",
    9: "1.833", 10: "1.812", 11: "1.796", 12: "1.782",
    15: "1.753", 20: "1.725", 25: "1.708", 30: "1.697",
    40: "1.684", 50: "1.676", 60: "1.671", 80: "1.664",
    100: "1.660", 120: "1.658", 200: "1.653", 500: "1.648",
}
_T_VALUES_95: Dict[int, str] = {
    1: "12.706", 2: "4.303", 3: "3.182", 4: "2.776",
    5: "2.571", 6: "2.447", 7: "2.365", 8: "2.306",
    9: "2.262", 10: "2.228", 11: "2.201", 12: "2.179",
    15: "2.131", 20: "2.086", 25: "2.060", 30: "2.042",
    40: "2.021", 50: "2.009", 60: "2.000", 80: "1.990",
    100: "1.984", 120: "1.980", 200: "1.972", 500: "1.965",
}

for df, val in _T_VALUES_68.items():
    T_TABLE[(df, 68)] = Decimal(val)
for df, val in _T_VALUES_90.items():
    T_TABLE[(df, 90)] = Decimal(val)
for df, val in _T_VALUES_95.items():
    T_TABLE[(df, 95)] = Decimal(val)

# Z-values for large samples.
Z_VALUES: Dict[int, Decimal] = {
    68: Decimal("1.000"),
    80: Decimal("1.282"),
    90: Decimal("1.645"),
    95: Decimal("1.960"),
    99: Decimal("2.576"),
}

# FSU grade thresholds.
FSU_GRADE_THRESHOLDS: List[Tuple[Decimal, UncertaintyGrade]] = [
    (Decimal("25"), UncertaintyGrade.EXCELLENT),
    (Decimal("50"), UncertaintyGrade.GOOD),
    (Decimal("75"), UncertaintyGrade.MARGINAL),
    (Decimal("9999"), UncertaintyGrade.POOR),
]

# ASHRAE 14 maximum acceptable FSU at 68% confidence.
ASHRAE14_MAX_FSU_68: Decimal = Decimal("50")

# Meter accuracy class defaults (% of reading).
METER_ACCURACY: Dict[str, Decimal] = {
    MeterAccuracyClass.CLASS_02.value: Decimal("0.002"),
    MeterAccuracyClass.CLASS_05.value: Decimal("0.005"),
    MeterAccuracyClass.CLASS_10.value: Decimal("0.010"),
    MeterAccuracyClass.CLASS_15.value: Decimal("0.015"),
    MeterAccuracyClass.CLASS_20.value: Decimal("0.020"),
    MeterAccuracyClass.CLASS_50.value: Decimal("0.050"),
}

# CT/PT combined accuracy defaults.
CT_PT_ACCURACY: Dict[str, Decimal] = {
    "revenue": Decimal("0.003"),
    "utility": Decimal("0.006"),
    "building": Decimal("0.010"),
    "portable": Decimal("0.020"),
}

# Calibration drift per year (% per year).
CALIBRATION_DRIFT_PER_YEAR: Decimal = Decimal("0.001")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class MeasurementUncertaintyInput(BaseModel):
    """Input for measurement uncertainty calculation.

    Attributes:
        meter_accuracy_class: Meter accuracy class.
        meter_accuracy_pct: Override meter accuracy (fraction, e.g. 0.005).
        ct_pt_class: CT/PT accuracy class.
        ct_pt_accuracy_pct: Override CT/PT accuracy (fraction).
        calibration_date: Date of last calibration.
        analysis_date: Date of analysis (for drift calculation).
        calibration_drift_pct_per_year: Calibration drift rate.
        additional_uncertainties: Additional uncertainty sources.
        total_metered_energy: Total energy through this meter.
    """
    meter_accuracy_class: MeterAccuracyClass = Field(
        default=MeterAccuracyClass.CLASS_10,
        description="Meter accuracy class",
    )
    meter_accuracy_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=1, description="Override accuracy (fraction)"
    )
    ct_pt_class: str = Field(
        default="building", description="CT/PT accuracy class"
    )
    ct_pt_accuracy_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=1, description="Override CT/PT accuracy"
    )
    calibration_date: Optional[datetime] = Field(
        default=None, description="Last calibration date"
    )
    analysis_date: datetime = Field(
        default_factory=utcnow, description="Analysis date"
    )
    calibration_drift_pct_per_year: Decimal = Field(
        default=CALIBRATION_DRIFT_PER_YEAR,
        description="Calibration drift rate"
    )
    additional_uncertainties: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Additional uncertainty sources {name: fraction}"
    )
    total_metered_energy: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total metered energy"
    )

class ModelUncertaintyInput(BaseModel):
    """Input for regression model uncertainty calculation.

    Attributes:
        cvrmse_pct: CVRMSE of the regression model (%).
        n_observations: Number of baseline data points.
        n_parameters: Number of model parameters (including intercept).
        baseline_energy_total: Total baseline-period energy (E_baseline).
        actual_energy_total: Total reporting-period energy (E_actual).
        residual_std_error: Standard error of the regression (RMSE).
        ss_residual: Residual sum of squares.
        mean_y: Mean of response variable.
    """
    cvrmse_pct: Decimal = Field(
        default=Decimal("0"), ge=0, description="CVRMSE %"
    )
    n_observations: int = Field(
        default=12, ge=3, description="Baseline data points"
    )
    n_parameters: int = Field(
        default=2, ge=1, description="Model parameters"
    )
    baseline_energy_total: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total baseline energy"
    )
    actual_energy_total: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total actual energy"
    )
    residual_std_error: Decimal = Field(
        default=Decimal("0"), ge=0, description="RMSE"
    )
    ss_residual: Decimal = Field(
        default=Decimal("0"), ge=0, description="SS residual"
    )
    mean_y: Decimal = Field(
        default=Decimal("0"), description="Mean of Y"
    )

class SamplingUncertaintyInput(BaseModel):
    """Input for sampling uncertainty calculation (Option A).

    Attributes:
        population_size: Total population of units.
        sample_size: Number of units sampled.
        coefficient_of_variation: CV of sampled parameter (fraction).
        sample_mean: Mean of sampled values.
        sample_std_dev: Standard deviation of sampled values.
        desired_precision_pct: Desired precision level (%).
        desired_confidence_pct: Desired confidence level (%).
        sampling_method: Sampling method used.
    """
    population_size: int = Field(
        default=100, ge=1, description="Total population"
    )
    sample_size: int = Field(
        default=10, ge=1, description="Sample size"
    )
    coefficient_of_variation: Decimal = Field(
        default=Decimal("0.20"), ge=0,
        description="CV of sampled parameter"
    )
    sample_mean: Decimal = Field(
        default=Decimal("0"), description="Sample mean"
    )
    sample_std_dev: Decimal = Field(
        default=Decimal("0"), ge=0, description="Sample std dev"
    )
    desired_precision_pct: Decimal = Field(
        default=Decimal("10"), ge=0, description="Desired precision %"
    )
    desired_confidence_pct: Decimal = Field(
        default=Decimal("90"), ge=0, le=99,
        description="Desired confidence %"
    )
    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.SIMPLE_RANDOM,
        description="Sampling method"
    )

class UncertaintyConfig(BaseModel):
    """Configuration for uncertainty analysis.

    Attributes:
        project_id: M&V project identifier.
        ecm_id: ECM identifier.
        facility_id: Facility identifier.
        confidence_levels: Confidence levels to evaluate.
        propagation_method: Uncertainty propagation method.
        energy_unit: Energy measurement unit.
        ashrae14_check: Check against ASHRAE 14 FSU limit.
        include_measurement: Include measurement uncertainty.
        include_model: Include model uncertainty.
        include_sampling: Include sampling uncertainty.
    """
    project_id: str = Field(default="", description="Project ID")
    ecm_id: str = Field(default="", description="ECM ID")
    facility_id: str = Field(default="", description="Facility ID")
    confidence_levels: List[ConfidenceLevel] = Field(
        default_factory=lambda: [
            ConfidenceLevel.CL_68, ConfidenceLevel.CL_90
        ],
        description="Confidence levels to evaluate",
    )
    propagation_method: PropagationMethod = Field(
        default=PropagationMethod.RSS,
        description="Propagation method"
    )
    energy_unit: str = Field(default="kWh", description="Energy unit")
    ashrae14_check: bool = Field(
        default=True, description="Check ASHRAE 14 FSU limit"
    )
    include_measurement: bool = Field(
        default=True, description="Include measurement uncertainty"
    )
    include_model: bool = Field(
        default=True, description="Include model uncertainty"
    )
    include_sampling: bool = Field(
        default=False, description="Include sampling uncertainty"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class MeasurementUncertaintyResult(BaseModel):
    """Result of measurement uncertainty calculation.

    Attributes:
        meter_uncertainty_pct: Meter uncertainty (fraction).
        ct_pt_uncertainty_pct: CT/PT uncertainty (fraction).
        calibration_uncertainty_pct: Calibration drift (fraction).
        additional_uncertainties_pct: Additional sources (fraction).
        combined_measurement_pct: Combined measurement (fraction).
        absolute_uncertainty: Absolute uncertainty in energy units.
        years_since_calibration: Years since last calibration.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    meter_uncertainty_pct: Decimal = Field(default=Decimal("0"))
    ct_pt_uncertainty_pct: Decimal = Field(default=Decimal("0"))
    calibration_uncertainty_pct: Decimal = Field(default=Decimal("0"))
    additional_uncertainties_pct: Decimal = Field(default=Decimal("0"))
    combined_measurement_pct: Decimal = Field(default=Decimal("0"))
    absolute_uncertainty: Decimal = Field(default=Decimal("0"))
    years_since_calibration: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class ModelUncertaintyResult(BaseModel):
    """Result of model (regression) uncertainty calculation.

    Attributes:
        cvrmse_pct: CVRMSE of model.
        t_statistic: t-statistic at specified confidence.
        n_observations: Data points.
        n_parameters: Model parameters.
        degrees_of_freedom: Degrees of freedom.
        correction_factor: sqrt((n+2)/n) correction factor.
        model_uncertainty_absolute: Absolute model uncertainty.
        model_uncertainty_pct: Model uncertainty as % of savings.
        prediction_interval_half_width: Half-width of prediction interval.
        confidence_level: Confidence level used.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    cvrmse_pct: Decimal = Field(default=Decimal("0"))
    t_statistic: Decimal = Field(default=Decimal("0"))
    n_observations: int = Field(default=0)
    n_parameters: int = Field(default=0)
    degrees_of_freedom: int = Field(default=0)
    correction_factor: Decimal = Field(default=Decimal("0"))
    model_uncertainty_absolute: Decimal = Field(default=Decimal("0"))
    model_uncertainty_pct: Decimal = Field(default=Decimal("0"))
    prediction_interval_half_width: Decimal = Field(default=Decimal("0"))
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.CL_68)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class SamplingUncertaintyResult(BaseModel):
    """Result of sampling uncertainty calculation.

    Attributes:
        population_size: Total population.
        sample_size: Actual sample size.
        required_sample_size: Minimum required sample size.
        is_adequate: Whether sample is adequate.
        coefficient_of_variation: CV of sampled parameter.
        precision_pct: Achieved precision.
        confidence_pct: Confidence level.
        sampling_uncertainty_pct: Sampling uncertainty (fraction).
        finite_population_correction: FPC factor.
        sampling_method: Sampling method used.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    population_size: int = Field(default=0)
    sample_size: int = Field(default=0)
    required_sample_size: int = Field(default=0)
    is_adequate: bool = Field(default=False)
    coefficient_of_variation: Decimal = Field(default=Decimal("0"))
    precision_pct: Decimal = Field(default=Decimal("0"))
    confidence_pct: Decimal = Field(default=Decimal("0"))
    sampling_uncertainty_pct: Decimal = Field(default=Decimal("0"))
    finite_population_correction: Decimal = Field(default=Decimal("1"))
    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.SIMPLE_RANDOM
    )
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class FSUResult(BaseModel):
    """Fractional savings uncertainty result at a specific confidence level.

    Attributes:
        confidence_level: Confidence level.
        fsu_pct: Fractional savings uncertainty (%).
        combined_uncertainty_absolute: Absolute combined uncertainty.
        measurement_component: Measurement uncertainty component.
        model_component: Model uncertainty component.
        sampling_component: Sampling uncertainty component.
        minimum_detectable_savings: MDS in energy units.
        minimum_detectable_savings_pct: MDS as % of baseline.
        savings_are_significant: True if savings > MDS.
        ashrae14_pass: Whether FSU meets ASHRAE 14 limit.
        uncertainty_grade: Qualitative grade.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.CL_68)
    fsu_pct: Decimal = Field(default=Decimal("0"))
    combined_uncertainty_absolute: Decimal = Field(default=Decimal("0"))
    measurement_component: Decimal = Field(default=Decimal("0"))
    model_component: Decimal = Field(default=Decimal("0"))
    sampling_component: Decimal = Field(default=Decimal("0"))
    minimum_detectable_savings: Decimal = Field(default=Decimal("0"))
    minimum_detectable_savings_pct: Decimal = Field(default=Decimal("0"))
    savings_are_significant: bool = Field(default=False)
    ashrae14_pass: bool = Field(default=False)
    uncertainty_grade: UncertaintyGrade = Field(
        default=UncertaintyGrade.INVALID
    )
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class UncertaintyResult(BaseModel):
    """Complete uncertainty analysis result.

    Attributes:
        analysis_id: Unique analysis identifier.
        project_id: M&V project identifier.
        ecm_id: ECM identifier.
        facility_id: Facility identifier.
        energy_savings: Energy savings being evaluated.
        baseline_energy: Total baseline energy.
        actual_energy: Total actual energy.
        measurement_result: Measurement uncertainty results.
        model_result: Model uncertainty results.
        sampling_result: Sampling uncertainty results (if applicable).
        fsu_results: FSU results at each confidence level.
        primary_fsu: Primary FSU (68% confidence).
        propagation_method: Method used for combining uncertainty.
        energy_unit: Energy measurement unit.
        overall_grade: Overall uncertainty quality grade.
        ashrae14_compliant: Whether analysis meets ASHRAE 14.
        warnings: Warnings generated.
        recommendations: Recommendations.
        processing_time_ms: Processing duration.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    analysis_id: str = Field(default_factory=_new_uuid)
    project_id: str = Field(default="")
    ecm_id: str = Field(default="")
    facility_id: str = Field(default="")
    energy_savings: Decimal = Field(default=Decimal("0"))
    baseline_energy: Decimal = Field(default=Decimal("0"))
    actual_energy: Decimal = Field(default=Decimal("0"))
    measurement_result: Optional[MeasurementUncertaintyResult] = Field(
        default=None
    )
    model_result: Optional[ModelUncertaintyResult] = Field(default=None)
    sampling_result: Optional[SamplingUncertaintyResult] = Field(default=None)
    fsu_results: List[FSUResult] = Field(default_factory=list)
    primary_fsu: Optional[FSUResult] = Field(default=None)
    propagation_method: PropagationMethod = Field(default=PropagationMethod.RSS)
    energy_unit: str = Field(default="kWh")
    overall_grade: UncertaintyGrade = Field(default=UncertaintyGrade.INVALID)
    ashrae14_compliant: bool = Field(default=False)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class UncertaintyEngine:
    """ASHRAE 14 fractional savings uncertainty engine for M&V.

    Computes measurement, model, and sampling uncertainty components,
    combines them via root-sum-square, and calculates fractional savings
    uncertainty (FSU) at multiple confidence levels.  Determines minimum
    detectable savings and validates against ASHRAE 14 limits.

    Usage::

        engine = UncertaintyEngine()
        config = UncertaintyConfig(project_id="PRJ-001")
        meas_input = MeasurementUncertaintyInput(
            meter_accuracy_class=MeterAccuracyClass.CLASS_10,
            total_metered_energy=Decimal("120000"),
        )
        model_input = ModelUncertaintyInput(
            cvrmse_pct=Decimal("15"),
            n_observations=12,
            n_parameters=3,
            baseline_energy_total=Decimal("120000"),
            actual_energy_total=Decimal("108000"),
        )
        result = engine.calculate_uncertainty(
            config, meas_input, model_input
        )
        print(f"FSU at 68%: {result.primary_fsu.fsu_pct}%")
        print(f"MDS: {result.primary_fsu.minimum_detectable_savings}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise UncertaintyEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - default_confidence (int): default confidence level (68)
                - max_fsu_pct (float): max acceptable FSU
                - default_meter_class (str): default meter accuracy class
        """
        self.config = config or {}
        self._default_confidence = int(
            self.config.get("default_confidence", 68)
        )
        self._max_fsu = _decimal(
            self.config.get("max_fsu_pct", ASHRAE14_MAX_FSU_68)
        )
        logger.info(
            "UncertaintyEngine v%s initialised (default_CL=%d%%, "
            "max_FSU=%.0f%%)",
            self.engine_version, self._default_confidence,
            float(self._max_fsu),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate_uncertainty(
        self,
        unc_config: UncertaintyConfig,
        measurement_input: Optional[MeasurementUncertaintyInput] = None,
        model_input: Optional[ModelUncertaintyInput] = None,
        sampling_input: Optional[SamplingUncertaintyInput] = None,
    ) -> UncertaintyResult:
        """Calculate comprehensive uncertainty analysis.

        Computes each component, combines via RSS, and evaluates FSU
        at all configured confidence levels.

        Args:
            unc_config: Uncertainty analysis configuration.
            measurement_input: Measurement uncertainty inputs.
            model_input: Model uncertainty inputs.
            sampling_input: Sampling uncertainty inputs (Option A).

        Returns:
            UncertaintyResult with complete analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating uncertainty: project=%s, ECM=%s",
            unc_config.project_id, unc_config.ecm_id,
        )

        # Determine savings
        baseline_energy = Decimal("0")
        actual_energy = Decimal("0")
        if model_input:
            baseline_energy = model_input.baseline_energy_total
            actual_energy = model_input.actual_energy_total
        energy_savings = baseline_energy - actual_energy

        # Measurement uncertainty
        meas_result: Optional[MeasurementUncertaintyResult] = None
        if unc_config.include_measurement and measurement_input:
            meas_result = self.calculate_measurement_uncertainty(
                measurement_input
            )

        # Model uncertainty (calculated at each confidence level below)
        model_result: Optional[ModelUncertaintyResult] = None
        if unc_config.include_model and model_input:
            model_result = self.calculate_model_uncertainty(
                model_input, ConfidenceLevel.CL_68,
            )

        # Sampling uncertainty
        samp_result: Optional[SamplingUncertaintyResult] = None
        if unc_config.include_sampling and sampling_input:
            samp_result = self.calculate_sampling_uncertainty(sampling_input)

        # Calculate FSU at each confidence level
        fsu_results: List[FSUResult] = []
        for cl in unc_config.confidence_levels:
            fsu = self._calculate_fsu(
                cl, energy_savings, baseline_energy,
                meas_result, model_input, samp_result,
                unc_config,
            )
            fsu_results.append(fsu)

        # Primary FSU (68% or first in list)
        primary_fsu = next(
            (f for f in fsu_results if f.confidence_level == ConfidenceLevel.CL_68),
            fsu_results[0] if fsu_results else None,
        )

        # Overall grade
        overall_grade = UncertaintyGrade.INVALID
        if primary_fsu:
            overall_grade = primary_fsu.uncertainty_grade

        # ASHRAE 14 compliance
        ashrae14 = False
        if primary_fsu:
            ashrae14 = primary_fsu.ashrae14_pass

        # Warnings and recommendations
        warnings = self._generate_warnings(
            primary_fsu, meas_result, model_result, samp_result,
            energy_savings, baseline_energy,
        )
        recommendations = self._generate_recommendations(
            primary_fsu, meas_result, model_result, samp_result,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = UncertaintyResult(
            project_id=unc_config.project_id,
            ecm_id=unc_config.ecm_id,
            facility_id=unc_config.facility_id,
            energy_savings=_round_val(energy_savings, 2),
            baseline_energy=_round_val(baseline_energy, 2),
            actual_energy=_round_val(actual_energy, 2),
            measurement_result=meas_result,
            model_result=model_result,
            sampling_result=samp_result,
            fsu_results=fsu_results,
            primary_fsu=primary_fsu,
            propagation_method=unc_config.propagation_method,
            energy_unit=unc_config.energy_unit,
            overall_grade=overall_grade,
            ashrae14_compliant=ashrae14,
            warnings=warnings,
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Uncertainty analysis: FSU_68=%.1f%% (%s), MDS=%.1f, "
            "ASHRAE14=%s, grade=%s, hash=%s (%.1f ms)",
            float(primary_fsu.fsu_pct) if primary_fsu else 0.0,
            primary_fsu.confidence_level.value if primary_fsu else "n/a",
            float(primary_fsu.minimum_detectable_savings) if primary_fsu else 0.0,
            "PASS" if ashrae14 else "FAIL",
            overall_grade.value,
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def calculate_measurement_uncertainty(
        self,
        meas_input: MeasurementUncertaintyInput,
    ) -> MeasurementUncertaintyResult:
        """Calculate measurement (metering) uncertainty.

        Combines meter accuracy, CT/PT accuracy, calibration drift,
        and any additional sources via root-sum-square.

        Args:
            meas_input: Measurement uncertainty inputs.

        Returns:
            MeasurementUncertaintyResult with combined measurement uncertainty.
        """
        t0 = time.perf_counter()
        logger.info("Calculating measurement uncertainty")

        # Meter accuracy
        meter_unc = meas_input.meter_accuracy_pct
        if meter_unc is None:
            meter_unc = METER_ACCURACY.get(
                meas_input.meter_accuracy_class.value, Decimal("0.01")
            )

        # CT/PT accuracy
        ct_pt_unc = meas_input.ct_pt_accuracy_pct
        if ct_pt_unc is None:
            ct_pt_unc = CT_PT_ACCURACY.get(
                meas_input.ct_pt_class, Decimal("0.01")
            )

        # Calibration drift
        years_since_cal = Decimal("0")
        if meas_input.calibration_date:
            delta = meas_input.analysis_date - meas_input.calibration_date
            years_since_cal = _decimal(delta.days) / Decimal("365.25")
        cal_unc = years_since_cal * meas_input.calibration_drift_pct_per_year

        # Additional uncertainties
        additional_unc_sq = Decimal("0")
        for _name, unc in meas_input.additional_uncertainties.items():
            additional_unc_sq += unc ** 2
        additional_unc = _decimal(math.sqrt(max(0, float(additional_unc_sq))))

        # Root-sum-square
        combined_sq = (
            meter_unc ** 2
            + ct_pt_unc ** 2
            + cal_unc ** 2
            + additional_unc_sq
        )
        combined = _decimal(math.sqrt(max(0, float(combined_sq))))

        # Absolute uncertainty
        absolute = combined * meas_input.total_metered_energy

        result = MeasurementUncertaintyResult(
            meter_uncertainty_pct=_round_val(meter_unc, 6),
            ct_pt_uncertainty_pct=_round_val(ct_pt_unc, 6),
            calibration_uncertainty_pct=_round_val(cal_unc, 6),
            additional_uncertainties_pct=_round_val(additional_unc, 6),
            combined_measurement_pct=_round_val(combined, 6),
            absolute_uncertainty=_round_val(absolute, 2),
            years_since_calibration=_round_val(years_since_cal, 2),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Measurement uncertainty: %.3f%% (meter=%.3f%%, CT/PT=%.3f%%, "
            "cal=%.3f%%), absolute=%.1f, hash=%s (%.1f ms)",
            float(combined) * 100, float(meter_unc) * 100,
            float(ct_pt_unc) * 100, float(cal_unc) * 100,
            float(absolute), result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_model_uncertainty(
        self,
        model_input: ModelUncertaintyInput,
        confidence_level: ConfidenceLevel = ConfidenceLevel.CL_68,
    ) -> ModelUncertaintyResult:
        """Calculate regression model uncertainty.

        Uses the ASHRAE 14 formula:
        u_model = t * sqrt((n+2)/n) * (CVRMSE/100) * E_baseline

        Args:
            model_input: Model uncertainty inputs.
            confidence_level: Confidence level for t-statistic.

        Returns:
            ModelUncertaintyResult with model uncertainty.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating model uncertainty (CL=%s%%)", confidence_level.value
        )

        n = model_input.n_observations
        p = model_input.n_parameters
        dof = max(1, n - p)

        # Get t-statistic
        cl_int = int(confidence_level.value)
        t_stat = self._lookup_t_value(dof, cl_int)

        # Correction factor: sqrt((n+2)/n)
        correction = _decimal(math.sqrt((n + 2) / max(1, n)))

        # Model uncertainty absolute
        cvrmse_frac = model_input.cvrmse_pct / Decimal("100")
        u_model = t_stat * correction * cvrmse_frac * model_input.baseline_energy_total

        # As percentage of savings
        savings = model_input.baseline_energy_total - model_input.actual_energy_total
        u_model_pct = _safe_pct(u_model, abs(savings)) if savings != Decimal("0") else Decimal("0")

        # Prediction interval half-width (approximate)
        rmse = model_input.residual_std_error
        pred_hw = t_stat * rmse * _decimal(math.sqrt(1 + 1 / max(1, n)))

        result = ModelUncertaintyResult(
            cvrmse_pct=model_input.cvrmse_pct,
            t_statistic=_round_val(t_stat, 4),
            n_observations=n,
            n_parameters=p,
            degrees_of_freedom=dof,
            correction_factor=_round_val(correction, 6),
            model_uncertainty_absolute=_round_val(u_model, 2),
            model_uncertainty_pct=_round_val(u_model_pct, 4),
            prediction_interval_half_width=_round_val(pred_hw, 2),
            confidence_level=confidence_level,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Model uncertainty: u=%.1f (%.1f%% of savings), t=%.3f, "
            "CVRMSE=%.1f%%, n=%d, dof=%d, hash=%s (%.1f ms)",
            float(u_model), float(u_model_pct), float(t_stat),
            float(model_input.cvrmse_pct), n, dof,
            result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_sampling_uncertainty(
        self,
        samp_input: SamplingUncertaintyInput,
    ) -> SamplingUncertaintyResult:
        """Calculate sampling uncertainty for Option A.

        Computes required sample size and achieved precision using
        the t-distribution.

        Args:
            samp_input: Sampling uncertainty inputs.

        Returns:
            SamplingUncertaintyResult with sampling analysis.
        """
        t0 = time.perf_counter()
        logger.info("Calculating sampling uncertainty")

        n = samp_input.sample_size
        N = samp_input.population_size
        cv = samp_input.coefficient_of_variation
        cl_int = int(samp_input.desired_confidence_pct)
        precision = samp_input.desired_precision_pct / Decimal("100")

        # Get t-statistic
        dof = max(1, n - 1)
        t_val = self._lookup_t_value(dof, cl_int)

        # Required sample size: n = (t * CV / precision)^2
        if precision > Decimal("0"):
            n_required_raw = (t_val * cv / precision) ** 2
            n_required = int(float(n_required_raw)) + 1
        else:
            n_required = N

        # Finite population correction
        fpc = Decimal("1")
        if N > 0 and n < N:
            fpc_sq = _safe_divide(
                _decimal(N - n), _decimal(max(1, N - 1))
            )
            fpc = _decimal(math.sqrt(max(0, float(fpc_sq))))

        # Achieved precision
        if n > 0:
            achieved_precision = t_val * cv / _decimal(math.sqrt(n)) * fpc
        else:
            achieved_precision = Decimal("9999")

        sampling_unc = achieved_precision

        is_adequate = n >= n_required

        result = SamplingUncertaintyResult(
            population_size=N,
            sample_size=n,
            required_sample_size=min(n_required, N),
            is_adequate=is_adequate,
            coefficient_of_variation=_round_val(cv, 4),
            precision_pct=_round_val(achieved_precision * Decimal("100"), 4),
            confidence_pct=samp_input.desired_confidence_pct,
            sampling_uncertainty_pct=_round_val(sampling_unc, 6),
            finite_population_correction=_round_val(fpc, 6),
            sampling_method=samp_input.sampling_method,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Sampling uncertainty: precision=%.1f%%, required_n=%d, "
            "actual_n=%d, adequate=%s, hash=%s (%.1f ms)",
            float(achieved_precision * 100), n_required, n,
            is_adequate, result.provenance_hash[:16], elapsed,
        )
        return result

    def calculate_required_sample_size(
        self,
        coefficient_of_variation: Decimal,
        desired_precision_pct: Decimal,
        confidence_level: ConfidenceLevel = ConfidenceLevel.CL_90,
        population_size: Optional[int] = None,
    ) -> int:
        """Calculate required sample size for Option A.

        n_required = (t * CV / precision)^2, adjusted for finite population.

        Args:
            coefficient_of_variation: CV of parameter being sampled.
            desired_precision_pct: Desired precision (%).
            confidence_level: Confidence level.
            population_size: Total population (for FPC).

        Returns:
            Required sample size.
        """
        t0 = time.perf_counter()

        cl_int = int(confidence_level.value)
        precision = desired_precision_pct / Decimal("100")

        if precision <= Decimal("0"):
            return population_size or 100

        # Use z-value for initial estimate (large sample)
        z = Z_VALUES.get(cl_int, Decimal("1.645"))
        n_initial = (z * coefficient_of_variation / precision) ** 2
        n_req = int(float(n_initial)) + 1

        # Adjust for finite population
        if population_size and population_size > 0:
            n_adj = _safe_divide(
                _decimal(n_req),
                Decimal("1") + _safe_divide(
                    _decimal(n_req), _decimal(population_size)
                ),
            )
            n_req = int(float(n_adj)) + 1
            n_req = min(n_req, population_size)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Required sample size: n=%d (CV=%.2f, precision=%.1f%%, "
            "CL=%d%%) (%.1f ms)",
            n_req, float(coefficient_of_variation),
            float(desired_precision_pct), cl_int, elapsed,
        )
        return max(1, n_req)

    def calculate_minimum_detectable_savings(
        self,
        model_input: ModelUncertaintyInput,
        confidence_level: ConfidenceLevel = ConfidenceLevel.CL_68,
    ) -> Decimal:
        """Calculate minimum detectable savings (MDS).

        MDS = 2 * u_model (at the specified confidence level).

        Args:
            model_input: Model uncertainty inputs.
            confidence_level: Confidence level.

        Returns:
            Minimum detectable savings in energy units.
        """
        t0 = time.perf_counter()

        model_result = self.calculate_model_uncertainty(
            model_input, confidence_level
        )
        mds = Decimal("2") * model_result.model_uncertainty_absolute

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Minimum detectable savings: %.1f (model_u=%.1f, CL=%s%%) "
            "(%.1f ms)",
            float(mds), float(model_result.model_uncertainty_absolute),
            confidence_level.value, elapsed,
        )
        return _round_val(mds, 2)

    # ------------------------------------------------------------------ #
    # Private: FSU Calculation                                             #
    # ------------------------------------------------------------------ #

    def _calculate_fsu(
        self,
        confidence_level: ConfidenceLevel,
        energy_savings: Decimal,
        baseline_energy: Decimal,
        meas_result: Optional[MeasurementUncertaintyResult],
        model_input: Optional[ModelUncertaintyInput],
        samp_result: Optional[SamplingUncertaintyResult],
        unc_config: UncertaintyConfig,
    ) -> FSUResult:
        """Calculate FSU at a specific confidence level."""
        cl_int = int(confidence_level.value)

        # Measurement component (absolute)
        u_meas = Decimal("0")
        if meas_result:
            u_meas = meas_result.absolute_uncertainty

        # Model component at this confidence level
        u_model = Decimal("0")
        if model_input:
            n = model_input.n_observations
            p = model_input.n_parameters
            dof = max(1, n - p)
            t_stat = self._lookup_t_value(dof, cl_int)
            correction = _decimal(math.sqrt((n + 2) / max(1, n)))
            cvrmse_frac = model_input.cvrmse_pct / Decimal("100")
            u_model = t_stat * correction * cvrmse_frac * baseline_energy

        # Sampling component (absolute)
        u_samp = Decimal("0")
        if samp_result:
            u_samp = abs(energy_savings) * samp_result.sampling_uncertainty_pct

        # Combined via RSS or linear
        if unc_config.propagation_method == PropagationMethod.LINEAR:
            u_combined = u_meas + u_model + u_samp
        else:
            u_combined_sq = u_meas ** 2 + u_model ** 2 + u_samp ** 2
            u_combined = _decimal(math.sqrt(max(0, float(u_combined_sq))))

        # FSU
        fsu_pct = _safe_pct(u_combined, abs(energy_savings)) if energy_savings != Decimal("0") else Decimal("9999")

        # MDS = 2 * u_combined
        mds = Decimal("2") * u_combined
        mds_pct = _safe_pct(mds, baseline_energy)

        # Significance check
        savings_significant = abs(energy_savings) > mds

        # ASHRAE 14 check (FSU < 50% at 68%)
        ashrae14_pass = False
        if cl_int == 68:
            ashrae14_pass = fsu_pct <= self._max_fsu

        # Grade
        grade = self._grade_fsu(fsu_pct)

        result = FSUResult(
            confidence_level=confidence_level,
            fsu_pct=_round_val(fsu_pct, 4),
            combined_uncertainty_absolute=_round_val(u_combined, 2),
            measurement_component=_round_val(u_meas, 2),
            model_component=_round_val(u_model, 2),
            sampling_component=_round_val(u_samp, 2),
            minimum_detectable_savings=_round_val(mds, 2),
            minimum_detectable_savings_pct=_round_val(mds_pct, 4),
            savings_are_significant=savings_significant,
            ashrae14_pass=ashrae14_pass,
            uncertainty_grade=grade,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Private: t-distribution Lookup                                       #
    # ------------------------------------------------------------------ #

    def _lookup_t_value(self, dof: int, confidence_pct: int) -> Decimal:
        """Look up t-value from table, interpolating for unlisted dof."""
        # Direct lookup
        if (dof, confidence_pct) in T_TABLE:
            return T_TABLE[(dof, confidence_pct)]

        # Find bracketing dof values
        available_dofs = sorted(set(
            d for (d, c) in T_TABLE if c == confidence_pct
        ))

        if not available_dofs:
            return Z_VALUES.get(confidence_pct, Decimal("1.645"))

        # Clamp to available range
        if dof >= available_dofs[-1]:
            return T_TABLE.get(
                (available_dofs[-1], confidence_pct),
                Z_VALUES.get(confidence_pct, Decimal("1.645")),
            )
        if dof <= available_dofs[0]:
            return T_TABLE.get(
                (available_dofs[0], confidence_pct),
                Z_VALUES.get(confidence_pct, Decimal("6.314")),
            )

        # Linear interpolation between nearest values
        lower_dof = max(d for d in available_dofs if d <= dof)
        upper_dof = min(d for d in available_dofs if d >= dof)

        if lower_dof == upper_dof:
            return T_TABLE[(lower_dof, confidence_pct)]

        t_lower = T_TABLE[(lower_dof, confidence_pct)]
        t_upper = T_TABLE[(upper_dof, confidence_pct)]

        # Linear interpolation
        frac = _safe_divide(
            _decimal(dof - lower_dof),
            _decimal(upper_dof - lower_dof),
        )
        return t_lower + frac * (t_upper - t_lower)

    # ------------------------------------------------------------------ #
    # Private: Grading                                                     #
    # ------------------------------------------------------------------ #

    def _grade_fsu(self, fsu_pct: Decimal) -> UncertaintyGrade:
        """Grade FSU percentage."""
        if fsu_pct >= Decimal("9999"):
            return UncertaintyGrade.INVALID
        for threshold, grade in FSU_GRADE_THRESHOLDS:
            if fsu_pct <= threshold:
                return grade
        return UncertaintyGrade.POOR

    # ------------------------------------------------------------------ #
    # Private: Warnings & Recommendations                                  #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        primary_fsu: Optional[FSUResult],
        meas_result: Optional[MeasurementUncertaintyResult],
        model_result: Optional[ModelUncertaintyResult],
        samp_result: Optional[SamplingUncertaintyResult],
        energy_savings: Decimal,
        baseline_energy: Decimal,
    ) -> List[str]:
        """Generate warnings for the uncertainty analysis."""
        warnings: List[str] = []

        if primary_fsu and not primary_fsu.ashrae14_pass:
            warnings.append(
                f"FSU of {float(primary_fsu.fsu_pct):.1f}% at 68% confidence "
                f"exceeds ASHRAE 14 limit of {float(self._max_fsu):.0f}%. "
                "Savings may not be statistically significant."
            )

        if primary_fsu and not primary_fsu.savings_are_significant:
            warnings.append(
                "Energy savings do not exceed the minimum detectable savings "
                f"({float(primary_fsu.minimum_detectable_savings):.0f}). "
                "Savings cannot be confirmed as statistically significant."
            )

        if energy_savings <= Decimal("0"):
            warnings.append(
                "Energy savings are zero or negative. Uncertainty "
                "analysis is not meaningful without positive savings."
            )

        if model_result and model_result.degrees_of_freedom < 6:
            warnings.append(
                f"Degrees of freedom ({model_result.degrees_of_freedom}) is "
                "low, resulting in large t-statistics and wide uncertainty "
                "bounds. Consider using more baseline data points."
            )

        if meas_result and meas_result.years_since_calibration > Decimal("2"):
            warnings.append(
                f"Meter was last calibrated "
                f"{float(meas_result.years_since_calibration):.1f} years ago. "
                "Recommend recalibration per ANSI C12.20."
            )

        if samp_result and not samp_result.is_adequate:
            warnings.append(
                f"Sample size ({samp_result.sample_size}) is below "
                f"the required minimum ({samp_result.required_sample_size}). "
                "Increase sample size for Option A compliance."
            )

        return warnings

    def _generate_recommendations(
        self,
        primary_fsu: Optional[FSUResult],
        meas_result: Optional[MeasurementUncertaintyResult],
        model_result: Optional[ModelUncertaintyResult],
        samp_result: Optional[SamplingUncertaintyResult],
    ) -> List[str]:
        """Generate recommendations for reducing uncertainty."""
        recs: List[str] = []

        if primary_fsu and primary_fsu.fsu_pct > Decimal("50"):
            # Identify dominant component
            components = {
                "measurement": primary_fsu.measurement_component,
                "model": primary_fsu.model_component,
                "sampling": primary_fsu.sampling_component,
            }
            dominant = max(components, key=lambda k: float(components[k]))
            recs.append(
                f"The dominant uncertainty source is '{dominant}'. "
                "Focus on reducing this component to improve overall FSU."
            )

            if dominant == "model":
                recs.append(
                    "To reduce model uncertainty: add more baseline data "
                    "points, include additional independent variables, or "
                    "use a higher-granularity model (daily vs. monthly)."
                )
            elif dominant == "measurement":
                recs.append(
                    "To reduce measurement uncertainty: upgrade meter "
                    "accuracy class, recalibrate meters, or verify CT/PT "
                    "ratios and accuracy."
                )
            elif dominant == "sampling":
                recs.append(
                    "To reduce sampling uncertainty: increase sample size, "
                    "use stratified sampling, or convert to census measurement."
                )

        if primary_fsu and primary_fsu.uncertainty_grade == UncertaintyGrade.EXCELLENT:
            recs.append(
                "Uncertainty is within excellent range. Proceed with "
                "savings verification and reporting."
            )

        return recs
