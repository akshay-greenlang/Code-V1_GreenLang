# -*- coding: utf-8 -*-
"""
WeatherNormalisationEngine - PACK-035 Energy Benchmark Engine 4
================================================================

Weather-normalises energy consumption data using degree-day regression
models so that year-on-year performance can be compared independently
of weather variations.  Implements ASHRAE Guideline 14 change-point
models (2P, 3PC, 3PH, 4P, 5P), model validation statistics (R-squared,
CV-RMSE, NMBE, t-ratio), and Typical Meteorological Year (TMY)
normalisation.

Change-Point Regression Models:
    2-Parameter (2P):
        E = a + b * X
        Simple linear model (energy vs one variable).

    3-Parameter Heating (3PH):
        E = a + b * max(0, Tbase - T)
        Energy increases below a heating change point.

    3-Parameter Cooling (3PC):
        E = a + b * max(0, T - Tbase)
        Energy increases above a cooling change point.

    4-Parameter (4P):
        E = a + b_h * max(0, T_h - T) + b_c * max(0, T - T_c)
        Combined heating and cooling with a dead band.

    5-Parameter (5P):
        E = a + b_h * max(0, T_h - T) + b_c * max(0, T - T_c)
        with T_h < T_c (heating change point < cooling change point).
        Full dual-slope model with explicit dead band.

Model Validation (ASHRAE Guideline 14-2014):
    R-squared >= 0.75 (monthly data)
    CV(RMSE)  <= 20%  (monthly data; 15% for good models)
    NMBE      <= 5%   (monthly data)
    t-ratio   >= 2.0  for each slope coefficient

Degree-Day Calculation:
    HDD = sum( max(0, T_base - T_daily_mean) ) for each day in period
    CDD = sum( max(0, T_daily_mean - T_base) ) for each day in period
    Base temperatures from ASHRAE Fundamentals and CIBSE Guide A.

TMY Normalisation:
    Normalised_Energy = Model_Prediction(TMY_degree_days)
    where TMY (Typical Meteorological Year) represents long-term average
    weather conditions for the building location.

Regulatory References:
    - ASHRAE Guideline 14-2014: Measurement of Energy, Demand, and Water Savings
    - ASHRAE Fundamentals (2021): Chapter 14 -- Climatic Design Information
    - CIBSE Guide A (2015): Environmental Design
    - ISO 15927-6:2007: Calculation of degree-days
    - IPMVP: International Performance Measurement & Verification Protocol

Zero-Hallucination:
    - All regression uses closed-form or iterative optimisation (no LLM)
    - Change-point models use grid search + golden section optimisation
    - Validation statistics are deterministic formulae
    - SHA-256 provenance hashing on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-035 Energy Benchmark
Engine:  4 of 10
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

class RegressionModelType(str, Enum):
    """Change-point regression model types per ASHRAE Guideline 14.

    TWO_PARAMETER:              E = a + b*X (simple linear).
    THREE_PARAMETER_HEATING:    E = a + b*max(0, Tcp - T).
    THREE_PARAMETER_COOLING:    E = a + b*max(0, T - Tcp).
    FOUR_PARAMETER:             E = a + bh*max(0,Th-T) + bc*max(0,T-Tc), Th=Tc.
    FIVE_PARAMETER:             E = a + bh*max(0,Th-T) + bc*max(0,T-Tc), Th<Tc.
    """
    TWO_PARAMETER = "two_parameter"
    THREE_PARAMETER_HEATING = "three_parameter_heating"
    THREE_PARAMETER_COOLING = "three_parameter_cooling"
    FOUR_PARAMETER = "four_parameter"
    FIVE_PARAMETER = "five_parameter"

class NormalisationMethod(str, Enum):
    """Weather normalisation approach.

    SIMPLE_RATIO:  Energy * (TMY_DD / actual_DD).
    REGRESSION:    Model prediction using TMY degree-days.
    CHANGE_POINT:  Change-point model prediction with TMY temperatures.
    """
    SIMPLE_RATIO = "simple_ratio"
    REGRESSION = "regression"
    CHANGE_POINT = "change_point"

class ValidationStatus(str, Enum):
    """Model validation status per ASHRAE Guideline 14 criteria.

    VALID:    R2 >= 0.75, CV-RMSE <= 20%, NMBE <= 5%.
    MARGINAL: Partially meets thresholds.
    INVALID:  Fails key thresholds.
    """
    VALID = "valid"
    MARGINAL = "marginal"
    INVALID = "invalid"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Degree-day base temperatures by country (Celsius).
# Sources: ASHRAE Fundamentals Ch.14; CIBSE Guide A Table 2.21; EN ISO 15927-6.
DEGREE_DAY_BASE_TEMPS: Dict[str, Dict[str, float]] = {
    "DE": {"heating_base_c": 15.5, "cooling_base_c": 18.0,
           "source": "CIBSE Guide A 2015 / VDI 3807"},
    "FR": {"heating_base_c": 15.5, "cooling_base_c": 18.0,
           "source": "CIBSE Guide A 2015"},
    "UK": {"heating_base_c": 15.5, "cooling_base_c": 18.0,
           "source": "CIBSE Guide A 2015, Table 2.21"},
    "IT": {"heating_base_c": 15.0, "cooling_base_c": 21.0,
           "source": "CTI (Comitato Termotecnico Italiano)"},
    "ES": {"heating_base_c": 15.0, "cooling_base_c": 21.0,
           "source": "IDAE (Instituto Diversificacion Ahorro Energia)"},
    "NL": {"heating_base_c": 15.5, "cooling_base_c": 18.0,
           "source": "KNMI / CIBSE Guide A"},
    "BE": {"heating_base_c": 15.5, "cooling_base_c": 18.0,
           "source": "IRM Belgium"},
    "AT": {"heating_base_c": 15.5, "cooling_base_c": 18.0,
           "source": "OeNORM B 8110-5"},
    "PL": {"heating_base_c": 15.0, "cooling_base_c": 18.0,
           "source": "PN-EN ISO 15927-6"},
    "SE": {"heating_base_c": 17.0, "cooling_base_c": 18.0,
           "source": "SMHI Sweden"},
    "FI": {"heating_base_c": 17.0, "cooling_base_c": 18.0,
           "source": "Finnish Meteorological Institute"},
    "NO": {"heating_base_c": 17.0, "cooling_base_c": 18.0,
           "source": "MET Norway"},
    "DK": {"heating_base_c": 17.0, "cooling_base_c": 18.0,
           "source": "DMI Denmark"},
    "US": {"heating_base_c": 18.3, "cooling_base_c": 18.3,
           "source": "ASHRAE Fundamentals 2021, base 65F"},
    "CA": {"heating_base_c": 18.0, "cooling_base_c": 18.0,
           "source": "ASHRAE Fundamentals / NRCan"},
    "JP": {"heating_base_c": 14.0, "cooling_base_c": 22.0,
           "source": "SHASE (Society of Heating/Air-Conditioning Engineers Japan)"},
    "AU": {"heating_base_c": 15.0, "cooling_base_c": 22.0,
           "source": "Bureau of Meteorology / NABERS"},
    "DEFAULT": {"heating_base_c": 15.5, "cooling_base_c": 18.0,
                "source": "CIBSE Guide A default"},
}
"""Degree-day base temperatures by country from ASHRAE/CIBSE."""

# ASHRAE Guideline 14-2014 statistical thresholds for model validation.
ASHRAE_14_THRESHOLDS: Dict[str, float] = {
    "R_SQUARED_MIN": 0.75,
    "CV_RMSE_MAX": 0.20,
    "NMBE_MAX": 0.05,
    "T_RATIO_MIN": 2.0,
    "MIN_DATA_POINTS": 12,
}
"""ASHRAE Guideline 14 model validation thresholds."""

# Change-point search parameters.
_CP_SEARCH_MIN_TEMP: float = -5.0
_CP_SEARCH_MAX_TEMP: float = 35.0
_CP_SEARCH_STEP: float = 0.5
_CP_GOLDEN_RATIO: float = (math.sqrt(5.0) - 1.0) / 2.0

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class WeatherStation(BaseModel):
    """Weather station metadata for degree-day sourcing.

    Attributes:
        station_id: Weather station identifier.
        name: Station name.
        country_code: ISO country code.
        latitude: Latitude (decimal degrees).
        longitude: Longitude (decimal degrees).
        elevation_m: Elevation above sea level (metres).
    """
    station_id: str = Field(..., min_length=1, description="Station ID")
    name: str = Field(default="", description="Station name")
    country_code: str = Field(
        default="DEFAULT", min_length=2, max_length=3, description="Country code"
    )
    latitude: Optional[float] = Field(
        None, ge=-90, le=90, description="Latitude"
    )
    longitude: Optional[float] = Field(
        None, ge=-180, le=180, description="Longitude"
    )
    elevation_m: Optional[float] = Field(
        None, ge=-500, le=9000, description="Elevation (m)"
    )

class DegreeDayData(BaseModel):
    """Degree-day data for a single period.

    Attributes:
        period: Time period label matching energy data (e.g. '2024-01').
        avg_temp_c: Average outdoor temperature for the period (Celsius).
        hdd: Heating Degree Days for the period.
        cdd: Cooling Degree Days for the period.
        num_days: Number of days in the period.
    """
    period: str = Field(..., min_length=4, description="Time period label")
    avg_temp_c: float = Field(..., ge=-60, le=60, description="Avg temperature (C)")
    hdd: Optional[float] = Field(None, ge=0, description="Heating Degree Days")
    cdd: Optional[float] = Field(None, ge=0, description="Cooling Degree Days")
    num_days: int = Field(default=30, ge=1, le=366, description="Days in period")

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class RegressionModel(BaseModel):
    """Fitted regression model parameters.

    Attributes:
        model_type: Type of change-point model.
        intercept: Base-load coefficient (a).
        slope_heating: Heating slope (b_h).
        slope_cooling: Cooling slope (b_c).
        change_point_heating: Heating change-point temperature (Celsius).
        change_point_cooling: Cooling change-point temperature (Celsius).
        r_squared: Coefficient of determination.
        cv_rmse: Coefficient of Variation of RMSE (fraction, not %).
        nmbe: Normalised Mean Bias Error (fraction).
        t_ratio_heating: t-ratio for heating slope.
        t_ratio_cooling: t-ratio for cooling slope.
        n_observations: Number of data points used.
    """
    model_type: str = Field(..., description="Model type")
    intercept: float = Field(default=0.0, description="Base-load (a)")
    slope_heating: float = Field(default=0.0, description="Heating slope (b_h)")
    slope_cooling: float = Field(default=0.0, description="Cooling slope (b_c)")
    change_point_heating: Optional[float] = Field(
        None, description="Heating change-point (C)"
    )
    change_point_cooling: Optional[float] = Field(
        None, description="Cooling change-point (C)"
    )
    r_squared: float = Field(default=0.0, description="R-squared")
    cv_rmse: float = Field(default=0.0, description="CV(RMSE) fraction")
    nmbe: float = Field(default=0.0, description="NMBE fraction")
    t_ratio_heating: float = Field(default=0.0, description="t-ratio heating")
    t_ratio_cooling: float = Field(default=0.0, description="t-ratio cooling")
    n_observations: int = Field(default=0, description="Data points")

class ModelValidation(BaseModel):
    """Model validation result per ASHRAE Guideline 14.

    Attributes:
        status: Overall validation status.
        r_squared_pass: Whether R2 >= threshold.
        cv_rmse_pass: Whether CV-RMSE <= threshold.
        nmbe_pass: Whether |NMBE| <= threshold.
        t_ratio_pass: Whether all t-ratios >= threshold.
        data_sufficiency_pass: Whether n >= minimum.
        issues: List of validation issues found.
    """
    status: str = Field(default=ValidationStatus.INVALID.value)
    r_squared_pass: bool = Field(default=False)
    cv_rmse_pass: bool = Field(default=False)
    nmbe_pass: bool = Field(default=False)
    t_ratio_pass: bool = Field(default=False)
    data_sufficiency_pass: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list)

class NormalisationResult(BaseModel):
    """Complete weather normalisation result with full provenance.

    Contains the fitted model, validation, normalised consumption per
    period, and TMY-normalised annual total.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    facility_id: str = Field(default="", description="Facility identifier")
    normalisation_method: str = Field(default="", description="Method used")

    best_model: Optional[RegressionModel] = Field(
        None, description="Best-fit regression model"
    )
    all_models: List[RegressionModel] = Field(
        default_factory=list, description="All fitted models"
    )
    model_validation: Optional[ModelValidation] = Field(
        None, description="Validation of best model"
    )

    actual_annual_kwh: float = Field(default=0.0, description="Actual annual energy")
    normalised_annual_kwh: float = Field(
        default=0.0, description="Weather-normalised annual energy"
    )
    adjustment_kwh: float = Field(
        default=0.0, description="Total weather adjustment (kWh)"
    )
    adjustment_pct: float = Field(
        default=0.0, description="Weather adjustment (%)"
    )

    normalised_periods: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-period normalised data"
    )

    tmy_hdd: Optional[float] = Field(None, description="TMY annual HDD")
    tmy_cdd: Optional[float] = Field(None, description="TMY annual CDD")

    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------

class WeatherNormalisationEngine:
    """Weather normalisation engine using degree-day regression.

    Provides deterministic, zero-hallucination weather normalisation:
    - Degree-day calculation (HDD/CDD) from temperature data
    - Change-point model fitting (2P, 3PH, 3PC, 4P, 5P)
    - ASHRAE Guideline 14 model validation
    - TMY normalisation for year-on-year comparison
    - Automatic best-model selection

    All regression uses closed-form solutions or deterministic grid search
    with golden-section refinement.  No LLM involvement in calculations.

    Usage::

        engine = WeatherNormalisationEngine()
        result = engine.normalise_consumption(
            facility_id="bldg-001",
            energy_by_period={"2024-01": 50000, ...},
            weather_data=[DegreeDayData(...)],
            country_code="UK",
        )
        print(f"Normalised: {result.normalised_annual_kwh} kWh")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise with embedded reference data."""
        self._dd_bases = DEGREE_DAY_BASE_TEMPS
        self._thresholds = ASHRAE_14_THRESHOLDS

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def calculate_degree_days(
        self,
        avg_temps: List[float],
        num_days_per_period: List[int],
        country_code: str = "DEFAULT",
        heating_base: Optional[float] = None,
        cooling_base: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """Calculate HDD and CDD from average temperatures.

        HDD = num_days * max(0, base - avg_temp)
        CDD = num_days * max(0, avg_temp - base)

        Args:
            avg_temps: Average temperature per period (Celsius).
            num_days_per_period: Number of days in each period.
            country_code: Country code for default base temperatures.
            heating_base: Override heating base temperature.
            cooling_base: Override cooling base temperature.

        Returns:
            List of dicts with 'hdd', 'cdd', 'avg_temp' per period.
        """
        bases = self._dd_bases.get(
            country_code.upper(), self._dd_bases["DEFAULT"]
        )
        h_base = heating_base if heating_base is not None else bases["heating_base_c"]
        c_base = cooling_base if cooling_base is not None else bases["cooling_base_c"]

        results: List[Dict[str, float]] = []
        for temp, days in zip(avg_temps, num_days_per_period):
            hdd = days * max(0.0, h_base - temp)
            cdd = days * max(0.0, temp - c_base)
            results.append({
                "avg_temp_c": _round2(temp),
                "hdd": _round2(hdd),
                "cdd": _round2(cdd),
                "num_days": days,
            })

        return results

    def fit_regression(
        self,
        energy_values: List[float],
        temperatures: List[float],
        model_type: RegressionModelType = RegressionModelType.THREE_PARAMETER_HEATING,
    ) -> RegressionModel:
        """Fit a specific change-point regression model.

        Args:
            energy_values: Energy consumption per period (kWh).
            temperatures: Average temperature per period (Celsius).
            model_type: Which model to fit.

        Returns:
            RegressionModel with fitted parameters and statistics.

        Raises:
            ValueError: If insufficient data.
        """
        n = min(len(energy_values), len(temperatures))
        if n < 3:
            raise ValueError(f"At least 3 data points required, got {n}")

        e = energy_values[:n]
        t = temperatures[:n]

        if model_type == RegressionModelType.TWO_PARAMETER:
            return self._fit_2p(e, t)
        elif model_type == RegressionModelType.THREE_PARAMETER_HEATING:
            return self._fit_3ph(e, t)
        elif model_type == RegressionModelType.THREE_PARAMETER_COOLING:
            return self._fit_3pc(e, t)
        elif model_type == RegressionModelType.FOUR_PARAMETER:
            return self._fit_4p(e, t)
        elif model_type == RegressionModelType.FIVE_PARAMETER:
            return self._fit_5p(e, t)
        else:
            return self._fit_3ph(e, t)

    def normalise_consumption(
        self,
        facility_id: str,
        energy_by_period: Dict[str, float],
        weather_data: List[DegreeDayData],
        country_code: str = "DEFAULT",
        tmy_hdd: Optional[float] = None,
        tmy_cdd: Optional[float] = None,
    ) -> NormalisationResult:
        """Weather-normalise energy consumption using best-fit model.

        Automatically fits all five model types, selects the best per
        ASHRAE Guideline 14, and applies TMY normalisation.

        Args:
            facility_id: Facility identifier.
            energy_by_period: Period -> energy (kWh) mapping.
            weather_data: Temperature/degree-day data per period.
            country_code: Country code for base temperatures.
            tmy_hdd: TMY annual HDD (optional; estimated if not provided).
            tmy_cdd: TMY annual CDD (optional; estimated if not provided).

        Returns:
            NormalisationResult with normalised data and provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Weather normalisation: facility=%s, periods=%d",
            facility_id, len(energy_by_period),
        )

        # Step 1: Align energy and weather data
        aligned_e: List[float] = []
        aligned_t: List[float] = []
        aligned_periods: List[str] = []
        aligned_hdd: List[float] = []
        aligned_cdd: List[float] = []

        weather_map: Dict[str, DegreeDayData] = {w.period: w for w in weather_data}

        for period in sorted(energy_by_period.keys()):
            if period in weather_map:
                wd = weather_map[period]
                aligned_e.append(energy_by_period[period])
                aligned_t.append(wd.avg_temp_c)
                aligned_periods.append(period)
                # Calculate HDD/CDD if not provided
                if wd.hdd is not None:
                    aligned_hdd.append(wd.hdd)
                else:
                    bases = self._dd_bases.get(
                        country_code.upper(), self._dd_bases["DEFAULT"]
                    )
                    aligned_hdd.append(
                        wd.num_days * max(0.0, bases["heating_base_c"] - wd.avg_temp_c)
                    )
                if wd.cdd is not None:
                    aligned_cdd.append(wd.cdd)
                else:
                    bases = self._dd_bases.get(
                        country_code.upper(), self._dd_bases["DEFAULT"]
                    )
                    aligned_cdd.append(
                        wd.num_days * max(0.0, wd.avg_temp_c - bases["cooling_base_c"])
                    )

        if len(aligned_e) < 3:
            raise ValueError(
                f"At least 3 matched periods required, got {len(aligned_e)}"
            )

        # Step 2: Fit all model types
        all_models: List[RegressionModel] = []
        model_types = [
            RegressionModelType.TWO_PARAMETER,
            RegressionModelType.THREE_PARAMETER_HEATING,
            RegressionModelType.THREE_PARAMETER_COOLING,
            RegressionModelType.FOUR_PARAMETER,
            RegressionModelType.FIVE_PARAMETER,
        ]

        for mt in model_types:
            try:
                model = self.fit_regression(aligned_e, aligned_t, mt)
                all_models.append(model)
            except Exception as exc:
                logger.debug("Failed to fit %s: %s", mt.value, str(exc))

        # Step 3: Select best model
        best_model = self.select_best_model(all_models)

        # Step 4: Validate best model
        validation = None
        if best_model is not None:
            validation = self.validate_model(best_model)

        # Step 5: TMY normalisation
        actual_annual = sum(aligned_e)

        # Estimate TMY HDD/CDD from data average if not provided
        total_actual_hdd = sum(aligned_hdd)
        total_actual_cdd = sum(aligned_cdd)
        n_periods = len(aligned_e)

        if tmy_hdd is None:
            tmy_hdd = total_actual_hdd  # Use actual as proxy
        if tmy_cdd is None:
            tmy_cdd = total_actual_cdd

        # Normalised annual using regression model
        normalised_annual = actual_annual
        normalised_periods: List[Dict[str, Any]] = []

        if best_model is not None:
            normalised_annual = self._predict_annual(
                best_model, aligned_t, aligned_hdd, aligned_cdd,
                tmy_hdd, tmy_cdd, n_periods,
            )
            # Per-period normalised values
            for i, period in enumerate(aligned_periods):
                predicted_actual = self._predict_single(
                    best_model, aligned_t[i],
                )
                # TMY-adjusted prediction (scale by TMY/actual ratio)
                actual_dd = aligned_hdd[i] + aligned_cdd[i]
                avg_tmy_dd = (tmy_hdd + tmy_cdd) / max(n_periods, 1)
                if actual_dd > 0:
                    tmy_ratio = avg_tmy_dd / actual_dd
                else:
                    tmy_ratio = 1.0
                normalised_period = aligned_e[i] * tmy_ratio
                normalised_periods.append({
                    "period": period,
                    "actual_kwh": _round2(aligned_e[i]),
                    "predicted_kwh": _round2(predicted_actual),
                    "normalised_kwh": _round2(normalised_period),
                })

        adjustment = normalised_annual - actual_annual
        adjustment_pct = 0.0
        if actual_annual > 0:
            adjustment_pct = _round2(adjustment / actual_annual * 100.0)

        # Step 6: Recommendations
        recommendations = self._generate_recommendations(
            best_model, validation, adjustment_pct,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = NormalisationResult(
            facility_id=facility_id,
            normalisation_method=(
                best_model.model_type if best_model else NormalisationMethod.SIMPLE_RATIO.value
            ),
            best_model=best_model,
            all_models=all_models,
            model_validation=validation,
            actual_annual_kwh=_round2(actual_annual),
            normalised_annual_kwh=_round2(normalised_annual),
            adjustment_kwh=_round2(adjustment),
            adjustment_pct=adjustment_pct,
            normalised_periods=normalised_periods,
            tmy_hdd=_round2(tmy_hdd),
            tmy_cdd=_round2(tmy_cdd),
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Normalisation complete: facility=%s, model=%s, actual=%.0f, "
            "normalised=%.0f, adj=%.1f%%, hash=%s (%.1f ms)",
            facility_id,
            best_model.model_type if best_model else "none",
            actual_annual, normalised_annual, adjustment_pct,
            result.provenance_hash[:16], elapsed_ms,
        )
        return result

    def validate_model(self, model: RegressionModel) -> ModelValidation:
        """Validate a regression model per ASHRAE Guideline 14.

        Args:
            model: Fitted regression model.

        Returns:
            ModelValidation with pass/fail for each criterion.
        """
        issues: List[str] = []

        r2_pass = model.r_squared >= self._thresholds["R_SQUARED_MIN"]
        if not r2_pass:
            issues.append(
                f"R-squared {_round4(model.r_squared)} < "
                f"{self._thresholds['R_SQUARED_MIN']}"
            )

        cv_pass = model.cv_rmse <= self._thresholds["CV_RMSE_MAX"]
        if not cv_pass:
            issues.append(
                f"CV(RMSE) {_round4(model.cv_rmse)} > "
                f"{self._thresholds['CV_RMSE_MAX']}"
            )

        nmbe_pass = abs(model.nmbe) <= self._thresholds["NMBE_MAX"]
        if not nmbe_pass:
            issues.append(
                f"|NMBE| {_round4(abs(model.nmbe))} > "
                f"{self._thresholds['NMBE_MAX']}"
            )

        t_min = self._thresholds["T_RATIO_MIN"]
        t_pass = True
        if model.slope_heating != 0.0 and model.t_ratio_heating < t_min:
            t_pass = False
            issues.append(
                f"Heating t-ratio {_round2(model.t_ratio_heating)} < {t_min}"
            )
        if model.slope_cooling != 0.0 and model.t_ratio_cooling < t_min:
            t_pass = False
            issues.append(
                f"Cooling t-ratio {_round2(model.t_ratio_cooling)} < {t_min}"
            )

        data_pass = model.n_observations >= self._thresholds["MIN_DATA_POINTS"]
        if not data_pass:
            issues.append(
                f"Only {model.n_observations} data points; minimum "
                f"{int(self._thresholds['MIN_DATA_POINTS'])} required"
            )

        if r2_pass and cv_pass and nmbe_pass and t_pass and data_pass:
            status = ValidationStatus.VALID.value
        elif (r2_pass or cv_pass) and data_pass:
            status = ValidationStatus.MARGINAL.value
        else:
            status = ValidationStatus.INVALID.value

        return ModelValidation(
            status=status,
            r_squared_pass=r2_pass,
            cv_rmse_pass=cv_pass,
            nmbe_pass=nmbe_pass,
            t_ratio_pass=t_pass,
            data_sufficiency_pass=data_pass,
            issues=issues,
        )

    def select_best_model(
        self, models: List[RegressionModel],
    ) -> Optional[RegressionModel]:
        """Select the best model from a list based on R2 and CV-RMSE.

        Priority: highest R2 among models meeting CV-RMSE threshold.
        Fallback: highest R2 regardless of CV-RMSE.

        Args:
            models: List of fitted regression models.

        Returns:
            Best model, or None if list is empty.
        """
        if not models:
            return None

        # First pass: models meeting ASHRAE thresholds
        valid = [
            m for m in models
            if m.r_squared >= self._thresholds["R_SQUARED_MIN"]
            and m.cv_rmse <= self._thresholds["CV_RMSE_MAX"]
        ]

        if valid:
            return max(valid, key=lambda m: m.r_squared)

        # Fallback: best R2
        return max(models, key=lambda m: m.r_squared)

    # -------------------------------------------------------------------
    # Internal: Model Fitting
    # -------------------------------------------------------------------

    def _fit_2p(
        self, energy: List[float], temps: List[float],
    ) -> RegressionModel:
        """Fit 2-parameter model: E = a + b*T.

        Args:
            energy: Energy values per period.
            temps: Average temperatures per period.

        Returns:
            RegressionModel with fitted parameters.
        """
        n = len(energy)
        sum_t = sum(temps)
        sum_e = sum(energy)
        sum_te = sum(t * e for t, e in zip(temps, energy))
        sum_t2 = sum(t * t for t in temps)

        t_mean = sum_t / n
        e_mean = sum_e / n

        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-12:
            return RegressionModel(
                model_type=RegressionModelType.TWO_PARAMETER.value,
                intercept=e_mean, n_observations=n,
            )

        b = (n * sum_te - sum_t * sum_e) / denom
        a = e_mean - b * t_mean

        predicted = [a + b * t for t in temps]
        stats = self._calc_stats(energy, predicted, n, 2)

        # t-ratio for slope
        se_t2 = sum((t - t_mean) ** 2 for t in temps)
        resid_std = stats["resid_std"]
        t_ratio = 0.0
        if se_t2 > 0 and resid_std > 0:
            se_b = resid_std / math.sqrt(se_t2)
            t_ratio = abs(b / se_b) if se_b > 0 else 0.0

        return RegressionModel(
            model_type=RegressionModelType.TWO_PARAMETER.value,
            intercept=_round2(a),
            slope_heating=_round4(b) if b < 0 else 0.0,
            slope_cooling=_round4(b) if b > 0 else 0.0,
            r_squared=_round4(stats["r_squared"]),
            cv_rmse=_round4(stats["cv_rmse"]),
            nmbe=_round4(stats["nmbe"]),
            t_ratio_heating=_round2(t_ratio) if b < 0 else 0.0,
            t_ratio_cooling=_round2(t_ratio) if b > 0 else 0.0,
            n_observations=n,
        )

    def _fit_3ph(
        self, energy: List[float], temps: List[float],
    ) -> RegressionModel:
        """Fit 3-parameter heating model: E = a + b*max(0, Tcp - T).

        Searches for optimal change-point via grid search.

        Args:
            energy: Energy values per period.
            temps: Average temperatures per period.

        Returns:
            RegressionModel with fitted parameters.
        """
        n = len(energy)
        best_r2 = -1.0
        best_cp = 15.0
        best_a = 0.0
        best_b = 0.0

        cp = _CP_SEARCH_MIN_TEMP
        while cp <= _CP_SEARCH_MAX_TEMP:
            x = [max(0.0, cp - t) for t in temps]
            a, b, r2 = self._ols_simple(x, energy)
            if b >= 0 and r2 > best_r2:
                best_r2 = r2
                best_cp = cp
                best_a = a
                best_b = b
            cp += _CP_SEARCH_STEP

        # Golden-section refinement around best_cp
        lo = best_cp - _CP_SEARCH_STEP
        hi = best_cp + _CP_SEARCH_STEP
        for _ in range(20):
            c1 = hi - _CP_GOLDEN_RATIO * (hi - lo)
            c2 = lo + _CP_GOLDEN_RATIO * (hi - lo)
            x1 = [max(0.0, c1 - t) for t in temps]
            x2 = [max(0.0, c2 - t) for t in temps]
            _, _, r2_1 = self._ols_simple(x1, energy)
            _, _, r2_2 = self._ols_simple(x2, energy)
            if r2_1 > r2_2:
                hi = c2
            else:
                lo = c1
            if abs(hi - lo) < 0.05:
                break

        final_cp = (lo + hi) / 2.0
        x_final = [max(0.0, final_cp - t) for t in temps]
        final_a, final_b, final_r2 = self._ols_simple(x_final, energy)
        if final_r2 > best_r2 and final_b >= 0:
            best_cp, best_a, best_b, best_r2 = final_cp, final_a, final_b, final_r2

        predicted = [best_a + best_b * max(0.0, best_cp - t) for t in temps]
        stats = self._calc_stats(energy, predicted, n, 3)
        t_ratio = self._calc_t_ratio(temps, energy, predicted, best_b, n, 3)

        return RegressionModel(
            model_type=RegressionModelType.THREE_PARAMETER_HEATING.value,
            intercept=_round2(best_a),
            slope_heating=_round4(best_b),
            change_point_heating=_round2(best_cp),
            r_squared=_round4(stats["r_squared"]),
            cv_rmse=_round4(stats["cv_rmse"]),
            nmbe=_round4(stats["nmbe"]),
            t_ratio_heating=_round2(t_ratio),
            n_observations=n,
        )

    def _fit_3pc(
        self, energy: List[float], temps: List[float],
    ) -> RegressionModel:
        """Fit 3-parameter cooling model: E = a + b*max(0, T - Tcp).

        Args:
            energy: Energy values per period.
            temps: Average temperatures per period.

        Returns:
            RegressionModel with fitted parameters.
        """
        n = len(energy)
        best_r2 = -1.0
        best_cp = 18.0
        best_a = 0.0
        best_b = 0.0

        cp = _CP_SEARCH_MIN_TEMP
        while cp <= _CP_SEARCH_MAX_TEMP:
            x = [max(0.0, t - cp) for t in temps]
            a, b, r2 = self._ols_simple(x, energy)
            if b >= 0 and r2 > best_r2:
                best_r2 = r2
                best_cp = cp
                best_a = a
                best_b = b
            cp += _CP_SEARCH_STEP

        predicted = [best_a + best_b * max(0.0, t - best_cp) for t in temps]
        stats = self._calc_stats(energy, predicted, n, 3)
        t_ratio = self._calc_t_ratio(temps, energy, predicted, best_b, n, 3)

        return RegressionModel(
            model_type=RegressionModelType.THREE_PARAMETER_COOLING.value,
            intercept=_round2(best_a),
            slope_cooling=_round4(best_b),
            change_point_cooling=_round2(best_cp),
            r_squared=_round4(stats["r_squared"]),
            cv_rmse=_round4(stats["cv_rmse"]),
            nmbe=_round4(stats["nmbe"]),
            t_ratio_cooling=_round2(t_ratio),
            n_observations=n,
        )

    def _fit_4p(
        self, energy: List[float], temps: List[float],
    ) -> RegressionModel:
        """Fit 4-parameter model: E = a + bh*max(0,Tcp-T) + bc*max(0,T-Tcp).

        Single change-point for both heating and cooling (no dead band).

        Args:
            energy: Energy values per period.
            temps: Average temperatures per period.

        Returns:
            RegressionModel with fitted parameters.
        """
        n = len(energy)
        best_r2 = -1.0
        best_cp = 15.0
        best_a = 0.0
        best_bh = 0.0
        best_bc = 0.0

        cp = _CP_SEARCH_MIN_TEMP
        while cp <= _CP_SEARCH_MAX_TEMP:
            x_h = [max(0.0, cp - t) for t in temps]
            x_c = [max(0.0, t - cp) for t in temps]
            a, bh, bc, r2 = self._ols_two_var(x_h, x_c, energy)
            if bh >= 0 and bc >= 0 and r2 > best_r2:
                best_r2 = r2
                best_cp = cp
                best_a = a
                best_bh = bh
                best_bc = bc
            cp += _CP_SEARCH_STEP

        predicted = [
            best_a + best_bh * max(0.0, best_cp - t) + best_bc * max(0.0, t - best_cp)
            for t in temps
        ]
        stats = self._calc_stats(energy, predicted, n, 4)

        return RegressionModel(
            model_type=RegressionModelType.FOUR_PARAMETER.value,
            intercept=_round2(best_a),
            slope_heating=_round4(best_bh),
            slope_cooling=_round4(best_bc),
            change_point_heating=_round2(best_cp),
            change_point_cooling=_round2(best_cp),
            r_squared=_round4(stats["r_squared"]),
            cv_rmse=_round4(stats["cv_rmse"]),
            nmbe=_round4(stats["nmbe"]),
            n_observations=n,
        )

    def _fit_5p(
        self, energy: List[float], temps: List[float],
    ) -> RegressionModel:
        """Fit 5-parameter model with separate heating/cooling change-points.

        E = a + bh*max(0, Th-T) + bc*max(0, T-Tc) where Th < Tc.

        Args:
            energy: Energy values per period.
            temps: Average temperatures per period.

        Returns:
            RegressionModel with fitted parameters.
        """
        n = len(energy)
        best_r2 = -1.0
        best_th = 12.0
        best_tc = 18.0
        best_a = 0.0
        best_bh = 0.0
        best_bc = 0.0

        th = _CP_SEARCH_MIN_TEMP
        while th <= _CP_SEARCH_MAX_TEMP - 2.0:
            tc = th + 2.0
            while tc <= _CP_SEARCH_MAX_TEMP:
                x_h = [max(0.0, th - t) for t in temps]
                x_c = [max(0.0, t - tc) for t in temps]
                a, bh, bc, r2 = self._ols_two_var(x_h, x_c, energy)
                if bh >= 0 and bc >= 0 and r2 > best_r2:
                    best_r2 = r2
                    best_th = th
                    best_tc = tc
                    best_a = a
                    best_bh = bh
                    best_bc = bc
                tc += 1.0
            th += 1.0

        predicted = [
            best_a + best_bh * max(0.0, best_th - t) + best_bc * max(0.0, t - best_tc)
            for t in temps
        ]
        stats = self._calc_stats(energy, predicted, n, 5)

        return RegressionModel(
            model_type=RegressionModelType.FIVE_PARAMETER.value,
            intercept=_round2(best_a),
            slope_heating=_round4(best_bh),
            slope_cooling=_round4(best_bc),
            change_point_heating=_round2(best_th),
            change_point_cooling=_round2(best_tc),
            r_squared=_round4(stats["r_squared"]),
            cv_rmse=_round4(stats["cv_rmse"]),
            nmbe=_round4(stats["nmbe"]),
            n_observations=n,
        )

    # -------------------------------------------------------------------
    # Internal: OLS Helpers
    # -------------------------------------------------------------------

    def _ols_simple(
        self, x: List[float], y: List[float],
    ) -> Tuple[float, float, float]:
        """Simple OLS: y = a + b*x. Returns (a, b, r_squared)."""
        n = len(x)
        if n < 2:
            return (sum(y) / max(n, 1), 0.0, 0.0)
        sx = sum(x)
        sy = sum(y)
        sxy = sum(xi * yi for xi, yi in zip(x, y))
        sx2 = sum(xi * xi for xi in x)
        xm = sx / n
        ym = sy / n
        denom = n * sx2 - sx * sx
        if abs(denom) < 1e-12:
            return (ym, 0.0, 0.0)
        b = (n * sxy - sx * sy) / denom
        a = ym - b * xm
        pred = [a + b * xi for xi in x]
        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, pred))
        ss_tot = sum((yi - ym) ** 2 for yi in y)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return (a, b, max(0.0, r2))

    def _ols_two_var(
        self, x1: List[float], x2: List[float], y: List[float],
    ) -> Tuple[float, float, float, float]:
        """OLS with two variables: y = a + b1*x1 + b2*x2. Returns (a, b1, b2, r2)."""
        n = len(y)
        if n < 4:
            return (sum(y) / max(n, 1), 0.0, 0.0, 0.0)
        sy = sum(y)
        s1 = sum(x1)
        s2 = sum(x2)
        s1y = sum(a * b for a, b in zip(x1, y))
        s2y = sum(a * b for a, b in zip(x2, y))
        s11 = sum(a * a for a in x1)
        s22 = sum(a * a for a in x2)
        s12 = sum(a * b for a, b in zip(x1, x2))
        det = (
            n * (s11 * s22 - s12 * s12)
            - s1 * (s1 * s22 - s12 * s2)
            + s2 * (s1 * s12 - s11 * s2)
        )
        if abs(det) < 1e-12:
            return (sy / n, 0.0, 0.0, 0.0)
        a = (
            sy * (s11 * s22 - s12 * s12)
            - s1 * (s1y * s22 - s12 * s2y)
            + s2 * (s1y * s12 - s11 * s2y)
        ) / det
        b1 = (
            n * (s1y * s22 - s12 * s2y)
            - sy * (s1 * s22 - s12 * s2)
            + s2 * (s1 * s2y - s1y * s2)
        ) / det
        b2 = (
            n * (s11 * s2y - s1y * s12)
            - s1 * (s1 * s2y - s1y * s2)
            + sy * (s1 * s12 - s11 * s2)
        ) / det
        ym = sy / n
        pred = [a + b1 * v1 + b2 * v2 for v1, v2 in zip(x1, x2)]
        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, pred))
        ss_tot = sum((yi - ym) ** 2 for yi in y)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return (a, b1, b2, max(0.0, r2))

    def _calc_stats(
        self, actual: List[float], predicted: List[float],
        n: int, p: int,
    ) -> Dict[str, float]:
        """Calculate R2, CV-RMSE, NMBE, residual std."""
        if n == 0:
            return {"r_squared": 0.0, "cv_rmse": 0.0, "nmbe": 0.0, "resid_std": 0.0}
        y_mean = sum(actual) / n
        ss_res = sum((a - p_) ** 2 for a, p_ in zip(actual, predicted))
        ss_tot = sum((a - y_mean) ** 2 for a in actual)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = math.sqrt(ss_res / max(n - p, 1))
        cv_rmse = rmse / y_mean if y_mean != 0 else 0.0
        bias = sum(a - p_ for a, p_ in zip(actual, predicted))
        nmbe = bias / (n * y_mean) if y_mean != 0 and n > 0 else 0.0
        resid_std = math.sqrt(ss_res / max(n - p, 1))
        return {
            "r_squared": max(0.0, r2),
            "cv_rmse": abs(cv_rmse),
            "nmbe": nmbe,
            "resid_std": resid_std,
        }

    def _calc_t_ratio(
        self, temps: List[float], actual: List[float],
        predicted: List[float], slope: float, n: int, p: int,
    ) -> float:
        """Calculate t-ratio for a slope coefficient."""
        if n <= p or slope == 0:
            return 0.0
        ss_res = sum((a - pr) ** 2 for a, pr in zip(actual, predicted))
        resid_std = math.sqrt(ss_res / max(n - p, 1))
        t_mean = sum(temps) / n
        se_x2 = sum((t - t_mean) ** 2 for t in temps)
        if se_x2 > 0 and resid_std > 0:
            se_b = resid_std / math.sqrt(se_x2)
            return abs(slope / se_b) if se_b > 0 else 0.0
        return 0.0

    # -------------------------------------------------------------------
    # Internal: Prediction
    # -------------------------------------------------------------------

    def _predict_single(
        self, model: RegressionModel, temp: float,
    ) -> float:
        """Predict energy for a single period using model.

        Args:
            model: Fitted regression model.
            temp: Average temperature for the period.

        Returns:
            Predicted energy (kWh).
        """
        e = model.intercept
        if model.change_point_heating is not None and model.slope_heating > 0:
            e += model.slope_heating * max(0.0, model.change_point_heating - temp)
        if model.change_point_cooling is not None and model.slope_cooling > 0:
            e += model.slope_cooling * max(0.0, temp - model.change_point_cooling)
        return max(0.0, e)

    def _predict_annual(
        self, model: RegressionModel,
        actual_temps: List[float],
        actual_hdd: List[float],
        actual_cdd: List[float],
        tmy_hdd: float,
        tmy_cdd: float,
        n_periods: int,
    ) -> float:
        """Predict TMY-normalised annual energy.

        Uses the ratio method: sum actual predictions, then scale by
        TMY/actual degree-day ratio.

        Args:
            model: Fitted model.
            actual_temps: Period temperatures.
            actual_hdd: Period HDD values.
            actual_cdd: Period CDD values.
            tmy_hdd: TMY annual HDD.
            tmy_cdd: TMY annual CDD.
            n_periods: Number of periods.

        Returns:
            TMY-normalised annual energy (kWh).
        """
        # Sum actual predictions
        total_predicted = sum(
            self._predict_single(model, t) for t in actual_temps
        )

        if total_predicted <= 0:
            return 0.0

        # Degree-day ratio scaling
        total_actual_dd = sum(actual_hdd) + sum(actual_cdd)
        tmy_total_dd = tmy_hdd + tmy_cdd

        if total_actual_dd > 0:
            dd_ratio = tmy_total_dd / total_actual_dd
        else:
            dd_ratio = 1.0

        # Weighted normalisation: base-load stays, weather-sensitive scales
        base_load = model.intercept * n_periods
        weather_load = total_predicted - base_load
        normalised = base_load + weather_load * dd_ratio

        return max(0.0, normalised)

    # -------------------------------------------------------------------
    # Internal: Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        model: Optional[RegressionModel],
        validation: Optional[ModelValidation],
        adjustment_pct: float,
    ) -> List[str]:
        """Generate recommendations based on normalisation results.

        Args:
            model: Best-fit model.
            validation: Model validation.
            adjustment_pct: Weather adjustment percentage.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if validation and validation.status == ValidationStatus.INVALID.value:
            recs.append(
                "Weather normalisation model does not meet ASHRAE Guideline 14 "
                "thresholds. Results should be treated as indicative only. "
                "Consider collecting higher-frequency data (weekly or daily) "
                "for a more robust model."
            )

        if model is None:
            recs.append(
                "No valid regression model could be fitted. Ensure at least "
                "12 months of matched energy and weather data are available."
            )
            return recs

        if abs(adjustment_pct) > 10.0:
            recs.append(
                f"Weather adjustment of {adjustment_pct}% indicates the "
                f"reporting year had significantly {'milder' if adjustment_pct > 0 else 'harsher'} "
                f"weather than the TMY reference. Weather-normalised values "
                f"should be used for year-on-year comparison."
            )

        if model.slope_heating > 0 and model.change_point_heating is not None:
            recs.append(
                f"Heating change-point at {_round2(model.change_point_heating)} C. "
                f"Improving building envelope insulation or heat recovery could "
                f"reduce the heating slope coefficient."
            )

        if model.slope_cooling > 0 and model.change_point_cooling is not None:
            recs.append(
                f"Cooling change-point at {_round2(model.change_point_cooling)} C. "
                f"Consider increasing cooling setpoints, improving shading, "
                f"or upgrading to higher-efficiency cooling equipment."
            )

        return recs
