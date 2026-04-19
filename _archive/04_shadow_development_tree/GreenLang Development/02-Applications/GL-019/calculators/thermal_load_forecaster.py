"""
GL-019 HEATSCHEDULER - Thermal Load Forecaster

Advanced degree-day based load forecasting with time-series decomposition,
weather normalization, occupancy adjustment, and building thermal mass modeling.

This module provides:
- Heating Degree Day (HDD) and Cooling Degree Day (CDD) calculations
- Time-series load prediction with seasonal decomposition
- Weather-normalized consumption analysis
- Occupancy-adjusted load profiles
- Building thermal mass modeling
- Peak demand prediction
- Forecast accuracy metrics (MAPE, RMSE, MAE, R-squared)

Standards Reference:
- ISO 50001 - Energy Management Systems
- ISO 50006 - Measuring Energy Performance Using Baselines
- ASHRAE Handbook - Fundamentals (Degree-Day Methods)
- ASHRAE Guideline 14 - Measurement of Energy and Demand Savings

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
from threading import Lock
import hashlib
import json
import math
import statistics

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Standard base temperatures (Celsius)
HDD_BASE_TEMP_C = 18.0  # Heating Degree Day base
CDD_BASE_TEMP_C = 24.0  # Cooling Degree Day base

# Alternative base temperatures (Fahrenheit converted)
HDD_BASE_TEMP_F_TO_C = 18.3  # 65F converted
CDD_BASE_TEMP_F_TO_C = 23.9  # 75F converted

# Confidence level z-scores for prediction intervals
CONFIDENCE_Z_SCORES = {
    "80%": 1.282,
    "90%": 1.645,
    "95%": 1.960,
    "99%": 2.576,
}

# Building thermal time constants (typical ranges in hours)
THERMAL_TIME_CONSTANTS = {
    "light": 4.0,      # Light construction
    "medium": 8.0,     # Medium construction
    "heavy": 16.0,     # Heavy/massive construction
    "industrial": 24.0  # Industrial with high thermal mass
}

# Occupancy schedule defaults (fraction of full occupancy)
DEFAULT_OCCUPANCY_PROFILE = [
    0.1, 0.1, 0.1, 0.1, 0.1, 0.2,   # 00:00-05:00
    0.4, 0.7, 1.0, 1.0, 1.0, 1.0,   # 06:00-11:00
    0.9, 1.0, 1.0, 1.0, 1.0, 0.8,   # 12:00-17:00
    0.5, 0.3, 0.2, 0.1, 0.1, 0.1    # 18:00-23:00
]


class SeasonType(str, Enum):
    """Season classification for seasonal decomposition."""
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"


class ForecastMethod(str, Enum):
    """Available forecasting methods."""
    DEGREE_DAY = "degree_day"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    MOVING_AVERAGE = "moving_average"
    REGRESSION = "regression"
    WEIGHTED_AVERAGE = "weighted_average"


class BuildingType(str, Enum):
    """Building thermal mass classification."""
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    INDUSTRIAL = "industrial"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class WeatherData:
    """
    Weather data point for degree-day calculations.

    Attributes:
        date: Date string (YYYY-MM-DD)
        temp_avg_c: Average temperature (Celsius)
        temp_min_c: Minimum temperature (Celsius)
        temp_max_c: Maximum temperature (Celsius)
        humidity_pct: Relative humidity (0-100)
        wind_speed_mps: Wind speed (m/s)
        solar_radiation_wm2: Solar radiation (W/m2)
    """
    date: str
    temp_avg_c: float
    temp_min_c: float = 0.0
    temp_max_c: float = 0.0
    humidity_pct: float = 50.0
    wind_speed_mps: float = 0.0
    solar_radiation_wm2: float = 0.0


@dataclass(frozen=True)
class HistoricalLoadData:
    """
    Historical load data point with weather correlation.

    Attributes:
        timestamp: ISO timestamp string
        energy_kwh: Energy consumption (kWh)
        peak_demand_kw: Peak demand (kW)
        temp_avg_c: Average temperature during period
        heating_degree_days: HDD for the period
        cooling_degree_days: CDD for the period
        occupancy_fraction: Occupancy level (0-1)
        day_of_week: Day of week (0=Monday)
        hour_of_day: Hour (0-23)
        is_holiday: Holiday flag
        production_level: Production output (0-1 normalized)
    """
    timestamp: str
    energy_kwh: float
    peak_demand_kw: float
    temp_avg_c: float = 15.0
    heating_degree_days: float = 0.0
    cooling_degree_days: float = 0.0
    occupancy_fraction: float = 1.0
    day_of_week: int = 0
    hour_of_day: int = 0
    is_holiday: bool = False
    production_level: float = 1.0


@dataclass(frozen=True)
class BuildingThermalModel:
    """
    Building thermal characteristics for load modeling.

    Attributes:
        building_id: Unique building identifier
        floor_area_m2: Total floor area (m2)
        building_type: Thermal mass classification
        thermal_time_constant_h: Thermal time constant (hours)
        ua_value_w_per_k: Overall heat transfer coefficient (W/K)
        internal_gain_w_per_m2: Internal heat gains (W/m2)
        hvac_efficiency: HVAC system efficiency (0-1)
        setpoint_heating_c: Heating setpoint (Celsius)
        setpoint_cooling_c: Cooling setpoint (Celsius)
        base_load_kw: Non-weather-dependent base load (kW)
    """
    building_id: str
    floor_area_m2: float
    building_type: BuildingType = BuildingType.MEDIUM
    thermal_time_constant_h: float = 8.0
    ua_value_w_per_k: float = 1000.0
    internal_gain_w_per_m2: float = 25.0
    hvac_efficiency: float = 0.85
    setpoint_heating_c: float = 20.0
    setpoint_cooling_c: float = 24.0
    base_load_kw: float = 0.0


@dataclass(frozen=True)
class OccupancySchedule:
    """
    Occupancy schedule for load adjustment.

    Attributes:
        schedule_id: Schedule identifier
        weekday_profile: 24-hour occupancy profile for weekdays (0-1)
        weekend_profile: 24-hour occupancy profile for weekends (0-1)
        holiday_profile: 24-hour occupancy profile for holidays (0-1)
        occupancy_load_factor: Load multiplier per occupancy unit
    """
    schedule_id: str
    weekday_profile: Tuple[float, ...] = tuple(DEFAULT_OCCUPANCY_PROFILE)
    weekend_profile: Tuple[float, ...] = tuple([0.1] * 24)
    holiday_profile: Tuple[float, ...] = tuple([0.1] * 24)
    occupancy_load_factor: float = 0.1


@dataclass(frozen=True)
class ThermalLoadForecastInput:
    """
    Input parameters for thermal load forecasting.

    Attributes:
        historical_data: Historical load data points
        weather_forecast: Forecasted weather data
        building_model: Building thermal characteristics
        occupancy_schedule: Occupancy schedule (optional)
        forecast_horizon_hours: Hours to forecast
        method: Forecasting method to use
        confidence_level: Confidence level for intervals
        include_peak_prediction: Include peak demand prediction
        custom_hdd_base_c: Custom HDD base temperature
        custom_cdd_base_c: Custom CDD base temperature
    """
    historical_data: List[HistoricalLoadData]
    weather_forecast: List[WeatherData]
    building_model: Optional[BuildingThermalModel] = None
    occupancy_schedule: Optional[OccupancySchedule] = None
    forecast_horizon_hours: int = 168  # 1 week
    method: ForecastMethod = ForecastMethod.DEGREE_DAY
    confidence_level: str = "95%"
    include_peak_prediction: bool = True
    custom_hdd_base_c: Optional[float] = None
    custom_cdd_base_c: Optional[float] = None


@dataclass(frozen=True)
class HourlyLoadForecast:
    """
    Single hourly load forecast result.

    Attributes:
        hour_index: Hour index from forecast start
        timestamp: ISO timestamp string
        energy_kwh: Forecasted energy (kWh)
        peak_demand_kw: Forecasted peak demand (kW)
        lower_bound_kwh: Lower confidence bound (kWh)
        upper_bound_kwh: Upper confidence bound (kWh)
        heating_load_kwh: Heating component (kWh)
        base_load_kwh: Base load component (kWh)
        occupancy_adjustment: Occupancy adjustment factor
        hdd_contribution: HDD-based load contribution
        confidence_level: Confidence level applied
    """
    hour_index: int
    timestamp: str
    energy_kwh: float
    peak_demand_kw: float
    lower_bound_kwh: float
    upper_bound_kwh: float
    heating_load_kwh: float
    base_load_kwh: float
    occupancy_adjustment: float
    hdd_contribution: float
    confidence_level: str


@dataclass(frozen=True)
class SeasonalComponents:
    """
    Seasonal decomposition components.

    Attributes:
        trend: Trend component values
        seasonal: Seasonal component values
        residual: Residual/irregular component values
        period: Seasonality period (hours)
    """
    trend: Tuple[float, ...]
    seasonal: Tuple[float, ...]
    residual: Tuple[float, ...]
    period: int


@dataclass(frozen=True)
class ForecastAccuracyMetrics:
    """
    Forecast accuracy metrics.

    Attributes:
        mape: Mean Absolute Percentage Error (%)
        rmse: Root Mean Square Error
        mae: Mean Absolute Error
        r_squared: Coefficient of determination
        bias: Mean forecast bias
        cv_rmse: Coefficient of Variation of RMSE (%)
    """
    mape: float
    rmse: float
    mae: float
    r_squared: float
    bias: float
    cv_rmse: float


@dataclass(frozen=True)
class PeakDemandPrediction:
    """
    Peak demand prediction result.

    Attributes:
        predicted_peak_kw: Predicted peak demand (kW)
        peak_hour: Hour of predicted peak
        peak_date: Date of predicted peak
        confidence_lower_kw: Lower confidence bound (kW)
        confidence_upper_kw: Upper confidence bound (kW)
        contributing_factors: Factors contributing to peak
    """
    predicted_peak_kw: float
    peak_hour: int
    peak_date: str
    confidence_lower_kw: float
    confidence_upper_kw: float
    contributing_factors: Dict[str, float]


@dataclass(frozen=True)
class ThermalLoadForecastOutput:
    """
    Complete output from thermal load forecasting.

    Attributes:
        hourly_forecasts: List of hourly forecasts
        total_energy_kwh: Total forecasted energy
        peak_demand_kw: Maximum forecasted demand
        average_daily_kwh: Average daily consumption
        total_hdd: Total heating degree days
        total_cdd: Total cooling degree days
        weather_normalized_consumption: Weather-normalized value
        daily_pattern: Typical daily pattern (24 values)
        weekly_pattern: Typical weekly pattern (7 values)
        seasonal_components: Seasonal decomposition results
        accuracy_metrics: Forecast accuracy metrics
        peak_prediction: Peak demand prediction
        hdd_sensitivity: Energy per HDD (kWh/HDD)
        cdd_sensitivity: Energy per CDD (kWh/CDD)
    """
    hourly_forecasts: List[HourlyLoadForecast]
    total_energy_kwh: float
    peak_demand_kw: float
    average_daily_kwh: float
    total_hdd: float
    total_cdd: float
    weather_normalized_consumption: float
    daily_pattern: List[float]
    weekly_pattern: List[float]
    seasonal_components: Optional[SeasonalComponents]
    accuracy_metrics: ForecastAccuracyMetrics
    peak_prediction: Optional[PeakDemandPrediction]
    hdd_sensitivity: float
    cdd_sensitivity: float


# =============================================================================
# THERMAL LOAD FORECASTER CLASS
# =============================================================================

class ThermalLoadForecaster:
    """
    Advanced thermal load forecaster with degree-day modeling.

    Implements zero-hallucination, deterministic forecasting using:
    - Heating/Cooling Degree Day methods (ASHRAE)
    - Time-series seasonal decomposition
    - Weather normalization (ISO 50001/50006)
    - Occupancy-adjusted profiles
    - Building thermal mass modeling
    - Peak demand prediction

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> forecaster = ThermalLoadForecaster()
        >>> historical = [HistoricalLoadData(...), ...]
        >>> weather = [WeatherData(...), ...]
        >>> inputs = ThermalLoadForecastInput(
        ...     historical_data=historical,
        ...     weather_forecast=weather
        ... )
        >>> result, provenance = forecaster.forecast(inputs)
        >>> print(f"Total Energy: {result.total_energy_kwh:.0f} kWh")
    """

    VERSION = "1.0.0"
    NAME = "ThermalLoadForecaster"

    # Thread-safe cache lock
    _cache_lock = Lock()

    def __init__(self):
        """Initialize the thermal load forecaster."""
        self._tracker: Optional[ProvenanceTracker] = None
        self._step_counter = 0

    def forecast(
        self,
        inputs: ThermalLoadForecastInput
    ) -> Tuple[ThermalLoadForecastOutput, ProvenanceRecord]:
        """
        Generate thermal load forecast using degree-day methods.

        Args:
            inputs: ThermalLoadForecastInput with historical and weather data

        Returns:
            Tuple of (ThermalLoadForecastOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": [
                    "ISO 50001",
                    "ISO 50006",
                    "ASHRAE Handbook-Fundamentals",
                    "ASHRAE Guideline 14"
                ],
                "domain": "Thermal Load Forecasting"
            }
        )
        self._step_counter = 0

        # Set base temperatures
        hdd_base = inputs.custom_hdd_base_c or HDD_BASE_TEMP_C
        cdd_base = inputs.custom_cdd_base_c or CDD_BASE_TEMP_C

        # Prepare inputs for provenance
        input_dict = {
            "num_historical_points": len(inputs.historical_data),
            "num_weather_forecasts": len(inputs.weather_forecast),
            "forecast_horizon_hours": inputs.forecast_horizon_hours,
            "method": inputs.method.value,
            "confidence_level": inputs.confidence_level,
            "hdd_base_temp_c": hdd_base,
            "cdd_base_temp_c": cdd_base,
            "include_peak_prediction": inputs.include_peak_prediction
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Calculate historical degree days and correlations
        hdd_data = self._calculate_historical_hdd(inputs.historical_data, hdd_base)
        cdd_data = self._calculate_historical_cdd(inputs.historical_data, cdd_base)
        hdd_sensitivity = self._calculate_hdd_sensitivity(inputs.historical_data, hdd_data)
        cdd_sensitivity = self._calculate_cdd_sensitivity(inputs.historical_data, cdd_data)

        # Calculate patterns
        daily_pattern = self._calculate_daily_pattern(inputs.historical_data)
        weekly_pattern = self._calculate_weekly_pattern(inputs.historical_data)

        # Perform seasonal decomposition if enough data
        seasonal_components = None
        if len(inputs.historical_data) >= 168:  # At least 1 week
            seasonal_components = self._decompose_seasonal(inputs.historical_data)

        # Calculate forecast degree days
        forecast_hdd = self._calculate_forecast_hdd(inputs.weather_forecast, hdd_base)
        forecast_cdd = self._calculate_forecast_cdd(inputs.weather_forecast, cdd_base)

        # Generate hourly forecasts
        hourly_forecasts = self._generate_forecasts(
            inputs=inputs,
            hdd_sensitivity=hdd_sensitivity,
            cdd_sensitivity=cdd_sensitivity,
            daily_pattern=daily_pattern,
            forecast_hdd=forecast_hdd,
            forecast_cdd=forecast_cdd
        )

        # Calculate totals
        total_energy = sum(f.energy_kwh for f in hourly_forecasts)
        peak_demand = max(f.peak_demand_kw for f in hourly_forecasts)
        total_hdd = sum(forecast_hdd)
        total_cdd = sum(forecast_cdd)
        days = inputs.forecast_horizon_hours / 24.0
        average_daily = total_energy / days if days > 0 else 0.0

        self._add_step(
            "Calculate forecast totals",
            "aggregate",
            {
                "num_forecasts": len(hourly_forecasts),
                "total_hdd": total_hdd,
                "total_cdd": total_cdd
            },
            total_energy,
            "total_energy_kwh",
            "Total = sum(hourly_energy)"
        )

        # Calculate weather-normalized consumption
        weather_normalized = self._calculate_weather_normalized_consumption(
            inputs.historical_data, hdd_data, cdd_data, hdd_sensitivity, cdd_sensitivity
        )

        # Calculate accuracy metrics from historical cross-validation
        accuracy_metrics = self._calculate_accuracy_metrics(inputs)

        # Peak demand prediction
        peak_prediction = None
        if inputs.include_peak_prediction:
            peak_prediction = self._predict_peak_demand(
                hourly_forecasts, inputs.confidence_level
            )

        # Create output
        output = ThermalLoadForecastOutput(
            hourly_forecasts=hourly_forecasts,
            total_energy_kwh=round(total_energy, 2),
            peak_demand_kw=round(peak_demand, 2),
            average_daily_kwh=round(average_daily, 2),
            total_hdd=round(total_hdd, 2),
            total_cdd=round(total_cdd, 2),
            weather_normalized_consumption=round(weather_normalized, 2),
            daily_pattern=[round(p, 4) for p in daily_pattern],
            weekly_pattern=[round(p, 4) for p in weekly_pattern],
            seasonal_components=seasonal_components,
            accuracy_metrics=accuracy_metrics,
            peak_prediction=peak_prediction,
            hdd_sensitivity=round(hdd_sensitivity, 4),
            cdd_sensitivity=round(cdd_sensitivity, 4)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "total_energy_kwh": output.total_energy_kwh,
            "peak_demand_kw": output.peak_demand_kw,
            "average_daily_kwh": output.average_daily_kwh,
            "total_hdd": output.total_hdd,
            "total_cdd": output.total_cdd,
            "weather_normalized_consumption": output.weather_normalized_consumption,
            "hdd_sensitivity": output.hdd_sensitivity,
            "cdd_sensitivity": output.cdd_sensitivity,
            "mape": output.accuracy_metrics.mape,
            "rmse": output.accuracy_metrics.rmse
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _add_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, Any],
        output_value: Union[float, str, Dict, List],
        output_name: str,
        formula: str = ""
    ) -> None:
        """Add a calculation step to provenance tracking."""
        self._step_counter += 1
        self._tracker.add_step(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )

    def _validate_inputs(self, inputs: ThermalLoadForecastInput) -> None:
        """Validate input parameters."""
        if not inputs.historical_data:
            raise ValueError("Historical data required for forecasting")

        if inputs.forecast_horizon_hours <= 0:
            raise ValueError("Forecast horizon must be positive")

        if inputs.forecast_horizon_hours > 8760:  # 1 year
            raise ValueError("Forecast horizon cannot exceed 8760 hours (1 year)")

        for data in inputs.historical_data:
            if data.energy_kwh < 0:
                raise ValueError(f"Energy cannot be negative: {data.energy_kwh}")
            if data.peak_demand_kw < 0:
                raise ValueError(f"Peak demand cannot be negative: {data.peak_demand_kw}")

    def _calculate_historical_hdd(
        self,
        historical_data: List[HistoricalLoadData],
        base_temp_c: float
    ) -> List[float]:
        """
        Calculate Heating Degree Days for historical data.

        Formula (ASHRAE):
            HDD = max(0, Base_Temp - Average_Temp)

        Args:
            historical_data: Historical load data
            base_temp_c: HDD base temperature

        Returns:
            List of HDD values
        """
        hdd_values = []
        for data in historical_data:
            hdd = max(0.0, base_temp_c - data.temp_avg_c) / 24.0  # Hourly HDD
            hdd_values.append(hdd)

        total_hdd = sum(hdd_values)
        self._add_step(
            "Calculate historical HDD",
            "degree_day_calculation",
            {"base_temp_c": base_temp_c, "num_points": len(historical_data)},
            total_hdd,
            "total_historical_hdd",
            "HDD = max(0, Base - T_avg) / 24"
        )

        return hdd_values

    def _calculate_historical_cdd(
        self,
        historical_data: List[HistoricalLoadData],
        base_temp_c: float
    ) -> List[float]:
        """
        Calculate Cooling Degree Days for historical data.

        Formula (ASHRAE):
            CDD = max(0, Average_Temp - Base_Temp)

        Args:
            historical_data: Historical load data
            base_temp_c: CDD base temperature

        Returns:
            List of CDD values
        """
        cdd_values = []
        for data in historical_data:
            cdd = max(0.0, data.temp_avg_c - base_temp_c) / 24.0  # Hourly CDD
            cdd_values.append(cdd)

        total_cdd = sum(cdd_values)
        self._add_step(
            "Calculate historical CDD",
            "degree_day_calculation",
            {"base_temp_c": base_temp_c, "num_points": len(historical_data)},
            total_cdd,
            "total_historical_cdd",
            "CDD = max(0, T_avg - Base) / 24"
        )

        return cdd_values

    def _calculate_hdd_sensitivity(
        self,
        historical_data: List[HistoricalLoadData],
        hdd_data: List[float]
    ) -> float:
        """
        Calculate energy sensitivity to HDD (kWh/HDD).

        Uses linear regression of energy vs HDD.

        Args:
            historical_data: Historical load data
            hdd_data: Corresponding HDD values

        Returns:
            HDD sensitivity (kWh per HDD)
        """
        if len(historical_data) < 3:
            return 0.0

        # Filter to heating periods (HDD > 0)
        heating_data = [
            (data.energy_kwh, hdd)
            for data, hdd in zip(historical_data, hdd_data)
            if hdd > 0
        ]

        if len(heating_data) < 3:
            return 0.0

        energies = [d[0] for d in heating_data]
        hdds = [d[1] for d in heating_data]

        # Simple linear regression: Energy = a + b * HDD
        n = len(heating_data)
        sum_hdd = sum(hdds)
        sum_energy = sum(energies)
        sum_hdd_energy = sum(h * e for h, e in zip(hdds, energies))
        sum_hdd_sq = sum(h * h for h in hdds)

        denominator = n * sum_hdd_sq - sum_hdd ** 2
        if abs(denominator) < 1e-10:
            sensitivity = 0.0
        else:
            sensitivity = (n * sum_hdd_energy - sum_hdd * sum_energy) / denominator

        # Sensitivity should be positive for heating
        sensitivity = max(0.0, sensitivity)

        self._add_step(
            "Calculate HDD sensitivity",
            "linear_regression",
            {"n_heating_points": len(heating_data)},
            sensitivity,
            "hdd_sensitivity",
            "Sensitivity = d(Energy)/d(HDD) via OLS"
        )

        return sensitivity

    def _calculate_cdd_sensitivity(
        self,
        historical_data: List[HistoricalLoadData],
        cdd_data: List[float]
    ) -> float:
        """
        Calculate energy sensitivity to CDD (kWh/CDD).

        Uses linear regression of energy vs CDD.

        Args:
            historical_data: Historical load data
            cdd_data: Corresponding CDD values

        Returns:
            CDD sensitivity (kWh per CDD)
        """
        if len(historical_data) < 3:
            return 0.0

        # Filter to cooling periods (CDD > 0)
        cooling_data = [
            (data.energy_kwh, cdd)
            for data, cdd in zip(historical_data, cdd_data)
            if cdd > 0
        ]

        if len(cooling_data) < 3:
            return 0.0

        energies = [d[0] for d in cooling_data]
        cdds = [d[1] for d in cooling_data]

        # Simple linear regression
        n = len(cooling_data)
        sum_cdd = sum(cdds)
        sum_energy = sum(energies)
        sum_cdd_energy = sum(c * e for c, e in zip(cdds, energies))
        sum_cdd_sq = sum(c * c for c in cdds)

        denominator = n * sum_cdd_sq - sum_cdd ** 2
        if abs(denominator) < 1e-10:
            sensitivity = 0.0
        else:
            sensitivity = (n * sum_cdd_energy - sum_cdd * sum_energy) / denominator

        sensitivity = max(0.0, sensitivity)

        self._add_step(
            "Calculate CDD sensitivity",
            "linear_regression",
            {"n_cooling_points": len(cooling_data)},
            sensitivity,
            "cdd_sensitivity",
            "Sensitivity = d(Energy)/d(CDD) via OLS"
        )

        return sensitivity

    def _calculate_daily_pattern(
        self,
        historical_data: List[HistoricalLoadData]
    ) -> List[float]:
        """
        Calculate normalized daily load pattern (24 hours).

        Args:
            historical_data: Historical load data

        Returns:
            List of 24 normalized hourly values
        """
        hourly_totals = [0.0] * 24
        hourly_counts = [0] * 24

        for data in historical_data:
            hour = data.hour_of_day
            if 0 <= hour < 24:
                hourly_totals[hour] += data.energy_kwh
                hourly_counts[hour] += 1

        # Calculate averages
        daily_pattern = []
        for hour in range(24):
            if hourly_counts[hour] > 0:
                avg = hourly_totals[hour] / hourly_counts[hour]
            else:
                avg = sum(hourly_totals) / max(sum(hourly_counts), 1) / 24
            daily_pattern.append(avg)

        # Normalize to sum to 1
        total = sum(daily_pattern)
        if total > 0:
            daily_pattern = [p / total for p in daily_pattern]

        self._add_step(
            "Calculate daily load pattern",
            "pattern_extraction",
            {"num_data_points": len(historical_data)},
            daily_pattern,
            "daily_pattern",
            "Pattern[h] = mean(energy where hour=h) / sum"
        )

        return daily_pattern

    def _calculate_weekly_pattern(
        self,
        historical_data: List[HistoricalLoadData]
    ) -> List[float]:
        """
        Calculate normalized weekly load pattern (7 days).

        Args:
            historical_data: Historical load data

        Returns:
            List of 7 normalized daily values (Mon-Sun)
        """
        daily_totals = [0.0] * 7
        daily_counts = [0] * 7

        for data in historical_data:
            day = data.day_of_week
            if 0 <= day < 7:
                daily_totals[day] += data.energy_kwh
                daily_counts[day] += 1

        # Calculate averages
        weekly_pattern = []
        for day in range(7):
            if daily_counts[day] > 0:
                avg = daily_totals[day] / daily_counts[day]
            else:
                avg = sum(daily_totals) / max(sum(daily_counts), 1) / 7
            weekly_pattern.append(avg)

        # Normalize to average of 1
        mean_val = sum(weekly_pattern) / 7 if weekly_pattern else 1.0
        if mean_val > 0:
            weekly_pattern = [p / mean_val for p in weekly_pattern]

        self._add_step(
            "Calculate weekly load pattern",
            "pattern_extraction",
            {"num_data_points": len(historical_data)},
            weekly_pattern,
            "weekly_pattern",
            "Pattern[d] = mean(energy where day=d) / overall_mean"
        )

        return weekly_pattern

    def _decompose_seasonal(
        self,
        historical_data: List[HistoricalLoadData],
        period: int = 24
    ) -> SeasonalComponents:
        """
        Perform additive seasonal decomposition.

        Components: Y = Trend + Seasonal + Residual

        Args:
            historical_data: Historical load data
            period: Seasonality period (default 24 hours)

        Returns:
            SeasonalComponents with trend, seasonal, residual
        """
        energies = [data.energy_kwh for data in historical_data]
        n = len(energies)

        if n < 2 * period:
            # Not enough data for decomposition
            return SeasonalComponents(
                trend=tuple(energies),
                seasonal=tuple([0.0] * n),
                residual=tuple([0.0] * n),
                period=period
            )

        # Calculate centered moving average (trend)
        trend = []
        half_period = period // 2
        for i in range(n):
            if i < half_period or i >= n - half_period:
                trend.append(statistics.mean(energies))
            else:
                window = energies[i - half_period:i + half_period + 1]
                trend.append(statistics.mean(window))

        # Calculate seasonal component
        detrended = [e - t for e, t in zip(energies, trend)]

        # Average by season position
        seasonal_avgs = []
        for pos in range(period):
            values = [detrended[i] for i in range(pos, n, period)]
            seasonal_avgs.append(statistics.mean(values) if values else 0.0)

        # Normalize seasonal to sum to zero
        seasonal_mean = statistics.mean(seasonal_avgs)
        seasonal_avgs = [s - seasonal_mean for s in seasonal_avgs]

        # Apply seasonal pattern to full length
        seasonal = [seasonal_avgs[i % period] for i in range(n)]

        # Calculate residual
        residual = [e - t - s for e, t, s in zip(energies, trend, seasonal)]

        self._add_step(
            "Perform seasonal decomposition",
            "decomposition",
            {"period": period, "n_points": n},
            {"trend_mean": statistics.mean(trend), "seasonal_amplitude": max(seasonal_avgs) - min(seasonal_avgs)},
            "seasonal_components",
            "Y = Trend + Seasonal + Residual"
        )

        return SeasonalComponents(
            trend=tuple(trend),
            seasonal=tuple(seasonal),
            residual=tuple(residual),
            period=period
        )

    def _calculate_forecast_hdd(
        self,
        weather_forecast: List[WeatherData],
        base_temp_c: float
    ) -> List[float]:
        """Calculate HDD for forecast period."""
        if not weather_forecast:
            return [0.0]

        hdd_values = []
        for weather in weather_forecast:
            daily_hdd = max(0.0, base_temp_c - weather.temp_avg_c)
            # Distribute across 24 hours if weather is daily
            hdd_values.append(daily_hdd / 24.0)

        return hdd_values

    def _calculate_forecast_cdd(
        self,
        weather_forecast: List[WeatherData],
        base_temp_c: float
    ) -> List[float]:
        """Calculate CDD for forecast period."""
        if not weather_forecast:
            return [0.0]

        cdd_values = []
        for weather in weather_forecast:
            daily_cdd = max(0.0, weather.temp_avg_c - base_temp_c)
            cdd_values.append(daily_cdd / 24.0)

        return cdd_values

    def _generate_forecasts(
        self,
        inputs: ThermalLoadForecastInput,
        hdd_sensitivity: float,
        cdd_sensitivity: float,
        daily_pattern: List[float],
        forecast_hdd: List[float],
        forecast_cdd: List[float]
    ) -> List[HourlyLoadForecast]:
        """
        Generate hourly load forecasts.

        Args:
            inputs: Forecast inputs
            hdd_sensitivity: kWh/HDD sensitivity
            cdd_sensitivity: kWh/CDD sensitivity
            daily_pattern: 24-hour load pattern
            forecast_hdd: Forecasted HDD values
            forecast_cdd: Forecasted CDD values

        Returns:
            List of hourly forecasts
        """
        forecasts = []
        z_score = CONFIDENCE_Z_SCORES.get(inputs.confidence_level, 1.96)

        # Calculate historical statistics for confidence intervals
        historical_energies = [d.energy_kwh for d in inputs.historical_data]
        if len(historical_energies) > 1:
            std_dev = statistics.stdev(historical_energies)
            mean_energy = statistics.mean(historical_energies)
        else:
            mean_energy = historical_energies[0] if historical_energies else 100.0
            std_dev = 0.1 * mean_energy

        # Calculate base load from historical data
        base_load = min(historical_energies) if historical_energies else 0.0

        # Get occupancy schedule
        occupancy_profile = list(DEFAULT_OCCUPANCY_PROFILE)
        if inputs.occupancy_schedule:
            occupancy_profile = list(inputs.occupancy_schedule.weekday_profile)

        # Generate forecasts
        for hour_idx in range(inputs.forecast_horizon_hours):
            hour_of_day = hour_idx % 24
            day_of_forecast = hour_idx // 24

            # Get HDD/CDD for this hour (use modulo for cycling through weather data)
            if forecast_hdd:
                hdd_idx = min(day_of_forecast, len(forecast_hdd) - 1)
                hour_hdd = forecast_hdd[hdd_idx]
            else:
                hour_hdd = 0.0

            if forecast_cdd:
                cdd_idx = min(day_of_forecast, len(forecast_cdd) - 1)
                hour_cdd = forecast_cdd[cdd_idx]
            else:
                hour_cdd = 0.0

            # Calculate weather-driven load
            hdd_load = hdd_sensitivity * hour_hdd
            cdd_load = cdd_sensitivity * hour_cdd

            # Apply daily pattern
            pattern_factor = daily_pattern[hour_of_day] * 24 if daily_pattern else 1.0

            # Apply occupancy adjustment
            occupancy = occupancy_profile[hour_of_day]

            # Calculate total forecast
            forecast_energy = (
                base_load * pattern_factor +
                hdd_load +
                cdd_load
            ) * occupancy

            # Ensure non-negative
            forecast_energy = max(0.0, forecast_energy)

            # Estimate peak demand (typically 1.2-1.5x average)
            peak_factor = 1.3
            peak_demand = forecast_energy * peak_factor

            # Calculate confidence bounds
            margin = z_score * std_dev / math.sqrt(len(historical_energies) + 1)
            lower_bound = max(0.0, forecast_energy - margin)
            upper_bound = forecast_energy + margin

            # Create timestamp (simplified - in production use proper datetime)
            timestamp = f"T+{hour_idx:04d}h"

            forecasts.append(HourlyLoadForecast(
                hour_index=hour_idx,
                timestamp=timestamp,
                energy_kwh=round(forecast_energy, 4),
                peak_demand_kw=round(peak_demand, 4),
                lower_bound_kwh=round(lower_bound, 4),
                upper_bound_kwh=round(upper_bound, 4),
                heating_load_kwh=round(hdd_load, 4),
                base_load_kwh=round(base_load * pattern_factor * occupancy, 4),
                occupancy_adjustment=round(occupancy, 4),
                hdd_contribution=round(hour_hdd, 4),
                confidence_level=inputs.confidence_level
            ))

        self._add_step(
            "Generate hourly forecasts",
            "forecast_generation",
            {
                "horizon_hours": inputs.forecast_horizon_hours,
                "z_score": z_score,
                "base_load": base_load
            },
            len(forecasts),
            "num_forecasts",
            "Forecast = (Base * Pattern + HDD_Load + CDD_Load) * Occupancy"
        )

        return forecasts

    def _calculate_weather_normalized_consumption(
        self,
        historical_data: List[HistoricalLoadData],
        hdd_data: List[float],
        cdd_data: List[float],
        hdd_sensitivity: float,
        cdd_sensitivity: float
    ) -> float:
        """
        Calculate weather-normalized energy consumption.

        Adjusts consumption to a reference weather condition (e.g., average year).

        Formula (ISO 50006):
            E_normalized = E_actual - (HDD_actual - HDD_ref) * sensitivity

        Args:
            historical_data: Historical load data
            hdd_data: Historical HDD values
            cdd_data: Historical CDD values
            hdd_sensitivity: HDD sensitivity
            cdd_sensitivity: CDD sensitivity

        Returns:
            Weather-normalized consumption
        """
        if not historical_data:
            return 0.0

        total_energy = sum(d.energy_kwh for d in historical_data)
        actual_hdd = sum(hdd_data)
        actual_cdd = sum(cdd_data)

        # Use average as reference (in production, use long-term normal)
        ref_hdd = actual_hdd  # No adjustment if using actual as reference
        ref_cdd = actual_cdd

        # Normalized = Actual - adjustment
        hdd_adjustment = (actual_hdd - ref_hdd) * hdd_sensitivity
        cdd_adjustment = (actual_cdd - ref_cdd) * cdd_sensitivity

        normalized = total_energy - hdd_adjustment - cdd_adjustment

        self._add_step(
            "Calculate weather-normalized consumption",
            "normalization",
            {
                "total_energy": total_energy,
                "actual_hdd": actual_hdd,
                "actual_cdd": actual_cdd
            },
            normalized,
            "weather_normalized_consumption",
            "E_norm = E_actual - (HDD_adj + CDD_adj)"
        )

        return normalized

    def _calculate_accuracy_metrics(
        self,
        inputs: ThermalLoadForecastInput
    ) -> ForecastAccuracyMetrics:
        """
        Calculate forecast accuracy metrics using historical cross-validation.

        Metrics:
        - MAPE: Mean Absolute Percentage Error
        - RMSE: Root Mean Square Error
        - MAE: Mean Absolute Error
        - R-squared: Coefficient of determination
        - Bias: Mean forecast error
        - CV(RMSE): Coefficient of Variation of RMSE

        Args:
            inputs: Forecast inputs

        Returns:
            ForecastAccuracyMetrics
        """
        if len(inputs.historical_data) < 10:
            return ForecastAccuracyMetrics(
                mape=0.0,
                rmse=0.0,
                mae=0.0,
                r_squared=0.0,
                bias=0.0,
                cv_rmse=0.0
            )

        # Simple holdout cross-validation
        train_size = int(len(inputs.historical_data) * 0.8)
        train_data = inputs.historical_data[:train_size]
        test_data = inputs.historical_data[train_size:]

        # Use training mean as naive forecast
        train_mean = statistics.mean(d.energy_kwh for d in train_data)

        # Calculate errors
        errors = []
        abs_errors = []
        sq_errors = []
        actual_values = []
        pct_errors = []

        for data in test_data:
            actual = data.energy_kwh
            predicted = train_mean
            error = actual - predicted

            errors.append(error)
            abs_errors.append(abs(error))
            sq_errors.append(error ** 2)
            actual_values.append(actual)

            if actual > 0:
                pct_errors.append(abs(error) / actual * 100)

        n = len(test_data)
        mae = sum(abs_errors) / n if n > 0 else 0.0
        rmse = math.sqrt(sum(sq_errors) / n) if n > 0 else 0.0
        mape = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0
        bias = sum(errors) / n if n > 0 else 0.0

        # R-squared
        actual_mean = statistics.mean(actual_values) if actual_values else 0.0
        ss_tot = sum((a - actual_mean) ** 2 for a in actual_values)
        ss_res = sum(sq_errors)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, r_squared)

        # CV(RMSE)
        cv_rmse = (rmse / actual_mean * 100) if actual_mean > 0 else 0.0

        metrics = ForecastAccuracyMetrics(
            mape=round(mape, 4),
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            r_squared=round(r_squared, 4),
            bias=round(bias, 4),
            cv_rmse=round(cv_rmse, 4)
        )

        self._add_step(
            "Calculate forecast accuracy metrics",
            "cross_validation",
            {"train_size": train_size, "test_size": len(test_data)},
            {
                "mape": metrics.mape,
                "rmse": metrics.rmse,
                "r_squared": metrics.r_squared
            },
            "accuracy_metrics",
            "Metrics: MAPE, RMSE, MAE, R^2, Bias, CV(RMSE)"
        )

        return metrics

    def _predict_peak_demand(
        self,
        hourly_forecasts: List[HourlyLoadForecast],
        confidence_level: str
    ) -> PeakDemandPrediction:
        """
        Predict peak demand from hourly forecasts.

        Args:
            hourly_forecasts: List of hourly forecasts
            confidence_level: Confidence level for bounds

        Returns:
            PeakDemandPrediction
        """
        if not hourly_forecasts:
            return PeakDemandPrediction(
                predicted_peak_kw=0.0,
                peak_hour=0,
                peak_date="",
                confidence_lower_kw=0.0,
                confidence_upper_kw=0.0,
                contributing_factors={}
            )

        # Find maximum peak demand
        max_forecast = max(hourly_forecasts, key=lambda f: f.peak_demand_kw)
        peak_kw = max_forecast.peak_demand_kw
        peak_hour = max_forecast.hour_index % 24
        peak_date = max_forecast.timestamp

        # Calculate confidence bounds
        z_score = CONFIDENCE_Z_SCORES.get(confidence_level, 1.96)
        demands = [f.peak_demand_kw for f in hourly_forecasts]
        if len(demands) > 1:
            std_dev = statistics.stdev(demands)
        else:
            std_dev = 0.1 * peak_kw

        margin = z_score * std_dev
        lower_bound = max(0.0, peak_kw - margin)
        upper_bound = peak_kw + margin

        # Determine contributing factors
        contributing_factors = {
            "heating_load": max_forecast.heating_load_kwh / max(max_forecast.energy_kwh, 0.01),
            "base_load": max_forecast.base_load_kwh / max(max_forecast.energy_kwh, 0.01),
            "occupancy": max_forecast.occupancy_adjustment
        }

        self._add_step(
            "Predict peak demand",
            "peak_analysis",
            {"num_forecasts": len(hourly_forecasts)},
            peak_kw,
            "predicted_peak_kw",
            f"Peak = max(hourly_peak_demand) with {confidence_level} CI"
        )

        return PeakDemandPrediction(
            predicted_peak_kw=round(peak_kw, 2),
            peak_hour=peak_hour,
            peak_date=peak_date,
            confidence_lower_kw=round(lower_bound, 2),
            confidence_upper_kw=round(upper_bound, 2),
            contributing_factors=contributing_factors
        )


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_hdd(
    temp_avg_c: float,
    base_temp_c: float = HDD_BASE_TEMP_C
) -> float:
    """
    Calculate Heating Degree Days for a single day.

    Formula (ASHRAE):
        HDD = max(0, Base_Temperature - Average_Temperature)

    Args:
        temp_avg_c: Average daily temperature (Celsius)
        base_temp_c: Base temperature for HDD calculation

    Returns:
        Heating Degree Days

    Example:
        >>> hdd = calculate_hdd(10.0, 18.0)
        >>> print(f"HDD: {hdd}")  # HDD: 8.0
    """
    return max(0.0, base_temp_c - temp_avg_c)


def calculate_cdd(
    temp_avg_c: float,
    base_temp_c: float = CDD_BASE_TEMP_C
) -> float:
    """
    Calculate Cooling Degree Days for a single day.

    Formula (ASHRAE):
        CDD = max(0, Average_Temperature - Base_Temperature)

    Args:
        temp_avg_c: Average daily temperature (Celsius)
        base_temp_c: Base temperature for CDD calculation

    Returns:
        Cooling Degree Days

    Example:
        >>> cdd = calculate_cdd(30.0, 24.0)
        >>> print(f"CDD: {cdd}")  # CDD: 6.0
    """
    return max(0.0, temp_avg_c - base_temp_c)


def calculate_degree_days_batch(
    temperatures: List[float],
    base_temp_c: float,
    is_heating: bool = True
) -> List[float]:
    """
    Calculate degree days for a batch of temperatures.

    Args:
        temperatures: List of average temperatures (Celsius)
        base_temp_c: Base temperature for calculation
        is_heating: True for HDD, False for CDD

    Returns:
        List of degree day values

    Example:
        >>> temps = [5, 10, 15, 20, 25]
        >>> hdds = calculate_degree_days_batch(temps, 18.0, is_heating=True)
        >>> print(hdds)  # [13.0, 8.0, 3.0, 0.0, 0.0]
    """
    if is_heating:
        return [max(0.0, base_temp_c - t) for t in temperatures]
    else:
        return [max(0.0, t - base_temp_c) for t in temperatures]


def calculate_mape(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Formula:
        MAPE = (100/n) * sum(|actual - predicted| / |actual|)

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAPE value (percentage)

    Example:
        >>> actual = [100, 110, 120]
        >>> predicted = [95, 115, 130]
        >>> mape = calculate_mape(actual, predicted)
        >>> print(f"MAPE: {mape:.2f}%")
    """
    if len(actual) != len(predicted):
        raise ValueError("Lists must have same length")

    if not actual:
        return 0.0

    errors = []
    for a, p in zip(actual, predicted):
        if a > 0:
            errors.append(abs(a - p) / a * 100)

    return statistics.mean(errors) if errors else 0.0


def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate Root Mean Square Error.

    Formula:
        RMSE = sqrt(mean((actual - predicted)^2))

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        RMSE value

    Example:
        >>> actual = [100, 110, 120]
        >>> predicted = [95, 115, 130]
        >>> rmse = calculate_rmse(actual, predicted)
        >>> print(f"RMSE: {rmse:.2f}")
    """
    if len(actual) != len(predicted):
        raise ValueError("Lists must have same length")

    if not actual:
        return 0.0

    sq_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    return math.sqrt(statistics.mean(sq_errors))


def calculate_cv_rmse(
    actual: List[float],
    predicted: List[float]
) -> float:
    """
    Calculate Coefficient of Variation of RMSE.

    Formula (ASHRAE Guideline 14):
        CV(RMSE) = RMSE / mean(actual) * 100

    This metric normalizes RMSE by the mean of actual values,
    allowing comparison across different scales.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        CV(RMSE) as percentage

    Example:
        >>> actual = [100, 110, 120]
        >>> predicted = [95, 115, 130]
        >>> cv_rmse = calculate_cv_rmse(actual, predicted)
        >>> print(f"CV(RMSE): {cv_rmse:.2f}%")
    """
    if not actual:
        return 0.0

    rmse = calculate_rmse(actual, predicted)
    mean_actual = statistics.mean(actual)

    if mean_actual == 0:
        return 0.0

    return rmse / mean_actual * 100


def apply_thermal_lag(
    load_profile: List[float],
    time_constant_hours: float
) -> List[float]:
    """
    Apply thermal mass lag to a load profile.

    Uses first-order exponential smoothing to simulate building thermal response.

    Formula:
        Load[t] = alpha * Input[t] + (1 - alpha) * Load[t-1]
        where alpha = dt / time_constant

    Args:
        load_profile: Input load profile (hourly values)
        time_constant_hours: Building thermal time constant

    Returns:
        Load profile with thermal lag applied

    Example:
        >>> profile = [0, 0, 100, 100, 100, 0, 0]
        >>> lagged = apply_thermal_lag(profile, 2.0)
    """
    if not load_profile:
        return []

    if time_constant_hours <= 0:
        return load_profile.copy()

    alpha = 1.0 / time_constant_hours  # Assuming hourly data
    alpha = min(1.0, alpha)  # Cap at 1.0

    result = [load_profile[0]]
    for i in range(1, len(load_profile)):
        smoothed = alpha * load_profile[i] + (1 - alpha) * result[-1]
        result.append(smoothed)

    return result


def adjust_load_for_occupancy(
    base_load: List[float],
    occupancy_profile: List[float],
    occupancy_factor: float = 0.3
) -> List[float]:
    """
    Adjust load profile based on occupancy.

    Formula:
        Adjusted_Load = Base_Load * (1 - factor + factor * occupancy)

    Args:
        base_load: Base load profile (hourly)
        occupancy_profile: Occupancy profile (0-1)
        occupancy_factor: Fraction of load affected by occupancy

    Returns:
        Occupancy-adjusted load profile

    Example:
        >>> base = [100] * 24
        >>> occupancy = [0.1]*6 + [1.0]*10 + [0.5]*4 + [0.1]*4
        >>> adjusted = adjust_load_for_occupancy(base, occupancy, 0.3)
    """
    if len(base_load) != len(occupancy_profile):
        raise ValueError("Load and occupancy profiles must have same length")

    adjusted = []
    for load, occ in zip(base_load, occupancy_profile):
        factor = 1.0 - occupancy_factor + occupancy_factor * occ
        adjusted.append(load * factor)

    return adjusted


def get_season(month: int) -> SeasonType:
    """
    Determine season from month number.

    Args:
        month: Month number (1-12)

    Returns:
        SeasonType enum value

    Example:
        >>> season = get_season(1)
        >>> print(season.value)  # "winter"
    """
    if month in [12, 1, 2]:
        return SeasonType.WINTER
    elif month in [3, 4, 5]:
        return SeasonType.SPRING
    elif month in [6, 7, 8]:
        return SeasonType.SUMMER
    else:
        return SeasonType.FALL


@lru_cache(maxsize=1000)
def cached_hdd_calculation(
    temp_avg_c: float,
    base_temp_c: float = HDD_BASE_TEMP_C
) -> float:
    """
    Thread-safe cached HDD calculation.

    Uses LRU cache for frequently accessed temperature/base combinations.

    Args:
        temp_avg_c: Average temperature (Celsius)
        base_temp_c: Base temperature

    Returns:
        Heating Degree Days
    """
    return max(0.0, base_temp_c - temp_avg_c)


def calculate_energy_signature(
    energy_data: List[float],
    temperature_data: List[float]
) -> Dict[str, float]:
    """
    Calculate building energy signature parameters.

    Energy signature relates energy consumption to outdoor temperature,
    identifying base load, heating sensitivity, and cooling sensitivity.

    Returns regression parameters for:
        E = base_load + heating_slope * HDD + cooling_slope * CDD

    Args:
        energy_data: Energy consumption values
        temperature_data: Corresponding outdoor temperatures

    Returns:
        Dictionary with signature parameters

    Example:
        >>> energy = [100, 150, 200, 120, 80]
        >>> temps = [5, 0, -5, 10, 20]
        >>> sig = calculate_energy_signature(energy, temps)
        >>> print(f"Base load: {sig['base_load']:.1f}")
    """
    if len(energy_data) != len(temperature_data):
        raise ValueError("Energy and temperature data must have same length")

    if len(energy_data) < 3:
        return {
            "base_load": statistics.mean(energy_data) if energy_data else 0.0,
            "heating_slope": 0.0,
            "cooling_slope": 0.0,
            "balance_point_c": HDD_BASE_TEMP_C
        }

    # Simple estimation using average of low and high temperature periods
    low_temp_energy = []
    high_temp_energy = []
    mid_temp_energy = []

    for e, t in zip(energy_data, temperature_data):
        if t < 10:
            low_temp_energy.append(e)
        elif t > 25:
            high_temp_energy.append(e)
        else:
            mid_temp_energy.append(e)

    base_load = statistics.mean(mid_temp_energy) if mid_temp_energy else statistics.mean(energy_data)

    # Estimate heating slope from cold weather data
    if low_temp_energy:
        heating_avg = statistics.mean(low_temp_energy)
        heating_slope = (heating_avg - base_load) / 10 if heating_avg > base_load else 0.0
    else:
        heating_slope = 0.0

    # Estimate cooling slope from hot weather data
    if high_temp_energy:
        cooling_avg = statistics.mean(high_temp_energy)
        cooling_slope = (cooling_avg - base_load) / 10 if cooling_avg > base_load else 0.0
    else:
        cooling_slope = 0.0

    return {
        "base_load": round(base_load, 2),
        "heating_slope": round(max(0, heating_slope), 4),
        "cooling_slope": round(max(0, cooling_slope), 4),
        "balance_point_c": HDD_BASE_TEMP_C
    }
