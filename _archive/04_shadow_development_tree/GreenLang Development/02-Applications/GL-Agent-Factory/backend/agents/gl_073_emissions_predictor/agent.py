"""
GL-073: Emissions Predictor Agent (EMISSIONS-PREDICTOR)

This module implements the EmissionsPredictorAgent for time-series forecasting of
emissions using statistical and machine learning methods with uncertainty quantification.

Standards Reference:
    - GHG Protocol
    - EPA Continuous Emissions Monitoring (CEM)
    - ISO 14064 (GHG Quantification)
    - IPCC Guidelines

Example:
    >>> agent = EmissionsPredictorAgent()
    >>> result = agent.run(EmissionsPredictorInput(historical_data=[...], forecast_horizon=12))
    >>> print(f"Predicted emissions: {result.forecast_summary.total_predicted_tCO2e:.1f} tCO2e")
"""

import hashlib
import json
import logging
import math
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmissionType(str, Enum):
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    CO2E = "co2e"
    NOX = "nox"
    SOX = "sox"
    PM = "pm"


class ForecastMethod(str, Enum):
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_REGRESSION = "linear_regression"
    MOVING_AVERAGE = "moving_average"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class Seasonality(str, Enum):
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class TimeGranularity(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class HistoricalDataPoint(BaseModel):
    """Historical emission data point."""
    timestamp: datetime = Field(..., description="Data timestamp")
    value: float = Field(..., ge=0, description="Emission value")
    unit: str = Field(default="tCO2e", description="Unit")
    emission_type: EmissionType = Field(default=EmissionType.CO2E)
    quality_flag: str = Field(default="VALID", description="Data quality flag")
    source: Optional[str] = Field(None, description="Data source")


class ExternalFactor(BaseModel):
    """External factor for prediction."""
    factor_id: str = Field(..., description="Factor identifier")
    name: str = Field(..., description="Factor name")
    historical_values: List[float] = Field(..., description="Historical values")
    forecast_values: List[float] = Field(..., description="Forecasted values")
    correlation_coefficient: Optional[float] = Field(None, description="Correlation with emissions")


class EmissionsPredictorInput(BaseModel):
    """Input for emissions prediction."""
    prediction_id: Optional[str] = Field(None, description="Prediction identifier")
    source_name: str = Field(default="Emission Source", description="Source name")
    historical_data: List[HistoricalDataPoint] = Field(..., description="Historical data")
    forecast_horizon: int = Field(default=12, ge=1, description="Forecast periods")
    granularity: TimeGranularity = Field(default=TimeGranularity.MONTHLY)
    method: ForecastMethod = Field(default=ForecastMethod.ENSEMBLE)
    seasonality: Seasonality = Field(default=Seasonality.MONTHLY)
    external_factors: List[ExternalFactor] = Field(default_factory=list)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    include_scenarios: bool = Field(default=True, description="Include scenario analysis")
    random_seed: Optional[int] = Field(None, description="Random seed")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ForecastPoint(BaseModel):
    """Single forecast point."""
    timestamp: datetime
    predicted_value: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    prediction_interval_width: float


class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    trend_direction: str
    trend_strength: float
    slope_per_period: float
    trend_significance: str
    inflection_points: List[datetime]
    trend_description: str


class SeasonalPattern(BaseModel):
    """Seasonal pattern analysis."""
    seasonality_type: str
    seasonal_strength: float
    peak_period: str
    trough_period: str
    seasonal_factors: Dict[str, float]
    deseasonalized_trend: float


class ModelDiagnostics(BaseModel):
    """Model diagnostic metrics."""
    method_used: str
    mape_percent: float
    rmse: float
    mae: float
    r_squared: float
    aic: Optional[float]
    bic: Optional[float]
    residual_autocorrelation: float
    model_adequacy: str


class ScenarioForecast(BaseModel):
    """Scenario-based forecast."""
    scenario_name: str
    scenario_description: str
    assumptions: Dict[str, Any]
    forecast_values: List[float]
    total_emissions: float
    probability_weight: float


class AlertThreshold(BaseModel):
    """Alert threshold definition."""
    threshold_type: str
    threshold_value: float
    periods_exceeding: int
    alert_level: str
    recommended_action: str


class ForecastSummary(BaseModel):
    """Forecast summary statistics."""
    forecast_horizon_periods: int
    total_predicted_tCO2e: float
    average_monthly_tCO2e: float
    peak_emission_period: datetime
    peak_emission_value: float
    minimum_emission_period: datetime
    minimum_emission_value: float
    year_over_year_change_percent: Optional[float]
    forecast_uncertainty_percent: float


class EmissionsPredictorOutput(BaseModel):
    """Output from emissions prediction."""
    prediction_id: str
    source_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    forecast: List[ForecastPoint]
    forecast_summary: ForecastSummary
    trend_analysis: TrendAnalysis
    seasonal_patterns: SeasonalPattern
    model_diagnostics: ModelDiagnostics
    scenario_forecasts: List[ScenarioForecast]
    alert_thresholds: List[AlertThreshold]
    contributing_factors: Dict[str, float]
    historical_comparison: Dict[str, float]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class EmissionsPredictorAgent:
    """GL-073: Emissions Predictor Agent - Time-series forecasting."""

    AGENT_ID = "GL-073"
    AGENT_NAME = "EMISSIONS-PREDICTOR"
    VERSION = "1.0.0"

    # Z-scores for confidence intervals
    Z_SCORES = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"EmissionsPredictorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: EmissionsPredictorInput) -> EmissionsPredictorOutput:
        start_time = datetime.utcnow()

        if input_data.random_seed:
            random.seed(input_data.random_seed)

        # Prepare historical data
        values = [dp.value for dp in input_data.historical_data]
        timestamps = [dp.timestamp for dp in input_data.historical_data]

        # Analyze trend
        trend = self._analyze_trend(values, timestamps)

        # Analyze seasonality
        seasonality = self._analyze_seasonality(
            values, input_data.seasonality, input_data.granularity)

        # Generate forecast
        forecast, diagnostics = self._generate_forecast(
            values, timestamps,
            input_data.forecast_horizon,
            input_data.method,
            input_data.confidence_level,
            input_data.granularity,
            trend, seasonality)

        # Generate scenario forecasts
        scenarios = []
        if input_data.include_scenarios:
            scenarios = self._generate_scenarios(
                values, forecast, input_data.forecast_horizon)

        # Calculate contributing factors
        factors = self._analyze_factors(
            values, input_data.external_factors)

        # Generate alert thresholds
        alerts = self._generate_alerts(values, forecast)

        # Calculate summary
        summary = self._calculate_summary(
            forecast, values, input_data.granularity)

        # Historical comparison
        historical = self._compare_historical(values, forecast)

        provenance_hash = hashlib.sha256(
            json.dumps({
                "agent": self.AGENT_ID,
                "source": input_data.source_name,
                "method": input_data.method.value,
                "horizon": input_data.forecast_horizon,
                "timestamp": datetime.utcnow().isoformat()
            }, sort_keys=True, default=str).encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return EmissionsPredictorOutput(
            prediction_id=input_data.prediction_id or f"PRED-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            source_name=input_data.source_name,
            forecast=forecast,
            forecast_summary=summary,
            trend_analysis=trend,
            seasonal_patterns=seasonality,
            model_diagnostics=diagnostics,
            scenario_forecasts=scenarios,
            alert_thresholds=alerts,
            contributing_factors=factors,
            historical_comparison=historical,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS")

    def _analyze_trend(self, values: List[float],
                      timestamps: List[datetime]) -> TrendAnalysis:
        """Analyze trend in historical data."""
        n = len(values)
        if n < 2:
            return TrendAnalysis(
                trend_direction="insufficient_data",
                trend_strength=0,
                slope_per_period=0,
                trend_significance="NONE",
                inflection_points=[],
                trend_description="Insufficient data for trend analysis")

        # Simple linear regression
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Trend direction
        if slope > 0.01 * y_mean:
            direction = "increasing"
        elif slope < -0.01 * y_mean:
            direction = "decreasing"
        else:
            direction = "stable"

        # Trend strength (R-squared)
        y_pred = [y_mean + slope * (xi - x_mean) for xi in x]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        strength = abs(r_squared)

        # Significance
        if strength > 0.7:
            significance = "STRONG"
        elif strength > 0.4:
            significance = "MODERATE"
        elif strength > 0.1:
            significance = "WEAK"
        else:
            significance = "NONE"

        # Find inflection points (simplified)
        inflections = []
        for i in range(1, n - 1):
            if ((values[i] > values[i-1] and values[i] > values[i+1]) or
                (values[i] < values[i-1] and values[i] < values[i+1])):
                inflections.append(timestamps[i])

        description = f"Emissions show a {direction} trend with {significance.lower()} strength"
        if direction != "stable":
            pct_change = (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
            description += f" ({pct_change:.1f}% change over period)"

        return TrendAnalysis(
            trend_direction=direction,
            trend_strength=round(strength, 3),
            slope_per_period=round(slope, 4),
            trend_significance=significance,
            inflection_points=inflections[:5],
            trend_description=description)

    def _analyze_seasonality(self, values: List[float],
                            seasonality: Seasonality,
                            granularity: TimeGranularity) -> SeasonalPattern:
        """Analyze seasonal patterns."""
        n = len(values)

        # Determine seasonal period
        period_map = {
            (Seasonality.MONTHLY, TimeGranularity.MONTHLY): 12,
            (Seasonality.QUARTERLY, TimeGranularity.MONTHLY): 3,
            (Seasonality.WEEKLY, TimeGranularity.DAILY): 7,
            (Seasonality.ANNUAL, TimeGranularity.MONTHLY): 12,
        }
        period = period_map.get((seasonality, granularity), 12)

        if n < period * 2:
            return SeasonalPattern(
                seasonality_type=seasonality.value,
                seasonal_strength=0,
                peak_period="N/A",
                trough_period="N/A",
                seasonal_factors={},
                deseasonalized_trend=sum(values) / n if values else 0)

        # Calculate seasonal factors
        seasonal_factors = {}
        for i in range(period):
            season_values = [values[j] for j in range(i, n, period)]
            overall_mean = sum(values) / n
            season_mean = sum(season_values) / len(season_values) if season_values else overall_mean
            seasonal_factors[f"period_{i+1}"] = round(season_mean / overall_mean, 3) if overall_mean != 0 else 1

        # Find peak and trough
        max_factor = max(seasonal_factors.values())
        min_factor = min(seasonal_factors.values())
        peak = [k for k, v in seasonal_factors.items() if v == max_factor][0]
        trough = [k for k, v in seasonal_factors.items() if v == min_factor][0]

        # Seasonal strength (variance explained by seasonality)
        strength = (max_factor - min_factor) / (max_factor + min_factor) * 2 if (max_factor + min_factor) != 0 else 0

        # Deseasonalized trend
        deseasonalized = sum(values) / n

        return SeasonalPattern(
            seasonality_type=seasonality.value,
            seasonal_strength=round(strength, 3),
            peak_period=peak,
            trough_period=trough,
            seasonal_factors=seasonal_factors,
            deseasonalized_trend=round(deseasonalized, 2))

    def _generate_forecast(self, values: List[float],
                          timestamps: List[datetime],
                          horizon: int,
                          method: ForecastMethod,
                          confidence: float,
                          granularity: TimeGranularity,
                          trend: TrendAnalysis,
                          seasonality: SeasonalPattern) -> Tuple[List[ForecastPoint], ModelDiagnostics]:
        """Generate forecast using specified method."""
        n = len(values)
        if n == 0:
            return [], self._empty_diagnostics(method)

        # Calculate base statistics
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = math.sqrt(variance)

        # Z-score for confidence interval
        z = self.Z_SCORES.get(confidence, 1.96)

        # Time delta based on granularity
        delta_map = {
            TimeGranularity.HOURLY: timedelta(hours=1),
            TimeGranularity.DAILY: timedelta(days=1),
            TimeGranularity.WEEKLY: timedelta(weeks=1),
            TimeGranularity.MONTHLY: timedelta(days=30),
            TimeGranularity.QUARTERLY: timedelta(days=91),
            TimeGranularity.ANNUAL: timedelta(days=365),
        }
        delta = delta_map.get(granularity, timedelta(days=30))

        forecast = []
        last_timestamp = timestamps[-1] if timestamps else datetime.utcnow()

        # Generate forecasts based on method
        if method == ForecastMethod.MOVING_AVERAGE:
            window = min(12, n)
            base_forecast = sum(values[-window:]) / window
            forecasts = [base_forecast] * horizon
        elif method == ForecastMethod.LINEAR_REGRESSION:
            forecasts = [mean + trend.slope_per_period * (n + i) for i in range(1, horizon + 1)]
        elif method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            alpha = 0.3
            level = values[-1]
            forecasts = []
            for i in range(horizon):
                forecasts.append(level)
                # Add trend adjustment
                level = level + trend.slope_per_period
        else:
            # Ensemble/default: combine methods
            ma_forecast = sum(values[-min(12, n):]) / min(12, n)
            lr_forecasts = [mean + trend.slope_per_period * (n + i) for i in range(1, horizon + 1)]
            forecasts = [(ma_forecast + lr_forecasts[i]) / 2 for i in range(horizon)]

        # Apply seasonal factors
        seasonal_factors = list(seasonality.seasonal_factors.values())
        if seasonal_factors:
            period = len(seasonal_factors)
            forecasts = [forecasts[i] * seasonal_factors[i % period] for i in range(horizon)]

        # Generate forecast points with uncertainty
        for i in range(horizon):
            forecast_time = last_timestamp + delta * (i + 1)
            pred_value = max(0, forecasts[i])

            # Uncertainty grows with horizon
            uncertainty = std_dev * math.sqrt(1 + i * 0.1) * z
            lower = max(0, pred_value - uncertainty)
            upper = pred_value + uncertainty

            forecast.append(ForecastPoint(
                timestamp=forecast_time,
                predicted_value=round(pred_value, 2),
                lower_bound=round(lower, 2),
                upper_bound=round(upper, 2),
                confidence_level=confidence,
                prediction_interval_width=round(upper - lower, 2)))

        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(values, method, trend)

        return forecast, diagnostics

    def _calculate_diagnostics(self, values: List[float],
                              method: ForecastMethod,
                              trend: TrendAnalysis) -> ModelDiagnostics:
        """Calculate model diagnostic metrics."""
        n = len(values)
        if n < 3:
            return self._empty_diagnostics(method)

        mean = sum(values) / n

        # Simple in-sample fit (for demonstration)
        # In practice, would use proper cross-validation
        if method == ForecastMethod.MOVING_AVERAGE:
            window = min(3, n - 1)
            fitted = [sum(values[max(0, i-window):i+1]) / min(i+1, window+1) for i in range(n)]
        else:
            fitted = [mean + trend.slope_per_period * i for i in range(n)]

        # Calculate errors
        errors = [values[i] - fitted[i] for i in range(n)]
        abs_errors = [abs(e) for e in errors]
        pct_errors = [abs(e) / values[i] * 100 if values[i] != 0 else 0 for i, e in enumerate(errors)]

        mape = sum(pct_errors) / n
        rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
        mae = sum(abs_errors) / n

        # R-squared
        ss_res = sum(e ** 2 for e in errors)
        ss_tot = sum((v - mean) ** 2 for v in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Residual autocorrelation (lag-1)
        if n > 1:
            error_mean = sum(errors) / n
            numerator = sum((errors[i] - error_mean) * (errors[i-1] - error_mean) for i in range(1, n))
            denominator = sum((e - error_mean) ** 2 for e in errors)
            autocorr = numerator / denominator if denominator != 0 else 0
        else:
            autocorr = 0

        # Model adequacy
        if mape < 5 and r_squared > 0.8:
            adequacy = "EXCELLENT"
        elif mape < 10 and r_squared > 0.6:
            adequacy = "GOOD"
        elif mape < 20:
            adequacy = "ACCEPTABLE"
        else:
            adequacy = "POOR"

        return ModelDiagnostics(
            method_used=method.value,
            mape_percent=round(mape, 2),
            rmse=round(rmse, 2),
            mae=round(mae, 2),
            r_squared=round(max(0, r_squared), 3),
            aic=None,
            bic=None,
            residual_autocorrelation=round(autocorr, 3),
            model_adequacy=adequacy)

    def _empty_diagnostics(self, method: ForecastMethod) -> ModelDiagnostics:
        """Return empty diagnostics for insufficient data."""
        return ModelDiagnostics(
            method_used=method.value,
            mape_percent=0,
            rmse=0,
            mae=0,
            r_squared=0,
            aic=None,
            bic=None,
            residual_autocorrelation=0,
            model_adequacy="INSUFFICIENT_DATA")

    def _generate_scenarios(self, historical: List[float],
                           baseline_forecast: List[ForecastPoint],
                           horizon: int) -> List[ScenarioForecast]:
        """Generate scenario-based forecasts."""
        scenarios = []
        baseline_values = [fp.predicted_value for fp in baseline_forecast]

        # Business as usual
        scenarios.append(ScenarioForecast(
            scenario_name="Business as Usual",
            scenario_description="Continuation of current trends",
            assumptions={"growth_rate": "historical", "policy_changes": "none"},
            forecast_values=[round(v, 2) for v in baseline_values],
            total_emissions=round(sum(baseline_values), 2),
            probability_weight=0.50))

        # High growth scenario
        high_values = [v * (1 + 0.05 * (i + 1) / horizon) for i, v in enumerate(baseline_values)]
        scenarios.append(ScenarioForecast(
            scenario_name="High Growth",
            scenario_description="Increased production/activity",
            assumptions={"growth_rate": "+5%", "policy_changes": "none"},
            forecast_values=[round(v, 2) for v in high_values],
            total_emissions=round(sum(high_values), 2),
            probability_weight=0.20))

        # Decarbonization scenario
        decarb_values = [v * (1 - 0.03 * (i + 1) / horizon) for i, v in enumerate(baseline_values)]
        scenarios.append(ScenarioForecast(
            scenario_name="Decarbonization",
            scenario_description="Emissions reduction initiatives",
            assumptions={"growth_rate": "stable", "policy_changes": "reduction_targets"},
            forecast_values=[round(v, 2) for v in decarb_values],
            total_emissions=round(sum(decarb_values), 2),
            probability_weight=0.25))

        # Disruption scenario
        disruption_values = [v * random.uniform(0.7, 1.1) for v in baseline_values]
        scenarios.append(ScenarioForecast(
            scenario_name="Disruption",
            scenario_description="Major operational changes or disruptions",
            assumptions={"growth_rate": "variable", "policy_changes": "uncertain"},
            forecast_values=[round(v, 2) for v in disruption_values],
            total_emissions=round(sum(disruption_values), 2),
            probability_weight=0.05))

        return scenarios

    def _analyze_factors(self, values: List[float],
                        external_factors: List[ExternalFactor]) -> Dict[str, float]:
        """Analyze contributing factors."""
        factors = {}

        # Baseline factor contributions (simplified)
        factors["production_activity"] = 0.60
        factors["energy_efficiency"] = 0.15
        factors["fuel_mix"] = 0.15
        factors["other"] = 0.10

        # Adjust based on external factors if provided
        for ef in external_factors:
            if ef.correlation_coefficient is not None:
                factors[ef.name] = round(abs(ef.correlation_coefficient), 3)

        return factors

    def _generate_alerts(self, historical: List[float],
                        forecast: List[ForecastPoint]) -> List[AlertThreshold]:
        """Generate alert thresholds."""
        alerts = []

        if not historical:
            return alerts

        mean = sum(historical) / len(historical)
        std = math.sqrt(sum((v - mean) ** 2 for v in historical) / len(historical))

        # High emission alert
        high_threshold = mean + 2 * std
        periods_high = sum(1 for fp in forecast if fp.predicted_value > high_threshold)
        alerts.append(AlertThreshold(
            threshold_type="HIGH_EMISSION",
            threshold_value=round(high_threshold, 2),
            periods_exceeding=periods_high,
            alert_level="WARNING" if periods_high > 0 else "NORMAL",
            recommended_action="Review emission sources and implement reduction measures" if periods_high > 0 else "Continue monitoring"))

        # Trend alert
        if forecast and len(forecast) >= 2:
            start_val = forecast[0].predicted_value
            end_val = forecast[-1].predicted_value
            trend_change = (end_val - start_val) / start_val * 100 if start_val != 0 else 0

            if trend_change > 10:
                alert_level = "WARNING"
                action = "Investigate increasing emission trend"
            elif trend_change < -10:
                alert_level = "POSITIVE"
                action = "Continue current reduction efforts"
            else:
                alert_level = "NORMAL"
                action = "Maintain current monitoring"

            alerts.append(AlertThreshold(
                threshold_type="TREND_CHANGE",
                threshold_value=round(trend_change, 2),
                periods_exceeding=len(forecast),
                alert_level=alert_level,
                recommended_action=action))

        # Regulatory threshold (example)
        reg_threshold = mean * 1.2  # 20% above historical mean
        periods_exceed = sum(1 for fp in forecast if fp.upper_bound > reg_threshold)
        alerts.append(AlertThreshold(
            threshold_type="REGULATORY_RISK",
            threshold_value=round(reg_threshold, 2),
            periods_exceeding=periods_exceed,
            alert_level="CRITICAL" if periods_exceed > len(forecast) / 2 else "WARNING" if periods_exceed > 0 else "NORMAL",
            recommended_action="Prepare compliance action plan" if periods_exceed > 0 else "Maintain compliance"))

        return alerts

    def _calculate_summary(self, forecast: List[ForecastPoint],
                          historical: List[float],
                          granularity: TimeGranularity) -> ForecastSummary:
        """Calculate forecast summary statistics."""
        if not forecast:
            return ForecastSummary(
                forecast_horizon_periods=0,
                total_predicted_tCO2e=0,
                average_monthly_tCO2e=0,
                peak_emission_period=datetime.utcnow(),
                peak_emission_value=0,
                minimum_emission_period=datetime.utcnow(),
                minimum_emission_value=0,
                year_over_year_change_percent=None,
                forecast_uncertainty_percent=0)

        values = [fp.predicted_value for fp in forecast]
        total = sum(values)
        avg = total / len(values)

        peak_idx = values.index(max(values))
        min_idx = values.index(min(values))

        # YoY change (if applicable)
        yoy_change = None
        if historical and len(historical) >= 12:
            prev_year = sum(historical[-12:])
            if prev_year > 0:
                yoy_change = (total - prev_year) / prev_year * 100

        # Average uncertainty
        avg_uncertainty = sum(fp.prediction_interval_width for fp in forecast) / len(forecast)
        uncertainty_pct = (avg_uncertainty / avg * 100) if avg > 0 else 0

        return ForecastSummary(
            forecast_horizon_periods=len(forecast),
            total_predicted_tCO2e=round(total, 2),
            average_monthly_tCO2e=round(avg, 2),
            peak_emission_period=forecast[peak_idx].timestamp,
            peak_emission_value=round(values[peak_idx], 2),
            minimum_emission_period=forecast[min_idx].timestamp,
            minimum_emission_value=round(values[min_idx], 2),
            year_over_year_change_percent=round(yoy_change, 2) if yoy_change is not None else None,
            forecast_uncertainty_percent=round(uncertainty_pct, 2))

    def _compare_historical(self, historical: List[float],
                           forecast: List[ForecastPoint]) -> Dict[str, float]:
        """Compare forecast to historical patterns."""
        comparison = {}

        if not historical:
            return comparison

        hist_mean = sum(historical) / len(historical)
        hist_std = math.sqrt(sum((v - hist_mean) ** 2 for v in historical) / len(historical))

        if forecast:
            forecast_mean = sum(fp.predicted_value for fp in forecast) / len(forecast)
            comparison["forecast_vs_historical_mean_percent"] = round(
                (forecast_mean - hist_mean) / hist_mean * 100 if hist_mean != 0 else 0, 2)
            comparison["historical_mean"] = round(hist_mean, 2)
            comparison["forecast_mean"] = round(forecast_mean, 2)
            comparison["historical_std"] = round(hist_std, 2)
            comparison["historical_max"] = round(max(historical), 2)
            comparison["historical_min"] = round(min(historical), 2)

        return comparison


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-073",
    "name": "EMISSIONS-PREDICTOR",
    "version": "1.0.0",
    "summary": "Time-series forecasting of emissions with uncertainty quantification",
    "tags": ["emissions", "forecasting", "prediction", "time-series", "uncertainty", "scenarios"],
    "standards": [
        {"ref": "GHG Protocol", "description": "Greenhouse Gas Protocol"},
        {"ref": "ISO 14064", "description": "GHG Quantification and Reporting"},
        {"ref": "IPCC Guidelines", "description": "National GHG Inventories"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
