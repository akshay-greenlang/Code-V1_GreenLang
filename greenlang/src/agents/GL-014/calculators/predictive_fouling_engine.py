"""
GL-014 EXCHANGER-PRO - Predictive Fouling Engine Module

This module implements advanced predictive analytics for heat exchanger fouling
with deterministic, reproducible calculations suitable for regulatory compliance.

Key Features:
- Time series analysis with seasonal decomposition
- Statistical forecasting (ARIMA, Exponential Smoothing)
- Anomaly detection (Z-score, IQR, CUSUM, Mahalanobis)
- Pattern recognition for fouling regimes
- Threshold prediction with confidence intervals
- Multi-exchanger correlation analysis
- Model validation with backtesting

Reference Standards:
- TEMA Standards (Heat Exchanger Design)
- ASME PTC 12.5 (Heat Exchanger Performance)
- ISO 13380 (Condition Monitoring)
- Box-Jenkins ARIMA methodology

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
from enum import Enum, auto
from datetime import datetime, timezone
import math
import hashlib
import json
import uuid


# =============================================================================
# CONSTANTS
# =============================================================================

# Default precision for Decimal calculations
DEFAULT_DECIMAL_PRECISION: int = 10

# Statistical z-scores for confidence intervals
Z_SCORES: Dict[str, Decimal] = {
    "80%": Decimal("1.282"),
    "85%": Decimal("1.440"),
    "90%": Decimal("1.645"),
    "95%": Decimal("1.960"),
    "99%": Decimal("2.576"),
    "99.9%": Decimal("3.291"),
}

# CUSUM default parameters
CUSUM_K_DEFAULT: Decimal = Decimal("0.5")  # Slack value
CUSUM_H_DEFAULT: Decimal = Decimal("5.0")  # Decision interval

# Fouling resistance thresholds (m2-K/W) per TEMA
FOULING_THRESHOLDS: Dict[str, Decimal] = {
    "clean": Decimal("0.0001"),
    "light": Decimal("0.0002"),
    "moderate": Decimal("0.0003"),
    "warning": Decimal("0.0004"),
    "critical": Decimal("0.0005"),
    "cleaning_required": Decimal("0.0006"),
}


# =============================================================================
# ENUMS
# =============================================================================

class FoulingRegime(Enum):
    """Fouling regime classification based on progression pattern."""
    LINEAR = auto()         # Constant rate fouling
    ASYMPTOTIC = auto()     # Decelerating fouling approaching limit
    ACCELERATING = auto()   # Increasing rate fouling
    FALLING_RATE = auto()   # Initial rapid then slowing
    SAWTOOTH = auto()       # Periodic cleaning cycles
    STABLE = auto()         # Minimal change


class AnomalyType(Enum):
    """Types of anomalies detected in fouling data."""
    POINT_ANOMALY = auto()      # Single outlier point
    LEVEL_SHIFT = auto()        # Sudden change in mean
    TREND_SHIFT = auto()        # Change in slope
    VARIANCE_CHANGE = auto()    # Change in variability
    SEASONAL_ANOMALY = auto()   # Unexpected seasonal pattern
    PROCESS_UPSET = auto()      # Operating condition anomaly


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    INFO = auto()        # Informational, monitor
    LOW = auto()         # Minor deviation
    MEDIUM = auto()      # Significant deviation
    HIGH = auto()        # Major deviation
    CRITICAL = auto()    # Immediate action required


class TrendDirection(Enum):
    """Direction of fouling trend."""
    INCREASING = auto()
    DECREASING = auto()
    STABLE = auto()
    FLUCTUATING = auto()


class SeasonalPeriod(Enum):
    """Seasonal period types."""
    DAILY = auto()       # 24 observations
    WEEKLY = auto()      # 168 observations (hourly)
    MONTHLY = auto()     # ~730 observations (hourly)
    QUARTERLY = auto()   # ~2190 observations (hourly)
    ANNUAL = auto()      # ~8760 observations (hourly)


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class TrendAnalysisResult:
    """
    Result of fouling resistance trend analysis.

    Attributes:
        direction: Overall trend direction
        slope: Rate of change per time unit
        slope_confidence: Confidence interval for slope
        intercept: Trend line intercept
        r_squared: Coefficient of determination
        is_stationary: Whether series is stationary (ADF test)
        adf_statistic: Augmented Dickey-Fuller test statistic
        adf_p_value: ADF test p-value
        autocorrelation_lag1: First-order autocorrelation
        mean_value: Mean fouling resistance
        std_deviation: Standard deviation
        provenance_hash: SHA-256 hash for audit
    """
    direction: TrendDirection
    slope: Decimal
    slope_confidence_lower: Decimal
    slope_confidence_upper: Decimal
    intercept: Decimal
    r_squared: Decimal
    is_stationary: bool
    adf_statistic: Decimal
    adf_p_value: Decimal
    autocorrelation_lag1: Decimal
    mean_value: Decimal
    std_deviation: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.name,
            "slope": str(self.slope),
            "slope_confidence_lower": str(self.slope_confidence_lower),
            "slope_confidence_upper": str(self.slope_confidence_upper),
            "intercept": str(self.intercept),
            "r_squared": str(self.r_squared),
            "is_stationary": self.is_stationary,
            "adf_statistic": str(self.adf_statistic),
            "adf_p_value": str(self.adf_p_value),
            "autocorrelation_lag1": str(self.autocorrelation_lag1),
            "mean_value": str(self.mean_value),
            "std_deviation": str(self.std_deviation),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class SeasonalDecompositionResult:
    """
    Result of seasonal decomposition analysis.

    Attributes:
        trend_component: Extracted trend values
        seasonal_component: Extracted seasonal values
        residual_component: Residual after decomposition
        seasonal_strength: Strength of seasonality (0-1)
        trend_strength: Strength of trend (0-1)
        dominant_period: Dominant seasonal period
        seasonal_amplitude: Amplitude of seasonal variation
        provenance_hash: SHA-256 hash
    """
    trend_component: Tuple[Decimal, ...]
    seasonal_component: Tuple[Decimal, ...]
    residual_component: Tuple[Decimal, ...]
    seasonal_strength: Decimal
    trend_strength: Decimal
    dominant_period: Optional[int]
    seasonal_amplitude: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class ForecastResult:
    """
    Result of fouling progression forecast.

    Attributes:
        forecast_values: Point forecasts
        forecast_lower: Lower confidence bound
        forecast_upper: Upper confidence bound
        forecast_horizon: Number of periods forecasted
        model_type: Type of model used (ARIMA, ETS)
        model_parameters: Model parameters used
        in_sample_mape: In-sample MAPE
        in_sample_rmse: In-sample RMSE
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        provenance_hash: SHA-256 hash
    """
    forecast_values: Tuple[Decimal, ...]
    forecast_lower: Tuple[Decimal, ...]
    forecast_upper: Tuple[Decimal, ...]
    forecast_horizon: int
    model_type: str
    model_parameters: Dict[str, Any]
    confidence_level: str
    in_sample_mape: Decimal
    in_sample_rmse: Decimal
    aic: Decimal
    bic: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forecast_values": [str(v) for v in self.forecast_values],
            "forecast_lower": [str(v) for v in self.forecast_lower],
            "forecast_upper": [str(v) for v in self.forecast_upper],
            "forecast_horizon": self.forecast_horizon,
            "model_type": self.model_type,
            "model_parameters": self.model_parameters,
            "confidence_level": self.confidence_level,
            "in_sample_mape": str(self.in_sample_mape),
            "in_sample_rmse": str(self.in_sample_rmse),
            "aic": str(self.aic),
            "bic": str(self.bic),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class AnomalyDetectionResult:
    """
    Result of fouling anomaly detection.

    Attributes:
        is_anomaly: Whether an anomaly was detected
        anomaly_type: Type of anomaly
        severity: Severity level
        anomaly_score: Composite anomaly score (0-1)
        z_score: Z-score of observation
        iqr_score: IQR-based score
        cusum_value: CUSUM statistic
        mahalanobis_distance: Mahalanobis distance (multivariate)
        explanation: Human-readable explanation
        provenance_hash: SHA-256 hash
    """
    is_anomaly: bool
    anomaly_type: Optional[AnomalyType]
    severity: AnomalySeverity
    anomaly_score: Decimal
    z_score: Decimal
    iqr_score: Decimal
    cusum_value: Decimal
    mahalanobis_distance: Optional[Decimal]
    explanation: str
    contributing_factors: Tuple[str, ...]
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type.name if self.anomaly_type else None,
            "severity": self.severity.name,
            "anomaly_score": str(self.anomaly_score),
            "z_score": str(self.z_score),
            "iqr_score": str(self.iqr_score),
            "cusum_value": str(self.cusum_value),
            "mahalanobis_distance": str(self.mahalanobis_distance) if self.mahalanobis_distance else None,
            "explanation": self.explanation,
            "contributing_factors": list(self.contributing_factors),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class PatternRecognitionResult:
    """
    Result of fouling pattern recognition.

    Attributes:
        identified_regime: Identified fouling regime
        regime_confidence: Confidence in regime identification
        regime_parameters: Parameters characterizing the regime
        operating_condition_correlation: Correlation with operating conditions
        seasonal_patterns: Identified seasonal patterns
        process_upset_indicators: Indicators of process upsets
        provenance_hash: SHA-256 hash
    """
    identified_regime: FoulingRegime
    regime_confidence: Decimal
    regime_parameters: Dict[str, Decimal]
    operating_condition_correlations: Dict[str, Decimal]
    seasonal_patterns: Tuple[SeasonalPeriod, ...]
    process_upset_indicators: Tuple[str, ...]
    provenance_hash: str = ""


@dataclass(frozen=True)
class ThresholdPredictionResult:
    """
    Result of threshold breach prediction.

    Attributes:
        time_to_warning: Predicted time to warning threshold
        time_to_critical: Predicted time to critical threshold
        time_to_cleaning: Predicted time to cleaning required
        probability_breach_30d: Probability of breach in 30 days
        probability_breach_90d: Probability of breach in 90 days
        confidence_interval_lower: Lower bound of time estimate
        confidence_interval_upper: Upper bound of time estimate
        current_fouling_resistance: Current fouling value
        warning_threshold: Warning threshold value
        critical_threshold: Critical threshold value
        provenance_hash: SHA-256 hash
    """
    time_to_warning_hours: Optional[Decimal]
    time_to_critical_hours: Optional[Decimal]
    time_to_cleaning_hours: Optional[Decimal]
    probability_breach_30d: Decimal
    probability_breach_90d: Decimal
    confidence_interval_lower: Decimal
    confidence_interval_upper: Decimal
    current_fouling_resistance: Decimal
    warning_threshold: Decimal
    critical_threshold: Decimal
    cleaning_threshold: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_to_warning_hours": str(self.time_to_warning_hours) if self.time_to_warning_hours else None,
            "time_to_critical_hours": str(self.time_to_critical_hours) if self.time_to_critical_hours else None,
            "time_to_cleaning_hours": str(self.time_to_cleaning_hours) if self.time_to_cleaning_hours else None,
            "probability_breach_30d": str(self.probability_breach_30d),
            "probability_breach_90d": str(self.probability_breach_90d),
            "confidence_interval_lower": str(self.confidence_interval_lower),
            "confidence_interval_upper": str(self.confidence_interval_upper),
            "current_fouling_resistance": str(self.current_fouling_resistance),
            "warning_threshold": str(self.warning_threshold),
            "critical_threshold": str(self.critical_threshold),
            "cleaning_threshold": str(self.cleaning_threshold),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class CorrelationResult:
    """
    Result of multi-exchanger correlation analysis.

    Attributes:
        exchanger_correlations: Correlation matrix between exchangers
        network_effects: Identified network effects
        upstream_dependencies: Upstream exchanger dependencies
        downstream_impacts: Downstream exchanger impacts
        common_causes: Identified common cause factors
        fleet_patterns: Fleet-wide patterns detected
        provenance_hash: SHA-256 hash
    """
    exchanger_correlations: Dict[str, Dict[str, Decimal]]
    network_effects: Tuple[str, ...]
    upstream_dependencies: Dict[str, Tuple[str, ...]]
    downstream_impacts: Dict[str, Tuple[str, ...]]
    common_causes: Tuple[str, ...]
    fleet_patterns: Dict[str, Any]
    provenance_hash: str = ""


@dataclass(frozen=True)
class ModelValidationResult:
    """
    Result of prediction model validation.

    Attributes:
        mape: Mean Absolute Percentage Error
        rmse: Root Mean Square Error
        mae: Mean Absolute Error
        r_squared: Coefficient of determination
        bias: Systematic bias in predictions
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        cross_validation_scores: K-fold cross-validation scores
        backtest_accuracy: Accuracy on historical data
        prediction_intervals_coverage: Coverage of prediction intervals
        provenance_hash: SHA-256 hash
    """
    mape: Decimal
    rmse: Decimal
    mae: Decimal
    r_squared: Decimal
    bias: Decimal
    aic: Decimal
    bic: Decimal
    cross_validation_scores: Tuple[Decimal, ...]
    backtest_accuracy: Decimal
    prediction_intervals_coverage: Decimal
    model_selected: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mape": str(self.mape),
            "rmse": str(self.rmse),
            "mae": str(self.mae),
            "r_squared": str(self.r_squared),
            "bias": str(self.bias),
            "aic": str(self.aic),
            "bic": str(self.bic),
            "cross_validation_scores": [str(s) for s in self.cross_validation_scores],
            "backtest_accuracy": str(self.backtest_accuracy),
            "prediction_intervals_coverage": str(self.prediction_intervals_coverage),
            "model_selected": self.model_selected,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class PredictionReportResult:
    """
    Comprehensive prediction report result.

    Attributes:
        report_id: Unique report identifier
        timestamp: Report generation timestamp
        exchanger_id: Heat exchanger identifier
        trend_analysis: Trend analysis results
        forecast: Forecast results
        anomalies: Detected anomalies
        pattern: Pattern recognition results
        threshold_prediction: Threshold predictions
        validation: Model validation results
        recommendations: Generated recommendations
        provenance_hash: SHA-256 hash
    """
    report_id: str
    timestamp: str
    exchanger_id: str
    trend_analysis: TrendAnalysisResult
    forecast: ForecastResult
    anomalies: Tuple[AnomalyDetectionResult, ...]
    pattern: PatternRecognitionResult
    threshold_prediction: ThresholdPredictionResult
    validation: ModelValidationResult
    recommendations: Tuple[str, ...]
    provenance_hash: str = ""


# =============================================================================
# CALCULATION STEP TRACKING
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """
    Immutable record of a single calculation step.
    """
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_name: str
    output_value: Any
    formula: str = ""
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "inputs": self._serialize_value(self.inputs),
            "output_name": self.output_name,
            "output_value": self._serialize_value(self.output_value),
            "formula": self.formula,
            "reference": self.reference,
        }

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a value to JSON-compatible format."""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, (list, tuple)):
            return [CalculationStep._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: CalculationStep._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, Enum):
            return value.name
        return value


# =============================================================================
# PREDICTIVE FOULING ENGINE
# =============================================================================

class PredictiveFoulingEngine:
    """
    Predictive analytics engine for heat exchanger fouling.

    Implements deterministic, reproducible calculations for:
    - Time series analysis and trend detection
    - Statistical forecasting (ARIMA, Holt-Winters)
    - Anomaly detection (Z-score, IQR, CUSUM, Mahalanobis)
    - Pattern recognition for fouling regimes
    - Threshold breach prediction
    - Multi-exchanger correlation analysis
    - Model validation with backtesting

    All calculations are:
    - Deterministic (zero hallucination)
    - Fully documented with provenance
    - Auditable with SHA-256 hashes

    Reference: TEMA Standards, ASME PTC 12.5

    Example:
        >>> engine = PredictiveFoulingEngine()
        >>> data = [Decimal("0.0001"), Decimal("0.00012"), ...]
        >>> trend = engine.analyze_fouling_trend(data)
        >>> forecast = engine.forecast_fouling_progression(data, horizon=30)
        >>> print(f"Days to cleaning: {forecast.time_to_cleaning_hours / 24}")
    """

    def __init__(
        self,
        precision: int = DEFAULT_DECIMAL_PRECISION,
        confidence_level: str = "95%",
        warning_threshold: Optional[Decimal] = None,
        critical_threshold: Optional[Decimal] = None,
        cleaning_threshold: Optional[Decimal] = None,
    ):
        """
        Initialize Predictive Fouling Engine.

        Args:
            precision: Decimal precision for calculations
            confidence_level: Default confidence level for intervals
            warning_threshold: Custom warning threshold (m2-K/W)
            critical_threshold: Custom critical threshold (m2-K/W)
            cleaning_threshold: Custom cleaning threshold (m2-K/W)
        """
        self._precision = precision
        self._confidence_level = confidence_level
        self._z_score = Z_SCORES.get(confidence_level, Decimal("1.96"))

        # Thresholds
        self._warning_threshold = warning_threshold or FOULING_THRESHOLDS["warning"]
        self._critical_threshold = critical_threshold or FOULING_THRESHOLDS["critical"]
        self._cleaning_threshold = cleaning_threshold or FOULING_THRESHOLDS["cleaning_required"]

        # State for CUSUM tracking
        self._cusum_state: Dict[str, Dict[str, Decimal]] = {}

        # Calculation steps for provenance
        self._calculation_steps: List[CalculationStep] = []

    # =========================================================================
    # TIME SERIES ANALYSIS
    # =========================================================================

    def analyze_fouling_trend(
        self,
        fouling_data: Sequence[Union[Decimal, float]],
        time_intervals_hours: Optional[Sequence[Union[Decimal, float]]] = None,
    ) -> TrendAnalysisResult:
        """
        Analyze fouling resistance trend with comprehensive statistics.

        Performs:
        - Linear regression for trend slope
        - Augmented Dickey-Fuller test for stationarity
        - Autocorrelation analysis
        - Trend direction classification

        Args:
            fouling_data: Sequence of fouling resistance values (m2-K/W)
            time_intervals_hours: Optional time intervals (default: unit intervals)

        Returns:
            TrendAnalysisResult with comprehensive trend analysis

        Reference:
            Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis

        Example:
            >>> engine = PredictiveFoulingEngine()
            >>> data = [0.0001, 0.00012, 0.00015, 0.00018, 0.0002]
            >>> result = engine.analyze_fouling_trend(data)
            >>> print(f"Trend: {result.direction.name}, Slope: {result.slope}")
        """
        self._calculation_steps = []

        # Convert to Decimal
        data = [self._to_decimal(v) for v in fouling_data]
        n = len(data)

        if n < 5:
            raise ValueError("Need at least 5 data points for trend analysis")

        # Generate time series if not provided
        if time_intervals_hours:
            times = [self._to_decimal(t) for t in time_intervals_hours]
        else:
            times = [Decimal(str(i)) for i in range(n)]

        # Step 1: Calculate basic statistics
        mean_val = sum(data) / Decimal(str(n))
        variance = sum((x - mean_val) ** 2 for x in data) / Decimal(str(n - 1))
        std_dev = self._sqrt(variance)

        self._add_step(
            1, "calculate", "Calculate basic statistics",
            {"n": n, "data_range": f"{min(data)} to {max(data)}"},
            "statistics", {"mean": mean_val, "std": std_dev},
            formula="mean = sum(x) / n, std = sqrt(variance)"
        )

        # Step 2: Linear regression for trend
        mean_t = sum(times) / Decimal(str(n))

        # Covariance and variance for regression
        cov_xy = sum((times[i] - mean_t) * (data[i] - mean_val) for i in range(n))
        var_x = sum((t - mean_t) ** 2 for t in times)

        if var_x > Decimal("0"):
            slope = cov_xy / var_x
            intercept = mean_val - slope * mean_t
        else:
            slope = Decimal("0")
            intercept = mean_val

        self._add_step(
            2, "regression", "Perform linear regression",
            {"cov_xy": cov_xy, "var_x": var_x},
            "regression_coefficients", {"slope": slope, "intercept": intercept},
            formula="slope = cov(x,y) / var(x), intercept = mean_y - slope * mean_x"
        )

        # Step 3: Calculate R-squared
        ss_tot = sum((y - mean_val) ** 2 for y in data)
        predicted = [intercept + slope * t for t in times]
        ss_res = sum((data[i] - predicted[i]) ** 2 for i in range(n))

        if ss_tot > Decimal("0"):
            r_squared = Decimal("1") - (ss_res / ss_tot)
        else:
            r_squared = Decimal("0")

        self._add_step(
            3, "calculate", "Calculate R-squared",
            {"ss_tot": ss_tot, "ss_res": ss_res},
            "r_squared", r_squared,
            formula="R^2 = 1 - SS_res / SS_tot"
        )

        # Step 4: Calculate slope confidence interval
        if n > 2 and var_x > Decimal("0"):
            mse = ss_res / Decimal(str(n - 2))
            se_slope = self._sqrt(mse / var_x)
            t_critical = self._t_critical(n - 2, self._confidence_level)
            slope_ci_lower = slope - t_critical * se_slope
            slope_ci_upper = slope + t_critical * se_slope
        else:
            slope_ci_lower = slope
            slope_ci_upper = slope

        self._add_step(
            4, "calculate", "Calculate slope confidence interval",
            {"se_slope": se_slope if n > 2 else Decimal("0"), "t_critical": t_critical if n > 2 else Decimal("0")},
            "slope_ci", {"lower": slope_ci_lower, "upper": slope_ci_upper},
            formula="CI = slope +/- t_critical * SE_slope"
        )

        # Step 5: Augmented Dickey-Fuller test for stationarity
        adf_stat, adf_p = self._adf_test(data)
        is_stationary = adf_p < Decimal("0.05")

        self._add_step(
            5, "test", "Augmented Dickey-Fuller stationarity test",
            {"data_length": n},
            "adf_result", {"statistic": adf_stat, "p_value": adf_p, "stationary": is_stationary},
            formula="ADF test with lag selection",
            reference="Dickey & Fuller (1979)"
        )

        # Step 6: Autocorrelation at lag 1
        acf_1 = self._autocorrelation(data, lag=1)

        self._add_step(
            6, "calculate", "Calculate lag-1 autocorrelation",
            {"lag": 1},
            "autocorrelation", acf_1,
            formula="r_k = sum((x_t - mean)(x_{t-k} - mean)) / sum((x_t - mean)^2)"
        )

        # Step 7: Classify trend direction
        slope_threshold = std_dev / Decimal(str(n)) if n > 0 else Decimal("0")

        if abs(slope) < slope_threshold:
            direction = TrendDirection.STABLE
        elif slope > Decimal("0"):
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Check for fluctuating pattern
        sign_changes = sum(1 for i in range(1, n) if (data[i] - data[i-1]) * (data[i-1] - data[i-2] if i > 1 else data[i] - data[i-1]) < Decimal("0"))
        if sign_changes > n // 3:
            direction = TrendDirection.FLUCTUATING

        self._add_step(
            7, "classify", "Classify trend direction",
            {"slope": slope, "threshold": slope_threshold, "sign_changes": sign_changes},
            "direction", direction.name
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash()

        return TrendAnalysisResult(
            direction=direction,
            slope=self._apply_precision(slope, 8),
            slope_confidence_lower=self._apply_precision(slope_ci_lower, 8),
            slope_confidence_upper=self._apply_precision(slope_ci_upper, 8),
            intercept=self._apply_precision(intercept, 8),
            r_squared=self._apply_precision(max(Decimal("0"), r_squared), 4),
            is_stationary=is_stationary,
            adf_statistic=self._apply_precision(adf_stat, 4),
            adf_p_value=self._apply_precision(adf_p, 4),
            autocorrelation_lag1=self._apply_precision(acf_1, 4),
            mean_value=self._apply_precision(mean_val, 8),
            std_deviation=self._apply_precision(std_dev, 8),
            provenance_hash=provenance_hash,
        )

    def decompose_seasonal(
        self,
        fouling_data: Sequence[Union[Decimal, float]],
        period: int = 24,
    ) -> SeasonalDecompositionResult:
        """
        Decompose time series into trend, seasonal, and residual components.

        Uses classical additive decomposition:
            Y_t = T_t + S_t + R_t

        Args:
            fouling_data: Sequence of fouling resistance values
            period: Seasonal period (default: 24 for daily with hourly data)

        Returns:
            SeasonalDecompositionResult with decomposed components

        Reference:
            Cleveland et al. (1990). STL: A Seasonal-Trend Decomposition
        """
        self._calculation_steps = []

        data = [self._to_decimal(v) for v in fouling_data]
        n = len(data)

        if n < 2 * period:
            raise ValueError(f"Need at least {2 * period} observations for period {period}")

        # Step 1: Extract trend using centered moving average
        trend = self._moving_average(data, period)

        self._add_step(
            1, "smooth", "Extract trend with centered moving average",
            {"period": period, "n": n},
            "trend", f"Trend extracted with {period}-period MA"
        )

        # Step 2: Calculate detrended series
        detrended = []
        for i in range(n):
            if trend[i] is not None:
                detrended.append(data[i] - trend[i])
            else:
                detrended.append(None)

        # Step 3: Calculate seasonal indices
        seasonal_indices = [Decimal("0")] * period
        for p in range(period):
            values = [detrended[i] for i in range(p, n, period) if detrended[i] is not None]
            if values:
                seasonal_indices[p] = sum(values) / Decimal(str(len(values)))

        # Normalize seasonal indices
        seasonal_mean = sum(seasonal_indices) / Decimal(str(period))
        seasonal_indices = [s - seasonal_mean for s in seasonal_indices]

        self._add_step(
            2, "calculate", "Calculate seasonal indices",
            {"period": period},
            "seasonal_indices", [str(s) for s in seasonal_indices[:5]]  # First 5
        )

        # Step 4: Extend seasonal component to full length
        seasonal = [seasonal_indices[i % period] for i in range(n)]

        # Step 5: Calculate residuals
        residual = []
        for i in range(n):
            if trend[i] is not None:
                residual.append(data[i] - trend[i] - seasonal[i])
            else:
                residual.append(data[i] - seasonal[i])

        self._add_step(
            3, "calculate", "Calculate residuals",
            {"method": "additive"},
            "residual_stats", {"mean": sum(r for r in residual if r) / Decimal(str(len([r for r in residual if r])))}
        )

        # Step 6: Calculate strength of components
        # Trend strength
        var_residual = self._variance([r for r in residual if r is not None])
        var_trend_residual = self._variance([
            (trend[i] if trend[i] else Decimal("0")) + residual[i]
            for i in range(n) if residual[i] is not None
        ])

        if var_trend_residual > Decimal("0"):
            trend_strength = max(Decimal("0"), Decimal("1") - var_residual / var_trend_residual)
        else:
            trend_strength = Decimal("0")

        # Seasonal strength
        var_seasonal_residual = self._variance([
            seasonal[i] + residual[i] for i in range(n) if residual[i] is not None
        ])

        if var_seasonal_residual > Decimal("0"):
            seasonal_strength = max(Decimal("0"), Decimal("1") - var_residual / var_seasonal_residual)
        else:
            seasonal_strength = Decimal("0")

        self._add_step(
            4, "calculate", "Calculate component strengths",
            {"var_residual": var_residual},
            "strengths", {"trend": trend_strength, "seasonal": seasonal_strength},
            formula="F_T = max(0, 1 - Var(R) / Var(T+R))"
        )

        # Seasonal amplitude
        seasonal_amplitude = max(seasonal_indices) - min(seasonal_indices)

        # Dominant period detection
        dominant_period = period if seasonal_strength > Decimal("0.1") else None

        provenance_hash = self._calculate_provenance_hash()

        return SeasonalDecompositionResult(
            trend_component=tuple(t if t else Decimal("0") for t in trend),
            seasonal_component=tuple(seasonal),
            residual_component=tuple(r if r else Decimal("0") for r in residual),
            seasonal_strength=self._apply_precision(seasonal_strength, 4),
            trend_strength=self._apply_precision(trend_strength, 4),
            dominant_period=dominant_period,
            seasonal_amplitude=self._apply_precision(seasonal_amplitude, 8),
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # STATISTICAL FORECASTING
    # =========================================================================

    def forecast_fouling_progression(
        self,
        fouling_data: Sequence[Union[Decimal, float]],
        horizon: int = 30,
        model_type: str = "auto",
        confidence_level: Optional[str] = None,
    ) -> ForecastResult:
        """
        Forecast fouling progression using statistical models.

        Implements:
        - Holt-Winters exponential smoothing (additive trend)
        - Simple exponential smoothing
        - ARIMA(p,d,q) with automatic order selection

        Args:
            fouling_data: Historical fouling resistance values
            horizon: Number of periods to forecast
            model_type: "auto", "holtwinters", "ses", or "arima"
            confidence_level: Confidence level for prediction intervals

        Returns:
            ForecastResult with forecasts and confidence intervals

        Reference:
            Hyndman, R.J. & Athanasopoulos, G. (2018). Forecasting: Principles and Practice
        """
        self._calculation_steps = []

        data = [self._to_decimal(v) for v in fouling_data]
        n = len(data)
        conf_level = confidence_level or self._confidence_level
        z = Z_SCORES.get(conf_level, Decimal("1.96"))

        if n < 10:
            raise ValueError("Need at least 10 data points for forecasting")

        # Step 1: Model selection based on data characteristics
        if model_type == "auto":
            # Analyze trend to select model
            trend_result = self.analyze_fouling_trend(data)
            if trend_result.is_stationary:
                selected_model = "ses"
            elif abs(trend_result.autocorrelation_lag1) > Decimal("0.5"):
                selected_model = "arima"
            else:
                selected_model = "holtwinters"
        else:
            selected_model = model_type

        self._add_step(
            1, "select", "Select forecasting model",
            {"model_type": model_type, "n": n},
            "selected_model", selected_model
        )

        # Step 2: Fit model and generate forecasts
        if selected_model == "holtwinters":
            forecasts, lower, upper, params, fitted = self._holt_winters_forecast(data, horizon, z)
        elif selected_model == "ses":
            forecasts, lower, upper, params, fitted = self._ses_forecast(data, horizon, z)
        else:  # arima
            forecasts, lower, upper, params, fitted = self._arima_forecast(data, horizon, z)

        self._add_step(
            2, "fit", f"Fit {selected_model} model",
            {"parameters": params},
            "model_fitted", f"Model fitted with {len(params)} parameters"
        )

        # Step 3: Calculate in-sample accuracy metrics
        residuals = [data[i] - fitted[i] for i in range(len(fitted))]

        mape = self._calculate_mape(data[:len(fitted)], fitted)
        rmse = self._calculate_rmse(residuals)

        self._add_step(
            3, "evaluate", "Calculate in-sample accuracy",
            {"n_fitted": len(fitted)},
            "accuracy", {"mape": mape, "rmse": rmse}
        )

        # Step 4: Calculate information criteria
        k = len(params)  # Number of parameters
        n_fit = len(fitted)
        ss_res = sum(r ** 2 for r in residuals)

        if n_fit > 0 and ss_res > Decimal("0"):
            log_likelihood = -n_fit / Decimal("2") * (Decimal("1") + self._log(ss_res / Decimal(str(n_fit))) + self._log(Decimal("2") * Decimal(str(math.pi))))
            aic = Decimal("2") * Decimal(str(k)) - Decimal("2") * log_likelihood
            bic = Decimal(str(k)) * self._log(Decimal(str(n_fit))) - Decimal("2") * log_likelihood
        else:
            aic = Decimal("0")
            bic = Decimal("0")

        self._add_step(
            4, "calculate", "Calculate information criteria",
            {"k": k, "n": n_fit, "ss_res": ss_res},
            "criteria", {"aic": aic, "bic": bic},
            formula="AIC = 2k - 2ln(L), BIC = k*ln(n) - 2ln(L)"
        )

        provenance_hash = self._calculate_provenance_hash()

        return ForecastResult(
            forecast_values=tuple(self._apply_precision(f, 8) for f in forecasts),
            forecast_lower=tuple(self._apply_precision(l, 8) for l in lower),
            forecast_upper=tuple(self._apply_precision(u, 8) for u in upper),
            forecast_horizon=horizon,
            model_type=selected_model,
            model_parameters=params,
            confidence_level=conf_level,
            in_sample_mape=self._apply_precision(mape, 4),
            in_sample_rmse=self._apply_precision(rmse, 8),
            aic=self._apply_precision(aic, 2),
            bic=self._apply_precision(bic, 2),
            provenance_hash=provenance_hash,
        )

    def _holt_winters_forecast(
        self,
        data: List[Decimal],
        horizon: int,
        z: Decimal,
    ) -> Tuple[List[Decimal], List[Decimal], List[Decimal], Dict[str, Any], List[Decimal]]:
        """
        Holt-Winters exponential smoothing with additive trend.

        Model: Y_t = l_{t-1} + b_{t-1} + e_t
        Level: l_t = alpha * y_t + (1-alpha) * (l_{t-1} + b_{t-1})
        Trend: b_t = beta * (l_t - l_{t-1}) + (1-beta) * b_{t-1}
        """
        n = len(data)

        # Optimize smoothing parameters (grid search for determinism)
        best_alpha = Decimal("0.3")
        best_beta = Decimal("0.1")
        best_sse = Decimal("Infinity")

        for alpha_int in range(1, 10):
            for beta_int in range(1, 10):
                alpha = Decimal(str(alpha_int)) / Decimal("10")
                beta = Decimal(str(beta_int)) / Decimal("10")

                # Initialize
                level = data[0]
                trend = (data[min(3, n-1)] - data[0]) / Decimal(str(min(3, n-1))) if n > 1 else Decimal("0")

                sse = Decimal("0")
                for i in range(1, n):
                    forecast = level + trend
                    error = data[i] - forecast
                    sse += error ** 2

                    level_new = alpha * data[i] + (Decimal("1") - alpha) * (level + trend)
                    trend = beta * (level_new - level) + (Decimal("1") - beta) * trend
                    level = level_new

                if sse < best_sse:
                    best_sse = sse
                    best_alpha = alpha
                    best_beta = beta

        # Refit with best parameters
        level = data[0]
        trend = (data[min(3, n-1)] - data[0]) / Decimal(str(min(3, n-1))) if n > 1 else Decimal("0")

        fitted = [level]
        for i in range(1, n):
            forecast = level + trend
            fitted.append(forecast)

            level_new = best_alpha * data[i] + (Decimal("1") - best_alpha) * (level + trend)
            trend = best_beta * (level_new - level) + (Decimal("1") - best_beta) * trend
            level = level_new

        # Generate forecasts
        forecasts = []
        for h in range(1, horizon + 1):
            forecasts.append(level + Decimal(str(h)) * trend)

        # Calculate prediction intervals
        residuals = [data[i] - fitted[i] for i in range(n)]
        sigma = self._sqrt(sum(r ** 2 for r in residuals) / Decimal(str(n - 2)))

        lower = []
        upper = []
        for h in range(1, horizon + 1):
            # Variance grows with horizon
            h_sigma = sigma * self._sqrt(Decimal(str(h)))
            lower.append(forecasts[h-1] - z * h_sigma)
            upper.append(forecasts[h-1] + z * h_sigma)

        params = {"alpha": float(best_alpha), "beta": float(best_beta), "method": "Holt-Winters"}

        return forecasts, lower, upper, params, fitted

    def _ses_forecast(
        self,
        data: List[Decimal],
        horizon: int,
        z: Decimal,
    ) -> Tuple[List[Decimal], List[Decimal], List[Decimal], Dict[str, Any], List[Decimal]]:
        """
        Simple exponential smoothing.

        Model: Y_t = alpha * y_{t-1} + (1-alpha) * Y_{t-1}
        """
        n = len(data)

        # Optimize alpha
        best_alpha = Decimal("0.3")
        best_sse = Decimal("Infinity")

        for alpha_int in range(1, 10):
            alpha = Decimal(str(alpha_int)) / Decimal("10")

            level = data[0]
            sse = Decimal("0")

            for i in range(1, n):
                forecast = level
                error = data[i] - forecast
                sse += error ** 2
                level = alpha * data[i] + (Decimal("1") - alpha) * level

            if sse < best_sse:
                best_sse = sse
                best_alpha = alpha

        # Refit
        level = data[0]
        fitted = [level]

        for i in range(1, n):
            fitted.append(level)
            level = best_alpha * data[i] + (Decimal("1") - best_alpha) * level

        # Forecast is constant
        forecasts = [level] * horizon

        # Prediction intervals
        residuals = [data[i] - fitted[i] for i in range(n)]
        sigma = self._sqrt(sum(r ** 2 for r in residuals) / Decimal(str(n - 1)))

        lower = [level - z * sigma * self._sqrt(Decimal(str(h))) for h in range(1, horizon + 1)]
        upper = [level + z * sigma * self._sqrt(Decimal(str(h))) for h in range(1, horizon + 1)]

        params = {"alpha": float(best_alpha), "method": "SES"}

        return forecasts, lower, upper, params, fitted

    def _arima_forecast(
        self,
        data: List[Decimal],
        horizon: int,
        z: Decimal,
    ) -> Tuple[List[Decimal], List[Decimal], List[Decimal], Dict[str, Any], List[Decimal]]:
        """
        Simplified ARIMA(1,1,0) - random walk with drift.

        Deterministic implementation suitable for regulatory compliance.
        """
        n = len(data)

        # Difference the series
        diff = [data[i] - data[i-1] for i in range(1, n)]

        # Calculate drift (mean of differences)
        drift = sum(diff) / Decimal(str(len(diff)))

        # AR(1) coefficient estimation
        if len(diff) > 2:
            mean_diff = drift
            numerator = sum((diff[i] - mean_diff) * (diff[i-1] - mean_diff) for i in range(1, len(diff)))
            denominator = sum((diff[i] - mean_diff) ** 2 for i in range(len(diff)))
            phi = numerator / denominator if denominator > Decimal("0") else Decimal("0")
            phi = max(Decimal("-0.99"), min(Decimal("0.99"), phi))  # Ensure stationarity
        else:
            phi = Decimal("0")

        # Fitted values
        fitted = [data[0]]
        for i in range(1, n):
            pred = data[i-1] + drift
            if i > 1:
                pred += phi * (data[i-1] - data[i-2] - drift)
            fitted.append(pred)

        # Forecasts
        forecasts = []
        last_val = data[-1]
        last_diff = data[-1] - data[-2] if n > 1 else Decimal("0")

        for h in range(horizon):
            if h == 0:
                forecast = last_val + drift + phi * (last_diff - drift)
            else:
                forecast = forecasts[-1] + drift
            forecasts.append(forecast)

        # Prediction intervals
        residuals = [data[i] - fitted[i] for i in range(n)]
        sigma = self._sqrt(sum(r ** 2 for r in residuals) / Decimal(str(n - 2))) if n > 2 else Decimal("0")

        lower = [forecasts[h] - z * sigma * self._sqrt(Decimal(str(h + 1))) for h in range(horizon)]
        upper = [forecasts[h] + z * sigma * self._sqrt(Decimal(str(h + 1))) for h in range(horizon)]

        params = {"phi": float(phi), "drift": float(drift), "method": "ARIMA(1,1,0)"}

        return forecasts, lower, upper, params, fitted

    # =========================================================================
    # ANOMALY DETECTION
    # =========================================================================

    def detect_fouling_anomalies(
        self,
        current_value: Union[Decimal, float],
        historical_data: Sequence[Union[Decimal, float]],
        exchanger_id: Optional[str] = None,
        operating_conditions: Optional[Dict[str, Union[Decimal, float]]] = None,
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies in fouling data using multiple methods.

        Implements:
        - Z-score based detection
        - IQR (Interquartile Range) method
        - CUSUM for change detection
        - Mahalanobis distance (if operating conditions provided)

        Args:
            current_value: Current fouling resistance value
            historical_data: Historical fouling resistance values
            exchanger_id: Identifier for CUSUM state tracking
            operating_conditions: Optional dict of operating parameters

        Returns:
            AnomalyDetectionResult with composite anomaly assessment

        Reference:
            ISO 13380:2018 Condition Monitoring
        """
        self._calculation_steps = []

        x = self._to_decimal(current_value)
        data = [self._to_decimal(v) for v in historical_data]
        n = len(data)

        if n < 10:
            raise ValueError("Need at least 10 historical data points")

        # Step 1: Z-score calculation
        mean_val = sum(data) / Decimal(str(n))
        std_dev = self._sqrt(sum((v - mean_val) ** 2 for v in data) / Decimal(str(n - 1)))

        if std_dev > Decimal("0"):
            z_score = (x - mean_val) / std_dev
        else:
            z_score = Decimal("0")

        self._add_step(
            1, "calculate", "Calculate Z-score",
            {"value": x, "mean": mean_val, "std": std_dev},
            "z_score", z_score,
            formula="z = (x - mean) / std"
        )

        # Step 2: IQR-based detection
        sorted_data = sorted(data)
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        lower_fence = q1 - Decimal("1.5") * iqr
        upper_fence = q3 + Decimal("1.5") * iqr

        if iqr > Decimal("0"):
            iqr_score = max(
                (lower_fence - x) / iqr if x < lower_fence else Decimal("0"),
                (x - upper_fence) / iqr if x > upper_fence else Decimal("0")
            )
        else:
            iqr_score = Decimal("0")

        self._add_step(
            2, "calculate", "Calculate IQR score",
            {"q1": q1, "q3": q3, "iqr": iqr},
            "iqr_score", iqr_score,
            formula="IQR = Q3 - Q1, fences at Q1/Q3 +/- 1.5*IQR"
        )

        # Step 3: CUSUM for trend shift detection
        ex_id = exchanger_id or "default"
        if ex_id not in self._cusum_state:
            self._cusum_state[ex_id] = {
                "c_plus": Decimal("0"),
                "c_minus": Decimal("0"),
                "target": mean_val,
            }

        state = self._cusum_state[ex_id]
        k = CUSUM_K_DEFAULT * std_dev
        h = CUSUM_H_DEFAULT * std_dev

        state["c_plus"] = max(Decimal("0"), state["c_plus"] + x - state["target"] - k)
        state["c_minus"] = max(Decimal("0"), state["c_minus"] - x + state["target"] - k)
        cusum_value = max(state["c_plus"], state["c_minus"])

        self._add_step(
            3, "update", "Update CUSUM statistics",
            {"c_plus": state["c_plus"], "c_minus": state["c_minus"], "h": h},
            "cusum_value", cusum_value,
            formula="C^+ = max(0, C^+ + x - target - k)",
            reference="Page (1954)"
        )

        # Step 4: Mahalanobis distance (if multivariate)
        mahalanobis_dist = None
        if operating_conditions and len(operating_conditions) >= 2:
            mahalanobis_dist = self._calculate_mahalanobis(x, operating_conditions, data)

            self._add_step(
                4, "calculate", "Calculate Mahalanobis distance",
                {"num_variables": len(operating_conditions) + 1},
                "mahalanobis", mahalanobis_dist
            )

        # Step 5: Composite anomaly score and classification
        z_anomaly = abs(z_score) > Decimal("3")
        iqr_anomaly = iqr_score > Decimal("1")
        cusum_anomaly = cusum_value > h

        # Weighted composite score
        anomaly_score = (
            Decimal("0.4") * min(abs(z_score) / Decimal("5"), Decimal("1")) +
            Decimal("0.3") * min(iqr_score / Decimal("3"), Decimal("1")) +
            Decimal("0.3") * min(cusum_value / (h * Decimal("2")), Decimal("1"))
        )

        is_anomaly = z_anomaly or iqr_anomaly or cusum_anomaly

        # Determine anomaly type
        if is_anomaly:
            if cusum_anomaly and not z_anomaly:
                anomaly_type = AnomalyType.TREND_SHIFT
            elif z_anomaly and abs(z_score) > Decimal("5"):
                anomaly_type = AnomalyType.POINT_ANOMALY
            elif iqr_anomaly:
                anomaly_type = AnomalyType.LEVEL_SHIFT
            else:
                anomaly_type = AnomalyType.POINT_ANOMALY
        else:
            anomaly_type = None

        # Determine severity
        if abs(z_score) > Decimal("5") or cusum_value > Decimal("2") * h:
            severity = AnomalySeverity.CRITICAL
        elif abs(z_score) > Decimal("4") or cusum_value > Decimal("1.5") * h:
            severity = AnomalySeverity.HIGH
        elif is_anomaly:
            severity = AnomalySeverity.MEDIUM
        elif anomaly_score > Decimal("0.3"):
            severity = AnomalySeverity.LOW
        else:
            severity = AnomalySeverity.INFO

        # Generate explanation
        factors = []
        if z_anomaly:
            direction = "above" if z_score > Decimal("0") else "below"
            factors.append(f"Z-score {z_score:.2f} ({direction} normal range)")
        if iqr_anomaly:
            factors.append(f"Outside IQR fences (score: {iqr_score:.2f})")
        if cusum_anomaly:
            factors.append(f"CUSUM threshold exceeded ({cusum_value:.4f} > {h:.4f})")

        if is_anomaly:
            explanation = f"Anomaly detected: {'; '.join(factors)}"
        else:
            explanation = f"Value within normal range (Z-score: {z_score:.2f})"

        self._add_step(
            5, "classify", "Classify anomaly",
            {"z_anomaly": z_anomaly, "iqr_anomaly": iqr_anomaly, "cusum_anomaly": cusum_anomaly},
            "classification", {"type": anomaly_type.name if anomaly_type else None, "severity": severity.name}
        )

        provenance_hash = self._calculate_provenance_hash()

        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            anomaly_score=self._apply_precision(anomaly_score, 4),
            z_score=self._apply_precision(z_score, 4),
            iqr_score=self._apply_precision(iqr_score, 4),
            cusum_value=self._apply_precision(cusum_value, 6),
            mahalanobis_distance=self._apply_precision(mahalanobis_dist, 4) if mahalanobis_dist else None,
            explanation=explanation,
            contributing_factors=tuple(factors),
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # PATTERN RECOGNITION
    # =========================================================================

    def identify_fouling_patterns(
        self,
        fouling_data: Sequence[Union[Decimal, float]],
        operating_conditions: Optional[Dict[str, Sequence[Union[Decimal, float]]]] = None,
    ) -> PatternRecognitionResult:
        """
        Identify fouling regime and patterns from historical data.

        Classifies fouling into regimes:
        - Linear: Constant rate
        - Asymptotic: Approaching limit
        - Accelerating: Increasing rate
        - Falling rate: Initial rapid then slowing
        - Sawtooth: Periodic cleaning
        - Stable: Minimal change

        Args:
            fouling_data: Historical fouling resistance values
            operating_conditions: Optional dict of operating parameters over time

        Returns:
            PatternRecognitionResult with identified patterns
        """
        self._calculation_steps = []

        data = [self._to_decimal(v) for v in fouling_data]
        n = len(data)

        if n < 20:
            raise ValueError("Need at least 20 data points for pattern recognition")

        # Step 1: Analyze trend characteristics
        trend = self.analyze_fouling_trend(data)

        # Step 2: Calculate rate of change over segments
        segment_size = max(5, n // 5)
        segment_slopes = []

        for i in range(0, n - segment_size + 1, segment_size):
            segment = data[i:i + segment_size]
            segment_trend = self.analyze_fouling_trend(segment)
            segment_slopes.append(segment_trend.slope)

        self._add_step(
            1, "analyze", "Calculate segment slopes",
            {"num_segments": len(segment_slopes), "segment_size": segment_size},
            "segment_slopes", [str(s) for s in segment_slopes]
        )

        # Step 3: Classify regime based on slope pattern
        if len(segment_slopes) >= 2:
            slope_changes = [segment_slopes[i+1] - segment_slopes[i] for i in range(len(segment_slopes) - 1)]
            mean_change = sum(slope_changes) / Decimal(str(len(slope_changes)))

            # Check for sawtooth pattern (sign reversals in data)
            reversals = sum(1 for i in range(1, n-1)
                          if (data[i] - data[i-1]) * (data[i+1] - data[i]) < Decimal("0"))
            reversal_rate = Decimal(str(reversals)) / Decimal(str(n))

            if reversal_rate > Decimal("0.3"):
                regime = FoulingRegime.SAWTOOTH
            elif abs(trend.slope) < trend.std_deviation / Decimal(str(n)):
                regime = FoulingRegime.STABLE
            elif mean_change > Decimal("0") and trend.slope > Decimal("0"):
                regime = FoulingRegime.ACCELERATING
            elif mean_change < Decimal("0") and trend.slope > Decimal("0"):
                if segment_slopes[-1] < segment_slopes[0] / Decimal("2"):
                    regime = FoulingRegime.ASYMPTOTIC
                else:
                    regime = FoulingRegime.FALLING_RATE
            else:
                regime = FoulingRegime.LINEAR
        else:
            regime = FoulingRegime.LINEAR

        self._add_step(
            2, "classify", "Classify fouling regime",
            {"mean_slope_change": mean_change if len(segment_slopes) >= 2 else Decimal("0")},
            "regime", regime.name
        )

        # Step 4: Calculate regime confidence
        if regime == FoulingRegime.LINEAR:
            confidence = min(Decimal("1"), trend.r_squared + Decimal("0.2"))
        elif regime == FoulingRegime.STABLE:
            confidence = Decimal("1") - min(Decimal("1"), abs(trend.slope) / trend.std_deviation * Decimal(str(n)))
        elif regime == FoulingRegime.SAWTOOTH:
            confidence = min(Decimal("1"), reversal_rate * Decimal("3"))
        else:
            # For other regimes, base on consistency of slope changes
            slope_change_variance = self._variance(slope_changes) if len(slope_changes) > 1 else Decimal("0")
            confidence = Decimal("1") - min(Decimal("1"), self._sqrt(slope_change_variance) / (abs(mean_change) + Decimal("0.001")))

        confidence = max(Decimal("0.1"), min(Decimal("0.99"), confidence))

        self._add_step(
            3, "calculate", "Calculate regime confidence",
            {"regime": regime.name},
            "confidence", confidence
        )

        # Step 5: Calculate operating condition correlations
        correlations = {}
        if operating_conditions:
            for param, values in operating_conditions.items():
                if len(values) == n:
                    param_data = [self._to_decimal(v) for v in values]
                    corr = self._pearson_correlation(data, param_data)
                    correlations[param] = self._apply_precision(corr, 4)

        self._add_step(
            4, "correlate", "Calculate operating condition correlations",
            {"num_conditions": len(correlations)},
            "correlations", {k: str(v) for k, v in correlations.items()}
        )

        # Step 6: Detect seasonal patterns
        seasonal_patterns = []
        decomp = self.decompose_seasonal(data, period=24)
        if decomp.seasonal_strength > Decimal("0.2"):
            seasonal_patterns.append(SeasonalPeriod.DAILY)

        if n >= 168:  # Weekly
            decomp_weekly = self.decompose_seasonal(data, period=168)
            if decomp_weekly.seasonal_strength > Decimal("0.2"):
                seasonal_patterns.append(SeasonalPeriod.WEEKLY)

        # Step 7: Identify process upset indicators
        upset_indicators = []
        anomaly_count = 0
        for i in range(min(10, n)):
            if i > 0:
                result = self.detect_fouling_anomalies(data[-(i+1)], data[:-i-1])
                if result.is_anomaly:
                    anomaly_count += 1

        if anomaly_count > 3:
            upset_indicators.append("Recent anomaly cluster detected")
        if trend.direction == TrendDirection.INCREASING and trend.slope > Decimal("3") * trend.std_deviation / Decimal(str(n)):
            upset_indicators.append("Rapid fouling acceleration")

        # Build regime parameters
        regime_params = {
            "mean_slope": trend.slope,
            "slope_variance": self._variance(segment_slopes) if segment_slopes else Decimal("0"),
            "reversal_rate": reversal_rate if 'reversal_rate' in dir() else Decimal("0"),
        }

        provenance_hash = self._calculate_provenance_hash()

        return PatternRecognitionResult(
            identified_regime=regime,
            regime_confidence=self._apply_precision(confidence, 4),
            regime_parameters={k: self._apply_precision(v, 6) for k, v in regime_params.items()},
            operating_condition_correlations=correlations,
            seasonal_patterns=tuple(seasonal_patterns),
            process_upset_indicators=tuple(upset_indicators),
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # THRESHOLD PREDICTION
    # =========================================================================

    def predict_threshold_breach(
        self,
        fouling_data: Sequence[Union[Decimal, float]],
        forecast_horizon_hours: int = 720,  # 30 days default
        warning_threshold: Optional[Decimal] = None,
        critical_threshold: Optional[Decimal] = None,
        cleaning_threshold: Optional[Decimal] = None,
    ) -> ThresholdPredictionResult:
        """
        Predict time to reach various fouling thresholds.

        Uses forecasting model to estimate:
        - Time to warning threshold
        - Time to critical threshold
        - Time to cleaning requirement
        - Probability distributions

        Args:
            fouling_data: Historical fouling resistance values
            forecast_horizon_hours: Maximum forecast horizon
            warning_threshold: Warning level (default: 0.0004 m2-K/W)
            critical_threshold: Critical level (default: 0.0005 m2-K/W)
            cleaning_threshold: Cleaning required (default: 0.0006 m2-K/W)

        Returns:
            ThresholdPredictionResult with time predictions
        """
        self._calculation_steps = []

        data = [self._to_decimal(v) for v in fouling_data]
        current_value = data[-1]

        warn_thresh = warning_threshold or self._warning_threshold
        crit_thresh = critical_threshold or self._critical_threshold
        clean_thresh = cleaning_threshold or self._cleaning_threshold

        # Step 1: Generate forecast
        forecast = self.forecast_fouling_progression(data, horizon=forecast_horizon_hours)

        self._add_step(
            1, "forecast", "Generate fouling forecast",
            {"horizon": forecast_horizon_hours, "model": forecast.model_type},
            "forecast_generated", True
        )

        # Step 2: Find time to each threshold
        time_to_warning = None
        time_to_critical = None
        time_to_cleaning = None

        for h, value in enumerate(forecast.forecast_values):
            if time_to_warning is None and value >= warn_thresh:
                time_to_warning = Decimal(str(h + 1))
            if time_to_critical is None and value >= crit_thresh:
                time_to_critical = Decimal(str(h + 1))
            if time_to_cleaning is None and value >= clean_thresh:
                time_to_cleaning = Decimal(str(h + 1))

        self._add_step(
            2, "find", "Find threshold crossing times",
            {"warn": warn_thresh, "crit": crit_thresh, "clean": clean_thresh},
            "crossing_times", {
                "warning": str(time_to_warning) if time_to_warning else "Not reached",
                "critical": str(time_to_critical) if time_to_critical else "Not reached",
                "cleaning": str(time_to_cleaning) if time_to_cleaning else "Not reached"
            }
        )

        # Step 3: Calculate probability of breach within time windows
        # Based on forecast uncertainty
        prob_30d = Decimal("0")
        prob_90d = Decimal("0")

        # Check if upper confidence bound crosses threshold within period
        hours_30d = min(720, forecast_horizon_hours)
        hours_90d = min(2160, forecast_horizon_hours)

        for h in range(hours_30d):
            if h < len(forecast.forecast_upper) and forecast.forecast_upper[h] >= warn_thresh:
                # Probability increases as we approach threshold
                margin = (forecast.forecast_upper[h] - warn_thresh) / (forecast.forecast_upper[h] - forecast.forecast_lower[h])
                prob_30d = max(prob_30d, min(Decimal("1"), Decimal("0.5") + margin * Decimal("0.5")))

        for h in range(hours_90d):
            if h < len(forecast.forecast_upper) and forecast.forecast_upper[h] >= warn_thresh:
                margin = (forecast.forecast_upper[h] - warn_thresh) / (forecast.forecast_upper[h] - forecast.forecast_lower[h])
                prob_90d = max(prob_90d, min(Decimal("1"), Decimal("0.5") + margin * Decimal("0.5")))

        self._add_step(
            3, "calculate", "Calculate breach probabilities",
            {"window_30d": hours_30d, "window_90d": hours_90d},
            "probabilities", {"30d": prob_30d, "90d": prob_90d}
        )

        # Step 4: Confidence intervals for time estimates
        # Use spread between upper and lower forecast bounds
        if time_to_warning:
            ci_lower = time_to_warning * Decimal("0.7")
            ci_upper = time_to_warning * Decimal("1.3")
        elif len(forecast.forecast_values) > 0:
            # Extrapolate based on trend
            trend = self.analyze_fouling_trend(data)
            if trend.slope > Decimal("0"):
                remaining = warn_thresh - current_value
                ci_lower = remaining / (trend.slope_confidence_upper + Decimal("0.0001"))
                ci_upper = remaining / (trend.slope_confidence_lower + Decimal("0.0001"))
            else:
                ci_lower = Decimal(str(forecast_horizon_hours))
                ci_upper = Decimal(str(forecast_horizon_hours * 2))
        else:
            ci_lower = Decimal("0")
            ci_upper = Decimal(str(forecast_horizon_hours))

        ci_lower = max(Decimal("0"), ci_lower)
        ci_upper = max(ci_lower, ci_upper)

        self._add_step(
            4, "calculate", "Calculate confidence intervals",
            {},
            "confidence_interval", {"lower": ci_lower, "upper": ci_upper}
        )

        provenance_hash = self._calculate_provenance_hash()

        return ThresholdPredictionResult(
            time_to_warning_hours=time_to_warning,
            time_to_critical_hours=time_to_critical,
            time_to_cleaning_hours=time_to_cleaning,
            probability_breach_30d=self._apply_precision(prob_30d, 4),
            probability_breach_90d=self._apply_precision(prob_90d, 4),
            confidence_interval_lower=self._apply_precision(ci_lower, 0),
            confidence_interval_upper=self._apply_precision(ci_upper, 0),
            current_fouling_resistance=self._apply_precision(current_value, 8),
            warning_threshold=warn_thresh,
            critical_threshold=crit_thresh,
            cleaning_threshold=clean_thresh,
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # MULTI-EXCHANGER CORRELATION
    # =========================================================================

    def correlate_with_operating_conditions(
        self,
        exchangers_data: Dict[str, Sequence[Union[Decimal, float]]],
        network_topology: Optional[Dict[str, List[str]]] = None,
    ) -> CorrelationResult:
        """
        Analyze correlations between multiple heat exchangers.

        Identifies:
        - Cross-correlation between exchanger fouling rates
        - Network effects (upstream/downstream dependencies)
        - Common cause factors
        - Fleet-wide patterns

        Args:
            exchangers_data: Dict mapping exchanger ID to fouling data
            network_topology: Optional dict mapping exchanger to downstream exchangers

        Returns:
            CorrelationResult with correlation analysis
        """
        self._calculation_steps = []

        exchanger_ids = list(exchangers_data.keys())
        n_exchangers = len(exchanger_ids)

        if n_exchangers < 2:
            raise ValueError("Need at least 2 exchangers for correlation analysis")

        # Convert all data
        data = {
            ex_id: [self._to_decimal(v) for v in values]
            for ex_id, values in exchangers_data.items()
        }

        # Step 1: Calculate pairwise correlations
        correlations = {}
        for i, ex1 in enumerate(exchanger_ids):
            correlations[ex1] = {}
            for ex2 in exchanger_ids:
                if ex1 != ex2:
                    # Align data lengths
                    min_len = min(len(data[ex1]), len(data[ex2]))
                    corr = self._pearson_correlation(data[ex1][-min_len:], data[ex2][-min_len:])
                    correlations[ex1][ex2] = self._apply_precision(corr, 4)

        self._add_step(
            1, "correlate", "Calculate pairwise correlations",
            {"n_exchangers": n_exchangers},
            "correlations", "Matrix calculated"
        )

        # Step 2: Identify network effects
        network_effects = []
        upstream_deps = {ex: [] for ex in exchanger_ids}
        downstream_impacts = {ex: [] for ex in exchanger_ids}

        if network_topology:
            for upstream, downstreams in network_topology.items():
                for downstream in downstreams:
                    if upstream in correlations and downstream in correlations[upstream]:
                        corr = correlations[upstream][downstream]
                        if corr > Decimal("0.5"):
                            network_effects.append(f"{upstream} -> {downstream} (r={corr})")
                            upstream_deps[downstream].append(upstream)
                            downstream_impacts[upstream].append(downstream)

        self._add_step(
            2, "analyze", "Analyze network effects",
            {"topology_provided": network_topology is not None},
            "network_effects", network_effects[:5]  # First 5
        )

        # Step 3: Identify common causes
        common_causes = []

        # Check for synchronized trends
        trends = {ex: self.analyze_fouling_trend(data[ex]) for ex in exchanger_ids}

        increasing_count = sum(1 for t in trends.values() if t.direction == TrendDirection.INCREASING)
        if increasing_count > n_exchangers * 0.7:
            common_causes.append("Fleet-wide fouling increase detected")

        # Check for simultaneous anomalies
        recent_anomalies = {}
        for ex in exchanger_ids:
            if len(data[ex]) > 10:
                result = self.detect_fouling_anomalies(data[ex][-1], data[ex][:-1])
                recent_anomalies[ex] = result.is_anomaly

        anomaly_count = sum(recent_anomalies.values())
        if anomaly_count > n_exchangers * 0.5:
            common_causes.append("Simultaneous anomalies suggest common cause")

        self._add_step(
            3, "identify", "Identify common causes",
            {"increasing_exchangers": increasing_count, "anomaly_count": anomaly_count},
            "common_causes", common_causes
        )

        # Step 4: Detect fleet-wide patterns
        fleet_patterns = {
            "mean_correlation": Decimal("0"),
            "trend_alignment": Decimal("0"),
            "anomaly_synchronization": Decimal("0"),
        }

        # Average correlation
        all_corrs = []
        for ex1 in exchanger_ids:
            for ex2 in exchanger_ids:
                if ex1 != ex2 and ex2 in correlations.get(ex1, {}):
                    all_corrs.append(correlations[ex1][ex2])

        if all_corrs:
            fleet_patterns["mean_correlation"] = self._apply_precision(
                sum(all_corrs) / Decimal(str(len(all_corrs))), 4
            )

        # Trend alignment
        if trends:
            same_direction = sum(1 for t in trends.values() if t.direction == list(trends.values())[0].direction)
            fleet_patterns["trend_alignment"] = self._apply_precision(
                Decimal(str(same_direction)) / Decimal(str(len(trends))), 4
            )

        self._add_step(
            4, "summarize", "Summarize fleet patterns",
            {},
            "fleet_patterns", fleet_patterns
        )

        provenance_hash = self._calculate_provenance_hash()

        return CorrelationResult(
            exchanger_correlations=correlations,
            network_effects=tuple(network_effects),
            upstream_dependencies={k: tuple(v) for k, v in upstream_deps.items()},
            downstream_impacts={k: tuple(v) for k, v in downstream_impacts.items()},
            common_causes=tuple(common_causes),
            fleet_patterns=fleet_patterns,
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # MODEL VALIDATION
    # =========================================================================

    def validate_prediction_model(
        self,
        fouling_data: Sequence[Union[Decimal, float]],
        test_size: int = 10,
        cv_folds: int = 5,
    ) -> ModelValidationResult:
        """
        Validate prediction model using backtesting and cross-validation.

        Implements:
        - Hold-out backtesting
        - K-fold cross-validation
        - Multiple accuracy metrics (MAPE, RMSE, MAE)
        - Model selection criteria (AIC, BIC)

        Args:
            fouling_data: Historical fouling data
            test_size: Number of observations for hold-out test
            cv_folds: Number of cross-validation folds

        Returns:
            ModelValidationResult with comprehensive validation metrics
        """
        self._calculation_steps = []

        data = [self._to_decimal(v) for v in fouling_data]
        n = len(data)

        if n < test_size + 20:
            raise ValueError(f"Need at least {test_size + 20} data points for validation")

        # Step 1: Hold-out backtesting
        train_data = data[:-test_size]
        test_data = data[-test_size:]

        forecast = self.forecast_fouling_progression(train_data, horizon=test_size)
        predictions = list(forecast.forecast_values)

        self._add_step(
            1, "backtest", "Perform hold-out backtesting",
            {"train_size": len(train_data), "test_size": test_size},
            "backtest_complete", True
        )

        # Step 2: Calculate accuracy metrics
        residuals = [test_data[i] - predictions[i] for i in range(test_size)]

        mape = self._calculate_mape(test_data, predictions)
        rmse = self._calculate_rmse(residuals)
        mae = sum(abs(r) for r in residuals) / Decimal(str(test_size))
        bias = sum(residuals) / Decimal(str(test_size))

        # R-squared
        ss_res = sum(r ** 2 for r in residuals)
        mean_test = sum(test_data) / Decimal(str(test_size))
        ss_tot = sum((y - mean_test) ** 2 for y in test_data)
        r_squared = Decimal("1") - (ss_res / ss_tot) if ss_tot > Decimal("0") else Decimal("0")

        self._add_step(
            2, "evaluate", "Calculate accuracy metrics",
            {"n_test": test_size},
            "metrics", {"mape": mape, "rmse": rmse, "mae": mae}
        )

        # Step 3: Cross-validation
        cv_scores = []
        fold_size = (n - 20) // cv_folds

        for fold in range(cv_folds):
            fold_start = fold * fold_size
            fold_end = fold_start + fold_size

            cv_train = data[:fold_start] + data[fold_end:]
            cv_test = data[fold_start:fold_end]

            if len(cv_train) >= 15 and len(cv_test) > 0:
                cv_forecast = self.forecast_fouling_progression(cv_train, horizon=len(cv_test))
                cv_preds = list(cv_forecast.forecast_values)[:len(cv_test)]

                if len(cv_preds) == len(cv_test):
                    fold_mape = self._calculate_mape(cv_test, cv_preds)
                    cv_scores.append(fold_mape)

        self._add_step(
            3, "cross_validate", f"Perform {cv_folds}-fold cross-validation",
            {"fold_size": fold_size},
            "cv_scores", [str(s) for s in cv_scores]
        )

        # Step 4: Check prediction interval coverage
        in_interval = sum(
            1 for i in range(test_size)
            if forecast.forecast_lower[i] <= test_data[i] <= forecast.forecast_upper[i]
        )
        coverage = Decimal(str(in_interval)) / Decimal(str(test_size))

        self._add_step(
            4, "coverage", "Check prediction interval coverage",
            {"in_interval": in_interval, "total": test_size},
            "coverage", coverage
        )

        # Step 5: Model selection
        models_tested = ["holtwinters", "ses", "arima"]
        best_model = forecast.model_type
        best_aic = forecast.aic
        best_bic = forecast.bic

        # Test alternative models
        for model in models_tested:
            if model != forecast.model_type:
                alt_forecast = self.forecast_fouling_progression(train_data, horizon=test_size, model_type=model)
                if alt_forecast.aic < best_aic:
                    best_model = model
                    best_aic = alt_forecast.aic
                    best_bic = alt_forecast.bic

        self._add_step(
            5, "select", "Select best model",
            {"models_tested": models_tested},
            "best_model", best_model
        )

        # Calculate backtest accuracy
        correct_direction = sum(
            1 for i in range(1, test_size)
            if (predictions[i] - predictions[i-1]) * (test_data[i] - test_data[i-1]) > Decimal("0")
        )
        backtest_accuracy = Decimal(str(correct_direction)) / Decimal(str(test_size - 1)) if test_size > 1 else Decimal("0")

        provenance_hash = self._calculate_provenance_hash()

        return ModelValidationResult(
            mape=self._apply_precision(mape, 4),
            rmse=self._apply_precision(rmse, 8),
            mae=self._apply_precision(mae, 8),
            r_squared=self._apply_precision(max(Decimal("0"), r_squared), 4),
            bias=self._apply_precision(bias, 8),
            aic=self._apply_precision(best_aic, 2),
            bic=self._apply_precision(best_bic, 2),
            cross_validation_scores=tuple(self._apply_precision(s, 4) for s in cv_scores),
            backtest_accuracy=self._apply_precision(backtest_accuracy, 4),
            prediction_intervals_coverage=self._apply_precision(coverage, 4),
            model_selected=best_model,
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================

    def generate_prediction_report(
        self,
        exchanger_id: str,
        fouling_data: Sequence[Union[Decimal, float]],
        operating_conditions: Optional[Dict[str, Sequence[Union[Decimal, float]]]] = None,
        forecast_horizon: int = 720,
    ) -> PredictionReportResult:
        """
        Generate comprehensive prediction report for a heat exchanger.

        Combines all analysis methods into a single report with:
        - Trend analysis
        - Forecasting
        - Anomaly detection
        - Pattern recognition
        - Threshold predictions
        - Model validation
        - Recommendations

        Args:
            exchanger_id: Heat exchanger identifier
            fouling_data: Historical fouling resistance values
            operating_conditions: Optional operating parameters
            forecast_horizon: Forecast horizon in hours

        Returns:
            PredictionReportResult with comprehensive analysis
        """
        report_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        data = [self._to_decimal(v) for v in fouling_data]

        # Run all analyses
        trend = self.analyze_fouling_trend(data)
        forecast = self.forecast_fouling_progression(data, horizon=forecast_horizon)

        # Detect recent anomalies
        anomalies = []
        for i in range(min(5, len(data) - 10)):
            idx = -(i + 1)
            result = self.detect_fouling_anomalies(data[idx], data[:idx], exchanger_id)
            if result.is_anomaly:
                anomalies.append(result)

        pattern = self.identify_fouling_patterns(data, operating_conditions)
        threshold = self.predict_threshold_breach(data, forecast_horizon)
        validation = self.validate_prediction_model(data)

        # Generate recommendations
        recommendations = []

        if threshold.time_to_warning_hours and threshold.time_to_warning_hours < Decimal("168"):
            recommendations.append("WARNING: Fouling approaching warning threshold within 7 days")

        if threshold.time_to_cleaning_hours and threshold.time_to_cleaning_hours < Decimal("720"):
            recommendations.append("Schedule cleaning within 30 days")

        if pattern.identified_regime == FoulingRegime.ACCELERATING:
            recommendations.append("Investigate cause of accelerating fouling rate")

        if len(anomalies) > 2:
            recommendations.append("Multiple recent anomalies detected - investigate process conditions")

        if validation.mape > Decimal("10"):
            recommendations.append("Model accuracy degraded - consider retraining or alternative model")

        for condition, corr in pattern.operating_condition_correlations.items():
            if abs(corr) > Decimal("0.7"):
                recommendations.append(f"Strong correlation with {condition} (r={corr})")

        if not recommendations:
            recommendations.append("No immediate actions required - continue monitoring")

        # Calculate overall provenance hash
        combined_data = {
            "report_id": report_id,
            "exchanger_id": exchanger_id,
            "trend_hash": trend.provenance_hash,
            "forecast_hash": forecast.provenance_hash,
            "pattern_hash": pattern.provenance_hash,
            "threshold_hash": threshold.provenance_hash,
            "validation_hash": validation.provenance_hash,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(combined_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return PredictionReportResult(
            report_id=report_id,
            timestamp=timestamp,
            exchanger_id=exchanger_id,
            trend_analysis=trend,
            forecast=forecast,
            anomalies=tuple(anomalies),
            pattern=pattern,
            threshold_prediction=threshold,
            validation=validation,
            recommendations=tuple(recommendations),
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None,
    ) -> Decimal:
        """Apply precision rounding."""
        prec = precision if precision is not None else self._precision
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _sqrt(self, x: Decimal) -> Decimal:
        """Calculate square root."""
        if x < Decimal("0"):
            raise ValueError("Cannot take square root of negative number")
        if x == Decimal("0"):
            return Decimal("0")
        return Decimal(str(math.sqrt(float(x))))

    def _log(self, x: Decimal) -> Decimal:
        """Calculate natural logarithm."""
        if x <= Decimal("0"):
            raise ValueError("Cannot take log of non-positive number")
        return Decimal(str(math.log(float(x))))

    def _exp(self, x: Decimal) -> Decimal:
        """Calculate e^x."""
        return Decimal(str(math.exp(float(x))))

    def _variance(self, data: List[Decimal]) -> Decimal:
        """Calculate sample variance."""
        n = len(data)
        if n < 2:
            return Decimal("0")
        mean = sum(data) / Decimal(str(n))
        return sum((x - mean) ** 2 for x in data) / Decimal(str(n - 1))

    def _moving_average(self, data: List[Decimal], period: int) -> List[Optional[Decimal]]:
        """Calculate centered moving average."""
        n = len(data)
        result = [None] * n
        half = period // 2

        for i in range(half, n - half):
            window = data[i - half:i + half + 1]
            result[i] = sum(window) / Decimal(str(len(window)))

        return result

    def _autocorrelation(self, data: List[Decimal], lag: int = 1) -> Decimal:
        """Calculate autocorrelation at specified lag."""
        n = len(data)
        if n <= lag:
            return Decimal("0")

        mean = sum(data) / Decimal(str(n))

        numerator = sum((data[i] - mean) * (data[i - lag] - mean) for i in range(lag, n))
        denominator = sum((x - mean) ** 2 for x in data)

        if denominator == Decimal("0"):
            return Decimal("0")

        return numerator / denominator

    def _adf_test(self, data: List[Decimal]) -> Tuple[Decimal, Decimal]:
        """
        Simplified Augmented Dickey-Fuller test for stationarity.

        Returns (test_statistic, p_value).
        """
        n = len(data)
        if n < 10:
            return Decimal("0"), Decimal("1")

        # Calculate first differences
        diff = [data[i] - data[i-1] for i in range(1, n)]

        # Regress diff on lagged level
        y = diff[1:]  # diff[t]
        x = [data[i] for i in range(1, n-1)]  # level[t-1]

        # Simple OLS for coefficient
        mean_x = sum(x) / Decimal(str(len(x)))
        mean_y = sum(y) / Decimal(str(len(y)))

        cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        var_x = sum((xi - mean_x) ** 2 for xi in x)

        if var_x == Decimal("0"):
            return Decimal("-10"), Decimal("0.01")

        rho = cov_xy / var_x

        # Calculate t-statistic
        residuals = [y[i] - rho * x[i] for i in range(len(x))]
        se = self._sqrt(sum(r ** 2 for r in residuals) / Decimal(str(len(x) - 1)))
        se_rho = se / self._sqrt(var_x) if var_x > Decimal("0") else Decimal("1")

        if se_rho == Decimal("0"):
            t_stat = Decimal("-10")
        else:
            t_stat = rho / se_rho

        # Approximate p-value using critical values
        # ADF critical values: 1% = -3.43, 5% = -2.86, 10% = -2.57
        if t_stat < Decimal("-3.43"):
            p_value = Decimal("0.01")
        elif t_stat < Decimal("-2.86"):
            p_value = Decimal("0.05")
        elif t_stat < Decimal("-2.57"):
            p_value = Decimal("0.10")
        else:
            p_value = Decimal("0.50")

        return t_stat, p_value

    def _t_critical(self, df: int, confidence: str) -> Decimal:
        """Get t-distribution critical value."""
        # Simplified t-critical values
        t_table = {
            "90%": Decimal("1.645"),
            "95%": Decimal("1.96"),
            "99%": Decimal("2.576"),
        }
        return t_table.get(confidence, Decimal("1.96"))

    def _pearson_correlation(self, x: List[Decimal], y: List[Decimal]) -> Decimal:
        """Calculate Pearson correlation coefficient."""
        n = min(len(x), len(y))
        if n < 3:
            return Decimal("0")

        mean_x = sum(x[:n]) / Decimal(str(n))
        mean_y = sum(y[:n]) / Decimal(str(n))

        cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((xi - mean_x) ** 2 for xi in x[:n])
        var_y = sum((yi - mean_y) ** 2 for yi in y[:n])

        denom = self._sqrt(var_x) * self._sqrt(var_y)
        if denom == Decimal("0"):
            return Decimal("0")

        return cov_xy / denom

    def _calculate_mape(self, actual: List[Decimal], predicted: List[Decimal]) -> Decimal:
        """Calculate Mean Absolute Percentage Error."""
        n = min(len(actual), len(predicted))
        if n == 0:
            return Decimal("0")

        ape_sum = Decimal("0")
        count = 0

        for i in range(n):
            if actual[i] != Decimal("0"):
                ape = abs(actual[i] - predicted[i]) / abs(actual[i])
                ape_sum += ape
                count += 1

        if count == 0:
            return Decimal("0")

        return (ape_sum / Decimal(str(count))) * Decimal("100")

    def _calculate_rmse(self, residuals: List[Decimal]) -> Decimal:
        """Calculate Root Mean Square Error."""
        n = len(residuals)
        if n == 0:
            return Decimal("0")

        mse = sum(r ** 2 for r in residuals) / Decimal(str(n))
        return self._sqrt(mse)

    def _calculate_mahalanobis(
        self,
        fouling_value: Decimal,
        operating_conditions: Dict[str, Union[Decimal, float]],
        historical_fouling: List[Decimal],
    ) -> Decimal:
        """Calculate simplified Mahalanobis distance."""
        # Simplified univariate approximation
        n = len(historical_fouling)
        if n < 5:
            return Decimal("0")

        mean = sum(historical_fouling) / Decimal(str(n))
        std = self._sqrt(sum((x - mean) ** 2 for x in historical_fouling) / Decimal(str(n - 1)))

        if std == Decimal("0"):
            return Decimal("0")

        return abs(fouling_value - mean) / std

    def _add_step(
        self,
        step_number: int,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: Any,
        formula: str = "",
        reference: str = "",
    ) -> None:
        """Add a calculation step for provenance tracking."""
        step = CalculationStep(
            step_number=step_number,
            operation=operation,
            description=description,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            formula=formula,
            reference=reference,
        )
        self._calculation_steps.append(step)

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of all calculation steps."""
        steps_data = [step.to_dict() for step in self._calculation_steps]
        json_str = json.dumps(steps_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def reset_cusum_state(self, exchanger_id: Optional[str] = None) -> None:
        """Reset CUSUM state for an exchanger or all exchangers."""
        if exchanger_id:
            if exchanger_id in self._cusum_state:
                del self._cusum_state[exchanger_id]
        else:
            self._cusum_state.clear()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_DECIMAL_PRECISION",
    "Z_SCORES",
    "FOULING_THRESHOLDS",

    # Enums
    "FoulingRegime",
    "AnomalyType",
    "AnomalySeverity",
    "TrendDirection",
    "SeasonalPeriod",

    # Result classes
    "TrendAnalysisResult",
    "SeasonalDecompositionResult",
    "ForecastResult",
    "AnomalyDetectionResult",
    "PatternRecognitionResult",
    "ThresholdPredictionResult",
    "CorrelationResult",
    "ModelValidationResult",
    "PredictionReportResult",
    "CalculationStep",

    # Main class
    "PredictiveFoulingEngine",
]
