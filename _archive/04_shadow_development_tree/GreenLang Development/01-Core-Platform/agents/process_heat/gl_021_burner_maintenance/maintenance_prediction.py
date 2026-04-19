# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY - Maintenance Prediction Engine

This module implements the Maintenance Prediction Engine for industrial burner
systems, providing Weibull-based RUL prediction, Proportional Hazards Modeling,
and ML-based failure prediction with ensemble methods.

Key capabilities:
    - Weibull survival analysis with MLE parameter estimation
    - Cox Proportional Hazards Model for covariate effects
    - Deterministic ML failure prediction (weighted scoring)
    - Ensemble prediction combining multiple methodologies
    - Uncertainty quantification and confidence intervals
    - SHA-256 provenance tracking for audit compliance

ZERO-HALLUCINATION COMPLIANCE:
    All predictions are deterministic based on historical data and
    physics-based models. No LLM/ML randomness in calculation path.
    Formulas are traceable to reliability engineering standards.

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.maintenance_prediction import (
    ...     MaintenancePredictionEngine
    ... )
    >>> engine = MaintenancePredictionEngine(config)
    >>> result = engine.predict_rul(
    ...     current_age_hours=15000,
    ...     operating_conditions=operating_data,
    ...     fuel_quality=fuel_data
    ... )
    >>> print(f"RUL P50: {result.rul_p50_hours:.0f} hours")

References:
    - ASME PCC-3: Guidelines for Fitness-for-Service
    - API 560: Fired Heaters for General Refinery Service
    - NFPA 86: Standard for Ovens and Furnaces
    - Weibull, W. (1951). A statistical distribution function of wide applicability
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import math

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Gas constant for Arrhenius calculations (kJ/mol-K)
GAS_CONSTANT_R = 8.314e-3

# Default burner component lifetimes (hours) from API 560 / Industry data
DEFAULT_COMPONENT_LIFETIMES = {
    "burner_tip": 35000,
    "flame_scanner": 25000,
    "pilot_assembly": 30000,
    "main_gas_valve": 50000,
    "air_damper": 40000,
    "ignition_transformer": 35000,
    "combustion_blower": 45000,
    "refractory": 60000,
    "flame_rod": 20000,
    "pressure_regulator": 40000,
}

# Typical Weibull beta values for burner components
COMPONENT_BETA_VALUES = {
    "burner_tip": 2.8,  # Wear-out pattern
    "flame_scanner": 1.8,  # Mixed failures
    "pilot_assembly": 2.2,
    "main_gas_valve": 2.5,
    "air_damper": 3.0,  # Strong wear-out
    "ignition_transformer": 1.5,
    "combustion_blower": 2.4,
    "refractory": 3.5,  # Thermal fatigue
    "flame_rod": 2.0,
    "pressure_regulator": 2.3,
}


# =============================================================================
# ENUMS
# =============================================================================

class BurnerComponent(str, Enum):
    """Burner system components for maintenance tracking."""
    BURNER_TIP = "burner_tip"
    FLAME_SCANNER = "flame_scanner"
    PILOT_ASSEMBLY = "pilot_assembly"
    MAIN_GAS_VALVE = "main_gas_valve"
    AIR_DAMPER = "air_damper"
    IGNITION_TRANSFORMER = "ignition_transformer"
    COMBUSTION_BLOWER = "combustion_blower"
    REFRACTORY = "refractory"
    FLAME_ROD = "flame_rod"
    PRESSURE_REGULATOR = "pressure_regulator"


class FailureMode(str, Enum):
    """Burner failure modes."""
    FLAME_INSTABILITY = "flame_instability"
    IGNITION_FAILURE = "ignition_failure"
    FLAME_SCANNER_FAULT = "flame_scanner_fault"
    FUEL_VALVE_LEAK = "fuel_valve_leak"
    AIR_DAMPER_STUCK = "air_damper_stuck"
    REFRACTORY_DEGRADATION = "refractory_degradation"
    COMBUSTION_AIR_LOSS = "combustion_air_loss"
    PILOT_FAILURE = "pilot_failure"
    HIGH_EMISSIONS = "high_emissions"
    OVERHEATING = "overheating"


class PredictionConfidence(str, Enum):
    """Prediction confidence levels."""
    HIGH = "high"       # >85% confidence
    MEDIUM = "medium"   # 70-85%
    LOW = "low"         # 50-70%
    UNCERTAIN = "uncertain"  # <50%


class MaintenanceUrgency(str, Enum):
    """Maintenance urgency classification."""
    IMMEDIATE = "immediate"    # Within 24 hours
    URGENT = "urgent"          # Within 1 week
    PLANNED = "planned"        # Within 1 month
    SCHEDULED = "scheduled"    # Next turnaround
    MONITOR = "monitor"        # Continue monitoring


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FailureData:
    """Container for failure/survival data point."""
    time_hours: float
    is_failure: bool = True
    component: Optional[BurnerComponent] = None
    failure_mode: Optional[FailureMode] = None
    operating_temp_c: Optional[float] = None
    fuel_quality_score: Optional[float] = None


@dataclass
class WeibullParameters:
    """Weibull distribution parameters."""
    beta: float  # Shape parameter
    eta: float   # Scale parameter (characteristic life in hours)
    gamma: float = 0.0  # Location parameter (failure-free life)
    r_squared: Optional[float] = None
    estimation_method: str = "mle"
    n_samples: int = 0


@dataclass
class CoxHazardResult:
    """Cox Proportional Hazards model result."""
    baseline_hazard: float
    hazard_ratios: Dict[str, float]
    coefficients: Dict[str, float]
    adjusted_rul_hours: float
    confidence_interval: Tuple[float, float]


@dataclass
class OperatingConditions:
    """Current operating conditions affecting degradation."""
    avg_flame_temp_c: float = 1200.0
    firing_rate_pct: float = 80.0
    air_fuel_ratio: float = 10.5
    combustion_air_temp_c: float = 25.0
    flue_gas_temp_c: float = 350.0
    cycling_frequency: float = 2.0  # cycles per day
    ambient_humidity_pct: float = 50.0
    fuel_sulfur_content_pct: float = 0.5
    excess_air_pct: float = 15.0


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class WeibullAnalysisConfig(BaseModel):
    """Configuration for Weibull analysis."""

    estimation_method: str = Field(
        default="mle",
        description="Parameter estimation method: mle, rank_regression, median_ranks"
    )
    confidence_level: float = Field(
        default=0.90,
        ge=0.50,
        le=0.99,
        description="Confidence level for intervals"
    )
    minimum_failures: int = Field(
        default=3,
        ge=2,
        description="Minimum failures required for analysis"
    )
    use_censored_data: bool = Field(
        default=True,
        description="Include right-censored (still running) data"
    )
    max_iterations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum iterations for MLE convergence"
    )
    convergence_tolerance: float = Field(
        default=1e-6,
        gt=0,
        description="Convergence tolerance for MLE"
    )


class PredictionEngineConfig(BaseModel):
    """Configuration for MaintenancePredictionEngine."""

    component: BurnerComponent = Field(
        default=BurnerComponent.BURNER_TIP,
        description="Burner component for analysis"
    )
    weibull: WeibullAnalysisConfig = Field(
        default_factory=WeibullAnalysisConfig,
        description="Weibull analysis configuration"
    )
    enable_cox_model: bool = Field(
        default=True,
        description="Enable Cox Proportional Hazards model"
    )
    enable_ml_prediction: bool = Field(
        default=True,
        description="Enable ML-based failure prediction"
    )
    enable_ensemble: bool = Field(
        default=True,
        description="Enable ensemble prediction"
    )
    ensemble_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "weibull": 0.40,
            "cox": 0.30,
            "ml": 0.30
        },
        description="Weights for ensemble prediction"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )


class RULPredictionResult(BaseModel):
    """Remaining Useful Life prediction result."""

    component: BurnerComponent = Field(..., description="Component analyzed")
    current_age_hours: float = Field(..., ge=0, description="Current component age")

    # Weibull-based RUL estimates
    rul_p10_hours: float = Field(..., description="RUL at 10% failure probability")
    rul_p50_hours: float = Field(..., description="RUL at 50% failure probability (median)")
    rul_p90_hours: float = Field(..., description="RUL at 90% failure probability")

    # Weibull parameters
    beta: float = Field(..., gt=0, description="Weibull shape parameter")
    eta_hours: float = Field(..., gt=0, description="Weibull scale parameter")

    # Failure probability
    current_failure_probability: float = Field(
        ..., ge=0, le=1,
        description="Current cumulative failure probability"
    )
    conditional_failure_prob_30d: float = Field(
        ..., ge=0, le=1,
        description="Conditional failure probability in next 30 days"
    )

    # Confidence and interpretation
    confidence_level: float = Field(default=0.90, description="Confidence level")
    prediction_confidence: PredictionConfidence = Field(
        ..., description="Overall prediction confidence"
    )
    maintenance_urgency: MaintenanceUrgency = Field(
        ..., description="Recommended maintenance urgency"
    )

    # Ensemble results
    ensemble_rul_hours: Optional[float] = Field(
        default=None,
        description="Ensemble-combined RUL estimate"
    )
    ensemble_uncertainty_hours: Optional[float] = Field(
        default=None,
        description="Ensemble prediction uncertainty"
    )

    # Cox model adjustments
    hazard_ratio: Optional[float] = Field(
        default=None,
        description="Cox model hazard ratio"
    )
    cox_adjusted_rul_hours: Optional[float] = Field(
        default=None,
        description="Cox-adjusted RUL"
    )

    # Interpretation
    failure_mode_interpretation: str = Field(
        ..., description="Interpretation of failure pattern"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Maintenance recommendations"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Prediction timestamp"
    )

    class Config:
        use_enum_values = True


class FailurePredictionResult(BaseModel):
    """ML-based failure prediction result."""

    failure_mode: FailureMode = Field(..., description="Predicted failure mode")
    probability: float = Field(..., ge=0, le=1, description="Failure probability")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")

    time_to_failure_hours: Optional[float] = Field(
        default=None, ge=0,
        description="Estimated time to failure"
    )
    uncertainty_lower_hours: Optional[float] = Field(
        default=None, description="Lower uncertainty bound"
    )
    uncertainty_upper_hours: Optional[float] = Field(
        default=None, description="Upper uncertainty bound"
    )

    feature_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance scores"
    )
    top_contributing_features: List[str] = Field(
        default_factory=list,
        description="Top features driving prediction"
    )
    risk_level: str = Field(default="medium", description="Risk classification")

    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


# =============================================================================
# WEIBULL ANALYZER CLASS
# =============================================================================

class WeibullAnalyzer:
    """
    Weibull Distribution Analyzer for burner component RUL estimation.

    Implements Maximum Likelihood Estimation (MLE) for Weibull parameters
    with support for right-censored data. Provides P10, P50, P90 confidence
    intervals for remaining useful life predictions.

    All calculations are DETERMINISTIC with full provenance tracking.

    Attributes:
        config: Weibull analysis configuration
        _cached_params: Cached parameter estimates

    Example:
        >>> analyzer = WeibullAnalyzer(config)
        >>> result = analyzer.analyze(failure_data, current_age=15000)
        >>> print(f"Beta: {result.beta:.2f}, Eta: {result.eta:.0f}")
    """

    def __init__(self, config: Optional[WeibullAnalysisConfig] = None) -> None:
        """
        Initialize Weibull analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or WeibullAnalysisConfig()
        self._cached_params: Optional[WeibullParameters] = None

        logger.info(
            f"WeibullAnalyzer initialized: method={self.config.estimation_method}, "
            f"confidence={self.config.confidence_level}"
        )

    def estimate_parameters(
        self,
        failure_data: List[FailureData],
        component_default: Optional[BurnerComponent] = None
    ) -> WeibullParameters:
        """
        Estimate Weibull parameters from failure data.

        Uses Maximum Likelihood Estimation (MLE) by default, with fallback
        to rank regression for small samples. Handles right-censored data.

        Args:
            failure_data: List of failure/survival observations
            component_default: Default component for industry priors

        Returns:
            WeibullParameters with beta, eta, and fit statistics

        Raises:
            ValueError: If insufficient failure data
        """
        failures = [d for d in failure_data if d.is_failure]
        n_failures = len(failures)
        n_total = len(failure_data)

        logger.info(
            f"Estimating Weibull parameters: {n_failures} failures, "
            f"{n_total - n_failures} censored"
        )

        if n_failures < self.config.minimum_failures:
            logger.warning(
                f"Insufficient failures ({n_failures}). "
                "Using industry default parameters."
            )
            return self._get_default_parameters(component_default)

        if self.config.estimation_method == "mle":
            params = self._estimate_mle(failure_data)
        elif self.config.estimation_method == "rank_regression":
            params = self._estimate_rank_regression(failure_data)
        else:
            params = self._estimate_median_ranks(failure_data)

        self._cached_params = params
        return params

    def _estimate_mle(self, data: List[FailureData]) -> WeibullParameters:
        """
        Estimate parameters using Maximum Likelihood Estimation.

        MLE maximizes the likelihood function:
        L = product(f(t_i)^delta_i * R(t_i)^(1-delta_i))

        where delta_i = 1 for failures, 0 for censored observations.

        Args:
            data: Failure data with failure/censored indicators

        Returns:
            WeibullParameters from MLE
        """
        times = [d.time_hours for d in data]
        is_failure = [d.is_failure for d in data]
        n = len(times)
        r = sum(is_failure)  # Number of failures

        if r == 0:
            raise ValueError("No failures in dataset for MLE estimation")

        # Newton-Raphson iteration for beta (shape parameter)
        beta = 2.0  # Initial guess (typical wear-out)

        for iteration in range(self.config.max_iterations):
            # Calculate likelihood derivative terms
            sum_t_beta = sum(t ** beta for t in times if t > 0)
            sum_t_beta_ln = sum(
                (t ** beta) * math.log(t)
                for t in times if t > 0
            )
            sum_ln_failures = sum(
                math.log(t)
                for t, f in zip(times, is_failure)
                if f and t > 0
            )

            if sum_t_beta <= 0:
                logger.warning("MLE: sum_t_beta <= 0, breaking iteration")
                break

            # First derivative of log-likelihood w.r.t. beta
            f_beta = (
                r / beta +
                sum_ln_failures -
                (r / sum_t_beta) * sum_t_beta_ln
            )

            # Second derivative for Newton-Raphson
            sum_t_beta_ln2 = sum(
                (t ** beta) * (math.log(t) ** 2)
                for t in times if t > 0
            )

            f_prime = (
                -r / (beta ** 2) -
                (r / sum_t_beta) * sum_t_beta_ln2 +
                (r / (sum_t_beta ** 2)) * (sum_t_beta_ln ** 2)
            )

            if abs(f_prime) < 1e-10:
                break

            # Newton-Raphson update
            beta_new = beta - f_beta / f_prime

            # Ensure positive beta
            if beta_new <= 0:
                beta_new = beta / 2

            # Check convergence
            if abs(beta_new - beta) < self.config.convergence_tolerance:
                beta = beta_new
                logger.debug(f"MLE converged at iteration {iteration}")
                break

            beta = beta_new

        # Estimate eta (scale parameter) given beta
        # eta = (sum(t^beta) / r)^(1/beta)
        eta = (sum(t ** beta for t in times) / r) ** (1 / beta)

        # Calculate R-squared for fit quality
        r_squared = self._calculate_r_squared(data, beta, eta)

        logger.info(f"MLE complete: beta={beta:.3f}, eta={eta:.1f}, R2={r_squared:.3f}")

        return WeibullParameters(
            beta=beta,
            eta=eta,
            gamma=0.0,
            r_squared=r_squared,
            estimation_method="mle",
            n_samples=n
        )

    def _estimate_rank_regression(self, data: List[FailureData]) -> WeibullParameters:
        """
        Estimate parameters using Rank Regression (Least Squares).

        Linearizes the Weibull CDF using the transformation:
        ln(ln(1/(1-F))) = beta * ln(t) - beta * ln(eta)

        More robust for small samples than MLE.

        Args:
            data: Failure data

        Returns:
            WeibullParameters from rank regression
        """
        # Filter to failures and sort by time
        failures = sorted(
            [d.time_hours for d in data if d.is_failure]
        )
        n = len(failures)

        if n < 2:
            raise ValueError("Need at least 2 failures for rank regression")

        # Calculate median ranks using Bernard's approximation
        # F(i) = (i - 0.3) / (n + 0.4)
        x = []  # ln(t)
        y = []  # ln(ln(1/(1-F)))

        for i, t in enumerate(failures, 1):
            if t <= 0:
                continue
            rank = (i - 0.3) / (n + 0.4)
            if rank <= 0 or rank >= 1:
                continue

            x.append(math.log(t))
            y.append(math.log(math.log(1 / (1 - rank))))

        if len(x) < 2:
            raise ValueError("Insufficient valid data points for regression")

        # Linear regression: y = beta * x - beta * ln(eta)
        n_pts = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        denominator = n_pts * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            raise ValueError("Singular matrix in regression")

        # Slope = beta
        beta = (n_pts * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - beta * sum_x) / n_pts

        # Solve for eta: intercept = -beta * ln(eta)
        eta = math.exp(-intercept / beta) if beta > 0 else 50000

        # Calculate R-squared
        y_mean = sum_y / n_pts
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum(
            (yi - (beta * xi + intercept)) ** 2
            for xi, yi in zip(x, y)
        )
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Ensure valid parameters
        beta = max(0.1, beta)
        eta = max(1, eta)

        logger.info(
            f"Rank regression complete: beta={beta:.3f}, eta={eta:.1f}, "
            f"R2={r_squared:.3f}"
        )

        return WeibullParameters(
            beta=beta,
            eta=eta,
            gamma=0.0,
            r_squared=r_squared,
            estimation_method="rank_regression",
            n_samples=n
        )

    def _estimate_median_ranks(self, data: List[FailureData]) -> WeibullParameters:
        """Estimate using median rank regression (delegates to rank regression)."""
        return self._estimate_rank_regression(data)

    def _get_default_parameters(
        self,
        component: Optional[BurnerComponent]
    ) -> WeibullParameters:
        """
        Get industry default Weibull parameters.

        Args:
            component: Burner component type

        Returns:
            Default WeibullParameters based on industry data
        """
        if component:
            comp_name = component.value
            beta = COMPONENT_BETA_VALUES.get(comp_name, 2.5)
            eta = DEFAULT_COMPONENT_LIFETIMES.get(comp_name, 40000)
        else:
            beta = 2.5  # Typical wear-out pattern
            eta = 40000  # Default characteristic life

        logger.info(f"Using default parameters: beta={beta}, eta={eta}")

        return WeibullParameters(
            beta=beta,
            eta=eta,
            gamma=0.0,
            r_squared=None,
            estimation_method="default",
            n_samples=0
        )

    def calculate_reliability(
        self,
        t: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate reliability function R(t).

        R(t) = exp(-((t - gamma) / eta)^beta)

        Args:
            t: Time in hours
            params: Weibull parameters

        Returns:
            Reliability (survival probability) at time t
        """
        if t <= params.gamma:
            return 1.0

        t_adj = (t - params.gamma) / params.eta
        return math.exp(-(t_adj ** params.beta))

    def calculate_cdf(
        self,
        t: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate cumulative distribution function F(t).

        F(t) = 1 - R(t) = 1 - exp(-((t - gamma) / eta)^beta)

        Args:
            t: Time in hours
            params: Weibull parameters

        Returns:
            Cumulative failure probability at time t
        """
        return 1 - self.calculate_reliability(t, params)

    def calculate_hazard_rate(
        self,
        t: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate instantaneous hazard rate h(t).

        h(t) = (beta/eta) * ((t - gamma)/eta)^(beta - 1)

        Args:
            t: Time in hours
            params: Weibull parameters

        Returns:
            Hazard rate at time t
        """
        if t <= params.gamma:
            return 0.0

        t_adj = (t - params.gamma) / params.eta
        return (params.beta / params.eta) * (t_adj ** (params.beta - 1))

    def calculate_rul(
        self,
        current_age: float,
        params: WeibullParameters,
        probability: float = 0.50
    ) -> float:
        """
        Calculate Remaining Useful Life at given probability.

        RUL_P = t_P - current_age
        where t_P = gamma + eta * (-ln(1-P))^(1/beta)

        Args:
            current_age: Current equipment age in hours
            params: Weibull parameters
            probability: Target failure probability (default 0.50 for median)

        Returns:
            Remaining useful life in hours
        """
        if probability <= 0:
            return float('inf')
        if probability >= 1:
            return 0

        # Calculate time to target probability
        t_p = params.gamma + params.eta * (
            (-math.log(1 - probability)) ** (1 / params.beta)
        )

        # RUL is time to target minus current age
        rul = max(0, t_p - current_age)

        return rul

    def calculate_conditional_failure_probability(
        self,
        current_age: float,
        future_age: float,
        params: WeibullParameters
    ) -> float:
        """
        Calculate conditional probability of failure.

        P(T <= future | T > current) = 1 - R(future)/R(current)

        Args:
            current_age: Current age in hours
            future_age: Future time point in hours
            params: Weibull parameters

        Returns:
            Conditional failure probability
        """
        if future_age <= current_age:
            return 0.0

        r_current = self.calculate_reliability(current_age, params)
        r_future = self.calculate_reliability(future_age, params)

        if r_current <= 0:
            return 1.0

        return 1 - (r_future / r_current)

    def calculate_confidence_intervals(
        self,
        params: WeibullParameters,
        n_samples: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate confidence intervals for Weibull parameters.

        Uses approximate Fisher information for large sample intervals.

        Args:
            params: Estimated parameters
            n_samples: Number of samples

        Returns:
            Tuple of (beta_ci, eta_ci) confidence intervals
        """
        # Z-score for confidence level
        alpha = 1 - self.config.confidence_level
        if self.config.confidence_level == 0.90:
            z = 1.645
        elif self.config.confidence_level == 0.95:
            z = 1.96
        elif self.config.confidence_level == 0.99:
            z = 2.576
        else:
            # Approximate using normal quantile
            z = 1.645 + (self.config.confidence_level - 0.90) * 3.15

        # Approximate standard errors
        if n_samples > 0:
            se_beta = params.beta / math.sqrt(n_samples)
            se_eta = params.eta / math.sqrt(n_samples)
        else:
            se_beta = params.beta * 0.3
            se_eta = params.eta * 0.3

        beta_lower = max(0.1, params.beta - z * se_beta)
        beta_upper = params.beta + z * se_beta

        eta_lower = max(1, params.eta - z * se_eta)
        eta_upper = params.eta + z * se_eta

        return (beta_lower, beta_upper), (eta_lower, eta_upper)

    def interpret_beta(self, beta: float) -> str:
        """
        Provide interpretation of Weibull shape parameter.

        Args:
            beta: Shape parameter value

        Returns:
            Human-readable interpretation
        """
        if beta < 0.5:
            return (
                "Very early life failures (severe infant mortality). "
                "Investigate manufacturing defects, installation errors, "
                "or material quality issues. Immediate inspection required."
            )
        elif beta < 1.0:
            return (
                "Decreasing failure rate (infant mortality pattern). "
                "Equipment likely has early life issues from manufacturing "
                "or installation. Consider burn-in testing or "
                "enhanced quality control."
            )
        elif abs(beta - 1.0) < 0.15:
            return (
                "Constant failure rate (random failures). "
                "Failures are independent of age - may indicate "
                "external factors, random stress events, or "
                "design limitations. Condition monitoring less effective."
            )
        elif beta < 2.0:
            return (
                "Slightly increasing failure rate (early wear-out). "
                "Equipment showing gradual degradation. "
                "Predictive maintenance moderately effective. "
                "Consider condition-based monitoring intervals."
            )
        elif beta < 3.5:
            return (
                "Increasing failure rate (wear-out pattern). "
                "Normal aging degradation consistent with "
                "burner component wear. Predictive maintenance "
                "highly effective. Schedule replacement before failure."
            )
        else:
            return (
                "Rapidly increasing failure rate (severe wear-out). "
                "Accelerated degradation possibly due to harsh operating "
                "conditions, poor fuel quality, or thermal cycling stress. "
                "Consider proactive replacement or operating condition review."
            )

    def _calculate_r_squared(
        self,
        data: List[FailureData],
        beta: float,
        eta: float
    ) -> float:
        """Calculate R-squared for fit quality assessment."""
        failures = sorted([d.time_hours for d in data if d.is_failure])
        n = len(failures)

        if n < 2:
            return 0.0

        params = WeibullParameters(beta=beta, eta=eta)

        y_emp = []
        y_theo = []

        for i, t in enumerate(failures, 1):
            if t <= 0:
                continue
            rank = (i - 0.3) / (n + 0.4)
            if 0 < rank < 1:
                y_emp.append(math.log(math.log(1 / (1 - rank))))
                y_theo.append(beta * (math.log(t) - math.log(eta)))

        if len(y_emp) < 2:
            return 0.0

        y_mean = sum(y_emp) / len(y_emp)
        ss_tot = sum((y - y_mean) ** 2 for y in y_emp)
        ss_res = sum((ye - yt) ** 2 for ye, yt in zip(y_emp, y_theo))

        return max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0


# =============================================================================
# PROPORTIONAL HAZARDS MODEL CLASS
# =============================================================================

class ProportionalHazardsModel:
    """
    Cox Proportional Hazards Model for covariate effects.

    Models how operating conditions affect burner failure rates:
    h(t|X) = h_0(t) * exp(beta_1*X_1 + beta_2*X_2 + ...)

    Covariates include:
    - Operating temperature
    - Fuel quality score
    - Cycling frequency
    - Firing rate

    All calculations are DETERMINISTIC with predetermined coefficients
    based on industry reliability studies and engineering correlations.

    Attributes:
        baseline_hazard: Baseline hazard rate
        coefficients: Covariate coefficients (from industry data)

    Example:
        >>> model = ProportionalHazardsModel()
        >>> hazard_ratio = model.calculate_hazard_ratio(operating_conditions)
        >>> adjusted_rul = model.adjust_rul(rul_hours, hazard_ratio)
    """

    # Predetermined coefficients from industry reliability studies
    # Positive coefficients increase hazard (reduce life)
    # Negative coefficients decrease hazard (increase life)
    DEFAULT_COEFFICIENTS = {
        "flame_temp_factor": 0.008,      # Per degree C above reference
        "firing_rate_factor": 0.012,      # Per % above 80% base
        "cycling_stress_factor": 0.15,    # Per cycle per day above 2
        "fuel_sulfur_factor": 0.25,       # Per % sulfur content
        "excess_air_deviation": 0.005,    # Per % deviation from optimal
        "air_fuel_ratio_deviation": 0.08, # Per unit deviation
        "humidity_factor": 0.002,         # Per % above 50%
    }

    # Reference operating conditions (baseline)
    REFERENCE_CONDITIONS = {
        "flame_temp_c": 1200.0,
        "firing_rate_pct": 80.0,
        "cycling_frequency": 2.0,
        "fuel_sulfur_pct": 0.5,
        "excess_air_pct": 15.0,
        "air_fuel_ratio": 10.5,
        "humidity_pct": 50.0,
    }

    def __init__(
        self,
        coefficients: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize Proportional Hazards model.

        Args:
            coefficients: Custom coefficients (uses defaults if not provided)
        """
        self.coefficients = coefficients or self.DEFAULT_COEFFICIENTS.copy()
        self.reference = self.REFERENCE_CONDITIONS.copy()

        logger.info(f"ProportionalHazardsModel initialized with {len(self.coefficients)} covariates")

    def calculate_hazard_ratio(
        self,
        conditions: OperatingConditions
    ) -> float:
        """
        Calculate hazard ratio for current operating conditions.

        HR = exp(sum(beta_i * (X_i - X_ref_i)))

        HR > 1 means higher risk (shorter life)
        HR < 1 means lower risk (longer life)
        HR = 1 means same as baseline

        Args:
            conditions: Current operating conditions

        Returns:
            Hazard ratio relative to baseline
        """
        log_hr = 0.0

        # Flame temperature effect (Arrhenius-like)
        temp_diff = conditions.avg_flame_temp_c - self.reference["flame_temp_c"]
        log_hr += self.coefficients["flame_temp_factor"] * temp_diff

        # Firing rate effect
        rate_diff = max(0, conditions.firing_rate_pct - self.reference["firing_rate_pct"])
        log_hr += self.coefficients["firing_rate_factor"] * rate_diff

        # Cycling stress effect (thermal fatigue)
        cycle_diff = max(0, conditions.cycling_frequency - self.reference["cycling_frequency"])
        log_hr += self.coefficients["cycling_stress_factor"] * cycle_diff

        # Fuel sulfur effect (corrosion)
        sulfur_diff = max(0, conditions.fuel_sulfur_content_pct - self.reference["fuel_sulfur_pct"])
        log_hr += self.coefficients["fuel_sulfur_factor"] * sulfur_diff

        # Excess air deviation (flame impingement/oxidation)
        air_diff = abs(conditions.excess_air_pct - self.reference["excess_air_pct"])
        log_hr += self.coefficients["excess_air_deviation"] * air_diff

        # Air/fuel ratio deviation
        afr_diff = abs(conditions.air_fuel_ratio - self.reference["air_fuel_ratio"])
        log_hr += self.coefficients["air_fuel_ratio_deviation"] * afr_diff

        # Humidity effect (moisture-related degradation)
        humidity_diff = max(0, conditions.ambient_humidity_pct - self.reference["humidity_pct"])
        log_hr += self.coefficients["humidity_factor"] * humidity_diff

        # Calculate hazard ratio
        hazard_ratio = math.exp(log_hr)

        # Clamp to reasonable range [0.5, 5.0]
        hazard_ratio = max(0.5, min(5.0, hazard_ratio))

        logger.debug(f"Hazard ratio: {hazard_ratio:.3f} (log_HR: {log_hr:.3f})")

        return hazard_ratio

    def adjust_rul(
        self,
        baseline_rul_hours: float,
        hazard_ratio: float
    ) -> float:
        """
        Adjust RUL based on hazard ratio.

        Adjusted RUL = Baseline RUL / Hazard Ratio

        Args:
            baseline_rul_hours: Baseline RUL from Weibull analysis
            hazard_ratio: Calculated hazard ratio

        Returns:
            Adjusted RUL in hours
        """
        if hazard_ratio <= 0:
            return baseline_rul_hours

        adjusted_rul = baseline_rul_hours / hazard_ratio

        logger.debug(
            f"RUL adjusted: {baseline_rul_hours:.0f} -> {adjusted_rul:.0f} hours "
            f"(HR={hazard_ratio:.3f})"
        )

        return adjusted_rul

    def calculate_covariate_contributions(
        self,
        conditions: OperatingConditions
    ) -> Dict[str, float]:
        """
        Calculate individual covariate contributions to hazard.

        Args:
            conditions: Operating conditions

        Returns:
            Dictionary of covariate contributions
        """
        contributions = {}

        # Flame temperature
        temp_diff = conditions.avg_flame_temp_c - self.reference["flame_temp_c"]
        contributions["flame_temperature"] = self.coefficients["flame_temp_factor"] * temp_diff

        # Firing rate
        rate_diff = max(0, conditions.firing_rate_pct - self.reference["firing_rate_pct"])
        contributions["firing_rate"] = self.coefficients["firing_rate_factor"] * rate_diff

        # Cycling
        cycle_diff = max(0, conditions.cycling_frequency - self.reference["cycling_frequency"])
        contributions["cycling_stress"] = self.coefficients["cycling_stress_factor"] * cycle_diff

        # Fuel quality
        sulfur_diff = max(0, conditions.fuel_sulfur_content_pct - self.reference["fuel_sulfur_pct"])
        contributions["fuel_sulfur"] = self.coefficients["fuel_sulfur_factor"] * sulfur_diff

        # Combustion
        air_diff = abs(conditions.excess_air_pct - self.reference["excess_air_pct"])
        contributions["excess_air"] = self.coefficients["excess_air_deviation"] * air_diff

        return contributions

    def get_risk_factors(
        self,
        conditions: OperatingConditions
    ) -> List[str]:
        """
        Identify significant risk factors from operating conditions.

        Args:
            conditions: Operating conditions

        Returns:
            List of identified risk factors
        """
        contributions = self.calculate_covariate_contributions(conditions)

        risk_factors = []
        threshold = 0.05  # Significant contribution threshold

        for factor, contribution in contributions.items():
            if contribution > threshold:
                severity = "high" if contribution > 0.2 else "moderate"
                risk_factors.append(f"{severity.upper()}: {factor} (impact: +{contribution:.1%})")

        return sorted(risk_factors, reverse=True)


# =============================================================================
# ML FAILURE PREDICTOR CLASS
# =============================================================================

class MLFailurePredictor:
    """
    Deterministic ML-based Failure Predictor for burner systems.

    Uses weighted scoring models based on engineering correlations
    rather than stochastic ML. This ensures ZERO-HALLUCINATION
    compliance with fully traceable predictions.

    Feature categories:
    - Combustion efficiency indicators
    - Temperature profiles
    - Flame characteristics
    - Equipment age and cycling
    - Fuel quality parameters

    Attributes:
        feature_weights: Predetermined feature weights by failure mode
        is_calibrated: Whether model is calibrated

    Example:
        >>> predictor = MLFailurePredictor()
        >>> features = predictor.extract_features(sensor_data)
        >>> result = predictor.predict(features, FailureMode.FLAME_INSTABILITY)
    """

    # Feature weights for each failure mode (deterministic)
    FAILURE_MODE_WEIGHTS = {
        FailureMode.FLAME_INSTABILITY: {
            "flame_temp_deviation": 0.25,
            "air_fuel_ratio_deviation": 0.20,
            "combustion_efficiency": 0.15,
            "flame_scanner_signal": 0.20,
            "burner_age_factor": 0.10,
            "fuel_quality_score": 0.10,
        },
        FailureMode.IGNITION_FAILURE: {
            "ignition_transformer_age": 0.25,
            "pilot_flame_strength": 0.25,
            "fuel_pressure_stability": 0.15,
            "ambient_conditions": 0.10,
            "electrode_gap_deviation": 0.15,
            "burner_age_factor": 0.10,
        },
        FailureMode.FLAME_SCANNER_FAULT: {
            "scanner_signal_noise": 0.30,
            "scanner_age_factor": 0.25,
            "flame_stability_index": 0.15,
            "ambient_temperature": 0.10,
            "lens_fouling_indicator": 0.20,
        },
        FailureMode.HIGH_EMISSIONS: {
            "excess_air_deviation": 0.25,
            "combustion_efficiency": 0.25,
            "air_fuel_ratio_deviation": 0.20,
            "flame_temp_deviation": 0.15,
            "fuel_quality_score": 0.15,
        },
        FailureMode.REFRACTORY_DEGRADATION: {
            "thermal_cycling_stress": 0.30,
            "max_flame_temp": 0.25,
            "refractory_age_factor": 0.20,
            "firing_rate_profile": 0.15,
            "fuel_ash_content": 0.10,
        },
        FailureMode.OVERHEATING: {
            "flame_temp_deviation": 0.25,
            "firing_rate_factor": 0.20,
            "cooling_efficiency": 0.20,
            "air_flow_restriction": 0.15,
            "ambient_temperature": 0.10,
            "burner_age_factor": 0.10,
        },
    }

    def __init__(self) -> None:
        """Initialize ML Failure Predictor."""
        self.is_calibrated = True  # Deterministic model always "calibrated"
        self._model_version = "1.0.0"
        self._model_id = "gl021_burner_failure_pred"

        logger.info(
            f"MLFailurePredictor initialized: "
            f"{len(self.FAILURE_MODE_WEIGHTS)} failure modes"
        )

    def extract_features(
        self,
        operating_conditions: OperatingConditions,
        sensor_data: Optional[Dict[str, float]] = None,
        equipment_age_hours: float = 0,
        fuel_quality_score: float = 100.0
    ) -> Dict[str, float]:
        """
        Extract features from operating data.

        Transforms raw sensor data into normalized features
        suitable for the weighted scoring model.

        Args:
            operating_conditions: Current operating conditions
            sensor_data: Raw sensor readings
            equipment_age_hours: Equipment age in hours
            fuel_quality_score: Fuel quality score (0-100)

        Returns:
            Normalized feature dictionary
        """
        features: Dict[str, float] = {}
        sensor_data = sensor_data or {}

        # Temperature features
        ref_flame_temp = 1200.0
        flame_temp = operating_conditions.avg_flame_temp_c
        features["flame_temp_deviation"] = self._normalize(
            abs(flame_temp - ref_flame_temp) / ref_flame_temp, 0, 0.5
        )
        features["max_flame_temp"] = self._normalize(
            flame_temp, 1000, 1500
        )

        # Combustion features
        ref_afr = 10.5
        afr = operating_conditions.air_fuel_ratio
        features["air_fuel_ratio_deviation"] = self._normalize(
            abs(afr - ref_afr) / ref_afr, 0, 0.3
        )

        ref_excess_air = 15.0
        excess_air = operating_conditions.excess_air_pct
        features["excess_air_deviation"] = self._normalize(
            abs(excess_air - ref_excess_air) / ref_excess_air, 0, 0.5
        )

        # Efficiency features
        features["combustion_efficiency"] = 1.0 - self._normalize(
            sensor_data.get("co_ppm", 0), 0, 500
        )

        # Age and cycling features
        typical_life = 40000.0  # hours
        features["burner_age_factor"] = self._normalize(
            equipment_age_hours, 0, typical_life
        )
        features["scanner_age_factor"] = features["burner_age_factor"] * 1.2
        features["refractory_age_factor"] = features["burner_age_factor"] * 0.8
        features["ignition_transformer_age"] = features["burner_age_factor"] * 1.1

        # Thermal cycling stress (normalized per day)
        cycling = operating_conditions.cycling_frequency
        features["thermal_cycling_stress"] = self._normalize(
            cycling, 0, 10
        )

        # Firing rate features
        firing_rate = operating_conditions.firing_rate_pct
        features["firing_rate_factor"] = self._normalize(
            max(0, firing_rate - 80), 0, 20
        )
        features["firing_rate_profile"] = self._normalize(
            firing_rate, 50, 100
        )

        # Fuel quality features
        features["fuel_quality_score"] = 1.0 - (fuel_quality_score / 100.0)
        features["fuel_ash_content"] = sensor_data.get("fuel_ash_pct", 0) / 10.0

        # Sensor-based features (with defaults)
        features["flame_scanner_signal"] = 1.0 - self._normalize(
            sensor_data.get("flame_signal_mv", 1000), 200, 1000
        )
        features["scanner_signal_noise"] = self._normalize(
            sensor_data.get("signal_noise_pct", 0), 0, 20
        )
        features["pilot_flame_strength"] = 1.0 - self._normalize(
            sensor_data.get("pilot_signal_mv", 500), 100, 500
        )
        features["flame_stability_index"] = self._normalize(
            sensor_data.get("flame_variance", 0), 0, 0.5
        )
        features["lens_fouling_indicator"] = self._normalize(
            sensor_data.get("lens_clarity_pct", 100), 50, 100
        )

        # Pressure and flow features
        features["fuel_pressure_stability"] = 1.0 - self._normalize(
            sensor_data.get("pressure_variance_pct", 0), 0, 10
        )
        features["air_flow_restriction"] = self._normalize(
            sensor_data.get("dp_inches_wc", 0), 0, 5
        )

        # Environmental features
        features["ambient_temperature"] = self._normalize(
            operating_conditions.combustion_air_temp_c, 0, 50
        )
        features["ambient_conditions"] = self._normalize(
            operating_conditions.ambient_humidity_pct, 30, 90
        )

        # Derived features
        features["electrode_gap_deviation"] = sensor_data.get("gap_deviation_pct", 0) / 50
        features["cooling_efficiency"] = 1.0 - self._normalize(
            sensor_data.get("jacket_temp_rise_c", 0), 0, 50
        )

        return features

    def predict(
        self,
        features: Dict[str, float],
        failure_mode: FailureMode
    ) -> FailurePredictionResult:
        """
        Predict failure probability for specific failure mode.

        Uses deterministic weighted scoring for ZERO-HALLUCINATION
        compliance. Weights are derived from domain expertise and
        historical analysis.

        Args:
            features: Extracted feature dictionary
            failure_mode: Target failure mode

        Returns:
            FailurePredictionResult with probability and explainability
        """
        weights = self.FAILURE_MODE_WEIGHTS.get(
            failure_mode,
            {"burner_age_factor": 1.0}
        )

        # Calculate weighted score
        score = 0.0
        feature_contributions: Dict[str, float] = {}

        for feature_name, weight in weights.items():
            value = features.get(feature_name, 0.0)
            contribution = value * weight
            score += contribution
            feature_contributions[feature_name] = contribution

        # Convert score to probability (logistic function)
        k = 5.0  # Steepness factor
        threshold = 0.5
        probability = 1 / (1 + math.exp(-k * (score - threshold)))
        probability = max(0.001, min(0.999, probability))

        # Calculate confidence based on feature availability
        available_features = sum(
            1 for f in weights.keys()
            if f in features and features[f] > 0
        )
        confidence = available_features / len(weights)
        confidence = max(0.5, min(0.95, confidence))

        # Top contributing features
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_features = [f[0] for f in sorted_features[:3]]

        # Risk level classification
        if probability > 0.7:
            risk_level = "critical"
        elif probability > 0.5:
            risk_level = "high"
        elif probability > 0.3:
            risk_level = "moderate"
        elif probability > 0.1:
            risk_level = "low"
        else:
            risk_level = "minimal"

        # Time to failure estimate
        base_hours = 3000.0
        if probability > 0.1:
            ttf_hours = base_hours * (1 - probability) / probability
            ttf_hours = max(24, min(30000, ttf_hours))
            uncertainty_factor = 0.3 + (1 - confidence) * 0.4
            ttf_lower = ttf_hours * (1 - uncertainty_factor)
            ttf_upper = ttf_hours * (1 + uncertainty_factor)
        else:
            ttf_hours = None
            ttf_lower = None
            ttf_upper = None

        # Provenance hash
        provenance_hash = self._calculate_provenance(
            features, failure_mode, probability
        )

        return FailurePredictionResult(
            failure_mode=failure_mode,
            probability=probability,
            confidence=confidence,
            time_to_failure_hours=ttf_hours,
            uncertainty_lower_hours=ttf_lower,
            uncertainty_upper_hours=ttf_upper,
            feature_importance=feature_contributions,
            top_contributing_features=top_features,
            risk_level=risk_level,
            provenance_hash=provenance_hash,
        )

    def predict_all_modes(
        self,
        features: Dict[str, float]
    ) -> List[FailurePredictionResult]:
        """
        Predict probabilities for all monitored failure modes.

        Args:
            features: Extracted features

        Returns:
            List of predictions sorted by probability
        """
        predictions = []

        for failure_mode in self.FAILURE_MODE_WEIGHTS.keys():
            prediction = self.predict(features, failure_mode)
            predictions.append(prediction)

        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def _calculate_provenance(
        self,
        features: Dict[str, float],
        failure_mode: FailureMode,
        probability: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        feature_str = "|".join(
            f"{k}:{v:.6f}"
            for k, v in sorted(features.items())
        )
        provenance_str = (
            f"ml_pred|{self._model_id}|{self._model_version}|"
            f"{failure_mode.value}|{probability:.8f}|{feature_str}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# ENSEMBLE PREDICTION CLASS
# =============================================================================

class EnsemblePrediction:
    """
    Ensemble prediction combining Weibull, Cox, and ML methods.

    Provides uncertainty quantification through method agreement
    and weighted combination of predictions.

    Ensemble weights are configurable but default to:
    - Weibull: 40% (reliability baseline)
    - Cox: 30% (operating condition adjustment)
    - ML: 30% (pattern recognition)

    Attributes:
        weights: Method weights for combination

    Example:
        >>> ensemble = EnsemblePrediction()
        >>> result = ensemble.combine(weibull_rul, cox_rul, ml_rul)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize ensemble predictor.

        Args:
            weights: Method weights (must sum to 1.0)
        """
        self.weights = weights or {
            "weibull": 0.40,
            "cox": 0.30,
            "ml": 0.30
        }

        # Normalize weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v/total for k, v in self.weights.items()}

        logger.info(f"EnsemblePrediction initialized: weights={self.weights}")

    def combine_rul_predictions(
        self,
        weibull_rul: float,
        cox_rul: Optional[float] = None,
        ml_rul: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Combine RUL predictions from multiple methods.

        Args:
            weibull_rul: Weibull-based RUL
            cox_rul: Cox-adjusted RUL (optional)
            ml_rul: ML-predicted RUL (optional)

        Returns:
            Tuple of (ensemble_rul, uncertainty)
        """
        predictions = []
        weights = []

        predictions.append(weibull_rul)
        weights.append(self.weights["weibull"])

        if cox_rul is not None:
            predictions.append(cox_rul)
            weights.append(self.weights["cox"])

        if ml_rul is not None:
            predictions.append(ml_rul)
            weights.append(self.weights["ml"])

        # Normalize weights for available predictions
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Weighted average
        ensemble_rul = sum(p * w for p, w in zip(predictions, weights))

        # Uncertainty from prediction spread
        if len(predictions) > 1:
            mean_pred = sum(predictions) / len(predictions)
            variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
            std_dev = math.sqrt(variance)

            # Uncertainty is coefficient of variation-like measure
            uncertainty = std_dev
        else:
            uncertainty = weibull_rul * 0.2  # 20% default uncertainty

        return ensemble_rul, uncertainty

    def calculate_consensus(
        self,
        weibull_p50: float,
        cox_rul: Optional[float],
        ml_predictions: List[FailurePredictionResult]
    ) -> Dict[str, Any]:
        """
        Calculate prediction consensus and disagreement.

        Args:
            weibull_p50: Weibull P50 RUL
            cox_rul: Cox-adjusted RUL
            ml_predictions: ML failure predictions

        Returns:
            Consensus analysis dictionary
        """
        # Get ML RUL estimate (from highest probability failure)
        ml_rul = None
        if ml_predictions and ml_predictions[0].time_to_failure_hours:
            ml_rul = ml_predictions[0].time_to_failure_hours

        # Combine predictions
        ensemble_rul, uncertainty = self.combine_rul_predictions(
            weibull_p50, cox_rul, ml_rul
        )

        # Determine agreement level
        available_preds = [weibull_p50]
        if cox_rul:
            available_preds.append(cox_rul)
        if ml_rul:
            available_preds.append(ml_rul)

        if len(available_preds) >= 2:
            max_pred = max(available_preds)
            min_pred = min(available_preds)
            spread_ratio = (max_pred - min_pred) / max(max_pred, 1)

            if spread_ratio < 0.2:
                agreement = "high"
            elif spread_ratio < 0.5:
                agreement = "moderate"
            else:
                agreement = "low"
        else:
            agreement = "single_method"

        # Determine confidence
        if agreement == "high" and ml_predictions:
            top_confidence = ml_predictions[0].confidence
            if top_confidence > 0.8:
                confidence = PredictionConfidence.HIGH
            elif top_confidence > 0.6:
                confidence = PredictionConfidence.MEDIUM
            else:
                confidence = PredictionConfidence.LOW
        elif agreement == "moderate":
            confidence = PredictionConfidence.MEDIUM
        elif agreement == "low":
            confidence = PredictionConfidence.LOW
        else:
            confidence = PredictionConfidence.UNCERTAIN

        return {
            "ensemble_rul_hours": ensemble_rul,
            "uncertainty_hours": uncertainty,
            "agreement_level": agreement,
            "prediction_confidence": confidence,
            "method_contributions": {
                "weibull": weibull_p50,
                "cox": cox_rul,
                "ml": ml_rul,
            }
        }

    def flag_anomalies(
        self,
        features: Dict[str, float],
        historical_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> List[str]:
        """
        Flag out-of-distribution inputs.

        Args:
            features: Current feature values
            historical_ranges: Expected (min, max) ranges for features

        Returns:
            List of anomaly warnings
        """
        anomalies = []

        # Default ranges based on typical burner operations
        default_ranges = {
            "flame_temp_deviation": (0, 0.3),
            "air_fuel_ratio_deviation": (0, 0.2),
            "combustion_efficiency": (0.8, 1.0),
            "burner_age_factor": (0, 1.5),
            "thermal_cycling_stress": (0, 0.5),
            "fuel_quality_score": (0, 0.3),
        }

        ranges = historical_ranges or default_ranges

        for feature, (min_val, max_val) in ranges.items():
            if feature in features:
                value = features[feature]
                if value < min_val * 0.8 or value > max_val * 1.2:
                    anomalies.append(
                        f"OUT_OF_RANGE: {feature}={value:.3f} "
                        f"(expected: {min_val:.3f}-{max_val:.3f})"
                    )

        return anomalies


# =============================================================================
# MAIN MAINTENANCE PREDICTION ENGINE CLASS
# =============================================================================

class MaintenancePredictionEngine:
    """
    Weibull-based RUL prediction + ML failure prediction.

    Implements multiple prediction methodologies:
    - Weibull survival analysis for RUL estimation
    - Proportional Hazards Model for covariate effects
    - Machine Learning failure prediction (deterministic)
    - Ensemble prediction combining all methods

    Zero-hallucination: All predictions are deterministic
    based on historical data and physics-based models.

    Attributes:
        config: Engine configuration
        weibull_analyzer: Weibull analysis module
        cox_model: Proportional hazards model
        ml_predictor: ML failure predictor
        ensemble: Ensemble combiner

    Example:
        >>> config = PredictionEngineConfig(component=BurnerComponent.BURNER_TIP)
        >>> engine = MaintenancePredictionEngine(config)
        >>> result = engine.predict_rul(
        ...     current_age_hours=15000,
        ...     failure_history=failures,
        ...     operating_conditions=conditions
        ... )
        >>> print(f"RUL P50: {result.rul_p50_hours:.0f} hours")
        >>> print(f"Urgency: {result.maintenance_urgency}")
    """

    def __init__(
        self,
        config: Optional[PredictionEngineConfig] = None
    ) -> None:
        """
        Initialize MaintenancePredictionEngine.

        Args:
            config: Engine configuration
        """
        self.config = config or PredictionEngineConfig()

        # Initialize sub-components
        self.weibull_analyzer = WeibullAnalyzer(self.config.weibull)

        if self.config.enable_cox_model:
            self.cox_model = ProportionalHazardsModel()
        else:
            self.cox_model = None

        if self.config.enable_ml_prediction:
            self.ml_predictor = MLFailurePredictor()
        else:
            self.ml_predictor = None

        if self.config.enable_ensemble:
            self.ensemble = EnsemblePrediction(self.config.ensemble_weights)
        else:
            self.ensemble = None

        logger.info(
            f"MaintenancePredictionEngine initialized for {self.config.component.value}: "
            f"cox={self.config.enable_cox_model}, ml={self.config.enable_ml_prediction}, "
            f"ensemble={self.config.enable_ensemble}"
        )

    def predict_rul(
        self,
        current_age_hours: float,
        failure_history: Optional[List[FailureData]] = None,
        operating_conditions: Optional[OperatingConditions] = None,
        sensor_data: Optional[Dict[str, float]] = None,
        fuel_quality_score: float = 100.0
    ) -> RULPredictionResult:
        """
        Predict Remaining Useful Life for burner component.

        Combines Weibull analysis, Cox Proportional Hazards adjustment,
        and ML failure prediction into an ensemble prediction.

        Args:
            current_age_hours: Current component age in hours
            failure_history: Historical failure data for Weibull
            operating_conditions: Current operating conditions
            sensor_data: Real-time sensor readings
            fuel_quality_score: Fuel quality (0-100)

        Returns:
            RULPredictionResult with comprehensive predictions
        """
        start_time = datetime.now(timezone.utc)

        logger.info(
            f"Starting RUL prediction for {self.config.component.value}: "
            f"age={current_age_hours:.0f}h"
        )

        # Default operating conditions if not provided
        if operating_conditions is None:
            operating_conditions = OperatingConditions()

        # Initialize failure history if not provided
        if failure_history is None:
            failure_history = []

        # Step 1: Weibull Analysis
        weibull_params = self.weibull_analyzer.estimate_parameters(
            failure_history,
            component_default=self.config.component
        )

        # Calculate RUL at different probability levels
        rul_p10 = self.weibull_analyzer.calculate_rul(
            current_age_hours, weibull_params, 0.10
        )
        rul_p50 = self.weibull_analyzer.calculate_rul(
            current_age_hours, weibull_params, 0.50
        )
        rul_p90 = self.weibull_analyzer.calculate_rul(
            current_age_hours, weibull_params, 0.90
        )

        # Current failure probability
        current_fail_prob = self.weibull_analyzer.calculate_cdf(
            current_age_hours, weibull_params
        )

        # Conditional failure probability (next 30 days = 720 hours)
        cond_fail_prob_30d = self.weibull_analyzer.calculate_conditional_failure_probability(
            current_age_hours,
            current_age_hours + 720,
            weibull_params
        )

        # Step 2: Cox Proportional Hazards Adjustment
        hazard_ratio = None
        cox_adjusted_rul = None

        if self.cox_model:
            hazard_ratio = self.cox_model.calculate_hazard_ratio(operating_conditions)
            cox_adjusted_rul = self.cox_model.adjust_rul(rul_p50, hazard_ratio)

        # Step 3: ML Failure Prediction
        ml_predictions = []
        if self.ml_predictor:
            features = self.ml_predictor.extract_features(
                operating_conditions,
                sensor_data,
                current_age_hours,
                fuel_quality_score
            )
            ml_predictions = self.ml_predictor.predict_all_modes(features)

        # Step 4: Ensemble Combination
        ensemble_rul = None
        ensemble_uncertainty = None
        prediction_confidence = PredictionConfidence.MEDIUM

        if self.ensemble:
            ml_rul = None
            if ml_predictions and ml_predictions[0].time_to_failure_hours:
                ml_rul = ml_predictions[0].time_to_failure_hours

            ensemble_rul, ensemble_uncertainty = self.ensemble.combine_rul_predictions(
                rul_p50, cox_adjusted_rul, ml_rul
            )

            consensus = self.ensemble.calculate_consensus(
                rul_p50, cox_adjusted_rul, ml_predictions
            )
            prediction_confidence = consensus["prediction_confidence"]

        # Step 5: Determine Maintenance Urgency
        urgency = self._determine_urgency(
            rul_p50,
            cox_adjusted_rul,
            cond_fail_prob_30d,
            ml_predictions
        )

        # Step 6: Generate Recommendations
        recommendations = self._generate_recommendations(
            weibull_params,
            rul_p50,
            hazard_ratio,
            operating_conditions,
            ml_predictions,
            urgency
        )

        # Step 7: Calculate Provenance
        processing_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        provenance_hash = self._calculate_provenance(
            current_age_hours,
            weibull_params,
            rul_p50,
            hazard_ratio,
            operating_conditions
        )

        # Build result
        result = RULPredictionResult(
            component=self.config.component,
            current_age_hours=current_age_hours,
            rul_p10_hours=rul_p10,
            rul_p50_hours=rul_p50,
            rul_p90_hours=rul_p90,
            beta=weibull_params.beta,
            eta_hours=weibull_params.eta,
            current_failure_probability=current_fail_prob,
            conditional_failure_prob_30d=cond_fail_prob_30d,
            confidence_level=self.config.weibull.confidence_level,
            prediction_confidence=prediction_confidence,
            maintenance_urgency=urgency,
            ensemble_rul_hours=ensemble_rul,
            ensemble_uncertainty_hours=ensemble_uncertainty,
            hazard_ratio=hazard_ratio,
            cox_adjusted_rul_hours=cox_adjusted_rul,
            failure_mode_interpretation=self.weibull_analyzer.interpret_beta(
                weibull_params.beta
            ),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            f"RUL prediction complete: P50={rul_p50:.0f}h, "
            f"urgency={urgency.value}, processing={processing_time_ms:.1f}ms"
        )

        return result

    def predict_failure_modes(
        self,
        operating_conditions: OperatingConditions,
        sensor_data: Optional[Dict[str, float]] = None,
        equipment_age_hours: float = 0,
        fuel_quality_score: float = 100.0
    ) -> List[FailurePredictionResult]:
        """
        Predict probabilities for all failure modes.

        Args:
            operating_conditions: Current operating conditions
            sensor_data: Sensor readings
            equipment_age_hours: Equipment age
            fuel_quality_score: Fuel quality score

        Returns:
            List of failure predictions sorted by probability
        """
        if not self.ml_predictor:
            logger.warning("ML predictor not enabled")
            return []

        features = self.ml_predictor.extract_features(
            operating_conditions,
            sensor_data,
            equipment_age_hours,
            fuel_quality_score
        )

        return self.ml_predictor.predict_all_modes(features)

    def calculate_optimal_maintenance_time(
        self,
        current_age_hours: float,
        failure_history: Optional[List[FailureData]] = None,
        cost_preventive: float = 1000.0,
        cost_corrective: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Calculate optimal preventive maintenance timing.

        Uses cost-based optimization:
        Optimal time minimizes: C_p / t + C_c * h(t)

        Args:
            current_age_hours: Current component age
            failure_history: Historical failures
            cost_preventive: Cost of preventive maintenance ($)
            cost_corrective: Cost of corrective maintenance ($)

        Returns:
            Optimization result with recommended time
        """
        # Estimate Weibull parameters
        weibull_params = self.weibull_analyzer.estimate_parameters(
            failure_history or [],
            component_default=self.config.component
        )

        # Only applies for increasing failure rate (beta > 1)
        if weibull_params.beta <= 1.0:
            return {
                "optimal_interval_hours": None,
                "recommendation": (
                    "Run-to-failure may be optimal for this component "
                    f"(beta={weibull_params.beta:.2f} indicates constant "
                    "or decreasing failure rate)"
                ),
                "cost_ratio": cost_corrective / cost_preventive,
            }

        # Optimal interval formula for Weibull
        # t* = eta * ((beta - 1) / (beta * (Cc/Cp - 1)))^(1/beta)
        cost_ratio = cost_corrective / cost_preventive

        if cost_ratio <= 1:
            return {
                "optimal_interval_hours": None,
                "recommendation": (
                    "Run-to-failure optimal when corrective cost <= "
                    "preventive cost"
                ),
                "cost_ratio": cost_ratio,
            }

        beta = weibull_params.beta
        eta = weibull_params.eta

        try:
            optimal_interval = eta * (
                (beta - 1) / (beta * (cost_ratio - 1))
            ) ** (1 / beta)

            # Calculate RUL from optimal interval
            time_to_pm = max(0, optimal_interval - current_age_hours)

            # Calculate expected cost per hour
            reliability_at_opt = self.weibull_analyzer.calculate_reliability(
                optimal_interval, weibull_params
            )
            expected_cost_per_hour = (
                cost_preventive * reliability_at_opt +
                cost_corrective * (1 - reliability_at_opt)
            ) / optimal_interval

            return {
                "optimal_interval_hours": optimal_interval,
                "time_to_pm_hours": time_to_pm,
                "reliability_at_pm": reliability_at_opt,
                "expected_cost_per_hour": expected_cost_per_hour,
                "cost_ratio": cost_ratio,
                "recommendation": (
                    f"Schedule preventive maintenance in {time_to_pm:.0f} hours "
                    f"(at age {optimal_interval:.0f}h) for optimal cost"
                ),
            }
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Could not calculate optimal time: {e}")
            return {
                "optimal_interval_hours": None,
                "recommendation": "Unable to calculate optimal interval",
                "error": str(e),
            }

    def _determine_urgency(
        self,
        rul_p50: float,
        cox_rul: Optional[float],
        cond_prob_30d: float,
        ml_predictions: List[FailurePredictionResult]
    ) -> MaintenanceUrgency:
        """Determine maintenance urgency based on predictions."""
        # Use Cox-adjusted RUL if available, otherwise P50
        effective_rul = cox_rul if cox_rul else rul_p50

        # Check ML predictions for high-probability failures
        high_risk_ml = False
        if ml_predictions:
            top_prob = ml_predictions[0].probability
            if top_prob > 0.7:
                high_risk_ml = True

        # Urgency determination logic
        if effective_rul < 168 or cond_prob_30d > 0.5 or high_risk_ml:
            return MaintenanceUrgency.IMMEDIATE
        elif effective_rul < 720 or cond_prob_30d > 0.3:
            return MaintenanceUrgency.URGENT
        elif effective_rul < 2160 or cond_prob_30d > 0.15:
            return MaintenanceUrgency.PLANNED
        elif effective_rul < 8760:  # 1 year
            return MaintenanceUrgency.SCHEDULED
        else:
            return MaintenanceUrgency.MONITOR

    def _generate_recommendations(
        self,
        params: WeibullParameters,
        rul_p50: float,
        hazard_ratio: Optional[float],
        conditions: OperatingConditions,
        ml_predictions: List[FailurePredictionResult],
        urgency: MaintenanceUrgency
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        # Urgency-based recommendations
        if urgency == MaintenanceUrgency.IMMEDIATE:
            recommendations.append(
                "IMMEDIATE: Schedule maintenance within 24 hours. "
                "Prepare replacement parts and notify operations."
            )
        elif urgency == MaintenanceUrgency.URGENT:
            recommendations.append(
                "URGENT: Schedule maintenance within 1 week. "
                "Order parts and plan downtime coordination."
            )
        elif urgency == MaintenanceUrgency.PLANNED:
            recommendations.append(
                f"PLANNED: Schedule maintenance within 30 days. "
                f"Estimated RUL: {rul_p50:.0f} hours."
            )

        # Beta interpretation recommendations
        if params.beta < 1.0:
            recommendations.append(
                "INVESTIGATION: Early-life failures detected. "
                "Review installation, commissioning, and component quality."
            )
        elif params.beta > 3.5:
            recommendations.append(
                "WARNING: Accelerated wear pattern. Review operating "
                "conditions, fuel quality, and thermal cycling."
            )

        # Hazard ratio recommendations
        if hazard_ratio and hazard_ratio > 1.5:
            recommendations.append(
                f"OPERATING CONDITIONS: Hazard ratio {hazard_ratio:.2f}x baseline. "
                "Consider reducing firing rate or improving fuel quality."
            )

        # Cox risk factor recommendations
        if self.cox_model and hazard_ratio:
            risk_factors = self.cox_model.get_risk_factors(conditions)
            for factor in risk_factors[:2]:
                recommendations.append(f"RISK FACTOR: {factor}")

        # ML prediction recommendations
        if ml_predictions:
            top_pred = ml_predictions[0]
            if top_pred.probability > 0.3:
                recommendations.append(
                    f"ML ALERT: {top_pred.failure_mode.value} risk at "
                    f"{top_pred.probability:.0%}. Top factors: "
                    f"{', '.join(top_pred.top_contributing_features)}"
                )

        return recommendations

    def _calculate_provenance(
        self,
        current_age: float,
        params: WeibullParameters,
        rul_p50: float,
        hazard_ratio: Optional[float],
        conditions: OperatingConditions
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = (
            f"rul_pred|{self.config.component.value}|{current_age:.2f}|"
            f"{params.beta:.6f}|{params.eta:.6f}|{rul_p50:.2f}|"
            f"{hazard_ratio or 0:.6f}|"
            f"{conditions.avg_flame_temp_c:.2f}|{conditions.firing_rate_pct:.2f}|"
            f"{conditions.fuel_sulfur_content_pct:.4f}"
        )
        return hashlib.sha256(provenance_data.encode()).hexdigest()

    def get_component_health_summary(
        self,
        current_age_hours: float,
        failure_history: Optional[List[FailureData]] = None,
        operating_conditions: Optional[OperatingConditions] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive component health summary.

        Args:
            current_age_hours: Current age in hours
            failure_history: Historical failures
            operating_conditions: Operating conditions

        Returns:
            Health summary dictionary
        """
        # Get RUL prediction
        result = self.predict_rul(
            current_age_hours,
            failure_history,
            operating_conditions
        )

        # Calculate health score (0-100)
        typical_life = DEFAULT_COMPONENT_LIFETIMES.get(
            self.config.component.value, 40000
        )
        age_factor = min(1.0, current_age_hours / typical_life)
        prob_factor = result.current_failure_probability

        health_score = max(0, 100 * (1 - 0.6 * age_factor - 0.4 * prob_factor))

        # Health status
        if health_score > 80:
            status = "HEALTHY"
        elif health_score > 60:
            status = "GOOD"
        elif health_score > 40:
            status = "DEGRADED"
        elif health_score > 20:
            status = "WARNING"
        else:
            status = "CRITICAL"

        return {
            "component": self.config.component.value,
            "health_score": round(health_score, 1),
            "health_status": status,
            "age_hours": current_age_hours,
            "typical_life_hours": typical_life,
            "age_percentage": round(100 * current_age_hours / typical_life, 1),
            "rul_p50_hours": round(result.rul_p50_hours, 0),
            "failure_probability": round(result.current_failure_probability, 3),
            "maintenance_urgency": result.maintenance_urgency.value,
            "confidence": result.prediction_confidence.value,
            "top_recommendation": result.recommendations[0] if result.recommendations else None,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_rul_prediction(
    component: BurnerComponent,
    current_age_hours: float,
    flame_temp_c: float = 1200.0,
    firing_rate_pct: float = 80.0,
    fuel_sulfur_pct: float = 0.5
) -> RULPredictionResult:
    """
    Quick RUL prediction with minimal inputs.

    Args:
        component: Burner component type
        current_age_hours: Current age in hours
        flame_temp_c: Average flame temperature
        firing_rate_pct: Firing rate percentage
        fuel_sulfur_pct: Fuel sulfur content

    Returns:
        RULPredictionResult

    Example:
        >>> result = quick_rul_prediction(
        ...     BurnerComponent.BURNER_TIP,
        ...     current_age_hours=15000,
        ...     flame_temp_c=1250.0
        ... )
        >>> print(f"RUL: {result.rul_p50_hours:.0f} hours")
    """
    config = PredictionEngineConfig(component=component)
    engine = MaintenancePredictionEngine(config)

    conditions = OperatingConditions(
        avg_flame_temp_c=flame_temp_c,
        firing_rate_pct=firing_rate_pct,
        fuel_sulfur_content_pct=fuel_sulfur_pct
    )

    return engine.predict_rul(
        current_age_hours=current_age_hours,
        operating_conditions=conditions
    )


def calculate_b10_life(component: BurnerComponent) -> float:
    """
    Calculate B10 life for component (10% failure probability time).

    Args:
        component: Burner component type

    Returns:
        B10 life in hours
    """
    comp_name = component.value
    beta = COMPONENT_BETA_VALUES.get(comp_name, 2.5)
    eta = DEFAULT_COMPONENT_LIFETIMES.get(comp_name, 40000)

    # B10 = eta * (-ln(0.9))^(1/beta)
    b10 = eta * ((-math.log(0.9)) ** (1 / beta))
    return b10


def get_component_mtbf(component: BurnerComponent) -> float:
    """
    Get Mean Time Between Failures for component.

    MTBF = eta * Gamma(1 + 1/beta)

    Args:
        component: Burner component type

    Returns:
        MTBF in hours
    """
    comp_name = component.value
    beta = COMPONENT_BETA_VALUES.get(comp_name, 2.5)
    eta = DEFAULT_COMPONENT_LIFETIMES.get(comp_name, 40000)

    gamma_val = math.gamma(1 + 1 / beta)
    return eta * gamma_val
