"""
Scenario Risk Engine - PACK-007 EUDR Professional

This module implements Monte Carlo risk simulation with scenario modeling
for EUDR compliance risk assessment.

Example:
    >>> config = ScenarioRiskConfig(num_simulations=10000)
    >>> engine = ScenarioRiskEngine(config)
    >>> result = engine.run_simulation(risk_inputs, config)
    >>> print(f"VaR 95%: {result.var_at_risk['95%']}")
"""

import hashlib
import json
import logging
import random
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class DistributionType(str, Enum):
    """Statistical distribution types for Monte Carlo simulation."""
    NORMAL = "NORMAL"
    UNIFORM = "UNIFORM"
    TRIANGULAR = "TRIANGULAR"
    LOGNORMAL = "LOGNORMAL"
    BETA = "BETA"


class RiskFactor(str, Enum):
    """Key risk factors for EUDR compliance."""
    DEFORESTATION = "DEFORESTATION"
    SUPPLIER_CERTIFICATION = "SUPPLIER_CERTIFICATION"
    TRACEABILITY_GAP = "TRACEABILITY_GAP"
    COUNTRY_RISK = "COUNTRY_RISK"
    PROTECTED_AREA = "PROTECTED_AREA"
    INDIGENOUS_RIGHTS = "INDIGENOUS_RIGHTS"
    DATA_QUALITY = "DATA_QUALITY"
    REGULATORY_CHANGE = "REGULATORY_CHANGE"
    SUPPLIER_FAILURE = "SUPPLIER_FAILURE"
    CERTIFICATION_REVOCATION = "CERTIFICATION_REVOCATION"


class ScenarioType(str, Enum):
    """Pre-defined scenario types."""
    BASELINE = "BASELINE"
    STRESS_TEST = "STRESS_TEST"
    BEST_CASE = "BEST_CASE"
    WORST_CASE = "WORST_CASE"
    REGULATORY_CHANGE = "REGULATORY_CHANGE"
    SUPPLIER_FAILURE = "SUPPLIER_FAILURE"
    CERTIFICATION_REVOCATION = "CERTIFICATION_REVOCATION"


class ScenarioRiskConfig(BaseModel):
    """Configuration for scenario risk analysis."""

    num_simulations: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo simulations"
    )
    confidence_levels: List[float] = Field(
        default=[0.90, 0.95, 0.99],
        description="Confidence levels for VaR calculation"
    )
    random_seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )
    enable_correlation: bool = Field(
        default=True,
        description="Enable correlation between risk factors"
    )
    time_horizon_days: int = Field(
        default=365,
        ge=1,
        le=3650,
        description="Time horizon for risk projection in days"
    )


class Distribution(BaseModel):
    """Statistical distribution parameters."""

    type: DistributionType = Field(..., description="Distribution type")
    params: Dict[str, float] = Field(..., description="Distribution parameters")

    @validator('params')
    def validate_params(cls, v, values):
        """Validate distribution parameters."""
        dist_type = values.get('type')

        if dist_type == DistributionType.NORMAL:
            if 'mean' not in v or 'std' not in v:
                raise ValueError("Normal distribution requires 'mean' and 'std'")
        elif dist_type == DistributionType.UNIFORM:
            if 'min' not in v or 'max' not in v:
                raise ValueError("Uniform distribution requires 'min' and 'max'")
        elif dist_type == DistributionType.TRIANGULAR:
            if 'min' not in v or 'mode' not in v or 'max' not in v:
                raise ValueError("Triangular distribution requires 'min', 'mode', 'max'")

        return v


class RiskInput(BaseModel):
    """Risk input with probability distribution."""

    factor: RiskFactor = Field(..., description="Risk factor")
    distribution: Distribution = Field(..., description="Probability distribution")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Factor weight")
    baseline_value: float = Field(..., description="Baseline risk value")


class Scenario(BaseModel):
    """Risk scenario with parameter overrides."""

    name: str = Field(..., description="Scenario name")
    scenario_type: ScenarioType = Field(..., description="Scenario type")
    description: str = Field(..., description="Scenario description")
    parameter_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameter overrides for this scenario"
    )
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Scenario probability"
    )


class SimulationResult(BaseModel):
    """Result of Monte Carlo simulation."""

    mean: float = Field(..., description="Mean risk score")
    median: float = Field(..., description="Median risk score")
    std: float = Field(..., ge=0.0, description="Standard deviation")
    percentiles: Dict[str, float] = Field(..., description="Risk percentiles")
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(
        ...,
        description="Confidence intervals"
    )
    var_at_risk: Dict[str, float] = Field(..., description="Value at Risk by confidence level")
    num_simulations: int = Field(..., description="Number of simulations run")
    risk_distribution: List[float] = Field(
        default_factory=list,
        description="Full risk distribution (sampled)"
    )


class SensitivityResult(BaseModel):
    """Sensitivity analysis result for a risk factor."""

    factor_name: str = Field(..., description="Risk factor name")
    impact: float = Field(..., description="Impact on overall risk (-1 to 1)")
    rank: int = Field(..., ge=1, description="Importance ranking")
    correlation: float = Field(..., ge=-1.0, le=1.0, description="Correlation coefficient")


class TrendProjection(BaseModel):
    """Risk trend projection."""

    projected_months: int = Field(..., ge=1, description="Months projected forward")
    baseline_risk: float = Field(..., description="Current baseline risk")
    projected_risk: List[float] = Field(..., description="Projected risk by month")
    trend_direction: str = Field(..., description="INCREASING/DECREASING/STABLE")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Projection confidence")


class PortfolioSimulation(BaseModel):
    """Portfolio-level simulation result."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    entities: List[str] = Field(..., description="Entity IDs in portfolio")
    aggregate_risk: SimulationResult = Field(..., description="Aggregate risk simulation")
    entity_risks: Dict[str, SimulationResult] = Field(..., description="Individual entity risks")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Inter-entity correlation"
    )


class ScenarioRiskEngine:
    """
    Scenario Risk Engine for PACK-007 EUDR Professional.

    This engine implements Monte Carlo risk simulation with scenario modeling
    for comprehensive EUDR compliance risk assessment. It follows GreenLang's
    zero-hallucination principle by using deterministic statistical methods
    with reproducible random seeds.

    Attributes:
        config: Engine configuration
        scenarios: Pre-defined risk scenarios

    Example:
        >>> config = ScenarioRiskConfig(num_simulations=10000)
        >>> engine = ScenarioRiskEngine(config)
        >>> result = engine.run_simulation(risk_inputs, config)
        >>> assert result.num_simulations == 10000
    """

    def __init__(self, config: ScenarioRiskConfig):
        """Initialize Scenario Risk Engine."""
        self.config = config

        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)

        # Initialize pre-defined scenarios
        self.scenarios = self._initialize_scenarios()

        logger.info(f"Initialized ScenarioRiskEngine with {config.num_simulations} simulations")

    def run_simulation(
        self,
        risk_inputs: List[RiskInput],
        config: Optional[ScenarioRiskConfig] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation on risk inputs.

        Args:
            risk_inputs: List of risk inputs with distributions
            config: Optional config override

        Returns:
            SimulationResult with statistical summary

        Raises:
            ValueError: If risk inputs are invalid
        """
        cfg = config or self.config

        if not risk_inputs:
            raise ValueError("Risk inputs cannot be empty")

        logger.info(f"Running {cfg.num_simulations} Monte Carlo simulations")

        # Run simulations
        risk_scores = []
        for i in range(cfg.num_simulations):
            # Sample from each risk factor distribution
            factor_values = []
            for risk_input in risk_inputs:
                value = self._sample_distribution(risk_input.distribution)
                weighted_value = value * risk_input.weight
                factor_values.append(weighted_value)

            # Calculate aggregate risk score (weighted average)
            total_weight = sum(r.weight for r in risk_inputs)
            risk_score = sum(factor_values) / total_weight if total_weight > 0 else 0.0

            risk_scores.append(risk_score)

            if (i + 1) % 1000 == 0:
                logger.debug(f"Completed {i + 1}/{cfg.num_simulations} simulations")

        # Calculate statistics
        risk_scores.sort()

        mean = sum(risk_scores) / len(risk_scores)
        median = risk_scores[len(risk_scores) // 2]

        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in risk_scores) / len(risk_scores)
        std = variance ** 0.5

        # Calculate percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            idx = int(len(risk_scores) * p / 100)
            percentiles[f"p{p}"] = risk_scores[idx]

        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in cfg.confidence_levels:
            lower_idx = int(len(risk_scores) * (1 - conf_level) / 2)
            upper_idx = int(len(risk_scores) * (1 + conf_level) / 2)
            confidence_intervals[f"{conf_level:.2f}"] = (
                risk_scores[lower_idx],
                risk_scores[upper_idx]
            )

        # Calculate Value at Risk (VaR)
        var_at_risk = {}
        for conf_level in cfg.confidence_levels:
            var_idx = int(len(risk_scores) * conf_level)
            var_at_risk[f"{conf_level:.2f}"] = risk_scores[var_idx]

        # Sample distribution for storage (every 10th value)
        sampled_distribution = risk_scores[::10]

        return SimulationResult(
            mean=mean,
            median=median,
            std=std,
            percentiles=percentiles,
            confidence_intervals=confidence_intervals,
            var_at_risk=var_at_risk,
            num_simulations=cfg.num_simulations,
            risk_distribution=sampled_distribution
        )

    def sensitivity_analysis(self, risk_inputs: List[RiskInput]) -> List[SensitivityResult]:
        """
        Perform sensitivity analysis on risk factors.

        Args:
            risk_inputs: List of risk inputs

        Returns:
            List of SensitivityResult ranked by impact
        """
        logger.info(f"Running sensitivity analysis on {len(risk_inputs)} factors")

        # Run baseline simulation
        baseline_result = self.run_simulation(risk_inputs)
        baseline_mean = baseline_result.mean

        sensitivity_results = []

        for i, risk_input in enumerate(risk_inputs):
            # Create modified inputs with this factor perturbed
            modified_inputs = risk_inputs.copy()

            # Increase this factor by 10%
            modified_factor = risk_input.copy()
            if modified_factor.distribution.type == DistributionType.NORMAL:
                modified_factor.distribution.params["mean"] *= 1.1
            elif modified_factor.distribution.type == DistributionType.UNIFORM:
                modified_factor.distribution.params["max"] *= 1.1

            modified_inputs[i] = modified_factor

            # Run modified simulation
            modified_result = self.run_simulation(modified_inputs)

            # Calculate impact
            impact = (modified_result.mean - baseline_mean) / baseline_mean if baseline_mean > 0 else 0.0

            # Calculate correlation (simplified)
            correlation = impact / 0.1  # Normalized by 10% perturbation

            sensitivity_results.append(
                SensitivityResult(
                    factor_name=risk_input.factor.value,
                    impact=impact,
                    rank=0,  # Will be set after sorting
                    correlation=min(1.0, max(-1.0, correlation))
                )
            )

        # Rank by absolute impact
        sensitivity_results.sort(key=lambda x: abs(x.impact), reverse=True)
        for rank, result in enumerate(sensitivity_results, 1):
            result.rank = rank

        logger.info(f"Sensitivity analysis complete. Top factor: {sensitivity_results[0].factor_name}")

        return sensitivity_results

    def stress_test(
        self,
        risk_inputs: List[RiskInput],
        scenarios: Optional[List[Scenario]] = None
    ) -> Dict[str, SimulationResult]:
        """
        Perform stress testing under different scenarios.

        Args:
            risk_inputs: Base risk inputs
            scenarios: Scenarios to test (uses pre-defined if None)

        Returns:
            Dictionary mapping scenario name to simulation result
        """
        test_scenarios = scenarios or list(self.scenarios.values())

        logger.info(f"Running stress test across {len(test_scenarios)} scenarios")

        results = {}

        for scenario in test_scenarios:
            # Apply scenario overrides
            modified_inputs = self._apply_scenario(risk_inputs, scenario)

            # Run simulation
            result = self.run_simulation(modified_inputs)
            results[scenario.name] = result

            logger.info(
                f"Scenario '{scenario.name}': Mean risk = {result.mean:.3f}, "
                f"VaR(95%) = {result.var_at_risk.get('0.95', 0):.3f}"
            )

        return results

    def calculate_var(self, simulations: List[float], confidence: float) -> float:
        """
        Calculate Value at Risk at given confidence level.

        Args:
            simulations: List of simulated risk values
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR value
        """
        if not simulations:
            return 0.0

        sorted_sims = sorted(simulations)
        var_idx = int(len(sorted_sims) * confidence)

        return sorted_sims[var_idx] if var_idx < len(sorted_sims) else sorted_sims[-1]

    def project_risk_trend(
        self,
        historical_data: List[Tuple[date, float]],
        months_forward: int
    ) -> TrendProjection:
        """
        Project risk trend based on historical data.

        Args:
            historical_data: List of (date, risk_score) tuples
            months_forward: Number of months to project

        Returns:
            TrendProjection with forecasted risk levels
        """
        if not historical_data or len(historical_data) < 2:
            raise ValueError("Need at least 2 historical data points")

        logger.info(f"Projecting risk trend {months_forward} months forward")

        # Sort by date
        historical_data.sort(key=lambda x: x[0])

        # Extract risk scores
        risk_scores = [score for _, score in historical_data]
        baseline_risk = risk_scores[-1]

        # Calculate simple linear trend
        n = len(risk_scores)
        x_values = list(range(n))
        y_values = risk_scores

        # Linear regression (simplified)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        else:
            slope = 0.0
            intercept = y_mean

        # Project forward
        projected_risk = []
        for month in range(1, months_forward + 1):
            projected_x = n + month
            projected_value = slope * projected_x + intercept
            # Add some uncertainty (increases with distance)
            uncertainty = 0.05 * month * abs(slope)
            projected_value += random.uniform(-uncertainty, uncertainty)
            projected_risk.append(max(0.0, min(1.0, projected_value)))

        # Determine trend direction
        if abs(slope) < 0.001:
            trend_direction = "STABLE"
        elif slope > 0:
            trend_direction = "INCREASING"
        else:
            trend_direction = "DECREASING"

        # Calculate confidence (decreases with projection distance)
        confidence = max(0.3, 1.0 - (months_forward * 0.05))

        return TrendProjection(
            projected_months=months_forward,
            baseline_risk=baseline_risk,
            projected_risk=projected_risk,
            trend_direction=trend_direction,
            confidence=confidence
        )

    def batch_simulation(
        self,
        portfolio: Dict[str, List[RiskInput]]
    ) -> Dict[str, SimulationResult]:
        """
        Run simulations on portfolio of entities.

        Args:
            portfolio: Dictionary mapping entity_id to risk_inputs

        Returns:
            Dictionary mapping entity_id to SimulationResult
        """
        logger.info(f"Running batch simulation for {len(portfolio)} entities")

        results = {}

        for entity_id, risk_inputs in portfolio.items():
            try:
                result = self.run_simulation(risk_inputs)
                results[entity_id] = result

                logger.debug(f"Entity {entity_id}: Mean risk = {result.mean:.3f}")

            except Exception as e:
                logger.error(f"Simulation failed for entity {entity_id}: {str(e)}")
                continue

        logger.info(f"Batch simulation complete: {len(results)}/{len(portfolio)} successful")

        return results

    # Helper methods

    def _initialize_scenarios(self) -> Dict[str, Scenario]:
        """Initialize pre-defined scenarios."""
        scenarios = {}

        # Baseline scenario (no changes)
        scenarios["baseline"] = Scenario(
            name="Baseline",
            scenario_type=ScenarioType.BASELINE,
            description="Current baseline conditions",
            parameter_overrides={},
            probability=0.50
        )

        # Stress test scenario (all risks +50%)
        scenarios["stress_test"] = Scenario(
            name="Stress Test",
            scenario_type=ScenarioType.STRESS_TEST,
            description="All risk factors increased by 50%",
            parameter_overrides={
                "risk_multiplier": 1.5
            },
            probability=0.05
        )

        # Best case scenario (all risks -30%)
        scenarios["best_case"] = Scenario(
            name="Best Case",
            scenario_type=ScenarioType.BEST_CASE,
            description="All risk factors decreased by 30%",
            parameter_overrides={
                "risk_multiplier": 0.7
            },
            probability=0.20
        )

        # Worst case scenario (all risks +100%)
        scenarios["worst_case"] = Scenario(
            name="Worst Case",
            scenario_type=ScenarioType.WORST_CASE,
            description="All risk factors doubled",
            parameter_overrides={
                "risk_multiplier": 2.0
            },
            probability=0.01
        )

        # Regulatory change scenario
        scenarios["regulatory_change"] = Scenario(
            name="Regulatory Change",
            scenario_type=ScenarioType.REGULATORY_CHANGE,
            description="Major regulatory requirements change",
            parameter_overrides={
                RiskFactor.REGULATORY_CHANGE.value: 0.8,
                RiskFactor.DATA_QUALITY.value: 0.6
            },
            probability=0.10
        )

        # Supplier failure scenario
        scenarios["supplier_failure"] = Scenario(
            name="Supplier Failure",
            scenario_type=ScenarioType.SUPPLIER_FAILURE,
            description="Major supplier compliance failure",
            parameter_overrides={
                RiskFactor.SUPPLIER_FAILURE.value: 0.9,
                RiskFactor.TRACEABILITY_GAP.value: 0.7
            },
            probability=0.08
        )

        # Certification revocation scenario
        scenarios["certification_revocation"] = Scenario(
            name="Certification Revocation",
            scenario_type=ScenarioType.CERTIFICATION_REVOCATION,
            description="Key supplier certifications revoked",
            parameter_overrides={
                RiskFactor.CERTIFICATION_REVOCATION.value: 0.9,
                RiskFactor.SUPPLIER_CERTIFICATION.value: 0.8
            },
            probability=0.06
        )

        return scenarios

    def _sample_distribution(self, distribution: Distribution) -> float:
        """
        Sample a value from the specified distribution.

        Args:
            distribution: Distribution specification

        Returns:
            Sampled value
        """
        params = distribution.params

        if distribution.type == DistributionType.NORMAL:
            mean = params["mean"]
            std = params["std"]
            # Box-Muller transform for normal distribution
            u1 = random.random()
            u2 = random.random()
            z = (-2.0 * (u1 if u1 > 0 else 0.0001)) ** 0.5 * ((2.0 * 3.14159 * u2) ** 0.5)
            return mean + std * z

        elif distribution.type == DistributionType.UNIFORM:
            return random.uniform(params["min"], params["max"])

        elif distribution.type == DistributionType.TRIANGULAR:
            return self._triangular(params["min"], params["mode"], params["max"])

        elif distribution.type == DistributionType.LOGNORMAL:
            # Sample from normal then exponentiate
            mean = params.get("mean", 0.0)
            std = params.get("std", 1.0)
            u1 = random.random()
            u2 = random.random()
            z = (-2.0 * (u1 if u1 > 0 else 0.0001)) ** 0.5
            normal_sample = mean + std * z
            return 2.71828 ** normal_sample

        elif distribution.type == DistributionType.BETA:
            # Simplified beta distribution (using uniform as approximation)
            alpha = params.get("alpha", 2.0)
            beta = params.get("beta", 2.0)
            # Very simplified - proper beta would need more complex sampling
            return random.random()

        else:
            return random.random()

    def _triangular(self, low: float, mode: float, high: float) -> float:
        """Sample from triangular distribution."""
        u = random.random()
        c = (mode - low) / (high - low)

        if u < c:
            return low + ((high - low) * (mode - low) * u) ** 0.5
        else:
            return high - ((high - low) * (high - mode) * (1 - u)) ** 0.5

    def _apply_scenario(
        self,
        risk_inputs: List[RiskInput],
        scenario: Scenario
    ) -> List[RiskInput]:
        """
        Apply scenario parameter overrides to risk inputs.

        Args:
            risk_inputs: Base risk inputs
            scenario: Scenario to apply

        Returns:
            Modified risk inputs
        """
        modified_inputs = []

        for risk_input in risk_inputs:
            modified = risk_input.copy(deep=True)

            # Check for global multiplier
            if "risk_multiplier" in scenario.parameter_overrides:
                multiplier = scenario.parameter_overrides["risk_multiplier"]

                # Apply to distribution parameters
                if modified.distribution.type == DistributionType.NORMAL:
                    modified.distribution.params["mean"] *= multiplier
                elif modified.distribution.type == DistributionType.UNIFORM:
                    modified.distribution.params["min"] *= multiplier
                    modified.distribution.params["max"] *= multiplier
                elif modified.distribution.type == DistributionType.TRIANGULAR:
                    modified.distribution.params["min"] *= multiplier
                    modified.distribution.params["mode"] *= multiplier
                    modified.distribution.params["max"] *= multiplier

            # Check for factor-specific overrides
            factor_key = risk_input.factor.value
            if factor_key in scenario.parameter_overrides:
                override_value = scenario.parameter_overrides[factor_key]

                # Replace distribution with fixed value (high certainty)
                modified.distribution = Distribution(
                    type=DistributionType.NORMAL,
                    params={"mean": override_value, "std": 0.05}
                )

            modified_inputs.append(modified)

        return modified_inputs
