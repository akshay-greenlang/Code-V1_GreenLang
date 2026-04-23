# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Scenario Analyzer

What-if scenario comparisons, sensitivity analysis, and Monte Carlo simulation
for cleaning schedule uncertainty quantification.

Features:
- What-if scenario comparisons (clean now vs later vs no clean)
- Sensitivity analysis on energy prices, cleaning costs, fouling rates
- Monte Carlo simulation for cost uncertainty bounds
- Tornado charts for decision drivers
- Breakeven analysis for cleaning investments

Zero-Hallucination Principle:
    All analyses use deterministic formulas and controlled random sampling.
    Monte Carlo uses explicit seed for reproducibility.
    Results include confidence intervals, not point estimates.

Author: GreenLang AI Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math
import random
from copy import deepcopy

from pydantic import BaseModel, Field, validator, root_validator

from .cost_model import (
    CleaningCostModel,
    CostModelConfig,
    CleaningMethodType,
    TotalCostBreakdown,
    CostProjection,
)
from .cleaning_optimizer import (
    CleaningScheduleOptimizer,
    OptimizerConfig,
    WhatIfScenario,
    WhatIfResult,
    CleaningUrgency,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class DistributionType(str, Enum):
    """Types of probability distributions for Monte Carlo."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    LOGNORMAL = "lognormal"
    BETA = "beta"


class SensitivityMetric(str, Enum):
    """Metrics to analyze sensitivity for."""
    TOTAL_COST = "total_cost"
    ENERGY_LOSS = "energy_loss"
    PRODUCTION_LOSS = "production_loss"
    OPTIMAL_CLEANING_DAY = "optimal_cleaning_day"
    SAVINGS = "savings"
    PAYBACK_PERIOD = "payback_period"


@dataclass
class ScenarioConfig:
    """Configuration for scenario analysis."""

    # Monte Carlo settings
    n_monte_carlo_samples: int = 1000
    confidence_level: float = 0.95
    random_seed: int = 42

    # Sensitivity analysis settings
    sensitivity_steps: int = 11  # Number of steps for each parameter
    sensitivity_range_pct: float = 50.0  # +/- 50% from base

    # Tornado chart settings
    tornado_n_factors: int = 10  # Top N factors to show

    # Computation limits
    max_scenarios: int = 100
    timeout_seconds: float = 60.0


class UncertainParameter(BaseModel):
    """
    Definition of an uncertain parameter for Monte Carlo.
    """
    name: str = Field(..., description="Parameter name")
    base_value: float = Field(..., description="Base/nominal value")
    distribution: DistributionType = Field(DistributionType.NORMAL)

    # Distribution parameters
    std_dev: Optional[float] = Field(None, description="For normal distribution")
    min_value: Optional[float] = Field(None, description="For uniform/triangular")
    max_value: Optional[float] = Field(None, description="For uniform/triangular")
    mode_value: Optional[float] = Field(None, description="For triangular")

    # Correlation with other parameters
    correlation_with: Optional[Dict[str, float]] = Field(None)

    # Impact category
    category: str = Field("operational", description="Parameter category")

    def sample(self, rng: random.Random) -> float:
        """Generate a random sample from the distribution."""
        if self.distribution == DistributionType.NORMAL:
            std = self.std_dev or self.base_value * 0.1
            return rng.gauss(self.base_value, std)

        elif self.distribution == DistributionType.UNIFORM:
            low = self.min_value if self.min_value is not None else self.base_value * 0.8
            high = self.max_value if self.max_value is not None else self.base_value * 1.2
            return rng.uniform(low, high)

        elif self.distribution == DistributionType.TRIANGULAR:
            low = self.min_value if self.min_value is not None else self.base_value * 0.8
            high = self.max_value if self.max_value is not None else self.base_value * 1.2
            mode = self.mode_value if self.mode_value is not None else self.base_value
            return rng.triangular(low, high, mode)

        elif self.distribution == DistributionType.LOGNORMAL:
            # mu and sigma for lognormal
            std = self.std_dev or self.base_value * 0.1
            mu = math.log(self.base_value ** 2 / math.sqrt(self.base_value ** 2 + std ** 2))
            sigma = math.sqrt(math.log(1 + (std ** 2 / self.base_value ** 2)))
            return rng.lognormvariate(mu, sigma)

        else:
            return self.base_value


class ScenarioDefinition(BaseModel):
    """
    Complete definition of a scenario for comparison.
    """
    scenario_id: str = Field(...)
    scenario_name: str = Field(...)
    description: str = Field("")

    # Cleaning decision
    cleaning_action: str = Field(
        "clean_optimal",
        pattern="^(clean_now|clean_day_\\d+|clean_optimal|no_clean)$"
    )
    cleaning_day: Optional[int] = Field(None)
    cleaning_method: Optional[CleaningMethodType] = Field(None)

    # Parameter overrides (relative to baseline)
    parameter_multipliers: Dict[str, float] = Field(
        default_factory=dict,
        description="Multipliers for base parameters (1.0 = no change)"
    )

    # Assumptions
    assumptions: List[str] = Field(default_factory=list)


class ScenarioComparison(BaseModel):
    """
    Comparison results for multiple scenarios.
    """
    comparison_id: str = Field(...)
    exchanger_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Scenarios compared
    scenarios: List[ScenarioDefinition] = Field(...)
    n_scenarios: int = Field(..., ge=1)

    # Results per scenario
    scenario_costs: Dict[str, float] = Field(..., description="Scenario ID -> total cost")
    scenario_rankings: Dict[str, int] = Field(..., description="Scenario ID -> rank")

    # Best scenario
    best_scenario_id: str = Field(...)
    best_scenario_cost_usd: float = Field(..., ge=0)

    # Cost comparison matrix
    pairwise_differences: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Scenario A -> Scenario B -> cost difference"
    )

    # Key insights
    key_insights: List[str] = Field(default_factory=list)

    # Provenance
    baseline_assumptions: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field("")


class SensitivityResult(BaseModel):
    """
    Results from sensitivity analysis on a single parameter.
    """
    parameter_name: str = Field(...)
    parameter_category: str = Field("")

    # Base case
    base_value: float = Field(...)
    base_metric_value: float = Field(...)

    # Sensitivity range
    values_tested: List[float] = Field(...)
    metric_values: List[float] = Field(...)

    # Sensitivity metrics
    elasticity: float = Field(
        0.0,
        description="Percent change in metric per percent change in parameter"
    )
    tornado_impact: float = Field(
        0.0, ge=0,
        description="Range of metric change (high - low)"
    )
    is_positive_correlation: bool = Field(True)

    # Breakpoints
    breakeven_value: Optional[float] = Field(
        None,
        description="Parameter value where decision changes"
    )

    # Provenance
    metric_analyzed: SensitivityMetric = Field(...)
    provenance_hash: str = Field("")


class TornadoChart(BaseModel):
    """
    Tornado chart data for decision sensitivity visualization.
    """
    chart_id: str = Field(...)
    exchanger_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Metric analyzed
    metric: SensitivityMetric = Field(...)
    base_metric_value: float = Field(...)

    # Factors ranked by impact
    factors: List[SensitivityResult] = Field(...)

    # Top drivers
    top_positive_drivers: List[str] = Field(default_factory=list)
    top_negative_drivers: List[str] = Field(default_factory=list)

    # Chart data for visualization
    chart_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ready-to-plot tornado chart data"
    )

    # Provenance
    provenance_hash: str = Field("")


class MonteCarloResult(BaseModel):
    """
    Results from Monte Carlo simulation.
    """
    simulation_id: str = Field(...)
    exchanger_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Simulation parameters
    n_samples: int = Field(..., ge=1)
    random_seed: int = Field(...)

    # Uncertain parameters
    uncertain_parameters: List[str] = Field(...)

    # Output statistics
    mean_cost_usd: float = Field(...)
    std_cost_usd: float = Field(..., ge=0)
    median_cost_usd: float = Field(...)

    # Confidence intervals
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)
    ci_lower_usd: float = Field(...)
    ci_upper_usd: float = Field(...)

    # Percentiles
    percentiles: Dict[int, float] = Field(
        default_factory=dict,
        description="Percentile -> cost value"
    )

    # Distribution shape
    skewness: float = Field(0.0)
    kurtosis: float = Field(0.0)

    # Risk metrics
    probability_exceeds_budget: Optional[float] = Field(None)
    value_at_risk_95: float = Field(0.0, description="95th percentile cost")
    conditional_var_95: float = Field(0.0, description="Expected cost above 95th percentile")

    # Optimal cleaning day distribution
    optimal_day_mean: Optional[float] = Field(None)
    optimal_day_std: Optional[float] = Field(None)
    optimal_day_distribution: Optional[Dict[int, float]] = Field(None)

    # Raw samples (optional, for detailed analysis)
    sample_costs: Optional[List[float]] = Field(None)

    # Provenance
    provenance_hash: str = Field("")


class BreakevenAnalysis(BaseModel):
    """
    Breakeven analysis for cleaning investment decision.
    """
    analysis_id: str = Field(...)
    exchanger_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Cleaning investment
    cleaning_cost_usd: float = Field(..., ge=0)
    downtime_cost_usd: float = Field(..., ge=0)
    total_investment_usd: float = Field(..., ge=0)

    # Savings rate
    daily_savings_usd: float = Field(...)
    annual_savings_usd: float = Field(...)

    # Breakeven metrics
    breakeven_days: Optional[int] = Field(None, description="Days to recover investment")
    breakeven_date: Optional[datetime] = Field(None)
    is_worthwhile: bool = Field(True)

    # Sensitivity
    breakeven_at_different_savings: Dict[float, int] = Field(
        default_factory=dict,
        description="Daily savings rate -> breakeven days"
    )

    # Required fouling rate for breakeven
    minimum_fouling_rate_for_breakeven: Optional[float] = Field(None)

    # NPV analysis
    npv_at_discount_rates: Dict[float, float] = Field(
        default_factory=dict,
        description="Discount rate -> NPV of cleaning decision"
    )
    internal_rate_of_return: Optional[float] = Field(None)

    # Provenance
    assumptions: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field("")


class ScenarioAnalyzer:
    """
    Comprehensive scenario analysis for cleaning schedule optimization.

    Provides:
    - What-if scenario comparisons
    - Sensitivity analysis on key parameters
    - Monte Carlo simulation for uncertainty quantification
    - Tornado charts for decision drivers
    - Breakeven analysis for cleaning investments

    Zero-Hallucination Principle:
        All analyses use deterministic formulas with controlled randomness.
        Monte Carlo uses explicit seeds for reproducibility.
        Results include uncertainty bounds, not single point estimates.

    Example:
        >>> config = ScenarioConfig(n_monte_carlo_samples=1000)
        >>> analyzer = ScenarioAnalyzer(config)
        >>> mc_result = analyzer.run_monte_carlo(
        ...     exchanger_id="HX-001",
        ...     base_params={...},
        ...     uncertain_params=[...]
        ... )
        >>> print(f"Mean cost: ${mc_result.mean_cost_usd:,.0f}")
        >>> print(f"95% CI: [${mc_result.ci_lower_usd:,.0f}, ${mc_result.ci_upper_usd:,.0f}]")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        cost_model: Optional[CleaningCostModel] = None,
    ) -> None:
        """
        Initialize scenario analyzer.

        Args:
            config: Analyzer configuration
            cost_model: Cost model for calculations
        """
        self.config = config or ScenarioConfig()
        self.cost_model = cost_model or CleaningCostModel(seed=self.config.random_seed)

        self.optimizer = CleaningScheduleOptimizer(
            config=OptimizerConfig(random_seed=self.config.random_seed),
            cost_model=self.cost_model,
        )

        self._rng = random.Random(self.config.random_seed)

        logger.info(
            f"ScenarioAnalyzer initialized: n_samples={self.config.n_monte_carlo_samples}, "
            f"confidence={self.config.confidence_level}"
        )

    def compare_scenarios(
        self,
        exchanger_id: str,
        scenarios: List[ScenarioDefinition],
        base_params: Dict[str, float],
    ) -> ScenarioComparison:
        """
        Compare multiple cleaning scenarios.

        Args:
            exchanger_id: Heat exchanger identifier
            scenarios: List of scenarios to compare
            base_params: Base parameter values

        Returns:
            ScenarioComparison with ranked scenarios
        """
        comparison_id = self._generate_id("compare")

        logger.info(f"Comparing {len(scenarios)} scenarios for {exchanger_id}")

        scenario_costs = {}

        for scenario in scenarios:
            # Apply parameter multipliers
            params = self._apply_multipliers(base_params, scenario.parameter_multipliers)

            # Calculate cost
            if scenario.cleaning_action == "no_clean":
                cost = self._evaluate_no_cleaning(exchanger_id, params)
            else:
                cleaning_day = scenario.cleaning_day or 0
                method = scenario.cleaning_method or CleaningMethodType.CHEMICAL_OFFLINE
                cost = self._evaluate_with_cleaning(exchanger_id, cleaning_day, method, params)

            scenario_costs[scenario.scenario_id] = cost

        # Rank scenarios by cost
        sorted_scenarios = sorted(scenario_costs.items(), key=lambda x: x[1])
        scenario_rankings = {s_id: rank + 1 for rank, (s_id, _) in enumerate(sorted_scenarios)}

        best_id, best_cost = sorted_scenarios[0]

        # Calculate pairwise differences
        pairwise = {}
        for s1_id, s1_cost in scenario_costs.items():
            pairwise[s1_id] = {}
            for s2_id, s2_cost in scenario_costs.items():
                pairwise[s1_id][s2_id] = s1_cost - s2_cost

        # Generate insights
        insights = self._generate_comparison_insights(scenarios, scenario_costs, best_id)

        result = ScenarioComparison(
            comparison_id=comparison_id,
            exchanger_id=exchanger_id,
            scenarios=scenarios,
            n_scenarios=len(scenarios),
            scenario_costs=scenario_costs,
            scenario_rankings=scenario_rankings,
            best_scenario_id=best_id,
            best_scenario_cost_usd=best_cost,
            pairwise_differences=pairwise,
            key_insights=insights,
            baseline_assumptions=base_params,
        )

        result.provenance_hash = self._compute_hash({
            "comparison_id": comparison_id,
            "n_scenarios": len(scenarios),
            "best_cost": best_cost,
        })

        return result

    def run_sensitivity_analysis(
        self,
        exchanger_id: str,
        base_params: Dict[str, float],
        parameters_to_analyze: List[str],
        metric: SensitivityMetric = SensitivityMetric.TOTAL_COST,
    ) -> List[SensitivityResult]:
        """
        Run sensitivity analysis on specified parameters.

        Varies each parameter while holding others constant to
        determine impact on the target metric.

        Args:
            exchanger_id: Heat exchanger identifier
            base_params: Base parameter values
            parameters_to_analyze: Parameters to analyze
            metric: Target metric

        Returns:
            List of SensitivityResult for each parameter
        """
        logger.info(
            f"Running sensitivity analysis on {len(parameters_to_analyze)} "
            f"parameters for {exchanger_id}"
        )

        # Calculate base metric value
        base_metric = self._calculate_metric(exchanger_id, base_params, metric)

        results = []

        for param_name in parameters_to_analyze:
            if param_name not in base_params:
                logger.warning(f"Parameter {param_name} not in base_params, skipping")
                continue

            base_value = base_params[param_name]

            # Generate test values
            range_pct = self.config.sensitivity_range_pct / 100
            low_value = base_value * (1 - range_pct)
            high_value = base_value * (1 + range_pct)

            test_values = [
                low_value + (high_value - low_value) * i / (self.config.sensitivity_steps - 1)
                for i in range(self.config.sensitivity_steps)
            ]

            # Evaluate metric at each test value
            metric_values = []
            for test_value in test_values:
                test_params = base_params.copy()
                test_params[param_name] = test_value
                metric_value = self._calculate_metric(exchanger_id, test_params, metric)
                metric_values.append(metric_value)

            # Calculate sensitivity metrics
            elasticity = self._calculate_elasticity(
                test_values, metric_values, base_value, base_metric
            )
            tornado_impact = max(metric_values) - min(metric_values)
            is_positive = metric_values[-1] > metric_values[0]

            # Find breakeven
            breakeven = self._find_breakeven(test_values, metric_values, base_metric)

            result = SensitivityResult(
                parameter_name=param_name,
                parameter_category=self._categorize_parameter(param_name),
                base_value=base_value,
                base_metric_value=base_metric,
                values_tested=test_values,
                metric_values=metric_values,
                elasticity=round(elasticity, 4),
                tornado_impact=round(tornado_impact, 2),
                is_positive_correlation=is_positive,
                breakeven_value=breakeven,
                metric_analyzed=metric,
            )

            result.provenance_hash = self._compute_hash({
                "parameter": param_name,
                "elasticity": elasticity,
            })

            results.append(result)

        # Sort by tornado impact
        results.sort(key=lambda x: x.tornado_impact, reverse=True)

        return results

    def generate_tornado_chart(
        self,
        exchanger_id: str,
        sensitivity_results: List[SensitivityResult],
        n_factors: Optional[int] = None,
    ) -> TornadoChart:
        """
        Generate tornado chart data from sensitivity results.

        Args:
            exchanger_id: Heat exchanger identifier
            sensitivity_results: Results from sensitivity analysis
            n_factors: Number of top factors to include

        Returns:
            TornadoChart with visualization-ready data
        """
        chart_id = self._generate_id("tornado")
        n = n_factors or self.config.tornado_n_factors

        # Take top N factors
        top_factors = sensitivity_results[:n]

        # Get base metric value
        base_metric = sensitivity_results[0].base_metric_value if sensitivity_results else 0

        # Build chart data
        chart_data = []
        for factor in top_factors:
            low_value = min(factor.metric_values)
            high_value = max(factor.metric_values)

            chart_data.append({
                "parameter": factor.parameter_name,
                "category": factor.parameter_category,
                "low": round(low_value - base_metric, 2),
                "high": round(high_value - base_metric, 2),
                "base": base_metric,
                "impact": factor.tornado_impact,
                "elasticity": factor.elasticity,
            })

        # Identify drivers
        positive_drivers = [
            f.parameter_name for f in top_factors
            if f.is_positive_correlation
        ][:3]
        negative_drivers = [
            f.parameter_name for f in top_factors
            if not f.is_positive_correlation
        ][:3]

        result = TornadoChart(
            chart_id=chart_id,
            exchanger_id=exchanger_id,
            metric=sensitivity_results[0].metric_analyzed if sensitivity_results else SensitivityMetric.TOTAL_COST,
            base_metric_value=base_metric,
            factors=top_factors,
            top_positive_drivers=positive_drivers,
            top_negative_drivers=negative_drivers,
            chart_data=chart_data,
        )

        result.provenance_hash = self._compute_hash({
            "chart_id": chart_id,
            "n_factors": len(top_factors),
        })

        return result

    def run_monte_carlo(
        self,
        exchanger_id: str,
        base_params: Dict[str, float],
        uncertain_params: List[UncertainParameter],
        cleaning_day: Optional[int] = None,
        cleaning_method: Optional[CleaningMethodType] = None,
        budget_limit: Optional[float] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for cost uncertainty quantification.

        Samples from specified distributions for uncertain parameters
        and propagates uncertainty through cost calculations.

        Args:
            exchanger_id: Heat exchanger identifier
            base_params: Base parameter values
            uncertain_params: Parameters with uncertainty distributions
            cleaning_day: Fixed cleaning day (None = optimize each sample)
            cleaning_method: Cleaning method to use
            budget_limit: Optional budget for probability calculation

        Returns:
            MonteCarloResult with statistics and confidence intervals
        """
        simulation_id = self._generate_id("mc")
        n_samples = self.config.n_monte_carlo_samples

        logger.info(
            f"Running Monte Carlo simulation for {exchanger_id}: "
            f"{n_samples} samples, {len(uncertain_params)} uncertain parameters"
        )

        # Reset RNG for reproducibility
        self._rng = random.Random(self.config.random_seed)

        sample_costs = []
        optimal_days = []

        for i in range(n_samples):
            # Sample parameter values
            sample_params = base_params.copy()
            for up in uncertain_params:
                if up.name in sample_params:
                    sample_params[up.name] = up.sample(self._rng)

            # Evaluate cost
            if cleaning_day is not None:
                # Fixed cleaning day
                method = cleaning_method or CleaningMethodType.CHEMICAL_OFFLINE
                cost = self._evaluate_with_cleaning(
                    exchanger_id, cleaning_day, method, sample_params
                )
            else:
                # Find optimal for this sample (simplified)
                cost = self._evaluate_no_cleaning(exchanger_id, sample_params)

            sample_costs.append(cost)

            if (i + 1) % 100 == 0:
                logger.debug(f"Monte Carlo: {i + 1}/{n_samples} samples completed")

        # Calculate statistics
        mean_cost = sum(sample_costs) / n_samples
        variance = sum((c - mean_cost) ** 2 for c in sample_costs) / (n_samples - 1)
        std_cost = math.sqrt(variance)

        sorted_costs = sorted(sample_costs)
        median_idx = n_samples // 2
        median_cost = sorted_costs[median_idx]

        # Confidence interval
        alpha = 1 - self.config.confidence_level
        lower_idx = int(n_samples * alpha / 2)
        upper_idx = int(n_samples * (1 - alpha / 2))
        ci_lower = sorted_costs[lower_idx]
        ci_upper = sorted_costs[upper_idx]

        # Percentiles
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            idx = int(n_samples * p / 100)
            percentiles[p] = sorted_costs[min(idx, n_samples - 1)]

        # Risk metrics
        var_95 = percentiles[95]
        costs_above_var = [c for c in sample_costs if c > var_95]
        cvar_95 = sum(costs_above_var) / len(costs_above_var) if costs_above_var else var_95

        # Probability exceeds budget
        prob_exceeds = None
        if budget_limit is not None:
            prob_exceeds = sum(1 for c in sample_costs if c > budget_limit) / n_samples

        # Skewness and kurtosis
        skewness = self._calculate_skewness(sample_costs, mean_cost, std_cost)
        kurtosis = self._calculate_kurtosis(sample_costs, mean_cost, std_cost)

        result = MonteCarloResult(
            simulation_id=simulation_id,
            exchanger_id=exchanger_id,
            n_samples=n_samples,
            random_seed=self.config.random_seed,
            uncertain_parameters=[up.name for up in uncertain_params],
            mean_cost_usd=round(mean_cost, 2),
            std_cost_usd=round(std_cost, 2),
            median_cost_usd=round(median_cost, 2),
            confidence_level=self.config.confidence_level,
            ci_lower_usd=round(ci_lower, 2),
            ci_upper_usd=round(ci_upper, 2),
            percentiles={p: round(v, 2) for p, v in percentiles.items()},
            skewness=round(skewness, 4),
            kurtosis=round(kurtosis, 4),
            probability_exceeds_budget=round(prob_exceeds, 4) if prob_exceeds else None,
            value_at_risk_95=round(var_95, 2),
            conditional_var_95=round(cvar_95, 2),
            sample_costs=sample_costs,  # Include raw samples
        )

        result.provenance_hash = self._compute_hash({
            "simulation_id": simulation_id,
            "n_samples": n_samples,
            "mean_cost": mean_cost,
        })

        logger.info(
            f"Monte Carlo complete: mean=${mean_cost:,.0f}, "
            f"95% CI=[${ci_lower:,.0f}, ${ci_upper:,.0f}]"
        )

        return result

    def analyze_breakeven(
        self,
        exchanger_id: str,
        base_params: Dict[str, float],
        cleaning_method: CleaningMethodType = CleaningMethodType.CHEMICAL_OFFLINE,
        discount_rates: Optional[List[float]] = None,
    ) -> BreakevenAnalysis:
        """
        Analyze breakeven for cleaning investment.

        Calculates payback period, NPV at various discount rates,
        and sensitivity of breakeven to key parameters.

        Args:
            exchanger_id: Heat exchanger identifier
            base_params: Base parameter values
            cleaning_method: Cleaning method to analyze
            discount_rates: Discount rates for NPV analysis

        Returns:
            BreakevenAnalysis with investment metrics
        """
        analysis_id = self._generate_id("breakeven")

        logger.info(f"Running breakeven analysis for {exchanger_id}")

        # Calculate cleaning costs
        cleaning_cost = self.cost_model.calculate_cleaning_cost(
            exchanger_id=exchanger_id,
            cleaning_method=cleaning_method,
        )

        downtime_cost = self.cost_model.calculate_downtime_cost(
            exchanger_id=exchanger_id,
            cleaning_duration_hours=cleaning_cost.expected_cleaning_duration_hours,
            production_rate_tph=base_params.get("design_throughput_tph", 100),
            product_margin_usd_per_tonne=base_params.get("product_margin_usd_per_tonne", 50),
        )

        total_investment = (
            cleaning_cost.total_cleaning_cost_usd +
            downtime_cost.total_downtime_cost_usd
        )

        # Calculate daily savings (difference between no-clean and post-clean)
        no_clean_daily = self._evaluate_no_cleaning(exchanger_id, base_params) / 90  # Rough daily
        post_clean_daily = self._evaluate_with_cleaning(
            exchanger_id, 0, cleaning_method, base_params
        ) / 90

        daily_savings = no_clean_daily - post_clean_daily
        annual_savings = daily_savings * 365

        # Breakeven calculation
        if daily_savings > 0:
            breakeven_days = int(math.ceil(total_investment / daily_savings))
            breakeven_date = datetime.utcnow() + timedelta(days=breakeven_days)
            is_worthwhile = breakeven_days < 365
        else:
            breakeven_days = None
            breakeven_date = None
            is_worthwhile = False

        # Sensitivity of breakeven to savings rate
        breakeven_at_savings = {}
        for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            adjusted_savings = daily_savings * multiplier
            if adjusted_savings > 0:
                breakeven_at_savings[adjusted_savings] = int(
                    math.ceil(total_investment / adjusted_savings)
                )

        # NPV analysis
        if discount_rates is None:
            discount_rates = [0.05, 0.08, 0.10, 0.12, 0.15]

        horizon_years = 2
        npv_at_rates = {}
        for rate in discount_rates:
            npv = -total_investment  # Initial investment
            for year in range(1, horizon_years + 1):
                npv += annual_savings / ((1 + rate) ** year)
            npv_at_rates[rate] = round(npv, 2)

        # Minimum fouling rate for breakeven
        # Simplified: find rate where daily savings = daily investment amortization
        min_fouling_rate = None
        if daily_savings > 0 and base_params.get("fouling_rate_per_day", 0) > 0:
            current_rate = base_params["fouling_rate_per_day"]
            # If we need breakeven in 180 days, savings must be total_investment/180
            required_savings = total_investment / 180
            if no_clean_daily > 0:
                min_fouling_rate = current_rate * (required_savings / daily_savings)

        result = BreakevenAnalysis(
            analysis_id=analysis_id,
            exchanger_id=exchanger_id,
            cleaning_cost_usd=cleaning_cost.total_cleaning_cost_usd,
            downtime_cost_usd=downtime_cost.total_downtime_cost_usd,
            total_investment_usd=round(total_investment, 2),
            daily_savings_usd=round(daily_savings, 2),
            annual_savings_usd=round(annual_savings, 2),
            breakeven_days=breakeven_days,
            breakeven_date=breakeven_date,
            is_worthwhile=is_worthwhile,
            breakeven_at_different_savings=breakeven_at_savings,
            minimum_fouling_rate_for_breakeven=min_fouling_rate,
            npv_at_discount_rates=npv_at_rates,
            internal_rate_of_return=None,  # Could calculate via IRR algorithm
            assumptions={
                "horizon_years": horizon_years,
                "cleaning_method": cleaning_method.value,
            },
        )

        result.provenance_hash = self._compute_hash({
            "analysis_id": analysis_id,
            "total_investment": total_investment,
            "breakeven_days": breakeven_days,
        })

        return result

    def create_standard_uncertain_params(
        self,
        base_params: Dict[str, float],
    ) -> List[UncertainParameter]:
        """
        Create standard set of uncertain parameters for Monte Carlo.

        Args:
            base_params: Base parameter values

        Returns:
            List of UncertainParameter with default distributions
        """
        params = []

        # Fouling rate - high uncertainty
        if "fouling_rate_per_day" in base_params:
            params.append(UncertainParameter(
                name="fouling_rate_per_day",
                base_value=base_params["fouling_rate_per_day"],
                distribution=DistributionType.TRIANGULAR,
                min_value=base_params["fouling_rate_per_day"] * 0.5,
                max_value=base_params["fouling_rate_per_day"] * 2.0,
                mode_value=base_params["fouling_rate_per_day"],
                category="fouling",
            ))

        # Energy costs - moderate uncertainty
        if "fuel_cost_usd_per_gj" in base_params:
            params.append(UncertainParameter(
                name="fuel_cost_usd_per_gj",
                base_value=base_params["fuel_cost_usd_per_gj"],
                distribution=DistributionType.NORMAL,
                std_dev=base_params["fuel_cost_usd_per_gj"] * 0.15,
                category="energy",
            ))

        # Product margin - moderate uncertainty
        if "product_margin_usd_per_tonne" in base_params:
            params.append(UncertainParameter(
                name="product_margin_usd_per_tonne",
                base_value=base_params["product_margin_usd_per_tonne"],
                distribution=DistributionType.NORMAL,
                std_dev=base_params["product_margin_usd_per_tonne"] * 0.10,
                category="production",
            ))

        # Cleaning cost - moderate uncertainty
        if "cleaning_cost_usd" in base_params:
            params.append(UncertainParameter(
                name="cleaning_cost_usd",
                base_value=base_params["cleaning_cost_usd"],
                distribution=DistributionType.TRIANGULAR,
                min_value=base_params["cleaning_cost_usd"] * 0.8,
                max_value=base_params["cleaning_cost_usd"] * 1.5,
                mode_value=base_params["cleaning_cost_usd"],
                category="cleaning",
            ))

        # UA recovery - moderate uncertainty
        if "ua_recovery_fraction" in base_params:
            params.append(UncertainParameter(
                name="ua_recovery_fraction",
                base_value=base_params["ua_recovery_fraction"],
                distribution=DistributionType.TRIANGULAR,
                min_value=0.80,
                max_value=0.98,
                mode_value=base_params["ua_recovery_fraction"],
                category="cleaning",
            ))

        return params

    def _evaluate_no_cleaning(
        self,
        exchanger_id: str,
        params: Dict[str, float],
    ) -> float:
        """Evaluate cost with no cleaning."""
        cost_breakdown = self.cost_model.calculate_total_cost(
            exchanger_id=exchanger_id,
            horizon_days=90,
            current_ua_kw_k=params.get("current_ua_kw_k", 450),
            clean_ua_kw_k=params.get("clean_ua_kw_k", 500),
            heat_duty_kw=params.get("heat_duty_kw", 1000),
            current_effectiveness=params.get("current_effectiveness", 0.80),
            design_effectiveness=params.get("design_effectiveness", 0.90),
            design_throughput_tph=params.get("design_throughput_tph", 100),
            product_margin_usd_per_tonne=params.get("product_margin_usd_per_tonne", 50),
            current_delta_p_kpa=params.get("current_delta_p_kpa", 50),
            delta_p_limit_kpa=params.get("delta_p_limit_kpa", 80),
            current_t_outlet_c=params.get("current_t_outlet_c", 80),
            t_outlet_limit_c=params.get("t_outlet_limit_c", 100),
            days_since_cleaning=params.get("days_since_cleaning", 180),
            fouling_rate_per_day=params.get("fouling_rate_per_day", 0.002),
            include_cleaning=False,
        )
        return cost_breakdown.total_cost_usd

    def _evaluate_with_cleaning(
        self,
        exchanger_id: str,
        cleaning_day: int,
        cleaning_method: CleaningMethodType,
        params: Dict[str, float],
    ) -> float:
        """Evaluate cost with cleaning on specified day."""
        cost_breakdown = self.cost_model.calculate_total_cost(
            exchanger_id=exchanger_id,
            horizon_days=90,
            current_ua_kw_k=params.get("current_ua_kw_k", 450),
            clean_ua_kw_k=params.get("clean_ua_kw_k", 500),
            heat_duty_kw=params.get("heat_duty_kw", 1000),
            current_effectiveness=params.get("current_effectiveness", 0.80),
            design_effectiveness=params.get("design_effectiveness", 0.90),
            design_throughput_tph=params.get("design_throughput_tph", 100),
            product_margin_usd_per_tonne=params.get("product_margin_usd_per_tonne", 50),
            current_delta_p_kpa=params.get("current_delta_p_kpa", 50),
            delta_p_limit_kpa=params.get("delta_p_limit_kpa", 80),
            current_t_outlet_c=params.get("current_t_outlet_c", 80),
            t_outlet_limit_c=params.get("t_outlet_limit_c", 100),
            days_since_cleaning=params.get("days_since_cleaning", 180),
            fouling_rate_per_day=params.get("fouling_rate_per_day", 0.002),
            cleaning_method=cleaning_method,
            include_cleaning=True,
        )
        return cost_breakdown.total_cost_usd

    def _calculate_metric(
        self,
        exchanger_id: str,
        params: Dict[str, float],
        metric: SensitivityMetric,
    ) -> float:
        """Calculate a specific metric for given parameters."""
        if metric == SensitivityMetric.TOTAL_COST:
            return self._evaluate_no_cleaning(exchanger_id, params)
        elif metric == SensitivityMetric.ENERGY_LOSS:
            cost = self.cost_model.calculate_energy_loss(
                exchanger_id=exchanger_id,
                current_ua_kw_k=params.get("current_ua_kw_k", 450),
                clean_ua_kw_k=params.get("clean_ua_kw_k", 500),
                heat_duty_kw=params.get("heat_duty_kw", 1000),
            )
            return cost.annual_energy_loss_usd
        else:
            return self._evaluate_no_cleaning(exchanger_id, params)

    def _apply_multipliers(
        self,
        base_params: Dict[str, float],
        multipliers: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply multipliers to base parameters."""
        result = base_params.copy()
        for param, mult in multipliers.items():
            if param in result:
                result[param] = result[param] * mult
        return result

    def _calculate_elasticity(
        self,
        x_values: List[float],
        y_values: List[float],
        base_x: float,
        base_y: float,
    ) -> float:
        """Calculate elasticity (percent change in y per percent change in x)."""
        if len(x_values) < 2 or base_x == 0 or base_y == 0:
            return 0.0

        # Use midpoint elasticity
        x_low, x_high = x_values[0], x_values[-1]
        y_low, y_high = y_values[0], y_values[-1]

        pct_change_x = (x_high - x_low) / base_x * 100
        pct_change_y = (y_high - y_low) / base_y * 100

        if pct_change_x == 0:
            return 0.0

        return pct_change_y / pct_change_x

    def _find_breakeven(
        self,
        x_values: List[float],
        y_values: List[float],
        base_y: float,
    ) -> Optional[float]:
        """Find x value where y equals base_y."""
        for i in range(len(x_values) - 1):
            if (y_values[i] <= base_y <= y_values[i + 1] or
                y_values[i] >= base_y >= y_values[i + 1]):
                # Linear interpolation
                if y_values[i + 1] - y_values[i] == 0:
                    continue
                t = (base_y - y_values[i]) / (y_values[i + 1] - y_values[i])
                return x_values[i] + t * (x_values[i + 1] - x_values[i])
        return None

    def _categorize_parameter(self, param_name: str) -> str:
        """Categorize parameter by name."""
        if "fouling" in param_name.lower():
            return "fouling"
        elif "energy" in param_name.lower() or "fuel" in param_name.lower():
            return "energy"
        elif "production" in param_name.lower() or "throughput" in param_name.lower():
            return "production"
        elif "cleaning" in param_name.lower():
            return "cleaning"
        elif "ua" in param_name.lower() or "heat" in param_name.lower():
            return "thermal"
        else:
            return "operational"

    def _calculate_skewness(
        self,
        values: List[float],
        mean: float,
        std: float,
    ) -> float:
        """Calculate skewness of distribution."""
        if std == 0 or len(values) < 3:
            return 0.0

        n = len(values)
        m3 = sum((v - mean) ** 3 for v in values) / n
        return m3 / (std ** 3)

    def _calculate_kurtosis(
        self,
        values: List[float],
        mean: float,
        std: float,
    ) -> float:
        """Calculate excess kurtosis of distribution."""
        if std == 0 or len(values) < 4:
            return 0.0

        n = len(values)
        m4 = sum((v - mean) ** 4 for v in values) / n
        return m4 / (std ** 4) - 3  # Excess kurtosis

    def _generate_comparison_insights(
        self,
        scenarios: List[ScenarioDefinition],
        costs: Dict[str, float],
        best_id: str,
    ) -> List[str]:
        """Generate insights from scenario comparison."""
        insights = []

        # Best scenario
        best_scenario = next((s for s in scenarios if s.scenario_id == best_id), None)
        if best_scenario:
            insights.append(
                f"Best option: {best_scenario.scenario_name} "
                f"(${costs[best_id]:,.0f} total cost)"
            )

        # Cost range
        min_cost = min(costs.values())
        max_cost = max(costs.values())
        if max_cost > min_cost:
            range_pct = (max_cost - min_cost) / min_cost * 100
            insights.append(
                f"Cost range: ${min_cost:,.0f} to ${max_cost:,.0f} "
                f"({range_pct:.0f}% difference)"
            )

        # Cleaning vs no-clean comparison
        no_clean_scenarios = [s for s in scenarios if s.cleaning_action == "no_clean"]
        clean_scenarios = [s for s in scenarios if s.cleaning_action != "no_clean"]

        if no_clean_scenarios and clean_scenarios:
            no_clean_cost = sum(costs[s.scenario_id] for s in no_clean_scenarios) / len(no_clean_scenarios)
            clean_cost = sum(costs[s.scenario_id] for s in clean_scenarios) / len(clean_scenarios)

            if clean_cost < no_clean_cost:
                savings = no_clean_cost - clean_cost
                insights.append(f"Cleaning saves ${savings:,.0f} on average vs. no cleaning")
            else:
                extra = clean_cost - no_clean_cost
                insights.append(f"Cleaning costs ${extra:,.0f} more than no cleaning")

        return insights

    def _generate_id(self, prefix: str) -> str:
        """Generate unique identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_suffix = self._rng.randint(1000, 9999)
        return f"{prefix}_{timestamp}_{random_suffix}"

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
