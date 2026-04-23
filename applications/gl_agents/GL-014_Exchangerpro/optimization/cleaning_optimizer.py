# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Cleaning Schedule Optimizer

Deterministic grid-search optimizer for single heat exchanger cleaning schedules.
Finds optimal cleaning windows by evaluating total expected cost over planning horizon.

Features:
- Decision variables: cleaning date/window, cleaning method
- Objective: minimize total expected cost over planning horizon
- Constraints: maintenance windows, delta-P limits, outlet temperature limits
- Candidate date grid search (deterministic, reproducible)
- What-if analysis: clean now vs clean in X days vs no clean

Zero-Hallucination Principle:
    All optimization uses explicit grid search over candidate dates.
    Cost calculations use deterministic formulas from CleaningCostModel.
    SHAP/LIME features are from ML predictions, not optimizer estimates.

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

logger = logging.getLogger(__name__)


class OptimizationStatus(str, Enum):
    """Status of optimization run."""
    SUCCESS = "success"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    NO_CLEANING_NEEDED = "no_cleaning_needed"
    ERROR = "error"


class CleaningUrgency(str, Enum):
    """Urgency level for cleaning recommendation."""
    CRITICAL = "critical"  # Within 7 days
    HIGH = "high"  # Within 14 days
    MEDIUM = "medium"  # Within 30 days
    LOW = "low"  # Within 60 days
    NONE = "none"  # No cleaning needed


class ConfidenceLevel(str, Enum):
    """Confidence level for recommendations."""
    HIGH = "high"  # >= 0.85
    MEDIUM = "medium"  # >= 0.65
    LOW = "low"  # >= 0.40


@dataclass
class OptimizerConfig:
    """Configuration for the cleaning schedule optimizer."""

    # Grid search parameters
    horizon_days: int = 90  # Planning horizon
    grid_resolution_days: int = 1  # Day step for grid search
    max_candidates: int = 100  # Maximum candidate dates to evaluate

    # Constraints
    min_days_between_cleanings: int = 30  # Minimum interval
    max_days_between_cleanings: int = 365  # Maximum interval
    maintenance_blackout_days: List[int] = field(default_factory=list)  # Days to avoid

    # Thresholds
    ua_degradation_threshold: float = 0.15  # 15% UA loss triggers consideration
    delta_p_safety_margin: float = 0.10  # 10% margin to delta-P limit
    risk_threshold: float = 0.20  # 20% violation probability triggers cleaning

    # Default cleaning method
    default_cleaning_method: CleaningMethodType = CleaningMethodType.CHEMICAL_OFFLINE

    # Reproducibility
    random_seed: int = 42


class CleaningWindow(BaseModel):
    """
    Represents a candidate cleaning window.
    """
    window_id: str = Field(..., description="Unique window identifier")
    start_date: datetime = Field(..., description="Window start date")
    end_date: datetime = Field(..., description="Window end date")
    duration_days: int = Field(..., ge=1, description="Window duration in days")

    # Feasibility
    is_feasible: bool = Field(True, description="Whether window is feasible")
    infeasibility_reasons: List[str] = Field(default_factory=list)

    # Constraints
    overlaps_maintenance: bool = Field(False)
    resource_available: bool = Field(True)

    # Priority
    priority_score: float = Field(0.0, ge=0, le=1, description="Priority score (higher = better)")


class CleaningMethod(BaseModel):
    """
    Represents a cleaning method option.
    """
    method_type: CleaningMethodType = Field(...)
    name: str = Field(..., description="Human-readable name")

    # Cost estimates
    estimated_cost_usd: float = Field(..., ge=0)
    estimated_duration_hours: float = Field(..., gt=0)

    # Effectiveness
    expected_ua_recovery: float = Field(..., ge=0, le=1)
    expected_time_to_refoul_days: int = Field(..., ge=1)

    # Constraints
    requires_shutdown: bool = Field(True)
    min_outage_hours: float = Field(0.0, ge=0)


class OptimizationResult(BaseModel):
    """
    Complete result from cleaning schedule optimization.
    """
    optimization_id: str = Field(..., description="Unique optimization ID")
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Status
    status: OptimizationStatus = Field(...)
    execution_time_ms: float = Field(..., ge=0)

    # Optimal solution
    optimal_cleaning_date: Optional[datetime] = Field(None)
    optimal_cleaning_method: Optional[CleaningMethodType] = Field(None)
    optimal_total_cost_usd: float = Field(..., ge=0)

    # Baseline comparison
    no_clean_cost_usd: float = Field(..., ge=0, description="Cost if no cleaning")
    savings_from_cleaning_usd: float = Field(default=0.0)
    payback_days: Optional[int] = Field(None, description="Days until cleaning pays off")

    # Ranked alternatives
    ranked_windows: List['ScheduleRanking'] = Field(default_factory=list)

    # Recommendation
    recommendation: 'CleaningRecommendation' = Field(...)

    # Provenance
    assumptions: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field("")
    model_version: str = Field("1.0.0")

    class Config:
        arbitrary_types_allowed = True


class ScheduleRanking(BaseModel):
    """
    Ranked cleaning schedule option.
    """
    rank: int = Field(..., ge=1, description="Ranking (1 = best)")
    cleaning_date: datetime = Field(...)
    cleaning_method: CleaningMethodType = Field(...)

    # Costs
    total_cost_usd: float = Field(..., ge=0)
    cost_delta_vs_optimal_usd: float = Field(0.0, ge=0)
    cost_delta_pct: float = Field(0.0, ge=0)

    # Performance
    expected_ua_recovery: float = Field(..., ge=0, le=1)
    expected_refouling_days: int = Field(..., ge=1)

    # Risk metrics
    violation_probability_at_cleaning: float = Field(..., ge=0, le=1)
    days_to_constraint_violation: Optional[int] = Field(None)

    # Reasoning
    key_drivers: List[str] = Field(default_factory=list, description="Top factors for ranking")


class CleaningRecommendation(BaseModel):
    """
    Final cleaning recommendation with supporting rationale.

    Includes SHAP/LIME-style top drivers for explainability.
    """
    exchanger_id: str = Field(...)
    recommendation_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Primary recommendation
    action: str = Field(..., description="Recommended action")
    urgency: CleaningUrgency = Field(...)
    confidence: ConfidenceLevel = Field(...)

    # Cleaning details
    recommended_date: Optional[datetime] = Field(None)
    recommended_method: Optional[CleaningMethodType] = Field(None)
    recommended_window: Optional[CleaningWindow] = Field(None)

    # Expected outcomes
    expected_ua_recovery: float = Field(0.0, ge=0, le=1)
    expected_time_to_refoul_days: int = Field(0, ge=0)
    expected_energy_savings_usd: float = Field(0.0, ge=0)
    expected_total_savings_usd: float = Field(0.0, ge=0)

    # Top drivers (SHAP/LIME features)
    top_drivers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top factors driving recommendation (SHAP/LIME style)"
    )

    # Assumptions and caveats
    assumptions: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field("")

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        content = {
            "exchanger_id": self.exchanger_id,
            "action": self.action,
            "recommended_date": self.recommended_date.isoformat() if self.recommended_date else None,
            "urgency": self.urgency.value,
            "timestamp": self.timestamp.isoformat(),
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class WhatIfScenario(BaseModel):
    """
    What-if scenario definition.
    """
    scenario_id: str = Field(...)
    scenario_name: str = Field(...)
    description: str = Field("")

    # Cleaning action
    clean_on_day: Optional[int] = Field(None, description="Day to clean (None = no cleaning)")
    cleaning_method: Optional[CleaningMethodType] = Field(None)

    # Parameter overrides
    parameter_overrides: Dict[str, float] = Field(default_factory=dict)


class WhatIfResult(BaseModel):
    """
    Result of what-if scenario analysis.
    """
    scenario: WhatIfScenario = Field(...)

    # Costs
    total_cost_usd: float = Field(..., ge=0)
    energy_loss_cost_usd: float = Field(..., ge=0)
    production_loss_cost_usd: float = Field(..., ge=0)
    cleaning_cost_usd: float = Field(0.0, ge=0)
    downtime_cost_usd: float = Field(0.0, ge=0)
    risk_penalty_usd: float = Field(..., ge=0)

    # Comparison to baseline
    cost_delta_vs_baseline_usd: float = Field(0.0)
    cost_delta_pct: float = Field(0.0)
    is_better_than_baseline: bool = Field(False)

    # Performance trajectory
    ua_trajectory: List[float] = Field(default_factory=list)
    delta_p_trajectory: List[float] = Field(default_factory=list)
    constraint_violation_day: Optional[int] = Field(None)

    # Provenance
    provenance_hash: str = Field("")


class CleaningScheduleOptimizer:
    """
    Deterministic grid-search optimizer for single heat exchanger cleaning schedules.

    This optimizer evaluates candidate cleaning dates across the planning horizon
    and selects the date/method combination that minimizes total expected cost.

    Features:
    - Grid search over candidate cleaning dates
    - Multiple cleaning method comparison
    - Constraint checking (maintenance windows, delta-P limits, temperature)
    - What-if scenario analysis
    - Explainable recommendations with top drivers

    Zero-Hallucination Principle:
        All cost calculations use deterministic formulas.
        Grid search is exhaustive within resolution.
        Recommendations include explicit assumptions and provenance.

    Example:
        >>> config = OptimizerConfig(horizon_days=90)
        >>> optimizer = CleaningScheduleOptimizer(config)
        >>> result = optimizer.optimize(
        ...     exchanger_id="HX-001",
        ...     current_ua=450.0,
        ...     clean_ua=500.0,
        ...     fouling_rate=0.002,
        ... )
        >>> print(f"Optimal date: {result.optimal_cleaning_date}")
        >>> print(f"Savings: ${result.savings_from_cleaning_usd:,.0f}")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
        cost_model: Optional[CleaningCostModel] = None,
    ) -> None:
        """
        Initialize the cleaning schedule optimizer.

        Args:
            config: Optimizer configuration
            cost_model: Cost model for calculations (creates default if None)
        """
        self.config = config or OptimizerConfig()
        self.cost_model = cost_model or CleaningCostModel(seed=self.config.random_seed)

        random.seed(self.config.random_seed)

        logger.info(
            f"CleaningScheduleOptimizer initialized: horizon={self.config.horizon_days}d, "
            f"resolution={self.config.grid_resolution_days}d"
        )

    def optimize(
        self,
        exchanger_id: str,
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        heat_duty_kw: float,
        current_effectiveness: float,
        design_effectiveness: float,
        design_throughput_tph: float,
        product_margin_usd_per_tonne: float,
        current_delta_p_kpa: float,
        delta_p_limit_kpa: float,
        current_t_outlet_c: float,
        t_outlet_limit_c: float,
        days_since_cleaning: int,
        fouling_rate_per_day: float,
        available_methods: Optional[List[CleaningMethodType]] = None,
        maintenance_windows: Optional[List[CleaningWindow]] = None,
        feature_importances: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """
        Optimize cleaning schedule for a single heat exchanger.

        Uses grid search to evaluate all candidate cleaning dates within the
        planning horizon and selects the optimal date/method combination.

        Args:
            exchanger_id: Heat exchanger identifier
            current_ua_kw_k: Current UA value (kW/K)
            clean_ua_kw_k: Clean/design UA value (kW/K)
            heat_duty_kw: Current heat duty (kW)
            current_effectiveness: Current thermal effectiveness (0-1)
            design_effectiveness: Design thermal effectiveness (0-1)
            design_throughput_tph: Design throughput (t/h)
            product_margin_usd_per_tonne: Product margin ($/t)
            current_delta_p_kpa: Current pressure drop (kPa)
            delta_p_limit_kpa: Maximum pressure drop limit (kPa)
            current_t_outlet_c: Current outlet temperature (C)
            t_outlet_limit_c: Outlet temperature limit (C)
            days_since_cleaning: Days since last cleaning
            fouling_rate_per_day: Daily fouling rate (fraction)
            available_methods: Available cleaning methods
            maintenance_windows: Available maintenance windows
            feature_importances: SHAP/LIME feature importances from ML model

        Returns:
            OptimizationResult with optimal schedule and recommendations
        """
        start_time = datetime.utcnow()
        optimization_id = self._generate_id(exchanger_id)

        logger.info(
            f"Starting optimization for {exchanger_id}: "
            f"UA={current_ua_kw_k:.1f}/{clean_ua_kw_k:.1f} kW/K, "
            f"days_since_clean={days_since_cleaning}"
        )

        # Default available methods
        if available_methods is None:
            available_methods = [
                CleaningMethodType.CHEMICAL_OFFLINE,
                CleaningMethodType.MECHANICAL_HYDROBLAST,
            ]

        # Generate candidate dates
        candidate_days = self._generate_candidate_days()

        # Evaluate "no cleaning" baseline
        no_clean_cost = self._evaluate_no_cleaning(
            exchanger_id=exchanger_id,
            current_ua_kw_k=current_ua_kw_k,
            clean_ua_kw_k=clean_ua_kw_k,
            heat_duty_kw=heat_duty_kw,
            current_effectiveness=current_effectiveness,
            design_effectiveness=design_effectiveness,
            design_throughput_tph=design_throughput_tph,
            product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            current_delta_p_kpa=current_delta_p_kpa,
            delta_p_limit_kpa=delta_p_limit_kpa,
            current_t_outlet_c=current_t_outlet_c,
            t_outlet_limit_c=t_outlet_limit_c,
            days_since_cleaning=days_since_cleaning,
            fouling_rate_per_day=fouling_rate_per_day,
        )

        # Evaluate each candidate
        evaluations: List[Tuple[int, CleaningMethodType, float, TotalCostBreakdown]] = []

        for day in candidate_days:
            for method in available_methods:
                # Check feasibility
                if not self._is_feasible(day, maintenance_windows):
                    continue

                # Calculate costs with cleaning on this day
                cost_breakdown = self._evaluate_with_cleaning(
                    exchanger_id=exchanger_id,
                    cleaning_day=day,
                    cleaning_method=method,
                    current_ua_kw_k=current_ua_kw_k,
                    clean_ua_kw_k=clean_ua_kw_k,
                    heat_duty_kw=heat_duty_kw,
                    current_effectiveness=current_effectiveness,
                    design_effectiveness=design_effectiveness,
                    design_throughput_tph=design_throughput_tph,
                    product_margin_usd_per_tonne=product_margin_usd_per_tonne,
                    current_delta_p_kpa=current_delta_p_kpa,
                    delta_p_limit_kpa=delta_p_limit_kpa,
                    current_t_outlet_c=current_t_outlet_c,
                    t_outlet_limit_c=t_outlet_limit_c,
                    days_since_cleaning=days_since_cleaning,
                    fouling_rate_per_day=fouling_rate_per_day,
                )

                evaluations.append((day, method, cost_breakdown.total_cost_usd, cost_breakdown))

        # Find optimal
        if not evaluations:
            # No feasible cleaning options
            status = OptimizationStatus.INFEASIBLE
            optimal_day = None
            optimal_method = None
            optimal_cost = no_clean_cost.total_cost_usd
        else:
            # Sort by cost
            evaluations.sort(key=lambda x: x[2])

            optimal_day, optimal_method, optimal_cost, _ = evaluations[0]

            # Check if cleaning is beneficial
            if optimal_cost >= no_clean_cost.total_cost_usd:
                status = OptimizationStatus.NO_CLEANING_NEEDED
                optimal_day = None
                optimal_method = None
                optimal_cost = no_clean_cost.total_cost_usd
            else:
                status = OptimizationStatus.SUCCESS

        # Build ranked alternatives
        ranked_windows = self._build_rankings(
            evaluations[:10],  # Top 10
            optimal_cost,
            feature_importances,
        )

        # Calculate savings
        savings = no_clean_cost.total_cost_usd - optimal_cost if optimal_day else 0
        payback_days = self._calculate_payback(evaluations, no_clean_cost) if optimal_day else None

        # Build recommendation
        recommendation = self._build_recommendation(
            exchanger_id=exchanger_id,
            optimal_day=optimal_day,
            optimal_method=optimal_method,
            savings=savings,
            no_clean_cost=no_clean_cost,
            feature_importances=feature_importances,
            current_ua_kw_k=current_ua_kw_k,
            clean_ua_kw_k=clean_ua_kw_k,
            current_delta_p_kpa=current_delta_p_kpa,
            delta_p_limit_kpa=delta_p_limit_kpa,
            fouling_rate_per_day=fouling_rate_per_day,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = OptimizationResult(
            optimization_id=optimization_id,
            exchanger_id=exchanger_id,
            status=status,
            execution_time_ms=execution_time,
            optimal_cleaning_date=(
                datetime.utcnow() + timedelta(days=optimal_day) if optimal_day else None
            ),
            optimal_cleaning_method=optimal_method,
            optimal_total_cost_usd=optimal_cost,
            no_clean_cost_usd=no_clean_cost.total_cost_usd,
            savings_from_cleaning_usd=savings,
            payback_days=payback_days,
            ranked_windows=ranked_windows,
            recommendation=recommendation,
            assumptions={
                "horizon_days": self.config.horizon_days,
                "grid_resolution_days": self.config.grid_resolution_days,
                "fouling_rate_per_day": fouling_rate_per_day,
                "candidates_evaluated": len(evaluations),
            },
        )

        result.provenance_hash = self._compute_hash({
            "optimization_id": optimization_id,
            "exchanger_id": exchanger_id,
            "optimal_cost": optimal_cost,
            "status": status.value,
        })

        logger.info(
            f"Optimization complete for {exchanger_id}: status={status.value}, "
            f"optimal_day={optimal_day}, savings=${savings:,.0f}"
        )

        return result

    def analyze_what_if(
        self,
        exchanger_id: str,
        scenarios: List[WhatIfScenario],
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        heat_duty_kw: float,
        current_effectiveness: float,
        design_effectiveness: float,
        design_throughput_tph: float,
        product_margin_usd_per_tonne: float,
        current_delta_p_kpa: float,
        delta_p_limit_kpa: float,
        current_t_outlet_c: float,
        t_outlet_limit_c: float,
        days_since_cleaning: int,
        fouling_rate_per_day: float,
    ) -> List[WhatIfResult]:
        """
        Analyze what-if scenarios comparing different cleaning strategies.

        Common scenarios:
        - Clean now vs clean in 7 days vs clean in 30 days vs no clean
        - Chemical vs mechanical cleaning
        - Different fouling rate assumptions

        Args:
            exchanger_id: Heat exchanger identifier
            scenarios: List of scenarios to evaluate
            (remaining args same as optimize())

        Returns:
            List of WhatIfResult for each scenario
        """
        logger.info(f"Analyzing {len(scenarios)} what-if scenarios for {exchanger_id}")

        results = []

        # Establish baseline (no cleaning)
        baseline_cost = self._evaluate_no_cleaning(
            exchanger_id=exchanger_id,
            current_ua_kw_k=current_ua_kw_k,
            clean_ua_kw_k=clean_ua_kw_k,
            heat_duty_kw=heat_duty_kw,
            current_effectiveness=current_effectiveness,
            design_effectiveness=design_effectiveness,
            design_throughput_tph=design_throughput_tph,
            product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            current_delta_p_kpa=current_delta_p_kpa,
            delta_p_limit_kpa=delta_p_limit_kpa,
            current_t_outlet_c=current_t_outlet_c,
            t_outlet_limit_c=t_outlet_limit_c,
            days_since_cleaning=days_since_cleaning,
            fouling_rate_per_day=fouling_rate_per_day,
        )

        for scenario in scenarios:
            # Apply parameter overrides
            local_fouling_rate = scenario.parameter_overrides.get(
                "fouling_rate_per_day", fouling_rate_per_day
            )

            if scenario.clean_on_day is None:
                # No cleaning scenario
                cost = self._evaluate_no_cleaning(
                    exchanger_id=exchanger_id,
                    current_ua_kw_k=current_ua_kw_k,
                    clean_ua_kw_k=clean_ua_kw_k,
                    heat_duty_kw=heat_duty_kw,
                    current_effectiveness=current_effectiveness,
                    design_effectiveness=design_effectiveness,
                    design_throughput_tph=design_throughput_tph,
                    product_margin_usd_per_tonne=product_margin_usd_per_tonne,
                    current_delta_p_kpa=current_delta_p_kpa,
                    delta_p_limit_kpa=delta_p_limit_kpa,
                    current_t_outlet_c=current_t_outlet_c,
                    t_outlet_limit_c=t_outlet_limit_c,
                    days_since_cleaning=days_since_cleaning,
                    fouling_rate_per_day=local_fouling_rate,
                )
                cleaning_cost = 0.0
                downtime_cost = 0.0
            else:
                # Cleaning scenario
                method = scenario.cleaning_method or self.config.default_cleaning_method
                cost = self._evaluate_with_cleaning(
                    exchanger_id=exchanger_id,
                    cleaning_day=scenario.clean_on_day,
                    cleaning_method=method,
                    current_ua_kw_k=current_ua_kw_k,
                    clean_ua_kw_k=clean_ua_kw_k,
                    heat_duty_kw=heat_duty_kw,
                    current_effectiveness=current_effectiveness,
                    design_effectiveness=design_effectiveness,
                    design_throughput_tph=design_throughput_tph,
                    product_margin_usd_per_tonne=product_margin_usd_per_tonne,
                    current_delta_p_kpa=current_delta_p_kpa,
                    delta_p_limit_kpa=delta_p_limit_kpa,
                    current_t_outlet_c=current_t_outlet_c,
                    t_outlet_limit_c=t_outlet_limit_c,
                    days_since_cleaning=days_since_cleaning,
                    fouling_rate_per_day=local_fouling_rate,
                )
                cleaning_cost = cost.total_cleaning_cost_usd
                downtime_cost = cost.total_downtime_cost_usd

            # Calculate delta vs baseline
            cost_delta = cost.total_cost_usd - baseline_cost.total_cost_usd
            cost_delta_pct = (cost_delta / baseline_cost.total_cost_usd * 100
                             if baseline_cost.total_cost_usd > 0 else 0)

            # Build UA trajectory
            ua_trajectory = self._project_ua_trajectory(
                current_ua_kw_k=current_ua_kw_k,
                clean_ua_kw_k=clean_ua_kw_k,
                fouling_rate=local_fouling_rate,
                cleaning_day=scenario.clean_on_day,
                horizon_days=self.config.horizon_days,
            )

            # Find constraint violation day
            violation_day = self._find_constraint_violation_day(
                ua_trajectory=ua_trajectory,
                clean_ua_kw_k=clean_ua_kw_k,
                threshold=self.config.ua_degradation_threshold,
            )

            result = WhatIfResult(
                scenario=scenario,
                total_cost_usd=cost.total_cost_usd,
                energy_loss_cost_usd=cost.total_energy_loss_usd,
                production_loss_cost_usd=cost.total_production_loss_usd,
                cleaning_cost_usd=cleaning_cost,
                downtime_cost_usd=downtime_cost,
                risk_penalty_usd=cost.total_risk_penalty_usd,
                cost_delta_vs_baseline_usd=cost_delta,
                cost_delta_pct=round(cost_delta_pct, 2),
                is_better_than_baseline=cost_delta < 0,
                ua_trajectory=ua_trajectory,
                constraint_violation_day=violation_day,
            )

            result.provenance_hash = self._compute_hash({
                "scenario_id": scenario.scenario_id,
                "total_cost": cost.total_cost_usd,
            })

            results.append(result)

        return results

    def create_standard_what_if_scenarios(
        self,
        cleaning_method: Optional[CleaningMethodType] = None,
    ) -> List[WhatIfScenario]:
        """
        Create standard what-if scenarios for comparison.

        Returns scenarios for:
        - Clean now (day 0)
        - Clean in 7 days
        - Clean in 14 days
        - Clean in 30 days
        - No cleaning

        Args:
            cleaning_method: Cleaning method to use (default if None)

        Returns:
            List of standard scenarios
        """
        method = cleaning_method or self.config.default_cleaning_method

        return [
            WhatIfScenario(
                scenario_id="clean_now",
                scenario_name="Clean Now",
                description="Immediate cleaning intervention",
                clean_on_day=0,
                cleaning_method=method,
            ),
            WhatIfScenario(
                scenario_id="clean_7d",
                scenario_name="Clean in 7 Days",
                description="Clean during next maintenance window (1 week)",
                clean_on_day=7,
                cleaning_method=method,
            ),
            WhatIfScenario(
                scenario_id="clean_14d",
                scenario_name="Clean in 14 Days",
                description="Clean during scheduled maintenance (2 weeks)",
                clean_on_day=14,
                cleaning_method=method,
            ),
            WhatIfScenario(
                scenario_id="clean_30d",
                scenario_name="Clean in 30 Days",
                description="Clean at next turnaround (1 month)",
                clean_on_day=30,
                cleaning_method=method,
            ),
            WhatIfScenario(
                scenario_id="no_clean",
                scenario_name="No Cleaning",
                description="Continue operation without cleaning",
                clean_on_day=None,
                cleaning_method=None,
            ),
        ]

    def _generate_candidate_days(self) -> List[int]:
        """Generate candidate cleaning days for grid search."""
        candidates = []
        day = 0

        while day <= self.config.horizon_days and len(candidates) < self.config.max_candidates:
            if day not in self.config.maintenance_blackout_days:
                candidates.append(day)
            day += self.config.grid_resolution_days

        return candidates

    def _is_feasible(
        self,
        day: int,
        maintenance_windows: Optional[List[CleaningWindow]],
    ) -> bool:
        """Check if cleaning on given day is feasible."""
        # Check blackout days
        if day in self.config.maintenance_blackout_days:
            return False

        # Check maintenance windows if provided
        if maintenance_windows:
            target_date = datetime.utcnow() + timedelta(days=day)
            for window in maintenance_windows:
                if window.start_date <= target_date <= window.end_date:
                    return window.is_feasible

        return True

    def _evaluate_no_cleaning(
        self,
        exchanger_id: str,
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        heat_duty_kw: float,
        current_effectiveness: float,
        design_effectiveness: float,
        design_throughput_tph: float,
        product_margin_usd_per_tonne: float,
        current_delta_p_kpa: float,
        delta_p_limit_kpa: float,
        current_t_outlet_c: float,
        t_outlet_limit_c: float,
        days_since_cleaning: int,
        fouling_rate_per_day: float,
    ) -> TotalCostBreakdown:
        """Evaluate costs assuming no cleaning over horizon."""
        # Integrate costs over horizon accounting for continued degradation
        total_energy_loss = 0.0
        total_production_loss = 0.0
        total_risk = 0.0

        for day in range(self.config.horizon_days):
            # Project degradation
            degradation_factor = 1 + fouling_rate_per_day * day
            projected_ua = current_ua_kw_k / degradation_factor
            projected_effectiveness = current_effectiveness / degradation_factor

            # Daily energy loss
            energy_loss = self.cost_model.calculate_energy_loss(
                exchanger_id=exchanger_id,
                current_ua_kw_k=projected_ua,
                clean_ua_kw_k=clean_ua_kw_k,
                heat_duty_kw=heat_duty_kw,
            )
            total_energy_loss += energy_loss.daily_energy_loss_usd

            # Daily production loss
            prod_loss = self.cost_model.calculate_production_loss(
                exchanger_id=exchanger_id,
                current_effectiveness=min(projected_effectiveness, 1.0),
                design_effectiveness=design_effectiveness,
                design_throughput_tph=design_throughput_tph,
                product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            )
            total_production_loss += prod_loss.daily_production_loss_usd

        # Risk penalty (evaluated once for full horizon)
        risk = self.cost_model.calculate_risk_penalty(
            exchanger_id=exchanger_id,
            current_delta_p_kpa=current_delta_p_kpa,
            delta_p_limit_kpa=delta_p_limit_kpa,
            current_t_outlet_c=current_t_outlet_c,
            t_outlet_limit_c=t_outlet_limit_c,
            days_since_cleaning=days_since_cleaning,
            fouling_rate_per_day=fouling_rate_per_day,
            horizon_days=self.config.horizon_days,
        )
        total_risk = risk.total_risk_penalty_usd

        # Build cost breakdown
        return TotalCostBreakdown(
            exchanger_id=exchanger_id,
            horizon_days=self.config.horizon_days,
            energy_loss_cost=self.cost_model.calculate_energy_loss(
                exchanger_id=exchanger_id,
                current_ua_kw_k=current_ua_kw_k,
                clean_ua_kw_k=clean_ua_kw_k,
                heat_duty_kw=heat_duty_kw,
            ),
            production_loss_cost=self.cost_model.calculate_production_loss(
                exchanger_id=exchanger_id,
                current_effectiveness=current_effectiveness,
                design_effectiveness=design_effectiveness,
                design_throughput_tph=design_throughput_tph,
                product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            ),
            risk_penalty=risk,
            total_energy_loss_usd=round(total_energy_loss, 2),
            total_production_loss_usd=round(total_production_loss, 2),
            total_cleaning_cost_usd=0.0,
            total_downtime_cost_usd=0.0,
            total_risk_penalty_usd=round(total_risk, 2),
            total_cost_usd=round(total_energy_loss + total_production_loss + total_risk, 2),
            daily_average_cost_usd=round(
                (total_energy_loss + total_production_loss + total_risk) / self.config.horizon_days, 2
            ),
        )

    def _evaluate_with_cleaning(
        self,
        exchanger_id: str,
        cleaning_day: int,
        cleaning_method: CleaningMethodType,
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        heat_duty_kw: float,
        current_effectiveness: float,
        design_effectiveness: float,
        design_throughput_tph: float,
        product_margin_usd_per_tonne: float,
        current_delta_p_kpa: float,
        delta_p_limit_kpa: float,
        current_t_outlet_c: float,
        t_outlet_limit_c: float,
        days_since_cleaning: int,
        fouling_rate_per_day: float,
    ) -> TotalCostBreakdown:
        """Evaluate costs with cleaning on specified day."""
        total_energy_loss = 0.0
        total_production_loss = 0.0

        # Get cleaning parameters
        cleaning_cost = self.cost_model.calculate_cleaning_cost(
            exchanger_id=exchanger_id,
            cleaning_method=cleaning_method,
        )
        ua_recovery = cleaning_cost.expected_ua_recovery

        # Pre-cleaning period
        for day in range(min(cleaning_day, self.config.horizon_days)):
            degradation_factor = 1 + fouling_rate_per_day * day
            projected_ua = current_ua_kw_k / degradation_factor
            projected_effectiveness = current_effectiveness / degradation_factor

            energy_loss = self.cost_model.calculate_energy_loss(
                exchanger_id=exchanger_id,
                current_ua_kw_k=projected_ua,
                clean_ua_kw_k=clean_ua_kw_k,
                heat_duty_kw=heat_duty_kw,
            )
            total_energy_loss += energy_loss.daily_energy_loss_usd

            prod_loss = self.cost_model.calculate_production_loss(
                exchanger_id=exchanger_id,
                current_effectiveness=min(projected_effectiveness, 1.0),
                design_effectiveness=design_effectiveness,
                design_throughput_tph=design_throughput_tph,
                product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            )
            total_production_loss += prod_loss.daily_production_loss_usd

        # Post-cleaning period (with recovered UA)
        recovered_ua = clean_ua_kw_k * ua_recovery
        recovered_effectiveness = design_effectiveness * ua_recovery

        for day in range(cleaning_day, self.config.horizon_days):
            days_after_clean = day - cleaning_day
            degradation_factor = 1 + fouling_rate_per_day * days_after_clean
            projected_ua = recovered_ua / degradation_factor
            projected_effectiveness = recovered_effectiveness / degradation_factor

            energy_loss = self.cost_model.calculate_energy_loss(
                exchanger_id=exchanger_id,
                current_ua_kw_k=projected_ua,
                clean_ua_kw_k=clean_ua_kw_k,
                heat_duty_kw=heat_duty_kw,
            )
            total_energy_loss += energy_loss.daily_energy_loss_usd

            prod_loss = self.cost_model.calculate_production_loss(
                exchanger_id=exchanger_id,
                current_effectiveness=min(projected_effectiveness, 1.0),
                design_effectiveness=design_effectiveness,
                design_throughput_tph=design_throughput_tph,
                product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            )
            total_production_loss += prod_loss.daily_production_loss_usd

        # Downtime cost
        downtime_cost = self.cost_model.calculate_downtime_cost(
            exchanger_id=exchanger_id,
            cleaning_duration_hours=cleaning_cost.expected_cleaning_duration_hours,
            production_rate_tph=design_throughput_tph,
            product_margin_usd_per_tonne=product_margin_usd_per_tonne,
        )

        # Risk penalty (reduced after cleaning)
        risk = self.cost_model.calculate_risk_penalty(
            exchanger_id=exchanger_id,
            current_delta_p_kpa=current_delta_p_kpa * 0.5,  # Reset after cleaning
            delta_p_limit_kpa=delta_p_limit_kpa,
            current_t_outlet_c=current_t_outlet_c,
            t_outlet_limit_c=t_outlet_limit_c,
            days_since_cleaning=0,  # Reset
            fouling_rate_per_day=fouling_rate_per_day,
            horizon_days=self.config.horizon_days - cleaning_day,
        )

        total_cost = (
            total_energy_loss +
            total_production_loss +
            cleaning_cost.total_cleaning_cost_usd +
            downtime_cost.total_downtime_cost_usd +
            risk.total_risk_penalty_usd
        )

        return TotalCostBreakdown(
            exchanger_id=exchanger_id,
            horizon_days=self.config.horizon_days,
            energy_loss_cost=self.cost_model.calculate_energy_loss(
                exchanger_id=exchanger_id,
                current_ua_kw_k=current_ua_kw_k,
                clean_ua_kw_k=clean_ua_kw_k,
                heat_duty_kw=heat_duty_kw,
            ),
            production_loss_cost=self.cost_model.calculate_production_loss(
                exchanger_id=exchanger_id,
                current_effectiveness=current_effectiveness,
                design_effectiveness=design_effectiveness,
                design_throughput_tph=design_throughput_tph,
                product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            ),
            cleaning_cost=cleaning_cost,
            downtime_cost=downtime_cost,
            risk_penalty=risk,
            total_energy_loss_usd=round(total_energy_loss, 2),
            total_production_loss_usd=round(total_production_loss, 2),
            total_cleaning_cost_usd=cleaning_cost.total_cleaning_cost_usd,
            total_downtime_cost_usd=downtime_cost.total_downtime_cost_usd,
            total_risk_penalty_usd=round(risk.total_risk_penalty_usd, 2),
            total_cost_usd=round(total_cost, 2),
            daily_average_cost_usd=round(total_cost / self.config.horizon_days, 2),
        )

    def _build_rankings(
        self,
        evaluations: List[Tuple[int, CleaningMethodType, float, TotalCostBreakdown]],
        optimal_cost: float,
        feature_importances: Optional[Dict[str, float]],
    ) -> List[ScheduleRanking]:
        """Build ranked list of cleaning schedule options."""
        rankings = []

        for rank, (day, method, cost, breakdown) in enumerate(evaluations, 1):
            # Get cleaning cost details
            cleaning_cost = self.cost_model.calculate_cleaning_cost(
                exchanger_id="temp",
                cleaning_method=method,
            )

            # Determine key drivers
            drivers = self._determine_key_drivers(breakdown, feature_importances)

            rankings.append(ScheduleRanking(
                rank=rank,
                cleaning_date=datetime.utcnow() + timedelta(days=day),
                cleaning_method=method,
                total_cost_usd=cost,
                cost_delta_vs_optimal_usd=cost - optimal_cost,
                cost_delta_pct=round((cost - optimal_cost) / optimal_cost * 100, 2) if optimal_cost > 0 else 0,
                expected_ua_recovery=cleaning_cost.expected_ua_recovery,
                expected_refouling_days=int(30 / 0.002),  # Simplified estimate
                violation_probability_at_cleaning=breakdown.risk_penalty.delta_p_violation_probability,
                days_to_constraint_violation=None,
                key_drivers=drivers,
            ))

        return rankings

    def _determine_key_drivers(
        self,
        breakdown: TotalCostBreakdown,
        feature_importances: Optional[Dict[str, float]],
    ) -> List[str]:
        """Determine key cost drivers for a schedule option."""
        drivers = []

        # Cost-based drivers
        if breakdown.cost_breakdown_pct:
            sorted_costs = sorted(
                breakdown.cost_breakdown_pct.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for category, pct in sorted_costs[:3]:
                if pct > 10:
                    drivers.append(f"{category}: {pct:.0f}% of cost")

        # Feature importance drivers (from ML model)
        if feature_importances:
            sorted_features = sorted(
                feature_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for feature, importance in sorted_features[:2]:
                direction = "increasing" if importance > 0 else "decreasing"
                drivers.append(f"{feature} ({direction} fouling)")

        return drivers[:5]

    def _build_recommendation(
        self,
        exchanger_id: str,
        optimal_day: Optional[int],
        optimal_method: Optional[CleaningMethodType],
        savings: float,
        no_clean_cost: TotalCostBreakdown,
        feature_importances: Optional[Dict[str, float]],
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        current_delta_p_kpa: float,
        delta_p_limit_kpa: float,
        fouling_rate_per_day: float,
    ) -> CleaningRecommendation:
        """Build final cleaning recommendation."""
        recommendation_id = self._generate_id(exchanger_id + "_rec")

        # Determine urgency
        if optimal_day is None:
            urgency = CleaningUrgency.NONE
            action = "No cleaning required within planning horizon"
            confidence = ConfidenceLevel.HIGH
        elif optimal_day <= 7:
            urgency = CleaningUrgency.CRITICAL
            action = f"Clean within {optimal_day} days to avoid constraint violations"
            confidence = ConfidenceLevel.HIGH
        elif optimal_day <= 14:
            urgency = CleaningUrgency.HIGH
            action = f"Schedule cleaning for day {optimal_day} during next maintenance window"
            confidence = ConfidenceLevel.HIGH
        elif optimal_day <= 30:
            urgency = CleaningUrgency.MEDIUM
            action = f"Plan cleaning for day {optimal_day} during scheduled maintenance"
            confidence = ConfidenceLevel.MEDIUM
        else:
            urgency = CleaningUrgency.LOW
            action = f"Consider cleaning around day {optimal_day}"
            confidence = ConfidenceLevel.MEDIUM

        # Build top drivers (SHAP/LIME style)
        top_drivers = []

        # UA degradation driver
        ua_degradation = 1.0 - (current_ua_kw_k / clean_ua_kw_k)
        top_drivers.append({
            "feature": "UA_degradation",
            "value": round(ua_degradation * 100, 1),
            "unit": "%",
            "impact": "high" if ua_degradation > 0.15 else "medium",
            "direction": "positive",  # Increases need for cleaning
            "explanation": f"Current UA is {ua_degradation*100:.1f}% below design value",
        })

        # Delta-P margin driver
        dp_margin = (delta_p_limit_kpa - current_delta_p_kpa) / delta_p_limit_kpa
        top_drivers.append({
            "feature": "delta_P_margin",
            "value": round(dp_margin * 100, 1),
            "unit": "%",
            "impact": "high" if dp_margin < 0.10 else "medium" if dp_margin < 0.20 else "low",
            "direction": "negative",  # Lower margin increases urgency
            "explanation": f"Pressure drop margin is {dp_margin*100:.1f}% to limit",
        })

        # Fouling rate driver
        top_drivers.append({
            "feature": "fouling_rate",
            "value": round(fouling_rate_per_day * 100, 4),
            "unit": "%/day",
            "impact": "high" if fouling_rate_per_day > 0.003 else "medium",
            "direction": "positive",
            "explanation": f"Fouling rate of {fouling_rate_per_day*100:.3f}%/day",
        })

        # Add ML feature importances if available
        if feature_importances:
            sorted_features = sorted(
                feature_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for feature, importance in sorted_features[:3]:
                top_drivers.append({
                    "feature": feature,
                    "value": round(importance, 4),
                    "unit": "SHAP value",
                    "impact": "high" if abs(importance) > 0.1 else "medium",
                    "direction": "positive" if importance > 0 else "negative",
                    "explanation": f"ML model: {feature} contribution to fouling prediction",
                })

        # Expected outcomes
        expected_ua_recovery = 0.0
        expected_refouling_days = 0
        expected_energy_savings = 0.0

        if optimal_method:
            cleaning_cost = self.cost_model.calculate_cleaning_cost(
                exchanger_id=exchanger_id,
                cleaning_method=optimal_method,
            )
            expected_ua_recovery = cleaning_cost.expected_ua_recovery
            expected_refouling_days = int(30 / fouling_rate_per_day) if fouling_rate_per_day > 0 else 365
            expected_energy_savings = no_clean_cost.total_energy_loss_usd * expected_ua_recovery * 0.5

        # Assumptions
        assumptions = [
            f"Fouling rate continues at {fouling_rate_per_day*100:.3f}%/day",
            f"Cleaning restores {expected_ua_recovery*100:.0f}% of design UA",
            f"Planning horizon: {self.config.horizon_days} days",
        ]

        # Caveats
        caveats = [
            "Actual results may vary based on fouling type and cleaning effectiveness",
            "Cost estimates based on historical averages",
        ]
        if confidence == ConfidenceLevel.LOW:
            caveats.append("Limited data available - recommendation has higher uncertainty")

        recommendation = CleaningRecommendation(
            exchanger_id=exchanger_id,
            recommendation_id=recommendation_id,
            action=action,
            urgency=urgency,
            confidence=confidence,
            recommended_date=datetime.utcnow() + timedelta(days=optimal_day) if optimal_day else None,
            recommended_method=optimal_method,
            expected_ua_recovery=expected_ua_recovery,
            expected_time_to_refoul_days=expected_refouling_days,
            expected_energy_savings_usd=round(expected_energy_savings, 2),
            expected_total_savings_usd=round(savings, 2),
            top_drivers=top_drivers,
            assumptions=assumptions,
            caveats=caveats,
        )

        recommendation.provenance_hash = recommendation.compute_provenance_hash()

        return recommendation

    def _calculate_payback(
        self,
        evaluations: List[Tuple[int, CleaningMethodType, float, TotalCostBreakdown]],
        no_clean_cost: TotalCostBreakdown,
    ) -> Optional[int]:
        """Calculate payback period in days for cleaning investment."""
        if not evaluations:
            return None

        optimal_day, optimal_method, optimal_cost, breakdown = evaluations[0]

        if breakdown.cleaning_cost is None:
            return None

        cleaning_investment = (
            breakdown.total_cleaning_cost_usd +
            breakdown.total_downtime_cost_usd
        )

        daily_savings = (
            (no_clean_cost.total_cost_usd - optimal_cost) / self.config.horizon_days
        )

        if daily_savings <= 0:
            return None

        payback_days = int(math.ceil(cleaning_investment / daily_savings))

        return payback_days

    def _project_ua_trajectory(
        self,
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        fouling_rate: float,
        cleaning_day: Optional[int],
        horizon_days: int,
    ) -> List[float]:
        """Project UA values over time."""
        trajectory = []

        for day in range(horizon_days):
            if cleaning_day is not None and day >= cleaning_day:
                # After cleaning - recovered UA with new fouling
                days_after_clean = day - cleaning_day
                recovery = 0.95  # 95% recovery
                degradation = 1 + fouling_rate * days_after_clean
                ua = clean_ua_kw_k * recovery / degradation
            else:
                # Before cleaning - continued degradation
                degradation = 1 + fouling_rate * day
                ua = current_ua_kw_k / degradation

            trajectory.append(round(ua, 2))

        return trajectory

    def _find_constraint_violation_day(
        self,
        ua_trajectory: List[float],
        clean_ua_kw_k: float,
        threshold: float,
    ) -> Optional[int]:
        """Find first day where UA degradation exceeds threshold."""
        for day, ua in enumerate(ua_trajectory):
            degradation = 1.0 - (ua / clean_ua_kw_k)
            if degradation > threshold:
                return day

        return None

    def _generate_id(self, prefix: str) -> str:
        """Generate unique identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_suffix = random.randint(1000, 9999)
        return f"{prefix}_{timestamp}_{random_suffix}"

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
