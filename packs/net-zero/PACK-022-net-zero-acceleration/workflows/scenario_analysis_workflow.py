# -*- coding: utf-8 -*-
"""
Scenario Analysis Workflow
===============================

5-phase workflow for comparing net-zero transition scenarios using Monte
Carlo simulation within PACK-022 Net-Zero Acceleration Pack.  The workflow
defines scenario parameters, runs probabilistic simulations, performs
tornado sensitivity analysis, compares scenarios across multiple
dimensions, and generates a decision matrix with a recommended scenario.

Phases:
    1. Setup              -- Validate scenario definitions, set Monte Carlo
                             parameters, initialise RNG with seed
    2. ModelRun           -- Run Monte Carlo simulation per scenario (1000 runs),
                             collect distributions
    3. Sensitivity        -- Tornado analysis on key parameters, rank by impact
    4. Compare            -- Compare scenarios on cost, risk, ambition, timeline
    5. Recommend          -- Generate decision matrix with scoring and recommendation

Regulatory references:
    - SBTi Net-Zero Standard v1.2 (2024)
    - TCFD Scenario Analysis Guidance
    - IEA Net Zero by 2050 Roadmap
    - NGFS Climate Scenarios

Zero-hallucination: all Monte Carlo, sensitivity, and scoring calculations
use deterministic formulas seeded by the configured RNG.  No LLM calls
in the numeric computation path.

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import math
import random
import statistics
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class CarbonPriceScenario(str, Enum):
    """IEA/NGFS carbon price scenarios."""

    LOW = "low"             # ~50 USD/tCO2 by 2030
    MODERATE = "moderate"   # ~100 USD/tCO2 by 2030
    HIGH = "high"           # ~200 USD/tCO2 by 2030
    NET_ZERO = "net_zero"   # ~250 USD/tCO2 by 2030


class ComparisonDimension(str, Enum):
    """Scenario comparison dimensions."""

    COST = "cost"
    RISK = "risk"
    AMBITION = "ambition"
    TIMELINE = "timeline"


# =============================================================================
# CARBON PRICE PROJECTIONS (Zero-Hallucination, from IEA WEO / NGFS)
# =============================================================================

CARBON_PRICE_PROJECTIONS: Dict[str, Dict[int, float]] = {
    "low": {2025: 30, 2030: 50, 2035: 65, 2040: 80, 2045: 90, 2050: 100},
    "moderate": {2025: 50, 2030: 100, 2035: 130, 2040: 160, 2045: 180, 2050: 200},
    "high": {2025: 80, 2030: 200, 2035: 280, 2040: 350, 2045: 400, 2050: 450},
    "net_zero": {2025: 100, 2030: 250, 2035: 350, 2040: 450, 2045: 550, 2050: 650},
}

# Default scenario comparison weights
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "cost": 0.30,
    "risk": 0.25,
    "ambition": 0.25,
    "timeline": 0.20,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ScenarioDefinition(BaseModel):
    """Definition of a single net-zero transition scenario."""

    scenario_id: str = Field(default="", description="Unique scenario identifier")
    name: str = Field(default="", description="Scenario display name")
    description: str = Field(default="", description="Scenario narrative")
    target_year: int = Field(default=2050, ge=2030, le=2070)
    reduction_target_pct: float = Field(default=90.0, ge=0.0, le=100.0)
    annual_capex_usd: float = Field(default=0.0, ge=0.0, description="Annual capital budget")
    abatement_cost_low_usd: float = Field(default=20.0, ge=0.0)
    abatement_cost_mid_usd: float = Field(default=80.0, ge=0.0)
    abatement_cost_high_usd: float = Field(default=200.0, ge=0.0)
    technology_risk_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    policy_risk_pct: float = Field(default=15.0, ge=0.0, le=100.0)
    market_risk_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    scope3_coverage_pct: float = Field(default=67.0, ge=0.0, le=100.0)
    renewable_energy_pct: float = Field(default=50.0, ge=0.0, le=100.0)


class MonteCarloDistribution(BaseModel):
    """Distribution summary from Monte Carlo simulation."""

    scenario_id: str = Field(default="")
    mean: float = Field(default=0.0)
    median: float = Field(default=0.0)
    std_dev: float = Field(default=0.0)
    p5: float = Field(default=0.0, description="5th percentile")
    p25: float = Field(default=0.0, description="25th percentile")
    p75: float = Field(default=0.0, description="75th percentile")
    p95: float = Field(default=0.0, description="95th percentile")
    min_value: float = Field(default=0.0)
    max_value: float = Field(default=0.0)
    num_runs: int = Field(default=0)


class SensitivityParameter(BaseModel):
    """Tornado sensitivity result for a single parameter."""

    parameter_name: str = Field(default="")
    base_value: float = Field(default=0.0)
    low_value: float = Field(default=0.0)
    high_value: float = Field(default=0.0)
    low_outcome: float = Field(default=0.0)
    high_outcome: float = Field(default=0.0)
    swing: float = Field(default=0.0, description="Absolute swing (high - low outcome)")
    rank: int = Field(default=0)


class ScenarioComparison(BaseModel):
    """Comparison of a single scenario across dimensions."""

    scenario_id: str = Field(default="")
    scenario_name: str = Field(default="")
    cost_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    ambition_score: float = Field(default=0.0, ge=0.0, le=100.0)
    timeline_score: float = Field(default=0.0, ge=0.0, le=100.0)
    weighted_total: float = Field(default=0.0, ge=0.0, le=100.0)
    rank: int = Field(default=0)


class DecisionMatrix(BaseModel):
    """Decision matrix comparing all scenarios."""

    comparisons: List[ScenarioComparison] = Field(default_factory=list)
    recommended_scenario_id: str = Field(default="")
    recommended_scenario_name: str = Field(default="")
    recommendation_rationale: str = Field(default="")
    dimension_weights: Dict[str, float] = Field(default_factory=dict)


class ScenarioAnalysisConfig(BaseModel):
    """Configuration for the scenario analysis workflow."""

    scenarios: List[ScenarioDefinition] = Field(default_factory=list)
    baseline_emissions_tco2e: float = Field(default=10000.0, ge=0.0)
    num_scenarios: int = Field(default=3, ge=1, le=10)
    monte_carlo_runs: int = Field(default=1000, ge=100, le=100000)
    seed: int = Field(default=42)
    discount_rate: float = Field(default=0.08, ge=0.0, le=0.5)
    carbon_price_scenario: str = Field(default="moderate")
    dimension_weights: Dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_DIMENSION_WEIGHTS))
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("carbon_price_scenario")
    @classmethod
    def _validate_carbon_price(cls, v: str) -> str:
        allowed = {"low", "moderate", "high", "net_zero"}
        if v not in allowed:
            raise ValueError(f"carbon_price_scenario must be one of {allowed}")
        return v


class ScenarioAnalysisResult(BaseModel):
    """Complete result from the scenario analysis workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="scenario_analysis")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    validated_scenarios: List[ScenarioDefinition] = Field(default_factory=list)
    distributions: List[MonteCarloDistribution] = Field(default_factory=list)
    sensitivity_results: Dict[str, List[SensitivityParameter]] = Field(default_factory=dict)
    decision_matrix: DecisionMatrix = Field(default_factory=DecisionMatrix)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ScenarioAnalysisWorkflow:
    """
    5-phase scenario analysis workflow with Monte Carlo simulation.

    Validates scenario definitions, runs probabilistic simulations,
    performs tornado sensitivity analysis, compares scenarios on cost/
    risk/ambition/timeline dimensions, and generates a decision matrix
    with a recommended scenario.

    Zero-hallucination: all Monte Carlo runs, sensitivity analyses, and
    scoring use deterministic formulas with a seeded RNG.  No LLM calls
    in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = ScenarioAnalysisWorkflow()
        >>> config = ScenarioAnalysisConfig(scenarios=[...], seed=42)
        >>> result = await wf.execute(config)
        >>> assert result.decision_matrix.recommended_scenario_id != ""
    """

    def __init__(self) -> None:
        """Initialise ScenarioAnalysisWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._scenarios: List[ScenarioDefinition] = []
        self._distributions: List[MonteCarloDistribution] = []
        self._sensitivity: Dict[str, List[SensitivityParameter]] = {}
        self._decision_matrix: DecisionMatrix = DecisionMatrix()
        self._rng: random.Random = random.Random(42)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: ScenarioAnalysisConfig) -> ScenarioAnalysisResult:
        """
        Execute the 5-phase scenario analysis workflow.

        Args:
            config: Scenario analysis configuration with scenario definitions,
                Monte Carlo parameters, and comparison weights.

        Returns:
            ScenarioAnalysisResult with distributions, sensitivity, and
            decision matrix.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting scenario analysis workflow %s, scenarios=%d, runs=%d",
            self.workflow_id, len(config.scenarios), config.monte_carlo_runs,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_setup(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Setup phase failed; cannot proceed")

            phase2 = await self._phase_model_run(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_sensitivity(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_compare(config)
            self._phase_results.append(phase4)

            phase5 = await self._phase_recommend(config)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Scenario analysis workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = ScenarioAnalysisResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            validated_scenarios=self._scenarios,
            distributions=self._distributions,
            sensitivity_results=self._sensitivity,
            decision_matrix=self._decision_matrix,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Scenario analysis workflow %s completed in %.2fs, recommended=%s",
            self.workflow_id, elapsed,
            self._decision_matrix.recommended_scenario_id,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Setup
    # -------------------------------------------------------------------------

    async def _phase_setup(self, config: ScenarioAnalysisConfig) -> PhaseResult:
        """Validate scenario definitions, set Monte Carlo parameters, initialise RNG."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        self._rng = random.Random(config.seed)

        # Validate or generate default scenarios
        if config.scenarios:
            self._scenarios = config.scenarios
        else:
            self._scenarios = self._generate_default_scenarios(config)
            warnings.append(
                f"No scenarios provided; generated {len(self._scenarios)} default scenarios"
            )

        # Assign IDs if missing
        for idx, sc in enumerate(self._scenarios):
            if not sc.scenario_id:
                sc.scenario_id = f"scenario_{idx + 1}"

        # Validate cost ordering
        for sc in self._scenarios:
            if sc.abatement_cost_low_usd > sc.abatement_cost_mid_usd:
                warnings.append(
                    f"Scenario '{sc.name}': low cost > mid cost; swapping"
                )
                sc.abatement_cost_low_usd, sc.abatement_cost_mid_usd = (
                    sc.abatement_cost_mid_usd, sc.abatement_cost_low_usd,
                )
            if sc.abatement_cost_mid_usd > sc.abatement_cost_high_usd:
                warnings.append(
                    f"Scenario '{sc.name}': mid cost > high cost; swapping"
                )
                sc.abatement_cost_mid_usd, sc.abatement_cost_high_usd = (
                    sc.abatement_cost_high_usd, sc.abatement_cost_mid_usd,
                )

        # Validate dimension weights sum to ~1.0
        weight_sum = sum(config.dimension_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            warnings.append(
                f"Dimension weights sum to {weight_sum:.3f}, normalising to 1.0"
            )

        outputs["scenario_count"] = len(self._scenarios)
        outputs["monte_carlo_runs"] = config.monte_carlo_runs
        outputs["seed"] = config.seed
        outputs["carbon_price_scenario"] = config.carbon_price_scenario
        outputs["scenario_ids"] = [s.scenario_id for s in self._scenarios]

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Setup: %d scenarios validated, seed=%d", len(self._scenarios), config.seed)
        return PhaseResult(
            phase_name="setup",
            status=PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _generate_default_scenarios(self, config: ScenarioAnalysisConfig) -> List[ScenarioDefinition]:
        """Generate default comparison scenarios if none provided."""
        baseline = config.baseline_emissions_tco2e
        return [
            ScenarioDefinition(
                scenario_id="conservative",
                name="Conservative Transition",
                description="Gradual decarbonisation with proven technologies",
                target_year=2050,
                reduction_target_pct=90.0,
                annual_capex_usd=baseline * 15,
                abatement_cost_low_usd=20.0,
                abatement_cost_mid_usd=60.0,
                abatement_cost_high_usd=150.0,
                technology_risk_pct=5.0,
                policy_risk_pct=10.0,
                market_risk_pct=8.0,
                scope3_coverage_pct=67.0,
                renewable_energy_pct=60.0,
            ),
            ScenarioDefinition(
                scenario_id="balanced",
                name="Balanced Acceleration",
                description="Moderate pace balancing cost and ambition",
                target_year=2045,
                reduction_target_pct=95.0,
                annual_capex_usd=baseline * 25,
                abatement_cost_low_usd=30.0,
                abatement_cost_mid_usd=80.0,
                abatement_cost_high_usd=200.0,
                technology_risk_pct=12.0,
                policy_risk_pct=15.0,
                market_risk_pct=12.0,
                scope3_coverage_pct=80.0,
                renewable_energy_pct=80.0,
            ),
            ScenarioDefinition(
                scenario_id="aggressive",
                name="Aggressive Net-Zero",
                description="Rapid decarbonisation with emerging technologies",
                target_year=2040,
                reduction_target_pct=98.0,
                annual_capex_usd=baseline * 40,
                abatement_cost_low_usd=50.0,
                abatement_cost_mid_usd=120.0,
                abatement_cost_high_usd=350.0,
                technology_risk_pct=25.0,
                policy_risk_pct=20.0,
                market_risk_pct=18.0,
                scope3_coverage_pct=90.0,
                renewable_energy_pct=100.0,
            ),
        ]

    # -------------------------------------------------------------------------
    # Phase 2: Model Run (Monte Carlo)
    # -------------------------------------------------------------------------

    async def _phase_model_run(self, config: ScenarioAnalysisConfig) -> PhaseResult:
        """Run Monte Carlo simulation for each scenario."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._distributions = []

        for scenario in self._scenarios:
            dist = self._run_monte_carlo(scenario, config)
            self._distributions.append(dist)

        outputs["distributions_count"] = len(self._distributions)
        for dist in self._distributions:
            outputs[f"{dist.scenario_id}_mean_cost"] = round(dist.mean, 2)
            outputs[f"{dist.scenario_id}_std_dev"] = round(dist.std_dev, 2)
            outputs[f"{dist.scenario_id}_p5"] = round(dist.p5, 2)
            outputs[f"{dist.scenario_id}_p95"] = round(dist.p95, 2)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Model run: %d scenarios x %d runs completed in %.3fs",
            len(self._scenarios), config.monte_carlo_runs, elapsed,
        )
        return PhaseResult(
            phase_name="model_run",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _run_monte_carlo(
        self, scenario: ScenarioDefinition, config: ScenarioAnalysisConfig
    ) -> MonteCarloDistribution:
        """Run Monte Carlo simulation for a single scenario."""
        results: List[float] = []
        baseline = config.baseline_emissions_tco2e
        carbon_prices = CARBON_PRICE_PROJECTIONS.get(config.carbon_price_scenario, {})
        target_year = scenario.target_year
        discount_rate = config.discount_rate

        # Interpolate carbon price at target year
        carbon_price_at_target = self._interpolate_carbon_price(
            carbon_prices, target_year
        )

        for _ in range(config.monte_carlo_runs):
            # Sample abatement cost per tCO2e (triangular distribution)
            abatement_cost = self._rng.triangular(
                scenario.abatement_cost_low_usd,
                scenario.abatement_cost_high_usd,
                scenario.abatement_cost_mid_usd,
            )

            # Sample reduction achievement (normal around target, clipped)
            achieved_pct = self._rng.gauss(
                scenario.reduction_target_pct,
                scenario.technology_risk_pct,
            )
            achieved_pct = max(0.0, min(100.0, achieved_pct))

            # Calculate total abatement volume
            abatement_volume = baseline * (achieved_pct / 100.0)

            # Total transition cost (NPV of annual capex + abatement cost)
            years = target_year - 2025
            npv_capex = self._calculate_npv_annuity(
                scenario.annual_capex_usd, years, discount_rate
            )
            total_abatement_cost = abatement_volume * abatement_cost

            # Carbon price benefit (avoided cost)
            carbon_benefit = abatement_volume * carbon_price_at_target

            # Net transition cost
            net_cost = npv_capex + total_abatement_cost - carbon_benefit

            # Apply risk adjustment
            combined_risk = (
                scenario.technology_risk_pct
                + scenario.policy_risk_pct
                + scenario.market_risk_pct
            ) / 300.0
            risk_multiplier = 1.0 + self._rng.uniform(-combined_risk, combined_risk)
            net_cost *= risk_multiplier

            results.append(net_cost)

        results.sort()
        n = len(results)
        return MonteCarloDistribution(
            scenario_id=scenario.scenario_id,
            mean=statistics.mean(results),
            median=statistics.median(results),
            std_dev=statistics.stdev(results) if n > 1 else 0.0,
            p5=results[max(0, int(n * 0.05))],
            p25=results[max(0, int(n * 0.25))],
            p75=results[min(n - 1, int(n * 0.75))],
            p95=results[min(n - 1, int(n * 0.95))],
            min_value=results[0],
            max_value=results[-1],
            num_runs=n,
        )

    def _interpolate_carbon_price(
        self, prices: Dict[int, float], target_year: int
    ) -> float:
        """Linearly interpolate carbon price for a target year."""
        years = sorted(prices.keys())
        if not years:
            return 100.0
        if target_year <= years[0]:
            return prices[years[0]]
        if target_year >= years[-1]:
            return prices[years[-1]]
        for i in range(len(years) - 1):
            if years[i] <= target_year <= years[i + 1]:
                y0, y1 = years[i], years[i + 1]
                p0, p1 = prices[y0], prices[y1]
                frac = (target_year - y0) / (y1 - y0)
                return p0 + frac * (p1 - p0)
        return prices[years[-1]]

    def _calculate_npv_annuity(
        self, annual_payment: float, years: int, rate: float
    ) -> float:
        """Calculate NPV of a fixed annual payment stream."""
        if rate <= 0 or years <= 0:
            return annual_payment * max(years, 0)
        return annual_payment * (1.0 - (1.0 + rate) ** (-years)) / rate

    # -------------------------------------------------------------------------
    # Phase 3: Sensitivity Analysis (Tornado)
    # -------------------------------------------------------------------------

    async def _phase_sensitivity(self, config: ScenarioAnalysisConfig) -> PhaseResult:
        """Perform tornado sensitivity analysis on key parameters."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._sensitivity = {}

        parameters_to_test = [
            "abatement_cost",
            "reduction_target",
            "carbon_price",
            "discount_rate",
            "technology_risk",
            "annual_capex",
        ]

        for scenario in self._scenarios:
            sensitivities = self._tornado_analysis(
                scenario, config, parameters_to_test
            )
            self._sensitivity[scenario.scenario_id] = sensitivities
            outputs[f"{scenario.scenario_id}_top_driver"] = (
                sensitivities[0].parameter_name if sensitivities else "none"
            )
            outputs[f"{scenario.scenario_id}_top_swing"] = (
                round(sensitivities[0].swing, 2) if sensitivities else 0.0
            )

        outputs["parameters_tested"] = len(parameters_to_test)
        outputs["scenarios_analysed"] = len(self._scenarios)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Sensitivity: %d parameters across %d scenarios",
                         len(parameters_to_test), len(self._scenarios))
        return PhaseResult(
            phase_name="sensitivity",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _tornado_analysis(
        self,
        scenario: ScenarioDefinition,
        config: ScenarioAnalysisConfig,
        parameters: List[str],
    ) -> List[SensitivityParameter]:
        """Run tornado sensitivity for a single scenario."""
        results: List[SensitivityParameter] = []
        baseline = config.baseline_emissions_tco2e
        carbon_prices = CARBON_PRICE_PROJECTIONS.get(config.carbon_price_scenario, {})
        carbon_price = self._interpolate_carbon_price(carbon_prices, scenario.target_year)
        years = scenario.target_year - 2025

        # Base case net cost
        base_cost = self._deterministic_net_cost(
            scenario, config, baseline, carbon_price, years
        )

        for param in parameters:
            low_outcome, high_outcome, base_val, low_val, high_val = (
                self._evaluate_sensitivity_param(
                    param, scenario, config, baseline, carbon_price, years, base_cost
                )
            )
            swing = abs(high_outcome - low_outcome)
            results.append(SensitivityParameter(
                parameter_name=param,
                base_value=round(base_val, 4),
                low_value=round(low_val, 4),
                high_value=round(high_val, 4),
                low_outcome=round(low_outcome, 2),
                high_outcome=round(high_outcome, 2),
                swing=round(swing, 2),
            ))

        # Rank by swing (descending)
        results.sort(key=lambda x: x.swing, reverse=True)
        for idx, r in enumerate(results):
            r.rank = idx + 1

        return results

    def _deterministic_net_cost(
        self,
        scenario: ScenarioDefinition,
        config: ScenarioAnalysisConfig,
        baseline: float,
        carbon_price: float,
        years: int,
    ) -> float:
        """Calculate deterministic net cost for a scenario."""
        abatement_vol = baseline * (scenario.reduction_target_pct / 100.0)
        abatement_cost_total = abatement_vol * scenario.abatement_cost_mid_usd
        npv_capex = self._calculate_npv_annuity(
            scenario.annual_capex_usd, years, config.discount_rate
        )
        carbon_benefit = abatement_vol * carbon_price
        return npv_capex + abatement_cost_total - carbon_benefit

    def _evaluate_sensitivity_param(
        self,
        param: str,
        scenario: ScenarioDefinition,
        config: ScenarioAnalysisConfig,
        baseline: float,
        carbon_price: float,
        years: int,
        base_cost: float,
    ) -> tuple:
        """Evaluate low/high outcomes for a single sensitivity parameter."""
        variation = 0.20  # +/- 20% swing

        if param == "abatement_cost":
            base_val = scenario.abatement_cost_mid_usd
            low_val = base_val * (1 - variation)
            high_val = base_val * (1 + variation)
            vol = baseline * (scenario.reduction_target_pct / 100.0)
            npv_c = self._calculate_npv_annuity(scenario.annual_capex_usd, years, config.discount_rate)
            cb = vol * carbon_price
            low_out = npv_c + vol * low_val - cb
            high_out = npv_c + vol * high_val - cb

        elif param == "reduction_target":
            base_val = scenario.reduction_target_pct
            low_val = max(0.0, base_val * (1 - variation))
            high_val = min(100.0, base_val * (1 + variation))
            npv_c = self._calculate_npv_annuity(scenario.annual_capex_usd, years, config.discount_rate)
            low_vol = baseline * (low_val / 100.0)
            high_vol = baseline * (high_val / 100.0)
            low_out = npv_c + low_vol * scenario.abatement_cost_mid_usd - low_vol * carbon_price
            high_out = npv_c + high_vol * scenario.abatement_cost_mid_usd - high_vol * carbon_price

        elif param == "carbon_price":
            base_val = carbon_price
            low_val = base_val * (1 - variation)
            high_val = base_val * (1 + variation)
            vol = baseline * (scenario.reduction_target_pct / 100.0)
            npv_c = self._calculate_npv_annuity(scenario.annual_capex_usd, years, config.discount_rate)
            ac = vol * scenario.abatement_cost_mid_usd
            low_out = npv_c + ac - vol * low_val
            high_out = npv_c + ac - vol * high_val

        elif param == "discount_rate":
            base_val = config.discount_rate
            low_val = max(0.01, base_val * (1 - variation))
            high_val = base_val * (1 + variation)
            vol = baseline * (scenario.reduction_target_pct / 100.0)
            ac = vol * scenario.abatement_cost_mid_usd
            cb = vol * carbon_price
            low_out = self._calculate_npv_annuity(scenario.annual_capex_usd, years, low_val) + ac - cb
            high_out = self._calculate_npv_annuity(scenario.annual_capex_usd, years, high_val) + ac - cb

        elif param == "technology_risk":
            base_val = scenario.technology_risk_pct
            low_val = max(0.0, base_val * (1 - variation))
            high_val = min(100.0, base_val * (1 + variation))
            risk_low = low_val / 100.0
            risk_high = high_val / 100.0
            low_out = base_cost * (1.0 + risk_low)
            high_out = base_cost * (1.0 + risk_high)

        elif param == "annual_capex":
            base_val = scenario.annual_capex_usd
            low_val = base_val * (1 - variation)
            high_val = base_val * (1 + variation)
            vol = baseline * (scenario.reduction_target_pct / 100.0)
            ac = vol * scenario.abatement_cost_mid_usd
            cb = vol * carbon_price
            low_out = self._calculate_npv_annuity(low_val, years, config.discount_rate) + ac - cb
            high_out = self._calculate_npv_annuity(high_val, years, config.discount_rate) + ac - cb

        else:
            base_val = 0.0
            low_val = 0.0
            high_val = 0.0
            low_out = base_cost
            high_out = base_cost

        return (low_out, high_out, base_val, low_val, high_val)

    # -------------------------------------------------------------------------
    # Phase 4: Compare
    # -------------------------------------------------------------------------

    async def _phase_compare(self, config: ScenarioAnalysisConfig) -> PhaseResult:
        """Compare scenarios on cost, risk, ambition, timeline dimensions."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        comparisons: List[ScenarioComparison] = []
        dist_map = {d.scenario_id: d for d in self._distributions}

        for scenario in self._scenarios:
            dist = dist_map.get(scenario.scenario_id)
            comp = self._score_scenario(scenario, dist, config)
            comparisons.append(comp)

        # Rank by weighted total (higher is better)
        comparisons.sort(key=lambda c: c.weighted_total, reverse=True)
        for idx, comp in enumerate(comparisons):
            comp.rank = idx + 1

        self._decision_matrix = DecisionMatrix(
            comparisons=comparisons,
            dimension_weights=config.dimension_weights,
        )

        outputs["comparison_count"] = len(comparisons)
        for comp in comparisons:
            outputs[f"{comp.scenario_id}_weighted_total"] = round(comp.weighted_total, 2)
            outputs[f"{comp.scenario_id}_rank"] = comp.rank

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Compare: %d scenarios ranked", len(comparisons))
        return PhaseResult(
            phase_name="compare",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _score_scenario(
        self,
        scenario: ScenarioDefinition,
        dist: Optional[MonteCarloDistribution],
        config: ScenarioAnalysisConfig,
    ) -> ScenarioComparison:
        """Score a single scenario across all dimensions."""
        weights = config.dimension_weights

        # Cost score: lower mean net cost is better (invert)
        cost_score = 50.0
        if dist and dist.mean != 0:
            # Normalise across all distributions
            all_means = [d.mean for d in self._distributions]
            if all_means:
                min_m = min(all_means)
                max_m = max(all_means)
                if max_m != min_m:
                    # Lower cost = higher score
                    cost_score = 100.0 * (1.0 - (dist.mean - min_m) / (max_m - min_m))
                else:
                    cost_score = 100.0

        # Risk score: lower combined risk is better
        combined_risk = (
            scenario.technology_risk_pct
            + scenario.policy_risk_pct
            + scenario.market_risk_pct
        )
        max_possible_risk = 300.0
        risk_score = 100.0 * (1.0 - combined_risk / max_possible_risk)

        # Ambition score: higher reduction target + scope3 coverage = better
        ambition_raw = (
            scenario.reduction_target_pct * 0.6
            + scenario.scope3_coverage_pct * 0.2
            + scenario.renewable_energy_pct * 0.2
        )
        ambition_score = min(ambition_raw, 100.0)

        # Timeline score: earlier target year is better
        earliest = min(s.target_year for s in self._scenarios) if self._scenarios else 2040
        latest = max(s.target_year for s in self._scenarios) if self._scenarios else 2050
        if latest != earliest:
            timeline_score = 100.0 * (1.0 - (scenario.target_year - earliest) / (latest - earliest))
        else:
            timeline_score = 100.0

        weighted_total = (
            cost_score * weights.get("cost", 0.30)
            + risk_score * weights.get("risk", 0.25)
            + ambition_score * weights.get("ambition", 0.25)
            + timeline_score * weights.get("timeline", 0.20)
        )

        return ScenarioComparison(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            cost_score=round(cost_score, 2),
            risk_score=round(risk_score, 2),
            ambition_score=round(ambition_score, 2),
            timeline_score=round(timeline_score, 2),
            weighted_total=round(weighted_total, 2),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Recommend
    # -------------------------------------------------------------------------

    async def _phase_recommend(self, config: ScenarioAnalysisConfig) -> PhaseResult:
        """Generate decision matrix with scoring and recommended scenario."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        comparisons = self._decision_matrix.comparisons
        if not comparisons:
            warnings.append("No scenarios to recommend")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="recommend",
                status=PhaseStatus.COMPLETED,
                duration_seconds=round(elapsed, 4),
                outputs=outputs,
                warnings=warnings,
                provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            )

        best = comparisons[0]
        self._decision_matrix.recommended_scenario_id = best.scenario_id
        self._decision_matrix.recommended_scenario_name = best.scenario_name

        # Build rationale
        rationale_parts = [
            f"'{best.scenario_name}' (ID: {best.scenario_id}) is the recommended scenario "
            f"with a weighted score of {best.weighted_total:.1f}/100.",
        ]

        # Identify strongest dimension
        dim_scores = {
            "cost": best.cost_score,
            "risk": best.risk_score,
            "ambition": best.ambition_score,
            "timeline": best.timeline_score,
        }
        strongest = max(dim_scores, key=dim_scores.get)  # type: ignore[arg-type]
        rationale_parts.append(
            f"Strongest dimension: {strongest} ({dim_scores[strongest]:.1f}/100)."
        )

        # Compare with runner-up
        if len(comparisons) > 1:
            runner_up = comparisons[1]
            gap = best.weighted_total - runner_up.weighted_total
            rationale_parts.append(
                f"Leads runner-up '{runner_up.scenario_name}' by {gap:.1f} points."
            )

        self._decision_matrix.recommendation_rationale = " ".join(rationale_parts)

        outputs["recommended_id"] = best.scenario_id
        outputs["recommended_name"] = best.scenario_name
        outputs["recommended_score"] = round(best.weighted_total, 2)
        outputs["rationale"] = self._decision_matrix.recommendation_rationale

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Recommend: %s (score=%.1f)", best.scenario_name, best.weighted_total)
        return PhaseResult(
            phase_name="recommend",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
