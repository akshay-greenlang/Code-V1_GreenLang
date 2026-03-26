# -*- coding: utf-8 -*-
"""
Scenario Analysis Workflow
====================================

3-phase workflow for scenario modelling and probability assessment within
PACK-046 Intensity Metrics Pack.

Phases:
    1. ScenarioDefinition         -- Define scenarios including efficiency
                                     improvement, business growth, structural
                                     change, methodology update, and combined
                                     scenarios with parameter ranges and
                                     probability weights.
    2. Simulation                 -- Run ScenarioEngine with Monte Carlo
                                     simulation for each defined scenario,
                                     propagate uncertainty through intensity
                                     calculations, generate distribution of
                                     possible intensity outcomes.
    3. ProbabilityAssessment      -- Assess probability of target achievement
                                     under each scenario, calculate confidence
                                     intervals, identify risk factors and
                                     sensitivity of outcomes to key parameters.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and random sampling with fixed
seeds for reproducibility. SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    TCFD Recommendations - Scenario analysis for climate risks
    ESRS E1-9 - Anticipated financial effects of climate-related risks
    SBTi Monitoring Reporting Verification - Progress assessment
    ISO 14064-1:2018 - Uncertainty assessment

Schedule: Annually or when strategic changes warrant new analysis
Estimated duration: 1-2 weeks

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

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


class ScenarioPhase(str, Enum):
    """Scenario analysis workflow phases."""

    SCENARIO_DEFINITION = "scenario_definition"
    SIMULATION = "simulation"
    PROBABILITY_ASSESSMENT = "probability_assessment"


class ScenarioType(str, Enum):
    """Type of intensity scenario."""

    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    BUSINESS_GROWTH = "business_growth"
    STRUCTURAL_CHANGE = "structural_change"
    METHODOLOGY_UPDATE = "methodology_update"
    COMBINED = "combined"
    CUSTOM = "custom"


class SimulationMethod(str, Enum):
    """Monte Carlo simulation method."""

    MONTE_CARLO = "monte_carlo"
    LATIN_HYPERCUBE = "latin_hypercube"
    DETERMINISTIC = "deterministic"


class ProbabilityBand(str, Enum):
    """Probability classification band."""

    VERY_LIKELY = "very_likely"       # >90%
    LIKELY = "likely"                 # 66-90%
    ABOUT_AS_LIKELY = "about_as_likely"  # 33-66%
    UNLIKELY = "unlikely"             # 10-33%
    VERY_UNLIKELY = "very_unlikely"   # <10%


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class ScenarioParameter(BaseModel):
    """A parameter within a scenario definition."""

    name: str = Field(..., description="Parameter name")
    base_value: float = Field(default=0.0)
    min_value: float = Field(default=0.0)
    max_value: float = Field(default=0.0)
    distribution: str = Field(default="normal", description="normal|uniform|triangular")
    std_dev: float = Field(default=0.0, ge=0.0, description="Standard deviation for normal")


class ScenarioDefinition(BaseModel):
    """Definition of a single scenario for analysis."""

    scenario_id: str = Field(default_factory=lambda: f"sc-{_new_uuid()[:8]}")
    name: str = Field(..., min_length=1)
    scenario_type: ScenarioType = Field(...)
    description: str = Field(default="")
    parameters: List[ScenarioParameter] = Field(default_factory=list)
    probability_weight: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Prior probability weight for this scenario",
    )
    emissions_change_pct: float = Field(
        default=0.0, description="Expected emissions change (negative=reduction)",
    )
    denominator_change_pct: float = Field(
        default=0.0, description="Expected denominator change",
    )


class SimulationResult(BaseModel):
    """Result from Monte Carlo simulation for one scenario."""

    scenario_id: str = Field(...)
    scenario_name: str = Field(default="")
    iterations: int = Field(default=0, ge=0)
    mean_intensity: float = Field(default=0.0)
    median_intensity: float = Field(default=0.0)
    std_dev: float = Field(default=0.0)
    p5_intensity: float = Field(default=0.0, description="5th percentile")
    p25_intensity: float = Field(default=0.0, description="25th percentile")
    p75_intensity: float = Field(default=0.0, description="75th percentile")
    p95_intensity: float = Field(default=0.0, description="95th percentile")
    min_intensity: float = Field(default=0.0)
    max_intensity: float = Field(default=0.0)
    intensity_change_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ProbabilityAssessment(BaseModel):
    """Probability of target achievement under a scenario."""

    scenario_id: str = Field(...)
    scenario_name: str = Field(default="")
    target_intensity: float = Field(default=0.0)
    probability_of_achievement_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    probability_band: ProbabilityBand = Field(default=ProbabilityBand.ABOUT_AS_LIKELY)
    confidence_interval_90: Tuple[float, float] = Field(default=(0.0, 0.0))
    key_risk_factors: List[str] = Field(default_factory=list)
    sensitivity_ranking: List[str] = Field(
        default_factory=list,
        description="Parameters ranked by sensitivity impact",
    )
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class ScenarioWorkflowInput(BaseModel):
    """Input data model for ScenarioAnalysisWorkflow."""

    organization_id: str = Field(..., min_length=1)
    current_intensity: float = Field(..., gt=0.0, description="Current intensity value")
    current_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    current_denominator_value: float = Field(default=0.0, ge=0.0)
    target_intensity: float = Field(
        default=0.0, ge=0.0,
        description="Target intensity for probability assessment",
    )
    target_year: int = Field(default=2030, ge=2025, le=2050)
    scenarios: List[ScenarioDefinition] = Field(
        default_factory=list, description="Scenarios to analyse",
    )
    simulation_iterations: int = Field(
        default=10000, ge=100, le=1000000,
        description="Number of Monte Carlo iterations",
    )
    simulation_method: SimulationMethod = Field(
        default=SimulationMethod.MONTE_CARLO,
    )
    random_seed: int = Field(
        default=42, ge=0,
        description="Random seed for reproducibility",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class ScenarioWorkflowResult(BaseModel):
    """Complete result from scenario analysis workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="scenario_analysis")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    scenarios_analysed: int = Field(default=0)
    simulation_results: List[SimulationResult] = Field(default_factory=list)
    probability_assessments: List[ProbabilityAssessment] = Field(default_factory=list)
    best_case_scenario: str = Field(default="")
    worst_case_scenario: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ScenarioAnalysisWorkflow:
    """
    3-phase workflow for scenario modelling and probability assessment.

    Defines scenarios, runs Monte Carlo simulation, and assesses probability
    of target achievement under each scenario.

    Zero-hallucination: all simulations use deterministic random sampling
    with fixed seeds; statistical calculations use standard formulas; no LLM
    calls in numeric paths; SHA-256 provenance on every result.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _validated_scenarios: Validated scenario definitions.
        _sim_results: Monte Carlo simulation results.
        _assessments: Probability assessments.

    Example:
        >>> wf = ScenarioAnalysisWorkflow()
        >>> scenario = ScenarioDefinition(
        ...     name="Efficiency",
        ...     scenario_type=ScenarioType.EFFICIENCY_IMPROVEMENT,
        ...     emissions_change_pct=-10.0,
        ... )
        >>> inp = ScenarioWorkflowInput(
        ...     organization_id="org-001",
        ...     current_intensity=50.0,
        ...     target_intensity=40.0,
        ...     scenarios=[scenario],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[ScenarioPhase] = [
        ScenarioPhase.SCENARIO_DEFINITION,
        ScenarioPhase.SIMULATION,
        ScenarioPhase.PROBABILITY_ASSESSMENT,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ScenarioAnalysisWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._validated_scenarios: List[ScenarioDefinition] = []
        self._sim_results: List[SimulationResult] = []
        self._assessments: List[ProbabilityAssessment] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: ScenarioWorkflowInput,
    ) -> ScenarioWorkflowResult:
        """
        Execute the 3-phase scenario analysis workflow.

        Args:
            input_data: Current intensity, scenarios, and target data.

        Returns:
            ScenarioWorkflowResult with simulations and probability assessments.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting scenario analysis %s org=%s scenarios=%d iterations=%d",
            self.workflow_id, input_data.organization_id,
            len(input_data.scenarios), input_data.simulation_iterations,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_scenario_definition,
            self._phase_2_simulation,
            self._phase_3_probability_assessment,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Scenario analysis failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        # Determine best/worst scenarios
        best_case = ""
        worst_case = ""
        if self._sim_results:
            best = min(self._sim_results, key=lambda s: s.mean_intensity)
            worst = max(self._sim_results, key=lambda s: s.mean_intensity)
            best_case = best.scenario_name
            worst_case = worst.scenario_name

        result = ScenarioWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            scenarios_analysed=len(self._validated_scenarios),
            simulation_results=self._sim_results,
            probability_assessments=self._assessments,
            best_case_scenario=best_case,
            worst_case_scenario=worst_case,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Scenario analysis %s completed in %.2fs status=%s scenarios=%d",
            self.workflow_id, elapsed, overall_status.value,
            len(self._validated_scenarios),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Scenario Definition
    # -------------------------------------------------------------------------

    async def _phase_1_scenario_definition(
        self, input_data: ScenarioWorkflowInput,
    ) -> PhaseResult:
        """Define and validate scenarios."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._validated_scenarios = []

        if not input_data.scenarios:
            # Generate default scenarios if none provided
            defaults = self._generate_default_scenarios(input_data)
            self._validated_scenarios = defaults
            warnings.append(
                f"No scenarios provided; generated {len(defaults)} default scenarios"
            )
        else:
            for scenario in input_data.scenarios:
                # Validate parameters
                valid = True
                for param in scenario.parameters:
                    if param.min_value > param.max_value:
                        warnings.append(
                            f"Scenario {scenario.name}: param {param.name} "
                            f"min > max ({param.min_value} > {param.max_value})"
                        )
                        valid = False
                if valid:
                    self._validated_scenarios.append(scenario)

        # Normalise probability weights
        total_weight = sum(s.probability_weight for s in self._validated_scenarios)
        if total_weight > 0:
            for s in self._validated_scenarios:
                s.probability_weight = round(s.probability_weight / total_weight, 4)

        outputs["scenarios_defined"] = len(self._validated_scenarios)
        outputs["scenario_names"] = [s.name for s in self._validated_scenarios]
        outputs["scenario_types"] = [s.scenario_type.value for s in self._validated_scenarios]
        outputs["default_scenarios_used"] = not bool(input_data.scenarios)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 ScenarioDefinition: %d scenarios validated",
            len(self._validated_scenarios),
        )
        return PhaseResult(
            phase_name="scenario_definition", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Simulation
    # -------------------------------------------------------------------------

    async def _phase_2_simulation(
        self, input_data: ScenarioWorkflowInput,
    ) -> PhaseResult:
        """Run Monte Carlo simulation for each scenario."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._sim_results = []
        rng = random.Random(input_data.random_seed)

        for scenario in self._validated_scenarios:
            intensities = self._run_monte_carlo(
                scenario, input_data.current_intensity,
                input_data.simulation_iterations, rng,
            )

            if not intensities:
                warnings.append(f"Scenario {scenario.name}: no valid simulation results")
                continue

            intensities.sort()
            n = len(intensities)
            mean_val = sum(intensities) / n
            median_val = intensities[n // 2]
            variance = sum((v - mean_val) ** 2 for v in intensities) / n
            std_val = math.sqrt(variance)

            change_pct = 0.0
            if input_data.current_intensity > 0:
                change_pct = ((mean_val - input_data.current_intensity)
                              / input_data.current_intensity) * 100.0

            sim_data = {
                "scenario": scenario.scenario_id,
                "mean": round(mean_val, 6),
                "iterations": n,
            }

            self._sim_results.append(SimulationResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                iterations=n,
                mean_intensity=round(mean_val, 6),
                median_intensity=round(median_val, 6),
                std_dev=round(std_val, 6),
                p5_intensity=round(self._percentile(intensities, 5.0), 6),
                p25_intensity=round(self._percentile(intensities, 25.0), 6),
                p75_intensity=round(self._percentile(intensities, 75.0), 6),
                p95_intensity=round(self._percentile(intensities, 95.0), 6),
                min_intensity=round(intensities[0], 6),
                max_intensity=round(intensities[-1], 6),
                intensity_change_pct=round(change_pct, 4),
                provenance_hash=_compute_hash(sim_data),
            ))

        outputs["simulations_completed"] = len(self._sim_results)
        outputs["total_iterations"] = sum(s.iterations for s in self._sim_results)
        outputs["simulation_method"] = input_data.simulation_method.value
        outputs["random_seed"] = input_data.random_seed

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 Simulation: %d scenarios simulated",
            len(self._sim_results),
        )
        return PhaseResult(
            phase_name="simulation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Probability Assessment
    # -------------------------------------------------------------------------

    async def _phase_3_probability_assessment(
        self, input_data: ScenarioWorkflowInput,
    ) -> PhaseResult:
        """Assess probability of target achievement under each scenario."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._assessments = []
        target = input_data.target_intensity

        if target <= 0:
            warnings.append("No target intensity specified; using current * 0.8 as proxy")
            target = input_data.current_intensity * 0.8

        rng = random.Random(input_data.random_seed)

        for sim in self._sim_results:
            # Re-run to get raw samples for probability counting
            scenario = next(
                (s for s in self._validated_scenarios if s.scenario_id == sim.scenario_id),
                None,
            )
            if not scenario:
                continue

            samples = self._run_monte_carlo(
                scenario, input_data.current_intensity,
                input_data.simulation_iterations, rng,
            )

            # Count samples meeting target
            below_target = sum(1 for s in samples if s <= target)
            prob_pct = round((below_target / max(len(samples), 1)) * 100.0, 2)

            # Classify probability
            if prob_pct >= 90.0:
                band = ProbabilityBand.VERY_LIKELY
            elif prob_pct >= 66.0:
                band = ProbabilityBand.LIKELY
            elif prob_pct >= 33.0:
                band = ProbabilityBand.ABOUT_AS_LIKELY
            elif prob_pct >= 10.0:
                band = ProbabilityBand.UNLIKELY
            else:
                band = ProbabilityBand.VERY_UNLIKELY

            # 90% confidence interval
            ci_lower = sim.p5_intensity
            ci_upper = sim.p95_intensity

            # Risk factors
            risk_factors: List[str] = []
            if prob_pct < 50.0:
                risk_factors.append("Low probability of target achievement")
            if sim.std_dev > 0.2 * sim.mean_intensity:
                risk_factors.append("High outcome uncertainty (CV > 20%)")
            if scenario.denominator_change_pct > 10.0:
                risk_factors.append("Significant denominator growth assumed")

            # Sensitivity ranking (by parameter std_dev relative impact)
            sensitivity = [
                p.name for p in sorted(
                    scenario.parameters,
                    key=lambda p: abs(p.max_value - p.min_value),
                    reverse=True,
                )
            ]

            assess_data = {
                "scenario": sim.scenario_id,
                "probability": prob_pct,
                "target": target,
            }

            self._assessments.append(ProbabilityAssessment(
                scenario_id=sim.scenario_id,
                scenario_name=sim.scenario_name,
                target_intensity=target,
                probability_of_achievement_pct=prob_pct,
                probability_band=band,
                confidence_interval_90=(round(ci_lower, 6), round(ci_upper, 6)),
                key_risk_factors=risk_factors,
                sensitivity_ranking=sensitivity[:5],
                provenance_hash=_compute_hash(assess_data),
            ))

        outputs["assessments_completed"] = len(self._assessments)
        outputs["target_intensity"] = target
        outputs["probability_summary"] = {
            a.scenario_name: a.probability_of_achievement_pct
            for a in self._assessments
        }

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 ProbabilityAssessment: %d assessments",
            len(self._assessments),
        )
        return PhaseResult(
            phase_name="probability_assessment", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Simulation Helpers
    # -------------------------------------------------------------------------

    def _run_monte_carlo(
        self, scenario: ScenarioDefinition, current_intensity: float,
        iterations: int, rng: random.Random,
    ) -> List[float]:
        """Run Monte Carlo simulation for a single scenario."""
        results: List[float] = []

        for _ in range(iterations):
            # Sample emission change
            emission_change = self._sample_parameter(
                scenario.emissions_change_pct, 5.0, rng,
            )
            # Sample denominator change
            denom_change = self._sample_parameter(
                scenario.denominator_change_pct, 3.0, rng,
            )

            # Sample additional parameters
            param_effect = 0.0
            for param in scenario.parameters:
                sampled = self._sample_from_distribution(param, rng)
                if param.base_value != 0:
                    param_effect += (sampled - param.base_value) / abs(param.base_value) * 100.0

            # Calculate intensity: I = E * (1 + emission_change%) / D * (1 + denom_change%)
            total_emission_change = 1.0 + (emission_change + param_effect * 0.5) / 100.0
            total_denom_change = 1.0 + denom_change / 100.0

            if total_denom_change > 0:
                new_intensity = current_intensity * total_emission_change / total_denom_change
                if new_intensity >= 0:
                    results.append(new_intensity)

        return results

    def _sample_parameter(
        self, mean: float, std_dev: float, rng: random.Random,
    ) -> float:
        """Sample a parameter from normal distribution."""
        return rng.gauss(mean, std_dev)

    def _sample_from_distribution(
        self, param: ScenarioParameter, rng: random.Random,
    ) -> float:
        """Sample a value from a parameter's specified distribution."""
        if param.distribution == "uniform":
            return rng.uniform(param.min_value, param.max_value)
        elif param.distribution == "triangular":
            return rng.triangular(param.min_value, param.max_value, param.base_value)
        else:
            # Normal distribution
            return rng.gauss(param.base_value, param.std_dev if param.std_dev > 0 else 1.0)

    def _percentile(self, sorted_vals: List[float], pct: float) -> float:
        """Calculate percentile from sorted list."""
        if not sorted_vals:
            return 0.0
        n = len(sorted_vals)
        k = (pct / 100.0) * (n - 1)
        f = int(math.floor(k))
        c = min(int(math.ceil(k)), n - 1)
        if f == c:
            return sorted_vals[f]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    def _generate_default_scenarios(
        self, input_data: ScenarioWorkflowInput,
    ) -> List[ScenarioDefinition]:
        """Generate default scenarios when none are provided."""
        return [
            ScenarioDefinition(
                name="Business as Usual",
                scenario_type=ScenarioType.CUSTOM,
                description="No significant changes to operations or efficiency",
                emissions_change_pct=2.0,
                denominator_change_pct=3.0,
                probability_weight=0.3,
            ),
            ScenarioDefinition(
                name="Moderate Efficiency",
                scenario_type=ScenarioType.EFFICIENCY_IMPROVEMENT,
                description="Moderate efficiency improvements across operations",
                emissions_change_pct=-8.0,
                denominator_change_pct=2.0,
                probability_weight=0.3,
            ),
            ScenarioDefinition(
                name="Aggressive Decarbonisation",
                scenario_type=ScenarioType.EFFICIENCY_IMPROVEMENT,
                description="Aggressive decarbonisation with capital investment",
                emissions_change_pct=-20.0,
                denominator_change_pct=1.0,
                probability_weight=0.2,
            ),
            ScenarioDefinition(
                name="High Growth",
                scenario_type=ScenarioType.BUSINESS_GROWTH,
                description="Rapid business growth outpacing efficiency gains",
                emissions_change_pct=5.0,
                denominator_change_pct=15.0,
                probability_weight=0.2,
            ),
        ]

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: ScenarioWorkflowInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._validated_scenarios = []
        self._sim_results = []
        self._assessments = []

    def _compute_provenance(self, result: ScenarioWorkflowResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.organization_id}|{result.scenarios_analysed}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
