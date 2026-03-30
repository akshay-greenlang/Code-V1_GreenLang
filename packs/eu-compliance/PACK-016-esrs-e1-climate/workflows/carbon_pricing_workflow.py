# -*- coding: utf-8 -*-
"""
Carbon Pricing Workflow
==============================

4-phase workflow for carbon pricing disclosure per ESRS E1-8.
Implements mechanism setup, coverage calculation, scenario analysis,
and report generation with full provenance tracking.

Phases:
    1. MechanismSetup         -- Register carbon pricing mechanisms
    2. CoverageCalculation    -- Calculate emissions coverage by mechanism
    3. ScenarioAnalysis       -- Analyze shadow pricing / price scenarios
    4. ReportGeneration       -- Produce E1-8 disclosure data

Author: GreenLang Team
Version: 16.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class WorkflowPhase(str, Enum):
    """Phases of the carbon pricing workflow."""
    MECHANISM_SETUP = "mechanism_setup"
    COVERAGE_CALCULATION = "coverage_calculation"
    SCENARIO_ANALYSIS = "scenario_analysis"
    REPORT_GENERATION = "report_generation"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class MechanismType(str, Enum):
    """Carbon pricing mechanism type."""
    ETS = "ets"
    CARBON_TAX = "carbon_tax"
    INTERNAL_CARBON_PRICE = "internal_carbon_price"
    SHADOW_PRICE = "shadow_price"
    CBAM = "cbam"
    OFFSET_PURCHASE = "offset_purchase"

class PriceScenario(str, Enum):
    """Carbon price scenario."""
    LOW = "low"
    CENTRAL = "central"
    HIGH = "high"
    IEA_NZE = "iea_nze"
    EU_ETS_CURRENT = "eu_ets_current"
    CUSTOM = "custom"

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

class PricingMechanism(BaseModel):
    """A carbon pricing mechanism."""
    mechanism_id: str = Field(default_factory=lambda: f"pm-{_new_uuid()[:8]}")
    name: str = Field(..., description="Mechanism name")
    mechanism_type: MechanismType = Field(..., description="Pricing type")
    jurisdiction: str = Field(default="", description="Applicable jurisdiction")
    price_per_tco2e_eur: float = Field(default=0.0, ge=0.0)
    covered_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    total_cost_eur: float = Field(default=0.0, ge=0.0)
    scopes_covered: List[str] = Field(default_factory=list)
    is_mandatory: bool = Field(default=True)
    effective_date: str = Field(default="")
    description: str = Field(default="")

class CoverageResult(BaseModel):
    """Coverage calculation result."""
    mechanism_id: str = Field(default="")
    mechanism_name: str = Field(default="")
    covered_tco2e: float = Field(default=0.0)
    coverage_pct: float = Field(default=0.0)
    cost_eur: float = Field(default=0.0)
    effective_price_eur: float = Field(default=0.0)

class ScenarioResult(BaseModel):
    """Scenario analysis result."""
    scenario: str = Field(default="")
    price_per_tco2e_eur: float = Field(default=0.0)
    total_cost_eur: float = Field(default=0.0)
    cost_as_pct_revenue: float = Field(default=0.0)
    delta_vs_current_eur: float = Field(default=0.0)

class CarbonPricingInput(BaseModel):
    """Input data model for CarbonPricingWorkflow."""
    mechanisms: List[PricingMechanism] = Field(
        default_factory=list, description="Carbon pricing mechanisms"
    )
    total_emissions_tco2e: float = Field(
        default=0.0, ge=0.0, description="Total GHG emissions"
    )
    revenue_eur: float = Field(
        default=0.0, ge=0.0, description="Annual revenue for cost ratios"
    )
    shadow_price_eur: float = Field(
        default=0.0, ge=0.0, description="Internal shadow carbon price"
    )
    price_scenarios: Dict[str, float] = Field(
        default_factory=lambda: {
            "low": 50.0,
            "central": 100.0,
            "high": 200.0,
            "iea_nze": 250.0,
        },
        description="Price scenarios (EUR/tCO2e)"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class CarbonPricingResult(BaseModel):
    """Complete result from carbon pricing workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="carbon_pricing")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, description="Number of phases completed")
    duration_ms: float = Field(default=0.0, description="Total duration in milliseconds")
    total_duration_seconds: float = Field(default=0.0)
    mechanisms: List[PricingMechanism] = Field(default_factory=list)
    coverage_results: List[CoverageResult] = Field(default_factory=list)
    scenario_results: List[ScenarioResult] = Field(default_factory=list)
    total_coverage_pct: float = Field(default=0.0)
    total_carbon_cost_eur: float = Field(default=0.0)
    weighted_avg_price_eur: float = Field(default=0.0)
    shadow_price_eur: float = Field(default=0.0)
    carbon_cost_as_pct_revenue: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class CarbonPricingWorkflow:
    """
    4-phase carbon pricing disclosure workflow for ESRS E1-8.

    Implements carbon pricing mechanism registration, emissions coverage
    calculation, scenario analysis with shadow pricing, and disclosure-ready
    output generation.

    Zero-hallucination: all coverage and cost calculations use
    deterministic arithmetic.

    Example:
        >>> wf = CarbonPricingWorkflow()
        >>> inp = CarbonPricingInput(mechanisms=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_coverage_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CarbonPricingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._mechanisms: List[PricingMechanism] = []
        self._coverage: List[CoverageResult] = []
        self._scenarios: List[ScenarioResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.MECHANISM_SETUP.value, "description": "Register carbon pricing mechanisms"},
            {"name": WorkflowPhase.COVERAGE_CALCULATION.value, "description": "Calculate emissions coverage"},
            {"name": WorkflowPhase.SCENARIO_ANALYSIS.value, "description": "Analyze pricing scenarios"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Produce E1-8 disclosure data"},
        ]

    def validate_inputs(self, input_data: CarbonPricingInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.mechanisms and input_data.shadow_price_eur <= 0:
            issues.append("No pricing mechanisms or shadow price provided")
        if input_data.total_emissions_tco2e <= 0:
            issues.append("Total emissions must be positive for coverage calculation")
        return issues

    async def execute(
        self,
        input_data: Optional[CarbonPricingInput] = None,
        mechanisms: Optional[List[PricingMechanism]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> CarbonPricingResult:
        """
        Execute the 4-phase carbon pricing workflow.

        Args:
            input_data: Full input model (preferred).
            mechanisms: Pricing mechanisms (fallback).
            config: Configuration overrides.

        Returns:
            CarbonPricingResult with coverage, costs, and scenario analysis.
        """
        if input_data is None:
            input_data = CarbonPricingInput(
                mechanisms=mechanisms or [],
                config=config or {},
            )

        started_at = utcnow()
        self.logger.info("Starting carbon pricing workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_mechanism_setup(input_data))
            phase_results.append(await self._phase_coverage_calculation(input_data))
            phase_results.append(await self._phase_scenario_analysis(input_data))
            phase_results.append(await self._phase_report_generation(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Carbon pricing workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)
        total_cost = sum(m.total_cost_eur for m in self._mechanisms)
        total_covered = sum(c.covered_tco2e for c in self._coverage)
        coverage_pct = round(
            (total_covered / input_data.total_emissions_tco2e * 100)
            if input_data.total_emissions_tco2e > 0 else 0.0, 2
        )
        weighted_price = round(
            (total_cost / total_covered) if total_covered > 0 else 0.0, 2
        )
        cost_pct_revenue = round(
            (total_cost / input_data.revenue_eur * 100)
            if input_data.revenue_eur > 0 else 0.0, 4
        )

        result = CarbonPricingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            mechanisms=self._mechanisms,
            coverage_results=self._coverage,
            scenario_results=self._scenarios,
            total_coverage_pct=coverage_pct,
            total_carbon_cost_eur=round(total_cost, 2),
            weighted_avg_price_eur=weighted_price,
            shadow_price_eur=input_data.shadow_price_eur,
            carbon_cost_as_pct_revenue=cost_pct_revenue,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Carbon pricing %s completed in %.2fs: %.1f%% covered, %.0f EUR total cost",
            self.workflow_id, elapsed, coverage_pct, total_cost,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Mechanism Setup
    # -------------------------------------------------------------------------

    async def _phase_mechanism_setup(
        self, input_data: CarbonPricingInput,
    ) -> PhaseResult:
        """Register carbon pricing mechanisms."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._mechanisms = list(input_data.mechanisms)

        type_counts: Dict[str, int] = {}
        for m in self._mechanisms:
            type_counts[m.mechanism_type.value] = type_counts.get(m.mechanism_type.value, 0) + 1

        outputs["mechanisms_registered"] = len(self._mechanisms)
        outputs["type_distribution"] = type_counts
        outputs["mandatory_count"] = sum(1 for m in self._mechanisms if m.is_mandatory)
        outputs["voluntary_count"] = sum(1 for m in self._mechanisms if not m.is_mandatory)

        if input_data.shadow_price_eur > 0:
            outputs["shadow_price_eur"] = input_data.shadow_price_eur

        if not self._mechanisms:
            warnings.append("No carbon pricing mechanisms registered")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 MechanismSetup: %d mechanisms registered", len(self._mechanisms))
        return PhaseResult(
            phase_name=WorkflowPhase.MECHANISM_SETUP.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Coverage Calculation
    # -------------------------------------------------------------------------

    async def _phase_coverage_calculation(
        self, input_data: CarbonPricingInput,
    ) -> PhaseResult:
        """Calculate emissions coverage by each mechanism."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._coverage = []

        total_emissions = input_data.total_emissions_tco2e

        for mech in self._mechanisms:
            covered = mech.covered_emissions_tco2e
            coverage_pct = round(
                (covered / total_emissions * 100) if total_emissions > 0 else 0.0, 2
            )
            cost = mech.total_cost_eur or (covered * mech.price_per_tco2e_eur)
            effective_price = round((cost / covered) if covered > 0 else 0.0, 2)

            self._coverage.append(CoverageResult(
                mechanism_id=mech.mechanism_id,
                mechanism_name=mech.name,
                covered_tco2e=round(covered, 2),
                coverage_pct=coverage_pct,
                cost_eur=round(cost, 2),
                effective_price_eur=effective_price,
            ))

        total_covered = sum(c.covered_tco2e for c in self._coverage)
        total_coverage_pct = round(
            (total_covered / total_emissions * 100) if total_emissions > 0 else 0.0, 2
        )

        outputs["total_covered_tco2e"] = round(total_covered, 2)
        outputs["total_coverage_pct"] = total_coverage_pct
        outputs["mechanisms_with_coverage"] = sum(1 for c in self._coverage if c.covered_tco2e > 0)

        if total_coverage_pct < 50:
            warnings.append(
                f"Only {total_coverage_pct:.1f}% of emissions covered by pricing mechanisms"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 CoverageCalculation: %.1f%% of emissions covered",
            total_coverage_pct,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.COVERAGE_CALCULATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Scenario Analysis
    # -------------------------------------------------------------------------

    async def _phase_scenario_analysis(
        self, input_data: CarbonPricingInput,
    ) -> PhaseResult:
        """Analyze carbon pricing scenarios."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._scenarios = []

        total_covered = sum(c.covered_tco2e for c in self._coverage)
        current_cost = sum(c.cost_eur for c in self._coverage)

        for scenario_name, price in sorted(input_data.price_scenarios.items()):
            scenario_cost = total_covered * price
            cost_pct_rev = round(
                (scenario_cost / input_data.revenue_eur * 100)
                if input_data.revenue_eur > 0 else 0.0, 4
            )

            self._scenarios.append(ScenarioResult(
                scenario=scenario_name,
                price_per_tco2e_eur=price,
                total_cost_eur=round(scenario_cost, 2),
                cost_as_pct_revenue=cost_pct_rev,
                delta_vs_current_eur=round(scenario_cost - current_cost, 2),
            ))

        outputs["scenarios_analyzed"] = len(self._scenarios)
        outputs["scenario_costs"] = {
            s.scenario: s.total_cost_eur for s in self._scenarios
        }

        # Check high scenario impact
        high_scenario = next(
            (s for s in self._scenarios if s.scenario == "high"), None
        )
        if high_scenario and high_scenario.cost_as_pct_revenue > 5.0:
            warnings.append(
                f"High price scenario would cost {high_scenario.cost_as_pct_revenue:.1f}% "
                f"of revenue ({high_scenario.total_cost_eur:,.0f} EUR)"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 ScenarioAnalysis: %d scenarios analyzed", len(self._scenarios))
        return PhaseResult(
            phase_name=WorkflowPhase.SCENARIO_ANALYSIS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: CarbonPricingInput,
    ) -> PhaseResult:
        """Generate E1-8 disclosure-ready output."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_cost = sum(c.cost_eur for c in self._coverage)
        total_covered = sum(c.covered_tco2e for c in self._coverage)

        outputs["e1_8_disclosure"] = {
            "mechanisms_count": len(self._mechanisms),
            "total_carbon_cost_eur": round(total_cost, 2),
            "total_covered_emissions_tco2e": round(total_covered, 2),
            "coverage_pct": round(
                (total_covered / input_data.total_emissions_tco2e * 100)
                if input_data.total_emissions_tco2e > 0 else 0.0, 2
            ),
            "weighted_avg_price_eur": round(
                (total_cost / total_covered) if total_covered > 0 else 0.0, 2
            ),
            "shadow_price_eur": input_data.shadow_price_eur,
            "uses_internal_carbon_price": input_data.shadow_price_eur > 0,
            "scenario_count": len(self._scenarios),
            "reporting_year": input_data.reporting_year,
        }

        outputs["report_ready"] = True

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 ReportGeneration: E1-8 disclosure ready")
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: CarbonPricingResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
