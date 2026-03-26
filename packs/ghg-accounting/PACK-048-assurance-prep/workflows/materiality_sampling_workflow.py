# -*- coding: utf-8 -*-
"""
Materiality & Sampling Workflow
====================================

5-phase workflow for GHG assurance materiality determination and sampling
plan development covering materiality calculation, population identification,
stratification, sample sizing, and selection plan within PACK-048 GHG
Assurance Prep Pack.

Phases:
    1. MaterialityCalculation      -- Calculate overall materiality,
                                      performance materiality, and clearly
                                      trivial threshold using total reported
                                      emissions, revenue, and sector-specific
                                      benchmarks with Decimal arithmetic.
    2. PopulationIdentification    -- Identify the complete population of
                                      data points, facilities, and emission
                                      sources that form the basis for
                                      verification sampling.
    3. Stratification              -- Stratify the population by scope,
                                      category, risk level, and materiality
                                      contribution, assigning each item to
                                      a stratum for differentiated sampling.
    4. SampleSizing               -- Calculate statistically-grounded sample
                                      sizes per stratum using confidence level,
                                      expected error rate, and population size
                                      to achieve the required assurance level.
    5. SelectionPlan               -- Produce the final selection plan with
                                      documentation of sampling methodology,
                                      stratum allocations, and selection
                                      criteria for the verifier.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ISAE 3410 (2012) - Materiality and sampling in GHG assurance
    ISA 320 (2009) - Materiality in planning and performing an audit
    ISA 530 (2009) - Audit sampling
    ISO 14064-3:2019 - Materiality for verification
    ESRS 1 (2024) - Materiality principles for sustainability reporting
    PCAF Global Standard (2022) - Data quality and materiality

Schedule: At engagement planning stage
Estimated duration: 1-2 weeks

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
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


class MaterialitySamplingPhase(str, Enum):
    """Materiality and sampling workflow phases."""

    MATERIALITY_CALCULATION = "materiality_calculation"
    POPULATION_IDENTIFICATION = "population_identification"
    STRATIFICATION = "stratification"
    SAMPLE_SIZING = "sample_sizing"
    SELECTION_PLAN = "selection_plan"


class AssuranceLevel(str, Enum):
    """Assurance engagement level."""

    LIMITED = "limited"
    REASONABLE = "reasonable"


class MaterialityBasis(str, Enum):
    """Basis for materiality calculation."""

    TOTAL_EMISSIONS = "total_emissions"
    REVENUE = "revenue"
    SCOPE_1_2 = "scope_1_2"
    SECTOR_BENCHMARK = "sector_benchmark"


class RiskLevel(str, Enum):
    """Risk level for stratification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StratumScope(str, Enum):
    """Scope classification for strata."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class SelectionMethod(str, Enum):
    """Sample selection method."""

    MONETARY_UNIT = "monetary_unit"
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    JUDGEMENTAL = "judgemental"
    ALL_ITEMS = "all_items"


# =============================================================================
# MATERIALITY REFERENCE DATA (Zero-Hallucination)
# =============================================================================

MATERIALITY_PERCENTAGES: Dict[str, Dict[str, Decimal]] = {
    "limited": {
        "overall_pct": Decimal("5.0"),
        "performance_pct": Decimal("75.0"),
        "trivial_pct": Decimal("3.0"),
    },
    "reasonable": {
        "overall_pct": Decimal("5.0"),
        "performance_pct": Decimal("60.0"),
        "trivial_pct": Decimal("2.0"),
    },
}

CONFIDENCE_LEVELS: Dict[str, Dict[str, Any]] = {
    "limited": {"confidence_pct": 80, "z_score": Decimal("1.28")},
    "reasonable": {"confidence_pct": 95, "z_score": Decimal("1.96")},
}

EXPECTED_ERROR_RATES: Dict[str, Decimal] = {
    "high": Decimal("5.0"),
    "medium": Decimal("3.0"),
    "low": Decimal("1.5"),
}


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


class MaterialityResult(BaseModel):
    """Materiality calculation result."""

    basis: MaterialityBasis = Field(default=MaterialityBasis.TOTAL_EMISSIONS)
    basis_value_tco2e: str = Field(default="0.00")
    overall_materiality_tco2e: str = Field(default="0.00")
    overall_materiality_pct: str = Field(default="0.00")
    performance_materiality_tco2e: str = Field(default="0.00")
    performance_materiality_pct: str = Field(default="0.00")
    clearly_trivial_tco2e: str = Field(default="0.00")
    clearly_trivial_pct: str = Field(default="0.00")
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    provenance_hash: str = Field(default="")


class PopulationItem(BaseModel):
    """A single item in the verification population."""

    item_id: str = Field(default_factory=lambda: f"pop-{_new_uuid()[:8]}")
    description: str = Field(default="")
    scope: StratumScope = Field(default=StratumScope.SCOPE_1)
    category: str = Field(default="")
    facility: str = Field(default="")
    emissions_tco2e: str = Field(default="0.00")
    pct_of_total: str = Field(default="0.00")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    stratum_id: str = Field(default="")
    provenance_hash: str = Field(default="")


class Stratum(BaseModel):
    """A stratum within the sampling plan."""

    stratum_id: str = Field(default_factory=lambda: f"str-{_new_uuid()[:8]}")
    name: str = Field(default="")
    scope: StratumScope = Field(default=StratumScope.SCOPE_1)
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    population_count: int = Field(default=0, ge=0)
    total_emissions_tco2e: str = Field(default="0.00")
    pct_of_total: str = Field(default="0.00")
    is_key_item: bool = Field(default=False)
    sample_size: int = Field(default=0, ge=0)
    selection_method: SelectionMethod = Field(default=SelectionMethod.RANDOM)
    provenance_hash: str = Field(default="")


class SamplingPlan(BaseModel):
    """Final sampling plan document."""

    plan_id: str = Field(default_factory=lambda: f"plan-{_new_uuid()[:8]}")
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    confidence_level_pct: int = Field(default=80)
    total_population: int = Field(default=0, ge=0)
    total_sample_size: int = Field(default=0, ge=0)
    coverage_pct: str = Field(default="0.00")
    strata_count: int = Field(default=0, ge=0)
    methodology_notes: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class MaterialitySamplingInput(BaseModel):
    """Input data model for MaterialitySamplingWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organisation identifier")
    organization_name: str = Field(default="", description="Organisation display name")
    total_emissions_tco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total reported emissions in tCO2e",
    )
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    revenue_usd_m: float = Field(default=0.0, ge=0.0)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    materiality_basis: MaterialityBasis = Field(
        default=MaterialityBasis.TOTAL_EMISSIONS,
    )
    population_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Data points / facilities / sources forming the population",
    )
    risk_assessments: Dict[str, str] = Field(
        default_factory=dict,
        description="Risk level per item ID (high/medium/low)",
    )
    reporting_period: str = Field(default="2025")
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class MaterialitySamplingResult(BaseModel):
    """Complete result from materiality and sampling workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="materiality_sampling")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    materiality: Optional[MaterialityResult] = Field(default=None)
    population: List[PopulationItem] = Field(default_factory=list)
    strata: List[Stratum] = Field(default_factory=list)
    sampling_plan: Optional[SamplingPlan] = Field(default=None)
    overall_materiality_tco2e: str = Field(default="0.00")
    total_sample_size: int = Field(default=0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MaterialitySamplingWorkflow:
    """
    5-phase workflow for materiality determination and sampling plan.

    Calculates overall, performance, and clearly trivial materiality,
    identifies the population, stratifies by risk and scope, calculates
    sample sizes, and produces the selection plan.

    Zero-hallucination: all materiality calculations use Decimal with
    ROUND_HALF_UP; sample sizes use statistical formulas with fixed
    parameters; no LLM calls in numeric paths; SHA-256 provenance on
    every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _materiality: Materiality calculation result.
        _population: Identified population items.
        _strata: Stratification results.
        _plan: Final sampling plan.

    Example:
        >>> wf = MaterialitySamplingWorkflow()
        >>> inp = MaterialitySamplingInput(
        ...     organization_id="org-001",
        ...     total_emissions_tco2e=100000.0,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[MaterialitySamplingPhase] = [
        MaterialitySamplingPhase.MATERIALITY_CALCULATION,
        MaterialitySamplingPhase.POPULATION_IDENTIFICATION,
        MaterialitySamplingPhase.STRATIFICATION,
        MaterialitySamplingPhase.SAMPLE_SIZING,
        MaterialitySamplingPhase.SELECTION_PLAN,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MaterialitySamplingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._materiality: Optional[MaterialityResult] = None
        self._population: List[PopulationItem] = []
        self._strata: List[Stratum] = []
        self._plan: Optional[SamplingPlan] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: MaterialitySamplingInput,
    ) -> MaterialitySamplingResult:
        """
        Execute the 5-phase materiality and sampling workflow.

        Args:
            input_data: Organisation emissions, population, and risk data.

        Returns:
            MaterialitySamplingResult with materiality and sampling plan.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting materiality sampling %s org=%s emissions=%.0f",
            self.workflow_id, input_data.organization_id,
            input_data.total_emissions_tco2e,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_materiality_calculation,
            self._phase_2_population_identification,
            self._phase_3_stratification,
            self._phase_4_sample_sizing,
            self._phase_5_selection_plan,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Materiality sampling failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        overall_mat = "0.00"
        if self._materiality:
            overall_mat = self._materiality.overall_materiality_tco2e

        total_sample = 0
        if self._plan:
            total_sample = self._plan.total_sample_size

        result = MaterialitySamplingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            materiality=self._materiality,
            population=self._population,
            strata=self._strata,
            sampling_plan=self._plan,
            overall_materiality_tco2e=overall_mat,
            total_sample_size=total_sample,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Materiality sampling %s completed in %.2fs status=%s materiality=%s sample=%d",
            self.workflow_id, elapsed, overall_status.value,
            overall_mat, total_sample,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Materiality Calculation
    # -------------------------------------------------------------------------

    async def _phase_1_materiality_calculation(
        self, input_data: MaterialitySamplingInput,
    ) -> PhaseResult:
        """Calculate overall, performance, and clearly trivial materiality."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        level_key = input_data.assurance_level.value
        mat_pcts = MATERIALITY_PERCENTAGES.get(level_key, MATERIALITY_PERCENTAGES["limited"])

        # Determine basis value
        if input_data.materiality_basis == MaterialityBasis.TOTAL_EMISSIONS:
            basis_val = Decimal(str(input_data.total_emissions_tco2e))
        elif input_data.materiality_basis == MaterialityBasis.SCOPE_1_2:
            basis_val = Decimal(str(input_data.scope1_tco2e + input_data.scope2_tco2e))
        elif input_data.materiality_basis == MaterialityBasis.REVENUE:
            basis_val = Decimal(str(input_data.total_emissions_tco2e))
            warnings.append("Revenue-based materiality uses total emissions as proxy")
        else:
            basis_val = Decimal(str(input_data.total_emissions_tco2e))

        if basis_val <= Decimal("0"):
            return PhaseResult(
                phase_name="materiality_calculation", phase_number=1,
                status=PhaseStatus.FAILED,
                errors=["Basis value is zero or negative; cannot calculate materiality"],
                duration_seconds=time.monotonic() - started,
            )

        # Overall materiality
        overall_pct = mat_pcts["overall_pct"]
        overall_mat = (
            basis_val * overall_pct / Decimal("100")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Performance materiality
        perf_pct_of_overall = mat_pcts["performance_pct"]
        perf_mat = (
            overall_mat * perf_pct_of_overall / Decimal("100")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Clearly trivial
        trivial_pct = mat_pcts["trivial_pct"]
        trivial_mat = (
            overall_mat * trivial_pct / Decimal("100")
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        mat_data = {
            "basis": str(basis_val), "overall": str(overall_mat),
            "performance": str(perf_mat), "trivial": str(trivial_mat),
        }
        self._materiality = MaterialityResult(
            basis=input_data.materiality_basis,
            basis_value_tco2e=str(basis_val),
            overall_materiality_tco2e=str(overall_mat),
            overall_materiality_pct=str(overall_pct),
            performance_materiality_tco2e=str(perf_mat),
            performance_materiality_pct=str(
                (perf_pct_of_overall * overall_pct / Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                ),
            ),
            clearly_trivial_tco2e=str(trivial_mat),
            clearly_trivial_pct=str(
                (trivial_pct * overall_pct / Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                ),
            ),
            assurance_level=input_data.assurance_level,
            provenance_hash=_compute_hash(mat_data),
        )

        outputs["basis"] = input_data.materiality_basis.value
        outputs["basis_value_tco2e"] = str(basis_val)
        outputs["overall_materiality_tco2e"] = str(overall_mat)
        outputs["performance_materiality_tco2e"] = str(perf_mat)
        outputs["clearly_trivial_tco2e"] = str(trivial_mat)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 MaterialityCalculation: overall=%s perf=%s trivial=%s",
            str(overall_mat), str(perf_mat), str(trivial_mat),
        )
        return PhaseResult(
            phase_name="materiality_calculation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Population Identification
    # -------------------------------------------------------------------------

    async def _phase_2_population_identification(
        self, input_data: MaterialitySamplingInput,
    ) -> PhaseResult:
        """Identify complete population of data points, facilities, sources."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._population = []
        total_emissions = Decimal(str(max(input_data.total_emissions_tco2e, 0.001)))

        for item_data in input_data.population_items:
            scope_str = item_data.get("scope", "scope_1")
            try:
                scope = StratumScope(scope_str)
            except ValueError:
                scope = StratumScope.SCOPE_1

            risk_str = input_data.risk_assessments.get(
                item_data.get("item_id", ""), "medium",
            )
            try:
                risk = RiskLevel(risk_str)
            except ValueError:
                risk = RiskLevel.MEDIUM

            emissions = Decimal(str(item_data.get("emissions_tco2e", 0.0)))
            pct_of_total = (emissions / total_emissions * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )

            item = PopulationItem(
                item_id=item_data.get("item_id", f"pop-{_new_uuid()[:8]}"),
                description=item_data.get("description", ""),
                scope=scope,
                category=item_data.get("category", ""),
                facility=item_data.get("facility", ""),
                emissions_tco2e=str(emissions),
                pct_of_total=str(pct_of_total),
                risk_level=risk,
            )
            item_hash = {"id": item.item_id, "emissions": str(emissions), "scope": scope.value}
            item.provenance_hash = _compute_hash(item_hash)
            self._population.append(item)

        # If no population provided, create synthetic from scope totals
        if not self._population:
            synthetic = [
                ("Scope 1 aggregate", StratumScope.SCOPE_1, input_data.scope1_tco2e),
                ("Scope 2 aggregate", StratumScope.SCOPE_2, input_data.scope2_tco2e),
                ("Scope 3 aggregate", StratumScope.SCOPE_3, input_data.scope3_tco2e),
            ]
            for desc, scope, val in synthetic:
                if val > 0:
                    emissions = Decimal(str(val))
                    pct = (emissions / total_emissions * Decimal("100")).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP,
                    )
                    self._population.append(PopulationItem(
                        description=desc, scope=scope,
                        emissions_tco2e=str(emissions), pct_of_total=str(pct),
                        risk_level=RiskLevel.MEDIUM,
                        provenance_hash=_compute_hash({"desc": desc, "val": str(emissions)}),
                    ))
            warnings.append("No population items provided; created synthetic from scope totals")

        outputs["population_size"] = len(self._population)
        outputs["total_emissions_covered"] = str(
            sum(Decimal(p.emissions_tco2e) for p in self._population),
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 PopulationIdentification: %d items", len(self._population),
        )
        return PhaseResult(
            phase_name="population_identification", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Stratification
    # -------------------------------------------------------------------------

    async def _phase_3_stratification(
        self, input_data: MaterialitySamplingInput,
    ) -> PhaseResult:
        """Stratify population by scope, risk, and materiality contribution."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Create strata by (scope, risk_level) combination
        stratum_map: Dict[str, List[PopulationItem]] = {}
        for item in self._population:
            key = f"{item.scope.value}|{item.risk_level.value}"
            if key not in stratum_map:
                stratum_map[key] = []
            stratum_map[key].append(item)

        total_emissions = Decimal(str(max(input_data.total_emissions_tco2e, 0.001)))
        self._strata = []

        for key, items in stratum_map.items():
            scope_str, risk_str = key.split("|")
            try:
                scope = StratumScope(scope_str)
            except ValueError:
                scope = StratumScope.SCOPE_1
            try:
                risk = RiskLevel(risk_str)
            except ValueError:
                risk = RiskLevel.MEDIUM

            stratum_emissions = sum(Decimal(i.emissions_tco2e) for i in items)
            stratum_pct = (
                stratum_emissions / total_emissions * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Mark as key item if > performance materiality
            is_key = False
            if self._materiality:
                perf_mat = Decimal(self._materiality.performance_materiality_tco2e)
                if stratum_emissions >= perf_mat:
                    is_key = True

            stratum = Stratum(
                name=f"{scope.value}_{risk.value}",
                scope=scope,
                risk_level=risk,
                population_count=len(items),
                total_emissions_tco2e=str(stratum_emissions),
                pct_of_total=str(stratum_pct),
                is_key_item=is_key,
            )

            # Assign items to stratum
            for item in items:
                item.stratum_id = stratum.stratum_id

            stratum_hash = {
                "name": stratum.name, "count": len(items),
                "emissions": str(stratum_emissions),
            }
            stratum.provenance_hash = _compute_hash(stratum_hash)
            self._strata.append(stratum)

        # Sort strata by emissions descending
        self._strata.sort(
            key=lambda s: Decimal(s.total_emissions_tco2e), reverse=True,
        )

        outputs["strata_count"] = len(self._strata)
        outputs["key_item_strata"] = sum(1 for s in self._strata if s.is_key_item)
        outputs["strata_summary"] = {
            s.name: {"count": s.population_count, "pct": s.pct_of_total}
            for s in self._strata
        }

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 Stratification: %d strata, %d key items",
            len(self._strata), outputs["key_item_strata"],
        )
        return PhaseResult(
            phase_name="stratification", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Sample Sizing
    # -------------------------------------------------------------------------

    async def _phase_4_sample_sizing(
        self, input_data: MaterialitySamplingInput,
    ) -> PhaseResult:
        """Calculate sample sizes per stratum."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        level_key = input_data.assurance_level.value
        conf = CONFIDENCE_LEVELS.get(level_key, CONFIDENCE_LEVELS["limited"])
        z = conf["z_score"]

        total_sample = 0
        for stratum in self._strata:
            n = stratum.population_count

            if n == 0:
                stratum.sample_size = 0
                continue

            # Key items: test all
            if stratum.is_key_item:
                stratum.sample_size = n
                stratum.selection_method = SelectionMethod.ALL_ITEMS
                total_sample += n
                continue

            # Statistical sample size: n0 = z^2 * p * (1-p) / e^2
            # Finite population correction: n = n0 / (1 + (n0-1)/N)
            expected_error = EXPECTED_ERROR_RATES.get(
                stratum.risk_level.value, Decimal("3.0"),
            ) / Decimal("100")
            tolerable_error = Decimal("0.05")  # 5% tolerable error

            if tolerable_error <= Decimal("0"):
                stratum.sample_size = n
                total_sample += n
                continue

            p = expected_error
            q = Decimal("1") - p
            e = tolerable_error

            n0 = (z * z * p * q) / (e * e)
            n0 = n0.quantize(Decimal("1"), rounding=ROUND_HALF_UP)

            # Finite population correction
            n_dec = Decimal(str(n))
            if n_dec > Decimal("0"):
                sample = (
                    n0 / (Decimal("1") + (n0 - Decimal("1")) / n_dec)
                ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            else:
                sample = n0

            sample = min(int(sample), n)
            sample = max(sample, 1)

            stratum.sample_size = sample
            stratum.selection_method = (
                SelectionMethod.MONETARY_UNIT if stratum.risk_level == RiskLevel.HIGH
                else SelectionMethod.RANDOM
            )
            total_sample += sample

        outputs["total_sample_size"] = total_sample
        outputs["total_population"] = sum(s.population_count for s in self._strata)
        outputs["sample_coverage_pct"] = str(
            (
                Decimal(str(total_sample))
                / Decimal(str(max(sum(s.population_count for s in self._strata), 1)))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 SampleSizing: total_sample=%d", total_sample,
        )
        return PhaseResult(
            phase_name="sample_sizing", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Selection Plan
    # -------------------------------------------------------------------------

    async def _phase_5_selection_plan(
        self, input_data: MaterialitySamplingInput,
    ) -> PhaseResult:
        """Produce final selection plan with documentation."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_pop = sum(s.population_count for s in self._strata)
        total_sample = sum(s.sample_size for s in self._strata)
        coverage = Decimal("0.00")
        if total_pop > 0:
            coverage = (
                Decimal(str(total_sample)) / Decimal(str(total_pop)) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        level_key = input_data.assurance_level.value
        conf = CONFIDENCE_LEVELS.get(level_key, CONFIDENCE_LEVELS["limited"])

        methodology = (
            f"Sampling methodology for {input_data.assurance_level.value} assurance. "
            f"Confidence level: {conf['confidence_pct']}%. "
            f"Population stratified into {len(self._strata)} strata by scope and risk. "
            f"Key items (exceeding performance materiality) tested 100%. "
            f"Non-key strata sampled using statistical formula with finite "
            f"population correction. "
            f"Overall materiality: {self._materiality.overall_materiality_tco2e} tCO2e. "
            f"Performance materiality: {self._materiality.performance_materiality_tco2e} tCO2e."
            if self._materiality else "Materiality not calculated."
        )

        plan_data = {
            "pop": total_pop, "sample": total_sample,
            "coverage": str(coverage), "strata": len(self._strata),
        }
        self._plan = SamplingPlan(
            assurance_level=input_data.assurance_level,
            confidence_level_pct=conf["confidence_pct"],
            total_population=total_pop,
            total_sample_size=total_sample,
            coverage_pct=str(coverage),
            strata_count=len(self._strata),
            methodology_notes=methodology,
            provenance_hash=_compute_hash(plan_data),
        )

        outputs["plan_id"] = self._plan.plan_id
        outputs["total_population"] = total_pop
        outputs["total_sample_size"] = total_sample
        outputs["coverage_pct"] = str(coverage)
        outputs["strata_count"] = len(self._strata)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 5 SelectionPlan: pop=%d sample=%d coverage=%s%%",
            total_pop, total_sample, str(coverage),
        )
        return PhaseResult(
            phase_name="selection_plan", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: MaterialitySamplingInput,
        phase_number: int,
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
        self._materiality = None
        self._population = []
        self._strata = []
        self._plan = None

    def _compute_provenance(self, result: MaterialitySamplingResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.overall_materiality_tco2e}|{result.total_sample_size}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
