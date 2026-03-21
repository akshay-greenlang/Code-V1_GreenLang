# -*- coding: utf-8 -*-
"""
FLAG Assessment Workflow
============================

4-phase workflow for Forest, Land and Agriculture (FLAG) target
assessment within PACK-023 SBTi Alignment Pack.  The workflow
identifies FLAG commodities and calculates their emissions
contribution, evaluates the 20% FLAG trigger threshold, calculates
the FLAG reduction pathway at 3.03%/yr, and validates
no-deforestation commitments per SBTi FLAG Guidance V1.1.

Phases:
    1. CommodityAssess     -- Identify FLAG commodities and assess emissions
    2. TriggerEval         -- Evaluate 20% FLAG trigger threshold
    3. PathwayCalc         -- Calculate FLAG reduction pathway (3.03%/yr)
    4. CommitmentValidate  -- Validate no-deforestation commitments

Regulatory references:
    - SBTi FLAG Guidance V1.1 (2022)
    - SBTi Corporate Manual V5.3 (2024)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - IPCC AR6 WG3 (2022) - AFOLU emission factors
    - GHG Protocol Agricultural Guidance (2014)
    - GHG Protocol Land Sector and Removals Guidance (2022)
    - Accountability Framework initiative (AFi) - No-deforestation
    - CDP Forests Questionnaire (2024) - Commodity disclosure

Zero-hallucination: FLAG rate (3.03%/yr) from SBTi FLAG Guidance
V1.1 Table 5.1.  20% trigger from SBTi FLAG Guidance V1.1 Section 3.
All reductions computed deterministically.  No LLM calls in the
numeric computation path.

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"


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


class FLAGTriggerStatus(str, Enum):
    """FLAG trigger evaluation result."""

    TRIGGERED = "triggered"          # >= 20% FLAG emissions
    NOT_TRIGGERED = "not_triggered"  # < 20% FLAG emissions
    BORDERLINE = "borderline"        # 15-20% (recommended but not required)
    INSUFFICIENT_DATA = "insufficient_data"


class CommitmentStatus(str, Enum):
    """No-deforestation commitment validation status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_ASSESSED = "not_assessed"


class CommodityRiskLevel(str, Enum):
    """Deforestation risk level for FLAG commodities."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination Lookups)
# =============================================================================

# FLAG commodity categories (11 per SBTi FLAG Guidance V1.1)
FLAG_COMMODITIES: List[str] = [
    "cattle", "soy", "palm_oil", "timber", "cocoa",
    "coffee", "rubber", "rice", "sugarcane", "maize", "wheat",
]

# Commodity display names
COMMODITY_DISPLAY_NAMES: Dict[str, str] = {
    "cattle": "Cattle (Beef & Dairy)",
    "soy": "Soy",
    "palm_oil": "Palm Oil",
    "timber": "Timber & Wood Products",
    "cocoa": "Cocoa",
    "coffee": "Coffee",
    "rubber": "Rubber",
    "rice": "Rice",
    "sugarcane": "Sugarcane",
    "maize": "Maize / Corn",
    "wheat": "Wheat",
}

# Deforestation risk levels by commodity (AFi / CDP Forests)
COMMODITY_DEFORESTATION_RISK: Dict[str, str] = {
    "cattle": "high",
    "soy": "high",
    "palm_oil": "high",
    "timber": "high",
    "cocoa": "high",
    "coffee": "medium",
    "rubber": "medium",
    "rice": "low",
    "sugarcane": "medium",
    "maize": "low",
    "wheat": "low",
}

# Emission source breakdown categories per commodity
EMISSION_CATEGORIES: List[str] = [
    "land_use_change",
    "agricultural_process",
    "input_production",
    "on_farm_energy",
    "post_harvest",
]

# IPCC AR6 average emission factors (tCO2e per tonne of commodity)
# These are indicative global averages for estimation purposes
COMMODITY_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "cattle": {
        "land_use_change": 12.5,
        "agricultural_process": 22.3,
        "input_production": 2.8,
        "on_farm_energy": 0.9,
        "post_harvest": 0.5,
    },
    "soy": {
        "land_use_change": 3.2,
        "agricultural_process": 0.8,
        "input_production": 0.3,
        "on_farm_energy": 0.1,
        "post_harvest": 0.1,
    },
    "palm_oil": {
        "land_use_change": 8.5,
        "agricultural_process": 2.1,
        "input_production": 0.4,
        "on_farm_energy": 0.2,
        "post_harvest": 0.3,
    },
    "timber": {
        "land_use_change": 5.0,
        "agricultural_process": 0.1,
        "input_production": 0.2,
        "on_farm_energy": 0.3,
        "post_harvest": 0.4,
    },
    "cocoa": {
        "land_use_change": 4.2,
        "agricultural_process": 0.5,
        "input_production": 0.2,
        "on_farm_energy": 0.1,
        "post_harvest": 0.2,
    },
    "coffee": {
        "land_use_change": 3.1,
        "agricultural_process": 0.4,
        "input_production": 0.3,
        "on_farm_energy": 0.2,
        "post_harvest": 0.3,
    },
    "rubber": {
        "land_use_change": 3.5,
        "agricultural_process": 0.3,
        "input_production": 0.2,
        "on_farm_energy": 0.1,
        "post_harvest": 0.2,
    },
    "rice": {
        "land_use_change": 0.5,
        "agricultural_process": 3.8,
        "input_production": 0.4,
        "on_farm_energy": 0.2,
        "post_harvest": 0.1,
    },
    "sugarcane": {
        "land_use_change": 1.8,
        "agricultural_process": 0.6,
        "input_production": 0.3,
        "on_farm_energy": 0.2,
        "post_harvest": 0.3,
    },
    "maize": {
        "land_use_change": 0.8,
        "agricultural_process": 1.2,
        "input_production": 0.5,
        "on_farm_energy": 0.2,
        "post_harvest": 0.1,
    },
    "wheat": {
        "land_use_change": 0.3,
        "agricultural_process": 0.9,
        "input_production": 0.4,
        "on_farm_energy": 0.2,
        "post_harvest": 0.1,
    },
}

# FLAG pathway constants
FLAG_ANNUAL_REDUCTION_RATE = 0.0303   # 3.03% per year (SBTi FLAG V1.1 Table 5.1)
FLAG_TRIGGER_THRESHOLD_PCT = 20.0     # 20% trigger (SBTi FLAG V1.1 Section 3)
FLAG_BORDERLINE_LOWER_PCT = 15.0      # Below trigger but recommended
NO_DEFORESTATION_DEADLINE_YEAR = 2025 # AFi/SBTi deadline for zero deforestation


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


class CommodityInput(BaseModel):
    """Input data for a single FLAG commodity."""

    commodity: str = Field(..., description="FLAG commodity identifier")
    production_tonnes: float = Field(default=0.0, ge=0.0,
                                     description="Annual production or procurement in tonnes")
    land_use_change_tco2e: float = Field(default=0.0, ge=0.0,
                                         description="LUC emissions in tCO2e")
    agricultural_process_tco2e: float = Field(default=0.0, ge=0.0,
                                              description="Agricultural process emissions")
    input_production_tco2e: float = Field(default=0.0, ge=0.0,
                                          description="Input production emissions")
    on_farm_energy_tco2e: float = Field(default=0.0, ge=0.0,
                                        description="On-farm energy emissions")
    post_harvest_tco2e: float = Field(default=0.0, ge=0.0,
                                      description="Post-harvest emissions")
    has_traceability: bool = Field(default=False,
                                   description="Whether supply chain traceability exists")
    supplier_count: int = Field(default=0, ge=0)
    sourcing_countries: List[str] = Field(default_factory=list)
    certified_pct: float = Field(default=0.0, ge=0.0, le=100.0,
                                  description="Percentage with sustainability certification")

    @field_validator("commodity")
    @classmethod
    def _validate_commodity(cls, v: str) -> str:
        normalized = v.lower().strip()
        if normalized not in FLAG_COMMODITIES:
            raise ValueError(
                f"Invalid FLAG commodity '{v}'. "
                f"Valid: {', '.join(FLAG_COMMODITIES)}"
            )
        return normalized

    @property
    def total_emissions_tco2e(self) -> float:
        """Total emissions across all categories for this commodity."""
        return (
            self.land_use_change_tco2e
            + self.agricultural_process_tco2e
            + self.input_production_tco2e
            + self.on_farm_energy_tco2e
            + self.post_harvest_tco2e
        )


class CommodityAssessment(BaseModel):
    """Assessment result for a single FLAG commodity."""

    commodity: str = Field(default="")
    commodity_name: str = Field(default="")
    total_emissions_tco2e: float = Field(default=0.0)
    land_use_change_tco2e: float = Field(default=0.0)
    agricultural_process_tco2e: float = Field(default=0.0)
    input_production_tco2e: float = Field(default=0.0)
    on_farm_energy_tco2e: float = Field(default=0.0)
    post_harvest_tco2e: float = Field(default=0.0)
    pct_of_flag_total: float = Field(default=0.0)
    pct_of_total_emissions: float = Field(default=0.0)
    deforestation_risk: CommodityRiskLevel = Field(default=CommodityRiskLevel.LOW)
    luc_fraction_pct: float = Field(default=0.0, description="% of commodity emissions from LUC")
    has_traceability: bool = Field(default=False)
    certified_pct: float = Field(default=0.0)


class TriggerEvaluation(BaseModel):
    """FLAG trigger evaluation result."""

    trigger_status: FLAGTriggerStatus = Field(default=FLAGTriggerStatus.INSUFFICIENT_DATA)
    flag_total_tco2e: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    flag_pct_of_total: float = Field(default=0.0)
    trigger_threshold_pct: float = Field(default=FLAG_TRIGGER_THRESHOLD_PCT)
    flag_target_required: bool = Field(default=False)
    flag_target_recommended: bool = Field(default=False)
    commodities_contributing: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class PathwayMilestone(BaseModel):
    """Annual milestone on the FLAG reduction pathway."""

    year: int = Field(...)
    target_emissions_tco2e: float = Field(default=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    luc_target_tco2e: float = Field(default=0.0, description="LUC reduction target")
    non_luc_target_tco2e: float = Field(default=0.0, description="Non-LUC reduction target")


class FLAGPathway(BaseModel):
    """Complete FLAG reduction pathway."""

    base_year: int = Field(default=2022)
    target_year: int = Field(default=2030)
    base_emissions_tco2e: float = Field(default=0.0)
    target_emissions_tco2e: float = Field(default=0.0)
    total_reduction_pct: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=FLAG_ANNUAL_REDUCTION_RATE * 100)
    milestones: List[PathwayMilestone] = Field(default_factory=list)
    zero_emissions_year: Optional[int] = Field(None,
                                                description="Year when FLAG emissions reach zero")
    luc_base_tco2e: float = Field(default=0.0)
    non_luc_base_tco2e: float = Field(default=0.0)


class DeforestationCommitment(BaseModel):
    """No-deforestation commitment assessment."""

    has_commitment: bool = Field(default=False)
    commitment_year: Optional[int] = Field(None, description="Year commitment was made")
    target_year: int = Field(default=NO_DEFORESTATION_DEADLINE_YEAR)
    covers_all_commodities: bool = Field(default=False)
    covered_commodities: List[str] = Field(default_factory=list)
    uncovered_commodities: List[str] = Field(default_factory=list)
    includes_conversion: bool = Field(default=False,
                                      description="Covers conversion of natural ecosystems")
    aligned_with_afi: bool = Field(default=False,
                                    description="Aligned with AFi definitions")
    has_monitoring: bool = Field(default=False)
    has_grievance_mechanism: bool = Field(default=False)


class CommitmentValidation(BaseModel):
    """Complete no-deforestation commitment validation result."""

    status: CommitmentStatus = Field(default=CommitmentStatus.NOT_ASSESSED)
    commitment: DeforestationCommitment = Field(default_factory=DeforestationCommitment)
    compliance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    blocking_submission: bool = Field(default=False)


class FLAGWorkflowConfig(BaseModel):
    """Configuration for the FLAG assessment workflow."""

    # Emissions context
    scope1_tco2e: float = Field(default=0.0, ge=0.0, description="Total Scope 1 emissions")
    scope2_tco2e: float = Field(default=0.0, ge=0.0, description="Total Scope 2 emissions")
    scope3_tco2e: float = Field(default=0.0, ge=0.0, description="Total Scope 3 emissions")

    # FLAG commodity data
    commodities: List[CommodityInput] = Field(default_factory=list)
    base_year: int = Field(default=2022, ge=2015, le=2050)
    target_year: int = Field(default=2030, ge=2025, le=2060)

    # No-deforestation commitment data
    has_no_deforestation_commitment: bool = Field(default=False)
    commitment_year: Optional[int] = Field(None)
    commitment_covers_all_commodities: bool = Field(default=False)
    commitment_covered_commodities: List[str] = Field(default_factory=list)
    includes_natural_ecosystem_conversion: bool = Field(default=False)
    aligned_with_afi: bool = Field(default=False)
    has_monitoring_system: bool = Field(default=False)
    has_grievance_mechanism: bool = Field(default=False)

    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class FLAGWorkflowResult(BaseModel):
    """Complete result from the FLAG assessment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="flag_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    commodity_assessments: List[CommodityAssessment] = Field(default_factory=list)
    trigger_evaluation: Optional[TriggerEvaluation] = Field(None)
    pathway: Optional[FLAGPathway] = Field(None)
    commitment_validation: Optional[CommitmentValidation] = Field(None)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FLAGWorkflow:
    """
    4-phase FLAG assessment workflow for SBTi alignment.

    Identifies FLAG commodities and assesses their emissions, evaluates
    the 20% trigger threshold, calculates the FLAG reduction pathway
    at 3.03%/yr, and validates no-deforestation commitments per SBTi
    FLAG Guidance V1.1.

    Zero-hallucination: FLAG rate (3.03%/yr) from SBTi FLAG Guidance
    V1.1 Table 5.1.  20% trigger from Section 3.  No LLM calls in
    the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = FLAGWorkflow()
        >>> config = FLAGWorkflowConfig(
        ...     scope1_tco2e=5000, scope2_tco2e=3000, scope3_tco2e=10000,
        ...     commodities=[CommodityInput(commodity="cattle",
        ...                                  land_use_change_tco2e=2000)],
        ... )
        >>> result = await wf.execute(config)
        >>> assert result.trigger_evaluation is not None
    """

    def __init__(self) -> None:
        """Initialise FLAGWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._commodity_assessments: List[CommodityAssessment] = []
        self._trigger: Optional[TriggerEvaluation] = None
        self._pathway: Optional[FLAGPathway] = None
        self._commitment: Optional[CommitmentValidation] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: FLAGWorkflowConfig) -> FLAGWorkflowResult:
        """
        Execute the 4-phase FLAG assessment workflow.

        Args:
            config: FLAG workflow configuration with commodity data,
                emissions context, and commitment information.

        Returns:
            FLAGWorkflowResult with commodity assessments, trigger
            evaluation, reduction pathway, and commitment validation.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting FLAG workflow %s, commodities=%d",
            self.workflow_id, len(config.commodities),
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Commodity Assessment
            phase1 = await self._phase_commodity_assess(config)
            self._phase_results.append(phase1)

            # Phase 2: Trigger Evaluation
            phase2 = await self._phase_trigger_eval(config)
            self._phase_results.append(phase2)

            # Phase 3: Pathway Calculation
            phase3 = await self._phase_pathway_calc(config)
            self._phase_results.append(phase3)

            # Phase 4: Commitment Validation
            phase4 = await self._phase_commitment_validate(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("FLAG workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = FLAGWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            commodity_assessments=self._commodity_assessments,
            trigger_evaluation=self._trigger,
            pathway=self._pathway,
            commitment_validation=self._commitment,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "FLAG workflow %s completed in %.2fs, triggered=%s",
            self.workflow_id, elapsed,
            self._trigger.flag_target_required if self._trigger else "unknown",
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Commodity Assessment
    # -------------------------------------------------------------------------

    async def _phase_commodity_assess(self, config: FLAGWorkflowConfig) -> PhaseResult:
        """Identify FLAG commodities and assess emissions contribution."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._commodity_assessments = []

        total_company = config.scope1_tco2e + config.scope2_tco2e + config.scope3_tco2e
        flag_total = 0.0

        # Assess each provided commodity
        for ci in config.commodities:
            commodity_emissions = ci.total_emissions_tco2e

            # If no emissions breakdown provided, estimate from production volume
            if commodity_emissions <= 0 and ci.production_tonnes > 0:
                ef = COMMODITY_EMISSION_FACTORS.get(ci.commodity, {})
                commodity_emissions = sum(ef.values()) * ci.production_tonnes
                warnings.append(
                    f"Commodity '{ci.commodity}': emissions estimated from production "
                    f"volume ({ci.production_tonnes} t) using IPCC AR6 average factors"
                )

            flag_total += commodity_emissions

            # LUC fraction
            luc_fraction = 0.0
            if commodity_emissions > 0:
                luc_fraction = (ci.land_use_change_tco2e / commodity_emissions) * 100.0

            # Deforestation risk
            risk_str = COMMODITY_DEFORESTATION_RISK.get(ci.commodity, "low")
            risk_level = CommodityRiskLevel(risk_str)

            self._commodity_assessments.append(CommodityAssessment(
                commodity=ci.commodity,
                commodity_name=COMMODITY_DISPLAY_NAMES.get(ci.commodity, ci.commodity),
                total_emissions_tco2e=round(commodity_emissions, 2),
                land_use_change_tco2e=round(ci.land_use_change_tco2e, 2),
                agricultural_process_tco2e=round(ci.agricultural_process_tco2e, 2),
                input_production_tco2e=round(ci.input_production_tco2e, 2),
                on_farm_energy_tco2e=round(ci.on_farm_energy_tco2e, 2),
                post_harvest_tco2e=round(ci.post_harvest_tco2e, 2),
                pct_of_flag_total=0.0,  # Updated after totals
                pct_of_total_emissions=round(
                    (commodity_emissions / total_company * 100.0)
                    if total_company > 0 else 0.0, 2
                ),
                deforestation_risk=risk_level,
                luc_fraction_pct=round(luc_fraction, 2),
                has_traceability=ci.has_traceability,
                certified_pct=ci.certified_pct,
            ))

        # Update pct_of_flag_total
        for ca in self._commodity_assessments:
            if flag_total > 0:
                ca.pct_of_flag_total = round(
                    ca.total_emissions_tco2e / flag_total * 100.0, 2
                )

        # Sort by emissions descending
        self._commodity_assessments.sort(
            key=lambda c: c.total_emissions_tco2e, reverse=True
        )

        # Identify missing high-risk commodities
        provided_commodities = {ci.commodity for ci in config.commodities}
        high_risk_missing = [
            c for c in FLAG_COMMODITIES
            if COMMODITY_DEFORESTATION_RISK.get(c) == "high" and c not in provided_commodities
        ]
        if high_risk_missing:
            warnings.append(
                f"High deforestation-risk commodities not assessed: "
                f"{', '.join(high_risk_missing)}. Confirm these are not in your value chain."
            )

        outputs["commodities_assessed"] = len(self._commodity_assessments)
        outputs["flag_total_tco2e"] = round(flag_total, 2)
        outputs["total_company_tco2e"] = round(total_company, 2)
        outputs["flag_pct_of_total"] = round(
            (flag_total / total_company * 100.0) if total_company > 0 else 0.0, 2
        )
        outputs["high_risk_commodities"] = sum(
            1 for ca in self._commodity_assessments
            if ca.deforestation_risk == CommodityRiskLevel.HIGH
        )
        outputs["total_luc_tco2e"] = round(
            sum(ca.land_use_change_tco2e for ca in self._commodity_assessments), 2
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Commodity assess: %d commodities, FLAG=%.2f tCO2e (%.1f%% of total)",
            len(self._commodity_assessments), flag_total,
            (flag_total / total_company * 100.0) if total_company > 0 else 0.0,
        )
        return PhaseResult(
            phase_name="commodity_assess",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Trigger Evaluation
    # -------------------------------------------------------------------------

    async def _phase_trigger_eval(self, config: FLAGWorkflowConfig) -> PhaseResult:
        """Evaluate 20% FLAG trigger threshold."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_company = config.scope1_tco2e + config.scope2_tco2e + config.scope3_tco2e
        flag_total = sum(ca.total_emissions_tco2e for ca in self._commodity_assessments)
        flag_pct = (flag_total / total_company * 100.0) if total_company > 0 else 0.0

        # Determine trigger status
        if total_company <= 0:
            trigger_status = FLAGTriggerStatus.INSUFFICIENT_DATA
            flag_required = False
            flag_recommended = False
            warnings.append("Total company emissions are zero; cannot evaluate FLAG trigger")
        elif flag_pct >= FLAG_TRIGGER_THRESHOLD_PCT:
            trigger_status = FLAGTriggerStatus.TRIGGERED
            flag_required = True
            flag_recommended = True
        elif flag_pct >= FLAG_BORDERLINE_LOWER_PCT:
            trigger_status = FLAGTriggerStatus.BORDERLINE
            flag_required = False
            flag_recommended = True
            warnings.append(
                f"FLAG emissions are {flag_pct:.1f}% of total "
                f"({FLAG_BORDERLINE_LOWER_PCT}%-{FLAG_TRIGGER_THRESHOLD_PCT}% range). "
                "Separate FLAG target not required but recommended."
            )
        else:
            trigger_status = FLAGTriggerStatus.NOT_TRIGGERED
            flag_required = False
            flag_recommended = False

        # Identify contributing commodities
        contributing = [
            ca.commodity for ca in self._commodity_assessments
            if ca.total_emissions_tco2e > 0
        ]

        trigger_notes: List[str] = []
        trigger_notes.append(
            f"FLAG emissions: {flag_total:.2f} tCO2e ({flag_pct:.1f}% of total)"
        )
        trigger_notes.append(
            f"Trigger threshold: {FLAG_TRIGGER_THRESHOLD_PCT}% (SBTi FLAG V1.1 Section 3)"
        )
        if flag_required:
            trigger_notes.append(
                "Separate FLAG target REQUIRED per SBTi FLAG Guidance V1.1"
            )

        self._trigger = TriggerEvaluation(
            trigger_status=trigger_status,
            flag_total_tco2e=round(flag_total, 2),
            total_emissions_tco2e=round(total_company, 2),
            flag_pct_of_total=round(flag_pct, 2),
            trigger_threshold_pct=FLAG_TRIGGER_THRESHOLD_PCT,
            flag_target_required=flag_required,
            flag_target_recommended=flag_recommended,
            commodities_contributing=contributing,
            notes=trigger_notes,
        )

        outputs["trigger_status"] = trigger_status.value
        outputs["flag_total_tco2e"] = round(flag_total, 2)
        outputs["total_emissions_tco2e"] = round(total_company, 2)
        outputs["flag_pct_of_total"] = round(flag_pct, 2)
        outputs["flag_target_required"] = flag_required
        outputs["flag_target_recommended"] = flag_recommended
        outputs["commodities_contributing"] = len(contributing)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Trigger eval: FLAG=%.1f%%, status=%s, required=%s",
            flag_pct, trigger_status.value, flag_required,
        )
        return PhaseResult(
            phase_name="trigger_eval",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Pathway Calculation
    # -------------------------------------------------------------------------

    async def _phase_pathway_calc(self, config: FLAGWorkflowConfig) -> PhaseResult:
        """Calculate FLAG reduction pathway at 3.03%/yr."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        flag_total = sum(ca.total_emissions_tco2e for ca in self._commodity_assessments)
        luc_total = sum(ca.land_use_change_tco2e for ca in self._commodity_assessments)
        non_luc_total = flag_total - luc_total

        if flag_total <= 0:
            warnings.append("FLAG emissions are zero; pathway will be empty")
            self._pathway = FLAGPathway(
                base_year=config.base_year,
                target_year=config.target_year,
            )
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="pathway_calc",
                status=PhaseStatus.COMPLETED,
                duration_seconds=round(elapsed, 4),
                outputs={"flag_total_tco2e": 0.0, "milestones_count": 0},
                warnings=warnings,
                provenance_hash=_compute_hash("{}"),
            )

        # FLAG pathway: linear 3.03%/yr reduction
        # E(t) = E(base) * max(0, 1 - 0.0303 * (t - base_year))
        rate = FLAG_ANNUAL_REDUCTION_RATE
        milestones: List[PathwayMilestone] = []
        zero_year: Optional[int] = None

        for year in range(config.base_year, config.target_year + 1):
            years_elapsed = year - config.base_year
            reduction_factor = max(0.0, 1.0 - rate * years_elapsed)
            target_emissions = flag_total * reduction_factor

            cumulative_pct = (1.0 - reduction_factor) * 100.0

            # Split LUC and non-LUC targets proportionally
            luc_target = luc_total * reduction_factor
            non_luc_target = non_luc_total * reduction_factor

            milestones.append(PathwayMilestone(
                year=year,
                target_emissions_tco2e=round(target_emissions, 2),
                cumulative_reduction_pct=round(cumulative_pct, 2),
                annual_reduction_rate_pct=round(rate * 100.0, 2),
                luc_target_tco2e=round(luc_target, 2),
                non_luc_target_tco2e=round(non_luc_target, 2),
            ))

            if target_emissions <= 0 and zero_year is None:
                zero_year = year

        # Calculate zero-emissions year (where linear pathway reaches zero)
        if zero_year is None:
            zero_year_calc = config.base_year + int(1.0 / rate) + 1
        else:
            zero_year_calc = zero_year

        # Target year values
        target_years = config.target_year - config.base_year
        target_reduction_factor = max(0.0, 1.0 - rate * target_years)
        target_emissions = flag_total * target_reduction_factor
        total_reduction_pct = (1.0 - target_reduction_factor) * 100.0

        self._pathway = FLAGPathway(
            base_year=config.base_year,
            target_year=config.target_year,
            base_emissions_tco2e=round(flag_total, 2),
            target_emissions_tco2e=round(target_emissions, 2),
            total_reduction_pct=round(total_reduction_pct, 2),
            annual_reduction_rate_pct=round(rate * 100.0, 2),
            milestones=milestones,
            zero_emissions_year=zero_year_calc,
            luc_base_tco2e=round(luc_total, 2),
            non_luc_base_tco2e=round(non_luc_total, 2),
        )

        # Warnings
        if total_reduction_pct < 20.0:
            warnings.append(
                f"FLAG pathway reduction of {total_reduction_pct:.1f}% by {config.target_year} "
                "may be considered insufficient for SBTi near-term target validation"
            )

        if luc_total / flag_total > 0.5 if flag_total > 0 else False:
            warnings.append(
                f"Land-use change emissions constitute {luc_total / flag_total * 100:.1f}% "
                "of FLAG total; no-deforestation commitment is critical"
            )

        outputs["base_emissions_tco2e"] = round(flag_total, 2)
        outputs["target_emissions_tco2e"] = round(target_emissions, 2)
        outputs["total_reduction_pct"] = round(total_reduction_pct, 2)
        outputs["annual_rate_pct"] = round(rate * 100.0, 2)
        outputs["milestones_count"] = len(milestones)
        outputs["zero_emissions_year"] = zero_year_calc
        outputs["luc_base_tco2e"] = round(luc_total, 2)
        outputs["non_luc_base_tco2e"] = round(non_luc_total, 2)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Pathway calc: base=%.2f, target=%.2f, reduction=%.1f%%, zero_year=%d",
            flag_total, target_emissions, total_reduction_pct, zero_year_calc,
        )
        return PhaseResult(
            phase_name="pathway_calc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Commitment Validation
    # -------------------------------------------------------------------------

    async def _phase_commitment_validate(self, config: FLAGWorkflowConfig) -> PhaseResult:
        """Validate no-deforestation commitments per SBTi FLAG Guidance V1.1."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Build commitment data
        assessed_commodities = [ca.commodity for ca in self._commodity_assessments]
        covered = config.commitment_covered_commodities
        uncovered = [c for c in assessed_commodities if c not in covered]

        commitment = DeforestationCommitment(
            has_commitment=config.has_no_deforestation_commitment,
            commitment_year=config.commitment_year,
            target_year=NO_DEFORESTATION_DEADLINE_YEAR,
            covers_all_commodities=config.commitment_covers_all_commodities,
            covered_commodities=covered,
            uncovered_commodities=uncovered,
            includes_conversion=config.includes_natural_ecosystem_conversion,
            aligned_with_afi=config.aligned_with_afi,
            has_monitoring=config.has_monitoring_system,
            has_grievance_mechanism=config.has_grievance_mechanism,
        )

        # Score compliance (weighted checklist)
        score_items = {
            "has_commitment": (config.has_no_deforestation_commitment, 25.0),
            "covers_all_commodities": (config.commitment_covers_all_commodities, 20.0),
            "includes_conversion": (config.includes_natural_ecosystem_conversion, 15.0),
            "aligned_with_afi": (config.aligned_with_afi, 15.0),
            "has_monitoring": (config.has_monitoring_system, 15.0),
            "has_grievance_mechanism": (config.has_grievance_mechanism, 10.0),
        }

        compliance_score = sum(
            weight for _, (met, weight) in score_items.items() if met
        )

        # Identify gaps
        gaps: List[str] = []
        recommendations: List[str] = []

        if not config.has_no_deforestation_commitment:
            gaps.append("No zero-deforestation commitment declared")
            recommendations.append(
                "Declare a public no-deforestation, no-conversion commitment "
                "covering all FLAG commodities in value chain by 2025"
            )

        if not config.commitment_covers_all_commodities and uncovered:
            gaps.append(
                f"Commitment does not cover all commodities: "
                f"uncovered = {', '.join(uncovered)}"
            )
            recommendations.append(
                f"Extend no-deforestation commitment to cover: {', '.join(uncovered)}"
            )

        if not config.includes_natural_ecosystem_conversion:
            gaps.append(
                "Commitment does not explicitly cover conversion of natural ecosystems "
                "(beyond forests)"
            )
            recommendations.append(
                "Expand commitment to explicitly include zero conversion of all "
                "natural ecosystems (wetlands, peatlands, savannas, grasslands)"
            )

        if not config.aligned_with_afi:
            gaps.append(
                "Commitment is not aligned with Accountability Framework initiative (AFi)"
            )
            recommendations.append(
                "Align commitment language and scope with AFi definitions and guidance"
            )

        if not config.has_monitoring_system:
            gaps.append("No monitoring/verification system for deforestation-free supply chain")
            recommendations.append(
                "Implement supply chain monitoring using satellite data, "
                "certification systems, and/or supplier audits"
            )

        if not config.has_grievance_mechanism:
            gaps.append("No grievance mechanism for reporting deforestation incidents")
            recommendations.append(
                "Establish a public grievance mechanism for stakeholders to report "
                "suspected deforestation or conversion incidents"
            )

        # Commitment year check
        if config.commitment_year and config.commitment_year > NO_DEFORESTATION_DEADLINE_YEAR:
            gaps.append(
                f"Commitment year ({config.commitment_year}) is after "
                f"SBTi deadline ({NO_DEFORESTATION_DEADLINE_YEAR})"
            )

        # Determine status
        blocking = False
        if compliance_score >= 80.0:
            status = CommitmentStatus.COMPLIANT
        elif compliance_score >= 40.0:
            status = CommitmentStatus.PARTIAL
            warnings.append(
                f"Partial compliance ({compliance_score:.0f}%): "
                f"{len(gaps)} gaps identified"
            )
        elif config.has_no_deforestation_commitment:
            status = CommitmentStatus.PARTIAL
            warnings.append(
                f"Weak compliance ({compliance_score:.0f}%): significant gaps remain"
            )
        else:
            status = CommitmentStatus.NON_COMPLIANT
            # Only blocking if FLAG target is required
            if self._trigger and self._trigger.flag_target_required:
                blocking = True
                warnings.append(
                    "No-deforestation commitment is REQUIRED for FLAG target submission "
                    "and is currently missing"
                )

        self._commitment = CommitmentValidation(
            status=status,
            commitment=commitment,
            compliance_score=round(compliance_score, 2),
            gaps=gaps,
            recommendations=recommendations,
            blocking_submission=blocking,
        )

        outputs["commitment_status"] = status.value
        outputs["compliance_score"] = round(compliance_score, 2)
        outputs["gaps_count"] = len(gaps)
        outputs["has_commitment"] = config.has_no_deforestation_commitment
        outputs["covers_all_commodities"] = config.commitment_covers_all_commodities
        outputs["includes_conversion"] = config.includes_natural_ecosystem_conversion
        outputs["aligned_with_afi"] = config.aligned_with_afi
        outputs["has_monitoring"] = config.has_monitoring_system
        outputs["blocking_submission"] = blocking
        outputs["uncovered_commodities"] = uncovered

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Commitment validate: status=%s, score=%.0f%%, gaps=%d, blocking=%s",
            status.value, compliance_score, len(gaps), blocking,
        )
        return PhaseResult(
            phase_name="commitment_validate",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
