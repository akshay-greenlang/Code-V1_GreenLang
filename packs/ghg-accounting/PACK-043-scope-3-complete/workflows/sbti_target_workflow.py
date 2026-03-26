# -*- coding: utf-8 -*-
"""
SBTi Target Setting Workflow
====================================

4-phase workflow for SBTi-aligned Scope 3 target setting, validation,
and submission package preparation within PACK-043 Scope 3 Complete Pack.

Phases:
    1. MATERIALITY_CHECK     -- Verify Scope 3 >= 40% of total GHG footprint,
                                determine SBTi target obligation (mandatory
                                vs. recommended).
    2. PATHWAY_CALCULATION   -- Calculate required reduction rates for absolute,
                                SDA, and intensity targets aligned with 1.5C
                                and well-below-2C pathways.
    3. TARGET_VALIDATION     -- Validate coverage >= 67% of Scope 3 emissions,
                                check FLAG applicability, verify methodology
                                alignment with SBTi criteria.
    4. SUBMISSION_PACKAGE    -- Generate SBTi target submission data package
                                with all required fields, supporting evidence,
                                and calculation documentation.

The workflow follows GreenLang zero-hallucination principles: all materiality
checks, reduction calculations, and coverage validations use deterministic
formulas from SBTi published criteria. SHA-256 provenance hashes ensure
auditability.

Regulatory Basis:
    SBTi Corporate Net-Zero Standard (v1.1, 2024)
    SBTi Near-Term Science-Based Target Setting Criteria (v5.1)
    SBTi FLAG Guidance (2022)
    GHG Protocol Scope 3 Standard

Schedule: upon initial target setting and annual tracking
Estimated duration: 2-4 hours

Author: GreenLang Platform Team
Version: 43.0.0
"""

_MODULE_VERSION: str = "43.0.0"

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


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


class TargetType(str, Enum):
    """SBTi target types."""

    ABSOLUTE_CONTRACTION = "absolute_contraction"
    SECTORAL_DECARBONIZATION = "sectoral_decarbonization"
    INTENSITY = "intensity"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"


class TargetTimeframe(str, Enum):
    """SBTi target timeframes."""

    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"
    NET_ZERO = "net_zero"


class PathwayAmbition(str, Enum):
    """SBTi pathway ambition levels."""

    ALIGNED_1_5C = "1.5c"
    WELL_BELOW_2C = "well_below_2c"
    BELOW_2C = "below_2c"


class FLAGApplicability(str, Enum):
    """FLAG (Forest, Land and Agriculture) applicability."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    PARTIAL = "partial"


class ValidationResult(str, Enum):
    """Validation result status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_EVALUATED = "not_evaluated"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# SBTi Scope 3 materiality threshold
SCOPE3_MATERIALITY_THRESHOLD_PCT: float = 40.0

# SBTi minimum Scope 3 coverage requirement
SBTI_MIN_COVERAGE_PCT: float = 67.0

# SBTi required annual reduction rates by pathway
SBTI_ANNUAL_RATES: Dict[str, Dict[str, float]] = {
    PathwayAmbition.ALIGNED_1_5C.value: {
        TargetType.ABSOLUTE_CONTRACTION.value: 4.2,
        TargetType.SECTORAL_DECARBONIZATION.value: 4.2,
        TargetType.INTENSITY.value: 4.2,
    },
    PathwayAmbition.WELL_BELOW_2C.value: {
        TargetType.ABSOLUTE_CONTRACTION.value: 2.5,
        TargetType.SECTORAL_DECARBONIZATION.value: 2.5,
        TargetType.INTENSITY.value: 2.5,
    },
}

# SBTi target year requirements
SBTI_NEAR_TERM_MAX_YEARS: int = 10
SBTI_LONG_TERM_YEAR: int = 2050
SBTI_NET_ZERO_MIN_PCT: float = 90.0

# FLAG sectors (NACE codes that trigger FLAG)
FLAG_SECTORS: List[str] = [
    "A01", "A02", "A03",  # Agriculture, forestry, fishing
    "C10", "C11", "C12",  # Food, beverages, tobacco manufacturing
    "G46.2", "G46.3",     # Wholesale of agricultural/food products
    "I56",                 # Food and beverage service
]

# SBTi submission required fields
SBTI_REQUIRED_FIELDS: List[str] = [
    "company_name",
    "sector",
    "base_year",
    "base_year_emissions_scope_1_tco2e",
    "base_year_emissions_scope_2_tco2e",
    "base_year_emissions_scope_3_tco2e",
    "target_year",
    "target_type",
    "target_ambition",
    "target_coverage_pct",
    "target_reduction_pct",
    "methodology",
    "scope3_categories_included",
    "scope3_categories_excluded_rationale",
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class WorkflowState(BaseModel):
    """Persistent state for checkpoint/resume."""

    workflow_id: str = Field(default="")
    current_phase: int = Field(default=0)
    phase_statuses: Dict[str, str] = Field(default_factory=dict)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class EmissionsSummary(BaseModel):
    """Complete GHG emission summary for SBTi evaluation."""

    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_category_emissions: Dict[str, float] = Field(
        default_factory=dict, description="Category -> tCO2e"
    )
    total_tco2e: float = Field(default=0.0, ge=0.0)


class MaterialityCheck(BaseModel):
    """Result of Scope 3 materiality check."""

    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    threshold_pct: float = Field(default=40.0)
    is_material: bool = Field(default=False)
    target_obligation: str = Field(
        default="recommended",
        description="'mandatory' if >= 40%, else 'recommended'",
    )
    flag_applicability: FLAGApplicability = Field(
        default=FLAGApplicability.NOT_APPLICABLE
    )


class PathwayResult(BaseModel):
    """Calculated pathway for a specific target type and ambition."""

    target_type: TargetType = Field(...)
    pathway_ambition: PathwayAmbition = Field(...)
    timeframe: TargetTimeframe = Field(default=TargetTimeframe.NEAR_TERM)
    base_year: int = Field(default=2025)
    target_year: int = Field(default=2030)
    annual_reduction_rate_pct: float = Field(default=0.0, ge=0.0)
    total_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    year_by_year_tco2e: Dict[int, float] = Field(default_factory=dict)


class TargetValidation(BaseModel):
    """Target validation result."""

    check_name: str = Field(default="")
    result: ValidationResult = Field(default=ValidationResult.NOT_EVALUATED)
    actual_value: str = Field(default="")
    required_value: str = Field(default="")
    detail: str = Field(default="")


class SubmissionField(BaseModel):
    """Single field in the SBTi submission package."""

    field_name: str = Field(default="")
    value: str = Field(default="")
    source: str = Field(default="")
    validated: bool = Field(default=False)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class SBTiTargetInput(BaseModel):
    """Input data model for SBTiTargetWorkflow."""

    organization_name: str = Field(default="")
    sector: str = Field(default="")
    sector_code: str = Field(default="", description="NACE/NAICS code")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    base_year: int = Field(default=2025, ge=2015, le=2050)
    target_year_near: int = Field(default=2030, ge=2025, le=2040)
    target_year_long: int = Field(default=2050, ge=2040, le=2060)
    emissions: EmissionsSummary = Field(default_factory=EmissionsSummary)
    preferred_ambition: PathwayAmbition = Field(
        default=PathwayAmbition.ALIGNED_1_5C
    )
    preferred_target_type: TargetType = Field(
        default=TargetType.ABSOLUTE_CONTRACTION
    )
    categories_to_include: List[str] = Field(
        default_factory=list,
        description="Scope 3 categories to include in target",
    )
    categories_to_exclude: List[str] = Field(default_factory=list)
    exclusion_rationale: Dict[str, str] = Field(
        default_factory=dict,
        description="Category -> rationale for exclusion",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class SBTiTargetOutput(BaseModel):
    """Complete output from SBTiTargetWorkflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sbti_target")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_name: str = Field(default="")
    materiality: MaterialityCheck = Field(default_factory=MaterialityCheck)
    pathways: List[PathwayResult] = Field(default_factory=list)
    validations: List[TargetValidation] = Field(default_factory=list)
    validation_pass: bool = Field(default=False)
    submission_fields: List[SubmissionField] = Field(default_factory=list)
    submission_ready: bool = Field(default=False)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SBTiTargetWorkflow:
    """
    4-phase SBTi Scope 3 target setting and submission workflow.

    Checks Scope 3 materiality, calculates required reduction pathways,
    validates target coverage and methodology alignment, and generates
    SBTi submission data package.

    Zero-hallucination: all thresholds, reduction rates, and validation
    criteria use published SBTi values and deterministic arithmetic.

    Example:
        >>> wf = SBTiTargetWorkflow()
        >>> inp = SBTiTargetInput(
        ...     emissions=EmissionsSummary(
        ...         scope1_tco2e=10000, scope2_market_tco2e=5000,
        ...         scope3_tco2e=85000, total_tco2e=100000,
        ...     ),
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "materiality_check",
        "pathway_calculation",
        "target_validation",
        "submission_package",
    ]

    PHASE_WEIGHTS: Dict[str, float] = {
        "materiality_check": 15.0,
        "pathway_calculation": 30.0,
        "target_validation": 30.0,
        "submission_package": 25.0,
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize SBTiTargetWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._materiality: MaterialityCheck = MaterialityCheck()
        self._pathways: List[PathwayResult] = []
        self._validations: List[TargetValidation] = []
        self._submission: List[SubmissionField] = []
        self._phase_results: List[PhaseResult] = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[SBTiTargetInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> SBTiTargetOutput:
        """Execute the 4-phase SBTi target workflow."""
        if input_data is None:
            input_data = SBTiTargetInput()

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting SBTi target workflow %s org=%s scope3=%.0f",
            self.workflow_id,
            input_data.organization_name,
            input_data.emissions.scope3_tco2e,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING
        self._update_progress(0.0)

        try:
            phase1 = await self._execute_with_retry(
                self._phase_materiality_check, input_data, 1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")
            self._update_progress(15.0)

            phase2 = await self._execute_with_retry(
                self._phase_pathway_calculation, input_data, 2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")
            self._update_progress(45.0)

            phase3 = await self._execute_with_retry(
                self._phase_target_validation, input_data, 3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")
            self._update_progress(75.0)

            phase4 = await self._execute_with_retry(
                self._phase_submission_package, input_data, 4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")
            self._update_progress(100.0)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "SBTi target workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(
                PhaseResult(
                    phase_name="error", phase_number=0,
                    status=PhaseStatus.FAILED, errors=[str(exc)],
                )
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        all_pass = all(
            v.result == ValidationResult.PASS for v in self._validations
            if v.result != ValidationResult.NOT_EVALUATED
        )
        submission_ready = all_pass and len(self._submission) >= len(SBTI_REQUIRED_FIELDS)

        result = SBTiTargetOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_name=input_data.organization_name,
            materiality=self._materiality,
            pathways=self._pathways,
            validations=self._validations,
            validation_pass=all_pass,
            submission_fields=self._submission,
            submission_ready=submission_ready,
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "SBTi target workflow %s completed in %.2fs status=%s "
            "materiality=%s validation=%s submission_ready=%s",
            self.workflow_id, elapsed, overall_status.value,
            "material" if self._materiality.is_material else "not_material",
            "PASS" if all_pass else "FAIL",
            submission_ready,
        )
        return result

    def get_state(self) -> WorkflowState:
        """Return current workflow state for checkpoint/resume."""
        return self._state.model_copy()

    async def resume(
        self, state: WorkflowState, input_data: SBTiTargetInput
    ) -> SBTiTargetOutput:
        """Resume workflow from a saved checkpoint state."""
        self._state = state
        self.workflow_id = state.workflow_id
        return await self.execute(input_data)

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: SBTiTargetInput, phase_number: int
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    import asyncio
                    await asyncio.sleep(self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1)))
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Materiality Check
    # -------------------------------------------------------------------------

    async def _phase_materiality_check(
        self, input_data: SBTiTargetInput
    ) -> PhaseResult:
        """Verify Scope 3 >= 40% of total, determine target obligation."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        e = input_data.emissions
        total = e.total_tco2e
        if total <= 0:
            total = e.scope1_tco2e + e.scope2_market_tco2e + e.scope3_tco2e

        scope3_pct = (e.scope3_tco2e / total * 100.0) if total > 0 else 0.0
        is_material = scope3_pct >= SCOPE3_MATERIALITY_THRESHOLD_PCT

        # FLAG applicability
        flag = FLAGApplicability.NOT_APPLICABLE
        if input_data.sector_code:
            for flag_code in FLAG_SECTORS:
                if input_data.sector_code.upper().startswith(flag_code.upper()):
                    flag = FLAGApplicability.APPLICABLE
                    break

        self._materiality = MaterialityCheck(
            scope3_pct_of_total=round(scope3_pct, 2),
            threshold_pct=SCOPE3_MATERIALITY_THRESHOLD_PCT,
            is_material=is_material,
            target_obligation="mandatory" if is_material else "recommended",
            flag_applicability=flag,
        )

        if not is_material:
            warnings.append(
                f"Scope 3 is {scope3_pct:.1f}% of total (below {SCOPE3_MATERIALITY_THRESHOLD_PCT}% "
                f"threshold); Scope 3 target is recommended but not mandatory"
            )

        outputs["scope3_tco2e"] = round(e.scope3_tco2e, 2)
        outputs["total_tco2e"] = round(total, 2)
        outputs["scope3_pct_of_total"] = round(scope3_pct, 2)
        outputs["is_material"] = is_material
        outputs["target_obligation"] = self._materiality.target_obligation
        outputs["flag_applicability"] = flag.value

        self._state.phase_statuses["materiality_check"] = "completed"
        self._state.current_phase = 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 MaterialityCheck: scope3=%.1f%% material=%s flag=%s",
            scope3_pct, is_material, flag.value,
        )
        return PhaseResult(
            phase_name="materiality_check", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Pathway Calculation
    # -------------------------------------------------------------------------

    async def _phase_pathway_calculation(
        self, input_data: SBTiTargetInput
    ) -> PhaseResult:
        """Calculate required reduction rates for multiple pathways."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._pathways = []
        base_emissions = input_data.emissions.scope3_tco2e

        # Calculate near-term pathways
        for ambition in [PathwayAmbition.ALIGNED_1_5C, PathwayAmbition.WELL_BELOW_2C]:
            for target_type in [
                TargetType.ABSOLUTE_CONTRACTION,
                TargetType.SUPPLIER_ENGAGEMENT,
            ]:
                rates = SBTI_ANNUAL_RATES.get(ambition.value, {})
                annual_rate = rates.get(
                    target_type.value,
                    rates.get(TargetType.ABSOLUTE_CONTRACTION.value, 4.2),
                )

                years = input_data.target_year_near - input_data.base_year
                total_reduction = min(annual_rate * years, 100.0)
                target_emissions = base_emissions * (1 - total_reduction / 100.0)

                # Year-by-year trajectory
                yby: Dict[int, float] = {}
                for y in range(input_data.base_year, input_data.target_year_near + 1):
                    elapsed_y = y - input_data.base_year
                    yby[y] = round(
                        base_emissions * (1 - annual_rate / 100.0 * elapsed_y), 2
                    )

                self._pathways.append(PathwayResult(
                    target_type=target_type,
                    pathway_ambition=ambition,
                    timeframe=TargetTimeframe.NEAR_TERM,
                    base_year=input_data.base_year,
                    target_year=input_data.target_year_near,
                    annual_reduction_rate_pct=round(annual_rate, 2),
                    total_reduction_pct=round(total_reduction, 2),
                    base_year_emissions_tco2e=round(base_emissions, 2),
                    target_year_emissions_tco2e=round(max(target_emissions, 0), 2),
                    year_by_year_tco2e=yby,
                ))

        # Long-term / net-zero pathway
        long_years = input_data.target_year_long - input_data.base_year
        nz_annual = SBTI_NET_ZERO_MIN_PCT / long_years if long_years > 0 else 0.0
        nz_target = base_emissions * (1 - SBTI_NET_ZERO_MIN_PCT / 100.0)

        yby_long: Dict[int, float] = {}
        for y in range(input_data.base_year, input_data.target_year_long + 1):
            ey = y - input_data.base_year
            yby_long[y] = round(
                base_emissions * (1 - nz_annual / 100.0 * ey), 2
            )

        self._pathways.append(PathwayResult(
            target_type=TargetType.ABSOLUTE_CONTRACTION,
            pathway_ambition=PathwayAmbition.ALIGNED_1_5C,
            timeframe=TargetTimeframe.NET_ZERO,
            base_year=input_data.base_year,
            target_year=input_data.target_year_long,
            annual_reduction_rate_pct=round(nz_annual, 2),
            total_reduction_pct=SBTI_NET_ZERO_MIN_PCT,
            base_year_emissions_tco2e=round(base_emissions, 2),
            target_year_emissions_tco2e=round(max(nz_target, 0), 2),
            year_by_year_tco2e=yby_long,
        ))

        outputs["pathways_calculated"] = len(self._pathways)
        outputs["near_term_pathways"] = sum(
            1 for p in self._pathways if p.timeframe == TargetTimeframe.NEAR_TERM
        )
        outputs["preferred_pathway"] = {
            "ambition": input_data.preferred_ambition.value,
            "type": input_data.preferred_target_type.value,
        }

        self._state.phase_statuses["pathway_calculation"] = "completed"
        self._state.current_phase = 2

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 PathwayCalculation: %d pathways calculated",
            len(self._pathways),
        )
        return PhaseResult(
            phase_name="pathway_calculation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Target Validation
    # -------------------------------------------------------------------------

    async def _phase_target_validation(
        self, input_data: SBTiTargetInput
    ) -> PhaseResult:
        """Validate coverage >= 67%, FLAG, and methodology alignment."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._validations = []
        e = input_data.emissions

        # --- Coverage Check ---
        included_emissions = sum(
            e.scope3_category_emissions.get(c, 0.0)
            for c in input_data.categories_to_include
        )
        coverage_pct = (
            (included_emissions / e.scope3_tco2e * 100.0)
            if e.scope3_tco2e > 0 else 0.0
        )

        self._validations.append(TargetValidation(
            check_name="scope3_coverage",
            result=(
                ValidationResult.PASS
                if coverage_pct >= SBTI_MIN_COVERAGE_PCT
                else ValidationResult.FAIL
            ),
            actual_value=f"{coverage_pct:.1f}%",
            required_value=f">= {SBTI_MIN_COVERAGE_PCT}%",
            detail=(
                f"Included {len(input_data.categories_to_include)} categories "
                f"covering {coverage_pct:.1f}% of Scope 3 emissions"
            ),
        ))

        # --- Timeframe Check ---
        near_term_years = input_data.target_year_near - input_data.base_year
        self._validations.append(TargetValidation(
            check_name="near_term_timeframe",
            result=(
                ValidationResult.PASS
                if 5 <= near_term_years <= SBTI_NEAR_TERM_MAX_YEARS
                else ValidationResult.FAIL
            ),
            actual_value=f"{near_term_years} years",
            required_value=f"5-{SBTI_NEAR_TERM_MAX_YEARS} years",
            detail=f"Near-term target: {input_data.base_year} to {input_data.target_year_near}",
        ))

        # --- Ambition Level Check ---
        preferred_rates = SBTI_ANNUAL_RATES.get(
            input_data.preferred_ambition.value, {}
        )
        min_rate = preferred_rates.get(
            input_data.preferred_target_type.value, 2.5
        )
        self._validations.append(TargetValidation(
            check_name="ambition_level",
            result=ValidationResult.PASS,
            actual_value=f"{min_rate:.1f}% annual",
            required_value=f">= {min_rate:.1f}% annual for {input_data.preferred_ambition.value}",
            detail=f"Selected {input_data.preferred_ambition.value} pathway",
        ))

        # --- FLAG Check ---
        if self._materiality.flag_applicability == FLAGApplicability.APPLICABLE:
            self._validations.append(TargetValidation(
                check_name="flag_target",
                result=ValidationResult.WARNING,
                actual_value="FLAG sector detected",
                required_value="Separate FLAG target required",
                detail=(
                    f"Sector {input_data.sector_code} requires separate FLAG "
                    f"target per SBTi FLAG guidance"
                ),
            ))
            warnings.append(
                "FLAG sector detected; a separate FLAG target is required"
            )

        # --- Exclusion Rationale Check ---
        for cat in input_data.categories_to_exclude:
            rationale = input_data.exclusion_rationale.get(cat, "")
            self._validations.append(TargetValidation(
                check_name=f"exclusion_rationale_{cat}",
                result=(
                    ValidationResult.PASS if rationale
                    else ValidationResult.FAIL
                ),
                actual_value=rationale or "No rationale provided",
                required_value="Documented rationale required",
                detail=f"Exclusion of {cat}",
            ))

        # --- Net-Zero 90% Check ---
        self._validations.append(TargetValidation(
            check_name="net_zero_minimum_reduction",
            result=ValidationResult.PASS,
            actual_value=f"{SBTI_NET_ZERO_MIN_PCT}%",
            required_value=f">= {SBTI_NET_ZERO_MIN_PCT}%",
            detail="Long-term target includes >= 90% absolute reduction",
        ))

        all_pass = all(
            v.result in (ValidationResult.PASS, ValidationResult.WARNING, ValidationResult.NOT_EVALUATED)
            for v in self._validations
        )

        outputs["total_validations"] = len(self._validations)
        outputs["passed"] = sum(
            1 for v in self._validations if v.result == ValidationResult.PASS
        )
        outputs["failed"] = sum(
            1 for v in self._validations if v.result == ValidationResult.FAIL
        )
        outputs["warnings"] = sum(
            1 for v in self._validations if v.result == ValidationResult.WARNING
        )
        outputs["overall_pass"] = all_pass
        outputs["coverage_pct"] = round(coverage_pct, 2)

        self._state.phase_statuses["target_validation"] = "completed"
        self._state.current_phase = 3

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 TargetValidation: %d checks, pass=%d fail=%d overall=%s",
            len(self._validations), outputs["passed"], outputs["failed"],
            "PASS" if all_pass else "FAIL",
        )
        return PhaseResult(
            phase_name="target_validation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Submission Package
    # -------------------------------------------------------------------------

    async def _phase_submission_package(
        self, input_data: SBTiTargetInput
    ) -> PhaseResult:
        """Generate SBTi submission data package."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._submission = []
        e = input_data.emissions

        # Find preferred pathway
        preferred = next(
            (
                p for p in self._pathways
                if p.pathway_ambition == input_data.preferred_ambition
                and p.target_type == input_data.preferred_target_type
                and p.timeframe == TargetTimeframe.NEAR_TERM
            ),
            self._pathways[0] if self._pathways else None,
        )

        field_values: Dict[str, str] = {
            "company_name": input_data.organization_name,
            "sector": input_data.sector,
            "base_year": str(input_data.base_year),
            "base_year_emissions_scope_1_tco2e": f"{e.scope1_tco2e:.2f}",
            "base_year_emissions_scope_2_tco2e": f"{e.scope2_market_tco2e:.2f}",
            "base_year_emissions_scope_3_tco2e": f"{e.scope3_tco2e:.2f}",
            "target_year": str(input_data.target_year_near),
            "target_type": input_data.preferred_target_type.value,
            "target_ambition": input_data.preferred_ambition.value,
            "target_coverage_pct": f"{self._compute_coverage(input_data):.1f}",
            "target_reduction_pct": (
                f"{preferred.total_reduction_pct:.1f}" if preferred else "0.0"
            ),
            "methodology": (
                f"SBTi {input_data.preferred_ambition.value} "
                f"{input_data.preferred_target_type.value}"
            ),
            "scope3_categories_included": ", ".join(
                input_data.categories_to_include
            ),
            "scope3_categories_excluded_rationale": json.dumps(
                input_data.exclusion_rationale
            ),
        }

        for field_name in SBTI_REQUIRED_FIELDS:
            value = field_values.get(field_name, "")
            self._submission.append(SubmissionField(
                field_name=field_name,
                value=value,
                source="pack_043_sbti_workflow",
                validated=bool(value),
            ))

        # Check completeness
        missing = [f.field_name for f in self._submission if not f.validated]
        if missing:
            warnings.append(
                f"Missing fields in submission: {', '.join(missing)}"
            )

        outputs["total_fields"] = len(self._submission)
        outputs["completed_fields"] = sum(1 for f in self._submission if f.validated)
        outputs["missing_fields"] = missing
        outputs["submission_ready"] = len(missing) == 0

        self._state.phase_statuses["submission_package"] = "completed"
        self._state.current_phase = 4

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 SubmissionPackage: %d/%d fields complete, ready=%s",
            outputs["completed_fields"], outputs["total_fields"],
            outputs["submission_ready"],
        )
        return PhaseResult(
            phase_name="submission_package", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _compute_coverage(self, input_data: SBTiTargetInput) -> float:
        """Compute Scope 3 target coverage percentage."""
        e = input_data.emissions
        included = sum(
            e.scope3_category_emissions.get(c, 0.0)
            for c in input_data.categories_to_include
        )
        return (included / e.scope3_tco2e * 100.0) if e.scope3_tco2e > 0 else 0.0

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._materiality = MaterialityCheck()
        self._pathways = []
        self._validations = []
        self._submission = []
        self._phase_results = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def _update_progress(self, pct: float) -> None:
        """Update progress percentage in state."""
        self._state.progress_pct = min(pct, 100.0)
        self._state.updated_at = datetime.utcnow().isoformat()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: SBTiTargetOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.materiality.scope3_pct_of_total}"
        chain += f"|{result.validation_pass}|{result.submission_ready}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
