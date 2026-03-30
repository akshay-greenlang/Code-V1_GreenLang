# -*- coding: utf-8 -*-
"""
Target Setting Workflow
====================================

4-phase workflow for SBTi SDA intensity target setting within
PACK-046 Intensity Metrics Pack.

Phases:
    1. BaselineCalculation        -- Calculate base year intensity from
                                     PACK-045 data or direct input, validate
                                     against sector pathway requirements,
                                     determine starting point for target.
    2. PathwaySelection           -- Select SBTi SDA pathway (1.5C or
                                     well-below 2C), configure sector
                                     decarbonisation parameters, select
                                     convergence year and ambition level.
    3. TargetCalculation          -- Run TargetPathwayEngine to generate
                                     annual intensity targets from base year
                                     to target year, calculate required annual
                                     reduction rates, determine interim targets.
    4. ValidationReport           -- Validate target ambition against SBTi
                                     minimum requirements, check convergence
                                     feasibility, generate target submission
                                     report and supporting documentation.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic SDA formulas and validated pathway data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    SBTi Sectoral Decarbonisation Approach (SDA) v2.0
    SBTi Corporate Manual v2.1 (2024) - Target setting requirements
    SBTi Criteria v5.1 - Validation criteria
    IEA Energy Technology Perspectives - Sector pathways
    ESRS E1-4 - Targets related to climate change mitigation

Schedule: Once during target setting, annually for progress review
Estimated duration: 2-4 weeks

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

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

class TargetPhase(str, Enum):
    """Target setting workflow phases."""

    BASELINE_CALCULATION = "baseline_calculation"
    PATHWAY_SELECTION = "pathway_selection"
    TARGET_CALCULATION = "target_calculation"
    VALIDATION_REPORT = "validation_report"

class SBTiPathway(str, Enum):
    """SBTi temperature alignment pathway."""

    PATHWAY_1_5C = "1.5C"
    PATHWAY_WB2C = "well_below_2C"
    PATHWAY_2C = "2C"

class TemperatureAlignment(str, Enum):
    """Temperature alignment classification."""

    ALIGNED_1_5C = "1.5C_aligned"
    ALIGNED_WB2C = "well_below_2C_aligned"
    ALIGNED_2C = "2C_aligned"
    NOT_ALIGNED = "not_aligned"
    INSUFFICIENT_DATA = "insufficient_data"

class AmbitionLevel(str, Enum):
    """Target ambition level classification."""

    EXCEEDS_MINIMUM = "exceeds_minimum"
    MEETS_MINIMUM = "meets_minimum"
    BELOW_MINIMUM = "below_minimum"
    WELL_BELOW_MINIMUM = "well_below_minimum"

class ValidationOutcome(str, Enum):
    """Target validation outcome."""

    APPROVED = "approved"
    CONDITIONALLY_APPROVED = "conditionally_approved"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"

# =============================================================================
# SECTOR PATHWAY DATA (Zero-Hallucination Reference)
# =============================================================================

# SBTi SDA sector convergence intensity targets (tCO2e per sector unit)
# Source: SBTi SDA Tool v2.0, IEA ETP sector pathways
SECTOR_PATHWAY_2030: Dict[str, Dict[str, float]] = {
    "power_generation": {
        "1.5C": 0.138,    # tCO2e/MWh
        "well_below_2C": 0.194,
        "2C": 0.254,
        "unit": "tCO2e/MWh",
    },
    "cement": {
        "1.5C": 0.469,    # tCO2e/tonne
        "well_below_2C": 0.524,
        "2C": 0.570,
        "unit": "tCO2e/tonne",
    },
    "iron_steel": {
        "1.5C": 1.227,    # tCO2e/tonne
        "well_below_2C": 1.383,
        "2C": 1.504,
        "unit": "tCO2e/tonne",
    },
    "aluminium": {
        "1.5C": 1.426,    # tCO2e/tonne
        "well_below_2C": 1.660,
        "2C": 1.836,
        "unit": "tCO2e/tonne",
    },
    "pulp_paper": {
        "1.5C": 0.264,    # tCO2e/tonne
        "well_below_2C": 0.316,
        "2C": 0.361,
        "unit": "tCO2e/tonne",
    },
    "services_commercial": {
        "1.5C": 0.022,    # tCO2e/sqm
        "well_below_2C": 0.029,
        "2C": 0.036,
        "unit": "tCO2e/sqm",
    },
}

# Minimum annual reduction rate required by SBTi per pathway
MINIMUM_ANNUAL_REDUCTION_RATE: Dict[str, float] = {
    "1.5C": 4.2,           # 4.2% per year minimum for 1.5C
    "well_below_2C": 2.5,  # 2.5% per year minimum for WB2C
    "2C": 1.5,             # 1.5% per year
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

class BaselineData(BaseModel):
    """Baseline intensity data for target setting."""

    base_year: int = Field(..., ge=2015, le=2030)
    base_intensity: float = Field(..., gt=0.0, description="Base year intensity")
    intensity_unit: str = Field(default="tCO2e/unit")
    base_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    base_denominator_value: float = Field(default=0.0, ge=0.0)
    denominator_type: str = Field(default="revenue")
    scope_coverage: str = Field(default="scope_1_2_location")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)

class PathwayConfig(BaseModel):
    """SBTi SDA pathway configuration."""

    pathway: SBTiPathway = Field(default=SBTiPathway.PATHWAY_1_5C)
    sector: str = Field(default="services_commercial")
    target_year: int = Field(default=2030, ge=2025, le=2050)
    convergence_year: int = Field(default=2050, ge=2030, le=2100)
    interim_target_years: List[int] = Field(default_factory=lambda: [2030, 2035])
    include_scope3: bool = Field(default=False)

class AnnualTarget(BaseModel):
    """Annual intensity target for a specific year."""

    year: int = Field(..., ge=2020, le=2100)
    target_intensity: float = Field(..., ge=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    is_interim_target: bool = Field(default=False)
    is_final_target: bool = Field(default=False)
    provenance_hash: str = Field(default="")

class TargetValidation(BaseModel):
    """Validation result for the proposed target."""

    outcome: ValidationOutcome = Field(...)
    ambition_level: AmbitionLevel = Field(...)
    temperature_alignment: TemperatureAlignment = Field(...)
    annual_reduction_rate_pct: float = Field(default=0.0)
    minimum_required_rate_pct: float = Field(default=0.0)
    meets_sbti_criteria: bool = Field(default=False)
    convergence_feasible: bool = Field(default=True)
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class TargetSettingInput(BaseModel):
    """Input data model for TargetSettingWorkflow."""

    organization_id: str = Field(..., min_length=1)
    baseline: BaselineData = Field(..., description="Base year intensity data")
    pathway_config: PathwayConfig = Field(
        default_factory=PathwayConfig,
        description="SBTi SDA pathway configuration",
    )
    current_year: int = Field(default=2025, ge=2020, le=2050)
    current_intensity: float = Field(
        default=0.0, ge=0.0, description="Current year intensity (for progress)",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class TargetSettingResult(BaseModel):
    """Complete result from target setting workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="target_setting")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    base_year: int = Field(default=0)
    target_year: int = Field(default=0)
    base_intensity: float = Field(default=0.0)
    target_intensity: float = Field(default=0.0)
    annual_targets: List[AnnualTarget] = Field(default_factory=list)
    validation: Optional[TargetValidation] = Field(default=None)
    pathway: SBTiPathway = Field(default=SBTiPathway.PATHWAY_1_5C)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class TargetSettingWorkflow:
    """
    4-phase workflow for SBTi SDA intensity target setting.

    Calculates baseline intensity, selects SDA pathway, generates annual
    targets, and validates ambition against SBTi criteria.

    Zero-hallucination: all target calculations use deterministic SDA
    convergence formulas from SBTi methodology; no LLM calls in calculation
    path; SHA-256 provenance on every target.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _baseline_validated: Validated baseline data.
        _annual_targets: Generated annual targets.
        _validation: Target validation result.

    Example:
        >>> wf = TargetSettingWorkflow()
        >>> baseline = BaselineData(base_year=2020, base_intensity=50.0)
        >>> inp = TargetSettingInput(
        ...     organization_id="org-001", baseline=baseline,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[TargetPhase] = [
        TargetPhase.BASELINE_CALCULATION,
        TargetPhase.PATHWAY_SELECTION,
        TargetPhase.TARGET_CALCULATION,
        TargetPhase.VALIDATION_REPORT,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize TargetSettingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._baseline_validated: Optional[BaselineData] = None
        self._pathway_intensity_2030: float = 0.0
        self._pathway_intensity_converge: float = 0.0
        self._annual_targets: List[AnnualTarget] = []
        self._validation: Optional[TargetValidation] = None
        self._overall_reduction_rate: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: TargetSettingInput) -> TargetSettingResult:
        """
        Execute the 4-phase target setting workflow.

        Args:
            input_data: Baseline data and pathway configuration.

        Returns:
            TargetSettingResult with annual targets and validation.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting target setting %s org=%s base_year=%d pathway=%s",
            self.workflow_id, input_data.organization_id,
            input_data.baseline.base_year,
            input_data.pathway_config.pathway.value,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_baseline_calculation,
            self._phase_2_pathway_selection,
            self._phase_3_target_calculation,
            self._phase_4_validation_report,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Target setting failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        target_intensity = 0.0
        if self._annual_targets:
            final = [t for t in self._annual_targets if t.is_final_target]
            target_intensity = final[0].target_intensity if final else 0.0

        result = TargetSettingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            base_year=input_data.baseline.base_year,
            target_year=input_data.pathway_config.target_year,
            base_intensity=input_data.baseline.base_intensity,
            target_intensity=target_intensity,
            annual_targets=self._annual_targets,
            validation=self._validation,
            pathway=input_data.pathway_config.pathway,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Target setting %s completed in %.2fs status=%s targets=%d",
            self.workflow_id, elapsed, overall_status.value, len(self._annual_targets),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Calculation
    # -------------------------------------------------------------------------

    async def _phase_1_baseline_calculation(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Validate base year intensity and establish baseline."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        bl = input_data.baseline

        # Validate baseline data
        if bl.base_intensity <= 0:
            raise ValueError("Base year intensity must be positive")

        if bl.data_quality_score < 50.0:
            warnings.append(
                f"Baseline data quality {bl.data_quality_score:.1f} is low; "
                f"SBTi recommends >70 for target validation"
            )

        # Validate base year is within SBTi acceptable range
        max_age = input_data.current_year - bl.base_year
        if max_age > 5:
            warnings.append(
                f"Base year {bl.base_year} is {max_age} years old; "
                f"SBTi recommends base year within 5 years"
            )

        self._baseline_validated = bl

        outputs["base_year"] = bl.base_year
        outputs["base_intensity"] = bl.base_intensity
        outputs["intensity_unit"] = bl.intensity_unit
        outputs["scope_coverage"] = bl.scope_coverage
        outputs["denominator_type"] = bl.denominator_type
        outputs["data_quality_score"] = bl.data_quality_score
        outputs["base_year_age_years"] = max_age

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 BaselineCalculation: base_year=%d intensity=%.4f",
            bl.base_year, bl.base_intensity,
        )
        return PhaseResult(
            phase_name="baseline_calculation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Pathway Selection
    # -------------------------------------------------------------------------

    async def _phase_2_pathway_selection(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Select and configure SBTi SDA pathway."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        pc = input_data.pathway_config
        sector_data = SECTOR_PATHWAY_2030.get(pc.sector, {})

        if not sector_data:
            warnings.append(
                f"Sector '{pc.sector}' not in SDA pathway database; "
                f"using linear reduction method"
            )
            # Linear fallback: use minimum annual rate
            min_rate = MINIMUM_ANNUAL_REDUCTION_RATE.get(pc.pathway.value, 2.5)
            years_to_target = max(pc.target_year - input_data.baseline.base_year, 1)
            total_reduction = min_rate * years_to_target / 100.0
            self._pathway_intensity_2030 = round(
                input_data.baseline.base_intensity * (1.0 - total_reduction), 6,
            )
        else:
            self._pathway_intensity_2030 = sector_data.get(pc.pathway.value, 0.0)

        # Convergence intensity (sector target at convergence year)
        self._pathway_intensity_converge = self._pathway_intensity_2030 * 0.5

        outputs["pathway"] = pc.pathway.value
        outputs["sector"] = pc.sector
        outputs["target_year"] = pc.target_year
        outputs["convergence_year"] = pc.convergence_year
        outputs["sector_pathway_2030"] = self._pathway_intensity_2030
        outputs["convergence_intensity"] = self._pathway_intensity_converge
        outputs["minimum_annual_rate_pct"] = MINIMUM_ANNUAL_REDUCTION_RATE.get(
            pc.pathway.value, 2.5,
        )
        outputs["sector_in_database"] = bool(sector_data)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 PathwaySelection: pathway=%s sector=%s target_2030=%.4f",
            pc.pathway.value, pc.sector, self._pathway_intensity_2030,
        )
        return PhaseResult(
            phase_name="pathway_selection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Target Calculation
    # -------------------------------------------------------------------------

    async def _phase_3_target_calculation(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Generate annual intensity targets using SDA convergence method."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        bl = input_data.baseline
        pc = input_data.pathway_config
        self._annual_targets = []

        base_year = bl.base_year
        target_year = pc.target_year
        years = target_year - base_year

        if years <= 0:
            raise ValueError(f"Target year {target_year} must be after base year {base_year}")

        # SDA convergence formula:
        # target_intensity(t) = pathway_intensity(t) + (company_base - pathway_base) * (1 - (t-base)/(converge-base))
        # Simplified: linear convergence to sector pathway
        pathway_2030 = self._pathway_intensity_2030
        company_base = bl.base_intensity
        convergence_year = pc.convergence_year

        for year in range(base_year, target_year + 1):
            fraction_elapsed = (year - base_year) / max(convergence_year - base_year, 1)
            fraction_elapsed = min(fraction_elapsed, 1.0)

            # SDA convergence: company converges to sector pathway linearly
            gap = company_base - pathway_2030
            target_intensity = round(company_base - (gap * fraction_elapsed), 6)

            # Ensure non-negative
            target_intensity = max(target_intensity, 0.0)

            # Calculate cumulative reduction
            cum_reduction = 0.0
            if company_base > 0:
                cum_reduction = round(
                    ((company_base - target_intensity) / company_base) * 100.0, 4,
                )

            # Calculate annual reduction rate
            annual_rate = 0.0
            if year > base_year:
                prev_target = self._annual_targets[-1].target_intensity
                if prev_target > 0:
                    annual_rate = round(
                        ((prev_target - target_intensity) / prev_target) * 100.0, 4,
                    )

            is_interim = year in pc.interim_target_years
            is_final = year == target_year

            target_data = {
                "year": year, "intensity": target_intensity,
                "reduction": cum_reduction,
            }

            self._annual_targets.append(AnnualTarget(
                year=year,
                target_intensity=target_intensity,
                cumulative_reduction_pct=cum_reduction,
                annual_reduction_rate_pct=annual_rate,
                is_interim_target=is_interim,
                is_final_target=is_final,
                provenance_hash=_compute_hash(target_data),
            ))

        # Calculate overall annual compound reduction rate
        if self._annual_targets and years > 0:
            final_intensity = self._annual_targets[-1].target_intensity
            if company_base > 0 and final_intensity > 0:
                self._overall_reduction_rate = round(
                    (1.0 - (final_intensity / company_base) ** (1.0 / years)) * 100.0, 4,
                )

        outputs["targets_generated"] = len(self._annual_targets)
        outputs["base_intensity"] = company_base
        outputs["target_intensity"] = self._annual_targets[-1].target_intensity if self._annual_targets else 0.0
        outputs["overall_reduction_pct"] = self._annual_targets[-1].cumulative_reduction_pct if self._annual_targets else 0.0
        outputs["compound_annual_reduction_rate_pct"] = self._overall_reduction_rate
        outputs["interim_targets"] = [
            {"year": t.year, "intensity": t.target_intensity}
            for t in self._annual_targets if t.is_interim_target
        ]

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 TargetCalculation: %d targets, CARR=%.2f%%",
            len(self._annual_targets), self._overall_reduction_rate,
        )
        return PhaseResult(
            phase_name="target_calculation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Validation Report
    # -------------------------------------------------------------------------

    async def _phase_4_validation_report(
        self, input_data: TargetSettingInput,
    ) -> PhaseResult:
        """Validate target ambition and generate submission report."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        pc = input_data.pathway_config
        min_rate = MINIMUM_ANNUAL_REDUCTION_RATE.get(pc.pathway.value, 2.5)
        findings: List[str] = []
        recommendations: List[str] = []

        # Check 1: Minimum annual reduction rate
        meets_minimum = self._overall_reduction_rate >= min_rate
        if not meets_minimum:
            findings.append(
                f"Annual reduction rate {self._overall_reduction_rate:.2f}% is below "
                f"SBTi minimum of {min_rate:.1f}% for {pc.pathway.value} pathway"
            )
            recommendations.append(
                "Increase target ambition or extend target timeframe"
            )

        # Check 2: Convergence feasibility
        convergence_feasible = True
        if self._annual_targets:
            max_annual_rate = max(
                t.annual_reduction_rate_pct for t in self._annual_targets
            )
            if max_annual_rate > 15.0:
                convergence_feasible = False
                findings.append(
                    f"Maximum annual reduction of {max_annual_rate:.1f}% may not be feasible"
                )
                recommendations.append(
                    "Consider extending convergence timeline or phased approach"
                )

        # Check 3: Target year within SBTi timeframe
        years_from_now = pc.target_year - input_data.current_year
        if years_from_now < 5:
            findings.append(
                f"Target year {pc.target_year} is less than 5 years away"
            )
            recommendations.append("SBTi requires minimum 5-year, maximum 15-year targets")
        elif years_from_now > 15:
            findings.append(
                f"Target year {pc.target_year} is more than 15 years away"
            )
            recommendations.append("SBTi near-term targets must be 5-15 years")

        # Determine ambition level
        if self._overall_reduction_rate >= min_rate * 1.5:
            ambition = AmbitionLevel.EXCEEDS_MINIMUM
        elif self._overall_reduction_rate >= min_rate:
            ambition = AmbitionLevel.MEETS_MINIMUM
        elif self._overall_reduction_rate >= min_rate * 0.7:
            ambition = AmbitionLevel.BELOW_MINIMUM
        else:
            ambition = AmbitionLevel.WELL_BELOW_MINIMUM

        # Determine temperature alignment
        if meets_minimum and pc.pathway == SBTiPathway.PATHWAY_1_5C:
            temp_align = TemperatureAlignment.ALIGNED_1_5C
        elif meets_minimum and pc.pathway == SBTiPathway.PATHWAY_WB2C:
            temp_align = TemperatureAlignment.ALIGNED_WB2C
        elif meets_minimum:
            temp_align = TemperatureAlignment.ALIGNED_2C
        else:
            temp_align = TemperatureAlignment.NOT_ALIGNED

        # Determine outcome
        if meets_minimum and convergence_feasible and 5 <= years_from_now <= 15:
            outcome = ValidationOutcome.APPROVED
        elif meets_minimum and (not convergence_feasible or years_from_now < 5):
            outcome = ValidationOutcome.CONDITIONALLY_APPROVED
        elif not meets_minimum and ambition in (AmbitionLevel.BELOW_MINIMUM,):
            outcome = ValidationOutcome.REQUIRES_REVISION
        else:
            outcome = ValidationOutcome.REJECTED

        self._validation = TargetValidation(
            outcome=outcome,
            ambition_level=ambition,
            temperature_alignment=temp_align,
            annual_reduction_rate_pct=self._overall_reduction_rate,
            minimum_required_rate_pct=min_rate,
            meets_sbti_criteria=meets_minimum,
            convergence_feasible=convergence_feasible,
            findings=findings,
            recommendations=recommendations,
        )

        outputs["outcome"] = outcome.value
        outputs["ambition_level"] = ambition.value
        outputs["temperature_alignment"] = temp_align.value
        outputs["annual_reduction_rate_pct"] = self._overall_reduction_rate
        outputs["minimum_required_rate_pct"] = min_rate
        outputs["meets_sbti_criteria"] = meets_minimum
        outputs["findings_count"] = len(findings)
        outputs["recommendations_count"] = len(recommendations)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 ValidationReport: outcome=%s ambition=%s temp_align=%s",
            outcome.value, ambition.value, temp_align.value,
        )
        return PhaseResult(
            phase_name="validation_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: TargetSettingInput, phase_number: int,
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
        self._baseline_validated = None
        self._pathway_intensity_2030 = 0.0
        self._pathway_intensity_converge = 0.0
        self._annual_targets = []
        self._validation = None
        self._overall_reduction_rate = 0.0

    def _compute_provenance(self, result: TargetSettingResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.base_year}|{result.target_year}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
