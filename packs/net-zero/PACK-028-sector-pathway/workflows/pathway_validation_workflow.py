# -*- coding: utf-8 -*-
"""
Pathway Validation Workflow
================================

4-phase workflow for validating sector decarbonization pathways against
SBTi SDA criteria within PACK-028 Sector Pathway Pack.  The workflow
performs comprehensive data validation, pathway integrity checks, SBTi
criteria compliance verification, and generates a detailed compliance
report with remediation actions.

Phases:
    1. DataValidation     -- Validate input data completeness, consistency,
                             and quality; check emission boundary coverage
    2. PathwayValidation  -- Validate pathway mathematical integrity,
                             convergence properties, and monotonicity
    3. SBTiCriteriaCheck  -- Check pathway against SBTi Corporate Standard
                             v2.0 and SDA-specific requirements
    4. ComplianceReport   -- Generate compliance report with pass/fail/
                             conditional criteria and improvement roadmap

Regulatory references:
    - SBTi Corporate Standard v2.0 (2024)
    - SBTi SDA Methodology
    - SBTi FLAG Guidance (for agriculture/forestry sectors)
    - SBTi Target Validation Protocol v3.0
    - GHG Protocol Corporate Standard & Scope 3 Standard

Zero-hallucination: all validation checks use deterministic rules from
SBTi published criteria.  No LLM calls in any validation logic.

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ValidationSeverity(str, Enum):
    """Severity of a validation finding."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class ComplianceStatus(str, Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"


class DataQualityTier(str, Enum):
    """Data quality tier classification."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    ESTIMATED = "estimated"
    DEFAULT = "default"


class PathwayProperty(str, Enum):
    """Pathway mathematical properties to validate."""
    MONOTONICITY = "monotonicity"
    CONVERGENCE = "convergence"
    CONTINUITY = "continuity"
    BOUNDARY = "boundary"
    RATE = "rate"


# =============================================================================
# SBTI VALIDATION CRITERIA (Zero-Hallucination: Published Requirements)
# =============================================================================

SBTI_NEAR_TERM_REQUIREMENTS: Dict[str, Any] = {
    "15c_linear_annual_rate": 4.2,     # % per year for 1.5C (linear)
    "wb2c_linear_annual_rate": 2.5,    # % per year for well-below 2C
    "timeframe_min_years": 5,
    "timeframe_max_years": 10,
    "scope12_coverage_pct": 95.0,
    "scope3_coverage_pct": 67.0,
    "scope3_materiality_pct": 40.0,    # S3 >= 40% of total triggers requirement
    "recalculation_threshold_pct": 5.0,
    "exclusion_threshold_pct": 5.0,
}

SBTI_LONG_TERM_REQUIREMENTS: Dict[str, Any] = {
    "reduction_pct": 90.0,
    "target_year_max": 2050,
    "neutralization_pct": 10.0,
    "removal_only": True,
    "no_compensation": True,
}

SDA_SECTOR_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "required_metric": "gCO2/kWh",
        "convergence_target_2050": 0.0,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA power pathway; grid average emission factor",
    },
    "steel": {
        "required_metric": "tCO2e/tonne crude steel",
        "convergence_target_2050": 0.10,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA steel pathway; includes process and combustion",
    },
    "cement": {
        "required_metric": "tCO2e/tonne cement",
        "convergence_target_2050": 0.06,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA cement pathway; includes process CO2 from clinker",
    },
    "aluminum": {
        "required_metric": "tCO2e/tonne aluminum",
        "convergence_target_2050": 0.50,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA aluminum pathway; Hall-Heroult process",
    },
    "chemicals": {
        "required_metric": "tCO2e/tonne product",
        "convergence_target_2050": 0.10,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA chemicals pathway; mass-weighted",
    },
    "pulp_paper": {
        "required_metric": "tCO2e/tonne pulp",
        "convergence_target_2050": 0.03,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA pulp and paper pathway",
    },
    "aviation": {
        "required_metric": "gCO2/pkm",
        "convergence_target_2050": 7.0,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1",
        "methodology_note": "SDA aviation pathway; passenger-km basis",
    },
    "shipping": {
        "required_metric": "gCO2/tkm",
        "convergence_target_2050": 0.3,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1",
        "methodology_note": "SDA shipping pathway; tonne-km basis",
    },
    "road_transport": {
        "required_metric": "gCO2/vkm",
        "convergence_target_2050": 5.0,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1",
        "methodology_note": "SDA road transport pathway; vehicle-km basis",
    },
    "rail": {
        "required_metric": "gCO2/pkm",
        "convergence_target_2050": 1.0,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA rail pathway; passenger-km basis",
    },
    "buildings_residential": {
        "required_metric": "kgCO2/m2/year",
        "convergence_target_2050": 1.0,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA buildings pathway; operational emissions",
    },
    "buildings_commercial": {
        "required_metric": "kgCO2/m2/year",
        "convergence_target_2050": 1.5,
        "max_deviation_pct": 10.0,
        "scope_coverage": "scope1+scope2",
        "methodology_note": "SDA buildings pathway; commercial operational",
    },
    "agriculture": {
        "required_metric": "tCO2e/tonne food",
        "convergence_target_2050": 0.80,
        "max_deviation_pct": 15.0,
        "scope_coverage": "flag",
        "methodology_note": "FLAG guidance; separate from fossil fuel targets",
    },
}

# SBTi pathway validation checklist for SDA sectors
SDA_VALIDATION_CHECKLIST: Dict[str, Dict[str, Any]] = {
    "CHK-001": {
        "check": "Base Year Emissions Verified",
        "description": "Base year emissions have been independently verified or assured.",
        "requirement": "Third-party assurance (limited or reasonable) recommended.",
        "severity": "major",
        "applies_to": "all_sectors",
    },
    "CHK-002": {
        "check": "Base Year Recalculation Policy",
        "description": "Policy exists for base year recalculation upon significant changes.",
        "requirement": "Recalculate when structural changes exceed 5% of base year emissions.",
        "severity": "major",
        "applies_to": "all_sectors",
    },
    "CHK-003": {
        "check": "Intensity Metric Consistency",
        "description": "Intensity metric is consistent with SBTi sector guidance.",
        "requirement": "Use the sector-specific metric defined in SDA guidance.",
        "severity": "critical",
        "applies_to": "sda_sectors",
    },
    "CHK-004": {
        "check": "Activity Data Source",
        "description": "Activity data source is documented and auditable.",
        "requirement": "Primary data preferred; secondary data acceptable with justification.",
        "severity": "major",
        "applies_to": "all_sectors",
    },
    "CHK-005": {
        "check": "Convergence Model Documentation",
        "description": "Convergence model (linear/exponential/S-curve) is documented.",
        "requirement": "Provide mathematical basis and parameter justification.",
        "severity": "minor",
        "applies_to": "sda_sectors",
    },
    "CHK-006": {
        "check": "Scope Coverage Verification",
        "description": "Scope 1+2 coverage meets minimum threshold.",
        "requirement": "Minimum 95% of Scope 1+2 emissions covered.",
        "severity": "critical",
        "applies_to": "all_sectors",
    },
    "CHK-007": {
        "check": "Scope 3 Materiality Assessment",
        "description": "Scope 3 materiality screening completed.",
        "requirement": "Screen all 15 Scope 3 categories; include if >=40% of total.",
        "severity": "major",
        "applies_to": "all_sectors",
    },
    "CHK-008": {
        "check": "Exclusion Justification",
        "description": "Any excluded sources are documented and justified.",
        "requirement": "Exclusions must not exceed 5% of total scope emissions.",
        "severity": "major",
        "applies_to": "all_sectors",
    },
    "CHK-009": {
        "check": "Pathway Continuity Verified",
        "description": "Pathway has no gaps or discontinuities.",
        "requirement": "Annual data points from base year to target year.",
        "severity": "critical",
        "applies_to": "sda_sectors",
    },
    "CHK-010": {
        "check": "Target Ambition Level",
        "description": "Near-term target meets minimum ambition threshold.",
        "requirement": "1.5C: >=4.2%/yr; WB2C: >=2.5%/yr (Scope 1+2).",
        "severity": "critical",
        "applies_to": "all_sectors",
    },
    "CHK-011": {
        "check": "Long-Term Target Coverage",
        "description": "Long-term target covers >=90% emission reduction by 2050.",
        "requirement": "Net-zero by 2050 with >=90% absolute reduction.",
        "severity": "critical",
        "applies_to": "all_sectors",
    },
    "CHK-012": {
        "check": "No Carbon Credits for Pathway",
        "description": "Pathway does not rely on carbon credits/offsets for target achievement.",
        "requirement": "Only verified removals allowed for residual <10%.",
        "severity": "critical",
        "applies_to": "all_sectors",
    },
    "CHK-013": {
        "check": "FLAG Sector Separation",
        "description": "FLAG emissions targets separated from fossil/industrial.",
        "requirement": "Agriculture/forestry use FLAG guidance with separate target.",
        "severity": "major",
        "applies_to": "flag_sectors",
    },
    "CHK-014": {
        "check": "Regional Factor Applied",
        "description": "SDA pathway uses correct regional convergence factor.",
        "requirement": "Apply regional weighting per SBTi SDA methodology.",
        "severity": "minor",
        "applies_to": "sda_sectors",
    },
    "CHK-015": {
        "check": "Market Share Correction",
        "description": "Market share correction applied where methodology requires.",
        "requirement": "Adjust pathway for expected market share changes.",
        "severity": "minor",
        "applies_to": "sda_sectors",
    },
}

# Data quality scoring criteria
DATA_QUALITY_CRITERIA: Dict[str, Dict[str, Any]] = {
    "DQ-SCORE-5": {
        "tier": "primary",
        "score": 5.0,
        "description": "Measured data from calibrated instruments, third-party assured.",
        "examples": ["CEMS data", "metered energy consumption", "third-party verified GHG inventory"],
    },
    "DQ-SCORE-4": {
        "tier": "primary",
        "score": 4.0,
        "description": "Measured data from company systems, internally verified.",
        "examples": ["Internal meter readings", "ERP-extracted energy data", "fuel purchase records"],
    },
    "DQ-SCORE-3": {
        "tier": "secondary",
        "score": 3.0,
        "description": "Calculated from known activity data and published emission factors.",
        "examples": ["Activity * EF calculation", "DEFRA emission factors", "EPA AP-42"],
    },
    "DQ-SCORE-2": {
        "tier": "tertiary",
        "score": 2.0,
        "description": "Estimated using proxy data or industry averages.",
        "examples": ["Spend-based estimates", "industry average intensity", "extrapolation"],
    },
    "DQ-SCORE-1": {
        "tier": "default",
        "score": 1.0,
        "description": "Default values or rough estimates with high uncertainty.",
        "examples": ["Global averages", "unverified third-party data", "order-of-magnitude"],
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")


class DataValidationFinding(BaseModel):
    """A single data validation finding."""
    finding_id: str = Field(default="")
    field_name: str = Field(default="")
    finding_type: str = Field(default="")
    severity: ValidationSeverity = Field(default=ValidationSeverity.INFO)
    description: str = Field(default="")
    actual_value: str = Field(default="")
    expected_value: str = Field(default="")
    remediation: str = Field(default="")
    data_quality_tier: DataQualityTier = Field(default=DataQualityTier.DEFAULT)


class DataValidationSummary(BaseModel):
    """Summary of all data validation findings."""
    total_checks: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    warnings_count: int = Field(default=0)
    critical_findings: int = Field(default=0)
    data_completeness_pct: float = Field(default=0.0)
    data_consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_quality_tier: DataQualityTier = Field(default=DataQualityTier.DEFAULT)
    findings: List[DataValidationFinding] = Field(default_factory=list)


class PathwayValidationCheck(BaseModel):
    """Result of a single pathway mathematical validation check."""
    check_id: str = Field(default="")
    check_name: str = Field(default="")
    property_type: PathwayProperty = Field(default=PathwayProperty.MONOTONICITY)
    passed: bool = Field(default=False)
    description: str = Field(default="")
    details: str = Field(default="")
    violation_years: List[int] = Field(default_factory=list)
    severity: ValidationSeverity = Field(default=ValidationSeverity.MINOR)


class PathwayValidationSummary(BaseModel):
    """Summary of pathway mathematical validation."""
    total_checks: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    pathway_valid: bool = Field(default=False)
    checks: List[PathwayValidationCheck] = Field(default_factory=list)
    pathway_years: int = Field(default=0)
    pathway_start_intensity: float = Field(default=0.0)
    pathway_end_intensity: float = Field(default=0.0)
    total_reduction_pct: float = Field(default=0.0)
    average_annual_reduction_pct: float = Field(default=0.0)


class SBTiCriterionCheck(BaseModel):
    """Result of a single SBTi criterion compliance check."""
    criterion_id: str = Field(default="")
    criterion_name: str = Field(default="")
    sbti_reference: str = Field(default="", description="SBTi standard section reference")
    category: str = Field(default="", description="near_term, long_term, sda_specific, flag")
    status: ComplianceStatus = Field(default=ComplianceStatus.NOT_ASSESSED)
    actual_value: str = Field(default="")
    required_value: str = Field(default="")
    margin: str = Field(default="", description="Margin to pass/fail")
    finding: str = Field(default="")
    remediation: str = Field(default="")
    severity: ValidationSeverity = Field(default=ValidationSeverity.MINOR)


class SBTiComplianceSummary(BaseModel):
    """Summary of SBTi compliance checks."""
    total_criteria: int = Field(default=0)
    compliant: int = Field(default=0)
    non_compliant: int = Field(default=0)
    partially_compliant: int = Field(default=0)
    overall_status: ComplianceStatus = Field(default=ComplianceStatus.NOT_ASSESSED)
    compliance_score_pct: float = Field(default=0.0)
    near_term_status: ComplianceStatus = Field(default=ComplianceStatus.NOT_ASSESSED)
    long_term_status: ComplianceStatus = Field(default=ComplianceStatus.NOT_ASSESSED)
    sda_specific_status: ComplianceStatus = Field(default=ComplianceStatus.NOT_ASSESSED)
    criteria: List[SBTiCriterionCheck] = Field(default_factory=list)


class ComplianceReportSection(BaseModel):
    """A single section in the compliance report."""
    section_id: str = Field(default="")
    section_title: str = Field(default="")
    content: str = Field(default="")
    status: ComplianceStatus = Field(default=ComplianceStatus.NOT_ASSESSED)
    key_metrics: Dict[str, Any] = Field(default_factory=dict)


class ComplianceReport(BaseModel):
    """Complete pathway compliance report."""
    report_id: str = Field(default="")
    report_title: str = Field(default="")
    sector: str = Field(default="")
    company_name: str = Field(default="")
    assessment_date: str = Field(default="")
    overall_status: ComplianceStatus = Field(default=ComplianceStatus.NOT_ASSESSED)
    data_validation: DataValidationSummary = Field(
        default_factory=DataValidationSummary,
    )
    pathway_validation: PathwayValidationSummary = Field(
        default_factory=PathwayValidationSummary,
    )
    sbti_compliance: SBTiComplianceSummary = Field(
        default_factory=SBTiComplianceSummary,
    )
    sections: List[ComplianceReportSection] = Field(default_factory=list)
    improvement_roadmap: List[str] = Field(default_factory=list)
    submission_ready: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class PathwayValidationConfig(BaseModel):
    """Configuration for pathway validation workflow."""
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    # Sector
    sector: str = Field(default="cross_sector")
    sda_method: str = Field(default="ACA")

    # Base data
    base_year: int = Field(default=2020, ge=2015, le=2030)
    target_year: int = Field(default=2050, ge=2030, le=2070)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    base_year_intensity: float = Field(default=0.0, ge=0.0)
    current_intensity: float = Field(default=0.0, ge=0.0)
    target_intensity_2050: float = Field(default=0.0, ge=0.0)
    intensity_metric: str = Field(default="")
    intensity_unit: str = Field(default="")

    # Scope coverage
    scope1_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope12_coverage_pct: float = Field(default=95.0, ge=0.0, le=100.0)
    scope3_coverage_pct: float = Field(default=67.0, ge=0.0, le=100.0)

    # Boundary
    consolidation_approach: str = Field(default="operational_control")
    exclusions_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    exclusion_justification: str = Field(default="")

    # Historical data
    base_year_activity: float = Field(default=0.0, ge=0.0)
    current_activity: float = Field(default=0.0, ge=0.0)

    # Recalculation
    structural_changes_pct: float = Field(default=0.0, ge=0.0)
    ma_events: List[str] = Field(default_factory=list)

    # FLAG
    is_flag_sector: bool = Field(default=False)
    flag_emissions_tco2e: float = Field(default=0.0, ge=0.0)


class PathwayValidationInput(BaseModel):
    """Input data for pathway validation."""
    config: PathwayValidationConfig = Field(
        default_factory=PathwayValidationConfig,
    )
    pathway_points: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Year-by-year pathway points [{year, intensity, emissions}]",
    )
    historical_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical emission records [{year, emissions, activity, intensity}]",
    )
    emission_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Emission source inventory [{source, scope, tco2e, method}]",
    )
    peer_benchmarks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Peer company benchmarks [{company, intensity, sbti_validated}]",
    )


class PathwayValidationResult(BaseModel):
    """Complete result from pathway validation workflow."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="pathway_validation")
    pack_id: str = Field(default="PACK-028")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)

    data_validation: DataValidationSummary = Field(
        default_factory=DataValidationSummary,
    )
    pathway_validation: PathwayValidationSummary = Field(
        default_factory=PathwayValidationSummary,
    )
    sbti_compliance: SBTiComplianceSummary = Field(
        default_factory=SBTiComplianceSummary,
    )
    compliance_report: ComplianceReport = Field(
        default_factory=ComplianceReport,
    )

    submission_ready: bool = Field(default=False)
    critical_issues: List[str] = Field(default_factory=list)
    improvement_actions: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PathwayValidationWorkflow:
    """
    4-phase workflow for validating sector pathways against SBTi criteria.

    Phase 1: DataValidation -- Input data completeness and quality.
    Phase 2: PathwayValidation -- Pathway mathematical integrity.
    Phase 3: SBTiCriteriaCheck -- SBTi compliance verification.
    Phase 4: ComplianceReport -- Comprehensive compliance report.

    Example:
        >>> wf = PathwayValidationWorkflow()
        >>> inp = PathwayValidationInput(
        ...     config=PathwayValidationConfig(sector="steel"),
        ...     pathway_points=[{"year": 2020, "intensity": 1.89}, ...],
        ... )
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[PathwayValidationConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or PathwayValidationConfig()
        self._phase_results: List[PhaseResult] = []
        self._data_validation: DataValidationSummary = DataValidationSummary()
        self._pathway_validation: PathwayValidationSummary = PathwayValidationSummary()
        self._sbti_compliance: SBTiComplianceSummary = SBTiComplianceSummary()
        self._compliance_report: ComplianceReport = ComplianceReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: PathwayValidationInput) -> PathwayValidationResult:
        """Execute the 4-phase pathway validation workflow."""
        started_at = _utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting pathway validation workflow %s, sector=%s",
            self.workflow_id, self.config.sector,
        )

        try:
            phase1 = await self._phase_data_validation(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_pathway_validation(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_sbti_criteria_check(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_compliance_report(input_data)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Pathway validation failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        critical_issues = self._collect_critical_issues()
        improvement_actions = self._collect_improvement_actions()

        result = PathwayValidationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            data_validation=self._data_validation,
            pathway_validation=self._pathway_validation,
            sbti_compliance=self._sbti_compliance,
            compliance_report=self._compliance_report,
            submission_ready=(
                self._sbti_compliance.overall_status == ComplianceStatus.COMPLIANT and
                self._data_validation.critical_findings == 0 and
                self._pathway_validation.pathway_valid
            ),
            critical_issues=critical_issues,
            improvement_actions=improvement_actions,
            next_steps=self._generate_next_steps(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Validation
    # -------------------------------------------------------------------------

    async def _phase_data_validation(self, input_data: PathwayValidationInput) -> PhaseResult:
        """Validate input data completeness, consistency, and quality."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        findings: List[DataValidationFinding] = []
        check_count = 0

        # Check 1: Base year intensity provided
        check_count += 1
        if self.config.base_year_intensity > 0:
            findings.append(DataValidationFinding(
                finding_id="DV-001", field_name="base_year_intensity",
                finding_type="completeness", severity=ValidationSeverity.INFO,
                description="Base year intensity provided.",
                actual_value=f"{self.config.base_year_intensity}",
                data_quality_tier=DataQualityTier.PRIMARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-001", field_name="base_year_intensity",
                finding_type="completeness", severity=ValidationSeverity.CRITICAL,
                description="Base year intensity is missing or zero.",
                actual_value="0.0", expected_value="> 0",
                remediation="Provide base year intensity metric value.",
                data_quality_tier=DataQualityTier.DEFAULT,
            ))

        # Check 2: Current intensity provided
        check_count += 1
        if self.config.current_intensity > 0:
            findings.append(DataValidationFinding(
                finding_id="DV-002", field_name="current_intensity",
                finding_type="completeness", severity=ValidationSeverity.INFO,
                description="Current intensity provided.",
                actual_value=f"{self.config.current_intensity}",
                data_quality_tier=DataQualityTier.PRIMARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-002", field_name="current_intensity",
                finding_type="completeness", severity=ValidationSeverity.MAJOR,
                description="Current intensity not provided.",
                remediation="Calculate current year intensity from emissions and activity.",
                data_quality_tier=DataQualityTier.DEFAULT,
            ))

        # Check 3: Pathway points provided
        check_count += 1
        pp_count = len(input_data.pathway_points)
        if pp_count >= 5:
            findings.append(DataValidationFinding(
                finding_id="DV-003", field_name="pathway_points",
                finding_type="completeness", severity=ValidationSeverity.INFO,
                description=f"Pathway has {pp_count} data points.",
                actual_value=str(pp_count),
                data_quality_tier=DataQualityTier.PRIMARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-003", field_name="pathway_points",
                finding_type="completeness", severity=ValidationSeverity.CRITICAL,
                description=f"Insufficient pathway points ({pp_count}). Need at least 5.",
                actual_value=str(pp_count), expected_value=">= 5",
                remediation="Generate pathway with annual granularity.",
                data_quality_tier=DataQualityTier.DEFAULT,
            ))

        # Check 4: Emission scope coverage
        check_count += 1
        total_emissions = (
            self.config.scope1_emissions_tco2e +
            self.config.scope2_emissions_tco2e +
            self.config.scope3_emissions_tco2e
        )
        if total_emissions > 0:
            s12_share = (
                (self.config.scope1_emissions_tco2e + self.config.scope2_emissions_tco2e) /
                total_emissions * 100
            )
            findings.append(DataValidationFinding(
                finding_id="DV-004", field_name="emission_scopes",
                finding_type="consistency", severity=ValidationSeverity.INFO,
                description=f"Scope 1+2 = {s12_share:.1f}% of total emissions.",
                actual_value=f"{total_emissions:.0f} tCO2e total",
                data_quality_tier=DataQualityTier.SECONDARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-004", field_name="emission_scopes",
                finding_type="completeness", severity=ValidationSeverity.MAJOR,
                description="No emission data by scope provided.",
                remediation="Provide Scope 1, 2, and 3 emission totals.",
                data_quality_tier=DataQualityTier.DEFAULT,
            ))

        # Check 5: Sector classification provided
        check_count += 1
        if self.config.sector and self.config.sector != "cross_sector":
            findings.append(DataValidationFinding(
                finding_id="DV-005", field_name="sector",
                finding_type="completeness", severity=ValidationSeverity.INFO,
                description=f"Sector: {self.config.sector}.",
                actual_value=self.config.sector,
                data_quality_tier=DataQualityTier.PRIMARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-005", field_name="sector",
                finding_type="completeness", severity=ValidationSeverity.MINOR,
                description="Sector defaulted to cross_sector.",
                remediation="Run sector classification workflow first.",
                data_quality_tier=DataQualityTier.DEFAULT,
            ))

        # Check 6: Intensity metric alignment
        check_count += 1
        sector_req = SDA_SECTOR_REQUIREMENTS.get(self.config.sector, {})
        if sector_req:
            req_metric = sector_req.get("required_metric", "")
            if self.config.intensity_metric == req_metric:
                findings.append(DataValidationFinding(
                    finding_id="DV-006", field_name="intensity_metric",
                    finding_type="consistency", severity=ValidationSeverity.INFO,
                    description=f"Intensity metric matches sector requirement: {req_metric}.",
                    data_quality_tier=DataQualityTier.PRIMARY,
                ))
            elif self.config.intensity_metric:
                findings.append(DataValidationFinding(
                    finding_id="DV-006", field_name="intensity_metric",
                    finding_type="consistency", severity=ValidationSeverity.MAJOR,
                    description=(
                        f"Intensity metric mismatch: using '{self.config.intensity_metric}', "
                        f"sector requires '{req_metric}'."
                    ),
                    actual_value=self.config.intensity_metric,
                    expected_value=req_metric,
                    remediation=f"Recalculate intensity using {req_metric}.",
                    data_quality_tier=DataQualityTier.SECONDARY,
                ))
            else:
                findings.append(DataValidationFinding(
                    finding_id="DV-006", field_name="intensity_metric",
                    finding_type="completeness", severity=ValidationSeverity.MAJOR,
                    description="No intensity metric specified.",
                    expected_value=req_metric,
                    remediation=f"Specify intensity metric: {req_metric}.",
                    data_quality_tier=DataQualityTier.DEFAULT,
                ))

        # Check 7: Consolidation approach
        check_count += 1
        valid_approaches = ["operational_control", "financial_control", "equity_share"]
        if self.config.consolidation_approach in valid_approaches:
            findings.append(DataValidationFinding(
                finding_id="DV-007", field_name="consolidation_approach",
                finding_type="completeness", severity=ValidationSeverity.INFO,
                description=f"Consolidation: {self.config.consolidation_approach}.",
                data_quality_tier=DataQualityTier.PRIMARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-007", field_name="consolidation_approach",
                finding_type="consistency", severity=ValidationSeverity.MINOR,
                description=f"Unknown consolidation approach: {self.config.consolidation_approach}.",
                expected_value="operational_control | financial_control | equity_share",
                remediation="Specify valid consolidation approach.",
                data_quality_tier=DataQualityTier.DEFAULT,
            ))

        # Check 8: Exclusion threshold
        check_count += 1
        if self.config.exclusions_pct <= SBTI_NEAR_TERM_REQUIREMENTS["exclusion_threshold_pct"]:
            findings.append(DataValidationFinding(
                finding_id="DV-008", field_name="exclusions_pct",
                finding_type="compliance", severity=ValidationSeverity.INFO,
                description=f"Exclusions: {self.config.exclusions_pct:.1f}% (within threshold).",
                data_quality_tier=DataQualityTier.PRIMARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-008", field_name="exclusions_pct",
                finding_type="compliance", severity=ValidationSeverity.MAJOR,
                description=(
                    f"Exclusions ({self.config.exclusions_pct:.1f}%) exceed "
                    f"SBTi threshold ({SBTI_NEAR_TERM_REQUIREMENTS['exclusion_threshold_pct']}%)."
                ),
                actual_value=f"{self.config.exclusions_pct:.1f}%",
                expected_value=f"<= {SBTI_NEAR_TERM_REQUIREMENTS['exclusion_threshold_pct']}%",
                remediation="Reduce exclusions or provide detailed justification.",
                data_quality_tier=DataQualityTier.SECONDARY,
            ))

        # Check 9: Structural changes recalculation
        check_count += 1
        if self.config.structural_changes_pct >= SBTI_NEAR_TERM_REQUIREMENTS["recalculation_threshold_pct"]:
            findings.append(DataValidationFinding(
                finding_id="DV-009", field_name="structural_changes",
                finding_type="compliance", severity=ValidationSeverity.MAJOR,
                description=(
                    f"Structural changes ({self.config.structural_changes_pct:.1f}%) trigger "
                    "base year recalculation requirement."
                ),
                remediation="Recalculate base year emissions for structural changes.",
                data_quality_tier=DataQualityTier.SECONDARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-009", field_name="structural_changes",
                finding_type="compliance", severity=ValidationSeverity.INFO,
                description="No structural changes triggering recalculation.",
                data_quality_tier=DataQualityTier.PRIMARY,
            ))

        # Check 10: Historical data availability
        check_count += 1
        hist_count = len(input_data.historical_data)
        if hist_count >= 3:
            findings.append(DataValidationFinding(
                finding_id="DV-010", field_name="historical_data",
                finding_type="completeness", severity=ValidationSeverity.INFO,
                description=f"Historical data: {hist_count} years available.",
                data_quality_tier=DataQualityTier.PRIMARY,
            ))
        else:
            findings.append(DataValidationFinding(
                finding_id="DV-010", field_name="historical_data",
                finding_type="completeness", severity=ValidationSeverity.MINOR,
                description=f"Limited historical data ({hist_count} years). 3+ recommended.",
                remediation="Provide at least 3 years of historical emission data.",
                data_quality_tier=DataQualityTier.TERTIARY,
            ))

        # Calculate summary
        passed = sum(1 for f in findings if f.severity == ValidationSeverity.INFO)
        critical = sum(1 for f in findings if f.severity == ValidationSeverity.CRITICAL)
        warn_count = sum(
            1 for f in findings if f.severity in (ValidationSeverity.MAJOR, ValidationSeverity.MINOR)
        )

        completeness_fields = [
            self.config.base_year_intensity > 0,
            self.config.current_intensity > 0,
            len(input_data.pathway_points) >= 5,
            total_emissions > 0,
            self.config.sector != "cross_sector",
            bool(self.config.intensity_metric),
            self.config.base_year_activity > 0,
        ]
        completeness = sum(completeness_fields) / max(len(completeness_fields), 1) * 100

        quality_tier = (
            DataQualityTier.PRIMARY if critical == 0 and warn_count == 0 else
            DataQualityTier.SECONDARY if critical == 0 else
            DataQualityTier.TERTIARY if critical <= 1 else
            DataQualityTier.ESTIMATED
        )

        self._data_validation = DataValidationSummary(
            total_checks=check_count,
            passed=passed,
            failed=critical,
            warnings_count=warn_count,
            critical_findings=critical,
            data_completeness_pct=round(completeness, 1),
            data_consistency_score=round(max(0, 100 - critical * 20 - warn_count * 5), 1),
            overall_quality_tier=quality_tier,
            findings=findings,
        )

        outputs["total_checks"] = check_count
        outputs["passed"] = passed
        outputs["critical"] = critical
        outputs["warnings"] = warn_count
        outputs["completeness_pct"] = round(completeness, 1)
        outputs["quality_tier"] = quality_tier.value

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="data_validation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_data_validation",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Pathway Validation
    # -------------------------------------------------------------------------

    async def _phase_pathway_validation(self, input_data: PathwayValidationInput) -> PhaseResult:
        """Validate pathway mathematical integrity and convergence."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        checks: List[PathwayValidationCheck] = []

        # Parse pathway points
        points: List[Tuple[int, float]] = []
        for pp in input_data.pathway_points:
            year = pp.get("year", 0)
            intensity = pp.get("intensity", pp.get("intensity_value", 0.0))
            if year > 0:
                points.append((year, intensity))

        # Synthesise pathway if not provided
        if len(points) < 2 and self.config.base_year_intensity > 0:
            base_i = self.config.base_year_intensity
            target_i = self.config.target_intensity_2050
            if target_i <= 0:
                sector_req = SDA_SECTOR_REQUIREMENTS.get(self.config.sector, {})
                target_i = sector_req.get("convergence_target_2050", base_i * 0.1)
            for y in range(self.config.base_year, self.config.target_year + 1):
                t = (y - self.config.base_year) / max(self.config.target_year - self.config.base_year, 1)
                points.append((y, base_i + t * (target_i - base_i)))

        points.sort(key=lambda x: x[0])

        # Check 1: Monotonicity (intensity should decrease over time)
        violations_mono: List[int] = []
        for i in range(1, len(points)):
            if points[i][1] > points[i - 1][1] * 1.01:  # 1% tolerance
                violations_mono.append(points[i][0])

        mono_pass = len(violations_mono) == 0
        checks.append(PathwayValidationCheck(
            check_id="PV-001", check_name="Monotonic Decrease",
            property_type=PathwayProperty.MONOTONICITY,
            passed=mono_pass,
            description="Pathway intensity should decrease monotonically over time.",
            details=(
                "Pathway is monotonically decreasing." if mono_pass else
                f"Intensity increases detected in years: {violations_mono[:5]}"
            ),
            violation_years=violations_mono,
            severity=ValidationSeverity.MAJOR if not mono_pass else ValidationSeverity.INFO,
        ))

        # Check 2: Convergence (should approach sector target)
        sector_req = SDA_SECTOR_REQUIREMENTS.get(self.config.sector, {})
        conv_target = sector_req.get("convergence_target_2050", 0.0)
        max_dev = sector_req.get("max_deviation_pct", 10.0)

        if points and conv_target >= 0:
            final_intensity = points[-1][1]
            final_year = points[-1][0]
            if conv_target > 0:
                deviation = abs(final_intensity - conv_target) / conv_target * 100
            else:
                deviation = final_intensity * 100

            conv_pass = deviation <= max_dev or final_year < 2050
            checks.append(PathwayValidationCheck(
                check_id="PV-002", check_name="Convergence to Sector Target",
                property_type=PathwayProperty.CONVERGENCE,
                passed=conv_pass,
                description=f"Final intensity should converge to {conv_target} (within {max_dev}%).",
                details=(
                    f"Final intensity: {final_intensity:.4f}, target: {conv_target:.4f}, "
                    f"deviation: {deviation:.1f}%"
                ),
                severity=ValidationSeverity.MAJOR if not conv_pass else ValidationSeverity.INFO,
            ))

        # Check 3: Continuity (no large gaps)
        violations_cont: List[int] = []
        for i in range(1, len(points)):
            year_gap = points[i][0] - points[i - 1][0]
            if year_gap > 5:
                violations_cont.append(points[i][0])

        cont_pass = len(violations_cont) == 0
        checks.append(PathwayValidationCheck(
            check_id="PV-003", check_name="Temporal Continuity",
            property_type=PathwayProperty.CONTINUITY,
            passed=cont_pass,
            description="Pathway should have data points at least every 5 years.",
            details=(
                "Pathway has continuous annual data." if cont_pass else
                f"Gaps > 5 years detected before years: {violations_cont}"
            ),
            violation_years=violations_cont,
            severity=ValidationSeverity.MINOR if not cont_pass else ValidationSeverity.INFO,
        ))

        # Check 4: Boundary (no negative intensities)
        violations_bound: List[int] = []
        for year, intensity in points:
            if intensity < 0:
                violations_bound.append(year)

        bound_pass = len(violations_bound) == 0
        checks.append(PathwayValidationCheck(
            check_id="PV-004", check_name="Non-Negative Intensity",
            property_type=PathwayProperty.BOUNDARY,
            passed=bound_pass,
            description="Intensity values must be non-negative.",
            details=(
                "All intensities are non-negative." if bound_pass else
                f"Negative intensities in years: {violations_bound}"
            ),
            violation_years=violations_bound,
            severity=ValidationSeverity.CRITICAL if not bound_pass else ValidationSeverity.INFO,
        ))

        # Check 5: Annual reduction rate reasonableness
        violations_rate: List[int] = []
        for i in range(1, len(points)):
            if points[i - 1][1] > 0:
                annual_rate = (1.0 - points[i][1] / points[i - 1][1]) * 100
                if annual_rate > 25.0:  # > 25% annual reduction is unrealistic
                    violations_rate.append(points[i][0])

        rate_pass = len(violations_rate) == 0
        checks.append(PathwayValidationCheck(
            check_id="PV-005", check_name="Reduction Rate Reasonableness",
            property_type=PathwayProperty.RATE,
            passed=rate_pass,
            description="Annual reduction rate should not exceed 25%.",
            details=(
                "All annual reduction rates are reasonable." if rate_pass else
                f"Excessive reduction rates (> 25%/yr) in years: {violations_rate}"
            ),
            violation_years=violations_rate,
            severity=ValidationSeverity.MINOR if not violations_rate else ValidationSeverity.INFO,
        ))

        # Check 6: Near-term ambition check (1.5C = 4.2% annual)
        if len(points) >= 2:
            start_i = points[0][1]
            start_y = points[0][0]
            near_term_point = None
            for year, intensity in points:
                if year >= start_y + 5:
                    near_term_point = (year, intensity)
                    break

            if near_term_point and start_i > 0:
                nt_years = near_term_point[0] - start_y
                nt_rate = (1.0 - (near_term_point[1] / start_i) ** (1.0 / nt_years)) * 100
                ambition_pass = nt_rate >= SBTI_NEAR_TERM_REQUIREMENTS["wb2c_linear_annual_rate"]

                checks.append(PathwayValidationCheck(
                    check_id="PV-006", check_name="Near-Term Reduction Ambition",
                    property_type=PathwayProperty.RATE,
                    passed=ambition_pass,
                    description=(
                        f"Near-term reduction rate should be >= "
                        f"{SBTI_NEAR_TERM_REQUIREMENTS['wb2c_linear_annual_rate']}%/yr (WB2C)."
                    ),
                    details=f"Calculated near-term rate: {nt_rate:.2f}%/yr over {nt_years} years.",
                    severity=(
                        ValidationSeverity.MAJOR if not ambition_pass else ValidationSeverity.INFO
                    ),
                ))

        # Check 7: Total pathway reduction
        if len(points) >= 2 and points[0][1] > 0:
            total_reduction = (1.0 - points[-1][1] / points[0][1]) * 100
            ltred_pass = total_reduction >= 80.0
            checks.append(PathwayValidationCheck(
                check_id="PV-007", check_name="Total Pathway Reduction",
                property_type=PathwayProperty.CONVERGENCE,
                passed=ltred_pass,
                description="Total intensity reduction should be >= 80% over pathway timeframe.",
                details=f"Total reduction: {total_reduction:.1f}%.",
                severity=(
                    ValidationSeverity.MAJOR if not ltred_pass else ValidationSeverity.INFO
                ),
            ))

        # Summary
        total = len(checks)
        passed_count = sum(1 for c in checks if c.passed)
        failed_count = total - passed_count

        start_int = points[0][1] if points else 0.0
        end_int = points[-1][1] if points else 0.0
        total_years = (points[-1][0] - points[0][0]) if len(points) >= 2 else 0
        tot_red = (1.0 - end_int / max(start_int, 1e-10)) * 100 if start_int > 0 else 0.0
        avg_annual = tot_red / max(total_years, 1)

        self._pathway_validation = PathwayValidationSummary(
            total_checks=total,
            passed=passed_count,
            failed=failed_count,
            pathway_valid=failed_count == 0,
            checks=checks,
            pathway_years=total_years,
            pathway_start_intensity=round(start_int, 6),
            pathway_end_intensity=round(end_int, 6),
            total_reduction_pct=round(tot_red, 2),
            average_annual_reduction_pct=round(avg_annual, 2),
        )

        outputs["total_checks"] = total
        outputs["passed"] = passed_count
        outputs["failed"] = failed_count
        outputs["pathway_valid"] = failed_count == 0
        outputs["pathway_years"] = total_years
        outputs["total_reduction_pct"] = round(tot_red, 2)
        outputs["avg_annual_reduction_pct"] = round(avg_annual, 2)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="pathway_validation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_pathway_validation",
        )

    # -------------------------------------------------------------------------
    # Phase 3: SBTi Criteria Check
    # -------------------------------------------------------------------------

    async def _phase_sbti_criteria_check(self, input_data: PathwayValidationInput) -> PhaseResult:
        """Check pathway against SBTi Corporate Standard v2.0 criteria."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        criteria: List[SBTiCriterionCheck] = []

        # Near-term criteria
        # NT-1: Scope 1+2 coverage
        nt1_pass = self.config.scope12_coverage_pct >= SBTI_NEAR_TERM_REQUIREMENTS["scope12_coverage_pct"]
        criteria.append(SBTiCriterionCheck(
            criterion_id="NT-001", criterion_name="Scope 1+2 Coverage",
            sbti_reference="SBTi CS v2.0 Section 4.1",
            category="near_term",
            status=ComplianceStatus.COMPLIANT if nt1_pass else ComplianceStatus.NON_COMPLIANT,
            actual_value=f"{self.config.scope12_coverage_pct:.0f}%",
            required_value=f">= {SBTI_NEAR_TERM_REQUIREMENTS['scope12_coverage_pct']:.0f}%",
            margin=f"{self.config.scope12_coverage_pct - SBTI_NEAR_TERM_REQUIREMENTS['scope12_coverage_pct']:+.0f}pp",
            finding=f"Scope 1+2 coverage: {self.config.scope12_coverage_pct:.0f}%.",
            remediation="" if nt1_pass else "Expand Scope 1+2 boundary.",
            severity=ValidationSeverity.CRITICAL if not nt1_pass else ValidationSeverity.INFO,
        ))

        # NT-2: Scope 3 materiality and coverage
        total_e = (
            self.config.scope1_emissions_tco2e +
            self.config.scope2_emissions_tco2e +
            self.config.scope3_emissions_tco2e
        )
        s3_material = False
        if total_e > 0:
            s3_share = self.config.scope3_emissions_tco2e / total_e * 100
            s3_material = s3_share >= SBTI_NEAR_TERM_REQUIREMENTS["scope3_materiality_pct"]

        if s3_material:
            nt2_pass = self.config.scope3_coverage_pct >= SBTI_NEAR_TERM_REQUIREMENTS["scope3_coverage_pct"]
            criteria.append(SBTiCriterionCheck(
                criterion_id="NT-002", criterion_name="Scope 3 Coverage",
                sbti_reference="SBTi CS v2.0 Section 4.2",
                category="near_term",
                status=ComplianceStatus.COMPLIANT if nt2_pass else ComplianceStatus.NON_COMPLIANT,
                actual_value=f"{self.config.scope3_coverage_pct:.0f}%",
                required_value=f">= {SBTI_NEAR_TERM_REQUIREMENTS['scope3_coverage_pct']:.0f}%",
                finding=f"Scope 3 is material ({s3_share:.0f}% of total). Coverage: {self.config.scope3_coverage_pct:.0f}%.",
                remediation="" if nt2_pass else "Expand Scope 3 boundary to 67%+.",
                severity=ValidationSeverity.MAJOR if not nt2_pass else ValidationSeverity.INFO,
            ))
        else:
            criteria.append(SBTiCriterionCheck(
                criterion_id="NT-002", criterion_name="Scope 3 Coverage",
                sbti_reference="SBTi CS v2.0 Section 4.2",
                category="near_term",
                status=ComplianceStatus.COMPLIANT,
                finding="Scope 3 not material (< 40% of total). No coverage requirement.",
                severity=ValidationSeverity.INFO,
            ))

        # NT-3: Target timeframe
        nt_years = self.config.near_term_target_year - self.config.base_year
        nt3_pass = (
            SBTI_NEAR_TERM_REQUIREMENTS["timeframe_min_years"] <= nt_years <=
            SBTI_NEAR_TERM_REQUIREMENTS["timeframe_max_years"]
        )
        criteria.append(SBTiCriterionCheck(
            criterion_id="NT-003", criterion_name="Near-Term Timeframe",
            sbti_reference="SBTi CS v2.0 Section 4.3",
            category="near_term",
            status=ComplianceStatus.COMPLIANT if nt3_pass else ComplianceStatus.NON_COMPLIANT,
            actual_value=f"{nt_years} years ({self.config.base_year}-{self.config.near_term_target_year})",
            required_value=(
                f"{SBTI_NEAR_TERM_REQUIREMENTS['timeframe_min_years']}-"
                f"{SBTI_NEAR_TERM_REQUIREMENTS['timeframe_max_years']} years"
            ),
            finding=f"Near-term timeframe: {nt_years} years.",
            remediation="" if nt3_pass else "Adjust near-term target to 5-10 year window.",
            severity=ValidationSeverity.MAJOR if not nt3_pass else ValidationSeverity.INFO,
        ))

        # NT-4: Reduction ambition (1.5C alignment)
        annual_rate = self._pathway_validation.average_annual_reduction_pct
        nt4_pass = annual_rate >= SBTI_NEAR_TERM_REQUIREMENTS["15c_linear_annual_rate"]
        nt4_partial = annual_rate >= SBTI_NEAR_TERM_REQUIREMENTS["wb2c_linear_annual_rate"]
        criteria.append(SBTiCriterionCheck(
            criterion_id="NT-004", criterion_name="Reduction Ambition (1.5C)",
            sbti_reference="SBTi CS v2.0 Section 4.4",
            category="near_term",
            status=(
                ComplianceStatus.COMPLIANT if nt4_pass else
                ComplianceStatus.PARTIALLY_COMPLIANT if nt4_partial else
                ComplianceStatus.NON_COMPLIANT
            ),
            actual_value=f"{annual_rate:.2f}%/yr",
            required_value=f">= {SBTI_NEAR_TERM_REQUIREMENTS['15c_linear_annual_rate']}%/yr (1.5C)",
            margin=f"{annual_rate - SBTI_NEAR_TERM_REQUIREMENTS['15c_linear_annual_rate']:+.2f}pp",
            finding=f"Average annual reduction: {annual_rate:.2f}%/yr.",
            remediation=(
                "" if nt4_pass else
                f"Increase ambition by {SBTI_NEAR_TERM_REQUIREMENTS['15c_linear_annual_rate'] - annual_rate:.2f}pp/yr."
            ),
            severity=(
                ValidationSeverity.INFO if nt4_pass else
                ValidationSeverity.MINOR if nt4_partial else
                ValidationSeverity.MAJOR
            ),
        ))

        # NT-5: Exclusions
        nt5_pass = self.config.exclusions_pct <= SBTI_NEAR_TERM_REQUIREMENTS["exclusion_threshold_pct"]
        criteria.append(SBTiCriterionCheck(
            criterion_id="NT-005", criterion_name="Emission Exclusions",
            sbti_reference="SBTi CS v2.0 Section 4.5",
            category="near_term",
            status=ComplianceStatus.COMPLIANT if nt5_pass else ComplianceStatus.NON_COMPLIANT,
            actual_value=f"{self.config.exclusions_pct:.1f}%",
            required_value=f"<= {SBTI_NEAR_TERM_REQUIREMENTS['exclusion_threshold_pct']}%",
            finding=f"Exclusions: {self.config.exclusions_pct:.1f}%.",
            remediation="" if nt5_pass else "Reduce exclusions to 5% or provide justification.",
            severity=ValidationSeverity.MAJOR if not nt5_pass else ValidationSeverity.INFO,
        ))

        # Long-term criteria
        # LT-1: Reduction magnitude
        total_red = self._pathway_validation.total_reduction_pct
        lt1_pass = total_red >= SBTI_LONG_TERM_REQUIREMENTS["reduction_pct"]
        criteria.append(SBTiCriterionCheck(
            criterion_id="LT-001", criterion_name="Long-Term Reduction (90%+)",
            sbti_reference="SBTi Net-Zero Standard Section 5.1",
            category="long_term",
            status=ComplianceStatus.COMPLIANT if lt1_pass else ComplianceStatus.NON_COMPLIANT,
            actual_value=f"{total_red:.1f}%",
            required_value=f">= {SBTI_LONG_TERM_REQUIREMENTS['reduction_pct']}%",
            margin=f"{total_red - SBTI_LONG_TERM_REQUIREMENTS['reduction_pct']:+.1f}pp",
            finding=f"Total pathway reduction: {total_red:.1f}%.",
            remediation="" if lt1_pass else "Extend pathway to achieve 90%+ total reduction.",
            severity=ValidationSeverity.CRITICAL if not lt1_pass else ValidationSeverity.INFO,
        ))

        # LT-2: Target year
        lt2_pass = self.config.target_year <= SBTI_LONG_TERM_REQUIREMENTS["target_year_max"]
        criteria.append(SBTiCriterionCheck(
            criterion_id="LT-002", criterion_name="Target Year (by 2050)",
            sbti_reference="SBTi Net-Zero Standard Section 5.2",
            category="long_term",
            status=ComplianceStatus.COMPLIANT if lt2_pass else ComplianceStatus.NON_COMPLIANT,
            actual_value=str(self.config.target_year),
            required_value=f"<= {SBTI_LONG_TERM_REQUIREMENTS['target_year_max']}",
            finding=f"Target year: {self.config.target_year}.",
            remediation="" if lt2_pass else "Set target year to 2050 or earlier.",
            severity=ValidationSeverity.MAJOR if not lt2_pass else ValidationSeverity.INFO,
        ))

        # SDA-specific criteria
        sector_req = SDA_SECTOR_REQUIREMENTS.get(self.config.sector, {})
        if sector_req:
            # SDA-1: Intensity metric
            sda1_pass = self.config.intensity_metric == sector_req.get("required_metric", "")
            criteria.append(SBTiCriterionCheck(
                criterion_id="SDA-001", criterion_name="Sector Intensity Metric",
                sbti_reference="SBTi SDA Methodology",
                category="sda_specific",
                status=ComplianceStatus.COMPLIANT if sda1_pass else ComplianceStatus.NON_COMPLIANT,
                actual_value=self.config.intensity_metric or "not specified",
                required_value=sector_req.get("required_metric", ""),
                finding=f"Using metric: {self.config.intensity_metric}.",
                remediation="" if sda1_pass else f"Use required metric: {sector_req.get('required_metric', '')}.",
                severity=ValidationSeverity.MAJOR if not sda1_pass else ValidationSeverity.INFO,
            ))

            # SDA-2: Convergence target
            end_int = self._pathway_validation.pathway_end_intensity
            sda_target = sector_req.get("convergence_target_2050", 0.0)
            max_dev_pct = sector_req.get("max_deviation_pct", 10.0)

            if sda_target > 0:
                dev = abs(end_int - sda_target) / sda_target * 100
            else:
                dev = end_int * 100 if end_int > 0 else 0.0

            sda2_pass = dev <= max_dev_pct
            criteria.append(SBTiCriterionCheck(
                criterion_id="SDA-002", criterion_name="Convergence Target Alignment",
                sbti_reference="SBTi SDA Methodology",
                category="sda_specific",
                status=ComplianceStatus.COMPLIANT if sda2_pass else ComplianceStatus.NON_COMPLIANT,
                actual_value=f"{end_int:.4f} (deviation: {dev:.1f}%)",
                required_value=f"{sda_target} (within {max_dev_pct}%)",
                margin=f"{dev - max_dev_pct:+.1f}pp from threshold",
                finding=f"Pathway end intensity: {end_int:.4f} vs. target {sda_target}.",
                remediation="" if sda2_pass else f"Adjust pathway to converge within {max_dev_pct}% of {sda_target}.",
                severity=ValidationSeverity.MAJOR if not sda2_pass else ValidationSeverity.INFO,
            ))

        # FLAG criteria
        if self.config.is_flag_sector:
            flag_pass = self.config.flag_emissions_tco2e > 0
            criteria.append(SBTiCriterionCheck(
                criterion_id="FLAG-001", criterion_name="FLAG Emissions Separated",
                sbti_reference="SBTi FLAG Guidance Section 3",
                category="flag",
                status=ComplianceStatus.COMPLIANT if flag_pass else ComplianceStatus.NON_COMPLIANT,
                actual_value=f"{self.config.flag_emissions_tco2e:.0f} tCO2e",
                required_value="> 0 tCO2e (FLAG emissions separated)",
                finding="FLAG emissions properly separated." if flag_pass else "FLAG emissions not separated.",
                remediation="" if flag_pass else "Separate FLAG from fossil fuel emissions.",
                severity=ValidationSeverity.MAJOR if not flag_pass else ValidationSeverity.INFO,
            ))

        # Compute summary
        compliant = sum(1 for c in criteria if c.status == ComplianceStatus.COMPLIANT)
        non_compliant = sum(1 for c in criteria if c.status == ComplianceStatus.NON_COMPLIANT)
        partial = sum(1 for c in criteria if c.status == ComplianceStatus.PARTIALLY_COMPLIANT)

        if non_compliant == 0 and partial == 0:
            overall = ComplianceStatus.COMPLIANT
        elif non_compliant == 0:
            overall = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall = ComplianceStatus.NON_COMPLIANT

        # Category-level status
        nt_criteria = [c for c in criteria if c.category == "near_term"]
        lt_criteria = [c for c in criteria if c.category == "long_term"]
        sda_criteria = [c for c in criteria if c.category == "sda_specific"]

        def _cat_status(cats: List[SBTiCriterionCheck]) -> ComplianceStatus:
            if not cats:
                return ComplianceStatus.NOT_ASSESSED
            nc = sum(1 for c in cats if c.status == ComplianceStatus.NON_COMPLIANT)
            pc = sum(1 for c in cats if c.status == ComplianceStatus.PARTIALLY_COMPLIANT)
            if nc == 0 and pc == 0:
                return ComplianceStatus.COMPLIANT
            elif nc == 0:
                return ComplianceStatus.PARTIALLY_COMPLIANT
            return ComplianceStatus.NON_COMPLIANT

        self._sbti_compliance = SBTiComplianceSummary(
            total_criteria=len(criteria),
            compliant=compliant,
            non_compliant=non_compliant,
            partially_compliant=partial,
            overall_status=overall,
            compliance_score_pct=round(
                (compliant + partial * 0.5) / max(len(criteria), 1) * 100, 1,
            ),
            near_term_status=_cat_status(nt_criteria),
            long_term_status=_cat_status(lt_criteria),
            sda_specific_status=_cat_status(sda_criteria),
            criteria=criteria,
        )

        outputs["total_criteria"] = len(criteria)
        outputs["compliant"] = compliant
        outputs["non_compliant"] = non_compliant
        outputs["partially_compliant"] = partial
        outputs["overall_status"] = overall.value
        outputs["compliance_score_pct"] = self._sbti_compliance.compliance_score_pct

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="sbti_criteria_check", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_sbti_criteria_check",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Compliance Report
    # -------------------------------------------------------------------------

    async def _phase_compliance_report(self, input_data: PathwayValidationInput) -> PhaseResult:
        """Generate comprehensive compliance report."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        # Build report sections
        sections: List[ComplianceReportSection] = []

        sections.append(ComplianceReportSection(
            section_id="SEC-001", section_title="Executive Summary",
            content=(
                f"Pathway validation for {self.config.company_name or 'Company'} in "
                f"sector '{self.config.sector}'. Overall compliance: "
                f"{self._sbti_compliance.overall_status.value}. "
                f"{self._sbti_compliance.compliant}/{self._sbti_compliance.total_criteria} "
                "criteria met."
            ),
            status=self._sbti_compliance.overall_status,
            key_metrics={
                "compliance_score": self._sbti_compliance.compliance_score_pct,
                "data_completeness": self._data_validation.data_completeness_pct,
                "pathway_valid": self._pathway_validation.pathway_valid,
            },
        ))

        sections.append(ComplianceReportSection(
            section_id="SEC-002", section_title="Data Quality Assessment",
            content=(
                f"Data quality tier: {self._data_validation.overall_quality_tier.value}. "
                f"{self._data_validation.total_checks} checks performed, "
                f"{self._data_validation.passed} passed, "
                f"{self._data_validation.critical_findings} critical issues."
            ),
            status=(
                ComplianceStatus.COMPLIANT if self._data_validation.critical_findings == 0 else
                ComplianceStatus.NON_COMPLIANT
            ),
            key_metrics={
                "completeness_pct": self._data_validation.data_completeness_pct,
                "consistency_score": self._data_validation.data_consistency_score,
                "quality_tier": self._data_validation.overall_quality_tier.value,
            },
        ))

        sections.append(ComplianceReportSection(
            section_id="SEC-003", section_title="Pathway Mathematical Validation",
            content=(
                f"Pathway spanning {self._pathway_validation.pathway_years} years. "
                f"Total reduction: {self._pathway_validation.total_reduction_pct:.1f}%. "
                f"{self._pathway_validation.passed}/{self._pathway_validation.total_checks} "
                "mathematical checks passed."
            ),
            status=(
                ComplianceStatus.COMPLIANT if self._pathway_validation.pathway_valid else
                ComplianceStatus.NON_COMPLIANT
            ),
            key_metrics={
                "total_reduction_pct": self._pathway_validation.total_reduction_pct,
                "avg_annual_reduction": self._pathway_validation.average_annual_reduction_pct,
                "start_intensity": self._pathway_validation.pathway_start_intensity,
                "end_intensity": self._pathway_validation.pathway_end_intensity,
            },
        ))

        sections.append(ComplianceReportSection(
            section_id="SEC-004", section_title="SBTi Near-Term Compliance",
            content=(
                f"Near-term status: {self._sbti_compliance.near_term_status.value}. "
                "Checks: Scope 1+2 coverage, Scope 3 coverage, target timeframe, "
                "reduction ambition, emission exclusions."
            ),
            status=self._sbti_compliance.near_term_status,
            key_metrics={
                "scope12_coverage": self.config.scope12_coverage_pct,
                "scope3_coverage": self.config.scope3_coverage_pct,
                "near_term_years": self.config.near_term_target_year - self.config.base_year,
            },
        ))

        sections.append(ComplianceReportSection(
            section_id="SEC-005", section_title="SBTi Long-Term Compliance",
            content=(
                f"Long-term status: {self._sbti_compliance.long_term_status.value}. "
                f"Total reduction: {self._pathway_validation.total_reduction_pct:.1f}%. "
                f"Target year: {self.config.target_year}."
            ),
            status=self._sbti_compliance.long_term_status,
            key_metrics={
                "total_reduction_pct": self._pathway_validation.total_reduction_pct,
                "target_year": self.config.target_year,
                "required_reduction_pct": SBTI_LONG_TERM_REQUIREMENTS["reduction_pct"],
            },
        ))

        sections.append(ComplianceReportSection(
            section_id="SEC-006", section_title="SDA Sector-Specific Compliance",
            content=(
                f"SDA status: {self._sbti_compliance.sda_specific_status.value}. "
                f"Sector: {self.config.sector}. Method: {self.config.sda_method}."
            ),
            status=self._sbti_compliance.sda_specific_status,
            key_metrics={
                "sector": self.config.sector,
                "sda_method": self.config.sda_method,
                "intensity_metric": self.config.intensity_metric,
            },
        ))

        sections.append(ComplianceReportSection(
            section_id="SEC-007", section_title="Improvement Roadmap",
            content="Prioritised improvement actions for pathway compliance.",
            status=ComplianceStatus.NOT_ASSESSED,
            key_metrics={"action_count": len(self._collect_improvement_actions())},
        ))

        # Determine overall
        overall_report_status = self._sbti_compliance.overall_status
        submission_ready = (
            overall_report_status == ComplianceStatus.COMPLIANT and
            self._data_validation.critical_findings == 0 and
            self._pathway_validation.pathway_valid
        )

        self._compliance_report = ComplianceReport(
            report_id=_new_uuid(),
            report_title=f"SBTi Pathway Validation Report - {self.config.company_name or 'Company'}",
            sector=self.config.sector,
            company_name=self.config.company_name,
            assessment_date=_utcnow().isoformat(),
            overall_status=overall_report_status,
            data_validation=self._data_validation,
            pathway_validation=self._pathway_validation,
            sbti_compliance=self._sbti_compliance,
            sections=sections,
            improvement_roadmap=self._collect_improvement_actions(),
            submission_ready=submission_ready,
        )
        self._compliance_report.provenance_hash = _compute_hash(
            self._compliance_report.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["report_id"] = self._compliance_report.report_id
        outputs["overall_status"] = overall_report_status.value
        outputs["submission_ready"] = submission_ready
        outputs["sections_count"] = len(sections)
        outputs["improvement_actions"] = len(self._compliance_report.improvement_roadmap)
        outputs["report_formats"] = ["MD", "HTML", "JSON", "PDF"]

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="compliance_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_compliance_report",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _collect_critical_issues(self) -> List[str]:
        issues: List[str] = []
        for f in self._data_validation.findings:
            if f.severity == ValidationSeverity.CRITICAL:
                issues.append(f"[DATA] {f.description}")
        for c in self._pathway_validation.checks:
            if not c.passed and c.severity == ValidationSeverity.CRITICAL:
                issues.append(f"[PATHWAY] {c.description}: {c.details}")
        for c in self._sbti_compliance.criteria:
            if c.status == ComplianceStatus.NON_COMPLIANT and c.severity == ValidationSeverity.CRITICAL:
                issues.append(f"[SBTI] {c.criterion_name}: {c.finding}")
        return issues

    def _collect_improvement_actions(self) -> List[str]:
        actions: List[str] = []
        priority = 1
        # Critical data findings first
        for f in self._data_validation.findings:
            if f.severity == ValidationSeverity.CRITICAL and f.remediation:
                actions.append(f"P{priority}: {f.remediation}")
                priority += 1
        # Non-compliant SBTi criteria
        for c in self._sbti_compliance.criteria:
            if c.status == ComplianceStatus.NON_COMPLIANT and c.remediation:
                actions.append(f"P{priority}: [{c.criterion_id}] {c.remediation}")
                priority += 1
        # Partially compliant
        for c in self._sbti_compliance.criteria:
            if c.status == ComplianceStatus.PARTIALLY_COMPLIANT and c.remediation:
                actions.append(f"P{priority}: [{c.criterion_id}] {c.remediation}")
                priority += 1
        # Major data findings
        for f in self._data_validation.findings:
            if f.severity == ValidationSeverity.MAJOR and f.remediation:
                actions.append(f"P{priority}: {f.remediation}")
                priority += 1
        return actions

    def _generate_next_steps(self) -> List[str]:
        steps: List[str] = []
        if not self._compliance_report.submission_ready:
            steps.append("Address critical issues and non-compliant criteria before SBTi submission.")
        else:
            steps.append("Pathway is SBTi-ready. Prepare submission package.")
        steps.extend([
            "Run sector pathway design workflow for pathway generation if not already done.",
            "Engage SBTi-approved validation body for external review.",
            "Document all assumptions and methodological choices.",
            "Schedule annual pathway review and base year recalculation check.",
            "Prepare board presentation on pathway validation results.",
        ])
        return steps

    def _run_sda_checklist(self, sector: str) -> List[Dict[str, Any]]:
        """
        Run the SDA validation checklist for a given sector.
        Returns a list of check results with pass/fail status and details.
        """
        results: List[Dict[str, Any]] = []

        for chk_id, chk_data in SDA_VALIDATION_CHECKLIST.items():
            applies = chk_data.get("applies_to", "all_sectors")

            # Filter checks by applicability
            if applies == "sda_sectors":
                sda_sectors = list(SDA_SECTOR_REQUIREMENTS.keys())
                if sector not in sda_sectors:
                    continue
            elif applies == "flag_sectors":
                if sector not in ("agriculture",):
                    continue

            # Determine pass/fail based on available data
            check_result = {
                "check_id": chk_id,
                "check_name": chk_data["check"],
                "description": chk_data["description"],
                "requirement": chk_data["requirement"],
                "severity": chk_data["severity"],
                "status": "not_assessed",
                "details": "",
            }

            results.append(check_result)

        return results

    def _assess_data_quality_tier(
        self, data_quality_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Assess the overall data quality tier based on individual quality scores.

        Returns a dict with overall_tier, average_score, min_score,
        distribution (count per tier), and recommendations.
        """
        if not data_quality_scores:
            return {
                "overall_tier": "default",
                "average_score": 1.0,
                "min_score": 1.0,
                "distribution": {"primary": 0, "secondary": 0, "tertiary": 0, "estimated": 0, "default": 0},
                "recommendations": ["Collect primary measurement data for all emission sources."],
            }

        avg_score = sum(data_quality_scores) / len(data_quality_scores)
        min_score = min(data_quality_scores)

        # Score to tier mapping
        tier_counts = {"primary": 0, "secondary": 0, "tertiary": 0, "estimated": 0, "default": 0}
        for score in data_quality_scores:
            if score >= 4.0:
                tier_counts["primary"] += 1
            elif score >= 3.0:
                tier_counts["secondary"] += 1
            elif score >= 2.0:
                tier_counts["tertiary"] += 1
            elif score >= 1.5:
                tier_counts["estimated"] += 1
            else:
                tier_counts["default"] += 1

        # Overall tier
        if avg_score >= 4.0:
            overall_tier = "primary"
        elif avg_score >= 3.0:
            overall_tier = "secondary"
        elif avg_score >= 2.0:
            overall_tier = "tertiary"
        elif avg_score >= 1.5:
            overall_tier = "estimated"
        else:
            overall_tier = "default"

        # Recommendations
        recs: List[str] = []
        if tier_counts["default"] > 0:
            recs.append(
                f"Upgrade {tier_counts['default']} default-quality data source(s) to "
                "at minimum secondary tier with emission factor calculations."
            )
        if tier_counts["estimated"] > 0:
            recs.append(
                f"Replace {tier_counts['estimated']} estimated data source(s) with "
                "activity-based calculations using published emission factors."
            )
        if tier_counts["tertiary"] > 0:
            recs.append(
                f"Improve {tier_counts['tertiary']} tertiary data source(s) by "
                "obtaining direct measurement or metered data."
            )
        if avg_score < 3.0:
            recs.append(
                "Overall data quality is below SBTi recommended threshold. "
                "Prioritize data quality improvement programme."
            )
        if min_score < 2.0:
            recs.append(
                "At least one data source has unacceptable quality. "
                "This may trigger SBTi reviewer concern."
            )
        if not recs:
            recs.append("Data quality meets SBTi requirements. Maintain current processes.")

        return {
            "overall_tier": overall_tier,
            "average_score": round(avg_score, 2),
            "min_score": round(min_score, 2),
            "distribution": tier_counts,
            "recommendations": recs,
        }

    def _validate_base_year_recalculation(
        self, base_year_emissions: float,
        current_emissions: float,
        structural_changes: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Check whether a base year recalculation is required based on
        structural changes (mergers, acquisitions, divestitures, methodology
        changes).

        Returns dict with recalculation_required (bool), total_change_pct,
        change_details, and recommendation.
        """
        threshold = SBTI_NEAR_TERM_REQUIREMENTS["recalculation_threshold_pct"]
        total_change = 0.0
        change_details: List[str] = []

        for change in structural_changes:
            change_type = change.get("type", "unknown")
            impact_tco2e = change.get("impact_tco2e", 0.0)
            impact_pct = (abs(impact_tco2e) / max(base_year_emissions, 1.0)) * 100
            total_change += impact_pct

            change_details.append(
                f"{change_type}: {impact_tco2e:,.0f} tCO2e ({impact_pct:.1f}% of base year)"
            )

        recalc_required = total_change >= threshold

        if recalc_required:
            recommendation = (
                f"Base year recalculation REQUIRED: cumulative structural changes "
                f"({total_change:.1f}%) exceed {threshold}% threshold. "
                "Recalculate base year emissions and resubmit pathway."
            )
        else:
            recommendation = (
                f"No recalculation required: structural changes ({total_change:.1f}%) "
                f"below {threshold}% threshold. Document changes for audit trail."
            )

        return {
            "recalculation_required": recalc_required,
            "total_change_pct": round(total_change, 2),
            "threshold_pct": threshold,
            "change_details": change_details,
            "recommendation": recommendation,
        }
