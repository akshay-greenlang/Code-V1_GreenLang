# -*- coding: utf-8 -*-
"""
Disclosure Workflow
========================

4-phase workflow for generating framework-specific Scope 3 disclosures
within PACK-042 Scope 3 Starter Pack.

Phases:
    1. FrameworkSelection   -- Choose target frameworks (GHG Protocol,
                               ESRS E1, CDP, SBTi, SEC, SB 253)
    2. ComplianceMapping    -- Map inventory data to each framework's
                               Scope 3 requirements
    3. GapAnalysis          -- Identify missing requirements per framework,
                               flag compliance gaps
    4. DisclosureOutput     -- Generate framework-specific reports, data
                               exports, and XBRL tags

The workflow follows GreenLang zero-hallucination principles: all compliance
requirements and gap analysis use deterministic rule matching against
published framework standards. SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    EU CSRD / ESRS E1 (E1-6 para 51)
    CDP Climate Change Questionnaire (C6.5, C6.7)
    SBTi Corporate Net-Zero Standard v1.1
    US SEC Climate Disclosure Rules
    California SB 253

Schedule: on-demand (after consolidation)
Estimated duration: 2-4 hours

Author: GreenLang Platform Team
Version: 42.0.0
"""

_MODULE_VERSION: str = "42.0.0"

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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


class DisclosureFramework(str, Enum):
    """Supported disclosure frameworks."""

    GHG_PROTOCOL = "ghg_protocol"
    ESRS_E1 = "esrs_e1"
    CDP_CLIMATE = "cdp_climate"
    SBTI = "sbti"
    SEC_CLIMATE = "sec_climate"
    SB_253 = "sb_253"
    ISO_14064 = "iso_14064"
    TCFD = "tcfd"


class ComplianceStatus(str, Enum):
    """Compliance status for a framework requirement."""

    MET = "met"
    PARTIALLY_MET = "partially_met"
    NOT_MET = "not_met"
    NOT_APPLICABLE = "not_applicable"


class OutputFormat(str, Enum):
    """Output format types."""

    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON_EXPORT = "json"
    XBRL = "xbrl"
    XML = "xml"


class GapSeverity(str, Enum):
    """Severity of a compliance gap."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    ADVISORY = "advisory"


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


class FrameworkRequirement(BaseModel):
    """A single disclosure requirement for a framework."""

    requirement_id: str = Field(default="")
    framework: DisclosureFramework = Field(...)
    section: str = Field(default="", description="Framework section reference")
    description: str = Field(default="")
    data_field: str = Field(default="", description="Required data field")
    is_mandatory: bool = Field(default=True)
    compliance_status: ComplianceStatus = Field(default=ComplianceStatus.NOT_MET)
    data_available: bool = Field(default=False)
    notes: str = Field(default="")


class ComplianceScore(BaseModel):
    """Compliance score for a framework."""

    framework: DisclosureFramework = Field(...)
    framework_name: str = Field(default="")
    total_requirements: int = Field(default=0, ge=0)
    requirements_met: int = Field(default=0, ge=0)
    requirements_partially_met: int = Field(default=0, ge=0)
    requirements_not_met: int = Field(default=0, ge=0)
    compliance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    mandatory_met_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class ComplianceGap(BaseModel):
    """Identified compliance gap."""

    gap_id: str = Field(
        default_factory=lambda: f"gap-{uuid.uuid4().hex[:8]}"
    )
    framework: DisclosureFramework = Field(...)
    requirement_id: str = Field(default="")
    description: str = Field(default="")
    severity: GapSeverity = Field(default=GapSeverity.MAJOR)
    remediation: str = Field(default="")
    estimated_effort_hours: float = Field(default=0.0, ge=0.0)


class DisclosureDocument(BaseModel):
    """Generated disclosure document."""

    document_id: str = Field(
        default_factory=lambda: f"doc-{uuid.uuid4().hex[:8]}"
    )
    framework: DisclosureFramework = Field(...)
    title: str = Field(default="")
    format: OutputFormat = Field(default=OutputFormat.JSON_EXPORT)
    content_summary: str = Field(default="")
    sections: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class InventoryData(BaseModel):
    """Scope 3 inventory data for disclosure."""

    total_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    upstream_tco2e: float = Field(default=0.0, ge=0.0)
    downstream_tco2e: float = Field(default=0.0, ge=0.0)
    category_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Category -> tCO2e"
    )
    methodology_tiers: Dict[str, str] = Field(
        default_factory=dict, description="Category -> tier"
    )
    data_quality_scores: Dict[str, float] = Field(
        default_factory=dict, description="Category -> DQR 1-5"
    )
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025)
    base_year: int = Field(default=2020)
    organization_name: str = Field(default="")
    sector: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class DisclosureInput(BaseModel):
    """Input data model for DisclosureWorkflow."""

    inventory_data: InventoryData = Field(
        default_factory=InventoryData, description="Complete inventory data"
    )
    target_frameworks: List[DisclosureFramework] = Field(
        default_factory=lambda: [DisclosureFramework.GHG_PROTOCOL],
        description="Target disclosure frameworks",
    )
    output_formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.JSON_EXPORT, OutputFormat.EXCEL],
    )
    include_xbrl_tags: bool = Field(default=False)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class DisclosureOutput(BaseModel):
    """Complete result from disclosure workflow."""

    workflow_id: str = Field(...)
    workflow_name: str = Field(default="disclosure")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    selected_frameworks: List[str] = Field(default_factory=list)
    compliance_scores: List[ComplianceScore] = Field(default_factory=list)
    compliance_gaps: List[ComplianceGap] = Field(default_factory=list)
    framework_requirements: List[FrameworkRequirement] = Field(default_factory=list)
    disclosure_documents: List[DisclosureDocument] = Field(default_factory=list)
    overall_compliance_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# FRAMEWORK REQUIREMENTS (Zero-Hallucination)
# =============================================================================

# Per-framework Scope 3 disclosure requirements
FRAMEWORK_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    "ghg_protocol": [
        {"id": "GHGP-S3-01", "section": "Chapter 5", "desc": "Total Scope 3 emissions (tCO2e)", "field": "total_scope3_tco2e", "mandatory": True},
        {"id": "GHGP-S3-02", "section": "Chapter 5", "desc": "Per-category breakdown (15 categories)", "field": "category_breakdown", "mandatory": True},
        {"id": "GHGP-S3-03", "section": "Chapter 7", "desc": "Screening methodology description", "field": "methodology_description", "mandatory": True},
        {"id": "GHGP-S3-04", "section": "Chapter 7", "desc": "Category relevance assessment", "field": "relevance_assessment", "mandatory": True},
        {"id": "GHGP-S3-05", "section": "Chapter 8", "desc": "Calculation methodology per category", "field": "methodology_tiers", "mandatory": True},
        {"id": "GHGP-S3-06", "section": "Chapter 8", "desc": "Emission factors used", "field": "emission_factors", "mandatory": True},
        {"id": "GHGP-S3-07", "section": "Chapter 9", "desc": "Data quality assessment", "field": "data_quality_scores", "mandatory": True},
        {"id": "GHGP-S3-08", "section": "Chapter 9", "desc": "Uncertainty assessment", "field": "uncertainty_assessment", "mandatory": False},
        {"id": "GHGP-S3-09", "section": "Chapter 10", "desc": "Base year recalculation policy", "field": "base_year", "mandatory": True},
        {"id": "GHGP-S3-10", "section": "Chapter 11", "desc": "Verification statement", "field": "verification_statement", "mandatory": False},
    ],
    "esrs_e1": [
        {"id": "ESRS-E1-01", "section": "E1-6 para 51", "desc": "Total Scope 3 GHG emissions (tCO2e)", "field": "total_scope3_tco2e", "mandatory": True},
        {"id": "ESRS-E1-02", "section": "E1-6 para 51", "desc": "Scope 3 upstream emissions", "field": "upstream_tco2e", "mandatory": True},
        {"id": "ESRS-E1-03", "section": "E1-6 para 51", "desc": "Scope 3 downstream emissions", "field": "downstream_tco2e", "mandatory": True},
        {"id": "ESRS-E1-04", "section": "E1-6 para 52", "desc": "Significant categories reported separately", "field": "category_breakdown", "mandatory": True},
        {"id": "ESRS-E1-05", "section": "E1-6 para 53", "desc": "Exclusion justification for omitted categories", "field": "exclusion_justification", "mandatory": True},
        {"id": "ESRS-E1-06", "section": "E1-6 para 55", "desc": "Estimation methods description", "field": "methodology_tiers", "mandatory": True},
        {"id": "ESRS-E1-07", "section": "E1-4", "desc": "GHG reduction targets including Scope 3", "field": "reduction_targets", "mandatory": False},
        {"id": "ESRS-E1-08", "section": "E1-1", "desc": "Transition plan with Scope 3 decarbonization", "field": "transition_plan", "mandatory": False},
    ],
    "cdp_climate": [
        {"id": "CDP-C6.5a", "section": "C6.5a", "desc": "Scope 3 per-category emissions", "field": "category_breakdown", "mandatory": True},
        {"id": "CDP-C6.5b", "section": "C6.5b", "desc": "Category relevance explanation", "field": "relevance_assessment", "mandatory": True},
        {"id": "CDP-C6.7", "section": "C6.7", "desc": "Scope 3 methodology description", "field": "methodology_tiers", "mandatory": True},
        {"id": "CDP-C6.7a", "section": "C6.7a", "desc": "Scope 3 verification status", "field": "verification_statement", "mandatory": False},
        {"id": "CDP-C6.10", "section": "C6.10", "desc": "Total global Scope 3 emissions", "field": "total_scope3_tco2e", "mandatory": True},
        {"id": "CDP-C12.1", "section": "C12.1", "desc": "Supplier engagement on Scope 3", "field": "supplier_engagement", "mandatory": False},
    ],
    "sbti": [
        {"id": "SBTi-S3-01", "section": "NZS v1.1 5.1", "desc": "Scope 3 screening results (all 15)", "field": "category_breakdown", "mandatory": True},
        {"id": "SBTi-S3-02", "section": "NZS v1.1 5.2", "desc": "Scope 3 near-term target (42% by 2030)", "field": "reduction_targets", "mandatory": True},
        {"id": "SBTi-S3-03", "section": "NZS v1.1 5.3", "desc": "Scope 3 long-term target (90%)", "field": "long_term_target", "mandatory": True},
        {"id": "SBTi-S3-04", "section": "NZS v1.1 5.4", "desc": "67%+ Scope 3 categories covered", "field": "category_coverage_pct", "mandatory": True},
        {"id": "SBTi-S3-05", "section": "NZS v1.1 5.5", "desc": "Supplier engagement target", "field": "supplier_engagement", "mandatory": False},
    ],
    "sec_climate": [
        {"id": "SEC-S3-01", "section": "Reg S-K Item 1504", "desc": "Scope 3 emissions if material", "field": "total_scope3_tco2e", "mandatory": False},
        {"id": "SEC-S3-02", "section": "Reg S-K Item 1504", "desc": "Scope 3 methodology and limitations", "field": "methodology_tiers", "mandatory": False},
        {"id": "SEC-S3-03", "section": "Reg S-K Item 1504", "desc": "Safe harbor attestation", "field": "safe_harbor", "mandatory": False},
    ],
    "sb_253": [
        {"id": "SB253-S3-01", "section": "Section 3", "desc": "Total Scope 3 emissions (required from 2027)", "field": "total_scope3_tco2e", "mandatory": True},
        {"id": "SB253-S3-02", "section": "Section 3", "desc": "Per-category Scope 3 breakdown", "field": "category_breakdown", "mandatory": True},
        {"id": "SB253-S3-03", "section": "Section 3", "desc": "Third-party assurance", "field": "assurance_statement", "mandatory": True},
    ],
    "iso_14064": [
        {"id": "ISO-S3-01", "section": "Clause 5.2.4", "desc": "Category 3-6 indirect GHG emissions", "field": "total_scope3_tco2e", "mandatory": True},
        {"id": "ISO-S3-02", "section": "Clause 5.2.4", "desc": "Quantification methodology", "field": "methodology_tiers", "mandatory": True},
        {"id": "ISO-S3-03", "section": "Clause 5.3", "desc": "Data quality management", "field": "data_quality_scores", "mandatory": True},
        {"id": "ISO-S3-04", "section": "Clause 5.4", "desc": "Uncertainty assessment", "field": "uncertainty_assessment", "mandatory": True},
    ],
    "tcfd": [
        {"id": "TCFD-M-01", "section": "Metrics M-a", "desc": "Scope 3 GHG emissions", "field": "total_scope3_tco2e", "mandatory": True},
        {"id": "TCFD-M-02", "section": "Metrics M-a", "desc": "Scope 3 breakdown by category", "field": "category_breakdown", "mandatory": False},
        {"id": "TCFD-T-01", "section": "Targets T-a", "desc": "Scope 3 reduction targets", "field": "reduction_targets", "mandatory": False},
    ],
}

FRAMEWORK_DISPLAY_NAMES: Dict[str, str] = {
    "ghg_protocol": "GHG Protocol Scope 3 Standard",
    "esrs_e1": "ESRS E1 (EU CSRD)",
    "cdp_climate": "CDP Climate Change",
    "sbti": "SBTi Corporate Net-Zero Standard",
    "sec_climate": "US SEC Climate Disclosure",
    "sb_253": "California SB 253",
    "iso_14064": "ISO 14064-1:2018",
    "tcfd": "TCFD Recommendations",
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DisclosureWorkflow:
    """
    4-phase Scope 3 disclosure workflow.

    Maps inventory data against framework-specific requirements, identifies
    compliance gaps, and generates disclosure-ready outputs including reports,
    data exports, and XBRL tags.

    Zero-hallucination: all compliance rules are derived from published framework
    standards. No LLM calls for compliance determination.

    Example:
        >>> wf = DisclosureWorkflow()
        >>> inv = InventoryData(total_scope3_tco2e=10000.0)
        >>> inp = DisclosureInput(
        ...     inventory_data=inv,
        ...     target_frameworks=[DisclosureFramework.GHG_PROTOCOL],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize DisclosureWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._requirements: List[FrameworkRequirement] = []
        self._scores: List[ComplianceScore] = []
        self._gaps: List[ComplianceGap] = []
        self._documents: List[DisclosureDocument] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[DisclosureInput] = None,
        inventory_data: Optional[InventoryData] = None,
        target_frameworks: Optional[List[DisclosureFramework]] = None,
    ) -> DisclosureOutput:
        """
        Execute the 4-phase disclosure workflow.

        Args:
            input_data: Full input model (preferred).
            inventory_data: Inventory data (fallback).
            target_frameworks: Target frameworks (fallback).

        Returns:
            DisclosureOutput with compliance scores, gaps, and documents.
        """
        if input_data is None:
            input_data = DisclosureInput(
                inventory_data=inventory_data or InventoryData(),
                target_frameworks=target_frameworks or [DisclosureFramework.GHG_PROTOCOL],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting disclosure workflow %s frameworks=%d",
            self.workflow_id, len(input_data.target_frameworks),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            for phase_num, phase_fn in enumerate(
                [
                    self._phase_framework_selection,
                    self._phase_compliance_mapping,
                    self._phase_gap_analysis,
                    self._phase_disclosure_output,
                ],
                start=1,
            ):
                phase = await self._execute_with_retry(phase_fn, input_data, phase_num)
                self._phase_results.append(phase)
                if phase.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {phase_num} failed: {phase.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Disclosure workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        overall_compliance = 0.0
        if self._scores:
            overall_compliance = sum(s.compliance_pct for s in self._scores) / len(self._scores)

        result = DisclosureOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            selected_frameworks=[f.value for f in input_data.target_frameworks],
            compliance_scores=self._scores,
            compliance_gaps=self._gaps,
            framework_requirements=self._requirements,
            disclosure_documents=self._documents,
            overall_compliance_pct=round(overall_compliance, 1),
            progress_pct=100.0,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Disclosure workflow %s completed in %.2fs frameworks=%d "
            "compliance=%.1f%% gaps=%d documents=%d",
            self.workflow_id, elapsed, len(input_data.target_frameworks),
            overall_compliance, len(self._gaps), len(self._documents),
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: DisclosureInput, phase_number: int,
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
            phase_number=phase_number, status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Framework Selection
    # -------------------------------------------------------------------------

    async def _phase_framework_selection(
        self, input_data: DisclosureInput
    ) -> PhaseResult:
        """Validate and confirm target disclosure frameworks."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        frameworks = input_data.target_frameworks
        if not frameworks:
            frameworks = [DisclosureFramework.GHG_PROTOCOL]
            warnings.append("No frameworks specified; defaulting to GHG Protocol")

        # Count total requirements across selected frameworks
        total_reqs = 0
        mandatory_reqs = 0
        for fw in frameworks:
            reqs = FRAMEWORK_REQUIREMENTS.get(fw.value, [])
            total_reqs += len(reqs)
            mandatory_reqs += sum(1 for r in reqs if r["mandatory"])

        outputs["selected_frameworks"] = [fw.value for fw in frameworks]
        outputs["framework_count"] = len(frameworks)
        outputs["total_requirements"] = total_reqs
        outputs["mandatory_requirements"] = mandatory_reqs
        outputs["framework_details"] = {
            fw.value: {
                "name": FRAMEWORK_DISPLAY_NAMES.get(fw.value, fw.value),
                "requirements": len(FRAMEWORK_REQUIREMENTS.get(fw.value, [])),
            }
            for fw in frameworks
        }

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 FrameworkSelection: %d frameworks, %d total requirements",
            len(frameworks), total_reqs,
        )
        return PhaseResult(
            phase_name="framework_selection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Compliance Mapping
    # -------------------------------------------------------------------------

    async def _phase_compliance_mapping(
        self, input_data: DisclosureInput
    ) -> PhaseResult:
        """Map inventory data to framework requirements."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        inv = input_data.inventory_data
        self._requirements = []

        # Build available data fields
        available_fields = self._get_available_fields(inv)

        for fw in input_data.target_frameworks:
            reqs = FRAMEWORK_REQUIREMENTS.get(fw.value, [])

            for req_def in reqs:
                field = req_def["field"]
                data_available = field in available_fields

                if data_available:
                    status = ComplianceStatus.MET
                elif not req_def["mandatory"]:
                    status = ComplianceStatus.NOT_MET
                else:
                    status = ComplianceStatus.NOT_MET

                self._requirements.append(FrameworkRequirement(
                    requirement_id=req_def["id"],
                    framework=fw,
                    section=req_def["section"],
                    description=req_def["desc"],
                    data_field=field,
                    is_mandatory=req_def["mandatory"],
                    compliance_status=status,
                    data_available=data_available,
                ))

        # Calculate compliance scores per framework
        self._scores = []
        for fw in input_data.target_frameworks:
            fw_reqs = [r for r in self._requirements if r.framework == fw]
            met = sum(1 for r in fw_reqs if r.compliance_status == ComplianceStatus.MET)
            partial = sum(1 for r in fw_reqs if r.compliance_status == ComplianceStatus.PARTIALLY_MET)
            not_met = sum(1 for r in fw_reqs if r.compliance_status == ComplianceStatus.NOT_MET)
            total = len(fw_reqs)

            compliance_pct = ((met + partial * 0.5) / total * 100.0) if total > 0 else 0.0

            mandatory_reqs = [r for r in fw_reqs if r.is_mandatory]
            mandatory_met = sum(1 for r in mandatory_reqs if r.compliance_status == ComplianceStatus.MET)
            mandatory_pct = (mandatory_met / len(mandatory_reqs) * 100.0) if mandatory_reqs else 100.0

            self._scores.append(ComplianceScore(
                framework=fw,
                framework_name=FRAMEWORK_DISPLAY_NAMES.get(fw.value, fw.value),
                total_requirements=total,
                requirements_met=met,
                requirements_partially_met=partial,
                requirements_not_met=not_met,
                compliance_pct=round(compliance_pct, 1),
                mandatory_met_pct=round(mandatory_pct, 1),
            ))

        outputs["frameworks_mapped"] = len(self._scores)
        outputs["total_requirements_checked"] = len(self._requirements)
        outputs["compliance_by_framework"] = {
            s.framework.value: {
                "compliance_pct": s.compliance_pct,
                "mandatory_met_pct": s.mandatory_met_pct,
                "met": s.requirements_met,
                "not_met": s.requirements_not_met,
            }
            for s in self._scores
        }

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 ComplianceMapping: %d requirements checked across %d frameworks",
            len(self._requirements), len(self._scores),
        )
        return PhaseResult(
            phase_name="compliance_mapping", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _get_available_fields(self, inv: InventoryData) -> set:
        """Determine which data fields are available from inventory."""
        available: set = set()

        if inv.total_scope3_tco2e > 0:
            available.add("total_scope3_tco2e")
        if inv.upstream_tco2e > 0:
            available.add("upstream_tco2e")
        if inv.downstream_tco2e > 0:
            available.add("downstream_tco2e")
        if inv.category_breakdown:
            available.add("category_breakdown")
            # Check coverage
            cats_with_data = sum(1 for v in inv.category_breakdown.values() if v > 0)
            if cats_with_data >= 10:
                available.add("category_coverage_pct")
        if inv.methodology_tiers:
            available.add("methodology_tiers")
            available.add("methodology_description")
        if inv.data_quality_scores:
            available.add("data_quality_scores")
        if inv.base_year > 0:
            available.add("base_year")
        if inv.scope1_tco2e > 0 or inv.scope2_location_tco2e > 0:
            available.add("scope_1_2_data")

        # Fields that require additional data not in inventory
        # These remain unavailable: relevance_assessment, emission_factors,
        # uncertainty_assessment, verification_statement, reduction_targets,
        # transition_plan, supplier_engagement, safe_harbor, assurance_statement,
        # exclusion_justification, long_term_target

        return available

    # -------------------------------------------------------------------------
    # Phase 3: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(
        self, input_data: DisclosureInput
    ) -> PhaseResult:
        """Identify missing requirements per framework."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gaps = []

        for req in self._requirements:
            if req.compliance_status in (ComplianceStatus.MET, ComplianceStatus.NOT_APPLICABLE):
                continue

            severity = GapSeverity.MAJOR if req.is_mandatory else GapSeverity.MINOR

            # Estimate remediation effort
            effort = self._estimate_gap_effort(req)

            remediation = self._suggest_remediation(req)

            self._gaps.append(ComplianceGap(
                framework=req.framework,
                requirement_id=req.requirement_id,
                description=f"Missing: {req.description}",
                severity=severity,
                remediation=remediation,
                estimated_effort_hours=effort,
            ))

        critical_gaps = sum(1 for g in self._gaps if g.severity == GapSeverity.CRITICAL)
        major_gaps = sum(1 for g in self._gaps if g.severity == GapSeverity.MAJOR)
        minor_gaps = sum(1 for g in self._gaps if g.severity == GapSeverity.MINOR)

        outputs["total_gaps"] = len(self._gaps)
        outputs["critical_gaps"] = critical_gaps
        outputs["major_gaps"] = major_gaps
        outputs["minor_gaps"] = minor_gaps
        outputs["total_remediation_hours"] = round(
            sum(g.estimated_effort_hours for g in self._gaps), 1
        )
        outputs["gaps_by_framework"] = {}
        for fw in input_data.target_frameworks:
            fw_gaps = [g for g in self._gaps if g.framework == fw]
            outputs["gaps_by_framework"][fw.value] = {
                "total": len(fw_gaps),
                "major": sum(1 for g in fw_gaps if g.severity == GapSeverity.MAJOR),
            }

        if major_gaps > 0:
            warnings.append(
                f"{major_gaps} major compliance gaps identified; "
                f"address before submission"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 GapAnalysis: %d gaps (critical=%d major=%d minor=%d)",
            len(self._gaps), critical_gaps, major_gaps, minor_gaps,
        )
        return PhaseResult(
            phase_name="gap_analysis", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _estimate_gap_effort(self, req: FrameworkRequirement) -> float:
        """Estimate remediation effort in hours for a gap."""
        effort_map: Dict[str, float] = {
            "verification_statement": 40.0,
            "assurance_statement": 40.0,
            "reduction_targets": 16.0,
            "long_term_target": 16.0,
            "transition_plan": 24.0,
            "supplier_engagement": 20.0,
            "uncertainty_assessment": 8.0,
            "emission_factors": 4.0,
            "relevance_assessment": 4.0,
            "exclusion_justification": 2.0,
            "safe_harbor": 4.0,
            "methodology_description": 4.0,
        }
        return effort_map.get(req.data_field, 8.0)

    def _suggest_remediation(self, req: FrameworkRequirement) -> str:
        """Suggest remediation action for a compliance gap."""
        remediation_map: Dict[str, str] = {
            "verification_statement": "Engage third-party verifier for Scope 3 limited assurance",
            "assurance_statement": "Engage accredited assurance provider per SB 253 requirements",
            "reduction_targets": "Set science-based targets using SBTi target-setting tool",
            "long_term_target": "Define long-term (2050) Scope 3 reduction target per SBTi NZS",
            "transition_plan": "Develop Scope 3 decarbonization strategy with milestones",
            "supplier_engagement": "Launch supplier carbon assessment and engagement program",
            "uncertainty_assessment": "Run Monte Carlo uncertainty analysis on category results",
            "emission_factors": "Document all emission factors with source references",
            "relevance_assessment": "Complete category relevance screening using PACK-042",
            "exclusion_justification": "Document rationale for any excluded Scope 3 categories",
            "safe_harbor": "Prepare safe harbor attestation per SEC climate rules",
            "methodology_description": "Document calculation methodology for each category",
        }
        return remediation_map.get(
            req.data_field,
            f"Collect data for {req.description}",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Disclosure Output
    # -------------------------------------------------------------------------

    async def _phase_disclosure_output(
        self, input_data: DisclosureInput
    ) -> PhaseResult:
        """Generate framework-specific reports and data exports."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._documents = []
        inv = input_data.inventory_data

        for fw in input_data.target_frameworks:
            fw_name = FRAMEWORK_DISPLAY_NAMES.get(fw.value, fw.value)

            for fmt in input_data.output_formats:
                sections = self._generate_document_sections(fw, inv)
                content_hash = self._hash_dict({
                    "framework": fw.value,
                    "format": fmt.value,
                    "total": inv.total_scope3_tco2e,
                    "year": inv.reporting_year,
                })

                self._documents.append(DisclosureDocument(
                    framework=fw,
                    title=f"Scope 3 Disclosure - {fw_name} ({inv.reporting_year})",
                    format=fmt,
                    content_summary=(
                        f"{fw_name} Scope 3 disclosure for {inv.organization_name or 'Organization'} "
                        f"reporting year {inv.reporting_year}: "
                        f"{inv.total_scope3_tco2e:.1f} tCO2e total"
                    ),
                    sections=sections,
                    provenance_hash=content_hash,
                ))

            # Generate XBRL if requested
            if input_data.include_xbrl_tags and fw == DisclosureFramework.ESRS_E1:
                xbrl_sections = [
                    "esrs:E1-6_Scope3GHGEmissions",
                    "esrs:E1-6_Scope3Upstream",
                    "esrs:E1-6_Scope3Downstream",
                    "esrs:E1-6_SignificantCategories",
                ]
                xbrl_hash = self._hash_dict({
                    "framework": fw.value, "format": "xbrl",
                    "total": inv.total_scope3_tco2e,
                })
                self._documents.append(DisclosureDocument(
                    framework=fw,
                    title=f"XBRL Tags - {fw_name} ({inv.reporting_year})",
                    format=OutputFormat.XBRL,
                    content_summary=f"XBRL-tagged data for {fw_name}",
                    sections=xbrl_sections,
                    provenance_hash=xbrl_hash,
                ))

        outputs["documents_generated"] = len(self._documents)
        outputs["formats_used"] = list({d.format.value for d in self._documents})
        outputs["frameworks_covered"] = list({d.framework.value for d in self._documents})

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 DisclosureOutput: %d documents generated",
            len(self._documents),
        )
        return PhaseResult(
            phase_name="disclosure_output", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_document_sections(
        self, framework: DisclosureFramework, inv: InventoryData
    ) -> List[str]:
        """Generate section list for a disclosure document."""
        common_sections = [
            "Executive Summary",
            "Organizational Boundary",
            "Scope 3 Methodology",
        ]

        fw_specific: Dict[str, List[str]] = {
            "ghg_protocol": [
                "Category Screening Results",
                "Per-Category Emissions Breakdown",
                "Data Quality Assessment",
                "Uncertainty Analysis",
                "Base Year Comparison",
                "Verification Statement",
            ],
            "esrs_e1": [
                "E1-6 Total Scope 3 Emissions",
                "Upstream / Downstream Split",
                "Significant Category Disclosures",
                "Estimation Methods",
                "Exclusion Justifications",
            ],
            "cdp_climate": [
                "C6.5a Category-Level Emissions",
                "C6.5b Relevance Explanations",
                "C6.7 Methodology Description",
                "C6.10 Global Scope 3 Total",
            ],
            "sbti": [
                "Scope 3 Screening Inventory",
                "Near-Term Target Progress",
                "Category Coverage Assessment",
                "Supplier Engagement Metrics",
            ],
            "sec_climate": [
                "Scope 3 Materiality Assessment",
                "Quantification Methodology",
                "Safe Harbor Statement",
            ],
            "sb_253": [
                "Annual Scope 3 Emissions Report",
                "Category-Level Disclosure",
                "Assurance Provider Statement",
            ],
            "iso_14064": [
                "Indirect GHG Emissions (Cat 3-6)",
                "Quantification Methodology",
                "Data Quality Management",
                "Uncertainty Assessment",
            ],
            "tcfd": [
                "Scope 3 Metrics",
                "Category Breakdown",
                "Reduction Targets",
            ],
        }

        return common_sections + fw_specific.get(framework.value, [])

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._requirements = []
        self._scores = []
        self._gaps = []
        self._documents = []
        self._phase_results = []

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: DisclosureOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.overall_compliance_pct}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
