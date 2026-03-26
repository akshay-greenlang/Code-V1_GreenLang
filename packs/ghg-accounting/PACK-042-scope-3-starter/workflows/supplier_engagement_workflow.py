# -*- coding: utf-8 -*-
"""
Supplier Engagement Workflow
==================================

4-phase workflow for managing supplier carbon data collection and engagement
within PACK-042 Scope 3 Starter Pack.

Phases:
    1. SupplierPrioritization     -- Rank suppliers by emission contribution
                                     (targeting top 80% of procurement spend)
    2. DataRequestGeneration      -- Create standardized questionnaires per
                                     supplier segment/industry
    3. ResponseCollection         -- Track responses (sent/opened/in-progress/
                                     completed/overdue), send automated reminders
    4. QualityAssessment          -- Score supplier data quality (5 levels),
                                     identify tier upgrade opportunities,
                                     calculate engagement ROI

The workflow follows GreenLang zero-hallucination principles: all supplier
ranking, quality scoring, and ROI calculations use deterministic formulas.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Scope 3 Standard -- Chapter 8 (Supplier-specific data)
    SBTi Corporate Net-Zero Standard -- Supplier engagement target
    CDP Supply Chain Program guidance

Schedule: ongoing (quarterly cycle)
Estimated duration: ongoing (quarterly cycle)

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


class SupplierTier(str, Enum):
    """Supplier engagement tier."""

    STRATEGIC = "strategic"
    KEY = "key"
    STANDARD = "standard"
    MINOR = "minor"


class ResponseStatus(str, Enum):
    """Status of a supplier data request."""

    NOT_SENT = "not_sent"
    SENT = "sent"
    OPENED = "opened"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    DECLINED = "declined"


class DataQualityLevel(str, Enum):
    """Supplier data quality levels (1-5)."""

    LEVEL_1_NONE = "level_1_none"
    LEVEL_2_SPEND = "level_2_spend"
    LEVEL_3_AVERAGE = "level_3_average"
    LEVEL_4_PRODUCT = "level_4_product"
    LEVEL_5_VERIFIED = "level_5_verified"


class QuestionnaireType(str, Enum):
    """Types of supplier questionnaires."""

    BASIC_CARBON = "basic_carbon"
    DETAILED_CARBON = "detailed_carbon"
    CDP_SUPPLY_CHAIN = "cdp_supply_chain"
    PRODUCT_SPECIFIC = "product_specific"
    SBTI_ALIGNMENT = "sbti_alignment"


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


class SupplierRecord(BaseModel):
    """Supplier record for engagement tracking."""

    supplier_id: str = Field(
        default_factory=lambda: f"sup-{uuid.uuid4().hex[:8]}"
    )
    supplier_name: str = Field(default="")
    industry: str = Field(default="")
    country: str = Field(default="", description="ISO 3166-1 alpha-2")
    annual_spend_usd: float = Field(default=0.0, ge=0.0)
    estimated_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_categories: List[str] = Field(default_factory=list)
    current_data_quality: DataQualityLevel = Field(
        default=DataQualityLevel.LEVEL_1_NONE
    )
    has_sbti_target: bool = Field(default=False)
    has_cdp_response: bool = Field(default=False)
    contact_email: str = Field(default="")
    last_engagement_date: str = Field(default="")


class SupplierPrioritization(BaseModel):
    """Prioritization result for a supplier."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    tier: SupplierTier = Field(default=SupplierTier.MINOR)
    spend_rank: int = Field(default=0, ge=0)
    emission_rank: int = Field(default=0, ge=0)
    spend_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    emission_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    cumulative_spend_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    priority_score: float = Field(default=0.0, ge=0.0, le=100.0)
    recommended_questionnaire: QuestionnaireType = Field(
        default=QuestionnaireType.BASIC_CARBON
    )


class DataRequest(BaseModel):
    """Data request sent to a supplier."""

    request_id: str = Field(
        default_factory=lambda: f"req-{uuid.uuid4().hex[:8]}"
    )
    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    questionnaire_type: QuestionnaireType = Field(
        default=QuestionnaireType.BASIC_CARBON
    )
    status: ResponseStatus = Field(default=ResponseStatus.NOT_SENT)
    sent_date: str = Field(default="")
    due_date: str = Field(default="")
    completed_date: str = Field(default="")
    reminder_count: int = Field(default=0, ge=0)
    questions_total: int = Field(default=0, ge=0)
    questions_answered: int = Field(default=0, ge=0)


class QualityAssessment(BaseModel):
    """Quality assessment for supplier-provided data."""

    supplier_id: str = Field(default="")
    supplier_name: str = Field(default="")
    previous_quality: DataQualityLevel = Field(
        default=DataQualityLevel.LEVEL_1_NONE
    )
    current_quality: DataQualityLevel = Field(
        default=DataQualityLevel.LEVEL_1_NONE
    )
    quality_score: float = Field(default=1.0, ge=1.0, le=5.0)
    improvement_from_previous: bool = Field(default=False)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    accuracy_flags: List[str] = Field(default_factory=list)
    tier_upgrade_possible: bool = Field(default=False)
    tier_upgrade_recommendation: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class SupplierEngagementInput(BaseModel):
    """Input data model for SupplierEngagementWorkflow."""

    suppliers: List[SupplierRecord] = Field(
        default_factory=list, description="Supplier master data"
    )
    existing_requests: List[DataRequest] = Field(
        default_factory=list, description="Pre-existing data requests"
    )
    spend_threshold_pct: float = Field(
        default=80.0, ge=50.0, le=100.0,
        description="Target % of spend to cover with engagement",
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    organization_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class SupplierEngagementOutput(BaseModel):
    """Complete result from supplier engagement workflow."""

    workflow_id: str = Field(...)
    workflow_name: str = Field(default="supplier_engagement")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    prioritized_suppliers: List[SupplierPrioritization] = Field(default_factory=list)
    data_requests: List[DataRequest] = Field(default_factory=list)
    quality_assessments: List[QualityAssessment] = Field(default_factory=list)
    # Summary metrics
    total_suppliers: int = Field(default=0, ge=0)
    suppliers_engaged: int = Field(default=0, ge=0)
    suppliers_responded: int = Field(default=0, ge=0)
    response_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    spend_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_data_quality_score: float = Field(default=1.0, ge=1.0, le=5.0)
    engagement_roi_tco2e: float = Field(
        default=0.0, ge=0.0,
        description="Emission reduction achieved through engagement",
    )
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# QUESTIONNAIRE TEMPLATES (Zero-Hallucination)
# =============================================================================

QUESTIONNAIRE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "basic_carbon": {
        "title": "Basic Carbon Data Request",
        "questions": 8,
        "fields": [
            "total_ghg_emissions_tco2e",
            "scope_1_emissions_tco2e",
            "scope_2_emissions_tco2e",
            "reporting_year",
            "methodology_used",
            "verification_status",
            "reduction_target",
            "contact_person",
        ],
        "estimated_completion_days": 14,
    },
    "detailed_carbon": {
        "title": "Detailed Carbon Footprint Request",
        "questions": 20,
        "fields": [
            "total_ghg_emissions_tco2e",
            "scope_1_by_source",
            "scope_2_location_based",
            "scope_2_market_based",
            "scope_3_upstream_tco2e",
            "reporting_year",
            "reporting_boundary",
            "methodology_standard",
            "emission_factors_source",
            "verification_status",
            "verifier_name",
            "reduction_target",
            "sbti_status",
            "renewable_energy_pct",
            "energy_consumption_kwh",
            "revenue_or_production_volume",
            "allocation_method",
            "product_carbon_footprint",
            "data_quality_self_assessment",
            "contact_person",
        ],
        "estimated_completion_days": 30,
    },
    "cdp_supply_chain": {
        "title": "CDP Supply Chain Questionnaire",
        "questions": 35,
        "fields": [
            "cdp_score",
            "total_emissions_tco2e",
            "scope_1_2_3_breakdown",
            "climate_risks",
            "reduction_initiatives",
            "sbti_status",
            "governance",
        ],
        "estimated_completion_days": 60,
    },
    "product_specific": {
        "title": "Product-Level Carbon Data Request",
        "questions": 15,
        "fields": [
            "product_name",
            "product_carbon_footprint_kgco2e",
            "cradle_to_gate_emissions",
            "gate_to_gate_emissions",
            "functional_unit",
            "lca_methodology",
            "epd_reference",
            "allocation_method",
            "data_source_primary_pct",
            "verification_status",
        ],
        "estimated_completion_days": 30,
    },
    "sbti_alignment": {
        "title": "SBTi Alignment Assessment",
        "questions": 12,
        "fields": [
            "sbti_commitment_status",
            "target_type",
            "target_year",
            "base_year",
            "reduction_target_pct",
            "current_emissions_tco2e",
            "progress_against_target_pct",
            "scope_coverage",
            "sector_classification",
            "validation_date",
            "net_zero_commitment",
            "transition_plan",
        ],
        "estimated_completion_days": 21,
    },
}

# Data quality level scoring criteria
QUALITY_LEVEL_CRITERIA: Dict[str, Dict[str, Any]] = {
    "level_1_none": {"score": 1.0, "description": "No supplier-specific data; rely on EEIO"},
    "level_2_spend": {"score": 2.0, "description": "Spend-based with sector EF only"},
    "level_3_average": {"score": 3.0, "description": "Average data with industry EF"},
    "level_4_product": {"score": 4.0, "description": "Product-level carbon footprint from supplier"},
    "level_5_verified": {"score": 5.0, "description": "Third-party verified supplier data"},
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SupplierEngagementWorkflow:
    """
    4-phase supplier engagement workflow for Scope 3 data collection.

    Ranks suppliers by emission contribution, generates standardized
    questionnaires, tracks responses and reminders, and assesses data
    quality with tier upgrade recommendations.

    Zero-hallucination: all ranking, scoring, and ROI calculations use
    deterministic formulas. No LLM calls in numeric paths.

    Example:
        >>> wf = SupplierEngagementWorkflow()
        >>> suppliers = [SupplierRecord(
        ...     supplier_name="ACME Corp",
        ...     annual_spend_usd=5_000_000,
        ... )]
        >>> inp = SupplierEngagementInput(suppliers=suppliers)
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize SupplierEngagementWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._prioritizations: List[SupplierPrioritization] = []
        self._data_requests: List[DataRequest] = []
        self._quality_assessments: List[QualityAssessment] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[SupplierEngagementInput] = None,
        suppliers: Optional[List[SupplierRecord]] = None,
    ) -> SupplierEngagementOutput:
        """
        Execute the 4-phase supplier engagement workflow.

        Args:
            input_data: Full input model (preferred).
            suppliers: Supplier records (fallback).

        Returns:
            SupplierEngagementOutput with prioritization, requests, quality.
        """
        if input_data is None:
            input_data = SupplierEngagementInput(
                suppliers=suppliers or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting supplier engagement workflow %s suppliers=%d",
            self.workflow_id, len(input_data.suppliers),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            for phase_num, phase_fn in enumerate(
                [
                    self._phase_supplier_prioritization,
                    self._phase_data_request_generation,
                    self._phase_response_collection,
                    self._phase_quality_assessment,
                ],
                start=1,
            ):
                phase = await self._execute_with_retry(phase_fn, input_data, phase_num)
                self._phase_results.append(phase)
                if phase.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {phase_num} failed: {phase.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Supplier engagement workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        # Calculate summary metrics
        total_suppliers = len(input_data.suppliers)
        engaged = sum(
            1 for dr in self._data_requests
            if dr.status != ResponseStatus.NOT_SENT
        )
        responded = sum(
            1 for dr in self._data_requests
            if dr.status == ResponseStatus.COMPLETED
        )
        response_rate = (responded / engaged * 100.0) if engaged > 0 else 0.0

        total_spend = sum(s.annual_spend_usd for s in input_data.suppliers)
        engaged_spend = sum(
            next(
                (s.annual_spend_usd for s in input_data.suppliers if s.supplier_id == dr.supplier_id),
                0.0,
            )
            for dr in self._data_requests
            if dr.status != ResponseStatus.NOT_SENT
        )
        spend_coverage = (engaged_spend / total_spend * 100.0) if total_spend > 0 else 0.0

        avg_quality = 1.0
        if self._quality_assessments:
            avg_quality = sum(
                qa.quality_score for qa in self._quality_assessments
            ) / len(self._quality_assessments)

        # Estimate engagement ROI: improved data quality reduces overestimation
        roi_tco2e = sum(
            qa.quality_score - QUALITY_LEVEL_CRITERIA[qa.previous_quality.value]["score"]
            for qa in self._quality_assessments
            if qa.improvement_from_previous
        ) * 50.0  # Approximate tCO2e reduction per quality level improvement

        result = SupplierEngagementOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            prioritized_suppliers=self._prioritizations,
            data_requests=self._data_requests,
            quality_assessments=self._quality_assessments,
            total_suppliers=total_suppliers,
            suppliers_engaged=engaged,
            suppliers_responded=responded,
            response_rate_pct=round(response_rate, 1),
            spend_coverage_pct=round(spend_coverage, 1),
            avg_data_quality_score=round(avg_quality, 2),
            engagement_roi_tco2e=round(roi_tco2e, 2),
            progress_pct=100.0,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Supplier engagement workflow %s completed in %.2fs "
            "suppliers=%d engaged=%d responded=%d rate=%.1f%%",
            self.workflow_id, elapsed, total_suppliers,
            engaged, responded, response_rate,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: SupplierEngagementInput,
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
                    import asyncio
                    await asyncio.sleep(self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1)))
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number, status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Supplier Prioritization
    # -------------------------------------------------------------------------

    async def _phase_supplier_prioritization(
        self, input_data: SupplierEngagementInput
    ) -> PhaseResult:
        """Rank suppliers by emission contribution targeting 80% of spend."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        suppliers = input_data.suppliers
        total_spend = sum(s.annual_spend_usd for s in suppliers)
        total_emissions = sum(s.estimated_emissions_tco2e for s in suppliers)

        if not suppliers:
            warnings.append("No suppliers provided")

        # Sort by spend descending
        sorted_suppliers = sorted(
            suppliers, key=lambda s: s.annual_spend_usd, reverse=True
        )

        self._prioritizations = []
        cumulative_spend_pct = 0.0

        for rank, sup in enumerate(sorted_suppliers, start=1):
            spend_pct = (
                (sup.annual_spend_usd / total_spend * 100.0)
                if total_spend > 0 else 0.0
            )
            emission_pct = (
                (sup.estimated_emissions_tco2e / total_emissions * 100.0)
                if total_emissions > 0 else 0.0
            )
            cumulative_spend_pct += spend_pct

            # Calculate priority score (weighted: 60% spend, 40% emissions)
            priority_score = spend_pct * 0.6 + emission_pct * 0.4

            # Determine tier
            if cumulative_spend_pct <= 50.0:
                tier = SupplierTier.STRATEGIC
            elif cumulative_spend_pct <= 80.0:
                tier = SupplierTier.KEY
            elif cumulative_spend_pct <= 95.0:
                tier = SupplierTier.STANDARD
            else:
                tier = SupplierTier.MINOR

            # Recommend questionnaire type based on tier
            questionnaire = self._select_questionnaire(tier, sup)

            # Determine emission rank
            emission_rank = sorted(
                suppliers,
                key=lambda s: s.estimated_emissions_tco2e,
                reverse=True,
            ).index(sup) + 1 if sup in suppliers else rank

            self._prioritizations.append(SupplierPrioritization(
                supplier_id=sup.supplier_id,
                supplier_name=sup.supplier_name,
                tier=tier,
                spend_rank=rank,
                emission_rank=emission_rank,
                spend_pct=round(spend_pct, 2),
                emission_pct=round(emission_pct, 2),
                cumulative_spend_pct=round(cumulative_spend_pct, 2),
                priority_score=round(priority_score, 2),
                recommended_questionnaire=questionnaire,
            ))

        strategic = sum(1 for p in self._prioritizations if p.tier == SupplierTier.STRATEGIC)
        key = sum(1 for p in self._prioritizations if p.tier == SupplierTier.KEY)

        outputs["total_suppliers"] = len(self._prioritizations)
        outputs["total_spend_usd"] = round(total_spend, 2)
        outputs["strategic_suppliers"] = strategic
        outputs["key_suppliers"] = key
        outputs["suppliers_in_80pct"] = sum(
            1 for p in self._prioritizations
            if p.cumulative_spend_pct <= input_data.spend_threshold_pct
        )
        outputs["top_5_by_spend"] = [
            {"name": p.supplier_name, "spend_pct": p.spend_pct, "tier": p.tier.value}
            for p in self._prioritizations[:5]
        ]

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 SupplierPrioritization: %d suppliers, strategic=%d key=%d",
            len(self._prioritizations), strategic, key,
        )
        return PhaseResult(
            phase_name="supplier_prioritization", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _select_questionnaire(
        self, tier: SupplierTier, supplier: SupplierRecord
    ) -> QuestionnaireType:
        """Select appropriate questionnaire type based on supplier tier."""
        if supplier.has_cdp_response:
            return QuestionnaireType.CDP_SUPPLY_CHAIN

        if tier == SupplierTier.STRATEGIC:
            if supplier.has_sbti_target:
                return QuestionnaireType.SBTI_ALIGNMENT
            return QuestionnaireType.DETAILED_CARBON

        if tier == SupplierTier.KEY:
            return QuestionnaireType.DETAILED_CARBON

        if tier == SupplierTier.STANDARD:
            return QuestionnaireType.BASIC_CARBON

        return QuestionnaireType.BASIC_CARBON

    # -------------------------------------------------------------------------
    # Phase 2: Data Request Generation
    # -------------------------------------------------------------------------

    async def _phase_data_request_generation(
        self, input_data: SupplierEngagementInput
    ) -> PhaseResult:
        """Create standardized data requests per supplier."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._data_requests = list(input_data.existing_requests)
        existing_supplier_ids = {dr.supplier_id for dr in self._data_requests}

        new_requests = 0
        for prio in self._prioritizations:
            # Skip if below spend threshold
            if prio.cumulative_spend_pct > input_data.spend_threshold_pct:
                continue

            # Skip if already has an existing request
            if prio.supplier_id in existing_supplier_ids:
                continue

            template = QUESTIONNAIRE_TEMPLATES.get(
                prio.recommended_questionnaire.value,
                QUESTIONNAIRE_TEMPLATES["basic_carbon"],
            )

            self._data_requests.append(DataRequest(
                supplier_id=prio.supplier_id,
                supplier_name=prio.supplier_name,
                questionnaire_type=prio.recommended_questionnaire,
                status=ResponseStatus.NOT_SENT,
                questions_total=template["questions"],
                questions_answered=0,
            ))
            new_requests += 1

        questionnaire_counts: Dict[str, int] = {}
        for dr in self._data_requests:
            qt = dr.questionnaire_type.value
            questionnaire_counts[qt] = questionnaire_counts.get(qt, 0) + 1

        outputs["total_requests"] = len(self._data_requests)
        outputs["new_requests_created"] = new_requests
        outputs["existing_requests"] = len(input_data.existing_requests)
        outputs["questionnaire_distribution"] = questionnaire_counts

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DataRequestGeneration: %d total requests (%d new)",
            len(self._data_requests), new_requests,
        )
        return PhaseResult(
            phase_name="data_request_generation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Response Collection
    # -------------------------------------------------------------------------

    async def _phase_response_collection(
        self, input_data: SupplierEngagementInput
    ) -> PhaseResult:
        """Track response status and identify overdue requests."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        now = datetime.utcnow().isoformat()

        # Track status counts
        status_counts: Dict[str, int] = {}
        overdue_suppliers: List[str] = []

        for dr in self._data_requests:
            status_counts[dr.status.value] = (
                status_counts.get(dr.status.value, 0) + 1
            )

            # Check for overdue requests
            if dr.status in (ResponseStatus.SENT, ResponseStatus.OPENED, ResponseStatus.IN_PROGRESS):
                if dr.due_date and dr.due_date < now:
                    dr.status = ResponseStatus.OVERDUE
                    overdue_suppliers.append(dr.supplier_name)
                    status_counts[ResponseStatus.OVERDUE.value] = (
                        status_counts.get(ResponseStatus.OVERDUE.value, 0) + 1
                    )
                    status_counts[dr.status.value] = max(
                        status_counts.get(dr.status.value, 1) - 1, 0
                    )

        completed = status_counts.get(ResponseStatus.COMPLETED.value, 0)
        sent_total = len(self._data_requests) - status_counts.get(
            ResponseStatus.NOT_SENT.value, 0
        )
        response_rate = (completed / sent_total * 100.0) if sent_total > 0 else 0.0

        if overdue_suppliers:
            warnings.append(
                f"{len(overdue_suppliers)} requests are overdue: "
                f"{', '.join(overdue_suppliers[:5])}"
                + (f" and {len(overdue_suppliers) - 5} more" if len(overdue_suppliers) > 5 else "")
            )

        outputs["status_distribution"] = status_counts
        outputs["response_rate_pct"] = round(response_rate, 1)
        outputs["completed"] = completed
        outputs["overdue"] = len(overdue_suppliers)
        outputs["not_sent"] = status_counts.get(ResponseStatus.NOT_SENT.value, 0)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ResponseCollection: completed=%d overdue=%d rate=%.1f%%",
            completed, len(overdue_suppliers), response_rate,
        )
        return PhaseResult(
            phase_name="response_collection", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Quality Assessment
    # -------------------------------------------------------------------------

    async def _phase_quality_assessment(
        self, input_data: SupplierEngagementInput
    ) -> PhaseResult:
        """Score supplier data quality and identify tier upgrade opportunities."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._quality_assessments = []
        supplier_map = {s.supplier_id: s for s in input_data.suppliers}

        for dr in self._data_requests:
            if dr.status != ResponseStatus.COMPLETED:
                continue

            supplier = supplier_map.get(dr.supplier_id)
            if not supplier:
                continue

            # Assess quality based on questionnaire responses
            completeness = (
                (dr.questions_answered / dr.questions_total * 100.0)
                if dr.questions_total > 0 else 0.0
            )

            # Determine new quality level based on questionnaire type and completeness
            new_quality = self._assess_quality_level(
                dr.questionnaire_type, completeness, supplier
            )
            prev_quality = supplier.current_data_quality
            quality_score = QUALITY_LEVEL_CRITERIA[new_quality.value]["score"]

            improved = quality_score > QUALITY_LEVEL_CRITERIA[prev_quality.value]["score"]

            # Check tier upgrade possibility
            can_upgrade = new_quality.value in (
                DataQualityLevel.LEVEL_3_AVERAGE.value,
                DataQualityLevel.LEVEL_4_PRODUCT.value,
            )
            upgrade_rec = ""
            if can_upgrade:
                if new_quality == DataQualityLevel.LEVEL_3_AVERAGE:
                    upgrade_rec = (
                        "Request product-level carbon footprint data for key SKUs "
                        "to upgrade to Level 4"
                    )
                elif new_quality == DataQualityLevel.LEVEL_4_PRODUCT:
                    upgrade_rec = (
                        "Encourage third-party verification to upgrade to Level 5"
                    )

            self._quality_assessments.append(QualityAssessment(
                supplier_id=dr.supplier_id,
                supplier_name=dr.supplier_name,
                previous_quality=prev_quality,
                current_quality=new_quality,
                quality_score=quality_score,
                improvement_from_previous=improved,
                completeness_pct=round(completeness, 1),
                tier_upgrade_possible=can_upgrade,
                tier_upgrade_recommendation=upgrade_rec,
            ))

        # Summary statistics
        avg_quality = 1.0
        if self._quality_assessments:
            avg_quality = sum(
                qa.quality_score for qa in self._quality_assessments
            ) / len(self._quality_assessments)

        improved_count = sum(
            1 for qa in self._quality_assessments if qa.improvement_from_previous
        )
        upgrade_possible = sum(
            1 for qa in self._quality_assessments if qa.tier_upgrade_possible
        )

        quality_distribution: Dict[str, int] = {}
        for qa in self._quality_assessments:
            level = qa.current_quality.value
            quality_distribution[level] = quality_distribution.get(level, 0) + 1

        outputs["suppliers_assessed"] = len(self._quality_assessments)
        outputs["avg_quality_score"] = round(avg_quality, 2)
        outputs["improved_from_previous"] = improved_count
        outputs["tier_upgrade_possible"] = upgrade_possible
        outputs["quality_distribution"] = quality_distribution

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 QualityAssessment: %d assessed, avg=%.1f, improved=%d",
            len(self._quality_assessments), avg_quality, improved_count,
        )
        return PhaseResult(
            phase_name="quality_assessment", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assess_quality_level(
        self,
        questionnaire: QuestionnaireType,
        completeness_pct: float,
        supplier: SupplierRecord,
    ) -> DataQualityLevel:
        """Determine data quality level from questionnaire response."""
        if completeness_pct < 30.0:
            return DataQualityLevel.LEVEL_1_NONE

        if questionnaire == QuestionnaireType.PRODUCT_SPECIFIC and completeness_pct >= 80.0:
            return DataQualityLevel.LEVEL_4_PRODUCT

        if questionnaire in (QuestionnaireType.DETAILED_CARBON, QuestionnaireType.CDP_SUPPLY_CHAIN):
            if completeness_pct >= 90.0:
                return DataQualityLevel.LEVEL_4_PRODUCT
            elif completeness_pct >= 60.0:
                return DataQualityLevel.LEVEL_3_AVERAGE
            else:
                return DataQualityLevel.LEVEL_2_SPEND

        if questionnaire == QuestionnaireType.SBTI_ALIGNMENT and completeness_pct >= 70.0:
            return DataQualityLevel.LEVEL_3_AVERAGE

        if questionnaire == QuestionnaireType.BASIC_CARBON:
            if completeness_pct >= 80.0:
                return DataQualityLevel.LEVEL_3_AVERAGE
            elif completeness_pct >= 50.0:
                return DataQualityLevel.LEVEL_2_SPEND
            else:
                return DataQualityLevel.LEVEL_1_NONE

        return DataQualityLevel.LEVEL_2_SPEND

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._prioritizations = []
        self._data_requests = []
        self._quality_assessments = []
        self._phase_results = []

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: SupplierEngagementOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.avg_data_quality_score}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
