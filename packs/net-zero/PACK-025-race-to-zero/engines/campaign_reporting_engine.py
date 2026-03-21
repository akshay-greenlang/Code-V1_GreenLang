# -*- coding: utf-8 -*-
"""
CampaignReportingEngine - PACK-025 Race to Zero Engine 8
=========================================================

Generates Race to Zero annual disclosure reports aligned to campaign
and partner-specific formats. Produces structured 10-section reports
covering entity profile, pledge status, Starting Line compliance,
emissions inventory, target progress, action plan summary, sector
alignment, HLEG credibility, partnership engagement, and forward
commitments. Maps outputs to partner-specific reporting formats
(CDP, GFANZ, C40, SBTi).

Calculation Methodology:
    Report Completeness (0-100):
        Each of 10 report sections scored 0-10
        completeness = sum(section_score * section_weight) * 10
        Weights emphasize emissions inventory (15%) and target progress (15%)

    Verification Badge:
        VERIFIED: All emission data third-party verified
        PENDING:  Verification in progress
        FAILED:   Verification failed or not started

    Credibility Score Display:
        Integrates score from CredibilityAssessmentEngine (Engine 9)
        Displayed as 0-100 with tier (HIGH/MODERATE/LOW/CRITICAL)

    Partner Format Mapping:
        CDP:    Maps to C4 (Targets), C6 (Emissions), C12 (Engagement)
        GFANZ:  Maps to Transition Plan sections
        C40:    Maps to Deadline 2020 reporting format
        SBTi:   Maps to progress report format

Regulatory References:
    - Race to Zero Annual Reporting Requirements (2022)
    - HLEG "Integrity Matters" (2022), Rec 8 (transparency)
    - CDP Climate Change Questionnaire (2024)
    - GFANZ Transition Plan Framework (2022)
    - C40 Deadline 2020 Reporting (2023)

Zero-Hallucination:
    - Report sections from Race to Zero campaign requirements
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------


class ReportSectionId(str, Enum):
    """Report section identifiers."""
    ENTITY_PROFILE = "entity_profile"
    PLEDGE_STATUS = "pledge_status"
    STARTING_LINE = "starting_line"
    EMISSIONS_INVENTORY = "emissions_inventory"
    TARGET_PROGRESS = "target_progress"
    ACTION_PLAN = "action_plan"
    SECTOR_ALIGNMENT = "sector_alignment"
    HLEG_CREDIBILITY = "hleg_credibility"
    PARTNERSHIP = "partnership"
    FORWARD_COMMITMENTS = "forward_commitments"


class VerificationBadge(str, Enum):
    """Verification badge status."""
    VERIFIED = "verified"
    PENDING = "pending"
    FAILED = "failed"
    NOT_STARTED = "not_started"


class PartnerFormat(str, Enum):
    """Partner-specific report format."""
    CDP = "cdp"
    GFANZ = "gfanz"
    C40 = "c40"
    SBTI = "sbti"
    UNIVERSAL = "universal"


SECTION_WEIGHTS: Dict[str, Decimal] = {
    ReportSectionId.ENTITY_PROFILE.value: Decimal("0.05"),
    ReportSectionId.PLEDGE_STATUS.value: Decimal("0.10"),
    ReportSectionId.STARTING_LINE.value: Decimal("0.10"),
    ReportSectionId.EMISSIONS_INVENTORY.value: Decimal("0.15"),
    ReportSectionId.TARGET_PROGRESS.value: Decimal("0.15"),
    ReportSectionId.ACTION_PLAN.value: Decimal("0.12"),
    ReportSectionId.SECTOR_ALIGNMENT.value: Decimal("0.08"),
    ReportSectionId.HLEG_CREDIBILITY.value: Decimal("0.10"),
    ReportSectionId.PARTNERSHIP.value: Decimal("0.07"),
    ReportSectionId.FORWARD_COMMITMENTS.value: Decimal("0.08"),
}

SECTION_LABELS: Dict[str, str] = {
    ReportSectionId.ENTITY_PROFILE.value: "1. Entity Profile",
    ReportSectionId.PLEDGE_STATUS.value: "2. Pledge Status",
    ReportSectionId.STARTING_LINE.value: "3. Starting Line Compliance",
    ReportSectionId.EMISSIONS_INVENTORY.value: "4. Emissions Inventory",
    ReportSectionId.TARGET_PROGRESS.value: "5. Target Progress",
    ReportSectionId.ACTION_PLAN.value: "6. Action Plan Summary",
    ReportSectionId.SECTOR_ALIGNMENT.value: "7. Sector Alignment",
    ReportSectionId.HLEG_CREDIBILITY.value: "8. HLEG Credibility",
    ReportSectionId.PARTNERSHIP.value: "9. Partnership Engagement",
    ReportSectionId.FORWARD_COMMITMENTS.value: "10. Forward Commitments",
}

# Partner format field mappings.
CDP_MAPPING: Dict[str, str] = {
    "emissions_inventory": "C6 - Emissions Data",
    "target_progress": "C4 - Targets and Performance",
    "partnership": "C12 - Engagement",
    "action_plan": "C3 - Business Strategy",
}

GFANZ_MAPPING: Dict[str, str] = {
    "pledge_status": "Section 1: Foundations",
    "target_progress": "Section 2: Implementation Strategy",
    "action_plan": "Section 3: Engagement Strategy",
    "emissions_inventory": "Section 4: Metrics and Targets",
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class ReportSectionInput(BaseModel):
    """Input for a single report section.

    Attributes:
        section_id: Section identifier.
        data_completeness: Data completeness (0-10).
        content_available: Whether content is available.
        data_source: Source of section data.
        notes: Section notes.
    """
    section_id: str = Field(...)
    data_completeness: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("10"))
    content_available: bool = Field(default=False)
    data_source: str = Field(default="")
    notes: str = Field(default="")

    @field_validator("section_id")
    @classmethod
    def validate_section(cls, v: str) -> str:
        valid = {s.value for s in ReportSectionId}
        if v not in valid:
            raise ValueError(f"Unknown section '{v}'.")
        return v


class CampaignReportingInput(BaseModel):
    """Complete input for campaign reporting.

    Attributes:
        entity_name: Entity name.
        actor_type: Actor type.
        reporting_year: Reporting year.
        reporting_period_start: Period start (YYYY-MM-DD).
        reporting_period_end: Period end (YYYY-MM-DD).
        scope1_tco2e: Scope 1 emissions.
        scope2_tco2e: Scope 2 emissions.
        scope3_tco2e: Scope 3 emissions.
        total_tco2e: Total emissions.
        baseline_year: Baseline year.
        baseline_tco2e: Baseline emissions.
        target_year: Target year.
        target_tco2e: Target emissions.
        cumulative_reduction_pct: Cumulative reduction from baseline.
        pledge_quality_score: From PledgeCommitmentEngine.
        starting_line_score: From StartingLineEngine.
        action_plan_score: From ActionPlanEngine.
        credibility_score: From CredibilityAssessmentEngine.
        credibility_tier: HLEG credibility tier.
        partnership_score: From PartnershipScoringEngine.
        sector_alignment_score: From SectorPathwayEngine.
        progress_status: From ProgressTrackingEngine (on_track/caution/off_track).
        verification_status: Emission data verification status.
        partner_formats: Partner formats to generate.
        sections: Per-section input data.
        forward_commitments: Forward-looking commitments for next year.
        include_executive_summary: Generate executive summary.
    """
    entity_name: str = Field(..., min_length=1, max_length=300)
    actor_type: str = Field(default="corporate")
    reporting_year: int = Field(..., ge=2020, le=2060)
    reporting_period_start: Optional[str] = Field(default=None)
    reporting_period_end: Optional[str] = Field(default=None)
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    baseline_year: int = Field(default=0, ge=0, le=2060)
    baseline_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    target_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    cumulative_reduction_pct: Decimal = Field(default=Decimal("0"))
    pledge_quality_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    starting_line_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    action_plan_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    credibility_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    credibility_tier: str = Field(default="low")
    partnership_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    sector_alignment_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    progress_status: str = Field(default="off_track")
    verification_status: str = Field(default="not_started")
    partner_formats: List[str] = Field(default_factory=lambda: ["universal"])
    sections: List[ReportSectionInput] = Field(default_factory=list)
    forward_commitments: List[str] = Field(default_factory=list)
    include_executive_summary: bool = Field(default=True)

    @field_validator("verification_status")
    @classmethod
    def validate_verification(cls, v: str) -> str:
        valid = {s.value for s in VerificationBadge}
        if v not in valid:
            raise ValueError(f"Unknown verification status '{v}'.")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class ReportSectionResult(BaseModel):
    """Result for a single report section."""
    section_id: str = Field(default="")
    section_name: str = Field(default="")
    completeness: Decimal = Field(default=Decimal("0"))
    weight: Decimal = Field(default=Decimal("0"))
    weighted_score: Decimal = Field(default=Decimal("0"))
    data_available: bool = Field(default=False)
    partner_mappings: Dict[str, str] = Field(default_factory=dict)


class PartnerFormatOutput(BaseModel):
    """Partner-specific formatted output."""
    format_id: str = Field(default="")
    format_name: str = Field(default="")
    sections_mapped: int = Field(default=0)
    field_mappings: Dict[str, str] = Field(default_factory=dict)
    submission_ready: bool = Field(default=False)


class ExecutiveSummary(BaseModel):
    """Executive summary for the campaign report."""
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=0)
    total_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_reduction_pct: Decimal = Field(default=Decimal("0"))
    progress_status: str = Field(default="")
    credibility_tier: str = Field(default="")
    verification_badge: str = Field(default="")
    key_achievements: List[str] = Field(default_factory=list)
    key_challenges: List[str] = Field(default_factory=list)


class CampaignReportingResult(BaseModel):
    """Complete campaign reporting result."""
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=0)
    report_completeness: Decimal = Field(default=Decimal("0"))
    verification_badge: str = Field(default=VerificationBadge.NOT_STARTED.value)
    section_results: List[ReportSectionResult] = Field(default_factory=list)
    partner_outputs: List[PartnerFormatOutput] = Field(default_factory=list)
    executive_summary: Optional[ExecutiveSummary] = Field(default=None)
    submission_ready: bool = Field(default=False)
    sections_complete: int = Field(default=0)
    sections_total: int = Field(default=10)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CampaignReportingEngine:
    """Race to Zero annual campaign disclosure reporting engine.

    Generates structured 10-section reports, maps to partner-specific
    formats, and assesses report completeness and submission readiness.

    Usage::

        engine = CampaignReportingEngine()
        result = engine.generate(input_data)
        print(f"Completeness: {result.report_completeness}/100")
        print(f"Badge: {result.verification_badge}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        logger.info("CampaignReportingEngine v%s initialised", self.engine_version)

    def generate(
        self, data: CampaignReportingInput,
    ) -> CampaignReportingResult:
        """Generate campaign disclosure report."""
        t0 = time.perf_counter()
        logger.info(
            "Campaign reporting: entity=%s, year=%d",
            data.entity_name, data.reporting_year,
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Build section input map
        section_map = {s.section_id: s for s in data.sections}

        # Step 1: Assess sections
        section_results = self._assess_sections(data, section_map)

        # Step 2: Completeness score
        completeness = Decimal("0")
        for sr in section_results:
            completeness += sr.weighted_score
        completeness = _round_val(completeness * Decimal("10"), 2)

        # Step 3: Sections complete count
        complete_count = sum(1 for sr in section_results if sr.completeness >= Decimal("7"))

        # Step 4: Verification badge
        badge = data.verification_status

        # Step 5: Partner format outputs
        partner_outputs = self._generate_partner_outputs(data, section_results)

        # Step 6: Submission readiness
        submission_ready = (
            completeness >= Decimal("70")
            and badge in (VerificationBadge.VERIFIED.value, VerificationBadge.PENDING.value)
            and complete_count >= 8
        )

        # Step 7: Executive summary
        exec_summary: Optional[ExecutiveSummary] = None
        if data.include_executive_summary:
            exec_summary = self._build_executive_summary(data, completeness, badge)

        # Step 8: Recommendations
        recommendations = self._generate_recommendations(
            data, section_results, completeness, badge
        )

        if badge == VerificationBadge.NOT_STARTED.value:
            warnings.append("Emission data verification not started.")
        if completeness < Decimal("50"):
            warnings.append("Report completeness below 50%.")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = CampaignReportingResult(
            entity_name=data.entity_name,
            reporting_year=data.reporting_year,
            report_completeness=completeness,
            verification_badge=badge,
            section_results=section_results,
            partner_outputs=partner_outputs,
            executive_summary=exec_summary,
            submission_ready=submission_ready,
            sections_complete=complete_count,
            sections_total=10,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _assess_sections(
        self, data: CampaignReportingInput, section_map: Dict[str, ReportSectionInput],
    ) -> List[ReportSectionResult]:
        results: List[ReportSectionResult] = []
        for sec in ReportSectionId:
            weight = SECTION_WEIGHTS.get(sec.value, Decimal("0.05"))
            label = SECTION_LABELS.get(sec.value, sec.value)

            if sec.value in section_map:
                score = section_map[sec.value].data_completeness
                available = section_map[sec.value].content_available
            else:
                score, available = self._auto_score_section(sec.value, data)

            # Partner mappings
            mappings: Dict[str, str] = {}
            if sec.value in CDP_MAPPING:
                mappings["cdp"] = CDP_MAPPING[sec.value]
            if sec.value in GFANZ_MAPPING:
                mappings["gfanz"] = GFANZ_MAPPING[sec.value]

            results.append(ReportSectionResult(
                section_id=sec.value,
                section_name=label,
                completeness=score,
                weight=weight,
                weighted_score=_round_val(score * weight, 4),
                data_available=available,
                partner_mappings=mappings,
            ))
        return results

    def _auto_score_section(
        self, sec_id: str, data: CampaignReportingInput,
    ) -> tuple:
        """Auto-score section from aggregate data."""
        score_map = {
            ReportSectionId.ENTITY_PROFILE.value: (Decimal("8"), True),
            ReportSectionId.PLEDGE_STATUS.value: (
                min(data.pledge_quality_score / Decimal("10"), Decimal("10")),
                data.pledge_quality_score > Decimal("0"),
            ),
            ReportSectionId.STARTING_LINE.value: (
                min(data.starting_line_score / Decimal("10"), Decimal("10")),
                data.starting_line_score > Decimal("0"),
            ),
            ReportSectionId.EMISSIONS_INVENTORY.value: (
                Decimal("8") if data.total_tco2e > Decimal("0") else Decimal("0"),
                data.total_tco2e > Decimal("0"),
            ),
            ReportSectionId.TARGET_PROGRESS.value: (
                Decimal("8") if data.cumulative_reduction_pct != Decimal("0") else Decimal("2"),
                data.target_tco2e > Decimal("0"),
            ),
            ReportSectionId.ACTION_PLAN.value: (
                min(data.action_plan_score / Decimal("10"), Decimal("10")),
                data.action_plan_score > Decimal("0"),
            ),
            ReportSectionId.SECTOR_ALIGNMENT.value: (
                min(data.sector_alignment_score / Decimal("10"), Decimal("10")),
                data.sector_alignment_score > Decimal("0"),
            ),
            ReportSectionId.HLEG_CREDIBILITY.value: (
                min(data.credibility_score / Decimal("10"), Decimal("10")),
                data.credibility_score > Decimal("0"),
            ),
            ReportSectionId.PARTNERSHIP.value: (
                min(data.partnership_score / Decimal("10"), Decimal("10")),
                data.partnership_score > Decimal("0"),
            ),
            ReportSectionId.FORWARD_COMMITMENTS.value: (
                Decimal("7") if data.forward_commitments else Decimal("2"),
                bool(data.forward_commitments),
            ),
        }
        return score_map.get(sec_id, (Decimal("0"), False))

    def _generate_partner_outputs(
        self, data: CampaignReportingInput, sections: List[ReportSectionResult],
    ) -> List[PartnerFormatOutput]:
        results: List[PartnerFormatOutput] = []
        for fmt in data.partner_formats:
            if fmt == PartnerFormat.CDP.value:
                mappings = CDP_MAPPING
                name = "CDP Climate Change Questionnaire"
            elif fmt == PartnerFormat.GFANZ.value:
                mappings = GFANZ_MAPPING
                name = "GFANZ Transition Plan"
            elif fmt == PartnerFormat.C40.value:
                mappings = {"action_plan": "CAP", "emissions_inventory": "GHG Inventory"}
                name = "C40 Deadline 2020"
            elif fmt == PartnerFormat.SBTI.value:
                mappings = {"target_progress": "Progress Report", "emissions_inventory": "GHG Data"}
                name = "SBTi Progress Report"
            else:
                mappings = {}
                name = "Universal Race to Zero Disclosure"

            mapped = sum(1 for s in sections if s.section_id in mappings)
            ready = mapped >= len(mappings) * 0.7 if mappings else True

            results.append(PartnerFormatOutput(
                format_id=fmt,
                format_name=name,
                sections_mapped=mapped,
                field_mappings=mappings,
                submission_ready=ready,
            ))
        return results

    def _build_executive_summary(
        self, data: CampaignReportingInput, completeness: Decimal, badge: str,
    ) -> ExecutiveSummary:
        achievements: List[str] = []
        challenges: List[str] = []

        if data.cumulative_reduction_pct > Decimal("0"):
            achievements.append(f"{data.cumulative_reduction_pct}% cumulative reduction achieved.")
        if data.credibility_tier in ("high", "moderate"):
            achievements.append(f"HLEG credibility tier: {data.credibility_tier.upper()}.")
        if badge == VerificationBadge.VERIFIED.value:
            achievements.append("Emission data third-party verified.")

        if data.progress_status in ("off_track", "critical"):
            challenges.append(f"Progress status: {data.progress_status.upper()}.")
        if data.credibility_tier in ("low", "critical"):
            challenges.append(f"HLEG credibility needs improvement: {data.credibility_tier}.")
        if completeness < Decimal("70"):
            challenges.append("Report completeness below target threshold.")

        return ExecutiveSummary(
            entity_name=data.entity_name,
            reporting_year=data.reporting_year,
            total_emissions_tco2e=data.total_tco2e,
            cumulative_reduction_pct=data.cumulative_reduction_pct,
            progress_status=data.progress_status,
            credibility_tier=data.credibility_tier,
            verification_badge=badge,
            key_achievements=achievements,
            key_challenges=challenges,
        )

    def _generate_recommendations(
        self, data: CampaignReportingInput, sections: List[ReportSectionResult],
        completeness: Decimal, badge: str,
    ) -> List[str]:
        recs: List[str] = []
        if completeness < Decimal("70"):
            recs.append("Increase report completeness to at least 70% for submission readiness.")
        if badge == VerificationBadge.NOT_STARTED.value:
            recs.append("Initiate third-party verification of emission data.")
        incomplete = [s for s in sections if s.completeness < Decimal("5")]
        for s in incomplete:
            recs.append(f"Complete '{s.section_name}' with required data.")
        return recs
