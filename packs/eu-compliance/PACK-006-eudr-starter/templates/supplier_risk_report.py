# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack: Supplier Risk Report Template
===========================================================

Generates a per-supplier risk assessment report covering supplier
profile, composite risk scores with trend analysis, country/supplier/
commodity/document risk breakdowns, geolocation summary, due diligence
status, and prioritized risk reduction recommendations.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

PACK_ID = "PACK-006-eudr-starter"
TEMPLATE_NAME = "supplier_risk_report"
TEMPLATE_VERSION = "1.0.0"


# =============================================================================
# ENUMS
# =============================================================================

class RiskClassification(str, Enum):
    """Risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskTrend(str, Enum):
    """Risk trend direction."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DETERIORATING = "DETERIORATING"


class CommodityType(str, Enum):
    """EUDR-regulated commodities."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    OIL_PALM = "OIL_PALM"
    RUBBER = "RUBBER"
    SOYA = "SOYA"
    WOOD = "WOOD"


class DDStatus(str, Enum):
    """Due diligence status."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    OVERDUE = "OVERDUE"
    REQUIRES_UPDATE = "REQUIRES_UPDATE"


class RecommendationPriority(str, Enum):
    """Recommendation priority."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SupplierProfile(BaseModel):
    """Section 1: Supplier profile information."""
    supplier_id: str = Field(..., description="Unique supplier identifier")
    supplier_name: str = Field(..., description="Legal entity name")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(..., description="Full country name")
    commodities: List[CommodityType] = Field(
        default_factory=list, description="Commodities supplied"
    )
    tier_level: int = Field(1, ge=1, le=10, description="Supply chain tier")
    eori_number: Optional[str] = Field(None, description="EORI number")
    contact_name: Optional[str] = Field(None, description="Primary contact")
    contact_email: Optional[str] = Field(None, description="Contact email")
    onboarded_date: Optional[date] = Field(None, description="Date onboarded")
    active: bool = Field(True, description="Whether supplier is active")


class RiskScoreSummary(BaseModel):
    """Section 2: Risk score summary with trend."""
    composite_score: float = Field(
        ..., ge=0.0, le=100.0, description="Composite risk score"
    )
    classification: RiskClassification = Field(
        ..., description="Risk classification"
    )
    trend: RiskTrend = Field(RiskTrend.STABLE, description="Risk trend")
    previous_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Previous assessment score"
    )
    assessment_date: date = Field(
        default_factory=date.today, description="Assessment date"
    )
    next_review_date: Optional[date] = Field(None, description="Next review date")


class CountryRiskDetail(BaseModel):
    """Section 3: Country risk details."""
    country_iso: str = Field(..., description="ISO 3166-1 alpha-2")
    country_name: str = Field(..., description="Country name")
    benchmark_classification: str = Field(..., description="LOW/STANDARD/HIGH")
    deforestation_rate_pct: Optional[float] = Field(
        None, ge=0.0, description="Annual deforestation rate %"
    )
    governance_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Governance effectiveness score"
    )
    corruption_index: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Corruption perception index"
    )
    country_risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Country risk score"
    )


class SupplierRiskDetail(BaseModel):
    """Section 4: Supplier-specific risk details."""
    certification_status: Optional[str] = Field(
        None, description="Current certification status"
    )
    certification_schemes: List[str] = Field(
        default_factory=list, description="Active certification schemes"
    )
    compliance_history_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Historical compliance score"
    )
    engagement_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Supplier engagement score"
    )
    years_in_relationship: int = Field(
        0, ge=0, description="Years of supplier relationship"
    )
    past_incidents: int = Field(0, ge=0, description="Past compliance incidents")
    supplier_risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Supplier risk score"
    )


class CommodityRiskEntry(BaseModel):
    """Risk detail for a single commodity."""
    commodity: CommodityType = Field(..., description="Commodity type")
    deforestation_correlation: float = Field(
        0.0, ge=0.0, le=100.0, description="Deforestation correlation score"
    )
    supply_chain_risk: float = Field(
        0.0, ge=0.0, le=100.0, description="Supply chain risk score"
    )
    volume_kg: Optional[float] = Field(None, ge=0, description="Volume in kg")
    commodity_risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Commodity risk score"
    )


class DocumentRiskDetail(BaseModel):
    """Section 6: Document risk details."""
    documentation_completeness_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Documentation completeness"
    )
    certificate_validity_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Certificate validity"
    )
    permit_status: str = Field("N/A", description="Permit status summary")
    documents_total: int = Field(0, ge=0, description="Total documents required")
    documents_valid: int = Field(0, ge=0, description="Documents valid")
    documents_expired: int = Field(0, ge=0, description="Documents expired")
    documents_missing: int = Field(0, ge=0, description="Documents missing")
    document_risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Document risk score"
    )


class GeolocationSummary(BaseModel):
    """Section 7: Geolocation summary."""
    total_plots: int = Field(0, ge=0, description="Total plots")
    total_area_ha: float = Field(0.0, ge=0.0, description="Total area in hectares")
    countries: List[str] = Field(
        default_factory=list, description="Countries with plots"
    )
    overlap_issues: int = Field(0, ge=0, description="Overlap issues detected")
    validation_status: str = Field("PENDING", description="Overall validation status")


class DDStatusDetail(BaseModel):
    """Section 8: Due diligence status."""
    status: DDStatus = Field(DDStatus.NOT_STARTED, description="DD status")
    completion_pct: float = Field(0.0, ge=0.0, le=100.0, description="Completion %")
    started_date: Optional[date] = Field(None, description="DD start date")
    completed_date: Optional[date] = Field(None, description="DD completion date")
    outstanding_items: List[str] = Field(
        default_factory=list, description="Outstanding DD items"
    )
    assessor: Optional[str] = Field(None, description="DD assessor")


class Recommendation(BaseModel):
    """Section 9: Risk reduction recommendation."""
    recommendation_id: str = Field(..., description="Recommendation identifier")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    priority: RecommendationPriority = Field(..., description="Priority level")
    category: str = Field("", description="Recommendation category")
    estimated_risk_reduction: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Estimated risk reduction points"
    )
    deadline: Optional[date] = Field(None, description="Recommended deadline")
    owner: Optional[str] = Field(None, description="Responsible party")


class SupplierRiskReportInput(BaseModel):
    """Complete input data for the Supplier Risk Report."""
    report_date: date = Field(
        default_factory=date.today, description="Report generation date"
    )
    profile: SupplierProfile = Field(..., description="Supplier profile")
    risk_summary: RiskScoreSummary = Field(..., description="Risk score summary")
    country_risk: CountryRiskDetail = Field(..., description="Country risk detail")
    supplier_risk: SupplierRiskDetail = Field(..., description="Supplier risk detail")
    commodity_risks: List[CommodityRiskEntry] = Field(
        default_factory=list, description="Per-commodity risk details"
    )
    document_risk: DocumentRiskDetail = Field(
        default_factory=DocumentRiskDetail, description="Document risk detail"
    )
    geolocation: GeolocationSummary = Field(
        default_factory=GeolocationSummary, description="Geolocation summary"
    )
    dd_status: DDStatusDetail = Field(
        default_factory=DDStatusDetail, description="DD status"
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list, description="Recommendations"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _risk_badge(classification: RiskClassification) -> str:
    """Text badge for risk classification."""
    return f"[{classification.value}]"


def _risk_css(classification: RiskClassification) -> str:
    """Inline CSS for risk classification."""
    mapping = {
        RiskClassification.LOW: "color:#1a7f37;font-weight:bold;",
        RiskClassification.MEDIUM: "color:#b08800;font-weight:bold;",
        RiskClassification.HIGH: "color:#e36209;font-weight:bold;",
        RiskClassification.CRITICAL: "color:#cf222e;font-weight:bold;",
    }
    return mapping.get(classification, "")


def _trend_badge(trend: RiskTrend) -> str:
    """Text indicator for risk trend."""
    mapping = {
        RiskTrend.IMPROVING: "[IMPROVING v]",
        RiskTrend.STABLE: "[STABLE =]",
        RiskTrend.DETERIORATING: "[DETERIORATING ^]",
    }
    return mapping.get(trend, "[UNKNOWN]")


def _trend_css(trend: RiskTrend) -> str:
    """Inline CSS for risk trend."""
    mapping = {
        RiskTrend.IMPROVING: "color:#1a7f37;",
        RiskTrend.STABLE: "color:#b08800;",
        RiskTrend.DETERIORATING: "color:#cf222e;",
    }
    return mapping.get(trend, "")


def _score_label(score: float) -> str:
    """Score to human-readable label."""
    if score <= 20:
        return "LOW"
    if score <= 40:
        return "MODERATE"
    if score <= 60:
        return "ELEVATED"
    if score <= 80:
        return "HIGH"
    return "CRITICAL"


def _fmt_commodity(commodity: CommodityType) -> str:
    """Human-readable commodity name."""
    mapping = {
        CommodityType.CATTLE: "Cattle",
        CommodityType.COCOA: "Cocoa",
        CommodityType.COFFEE: "Coffee",
        CommodityType.OIL_PALM: "Oil Palm",
        CommodityType.RUBBER: "Rubber",
        CommodityType.SOYA: "Soya",
        CommodityType.WOOD: "Wood",
    }
    return mapping.get(commodity, commodity.value)


def _fmt_volume(kg: Optional[float]) -> str:
    """Format volume."""
    if kg is None:
        return "N/A"
    if kg >= 1_000_000:
        return f"{kg / 1_000:,.0f} t"
    if kg >= 1_000:
        return f"{kg / 1_000:,.1f} t"
    return f"{kg:,.0f} kg"


def _priority_sort(priority: RecommendationPriority) -> int:
    """Numeric sort key for priority."""
    return {
        RecommendationPriority.CRITICAL: 0,
        RecommendationPriority.HIGH: 1,
        RecommendationPriority.MEDIUM: 2,
        RecommendationPriority.LOW: 3,
    }.get(priority, 99)


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class SupplierRiskReport:
    """Generate per-supplier risk assessment report.

    Sections:
        1. Supplier Profile - Name, country, commodities, tier
        2. Risk Score Summary - Composite score, trend, classification
        3. Country Risk Detail - Benchmark, deforestation, governance
        4. Supplier Risk Detail - Certification, history, engagement
        5. Commodity Risk Detail - Per-commodity deforestation correlation
        6. Document Risk Detail - Completeness, validity, permits
        7. Geolocation Summary - Plots, area, countries, overlaps
        8. DD Status - Current status, completion, outstanding items
        9. Recommendations - Prioritized risk reduction actions

    Example:
        >>> report = SupplierRiskReport()
        >>> data = SupplierRiskReportInput(...)
        >>> md = report.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize the Supplier Risk Report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: SupplierRiskReportInput) -> str:
        """Render the supplier risk report as Markdown.

        Args:
            data: Validated supplier risk input data.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_supplier_profile(data),
            self._md_risk_summary(data),
            self._md_country_risk(data),
            self._md_supplier_risk(data),
            self._md_commodity_risk(data),
            self._md_document_risk(data),
            self._md_geolocation(data),
            self._md_dd_status(data),
            self._md_recommendations(data),
            self._md_provenance(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: SupplierRiskReportInput) -> str:
        """Render the supplier risk report as HTML.

        Args:
            data: Validated supplier risk input data.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_supplier_profile(data),
            self._html_risk_summary(data),
            self._html_country_risk(data),
            self._html_supplier_risk(data),
            self._html_commodity_risk(data),
            self._html_document_risk(data),
            self._html_geolocation(data),
            self._html_dd_status(data),
            self._html_recommendations(data),
            self._html_provenance(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: SupplierRiskReportInput) -> Dict[str, Any]:
        """Render as JSON-serializable dictionary.

        Args:
            data: Validated supplier risk input data.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance_hash = self._compute_provenance_hash(data)

        return {
            "metadata": {
                "pack_id": PACK_ID,
                "template_name": TEMPLATE_NAME,
                "version": TEMPLATE_VERSION,
                "generated_at": self._render_timestamp.isoformat(),
                "provenance_hash": provenance_hash,
            },
            "report_date": data.report_date.isoformat(),
            "profile": data.profile.model_dump(mode="json"),
            "risk_summary": data.risk_summary.model_dump(mode="json"),
            "country_risk": data.country_risk.model_dump(mode="json"),
            "supplier_risk": data.supplier_risk.model_dump(mode="json"),
            "commodity_risks": [
                c.model_dump(mode="json") for c in data.commodity_risks
            ],
            "document_risk": data.document_risk.model_dump(mode="json"),
            "geolocation": data.geolocation.model_dump(mode="json"),
            "dd_status": data.dd_status.model_dump(mode="json"),
            "recommendations": [
                r.model_dump(mode="json") for r in data.recommendations
            ],
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance_hash(self, data: SupplierRiskReportInput) -> str:
        """Compute SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: SupplierRiskReportInput) -> str:
        """Report header."""
        p = data.profile
        rs = data.risk_summary
        return (
            f"# Supplier Risk Report: {p.supplier_name}\n"
            f"**Supplier ID:** {p.supplier_id} | "
            f"**Country:** {p.country_name} ({p.country}) | "
            f"**Risk:** {_risk_badge(rs.classification)} "
            f"({rs.composite_score:.1f}/100)\n"
            f"**Report Date:** {data.report_date.isoformat()}\n\n---"
        )

    def _md_supplier_profile(self, data: SupplierRiskReportInput) -> str:
        """Section 1: Supplier Profile."""
        p = data.profile
        commodities = ", ".join(_fmt_commodity(c) for c in p.commodities) or "N/A"
        eori = p.eori_number or "N/A"
        contact = p.contact_name or "N/A"
        email = p.contact_email or "N/A"
        onboarded = p.onboarded_date.isoformat() if p.onboarded_date else "N/A"
        active = "Active" if p.active else "Inactive"
        return (
            "## 1. Supplier Profile\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Supplier Name | {p.supplier_name} |\n"
            f"| Supplier ID | {p.supplier_id} |\n"
            f"| Country | {p.country_name} ({p.country}) |\n"
            f"| Commodities | {commodities} |\n"
            f"| Tier Level | {p.tier_level} |\n"
            f"| EORI Number | {eori} |\n"
            f"| Contact | {contact} |\n"
            f"| Email | {email} |\n"
            f"| Onboarded | {onboarded} |\n"
            f"| Status | {active} |"
        )

    def _md_risk_summary(self, data: SupplierRiskReportInput) -> str:
        """Section 2: Risk Score Summary."""
        rs = data.risk_summary
        prev = f"{rs.previous_score:.1f}" if rs.previous_score is not None else "N/A"
        change = ""
        if rs.previous_score is not None:
            delta = rs.composite_score - rs.previous_score
            sign = "+" if delta > 0 else ""
            change = f" ({sign}{delta:.1f})"
        next_rev = rs.next_review_date.isoformat() if rs.next_review_date else "N/A"
        return (
            "## 2. Risk Score Summary\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Composite Score | **{rs.composite_score:.1f}/100** "
            f"({_score_label(rs.composite_score)}) |\n"
            f"| Classification | {_risk_badge(rs.classification)} |\n"
            f"| Trend | {_trend_badge(rs.trend)} |\n"
            f"| Previous Score | {prev}{change} |\n"
            f"| Assessment Date | {rs.assessment_date.isoformat()} |\n"
            f"| Next Review | {next_rev} |"
        )

    def _md_country_risk(self, data: SupplierRiskReportInput) -> str:
        """Section 3: Country Risk Detail."""
        cr = data.country_risk
        deforest = f"{cr.deforestation_rate_pct:.2f}%" if cr.deforestation_rate_pct is not None else "N/A"
        governance = f"{cr.governance_score:.1f}/100" if cr.governance_score is not None else "N/A"
        corruption = f"{cr.corruption_index:.1f}/100" if cr.corruption_index is not None else "N/A"
        return (
            "## 3. Country Risk Detail\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Country | {cr.country_name} ({cr.country_iso}) |\n"
            f"| Benchmark | [{cr.benchmark_classification}] |\n"
            f"| Deforestation Rate | {deforest} |\n"
            f"| Governance Score | {governance} |\n"
            f"| Corruption Index | {corruption} |\n"
            f"| Country Risk Score | {cr.country_risk_score:.1f}/100 "
            f"({_score_label(cr.country_risk_score)}) |"
        )

    def _md_supplier_risk(self, data: SupplierRiskReportInput) -> str:
        """Section 4: Supplier Risk Detail."""
        sr = data.supplier_risk
        cert = sr.certification_status or "N/A"
        schemes = ", ".join(sr.certification_schemes) if sr.certification_schemes else "N/A"
        return (
            "## 4. Supplier Risk Detail\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Certification Status | {cert} |\n"
            f"| Certification Schemes | {schemes} |\n"
            f"| Compliance History | {sr.compliance_history_score:.1f}/100 |\n"
            f"| Engagement Score | {sr.engagement_score:.1f}/100 |\n"
            f"| Years in Relationship | {sr.years_in_relationship} |\n"
            f"| Past Incidents | {sr.past_incidents} |\n"
            f"| Supplier Risk Score | {sr.supplier_risk_score:.1f}/100 "
            f"({_score_label(sr.supplier_risk_score)}) |"
        )

    def _md_commodity_risk(self, data: SupplierRiskReportInput) -> str:
        """Section 5: Commodity Risk Detail."""
        lines = [
            "## 5. Commodity Risk Detail\n",
            "| Commodity | Deforestation Corr. | Supply Chain Risk | Volume | Score |",
            "|-----------|--------------------|--------------------|--------|-------|",
        ]
        for c in data.commodity_risks:
            volume = _fmt_volume(c.volume_kg)
            lines.append(
                f"| {_fmt_commodity(c.commodity)} | {c.deforestation_correlation:.1f} "
                f"| {c.supply_chain_risk:.1f} | {volume} "
                f"| {c.commodity_risk_score:.1f} |"
            )
        if not data.commodity_risks:
            lines.append("| - | No commodity data | - | - | - |")
        return "\n".join(lines)

    def _md_document_risk(self, data: SupplierRiskReportInput) -> str:
        """Section 6: Document Risk Detail."""
        dr = data.document_risk
        return (
            "## 6. Document Risk Detail\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Documentation Completeness | {dr.documentation_completeness_pct:.1f}% |\n"
            f"| Certificate Validity | {dr.certificate_validity_pct:.1f}% |\n"
            f"| Permit Status | {dr.permit_status} |\n"
            f"| Documents Total | {dr.documents_total} |\n"
            f"| Documents Valid | {dr.documents_valid} |\n"
            f"| Documents Expired | {dr.documents_expired} |\n"
            f"| Documents Missing | {dr.documents_missing} |\n"
            f"| Document Risk Score | {dr.document_risk_score:.1f}/100 "
            f"({_score_label(dr.document_risk_score)}) |"
        )

    def _md_geolocation(self, data: SupplierRiskReportInput) -> str:
        """Section 7: Geolocation Summary."""
        g = data.geolocation
        countries = ", ".join(g.countries) if g.countries else "N/A"
        return (
            "## 7. Geolocation Summary\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Total Plots | {g.total_plots} |\n"
            f"| Total Area | {g.total_area_ha:,.1f} ha |\n"
            f"| Countries | {countries} |\n"
            f"| Overlap Issues | {g.overlap_issues} |\n"
            f"| Validation Status | {g.validation_status} |"
        )

    def _md_dd_status(self, data: SupplierRiskReportInput) -> str:
        """Section 8: DD Status."""
        dd = data.dd_status
        started = dd.started_date.isoformat() if dd.started_date else "N/A"
        completed = dd.completed_date.isoformat() if dd.completed_date else "N/A"
        assessor = dd.assessor or "Unassigned"
        items = (
            "\n".join(f"- {item}" for item in dd.outstanding_items)
            if dd.outstanding_items
            else "- None"
        )
        return (
            "## 8. Due Diligence Status\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Status | [{dd.status.value}] |\n"
            f"| Completion | {dd.completion_pct:.0f}% |\n"
            f"| Started | {started} |\n"
            f"| Completed | {completed} |\n"
            f"| Assessor | {assessor} |\n\n"
            f"**Outstanding Items:**\n\n{items}"
        )

    def _md_recommendations(self, data: SupplierRiskReportInput) -> str:
        """Section 9: Recommendations."""
        sorted_recs = sorted(
            data.recommendations, key=lambda r: _priority_sort(r.priority)
        )
        lines = [
            "## 9. Recommendations\n",
            "| # | Priority | Title | Category | Risk Reduction | Deadline | Owner |",
            "|---|----------|-------|----------|---------------|----------|-------|",
        ]
        for idx, r in enumerate(sorted_recs, 1):
            reduction = (
                f"{r.estimated_risk_reduction:.0f} pts"
                if r.estimated_risk_reduction is not None
                else "N/A"
            )
            deadline = r.deadline.isoformat() if r.deadline else "TBD"
            owner = r.owner or "TBD"
            lines.append(
                f"| {idx} | [{r.priority.value}] | {r.title} "
                f"| {r.category} | {reduction} | {deadline} | {owner} |"
            )
        if not data.recommendations:
            lines.append("| - | No recommendations | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_provenance(self, data: SupplierRiskReportInput) -> str:
        """Provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, data: SupplierRiskReportInput, body: str) -> str:
        """Wrap body in HTML document."""
        p = data.profile
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Supplier Risk Report - {p.supplier_name}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "max-width:1100px;color:#222;line-height:1.5;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "h1{color:#1a365d;border-bottom:3px solid #2b6cb0;padding-bottom:0.5rem;}\n"
            "h2{color:#2b6cb0;margin-top:2rem;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".risk-low{color:#1a7f37;font-weight:bold;}\n"
            ".risk-medium{color:#b08800;font-weight:bold;}\n"
            ".risk-high{color:#e36209;font-weight:bold;}\n"
            ".risk-critical{color:#cf222e;font-weight:bold;}\n"
            ".trend-improving{color:#1a7f37;}\n"
            ".trend-stable{color:#b08800;}\n"
            ".trend-deteriorating{color:#cf222e;}\n"
            ".score-bar{height:18px;border-radius:4px;display:inline-block;}\n"
            ".provenance{font-size:0.85rem;color:#666;}\n"
            "code{background:#f5f5f5;padding:0.2rem 0.4rem;border-radius:3px;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: SupplierRiskReportInput) -> str:
        """HTML report header."""
        p = data.profile
        rs = data.risk_summary
        css = _risk_css(rs.classification)
        return (
            '<div class="section">\n'
            f"<h1>Supplier Risk Report &mdash; {p.supplier_name}</h1>\n"
            f"<p><strong>Supplier ID:</strong> {p.supplier_id} | "
            f"<strong>Country:</strong> {p.country_name} ({p.country}) | "
            f'<strong>Risk:</strong> <span style="{css}">'
            f"{rs.classification.value} ({rs.composite_score:.1f}/100)</span></p>\n"
            f"<p><strong>Report Date:</strong> "
            f"{data.report_date.isoformat()}</p>\n<hr>\n</div>"
        )

    def _html_supplier_profile(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 1: Supplier Profile."""
        p = data.profile
        commodities = ", ".join(_fmt_commodity(c) for c in p.commodities) or "N/A"
        eori = p.eori_number or "N/A"
        contact = p.contact_name or "N/A"
        email = p.contact_email or "N/A"
        onboarded = p.onboarded_date.isoformat() if p.onboarded_date else "N/A"
        active = "Active" if p.active else "Inactive"
        return (
            '<div class="section">\n<h2>1. Supplier Profile</h2>\n'
            "<table><tbody>"
            f"<tr><th>Supplier Name</th><td>{p.supplier_name}</td></tr>"
            f"<tr><th>Supplier ID</th><td>{p.supplier_id}</td></tr>"
            f"<tr><th>Country</th><td>{p.country_name} ({p.country})</td></tr>"
            f"<tr><th>Commodities</th><td>{commodities}</td></tr>"
            f"<tr><th>Tier Level</th><td>{p.tier_level}</td></tr>"
            f"<tr><th>EORI Number</th><td>{eori}</td></tr>"
            f"<tr><th>Contact</th><td>{contact}</td></tr>"
            f"<tr><th>Email</th><td>{email}</td></tr>"
            f"<tr><th>Onboarded</th><td>{onboarded}</td></tr>"
            f"<tr><th>Status</th><td>{active}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_risk_summary(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 2: Risk Score Summary."""
        rs = data.risk_summary
        css_risk = _risk_css(rs.classification)
        css_trend = _trend_css(rs.trend)
        prev = f"{rs.previous_score:.1f}" if rs.previous_score is not None else "N/A"
        next_rev = rs.next_review_date.isoformat() if rs.next_review_date else "N/A"
        return (
            '<div class="section">\n<h2>2. Risk Score Summary</h2>\n'
            "<table><tbody>"
            f'<tr><th>Composite Score</th><td style="{css_risk}">'
            f"<strong>{rs.composite_score:.1f}/100</strong> "
            f"({_score_label(rs.composite_score)})</td></tr>"
            f'<tr><th>Classification</th><td style="{css_risk}">'
            f"{rs.classification.value}</td></tr>"
            f'<tr><th>Trend</th><td style="{css_trend}">'
            f"{rs.trend.value}</td></tr>"
            f"<tr><th>Previous Score</th><td>{prev}</td></tr>"
            f"<tr><th>Assessment Date</th>"
            f"<td>{rs.assessment_date.isoformat()}</td></tr>"
            f"<tr><th>Next Review</th><td>{next_rev}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_country_risk(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 3: Country Risk Detail."""
        cr = data.country_risk
        deforest = f"{cr.deforestation_rate_pct:.2f}%" if cr.deforestation_rate_pct is not None else "N/A"
        governance = f"{cr.governance_score:.1f}/100" if cr.governance_score is not None else "N/A"
        corruption = f"{cr.corruption_index:.1f}/100" if cr.corruption_index is not None else "N/A"
        return (
            '<div class="section">\n<h2>3. Country Risk Detail</h2>\n'
            "<table><tbody>"
            f"<tr><th>Country</th><td>{cr.country_name} ({cr.country_iso})</td></tr>"
            f"<tr><th>Benchmark</th><td>{cr.benchmark_classification}</td></tr>"
            f"<tr><th>Deforestation Rate</th><td>{deforest}</td></tr>"
            f"<tr><th>Governance Score</th><td>{governance}</td></tr>"
            f"<tr><th>Corruption Index</th><td>{corruption}</td></tr>"
            f"<tr><th>Country Risk Score</th><td>{cr.country_risk_score:.1f}/100</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_supplier_risk(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 4: Supplier Risk Detail."""
        sr = data.supplier_risk
        cert = sr.certification_status or "N/A"
        schemes = ", ".join(sr.certification_schemes) if sr.certification_schemes else "N/A"
        return (
            '<div class="section">\n<h2>4. Supplier Risk Detail</h2>\n'
            "<table><tbody>"
            f"<tr><th>Certification Status</th><td>{cert}</td></tr>"
            f"<tr><th>Certification Schemes</th><td>{schemes}</td></tr>"
            f"<tr><th>Compliance History</th><td>{sr.compliance_history_score:.1f}/100</td></tr>"
            f"<tr><th>Engagement Score</th><td>{sr.engagement_score:.1f}/100</td></tr>"
            f"<tr><th>Years in Relationship</th><td>{sr.years_in_relationship}</td></tr>"
            f"<tr><th>Past Incidents</th><td>{sr.past_incidents}</td></tr>"
            f"<tr><th>Supplier Risk Score</th><td>{sr.supplier_risk_score:.1f}/100</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_commodity_risk(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 5: Commodity Risk Detail."""
        rows = ""
        for c in data.commodity_risks:
            volume = _fmt_volume(c.volume_kg)
            rows += (
                f"<tr><td>{_fmt_commodity(c.commodity)}</td>"
                f"<td>{c.deforestation_correlation:.1f}</td>"
                f"<td>{c.supply_chain_risk:.1f}</td>"
                f"<td>{volume}</td><td>{c.commodity_risk_score:.1f}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="5">No commodity data</td></tr>'
        return (
            '<div class="section">\n<h2>5. Commodity Risk Detail</h2>\n'
            "<table><thead><tr><th>Commodity</th><th>Deforestation Corr.</th>"
            "<th>Supply Chain Risk</th><th>Volume</th>"
            f"<th>Score</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_document_risk(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 6: Document Risk Detail."""
        dr = data.document_risk
        return (
            '<div class="section">\n<h2>6. Document Risk Detail</h2>\n'
            "<table><tbody>"
            f"<tr><th>Documentation Completeness</th><td>{dr.documentation_completeness_pct:.1f}%</td></tr>"
            f"<tr><th>Certificate Validity</th><td>{dr.certificate_validity_pct:.1f}%</td></tr>"
            f"<tr><th>Permit Status</th><td>{dr.permit_status}</td></tr>"
            f"<tr><th>Documents Total</th><td>{dr.documents_total}</td></tr>"
            f"<tr><th>Documents Valid</th><td>{dr.documents_valid}</td></tr>"
            f"<tr><th>Documents Expired</th><td>{dr.documents_expired}</td></tr>"
            f"<tr><th>Documents Missing</th><td>{dr.documents_missing}</td></tr>"
            f"<tr><th>Document Risk Score</th><td>{dr.document_risk_score:.1f}/100</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_geolocation(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 7: Geolocation Summary."""
        g = data.geolocation
        countries = ", ".join(g.countries) if g.countries else "N/A"
        return (
            '<div class="section">\n<h2>7. Geolocation Summary</h2>\n'
            "<table><tbody>"
            f"<tr><th>Total Plots</th><td>{g.total_plots}</td></tr>"
            f"<tr><th>Total Area</th><td>{g.total_area_ha:,.1f} ha</td></tr>"
            f"<tr><th>Countries</th><td>{countries}</td></tr>"
            f"<tr><th>Overlap Issues</th><td>{g.overlap_issues}</td></tr>"
            f"<tr><th>Validation Status</th><td>{g.validation_status}</td></tr>"
            "</tbody></table>\n</div>"
        )

    def _html_dd_status(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 8: DD Status."""
        dd = data.dd_status
        started = dd.started_date.isoformat() if dd.started_date else "N/A"
        completed = dd.completed_date.isoformat() if dd.completed_date else "N/A"
        assessor = dd.assessor or "Unassigned"
        items_html = "".join(f"<li>{item}</li>" for item in dd.outstanding_items)
        items_list = f"<ul>{items_html}</ul>" if items_html else "<p>None</p>"
        return (
            '<div class="section">\n<h2>8. Due Diligence Status</h2>\n'
            "<table><tbody>"
            f"<tr><th>Status</th><td>{dd.status.value}</td></tr>"
            f"<tr><th>Completion</th><td>{dd.completion_pct:.0f}%</td></tr>"
            f"<tr><th>Started</th><td>{started}</td></tr>"
            f"<tr><th>Completed</th><td>{completed}</td></tr>"
            f"<tr><th>Assessor</th><td>{assessor}</td></tr>"
            "</tbody></table>\n"
            f"<h3>Outstanding Items</h3>\n{items_list}\n</div>"
        )

    def _html_recommendations(self, data: SupplierRiskReportInput) -> str:
        """HTML Section 9: Recommendations."""
        sorted_recs = sorted(
            data.recommendations, key=lambda r: _priority_sort(r.priority)
        )
        rows = ""
        for idx, r in enumerate(sorted_recs, 1):
            reduction = (
                f"{r.estimated_risk_reduction:.0f} pts"
                if r.estimated_risk_reduction is not None
                else "N/A"
            )
            deadline = r.deadline.isoformat() if r.deadline else "TBD"
            owner = r.owner or "TBD"
            rows += (
                f"<tr><td>{idx}</td><td>{r.priority.value}</td>"
                f"<td>{r.title}</td><td>{r.category}</td>"
                f"<td>{reduction}</td><td>{deadline}</td><td>{owner}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="7">No recommendations</td></tr>'
        return (
            '<div class="section">\n<h2>9. Recommendations</h2>\n'
            "<table><thead><tr><th>#</th><th>Priority</th><th>Title</th>"
            "<th>Category</th><th>Risk Reduction</th><th>Deadline</th>"
            f"<th>Owner</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_provenance(self, data: SupplierRiskReportInput) -> str:
        """HTML provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section provenance">\n<hr>\n'
            f"<p>Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} "
            f"| {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
