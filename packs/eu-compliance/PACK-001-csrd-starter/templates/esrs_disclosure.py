# -*- coding: utf-8 -*-
"""
PACK-001 Phase 3: ESRS Disclosure Template
===========================================

Generates full ESRS disclosure narratives for all 12 standards (E1-E5,
S1-S4, G1, ESRS-1, ESRS-2). Each standard section includes disclosure
requirements met/unmet, metric values with provenance, narrative
sections, cross-references, and data quality indicators.

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


# =============================================================================
# ENUMS
# =============================================================================

class ESRSStandardId(str, Enum):
    """Identifiers for ESRS standards."""
    ESRS_1 = "ESRS 1"
    ESRS_2 = "ESRS 2"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    G1 = "G1"


ESRS_STANDARD_NAMES: Dict[ESRSStandardId, str] = {
    ESRSStandardId.ESRS_1: "General Requirements",
    ESRSStandardId.ESRS_2: "General Disclosures",
    ESRSStandardId.E1: "Climate Change",
    ESRSStandardId.E2: "Pollution",
    ESRSStandardId.E3: "Water and Marine Resources",
    ESRSStandardId.E4: "Biodiversity and Ecosystems",
    ESRSStandardId.E5: "Resource Use and Circular Economy",
    ESRSStandardId.S1: "Own Workforce",
    ESRSStandardId.S2: "Workers in the Value Chain",
    ESRSStandardId.S3: "Affected Communities",
    ESRSStandardId.S4: "Consumers and End-users",
    ESRSStandardId.G1: "Business Conduct",
}


class DisclosureStatus(str, Enum):
    """Status of a disclosure requirement."""
    MET = "MET"
    PARTIALLY_MET = "PARTIALLY_MET"
    NOT_MET = "NOT_MET"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DataQualityLevel(str, Enum):
    """Data quality tier following ESRS guidance."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    ESTIMATED = "ESTIMATED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class NarrativeReviewStatus(str, Enum):
    """Review status for AI-generated narrative content."""
    DRAFT = "DRAFT"
    PENDING_REVIEW = "PENDING_REVIEW"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CrossReference(BaseModel):
    """Cross-reference to another ESRS standard or external framework."""
    target_standard: str = Field(..., description="Target standard or framework")
    target_section: str = Field(..., description="Specific section or paragraph")
    relationship: str = Field("related", description="Relationship type")
    description: Optional[str] = Field(None, description="Brief description of the link")


class DataQualityIndicator(BaseModel):
    """Data quality assessment for a metric or disclosure."""
    quality_level: DataQualityLevel = Field(
        DataQualityLevel.NOT_AVAILABLE, description="Quality tier"
    )
    completeness_pct: float = Field(100.0, ge=0.0, le=100.0, description="Data completeness %")
    source_description: str = Field("", description="Data source description")
    estimation_methodology: Optional[str] = Field(None, description="Estimation method if applicable")
    last_updated: Optional[date] = Field(None, description="Last data update date")


class MetricValue(BaseModel):
    """A single metric value with provenance and quality metadata."""
    metric_id: str = Field(..., description="Metric identifier (e.g. E1-6 para 48a)")
    metric_name: str = Field(..., description="Human-readable metric name")
    value: Optional[float] = Field(None, description="Numeric value (None if not reported)")
    unit: str = Field("", description="Unit of measurement (tCO2e, EUR, %, etc.)")
    reporting_year: Optional[int] = Field(None, description="Year the value pertains to")
    comparative_value: Optional[float] = Field(None, description="Prior year value for comparison")
    xbrl_tag: Optional[str] = Field(None, description="ESRS XBRL taxonomy tag reference")
    data_quality: DataQualityIndicator = Field(
        default_factory=DataQualityIndicator, description="Quality metadata"
    )
    provenance_hash: Optional[str] = Field(None, description="SHA-256 of source calculation")

    def formatted_value(self) -> str:
        """Return human-readable formatted value string."""
        if self.value is None:
            return "Not reported"
        if self.unit == "%":
            return f"{self.value:.1f}%"
        if self.unit in ("EUR", "USD", "GBP"):
            if abs(self.value) >= 1_000_000:
                return f"{self.unit} {self.value / 1_000_000:,.1f}M"
            return f"{self.unit} {self.value:,.0f}"
        if self.unit == "tCO2e":
            if abs(self.value) >= 1_000_000:
                return f"{self.value / 1_000_000:,.1f}M tCO2e"
            return f"{self.value:,.1f} tCO2e"
        return f"{self.value:,.2f} {self.unit}".strip()

    def yoy_change(self) -> Optional[str]:
        """Return year-over-year change string if comparative data exists."""
        if self.value is None or self.comparative_value is None:
            return None
        if self.comparative_value == 0:
            return "N/A (base year zero)"
        change_pct = ((self.value - self.comparative_value) / abs(self.comparative_value)) * 100
        sign = "+" if change_pct > 0 else ""
        return f"{sign}{change_pct:.1f}%"


class DisclosureRequirement(BaseModel):
    """A single disclosure requirement within an ESRS standard."""
    requirement_id: str = Field(..., description="Requirement identifier (e.g. E1-1, DR-1)")
    requirement_name: str = Field(..., description="Requirement name")
    paragraph_ref: Optional[str] = Field(None, description="ESRS paragraph reference")
    status: DisclosureStatus = Field(..., description="Met/unmet status")
    metrics: List[MetricValue] = Field(default_factory=list, description="Related metric values")
    narrative: Optional[str] = Field(None, description="Narrative disclosure text")
    narrative_review_status: NarrativeReviewStatus = Field(
        NarrativeReviewStatus.DRAFT, description="Review status if narrative is AI-generated"
    )
    cross_references: List[CrossReference] = Field(
        default_factory=list, description="Cross-references"
    )
    evidence_references: List[str] = Field(
        default_factory=list, description="Supporting evidence references"
    )


class StandardDisclosure(BaseModel):
    """Complete disclosure for a single ESRS standard."""
    standard_id: ESRSStandardId = Field(..., description="ESRS standard identifier")
    standard_name: str = Field("", description="Full standard name")
    is_material: bool = Field(True, description="Whether standard is material for the entity")
    materiality_rationale: Optional[str] = Field(None, description="Rationale for materiality decision")
    requirements: List[DisclosureRequirement] = Field(
        default_factory=list, description="Disclosure requirements"
    )
    summary_narrative: Optional[str] = Field(None, description="High-level standard summary")

    def model_post_init(self, __context: Any) -> None:
        """Auto-fill standard name from lookup if empty."""
        if not self.standard_name:
            self.standard_name = ESRS_STANDARD_NAMES.get(self.standard_id, self.standard_id.value)

    @property
    def requirements_met(self) -> int:
        """Count of requirements with MET status."""
        return sum(1 for r in self.requirements if r.status == DisclosureStatus.MET)

    @property
    def requirements_total(self) -> int:
        """Total requirement count excluding NOT_APPLICABLE."""
        return sum(1 for r in self.requirements if r.status != DisclosureStatus.NOT_APPLICABLE)

    @property
    def completion_pct(self) -> float:
        """Completion percentage."""
        total = self.requirements_total
        if total == 0:
            return 100.0 if self.is_material else 0.0
        return (self.requirements_met / total) * 100.0


class ESRSDisclosureInput(BaseModel):
    """Complete input for ESRS disclosure generation."""
    company_name: str = Field(..., description="Reporting entity name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Fiscal year")
    report_date: date = Field(default_factory=date.today, description="Generation date")
    standards: List[StandardDisclosure] = Field(
        default_factory=list, description="Per-standard disclosures"
    )
    general_methodology_note: Optional[str] = Field(
        None, description="Overarching methodology description"
    )
    assurance_level: Optional[str] = Field(
        None, description="Limited or reasonable assurance statement"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _status_icon(status: DisclosureStatus) -> str:
    """Plain-text icon for disclosure status."""
    return {
        DisclosureStatus.MET: "[MET]",
        DisclosureStatus.PARTIALLY_MET: "[PARTIAL]",
        DisclosureStatus.NOT_MET: "[NOT MET]",
        DisclosureStatus.NOT_APPLICABLE: "[N/A]",
    }.get(status, "[?]")


def _quality_label(level: DataQualityLevel) -> str:
    """Render quality badge."""
    return f"[{level.value}]"


def _status_css(status: DisclosureStatus) -> str:
    """CSS class for disclosure status."""
    return f"disclosure-{status.value.lower().replace('_', '-')}"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ESRSDisclosureTemplate:
    """Generate full ESRS disclosure narrative for all 12 standards.

    For each standard (E1-E5, S1-S4, G1, ESRS-1, ESRS-2):
        - Disclosure requirements met/unmet
        - Metric values with provenance
        - Narrative sections (AI-generated, marked for review)
        - Cross-references to other standards
        - Data quality indicators

    Example:
        >>> template = ESRSDisclosureTemplate()
        >>> data = ESRSDisclosureInput(company_name="Acme", reporting_year=2025, ...)
        >>> full_md = template.full_disclosure(data)
        >>> e1_md = template.per_standard_disclosure(data, ESRSStandardId.E1)
        >>> gaps = template.gap_summary(data)
    """

    def __init__(self) -> None:
        """Initialize the ESRS disclosure template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def per_standard_disclosure(self, data: ESRSDisclosureInput, standard: ESRSStandardId) -> str:
        """Render disclosure for a single ESRS standard as Markdown.

        Args:
            data: Full disclosure input.
            standard: Which standard to render.

        Returns:
            Markdown string for the requested standard, or a notice if not found.
        """
        match = next((s for s in data.standards if s.standard_id == standard), None)
        if match is None:
            return f"## {standard.value}\n\nNo disclosure data available for {standard.value}."
        return self._md_standard_section(match)

    def full_disclosure(self, data: ESRSDisclosureInput) -> str:
        """Render complete ESRS disclosure document as Markdown.

        Args:
            data: Full disclosure input.

        Returns:
            Complete Markdown document.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [self._md_cover(data)]
        if data.general_methodology_note:
            sections.append(f"## Methodology\n\n{data.general_methodology_note}")
        if data.assurance_level:
            sections.append(f"## Assurance\n\n{data.assurance_level}")
        sections.append(self._md_gap_summary_section(data))
        for std in self._ordered_standards(data):
            sections.append(self._md_standard_section(std))
        sections.append(self._md_footer(data))
        return "\n\n".join(sections)

    def gap_summary(self, data: ESRSDisclosureInput) -> Dict[str, Any]:
        """Return a structured gap analysis summary.

        Args:
            data: Full disclosure input.

        Returns:
            Dictionary with per-standard gap counts and overall statistics.
        """
        gaps: Dict[str, Any] = {"standards": {}, "totals": {}}
        total_met = 0
        total_partial = 0
        total_not_met = 0
        total_na = 0

        for std in data.standards:
            met = sum(1 for r in std.requirements if r.status == DisclosureStatus.MET)
            partial = sum(1 for r in std.requirements if r.status == DisclosureStatus.PARTIALLY_MET)
            not_met = sum(1 for r in std.requirements if r.status == DisclosureStatus.NOT_MET)
            na = sum(1 for r in std.requirements if r.status == DisclosureStatus.NOT_APPLICABLE)
            gaps["standards"][std.standard_id.value] = {
                "name": std.standard_name,
                "is_material": std.is_material,
                "met": met,
                "partially_met": partial,
                "not_met": not_met,
                "not_applicable": na,
                "completion_pct": std.completion_pct,
            }
            total_met += met
            total_partial += partial
            total_not_met += not_met
            total_na += na

        applicable_total = total_met + total_partial + total_not_met
        gaps["totals"] = {
            "met": total_met,
            "partially_met": total_partial,
            "not_met": total_not_met,
            "not_applicable": total_na,
            "overall_completion_pct": (
                (total_met / applicable_total * 100.0) if applicable_total > 0 else 0.0
            ),
        }
        return gaps

    def render_markdown(self, data: ESRSDisclosureInput) -> str:
        """Alias for full_disclosure for interface consistency."""
        return self.full_disclosure(data)

    def render_html(self, data: ESRSDisclosureInput) -> str:
        """Render the full ESRS disclosure as HTML.

        Args:
            data: Full disclosure input.

        Returns:
            HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [self._html_cover(data)]
        if data.general_methodology_note:
            body_parts.append(
                f'<div class="section"><h2>Methodology</h2>'
                f"<p>{data.general_methodology_note}</p></div>"
            )
        body_parts.append(self._html_gap_summary(data))
        for std in self._ordered_standards(data):
            body_parts.append(self._html_standard_section(std))
        body_parts.append(self._html_footer(data))
        body = "\n".join(body_parts)
        return self._wrap_html(data.company_name, data.reporting_year, body)

    def render_json(self, data: ESRSDisclosureInput) -> Dict[str, Any]:
        """Render the full disclosure as a JSON-serializable dict.

        Args:
            data: Full disclosure input.

        Returns:
            Dictionary ready for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        gap = self.gap_summary(data)

        return {
            "template": "esrs_disclosure",
            "version": "1.0.0",
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "company_name": data.company_name,
            "reporting_year": data.reporting_year,
            "assurance_level": data.assurance_level,
            "gap_summary": gap,
            "standards": [s.model_dump(mode="json") for s in data.standards],
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: ESRSDisclosureInput) -> str:
        """Compute SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # ORDERING HELPER
    # --------------------------------------------------------------------- #

    def _ordered_standards(self, data: ESRSDisclosureInput) -> List[StandardDisclosure]:
        """Return standards in canonical ESRS order."""
        order = list(ESRSStandardId)
        indexed = {s.standard_id: s for s in data.standards}
        return [indexed[sid] for sid in order if sid in indexed]

    # --------------------------------------------------------------------- #
    # MARKDOWN RENDERERS
    # --------------------------------------------------------------------- #

    def _md_cover(self, data: ESRSDisclosureInput) -> str:
        """Render Markdown cover page."""
        return (
            f"# ESRS Disclosure Report - {data.company_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()}\n\n---"
        )

    def _md_gap_summary_section(self, data: ESRSDisclosureInput) -> str:
        """Render the gap summary as a Markdown table."""
        gap = self.gap_summary(data)
        lines = [
            "## Gap Summary",
            "",
            "| Standard | Name | Material | Met | Partial | Not Met | Completion |",
            "|----------|------|----------|-----|---------|---------|------------|",
        ]
        for sid, info in gap["standards"].items():
            mat = "Yes" if info["is_material"] else "No"
            lines.append(
                f"| {sid} | {info['name']} | {mat} | {info['met']} | "
                f"{info['partially_met']} | {info['not_met']} | {info['completion_pct']:.0f}% |"
            )
        totals = gap["totals"]
        lines.append(
            f"| **Total** | | | **{totals['met']}** | **{totals['partially_met']}** | "
            f"**{totals['not_met']}** | **{totals['overall_completion_pct']:.0f}%** |"
        )
        return "\n".join(lines)

    def _md_standard_section(self, std: StandardDisclosure) -> str:
        """Render a single standard's disclosure as Markdown."""
        lines = [
            f"## {std.standard_id.value} - {std.standard_name}",
            "",
            f"**Material:** {'Yes' if std.is_material else 'No'} | "
            f"**Completion:** {std.completion_pct:.0f}% "
            f"({std.requirements_met}/{std.requirements_total})",
        ]
        if std.materiality_rationale:
            lines.append(f"\n*Materiality Rationale:* {std.materiality_rationale}")
        if std.summary_narrative:
            lines.append(f"\n{std.summary_narrative}")
        if not std.is_material:
            lines.append("\n*This standard has been assessed as not material.*")
            return "\n".join(lines)
        for req in std.requirements:
            lines.append(self._md_requirement(req))
        return "\n".join(lines)

    def _md_requirement(self, req: DisclosureRequirement) -> str:
        """Render a single disclosure requirement as Markdown."""
        para = f" (Para. {req.paragraph_ref})" if req.paragraph_ref else ""
        lines = [
            f"\n### {req.requirement_id} - {req.requirement_name}{para}",
            f"**Status:** {_status_icon(req.status)}",
        ]
        if req.narrative:
            review = req.narrative_review_status.value.replace("_", " ").title()
            lines.append(f"\n> **Narrative** *[{review}]*:\n> {req.narrative}")
        if req.metrics:
            lines.append("\n| Metric | Value | YoY | Unit | Quality | XBRL Tag |")
            lines.append("|--------|-------|-----|------|---------|----------|")
            for m in req.metrics:
                yoy = m.yoy_change() or "-"
                quality = _quality_label(m.data_quality.quality_level)
                xbrl = m.xbrl_tag or "-"
                lines.append(
                    f"| {m.metric_name} | {m.formatted_value()} | {yoy} "
                    f"| {m.unit} | {quality} | `{xbrl}` |"
                )
        if req.cross_references:
            refs = "; ".join(
                f"{cr.target_standard} {cr.target_section}" for cr in req.cross_references
            )
            lines.append(f"\n*Cross-references:* {refs}")
        if req.evidence_references:
            lines.append(f"\n*Evidence:* {', '.join(req.evidence_references)}")
        return "\n".join(lines)

    def _md_footer(self, data: ESRSDisclosureInput) -> str:
        """Render Markdown footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML RENDERERS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, company: str, year: int, body: str) -> str:
        """Wrap content in HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>ESRS Disclosure - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:Arial,Helvetica,sans-serif;margin:2rem;color:#222;max-width:960px;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f5f5f5;}\n"
            ".disclosure-met{color:#1a7f37;font-weight:bold;}\n"
            ".disclosure-partially-met{color:#b08800;font-weight:bold;}\n"
            ".disclosure-not-met{color:#cf222e;font-weight:bold;}\n"
            ".disclosure-not-applicable{color:#888;}\n"
            "blockquote{border-left:3px solid #ddd;padding-left:1rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".review-badge{font-size:0.8rem;padding:0.1rem 0.4rem;border-radius:3px;"
            "background:#e8e8e8;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_cover(self, data: ESRSDisclosureInput) -> str:
        """Render HTML cover."""
        return (
            '<div class="section">\n'
            f"<h1>ESRS Disclosure Report &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Report Date:</strong> {data.report_date.isoformat()}</p>\n"
            "<hr>\n</div>"
        )

    def _html_gap_summary(self, data: ESRSDisclosureInput) -> str:
        """Render gap summary as HTML table."""
        gap = self.gap_summary(data)
        rows = []
        for sid, info in gap["standards"].items():
            mat = "Yes" if info["is_material"] else "No"
            rows.append(
                f"<tr><td>{sid}</td><td>{info['name']}</td><td>{mat}</td>"
                f"<td>{info['met']}</td><td>{info['partially_met']}</td>"
                f"<td>{info['not_met']}</td><td>{info['completion_pct']:.0f}%</td></tr>"
            )
        totals = gap["totals"]
        rows.append(
            f"<tr style='font-weight:bold'><td>Total</td><td></td><td></td>"
            f"<td>{totals['met']}</td><td>{totals['partially_met']}</td>"
            f"<td>{totals['not_met']}</td><td>{totals['overall_completion_pct']:.0f}%</td></tr>"
        )
        return (
            '<div class="section">\n<h2>Gap Summary</h2>\n'
            "<table><thead><tr><th>Standard</th><th>Name</th><th>Material</th>"
            "<th>Met</th><th>Partial</th><th>Not Met</th><th>Completion</th>"
            f"</tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_standard_section(self, std: StandardDisclosure) -> str:
        """Render a single standard as HTML."""
        parts = [
            f'<div class="section">\n'
            f"<h2>{std.standard_id.value} &mdash; {std.standard_name}</h2>\n"
            f"<p><strong>Material:</strong> {'Yes' if std.is_material else 'No'} | "
            f"<strong>Completion:</strong> {std.completion_pct:.0f}% "
            f"({std.requirements_met}/{std.requirements_total})</p>"
        ]
        if std.materiality_rationale:
            parts.append(f"<p><em>Materiality Rationale:</em> {std.materiality_rationale}</p>")
        if std.summary_narrative:
            parts.append(f"<p>{std.summary_narrative}</p>")
        if not std.is_material:
            parts.append("<p><em>This standard has been assessed as not material.</em></p>")
            parts.append("</div>")
            return "\n".join(parts)
        for req in std.requirements:
            parts.append(self._html_requirement(req))
        parts.append("</div>")
        return "\n".join(parts)

    def _html_requirement(self, req: DisclosureRequirement) -> str:
        """Render a single requirement as HTML."""
        css = _status_css(req.status)
        para = f" (Para. {req.paragraph_ref})" if req.paragraph_ref else ""
        parts = [
            f"<h3>{req.requirement_id} &mdash; {req.requirement_name}{para}</h3>",
            f'<p><strong>Status:</strong> <span class="{css}">{req.status.value}</span></p>',
        ]
        if req.narrative:
            review = req.narrative_review_status.value.replace("_", " ").title()
            parts.append(
                f'<blockquote><span class="review-badge">{review}</span><br>'
                f"{req.narrative}</blockquote>"
            )
        if req.metrics:
            rows = []
            for m in req.metrics:
                yoy = m.yoy_change() or "-"
                quality = m.data_quality.quality_level.value
                xbrl = m.xbrl_tag or "-"
                rows.append(
                    f"<tr><td>{m.metric_name}</td><td>{m.formatted_value()}</td>"
                    f"<td>{yoy}</td><td>{m.unit}</td><td>{quality}</td>"
                    f"<td><code>{xbrl}</code></td></tr>"
                )
            parts.append(
                "<table><thead><tr><th>Metric</th><th>Value</th><th>YoY</th>"
                "<th>Unit</th><th>Quality</th><th>XBRL</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
            )
        if req.cross_references:
            refs = "; ".join(f"{cr.target_standard} {cr.target_section}" for cr in req.cross_references)
            parts.append(f"<p><em>Cross-references:</em> {refs}</p>")
        return "\n".join(parts)

    def _html_footer(self, data: ESRSDisclosureInput) -> str:
        """Render HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
