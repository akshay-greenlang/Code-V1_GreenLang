# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Cross-Framework Alignment Report Template
=============================================================

Cross-framework alignment map showing coverage across CSRD/ESRS, CDP,
SBTi, EU Taxonomy, and other reporting frameworks. Includes gap
analysis, scoring predictions, and temperature pathway alignment.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 2.0.0
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

class AlignmentStatus(str, Enum):
    """Framework alignment status."""
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    GAP = "GAP"
    NOT_APPLICABLE = "N/A"


class GapPriority(str, Enum):
    """Priority level for gap remediation."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CDPScoreGrade(str, Enum):
    """CDP score grades."""
    A = "A"
    A_MINUS = "A-"
    B = "B"
    B_MINUS = "B-"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    D_MINUS = "D-"
    F = "F"


class SBTiTargetStatus(str, Enum):
    """SBTi target validation status."""
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    IN_PROGRESS = "IN_PROGRESS"
    NOT_SET = "NOT_SET"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FrameworkAlignment(BaseModel):
    """Alignment status for a single reporting framework."""
    framework_name: str = Field(..., description="Framework name (e.g. CDP, SBTi)")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage %")
    aligned_count: int = Field(0, ge=0, description="Number of aligned requirements")
    gap_count: int = Field(0, ge=0, description="Number of gaps")
    total_requirements: int = Field(0, ge=0, description="Total requirements")
    notes: Optional[str] = Field(None, description="Additional notes")


class GapEntry(BaseModel):
    """Individual gap item with remediation guidance."""
    gap_id: str = Field(..., description="Gap identifier")
    requirement: str = Field(..., description="Requirement description")
    framework_reference: str = Field(..., description="Framework reference code")
    priority: GapPriority = Field(GapPriority.MEDIUM, description="Priority level")
    remediation: str = Field("", description="Remediation recommendation")
    estimated_effort_days: Optional[int] = Field(
        None, ge=0, description="Estimated effort in days"
    )
    esrs_mapping: Optional[str] = Field(
        None, description="Corresponding ESRS standard"
    )


class CDPScoringResult(BaseModel):
    """CDP scoring prediction result."""
    predicted_score: CDPScoreGrade = Field(..., description="Predicted CDP score")
    confidence_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Prediction confidence"
    )
    category_scores: Dict[str, str] = Field(
        default_factory=dict,
        description="Score per CDP category (Governance, Risks, etc.)",
    )
    improvement_areas: List[str] = Field(
        default_factory=list, description="Areas for score improvement"
    )


class SBTiResult(BaseModel):
    """SBTi alignment result."""
    implied_temperature: float = Field(
        ..., description="Implied temperature rise in degrees C"
    )
    target_status: SBTiTargetStatus = Field(
        SBTiTargetStatus.NOT_SET, description="Target validation status"
    )
    near_term_target: Optional[str] = Field(None, description="Near-term target desc")
    long_term_target: Optional[str] = Field(None, description="Long-term target desc")
    progress_pct: float = Field(0.0, ge=0.0, le=100.0, description="Progress %")
    pathway: Optional[str] = Field(
        None, description="Temperature pathway (1.5C, WB2C, 2C)"
    )


class TaxonomyResult(BaseModel):
    """EU Taxonomy alignment result."""
    gar: float = Field(0.0, ge=0.0, le=100.0, description="Green Asset Ratio %")
    btar: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Banking Book Taxonomy Alignment Ratio %",
    )
    eligible_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Taxonomy-eligible %"
    )
    aligned_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Taxonomy-aligned %"
    )
    revenue_eligible_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Revenue eligible %"
    )
    capex_eligible_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="CapEx eligible %"
    )
    opex_eligible_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="OpEx eligible %"
    )


class CrossFrameworkReportInput(BaseModel):
    """Complete input for the cross-framework alignment report."""
    organization_name: str = Field(..., description="Organization name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    report_date: date = Field(
        default_factory=date.today, description="Report generation date"
    )
    frameworks: List[FrameworkAlignment] = Field(
        default_factory=list, description="Framework alignment summaries"
    )
    coverage_matrix: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Framework x standard coverage matrix",
    )
    gaps_by_framework: Dict[str, List[GapEntry]] = Field(
        default_factory=dict, description="Gaps organized by framework"
    )
    cdp_scoring: Optional[CDPScoringResult] = Field(
        None, description="CDP scoring prediction"
    )
    sbti_temperature: Optional[SBTiResult] = Field(
        None, description="SBTi temperature alignment"
    )
    taxonomy_kpis: Optional[TaxonomyResult] = Field(
        None, description="EU Taxonomy KPIs"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage value."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _fmt_temp(value: Optional[float]) -> str:
    """Format temperature value."""
    if value is None:
        return "N/A"
    return f"{value:.1f} C"


def _priority_sort(priority: GapPriority) -> int:
    """Sort key for gap priority."""
    return {
        GapPriority.CRITICAL: 0,
        GapPriority.HIGH: 1,
        GapPriority.MEDIUM: 2,
        GapPriority.LOW: 3,
    }.get(priority, 99)


def _alignment_badge(status: str) -> str:
    """Badge for alignment status string."""
    return f"[{status.upper()}]"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class CrossFrameworkReportTemplate:
    """Generate cross-framework alignment map report.

    Sections:
        1. Framework Coverage Summary
        2. Coverage Matrix (framework x standard)
        3. Per-Framework Detail
        4. CDP Scoring Breakdown
        5. SBTi Temperature & Pathway
        6. EU Taxonomy KPIs
        7. Gap Analysis by Priority
        8. Recommendations

    Example:
        >>> template = CrossFrameworkReportTemplate()
        >>> data = CrossFrameworkReportInput(organization_name="Acme", reporting_year=2025)
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "cross_framework_report"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the cross-framework report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC RENDER METHODS
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: CrossFrameworkReportInput) -> str:
        """Render as Markdown.

        Args:
            data: Validated cross-framework report input.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_framework_summary(data),
            self._md_coverage_matrix(data),
            self._md_per_framework_detail(data),
            self._md_cdp_scoring(data),
            self._md_sbti_temperature(data),
            self._md_taxonomy_kpis(data),
            self._md_gap_analysis(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: CrossFrameworkReportInput) -> str:
        """Render as HTML document.

        Args:
            data: Validated cross-framework report input.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_framework_summary(data),
            self._html_coverage_matrix(data),
            self._html_per_framework_detail(data),
            self._html_cdp_scoring(data),
            self._html_sbti_temperature(data),
            self._html_taxonomy_kpis(data),
            self._html_gap_analysis(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, data.reporting_year, body)

    def render_json(self, data: CrossFrameworkReportInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict.

        Args:
            data: Validated cross-framework report input.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)

        result: Dict[str, Any] = {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "organization_name": data.organization_name,
            "reporting_year": data.reporting_year,
            "frameworks": [f.model_dump(mode="json") for f in data.frameworks],
            "coverage_matrix": data.coverage_matrix,
            "gaps_by_framework": {
                k: [g.model_dump(mode="json") for g in v]
                for k, v in data.gaps_by_framework.items()
            },
        }
        if data.cdp_scoring:
            result["cdp_scoring"] = data.cdp_scoring.model_dump(mode="json")
        if data.sbti_temperature:
            result["sbti_temperature"] = data.sbti_temperature.model_dump(mode="json")
        if data.taxonomy_kpis:
            result["taxonomy_kpis"] = data.taxonomy_kpis.model_dump(mode="json")
        return result

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: CrossFrameworkReportInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: CrossFrameworkReportInput) -> str:
        """Markdown header."""
        return (
            f"# Cross-Framework Alignment Report - {data.organization_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()} | "
            f"**Frameworks Analyzed:** {len(data.frameworks)}\n\n---"
        )

    def _md_framework_summary(self, data: CrossFrameworkReportInput) -> str:
        """Framework coverage summary table."""
        lines = [
            "## 1. Framework Coverage Summary",
            "",
            "| Framework | Coverage | Aligned | Gaps | Total | Notes |",
            "|-----------|----------|---------|------|-------|-------|",
        ]
        for fw in data.frameworks:
            notes = fw.notes or "-"
            lines.append(
                f"| {fw.framework_name} | {_fmt_pct(fw.coverage_pct)} "
                f"| {fw.aligned_count} | {fw.gap_count} "
                f"| {fw.total_requirements} | {notes} |"
            )
        if not data.frameworks:
            lines.append("| - | No frameworks analyzed | - | - | - | - |")
        return "\n".join(lines)

    def _md_coverage_matrix(self, data: CrossFrameworkReportInput) -> str:
        """Coverage matrix (framework x standard)."""
        if not data.coverage_matrix:
            return "## 2. Coverage Matrix\n\nNo coverage matrix data available."
        frameworks = sorted(data.coverage_matrix.keys())
        all_standards: List[str] = []
        for fw in frameworks:
            for std in data.coverage_matrix[fw]:
                if std not in all_standards:
                    all_standards.append(std)
        all_standards.sort()

        header = "| Framework | " + " | ".join(all_standards) + " |"
        sep = "|-----------|" + "|".join("---" for _ in all_standards) + "|"
        lines = ["## 2. Coverage Matrix", "", header, sep]
        for fw in frameworks:
            cells = []
            for std in all_standards:
                status = data.coverage_matrix[fw].get(std, "-")
                cells.append(_alignment_badge(status) if status != "-" else "-")
            lines.append(f"| {fw} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _md_per_framework_detail(self, data: CrossFrameworkReportInput) -> str:
        """Per-framework detail sections."""
        if not data.frameworks:
            return ""
        lines = ["## 3. Per-Framework Detail", ""]
        for fw in data.frameworks:
            lines.extend([
                f"### {fw.framework_name}",
                f"- **Coverage:** {_fmt_pct(fw.coverage_pct)}",
                f"- **Aligned Requirements:** {fw.aligned_count} / {fw.total_requirements}",
                f"- **Identified Gaps:** {fw.gap_count}",
            ])
            if fw.notes:
                lines.append(f"- **Notes:** {fw.notes}")
            lines.append("")
        return "\n".join(lines)

    def _md_cdp_scoring(self, data: CrossFrameworkReportInput) -> str:
        """CDP scoring breakdown."""
        if not data.cdp_scoring:
            return "## 4. CDP Scoring Breakdown\n\nNo CDP scoring data available."
        cdp = data.cdp_scoring
        lines = [
            "## 4. CDP Scoring Breakdown",
            "",
            f"**Predicted Score:** {cdp.predicted_score.value} "
            f"(Confidence: {_fmt_pct(cdp.confidence_pct)})",
            "",
        ]
        if cdp.category_scores:
            lines.extend([
                "| Category | Score |",
                "|----------|-------|",
            ])
            for cat, score in cdp.category_scores.items():
                lines.append(f"| {cat} | {score} |")
        if cdp.improvement_areas:
            lines.extend(["", "**Improvement Areas:**"])
            for area in cdp.improvement_areas:
                lines.append(f"- {area}")
        return "\n".join(lines)

    def _md_sbti_temperature(self, data: CrossFrameworkReportInput) -> str:
        """SBTi temperature and pathway."""
        if not data.sbti_temperature:
            return "## 5. SBTi Temperature & Pathway\n\nNo SBTi data available."
        sbti = data.sbti_temperature
        lines = [
            "## 5. SBTi Temperature & Pathway",
            "",
            f"- **Implied Temperature Rise:** {_fmt_temp(sbti.implied_temperature)}",
            f"- **Target Status:** {sbti.target_status.value}",
            f"- **Progress:** {_fmt_pct(sbti.progress_pct)}",
            f"- **Pathway:** {sbti.pathway or 'Not specified'}",
        ]
        if sbti.near_term_target:
            lines.append(f"- **Near-term Target:** {sbti.near_term_target}")
        if sbti.long_term_target:
            lines.append(f"- **Long-term Target:** {sbti.long_term_target}")
        return "\n".join(lines)

    def _md_taxonomy_kpis(self, data: CrossFrameworkReportInput) -> str:
        """EU Taxonomy KPIs."""
        if not data.taxonomy_kpis:
            return "## 6. EU Taxonomy KPIs\n\nNo EU Taxonomy data available."
        tax = data.taxonomy_kpis
        lines = [
            "## 6. EU Taxonomy KPIs",
            "",
            "| KPI | Value |",
            "|-----|-------|",
            f"| Green Asset Ratio (GAR) | {_fmt_pct(tax.gar)} |",
            f"| BTAR | {_fmt_pct(tax.btar)} |",
            f"| Taxonomy Eligible | {_fmt_pct(tax.eligible_pct)} |",
            f"| Taxonomy Aligned | {_fmt_pct(tax.aligned_pct)} |",
            f"| Revenue Eligible | {_fmt_pct(tax.revenue_eligible_pct)} |",
            f"| CapEx Eligible | {_fmt_pct(tax.capex_eligible_pct)} |",
            f"| OpEx Eligible | {_fmt_pct(tax.opex_eligible_pct)} |",
        ]
        return "\n".join(lines)

    def _md_gap_analysis(self, data: CrossFrameworkReportInput) -> str:
        """Gap analysis by priority."""
        all_gaps: List[tuple] = []
        for fw_name, gaps in data.gaps_by_framework.items():
            for gap in gaps:
                all_gaps.append((fw_name, gap))
        if not all_gaps:
            return "## 7. Gap Analysis\n\nNo gaps identified."
        all_gaps.sort(key=lambda x: _priority_sort(x[1].priority))
        lines = [
            "## 7. Gap Analysis by Priority",
            "",
            "| Priority | Framework | ID | Requirement | Remediation | Effort (days) | ESRS |",
            "|----------|-----------|-----|-------------|-------------|---------------|------|",
        ]
        for fw_name, gap in all_gaps:
            effort = str(gap.estimated_effort_days) if gap.estimated_effort_days else "TBD"
            esrs = gap.esrs_mapping or "-"
            lines.append(
                f"| {gap.priority.value} | {fw_name} | {gap.gap_id} "
                f"| {gap.requirement} | {gap.remediation} | {effort} | {esrs} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: CrossFrameworkReportInput) -> str:
        """Recommendations section."""
        lines = ["## 8. Recommendations", ""]
        recs = []
        # Auto-generate recommendations based on data
        total_gaps = sum(
            len(gaps) for gaps in data.gaps_by_framework.values()
        )
        critical_gaps = sum(
            1 for gaps in data.gaps_by_framework.values()
            for g in gaps if g.priority == GapPriority.CRITICAL
        )
        if critical_gaps > 0:
            recs.append(
                f"Address {critical_gaps} critical gap(s) immediately to "
                f"prevent regulatory non-compliance."
            )
        if data.cdp_scoring and data.cdp_scoring.improvement_areas:
            recs.append(
                f"Focus on CDP improvement areas to raise predicted score "
                f"from {data.cdp_scoring.predicted_score.value}."
            )
        if data.sbti_temperature:
            if data.sbti_temperature.implied_temperature > 2.0:
                recs.append(
                    "Strengthen emissions reduction targets to align with "
                    "1.5C pathway as per SBTi recommendations."
                )
        if data.taxonomy_kpis:
            if data.taxonomy_kpis.aligned_pct < 50.0:
                recs.append(
                    "Increase EU Taxonomy-aligned activities to improve "
                    "green asset ratio and investor reporting."
                )
        if not recs:
            recs.append(
                "Continue monitoring framework alignment and address "
                "emerging gaps proactively."
            )
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: CrossFrameworkReportInput) -> str:
        """Markdown footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, org: str, year: int, body: str) -> str:
        """HTML wrapper."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Cross-Framework Report - {org} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1200px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "h3{color:#533483;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".status-full{color:#1a7f37;font-weight:bold;}\n"
            ".status-partial{color:#b08800;font-weight:bold;}\n"
            ".status-gap{color:#cf222e;font-weight:bold;}\n"
            ".priority-critical{color:#cf222e;font-weight:bold;}\n"
            ".priority-high{color:#e36209;font-weight:bold;}\n"
            ".priority-medium{color:#b08800;}\n"
            ".priority-low{color:#1a7f37;}\n"
            ".metric-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".metric-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".metric-label{font-size:0.85rem;color:#666;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: CrossFrameworkReportInput) -> str:
        """HTML header."""
        return (
            '<div class="section">\n'
            f"<h1>Cross-Framework Alignment Report &mdash; {data.organization_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Report Date:</strong> {data.report_date.isoformat()} | "
            f"<strong>Frameworks:</strong> {len(data.frameworks)}</p>\n"
            "<hr>\n</div>"
        )

    def _html_framework_summary(self, data: CrossFrameworkReportInput) -> str:
        """HTML framework summary."""
        rows = []
        for fw in data.frameworks:
            notes = fw.notes or "-"
            rows.append(
                f"<tr><td>{fw.framework_name}</td><td>{_fmt_pct(fw.coverage_pct)}</td>"
                f"<td>{fw.aligned_count}</td><td>{fw.gap_count}</td>"
                f"<td>{fw.total_requirements}</td><td>{notes}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">No frameworks analyzed</td></tr>')
        return (
            '<div class="section">\n<h2>1. Framework Coverage Summary</h2>\n'
            "<table><thead><tr><th>Framework</th><th>Coverage</th>"
            "<th>Aligned</th><th>Gaps</th><th>Total</th>"
            f"<th>Notes</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_coverage_matrix(self, data: CrossFrameworkReportInput) -> str:
        """HTML coverage matrix."""
        if not data.coverage_matrix:
            return (
                '<div class="section"><h2>2. Coverage Matrix</h2>'
                "<p>No data available.</p></div>"
            )
        frameworks = sorted(data.coverage_matrix.keys())
        all_standards: List[str] = []
        for fw in frameworks:
            for std in data.coverage_matrix[fw]:
                if std not in all_standards:
                    all_standards.append(std)
        all_standards.sort()

        header_cells = "".join(f"<th>{std}</th>" for std in all_standards)
        rows = []
        for fw in frameworks:
            cells = []
            for std in all_standards:
                status = data.coverage_matrix[fw].get(std, "-")
                if status.upper() == "FULL":
                    css = "status-full"
                elif status.upper() == "PARTIAL":
                    css = "status-partial"
                elif status.upper() == "GAP":
                    css = "status-gap"
                else:
                    css = ""
                cells.append(
                    f'<td class="{css}">{status.upper()}</td>'
                    if css else f"<td>{status}</td>"
                )
            rows.append(f"<tr><td>{fw}</td>{''.join(cells)}</tr>")
        return (
            '<div class="section">\n<h2>2. Coverage Matrix</h2>\n'
            f"<table><thead><tr><th>Framework</th>{header_cells}</tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_per_framework_detail(self, data: CrossFrameworkReportInput) -> str:
        """HTML per-framework detail."""
        if not data.frameworks:
            return ""
        parts = ['<div class="section">\n<h2>3. Per-Framework Detail</h2>\n']
        for fw in data.frameworks:
            parts.append(
                f"<h3>{fw.framework_name}</h3>\n<ul>\n"
                f"<li><strong>Coverage:</strong> {_fmt_pct(fw.coverage_pct)}</li>\n"
                f"<li><strong>Aligned:</strong> {fw.aligned_count} / {fw.total_requirements}</li>\n"
                f"<li><strong>Gaps:</strong> {fw.gap_count}</li>\n"
            )
            if fw.notes:
                parts.append(f"<li><strong>Notes:</strong> {fw.notes}</li>\n")
            parts.append("</ul>\n")
        parts.append("</div>")
        return "".join(parts)

    def _html_cdp_scoring(self, data: CrossFrameworkReportInput) -> str:
        """HTML CDP scoring."""
        if not data.cdp_scoring:
            return (
                '<div class="section"><h2>4. CDP Scoring</h2>'
                "<p>No CDP scoring data available.</p></div>"
            )
        cdp = data.cdp_scoring
        parts = [
            '<div class="section">\n<h2>4. CDP Scoring Breakdown</h2>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{cdp.predicted_score.value}</div>'
            f'<div class="metric-label">Predicted Score</div></div>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{_fmt_pct(cdp.confidence_pct)}</div>'
            f'<div class="metric-label">Confidence</div></div>\n',
        ]
        if cdp.category_scores:
            rows = "".join(
                f"<tr><td>{cat}</td><td>{score}</td></tr>"
                for cat, score in cdp.category_scores.items()
            )
            parts.append(
                "<table><thead><tr><th>Category</th><th>Score</th></tr></thead>\n"
                f"<tbody>{rows}</tbody></table>\n"
            )
        if cdp.improvement_areas:
            items = "".join(f"<li>{a}</li>" for a in cdp.improvement_areas)
            parts.append(f"<p><strong>Improvement Areas:</strong></p><ul>{items}</ul>\n")
        parts.append("</div>")
        return "".join(parts)

    def _html_sbti_temperature(self, data: CrossFrameworkReportInput) -> str:
        """HTML SBTi temperature."""
        if not data.sbti_temperature:
            return (
                '<div class="section"><h2>5. SBTi Temperature</h2>'
                "<p>No SBTi data available.</p></div>"
            )
        sbti = data.sbti_temperature
        parts = [
            '<div class="section">\n<h2>5. SBTi Temperature &amp; Pathway</h2>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{_fmt_temp(sbti.implied_temperature)}</div>'
            f'<div class="metric-label">Implied Temperature</div></div>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{sbti.target_status.value}</div>'
            f'<div class="metric-label">Target Status</div></div>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{_fmt_pct(sbti.progress_pct)}</div>'
            f'<div class="metric-label">Progress</div></div>\n',
            "<ul>\n",
            f"<li><strong>Pathway:</strong> {sbti.pathway or 'Not specified'}</li>\n",
        ]
        if sbti.near_term_target:
            parts.append(
                f"<li><strong>Near-term Target:</strong> {sbti.near_term_target}</li>\n"
            )
        if sbti.long_term_target:
            parts.append(
                f"<li><strong>Long-term Target:</strong> {sbti.long_term_target}</li>\n"
            )
        parts.append("</ul>\n</div>")
        return "".join(parts)

    def _html_taxonomy_kpis(self, data: CrossFrameworkReportInput) -> str:
        """HTML EU Taxonomy KPIs."""
        if not data.taxonomy_kpis:
            return (
                '<div class="section"><h2>6. EU Taxonomy KPIs</h2>'
                "<p>No EU Taxonomy data available.</p></div>"
            )
        tax = data.taxonomy_kpis
        rows = [
            f"<tr><td>Green Asset Ratio (GAR)</td><td>{_fmt_pct(tax.gar)}</td></tr>",
            f"<tr><td>BTAR</td><td>{_fmt_pct(tax.btar)}</td></tr>",
            f"<tr><td>Taxonomy Eligible</td><td>{_fmt_pct(tax.eligible_pct)}</td></tr>",
            f"<tr><td>Taxonomy Aligned</td><td>{_fmt_pct(tax.aligned_pct)}</td></tr>",
            f"<tr><td>Revenue Eligible</td><td>{_fmt_pct(tax.revenue_eligible_pct)}</td></tr>",
            f"<tr><td>CapEx Eligible</td><td>{_fmt_pct(tax.capex_eligible_pct)}</td></tr>",
            f"<tr><td>OpEx Eligible</td><td>{_fmt_pct(tax.opex_eligible_pct)}</td></tr>",
        ]
        return (
            '<div class="section">\n<h2>6. EU Taxonomy KPIs</h2>\n'
            "<table><thead><tr><th>KPI</th><th>Value</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_gap_analysis(self, data: CrossFrameworkReportInput) -> str:
        """HTML gap analysis."""
        all_gaps: List[tuple] = []
        for fw_name, gaps in data.gaps_by_framework.items():
            for gap in gaps:
                all_gaps.append((fw_name, gap))
        if not all_gaps:
            return (
                '<div class="section"><h2>7. Gap Analysis</h2>'
                "<p>No gaps identified.</p></div>"
            )
        all_gaps.sort(key=lambda x: _priority_sort(x[1].priority))
        rows = []
        for fw_name, gap in all_gaps:
            css = f"priority-{gap.priority.value.lower()}"
            effort = str(gap.estimated_effort_days) if gap.estimated_effort_days else "TBD"
            esrs = gap.esrs_mapping or "-"
            rows.append(
                f'<tr><td class="{css}">{gap.priority.value}</td>'
                f"<td>{fw_name}</td><td>{gap.gap_id}</td>"
                f"<td>{gap.requirement}</td><td>{gap.remediation}</td>"
                f"<td>{effort}</td><td>{esrs}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>7. Gap Analysis by Priority</h2>\n'
            "<table><thead><tr><th>Priority</th><th>Framework</th><th>ID</th>"
            "<th>Requirement</th><th>Remediation</th><th>Effort</th>"
            f"<th>ESRS</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_recommendations(self, data: CrossFrameworkReportInput) -> str:
        """HTML recommendations."""
        recs = []
        total_gaps = sum(len(g) for g in data.gaps_by_framework.values())
        critical_gaps = sum(
            1 for gaps in data.gaps_by_framework.values()
            for g in gaps if g.priority == GapPriority.CRITICAL
        )
        if critical_gaps > 0:
            recs.append(
                f"Address {critical_gaps} critical gap(s) immediately."
            )
        if data.cdp_scoring and data.cdp_scoring.improvement_areas:
            recs.append(
                f"Focus on CDP improvement areas to raise score from "
                f"{data.cdp_scoring.predicted_score.value}."
            )
        if data.sbti_temperature and data.sbti_temperature.implied_temperature > 2.0:
            recs.append(
                "Strengthen targets to align with 1.5C pathway."
            )
        if data.taxonomy_kpis and data.taxonomy_kpis.aligned_pct < 50.0:
            recs.append("Increase Taxonomy-aligned activities.")
        if not recs:
            recs.append("Continue monitoring framework alignment proactively.")
        items = "".join(f"<li>{r}</li>" for r in recs)
        return (
            '<div class="section">\n<h2>8. Recommendations</h2>\n'
            f"<ol>{items}</ol>\n</div>"
        )

    def _html_footer(self, data: CrossFrameworkReportInput) -> str:
        """HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
