# -*- coding: utf-8 -*-
"""
MaterialityAssessmentReport - Materiality Assessment for PACK-048.

Generates a materiality assessment report for GHG assurance with
quantitative materiality thresholds (overall, performance, clearly
trivial), scope-specific materiality breakdown, qualitative factors
assessment, materiality methodology narrative, and comparison to
prior-period materiality.

Regulatory References:
    - ISAE 3410 para 18-22: Materiality in GHG assurance
    - ISO 14064-3 clause 6.2.2: Materiality considerations
    - AA1000AS v3: Materiality principle
    - GHG Protocol: Significance thresholds
    - IAASB ISA 320 / ISRE 2410 analogy

Sections:
    1. Quantitative Materiality Thresholds
    2. Scope-Specific Materiality Breakdown
    3. Qualitative Factors Assessment
    4. Materiality Methodology
    5. Prior Period Comparison
    6. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    MD = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"

class MaterialityBasis(str, Enum):
    """Basis for materiality calculation."""
    TOTAL_EMISSIONS = "total_emissions"
    SCOPE_EMISSIONS = "scope_emissions"
    REVENUE = "revenue"
    CUSTOM = "custom"

class QualitativeImpact(str, Enum):
    """Qualitative impact level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_APPLICABLE = "not_applicable"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class MaterialityThreshold(BaseModel):
    """Quantitative materiality threshold."""
    threshold_name: str = Field(..., description="Threshold name")
    threshold_type: str = Field("overall", description="Type: overall / performance / clearly_trivial")
    value_tco2e: Optional[float] = Field(None, ge=0, description="Threshold in tCO2e")
    value_pct: Optional[float] = Field(None, ge=0, le=100, description="Threshold as % of base")
    base_amount_tco2e: Optional[float] = Field(None, ge=0, description="Base amount (tCO2e)")
    basis: MaterialityBasis = Field(
        MaterialityBasis.TOTAL_EMISSIONS, description="Calculation basis"
    )
    rationale: str = Field("", description="Rationale for threshold")

class ScopeMateriality(BaseModel):
    """Scope-specific materiality breakdown."""
    scope: str = Field(..., description="Scope label (Scope 1, Scope 2, Scope 3)")
    total_emissions_tco2e: float = Field(0.0, ge=0, description="Total scope emissions")
    materiality_tco2e: float = Field(0.0, ge=0, description="Materiality threshold tCO2e")
    materiality_pct: float = Field(0.0, ge=0, le=100, description="Materiality as % of scope")
    clearly_trivial_tco2e: float = Field(0.0, ge=0, description="Clearly trivial threshold")
    performance_materiality_tco2e: float = Field(0.0, ge=0, description="Performance materiality")
    notes: str = Field("", description="Scope-specific notes")

class QualitativeFactor(BaseModel):
    """Single qualitative materiality factor."""
    factor_name: str = Field(..., description="Factor name")
    description: str = Field("", description="Factor description")
    impact: QualitativeImpact = Field(
        QualitativeImpact.MEDIUM, description="Impact level"
    )
    consideration: str = Field("", description="Materiality consideration")
    affects_threshold: bool = Field(False, description="Whether this affects quantitative threshold")
    adjustment_direction: str = Field("", description="Increase/decrease/none")

class MaterialityMethodology(BaseModel):
    """Materiality methodology description."""
    approach: str = Field("", description="Overall approach description")
    benchmark_basis: str = Field("", description="Benchmark or basis used")
    percentage_applied: Optional[float] = Field(None, ge=0, le=100, description="% applied to base")
    adjustments: List[str] = Field(default_factory=list, description="Adjustments made")
    professional_judgement_factors: List[str] = Field(
        default_factory=list, description="Professional judgement factors"
    )
    standard_reference: str = Field("", description="Standard reference (e.g., ISAE 3410 para 18)")
    limitations: List[str] = Field(default_factory=list, description="Known limitations")

class PriorPeriodComparison(BaseModel):
    """Prior-period materiality comparison."""
    prior_period: str = Field("", description="Prior reporting period label")
    prior_overall_tco2e: Optional[float] = Field(None, ge=0, description="Prior overall materiality")
    prior_overall_pct: Optional[float] = Field(None, ge=0, le=100, description="Prior overall %")
    current_overall_tco2e: Optional[float] = Field(None, ge=0, description="Current overall materiality")
    current_overall_pct: Optional[float] = Field(None, ge=0, le=100, description="Current overall %")
    change_tco2e: Optional[float] = Field(None, description="Change in tCO2e")
    change_pct: Optional[float] = Field(None, description="Change in %")
    reason_for_change: str = Field("", description="Reason for change")

class MaterialityAssessmentInput(BaseModel):
    """Complete input model for MaterialityAssessmentReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    assurance_level: str = Field("limited", description="Assurance level")
    thresholds: List[MaterialityThreshold] = Field(
        default_factory=list, description="Quantitative thresholds"
    )
    scope_materiality: List[ScopeMateriality] = Field(
        default_factory=list, description="Scope-specific materiality"
    )
    qualitative_factors: List[QualitativeFactor] = Field(
        default_factory=list, description="Qualitative factors"
    )
    methodology: Optional[MaterialityMethodology] = Field(
        None, description="Methodology description"
    )
    prior_comparison: Optional[PriorPeriodComparison] = Field(
        None, description="Prior period comparison"
    )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _impact_label(impact: str) -> str:
    """Return display label for impact."""
    return impact.replace("_", " ").upper()

def _format_tco2e(value: Optional[float]) -> str:
    """Format tCO2e value or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:,.1f}"

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class MaterialityAssessmentReport:
    """
    Materiality assessment report template for PACK-048.

    Renders quantitative and qualitative materiality assessments with
    scope-specific breakdowns, methodology narrative, and prior-period
    comparison. All outputs include SHA-256 provenance hashing for
    audit-trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = MaterialityAssessmentReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MaterialityAssessmentReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Any:
        """Render in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format: {fmt}")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render materiality assessment as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render materiality assessment as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render materiality assessment as JSON dict."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def to_markdown(self, data: Dict[str, Any]) -> str:
        """Alias for render_markdown."""
        return self.render_markdown(data)

    def to_html(self, data: Dict[str, Any]) -> str:
        """Alias for render_html."""
        return self.render_html(data)

    def to_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for render_json."""
        return self.render_json(data)

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_thresholds(data),
            self._md_scope_materiality(data),
            self._md_qualitative_factors(data),
            self._md_methodology(data),
            self._md_prior_comparison(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        level = self._get_val(data, "assurance_level", "limited")
        return (
            f"# Materiality Assessment Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Assurance Level:** {level.title()} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_thresholds(self, data: Dict[str, Any]) -> str:
        """Render Markdown quantitative materiality thresholds."""
        thresholds = data.get("thresholds", [])
        if not thresholds:
            return ""
        lines = [
            "## 1. Quantitative Materiality Thresholds",
            "",
            "| Threshold | Type | Value (tCO2e) | % of Base | Base (tCO2e) | Basis |",
            "|-----------|------|---------------|-----------|-------------|-------|",
        ]
        for t in thresholds:
            lines.append(
                f"| {t.get('threshold_name', '')} | "
                f"{t.get('threshold_type', 'overall').replace('_', ' ').title()} | "
                f"{_format_tco2e(t.get('value_tco2e'))} | "
                f"{t.get('value_pct', 'N/A')}% | "
                f"{_format_tco2e(t.get('base_amount_tco2e'))} | "
                f"{t.get('basis', 'total_emissions').replace('_', ' ').title()} |"
            )
        return "\n".join(lines)

    def _md_scope_materiality(self, data: Dict[str, Any]) -> str:
        """Render Markdown scope-specific materiality."""
        scopes = data.get("scope_materiality", [])
        if not scopes:
            return ""
        lines = [
            "## 2. Scope-Specific Materiality",
            "",
            "| Scope | Total (tCO2e) | Materiality (tCO2e) | Mat. % | Performance (tCO2e) | Trivial (tCO2e) |",
            "|-------|---------------|---------------------|--------|---------------------|--------------------|",
        ]
        for s in scopes:
            lines.append(
                f"| {s.get('scope', '')} | "
                f"{s.get('total_emissions_tco2e', 0):,.1f} | "
                f"{s.get('materiality_tco2e', 0):,.1f} | "
                f"{s.get('materiality_pct', 0):.1f}% | "
                f"{s.get('performance_materiality_tco2e', 0):,.1f} | "
                f"{s.get('clearly_trivial_tco2e', 0):,.1f} |"
            )
        return "\n".join(lines)

    def _md_qualitative_factors(self, data: Dict[str, Any]) -> str:
        """Render Markdown qualitative factors."""
        factors = data.get("qualitative_factors", [])
        if not factors:
            return ""
        lines = [
            "## 3. Qualitative Factors Assessment",
            "",
            "| Factor | Impact | Consideration | Affects Threshold | Direction |",
            "|--------|--------|---------------|-------------------|-----------|",
        ]
        for f in factors:
            affects = "Yes" if f.get("affects_threshold", False) else "No"
            lines.append(
                f"| {f.get('factor_name', '')} | "
                f"**{_impact_label(f.get('impact', 'medium'))}** | "
                f"{f.get('consideration', '')} | "
                f"{affects} | "
                f"{f.get('adjustment_direction', '-')} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology narrative."""
        meth = data.get("methodology")
        if not meth:
            return ""
        lines = ["## 4. Materiality Methodology", ""]
        approach = meth.get("approach", "")
        if approach:
            lines.append(f"**Approach:** {approach}")
            lines.append("")
        benchmark = meth.get("benchmark_basis", "")
        if benchmark:
            lines.append(f"**Benchmark Basis:** {benchmark}")
        pct = meth.get("percentage_applied")
        if pct is not None:
            lines.append(f"**Percentage Applied:** {pct:.1f}%")
        std_ref = meth.get("standard_reference", "")
        if std_ref:
            lines.append(f"**Standard Reference:** {std_ref}")
        lines.append("")
        adjustments = meth.get("adjustments", [])
        if adjustments:
            lines.append("**Adjustments:**")
            for a in adjustments:
                lines.append(f"- {a}")
            lines.append("")
        pj_factors = meth.get("professional_judgement_factors", [])
        if pj_factors:
            lines.append("**Professional Judgement Factors:**")
            for pj in pj_factors:
                lines.append(f"- {pj}")
            lines.append("")
        limitations = meth.get("limitations", [])
        if limitations:
            lines.append("**Limitations:**")
            for lim in limitations:
                lines.append(f"- {lim}")
        return "\n".join(lines)

    def _md_prior_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown prior-period comparison."""
        comp = data.get("prior_comparison")
        if not comp:
            return ""
        lines = [
            "## 5. Prior Period Comparison",
            "",
            "| Metric | Prior Period | Current Period | Change |",
            "|--------|-------------|----------------|--------|",
        ]
        prior_tco2e = comp.get("prior_overall_tco2e")
        current_tco2e = comp.get("current_overall_tco2e")
        change_tco2e = comp.get("change_tco2e")
        lines.append(
            f"| Overall Materiality (tCO2e) | "
            f"{_format_tco2e(prior_tco2e)} | "
            f"{_format_tco2e(current_tco2e)} | "
            f"{_format_tco2e(change_tco2e)} |"
        )
        prior_pct = comp.get("prior_overall_pct")
        current_pct = comp.get("current_overall_pct")
        change_pct = comp.get("change_pct")
        pp_str = f"{prior_pct:.1f}%" if prior_pct is not None else "N/A"
        cp_str = f"{current_pct:.1f}%" if current_pct is not None else "N/A"
        ch_str = f"{change_pct:+.1f}pp" if change_pct is not None else "N/A"
        lines.append(f"| Overall Materiality (%) | {pp_str} | {cp_str} | {ch_str} |")
        reason = comp.get("reason_for_change", "")
        if reason:
            lines.append("")
            lines.append(f"**Reason for Change:** {reason}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-048 Assurance Prep v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_thresholds(data),
            self._html_scope_materiality(data),
            self._html_qualitative_factors(data),
            self._html_methodology(data),
            self._html_prior_comparison(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Materiality Assessment - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".impact-high{color:#e76f51;font-weight:700;}\n"
            ".impact-medium{color:#e9c46a;font-weight:700;}\n"
            ".impact-low{color:#2a9d8f;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        level = self._get_val(data, "assurance_level", "limited")
        return (
            '<div class="section">\n'
            f"<h1>Materiality Assessment Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Assurance Level:</strong> {level.title()} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_thresholds(self, data: Dict[str, Any]) -> str:
        """Render HTML quantitative thresholds."""
        thresholds = data.get("thresholds", [])
        if not thresholds:
            return ""
        rows = ""
        for t in thresholds:
            rows += (
                f"<tr><td>{t.get('threshold_name', '')}</td>"
                f"<td>{t.get('threshold_type', 'overall').replace('_', ' ').title()}</td>"
                f"<td>{_format_tco2e(t.get('value_tco2e'))}</td>"
                f"<td>{t.get('value_pct', 'N/A')}%</td>"
                f"<td>{_format_tco2e(t.get('base_amount_tco2e'))}</td>"
                f"<td>{t.get('basis', 'total_emissions').replace('_', ' ').title()}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>1. Quantitative Materiality Thresholds</h2>\n'
            "<table><thead><tr><th>Threshold</th><th>Type</th><th>Value (tCO2e)</th>"
            "<th>% of Base</th><th>Base (tCO2e)</th><th>Basis</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scope_materiality(self, data: Dict[str, Any]) -> str:
        """Render HTML scope-specific materiality."""
        scopes = data.get("scope_materiality", [])
        if not scopes:
            return ""
        rows = ""
        for s in scopes:
            rows += (
                f"<tr><td><strong>{s.get('scope', '')}</strong></td>"
                f"<td>{s.get('total_emissions_tco2e', 0):,.1f}</td>"
                f"<td>{s.get('materiality_tco2e', 0):,.1f}</td>"
                f"<td>{s.get('materiality_pct', 0):.1f}%</td>"
                f"<td>{s.get('performance_materiality_tco2e', 0):,.1f}</td>"
                f"<td>{s.get('clearly_trivial_tco2e', 0):,.1f}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Scope-Specific Materiality</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Total (tCO2e)</th>"
            "<th>Materiality (tCO2e)</th><th>Mat. %</th>"
            "<th>Performance (tCO2e)</th><th>Trivial (tCO2e)</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_qualitative_factors(self, data: Dict[str, Any]) -> str:
        """Render HTML qualitative factors."""
        factors = data.get("qualitative_factors", [])
        if not factors:
            return ""
        rows = ""
        for f in factors:
            impact = f.get("impact", "medium")
            css = f"impact-{impact}" if impact in ("high", "medium", "low") else ""
            affects = "Yes" if f.get("affects_threshold", False) else "No"
            rows += (
                f"<tr><td>{f.get('factor_name', '')}</td>"
                f'<td class="{css}">{_impact_label(impact)}</td>'
                f"<td>{f.get('consideration', '')}</td>"
                f"<td>{affects}</td>"
                f"<td>{f.get('adjustment_direction', '-')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Qualitative Factors Assessment</h2>\n'
            "<table><thead><tr><th>Factor</th><th>Impact</th>"
            "<th>Consideration</th><th>Affects Threshold</th>"
            "<th>Direction</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology section."""
        meth = data.get("methodology")
        if not meth:
            return ""
        content = ""
        approach = meth.get("approach", "")
        if approach:
            content += f"<p><strong>Approach:</strong> {approach}</p>\n"
        benchmark = meth.get("benchmark_basis", "")
        if benchmark:
            content += f"<p><strong>Benchmark Basis:</strong> {benchmark}</p>\n"
        pct = meth.get("percentage_applied")
        if pct is not None:
            content += f"<p><strong>Percentage Applied:</strong> {pct:.1f}%</p>\n"
        adjustments = meth.get("adjustments", [])
        if adjustments:
            content += "<p><strong>Adjustments:</strong></p><ul>"
            for a in adjustments:
                content += f"<li>{a}</li>"
            content += "</ul>\n"
        limitations = meth.get("limitations", [])
        if limitations:
            content += "<p><strong>Limitations:</strong></p><ul>"
            for lim in limitations:
                content += f"<li>{lim}</li>"
            content += "</ul>\n"
        return f'<div class="section">\n<h2>4. Materiality Methodology</h2>\n{content}</div>'

    def _html_prior_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML prior-period comparison."""
        comp = data.get("prior_comparison")
        if not comp:
            return ""
        rows = (
            f"<tr><td>Overall Materiality (tCO2e)</td>"
            f"<td>{_format_tco2e(comp.get('prior_overall_tco2e'))}</td>"
            f"<td>{_format_tco2e(comp.get('current_overall_tco2e'))}</td>"
            f"<td>{_format_tco2e(comp.get('change_tco2e'))}</td></tr>\n"
        )
        pp = comp.get("prior_overall_pct")
        cp = comp.get("current_overall_pct")
        ch = comp.get("change_pct")
        pp_str = f"{pp:.1f}%" if pp is not None else "N/A"
        cp_str = f"{cp:.1f}%" if cp is not None else "N/A"
        ch_str = f"{ch:+.1f}pp" if ch is not None else "N/A"
        rows += f"<tr><td>Overall Materiality (%)</td><td>{pp_str}</td><td>{cp_str}</td><td>{ch_str}</td></tr>\n"
        return (
            '<div class="section">\n<h2>5. Prior Period Comparison</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Prior</th>"
            "<th>Current</th><th>Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-048 Assurance Prep v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render materiality assessment as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "materiality_assessment_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "assurance_level": self._get_val(data, "assurance_level", "limited"),
            "thresholds": data.get("thresholds", []),
            "scope_materiality": data.get("scope_materiality", []),
            "qualitative_factors": data.get("qualitative_factors", []),
            "methodology": data.get("methodology"),
            "prior_comparison": data.get("prior_comparison"),
            "chart_data": {
                "scope_materiality_bar": self._build_scope_bar(data),
                "qualitative_impact_pie": self._build_impact_pie(data),
                "threshold_waterfall": self._build_threshold_waterfall(data),
            },
        }

    def _build_scope_bar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build scope materiality bar chart data."""
        scopes = data.get("scope_materiality", [])
        if not scopes:
            return {}
        return {
            "labels": [s.get("scope", "") for s in scopes],
            "total_emissions": [s.get("total_emissions_tco2e", 0) for s in scopes],
            "materiality": [s.get("materiality_tco2e", 0) for s in scopes],
        }

    def _build_impact_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build qualitative impact pie chart data."""
        factors = data.get("qualitative_factors", [])
        if not factors:
            return {}
        dist: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "not_applicable": 0}
        for f in factors:
            imp = f.get("impact", "medium")
            if imp in dist:
                dist[imp] += 1
        return {"labels": list(dist.keys()), "values": list(dist.values())}

    def _build_threshold_waterfall(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build threshold waterfall chart data."""
        thresholds = data.get("thresholds", [])
        if not thresholds:
            return {}
        return {
            "labels": [t.get("threshold_name", "") for t in thresholds],
            "values": [t.get("value_tco2e", 0) for t in thresholds],
            "types": [t.get("threshold_type", "overall") for t in thresholds],
        }
