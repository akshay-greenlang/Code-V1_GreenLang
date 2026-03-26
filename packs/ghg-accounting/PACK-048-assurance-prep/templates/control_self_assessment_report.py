# -*- coding: utf-8 -*-
"""
ControlSelfAssessmentReport - Control Self-Assessment for PACK-048.

Generates a control self-assessment report with a 25-control register
covering design and operating effectiveness ratings, a control maturity
heatmap (category x maturity level), a deficiency log with severity
and remediation status, control coverage summary, and improvement
recommendations.

Regulatory References:
    - ISAE 3410 para 23-29: Understanding entity controls
    - ISO 14064-3 clause 6.3.4: Evaluation of GHG information system
    - COSO Internal Control Framework
    - AA1000AS v3: Information management systems
    - SOX Section 404 analogy for GHG controls

Sections:
    1. Controls Register (25 controls)
    2. Control Maturity Heatmap Data
    3. Deficiency Log
    4. Control Coverage Summary
    5. Recommendations
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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


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


class DesignEffectiveness(str, Enum):
    """Control design effectiveness rating."""
    EFFECTIVE = "effective"
    PARTIALLY_EFFECTIVE = "partially_effective"
    INEFFECTIVE = "ineffective"
    NOT_TESTED = "not_tested"


class OperatingEffectiveness(str, Enum):
    """Control operating effectiveness rating."""
    EFFECTIVE = "effective"
    PARTIALLY_EFFECTIVE = "partially_effective"
    INEFFECTIVE = "ineffective"
    NOT_TESTED = "not_tested"


class MaturityLevel(str, Enum):
    """Control maturity level."""
    INITIAL = "initial"
    REPEATABLE = "repeatable"
    DEFINED = "defined"
    MANAGED = "managed"
    OPTIMIZING = "optimizing"


class DeficiencySeverity(str, Enum):
    """Control deficiency severity."""
    SIGNIFICANT = "significant"
    MATERIAL_WEAKNESS = "material_weakness"
    DEFICIENCY = "deficiency"
    OBSERVATION = "observation"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class ControlEntry(BaseModel):
    """Single control in the controls register."""
    control_id: str = Field(..., description="Control identifier (e.g., CTL-001)")
    control_name: str = Field(..., description="Control name")
    category: str = Field("", description="Control category (e.g., Data Collection)")
    description: str = Field("", description="Control description")
    control_type: str = Field("preventive", description="Control type (preventive/detective/corrective)")
    frequency: str = Field("", description="Control frequency (e.g., continuous, monthly)")
    owner: str = Field("", description="Control owner")
    design_effectiveness: DesignEffectiveness = Field(
        DesignEffectiveness.NOT_TESTED, description="Design effectiveness"
    )
    operating_effectiveness: OperatingEffectiveness = Field(
        OperatingEffectiveness.NOT_TESTED, description="Operating effectiveness"
    )
    maturity_level: MaturityLevel = Field(
        MaturityLevel.INITIAL, description="Maturity level"
    )
    evidence_ref: str = Field("", description="Reference to supporting evidence")
    last_tested: Optional[str] = Field(None, description="Last tested date (ISO)")
    notes: str = Field("", description="Additional notes")


class MaturityHeatmapCell(BaseModel):
    """Single cell in the control maturity heatmap."""
    category: str = Field(..., description="Control category")
    maturity_level: MaturityLevel = Field(..., description="Maturity level")
    control_count: int = Field(0, ge=0, description="Number of controls at this level")


class DeficiencyEntry(BaseModel):
    """Single deficiency in the deficiency log."""
    deficiency_id: str = Field(default_factory=lambda: f"DEF-{_new_uuid()[:6]}", description="Deficiency ID")
    control_id: str = Field("", description="Related control ID")
    title: str = Field(..., description="Deficiency title")
    description: str = Field("", description="Deficiency description")
    severity: DeficiencySeverity = Field(
        DeficiencySeverity.DEFICIENCY, description="Severity level"
    )
    root_cause: str = Field("", description="Root cause analysis")
    remediation_action: str = Field("", description="Planned remediation")
    responsible: str = Field("", description="Responsible person / team")
    target_date: Optional[str] = Field(None, description="Target remediation date (ISO)")
    status: str = Field("open", description="Status (open/in_progress/closed)")


class ControlCoverageSummary(BaseModel):
    """Control coverage summary statistics."""
    total_controls: int = Field(0, ge=0, description="Total controls in register")
    controls_tested: int = Field(0, ge=0, description="Controls tested")
    controls_effective: int = Field(0, ge=0, description="Controls rated effective")
    controls_partially_effective: int = Field(0, ge=0, description="Partially effective")
    controls_ineffective: int = Field(0, ge=0, description="Controls rated ineffective")
    tested_pct: float = Field(0.0, ge=0, le=100, description="% tested")
    effective_pct: float = Field(0.0, ge=0, le=100, description="% effective (of tested)")


class Recommendation(BaseModel):
    """Single recommendation for control improvement."""
    priority: int = Field(1, ge=1, description="Priority rank")
    area: str = Field("", description="Improvement area")
    recommendation: str = Field(..., description="Recommendation text")
    expected_benefit: str = Field("", description="Expected benefit")
    effort: str = Field("", description="Estimated effort (low/medium/high)")


class ControlAssessmentInput(BaseModel):
    """Complete input model for ControlSelfAssessmentReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    controls: List[ControlEntry] = Field(
        default_factory=list, description="Controls register (25 controls)"
    )
    maturity_heatmap: List[MaturityHeatmapCell] = Field(
        default_factory=list, description="Maturity heatmap data"
    )
    deficiencies: List[DeficiencyEntry] = Field(
        default_factory=list, description="Deficiency log"
    )
    coverage_summary: Optional[ControlCoverageSummary] = Field(
        None, description="Coverage summary"
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list, description="Improvement recommendations"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _effectiveness_label(value: str) -> str:
    """Return display label for effectiveness."""
    return value.replace("_", " ").title()


def _severity_css(severity: str) -> str:
    """Return CSS class for deficiency severity."""
    mapping = {
        "significant": "sev-significant",
        "material_weakness": "sev-material",
        "deficiency": "sev-deficiency",
        "observation": "sev-observation",
    }
    return mapping.get(severity, "sev-deficiency")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ControlSelfAssessmentReport:
    """
    Control self-assessment report template for PACK-048.

    Renders a 25-control register with design and operating effectiveness
    ratings, maturity heatmap data, deficiency log with severity and
    remediation status, coverage summary, and improvement recommendations.
    All outputs include SHA-256 provenance hashing for audit-trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = ControlSelfAssessmentReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ControlSelfAssessmentReport."""
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
        """Render control assessment as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render control assessment as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render control assessment as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
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
            self._md_controls_register(data),
            self._md_maturity_heatmap(data),
            self._md_deficiency_log(data),
            self._md_coverage_summary(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Control Self-Assessment Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_controls_register(self, data: Dict[str, Any]) -> str:
        """Render Markdown controls register."""
        controls = data.get("controls", [])
        if not controls:
            return "## 1. Controls Register\n\nNo controls defined."
        lines = [
            "## 1. Controls Register",
            "",
            "| ID | Name | Category | Type | Design | Operating | Maturity |",
            "|----|------|----------|------|--------|-----------|----------|",
        ]
        for c in controls:
            lines.append(
                f"| {c.get('control_id', '')} | "
                f"{c.get('control_name', '')} | "
                f"{c.get('category', '')} | "
                f"{c.get('control_type', '')} | "
                f"{_effectiveness_label(c.get('design_effectiveness', 'not_tested'))} | "
                f"{_effectiveness_label(c.get('operating_effectiveness', 'not_tested'))} | "
                f"{c.get('maturity_level', 'initial').title()} |"
            )
        return "\n".join(lines)

    def _md_maturity_heatmap(self, data: Dict[str, Any]) -> str:
        """Render Markdown maturity heatmap table."""
        cells = data.get("maturity_heatmap", [])
        if not cells:
            return ""
        categories = sorted(set(c.get("category", "") for c in cells))
        levels = ["initial", "repeatable", "defined", "managed", "optimizing"]
        lookup: Dict[str, Dict[str, int]] = {}
        for cell in cells:
            cat = cell.get("category", "")
            lvl = cell.get("maturity_level", "initial")
            if cat not in lookup:
                lookup[cat] = {}
            lookup[cat][lvl] = cell.get("control_count", 0)
        lines = [
            "## 2. Control Maturity Heatmap",
            "",
            "| Category | Initial | Repeatable | Defined | Managed | Optimizing |",
            "|----------|---------|------------|---------|---------|------------|",
        ]
        for cat in categories:
            row = f"| {cat}"
            for lvl in levels:
                count = lookup.get(cat, {}).get(lvl, 0)
                row += f" | {count}"
            row += " |"
            lines.append(row)
        return "\n".join(lines)

    def _md_deficiency_log(self, data: Dict[str, Any]) -> str:
        """Render Markdown deficiency log."""
        deficiencies = data.get("deficiencies", [])
        if not deficiencies:
            return "## 3. Deficiency Log\n\nNo deficiencies identified."
        lines = [
            "## 3. Deficiency Log",
            "",
            "| ID | Control | Title | Severity | Remediation | Owner | Target | Status |",
            "|----|---------|-------|----------|-------------|-------|--------|--------|",
        ]
        for d in deficiencies:
            severity = d.get("severity", "deficiency").replace("_", " ").title()
            lines.append(
                f"| {d.get('deficiency_id', '')} | "
                f"{d.get('control_id', '')} | "
                f"{d.get('title', '')} | "
                f"**{severity}** | "
                f"{d.get('remediation_action', '')} | "
                f"{d.get('responsible', '')} | "
                f"{d.get('target_date', '-')} | "
                f"{d.get('status', 'open')} |"
            )
        return "\n".join(lines)

    def _md_coverage_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown coverage summary."""
        cs = data.get("coverage_summary")
        if not cs:
            return ""
        lines = [
            "## 4. Control Coverage Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Controls | {cs.get('total_controls', 0)} |",
            f"| Controls Tested | {cs.get('controls_tested', 0)} |",
            f"| Effective | {cs.get('controls_effective', 0)} |",
            f"| Partially Effective | {cs.get('controls_partially_effective', 0)} |",
            f"| Ineffective | {cs.get('controls_ineffective', 0)} |",
            f"| % Tested | {cs.get('tested_pct', 0):.1f}% |",
            f"| % Effective (of tested) | {cs.get('effective_pct', 0):.1f}% |",
        ]
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        lines = ["## 5. Recommendations", ""]
        for r in recs:
            lines.append(
                f"**{r.get('priority', '')}.** {r.get('recommendation', '')}"
            )
            area = r.get("area", "")
            benefit = r.get("expected_benefit", "")
            effort = r.get("effort", "")
            if area:
                lines.append(f"  - Area: {area}")
            if benefit:
                lines.append(f"  - Benefit: {benefit}")
            if effort:
                lines.append(f"  - Effort: {effort}")
            lines.append("")
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
            self._html_controls_register(data),
            self._html_maturity_heatmap(data),
            self._html_deficiency_log(data),
            self._html_coverage_summary(data),
            self._html_recommendations(data),
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
            f"<title>Control Self-Assessment - {company}</title>\n"
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
            ".eff-effective{color:#2a9d8f;font-weight:700;}\n"
            ".eff-partial{color:#e9c46a;font-weight:700;}\n"
            ".eff-ineffective{color:#e76f51;font-weight:700;}\n"
            ".sev-significant{color:#e76f51;font-weight:700;}\n"
            ".sev-material{color:#d62828;font-weight:700;}\n"
            ".sev-deficiency{color:#e9c46a;font-weight:700;}\n"
            ".sev-observation{color:#2a9d8f;}\n"
            ".heatmap-cell{text-align:center;font-weight:600;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            '<div class="section">\n'
            f"<h1>Control Self-Assessment Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_controls_register(self, data: Dict[str, Any]) -> str:
        """Render HTML controls register."""
        controls = data.get("controls", [])
        if not controls:
            return ""
        rows = ""
        for c in controls:
            de = c.get("design_effectiveness", "not_tested")
            oe = c.get("operating_effectiveness", "not_tested")
            de_css = "eff-effective" if de == "effective" else (
                "eff-partial" if de == "partially_effective" else (
                    "eff-ineffective" if de == "ineffective" else ""
                )
            )
            oe_css = "eff-effective" if oe == "effective" else (
                "eff-partial" if oe == "partially_effective" else (
                    "eff-ineffective" if oe == "ineffective" else ""
                )
            )
            rows += (
                f"<tr><td>{c.get('control_id', '')}</td>"
                f"<td>{c.get('control_name', '')}</td>"
                f"<td>{c.get('category', '')}</td>"
                f"<td>{c.get('control_type', '')}</td>"
                f'<td class="{de_css}">{_effectiveness_label(de)}</td>'
                f'<td class="{oe_css}">{_effectiveness_label(oe)}</td>'
                f"<td>{c.get('maturity_level', 'initial').title()}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>1. Controls Register</h2>\n'
            "<table><thead><tr><th>ID</th><th>Name</th><th>Category</th>"
            "<th>Type</th><th>Design</th><th>Operating</th><th>Maturity</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_maturity_heatmap(self, data: Dict[str, Any]) -> str:
        """Render HTML maturity heatmap."""
        cells = data.get("maturity_heatmap", [])
        if not cells:
            return ""
        categories = sorted(set(c.get("category", "") for c in cells))
        levels = ["initial", "repeatable", "defined", "managed", "optimizing"]
        lookup: Dict[str, Dict[str, int]] = {}
        for cell in cells:
            cat = cell.get("category", "")
            lvl = cell.get("maturity_level", "initial")
            if cat not in lookup:
                lookup[cat] = {}
            lookup[cat][lvl] = cell.get("control_count", 0)
        rows = ""
        for cat in categories:
            cells_html = ""
            for lvl in levels:
                count = lookup.get(cat, {}).get(lvl, 0)
                bg = "#f0f4f8" if count == 0 else "#d4edda" if count > 0 else "#f0f4f8"
                cells_html += f'<td class="heatmap-cell" style="background:{bg};">{count}</td>'
            rows += f"<tr><td><strong>{cat}</strong></td>{cells_html}</tr>\n"
        return (
            '<div class="section">\n<h2>2. Control Maturity Heatmap</h2>\n'
            "<table><thead><tr><th>Category</th><th>Initial</th><th>Repeatable</th>"
            "<th>Defined</th><th>Managed</th><th>Optimizing</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_deficiency_log(self, data: Dict[str, Any]) -> str:
        """Render HTML deficiency log."""
        deficiencies = data.get("deficiencies", [])
        if not deficiencies:
            return ""
        rows = ""
        for d in deficiencies:
            severity = d.get("severity", "deficiency")
            css = _severity_css(severity)
            rows += (
                f"<tr><td>{d.get('deficiency_id', '')}</td>"
                f"<td>{d.get('control_id', '')}</td>"
                f"<td>{d.get('title', '')}</td>"
                f'<td class="{css}">{severity.replace("_", " ").title()}</td>'
                f"<td>{d.get('remediation_action', '')}</td>"
                f"<td>{d.get('responsible', '')}</td>"
                f"<td>{d.get('target_date', '-')}</td>"
                f"<td>{d.get('status', 'open')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Deficiency Log</h2>\n'
            "<table><thead><tr><th>ID</th><th>Control</th><th>Title</th>"
            "<th>Severity</th><th>Remediation</th><th>Owner</th>"
            "<th>Target</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_coverage_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML coverage summary."""
        cs = data.get("coverage_summary")
        if not cs:
            return ""
        rows = (
            f"<tr><td>Total Controls</td><td>{cs.get('total_controls', 0)}</td></tr>\n"
            f"<tr><td>Controls Tested</td><td>{cs.get('controls_tested', 0)}</td></tr>\n"
            f"<tr><td>Effective</td><td>{cs.get('controls_effective', 0)}</td></tr>\n"
            f"<tr><td>Partially Effective</td><td>{cs.get('controls_partially_effective', 0)}</td></tr>\n"
            f"<tr><td>Ineffective</td><td>{cs.get('controls_ineffective', 0)}</td></tr>\n"
            f"<tr><td>% Tested</td><td>{cs.get('tested_pct', 0):.1f}%</td></tr>\n"
            f"<tr><td>% Effective (of tested)</td><td>{cs.get('effective_pct', 0):.1f}%</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>4. Control Coverage Summary</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        content = ""
        for r in recs:
            area = r.get("area", "")
            benefit = r.get("expected_benefit", "")
            effort = r.get("effort", "")
            details = ""
            if area:
                details += f"<br><small>Area: {area}</small>"
            if benefit:
                details += f"<br><small>Benefit: {benefit}</small>"
            if effort:
                details += f"<br><small>Effort: {effort}</small>"
            content += (
                f'<div style="background:#f0f4f8;border-left:3px solid #264653;'
                f'padding:0.6rem 1rem;margin:0.5rem 0;border-radius:4px;">'
                f"<strong>{r.get('priority', '')}.</strong> "
                f"{r.get('recommendation', '')}{details}</div>\n"
            )
        return f'<div class="section">\n<h2>5. Recommendations</h2>\n{content}</div>'

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
        """Render control assessment as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "control_self_assessment_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "controls": data.get("controls", []),
            "maturity_heatmap": data.get("maturity_heatmap", []),
            "deficiencies": data.get("deficiencies", []),
            "coverage_summary": data.get("coverage_summary"),
            "recommendations": data.get("recommendations", []),
            "chart_data": {
                "maturity_heatmap": self._build_heatmap_chart(data),
                "effectiveness_pie": self._build_effectiveness_pie(data),
                "deficiency_severity_bar": self._build_deficiency_bar(data),
            },
        }

    def _build_heatmap_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build heatmap chart data."""
        cells = data.get("maturity_heatmap", [])
        if not cells:
            return {}
        categories = sorted(set(c.get("category", "") for c in cells))
        levels = ["initial", "repeatable", "defined", "managed", "optimizing"]
        matrix: List[List[int]] = []
        for cat in categories:
            row = []
            for lvl in levels:
                count = next(
                    (c.get("control_count", 0) for c in cells
                     if c.get("category") == cat and c.get("maturity_level") == lvl),
                    0,
                )
                row.append(count)
            matrix.append(row)
        return {"categories": categories, "levels": levels, "matrix": matrix}

    def _build_effectiveness_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build effectiveness distribution pie chart data."""
        controls = data.get("controls", [])
        if not controls:
            return {}
        dist: Dict[str, int] = {
            "effective": 0, "partially_effective": 0,
            "ineffective": 0, "not_tested": 0,
        }
        for c in controls:
            de = c.get("design_effectiveness", "not_tested")
            if de in dist:
                dist[de] += 1
        return {"labels": list(dist.keys()), "values": list(dist.values())}

    def _build_deficiency_bar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build deficiency severity bar chart data."""
        deficiencies = data.get("deficiencies", [])
        if not deficiencies:
            return {}
        dist: Dict[str, int] = {
            "significant": 0, "material_weakness": 0,
            "deficiency": 0, "observation": 0,
        }
        for d in deficiencies:
            sev = d.get("severity", "deficiency")
            if sev in dist:
                dist[sev] += 1
        return {"labels": list(dist.keys()), "values": list(dist.values())}
