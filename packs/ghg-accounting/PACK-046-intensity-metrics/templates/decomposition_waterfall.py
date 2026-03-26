# -*- coding: utf-8 -*-
"""
DecompositionWaterfallReport - LMDI Decomposition Waterfall for PACK-046.

Generates a decomposition waterfall report based on Logarithmic Mean
Divisia Index (LMDI) analysis, showing activity, structure, and intensity
effects with entity-level contributions, chart data, narrative
interpretation, and closure validation.

Sections:
    1. Methodology Description (LMDI explanation)
    2. Period Comparison
    3. Waterfall Data (activity / structure / intensity bars)
    4. Entity Contributions (which entities drove each effect)
    5. Interpretation (narrative text)
    6. Closure Validation

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured with waterfall chart data)

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

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


class EffectType(str, Enum):
    """Decomposition effect types."""
    ACTIVITY = "activity"
    STRUCTURE = "structure"
    INTENSITY = "intensity"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class PeriodSummary(BaseModel):
    """Summary of a single period for comparison."""
    year: int = Field(..., description="Reporting year")
    total_emissions_tco2e: float = Field(0.0, description="Total emissions")
    total_activity: float = Field(0.0, description="Total activity measure")
    activity_unit: str = Field("", description="Activity unit")
    overall_intensity: float = Field(0.0, description="Overall intensity")
    intensity_unit: str = Field("", description="Intensity unit")


class WaterfallBar(BaseModel):
    """Single bar in the waterfall chart."""
    label: str = Field(..., description="Bar label")
    effect_type: str = Field("", description="Effect type (activity/structure/intensity/total)")
    value: float = Field(0.0, description="Absolute change value (tCO2e)")
    pct_of_base: float = Field(0.0, description="Change as percentage of base period")
    cumulative_start: float = Field(0.0, description="Cumulative start position for chart")
    cumulative_end: float = Field(0.0, description="Cumulative end position for chart")
    color: str = Field("#888", description="Suggested bar colour hex code")


class EntityContribution(BaseModel):
    """Entity-level contribution to a decomposition effect."""
    entity_name: str = Field(..., description="Entity name")
    effect_type: str = Field(..., description="Effect type")
    contribution_tco2e: float = Field(0.0, description="Absolute contribution in tCO2e")
    contribution_pct: float = Field(0.0, description="Share of total effect (%)")
    direction: str = Field("neutral", description="Increase / decrease / neutral")


class ClosureValidation(BaseModel):
    """Validation that decomposition closes to observed change."""
    observed_change_tco2e: float = Field(0.0, description="Observed total change")
    decomposed_total_tco2e: float = Field(0.0, description="Sum of decomposed effects")
    residual_tco2e: float = Field(0.0, description="Residual (should be near zero)")
    residual_pct: float = Field(0.0, description="Residual as % of observed change")
    closure_passed: bool = Field(True, description="Whether closure test passed")
    tolerance_pct: float = Field(1.0, description="Acceptable tolerance %")


class WaterfallInput(BaseModel):
    """Complete input model for DecompositionWaterfallReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period label")
    methodology_description: str = Field(
        "",
        description="LMDI methodology description narrative",
    )
    base_period: Optional[PeriodSummary] = Field(None, description="Base period data")
    current_period: Optional[PeriodSummary] = Field(None, description="Current period data")
    waterfall_bars: List[WaterfallBar] = Field(
        default_factory=list, description="Waterfall chart bar data"
    )
    entity_contributions: List[EntityContribution] = Field(
        default_factory=list, description="Entity contributions to effects"
    )
    interpretation: str = Field("", description="Narrative interpretation text")
    closure_validation: Optional[ClosureValidation] = Field(
        None, description="Closure validation results"
    )


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class DecompositionWaterfallReport:
    """
    LMDI decomposition waterfall report template.

    Renders decomposition analysis results as a waterfall chart with
    activity, structure, and intensity effects. Includes entity-level
    contribution breakdowns, narrative interpretation, and closure
    validation. All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = DecompositionWaterfallReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DecompositionWaterfallReport."""
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

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render decomposition waterfall as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render decomposition waterfall as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render decomposition waterfall as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_methodology(data),
            self._md_period_comparison(data),
            self._md_waterfall(data),
            self._md_entity_contributions(data),
            self._md_interpretation(data),
            self._md_closure(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Decomposition Waterfall Analysis - {company}\n\n"
            f"**Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown LMDI methodology description."""
        desc = self._get_val(data, "methodology_description", "")
        if not desc:
            desc = (
                "This report uses the Logarithmic Mean Divisia Index (LMDI) "
                "decomposition method to separate total emission changes into "
                "activity, structure, and intensity effects. LMDI is preferred "
                "for its perfect decomposition property (no residual term) and "
                "ability to handle zero values."
            )
        return f"## 1. Methodology\n\n{desc}"

    def _md_period_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown period comparison table."""
        base = data.get("base_period")
        current = data.get("current_period")
        if not base and not current:
            return "## 2. Period Comparison\n\nNo period data available."
        lines = [
            "## 2. Period Comparison",
            "",
            "| Metric | Base Period | Current Period | Change |",
            "|--------|-----------|---------------|--------|",
        ]
        if base and current:
            b_year = base.get("year", "")
            c_year = current.get("year", "")
            b_em = base.get("total_emissions_tco2e", 0)
            c_em = current.get("total_emissions_tco2e", 0)
            em_chg = c_em - b_em
            b_act = base.get("total_activity", 0)
            c_act = current.get("total_activity", 0)
            act_unit = base.get("activity_unit", "")
            b_int = base.get("overall_intensity", 0)
            c_int = current.get("overall_intensity", 0)
            int_unit = base.get("intensity_unit", "")
            lines.append(f"| Year | {b_year} | {c_year} | - |")
            lines.append(
                f"| Total Emissions (tCO2e) | {b_em:,.1f} | "
                f"{c_em:,.1f} | {em_chg:+,.1f} |"
            )
            lines.append(
                f"| Total Activity ({act_unit}) | {b_act:,.2f} | "
                f"{c_act:,.2f} | {c_act - b_act:+,.2f} |"
            )
            lines.append(
                f"| Overall Intensity ({int_unit}) | {b_int:,.4f} | "
                f"{c_int:,.4f} | {c_int - b_int:+,.4f} |"
            )
        return "\n".join(lines)

    def _md_waterfall(self, data: Dict[str, Any]) -> str:
        """Render Markdown waterfall data table."""
        bars = data.get("waterfall_bars", [])
        if not bars:
            return "## 3. Waterfall Decomposition\n\nNo waterfall data available."
        lines = [
            "## 3. Waterfall Decomposition",
            "",
            "| Component | Change (tCO2e) | % of Base | Direction |",
            "|-----------|---------------|-----------|-----------|",
        ]
        for b in bars:
            label = b.get("label", "")
            value = b.get("value", 0)
            pct = b.get("pct_of_base", 0)
            direction = "Increase" if value > 0 else ("Decrease" if value < 0 else "Neutral")
            lines.append(
                f"| {label} | {value:+,.1f} | {pct:+.1f}% | {direction} |"
            )
        return "\n".join(lines)

    def _md_entity_contributions(self, data: Dict[str, Any]) -> str:
        """Render Markdown entity contributions table."""
        contributions = data.get("entity_contributions", [])
        if not contributions:
            return ""
        lines = [
            "## 4. Entity Contributions",
            "",
            "| Entity | Effect | Contribution (tCO2e) | Share (%) | Direction |",
            "|--------|--------|---------------------|-----------|-----------|",
        ]
        for c in contributions:
            entity = c.get("entity_name", "")
            effect = c.get("effect_type", "")
            value = c.get("contribution_tco2e", 0)
            share = c.get("contribution_pct", 0)
            direction = c.get("direction", "neutral")
            lines.append(
                f"| {entity} | {effect.title()} | {value:+,.1f} | "
                f"{share:.1f}% | {direction.title()} |"
            )
        return "\n".join(lines)

    def _md_interpretation(self, data: Dict[str, Any]) -> str:
        """Render Markdown interpretation narrative."""
        text = self._get_val(data, "interpretation", "")
        if not text:
            return ""
        return f"## 5. Interpretation\n\n{text}"

    def _md_closure(self, data: Dict[str, Any]) -> str:
        """Render Markdown closure validation."""
        closure = data.get("closure_validation")
        if not closure:
            return ""
        observed = closure.get("observed_change_tco2e", 0)
        decomposed = closure.get("decomposed_total_tco2e", 0)
        residual = closure.get("residual_tco2e", 0)
        residual_pct = closure.get("residual_pct", 0)
        passed = closure.get("closure_passed", True)
        tolerance = closure.get("tolerance_pct", 1.0)
        status = "PASSED" if passed else "FAILED"
        lines = [
            "## 6. Closure Validation",
            "",
            "| Check | Value |",
            "|-------|-------|",
            f"| Observed Change | {observed:+,.1f} tCO2e |",
            f"| Decomposed Total | {decomposed:+,.1f} tCO2e |",
            f"| Residual | {residual:+,.2f} tCO2e ({residual_pct:+.3f}%) |",
            f"| Tolerance | {tolerance:.1f}% |",
            f"| **Status** | **{status}** |",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_methodology(data),
            self._html_period_comparison(data),
            self._html_waterfall(data),
            self._html_entity_contributions(data),
            self._html_interpretation(data),
            self._html_closure(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Decomposition Waterfall - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".bar-positive{background:#e76f51;}\n"
            ".bar-negative{background:#2a9d8f;}\n"
            ".bar-neutral{background:#888;}\n"
            ".waterfall-bar{display:inline-block;height:24px;border-radius:3px;"
            "min-width:4px;vertical-align:middle;}\n"
            ".closure-pass{color:#2a9d8f;font-weight:700;}\n"
            ".closure-fail{color:#e76f51;font-weight:700;}\n"
            ".interpretation-box{background:#f8f9fa;border-left:4px solid #264653;"
            "padding:1rem 1.5rem;margin:1rem 0;}\n"
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
            f"<h1>Decomposition Waterfall Analysis &mdash; {company}</h1>\n"
            f"<p><strong>Period:</strong> {period}</p>\n<hr>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology section."""
        desc = self._get_val(data, "methodology_description", "")
        if not desc:
            desc = (
                "This report uses the Logarithmic Mean Divisia Index (LMDI) "
                "decomposition method to separate total emission changes into "
                "activity, structure, and intensity effects."
            )
        return (
            '<div class="section">\n<h2>1. Methodology</h2>\n'
            f"<p>{desc}</p>\n</div>"
        )

    def _html_period_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML period comparison table."""
        base = data.get("base_period")
        current = data.get("current_period")
        if not base or not current:
            return ""
        b_em = base.get("total_emissions_tco2e", 0)
        c_em = current.get("total_emissions_tco2e", 0)
        b_int = base.get("overall_intensity", 0)
        c_int = current.get("overall_intensity", 0)
        rows = (
            f"<tr><td>Year</td><td>{base.get('year', '')}</td>"
            f"<td>{current.get('year', '')}</td><td>-</td></tr>\n"
            f"<tr><td>Total Emissions (tCO2e)</td><td>{b_em:,.1f}</td>"
            f"<td>{c_em:,.1f}</td><td>{c_em - b_em:+,.1f}</td></tr>\n"
            f"<tr><td>Overall Intensity</td><td>{b_int:,.4f}</td>"
            f"<td>{c_int:,.4f}</td><td>{c_int - b_int:+,.4f}</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>2. Period Comparison</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Base</th>"
            "<th>Current</th><th>Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_waterfall(self, data: Dict[str, Any]) -> str:
        """Render HTML waterfall data table with inline bar indicators."""
        bars = data.get("waterfall_bars", [])
        if not bars:
            return ""
        rows = ""
        for b in bars:
            label = b.get("label", "")
            value = b.get("value", 0)
            pct = b.get("pct_of_base", 0)
            bar_class = "bar-positive" if value > 0 else ("bar-negative" if value < 0 else "bar-neutral")
            bar_width = min(abs(pct) * 3, 200)
            rows += (
                f"<tr><td>{label}</td><td>{value:+,.1f}</td><td>{pct:+.1f}%</td>"
                f'<td><span class="waterfall-bar {bar_class}" '
                f'style="width:{bar_width}px;"></span></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Waterfall Decomposition</h2>\n'
            "<table><thead><tr><th>Component</th><th>Change (tCO2e)</th>"
            "<th>% of Base</th><th>Visual</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_entity_contributions(self, data: Dict[str, Any]) -> str:
        """Render HTML entity contributions table."""
        contributions = data.get("entity_contributions", [])
        if not contributions:
            return ""
        rows = ""
        for c in contributions:
            entity = c.get("entity_name", "")
            effect = c.get("effect_type", "").title()
            value = c.get("contribution_tco2e", 0)
            share = c.get("contribution_pct", 0)
            direction = c.get("direction", "neutral").title()
            rows += (
                f"<tr><td>{entity}</td><td>{effect}</td>"
                f"<td>{value:+,.1f}</td><td>{share:.1f}%</td>"
                f"<td>{direction}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Entity Contributions</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Effect</th>"
            "<th>Contribution (tCO2e)</th><th>Share</th>"
            "<th>Direction</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_interpretation(self, data: Dict[str, Any]) -> str:
        """Render HTML interpretation narrative."""
        text = self._get_val(data, "interpretation", "")
        if not text:
            return ""
        return (
            '<div class="section">\n<h2>5. Interpretation</h2>\n'
            f'<div class="interpretation-box"><p>{text}</p></div>\n</div>'
        )

    def _html_closure(self, data: Dict[str, Any]) -> str:
        """Render HTML closure validation."""
        closure = data.get("closure_validation")
        if not closure:
            return ""
        passed = closure.get("closure_passed", True)
        css = "closure-pass" if passed else "closure-fail"
        status = "PASSED" if passed else "FAILED"
        observed = closure.get("observed_change_tco2e", 0)
        decomposed = closure.get("decomposed_total_tco2e", 0)
        residual = closure.get("residual_tco2e", 0)
        residual_pct = closure.get("residual_pct", 0)
        rows = (
            f"<tr><td>Observed Change</td><td>{observed:+,.1f} tCO2e</td></tr>\n"
            f"<tr><td>Decomposed Total</td><td>{decomposed:+,.1f} tCO2e</td></tr>\n"
            f"<tr><td>Residual</td><td>{residual:+,.2f} tCO2e ({residual_pct:+.3f}%)</td></tr>\n"
            f'<tr><td><strong>Status</strong></td>'
            f'<td class="{css}"><strong>{status}</strong></td></tr>\n'
        )
        return (
            '<div class="section">\n<h2>6. Closure Validation</h2>\n'
            "<table><thead><tr><th>Check</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render decomposition waterfall as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "decomposition_waterfall",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "methodology_description": self._get_val(data, "methodology_description", ""),
            "base_period": data.get("base_period"),
            "current_period": data.get("current_period"),
            "waterfall_bars": data.get("waterfall_bars", []),
            "entity_contributions": data.get("entity_contributions", []),
            "interpretation": self._get_val(data, "interpretation", ""),
            "closure_validation": data.get("closure_validation"),
            "chart_data": {
                "waterfall": self._build_waterfall_chart(data),
            },
        }

    def _build_waterfall_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build waterfall chart data structure for frontend rendering."""
        bars = data.get("waterfall_bars", [])
        if not bars:
            return {}
        return {
            "labels": [b.get("label", "") for b in bars],
            "values": [b.get("value", 0) for b in bars],
            "cumulative_starts": [b.get("cumulative_start", 0) for b in bars],
            "cumulative_ends": [b.get("cumulative_end", 0) for b in bars],
            "colors": [b.get("color", "#888") for b in bars],
        }
