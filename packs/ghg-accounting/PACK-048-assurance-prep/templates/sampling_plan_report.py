# -*- coding: utf-8 -*-
"""
SamplingPlanReport - Sampling Plan Report for PACK-048.

Generates a statistical sampling plan report with population description,
stratification table (stratum, count, value, sample size), high-value
items requiring 100% testing, key items for judgmental review, sample
selection methodology and parameters, and statistical parameters
(confidence level, tolerable misstatement).

Regulatory References:
    - ISAE 3410 para 46-47: Sampling and analytical procedures
    - ISO 14064-3 clause 6.3.3: Sampling approach
    - IAASB ISA 530: Audit sampling analogy
    - AA1000AS v3: Evidence gathering methodology

Sections:
    1. Population Description
    2. Stratification Table
    3. High-Value Items (100% Testing)
    4. Key Items (Judgmental Review)
    5. Sample Selection Methodology
    6. Statistical Parameters
    7. Provenance Footer

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


class SamplingMethod(str, Enum):
    """Sampling method classification."""
    MONETARY_UNIT = "monetary_unit"
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    JUDGMENTAL = "judgmental"


class SelectionReason(str, Enum):
    """Reason for item selection."""
    HIGH_VALUE = "high_value"
    HIGH_RISK = "high_risk"
    UNUSUAL_ITEM = "unusual_item"
    JUDGMENTAL = "judgmental"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class PopulationDescription(BaseModel):
    """Population description for sampling."""
    population_name: str = Field("", description="Population name")
    total_items: int = Field(0, ge=0, description="Total items in population")
    total_value_tco2e: float = Field(0.0, ge=0, description="Total value in tCO2e")
    scope: str = Field("", description="GHG scope covered")
    description: str = Field("", description="Population description")
    data_source: str = Field("", description="Data source")
    period_start: Optional[str] = Field(None, description="Period start (ISO)")
    period_end: Optional[str] = Field(None, description="Period end (ISO)")


class StratumEntry(BaseModel):
    """Single stratum in the stratification table."""
    stratum_id: str = Field("", description="Stratum identifier")
    stratum_name: str = Field(..., description="Stratum name")
    item_count: int = Field(0, ge=0, description="Items in stratum")
    total_value_tco2e: float = Field(0.0, ge=0, description="Stratum total tCO2e")
    pct_of_population: float = Field(0.0, ge=0, le=100, description="% of population value")
    sample_size: int = Field(0, ge=0, description="Planned sample size")
    sampling_method: SamplingMethod = Field(
        SamplingMethod.RANDOM, description="Sampling method"
    )
    coverage_pct: float = Field(0.0, ge=0, le=100, description="Coverage % by value")


class HighValueItem(BaseModel):
    """Item requiring 100% testing."""
    item_id: str = Field("", description="Item identifier")
    description: str = Field(..., description="Item description")
    value_tco2e: float = Field(0.0, ge=0, description="Item value tCO2e")
    scope: str = Field("", description="GHG scope")
    category: str = Field("", description="Emissions category")
    reason: str = Field("high_value", description="Reason for 100% testing")
    threshold_exceeded: Optional[float] = Field(None, description="Threshold exceeded (tCO2e)")


class KeyItem(BaseModel):
    """Item selected for judgmental review."""
    item_id: str = Field("", description="Item identifier")
    description: str = Field(..., description="Item description")
    value_tco2e: float = Field(0.0, ge=0, description="Item value tCO2e")
    selection_reason: SelectionReason = Field(
        SelectionReason.JUDGMENTAL, description="Selection reason"
    )
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors identified")
    notes: str = Field("", description="Review notes")


class SampleMethodology(BaseModel):
    """Sample selection methodology description."""
    primary_method: SamplingMethod = Field(
        SamplingMethod.STRATIFIED, description="Primary sampling method"
    )
    description: str = Field("", description="Methodology description")
    software_used: str = Field("", description="Sampling software / tool used")
    seed_value: Optional[int] = Field(None, description="Random seed (for reproducibility)")
    selection_criteria: List[str] = Field(
        default_factory=list, description="Selection criteria"
    )


class StatisticalParameters(BaseModel):
    """Statistical sampling parameters."""
    confidence_level_pct: float = Field(95.0, ge=0, le=100, description="Confidence level (%)")
    tolerable_misstatement_tco2e: float = Field(
        0.0, ge=0, description="Tolerable misstatement (tCO2e)"
    )
    tolerable_misstatement_pct: float = Field(
        0.0, ge=0, le=100, description="Tolerable misstatement (% of population)"
    )
    expected_misstatement_tco2e: float = Field(
        0.0, ge=0, description="Expected misstatement (tCO2e)"
    )
    expected_misstatement_pct: float = Field(
        0.0, ge=0, le=100, description="Expected misstatement (%)"
    )
    precision_tco2e: Optional[float] = Field(None, ge=0, description="Precision (tCO2e)")
    risk_of_material_misstatement: str = Field("", description="Risk assessment")


class SamplingPlanInput(BaseModel):
    """Complete input model for SamplingPlanReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    population: Optional[PopulationDescription] = Field(
        None, description="Population description"
    )
    strata: List[StratumEntry] = Field(
        default_factory=list, description="Stratification table"
    )
    high_value_items: List[HighValueItem] = Field(
        default_factory=list, description="High-value items (100% testing)"
    )
    key_items: List[KeyItem] = Field(
        default_factory=list, description="Key items (judgmental)"
    )
    methodology: Optional[SampleMethodology] = Field(
        None, description="Sample selection methodology"
    )
    statistical_params: Optional[StatisticalParameters] = Field(
        None, description="Statistical parameters"
    )


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class SamplingPlanReport:
    """
    Sampling plan report template for PACK-048.

    Renders a sampling plan with population description, stratification,
    high-value and key items, methodology, and statistical parameters.
    All outputs include SHA-256 provenance hashing for audit-trail
    integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = SamplingPlanReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SamplingPlanReport."""
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
        """Render sampling plan as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render sampling plan as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render sampling plan as JSON dict."""
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
            self._md_population(data),
            self._md_stratification(data),
            self._md_high_value_items(data),
            self._md_key_items(data),
            self._md_methodology(data),
            self._md_statistical_params(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Sampling Plan Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_population(self, data: Dict[str, Any]) -> str:
        """Render Markdown population description."""
        pop = data.get("population")
        if not pop:
            return ""
        lines = [
            "## 1. Population Description",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Population | {pop.get('population_name', '')} |",
            f"| Total Items | {pop.get('total_items', 0):,} |",
            f"| Total Value | {pop.get('total_value_tco2e', 0):,.1f} tCO2e |",
            f"| Scope | {pop.get('scope', '')} |",
            f"| Data Source | {pop.get('data_source', '')} |",
        ]
        desc = pop.get("description", "")
        if desc:
            lines.append("")
            lines.append(f"**Description:** {desc}")
        return "\n".join(lines)

    def _md_stratification(self, data: Dict[str, Any]) -> str:
        """Render Markdown stratification table."""
        strata = data.get("strata", [])
        if not strata:
            return ""
        lines = [
            "## 2. Stratification Table",
            "",
            "| Stratum | Items | Value (tCO2e) | % of Pop | Sample Size | Method | Coverage % |",
            "|---------|-------|---------------|----------|-------------|--------|------------|",
        ]
        for s in strata:
            method = s.get("sampling_method", "random").replace("_", " ").title()
            lines.append(
                f"| {s.get('stratum_name', '')} | "
                f"{s.get('item_count', 0):,} | "
                f"{s.get('total_value_tco2e', 0):,.1f} | "
                f"{s.get('pct_of_population', 0):.1f}% | "
                f"{s.get('sample_size', 0):,} | "
                f"{method} | "
                f"{s.get('coverage_pct', 0):.1f}% |"
            )
        return "\n".join(lines)

    def _md_high_value_items(self, data: Dict[str, Any]) -> str:
        """Render Markdown high-value items."""
        items = data.get("high_value_items", [])
        if not items:
            return ""
        lines = [
            "## 3. High-Value Items (100% Testing)",
            "",
            "| ID | Description | Value (tCO2e) | Scope | Category | Reason |",
            "|----|-------------|---------------|-------|----------|--------|",
        ]
        for item in items:
            lines.append(
                f"| {item.get('item_id', '')} | "
                f"{item.get('description', '')} | "
                f"{item.get('value_tco2e', 0):,.1f} | "
                f"{item.get('scope', '')} | "
                f"{item.get('category', '')} | "
                f"{item.get('reason', '')} |"
            )
        return "\n".join(lines)

    def _md_key_items(self, data: Dict[str, Any]) -> str:
        """Render Markdown key items for judgmental review."""
        items = data.get("key_items", [])
        if not items:
            return ""
        lines = [
            "## 4. Key Items (Judgmental Review)",
            "",
            "| ID | Description | Value (tCO2e) | Reason | Risk Factors |",
            "|----|-------------|---------------|--------|-------------|",
        ]
        for item in items:
            reason = item.get("selection_reason", "judgmental").replace("_", " ").title()
            risks = "; ".join(item.get("risk_factors", [])[:3]) or "-"
            lines.append(
                f"| {item.get('item_id', '')} | "
                f"{item.get('description', '')} | "
                f"{item.get('value_tco2e', 0):,.1f} | "
                f"{reason} | "
                f"{risks} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown sample methodology."""
        meth = data.get("methodology")
        if not meth:
            return ""
        method = meth.get("primary_method", "stratified").replace("_", " ").title()
        lines = [
            "## 5. Sample Selection Methodology",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Primary Method | {method} |",
        ]
        software = meth.get("software_used", "")
        if software:
            lines.append(f"| Software | {software} |")
        seed = meth.get("seed_value")
        if seed is not None:
            lines.append(f"| Random Seed | {seed} |")
        desc = meth.get("description", "")
        if desc:
            lines.append("")
            lines.append(f"**Description:** {desc}")
        criteria = meth.get("selection_criteria", [])
        if criteria:
            lines.append("")
            lines.append("**Selection Criteria:**")
            for c in criteria:
                lines.append(f"- {c}")
        return "\n".join(lines)

    def _md_statistical_params(self, data: Dict[str, Any]) -> str:
        """Render Markdown statistical parameters."""
        sp = data.get("statistical_params")
        if not sp:
            return ""
        lines = [
            "## 6. Statistical Parameters",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Confidence Level | {sp.get('confidence_level_pct', 95):.0f}% |",
            f"| Tolerable Misstatement | {sp.get('tolerable_misstatement_tco2e', 0):,.1f} tCO2e ({sp.get('tolerable_misstatement_pct', 0):.1f}%) |",
            f"| Expected Misstatement | {sp.get('expected_misstatement_tco2e', 0):,.1f} tCO2e ({sp.get('expected_misstatement_pct', 0):.1f}%) |",
        ]
        precision = sp.get("precision_tco2e")
        if precision is not None:
            lines.append(f"| Precision | {precision:,.1f} tCO2e |")
        risk = sp.get("risk_of_material_misstatement", "")
        if risk:
            lines.append(f"| Risk of Material Misstatement | {risk} |")
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
            self._html_population(data),
            self._html_stratification(data),
            self._html_high_value_items(data),
            self._html_key_items(data),
            self._html_methodology(data),
            self._html_statistical_params(data),
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
            f"<title>Sampling Plan - {company}</title>\n"
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
            ".high-value{background:#fff3e0;}\n"
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
            f"<h1>Sampling Plan Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_population(self, data: Dict[str, Any]) -> str:
        """Render HTML population description."""
        pop = data.get("population")
        if not pop:
            return ""
        rows = (
            f"<tr><td>Population</td><td>{pop.get('population_name', '')}</td></tr>\n"
            f"<tr><td>Total Items</td><td>{pop.get('total_items', 0):,}</td></tr>\n"
            f"<tr><td>Total Value</td><td>{pop.get('total_value_tco2e', 0):,.1f} tCO2e</td></tr>\n"
            f"<tr><td>Scope</td><td>{pop.get('scope', '')}</td></tr>\n"
            f"<tr><td>Data Source</td><td>{pop.get('data_source', '')}</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>1. Population Description</h2>\n'
            "<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_stratification(self, data: Dict[str, Any]) -> str:
        """Render HTML stratification table."""
        strata = data.get("strata", [])
        if not strata:
            return ""
        rows = ""
        for s in strata:
            method = s.get("sampling_method", "random").replace("_", " ").title()
            rows += (
                f"<tr><td>{s.get('stratum_name', '')}</td>"
                f"<td>{s.get('item_count', 0):,}</td>"
                f"<td>{s.get('total_value_tco2e', 0):,.1f}</td>"
                f"<td>{s.get('pct_of_population', 0):.1f}%</td>"
                f"<td>{s.get('sample_size', 0):,}</td>"
                f"<td>{method}</td>"
                f"<td>{s.get('coverage_pct', 0):.1f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Stratification Table</h2>\n'
            "<table><thead><tr><th>Stratum</th><th>Items</th><th>Value (tCO2e)</th>"
            "<th>% of Pop</th><th>Sample Size</th><th>Method</th>"
            "<th>Coverage %</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_high_value_items(self, data: Dict[str, Any]) -> str:
        """Render HTML high-value items."""
        items = data.get("high_value_items", [])
        if not items:
            return ""
        rows = ""
        for item in items:
            rows += (
                f'<tr class="high-value"><td>{item.get("item_id", "")}</td>'
                f"<td>{item.get('description', '')}</td>"
                f"<td>{item.get('value_tco2e', 0):,.1f}</td>"
                f"<td>{item.get('scope', '')}</td>"
                f"<td>{item.get('category', '')}</td>"
                f"<td>{item.get('reason', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. High-Value Items (100% Testing)</h2>\n'
            "<table><thead><tr><th>ID</th><th>Description</th><th>Value (tCO2e)</th>"
            "<th>Scope</th><th>Category</th><th>Reason</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_key_items(self, data: Dict[str, Any]) -> str:
        """Render HTML key items."""
        items = data.get("key_items", [])
        if not items:
            return ""
        rows = ""
        for item in items:
            reason = item.get("selection_reason", "judgmental").replace("_", " ").title()
            risks = "; ".join(item.get("risk_factors", [])[:3]) or "-"
            rows += (
                f"<tr><td>{item.get('item_id', '')}</td>"
                f"<td>{item.get('description', '')}</td>"
                f"<td>{item.get('value_tco2e', 0):,.1f}</td>"
                f"<td>{reason}</td>"
                f"<td>{risks}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Key Items (Judgmental Review)</h2>\n'
            "<table><thead><tr><th>ID</th><th>Description</th><th>Value (tCO2e)</th>"
            "<th>Reason</th><th>Risk Factors</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology section."""
        meth = data.get("methodology")
        if not meth:
            return ""
        method = meth.get("primary_method", "stratified").replace("_", " ").title()
        rows = f"<tr><td>Primary Method</td><td>{method}</td></tr>\n"
        software = meth.get("software_used", "")
        if software:
            rows += f"<tr><td>Software</td><td>{software}</td></tr>\n"
        seed = meth.get("seed_value")
        if seed is not None:
            rows += f"<tr><td>Random Seed</td><td>{seed}</td></tr>\n"
        return (
            '<div class="section">\n<h2>5. Sample Selection Methodology</h2>\n'
            "<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_statistical_params(self, data: Dict[str, Any]) -> str:
        """Render HTML statistical parameters."""
        sp = data.get("statistical_params")
        if not sp:
            return ""
        rows = (
            f"<tr><td>Confidence Level</td><td>{sp.get('confidence_level_pct', 95):.0f}%</td></tr>\n"
            f"<tr><td>Tolerable Misstatement</td><td>{sp.get('tolerable_misstatement_tco2e', 0):,.1f} tCO2e ({sp.get('tolerable_misstatement_pct', 0):.1f}%)</td></tr>\n"
            f"<tr><td>Expected Misstatement</td><td>{sp.get('expected_misstatement_tco2e', 0):,.1f} tCO2e ({sp.get('expected_misstatement_pct', 0):.1f}%)</td></tr>\n"
        )
        precision = sp.get("precision_tco2e")
        if precision is not None:
            rows += f"<tr><td>Precision</td><td>{precision:,.1f} tCO2e</td></tr>\n"
        risk = sp.get("risk_of_material_misstatement", "")
        if risk:
            rows += f"<tr><td>Risk of Material Misstatement</td><td>{risk}</td></tr>\n"
        return (
            '<div class="section">\n<h2>6. Statistical Parameters</h2>\n'
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n"
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
        """Render sampling plan as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "sampling_plan_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "population": data.get("population"),
            "strata": data.get("strata", []),
            "high_value_items": data.get("high_value_items", []),
            "key_items": data.get("key_items", []),
            "methodology": data.get("methodology"),
            "statistical_params": data.get("statistical_params"),
            "chart_data": {
                "stratification_pie": self._build_strata_pie(data),
                "sample_coverage_bar": self._build_coverage_bar(data),
            },
        }

    def _build_strata_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build stratification pie chart data."""
        strata = data.get("strata", [])
        if not strata:
            return {}
        return {
            "labels": [s.get("stratum_name", "") for s in strata],
            "values": [s.get("total_value_tco2e", 0) for s in strata],
        }

    def _build_coverage_bar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build sample coverage bar chart data."""
        strata = data.get("strata", [])
        if not strata:
            return {}
        return {
            "labels": [s.get("stratum_name", "") for s in strata],
            "coverage_pct": [s.get("coverage_pct", 0) for s in strata],
            "sample_sizes": [s.get("sample_size", 0) for s in strata],
        }
