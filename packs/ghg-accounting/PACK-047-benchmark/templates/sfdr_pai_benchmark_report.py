# -*- coding: utf-8 -*-
"""
SFDRPAIBenchmarkReport - SFDR PAI Benchmark Report for PACK-047.

Generates an SFDR-compliant Principal Adverse Impact (PAI) benchmark
report covering PAI indicators 1-3 (GHG emissions, carbon footprint,
GHG intensity of investee companies), portfolio vs benchmark index
comparison per PAI, Article 8/9 fund benchmark compliance, and EU
Taxonomy eligibility/alignment benchmark.

Regulatory References:
    - SFDR Regulation (EU) 2019/2088: Sustainability-related disclosures
    - SFDR RTS (Delegated Regulation 2022/1288): PAI indicators
    - EU Taxonomy Regulation 2020/852: Alignment and eligibility
    - SFDR Article 8: Environmental/social characteristics funds
    - SFDR Article 9: Sustainable investment objective funds
    - PCAF Global GHG Accounting Standard: Financed emissions

Sections:
    1. PAI Indicator 1: GHG Emissions (Scope 1, 2, 3, Total)
    2. PAI Indicator 2: Carbon Footprint
    3. PAI Indicator 3: GHG Intensity of Investee Companies
    4. Portfolio vs Benchmark Index Comparison
    5. Article 8/9 Fund Benchmark Compliance
    6. EU Taxonomy Eligibility/Alignment Benchmark
    7. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured SFDR reporting data)

Author: GreenLang Team
Version: 47.0.0
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

class TrafficLight(str, Enum):
    """Traffic light status indicators."""
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

class FundClassification(str, Enum):
    """SFDR fund classification."""
    ARTICLE_6 = "article_6"
    ARTICLE_8 = "article_8"
    ARTICLE_8_PLUS = "article_8_plus"
    ARTICLE_9 = "article_9"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class PAIIndicator1Row(BaseModel):
    """PAI Indicator 1: GHG emissions breakdown."""
    scope_label: str = Field(..., description="Scope label (1, 2, 3, Total)")
    portfolio_emissions_tco2e: float = Field(0.0, description="Portfolio financed emissions")
    index_emissions_tco2e: Optional[float] = Field(None, description="Index financed emissions")
    difference: Optional[float] = Field(None, description="Absolute difference")
    difference_pct: Optional[float] = Field(None, description="Relative difference (%)")
    prior_year_portfolio: Optional[float] = Field(None, description="Prior year portfolio")
    yoy_change_pct: Optional[float] = Field(None, description="Year-over-year change (%)")

class PAIIndicator2(BaseModel):
    """PAI Indicator 2: Carbon footprint."""
    portfolio_carbon_footprint: float = Field(
        0.0, description="Portfolio carbon footprint (tCO2e/M EUR invested)"
    )
    index_carbon_footprint: Optional[float] = Field(
        None, description="Index carbon footprint"
    )
    difference_pct: Optional[float] = Field(None, description="Relative difference (%)")
    prior_year: Optional[float] = Field(None, description="Prior year value")
    yoy_change_pct: Optional[float] = Field(None, description="YoY change (%)")
    currency: str = Field("EUR", description="Currency")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Status")

class PAIIndicator3(BaseModel):
    """PAI Indicator 3: GHG intensity of investee companies."""
    portfolio_ghg_intensity: float = Field(
        0.0, description="Portfolio WACI (tCO2e/M EUR revenue)"
    )
    index_ghg_intensity: Optional[float] = Field(
        None, description="Index WACI"
    )
    difference_pct: Optional[float] = Field(None, description="Relative difference (%)")
    prior_year: Optional[float] = Field(None, description="Prior year value")
    yoy_change_pct: Optional[float] = Field(None, description="YoY change (%)")
    currency: str = Field("EUR", description="Revenue currency")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Status")

class IndexComparisonEntry(BaseModel):
    """Portfolio vs benchmark index comparison for a single PAI."""
    pai_indicator: str = Field(..., description="PAI indicator label")
    portfolio_value: float = Field(0.0, description="Portfolio value")
    index_value: Optional[float] = Field(None, description="Index value")
    index_name: str = Field("", description="Index name")
    difference_pct: Optional[float] = Field(None, description="Difference (%)")
    unit: str = Field("", description="Unit")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Status")

class ArticleCompliance(BaseModel):
    """Article 8/9 fund benchmark compliance."""
    fund_name: str = Field("", description="Fund name")
    sfdr_classification: FundClassification = Field(
        FundClassification.ARTICLE_8, description="SFDR classification"
    )
    pai_1_compliant: bool = Field(False, description="PAI 1 benchmark compliant")
    pai_2_compliant: bool = Field(False, description="PAI 2 benchmark compliant")
    pai_3_compliant: bool = Field(False, description="PAI 3 benchmark compliant")
    overall_compliant: bool = Field(False, description="Overall benchmark compliant")
    benchmark_index: str = Field("", description="Benchmark index used")
    notes: str = Field("", description="Compliance notes")

class TaxonomyBenchmarkEntry(BaseModel):
    """EU Taxonomy eligibility/alignment benchmark."""
    metric_name: str = Field(..., description="Metric (eligibility or alignment)")
    portfolio_pct: float = Field(0.0, description="Portfolio percentage")
    index_pct: Optional[float] = Field(None, description="Index percentage")
    index_name: str = Field("", description="Index name")
    difference_pct: Optional[float] = Field(None, description="Difference (pp)")
    environmental_objective: str = Field(
        "climate_mitigation", description="Environmental objective"
    )
    status: TrafficLight = Field(TrafficLight.AMBER, description="Status")

class SFDRPAIBenchmarkInput(BaseModel):
    """Complete input model for SFDRPAIBenchmarkReport."""
    company_name: str = Field("Organization", description="Company / fund manager name")
    fund_name: str = Field("", description="Fund name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    pai_indicator_1: List[PAIIndicator1Row] = Field(
        default_factory=list, description="PAI 1: GHG emissions"
    )
    pai_indicator_2: Optional[PAIIndicator2] = Field(
        None, description="PAI 2: Carbon footprint"
    )
    pai_indicator_3: Optional[PAIIndicator3] = Field(
        None, description="PAI 3: GHG intensity"
    )
    index_comparisons: List[IndexComparisonEntry] = Field(
        default_factory=list, description="Index comparison per PAI"
    )
    article_compliance: List[ArticleCompliance] = Field(
        default_factory=list, description="Article 8/9 compliance"
    )
    taxonomy_benchmark: List[TaxonomyBenchmarkEntry] = Field(
        default_factory=list, description="Taxonomy benchmark"
    )
    index_name: str = Field("", description="Primary benchmark index name")
    currency: str = Field("EUR", description="Currency")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tl_label(status: TrafficLight) -> str:
    """Return uppercase label for traffic light."""
    return status.value.upper()

def _tl_color(status: TrafficLight) -> str:
    """Return hex colour for traffic light."""
    mapping = {
        TrafficLight.GREEN: "#2a9d8f",
        TrafficLight.AMBER: "#e9c46a",
        TrafficLight.RED: "#e76f51",
    }
    return mapping.get(status, "#e9c46a")

def _classification_label(cls: FundClassification) -> str:
    """Return human-readable SFDR classification label."""
    mapping = {
        FundClassification.ARTICLE_6: "Article 6",
        FundClassification.ARTICLE_8: "Article 8",
        FundClassification.ARTICLE_8_PLUS: "Article 8+",
        FundClassification.ARTICLE_9: "Article 9",
    }
    return mapping.get(cls, "Unknown")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class SFDRPAIBenchmarkReport:
    """
    SFDR PAI benchmark report template.

    Renders SFDR-compliant PAI indicators 1-3 with portfolio vs index
    comparison, Article 8/9 fund benchmark compliance, and EU Taxonomy
    eligibility/alignment benchmarks. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = SFDRPAIBenchmarkReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SFDRPAIBenchmarkReport."""
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
        """Render SFDR PAI benchmark as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render SFDR PAI benchmark as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render SFDR PAI benchmark as JSON dict."""
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
            self._md_pai_1(data),
            self._md_pai_2(data),
            self._md_pai_3(data),
            self._md_index_comparison(data),
            self._md_article_compliance(data),
            self._md_taxonomy_benchmark(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        fund = self._get_val(data, "fund_name", "")
        period = self._get_val(data, "reporting_period", "")
        title = f"# SFDR PAI Benchmark Report - {company}"
        if fund:
            title += f" ({fund})"
        return (
            f"{title}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_pai_1(self, data: Dict[str, Any]) -> str:
        """Render Markdown PAI Indicator 1: GHG Emissions."""
        rows = data.get("pai_indicator_1", [])
        if not rows:
            return ""
        index_name = self._get_val(data, "index_name", "Index")
        lines = [
            "## 1. PAI Indicator 1: GHG Emissions",
            "",
            f"| Scope | Portfolio (tCO2e) | {index_name} (tCO2e) | Difference | YoY Change |",
            f"|-------|-------------------|{'--' * len(index_name)}------------|------------|------------|",
        ]
        for r in rows:
            idx = r.get("index_emissions_tco2e")
            idx_str = f"{idx:,.0f}" if idx is not None else "-"
            diff_pct = r.get("difference_pct")
            diff_str = f"{diff_pct:+.1f}%" if diff_pct is not None else "-"
            yoy = r.get("yoy_change_pct")
            yoy_str = f"{yoy:+.1f}%" if yoy is not None else "-"
            lines.append(
                f"| {r.get('scope_label', '')} | "
                f"{r.get('portfolio_emissions_tco2e', 0):,.0f} | "
                f"{idx_str} | {diff_str} | {yoy_str} |"
            )
        return "\n".join(lines)

    def _md_pai_2(self, data: Dict[str, Any]) -> str:
        """Render Markdown PAI Indicator 2: Carbon Footprint."""
        pai2 = data.get("pai_indicator_2")
        if not pai2:
            return ""
        currency = pai2.get("currency", "EUR")
        status = TrafficLight(pai2.get("status", "amber"))
        idx = pai2.get("index_carbon_footprint")
        idx_str = f"{idx:,.2f}" if idx is not None else "-"
        diff = pai2.get("difference_pct")
        diff_str = f"{diff:+.1f}%" if diff is not None else "-"
        yoy = pai2.get("yoy_change_pct")
        yoy_str = f"{yoy:+.1f}%" if yoy is not None else "-"
        lines = [
            "## 2. PAI Indicator 2: Carbon Footprint",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Portfolio Carbon Footprint | {pai2.get('portfolio_carbon_footprint', 0):,.2f} tCO2e/M {currency} invested |",
            f"| Index Carbon Footprint | {idx_str} |",
            f"| Difference | {diff_str} |",
            f"| YoY Change | {yoy_str} |",
            f"| Status | **{_tl_label(status)}** |",
        ]
        return "\n".join(lines)

    def _md_pai_3(self, data: Dict[str, Any]) -> str:
        """Render Markdown PAI Indicator 3: GHG Intensity."""
        pai3 = data.get("pai_indicator_3")
        if not pai3:
            return ""
        currency = pai3.get("currency", "EUR")
        status = TrafficLight(pai3.get("status", "amber"))
        idx = pai3.get("index_ghg_intensity")
        idx_str = f"{idx:,.2f}" if idx is not None else "-"
        diff = pai3.get("difference_pct")
        diff_str = f"{diff:+.1f}%" if diff is not None else "-"
        yoy = pai3.get("yoy_change_pct")
        yoy_str = f"{yoy:+.1f}%" if yoy is not None else "-"
        lines = [
            "## 3. PAI Indicator 3: GHG Intensity of Investee Companies",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Portfolio WACI | {pai3.get('portfolio_ghg_intensity', 0):,.2f} tCO2e/M {currency} revenue |",
            f"| Index WACI | {idx_str} |",
            f"| Difference | {diff_str} |",
            f"| YoY Change | {yoy_str} |",
            f"| Status | **{_tl_label(status)}** |",
        ]
        return "\n".join(lines)

    def _md_index_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown index comparison summary."""
        entries = data.get("index_comparisons", [])
        if not entries:
            return ""
        lines = [
            "## 4. Portfolio vs Benchmark Index",
            "",
            "| PAI Indicator | Portfolio | Index | Index Name | Diff (%) | Unit | Status |",
            "|---------------|----------|-------|-----------|----------|------|--------|",
        ]
        for e in entries:
            idx = e.get("index_value")
            idx_str = f"{idx:,.2f}" if idx is not None else "-"
            diff = e.get("difference_pct")
            diff_str = f"{diff:+.1f}%" if diff is not None else "-"
            status = TrafficLight(e.get("status", "amber"))
            lines.append(
                f"| {e.get('pai_indicator', '')} | "
                f"{e.get('portfolio_value', 0):,.2f} | {idx_str} | "
                f"{e.get('index_name', '')} | {diff_str} | "
                f"{e.get('unit', '')} | **{_tl_label(status)}** |"
            )
        return "\n".join(lines)

    def _md_article_compliance(self, data: Dict[str, Any]) -> str:
        """Render Markdown Article 8/9 fund benchmark compliance."""
        entries = data.get("article_compliance", [])
        if not entries:
            return ""
        lines = [
            "## 5. Article 8/9 Fund Benchmark Compliance",
            "",
            "| Fund | Classification | PAI 1 | PAI 2 | PAI 3 | Overall | Index |",
            "|------|---------------|-------|-------|-------|---------|-------|",
        ]
        for e in entries:
            cls = FundClassification(e.get("sfdr_classification", "article_8"))
            pai1 = "Pass" if e.get("pai_1_compliant", False) else "Fail"
            pai2 = "Pass" if e.get("pai_2_compliant", False) else "Fail"
            pai3 = "Pass" if e.get("pai_3_compliant", False) else "Fail"
            overall = "**PASS**" if e.get("overall_compliant", False) else "**FAIL**"
            lines.append(
                f"| {e.get('fund_name', '')} | {_classification_label(cls)} | "
                f"{pai1} | {pai2} | {pai3} | {overall} | "
                f"{e.get('benchmark_index', '')} |"
            )
        return "\n".join(lines)

    def _md_taxonomy_benchmark(self, data: Dict[str, Any]) -> str:
        """Render Markdown EU Taxonomy benchmark."""
        entries = data.get("taxonomy_benchmark", [])
        if not entries:
            return ""
        lines = [
            "## 6. EU Taxonomy Eligibility/Alignment Benchmark",
            "",
            "| Metric | Portfolio (%) | Index (%) | Index Name | Diff (pp) | Status |",
            "|--------|--------------|----------|-----------|-----------|--------|",
        ]
        for e in entries:
            idx_pct = e.get("index_pct")
            idx_str = f"{idx_pct:.1f}%" if idx_pct is not None else "-"
            diff = e.get("difference_pct")
            diff_str = f"{diff:+.1f}pp" if diff is not None else "-"
            status = TrafficLight(e.get("status", "amber"))
            lines.append(
                f"| {e.get('metric_name', '')} | "
                f"{e.get('portfolio_pct', 0):.1f}% | {idx_str} | "
                f"{e.get('index_name', '')} | {diff_str} | "
                f"**{_tl_label(status)}** |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-047 GHG Benchmark v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_pai_1(data),
            self._html_pai_2(data),
            self._html_pai_3(data),
            self._html_index_comparison(data),
            self._html_article_compliance(data),
            self._html_taxonomy_benchmark(data),
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
            f"<title>SFDR PAI Benchmark - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #003399;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#e8eef6;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".tl-green{color:#2a9d8f;font-weight:700;}\n"
            ".tl-amber{color:#e9c46a;font-weight:700;}\n"
            ".tl-red{color:#e76f51;font-weight:700;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".pai-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:220px;"
            "border-left:4px solid #003399;vertical-align:top;}\n"
            ".pai-value{font-size:1.4rem;font-weight:700;color:#1b263b;}\n"
            ".pai-label{font-size:0.85rem;color:#555;}\n"
            ".compliant{color:#2a9d8f;font-weight:700;}\n"
            ".non-compliant{color:#e76f51;font-weight:700;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        fund = self._get_val(data, "fund_name", "")
        period = self._get_val(data, "reporting_period", "")
        title = f"SFDR PAI Benchmark Report &mdash; {company}"
        if fund:
            title += f" ({fund})"
        return (
            '<div class="section">\n'
            f"<h1>{title}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_pai_1(self, data: Dict[str, Any]) -> str:
        """Render HTML PAI Indicator 1."""
        rows_data = data.get("pai_indicator_1", [])
        if not rows_data:
            return ""
        rows = ""
        for r in rows_data:
            idx = r.get("index_emissions_tco2e")
            idx_str = f"{idx:,.0f}" if idx is not None else "-"
            diff = r.get("difference_pct")
            diff_str = f"{diff:+.1f}%" if diff is not None else "-"
            rows += (
                f"<tr><td>{r.get('scope_label', '')}</td>"
                f"<td>{r.get('portfolio_emissions_tco2e', 0):,.0f}</td>"
                f"<td>{idx_str}</td>"
                f"<td>{diff_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>1. PAI 1: GHG Emissions</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Portfolio (tCO2e)</th>"
            "<th>Index (tCO2e)</th><th>Difference</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_pai_2(self, data: Dict[str, Any]) -> str:
        """Render HTML PAI Indicator 2."""
        pai2 = data.get("pai_indicator_2")
        if not pai2:
            return ""
        status = TrafficLight(pai2.get("status", "amber"))
        color = _tl_color(status)
        currency = pai2.get("currency", "EUR")
        return (
            '<div class="section">\n<h2>2. PAI 2: Carbon Footprint</h2>\n'
            f'<div class="pai-card" style="border-left-color:{color};">'
            f'<div class="pai-value">'
            f'{pai2.get("portfolio_carbon_footprint", 0):,.2f}</div>'
            f'<div class="pai-label">tCO2e/M {currency} invested</div></div>\n</div>'
        )

    def _html_pai_3(self, data: Dict[str, Any]) -> str:
        """Render HTML PAI Indicator 3."""
        pai3 = data.get("pai_indicator_3")
        if not pai3:
            return ""
        status = TrafficLight(pai3.get("status", "amber"))
        color = _tl_color(status)
        currency = pai3.get("currency", "EUR")
        return (
            '<div class="section">\n<h2>3. PAI 3: GHG Intensity</h2>\n'
            f'<div class="pai-card" style="border-left-color:{color};">'
            f'<div class="pai-value">'
            f'{pai3.get("portfolio_ghg_intensity", 0):,.2f}</div>'
            f'<div class="pai-label">WACI (tCO2e/M {currency} revenue)</div></div>\n</div>'
        )

    def _html_index_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML index comparison."""
        entries = data.get("index_comparisons", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            idx = e.get("index_value")
            idx_str = f"{idx:,.2f}" if idx is not None else "-"
            diff = e.get("difference_pct")
            diff_str = f"{diff:+.1f}%" if diff is not None else "-"
            status = TrafficLight(e.get("status", "amber"))
            css = f"tl-{status.value}"
            rows += (
                f"<tr><td>{e.get('pai_indicator', '')}</td>"
                f"<td>{e.get('portfolio_value', 0):,.2f}</td>"
                f"<td>{idx_str}</td>"
                f"<td>{diff_str}</td>"
                f"<td>{e.get('unit', '')}</td>"
                f'<td class="{css}"><strong>{_tl_label(status)}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>4. Portfolio vs Index</h2>\n'
            "<table><thead><tr><th>PAI</th><th>Portfolio</th>"
            "<th>Index</th><th>Diff</th><th>Unit</th>"
            "<th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_article_compliance(self, data: Dict[str, Any]) -> str:
        """Render HTML Article 8/9 compliance."""
        entries = data.get("article_compliance", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            cls = FundClassification(e.get("sfdr_classification", "article_8"))
            overall = e.get("overall_compliant", False)
            css = "compliant" if overall else "non-compliant"
            label = "PASS" if overall else "FAIL"
            rows += (
                f"<tr><td>{e.get('fund_name', '')}</td>"
                f"<td>{_classification_label(cls)}</td>"
                f'<td class="{css}"><strong>{label}</strong></td>'
                f"<td>{e.get('benchmark_index', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Article 8/9 Compliance</h2>\n'
            "<table><thead><tr><th>Fund</th><th>Classification</th>"
            "<th>Status</th><th>Benchmark</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_taxonomy_benchmark(self, data: Dict[str, Any]) -> str:
        """Render HTML Taxonomy benchmark."""
        entries = data.get("taxonomy_benchmark", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            idx_pct = e.get("index_pct")
            idx_str = f"{idx_pct:.1f}%" if idx_pct is not None else "-"
            status = TrafficLight(e.get("status", "amber"))
            css = f"tl-{status.value}"
            rows += (
                f"<tr><td>{e.get('metric_name', '')}</td>"
                f"<td>{e.get('portfolio_pct', 0):.1f}%</td>"
                f"<td>{idx_str}</td>"
                f'<td class="{css}"><strong>{_tl_label(status)}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>6. EU Taxonomy Benchmark</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Portfolio</th>"
            "<th>Index</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-047 GHG Benchmark v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render SFDR PAI benchmark as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "sfdr_pai_benchmark_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "fund_name": self._get_val(data, "fund_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "pai_indicator_1": data.get("pai_indicator_1", []),
            "pai_indicator_2": data.get("pai_indicator_2"),
            "pai_indicator_3": data.get("pai_indicator_3"),
            "index_comparisons": data.get("index_comparisons", []),
            "article_compliance": data.get("article_compliance", []),
            "taxonomy_benchmark": data.get("taxonomy_benchmark", []),
        }
