# -*- coding: utf-8 -*-
"""
PortfolioBenchmarkReport - Portfolio Benchmark Report for PACK-047.

Generates a portfolio-level benchmark report with WACI comparison
(portfolio vs index), carbon footprint trends, sector attribution
waterfall, top 10 contributors table, PCAF data quality distribution,
and holdings-level heatmap data.

Regulatory References:
    - TCFD Recommendations: Portfolio carbon metrics
    - PCAF Global GHG Accounting Standard for Financial Industry
    - EU SFDR: PAI indicators for investment portfolios
    - EU Benchmark Regulation 2019/2089: Climate benchmarks
    - Task Force on Climate-Related Financial Disclosures (TCFD)

Sections:
    1. WACI Comparison (Portfolio vs Index)
    2. Carbon Footprint Trends
    3. Sector Attribution Waterfall
    4. Top 10 Contributors
    5. PCAF Data Quality Distribution
    6. Holdings-Level Heatmap Data
    7. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

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


class PCAFScore(int, Enum):
    """PCAF data quality score (1=highest, 5=lowest)."""
    SCORE_1 = 1
    SCORE_2 = 2
    SCORE_3 = 3
    SCORE_4 = 4
    SCORE_5 = 5


class WaterfallBarType(str, Enum):
    """Waterfall chart bar type."""
    INCREASE = "increase"
    DECREASE = "decrease"
    TOTAL = "total"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class WACIComparison(BaseModel):
    """Weighted Average Carbon Intensity comparison."""
    portfolio_waci: float = Field(0.0, description="Portfolio WACI (tCO2e/M revenue)")
    index_waci: float = Field(0.0, description="Benchmark index WACI")
    index_name: str = Field("", description="Benchmark index name")
    difference: float = Field(0.0, description="Absolute difference")
    difference_pct: float = Field(0.0, description="Relative difference (%)")
    portfolio_name: str = Field("", description="Portfolio name")
    currency: str = Field("EUR", description="Revenue currency")
    as_of_date: str = Field("", description="Measurement date (ISO)")


class CarbonFootprintTrend(BaseModel):
    """Carbon footprint for a single period."""
    period: str = Field(..., description="Period label (e.g., 2023, Q1 2024)")
    portfolio_footprint: float = Field(0.0, description="Portfolio carbon footprint (tCO2e)")
    index_footprint: Optional[float] = Field(None, description="Index carbon footprint")
    portfolio_waci: float = Field(0.0, description="Portfolio WACI")
    index_waci: Optional[float] = Field(None, description="Index WACI")
    aum: Optional[float] = Field(None, description="Assets under management")


class SectorAttributionEntry(BaseModel):
    """Sector attribution waterfall entry."""
    sector: str = Field(..., description="Sector name / GICS code")
    allocation_effect: float = Field(0.0, description="Allocation effect (tCO2e/M)")
    selection_effect: float = Field(0.0, description="Selection effect (tCO2e/M)")
    interaction_effect: float = Field(0.0, description="Interaction effect (tCO2e/M)")
    total_effect: float = Field(0.0, description="Total attribution (tCO2e/M)")
    bar_type: WaterfallBarType = Field(
        WaterfallBarType.INCREASE, description="Waterfall bar type"
    )


class TopContributor(BaseModel):
    """Top contributor to portfolio carbon footprint."""
    rank: int = Field(0, ge=0, description="Rank (1=highest contributor)")
    entity_name: str = Field(..., description="Company / issuer name")
    sector: str = Field("", description="Sector")
    weight_pct: float = Field(0.0, description="Portfolio weight (%)")
    emissions_tco2e: float = Field(0.0, description="Financed emissions (tCO2e)")
    contribution_pct: float = Field(0.0, description="% of portfolio carbon footprint")
    intensity: Optional[float] = Field(None, description="Carbon intensity")
    pcaf_score: Optional[int] = Field(None, ge=1, le=5, description="PCAF data quality")


class PCAFDistribution(BaseModel):
    """PCAF data quality score distribution."""
    score_1_pct: float = Field(0.0, description="% of portfolio at PCAF Score 1")
    score_2_pct: float = Field(0.0, description="% at PCAF Score 2")
    score_3_pct: float = Field(0.0, description="% at PCAF Score 3")
    score_4_pct: float = Field(0.0, description="% at PCAF Score 4")
    score_5_pct: float = Field(0.0, description="% at PCAF Score 5")
    weighted_average_score: float = Field(0.0, description="Weighted average PCAF score")
    coverage_pct: float = Field(0.0, description="% of portfolio with emissions data")


class HoldingHeatmapEntry(BaseModel):
    """Single holding in the heatmap."""
    entity_name: str = Field(..., description="Issuer name")
    sector: str = Field("", description="Sector")
    weight_pct: float = Field(0.0, description="Portfolio weight (%)")
    carbon_intensity: float = Field(0.0, description="Carbon intensity")
    yoy_change_pct: Optional[float] = Field(None, description="YoY intensity change (%)")
    pcaf_score: Optional[int] = Field(None, ge=1, le=5, description="PCAF score")
    heat_category: str = Field("medium", description="Heat category (low/medium/high)")


class PortfolioBenchmarkInput(BaseModel):
    """Complete input model for PortfolioBenchmarkReport."""
    company_name: str = Field("Organization", description="Company / fund manager name")
    portfolio_name: str = Field("", description="Portfolio name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    waci_comparison: Optional[WACIComparison] = Field(
        None, description="WACI comparison data"
    )
    carbon_footprint_trends: List[CarbonFootprintTrend] = Field(
        default_factory=list, description="Carbon footprint trend data"
    )
    sector_attribution: List[SectorAttributionEntry] = Field(
        default_factory=list, description="Sector attribution waterfall"
    )
    top_contributors: List[TopContributor] = Field(
        default_factory=list, description="Top 10 contributors"
    )
    pcaf_distribution: Optional[PCAFDistribution] = Field(
        None, description="PCAF quality distribution"
    )
    holdings_heatmap: List[HoldingHeatmapEntry] = Field(
        default_factory=list, description="Holdings heatmap data"
    )


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class PortfolioBenchmarkReport:
    """
    Portfolio benchmark report template for GHG emissions benchmarking.

    Renders portfolio-level carbon metrics with WACI comparison, footprint
    trends, sector attribution, top contributors, PCAF quality distribution,
    and holdings heatmap. All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = PortfolioBenchmarkReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PortfolioBenchmarkReport."""
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
        """Render portfolio benchmark as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render portfolio benchmark as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render portfolio benchmark as JSON dict."""
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
            self._md_waci(data),
            self._md_footprint_trends(data),
            self._md_sector_attribution(data),
            self._md_top_contributors(data),
            self._md_pcaf(data),
            self._md_heatmap(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        portfolio = self._get_val(data, "portfolio_name", "")
        period = self._get_val(data, "reporting_period", "")
        title = f"# Portfolio Benchmark Report - {company}"
        if portfolio:
            title += f" ({portfolio})"
        return (
            f"{title}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_waci(self, data: Dict[str, Any]) -> str:
        """Render Markdown WACI comparison."""
        waci = data.get("waci_comparison")
        if not waci:
            return ""
        currency = waci.get("currency", "EUR")
        lines = [
            "## 1. WACI Comparison (Portfolio vs Index)",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Portfolio WACI | {waci.get('portfolio_waci', 0):,.2f} tCO2e/M {currency} |",
            f"| Index ({waci.get('index_name', '')}) | {waci.get('index_waci', 0):,.2f} tCO2e/M {currency} |",
            f"| Difference | {waci.get('difference', 0):+,.2f} ({waci.get('difference_pct', 0):+.1f}%) |",
            f"| As of Date | {waci.get('as_of_date', '')} |",
        ]
        return "\n".join(lines)

    def _md_footprint_trends(self, data: Dict[str, Any]) -> str:
        """Render Markdown carbon footprint trends table."""
        trends = data.get("carbon_footprint_trends", [])
        if not trends:
            return ""
        lines = [
            "## 2. Carbon Footprint Trends",
            "",
            "| Period | Portfolio Footprint (tCO2e) | Index Footprint | Portfolio WACI | Index WACI |",
            "|--------|----------------------------|-----------------|----------------|------------|",
        ]
        for t in trends:
            idx_fp = t.get("index_footprint")
            idx_fp_str = f"{idx_fp:,.0f}" if idx_fp is not None else "-"
            idx_waci = t.get("index_waci")
            idx_waci_str = f"{idx_waci:,.2f}" if idx_waci is not None else "-"
            lines.append(
                f"| {t.get('period', '')} | {t.get('portfolio_footprint', 0):,.0f} | "
                f"{idx_fp_str} | {t.get('portfolio_waci', 0):,.2f} | {idx_waci_str} |"
            )
        return "\n".join(lines)

    def _md_sector_attribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown sector attribution waterfall."""
        entries = data.get("sector_attribution", [])
        if not entries:
            return ""
        lines = [
            "## 3. Sector Attribution Waterfall",
            "",
            "| Sector | Allocation | Selection | Interaction | Total |",
            "|--------|------------|-----------|-------------|-------|",
        ]
        for e in entries:
            lines.append(
                f"| {e.get('sector', '')} | {e.get('allocation_effect', 0):+,.2f} | "
                f"{e.get('selection_effect', 0):+,.2f} | "
                f"{e.get('interaction_effect', 0):+,.2f} | "
                f"{e.get('total_effect', 0):+,.2f} |"
            )
        return "\n".join(lines)

    def _md_top_contributors(self, data: Dict[str, Any]) -> str:
        """Render Markdown top 10 contributors table."""
        contributors = data.get("top_contributors", [])
        if not contributors:
            return ""
        lines = [
            "## 4. Top 10 Contributors",
            "",
            "| Rank | Entity | Sector | Weight (%) | Emissions (tCO2e) | Contribution (%) | PCAF |",
            "|------|--------|--------|-----------|-------------------|-------------------|------|",
        ]
        for c in contributors[:10]:
            pcaf = c.get("pcaf_score")
            pcaf_str = str(pcaf) if pcaf is not None else "-"
            lines.append(
                f"| {c.get('rank', 0)} | {c.get('entity_name', '')} | "
                f"{c.get('sector', '')} | {c.get('weight_pct', 0):.1f} | "
                f"{c.get('emissions_tco2e', 0):,.0f} | "
                f"{c.get('contribution_pct', 0):.1f} | {pcaf_str} |"
            )
        return "\n".join(lines)

    def _md_pcaf(self, data: Dict[str, Any]) -> str:
        """Render Markdown PCAF data quality distribution."""
        pcaf = data.get("pcaf_distribution")
        if not pcaf:
            return ""
        lines = [
            "## 5. PCAF Data Quality Distribution",
            "",
            "| Score | Description | Portfolio (%) |",
            "|-------|-------------|---------------|",
            f"| 1 | Verified emissions | {pcaf.get('score_1_pct', 0):.1f}% |",
            f"| 2 | Reported emissions | {pcaf.get('score_2_pct', 0):.1f}% |",
            f"| 3 | Physical activity data | {pcaf.get('score_3_pct', 0):.1f}% |",
            f"| 4 | Economic activity data | {pcaf.get('score_4_pct', 0):.1f}% |",
            f"| 5 | Estimated data | {pcaf.get('score_5_pct', 0):.1f}% |",
            "",
            f"**Weighted Average Score:** {pcaf.get('weighted_average_score', 0):.1f}",
            f"**Data Coverage:** {pcaf.get('coverage_pct', 0):.1f}%",
        ]
        return "\n".join(lines)

    def _md_heatmap(self, data: Dict[str, Any]) -> str:
        """Render Markdown holdings heatmap data."""
        holdings = data.get("holdings_heatmap", [])
        if not holdings:
            return ""
        lines = [
            "## 6. Holdings Heatmap",
            "",
            "| Entity | Sector | Weight (%) | Intensity | YoY Change | Heat |",
            "|--------|--------|-----------|-----------|------------|------|",
        ]
        for h in holdings:
            yoy = h.get("yoy_change_pct")
            yoy_str = f"{yoy:+.1f}%" if yoy is not None else "-"
            lines.append(
                f"| {h.get('entity_name', '')} | {h.get('sector', '')} | "
                f"{h.get('weight_pct', 0):.1f} | {h.get('carbon_intensity', 0):,.2f} | "
                f"{yoy_str} | {h.get('heat_category', '')} |"
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
            self._html_waci(data),
            self._html_footprint_trends(data),
            self._html_sector_attribution(data),
            self._html_top_contributors(data),
            self._html_pcaf(data),
            self._html_heatmap(data),
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
            f"<title>Portfolio Benchmark - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #1b4332;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".waci-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:200px;"
            "border-left:4px solid #1b4332;vertical-align:top;}\n"
            ".waci-value{font-size:1.6rem;font-weight:700;color:#1b263b;}\n"
            ".waci-label{font-size:0.85rem;color:#555;}\n"
            ".heat-low{background:#e8f5e9;}\n"
            ".heat-medium{background:#fff8e1;}\n"
            ".heat-high{background:#ffebee;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        portfolio = self._get_val(data, "portfolio_name", "")
        period = self._get_val(data, "reporting_period", "")
        title = f"Portfolio Benchmark Report &mdash; {company}"
        if portfolio:
            title += f" ({portfolio})"
        return (
            '<div class="section">\n'
            f"<h1>{title}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_waci(self, data: Dict[str, Any]) -> str:
        """Render HTML WACI comparison cards."""
        waci = data.get("waci_comparison")
        if not waci:
            return ""
        currency = waci.get("currency", "EUR")
        diff_pct = waci.get("difference_pct", 0)
        diff_color = "#2a9d8f" if diff_pct < 0 else "#e76f51"
        cards = (
            f'<div class="waci-card">'
            f'<div class="waci-value">{waci.get("portfolio_waci", 0):,.2f}</div>'
            f'<div class="waci-label">Portfolio WACI (tCO2e/M {currency})</div></div>\n'
            f'<div class="waci-card">'
            f'<div class="waci-value">{waci.get("index_waci", 0):,.2f}</div>'
            f'<div class="waci-label">{waci.get("index_name", "Index")} WACI</div></div>\n'
            f'<div class="waci-card" style="border-left-color:{diff_color};">'
            f'<div class="waci-value">{diff_pct:+.1f}%</div>'
            f'<div class="waci-label">Difference</div></div>\n'
        )
        return (
            '<div class="section">\n<h2>1. WACI Comparison</h2>\n'
            f"<div>{cards}</div>\n</div>"
        )

    def _html_footprint_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML carbon footprint trends."""
        trends = data.get("carbon_footprint_trends", [])
        if not trends:
            return ""
        rows = ""
        for t in trends:
            idx_fp = t.get("index_footprint")
            idx_fp_str = f"{idx_fp:,.0f}" if idx_fp is not None else "-"
            rows += (
                f"<tr><td>{t.get('period', '')}</td>"
                f"<td>{t.get('portfolio_footprint', 0):,.0f}</td>"
                f"<td>{idx_fp_str}</td>"
                f"<td>{t.get('portfolio_waci', 0):,.2f}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Carbon Footprint Trends</h2>\n'
            "<table><thead><tr><th>Period</th><th>Portfolio (tCO2e)</th>"
            "<th>Index (tCO2e)</th><th>Portfolio WACI</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sector_attribution(self, data: Dict[str, Any]) -> str:
        """Render HTML sector attribution waterfall."""
        entries = data.get("sector_attribution", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            rows += (
                f"<tr><td>{e.get('sector', '')}</td>"
                f"<td>{e.get('allocation_effect', 0):+,.2f}</td>"
                f"<td>{e.get('selection_effect', 0):+,.2f}</td>"
                f"<td>{e.get('interaction_effect', 0):+,.2f}</td>"
                f"<td><strong>{e.get('total_effect', 0):+,.2f}</strong></td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Sector Attribution</h2>\n'
            "<table><thead><tr><th>Sector</th><th>Allocation</th>"
            "<th>Selection</th><th>Interaction</th><th>Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_top_contributors(self, data: Dict[str, Any]) -> str:
        """Render HTML top 10 contributors."""
        contributors = data.get("top_contributors", [])
        if not contributors:
            return ""
        rows = ""
        for c in contributors[:10]:
            pcaf = c.get("pcaf_score")
            pcaf_str = str(pcaf) if pcaf is not None else "-"
            rows += (
                f"<tr><td>{c.get('rank', 0)}</td>"
                f"<td>{c.get('entity_name', '')}</td>"
                f"<td>{c.get('sector', '')}</td>"
                f"<td>{c.get('weight_pct', 0):.1f}%</td>"
                f"<td>{c.get('emissions_tco2e', 0):,.0f}</td>"
                f"<td>{c.get('contribution_pct', 0):.1f}%</td>"
                f"<td>{pcaf_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Top 10 Contributors</h2>\n'
            "<table><thead><tr><th>Rank</th><th>Entity</th><th>Sector</th>"
            "<th>Weight</th><th>Emissions</th><th>Contribution</th>"
            "<th>PCAF</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_pcaf(self, data: Dict[str, Any]) -> str:
        """Render HTML PCAF distribution."""
        pcaf = data.get("pcaf_distribution")
        if not pcaf:
            return ""
        rows = (
            f"<tr><td>1</td><td>Verified emissions</td><td>{pcaf.get('score_1_pct', 0):.1f}%</td></tr>\n"
            f"<tr><td>2</td><td>Reported emissions</td><td>{pcaf.get('score_2_pct', 0):.1f}%</td></tr>\n"
            f"<tr><td>3</td><td>Physical activity data</td><td>{pcaf.get('score_3_pct', 0):.1f}%</td></tr>\n"
            f"<tr><td>4</td><td>Economic activity data</td><td>{pcaf.get('score_4_pct', 0):.1f}%</td></tr>\n"
            f"<tr><td>5</td><td>Estimated data</td><td>{pcaf.get('score_5_pct', 0):.1f}%</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>5. PCAF Data Quality</h2>\n'
            "<table><thead><tr><th>Score</th><th>Description</th>"
            "<th>Portfolio (%)</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n"
            f"<p><strong>Weighted Average:</strong> {pcaf.get('weighted_average_score', 0):.1f} | "
            f"<strong>Coverage:</strong> {pcaf.get('coverage_pct', 0):.1f}%</p>\n</div>"
        )

    def _html_heatmap(self, data: Dict[str, Any]) -> str:
        """Render HTML holdings heatmap."""
        holdings = data.get("holdings_heatmap", [])
        if not holdings:
            return ""
        rows = ""
        for h in holdings:
            heat = h.get("heat_category", "medium")
            css = f"heat-{heat}"
            yoy = h.get("yoy_change_pct")
            yoy_str = f"{yoy:+.1f}%" if yoy is not None else "-"
            rows += (
                f'<tr class="{css}"><td>{h.get("entity_name", "")}</td>'
                f"<td>{h.get('sector', '')}</td>"
                f"<td>{h.get('weight_pct', 0):.1f}%</td>"
                f"<td>{h.get('carbon_intensity', 0):,.2f}</td>"
                f"<td>{yoy_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. Holdings Heatmap</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Sector</th>"
            "<th>Weight</th><th>Intensity</th><th>YoY Change</th></tr></thead>\n"
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
        """Render portfolio benchmark as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "portfolio_benchmark_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "portfolio_name": self._get_val(data, "portfolio_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "waci_comparison": data.get("waci_comparison"),
            "carbon_footprint_trends": data.get("carbon_footprint_trends", []),
            "sector_attribution": data.get("sector_attribution", []),
            "top_contributors": data.get("top_contributors", []),
            "pcaf_distribution": data.get("pcaf_distribution"),
            "holdings_heatmap": data.get("holdings_heatmap", []),
            "chart_data": {
                "waci_comparison_bar": self._build_waci_chart(data),
                "footprint_trend_line": self._build_trend_chart(data),
                "sector_waterfall": self._build_waterfall_chart(data),
                "pcaf_pie": self._build_pcaf_chart(data),
            },
        }

    def _build_waci_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build WACI comparison chart data."""
        waci = data.get("waci_comparison")
        if not waci:
            return {}
        return {
            "labels": ["Portfolio", waci.get("index_name", "Index")],
            "values": [waci.get("portfolio_waci", 0), waci.get("index_waci", 0)],
        }

    def _build_trend_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build footprint trend chart data."""
        trends = data.get("carbon_footprint_trends", [])
        if not trends:
            return {}
        return {
            "periods": [t.get("period", "") for t in trends],
            "portfolio_waci": [t.get("portfolio_waci", 0) for t in trends],
            "index_waci": [t.get("index_waci") for t in trends],
        }

    def _build_waterfall_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build sector attribution waterfall chart data."""
        entries = data.get("sector_attribution", [])
        if not entries:
            return {}
        return {
            "sectors": [e.get("sector", "") for e in entries],
            "allocation": [e.get("allocation_effect", 0) for e in entries],
            "selection": [e.get("selection_effect", 0) for e in entries],
            "interaction": [e.get("interaction_effect", 0) for e in entries],
            "total": [e.get("total_effect", 0) for e in entries],
        }

    def _build_pcaf_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build PCAF distribution pie chart data."""
        pcaf = data.get("pcaf_distribution")
        if not pcaf:
            return {}
        return {
            "labels": ["Score 1", "Score 2", "Score 3", "Score 4", "Score 5"],
            "values": [
                pcaf.get("score_1_pct", 0),
                pcaf.get("score_2_pct", 0),
                pcaf.get("score_3_pct", 0),
                pcaf.get("score_4_pct", 0),
                pcaf.get("score_5_pct", 0),
            ],
        }
