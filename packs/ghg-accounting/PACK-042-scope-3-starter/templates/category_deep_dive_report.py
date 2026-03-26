# -*- coding: utf-8 -*-
"""
CategoryDeepDiveReportTemplate - Single-Category Deep Dive for PACK-042.

Generates a detailed analysis report for any single Scope 3 category
(parameterized for any of the 15 GHG Protocol categories). Includes
category overview, sub-category breakdown, emission factor sources,
supplier/product contributions, methodology description, data quality
assessment, uncertainty range, and reduction opportunities.

Sections:
    1. Category Overview
    2. Sub-Category Breakdown
    3. Emission Factor Sources
    4. Supplier/Product Contributions
    5. Methodology Description
    6. Data Quality Assessment
    7. Uncertainty Range
    8. Reduction Opportunities

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, category-specific accent)
    - JSON (structured with drill-down data)

Regulatory References:
    - GHG Protocol Scope 3 Standard, Category-specific guidance
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions

Author: GreenLang Team
Version: 42.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "42.0.0"

# Category-specific accent colors
CATEGORY_COLORS = {
    1: "#0D7377",   # Purchased Goods - deep teal
    2: "#1A5276",   # Capital Goods - steel blue
    3: "#6C3483",   # Fuel & Energy - purple
    4: "#2E86C1",   # Upstream Transport - cerulean
    5: "#117A65",   # Waste - forest green
    6: "#D4AC0D",   # Business Travel - gold
    7: "#CA6F1E",   # Employee Commuting - amber
    8: "#5B2C6F",   # Upstream Leased - plum
    9: "#1F618D",   # Downstream Transport - navy
    10: "#148F77",  # Processing - jade
    11: "#B7950B",  # Use of Sold Products - olive
    12: "#873600",  # End-of-Life - brown
    13: "#1B4F72",  # Downstream Leased - dark blue
    14: "#7D6608",  # Franchises - dark gold
    15: "#4A235A",  # Investments - dark purple
}


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    return f"{_fmt_num(value)} tCO2e"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _pct_of(part: float, total: float) -> str:
    """Percentage of total, formatted."""
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


class CategoryDeepDiveReportTemplate:
    """
    Detailed single-category analysis report template for Scope 3.

    Renders a deep dive report for any one of the 15 Scope 3 categories.
    The report is parameterized via the category field in the input data
    and includes sub-category breakdowns, emission factor sources,
    supplier/product contributions, methodology descriptions, data quality
    assessments, uncertainty ranges, and reduction opportunities. Uses
    category-specific accent colors for HTML output. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = CategoryDeepDiveReportTemplate()
        >>> data = {"category_number": 1, "category_name": "Purchased Goods and Services", ...}
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CategoryDeepDiveReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def _category_total(self, data: Dict[str, Any]) -> float:
        """Get total emissions for the category."""
        return data.get("total_emissions_tco2e", 0.0)

    def _accent_color(self, data: Dict[str, Any]) -> str:
        """Get category-specific accent color."""
        cat_num = data.get("category_number", 1)
        return CATEGORY_COLORS.get(cat_num, "#0D7377")

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render category deep dive report as Markdown.

        Args:
            data: Validated category deep dive data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_category_overview(data),
            self._md_subcategory_breakdown(data),
            self._md_ef_sources(data),
            self._md_supplier_contributions(data),
            self._md_methodology(data),
            self._md_data_quality(data),
            self._md_uncertainty(data),
            self._md_reduction_opportunities(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render category deep dive report as HTML.

        Args:
            data: Validated category deep dive data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_category_overview(data),
            self._html_subcategory_breakdown(data),
            self._html_ef_sources(data),
            self._html_supplier_contributions(data),
            self._html_methodology(data),
            self._html_data_quality(data),
            self._html_uncertainty(data),
            self._html_reduction_opportunities(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render category deep dive report as JSON-serializable dict.

        Args:
            data: Validated category deep dive data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "category_deep_dive_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "category_number": data.get("category_number"),
            "category_name": data.get("category_name"),
            "total_emissions_tco2e": self._category_total(data),
            "subcategories": data.get("subcategories", []),
            "ef_sources": data.get("ef_sources", []),
            "supplier_contributions": data.get("supplier_contributions", []),
            "methodology": data.get("methodology", {}),
            "data_quality": data.get("data_quality", {}),
            "uncertainty": data.get("uncertainty", {}),
            "reduction_opportunities": data.get("reduction_opportunities", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        cat_num = data.get("category_number", "?")
        cat_name = data.get("category_name", "Unknown Category")
        return (
            f"# Category {cat_num} Deep Dive: {cat_name}\n\n"
            f"**Organization:** {company} | **Reporting Year:** {year}\n\n"
            "---"
        )

    def _md_category_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown category overview."""
        total = self._category_total(data)
        s3_total = data.get("scope3_total_tco2e", 0.0)
        pct_of_s3 = _pct_of(total, s3_total) if s3_total > 0 else "N/A"
        rank = data.get("category_rank")
        yoy = data.get("yoy_change_pct")
        tier = data.get("methodology_tier", "-")
        dq = data.get("data_quality_rating", "-")
        lines = [
            "## 1. Category Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Emissions | {_fmt_tco2e(total)} |",
            f"| % of Scope 3 | {pct_of_s3} |",
        ]
        if rank is not None:
            lines.append(f"| Rank (of 15) | #{rank} |")
        lines.append(f"| Methodology Tier | {tier} |")
        lines.append(f"| Data Quality Rating | {dq} |")
        if yoy is not None:
            lines.append(f"| Year-over-Year Change | {_fmt_pct(yoy)} |")
        description = data.get("category_description", "")
        if description:
            lines.append(f"\n{description}")
        return "\n".join(lines)

    def _md_subcategory_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown sub-category breakdown."""
        subcats = data.get("subcategories", [])
        if not subcats:
            return "## 2. Sub-Category Breakdown\n\nNo sub-category data available."
        total = self._category_total(data)
        lines = [
            "## 2. Sub-Category Breakdown",
            "",
            "| Sub-Category | tCO2e | % of Category | Method | Data Source |",
            "|-------------|-------|---------------|--------|-------------|",
        ]
        for sc in sorted(subcats, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = sc.get("name", "")
            em = sc.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, total)
            method = sc.get("method", "-")
            source = sc.get("data_source", "-")
            lines.append(
                f"| {name} | {_fmt_tco2e(em)} | {pct} | {method} | {source} |"
            )
        return "\n".join(lines)

    def _md_ef_sources(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission factor sources."""
        sources = data.get("ef_sources", [])
        if not sources:
            return "## 3. Emission Factor Sources\n\nNo EF sources documented."
        lines = [
            "## 3. Emission Factor Sources",
            "",
            "| Factor Name | Value | Unit | Source | Version | Geography | Provenance |",
            "|------------|-------|------|--------|---------|-----------|-----------|",
        ]
        for src in sources:
            name = src.get("factor_name", "")
            val = src.get("value")
            val_str = f"{val:.6f}" if val is not None else "-"
            unit = src.get("unit", "")
            source = src.get("source", "")
            version = src.get("version", "-")
            geo = src.get("geography", "Global")
            phash = src.get("provenance_hash", "-")
            if len(str(phash)) > 12:
                phash = f"{str(phash)[:12]}..."
            lines.append(
                f"| {name} | {val_str} | {unit} | {source} | {version} | {geo} | `{phash}` |"
            )
        return "\n".join(lines)

    def _md_supplier_contributions(self, data: Dict[str, Any]) -> str:
        """Render Markdown supplier/product contributions."""
        contributions = data.get("supplier_contributions", [])
        if not contributions:
            return "## 4. Supplier/Product Contributions\n\nNo supplier contribution data available."
        total = self._category_total(data)
        lines = [
            "## 4. Supplier/Product Contributions",
            "",
            "| Supplier/Product | tCO2e | % of Category | Data Quality | Engagement Status |",
            "|-----------------|-------|---------------|--------------|-------------------|",
        ]
        for c in sorted(contributions, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = c.get("name", "")
            em = c.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, total)
            dq = c.get("data_quality", "-")
            engagement = c.get("engagement_status", "-")
            lines.append(
                f"| {name} | {_fmt_tco2e(em)} | {pct} | {dq} | {engagement} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology description."""
        methodology = data.get("methodology", {})
        if not methodology:
            return "## 5. Methodology Description\n\nNo methodology description provided."
        method = methodology.get("calculation_method", "")
        tier = methodology.get("tier", "-")
        description = methodology.get("description", "")
        boundary = methodology.get("boundary", "")
        data_sources = methodology.get("data_sources", [])
        lines = [
            "## 5. Methodology Description",
            "",
            f"**Calculation Method:** {method}",
            f"**Methodology Tier:** {tier}",
        ]
        if boundary:
            lines.append(f"**Category Boundary:** {boundary}")
        if description:
            lines.append(f"\n{description}")
        if data_sources:
            lines.append("\n**Data Sources:**")
            for ds in data_sources:
                lines.append(f"- {ds}")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality assessment."""
        quality = data.get("data_quality", {})
        if not quality:
            return "## 6. Data Quality Assessment\n\nNo data quality assessment available."
        overall = quality.get("overall_score")
        lines = [
            "## 6. Data Quality Assessment",
            "",
        ]
        if overall is not None:
            lines.append(f"**Overall DQR Score:** {overall:.1f} / 5.0")
            lines.append("")
        indicators = quality.get("indicators", [])
        if indicators:
            lines.append("| Indicator | Score | Description |")
            lines.append("|-----------|-------|------------|")
            for ind in indicators:
                name = ind.get("name", "")
                score = ind.get("score")
                score_str = f"{score:.1f}" if score is not None else "-"
                desc = ind.get("description", "-")
                lines.append(f"| {name} | {score_str} | {desc} |")
        improvements = quality.get("improvement_actions", [])
        if improvements:
            lines.append("\n**Improvement Actions:**")
            for imp in improvements:
                lines.append(f"- {imp}")
        return "\n".join(lines)

    def _md_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render Markdown uncertainty range."""
        uncertainty = data.get("uncertainty", {})
        if not uncertainty:
            return "## 7. Uncertainty Range\n\nNo uncertainty analysis available."
        central = uncertainty.get("central_tco2e")
        lower = uncertainty.get("lower_bound_tco2e")
        upper = uncertainty.get("upper_bound_tco2e")
        unc_pct = uncertainty.get("uncertainty_pct")
        method = uncertainty.get("method", "-")
        confidence = uncertainty.get("confidence_level", "95%")
        lines = [
            "## 7. Uncertainty Range",
            "",
            f"**Method:** {method} | **Confidence Level:** {confidence}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Central Estimate | {_fmt_tco2e(central)} |",
            f"| Lower Bound | {_fmt_tco2e(lower)} |",
            f"| Upper Bound | {_fmt_tco2e(upper)} |",
        ]
        if unc_pct is not None:
            lines.append(f"| Uncertainty | +/-{unc_pct:.1f}% |")
        sources = uncertainty.get("key_drivers", [])
        if sources:
            lines.append("\n**Key Uncertainty Drivers:**")
            for s in sources:
                lines.append(f"- {s}")
        return "\n".join(lines)

    def _md_reduction_opportunities(self, data: Dict[str, Any]) -> str:
        """Render Markdown reduction opportunities."""
        opportunities = data.get("reduction_opportunities", [])
        if not opportunities:
            return "## 8. Reduction Opportunities\n\nNo reduction opportunities identified."
        lines = [
            "## 8. Reduction Opportunities",
            "",
            "| Opportunity | Potential tCO2e Reduction | Effort | Timeline | ROI |",
            "|-----------|-------------------------|--------|----------|-----|",
        ]
        for opp in opportunities:
            name = opp.get("name", "")
            reduction = _fmt_tco2e(opp.get("potential_reduction_tco2e"))
            effort = opp.get("effort", "-")
            timeline = opp.get("timeline", "-")
            roi = opp.get("roi", "-")
            lines.append(
                f"| {name} | {reduction} | {effort} | {timeline} | {roi} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        cat_num = data.get("category_number", "?")
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | "
            f"Category {cat_num} Deep Dive | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        cat_num = data.get("category_number", "?")
        cat_name = data.get("category_name", "Category")
        accent = self._accent_color(data)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Cat {cat_num} Deep Dive - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            f"h1{{color:{accent};border-bottom:3px solid {accent};padding-bottom:0.5rem;}}\n"
            f"h2{{color:{accent};margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}}\n"
            f"h3{{color:{accent};}}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            f"th{{background:#f0f4f8;font-weight:600;color:{accent};}}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".total-row{font-weight:bold;background:#e8eef4;}\n"
            f".metric-card{{display:inline-block;background:#f0f4f8;border-radius:8px;"
            f"padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:160px;"
            f"border-top:3px solid {accent};}}\n"
            f".metric-value{{font-size:1.5rem;font-weight:700;color:{accent};}}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        cat_num = data.get("category_number", "?")
        cat_name = data.get("category_name", "Unknown Category")
        return (
            '<div class="section">\n'
            f"<h1>Category {cat_num} Deep Dive: {cat_name}</h1>\n"
            f"<p><strong>Organization:</strong> {company} | "
            f"<strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_category_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML category overview with metric cards."""
        total = self._category_total(data)
        s3_total = data.get("scope3_total_tco2e", 0.0)
        pct = _pct_of(total, s3_total) if s3_total > 0 else "N/A"
        rank = data.get("category_rank")
        tier = data.get("methodology_tier", "-")
        cards = [
            ("Total Emissions", _fmt_tco2e(total)),
            ("% of Scope 3", pct),
            ("Methodology Tier", tier),
        ]
        if rank is not None:
            cards.append(("Rank", f"#{rank} of 15"))
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        desc = data.get("category_description", "")
        desc_html = f"<p>{desc}</p>" if desc else ""
        return (
            '<div class="section">\n'
            "<h2>1. Category Overview</h2>\n"
            f"<div>{card_html}</div>\n{desc_html}\n</div>"
        )

    def _html_subcategory_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML sub-category breakdown."""
        subcats = data.get("subcategories", [])
        if not subcats:
            return '<div class="section">\n<h2>2. Sub-Category Breakdown</h2>\n<p>No data available.</p>\n</div>'
        total = self._category_total(data)
        rows = ""
        for sc in sorted(subcats, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = sc.get("name", "")
            em = sc.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, total)
            method = sc.get("method", "-")
            source = sc.get("data_source", "-")
            rows += f"<tr><td>{name}</td><td>{_fmt_tco2e(em)}</td><td>{pct}</td><td>{method}</td><td>{source}</td></tr>\n"
        return (
            '<div class="section">\n<h2>2. Sub-Category Breakdown</h2>\n'
            "<table><thead><tr><th>Sub-Category</th><th>tCO2e</th><th>%</th>"
            f"<th>Method</th><th>Source</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_ef_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML emission factor sources."""
        sources = data.get("ef_sources", [])
        if not sources:
            return ""
        rows = ""
        for src in sources:
            name = src.get("factor_name", "")
            val = src.get("value")
            val_str = f"{val:.6f}" if val is not None else "-"
            unit = src.get("unit", "")
            source = src.get("source", "")
            version = src.get("version", "-")
            geo = src.get("geography", "Global")
            phash = src.get("provenance_hash", "-")
            if len(str(phash)) > 12:
                phash = f"{str(phash)[:12]}..."
            rows += (
                f"<tr><td>{name}</td><td>{val_str}</td><td>{unit}</td>"
                f"<td>{source}</td><td>{version}</td><td>{geo}</td>"
                f"<td><code>{phash}</code></td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Emission Factor Sources</h2>\n'
            "<table><thead><tr><th>Factor</th><th>Value</th><th>Unit</th>"
            "<th>Source</th><th>Version</th><th>Geography</th>"
            f"<th>Hash</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_supplier_contributions(self, data: Dict[str, Any]) -> str:
        """Render HTML supplier/product contributions."""
        contributions = data.get("supplier_contributions", [])
        if not contributions:
            return ""
        total = self._category_total(data)
        rows = ""
        for c in sorted(contributions, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = c.get("name", "")
            em = c.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, total)
            dq = c.get("data_quality", "-")
            engagement = c.get("engagement_status", "-")
            rows += (
                f"<tr><td>{name}</td><td>{_fmt_tco2e(em)}</td><td>{pct}</td>"
                f"<td>{dq}</td><td>{engagement}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Supplier/Product Contributions</h2>\n'
            "<table><thead><tr><th>Supplier/Product</th><th>tCO2e</th><th>%</th>"
            f"<th>DQ</th><th>Engagement</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology description."""
        methodology = data.get("methodology", {})
        if not methodology:
            return ""
        method = methodology.get("calculation_method", "")
        tier = methodology.get("tier", "-")
        description = methodology.get("description", "")
        boundary = methodology.get("boundary", "")
        data_sources = methodology.get("data_sources", [])
        sources_html = ""
        if data_sources:
            sources_html = "<p><strong>Data Sources:</strong></p><ul>\n"
            for ds in data_sources:
                sources_html += f"<li>{ds}</li>\n"
            sources_html += "</ul>\n"
        return (
            '<div class="section">\n<h2>5. Methodology Description</h2>\n'
            f"<p><strong>Calculation Method:</strong> {method} | "
            f"<strong>Tier:</strong> {tier}</p>\n"
            + (f"<p><strong>Boundary:</strong> {boundary}</p>\n" if boundary else "")
            + (f"<p>{description}</p>\n" if description else "")
            + f"{sources_html}</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality assessment."""
        quality = data.get("data_quality", {})
        if not quality:
            return ""
        overall = quality.get("overall_score")
        overall_str = f"<p><strong>Overall DQR Score:</strong> {overall:.1f} / 5.0</p>\n" if overall else ""
        indicators = quality.get("indicators", [])
        rows = ""
        for ind in indicators:
            name = ind.get("name", "")
            score = ind.get("score")
            score_str = f"{score:.1f}" if score is not None else "-"
            desc = ind.get("description", "-")
            rows += f"<tr><td>{name}</td><td>{score_str}</td><td>{desc}</td></tr>\n"
        table = ""
        if rows:
            table = (
                "<table><thead><tr><th>Indicator</th><th>Score</th>"
                f"<th>Description</th></tr></thead>\n<tbody>{rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n<h2>6. Data Quality Assessment</h2>\n'
            f"{overall_str}{table}</div>"
        )

    def _html_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty range."""
        uncertainty = data.get("uncertainty", {})
        if not uncertainty:
            return ""
        central = _fmt_tco2e(uncertainty.get("central_tco2e"))
        lower = _fmt_tco2e(uncertainty.get("lower_bound_tco2e"))
        upper = _fmt_tco2e(uncertainty.get("upper_bound_tco2e"))
        unc_pct = uncertainty.get("uncertainty_pct")
        method = uncertainty.get("method", "-")
        confidence = uncertainty.get("confidence_level", "95%")
        return (
            '<div class="section">\n<h2>7. Uncertainty Range</h2>\n'
            f"<p><strong>Method:</strong> {method} | <strong>Confidence:</strong> {confidence}</p>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>Central Estimate</td><td>{central}</td></tr>\n"
            f"<tr><td>Lower Bound</td><td>{lower}</td></tr>\n"
            f"<tr><td>Upper Bound</td><td>{upper}</td></tr>\n"
            + (f"<tr><td>Uncertainty</td><td>+/-{unc_pct:.1f}%</td></tr>\n" if unc_pct else "")
            + "</tbody></table>\n</div>"
        )

    def _html_reduction_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML reduction opportunities."""
        opportunities = data.get("reduction_opportunities", [])
        if not opportunities:
            return ""
        rows = ""
        for opp in opportunities:
            name = opp.get("name", "")
            reduction = _fmt_tco2e(opp.get("potential_reduction_tco2e"))
            effort = opp.get("effort", "-")
            timeline = opp.get("timeline", "-")
            roi = opp.get("roi", "-")
            rows += (
                f"<tr><td>{name}</td><td>{reduction}</td><td>{effort}</td>"
                f"<td>{timeline}</td><td>{roi}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>8. Reduction Opportunities</h2>\n'
            "<table><thead><tr><th>Opportunity</th><th>Potential Reduction</th>"
            f"<th>Effort</th><th>Timeline</th><th>ROI</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        cat_num = data.get("category_number", "?")
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | "
            f"Category {cat_num} Deep Dive | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
