# -*- coding: utf-8 -*-
"""
HotspotReportTemplate - Pareto Analysis and Prioritization for PACK-042.

Generates a hotspot analysis report with Pareto chart data (categories
ranked by contribution), materiality matrix data, supplier concentration
(top N suppliers), geographic distribution, product carbon intensity
ranking, reduction opportunities with ROI, and tier upgrade impact
quantification.

Sections:
    1. Pareto Analysis
    2. Materiality Matrix
    3. Supplier Concentration
    4. Geographic Distribution
    5. Product Carbon Intensity
    6. Reduction Opportunities with ROI
    7. Tier Upgrade Impact

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, warm orange theme)
    - JSON (structured with chart-ready data)

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


class HotspotReportTemplate:
    """
    Pareto analysis and prioritization report template.

    Renders hotspot analysis reports with Pareto-ranked categories,
    materiality matrices, supplier concentration analysis, geographic
    distribution, product carbon intensity rankings, reduction
    opportunities with ROI, and tier upgrade impact quantification.
    All outputs include SHA-256 provenance hashing for audit trails.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = HotspotReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize HotspotReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def _scope3_total(self, data: Dict[str, Any]) -> float:
        """Get total Scope 3 emissions."""
        return data.get("scope3_total_tco2e", 0.0)

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render hotspot report as Markdown.

        Args:
            data: Validated hotspot analysis data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_pareto_analysis(data),
            self._md_materiality_matrix(data),
            self._md_supplier_concentration(data),
            self._md_geographic_distribution(data),
            self._md_product_intensity(data),
            self._md_reduction_opportunities(data),
            self._md_tier_upgrade_impact(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render hotspot report as HTML.

        Args:
            data: Validated hotspot analysis data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_pareto_analysis(data),
            self._html_materiality_matrix(data),
            self._html_supplier_concentration(data),
            self._html_geographic_distribution(data),
            self._html_product_intensity(data),
            self._html_reduction_opportunities(data),
            self._html_tier_upgrade_impact(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render hotspot report as JSON-serializable dict.

        Args:
            data: Validated hotspot analysis data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        s3_total = self._scope3_total(data)
        pareto = self._json_pareto(data, s3_total)
        return {
            "template": "hotspot_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "scope3_total_tco2e": s3_total,
            "pareto_analysis": pareto,
            "materiality_matrix": data.get("materiality_matrix", []),
            "supplier_concentration": data.get("supplier_concentration", {}),
            "geographic_distribution": data.get("geographic_distribution", []),
            "product_intensity": data.get("product_intensity", []),
            "reduction_opportunities": data.get("reduction_opportunities", []),
            "tier_upgrade_impact": data.get("tier_upgrade_impact", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 3 Hotspot Analysis - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_pareto_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown Pareto analysis."""
        categories = data.get("pareto_categories", [])
        s3_total = self._scope3_total(data)
        if not categories:
            return "## 1. Pareto Analysis\n\nNo category data available."
        sorted_cats = sorted(
            categories, key=lambda c: c.get("emissions_tco2e", 0), reverse=True
        )
        lines = [
            "## 1. Pareto Analysis",
            "",
            "| Rank | Category | tCO2e | % of Total | Cumulative % |",
            "|------|----------|-------|-----------|-------------|",
        ]
        cumulative = 0.0
        for i, cat in enumerate(sorted_cats, 1):
            name = cat.get("category_name", "Unknown")
            em = cat.get("emissions_tco2e", 0.0)
            cumulative += em
            pct = _pct_of(em, s3_total)
            cum_pct = _pct_of(cumulative, s3_total)
            lines.append(
                f"| {i} | {name} | {_fmt_tco2e(em)} | {pct} | {cum_pct} |"
            )
        # Find 80% threshold
        threshold_80 = 0.0
        count_80 = 0
        for cat in sorted_cats:
            threshold_80 += cat.get("emissions_tco2e", 0.0)
            count_80 += 1
            if s3_total > 0 and (threshold_80 / s3_total) >= 0.80:
                break
        lines.append(
            f"\n*{count_80} categories account for 80%+ of Scope 3 emissions (Pareto principle).*"
        )
        return "\n".join(lines)

    def _md_materiality_matrix(self, data: Dict[str, Any]) -> str:
        """Render Markdown materiality matrix."""
        matrix = data.get("materiality_matrix", [])
        if not matrix:
            return "## 2. Materiality Matrix\n\nNo materiality assessment available."
        lines = [
            "## 2. Materiality Matrix",
            "",
            "| Category | Emission Impact | Data Availability | Reduction Potential | Materiality |",
            "|----------|----------------|-------------------|--------------------|-----------:|",
        ]
        for item in matrix:
            cat = item.get("category_name", "")
            impact = item.get("emission_impact", "-")
            avail = item.get("data_availability", "-")
            potential = item.get("reduction_potential", "-")
            materiality = item.get("materiality_score", "-")
            lines.append(
                f"| {cat} | {impact} | {avail} | {potential} | {materiality} |"
            )
        return "\n".join(lines)

    def _md_supplier_concentration(self, data: Dict[str, Any]) -> str:
        """Render Markdown supplier concentration."""
        conc = data.get("supplier_concentration", {})
        if not conc:
            return "## 3. Supplier Concentration\n\nNo supplier data available."
        suppliers = conc.get("top_suppliers", [])
        s3_total = self._scope3_total(data)
        lines = [
            "## 3. Supplier Concentration",
            "",
        ]
        summary = conc.get("summary", {})
        if summary:
            total_suppliers = summary.get("total_suppliers", 0)
            top_n = summary.get("top_n", 10)
            top_pct = summary.get("top_n_pct")
            lines.append(
                f"**Total Suppliers:** {total_suppliers} | "
                f"**Top {top_n} Cover:** {top_pct:.1f}%" if top_pct else ""
            )
            lines.append("")
        if suppliers:
            lines.append(
                "| Supplier | tCO2e | % of Scope 3 | Category | Data Quality |"
            )
            lines.append(
                "|----------|-------|-------------|----------|-------------|"
            )
            for s in suppliers:
                name = s.get("supplier_name", "")
                em = s.get("emissions_tco2e", 0.0)
                pct = _pct_of(em, s3_total)
                cat = s.get("primary_category", "-")
                dq = s.get("data_quality", "-")
                lines.append(
                    f"| {name} | {_fmt_tco2e(em)} | {pct} | {cat} | {dq} |"
                )
        return "\n".join(lines)

    def _md_geographic_distribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown geographic distribution."""
        geo = data.get("geographic_distribution", [])
        if not geo:
            return "## 4. Geographic Distribution\n\nNo geographic data available."
        s3_total = self._scope3_total(data)
        lines = [
            "## 4. Geographic Distribution",
            "",
            "| Region/Country | tCO2e | % of Scope 3 | Primary Categories | Data Quality |",
            "|---------------|-------|-------------|-------------------|-------------|",
        ]
        for g in sorted(geo, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            region = g.get("region", "")
            em = g.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            cats = g.get("primary_categories", "-")
            if isinstance(cats, list):
                cats = ", ".join(cats)
            dq = g.get("data_quality", "-")
            lines.append(f"| {region} | {_fmt_tco2e(em)} | {pct} | {cats} | {dq} |")
        return "\n".join(lines)

    def _md_product_intensity(self, data: Dict[str, Any]) -> str:
        """Render Markdown product carbon intensity ranking."""
        products = data.get("product_intensity", [])
        if not products:
            return "## 5. Product Carbon Intensity\n\nNo product intensity data available."
        lines = [
            "## 5. Product Carbon Intensity",
            "",
            "| Rank | Product/Service | Intensity (kgCO2e/unit) | Total tCO2e | Revenue Share |",
            "|------|----------------|------------------------|------------|---------------|",
        ]
        for i, p in enumerate(
            sorted(products, key=lambda x: x.get("intensity_kgco2e", 0), reverse=True), 1
        ):
            name = p.get("product_name", "")
            intensity = p.get("intensity_kgco2e")
            int_str = f"{intensity:,.2f}" if intensity is not None else "-"
            total = _fmt_tco2e(p.get("total_tco2e"))
            revenue = p.get("revenue_share_pct")
            rev_str = f"{revenue:.1f}%" if revenue is not None else "-"
            lines.append(f"| {i} | {name} | {int_str} | {total} | {rev_str} |")
        return "\n".join(lines)

    def _md_reduction_opportunities(self, data: Dict[str, Any]) -> str:
        """Render Markdown reduction opportunities with ROI."""
        opportunities = data.get("reduction_opportunities", [])
        if not opportunities:
            return "## 6. Reduction Opportunities\n\nNo reduction opportunities identified."
        lines = [
            "## 6. Reduction Opportunities with ROI",
            "",
            "| Priority | Opportunity | Reduction tCO2e | Investment | Annual Savings | Payback | ROI |",
            "|----------|-----------|----------------|-----------|---------------|---------|-----|",
        ]
        for i, opp in enumerate(opportunities, 1):
            name = opp.get("name", "")
            reduction = _fmt_tco2e(opp.get("reduction_tco2e"))
            investment = opp.get("investment", "-")
            savings = opp.get("annual_savings", "-")
            payback = opp.get("payback_period", "-")
            roi = opp.get("roi", "-")
            lines.append(
                f"| {i} | {name} | {reduction} | {investment} | {savings} | {payback} | {roi} |"
            )
        return "\n".join(lines)

    def _md_tier_upgrade_impact(self, data: Dict[str, Any]) -> str:
        """Render Markdown tier upgrade impact quantification."""
        upgrades = data.get("tier_upgrade_impact", [])
        if not upgrades:
            return "## 7. Tier Upgrade Impact\n\nNo tier upgrade analysis available."
        lines = [
            "## 7. Tier Upgrade Impact",
            "",
            "| Category | Current Tier | Target Tier | Uncertainty Reduction | Effort | Priority |",
            "|----------|-------------|-------------|---------------------|--------|----------|",
        ]
        for u in upgrades:
            cat = u.get("category_name", "")
            current = u.get("current_tier", "-")
            target = u.get("target_tier", "-")
            unc_red = u.get("uncertainty_reduction_pct")
            unc_str = f"{unc_red:.0f}%" if unc_red is not None else "-"
            effort = u.get("effort", "-")
            priority = u.get("priority", "-")
            lines.append(
                f"| {cat} | {current} | {target} | {unc_str} | {effort} | {priority} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Scope 3 Hotspot Analysis - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#E67E22;border-bottom:3px solid #E67E22;padding-bottom:0.5rem;}\n"
            "h2{color:#CA6F1E;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}\n"
            "h3{color:#E67E22;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#fdf2e9;font-weight:600;color:#CA6F1E;}\n"
            "tr:nth-child(even){background:#fef9f4;}\n"
            ".total-row{font-weight:bold;background:#f5e0c3;}\n"
            ".hotspot{border-left:4px solid #E67E22;}\n"
            ".metric-card{display:inline-block;background:#fdf2e9;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:170px;"
            "border-top:3px solid #E67E22;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#CA6F1E;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".priority-high{color:#e74c3c;font-weight:700;}\n"
            ".priority-medium{color:#E67E22;font-weight:700;}\n"
            ".priority-low{color:#27ae60;font-weight:700;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Scope 3 Hotspot Analysis &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_pareto_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML Pareto analysis."""
        categories = data.get("pareto_categories", [])
        s3_total = self._scope3_total(data)
        if not categories:
            return ""
        sorted_cats = sorted(
            categories, key=lambda c: c.get("emissions_tco2e", 0), reverse=True
        )
        rows = ""
        cumulative = 0.0
        for i, cat in enumerate(sorted_cats, 1):
            name = cat.get("category_name", "Unknown")
            em = cat.get("emissions_tco2e", 0.0)
            cumulative += em
            pct = _pct_of(em, s3_total)
            cum_pct = _pct_of(cumulative, s3_total)
            rows += (
                f'<tr class="hotspot"><td>{i}</td><td>{name}</td>'
                f"<td>{_fmt_tco2e(em)}</td><td>{pct}</td><td>{cum_pct}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>1. Pareto Analysis</h2>\n"
            "<table><thead><tr><th>Rank</th><th>Category</th><th>tCO2e</th>"
            f"<th>%</th><th>Cumulative %</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_materiality_matrix(self, data: Dict[str, Any]) -> str:
        """Render HTML materiality matrix."""
        matrix = data.get("materiality_matrix", [])
        if not matrix:
            return ""
        rows = ""
        for item in matrix:
            cat = item.get("category_name", "")
            impact = item.get("emission_impact", "-")
            avail = item.get("data_availability", "-")
            potential = item.get("reduction_potential", "-")
            materiality = item.get("materiality_score", "-")
            rows += (
                f"<tr><td>{cat}</td><td>{impact}</td><td>{avail}</td>"
                f"<td>{potential}</td><td>{materiality}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Materiality Matrix</h2>\n"
            "<table><thead><tr><th>Category</th><th>Impact</th><th>Data Avail.</th>"
            f"<th>Reduction Pot.</th><th>Materiality</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_supplier_concentration(self, data: Dict[str, Any]) -> str:
        """Render HTML supplier concentration."""
        conc = data.get("supplier_concentration", {})
        suppliers = conc.get("top_suppliers", [])
        if not suppliers:
            return ""
        s3_total = self._scope3_total(data)
        rows = ""
        for s in suppliers:
            name = s.get("supplier_name", "")
            em = s.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            cat = s.get("primary_category", "-")
            dq = s.get("data_quality", "-")
            rows += (
                f"<tr><td>{name}</td><td>{_fmt_tco2e(em)}</td><td>{pct}</td>"
                f"<td>{cat}</td><td>{dq}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Supplier Concentration</h2>\n"
            "<table><thead><tr><th>Supplier</th><th>tCO2e</th><th>%</th>"
            f"<th>Category</th><th>DQ</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_geographic_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML geographic distribution."""
        geo = data.get("geographic_distribution", [])
        if not geo:
            return ""
        s3_total = self._scope3_total(data)
        rows = ""
        for g in sorted(geo, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            region = g.get("region", "")
            em = g.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            cats = g.get("primary_categories", "-")
            if isinstance(cats, list):
                cats = ", ".join(cats)
            dq = g.get("data_quality", "-")
            rows += (
                f"<tr><td>{region}</td><td>{_fmt_tco2e(em)}</td><td>{pct}</td>"
                f"<td>{cats}</td><td>{dq}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Geographic Distribution</h2>\n"
            "<table><thead><tr><th>Region</th><th>tCO2e</th><th>%</th>"
            f"<th>Categories</th><th>DQ</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_product_intensity(self, data: Dict[str, Any]) -> str:
        """Render HTML product carbon intensity."""
        products = data.get("product_intensity", [])
        if not products:
            return ""
        rows = ""
        for i, p in enumerate(
            sorted(products, key=lambda x: x.get("intensity_kgco2e", 0), reverse=True), 1
        ):
            name = p.get("product_name", "")
            intensity = p.get("intensity_kgco2e")
            int_str = f"{intensity:,.2f}" if intensity is not None else "-"
            total = _fmt_tco2e(p.get("total_tco2e"))
            revenue = p.get("revenue_share_pct")
            rev_str = f"{revenue:.1f}%" if revenue is not None else "-"
            rows += (
                f"<tr><td>{i}</td><td>{name}</td><td>{int_str}</td>"
                f"<td>{total}</td><td>{rev_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>5. Product Carbon Intensity</h2>\n"
            "<table><thead><tr><th>Rank</th><th>Product</th><th>kgCO2e/unit</th>"
            f"<th>Total</th><th>Revenue %</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_reduction_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML reduction opportunities with ROI."""
        opportunities = data.get("reduction_opportunities", [])
        if not opportunities:
            return ""
        rows = ""
        for i, opp in enumerate(opportunities, 1):
            name = opp.get("name", "")
            reduction = _fmt_tco2e(opp.get("reduction_tco2e"))
            investment = opp.get("investment", "-")
            savings = opp.get("annual_savings", "-")
            payback = opp.get("payback_period", "-")
            roi = opp.get("roi", "-")
            rows += (
                f"<tr><td>{i}</td><td>{name}</td><td>{reduction}</td>"
                f"<td>{investment}</td><td>{savings}</td><td>{payback}</td>"
                f"<td>{roi}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Reduction Opportunities with ROI</h2>\n"
            "<table><thead><tr><th>#</th><th>Opportunity</th><th>Reduction</th>"
            "<th>Investment</th><th>Savings/yr</th><th>Payback</th>"
            f"<th>ROI</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_tier_upgrade_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML tier upgrade impact."""
        upgrades = data.get("tier_upgrade_impact", [])
        if not upgrades:
            return ""
        rows = ""
        for u in upgrades:
            cat = u.get("category_name", "")
            current = u.get("current_tier", "-")
            target = u.get("target_tier", "-")
            unc_red = u.get("uncertainty_reduction_pct")
            unc_str = f"{unc_red:.0f}%" if unc_red is not None else "-"
            effort = u.get("effort", "-")
            priority = u.get("priority", "-")
            p_css = f"priority-{priority.lower()}" if priority in ("High", "Medium", "Low") else ""
            rows += (
                f"<tr><td>{cat}</td><td>{current}</td><td>{target}</td>"
                f'<td>{unc_str}</td><td>{effort}</td><td class="{p_css}">{priority}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>7. Tier Upgrade Impact</h2>\n"
            "<table><thead><tr><th>Category</th><th>Current</th><th>Target</th>"
            "<th>Unc. Reduction</th><th>Effort</th>"
            f"<th>Priority</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON HELPERS
    # ==================================================================

    def _json_pareto(
        self, data: Dict[str, Any], s3_total: float
    ) -> List[Dict[str, Any]]:
        """Build Pareto chart data."""
        categories = data.get("pareto_categories", [])
        sorted_cats = sorted(
            categories, key=lambda c: c.get("emissions_tco2e", 0), reverse=True
        )
        result = []
        cumulative = 0.0
        for i, cat in enumerate(sorted_cats, 1):
            em = cat.get("emissions_tco2e", 0.0)
            cumulative += em
            result.append({
                "rank": i,
                "category_name": cat.get("category_name"),
                "category_number": cat.get("category_number"),
                "emissions_tco2e": em,
                "pct_of_total": (em / s3_total * 100) if s3_total > 0 else 0.0,
                "cumulative_pct": (cumulative / s3_total * 100) if s3_total > 0 else 0.0,
            })
        return result
