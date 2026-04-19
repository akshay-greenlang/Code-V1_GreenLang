# -*- coding: utf-8 -*-
"""
InventorySummaryReport - Base Year Inventory Summary for PACK-045.

Generates a base year inventory summary covering emissions by scope,
category and gas type, totals with percentage breakdowns, methodology
notes, and data quality summary.

Sections:
    1. Inventory Overview
    2. Scope 1 Emissions Breakdown
    3. Scope 2 Emissions Breakdown
    4. Scope 3 Emissions Breakdown
    5. Emissions by Gas Type
    6. Methodology Summary

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "45.0.0"


def _pct_str(value: float, total: float) -> str:
    """Return percentage string of value over total."""
    if total <= 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


class InventorySummaryReport:
    """
    Base year inventory summary report template.

    Renders emissions breakdowns by scope, category, and gas type for the
    selected base year. Includes totals, percentage contributions, methodology
    references, and data quality indicators. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = InventorySummaryReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize InventorySummaryReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

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
        """Render inventory summary report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_scope1(data),
            self._md_scope2(data),
            self._md_scope3(data),
            self._md_gas_breakdown(data),
            self._md_methodology(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render inventory summary report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
            self._html_scope1(data),
            self._html_scope2(data),
            self._html_scope3(data),
            self._html_gas_breakdown(data),
            self._html_methodology(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render inventory summary report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        total = data.get("total_emissions_tco2e", 0)
        return {
            "template": "inventory_summary_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "total_emissions_tco2e": total,
            "scope1": data.get("scope1", {}),
            "scope2": data.get("scope2", {}),
            "scope3": data.get("scope3", {}),
            "gas_breakdown": data.get("gas_breakdown", []),
            "methodology": data.get("methodology", {}),
            "data_quality": data.get("data_quality", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        total = data.get("total_emissions_tco2e", 0)
        return (
            f"# Base Year Inventory Summary - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Total Emissions:** {total:,.1f} tCO2e | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown inventory overview."""
        total = data.get("total_emissions_tco2e", 0)
        s1 = data.get("scope1", {}).get("total_tco2e", 0)
        s2 = data.get("scope2", {}).get("total_tco2e", 0)
        s3 = data.get("scope3", {}).get("total_tco2e", 0)
        lines = [
            "## 1. Inventory Overview",
            "",
            "| Scope | Emissions (tCO2e) | % of Total |",
            "|-------|------------------|------------|",
            f"| Scope 1 (Direct) | {s1:,.1f} | {_pct_str(s1, total)} |",
            f"| Scope 2 (Indirect Energy) | {s2:,.1f} | {_pct_str(s2, total)} |",
            f"| Scope 3 (Value Chain) | {s3:,.1f} | {_pct_str(s3, total)} |",
            f"| **Total** | **{total:,.1f}** | **100.0%** |",
        ]
        return "\n".join(lines)

    def _md_scope1(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 1 breakdown."""
        scope1 = data.get("scope1", {})
        categories = scope1.get("categories", [])
        if not categories:
            return "## 2. Scope 1 Emissions\n\nNo Scope 1 data available."
        total = scope1.get("total_tco2e", 0)
        lines = [
            "## 2. Scope 1 Emissions (Direct)",
            "",
            "| Category | Emissions (tCO2e) | % of Scope 1 | Methodology |",
            "|----------|------------------|-------------|-------------|",
        ]
        for cat in categories:
            name = cat.get("name", "")
            emissions = cat.get("emissions_tco2e", 0)
            method = cat.get("methodology", "")
            lines.append(f"| {name} | {emissions:,.1f} | {_pct_str(emissions, total)} | {method} |")
        lines.append(f"| **Total Scope 1** | **{total:,.1f}** | **100.0%** | |")
        return "\n".join(lines)

    def _md_scope2(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 2 breakdown."""
        scope2 = data.get("scope2", {})
        location = scope2.get("location_based_tco2e", 0)
        market = scope2.get("market_based_tco2e", 0)
        categories = scope2.get("categories", [])
        total = scope2.get("total_tco2e", 0)
        lines = [
            "## 3. Scope 2 Emissions (Indirect Energy)",
            "",
            f"**Location-Based:** {location:,.1f} tCO2e | "
            f"**Market-Based:** {market:,.1f} tCO2e",
            "",
        ]
        if categories:
            lines.extend([
                "| Category | Emissions (tCO2e) | % of Scope 2 | Method |",
                "|----------|------------------|-------------|--------|",
            ])
            for cat in categories:
                name = cat.get("name", "")
                emissions = cat.get("emissions_tco2e", 0)
                method = cat.get("method", "")
                lines.append(f"| {name} | {emissions:,.1f} | {_pct_str(emissions, total)} | {method} |")
        return "\n".join(lines)

    def _md_scope3(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 3 breakdown."""
        scope3 = data.get("scope3", {})
        categories = scope3.get("categories", [])
        if not categories:
            return "## 4. Scope 3 Emissions\n\nNo Scope 3 data available."
        total = scope3.get("total_tco2e", 0)
        lines = [
            "## 4. Scope 3 Emissions (Value Chain)",
            "",
            "| Cat # | Category | Emissions (tCO2e) | % of Scope 3 | Method | DQ Score |",
            "|-------|----------|------------------|-------------|--------|----------|",
        ]
        for cat in categories:
            num = cat.get("category_number", "")
            name = cat.get("name", "")
            emissions = cat.get("emissions_tco2e", 0)
            method = cat.get("methodology", "")
            dq = cat.get("data_quality_score", 0)
            lines.append(
                f"| {num} | {name} | {emissions:,.1f} | "
                f"{_pct_str(emissions, total)} | {method} | {dq:.0f}/5 |"
            )
        lines.append(f"| | **Total Scope 3** | **{total:,.1f}** | **100.0%** | | |")
        return "\n".join(lines)

    def _md_gas_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown emissions by gas type."""
        gases = data.get("gas_breakdown", [])
        if not gases:
            return ""
        total = data.get("total_emissions_tco2e", 0)
        lines = [
            "## 5. Emissions by Gas Type",
            "",
            "| Gas | Emissions (tCO2e) | % of Total | GWP Applied |",
            "|-----|------------------|------------|------------|",
        ]
        for g in gases:
            name = g.get("gas", "")
            emissions = g.get("emissions_tco2e", 0)
            gwp = g.get("gwp", "")
            lines.append(f"| {name} | {emissions:,.1f} | {_pct_str(emissions, total)} | {gwp} |")
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology summary."""
        meth = data.get("methodology", {})
        if not meth:
            return "## 6. Methodology Summary\n\nNo methodology information available."
        framework = meth.get("framework", "GHG Protocol")
        consolidation = meth.get("consolidation_approach", "")
        gwp_source = meth.get("gwp_source", "IPCC AR5")
        notes = meth.get("notes", [])
        lines = [
            "## 6. Methodology Summary",
            "",
            f"- **Framework:** {framework}",
            f"- **Consolidation Approach:** {consolidation}",
            f"- **GWP Source:** {gwp_source}",
        ]
        if notes:
            lines.append("")
            lines.append("**Notes:**")
            for n in notes:
                lines.append(f"- {n}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-045 Base Year Management v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Base Year Inventory - {company}</title>\n"
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
            ".scope-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;}\n"
            ".scope-value{font-size:1.4rem;font-weight:700;color:#1b263b;}\n"
            ".scope-label{font-size:0.85rem;color:#555;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        total = data.get("total_emissions_tco2e", 0)
        return (
            '<div class="section">\n'
            f"<h1>Base Year Inventory Summary &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year} | "
            f"<strong>Total:</strong> {total:,.1f} tCO2e</p>\n<hr>\n</div>"
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML inventory overview with scope cards."""
        total = data.get("total_emissions_tco2e", 0)
        s1 = data.get("scope1", {}).get("total_tco2e", 0)
        s2 = data.get("scope2", {}).get("total_tco2e", 0)
        s3 = data.get("scope3", {}).get("total_tco2e", 0)
        return (
            '<div class="section">\n<h2>1. Inventory Overview</h2>\n<div>\n'
            f'<div class="scope-card"><div class="scope-value">{s1:,.0f}</div>'
            f'<div class="scope-label">Scope 1 ({_pct_str(s1, total)})</div></div>\n'
            f'<div class="scope-card"><div class="scope-value">{s2:,.0f}</div>'
            f'<div class="scope-label">Scope 2 ({_pct_str(s2, total)})</div></div>\n'
            f'<div class="scope-card"><div class="scope-value">{s3:,.0f}</div>'
            f'<div class="scope-label">Scope 3 ({_pct_str(s3, total)})</div></div>\n'
            f'<div class="scope-card"><div class="scope-value">{total:,.0f}</div>'
            f'<div class="scope-label">Total tCO2e</div></div>\n'
            "</div>\n</div>"
        )

    def _html_scope1(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 1 breakdown."""
        categories = data.get("scope1", {}).get("categories", [])
        if not categories:
            return ""
        total = data.get("scope1", {}).get("total_tco2e", 0)
        rows = ""
        for cat in categories:
            name = cat.get("name", "")
            emissions = cat.get("emissions_tco2e", 0)
            method = cat.get("methodology", "")
            rows += f"<tr><td>{name}</td><td>{emissions:,.1f}</td><td>{_pct_str(emissions, total)}</td><td>{method}</td></tr>\n"
        return (
            '<div class="section">\n<h2>2. Scope 1 Emissions</h2>\n'
            "<table><thead><tr><th>Category</th><th>tCO2e</th><th>%</th>"
            "<th>Methodology</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scope2(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 2 breakdown."""
        scope2 = data.get("scope2", {})
        location = scope2.get("location_based_tco2e", 0)
        market = scope2.get("market_based_tco2e", 0)
        return (
            '<div class="section">\n<h2>3. Scope 2 Emissions</h2>\n'
            f"<p><strong>Location-Based:</strong> {location:,.1f} tCO2e | "
            f"<strong>Market-Based:</strong> {market:,.1f} tCO2e</p>\n</div>"
        )

    def _html_scope3(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 3 breakdown."""
        categories = data.get("scope3", {}).get("categories", [])
        if not categories:
            return ""
        total = data.get("scope3", {}).get("total_tco2e", 0)
        rows = ""
        for cat in categories:
            num = cat.get("category_number", "")
            name = cat.get("name", "")
            emissions = cat.get("emissions_tco2e", 0)
            rows += f"<tr><td>{num}</td><td>{name}</td><td>{emissions:,.1f}</td><td>{_pct_str(emissions, total)}</td></tr>\n"
        return (
            '<div class="section">\n<h2>4. Scope 3 Emissions</h2>\n'
            "<table><thead><tr><th>Cat#</th><th>Category</th><th>tCO2e</th>"
            "<th>%</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_gas_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML gas type breakdown."""
        gases = data.get("gas_breakdown", [])
        if not gases:
            return ""
        total = data.get("total_emissions_tco2e", 0)
        rows = ""
        for g in gases:
            name = g.get("gas", "")
            emissions = g.get("emissions_tco2e", 0)
            gwp = g.get("gwp", "")
            rows += f"<tr><td>{name}</td><td>{emissions:,.1f}</td><td>{_pct_str(emissions, total)}</td><td>{gwp}</td></tr>\n"
        return (
            '<div class="section">\n<h2>5. Emissions by Gas</h2>\n'
            "<table><thead><tr><th>Gas</th><th>tCO2e</th><th>%</th>"
            "<th>GWP</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology summary."""
        meth = data.get("methodology", {})
        if not meth:
            return ""
        framework = meth.get("framework", "GHG Protocol")
        consolidation = meth.get("consolidation_approach", "")
        gwp_source = meth.get("gwp_source", "IPCC AR5")
        return (
            '<div class="section">\n<h2>6. Methodology</h2>\n'
            f"<p><strong>Framework:</strong> {framework}</p>\n"
            f"<p><strong>Consolidation:</strong> {consolidation}</p>\n"
            f"<p><strong>GWP Source:</strong> {gwp_source}</p>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-045 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
