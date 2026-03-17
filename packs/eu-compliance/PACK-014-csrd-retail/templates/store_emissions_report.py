# -*- coding: utf-8 -*-
"""
StoreEmissionsReportTemplate - Store-level emissions report for PACK-014.

Renders store-by-store GHG emission breakdowns, Scope 1/2 detail,
multi-store consolidation, energy intensity KPIs, and F-Gas phase-down status.

Sections:
    1. Executive Summary
    2. Store-by-Store Breakdown
    3. Scope 1 Detail (heating, refrigerant, fleet)
    4. Scope 2 Detail (location vs market)
    5. Multi-Store Consolidation
    6. Energy Intensity KPIs
    7. F-Gas Phase-Down Status

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StoreEmissionsReportTemplate:
    """
    Store-level emissions report template.

    Renders multi-store GHG emissions with Scope 1/2 breakdowns,
    intensity KPIs, and F-Gas tracking across markdown, HTML, and JSON.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize StoreEmissionsReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render store emissions report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_executive_summary(data),
            self._md_store_breakdown(data),
            self._md_scope1_detail(data),
            self._md_scope2_detail(data),
            self._md_consolidation(data),
            self._md_kpis(data),
            self._md_fgas_status(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render store emissions report as self-contained HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_executive_summary(data),
            self._html_store_breakdown(data),
            self._html_scope1_detail(data),
            self._html_scope2_detail(data),
            self._html_consolidation(data),
            self._html_kpis(data),
            self._html_fgas_status(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Store Emissions Report</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render store emissions report as structured JSON."""
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "store_emissions_report",
            "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_summary(data),
            "store_breakdown": data.get("per_store_results", []),
            "scope1": data.get("scope1_detail", {}),
            "scope2": data.get("scope2_detail", {}),
            "consolidated": data.get("consolidated", {}),
            "kpis": data.get("kpis", {}),
            "fgas_status": data.get("fgas_status", {}),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # -- Markdown sections --

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary in markdown."""
        title = data.get("title", "Store Emissions Report")
        year = data.get("reporting_year", "")
        total = data.get("total_emissions_tco2e", 0)
        s1 = data.get("scope1_total_tco2e", 0)
        s2 = data.get("scope2_total_tco2e", 0)
        stores = data.get("store_count", 0)
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# {title}\n\n**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n**Stores:** {stores}\n\n---\n\n"
            f"## Executive Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Emissions | {self._fmt(total)} tCO2e |\n"
            f"| Scope 1 | {self._fmt(s1)} tCO2e |\n"
            f"| Scope 2 (Location) | {self._fmt(s2)} tCO2e |"
        )

    def _md_store_breakdown(self, data: Dict[str, Any]) -> str:
        """Render store-by-store breakdown table."""
        stores = data.get("per_store_results", [])
        if not stores:
            return "## Store-by-Store Breakdown\n\n_No store data available._"
        lines = [
            "## Store-by-Store Breakdown", "",
            "| Store | Scope 1 | Scope 2 (Loc) | Total | Per sqm | Per Employee |",
            "|-------|---------|---------------|-------|---------|-------------|",
        ]
        for s in stores:
            name = s.get("store_name", s.get("store_id", "-"))
            lines.append(
                f"| {name} | {self._fmt(s.get('scope1_total_tco2e', 0))} | "
                f"{self._fmt(s.get('scope2_location_tco2e', 0))} | "
                f"{self._fmt(s.get('total_tco2e', 0))} | "
                f"{self._fmt(s.get('emissions_per_sqm', 0))} | "
                f"{self._fmt(s.get('emissions_per_employee', 0))} |"
            )
        return "\n".join(lines)

    def _md_scope1_detail(self, data: Dict[str, Any]) -> str:
        """Render Scope 1 detail section."""
        detail = data.get("scope1_detail", {})
        heating = detail.get("heating_tco2e", data.get("heating_total", 0))
        refrig = detail.get("refrigerant_tco2e", data.get("refrigerant_total", 0))
        fleet = detail.get("fleet_tco2e", data.get("fleet_total", 0))
        return (
            "## Scope 1 Detail\n\n"
            "| Source | Emissions (tCO2e) |\n|--------|------------------|\n"
            f"| Heating (Combustion) | {self._fmt(heating)} |\n"
            f"| Refrigerant Leakage | {self._fmt(refrig)} |\n"
            f"| Fleet Vehicles | {self._fmt(fleet)} |"
        )

    def _md_scope2_detail(self, data: Dict[str, Any]) -> str:
        """Render Scope 2 detail section."""
        loc = data.get("scope2_location_tco2e", 0)
        mkt = data.get("scope2_market_tco2e", 0)
        return (
            "## Scope 2 Detail\n\n"
            "| Method | Emissions (tCO2e) |\n|--------|------------------|\n"
            f"| Location-Based | {self._fmt(loc)} |\n"
            f"| Market-Based | {self._fmt(mkt)} |"
        )

    def _md_consolidation(self, data: Dict[str, Any]) -> str:
        """Render multi-store consolidation."""
        c = data.get("consolidated", {})
        return (
            "## Multi-Store Consolidation\n\n"
            f"- **Total Floor Area:** {self._fmt(c.get('total_floor_area_sqm', 0))} sqm\n"
            f"- **Total Employees:** {c.get('total_employees', 0)}\n"
            f"- **Highest Emitting Store:** {c.get('highest_emitting_store', '-')}\n"
            f"- **Lowest Emitting Store:** {c.get('lowest_emitting_store', '-')}"
        )

    def _md_kpis(self, data: Dict[str, Any]) -> str:
        """Render energy intensity KPIs."""
        kpis = data.get("kpis", {})
        return (
            "## Energy Intensity KPIs\n\n"
            "| KPI | Value |\n|-----|-------|\n"
            f"| Emissions per sqm | {self._fmt(kpis.get('emissions_per_sqm', 0))} tCO2e/sqm |\n"
            f"| Emissions per employee | {self._fmt(kpis.get('emissions_per_employee', 0))} tCO2e/emp |\n"
            f"| Refrigerant Share | {self._fmt(kpis.get('refrigerant_share_pct', 0))}% |\n"
            f"| Fleet Share | {self._fmt(kpis.get('fleet_share_pct', 0))}% |\n"
            f"| Renewable Electricity | {self._fmt(kpis.get('renewable_pct', 0))}% |"
        )

    def _md_fgas_status(self, data: Dict[str, Any]) -> str:
        """Render F-Gas phase-down status."""
        fgas = data.get("fgas_status", {})
        if not fgas:
            return "## F-Gas Phase-Down Status\n\n_No F-Gas data available._"
        lines = ["## F-Gas Phase-Down Status", ""]
        for ref, info in fgas.items():
            lines.append(f"- **{ref}**: GWP={info.get('gwp', '-')}, "
                         f"Charge={self._fmt(info.get('charge_kg', 0))} kg, "
                         f"Leakage={self._fmt(info.get('leakage_kg', 0))} kg")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-014 CSRD Retail Pack*"

    # -- HTML sections --

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary."""
        title = data.get("title", "Store Emissions Report")
        total = self._fmt(data.get("total_emissions_tco2e", 0))
        s1 = self._fmt(data.get("scope1_total_tco2e", 0))
        s2 = self._fmt(data.get("scope2_total_tco2e", 0))
        return (
            f'<h1>{title}</h1>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total</span><span class="value">{total} tCO2e</span></div>\n'
            f'  <div class="card"><span class="label">Scope 1</span><span class="value">{s1} tCO2e</span></div>\n'
            f'  <div class="card"><span class="label">Scope 2</span><span class="value">{s2} tCO2e</span></div>\n'
            f'</div>'
        )

    def _html_store_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML store breakdown table."""
        stores = data.get("per_store_results", [])
        rows = ""
        for s in stores:
            name = s.get("store_name", s.get("store_id", "-"))
            rows += (
                f'<tr><td>{name}</td><td>{self._fmt(s.get("scope1_total_tco2e", 0))}</td>'
                f'<td>{self._fmt(s.get("scope2_location_tco2e", 0))}</td>'
                f'<td>{self._fmt(s.get("total_tco2e", 0))}</td></tr>\n'
            )
        return (
            f'<h2>Store-by-Store Breakdown</h2>\n<table>\n'
            f'<tr><th>Store</th><th>Scope 1</th><th>Scope 2</th><th>Total</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope1_detail(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 1 detail."""
        return f'<h2>Scope 1 Detail</h2>\n<p>Heating, refrigerant leakage, fleet emissions.</p>'

    def _html_scope2_detail(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 2 detail."""
        return f'<h2>Scope 2 Detail</h2>\n<p>Location-based vs market-based electricity emissions.</p>'

    def _html_consolidation(self, data: Dict[str, Any]) -> str:
        """Render HTML consolidation."""
        return f'<h2>Multi-Store Consolidation</h2>'

    def _html_kpis(self, data: Dict[str, Any]) -> str:
        """Render HTML KPIs."""
        return f'<h2>Energy Intensity KPIs</h2>'

    def _html_fgas_status(self, data: Dict[str, Any]) -> str:
        """Render HTML F-Gas status."""
        return f'<h2>F-Gas Phase-Down Status</h2>'

    # -- JSON --

    def _json_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        return {
            "total_emissions_tco2e": data.get("total_emissions_tco2e", 0),
            "scope1_total_tco2e": data.get("scope1_total_tco2e", 0),
            "scope2_total_tco2e": data.get("scope2_total_tco2e", 0),
            "store_count": data.get("store_count", 0),
            "reporting_year": data.get("reporting_year", ""),
        }

    # -- Helpers --

    def _css(self) -> str:
        """Build inline CSS."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
