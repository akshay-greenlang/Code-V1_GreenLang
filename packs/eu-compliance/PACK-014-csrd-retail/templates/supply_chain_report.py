# -*- coding: utf-8 -*-
"""
SupplyChainReportTemplate - Scope 3 supply chain report for PACK-014.

Sections:
    1. Scope 3 Overview
    2. Category Breakdown (15 categories)
    3. Hotspot Analysis (top 20 suppliers/products)
    4. Data Quality Assessment
    5. Supplier Engagement Scorecard
    6. Recommendations

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SupplyChainReportTemplate:
    """
    Scope 3 supply chain report template for retail.

    Renders category breakdowns, hotspot analysis, data quality
    metrics, and supplier engagement scorecards.
    """

    SCOPE3_CATEGORIES: List[str] = [
        "Cat 1: Purchased Goods & Services", "Cat 2: Capital Goods",
        "Cat 3: Fuel & Energy", "Cat 4: Upstream Transport",
        "Cat 5: Waste", "Cat 6: Business Travel",
        "Cat 7: Commuting", "Cat 8: Upstream Leased",
        "Cat 9: Downstream Transport", "Cat 10: Processing",
        "Cat 11: Use of Sold Products", "Cat 12: End-of-Life",
        "Cat 13: Downstream Leased", "Cat 14: Franchises",
        "Cat 15: Investments",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SupplyChainReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render supply chain report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_scope3_overview(data),
            self._md_category_breakdown(data),
            self._md_hotspot_analysis(data),
            self._md_data_quality(data),
            self._md_engagement_scorecard(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render supply chain report as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_scope3_overview(data),
            self._html_category_breakdown(data),
            self._html_hotspot_analysis(data),
            self._html_data_quality(data),
            self._html_engagement_scorecard(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Supply Chain Report</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render supply chain report as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "supply_chain_report",
            "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "scope3_total_tco2e": data.get("scope3_total_tco2e", 0),
            "category_breakdown": data.get("category_breakdown", []),
            "hotspots": data.get("hotspots", []),
            "data_quality": data.get("data_quality_summary", {}),
            "engagement_plan": data.get("engagement_plan", []),
            "supplier_count": data.get("supplier_count", 0),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Supply Chain Emissions Report\n\n**Generated:** {ts}\n\n---"

    def _md_scope3_overview(self, data: Dict[str, Any]) -> str:
        """Render Scope 3 overview."""
        total = data.get("scope3_total_tco2e", 0)
        sups = data.get("supplier_count", 0)
        return (
            "## Scope 3 Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Scope 3 | {self._fmt(total)} tCO2e |\n"
            f"| Suppliers Assessed | {sups} |"
        )

    def _md_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Render category breakdown table."""
        cats = data.get("category_breakdown", [])
        if not cats:
            return "## Category Breakdown\n\n_No category data available._"
        lines = [
            "## Scope 3 Category Breakdown", "",
            "| Category | Emissions (tCO2e) | Method | Data Quality |",
            "|----------|------------------|--------|-------------|",
        ]
        for c in cats:
            name = c.get("category_name", c.get("category_id", "-"))
            lines.append(
                f"| {name} | {self._fmt(c.get('emissions_tco2e', 0))} | "
                f"{c.get('method', '-')} | {c.get('data_quality', '-')} |"
            )
        return "\n".join(lines)

    def _md_hotspot_analysis(self, data: Dict[str, Any]) -> str:
        """Render hotspot analysis."""
        hotspots = data.get("hotspots", [])
        if not hotspots:
            return "## Hotspot Analysis\n\n_No hotspots identified._"
        lines = [
            "## Hotspot Analysis (Top Emitters)", "",
            "| Rank | Type | Name | Emissions (tCO2e) | Share |",
            "|------|------|------|------------------|-------|",
        ]
        for h in hotspots[:20]:
            lines.append(
                f"| {h.get('rank', '-')} | {h.get('entity_type', '-')} | "
                f"{h.get('entity_name', '-')} | {self._fmt(h.get('emissions_tco2e', 0))} | "
                f"{self._fmt(h.get('share_of_total_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render data quality assessment."""
        dq = data.get("data_quality_summary", {})
        if not dq:
            return "## Data Quality Assessment\n\n_No data quality summary available._"
        lines = ["## Data Quality Assessment", "", "| Level | Supplier Count |", "|-------|---------------|"]
        for level, count in sorted(dq.items()):
            lines.append(f"| {level} | {count} |")
        return "\n".join(lines)

    def _md_engagement_scorecard(self, data: Dict[str, Any]) -> str:
        """Render supplier engagement scorecard."""
        plan = data.get("engagement_plan", [])
        if not plan:
            return "## Supplier Engagement Scorecard\n\n_No engagement plan._"
        lines = [
            "## Supplier Engagement Scorecard", "",
            "| Supplier | Priority | Action | Expected Reduction |",
            "|----------|----------|--------|-------------------|",
        ]
        for a in plan[:15]:
            lines.append(
                f"| {a.get('supplier_name', '-')} | {a.get('priority', '-')} | "
                f"{a.get('action_type', '-')} | {self._fmt(a.get('expected_reduction_tco2e', 0))} tCO2e |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## Recommendations\n\n- Increase primary data collection from top 20 suppliers\n- Set SBTi-aligned scope 3 targets\n- Implement supplier engagement program"
        lines = ["## Recommendations", ""]
        for i, r in enumerate(recs, 1):
            lines.append(f"{i}. {r}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-014 CSRD Retail Pack*"

    # -- HTML --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>Supply Chain Emissions Report</h1>'

    def _html_scope3_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview."""
        total = self._fmt(data.get("scope3_total_tco2e", 0))
        return f'<h2>Scope 3 Overview</h2>\n<p>Total: {total} tCO2e</p>'

    def _html_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML category table."""
        cats = data.get("category_breakdown", [])
        rows = ""
        for c in cats:
            rows += f'<tr><td>{c.get("category_name", "-")}</td><td>{self._fmt(c.get("emissions_tco2e", 0))}</td></tr>\n'
        return f'<h2>Category Breakdown</h2>\n<table><tr><th>Category</th><th>Emissions</th></tr>\n{rows}</table>'

    def _html_hotspot_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML hotspot analysis."""
        return '<h2>Hotspot Analysis</h2>'

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality."""
        return '<h2>Data Quality Assessment</h2>'

    def _html_engagement_scorecard(self, data: Dict[str, Any]) -> str:
        """Render HTML engagement scorecard."""
        return '<h2>Supplier Engagement Scorecard</h2>'

    # -- Helpers --

    def _css(self) -> str:
        """Build CSS."""
        return (
            "body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format numeric value."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
