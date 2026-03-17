# -*- coding: utf-8 -*-
"""
FinancialMaterialityReportTemplate - Financial assessment report for PACK-015.

Sections:
    1. Financial Assessment Overview
    2. Risk/Opportunity Analysis
    3. KPI Impact Summary
    4. Material Financial Items
    5. Time Horizon Analysis
    6. Methodology Notes

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FinancialMaterialityReportTemplate:
    """
    Financial materiality assessment report template.

    Renders risk/opportunity analysis, KPI impact mapping, material
    financial rankings, and time horizon analysis for the outside-in
    dimension of double materiality per ESRS 1.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FinancialMaterialityReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render financial materiality report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overview(data),
            self._md_risk_opportunity(data),
            self._md_kpi_impact(data),
            self._md_material_items(data),
            self._md_time_horizon(data),
            self._md_methodology(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render financial materiality report as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_risk_opportunity(data),
            self._html_material_items(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Financial Materiality Report</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render financial materiality report as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "financial_materiality_report",
            "version": "15.0.0",
            "generated_at": self.generated_at.isoformat(),
            "exposures_assessed": data.get("exposures_assessed", 0),
            "material_items": data.get("material_items", 0),
            "total_exposure_eur": data.get("total_exposure_eur", 0),
            "financial_scores": data.get("financial_scores", []),
            "ranked_items": data.get("ranked_items", []),
            "risk_type_distribution": data.get("risk_type_distribution", {}),
            "kpi_impact_summary": data.get("kpi_impact_summary", {}),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"# Financial Materiality Assessment Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render assessment overview."""
        assessed = data.get("exposures_assessed", 0)
        material = data.get("material_items", 0)
        total_eur = data.get("total_exposure_eur", 0)
        return (
            "## Financial Assessment Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Financial Exposures Assessed | {assessed} |\n"
            f"| Material Financial Items | {material} |\n"
            f"| Total Estimated Exposure | {self._fmt_eur(total_eur)} |\n"
            f"| Materiality Rate | {self._pct(material, assessed)} |"
        )

    def _md_risk_opportunity(self, data: Dict[str, Any]) -> str:
        """Render risk/opportunity distribution."""
        dist = data.get("risk_type_distribution", {})
        if not dist:
            return "## Risk/Opportunity Analysis\n\n_No risk type data available._"
        lines = [
            "## Risk/Opportunity Distribution", "",
            "| Risk/Opportunity Type | Count |",
            "|----------------------|-------|",
        ]
        for rtype, count in sorted(dist.items()):
            label = rtype.replace("_", " ").title()
            lines.append(f"| {label} | {count} |")
        return "\n".join(lines)

    def _md_kpi_impact(self, data: Dict[str, Any]) -> str:
        """Render KPI impact summary."""
        kpi = data.get("kpi_impact_summary", {})
        if not kpi:
            return "## KPI Impact Summary\n\n_No KPI impact data available._"
        lines = [
            "## KPI Impact Summary", "",
            "| Financial KPI | Estimated Impact (EUR) |",
            "|--------------|----------------------|",
        ]
        for kpi_name, impact in sorted(kpi.items(), key=lambda x: -abs(x[1])):
            label = kpi_name.replace("_", " ").title()
            lines.append(f"| {label} | {self._fmt_eur(impact)} |")
        return "\n".join(lines)

    def _md_material_items(self, data: Dict[str, Any]) -> str:
        """Render material financial items ranking."""
        ranked = data.get("ranked_items", [])
        material = [r for r in ranked if r.get("is_material", False)]
        if not material:
            return "## Material Financial Items\n\n_No material financial items identified._"
        lines = [
            "## Material Financial Items (Ranked)", "",
            "| Rank | Matter | Risk Type | Impact (EUR) | Score |",
            "|------|--------|-----------|-------------|-------|",
        ]
        for r in material:
            lines.append(
                f"| {r.get('rank', '-')} | {r.get('matter_name', '-')} | "
                f"{r.get('risk_type', '-')} | "
                f"{self._fmt_eur(r.get('estimated_impact_eur', 0))} | "
                f"**{self._fmt(r.get('composite_score', 0))}** |"
            )
        return "\n".join(lines)

    def _md_time_horizon(self, data: Dict[str, Any]) -> str:
        """Render time horizon analysis."""
        ranked = data.get("ranked_items", [])
        if not ranked:
            return "## Time Horizon Analysis\n\n_No data available._"

        # Count by estimated time horizon from scores
        scores = data.get("financial_scores", [])
        horizon_counts: Dict[str, int] = {"short_term": 0, "medium_term": 0, "long_term": 0}
        for s in scores:
            th_score = s.get("time_horizon_score", 0)
            if th_score >= 4.0:
                horizon_counts["short_term"] += 1
            elif th_score >= 2.5:
                horizon_counts["medium_term"] += 1
            else:
                horizon_counts["long_term"] += 1

        lines = [
            "## Time Horizon Analysis", "",
            "| Time Horizon | Exposures |",
            "|-------------|-----------|",
        ]
        for horizon, count in horizon_counts.items():
            label = horizon.replace("_", " ").title()
            lines.append(f"| {label} | {count} |")
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render methodology notes."""
        return (
            "## Methodology\n\n"
            "This financial materiality assessment follows ESRS 1 Chapter 3 "
            "(outside-in perspective). Scoring dimensions: magnitude (impact "
            "as % of revenue), likelihood (confidence-weighted), and time "
            "horizon (urgency-weighted). Composite scores use configurable "
            "weighted averages."
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-015 Double Materiality Pack*"

    # -- HTML sections --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>Financial Materiality Assessment Report</h1>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview."""
        total = self._fmt_eur(data.get("total_exposure_eur", 0))
        return f'<h2>Overview</h2>\n<p>Total Exposure: {total}</p>'

    def _html_risk_opportunity(self, data: Dict[str, Any]) -> str:
        """Render HTML risk/opportunity distribution."""
        return '<h2>Risk/Opportunity Distribution</h2>'

    def _html_material_items(self, data: Dict[str, Any]) -> str:
        """Render HTML material items."""
        ranked = data.get("ranked_items", [])
        material = [r for r in ranked if r.get("is_material", False)]
        rows = ""
        for r in material:
            rows += (
                f'<tr><td>{r.get("rank", "-")}</td>'
                f'<td>{r.get("matter_name", "-")}</td>'
                f'<td>{self._fmt(r.get("composite_score", 0))}</td></tr>\n'
            )
        return (
            f'<h2>Material Financial Items</h2>\n'
            f'<table><tr><th>Rank</th><th>Matter</th><th>Score</th></tr>\n'
            f'{rows}</table>'
        )

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

    def _fmt_eur(self, val: Any) -> str:
        """Format EUR currency value."""
        if isinstance(val, (int, float)):
            return f"EUR {val:,.0f}"
        return str(val)

    def _pct(self, part: int, total: int) -> str:
        """Format percentage."""
        if total == 0:
            return "0.0%"
        return f"{part / total * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
