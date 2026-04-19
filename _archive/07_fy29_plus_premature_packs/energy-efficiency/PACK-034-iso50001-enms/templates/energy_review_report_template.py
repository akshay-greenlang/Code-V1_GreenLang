# -*- coding: utf-8 -*-
"""
EnergyReviewReportTemplate - ISO 50001 Clause 6.3 Energy Review for PACK-034.

Generates comprehensive energy review reports aligned with ISO 50001:2018
Clause 6.3. Covers energy consumption overview, Significant Energy Use (SEU)
analysis with Pareto chart data, energy drivers identification, baseline
status, EnPI performance, improvement opportunities, and data quality
assessment.

Sections:
    1. Executive Summary
    2. Energy Consumption Overview
    3. Significant Energy Uses (SEU) Analysis
    4. Energy Drivers
    5. Baseline Status
    6. EnPI Performance
    7. Improvement Opportunities
    8. Data Quality Assessment
    9. Conclusions

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnergyReviewReportTemplate:
    """
    ISO 50001 energy review report template.

    Renders energy review reports aligned with ISO 50001:2018 Clause 6.3,
    covering energy consumption analysis, SEU identification, energy driver
    analysis, EnPI performance, and improvement opportunities across
    markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyReviewReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render energy review report as Markdown.

        Args:
            data: Energy review data including facility_name, review_period,
                  total_consumption, seus, drivers, baselines, enpis,
                  and opportunities.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_consumption_overview(data),
            self._md_seu_analysis(data),
            self._md_energy_drivers(data),
            self._md_baseline_status(data),
            self._md_enpi_performance(data),
            self._md_improvement_opportunities(data),
            self._md_data_quality(data),
            self._md_conclusions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render energy review report as self-contained HTML.

        Args:
            data: Energy review data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_consumption_overview(data),
            self._html_seu_analysis(data),
            self._html_energy_drivers(data),
            self._html_enpi_performance(data),
            self._html_improvement_opportunities(data),
            self._html_data_quality(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Review Report - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render energy review report as structured JSON.

        Args:
            data: Energy review data dict.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "energy_review_report",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "facility_name": data.get("facility_name", ""),
            "review_period": data.get("review_period", ""),
            "executive_summary": self._json_executive_summary(data),
            "consumption_overview": self._json_consumption_overview(data),
            "seus": data.get("seus", []),
            "energy_drivers": data.get("drivers", []),
            "baselines": data.get("baselines", []),
            "enpis": data.get("enpis", []),
            "improvement_opportunities": data.get("opportunities", []),
            "data_quality": data.get("data_quality", {}),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with review metadata."""
        facility = data.get("facility_name", "Facility")
        period = data.get("review_period", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Energy Review Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Review Period:** {period}  \n"
            f"**ISO 50001:2018 Clause:** 6.3  \n"
            f"**Review Type:** {data.get('review_type', 'Comprehensive Energy Review')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 EnergyReviewReportTemplate v34.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary section."""
        total = data.get("total_consumption", {})
        seus = data.get("seus", [])
        opps = data.get("opportunities", [])
        seu_count = len(seus)
        opp_count = len(opps)
        total_savings = sum(o.get("estimated_savings_mwh", 0) for o in opps)
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Energy Consumption | {self._format_energy(total.get('total_mwh', 0))} |\n"
            f"| Total Energy Cost | {self._format_currency(total.get('total_cost', 0))} |\n"
            f"| Significant Energy Uses Identified | {seu_count} |\n"
            f"| SEU Share of Total Consumption | {self._fmt(total.get('seu_share_pct', 0))}% |\n"
            f"| Improvement Opportunities Found | {opp_count} |\n"
            f"| Total Identified Savings Potential | {self._format_energy(total_savings)} |\n"
            f"| Energy Performance vs Baseline | {data.get('performance_vs_baseline', '-')} |"
        )

    def _md_consumption_overview(self, data: Dict[str, Any]) -> str:
        """Render energy consumption overview section."""
        total = data.get("total_consumption", {})
        by_source = total.get("by_source", [])
        by_end_use = total.get("by_end_use", [])
        lines = [
            "## 2. Energy Consumption Overview\n",
            f"**Total Consumption:** {self._format_energy(total.get('total_mwh', 0))}  ",
            f"**Total Cost:** {self._format_currency(total.get('total_cost', 0))}  ",
            f"**Reporting Period:** {data.get('review_period', '-')}\n",
        ]
        if by_source:
            lines.extend([
                "### By Energy Source\n",
                "| Source | Consumption (MWh) | Share (%) | Cost |",
                "|--------|------------------|-----------|------|",
            ])
            for s in by_source:
                lines.append(
                    f"| {s.get('source', '-')} "
                    f"| {self._fmt(s.get('consumption_mwh', 0))} "
                    f"| {self._fmt(s.get('share_pct', 0))}% "
                    f"| {self._format_currency(s.get('cost', 0))} |"
                )
        if by_end_use:
            lines.extend([
                "\n### By End Use\n",
                "| End Use | Consumption (MWh) | Share (%) |",
                "|---------|------------------|-----------|",
            ])
            for e in by_end_use:
                lines.append(
                    f"| {e.get('end_use', '-')} "
                    f"| {self._fmt(e.get('consumption_mwh', 0))} "
                    f"| {self._fmt(e.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_seu_analysis(self, data: Dict[str, Any]) -> str:
        """Render SEU analysis section with Pareto data."""
        seus = data.get("seus", [])
        if not seus:
            return "## 3. Significant Energy Uses (SEU) Analysis\n\n_No SEUs identified._"
        lines = [
            "## 3. Significant Energy Uses (SEU) Analysis\n",
            "Significant Energy Uses are identified per ISO 50001:2018 Clause 6.3 based on "
            "substantial energy consumption and/or considerable potential for improvement.\n",
            "| # | SEU | Energy Type | Consumption (MWh) | Share (%) | Cumulative (%) | Criteria Met |",
            "|---|-----|------------|------------------|-----------|----------------|-------------|",
        ]
        cumulative = 0.0
        for i, seu in enumerate(seus, 1):
            share = seu.get("share_pct", 0)
            cumulative += share
            lines.append(
                f"| {i} | {seu.get('name', '-')} "
                f"| {seu.get('energy_type', '-')} "
                f"| {self._fmt(seu.get('consumption_mwh', 0))} "
                f"| {self._fmt(share)}% "
                f"| {self._fmt(cumulative)}% "
                f"| {seu.get('criteria_met', '-')} |"
            )
        lines.extend([
            "",
            "### SEU Selection Criteria\n",
            "- Substantial share of total energy consumption (typically >5%)",
            "- Considerable potential for energy performance improvement",
            "- Regulatory or stakeholder significance",
        ])
        return "\n".join(lines)

    def _md_energy_drivers(self, data: Dict[str, Any]) -> str:
        """Render energy drivers section."""
        drivers = data.get("drivers", [])
        if not drivers:
            return "## 4. Energy Drivers\n\n_No energy drivers identified._"
        lines = [
            "## 4. Energy Drivers\n",
            "Relevant variables affecting energy consumption for each SEU:\n",
            "| SEU | Driver | Type | Correlation (R2) | Impact Level |",
            "|-----|--------|------|------------------|-------------|",
        ]
        for d in drivers:
            lines.append(
                f"| {d.get('seu', '-')} "
                f"| {d.get('driver', '-')} "
                f"| {d.get('type', '-')} "
                f"| {self._fmt(d.get('r_squared', 0), 3)} "
                f"| {d.get('impact_level', '-')} |"
            )
        return "\n".join(lines)

    def _md_baseline_status(self, data: Dict[str, Any]) -> str:
        """Render baseline status section."""
        baselines = data.get("baselines", [])
        if not baselines:
            return "## 5. Baseline Status\n\n_No baselines established._"
        lines = [
            "## 5. Baseline Status\n",
            "Energy baselines (EnBs) represent the quantitative reference for "
            "comparison of energy performance per ISO 50001:2018 Clause 6.5.\n",
            "| SEU / Metric | Baseline Period | Baseline Value | Model Type | R2 | Status |",
            "|-------------|----------------|---------------|------------|-----|--------|",
        ]
        for b in baselines:
            lines.append(
                f"| {b.get('metric', '-')} "
                f"| {b.get('baseline_period', '-')} "
                f"| {self._fmt(b.get('baseline_value', 0))} "
                f"| {b.get('model_type', '-')} "
                f"| {self._fmt(b.get('r_squared', 0), 3)} "
                f"| {b.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_enpi_performance(self, data: Dict[str, Any]) -> str:
        """Render EnPI performance section."""
        enpis = data.get("enpis", [])
        if not enpis:
            return "## 6. EnPI Performance\n\n_No EnPIs defined._"
        lines = [
            "## 6. EnPI Performance\n",
            "| EnPI | Current Value | Baseline Value | Change (%) | Target | Status |",
            "|------|-------------|---------------|-----------|--------|--------|",
        ]
        for e in enpis:
            baseline_val = e.get("baseline_value", 0)
            current_val = e.get("current_value", 0)
            change_pct = 0.0
            if baseline_val and baseline_val != 0:
                change_pct = ((current_val - baseline_val) / abs(baseline_val)) * 100
            lines.append(
                f"| {e.get('name', '-')} "
                f"| {self._fmt(current_val)} {e.get('unit', '')} "
                f"| {self._fmt(baseline_val)} {e.get('unit', '')} "
                f"| {self._fmt(change_pct)}% "
                f"| {e.get('target', '-')} "
                f"| {e.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_improvement_opportunities(self, data: Dict[str, Any]) -> str:
        """Render improvement opportunities section."""
        opps = data.get("opportunities", [])
        if not opps:
            return "## 7. Improvement Opportunities\n\n_No improvement opportunities identified._"
        lines = [
            "## 7. Improvement Opportunities\n",
            "| # | Opportunity | SEU | Savings (MWh/yr) | Cost Savings (/yr) | Investment | Payback (mo) | Priority |",
            "|---|-----------|-----|-----------------|-------------------|-----------|-------------|----------|",
        ]
        for i, o in enumerate(opps, 1):
            lines.append(
                f"| {i} | {o.get('description', '-')} "
                f"| {o.get('seu', '-')} "
                f"| {self._fmt(o.get('estimated_savings_mwh', 0))} "
                f"| {self._format_currency(o.get('cost_savings', 0))} "
                f"| {self._format_currency(o.get('investment', 0))} "
                f"| {self._fmt(o.get('payback_months', 0), 1)} "
                f"| {o.get('priority', '-')} |"
            )
        total_savings = sum(o.get("estimated_savings_mwh", 0) for o in opps)
        total_cost = sum(o.get("cost_savings", 0) for o in opps)
        lines.append(
            f"| | **TOTAL** | | **{self._fmt(total_savings)}** "
            f"| **{self._format_currency(total_cost)}** | | | |"
        )
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render data quality assessment section."""
        dq = data.get("data_quality", {})
        metrics = dq.get("metrics", [])
        lines = [
            "## 8. Data Quality Assessment\n",
            f"**Overall Data Quality Score:** {self._fmt(dq.get('overall_score', 0))}%  ",
            f"**Data Completeness:** {self._fmt(dq.get('completeness_pct', 0))}%  ",
            f"**Measurement Coverage:** {self._fmt(dq.get('measurement_coverage_pct', 0))}%\n",
        ]
        if metrics:
            lines.extend([
                "| Data Source | Completeness (%) | Accuracy | Method | Gaps |",
                "|------------|-----------------|----------|--------|------|",
            ])
            for m in metrics:
                lines.append(
                    f"| {m.get('source', '-')} "
                    f"| {self._fmt(m.get('completeness_pct', 0))}% "
                    f"| {m.get('accuracy', '-')} "
                    f"| {m.get('method', '-')} "
                    f"| {m.get('gaps', '-')} |"
                )
        return "\n".join(lines)

    def _md_conclusions(self, data: Dict[str, Any]) -> str:
        """Render conclusions section."""
        conclusions = data.get("conclusions", [])
        if not conclusions:
            conclusions = [
                "Energy review completed per ISO 50001:2018 Clause 6.3 requirements",
                "Significant Energy Uses identified and prioritized",
                "Energy baselines and EnPIs established or validated",
                "Improvement opportunities quantified and prioritized",
            ]
        lines = ["## 9. Conclusions\n"]
        for i, c in enumerate(conclusions, 1):
            lines.append(f"{i}. {c}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Energy Review Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("review_period", "-")} | '
            f'ISO 50001 Clause 6.3</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        total = data.get("total_consumption", {})
        seus = data.get("seus", [])
        opps = data.get("opportunities", [])
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Consumption</span>'
            f'<span class="value">{self._format_energy(total.get("total_mwh", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Total Cost</span>'
            f'<span class="value">{self._format_currency(total.get("total_cost", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">SEUs Identified</span>'
            f'<span class="value">{len(seus)}</span></div>\n'
            f'  <div class="card"><span class="label">Opportunities</span>'
            f'<span class="value">{len(opps)}</span></div>\n'
            '</div>'
        )

    def _html_consumption_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML consumption overview."""
        by_source = data.get("total_consumption", {}).get("by_source", [])
        rows = ""
        for s in by_source:
            rows += (
                f'<tr><td>{s.get("source", "-")}</td>'
                f'<td>{self._fmt(s.get("consumption_mwh", 0))} MWh</td>'
                f'<td>{self._fmt(s.get("share_pct", 0))}%</td>'
                f'<td>{self._format_currency(s.get("cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>Energy Consumption Overview</h2>\n'
            '<table>\n<tr><th>Source</th><th>Consumption</th>'
            f'<th>Share</th><th>Cost</th></tr>\n{rows}</table>'
        )

    def _html_seu_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML SEU analysis table."""
        seus = data.get("seus", [])
        rows = ""
        cumulative = 0.0
        for seu in seus:
            share = seu.get("share_pct", 0)
            cumulative += share
            rows += (
                f'<tr><td><strong>{seu.get("name", "-")}</strong></td>'
                f'<td>{seu.get("energy_type", "-")}</td>'
                f'<td>{self._fmt(seu.get("consumption_mwh", 0))} MWh</td>'
                f'<td>{self._fmt(share)}%</td>'
                f'<td>{self._fmt(cumulative)}%</td></tr>\n'
            )
        return (
            '<h2>Significant Energy Uses (SEU)</h2>\n'
            '<table>\n<tr><th>SEU</th><th>Energy Type</th>'
            f'<th>Consumption</th><th>Share</th><th>Cumulative</th></tr>\n{rows}</table>'
        )

    def _html_energy_drivers(self, data: Dict[str, Any]) -> str:
        """Render HTML energy drivers."""
        drivers = data.get("drivers", [])
        rows = ""
        for d in drivers:
            rows += (
                f'<tr><td>{d.get("seu", "-")}</td>'
                f'<td>{d.get("driver", "-")}</td>'
                f'<td>{d.get("type", "-")}</td>'
                f'<td>{self._fmt(d.get("r_squared", 0), 3)}</td></tr>\n'
            )
        return (
            '<h2>Energy Drivers</h2>\n'
            '<table>\n<tr><th>SEU</th><th>Driver</th>'
            f'<th>Type</th><th>R2</th></tr>\n{rows}</table>'
        )

    def _html_enpi_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI performance."""
        enpis = data.get("enpis", [])
        rows = ""
        for e in enpis:
            status = e.get("status", "").lower()
            cls = "status-improved" if status == "improved" else "status-declined" if status == "declined" else ""
            rows += (
                f'<tr><td>{e.get("name", "-")}</td>'
                f'<td>{self._fmt(e.get("current_value", 0))} {e.get("unit", "")}</td>'
                f'<td>{self._fmt(e.get("baseline_value", 0))} {e.get("unit", "")}</td>'
                f'<td class="{cls}">{e.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>EnPI Performance</h2>\n'
            '<table>\n<tr><th>EnPI</th><th>Current</th>'
            f'<th>Baseline</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_improvement_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement opportunities."""
        opps = data.get("opportunities", [])
        rows = ""
        for o in opps:
            rows += (
                f'<tr><td>{o.get("description", "-")}</td>'
                f'<td>{o.get("seu", "-")}</td>'
                f'<td>{self._fmt(o.get("estimated_savings_mwh", 0))} MWh</td>'
                f'<td>{self._format_currency(o.get("cost_savings", 0))}</td>'
                f'<td><span class="priority-{o.get("priority", "medium").lower()}">'
                f'{o.get("priority", "-")}</span></td></tr>\n'
            )
        return (
            '<h2>Improvement Opportunities</h2>\n'
            '<table>\n<tr><th>Opportunity</th><th>SEU</th>'
            f'<th>Savings</th><th>Cost Savings</th><th>Priority</th></tr>\n{rows}</table>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality assessment."""
        dq = data.get("data_quality", {})
        return (
            '<h2>Data Quality Assessment</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Overall Score</span>'
            f'<span class="value">{self._fmt(dq.get("overall_score", 0), 0)}%</span></div>\n'
            f'  <div class="card"><span class="label">Completeness</span>'
            f'<span class="value">{self._fmt(dq.get("completeness_pct", 0), 0)}%</span></div>\n'
            f'  <div class="card"><span class="label">Measurement Coverage</span>'
            f'<span class="value">{self._fmt(dq.get("measurement_coverage_pct", 0), 0)}%</span></div>\n'
            '</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        total = data.get("total_consumption", {})
        seus = data.get("seus", [])
        opps = data.get("opportunities", [])
        return {
            "total_consumption_mwh": total.get("total_mwh", 0),
            "total_cost": total.get("total_cost", 0),
            "seu_count": len(seus),
            "seu_share_pct": total.get("seu_share_pct", 0),
            "opportunity_count": len(opps),
            "total_savings_potential_mwh": sum(
                o.get("estimated_savings_mwh", 0) for o in opps
            ),
        }

    def _json_consumption_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON consumption overview."""
        total = data.get("total_consumption", {})
        return {
            "total_mwh": total.get("total_mwh", 0),
            "total_cost": total.get("total_cost", 0),
            "by_source": total.get("by_source", []),
            "by_end_use": total.get("by_end_use", []),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        seus = data.get("seus", [])
        by_source = data.get("total_consumption", {}).get("by_source", [])
        enpis = data.get("enpis", [])
        return {
            "source_pie": {
                "type": "pie",
                "labels": [s.get("source", "") for s in by_source],
                "values": [s.get("consumption_mwh", 0) for s in by_source],
            },
            "seu_pareto": {
                "type": "pareto",
                "labels": [s.get("name", "") for s in seus],
                "values": [s.get("consumption_mwh", 0) for s in seus],
                "cumulative_pct": self._compute_cumulative_pct(seus),
            },
            "enpi_comparison": {
                "type": "grouped_bar",
                "labels": [e.get("name", "") for e in enpis],
                "series": {
                    "baseline": [e.get("baseline_value", 0) for e in enpis],
                    "current": [e.get("current_value", 0) for e in enpis],
                },
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_cumulative_pct(self, seus: List[Dict[str, Any]]) -> List[float]:
        """Compute cumulative percentage for Pareto chart."""
        cumulative = 0.0
        result: List[float] = []
        for seu in seus:
            cumulative += seu.get("share_pct", 0)
            result.append(round(cumulative, 2))
        return result

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".status-improved{color:#198754;font-weight:600;}"
            ".status-declined{color:#dc3545;font-weight:600;}"
            ".priority-high{color:#dc3545;font-weight:700;}"
            ".priority-medium{color:#fd7e14;font-weight:600;}"
            ".priority-low{color:#198754;font-weight:500;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string.
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} MWh"
        return str(val)

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
