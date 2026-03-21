# -*- coding: utf-8 -*-
"""
ScenarioComparisonReportTemplate - Multi-scenario comparison for PACK-022.

Renders a multi-scenario comparison report with tornado chart data, Monte Carlo
simulation results, sensitivity analysis, and a decision matrix scoring cost,
risk, and ambition for each scenario pathway.

Sections:
    1. Executive Summary
    2. Scenario Definitions
    3. Monte Carlo Results (P10/P25/median/P75/P90)
    4. Scenario Comparison (delta table)
    5. Sensitivity Analysis (tornado ranking)
    6. Decision Matrix (cost vs risk vs ambition)
    7. Recommended Pathway
    8. Key Assumptions

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    """Format a value as a Decimal string with fixed decimal places."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 2) -> str:
    """Format a Decimal value with thousands separator."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)


class ScenarioComparisonReportTemplate:
    """
    Multi-scenario comparison report template.

    Compares net-zero scenarios across cost, risk, ambition, and Monte Carlo
    distributions. Provides tornado chart rankings for sensitivity analysis
    and a decision matrix to support pathway selection.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ScenarioComparisonReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render scenario comparison report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_scenario_definitions(data),
            self._md_monte_carlo(data),
            self._md_scenario_comparison(data),
            self._md_sensitivity_analysis(data),
            self._md_decision_matrix(data),
            self._md_recommended_pathway(data),
            self._md_key_assumptions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render scenario comparison report as self-contained HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_scenario_definitions(data),
            self._html_monte_carlo(data),
            self._html_scenario_comparison(data),
            self._html_sensitivity_analysis(data),
            self._html_decision_matrix(data),
            self._html_recommended_pathway(data),
            self._html_key_assumptions(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Scenario Comparison Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render scenario comparison report as structured JSON."""
        self.generated_at = _utcnow()
        scenarios = data.get("scenarios", [])
        monte_carlo = data.get("monte_carlo", {})
        sensitivity = data.get("sensitivity", [])
        decision_matrix = data.get("decision_matrix", [])
        recommended = data.get("recommended_pathway", {})

        result: Dict[str, Any] = {
            "template": "scenario_comparison_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "executive_summary": {
                "scenario_count": len(scenarios),
                "recommended_scenario": recommended.get("name", ""),
                "recommended_score": str(recommended.get("total_score", 0)),
                "simulation_runs": monte_carlo.get("runs", 0),
            },
            "scenarios": scenarios,
            "monte_carlo": monte_carlo,
            "sensitivity": sensitivity,
            "decision_matrix": decision_matrix,
            "recommended_pathway": recommended,
            "assumptions": data.get("assumptions", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Scenario Comparison Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", [])
        recommended = data.get("recommended_pathway", {})
        monte_carlo = data.get("monte_carlo", {})
        return (
            "## 1. Executive Summary\n\n"
            f"This report compares **{len(scenarios)}** net-zero scenarios for "
            f"**{data.get('org_name', 'the organization')}** using Monte Carlo "
            f"simulation ({_dec_comma(monte_carlo.get('runs', 0), 0)} runs), "
            f"sensitivity analysis, and multi-criteria decision scoring.\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Scenarios Analyzed | {len(scenarios)} |\n"
            f"| Simulation Runs | {_dec_comma(monte_carlo.get('runs', 0), 0)} |\n"
            f"| Sensitivity Factors | {len(data.get('sensitivity', []))} |\n"
            f"| Recommended Pathway | {recommended.get('name', 'N/A')} |\n"
            f"| Decision Score | {_dec(recommended.get('total_score', 0))} / 10.00 |"
        )

    def _md_scenario_definitions(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", [])
        lines = [
            "## 2. Scenario Definitions\n",
            "| # | Scenario | Description | Target Year | Ambition | Key Lever |",
            "|---|----------|-------------|:-----------:|----------|-----------|",
        ]
        for i, sc in enumerate(scenarios, 1):
            lines.append(
                f"| {i} | {sc.get('name', '-')} | {sc.get('description', '-')} "
                f"| {sc.get('target_year', '-')} | {sc.get('ambition', '-')} "
                f"| {sc.get('key_lever', '-')} |"
            )
        if not scenarios:
            lines.append("| - | _No scenarios defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_monte_carlo(self, data: Dict[str, Any]) -> str:
        mc = data.get("monte_carlo", {})
        runs = mc.get("runs", 0)
        results = mc.get("results", [])
        lines = [
            "## 3. Monte Carlo Simulation Results\n",
            f"**Simulation Runs:** {_dec_comma(runs, 0)}  \n"
            f"**Distribution Type:** {mc.get('distribution', 'Normal')}  \n"
            f"**Confidence Level:** {mc.get('confidence_pct', 95)}%\n",
            "| Scenario | P10 (tCO2e) | P25 (tCO2e) | Median (tCO2e) | P75 (tCO2e) | P90 (tCO2e) | Std Dev |",
            "|----------|------------:|------------:|---------------:|------------:|------------:|--------:|",
        ]
        for r in results:
            lines.append(
                f"| {r.get('scenario', '-')} "
                f"| {_dec_comma(r.get('p10', 0))} "
                f"| {_dec_comma(r.get('p25', 0))} "
                f"| {_dec_comma(r.get('median', 0))} "
                f"| {_dec_comma(r.get('p75', 0))} "
                f"| {_dec_comma(r.get('p90', 0))} "
                f"| {_dec_comma(r.get('std_dev', 0))} |"
            )
        if not results:
            lines.append("| _No simulation results_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_scenario_comparison(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", [])
        base = data.get("base_scenario", "")
        lines = [
            "## 4. Scenario Comparison (Delta Analysis)\n",
            f"**Base Scenario:** {base if base else 'First scenario listed'}\n",
            "| Scenario | Emissions Delta (tCO2e) | Cost Delta (EUR) | Risk Delta | Timeline Delta (yrs) | Ambition Delta |",
            "|----------|------------------------:|------------------:|:----------:|---------------------:|:--------------:|",
        ]
        for sc in scenarios:
            delta = sc.get("delta", {})
            lines.append(
                f"| {sc.get('name', '-')} "
                f"| {_dec_comma(delta.get('emissions_tco2e', 0))} "
                f"| {_dec_comma(delta.get('cost_eur', 0), 0)} "
                f"| {delta.get('risk', '-')} "
                f"| {delta.get('timeline_years', 0)} "
                f"| {delta.get('ambition', '-')} |"
            )
        if not scenarios:
            lines.append("| _No scenario comparisons_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        sensitivity = data.get("sensitivity", [])
        lines = [
            "## 5. Sensitivity Analysis (Tornado Ranking)\n",
            "Factors ranked by impact magnitude on net-zero target achievement.\n",
            "| Rank | Factor | Low Case Impact (tCO2e) | High Case Impact (tCO2e) | Swing (tCO2e) | Most Affected Scenario |",
            "|:----:|--------|------------------------:|-------------------------:|--------------:|-----------------------|",
        ]
        sorted_sens = sorted(sensitivity, key=lambda x: abs(x.get("swing_tco2e", 0)), reverse=True)
        for i, sf in enumerate(sorted_sens, 1):
            lines.append(
                f"| {i} | {sf.get('factor', '-')} "
                f"| {_dec_comma(sf.get('low_impact_tco2e', 0))} "
                f"| {_dec_comma(sf.get('high_impact_tco2e', 0))} "
                f"| {_dec_comma(sf.get('swing_tco2e', 0))} "
                f"| {sf.get('most_affected_scenario', '-')} |"
            )
        if not sensitivity:
            lines.append("| - | _No sensitivity factors_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_decision_matrix(self, data: Dict[str, Any]) -> str:
        matrix = data.get("decision_matrix", [])
        lines = [
            "## 6. Decision Matrix\n",
            "Scenarios scored on cost (0-10), risk (0-10), and ambition (0-10). "
            "Weights applied per organizational priorities.\n",
            "| Scenario | Cost Score | Risk Score | Ambition Score | Weighted Total | Rank |",
            "|----------|:---------:|:---------:|:--------------:|:--------------:|:----:|",
        ]
        sorted_matrix = sorted(matrix, key=lambda x: x.get("total_score", 0), reverse=True)
        for rank, dm in enumerate(sorted_matrix, 1):
            lines.append(
                f"| {dm.get('scenario', '-')} "
                f"| {_dec(dm.get('cost_score', 0), 1)} "
                f"| {_dec(dm.get('risk_score', 0), 1)} "
                f"| {_dec(dm.get('ambition_score', 0), 1)} "
                f"| **{_dec(dm.get('total_score', 0), 1)}** "
                f"| {rank} |"
            )
        if not matrix:
            lines.append("| _No decision matrix data_ | - | - | - | - | - |")

        weights = data.get("weights", {})
        if weights:
            lines.append("")
            lines.append(
                f"**Weights Applied:** Cost={_dec(weights.get('cost', 0.33), 2)}, "
                f"Risk={_dec(weights.get('risk', 0.33), 2)}, "
                f"Ambition={_dec(weights.get('ambition', 0.34), 2)}"
            )
        return "\n".join(lines)

    def _md_recommended_pathway(self, data: Dict[str, Any]) -> str:
        rec = data.get("recommended_pathway", {})
        lines = [
            "## 7. Recommended Pathway\n",
            f"**Scenario:** {rec.get('name', 'N/A')}  \n"
            f"**Total Score:** {_dec(rec.get('total_score', 0), 1)} / 10.0  \n"
            f"**Rationale:** {rec.get('rationale', 'N/A')}\n",
        ]
        key_actions = rec.get("key_actions", [])
        if key_actions:
            lines.append("### Key Actions")
            lines.append("")
            for i, action in enumerate(key_actions, 1):
                lines.append(f"{i}. {action}")
        milestones = rec.get("milestones", [])
        if milestones:
            lines.append("")
            lines.append("### Implementation Milestones")
            lines.append("")
            lines.append("| Year | Milestone | Expected Outcome |")
            lines.append("|------|-----------|-----------------|")
            for ms in milestones:
                lines.append(
                    f"| {ms.get('year', '-')} | {ms.get('milestone', '-')} "
                    f"| {ms.get('outcome', '-')} |"
                )
        return "\n".join(lines)

    def _md_key_assumptions(self, data: Dict[str, Any]) -> str:
        assumptions = data.get("assumptions", [])
        lines = [
            "## 8. Key Assumptions\n",
        ]
        if assumptions:
            lines.append("| # | Assumption | Category | Impact if Wrong |")
            lines.append("|---|------------|----------|----------------|")
            for i, a in enumerate(assumptions, 1):
                lines.append(
                    f"| {i} | {a.get('assumption', '-')} "
                    f"| {a.get('category', '-')} "
                    f"| {a.get('impact_if_wrong', '-')} |"
                )
        else:
            lines.append("_No key assumptions documented._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*Scenario analysis methodology aligned with NGFS and IEA frameworks.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".tornado-bar{display:flex;align-items:center;height:24px;margin:2px 0;}"
            ".tornado-low{background:#ef5350;height:100%;border-radius:4px 0 0 4px;}"
            ".tornado-high{background:#43a047;height:100%;border-radius:0 4px 4px 0;}"
            ".rank-1{background:#c8e6c9;font-weight:700;}"
            ".rank-badge{display:inline-block;background:#2e7d32;color:#fff;"
            "border-radius:50%;width:28px;height:28px;line-height:28px;text-align:center;"
            "font-weight:700;font-size:0.9em;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Scenario Comparison Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", [])
        recommended = data.get("recommended_pathway", {})
        mc = data.get("monte_carlo", {})
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Scenarios</div>'
            f'<div class="card-value">{len(scenarios)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Simulation Runs</div>'
            f'<div class="card-value">{_dec_comma(mc.get("runs", 0), 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Sensitivity Factors</div>'
            f'<div class="card-value">{len(data.get("sensitivity", []))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Recommended</div>'
            f'<div class="card-value">{recommended.get("name", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Decision Score</div>'
            f'<div class="card-value">{_dec(recommended.get("total_score", 0), 1)}</div>'
            f'<div class="card-unit">/ 10.0</div></div>\n'
            f'</div>'
        )

    def _html_scenario_definitions(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", [])
        rows = ""
        for i, sc in enumerate(scenarios, 1):
            rows += (
                f'<tr><td>{i}</td><td><strong>{sc.get("name", "-")}</strong></td>'
                f'<td>{sc.get("description", "-")}</td>'
                f'<td>{sc.get("target_year", "-")}</td>'
                f'<td>{sc.get("ambition", "-")}</td>'
                f'<td>{sc.get("key_lever", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Scenario Definitions</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Scenario</th><th>Description</th>'
            f'<th>Target Year</th><th>Ambition</th><th>Key Lever</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_monte_carlo(self, data: Dict[str, Any]) -> str:
        mc = data.get("monte_carlo", {})
        results = mc.get("results", [])
        rows = ""
        for r in results:
            rows += (
                f'<tr><td>{r.get("scenario", "-")}</td>'
                f'<td>{_dec_comma(r.get("p10", 0))}</td>'
                f'<td>{_dec_comma(r.get("p25", 0))}</td>'
                f'<td><strong>{_dec_comma(r.get("median", 0))}</strong></td>'
                f'<td>{_dec_comma(r.get("p75", 0))}</td>'
                f'<td>{_dec_comma(r.get("p90", 0))}</td>'
                f'<td>{_dec_comma(r.get("std_dev", 0))}</td></tr>\n'
            )
        return (
            f'<h2>3. Monte Carlo Simulation Results</h2>\n'
            f'<p><strong>Runs:</strong> {_dec_comma(mc.get("runs", 0), 0)} | '
            f'<strong>Distribution:</strong> {mc.get("distribution", "Normal")} | '
            f'<strong>Confidence:</strong> {mc.get("confidence_pct", 95)}%</p>\n'
            f'<table>\n'
            f'<tr><th>Scenario</th><th>P10</th><th>P25</th><th>Median</th>'
            f'<th>P75</th><th>P90</th><th>Std Dev</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scenario_comparison(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", [])
        base = data.get("base_scenario", "")
        rows = ""
        for sc in scenarios:
            delta = sc.get("delta", {})
            rows += (
                f'<tr><td>{sc.get("name", "-")}</td>'
                f'<td>{_dec_comma(delta.get("emissions_tco2e", 0))}</td>'
                f'<td>{_dec_comma(delta.get("cost_eur", 0), 0)}</td>'
                f'<td>{delta.get("risk", "-")}</td>'
                f'<td>{delta.get("timeline_years", 0)}</td>'
                f'<td>{delta.get("ambition", "-")}</td></tr>\n'
            )
        return (
            f'<h2>4. Scenario Comparison (Delta Analysis)</h2>\n'
            f'<p><strong>Base Scenario:</strong> {base if base else "First scenario listed"}</p>\n'
            f'<table>\n'
            f'<tr><th>Scenario</th><th>Emissions Delta (tCO2e)</th>'
            f'<th>Cost Delta (EUR)</th><th>Risk Delta</th>'
            f'<th>Timeline Delta (yrs)</th><th>Ambition Delta</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        sensitivity = data.get("sensitivity", [])
        sorted_sens = sorted(sensitivity, key=lambda x: abs(x.get("swing_tco2e", 0)), reverse=True)
        rows = ""
        max_swing = max((abs(s.get("swing_tco2e", 0)) for s in sorted_sens), default=1) or 1
        for i, sf in enumerate(sorted_sens, 1):
            swing = abs(sf.get("swing_tco2e", 0))
            bar_width = min(int(swing / max_swing * 100), 100)
            rows += (
                f'<tr><td><span class="rank-badge">{i}</span></td>'
                f'<td>{sf.get("factor", "-")}</td>'
                f'<td>{_dec_comma(sf.get("low_impact_tco2e", 0))}</td>'
                f'<td>{_dec_comma(sf.get("high_impact_tco2e", 0))}</td>'
                f'<td><div class="tornado-bar">'
                f'<div class="tornado-high" style="width:{bar_width}%"></div>'
                f'</div> {_dec_comma(sf.get("swing_tco2e", 0))}</td>'
                f'<td>{sf.get("most_affected_scenario", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. Sensitivity Analysis (Tornado Ranking)</h2>\n'
            f'<table>\n'
            f'<tr><th>Rank</th><th>Factor</th><th>Low Case (tCO2e)</th>'
            f'<th>High Case (tCO2e)</th><th>Swing</th><th>Most Affected</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_decision_matrix(self, data: Dict[str, Any]) -> str:
        matrix = data.get("decision_matrix", [])
        sorted_matrix = sorted(matrix, key=lambda x: x.get("total_score", 0), reverse=True)
        rows = ""
        for rank, dm in enumerate(sorted_matrix, 1):
            row_cls = ' class="rank-1"' if rank == 1 else ""
            rows += (
                f'<tr{row_cls}><td>{dm.get("scenario", "-")}</td>'
                f'<td>{_dec(dm.get("cost_score", 0), 1)}</td>'
                f'<td>{_dec(dm.get("risk_score", 0), 1)}</td>'
                f'<td>{_dec(dm.get("ambition_score", 0), 1)}</td>'
                f'<td><strong>{_dec(dm.get("total_score", 0), 1)}</strong></td>'
                f'<td><span class="rank-badge">{rank}</span></td></tr>\n'
            )
        weights = data.get("weights", {})
        weights_html = ""
        if weights:
            weights_html = (
                f'<p><strong>Weights:</strong> Cost={_dec(weights.get("cost", 0.33), 2)}, '
                f'Risk={_dec(weights.get("risk", 0.33), 2)}, '
                f'Ambition={_dec(weights.get("ambition", 0.34), 2)}</p>\n'
            )
        return (
            f'<h2>6. Decision Matrix</h2>\n'
            f'{weights_html}'
            f'<table>\n'
            f'<tr><th>Scenario</th><th>Cost Score</th><th>Risk Score</th>'
            f'<th>Ambition Score</th><th>Total</th><th>Rank</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_recommended_pathway(self, data: Dict[str, Any]) -> str:
        rec = data.get("recommended_pathway", {})
        actions_html = ""
        key_actions = rec.get("key_actions", [])
        if key_actions:
            actions_html = "<h3>Key Actions</h3>\n<ol>" + "".join(
                f"<li>{a}</li>" for a in key_actions
            ) + "</ol>\n"
        milestones = rec.get("milestones", [])
        ms_rows = ""
        for ms in milestones:
            ms_rows += (
                f'<tr><td>{ms.get("year", "-")}</td>'
                f'<td>{ms.get("milestone", "-")}</td>'
                f'<td>{ms.get("outcome", "-")}</td></tr>\n'
            )
        ms_html = ""
        if milestones:
            ms_html = (
                f'<h3>Implementation Milestones</h3>\n'
                f'<table><tr><th>Year</th><th>Milestone</th><th>Expected Outcome</th></tr>\n'
                f'{ms_rows}</table>\n'
            )
        return (
            f'<h2>7. Recommended Pathway</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Scenario</div>'
            f'<div class="card-value">{rec.get("name", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Score</div>'
            f'<div class="card-value">{_dec(rec.get("total_score", 0), 1)}</div>'
            f'<div class="card-unit">/ 10.0</div></div>\n'
            f'</div>\n'
            f'<p><strong>Rationale:</strong> {rec.get("rationale", "N/A")}</p>\n'
            f'{actions_html}{ms_html}'
        )

    def _html_key_assumptions(self, data: Dict[str, Any]) -> str:
        assumptions = data.get("assumptions", [])
        rows = ""
        for i, a in enumerate(assumptions, 1):
            rows += (
                f'<tr><td>{i}</td><td>{a.get("assumption", "-")}</td>'
                f'<td>{a.get("category", "-")}</td>'
                f'<td>{a.get("impact_if_wrong", "-")}</td></tr>\n'
            )
        return (
            f'<h2>8. Key Assumptions</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Assumption</th><th>Category</th>'
            f'<th>Impact if Wrong</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'Scenario methodology aligned with NGFS and IEA frameworks.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
