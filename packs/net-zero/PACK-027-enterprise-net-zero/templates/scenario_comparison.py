# -*- coding: utf-8 -*-
"""
ScenarioComparisonTemplate - 1.5C vs 2C vs BAU comparison for PACK-027.

Renders a board-ready scenario comparison document with fan charts
(P10-P90), tornado sensitivity analysis, investment requirement
comparison, probability of target achievement, and carbon budget
consumption analysis.

Sections:
    1. Executive Summary
    2. Scenario Definitions (1.5C, 2C, BAU)
    3. Emission Trajectories (P10/P25/P50/P75/P90 per scenario)
    4. Sensitivity Analysis (Sobol indices, tornado chart data)
    5. Investment Requirements by Scenario
    6. Probability of Target Achievement
    7. Carbon Budget Analysis
    8. Stranded Asset Risk
    9. Strategic Recommendations
   10. Citations

Output: Markdown, HTML, JSON
Provenance: SHA-256 hash on all outputs

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"
_TEMPLATE_ID = "scenario_comparison"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"
_SCENARIO_COLORS = {"1.5C": "#2e7d32", "2C": "#ef6c00", "BAU": "#c62828"}

SCENARIO_DEFINITIONS = [
    {
        "id": "aggressive_15c",
        "name": "Aggressive (1.5C)",
        "temp": "1.5C by 2100",
        "key_assumptions": "Rapid electrification, 100% RE by 2035, high carbon price ($150+/tCO2e by 2030)",
        "use_case": "Stretch target, investor communication",
    },
    {
        "id": "moderate_2c",
        "name": "Moderate (2C)",
        "temp": "Well-below 2C",
        "key_assumptions": "Steady transition, 80% RE by 2035, medium carbon price ($75-100/tCO2e by 2030)",
        "use_case": "Base case for planning",
    },
    {
        "id": "conservative_bau",
        "name": "Conservative (BAU)",
        "temp": "3-4C",
        "key_assumptions": "Current policies only, low carbon price ($25-50/tCO2e by 2030)",
        "use_case": "Risk scenario, stranded asset analysis",
    },
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)


def _pct(val: Any) -> str:
    try:
        return str(round(float(val), 1)) + "%"
    except Exception:
        return str(val)


def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default


class ScenarioComparisonTemplate:
    """
    1.5C vs 2C vs BAU scenario comparison template.

    Board-ready strategic decision document with Monte Carlo results,
    sensitivity analysis, investment comparison, and carbon budget.
    Supports Markdown, HTML, and JSON output.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_scenario_definitions(data),
            self._md_emission_trajectories(data),
            self._md_sensitivity_analysis(data),
            self._md_investment_requirements(data),
            self._md_probability_achievement(data),
            self._md_carbon_budget(data),
            self._md_stranded_assets(data),
            self._md_recommendations(data),
            self._md_citations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(s for s in sections if s)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- SHA-256 Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:1100px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".scenario-card{{padding:16px;border-radius:8px;margin:8px 0;border-left:4px solid;}}"
            f".s-15c{{border-color:{_SCENARIO_COLORS['1.5C']};background:#e8f5e9;}}"
            f".s-2c{{border-color:{_SCENARIO_COLORS['2C']};background:#fff3e0;}}"
            f".s-bau{{border-color:{_SCENARIO_COLORS['BAU']};background:#ffebee;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        body = (
            f'<h1>Scenario Comparison</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | 1.5C vs 2C vs BAU | '
            f'Monte Carlo: {_dec_comma(data.get("monte_carlo_runs", 10000))} runs</p>\n'
            f'<div class="scenario-card s-15c"><strong>1.5C (Aggressive)</strong>: '
            f'Rapid decarbonization, net-zero by 2050</div>\n'
            f'<div class="scenario-card s-2c"><strong>2C (Moderate)</strong>: '
            f'Orderly transition, significant reduction by 2050</div>\n'
            f'<div class="scenario-card s-bau"><strong>BAU (Conservative)</strong>: '
            f'Current policies, limited additional action</div>\n'
            f'{self._html_comparison_table(data)}\n'
            f'<div class="footer">Generated by GreenLang PACK-027 | Scenario Analysis | SHA-256</div>'
        )
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>Scenario Comparison</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        scenarios = data.get("scenarios", {})
        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": data.get("org_name", ""),
            "base_year": data.get("base_year", ""),
            "base_year_tco2e": data.get("base_year_tco2e", 0),
            "monte_carlo_runs": data.get("monte_carlo_runs", 10000),
            "scenarios": {
                sid["id"]: {
                    "name": sid["name"],
                    "temperature": sid["temp"],
                    "trajectories": scenarios.get(sid["id"], {}).get("trajectories", []),
                    "p10_2030": scenarios.get(sid["id"], {}).get("p10_2030", 0),
                    "p50_2030": scenarios.get(sid["id"], {}).get("p50_2030", 0),
                    "p90_2030": scenarios.get(sid["id"], {}).get("p90_2030", 0),
                    "p50_2050": scenarios.get(sid["id"], {}).get("p50_2050", 0),
                    "investment_p50": scenarios.get(sid["id"], {}).get("investment_p50", 0),
                    "investment_p90": scenarios.get(sid["id"], {}).get("investment_p90", 0),
                    "prob_target_achievement": scenarios.get(sid["id"], {}).get("prob_target", 0),
                }
                for sid in SCENARIO_DEFINITIONS
            },
            "sensitivity": data.get("sensitivity", {}),
            "carbon_budget": data.get("carbon_budget", {}),
            "stranded_assets": data.get("stranded_assets", {}),
            "recommendations": data.get("recommendations", []),
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Scenario Comparison Report\n\n"
            f"## {data.get('org_name', 'Enterprise')} -- 1.5C vs 2C vs BAU\n\n"
            f"**Monte Carlo Runs:** {_dec_comma(data.get('monte_carlo_runs', 10000))}  \n"
            f"**Base Year:** {data.get('base_year', '')} "
            f"({_dec_comma(data.get('base_year_tco2e', 0))} tCO2e)  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | 1.5C (Aggressive) | 2C (Moderate) | BAU (Conservative) |",
            "|--------|:-----------------:|:-------------:|:------------------:|",
        ]
        for metric_key, label in [
            ("p50_2030", "2030 Emissions (P50, tCO2e)"),
            ("p50_2050", "2050 Emissions (P50, tCO2e)"),
            ("investment_p50", "Investment Required (P50)"),
            ("prob_target", "Probability of SBTi Achievement"),
        ]:
            vals = []
            for sd in SCENARIO_DEFINITIONS:
                sc = scenarios.get(sd["id"], {})
                v = sc.get(metric_key, 0)
                if "investment" in metric_key:
                    vals.append(f"${_dec_comma(v)}")
                elif "prob" in metric_key:
                    vals.append(_pct(v))
                else:
                    vals.append(_dec_comma(v))
            lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")
        return "\n".join(lines)

    def _md_scenario_definitions(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 2. Scenario Definitions\n",
            "| Scenario | Temperature | Key Assumptions | Use Case |",
            "|----------|:----------:|-----------------|----------|",
        ]
        for sd in SCENARIO_DEFINITIONS:
            lines.append(
                f"| {sd['name']} | {sd['temp']} | {sd['key_assumptions']} | {sd['use_case']} |"
            )
        return "\n".join(lines)

    def _md_emission_trajectories(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        lines = [
            "## 3. Emission Trajectories\n",
            "### Confidence Bands (P10 / P50 / P90)\n",
        ]
        for sd in SCENARIO_DEFINITIONS:
            sc = scenarios.get(sd["id"], {})
            trajectories = sc.get("trajectories", [])
            if trajectories:
                lines.append(f"\n#### {sd['name']}\n")
                lines.append("| Year | P10 (tCO2e) | P25 | P50 | P75 | P90 |")
                lines.append("|------|------------:|----:|----:|----:|----:|")
                for yr in trajectories:
                    lines.append(
                        f"| {yr.get('year', '')} "
                        f"| {_dec_comma(yr.get('p10', 0))} "
                        f"| {_dec_comma(yr.get('p25', 0))} "
                        f"| {_dec_comma(yr.get('p50', 0))} "
                        f"| {_dec_comma(yr.get('p75', 0))} "
                        f"| {_dec_comma(yr.get('p90', 0))} |"
                    )
            else:
                lines.append(f"\n#### {sd['name']}\n")
                lines.append(
                    f"P50 2030: {_dec_comma(sc.get('p50_2030', 0))} tCO2e | "
                    f"P50 2050: {_dec_comma(sc.get('p50_2050', 0))} tCO2e"
                )
        return "\n".join(lines)

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        sensitivity = data.get("sensitivity", {})
        drivers = sensitivity.get("top_drivers", [])
        if not drivers:
            return "## 4. Sensitivity Analysis\n\nSensitivity data not yet computed."
        lines = [
            "## 4. Sensitivity Analysis (Tornado Chart Data)\n",
            "| Rank | Driver | Sobol Index | Low Impact (tCO2e) | High Impact (tCO2e) |",
            "|:----:|--------|:-----------:|-------------------:|--------------------:|",
        ]
        for i, d in enumerate(drivers[:10], 1):
            lines.append(
                f"| {i} | {d.get('name', '')} | {d.get('sobol_index', '-')} "
                f"| {_dec_comma(d.get('low_impact', 0))} "
                f"| {_dec_comma(d.get('high_impact', 0))} |"
            )
        return "\n".join(lines)

    def _md_investment_requirements(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        currency = data.get("currency", "USD")
        lines = [
            "## 5. Investment Requirements by Scenario\n",
            f"| Scenario | Cumulative Investment (P50) | Cumulative Investment (P90) | Annual Average |",
            f"|----------|---------------------------:|---------------------------:|---------------:|",
        ]
        for sd in SCENARIO_DEFINITIONS:
            sc = scenarios.get(sd["id"], {})
            lines.append(
                f"| {sd['name']} "
                f"| {currency} {_dec_comma(sc.get('investment_p50', 0))} "
                f"| {currency} {_dec_comma(sc.get('investment_p90', 0))} "
                f"| {currency} {_dec_comma(sc.get('investment_annual', 0))} |"
            )
        return "\n".join(lines)

    def _md_probability_achievement(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        lines = [
            "## 6. Probability of Target Achievement\n",
            "| Scenario | P(SBTi Near-Term) | P(SBTi Net-Zero) | P(1.5C Budget) |",
            "|----------|:-----------------:|:----------------:|:--------------:|",
        ]
        for sd in SCENARIO_DEFINITIONS:
            sc = scenarios.get(sd["id"], {})
            lines.append(
                f"| {sd['name']} "
                f"| {_pct(sc.get('prob_target', 0))} "
                f"| {_pct(sc.get('prob_net_zero', 0))} "
                f"| {_pct(sc.get('prob_15c_budget', 0))} |"
            )
        return "\n".join(lines)

    def _md_carbon_budget(self, data: Dict[str, Any]) -> str:
        cb = data.get("carbon_budget", {})
        return (
            f"## 7. Carbon Budget Analysis\n\n"
            f"| Metric | Value |\n"
            f"|--------|------:|\n"
            f"| Remaining 1.5C budget (global) | {_dec_comma(cb.get('global_budget_gtco2', 0))} GtCO2 |\n"
            f"| Company fair share budget | {_dec_comma(cb.get('company_budget_tco2e', 0))} tCO2e |\n"
            f"| Budget consumed to date | {_pct(cb.get('consumed_pct', 0))} |\n"
            f"| Years remaining at current rate | {cb.get('years_remaining', 'N/A')} |\n"
            f"| Budget exhaustion year (BAU) | {cb.get('exhaustion_year_bau', 'N/A')} |\n"
            f"| Budget exhaustion year (1.5C) | {cb.get('exhaustion_year_15c', 'N/A')} |"
        )

    def _md_stranded_assets(self, data: Dict[str, Any]) -> str:
        sa = data.get("stranded_assets", {})
        assets = sa.get("at_risk", [])
        if not assets:
            return "## 8. Stranded Asset Risk\n\nNo significant stranded asset risk identified."
        lines = [
            "## 8. Stranded Asset Risk\n",
            "| Asset Category | Book Value | Stranding Risk (1.5C) | Stranding Risk (2C) | Stranding Year |",
            "|----------------|----------:|:---------------------:|:-------------------:|:--------------:|",
        ]
        for a in assets:
            lines.append(
                f"| {a.get('category', '')} "
                f"| {data.get('currency', '$')}{_dec_comma(a.get('book_value', 0))} "
                f"| {a.get('risk_15c', 'Low')} "
                f"| {a.get('risk_2c', 'Low')} "
                f"| {a.get('stranding_year', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [
            "Adopt the 2C (Moderate) scenario as base case for strategic planning",
            "Set internal carbon price at $100/tCO2e with annual escalation",
            "Prioritize top 5 sensitivity drivers for emission reduction",
            "Allocate CapEx for P50 investment requirement of the chosen scenario",
            "Conduct annual scenario refresh with updated parameters",
        ])
        lines = ["## 9. Strategic Recommendations\n"]
        for i, r in enumerate(recs, 1):
            lines.append(f"{i}. {r}")
        return "\n".join(lines)

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "SCEN-001", "source": "IPCC AR6 WG III Mitigation Pathways", "year": "2022"},
            {"ref": "SCEN-002", "source": "IEA Net Zero Roadmap", "year": "2023"},
            {"ref": "SCEN-003", "source": "NGFS Climate Scenarios", "year": "2024"},
        ])
        lines = ["## 10. Citations\n"]
        for c in citations:
            lines.append(f"- [{c.get('ref', '')}] {c.get('source', '')} ({c.get('year', '')})")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*Monte Carlo scenario analysis. SHA-256 provenance.*"
        )

    def _html_comparison_table(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        rows = ""
        for metric, label in [
            ("p50_2030", "2030 Emissions (P50)"),
            ("p50_2050", "2050 Emissions (P50)"),
            ("prob_target", "P(SBTi Achievement)"),
        ]:
            vals = []
            for sd in SCENARIO_DEFINITIONS:
                sc = scenarios.get(sd["id"], {})
                v = sc.get(metric, 0)
                vals.append(_pct(v) if "prob" in metric else _dec_comma(v))
            rows += f'<tr><td>{label}</td><td>{vals[0]}</td><td>{vals[1]}</td><td>{vals[2]}</td></tr>\n'
        return (
            f'<h2>Scenario Comparison</h2>\n'
            f'<table><tr><th>Metric</th><th>1.5C</th><th>2C</th><th>BAU</th></tr>\n{rows}</table>'
        )
