# -*- coding: utf-8 -*-
"""
TemperatureAlignmentReportTemplate - Portfolio temperature scoring for PACK-022.

Renders a temperature alignment report with portfolio-level temperature scores,
multiple aggregation methods (WATS/TETS/MOTS/EOTS), entity-level scores,
target classification, contribution analysis, temperature band distribution,
what-if scenarios, and methodology notes.

Sections:
    1. Portfolio Temperature Score
    2. Score by Aggregation Method (WATS/TETS/MOTS/EOTS)
    3. Entity-Level Scores
    4. Target Classification Matrix
    5. Contribution Analysis
    6. Temperature Band Distribution
    7. What-If Improvement Scenarios
    8. Methodology Notes

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
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 2) -> str:
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


def _pct_of(part: Any, total: Any) -> Decimal:
    try:
        p = Decimal(str(part))
        t = Decimal(str(total))
        if t == 0:
            return Decimal("0.00")
        return (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception:
        return Decimal("0.00")


def _temp_color(temp: float) -> str:
    """Return CSS class based on temperature score."""
    if temp <= 1.5:
        return "temp-15"
    if temp <= 2.0:
        return "temp-20"
    if temp <= 2.5:
        return "temp-25"
    return "temp-high"


class TemperatureAlignmentReportTemplate:
    """
    Portfolio temperature scoring report template.

    Calculates and presents implied temperature rise (ITR) scores using
    multiple aggregation methods, entity-level attribution, target
    classification, and what-if improvement scenarios.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_portfolio_score(data),
            self._md_aggregation_methods(data),
            self._md_entity_scores(data),
            self._md_target_classification(data),
            self._md_contribution_analysis(data),
            self._md_band_distribution(data),
            self._md_what_if(data),
            self._md_methodology(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_portfolio_score(data),
            self._html_aggregation_methods(data),
            self._html_entity_scores(data),
            self._html_target_classification(data),
            self._html_contribution_analysis(data),
            self._html_band_distribution(data),
            self._html_what_if(data),
            self._html_methodology(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Temperature Alignment Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "temperature_alignment_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "portfolio_score": data.get("portfolio_score", {}),
            "aggregation_methods": data.get("aggregation_methods", []),
            "entity_scores": data.get("entity_scores", []),
            "target_classification": data.get("target_classification", {}),
            "contribution_analysis": data.get("contribution_analysis", []),
            "band_distribution": data.get("band_distribution", []),
            "what_if_scenarios": data.get("what_if_scenarios", []),
            "methodology": data.get("methodology", {}),
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
            f"# Temperature Alignment Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_portfolio_score(self, data: Dict[str, Any]) -> str:
        score = data.get("portfolio_score", {})
        temp = score.get("temperature_c", 0)
        alignment = "1.5C aligned" if temp <= 1.5 else "2C aligned" if temp <= 2.0 else "Above 2C"
        return (
            "## 1. Portfolio Temperature Score\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Portfolio Temperature | {_dec(temp, 1)} C |\n"
            f"| Alignment Status | {alignment} |\n"
            f"| Primary Method | {score.get('primary_method', 'WATS')} |\n"
            f"| Scope Coverage | {score.get('scope_coverage', 'S1+S2')} |\n"
            f"| Entities in Scope | {score.get('entity_count', 0)} |\n"
            f"| Data Coverage | {_dec(score.get('data_coverage_pct', 0))}% |\n"
            f"| Benchmark (sector avg) | {_dec(score.get('benchmark_c', 0), 1)} C |"
        )

    def _md_aggregation_methods(self, data: Dict[str, Any]) -> str:
        methods = data.get("aggregation_methods", [])
        lines = [
            "## 2. Score by Aggregation Method\n",
            "| Method | Full Name | Temperature (C) | Description |",
            "|--------|-----------|:---------------:|-------------|",
        ]
        for m in methods:
            lines.append(
                f"| {m.get('code', '-')} | {m.get('name', '-')} "
                f"| {_dec(m.get('temperature_c', 0), 1)} "
                f"| {m.get('description', '-')} |"
            )
        if not methods:
            lines.append("| _No methods_ | - | - | - |")
        return "\n".join(lines)

    def _md_entity_scores(self, data: Dict[str, Any]) -> str:
        entities = data.get("entity_scores", [])
        lines = [
            "## 3. Entity-Level Scores\n",
            "| Entity | Sector | Weight (%) | Temperature (C) | Target Type | Coverage (%) |",
            "|--------|--------|:----------:|:---------------:|:-----------:|:------------:|",
        ]
        for e in entities:
            lines.append(
                f"| {e.get('name', '-')} | {e.get('sector', '-')} "
                f"| {_dec(e.get('weight_pct', 0))}% "
                f"| {_dec(e.get('temperature_c', 0), 1)} "
                f"| {e.get('target_type', '-')} "
                f"| {_dec(e.get('coverage_pct', 0))}% |"
            )
        if not entities:
            lines.append("| _No entity data_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_target_classification(self, data: Dict[str, Any]) -> str:
        tc = data.get("target_classification", {})
        matrix = tc.get("matrix", [])
        lines = [
            "## 4. Target Classification Matrix\n",
            f"**Total Entities:** {tc.get('total_entities', 0)}  \n"
            f"**With Targets:** {tc.get('with_targets', 0)}  \n"
            f"**Without Targets:** {tc.get('without_targets', 0)}\n",
            "| Classification | Count | Share (%) | Avg Temperature (C) |",
            "|---------------|:-----:|:---------:|:-------------------:|",
        ]
        total = tc.get("total_entities", 0)
        for row in matrix:
            cnt = row.get("count", 0)
            lines.append(
                f"| {row.get('classification', '-')} "
                f"| {cnt} "
                f"| {_dec(_pct_of(cnt, total))}% "
                f"| {_dec(row.get('avg_temperature_c', 0), 1)} |"
            )
        if not matrix:
            lines.append("| _No classification data_ | - | - | - |")
        return "\n".join(lines)

    def _md_contribution_analysis(self, data: Dict[str, Any]) -> str:
        contributions = data.get("contribution_analysis", [])
        lines = [
            "## 5. Contribution Analysis\n",
            "Entities driving the portfolio temperature score.\n",
            "| Entity | Temperature (C) | Weight (%) | Contribution to Score (C) | Cumulative (%) |",
            "|--------|:---------------:|:----------:|:-------------------------:|:--------------:|",
        ]
        for c in contributions:
            lines.append(
                f"| {c.get('name', '-')} "
                f"| {_dec(c.get('temperature_c', 0), 1)} "
                f"| {_dec(c.get('weight_pct', 0))}% "
                f"| {_dec(c.get('contribution_c', 0), 2)} "
                f"| {_dec(c.get('cumulative_pct', 0))}% |"
            )
        if not contributions:
            lines.append("| _No contribution data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_band_distribution(self, data: Dict[str, Any]) -> str:
        bands = data.get("band_distribution", [])
        lines = [
            "## 6. Temperature Band Distribution\n",
            "| Temperature Band | Entity Count | Weight (%) | Emissions Share (%) |",
            "|-----------------|:------------:|:----------:|:-------------------:|",
        ]
        for b in bands:
            lines.append(
                f"| {b.get('band', '-')} "
                f"| {b.get('count', 0)} "
                f"| {_dec(b.get('weight_pct', 0))}% "
                f"| {_dec(b.get('emissions_share_pct', 0))}% |"
            )
        if not bands:
            lines.append("| _No band data_ | - | - | - |")
        return "\n".join(lines)

    def _md_what_if(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("what_if_scenarios", [])
        lines = [
            "## 7. What-If Improvement Scenarios\n",
            "| Scenario | Description | Portfolio Impact (C) | New Score (C) | Feasibility |",
            "|----------|-------------|:--------------------:|:-------------:|:-----------:|",
        ]
        for sc in scenarios:
            lines.append(
                f"| {sc.get('name', '-')} "
                f"| {sc.get('description', '-')} "
                f"| {_dec(sc.get('impact_c', 0), 2)} "
                f"| {_dec(sc.get('new_score_c', 0), 1)} "
                f"| {sc.get('feasibility', '-')} |"
            )
        if not scenarios:
            lines.append("| _No scenarios_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        meth = data.get("methodology", {})
        return (
            "## 8. Methodology Notes\n\n"
            f"- **Framework:** {meth.get('framework', 'CDP-WWF Temperature Rating')}\n"
            f"- **Scenario:** {meth.get('scenario', 'IPCC SR1.5')}\n"
            f"- **Time Horizon:** {meth.get('time_horizon', 'Short-term (2030)')}\n"
            f"- **Scope:** {meth.get('scope', 'S1+S2')}\n"
            f"- **Default Score:** {_dec(meth.get('default_score_c', 3.2), 1)} C (for entities without targets)\n"
            f"- **Overshoot Allowed:** {'Yes' if meth.get('overshoot_allowed', False) else 'No'}\n"
            f"- **Data Sources:** {meth.get('data_sources', 'CDP, SBTi, UNFCCC')}\n"
            f"- **Last Updated:** {meth.get('last_updated', 'N/A')}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*Temperature scoring per CDP-WWF methodology.*"
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
            ".temp-15{color:#1b5e20;font-weight:700;}"
            ".temp-20{color:#33691e;font-weight:700;}"
            ".temp-25{color:#e65100;font-weight:700;}"
            ".temp-high{color:#c62828;font-weight:700;}"
            ".temp-badge{display:inline-block;padding:6px 16px;border-radius:20px;"
            "font-weight:700;font-size:1.1em;}"
            ".badge-15{background:#c8e6c9;color:#1b5e20;}"
            ".badge-20{background:#dcedc8;color:#33691e;}"
            ".badge-25{background:#ffe0b2;color:#e65100;}"
            ".badge-high{background:#ffcdd2;color:#c62828;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Temperature Alignment Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_portfolio_score(self, data: Dict[str, Any]) -> str:
        score = data.get("portfolio_score", {})
        temp = float(Decimal(str(score.get("temperature_c", 0))))
        if temp <= 1.5:
            badge_cls = "badge-15"
        elif temp <= 2.0:
            badge_cls = "badge-20"
        elif temp <= 2.5:
            badge_cls = "badge-25"
        else:
            badge_cls = "badge-high"
        alignment = "1.5C Aligned" if temp <= 1.5 else "2C Aligned" if temp <= 2.0 else "Above 2C"
        return (
            f'<h2>1. Portfolio Temperature Score</h2>\n'
            f'<div style="text-align:center;margin:20px 0;">'
            f'<span class="temp-badge {badge_cls}">{_dec(temp, 1)} C</span>'
            f'<p style="margin-top:8px;font-size:1.1em;">{alignment}</p></div>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Primary Method</div>'
            f'<div class="card-value">{score.get("primary_method", "WATS")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope</div>'
            f'<div class="card-value">{score.get("scope_coverage", "S1+S2")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Entities</div>'
            f'<div class="card-value">{score.get("entity_count", 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Data Coverage</div>'
            f'<div class="card-value">{_dec(score.get("data_coverage_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Sector Benchmark</div>'
            f'<div class="card-value">{_dec(score.get("benchmark_c", 0), 1)} C</div></div>\n'
            f'</div>'
        )

    def _html_aggregation_methods(self, data: Dict[str, Any]) -> str:
        methods = data.get("aggregation_methods", [])
        rows = ""
        for m in methods:
            temp = float(Decimal(str(m.get("temperature_c", 0))))
            cls = _temp_color(temp)
            rows += (
                f'<tr><td><strong>{m.get("code", "-")}</strong></td>'
                f'<td>{m.get("name", "-")}</td>'
                f'<td class="{cls}">{_dec(temp, 1)} C</td>'
                f'<td>{m.get("description", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Score by Aggregation Method</h2>\n'
            f'<table>\n'
            f'<tr><th>Method</th><th>Full Name</th><th>Temperature</th>'
            f'<th>Description</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_entity_scores(self, data: Dict[str, Any]) -> str:
        entities = data.get("entity_scores", [])
        rows = ""
        for e in entities:
            temp = float(Decimal(str(e.get("temperature_c", 0))))
            cls = _temp_color(temp)
            rows += (
                f'<tr><td><strong>{e.get("name", "-")}</strong></td>'
                f'<td>{e.get("sector", "-")}</td>'
                f'<td>{_dec(e.get("weight_pct", 0))}%</td>'
                f'<td class="{cls}">{_dec(temp, 1)} C</td>'
                f'<td>{e.get("target_type", "-")}</td>'
                f'<td>{_dec(e.get("coverage_pct", 0))}%</td></tr>\n'
            )
        return (
            f'<h2>3. Entity-Level Scores</h2>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Sector</th><th>Weight</th>'
            f'<th>Temperature</th><th>Target Type</th><th>Coverage</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_target_classification(self, data: Dict[str, Any]) -> str:
        tc = data.get("target_classification", {})
        matrix = tc.get("matrix", [])
        total = tc.get("total_entities", 0)
        rows = ""
        for row in matrix:
            cnt = row.get("count", 0)
            temp = float(Decimal(str(row.get("avg_temperature_c", 0))))
            cls = _temp_color(temp)
            rows += (
                f'<tr><td>{row.get("classification", "-")}</td>'
                f'<td>{cnt}</td>'
                f'<td>{_dec(_pct_of(cnt, total))}%</td>'
                f'<td class="{cls}">{_dec(temp, 1)} C</td></tr>\n'
            )
        return (
            f'<h2>4. Target Classification Matrix</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Entities</div>'
            f'<div class="card-value">{tc.get("total_entities", 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">With Targets</div>'
            f'<div class="card-value">{tc.get("with_targets", 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Without Targets</div>'
            f'<div class="card-value">{tc.get("without_targets", 0)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Classification</th><th>Count</th><th>Share</th>'
            f'<th>Avg Temperature</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_contribution_analysis(self, data: Dict[str, Any]) -> str:
        contributions = data.get("contribution_analysis", [])
        rows = ""
        for c in contributions:
            temp = float(Decimal(str(c.get("temperature_c", 0))))
            cls = _temp_color(temp)
            rows += (
                f'<tr><td><strong>{c.get("name", "-")}</strong></td>'
                f'<td class="{cls}">{_dec(temp, 1)} C</td>'
                f'<td>{_dec(c.get("weight_pct", 0))}%</td>'
                f'<td>{_dec(c.get("contribution_c", 0), 2)} C</td>'
                f'<td>{_dec(c.get("cumulative_pct", 0))}%</td></tr>\n'
            )
        return (
            f'<h2>5. Contribution Analysis</h2>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Temperature</th><th>Weight</th>'
            f'<th>Contribution</th><th>Cumulative</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_band_distribution(self, data: Dict[str, Any]) -> str:
        bands = data.get("band_distribution", [])
        rows = ""
        for b in bands:
            band_name = b.get("band", "-")
            if "1.5" in str(band_name):
                cls = "temp-15"
            elif "2.0" in str(band_name) or "2C" in str(band_name):
                cls = "temp-20"
            elif "2.5" in str(band_name):
                cls = "temp-25"
            else:
                cls = "temp-high"
            rows += (
                f'<tr><td class="{cls}">{band_name}</td>'
                f'<td>{b.get("count", 0)}</td>'
                f'<td>{_dec(b.get("weight_pct", 0))}%</td>'
                f'<td>{_dec(b.get("emissions_share_pct", 0))}%</td></tr>\n'
            )
        return (
            f'<h2>6. Temperature Band Distribution</h2>\n'
            f'<table>\n'
            f'<tr><th>Band</th><th>Entities</th><th>Weight</th>'
            f'<th>Emissions Share</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_what_if(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("what_if_scenarios", [])
        rows = ""
        for sc in scenarios:
            new_temp = float(Decimal(str(sc.get("new_score_c", 0))))
            cls = _temp_color(new_temp)
            rows += (
                f'<tr><td><strong>{sc.get("name", "-")}</strong></td>'
                f'<td>{sc.get("description", "-")}</td>'
                f'<td>{_dec(sc.get("impact_c", 0), 2)} C</td>'
                f'<td class="{cls}">{_dec(new_temp, 1)} C</td>'
                f'<td>{sc.get("feasibility", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. What-If Improvement Scenarios</h2>\n'
            f'<table>\n'
            f'<tr><th>Scenario</th><th>Description</th><th>Impact</th>'
            f'<th>New Score</th><th>Feasibility</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        meth = data.get("methodology", {})
        return (
            f'<h2>8. Methodology Notes</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Framework</td><td>{meth.get("framework", "CDP-WWF Temperature Rating")}</td></tr>\n'
            f'<tr><td>Scenario</td><td>{meth.get("scenario", "IPCC SR1.5")}</td></tr>\n'
            f'<tr><td>Time Horizon</td><td>{meth.get("time_horizon", "Short-term (2030)")}</td></tr>\n'
            f'<tr><td>Scope</td><td>{meth.get("scope", "S1+S2")}</td></tr>\n'
            f'<tr><td>Default Score</td><td>{_dec(meth.get("default_score_c", 3.2), 1)} C</td></tr>\n'
            f'<tr><td>Overshoot Allowed</td><td>{"Yes" if meth.get("overshoot_allowed", False) else "No"}</td></tr>\n'
            f'<tr><td>Data Sources</td><td>{meth.get("data_sources", "CDP, SBTi, UNFCCC")}</td></tr>\n'
            f'</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'Temperature scoring per CDP-WWF methodology.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
