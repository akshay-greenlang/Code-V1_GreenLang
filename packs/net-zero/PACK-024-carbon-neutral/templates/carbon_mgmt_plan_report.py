# -*- coding: utf-8 -*-
"""
CarbonMgmtPlanReportTemplate - Carbon management plan report for PACK-024.

Renders the PAS 2060 required carbon management plan documentation including
reduction targets, abatement actions, MACC analysis, residual emissions
forecast, implementation timeline, and annual review schedule.

Sections:
    1. Plan Overview (objectives, scope, period)
    2. Baseline Profile (emissions by scope)
    3. Reduction Targets (science-aligned targets)
    4. Abatement Actions (prioritized MACC)
    5. Residual Forecast (year-by-year projection)
    6. Implementation Timeline
    7. Review Schedule

Author: GreenLang Team
Version: 24.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_MODULE_VERSION = "24.0.0"


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
        neg = int_part.startswith("-")
        if neg:
            int_part = int_part[1:]
        fmt = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                fmt = "," + fmt
            fmt = ch + fmt
        if neg:
            fmt = "-" + fmt
        if len(parts) > 1:
            fmt += "." + parts[1]
        return fmt
    except Exception:
        return str(val)

def _pct(val: Any) -> str:
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)


class CarbonMgmtPlanReportTemplate:
    """Carbon management plan report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_plan_overview(data),
            self._md_baseline(data),
            self._md_targets(data),
            self._md_abatement(data),
            self._md_residual_forecast(data),
            self._md_timeline(data),
            self._md_review(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = (
            "body{font-family:'Segoe UI',sans-serif;padding:20px;background:#f0f4f0;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;border-left:4px solid #43a047;padding-left:12px;margin-top:35px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #c8e6c9;padding:10px;text-align:left;}"
            "th{background:#e8f5e9;color:#1b5e20;}"
            ".footer{margin-top:40px;border-top:2px solid #c8e6c9;padding-top:20px;"
            "color:#689f38;text-align:center;font-size:0.85em;}"
        )
        body = "\n".join([
            self._html_header(data),
            self._html_targets(data),
            self._html_abatement(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Carbon Management Plan</title>\n<style>\n{css}\n</style>\n</head>\n'
            f'<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result = {
            "template": "carbon_mgmt_plan_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "plan_period": data.get("plan_period", {}),
            "targets": data.get("targets", []),
            "actions": data.get("actions", []),
            "residual_forecast": data.get("residual_forecast", []),
            "timeline": data.get("timeline", {}),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Carbon Management Plan\n\n"
            f"**Organization:** {org}  \n"
            f"**Generated:** {ts}  \n"
            f"**Compliance:** PAS 2060:2014 Section 7\n\n---"
        )

    def _md_plan_overview(self, data: Dict[str, Any]) -> str:
        period = data.get("plan_period", {})
        return (
            f"## 1. Plan Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Plan Start | {period.get('start_year', 'N/A')} |\n"
            f"| Plan End | {period.get('end_year', 'N/A')} |\n"
            f"| Target Reduction | {_pct(data.get('target_reduction_pct', 0))} |\n"
            f"| Science Alignment | {data.get('science_alignment', 'PAS 2060')} |\n"
            f"| Total Actions | {len(data.get('actions', []))} |"
        )

    def _md_baseline(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        return (
            f"## 2. Baseline Profile\n\n"
            f"| Scope | Emissions (tCO2e) | % of Total |\n"
            f"|-------|------------------:|:----------:|\n"
            f"| Scope 1 | {_dec_comma(baseline.get('scope1_tco2e', 0), 0)} | {_pct(baseline.get('scope1_pct', 0))} |\n"
            f"| Scope 2 | {_dec_comma(baseline.get('scope2_tco2e', 0), 0)} | {_pct(baseline.get('scope2_pct', 0))} |\n"
            f"| Scope 3 | {_dec_comma(baseline.get('scope3_tco2e', 0), 0)} | {_pct(baseline.get('scope3_pct', 0))} |\n"
            f"| **Total** | **{_dec_comma(baseline.get('total_tco2e', 0), 0)}** | **100%** |"
        )

    def _md_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 3. Reduction Targets\n",
            "| # | Scope | Base Year | Target Year | Reduction | Annual Rate | Aligned |",
            "|---|-------|:---------:|:-----------:|:---------:|:-----------:|:-------:|",
        ]
        for i, t in enumerate(targets, 1):
            lines.append(
                f"| {i} | {t.get('scope', '-')} | {t.get('base_year', '-')} "
                f"| {t.get('target_year', '-')} | {_pct(t.get('reduction_pct', 0))} "
                f"| {_pct(t.get('annual_rate', 0))} | {t.get('aligned', '-')} |"
            )
        return "\n".join(lines)

    def _md_abatement(self, data: Dict[str, Any]) -> str:
        actions = data.get("actions", [])
        lines = [
            "## 4. Abatement Actions (MACC)\n",
            "| # | Action | Strategy | Priority | Potential (tCO2e) | Cost ($/tCO2e) | Payback (yr) |",
            "|---|--------|----------|:--------:|------------------:|:--------------:|:------------:|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('name', '-')} | {a.get('strategy', '-')} "
                f"| {a.get('priority', '-')} | {_dec_comma(a.get('potential_tco2e', 0), 0)} "
                f"| {_dec(a.get('cost_per_tco2e', 0))} | {_dec(a.get('payback_years', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_residual_forecast(self, data: Dict[str, Any]) -> str:
        forecast = data.get("residual_forecast", [])
        lines = [
            "## 5. Residual Emissions Forecast\n",
            "| Year | Emissions (tCO2e) | Reductions | Residual | Credits Needed |",
            "|:----:|------------------:|:----------:|:--------:|:--------------:|",
        ]
        for f in forecast:
            lines.append(
                f"| {f.get('year', '-')} "
                f"| {_dec_comma(f.get('emissions_tco2e', 0), 0)} "
                f"| {_dec_comma(f.get('reductions_tco2e', 0), 0)} "
                f"| {_dec_comma(f.get('residual_tco2e', 0), 0)} "
                f"| {_dec_comma(f.get('credits_needed_tco2e', 0), 0)} |"
            )
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        timeline = data.get("timeline", {})
        milestones = timeline.get("milestones", [])
        lines = ["## 6. Implementation Timeline\n"]
        if milestones:
            lines.append("| Year | Type | Description |")
            lines.append("|:----:|------|-------------|")
            for m in milestones:
                lines.append(f"| {m.get('year', '-')} | {m.get('type', '-')} | {m.get('description', '-')} |")
        else:
            lines.append("_Timeline milestones to be defined._")
        return "\n".join(lines)

    def _md_review(self, data: Dict[str, Any]) -> str:
        return (
            "## 7. Review Schedule\n\n"
            "- **Frequency:** Annual\n"
            "- **Triggers:** Significant emission change (>5%), restructuring, methodology update\n"
            "- **Responsible:** Sustainability Director\n"
            "- **Next Review:** End of current reporting year"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*  \n"
            f"*Plan compliant with PAS 2060:2014 Section 7 requirements.*"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Carbon Management Plan</h1>\n<p><strong>{org}</strong> | {ts}</p>'

    def _html_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        rows = ""
        for t in targets:
            rows += (
                f'<tr><td>{t.get("scope", "-")}</td><td>{t.get("target_year", "-")}</td>'
                f'<td>{_pct(t.get("reduction_pct", 0))}</td></tr>\n'
            )
        return f'<h2>Reduction Targets</h2>\n<table><tr><th>Scope</th><th>Target Year</th><th>Reduction</th></tr>\n{rows}</table>'

    def _html_abatement(self, data: Dict[str, Any]) -> str:
        actions = data.get("actions", [])
        rows = ""
        for a in actions:
            rows += (
                f'<tr><td>{a.get("name", "-")}</td><td>{a.get("strategy", "-")}</td>'
                f'<td>{_dec_comma(a.get("potential_tco2e", 0), 0)}</td>'
                f'<td>{_dec(a.get("cost_per_tco2e", 0))}</td></tr>\n'
            )
        return f'<h2>Abatement Actions</h2>\n<table><tr><th>Action</th><th>Strategy</th><th>Potential (tCO2e)</th><th>Cost ($/t)</th></tr>\n{rows}</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by PACK-024 on {ts}</div>'
