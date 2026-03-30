# -*- coding: utf-8 -*-
"""
NetZeroStrategyReportTemplate - Executive net-zero strategy report for PACK-021.

Renders a comprehensive net-zero strategy document suitable for board-level
review, investor disclosure, and internal planning. Covers baseline summary,
target setting, reduction pathway, offset strategy, implementation timeline,
investment requirements, risk assessment, and governance.

Sections:
    1. Executive Summary
    2. Organization Profile
    3. GHG Baseline Summary (scope split)
    4. Net Zero Targets (near-term + long-term)
    5. Reduction Pathway (top 10 abatement actions)
    6. Offset / Neutralization Strategy
    7. Implementation Timeline
    8. Investment Requirements
    9. Risk Assessment
   10. Governance Framework

Author: GreenLang Team
Version: 21.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"

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
        quantize_str = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _dec_comma(val: Any, places: int = 2) -> str:
    """Format a Decimal value with thousands separator."""
    try:
        d = Decimal(str(val))
        quantize_str = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
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

class NetZeroStrategyReportTemplate:
    """
    Executive-level net-zero strategy report template.

    Renders a comprehensive net-zero strategy document with baseline
    summary, targets, reduction pathway, offset strategy, implementation
    timeline, investment requirements, risk assessment, and governance
    framework across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize NetZeroStrategyReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render net-zero strategy report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_org_profile(data),
            self._md_baseline_summary(data),
            self._md_targets(data),
            self._md_reduction_pathway(data),
            self._md_offset_strategy(data),
            self._md_timeline(data),
            self._md_investment(data),
            self._md_risk_assessment(data),
            self._md_governance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render net-zero strategy report as self-contained HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_org_profile(data),
            self._html_baseline_summary(data),
            self._html_targets(data),
            self._html_reduction_pathway(data),
            self._html_offset_strategy(data),
            self._html_timeline(data),
            self._html_investment(data),
            self._html_risk_assessment(data),
            self._html_governance(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Net Zero Strategy Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render net-zero strategy report as structured JSON."""
        self.generated_at = utcnow()
        baseline = data.get("baseline", {})
        targets = data.get("targets", [])
        pathway = data.get("pathway", [])
        offsets = data.get("offsets", {})
        timeline_items = data.get("timeline", [])
        investment = data.get("investment", {})
        risks = data.get("risks", [])
        governance = data.get("governance", {})

        scope1 = Decimal(str(baseline.get("scope1_tco2e", 0)))
        scope2 = Decimal(str(baseline.get("scope2_tco2e", 0)))
        scope3 = Decimal(str(baseline.get("scope3_tco2e", 0)))
        total = scope1 + scope2 + scope3

        result: Dict[str, Any] = {
            "template": "net_zero_strategy_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "executive_summary": {
                "total_baseline_tco2e": str(total),
                "scope1_tco2e": str(scope1),
                "scope2_tco2e": str(scope2),
                "scope3_tco2e": str(scope3),
                "target_count": len(targets),
                "pathway_actions_count": len(pathway),
                "net_zero_target_year": data.get("net_zero_target_year", ""),
            },
            "baseline": {
                "base_year": baseline.get("base_year", ""),
                "scope1_tco2e": str(scope1),
                "scope2_tco2e": str(scope2),
                "scope3_tco2e": str(scope3),
                "total_tco2e": str(total),
                "scope_split_pct": {
                    "scope1": str(self._pct_of(scope1, total)),
                    "scope2": str(self._pct_of(scope2, total)),
                    "scope3": str(self._pct_of(scope3, total)),
                },
                "sources": baseline.get("sources", []),
            },
            "targets": targets,
            "pathway": pathway,
            "offsets": offsets,
            "timeline": timeline_items,
            "investment": investment,
            "risks": risks,
            "governance": governance,
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
            f"# Net Zero Strategy Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        scope1 = Decimal(str(baseline.get("scope1_tco2e", 0)))
        scope2 = Decimal(str(baseline.get("scope2_tco2e", 0)))
        scope3 = Decimal(str(baseline.get("scope3_tco2e", 0)))
        total = scope1 + scope2 + scope3
        target_year = data.get("net_zero_target_year", "N/A")
        targets = data.get("targets", [])
        pathway = data.get("pathway", [])
        return (
            f"## 1. Executive Summary\n\n"
            f"This document presents the net-zero strategy for "
            f"**{data.get('org_name', 'the organization')}**. "
            f"The organization has committed to achieving net-zero greenhouse gas "
            f"emissions by **{target_year}**.\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Baseline Emissions | {_dec_comma(total)} tCO2e |\n"
            f"| Scope 1 | {_dec_comma(scope1)} tCO2e |\n"
            f"| Scope 2 | {_dec_comma(scope2)} tCO2e |\n"
            f"| Scope 3 | {_dec_comma(scope3)} tCO2e |\n"
            f"| Net Zero Target Year | {target_year} |\n"
            f"| Targets Defined | {len(targets)} |\n"
            f"| Reduction Actions | {len(pathway)} |"
        )

    def _md_org_profile(self, data: Dict[str, Any]) -> str:
        profile = data.get("org_profile", {})
        return (
            f"## 2. Organization Profile\n\n"
            f"- **Name:** {data.get('org_name', '')}\n"
            f"- **Sector:** {profile.get('sector', 'N/A')}\n"
            f"- **Employees:** {profile.get('employees', 'N/A')}\n"
            f"- **Revenue:** {profile.get('revenue', 'N/A')}\n"
            f"- **Operating Countries:** {profile.get('countries', 'N/A')}\n"
            f"- **Reporting Standard:** {profile.get('reporting_standard', 'GHG Protocol')}\n"
            f"- **Consolidation Approach:** {profile.get('consolidation', 'Operational Control')}"
        )

    def _md_baseline_summary(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        scope1 = Decimal(str(baseline.get("scope1_tco2e", 0)))
        scope2 = Decimal(str(baseline.get("scope2_tco2e", 0)))
        scope3 = Decimal(str(baseline.get("scope3_tco2e", 0)))
        total = scope1 + scope2 + scope3
        base_year = baseline.get("base_year", "N/A")
        sources = baseline.get("sources", [])

        lines = [
            f"## 3. GHG Baseline Summary\n",
            f"**Base Year:** {base_year}\n",
            "| Scope | Emissions (tCO2e) | Share (%) |",
            "|-------|------------------:|----------:|",
            f"| Scope 1 (Direct) | {_dec_comma(scope1)} | {_dec(self._pct_of(scope1, total))}% |",
            f"| Scope 2 (Indirect Energy) | {_dec_comma(scope2)} | {_dec(self._pct_of(scope2, total))}% |",
            f"| Scope 3 (Value Chain) | {_dec_comma(scope3)} | {_dec(self._pct_of(scope3, total))}% |",
            f"| **Total** | **{_dec_comma(total)}** | **100.00%** |",
        ]

        if sources:
            lines.append("")
            lines.append("### Emission Sources")
            lines.append("")
            lines.append("| Source | Scope | Emissions (tCO2e) | Share (%) |")
            lines.append("|--------|-------|------------------:|----------:|")
            for src in sources:
                src_emissions = Decimal(str(src.get("emissions_tco2e", 0)))
                lines.append(
                    f"| {src.get('name', '-')} | {src.get('scope', '-')} "
                    f"| {_dec_comma(src_emissions)} "
                    f"| {_dec(self._pct_of(src_emissions, total))}% |"
                )

        return "\n".join(lines)

    def _md_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 4. Net Zero Targets\n",
            "| Target | Type | Scope | Base Year | Target Year | Reduction (%) | Pathway |",
            "|--------|------|-------|-----------|-------------|-------------:|---------|",
        ]
        for t in targets:
            lines.append(
                f"| {t.get('name', '-')} | {t.get('type', '-')} "
                f"| {t.get('scope_coverage', '-')} | {t.get('base_year', '-')} "
                f"| {t.get('target_year', '-')} | {_dec(t.get('reduction_pct', 0))}% "
                f"| {t.get('pathway', '-')} |"
            )
        if not targets:
            lines.append("| _No targets defined_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_reduction_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("pathway", [])
        lines = [
            "## 5. Reduction Pathway\n",
            "### Top Abatement Actions\n",
            "| # | Action | Scope | Abatement (tCO2e/yr) | Cost (EUR/tCO2e) | Timeline | TRL |",
            "|---|--------|-------|---------------------:|----------------:|----------|----:|",
        ]
        for i, action in enumerate(pathway[:10], 1):
            lines.append(
                f"| {i} | {action.get('name', '-')} | {action.get('scope', '-')} "
                f"| {_dec_comma(action.get('abatement_tco2e', 0))} "
                f"| {_dec_comma(action.get('cost_per_tco2e', 0))} "
                f"| {action.get('timeline', '-')} "
                f"| {action.get('trl', '-')} |"
            )
        if not pathway:
            lines.append("| - | _No actions defined_ | - | - | - | - | - |")
        total_abatement = sum(
            Decimal(str(a.get("abatement_tco2e", 0))) for a in pathway
        )
        lines.append(f"\n**Total Projected Abatement:** {_dec_comma(total_abatement)} tCO2e/yr")
        return "\n".join(lines)

    def _md_offset_strategy(self, data: Dict[str, Any]) -> str:
        offsets = data.get("offsets", {})
        residual = offsets.get("residual_emissions_tco2e", 0)
        strategy = offsets.get("strategy", "N/A")
        credits_list = offsets.get("credits", [])

        lines = [
            "## 6. Offset / Neutralization Strategy\n",
            f"**Residual Emissions:** {_dec_comma(residual)} tCO2e  \n"
            f"**Strategy:** {strategy}\n",
        ]
        if credits_list:
            lines.append("| Credit Type | Registry | Volume (tCO2e) | Vintage | Quality Score |")
            lines.append("|-------------|----------|---------------:|---------|-------------:|")
            for c in credits_list:
                lines.append(
                    f"| {c.get('type', '-')} | {c.get('registry', '-')} "
                    f"| {_dec_comma(c.get('volume_tco2e', 0))} "
                    f"| {c.get('vintage', '-')} "
                    f"| {_dec(c.get('quality_score', 0))} |"
                )
        else:
            lines.append("_No offset credits currently allocated._")

        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        timeline_items = data.get("timeline", [])
        lines = [
            "## 7. Implementation Timeline\n",
            "| Phase | Period | Key Milestones | Status |",
            "|-------|--------|----------------|--------|",
        ]
        for item in timeline_items:
            milestones = "; ".join(item.get("milestones", []))
            lines.append(
                f"| {item.get('phase', '-')} | {item.get('period', '-')} "
                f"| {milestones} | {item.get('status', '-')} |"
            )
        if not timeline_items:
            lines.append("| - | - | _No timeline defined_ | - |")
        return "\n".join(lines)

    def _md_investment(self, data: Dict[str, Any]) -> str:
        investment = data.get("investment", {})
        total_capex = investment.get("total_capex_eur", 0)
        total_opex = investment.get("total_opex_eur", 0)
        phases = investment.get("phases", [])

        lines = [
            "## 8. Investment Requirements\n",
            f"**Total CapEx:** EUR {_dec_comma(total_capex, 0)}  \n"
            f"**Total OpEx (annual):** EUR {_dec_comma(total_opex, 0)}\n",
        ]
        if phases:
            lines.append("| Phase | CapEx (EUR) | OpEx (EUR/yr) | Expected Savings (EUR/yr) | Payback |")
            lines.append("|-------|------------:|--------------:|-------------------------:|---------|")
            for p in phases:
                lines.append(
                    f"| {p.get('name', '-')} | {_dec_comma(p.get('capex_eur', 0), 0)} "
                    f"| {_dec_comma(p.get('opex_eur', 0), 0)} "
                    f"| {_dec_comma(p.get('savings_eur', 0), 0)} "
                    f"| {p.get('payback_years', '-')} yrs |"
                )
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 9. Risk Assessment\n",
            "| Risk | Category | Likelihood | Impact | Mitigation |",
            "|------|----------|-----------|--------|------------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('name', '-')} | {r.get('category', '-')} "
                f"| {r.get('likelihood', '-')} | {r.get('impact', '-')} "
                f"| {r.get('mitigation', '-')} |"
            )
        if not risks:
            lines.append("| _No risks identified_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_governance(self, data: Dict[str, Any]) -> str:
        governance = data.get("governance", {})
        board_oversight = governance.get("board_oversight", "N/A")
        committee = governance.get("committee", "N/A")
        reporting_freq = governance.get("reporting_frequency", "N/A")
        roles = governance.get("roles", [])

        lines = [
            "## 10. Governance Framework\n",
            f"- **Board Oversight:** {board_oversight}\n"
            f"- **Committee:** {committee}\n"
            f"- **Reporting Frequency:** {reporting_freq}\n",
        ]
        if roles:
            lines.append("| Role | Responsibility | Accountable To |")
            lines.append("|------|---------------|---------------|")
            for role in roles:
                lines.append(
                    f"| {role.get('title', '-')} | {role.get('responsibility', '-')} "
                    f"| {role.get('accountable_to', '-')} |"
                )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}*  \n"
            f"*This report is intended for internal planning and stakeholder engagement.*"
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
            ".progress-bar{background:#e0e0e0;border-radius:6px;height:20px;margin:4px 0;"
            "overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:6px;transition:width 0.3s;}"
            ".fill-green{background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".fill-amber{background:linear-gradient(90deg,#ff8f00,#ffb300);}"
            ".fill-red{background:linear-gradient(90deg,#e53935,#ef5350);}"
            ".risk-high{color:#c62828;font-weight:600;}"
            ".risk-medium{color:#e65100;font-weight:600;}"
            ".risk-low{color:#2e7d32;font-weight:600;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Net Zero Strategy Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        scope1 = Decimal(str(baseline.get("scope1_tco2e", 0)))
        scope2 = Decimal(str(baseline.get("scope2_tco2e", 0)))
        scope3 = Decimal(str(baseline.get("scope3_tco2e", 0)))
        total = scope1 + scope2 + scope3
        target_year = data.get("net_zero_target_year", "N/A")
        targets = data.get("targets", [])
        pathway = data.get("pathway", [])
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Baseline</div>'
            f'<div class="card-value">{_dec_comma(total)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Net Zero Year</div>'
            f'<div class="card-value">{target_year}</div></div>\n'
            f'  <div class="card"><div class="card-label">Targets</div>'
            f'<div class="card-value">{len(targets)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Actions</div>'
            f'<div class="card-value">{len(pathway)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(scope1)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 2</div>'
            f'<div class="card-value">{_dec_comma(scope2)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(scope3)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_org_profile(self, data: Dict[str, Any]) -> str:
        profile = data.get("org_profile", {})
        return (
            f'<h2>2. Organization Profile</h2>\n'
            f'<table>\n'
            f'<tr><th>Attribute</th><th>Value</th></tr>\n'
            f'<tr><td>Name</td><td>{data.get("org_name", "")}</td></tr>\n'
            f'<tr><td>Sector</td><td>{profile.get("sector", "N/A")}</td></tr>\n'
            f'<tr><td>Employees</td><td>{profile.get("employees", "N/A")}</td></tr>\n'
            f'<tr><td>Revenue</td><td>{profile.get("revenue", "N/A")}</td></tr>\n'
            f'<tr><td>Operating Countries</td><td>{profile.get("countries", "N/A")}</td></tr>\n'
            f'<tr><td>Reporting Standard</td><td>{profile.get("reporting_standard", "GHG Protocol")}</td></tr>\n'
            f'<tr><td>Consolidation</td><td>{profile.get("consolidation", "Operational Control")}</td></tr>\n'
            f'</table>'
        )

    def _html_baseline_summary(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        scope1 = Decimal(str(baseline.get("scope1_tco2e", 0)))
        scope2 = Decimal(str(baseline.get("scope2_tco2e", 0)))
        scope3 = Decimal(str(baseline.get("scope3_tco2e", 0)))
        total = scope1 + scope2 + scope3
        s1_pct = float(self._pct_of(scope1, total))
        s2_pct = float(self._pct_of(scope2, total))
        s3_pct = float(self._pct_of(scope3, total))

        scope_rows = (
            f'<tr><td>Scope 1 (Direct)</td><td>{_dec_comma(scope1)}</td>'
            f'<td>{_dec(self._pct_of(scope1, total))}%</td>'
            f'<td><div class="progress-bar"><div class="progress-fill fill-green" '
            f'style="width:{s1_pct}%"></div></div></td></tr>\n'
            f'<tr><td>Scope 2 (Indirect Energy)</td><td>{_dec_comma(scope2)}</td>'
            f'<td>{_dec(self._pct_of(scope2, total))}%</td>'
            f'<td><div class="progress-bar"><div class="progress-fill fill-amber" '
            f'style="width:{s2_pct}%"></div></div></td></tr>\n'
            f'<tr><td>Scope 3 (Value Chain)</td><td>{_dec_comma(scope3)}</td>'
            f'<td>{_dec(self._pct_of(scope3, total))}%</td>'
            f'<td><div class="progress-bar"><div class="progress-fill fill-red" '
            f'style="width:{s3_pct}%"></div></div></td></tr>\n'
            f'<tr><th>Total</th><th>{_dec_comma(total)}</th><th>100.00%</th><th></th></tr>'
        )

        return (
            f'<h2>3. GHG Baseline Summary</h2>\n'
            f'<p><strong>Base Year:</strong> {baseline.get("base_year", "N/A")}</p>\n'
            f'<table>\n'
            f'<tr><th>Scope</th><th>Emissions (tCO2e)</th><th>Share (%)</th><th>Distribution</th></tr>\n'
            f'{scope_rows}\n</table>'
        )

    def _html_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        rows = ""
        for t in targets:
            reduction = float(Decimal(str(t.get("reduction_pct", 0))))
            bar_color = "fill-green" if reduction >= 90 else "fill-amber" if reduction >= 50 else "fill-red"
            rows += (
                f'<tr><td>{t.get("name", "-")}</td><td>{t.get("type", "-")}</td>'
                f'<td>{t.get("scope_coverage", "-")}</td>'
                f'<td>{t.get("base_year", "-")}</td><td>{t.get("target_year", "-")}</td>'
                f'<td>{_dec(t.get("reduction_pct", 0))}%'
                f'<div class="progress-bar"><div class="progress-fill {bar_color}" '
                f'style="width:{min(reduction, 100)}%"></div></div></td>'
                f'<td>{t.get("pathway", "-")}</td></tr>\n'
            )
        return (
            f'<h2>4. Net Zero Targets</h2>\n'
            f'<table>\n'
            f'<tr><th>Target</th><th>Type</th><th>Scope</th><th>Base Year</th>'
            f'<th>Target Year</th><th>Reduction</th><th>Pathway</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_reduction_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("pathway", [])
        rows = ""
        for i, action in enumerate(pathway[:10], 1):
            rows += (
                f'<tr><td>{i}</td><td>{action.get("name", "-")}</td>'
                f'<td>{action.get("scope", "-")}</td>'
                f'<td>{_dec_comma(action.get("abatement_tco2e", 0))}</td>'
                f'<td>{_dec_comma(action.get("cost_per_tco2e", 0))}</td>'
                f'<td>{action.get("timeline", "-")}</td>'
                f'<td>{action.get("trl", "-")}</td></tr>\n'
            )
        total_abatement = sum(
            Decimal(str(a.get("abatement_tco2e", 0))) for a in pathway
        )
        return (
            f'<h2>5. Reduction Pathway</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Action</th><th>Scope</th><th>Abatement (tCO2e/yr)</th>'
            f'<th>Cost (EUR/tCO2e)</th><th>Timeline</th><th>TRL</th></tr>\n'
            f'{rows}</table>\n'
            f'<p><strong>Total Projected Abatement:</strong> {_dec_comma(total_abatement)} tCO2e/yr</p>'
        )

    def _html_offset_strategy(self, data: Dict[str, Any]) -> str:
        offsets = data.get("offsets", {})
        residual = offsets.get("residual_emissions_tco2e", 0)
        strategy = offsets.get("strategy", "N/A")
        credits_list = offsets.get("credits", [])
        rows = ""
        for c in credits_list:
            rows += (
                f'<tr><td>{c.get("type", "-")}</td><td>{c.get("registry", "-")}</td>'
                f'<td>{_dec_comma(c.get("volume_tco2e", 0))}</td>'
                f'<td>{c.get("vintage", "-")}</td>'
                f'<td>{_dec(c.get("quality_score", 0))}</td></tr>\n'
            )
        return (
            f'<h2>6. Offset / Neutralization Strategy</h2>\n'
            f'<p><strong>Residual Emissions:</strong> {_dec_comma(residual)} tCO2e | '
            f'<strong>Strategy:</strong> {strategy}</p>\n'
            f'<table>\n'
            f'<tr><th>Credit Type</th><th>Registry</th><th>Volume (tCO2e)</th>'
            f'<th>Vintage</th><th>Quality Score</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_timeline(self, data: Dict[str, Any]) -> str:
        timeline_items = data.get("timeline", [])
        rows = ""
        for item in timeline_items:
            milestones = "; ".join(item.get("milestones", []))
            status = item.get("status", "-")
            status_cls = (
                "risk-low" if status.lower() in ("completed", "on_track", "on track")
                else "risk-medium" if status.lower() in ("in_progress", "in progress")
                else "risk-high"
            )
            rows += (
                f'<tr><td>{item.get("phase", "-")}</td><td>{item.get("period", "-")}</td>'
                f'<td>{milestones}</td><td class="{status_cls}">{status}</td></tr>\n'
            )
        return (
            f'<h2>7. Implementation Timeline</h2>\n'
            f'<table>\n'
            f'<tr><th>Phase</th><th>Period</th><th>Key Milestones</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_investment(self, data: Dict[str, Any]) -> str:
        investment = data.get("investment", {})
        total_capex = investment.get("total_capex_eur", 0)
        total_opex = investment.get("total_opex_eur", 0)
        phases = investment.get("phases", [])
        rows = ""
        for p in phases:
            rows += (
                f'<tr><td>{p.get("name", "-")}</td>'
                f'<td>{_dec_comma(p.get("capex_eur", 0), 0)}</td>'
                f'<td>{_dec_comma(p.get("opex_eur", 0), 0)}</td>'
                f'<td>{_dec_comma(p.get("savings_eur", 0), 0)}</td>'
                f'<td>{p.get("payback_years", "-")} yrs</td></tr>\n'
            )
        return (
            f'<h2>8. Investment Requirements</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total CapEx</div>'
            f'<div class="card-value">EUR {_dec_comma(total_capex, 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Total OpEx (annual)</div>'
            f'<div class="card-value">EUR {_dec_comma(total_opex, 0)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Phase</th><th>CapEx (EUR)</th><th>OpEx (EUR/yr)</th>'
            f'<th>Savings (EUR/yr)</th><th>Payback</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = ""
        for r in risks:
            impact = r.get("impact", "Low")
            impact_cls = (
                "risk-high" if impact.lower() == "high"
                else "risk-medium" if impact.lower() == "medium"
                else "risk-low"
            )
            rows += (
                f'<tr><td>{r.get("name", "-")}</td><td>{r.get("category", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td class="{impact_cls}">{impact}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            f'<h2>9. Risk Assessment</h2>\n'
            f'<table>\n'
            f'<tr><th>Risk</th><th>Category</th><th>Likelihood</th>'
            f'<th>Impact</th><th>Mitigation</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_governance(self, data: Dict[str, Any]) -> str:
        governance = data.get("governance", {})
        roles = governance.get("roles", [])
        rows = ""
        for role in roles:
            rows += (
                f'<tr><td>{role.get("title", "-")}</td>'
                f'<td>{role.get("responsibility", "-")}</td>'
                f'<td>{role.get("accountable_to", "-")}</td></tr>\n'
            )
        return (
            f'<h2>10. Governance Framework</h2>\n'
            f'<table>\n'
            f'<tr><th>Attribute</th><th>Value</th></tr>\n'
            f'<tr><td>Board Oversight</td><td>{governance.get("board_oversight", "N/A")}</td></tr>\n'
            f'<tr><td>Committee</td><td>{governance.get("committee", "N/A")}</td></tr>\n'
            f'<tr><td>Reporting Frequency</td><td>{governance.get("reporting_frequency", "N/A")}</td></tr>\n'
            f'</table>\n'
            f'<h3>Roles & Responsibilities</h3>\n'
            f'<table>\n'
            f'<tr><th>Role</th><th>Responsibility</th><th>Accountable To</th></tr>\n'
            f'{rows}</table>\n'
            f'<div class="footer">Generated by GreenLang PACK-021 Net Zero Starter Pack</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pct_of(self, part: Decimal, total: Decimal) -> Decimal:
        """Calculate percentage of part relative to total."""
        if total == 0:
            return Decimal("0.00")
        return (part / total * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
