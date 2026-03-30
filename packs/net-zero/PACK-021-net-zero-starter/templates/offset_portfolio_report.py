# -*- coding: utf-8 -*-
"""
OffsetPortfolioReportTemplate - Carbon credit portfolio quality report for PACK-021.

Renders a carbon credit portfolio report covering residual emissions budget,
portfolio composition, quality assessment, SBTi compliance (compensation vs
neutralization), VCMI claims alignment, credit retirement schedule, cost
projections, and recommendations.

Sections:
    1. Residual Emissions Budget
    2. Portfolio Composition (by credit type)
    3. Quality Assessment (dimension scores)
    4. SBTi Compliance (compensation vs neutralization)
    5. VCMI Claims Alignment
    6. Credit Retirement Schedule
    7. Cost Projection
    8. Recommendations

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

_QUALITY_DIMENSIONS: List[Dict[str, str]] = [
    {"id": "additionality", "name": "Additionality", "description": "Would reductions occur without carbon finance?"},
    {"id": "permanence", "name": "Permanence", "description": "How long will carbon be stored?"},
    {"id": "leakage", "name": "Leakage Risk", "description": "Risk of displacement to other areas"},
    {"id": "measurement", "name": "MRV Robustness", "description": "Measurement, reporting, verification quality"},
    {"id": "co_benefits", "name": "Co-Benefits", "description": "Social, biodiversity, SDG alignment"},
    {"id": "registry", "name": "Registry Standard", "description": "Certification standard quality"},
]

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
    p = Decimal(str(part))
    t = Decimal(str(total))
    if t == 0:
        return Decimal("0.00")
    return (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

class OffsetPortfolioReportTemplate:
    """
    Carbon credit portfolio and quality report template.

    Renders a comprehensive portfolio report covering residual emissions,
    credit composition, quality assessment, SBTi and VCMI compliance,
    retirement schedules, cost projections, and recommendations.

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
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_residual_budget(data),
            self._md_portfolio_composition(data),
            self._md_quality_assessment(data),
            self._md_sbti_compliance(data),
            self._md_vcmi_alignment(data),
            self._md_retirement_schedule(data),
            self._md_cost_projection(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_residual_budget(data),
            self._html_portfolio_composition(data),
            self._html_quality_assessment(data),
            self._html_sbti_compliance(data),
            self._html_vcmi_alignment(data),
            self._html_retirement_schedule(data),
            self._html_cost_projection(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Offset Portfolio Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        credits_list = data.get("credits", [])
        total_volume = sum(Decimal(str(c.get("volume_tco2e", 0))) for c in credits_list)
        removal_volume = sum(
            Decimal(str(c.get("volume_tco2e", 0)))
            for c in credits_list if c.get("category") == "removal"
        )

        result: Dict[str, Any] = {
            "template": "offset_portfolio_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "residual_budget": {
                "baseline_tco2e": str(Decimal(str(data.get("baseline_tco2e", 0)))),
                "abatement_tco2e": str(Decimal(str(data.get("abatement_tco2e", 0)))),
                "residual_tco2e": str(Decimal(str(data.get("residual_tco2e", 0)))),
            },
            "portfolio": {
                "total_volume_tco2e": str(total_volume),
                "removal_volume_tco2e": str(removal_volume),
                "removal_share_pct": str(_pct_of(removal_volume, total_volume)),
                "credits": credits_list,
            },
            "quality_assessment": data.get("quality_assessment", {}),
            "sbti_compliance": data.get("sbti_compliance", {}),
            "vcmi_alignment": data.get("vcmi_alignment", {}),
            "retirement_schedule": data.get("retirement_schedule", []),
            "cost_projection": data.get("cost_projection", []),
            "recommendations": data.get("recommendations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Carbon Credit Portfolio Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_residual_budget(self, data: Dict[str, Any]) -> str:
        baseline = Decimal(str(data.get("baseline_tco2e", 0)))
        abatement = Decimal(str(data.get("abatement_tco2e", 0)))
        residual = Decimal(str(data.get("residual_tco2e", 0)))
        residual_pct = _pct_of(residual, baseline) if baseline > 0 else Decimal("0")
        return (
            f"## 1. Residual Emissions Budget\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Baseline Emissions | {_dec_comma(baseline)} tCO2e |\n"
            f"| Planned Abatement | {_dec_comma(abatement)} tCO2e |\n"
            f"| Residual Emissions | {_dec_comma(residual)} tCO2e |\n"
            f"| Residual Share | {_dec(residual_pct)}% of baseline |\n\n"
            f"*SBTi requires residual emissions to be no more than 10% of baseline "
            f"before offsetting.*"
        )

    def _md_portfolio_composition(self, data: Dict[str, Any]) -> str:
        credits_list = data.get("credits", [])
        total_volume = sum(Decimal(str(c.get("volume_tco2e", 0))) for c in credits_list)
        lines = [
            "## 2. Portfolio Composition\n",
            "| # | Credit Type | Category | Registry | Volume (tCO2e) | Share (%) | Vintage | Quality |",
            "|---|-----------|----------|----------|---------------:|----------:|---------|--------:|",
        ]
        for i, c in enumerate(credits_list, 1):
            volume = Decimal(str(c.get("volume_tco2e", 0)))
            lines.append(
                f"| {i} | {c.get('type', '-')} | {c.get('category', '-')} "
                f"| {c.get('registry', '-')} | {_dec_comma(volume)} "
                f"| {_dec(_pct_of(volume, total_volume))}% "
                f"| {c.get('vintage', '-')} | {_dec(c.get('quality_score', 0))} |"
            )
        if not credits_list:
            lines.append("| - | _No credits allocated_ | - | - | - | - | - | - |")
        lines.append(f"\n**Total Portfolio:** {_dec_comma(total_volume)} tCO2e")
        return "\n".join(lines)

    def _md_quality_assessment(self, data: Dict[str, Any]) -> str:
        qa = data.get("quality_assessment", {})
        overall = qa.get("overall_score", 0)
        dimension_scores = qa.get("dimension_scores", {})
        lines = [
            "## 3. Quality Assessment\n",
            f"**Overall Quality Score:** {_dec(overall)} / 100\n",
            "| Dimension | Score | Rating | Description |",
            "|-----------|------:|--------|------------|",
        ]
        for dim in _QUALITY_DIMENSIONS:
            score = dimension_scores.get(dim["id"], 0)
            rating = "High" if score >= 80 else "Medium" if score >= 50 else "Low"
            lines.append(
                f"| {dim['name']} | {_dec(score)} | {rating} | {dim['description']} |"
            )
        return "\n".join(lines)

    def _md_sbti_compliance(self, data: Dict[str, Any]) -> str:
        sbti = data.get("sbti_compliance", {})
        credits_list = data.get("credits", [])
        removal_volume = sum(
            Decimal(str(c.get("volume_tco2e", 0)))
            for c in credits_list if c.get("category") == "removal"
        )
        avoidance_volume = sum(
            Decimal(str(c.get("volume_tco2e", 0)))
            for c in credits_list if c.get("category") != "removal"
        )
        return (
            "## 4. SBTi Compliance\n\n"
            "### Neutralization vs Compensation\n\n"
            "| Category | Volume (tCO2e) | SBTi Role |\n"
            "|----------|---------------:|----------|\n"
            f"| Carbon Removals | {_dec_comma(removal_volume)} | Neutralization (required for net-zero) |\n"
            f"| Avoidance/Reduction Credits | {_dec_comma(avoidance_volume)} | Compensation (beyond value chain mitigation) |\n\n"
            f"- **Neutralization Compliant:** {sbti.get('neutralization_compliant', 'N/A')}\n"
            f"- **BVCM Strategy:** {sbti.get('bvcm_strategy', 'N/A')}\n"
            f"- **Abatement First Verified:** {sbti.get('abatement_first', 'N/A')}"
        )

    def _md_vcmi_alignment(self, data: Dict[str, Any]) -> str:
        vcmi = data.get("vcmi_alignment", {})
        claim_level = vcmi.get("claim_level", "N/A")
        prerequisites = vcmi.get("prerequisites_met", [])
        gaps = vcmi.get("gaps", [])
        lines = [
            "## 5. VCMI Claims Alignment\n",
            f"**Claim Level:** {claim_level}\n",
            "### Prerequisites Met\n",
        ]
        if prerequisites:
            for p in prerequisites:
                lines.append(f"- [x] {p}")
        else:
            lines.append("- _No prerequisites assessed_")
        if gaps:
            lines.append("\n### Gaps to Address\n")
            for g in gaps:
                lines.append(f"- [ ] {g}")
        return "\n".join(lines)

    def _md_retirement_schedule(self, data: Dict[str, Any]) -> str:
        schedule = data.get("retirement_schedule", [])
        lines = [
            "## 6. Credit Retirement Schedule\n",
            "| Year | Volume to Retire (tCO2e) | Source | Registry | Serial Range |",
            "|------|-------------------------:|--------|----------|-------------|",
        ]
        for item in schedule:
            lines.append(
                f"| {item.get('year', '-')} "
                f"| {_dec_comma(item.get('volume_tco2e', 0))} "
                f"| {item.get('source', '-')} "
                f"| {item.get('registry', '-')} "
                f"| {item.get('serial_range', '-')} |"
            )
        if not schedule:
            lines.append("| - | - | - | - | - |")
        return "\n".join(lines)

    def _md_cost_projection(self, data: Dict[str, Any]) -> str:
        projection = data.get("cost_projection", [])
        lines = [
            "## 7. Cost Projection\n",
            "| Year | Volume (tCO2e) | Avg Price (EUR/tCO2e) | Total Cost (EUR) | Type Mix |",
            "|------|---------------:|---------------------:|----------------:|----------|",
        ]
        total_cost = Decimal("0")
        for yr in projection:
            cost = Decimal(str(yr.get("total_cost_eur", 0)))
            total_cost += cost
            lines.append(
                f"| {yr.get('year', '-')} "
                f"| {_dec_comma(yr.get('volume_tco2e', 0))} "
                f"| {_dec_comma(yr.get('avg_price_eur', 0))} "
                f"| {_dec_comma(cost, 0)} "
                f"| {yr.get('type_mix', '-')} |"
            )
        if projection:
            lines.append(f"\n**Total Projected Cost:** EUR {_dec_comma(total_cost, 0)}")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = ["## 8. Recommendations\n"]
        if recs:
            for i, rec in enumerate(recs, 1):
                lines.append(f"### {i}. {rec.get('title', '')}\n")
                lines.append(f"{rec.get('description', '')}\n")
                if rec.get("actions"):
                    for action in rec["actions"]:
                        lines.append(f"  - {action}")
                lines.append("")
        else:
            lines.append("_Portfolio meets current quality and compliance standards._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}*  \n"
            f"*SBTi Corporate Net-Zero Standard and VCMI Claims Code applied.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f5f7f5;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "h3{color:#388e3c;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".gauge{width:100%;height:24px;background:#e0e0e0;border-radius:12px;"
            "overflow:hidden;margin:6px 0;}"
            ".gauge-fill{height:100%;border-radius:12px;transition:width 0.3s;}"
            ".gauge-green{background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".gauge-amber{background:linear-gradient(90deg,#ff8f00,#ffb300);}"
            ".gauge-red{background:linear-gradient(90deg,#e53935,#ef5350);}"
            ".removal{background:#e8f5e9;}"
            ".avoidance{background:#fff9c4;}"
            ".quality-high{color:#1b5e20;font-weight:600;}"
            ".quality-medium{color:#e65100;font-weight:600;}"
            ".quality-low{color:#c62828;font-weight:600;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        credits_list = data.get("credits", [])
        total_volume = sum(Decimal(str(c.get("volume_tco2e", 0))) for c in credits_list)
        residual = Decimal(str(data.get("residual_tco2e", 0)))
        return (
            f'<h1>Carbon Credit Portfolio Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Generated:</strong> {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Residual Emissions</div>'
            f'<div class="card-value">{_dec_comma(residual)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Portfolio Volume</div>'
            f'<div class="card-value">{_dec_comma(total_volume)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Credits Count</div>'
            f'<div class="card-value">{len(credits_list)}</div></div>\n'
            f'</div>'
        )

    def _html_residual_budget(self, data: Dict[str, Any]) -> str:
        baseline = Decimal(str(data.get("baseline_tco2e", 0)))
        abatement = Decimal(str(data.get("abatement_tco2e", 0)))
        residual = Decimal(str(data.get("residual_tco2e", 0)))
        residual_pct = float(_pct_of(residual, baseline)) if baseline > 0 else 0
        gauge_color = "gauge-green" if residual_pct <= 10 else "gauge-amber" if residual_pct <= 20 else "gauge-red"
        return (
            f'<h2>1. Residual Emissions Budget</h2>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>Baseline Emissions</td><td>{_dec_comma(baseline)} tCO2e</td></tr>\n'
            f'<tr><td>Planned Abatement</td><td>{_dec_comma(abatement)} tCO2e</td></tr>\n'
            f'<tr><td>Residual Emissions</td><td>{_dec_comma(residual)} tCO2e</td></tr>\n'
            f'<tr><td>Residual Share</td><td>'
            f'<div class="gauge"><div class="gauge-fill {gauge_color}" '
            f'style="width:{min(residual_pct, 100)}%"></div></div>'
            f'{_dec(residual_pct)}% of baseline</td></tr>\n'
            f'</table>\n'
            f'<p><em>SBTi requires residual &le; 10% before neutralization.</em></p>'
        )

    def _html_portfolio_composition(self, data: Dict[str, Any]) -> str:
        credits_list = data.get("credits", [])
        total_volume = sum(Decimal(str(c.get("volume_tco2e", 0))) for c in credits_list)
        rows = ""
        for i, c in enumerate(credits_list, 1):
            volume = Decimal(str(c.get("volume_tco2e", 0)))
            cat_cls = "removal" if c.get("category") == "removal" else "avoidance"
            rows += (
                f'<tr class="{cat_cls}"><td>{i}</td><td>{c.get("type", "-")}</td>'
                f'<td>{c.get("category", "-")}</td><td>{c.get("registry", "-")}</td>'
                f'<td>{_dec_comma(volume)}</td>'
                f'<td>{_dec(_pct_of(volume, total_volume))}%</td>'
                f'<td>{c.get("vintage", "-")}</td>'
                f'<td>{_dec(c.get("quality_score", 0))}</td></tr>\n'
            )
        return (
            f'<h2>2. Portfolio Composition</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Type</th><th>Category</th><th>Registry</th>'
            f'<th>Volume (tCO2e)</th><th>Share</th><th>Vintage</th><th>Quality</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_quality_assessment(self, data: Dict[str, Any]) -> str:
        qa = data.get("quality_assessment", {})
        overall = qa.get("overall_score", 0)
        dimension_scores = qa.get("dimension_scores", {})
        rows = ""
        for dim in _QUALITY_DIMENSIONS:
            score = float(Decimal(str(dimension_scores.get(dim["id"], 0))))
            gauge_color = "gauge-green" if score >= 80 else "gauge-amber" if score >= 50 else "gauge-red"
            rating_cls = "quality-high" if score >= 80 else "quality-medium" if score >= 50 else "quality-low"
            rating = "High" if score >= 80 else "Medium" if score >= 50 else "Low"
            rows += (
                f'<tr><td>{dim["name"]}</td>'
                f'<td><div class="gauge"><div class="gauge-fill {gauge_color}" '
                f'style="width:{score}%"></div></div></td>'
                f'<td>{_dec(score)}</td>'
                f'<td class="{rating_cls}">{rating}</td></tr>\n'
            )
        return (
            f'<h2>3. Quality Assessment</h2>\n'
            f'<p><strong>Overall Score:</strong> {_dec(overall)} / 100</p>\n'
            f'<table>\n'
            f'<tr><th>Dimension</th><th>Score Bar</th><th>Score</th><th>Rating</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_sbti_compliance(self, data: Dict[str, Any]) -> str:
        sbti = data.get("sbti_compliance", {})
        credits_list = data.get("credits", [])
        removal = sum(
            Decimal(str(c.get("volume_tco2e", 0)))
            for c in credits_list if c.get("category") == "removal"
        )
        avoidance = sum(
            Decimal(str(c.get("volume_tco2e", 0)))
            for c in credits_list if c.get("category") != "removal"
        )
        return (
            f'<h2>4. SBTi Compliance</h2>\n'
            f'<h3>Neutralization vs Compensation</h3>\n'
            f'<table>\n'
            f'<tr><th>Category</th><th>Volume (tCO2e)</th><th>SBTi Role</th></tr>\n'
            f'<tr class="removal"><td>Carbon Removals</td><td>{_dec_comma(removal)}</td>'
            f'<td>Neutralization (required for net-zero)</td></tr>\n'
            f'<tr class="avoidance"><td>Avoidance/Reduction</td><td>{_dec_comma(avoidance)}</td>'
            f'<td>Compensation (beyond value chain mitigation)</td></tr>\n'
            f'</table>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Status</th></tr>\n'
            f'<tr><td>Neutralization Compliant</td><td>{sbti.get("neutralization_compliant", "N/A")}</td></tr>\n'
            f'<tr><td>BVCM Strategy</td><td>{sbti.get("bvcm_strategy", "N/A")}</td></tr>\n'
            f'<tr><td>Abatement First Verified</td><td>{sbti.get("abatement_first", "N/A")}</td></tr>\n'
            f'</table>'
        )

    def _html_vcmi_alignment(self, data: Dict[str, Any]) -> str:
        vcmi = data.get("vcmi_alignment", {})
        claim_level = vcmi.get("claim_level", "N/A")
        prerequisites = vcmi.get("prerequisites_met", [])
        gaps = vcmi.get("gaps", [])
        prereq_html = ""
        for p in prerequisites:
            prereq_html += f'<li style="color:#1b5e20;">&#10004; {p}</li>'
        gap_html = ""
        for g in gaps:
            gap_html += f'<li style="color:#c62828;">&#10008; {g}</li>'
        return (
            f'<h2>5. VCMI Claims Alignment</h2>\n'
            f'<p><strong>Claim Level:</strong> {claim_level}</p>\n'
            f'<h3>Prerequisites Met</h3>\n<ul>{prereq_html}</ul>\n'
            + (f'<h3>Gaps</h3>\n<ul>{gap_html}</ul>\n' if gaps else '')
        )

    def _html_retirement_schedule(self, data: Dict[str, Any]) -> str:
        schedule = data.get("retirement_schedule", [])
        rows = ""
        for item in schedule:
            rows += (
                f'<tr><td>{item.get("year", "-")}</td>'
                f'<td>{_dec_comma(item.get("volume_tco2e", 0))}</td>'
                f'<td>{item.get("source", "-")}</td>'
                f'<td>{item.get("registry", "-")}</td>'
                f'<td>{item.get("serial_range", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Credit Retirement Schedule</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Volume (tCO2e)</th><th>Source</th>'
            f'<th>Registry</th><th>Serial Range</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_cost_projection(self, data: Dict[str, Any]) -> str:
        projection = data.get("cost_projection", [])
        rows = ""
        total_cost = Decimal("0")
        for yr in projection:
            cost = Decimal(str(yr.get("total_cost_eur", 0)))
            total_cost += cost
            rows += (
                f'<tr><td>{yr.get("year", "-")}</td>'
                f'<td>{_dec_comma(yr.get("volume_tco2e", 0))}</td>'
                f'<td>{_dec_comma(yr.get("avg_price_eur", 0))}</td>'
                f'<td>{_dec_comma(cost, 0)}</td>'
                f'<td>{yr.get("type_mix", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. Cost Projection</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Volume (tCO2e)</th><th>Avg Price (EUR/tCO2e)</th>'
            f'<th>Total Cost (EUR)</th><th>Type Mix</th></tr>\n'
            f'{rows}</table>\n'
            f'<p><strong>Total Projected Cost:</strong> EUR {_dec_comma(total_cost, 0)}</p>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        items = ""
        for i, rec in enumerate(recs, 1):
            actions_html = ""
            if rec.get("actions"):
                actions_html = "<ul>" + "".join(f"<li>{a}</li>" for a in rec["actions"]) + "</ul>"
            items += (
                f'<div style="margin:12px 0;padding:16px;border:1px solid #c8e6c9;'
                f'border-radius:8px;border-left:4px solid #2e7d32;">'
                f'<strong>{i}. {rec.get("title", "")}</strong>'
                f'<p>{rec.get("description", "")}</p>'
                f'{actions_html}</div>\n'
            )
        return f'<h2>8. Recommendations</h2>\n{items}'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}<br>'
            f'SBTi Corporate Net-Zero Standard &amp; VCMI Claims Code</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
