# -*- coding: utf-8 -*-
"""
SDAPathwayReportTemplate - Sectoral Decarbonization Approach pathway for PACK-022.

Renders a sector-specific decarbonization pathway report with SDA convergence
curves, activity growth projections, absolute emissions trajectories, ACA vs
SDA comparisons, and IEA NZE benchmark alignment.

Sections:
    1. Sector Classification
    2. Base Year Profile (intensity + absolute)
    3. SDA Convergence Pathway (year-by-year)
    4. Activity Growth Projections
    5. Absolute Emissions Trajectory
    6. ACA vs SDA Comparison
    7. IEA NZE Benchmark
    8. SBTi Validation Status
    9. Key Assumptions

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"

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

class SDAPathwayReportTemplate:
    """
    Sectoral Decarbonization Approach pathway report template.

    Renders SDA convergence pathways with intensity targets, activity
    growth adjustments, and comparisons against ACA and IEA NZE benchmarks.

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
            self._md_sector_classification(data),
            self._md_base_year_profile(data),
            self._md_convergence_pathway(data),
            self._md_activity_growth(data),
            self._md_absolute_trajectory(data),
            self._md_aca_vs_sda(data),
            self._md_iea_benchmark(data),
            self._md_sbti_validation(data),
            self._md_key_assumptions(data),
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
            self._html_sector_classification(data),
            self._html_base_year_profile(data),
            self._html_convergence_pathway(data),
            self._html_activity_growth(data),
            self._html_absolute_trajectory(data),
            self._html_aca_vs_sda(data),
            self._html_iea_benchmark(data),
            self._html_sbti_validation(data),
            self._html_key_assumptions(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>SDA Pathway Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        sector = data.get("sector", {})
        base_year = data.get("base_year_profile", {})
        convergence = data.get("convergence_pathway", [])
        activity = data.get("activity_growth", [])
        trajectory = data.get("absolute_trajectory", [])
        aca_sda = data.get("aca_vs_sda", {})
        iea = data.get("iea_benchmark", {})
        sbti = data.get("sbti_validation", {})

        result: Dict[str, Any] = {
            "template": "sda_pathway_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "sector": sector,
            "base_year_profile": base_year,
            "convergence_pathway": convergence,
            "activity_growth": activity,
            "absolute_trajectory": trajectory,
            "aca_vs_sda": aca_sda,
            "iea_benchmark": iea,
            "sbti_validation": sbti,
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
            f"# SDA Pathway Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_sector_classification(self, data: Dict[str, Any]) -> str:
        sector = data.get("sector", {})
        return (
            "## 1. Sector Classification\n\n"
            f"- **Sector:** {sector.get('name', 'N/A')}\n"
            f"- **ISIC Code:** {sector.get('isic_code', 'N/A')}\n"
            f"- **SDA Category:** {sector.get('sda_category', 'N/A')}\n"
            f"- **Homogeneous/Heterogeneous:** {sector.get('type', 'N/A')}\n"
            f"- **Intensity Metric:** {sector.get('intensity_metric', 'tCO2e per unit')}\n"
            f"- **Convergence Year:** {sector.get('convergence_year', '2050')}\n"
            f"- **Global Budget Allocation:** {_dec(sector.get('budget_share_pct', 0))}%"
        )

    def _md_base_year_profile(self, data: Dict[str, Any]) -> str:
        profile = data.get("base_year_profile", {})
        intensity = profile.get("intensity", 0)
        absolute = profile.get("absolute_tco2e", 0)
        activity = profile.get("activity_level", 0)
        activity_unit = profile.get("activity_unit", "units")
        return (
            "## 2. Base Year Profile\n\n"
            f"**Base Year:** {profile.get('base_year', 'N/A')}\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Absolute Emissions | {_dec_comma(absolute)} tCO2e |\n"
            f"| Activity Level | {_dec_comma(activity)} {activity_unit} |\n"
            f"| Emission Intensity | {_dec(intensity, 4)} tCO2e/{activity_unit} |\n"
            f"| Sector Average Intensity | {_dec(profile.get('sector_avg_intensity', 0), 4)} tCO2e/{activity_unit} |\n"
            f"| Global Convergence Target | {_dec(profile.get('convergence_intensity', 0), 4)} tCO2e/{activity_unit} |"
        )

    def _md_convergence_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("convergence_pathway", [])
        lines = [
            "## 3. SDA Convergence Pathway\n",
            "Year-by-year intensity targets converging to the sector global budget.\n",
            "| Year | Target Intensity | Reduction from Base (%) | Convergence Gap | Absolute Target (tCO2e) |",
            "|:----:|:----------------:|------------------------:|:---------------:|------------------------:|",
        ]
        for row in pathway:
            lines.append(
                f"| {row.get('year', '-')} "
                f"| {_dec(row.get('target_intensity', 0), 4)} "
                f"| {_dec(row.get('reduction_pct', 0))}% "
                f"| {_dec(row.get('convergence_gap', 0), 4)} "
                f"| {_dec_comma(row.get('absolute_target_tco2e', 0))} |"
            )
        if not pathway:
            lines.append("| - | _No convergence data_ | - | - | - |")
        return "\n".join(lines)

    def _md_activity_growth(self, data: Dict[str, Any]) -> str:
        growth = data.get("activity_growth", [])
        lines = [
            "## 4. Activity Growth Projections\n",
            "| Year | Projected Activity | Growth Rate (%) | Source |",
            "|:----:|--------------------|----------------:|--------|",
        ]
        for row in growth:
            lines.append(
                f"| {row.get('year', '-')} "
                f"| {_dec_comma(row.get('activity_level', 0))} "
                f"| {_dec(row.get('growth_rate_pct', 0))}% "
                f"| {row.get('source', '-')} |"
            )
        if not growth:
            lines.append("| - | _No growth projections_ | - | - |")
        return "\n".join(lines)

    def _md_absolute_trajectory(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("absolute_trajectory", [])
        lines = [
            "## 5. Absolute Emissions Trajectory\n",
            "| Year | Projected Emissions (tCO2e) | SDA Target (tCO2e) | Gap (tCO2e) | On Track |",
            "|:----:|----------------------------:|--------------------:|------------:|:--------:|",
        ]
        for row in trajectory:
            on_track = "YES" if row.get("on_track", False) else "NO"
            lines.append(
                f"| {row.get('year', '-')} "
                f"| {_dec_comma(row.get('projected_tco2e', 0))} "
                f"| {_dec_comma(row.get('sda_target_tco2e', 0))} "
                f"| {_dec_comma(row.get('gap_tco2e', 0))} "
                f"| {on_track} |"
            )
        if not trajectory:
            lines.append("| - | _No trajectory data_ | - | - | - |")
        return "\n".join(lines)

    def _md_aca_vs_sda(self, data: Dict[str, Any]) -> str:
        comp = data.get("aca_vs_sda", {})
        aca = comp.get("aca", {})
        sda = comp.get("sda", {})
        return (
            "## 6. ACA vs SDA Comparison\n\n"
            "| Metric | ACA (Absolute Contraction) | SDA (Sectoral Decarbonization) |\n"
            "|--------|:-------------------------:|:------------------------------:|\n"
            f"| Method | Absolute reduction | Intensity convergence |\n"
            f"| Annual Rate | {_dec(aca.get('annual_rate_pct', 0))}% | {_dec(sda.get('annual_rate_pct', 0))}% |\n"
            f"| 2030 Target (tCO2e) | {_dec_comma(aca.get('target_2030_tco2e', 0))} | {_dec_comma(sda.get('target_2030_tco2e', 0))} |\n"
            f"| 2050 Target (tCO2e) | {_dec_comma(aca.get('target_2050_tco2e', 0))} | {_dec_comma(sda.get('target_2050_tco2e', 0))} |\n"
            f"| Accounts for Growth | No | Yes |\n"
            f"| Sector Benchmark | N/A | {sda.get('benchmark', 'IEA NZE')} |\n"
            f"| Recommended | {aca.get('recommended', 'N/A')} | {sda.get('recommended', 'N/A')} |\n\n"
            f"**Assessment:** {comp.get('assessment', 'N/A')}"
        )

    def _md_iea_benchmark(self, data: Dict[str, Any]) -> str:
        iea = data.get("iea_benchmark", {})
        benchmarks = iea.get("benchmarks", [])
        lines = [
            "## 7. IEA NZE Benchmark\n",
            f"**Scenario:** {iea.get('scenario', 'IEA Net Zero Emissions by 2050')}  \n"
            f"**Data Source:** {iea.get('source', 'IEA World Energy Outlook')}\n",
            "| Year | IEA Sector Intensity | Company Intensity | Alignment Status |",
            "|:----:|---------------------:|------------------:|:----------------:|",
        ]
        for b in benchmarks:
            status = b.get("alignment", "N/A")
            lines.append(
                f"| {b.get('year', '-')} "
                f"| {_dec(b.get('iea_intensity', 0), 4)} "
                f"| {_dec(b.get('company_intensity', 0), 4)} "
                f"| {status} |"
            )
        if not benchmarks:
            lines.append("| - | _No benchmark data_ | - | - |")
        return "\n".join(lines)

    def _md_sbti_validation(self, data: Dict[str, Any]) -> str:
        sbti = data.get("sbti_validation", {})
        checks = sbti.get("checks", [])
        lines = [
            "## 8. SBTi Validation Status\n",
            f"**Overall Status:** {sbti.get('status', 'N/A')}  \n"
            f"**Submission Date:** {sbti.get('submission_date', 'N/A')}  \n"
            f"**Validation Body:** {sbti.get('validation_body', 'SBTi')}\n",
            "| Check | Requirement | Status | Notes |",
            "|-------|------------|:------:|-------|",
        ]
        for c in checks:
            status = "PASS" if c.get("pass", False) else "FAIL"
            lines.append(
                f"| {c.get('name', '-')} | {c.get('requirement', '-')} "
                f"| {status} | {c.get('notes', '-')} |"
            )
        if not checks:
            lines.append("| _No validation checks_ | - | - | - |")
        return "\n".join(lines)

    def _md_key_assumptions(self, data: Dict[str, Any]) -> str:
        assumptions = data.get("assumptions", [])
        lines = ["## 9. Key Assumptions\n"]
        if assumptions:
            lines.append("| # | Assumption | Category | Sensitivity |")
            lines.append("|---|------------|----------|-------------|")
            for i, a in enumerate(assumptions, 1):
                lines.append(
                    f"| {i} | {a.get('assumption', '-')} "
                    f"| {a.get('category', '-')} "
                    f"| {a.get('sensitivity', '-')} |"
                )
        else:
            lines.append("_No key assumptions documented._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*SDA methodology per SBTi Sectoral Decarbonization Approach (v2).*"
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
            ".pass{color:#1b5e20;font-weight:700;}"
            ".fail{color:#c62828;font-weight:700;}"
            ".on-track{color:#1b5e20;font-weight:600;}"
            ".off-track{color:#c62828;font-weight:600;}"
            ".progress-bar{background:#e0e0e0;border-radius:6px;height:18px;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:6px;}"
            ".fill-green{background:#43a047;}"
            ".fill-amber{background:#ff8f00;}"
            ".fill-red{background:#e53935;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>SDA Pathway Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_sector_classification(self, data: Dict[str, Any]) -> str:
        sector = data.get("sector", {})
        return (
            f'<h2>1. Sector Classification</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Sector</div>'
            f'<div class="card-value">{sector.get("name", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">SDA Category</div>'
            f'<div class="card-value">{sector.get("sda_category", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Convergence Year</div>'
            f'<div class="card-value">{sector.get("convergence_year", "2050")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Budget Share</div>'
            f'<div class="card-value">{_dec(sector.get("budget_share_pct", 0))}%</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Attribute</th><th>Value</th></tr>\n'
            f'<tr><td>ISIC Code</td><td>{sector.get("isic_code", "N/A")}</td></tr>\n'
            f'<tr><td>Type</td><td>{sector.get("type", "N/A")}</td></tr>\n'
            f'<tr><td>Intensity Metric</td><td>{sector.get("intensity_metric", "tCO2e per unit")}</td></tr>\n'
            f'</table>'
        )

    def _html_base_year_profile(self, data: Dict[str, Any]) -> str:
        profile = data.get("base_year_profile", {})
        activity_unit = profile.get("activity_unit", "units")
        return (
            f'<h2>2. Base Year Profile</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Base Year</div>'
            f'<div class="card-value">{profile.get("base_year", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Absolute Emissions</div>'
            f'<div class="card-value">{_dec_comma(profile.get("absolute_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Intensity</div>'
            f'<div class="card-value">{_dec(profile.get("intensity", 0), 4)}</div>'
            f'<div class="card-unit">tCO2e/{activity_unit}</div></div>\n'
            f'  <div class="card"><div class="card-label">Activity Level</div>'
            f'<div class="card-value">{_dec_comma(profile.get("activity_level", 0))}</div>'
            f'<div class="card-unit">{activity_unit}</div></div>\n'
            f'</div>'
        )

    def _html_convergence_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("convergence_pathway", [])
        rows = ""
        for row in pathway:
            rows += (
                f'<tr><td>{row.get("year", "-")}</td>'
                f'<td>{_dec(row.get("target_intensity", 0), 4)}</td>'
                f'<td>{_dec(row.get("reduction_pct", 0))}%</td>'
                f'<td>{_dec(row.get("convergence_gap", 0), 4)}</td>'
                f'<td>{_dec_comma(row.get("absolute_target_tco2e", 0))}</td></tr>\n'
            )
        return (
            f'<h2>3. SDA Convergence Pathway</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Target Intensity</th><th>Reduction (%)</th>'
            f'<th>Convergence Gap</th><th>Absolute Target (tCO2e)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_activity_growth(self, data: Dict[str, Any]) -> str:
        growth = data.get("activity_growth", [])
        rows = ""
        for row in growth:
            rows += (
                f'<tr><td>{row.get("year", "-")}</td>'
                f'<td>{_dec_comma(row.get("activity_level", 0))}</td>'
                f'<td>{_dec(row.get("growth_rate_pct", 0))}%</td>'
                f'<td>{row.get("source", "-")}</td></tr>\n'
            )
        return (
            f'<h2>4. Activity Growth Projections</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Projected Activity</th>'
            f'<th>Growth Rate</th><th>Source</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_absolute_trajectory(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("absolute_trajectory", [])
        rows = ""
        for row in trajectory:
            on_track = row.get("on_track", False)
            cls = "on-track" if on_track else "off-track"
            label = "On Track" if on_track else "Off Track"
            rows += (
                f'<tr><td>{row.get("year", "-")}</td>'
                f'<td>{_dec_comma(row.get("projected_tco2e", 0))}</td>'
                f'<td>{_dec_comma(row.get("sda_target_tco2e", 0))}</td>'
                f'<td>{_dec_comma(row.get("gap_tco2e", 0))}</td>'
                f'<td class="{cls}">{label}</td></tr>\n'
            )
        return (
            f'<h2>5. Absolute Emissions Trajectory</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Projected (tCO2e)</th><th>SDA Target (tCO2e)</th>'
            f'<th>Gap (tCO2e)</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_aca_vs_sda(self, data: Dict[str, Any]) -> str:
        comp = data.get("aca_vs_sda", {})
        aca = comp.get("aca", {})
        sda = comp.get("sda", {})
        return (
            f'<h2>6. ACA vs SDA Comparison</h2>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>ACA (Absolute Contraction)</th>'
            f'<th>SDA (Sectoral Decarbonization)</th></tr>\n'
            f'<tr><td>Method</td><td>Absolute reduction</td><td>Intensity convergence</td></tr>\n'
            f'<tr><td>Annual Rate</td><td>{_dec(aca.get("annual_rate_pct", 0))}%</td>'
            f'<td>{_dec(sda.get("annual_rate_pct", 0))}%</td></tr>\n'
            f'<tr><td>2030 Target</td><td>{_dec_comma(aca.get("target_2030_tco2e", 0))} tCO2e</td>'
            f'<td>{_dec_comma(sda.get("target_2030_tco2e", 0))} tCO2e</td></tr>\n'
            f'<tr><td>2050 Target</td><td>{_dec_comma(aca.get("target_2050_tco2e", 0))} tCO2e</td>'
            f'<td>{_dec_comma(sda.get("target_2050_tco2e", 0))} tCO2e</td></tr>\n'
            f'<tr><td>Accounts for Growth</td><td>No</td><td>Yes</td></tr>\n'
            f'<tr><td>Recommended</td><td>{aca.get("recommended", "N/A")}</td>'
            f'<td>{sda.get("recommended", "N/A")}</td></tr>\n'
            f'</table>\n'
            f'<p><strong>Assessment:</strong> {comp.get("assessment", "N/A")}</p>'
        )

    def _html_iea_benchmark(self, data: Dict[str, Any]) -> str:
        iea = data.get("iea_benchmark", {})
        benchmarks = iea.get("benchmarks", [])
        rows = ""
        for b in benchmarks:
            alignment = b.get("alignment", "N/A")
            cls = "on-track" if alignment.lower() in ("aligned", "on track") else "off-track"
            rows += (
                f'<tr><td>{b.get("year", "-")}</td>'
                f'<td>{_dec(b.get("iea_intensity", 0), 4)}</td>'
                f'<td>{_dec(b.get("company_intensity", 0), 4)}</td>'
                f'<td class="{cls}">{alignment}</td></tr>\n'
            )
        return (
            f'<h2>7. IEA NZE Benchmark</h2>\n'
            f'<p><strong>Scenario:</strong> {iea.get("scenario", "IEA NZE")} | '
            f'<strong>Source:</strong> {iea.get("source", "IEA WEO")}</p>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>IEA Sector Intensity</th>'
            f'<th>Company Intensity</th><th>Alignment</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_sbti_validation(self, data: Dict[str, Any]) -> str:
        sbti = data.get("sbti_validation", {})
        checks = sbti.get("checks", [])
        rows = ""
        for c in checks:
            passed = c.get("pass", False)
            cls = "pass" if passed else "fail"
            icon = "&#10004;" if passed else "&#10008;"
            rows += (
                f'<tr><td>{c.get("name", "-")}</td>'
                f'<td>{c.get("requirement", "-")}</td>'
                f'<td class="{cls}">{icon}</td>'
                f'<td>{c.get("notes", "-")}</td></tr>\n'
            )
        status = sbti.get("status", "N/A")
        status_cls = "on-track" if status.lower() in ("validated", "approved") else "off-track"
        return (
            f'<h2>8. SBTi Validation Status</h2>\n'
            f'<p><strong>Status:</strong> <span class="{status_cls}">{status}</span> | '
            f'<strong>Submission:</strong> {sbti.get("submission_date", "N/A")}</p>\n'
            f'<table>\n'
            f'<tr><th>Check</th><th>Requirement</th><th>Status</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_key_assumptions(self, data: Dict[str, Any]) -> str:
        assumptions = data.get("assumptions", [])
        rows = ""
        for i, a in enumerate(assumptions, 1):
            rows += (
                f'<tr><td>{i}</td><td>{a.get("assumption", "-")}</td>'
                f'<td>{a.get("category", "-")}</td>'
                f'<td>{a.get("sensitivity", "-")}</td></tr>\n'
            )
        return (
            f'<h2>9. Key Assumptions</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Assumption</th><th>Category</th><th>Sensitivity</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'SDA methodology per SBTi Sectoral Decarbonization Approach (v2).</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
