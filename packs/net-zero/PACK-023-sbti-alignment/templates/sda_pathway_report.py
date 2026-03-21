# -*- coding: utf-8 -*-
"""
SDAPathwayReportTemplate - Sectoral Decarbonization Approach report for PACK-023.

Renders a sector-specific SDA intensity convergence report with sector
classification, base year intensity profile, year-by-year convergence pathway,
absolute emissions trajectory, ACA vs SDA comparison, IEA NZE benchmarks,
and SBTi validation status.

Sections:
    1. Sector Classification
    2. Base Year Intensity Profile
    3. Convergence Pathway (year-by-year intensity table)
    4. Absolute Emissions Trajectory
    5. ACA vs SDA Comparison
    6. IEA NZE Benchmark
    7. SBTi Validation Status

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _pct(val: Any) -> str:
    """Format a value as percentage string."""
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)


class SDAPathwayReportTemplate:
    """
    SDA sector pathway report template for SBTi alignment.

    Renders the Sectoral Decarbonization Approach convergence pathway
    for one of 12 SBTi sectors, comparing company intensity with sector
    benchmarks and IEA Net Zero Emissions (NZE) scenarios.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SDAPathwayReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render SDA pathway report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_sector_classification(data),
            self._md_base_year_intensity(data),
            self._md_convergence_pathway(data),
            self._md_absolute_trajectory(data),
            self._md_aca_vs_sda(data),
            self._md_iea_benchmark(data),
            self._md_validation_status(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render SDA pathway report as self-contained HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_sector_classification(data),
            self._html_base_year_intensity(data),
            self._html_convergence_pathway(data),
            self._html_absolute_trajectory(data),
            self._html_aca_vs_sda(data),
            self._html_iea_benchmark(data),
            self._html_validation_status(data),
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
        """Render SDA pathway report as structured JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "sda_pathway_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "sector": data.get("sector", {}),
            "base_year_intensity": data.get("base_year_intensity", {}),
            "convergence_pathway": data.get("convergence_pathway", []),
            "absolute_trajectory": data.get("absolute_trajectory", []),
            "aca_vs_sda": data.get("aca_vs_sda", {}),
            "iea_benchmark": data.get("iea_benchmark", {}),
            "validation_status": data.get("validation_status", {}),
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
            f"## 1. Sector Classification\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Sector | {sector.get('name', 'N/A')} |\n"
            f"| SBTi Sector Code | {sector.get('code', 'N/A')} |\n"
            f"| ISIC Classification | {sector.get('isic', 'N/A')} |\n"
            f"| Intensity Metric | {sector.get('intensity_metric', 'N/A')} |\n"
            f"| Convergence Year | {sector.get('convergence_year', 2050)} |\n"
            f"| 2050 Benchmark | {_dec(sector.get('benchmark_2050', 0), 4)} "
            f"{sector.get('intensity_unit', '')} |\n"
            f"| SDA Mandatory | {sector.get('sda_mandatory', 'No')} |\n"
            f"| Pathway Method | SDA (Sectoral Decarbonization Approach) |"
        )

    def _md_base_year_intensity(self, data: Dict[str, Any]) -> str:
        byi = data.get("base_year_intensity", {})
        return (
            f"## 2. Base Year Intensity Profile\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Base Year | {byi.get('year', 'N/A')} |\n"
            f"| Company Intensity | {_dec(byi.get('company_intensity', 0), 4)} "
            f"{byi.get('unit', '')} |\n"
            f"| Sector Average Intensity | {_dec(byi.get('sector_intensity', 0), 4)} "
            f"{byi.get('unit', '')} |\n"
            f"| Activity Data | {_dec_comma(byi.get('activity_data', 0), 0)} "
            f"{byi.get('activity_unit', '')} |\n"
            f"| Total Emissions (tCO2e) | {_dec_comma(byi.get('total_emissions_tco2e', 0), 0)} |\n"
            f"| Position vs Sector | {byi.get('position_vs_sector', 'N/A')} |"
        )

    def _md_convergence_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("convergence_pathway", [])
        unit = data.get("base_year_intensity", {}).get("unit", "")
        lines = [
            "## 3. Convergence Pathway\n",
            f"Year-by-year intensity convergence from base year to {data.get('sector', {}).get('convergence_year', 2050)}.\n",
            f"| Year | Company Target ({unit}) | Sector Benchmark ({unit}) "
            f"| Gap | Reduction from Base (%) |",
            f"|:----:|:----------------------:|:---------------------:"
            f"|:---:|:----------------------:|",
        ]
        for p in pathway:
            lines.append(
                f"| {p.get('year', '-')} "
                f"| {_dec(p.get('company_target', 0), 4)} "
                f"| {_dec(p.get('sector_benchmark', 0), 4)} "
                f"| {_dec(p.get('gap', 0), 4)} "
                f"| {_pct(p.get('reduction_from_base_pct', 0))} |"
            )
        if not pathway:
            lines.append("| - | _No pathway data_ | - | - | - |")
        return "\n".join(lines)

    def _md_absolute_trajectory(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("absolute_trajectory", [])
        lines = [
            "## 4. Absolute Emissions Trajectory\n",
            "| Year | Projected Activity | Intensity Target "
            "| Absolute Target (tCO2e) | YoY Change (%) |",
            "|:----:|:------------------:|:--------------:"
            "|:----------------------:|:--------------:|",
        ]
        for t in trajectory:
            lines.append(
                f"| {t.get('year', '-')} "
                f"| {_dec_comma(t.get('projected_activity', 0), 0)} "
                f"| {_dec(t.get('intensity_target', 0), 4)} "
                f"| {_dec_comma(t.get('absolute_target_tco2e', 0), 0)} "
                f"| {_pct(t.get('yoy_change_pct', 0))} |"
            )
        if not trajectory:
            lines.append("| - | _No trajectory data_ | - | - | - |")
        return "\n".join(lines)

    def _md_aca_vs_sda(self, data: Dict[str, Any]) -> str:
        comp = data.get("aca_vs_sda", {})
        aca = comp.get("aca", {})
        sda = comp.get("sda", {})
        return (
            f"## 5. ACA vs SDA Comparison\n\n"
            f"| Metric | ACA | SDA |\n|--------|-----|-----|\n"
            f"| Method | Absolute Contraction | Sectoral Decarbonization |\n"
            f"| Annual Reduction Rate | {_pct(aca.get('annual_rate', 0))} "
            f"| {_pct(sda.get('annual_rate', 0))} |\n"
            f"| 2030 Target (tCO2e) | {_dec_comma(aca.get('target_2030', 0), 0)} "
            f"| {_dec_comma(sda.get('target_2030', 0), 0)} |\n"
            f"| 2050 Target (tCO2e) | {_dec_comma(aca.get('target_2050', 0), 0)} "
            f"| {_dec_comma(sda.get('target_2050', 0), 0)} |\n"
            f"| Accounts for Growth | No | Yes |\n"
            f"| Sector-Specific | No | Yes |\n"
            f"| Recommended | {aca.get('recommended', 'N/A')} "
            f"| {sda.get('recommended', 'N/A')} |\n\n"
            f"**Selected Pathway:** {comp.get('selected', 'N/A')}  \n"
            f"**Rationale:** {comp.get('rationale', 'N/A')}"
        )

    def _md_iea_benchmark(self, data: Dict[str, Any]) -> str:
        iea = data.get("iea_benchmark", {})
        scenarios = iea.get("scenarios", [])
        lines = [
            "## 6. IEA NZE Benchmark\n",
            f"**IEA Scenario:** {iea.get('scenario_name', 'Net Zero Emissions by 2050')}  \n"
            f"**Sector Benchmark Source:** {iea.get('source', 'IEA World Energy Outlook')}\n",
            "| Year | IEA NZE Benchmark | SBTi SDA Target | Company Target "
            "| Alignment |",
            "|:----:|:-----------------:|:---------------:|:--------------:"
            "|:---------:|",
        ]
        for s in scenarios:
            lines.append(
                f"| {s.get('year', '-')} "
                f"| {_dec(s.get('iea_benchmark', 0), 4)} "
                f"| {_dec(s.get('sbti_target', 0), 4)} "
                f"| {_dec(s.get('company_target', 0), 4)} "
                f"| {s.get('alignment', '-')} |"
            )
        if not scenarios:
            lines.append("| - | _No benchmark data_ | - | - | - |")
        return "\n".join(lines)

    def _md_validation_status(self, data: Dict[str, Any]) -> str:
        vs = data.get("validation_status", {})
        checks = vs.get("checks", [])
        lines = [
            "## 7. SBTi Validation Status\n",
            f"**Overall Status:** {vs.get('overall', 'N/A')}  \n"
            f"**Pathway Valid:** {vs.get('pathway_valid', 'N/A')}  \n"
            f"**Ambition Level:** {vs.get('ambition_level', 'N/A')}\n",
            "| Check | Status | Detail |",
            "|-------|:------:|--------|",
        ]
        for c in checks:
            lines.append(
                f"| {c.get('check', '-')} "
                f"| {c.get('status', '-')} "
                f"| {c.get('detail', '-')} |"
            )
        if not checks:
            lines.append("| _No validation checks_ | - | - |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*SDA pathway per SBTi SDA Tool V3.0 with IEA NZE benchmarks.*"
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
            ".badge-pass{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-fail{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".highlight-row{background:#e8f5e9 !important;font-weight:600;}"
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
            f'  <div class="card"><div class="card-label">Intensity Metric</div>'
            f'<div class="card-value">{sector.get("intensity_metric", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">2050 Benchmark</div>'
            f'<div class="card-value">{_dec(sector.get("benchmark_2050", 0), 4)}</div>'
            f'<div class="card-unit">{sector.get("intensity_unit", "")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Convergence Year</div>'
            f'<div class="card-value">{sector.get("convergence_year", 2050)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>SBTi Sector Code</td><td>{sector.get("code", "N/A")}</td></tr>\n'
            f'<tr><td>ISIC Classification</td><td>{sector.get("isic", "N/A")}</td></tr>\n'
            f'<tr><td>SDA Mandatory</td><td>{sector.get("sda_mandatory", "No")}</td></tr>\n'
            f'</table>'
        )

    def _html_base_year_intensity(self, data: Dict[str, Any]) -> str:
        byi = data.get("base_year_intensity", {})
        return (
            f'<h2>2. Base Year Intensity Profile</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Base Year</div>'
            f'<div class="card-value">{byi.get("year", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Company Intensity</div>'
            f'<div class="card-value">{_dec(byi.get("company_intensity", 0), 4)}</div>'
            f'<div class="card-unit">{byi.get("unit", "")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Sector Average</div>'
            f'<div class="card-value">{_dec(byi.get("sector_intensity", 0), 4)}</div>'
            f'<div class="card-unit">{byi.get("unit", "")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Total Emissions</div>'
            f'<div class="card-value">{_dec_comma(byi.get("total_emissions_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_convergence_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("convergence_pathway", [])
        unit = data.get("base_year_intensity", {}).get("unit", "")
        rows = ""
        for p in pathway:
            rows += (
                f'<tr><td>{p.get("year", "-")}</td>'
                f'<td>{_dec(p.get("company_target", 0), 4)}</td>'
                f'<td>{_dec(p.get("sector_benchmark", 0), 4)}</td>'
                f'<td>{_dec(p.get("gap", 0), 4)}</td>'
                f'<td>{_pct(p.get("reduction_from_base_pct", 0))}</td></tr>\n'
            )
        return (
            f'<h2>3. Convergence Pathway</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Company Target ({unit})</th>'
            f'<th>Sector Benchmark ({unit})</th><th>Gap</th>'
            f'<th>Reduction from Base</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_absolute_trajectory(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("absolute_trajectory", [])
        rows = ""
        for t in trajectory:
            rows += (
                f'<tr><td>{t.get("year", "-")}</td>'
                f'<td>{_dec_comma(t.get("projected_activity", 0), 0)}</td>'
                f'<td>{_dec(t.get("intensity_target", 0), 4)}</td>'
                f'<td>{_dec_comma(t.get("absolute_target_tco2e", 0), 0)}</td>'
                f'<td>{_pct(t.get("yoy_change_pct", 0))}</td></tr>\n'
            )
        return (
            f'<h2>4. Absolute Emissions Trajectory</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Projected Activity</th>'
            f'<th>Intensity Target</th><th>Absolute (tCO2e)</th>'
            f'<th>YoY Change</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_aca_vs_sda(self, data: Dict[str, Any]) -> str:
        comp = data.get("aca_vs_sda", {})
        aca = comp.get("aca", {})
        sda = comp.get("sda", {})
        return (
            f'<h2>5. ACA vs SDA Comparison</h2>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>ACA</th><th>SDA</th></tr>\n'
            f'<tr><td>Method</td><td>Absolute Contraction</td>'
            f'<td>Sectoral Decarbonization</td></tr>\n'
            f'<tr><td>Annual Reduction Rate</td>'
            f'<td>{_pct(aca.get("annual_rate", 0))}</td>'
            f'<td>{_pct(sda.get("annual_rate", 0))}</td></tr>\n'
            f'<tr><td>2030 Target (tCO2e)</td>'
            f'<td>{_dec_comma(aca.get("target_2030", 0), 0)}</td>'
            f'<td>{_dec_comma(sda.get("target_2030", 0), 0)}</td></tr>\n'
            f'<tr><td>2050 Target (tCO2e)</td>'
            f'<td>{_dec_comma(aca.get("target_2050", 0), 0)}</td>'
            f'<td>{_dec_comma(sda.get("target_2050", 0), 0)}</td></tr>\n'
            f'<tr><td>Accounts for Growth</td><td>No</td><td>Yes</td></tr>\n'
            f'<tr><td>Sector-Specific</td><td>No</td><td>Yes</td></tr>\n'
            f'</table>\n'
            f'<p><strong>Selected:</strong> {comp.get("selected", "N/A")} | '
            f'<strong>Rationale:</strong> {comp.get("rationale", "N/A")}</p>'
        )

    def _html_iea_benchmark(self, data: Dict[str, Any]) -> str:
        iea = data.get("iea_benchmark", {})
        scenarios = iea.get("scenarios", [])
        rows = ""
        for s in scenarios:
            rows += (
                f'<tr><td>{s.get("year", "-")}</td>'
                f'<td>{_dec(s.get("iea_benchmark", 0), 4)}</td>'
                f'<td>{_dec(s.get("sbti_target", 0), 4)}</td>'
                f'<td>{_dec(s.get("company_target", 0), 4)}</td>'
                f'<td>{s.get("alignment", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. IEA NZE Benchmark</h2>\n'
            f'<p><strong>Scenario:</strong> '
            f'{iea.get("scenario_name", "Net Zero Emissions by 2050")} | '
            f'<strong>Source:</strong> {iea.get("source", "IEA")}</p>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>IEA NZE</th><th>SBTi SDA</th>'
            f'<th>Company</th><th>Alignment</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_validation_status(self, data: Dict[str, Any]) -> str:
        vs = data.get("validation_status", {})
        checks = vs.get("checks", [])
        rows = ""
        for c in checks:
            status = str(c.get("status", "")).upper()
            badge = (
                '<span class="badge-pass">PASS</span>'
                if status == "PASS"
                else '<span class="badge-fail">FAIL</span>'
            )
            rows += (
                f'<tr><td>{c.get("check", "-")}</td>'
                f'<td>{badge}</td>'
                f'<td>{c.get("detail", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. SBTi Validation Status</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Overall</div>'
            f'<div class="card-value">{vs.get("overall", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Pathway Valid</div>'
            f'<div class="card-value">{vs.get("pathway_valid", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Ambition</div>'
            f'<div class="card-value">{vs.get("ambition_level", "N/A")}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Check</th><th>Status</th><th>Detail</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'SDA pathway per SBTi SDA Tool V3.0 with IEA NZE benchmarks.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
