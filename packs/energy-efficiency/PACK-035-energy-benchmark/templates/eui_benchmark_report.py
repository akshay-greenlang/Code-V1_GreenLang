# -*- coding: utf-8 -*-
"""
EUIBenchmarkReportTemplate - Energy Use Intensity benchmark report for PACK-035.

Generates comprehensive EUI benchmark reports with site/source/primary EUI
calculations, weather normalisation, rolling 12-month trends, and
occupancy-adjusted results. Designed for facility energy managers who need
to track energy performance against baselines and normalised conditions.

Sections:
    1. Header & Facility Summary
    2. EUI Calculation Methodology
    3. Site / Source / Primary EUI Results
    4. Weather Normalisation Details
    5. Rolling 12-Month Trend
    6. Occupancy-Adjusted EUI
    7. Methodology Notes
    8. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EUIBenchmarkReportTemplate:
    """
    Energy Use Intensity benchmark report template.

    Renders EUI benchmark reports with site, source, and primary energy
    intensity metrics, weather normalisation, rolling trends, and
    occupancy adjustments across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EUIBenchmarkReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render EUI benchmark report as Markdown.

        Args:
            data: EUI benchmark data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_facility_summary(data),
            self._md_methodology(data),
            self._md_eui_results(data),
            self._md_weather_normalisation(data),
            self._md_rolling_trend(data),
            self._md_occupancy_adjusted(data),
            self._md_methodology_notes(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render EUI benchmark report as self-contained HTML.

        Args:
            data: EUI benchmark data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_facility_summary(data),
            self._html_methodology(data),
            self._html_eui_results(data),
            self._html_weather_normalisation(data),
            self._html_rolling_trend(data),
            self._html_occupancy_adjusted(data),
            self._html_methodology_notes(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>EUI Benchmark Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render EUI benchmark report as structured JSON.

        Args:
            data: EUI benchmark data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "eui_benchmark_report",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "facility": data.get("facility", {}),
            "methodology": data.get("methodology", {}),
            "eui_results": self._json_eui_results(data),
            "weather_normalisation": data.get("weather_normalisation", {}),
            "rolling_trend": data.get("rolling_trend", []),
            "occupancy_adjusted": data.get("occupancy_adjusted", {}),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with report metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# EUI Benchmark Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Benchmark Standard:** {data.get('benchmark_standard', 'ENERGY STAR / CIBSE TM46')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 EUIBenchmarkReportTemplate v35.0.0\n\n---"
        )

    def _md_facility_summary(self, data: Dict[str, Any]) -> str:
        """Render facility summary section."""
        f = data.get("facility", {})
        return (
            "## 1. Facility Summary\n\n"
            "| Property | Value |\n|----------|-------|\n"
            f"| Name | {f.get('name', '-')} |\n"
            f"| Address | {f.get('address', '-')} |\n"
            f"| Building Type | {f.get('building_type', '-')} |\n"
            f"| Gross Floor Area | {self._fmt(f.get('gross_floor_area_sqm', 0), 0)} m2 |\n"
            f"| Conditioned Area | {self._fmt(f.get('conditioned_area_sqm', 0), 0)} m2 |\n"
            f"| Year Built | {f.get('year_built', '-')} |\n"
            f"| Climate Zone | {f.get('climate_zone', '-')} |\n"
            f"| Operating Hours | {self._fmt(f.get('operating_hours_yr', 0), 0)} hrs/yr |\n"
            f"| Occupancy | {self._fmt(f.get('occupancy_persons', 0), 0)} persons |"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render EUI calculation methodology section."""
        m = data.get("methodology", {})
        return (
            "## 2. EUI Calculation Methodology\n\n"
            f"**Site EUI:** Total energy consumed at the meter divided by gross "
            f"floor area (kWh/m2/yr).  \n"
            f"**Source EUI:** Site energy multiplied by source-site ratios to "
            f"account for upstream losses (kWh/m2/yr).  \n"
            f"**Primary EUI:** Primary energy factor applied per EN 15603 "
            f"methodology (kWh/m2/yr).\n\n"
            f"**Source-Site Ratio (Electricity):** {self._fmt(m.get('source_site_ratio_elec', 2.80), 2)}  \n"
            f"**Source-Site Ratio (Gas):** {self._fmt(m.get('source_site_ratio_gas', 1.05), 2)}  \n"
            f"**Primary Energy Factor (Electricity):** {self._fmt(m.get('pef_electricity', 2.50), 2)}  \n"
            f"**Primary Energy Factor (Gas):** {self._fmt(m.get('pef_gas', 1.10), 2)}  \n"
            f"**Weather Normalisation Method:** {m.get('normalisation_method', 'Degree-Day Regression')}"
        )

    def _md_eui_results(self, data: Dict[str, Any]) -> str:
        """Render site/source/primary EUI results section."""
        eui = data.get("eui_results", {})
        lines = [
            "## 3. EUI Results\n",
            "| Metric | Value (kWh/m2/yr) | vs Benchmark | Status |",
            "|--------|-------------------|-------------|--------|",
        ]
        for metric in ["site_eui", "source_eui", "primary_eui"]:
            m = eui.get(metric, {})
            label = metric.replace("_", " ").title()
            lines.append(
                f"| {label} | {self._fmt(m.get('value', 0))} "
                f"| {self._fmt(m.get('benchmark', 0))} "
                f"| {m.get('status', '-')} |"
            )
        total_energy = eui.get("total_energy_kwh", 0)
        total_cost = eui.get("total_energy_cost", 0)
        lines.extend([
            "",
            f"**Total Energy Consumed:** {self._fmt(total_energy, 0)} kWh/yr  ",
            f"**Total Energy Cost:** EUR {self._fmt(total_cost)} /yr  ",
            f"**Energy Cost Intensity:** EUR {self._fmt(eui.get('cost_intensity', 0))} /m2/yr",
        ])
        breakdown = eui.get("by_fuel", [])
        if breakdown:
            lines.extend([
                "\n### By Fuel Type\n",
                "| Fuel | kWh/yr | kWh/m2/yr | Share (%) | Cost (EUR) |",
                "|------|--------|-----------|-----------|------------|",
            ])
            for b in breakdown:
                lines.append(
                    f"| {b.get('fuel', '-')} "
                    f"| {self._fmt(b.get('kwh_yr', 0), 0)} "
                    f"| {self._fmt(b.get('kwh_m2_yr', 0))} "
                    f"| {self._fmt(b.get('share_pct', 0))}% "
                    f"| {self._fmt(b.get('cost_eur', 0))} |"
                )
        return "\n".join(lines)

    def _md_weather_normalisation(self, data: Dict[str, Any]) -> str:
        """Render weather normalisation details section."""
        wn = data.get("weather_normalisation", {})
        lines = [
            "## 4. Weather Normalisation\n",
            f"**Method:** {wn.get('method', 'Degree-Day Regression')}  ",
            f"**Heating Degree Days (Actual):** {self._fmt(wn.get('hdd_actual', 0), 0)}  ",
            f"**Heating Degree Days (Normal):** {self._fmt(wn.get('hdd_normal', 0), 0)}  ",
            f"**Cooling Degree Days (Actual):** {self._fmt(wn.get('cdd_actual', 0), 0)}  ",
            f"**Cooling Degree Days (Normal):** {self._fmt(wn.get('cdd_normal', 0), 0)}  ",
            f"**Base Temperature (Heating):** {self._fmt(wn.get('base_temp_heating_c', 15.5), 1)} C  ",
            f"**Base Temperature (Cooling):** {self._fmt(wn.get('base_temp_cooling_c', 18.0), 1)} C  ",
            f"**R-squared:** {self._fmt(wn.get('r_squared', 0), 3)}  ",
            f"**CVRMSE:** {self._fmt(wn.get('cvrmse_pct', 0), 1)}%\n",
            "| Metric | Actual | Normalised | Adjustment |",
            "|--------|--------|------------|------------|",
        ]
        for row in wn.get("comparison", []):
            lines.append(
                f"| {row.get('metric', '-')} "
                f"| {self._fmt(row.get('actual', 0))} "
                f"| {self._fmt(row.get('normalised', 0))} "
                f"| {self._fmt(row.get('adjustment', 0))} |"
            )
        return "\n".join(lines)

    def _md_rolling_trend(self, data: Dict[str, Any]) -> str:
        """Render rolling 12-month trend section."""
        trend = data.get("rolling_trend", [])
        if not trend:
            return "## 5. Rolling 12-Month EUI Trend\n\n_No trend data available._"
        lines = [
            "## 5. Rolling 12-Month EUI Trend\n",
            "| Period End | Site EUI | Source EUI | Normalised EUI | vs Benchmark |",
            "|-----------|----------|-----------|----------------|-------------|",
        ]
        for t in trend:
            lines.append(
                f"| {t.get('period_end', '-')} "
                f"| {self._fmt(t.get('site_eui', 0))} "
                f"| {self._fmt(t.get('source_eui', 0))} "
                f"| {self._fmt(t.get('normalised_eui', 0))} "
                f"| {t.get('vs_benchmark', '-')} |"
            )
        return "\n".join(lines)

    def _md_occupancy_adjusted(self, data: Dict[str, Any]) -> str:
        """Render occupancy-adjusted EUI section."""
        oa = data.get("occupancy_adjusted", {})
        return (
            "## 6. Occupancy-Adjusted EUI\n\n"
            f"**Average Occupancy Rate:** {self._fmt(oa.get('avg_occupancy_pct', 0))}%  \n"
            f"**Design Occupancy:** {self._fmt(oa.get('design_occupancy', 0), 0)} persons  \n"
            f"**Actual Average Occupancy:** {self._fmt(oa.get('actual_occupancy', 0), 0)} persons  \n"
            f"**Occupancy Adjustment Factor:** {self._fmt(oa.get('adjustment_factor', 1.0), 3)}  \n\n"
            "| Metric | Unadjusted | Adjusted | Difference |\n"
            "|--------|-----------|----------|------------|\n"
            f"| Site EUI (kWh/m2/yr) "
            f"| {self._fmt(oa.get('site_eui_unadjusted', 0))} "
            f"| {self._fmt(oa.get('site_eui_adjusted', 0))} "
            f"| {self._fmt(oa.get('site_eui_diff', 0))} |\n"
            f"| Source EUI (kWh/m2/yr) "
            f"| {self._fmt(oa.get('source_eui_unadjusted', 0))} "
            f"| {self._fmt(oa.get('source_eui_adjusted', 0))} "
            f"| {self._fmt(oa.get('source_eui_diff', 0))} |\n"
            f"| Energy per Person (kWh/person/yr) "
            f"| {self._fmt(oa.get('kwh_per_person_unadjusted', 0))} "
            f"| {self._fmt(oa.get('kwh_per_person_adjusted', 0))} "
            f"| {self._fmt(oa.get('kwh_per_person_diff', 0))} |"
        )

    def _md_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render methodology notes section."""
        notes = data.get("methodology_notes", [])
        if not notes:
            notes = [
                "EUI calculations follow ASHRAE Standard 105 / EN 15603 methodology.",
                "Source-site ratios derived from national grid conversion factors.",
                "Weather normalisation uses TMY3 / CIBSE TRY long-term averages.",
                "Occupancy adjustment applies linear scaling per ASHRAE 90.1 Appendix G.",
            ]
        lines = ["## 7. Methodology Notes\n"]
        for n in notes:
            lines.append(f"- {n}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-035 Energy Benchmark Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>EUI Benchmark Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("reporting_period", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_facility_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML facility summary."""
        f = data.get("facility", {})
        fields = [
            ("Name", f.get("name", "-")),
            ("Building Type", f.get("building_type", "-")),
            ("Gross Floor Area", f"{self._fmt(f.get('gross_floor_area_sqm', 0), 0)} m2"),
            ("Climate Zone", f.get("climate_zone", "-")),
            ("Year Built", f.get("year_built", "-")),
            ("Operating Hours", f"{self._fmt(f.get('operating_hours_yr', 0), 0)} hrs/yr"),
        ]
        rows = "".join(
            f'<tr><td>{label}</td><td>{val}</td></tr>\n' for label, val in fields
        )
        return (
            '<h2>Facility Summary</h2>\n'
            f'<table>\n<tr><th>Property</th><th>Value</th></tr>\n{rows}</table>'
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology overview."""
        m = data.get("methodology", {})
        return (
            '<h2>EUI Calculation Methodology</h2>\n'
            '<div class="info-box">'
            f'<p><strong>Site EUI:</strong> Metered energy / floor area</p>'
            f'<p><strong>Source EUI:</strong> Site EUI x source-site ratio '
            f'(Elec: {self._fmt(m.get("source_site_ratio_elec", 2.80), 2)}, '
            f'Gas: {self._fmt(m.get("source_site_ratio_gas", 1.05), 2)})</p>'
            f'<p><strong>Primary EUI:</strong> EN 15603 primary energy factors</p>'
            f'<p><strong>Normalisation:</strong> {m.get("normalisation_method", "Degree-Day Regression")}</p>'
            '</div>'
        )

    def _html_eui_results(self, data: Dict[str, Any]) -> str:
        """Render HTML EUI results with summary cards and fuel table."""
        eui = data.get("eui_results", {})
        cards = ""
        for metric in ["site_eui", "source_eui", "primary_eui"]:
            m = eui.get(metric, {})
            label = metric.replace("_", " ").title()
            status = m.get("status", "-")
            cls = "card-green" if status == "PASS" else ("card-red" if status == "FAIL" else "")
            cards += (
                f'  <div class="card {cls}"><span class="label">{label}</span>'
                f'<span class="value">{self._fmt(m.get("value", 0))}</span>'
                f'<span class="label">kWh/m2/yr (BM: {self._fmt(m.get("benchmark", 0))})</span></div>\n'
            )
        fuel_rows = ""
        for b in eui.get("by_fuel", []):
            fuel_rows += (
                f'<tr><td>{b.get("fuel", "-")}</td>'
                f'<td>{self._fmt(b.get("kwh_yr", 0), 0)}</td>'
                f'<td>{self._fmt(b.get("kwh_m2_yr", 0))}</td>'
                f'<td>{self._fmt(b.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>EUI Results</h2>\n'
            f'<div class="summary-cards">\n{cards}</div>\n'
            '<table>\n<tr><th>Fuel</th><th>kWh/yr</th><th>kWh/m2/yr</th>'
            f'<th>Share</th></tr>\n{fuel_rows}</table>'
        )

    def _html_weather_normalisation(self, data: Dict[str, Any]) -> str:
        """Render HTML weather normalisation section."""
        wn = data.get("weather_normalisation", {})
        rows = ""
        for row in wn.get("comparison", []):
            rows += (
                f'<tr><td>{row.get("metric", "-")}</td>'
                f'<td>{self._fmt(row.get("actual", 0))}</td>'
                f'<td>{self._fmt(row.get("normalised", 0))}</td>'
                f'<td>{self._fmt(row.get("adjustment", 0))}</td></tr>\n'
            )
        return (
            '<h2>Weather Normalisation</h2>\n'
            f'<p>Method: {wn.get("method", "-")} | '
            f'HDD: {self._fmt(wn.get("hdd_actual", 0), 0)} actual / '
            f'{self._fmt(wn.get("hdd_normal", 0), 0)} normal | '
            f'R2: {self._fmt(wn.get("r_squared", 0), 3)}</p>\n'
            '<table>\n<tr><th>Metric</th><th>Actual</th><th>Normalised</th>'
            f'<th>Adjustment</th></tr>\n{rows}</table>'
        )

    def _html_rolling_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML rolling trend table."""
        trend = data.get("rolling_trend", [])
        rows = ""
        for t in trend:
            rows += (
                f'<tr><td>{t.get("period_end", "-")}</td>'
                f'<td>{self._fmt(t.get("site_eui", 0))}</td>'
                f'<td>{self._fmt(t.get("source_eui", 0))}</td>'
                f'<td>{self._fmt(t.get("normalised_eui", 0))}</td>'
                f'<td>{t.get("vs_benchmark", "-")}</td></tr>\n'
            )
        return (
            '<h2>Rolling 12-Month EUI Trend</h2>\n'
            '<table>\n<tr><th>Period End</th><th>Site EUI</th><th>Source EUI</th>'
            f'<th>Normalised EUI</th><th>vs Benchmark</th></tr>\n{rows}</table>'
        )

    def _html_occupancy_adjusted(self, data: Dict[str, Any]) -> str:
        """Render HTML occupancy-adjusted EUI section."""
        oa = data.get("occupancy_adjusted", {})
        return (
            '<h2>Occupancy-Adjusted EUI</h2>\n'
            f'<p>Occupancy Rate: {self._fmt(oa.get("avg_occupancy_pct", 0))}% | '
            f'Adjustment Factor: {self._fmt(oa.get("adjustment_factor", 1.0), 3)}</p>\n'
            '<table>\n<tr><th>Metric</th><th>Unadjusted</th>'
            '<th>Adjusted</th><th>Difference</th></tr>\n'
            f'<tr><td>Site EUI</td>'
            f'<td>{self._fmt(oa.get("site_eui_unadjusted", 0))}</td>'
            f'<td>{self._fmt(oa.get("site_eui_adjusted", 0))}</td>'
            f'<td>{self._fmt(oa.get("site_eui_diff", 0))}</td></tr>\n'
            f'<tr><td>Source EUI</td>'
            f'<td>{self._fmt(oa.get("source_eui_unadjusted", 0))}</td>'
            f'<td>{self._fmt(oa.get("source_eui_adjusted", 0))}</td>'
            f'<td>{self._fmt(oa.get("source_eui_diff", 0))}</td></tr>\n'
            '</table>'
        )

    def _html_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology notes."""
        notes = data.get("methodology_notes", [
            "EUI calculations follow ASHRAE Standard 105 / EN 15603 methodology.",
            "Source-site ratios derived from national grid conversion factors.",
            "Weather normalisation uses TMY3 / CIBSE TRY long-term averages.",
        ])
        items = "".join(f'<li>{n}</li>\n' for n in notes)
        return f'<h2>Methodology Notes</h2>\n<ul>\n{items}</ul>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_eui_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON EUI results data."""
        eui = data.get("eui_results", {})
        return {
            "site_eui": eui.get("site_eui", {}),
            "source_eui": eui.get("source_eui", {}),
            "primary_eui": eui.get("primary_eui", {}),
            "total_energy_kwh": eui.get("total_energy_kwh", 0),
            "total_energy_cost": eui.get("total_energy_cost", 0),
            "cost_intensity": eui.get("cost_intensity", 0),
            "by_fuel": eui.get("by_fuel", []),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        trend = data.get("rolling_trend", [])
        fuels = data.get("eui_results", {}).get("by_fuel", [])
        return {
            "eui_trend_line": {
                "type": "line",
                "labels": [t.get("period_end", "") for t in trend],
                "series": {
                    "site_eui": [t.get("site_eui", 0) for t in trend],
                    "source_eui": [t.get("source_eui", 0) for t in trend],
                    "normalised_eui": [t.get("normalised_eui", 0) for t in trend],
                },
            },
            "fuel_mix_pie": {
                "type": "pie",
                "labels": [f.get("fuel", "") for f in fuels],
                "values": [f.get("kwh_yr", 0) for f in fuels],
            },
            "eui_comparison_bar": {
                "type": "bar",
                "labels": ["Site EUI", "Source EUI", "Primary EUI"],
                "series": {
                    "actual": [
                        data.get("eui_results", {}).get("site_eui", {}).get("value", 0),
                        data.get("eui_results", {}).get("source_eui", {}).get("value", 0),
                        data.get("eui_results", {}).get("primary_eui", {}).get("value", 0),
                    ],
                    "benchmark": [
                        data.get("eui_results", {}).get("site_eui", {}).get("benchmark", 0),
                        data.get("eui_results", {}).get("source_eui", {}).get("benchmark", 0),
                        data.get("eui_results", {}).get("primary_eui", {}).get("benchmark", 0),
                    ],
                },
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:160px;}"
            ".card-green{background:#d1e7dd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

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

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
