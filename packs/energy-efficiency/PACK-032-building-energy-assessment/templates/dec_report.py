# -*- coding: utf-8 -*-
"""
DECReportTemplate - Display Energy Certificate report for PACK-032.

Generates Display Energy Certificate (DEC) reports for public buildings
as required under the EPBD recast. Includes operational rating on a
0-150+ scale, building details, electricity/heating/renewable breakdowns,
previous ratings comparison, and advisory report references.

Sections:
    1. Header & Certificate Details
    2. Operational Rating (0-150+ Scale)
    3. Building Details
    4. Electricity Breakdown
    5. Heating/Cooling Breakdown
    6. Renewable Energy Contribution
    7. Previous Ratings Comparison
    8. Advisory Report Reference
    9. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# DEC operational rating bands
DEC_BANDS: List[Dict[str, Any]] = [
    {"band": "A", "min": 0, "max": 25, "color": "#00a651", "label": "Excellent"},
    {"band": "B", "min": 26, "max": 50, "color": "#50b848", "label": "Good"},
    {"band": "C", "min": 51, "max": 75, "color": "#b2d235", "label": "Fairly Good"},
    {"band": "D", "min": 76, "max": 100, "color": "#fff200", "label": "Typical"},
    {"band": "E", "min": 101, "max": 125, "color": "#f7941d", "label": "Fairly Poor"},
    {"band": "F", "min": 126, "max": 150, "color": "#ed1c24", "label": "Poor"},
    {"band": "G", "min": 151, "max": 999, "color": "#be1e2d", "label": "Very Poor"},
]


class DECReportTemplate:
    """
    Display Energy Certificate report template.

    Renders DEC reports with operational ratings, energy breakdowns by
    electricity/heating/renewables, historical comparisons, and advisory
    report references across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    DEC_SECTIONS: List[str] = [
        "Certificate Details",
        "Operational Rating",
        "Building Details",
        "Electricity Breakdown",
        "Heating Breakdown",
        "Renewable Contribution",
        "Previous Ratings",
        "Advisory Report",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DECReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render DEC report as Markdown.

        Args:
            data: DEC assessment data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_operational_rating(data),
            self._md_building_details(data),
            self._md_electricity_breakdown(data),
            self._md_heating_breakdown(data),
            self._md_renewable_contribution(data),
            self._md_previous_ratings(data),
            self._md_advisory_report(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render DEC report as self-contained HTML.

        Args:
            data: DEC assessment data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_operational_rating(data),
            self._html_building_details(data),
            self._html_electricity_breakdown(data),
            self._html_heating_breakdown(data),
            self._html_renewable_contribution(data),
            self._html_previous_ratings(data),
            self._html_advisory_report(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Display Energy Certificate (DEC)</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render DEC report as structured JSON.

        Args:
            data: DEC assessment data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "dec_report",
            "version": "32.0.0",
            "standard": "EPBD_DEC",
            "generated_at": self.generated_at.isoformat(),
            "certificate": self._json_certificate(data),
            "operational_rating": self._json_operational_rating(data),
            "building": data.get("building", {}),
            "electricity": data.get("electricity", {}),
            "heating": data.get("heating", {}),
            "renewables": data.get("renewables", {}),
            "previous_ratings": data.get("previous_ratings", []),
            "advisory_report": data.get("advisory_report", {}),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with DEC metadata."""
        address = data.get("address", "")
        ref = data.get("certificate_reference", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        valid_until = data.get("valid_until", "")
        occupier = data.get("occupier", "")
        return (
            "# Display Energy Certificate (DEC)\n\n"
            f"**Address:** {address}  \n"
            f"**Occupier:** {occupier}  \n"
            f"**Certificate Reference:** {ref}  \n"
            f"**Valid Until:** {valid_until}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 DECReportTemplate v32.0.0\n\n---"
        )

    def _md_operational_rating(self, data: Dict[str, Any]) -> str:
        """Render operational rating section with 0-150+ scale."""
        rating = data.get("operational_rating", {})
        score = rating.get("score", 100)
        band = rating.get("band", "D")
        typical = rating.get("typical_score", 100)
        lines = [
            "## 1. Operational Rating\n",
            f"**Operational Rating:** {score}  ",
            f"**Band:** {band}  ",
            f"**Typical Building Score:** {typical}  ",
            f"**Building vs Typical:** {self._pct(score, typical)} of typical\n",
            "### Rating Scale\n",
            "| Band | Range | Label | Current |",
            "|------|-------|-------|---------|",
        ]
        for b in DEC_BANDS:
            marker = " <-- " if b["band"] == band else ""
            lines.append(
                f"| **{b['band']}** | {b['min']}-{b['max']} | {b['label']} | {marker} |"
            )
        return "\n".join(lines)

    def _md_building_details(self, data: Dict[str, Any]) -> str:
        """Render building details section."""
        bld = data.get("building", {})
        return (
            "## 2. Building Details\n\n"
            f"| Property | Value |\n|----------|-------|\n"
            f"| Category | {bld.get('category', '-')} |\n"
            f"| Main Activity | {bld.get('main_activity', '-')} |\n"
            f"| Total Useful Floor Area | {self._fmt(bld.get('floor_area_sqm', 0), 0)} m2 |\n"
            f"| Number of Floors | {bld.get('num_floors', '-')} |\n"
            f"| Occupancy Hours | {self._fmt(bld.get('occupancy_hours_yr', 0), 0)} hrs/yr |\n"
            f"| Typical Occupancy | {bld.get('typical_occupancy', '-')} |\n"
            f"| Primary Heating Fuel | {bld.get('primary_heating_fuel', '-')} |\n"
            f"| Air Conditioning | {bld.get('air_conditioning', '-')} |\n"
            f"| Renewable Systems | {bld.get('renewable_systems', '-')} |"
        )

    def _md_electricity_breakdown(self, data: Dict[str, Any]) -> str:
        """Render electricity breakdown section."""
        elec = data.get("electricity", {})
        annual = elec.get("annual_kwh", 0)
        intensity = elec.get("kwh_m2_yr", 0)
        benchmark = elec.get("benchmark_kwh_m2", 0)
        monthly = elec.get("monthly", [])
        lines = [
            "## 3. Electricity Breakdown\n",
            f"**Annual Consumption:** {self._fmt(annual, 0)} kWh  ",
            f"**Electricity Intensity:** {self._fmt(intensity)} kWh/m2/yr  ",
            f"**Benchmark:** {self._fmt(benchmark)} kWh/m2/yr  ",
            f"**vs Benchmark:** {self._pct(intensity, benchmark)} of benchmark",
        ]
        if monthly:
            lines.extend([
                "\n### Monthly Electricity Consumption\n",
                "| Month | kWh | Cost | vs Previous Year |",
                "|-------|-----|------|-----------------|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('kwh', 0), 0)} "
                    f"| {m.get('cost', '-')} "
                    f"| {m.get('vs_previous', '-')} |"
                )
        return "\n".join(lines)

    def _md_heating_breakdown(self, data: Dict[str, Any]) -> str:
        """Render heating/cooling breakdown section."""
        heating = data.get("heating", {})
        annual = heating.get("annual_kwh", 0)
        intensity = heating.get("kwh_m2_yr", 0)
        benchmark = heating.get("benchmark_kwh_m2", 0)
        fuel = heating.get("fuel_type", "-")
        monthly = heating.get("monthly", [])
        lines = [
            "## 4. Heating & Cooling Breakdown\n",
            f"**Primary Fuel:** {fuel}  ",
            f"**Annual Consumption:** {self._fmt(annual, 0)} kWh  ",
            f"**Heating Intensity:** {self._fmt(intensity)} kWh/m2/yr  ",
            f"**Benchmark:** {self._fmt(benchmark)} kWh/m2/yr  ",
            f"**vs Benchmark:** {self._pct(intensity, benchmark)} of benchmark",
        ]
        cooling = heating.get("cooling", {})
        if cooling:
            lines.extend([
                f"\n**Cooling Consumption:** {self._fmt(cooling.get('annual_kwh', 0), 0)} kWh  ",
                f"**Cooling Intensity:** {self._fmt(cooling.get('kwh_m2_yr', 0))} kWh/m2/yr",
            ])
        if monthly:
            lines.extend([
                "\n### Monthly Heating Consumption\n",
                "| Month | kWh | Degree Days | vs Previous Year |",
                "|-------|-----|------------|-----------------|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('kwh', 0), 0)} "
                    f"| {self._fmt(m.get('degree_days', 0), 0)} "
                    f"| {m.get('vs_previous', '-')} |"
                )
        return "\n".join(lines)

    def _md_renewable_contribution(self, data: Dict[str, Any]) -> str:
        """Render renewable energy contribution section."""
        renewables = data.get("renewables", {})
        total_gen = renewables.get("total_generation_kwh", 0)
        share = renewables.get("share_pct", 0)
        systems = renewables.get("systems", [])
        lines = [
            "## 5. Renewable Energy Contribution\n",
            f"**Total Renewable Generation:** {self._fmt(total_gen, 0)} kWh/yr  ",
            f"**Renewable Share:** {self._fmt(share)}%  ",
            f"**Grid Export:** {self._fmt(renewables.get('grid_export_kwh', 0), 0)} kWh/yr",
        ]
        if systems:
            lines.extend([
                "\n### Renewable Systems\n",
                "| System | Capacity | Generation (kWh/yr) | Share (%) |",
                "|--------|----------|--------------------| ----------|",
            ])
            for s in systems:
                lines.append(
                    f"| {s.get('system', '-')} "
                    f"| {s.get('capacity', '-')} "
                    f"| {self._fmt(s.get('generation_kwh', 0), 0)} "
                    f"| {self._fmt(s.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_previous_ratings(self, data: Dict[str, Any]) -> str:
        """Render previous ratings comparison section."""
        ratings = data.get("previous_ratings", [])
        if not ratings:
            return "## 6. Previous Ratings\n\n_No previous ratings available._"
        lines = [
            "## 6. Previous Ratings Comparison\n",
            "| Year | Score | Band | Electricity (kWh/m2) | Heating (kWh/m2) | Change |",
            "|------|-------|------|---------------------|-----------------|--------|",
        ]
        for r in ratings:
            lines.append(
                f"| {r.get('year', '-')} "
                f"| {r.get('score', '-')} "
                f"| {r.get('band', '-')} "
                f"| {self._fmt(r.get('electricity_kwh_m2', 0))} "
                f"| {self._fmt(r.get('heating_kwh_m2', 0))} "
                f"| {r.get('change', '-')} |"
            )
        return "\n".join(lines)

    def _md_advisory_report(self, data: Dict[str, Any]) -> str:
        """Render advisory report reference section."""
        advisory = data.get("advisory_report", {})
        return (
            "## 7. Advisory Report\n\n"
            f"**Advisory Report Reference:** {advisory.get('reference', '-')}  \n"
            f"**Report Date:** {advisory.get('date', '-')}  \n"
            f"**Assessor:** {advisory.get('assessor', '-')}  \n"
            f"**Accreditation:** {advisory.get('accreditation', '-')}  \n"
            f"**Key Recommendations:**\n\n"
            + "\n".join(
                f"- {r}" for r in advisory.get("key_recommendations", [])
            ) if advisory.get("key_recommendations") else
            "## 7. Advisory Report\n\n"
            f"**Advisory Report Reference:** {advisory.get('reference', '-')}  \n"
            f"**Report Date:** {advisory.get('date', '-')}  \n"
            f"**Assessor:** {advisory.get('assessor', '-')}  \n"
            f"**Accreditation:** {advisory.get('accreditation', '-')}  \n"
            f"_No key recommendations listed._"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 DECReportTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        address = data.get("address", "")
        occupier = data.get("occupier", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Display Energy Certificate (DEC)</h1>\n'
            f'<p class="subtitle">Address: {address} | Occupier: {occupier} | '
            f'Generated: {ts}</p>'
        )

    def _html_operational_rating(self, data: Dict[str, Any]) -> str:
        """Render HTML operational rating with visual scale."""
        rating = data.get("operational_rating", {})
        score = rating.get("score", 100)
        band = rating.get("band", "D")
        bands_html = ""
        for b in DEC_BANDS:
            is_current = b["band"] == band
            marker = f' <strong>[{score}]</strong>' if is_current else ""
            width = max(30, 100 - (DEC_BANDS.index(b) * 8))
            bands_html += (
                f'<div class="dec-band" style="background:{b["color"]};'
                f'width:{width}%;padding:6px 12px;margin:2px 0;color:#fff;'
                f'font-weight:bold;">{b["band"]} ({b["min"]}-{b["max"]}) '
                f'{b["label"]}{marker}</div>\n'
            )
        return (
            '<h2>Operational Rating</h2>\n'
            f'<div class="dec-scale">\n{bands_html}</div>'
        )

    def _html_building_details(self, data: Dict[str, Any]) -> str:
        """Render HTML building details."""
        bld = data.get("building", {})
        fields = [
            ("Category", bld.get("category", "-")),
            ("Main Activity", bld.get("main_activity", "-")),
            ("Floor Area", f"{self._fmt(bld.get('floor_area_sqm', 0), 0)} m2"),
            ("Floors", bld.get("num_floors", "-")),
            ("Occupancy Hours", f"{self._fmt(bld.get('occupancy_hours_yr', 0), 0)} hrs/yr"),
            ("Primary Heating Fuel", bld.get("primary_heating_fuel", "-")),
        ]
        rows = ""
        for label, val in fields:
            rows += f'<tr><td>{label}</td><td>{val}</td></tr>\n'
        return (
            '<h2>Building Details</h2>\n'
            f'<table>\n<tr><th>Property</th><th>Value</th></tr>\n{rows}</table>'
        )

    def _html_electricity_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML electricity breakdown."""
        elec = data.get("electricity", {})
        monthly = elec.get("monthly", [])
        rows = ""
        for m in monthly:
            rows += (
                f'<tr><td>{m.get("month", "-")}</td>'
                f'<td>{self._fmt(m.get("kwh", 0), 0)}</td>'
                f'<td>{m.get("cost", "-")}</td></tr>\n'
            )
        return (
            '<h2>Electricity Breakdown</h2>\n'
            f'<p>Annual: {self._fmt(elec.get("annual_kwh", 0), 0)} kWh | '
            f'Intensity: {self._fmt(elec.get("kwh_m2_yr", 0))} kWh/m2/yr</p>\n'
            '<table>\n<tr><th>Month</th><th>kWh</th><th>Cost</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_heating_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML heating breakdown."""
        heating = data.get("heating", {})
        monthly = heating.get("monthly", [])
        rows = ""
        for m in monthly:
            rows += (
                f'<tr><td>{m.get("month", "-")}</td>'
                f'<td>{self._fmt(m.get("kwh", 0), 0)}</td>'
                f'<td>{self._fmt(m.get("degree_days", 0), 0)}</td></tr>\n'
            )
        return (
            '<h2>Heating &amp; Cooling Breakdown</h2>\n'
            f'<p>Annual: {self._fmt(heating.get("annual_kwh", 0), 0)} kWh | '
            f'Intensity: {self._fmt(heating.get("kwh_m2_yr", 0))} kWh/m2/yr</p>\n'
            '<table>\n<tr><th>Month</th><th>kWh</th><th>Degree Days</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_renewable_contribution(self, data: Dict[str, Any]) -> str:
        """Render HTML renewable contribution."""
        renewables = data.get("renewables", {})
        systems = renewables.get("systems", [])
        rows = ""
        for s in systems:
            rows += (
                f'<tr><td>{s.get("system", "-")}</td>'
                f'<td>{s.get("capacity", "-")}</td>'
                f'<td>{self._fmt(s.get("generation_kwh", 0), 0)}</td>'
                f'<td>{self._fmt(s.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Renewable Energy Contribution</h2>\n'
            f'<p>Total: {self._fmt(renewables.get("total_generation_kwh", 0), 0)} kWh/yr | '
            f'Share: {self._fmt(renewables.get("share_pct", 0))}%</p>\n'
            '<table>\n<tr><th>System</th><th>Capacity</th><th>Generation</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_previous_ratings(self, data: Dict[str, Any]) -> str:
        """Render HTML previous ratings comparison."""
        ratings = data.get("previous_ratings", [])
        rows = ""
        for r in ratings:
            rows += (
                f'<tr><td>{r.get("year", "-")}</td>'
                f'<td>{r.get("score", "-")}</td>'
                f'<td>{r.get("band", "-")}</td>'
                f'<td>{r.get("change", "-")}</td></tr>\n'
            )
        return (
            '<h2>Previous Ratings</h2>\n'
            '<table>\n<tr><th>Year</th><th>Score</th><th>Band</th>'
            f'<th>Change</th></tr>\n{rows}</table>'
        )

    def _html_advisory_report(self, data: Dict[str, Any]) -> str:
        """Render HTML advisory report reference."""
        advisory = data.get("advisory_report", {})
        recs = advisory.get("key_recommendations", [])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return (
            '<h2>Advisory Report</h2>\n'
            f'<p>Reference: {advisory.get("reference", "-")} | '
            f'Date: {advisory.get("date", "-")} | '
            f'Assessor: {advisory.get("assessor", "-")}</p>\n'
            f'<ul>\n{items}</ul>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_certificate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON certificate metadata."""
        return {
            "address": data.get("address", ""),
            "occupier": data.get("occupier", ""),
            "reference": data.get("certificate_reference", ""),
            "valid_until": data.get("valid_until", ""),
            "reporting_period": data.get("reporting_period", ""),
        }

    def _json_operational_rating(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON operational rating."""
        rating = data.get("operational_rating", {})
        return {
            "score": rating.get("score", 100),
            "band": rating.get("band", "D"),
            "typical_score": rating.get("typical_score", 100),
            "bands": DEC_BANDS,
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
            ".dec-band{border-radius:4px;font-size:0.95em;}"
            ".dec-scale{margin:15px 0;}"
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
