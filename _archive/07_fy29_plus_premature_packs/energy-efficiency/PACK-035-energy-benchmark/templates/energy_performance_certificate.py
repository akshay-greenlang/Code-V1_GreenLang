# -*- coding: utf-8 -*-
"""
EnergyPerformanceCertificateTemplate - EPC-style certificate for PACK-035.

Generates Energy Performance Certificate reports with an A-G rating scale,
coloured visual bars, primary energy value, CO2 emissions, improvement
recommendations, validity period, and assessor information. Follows the
EPBD framework for building energy labelling.

Sections:
    1. Certificate Header
    2. Building Information
    3. Energy Rating (A-G visual scale)
    4. Primary Energy Value
    5. CO2 Emissions
    6. Recommendations
    7. Validity Period
    8. Assessor Information
    9. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# EPC rating band definitions (A = best, G = worst)
EPC_BANDS: List[Dict[str, Any]] = [
    {"band": "A", "min": 0, "max": 25, "color": "#00703c", "label": "Very efficient"},
    {"band": "B", "min": 26, "max": 50, "color": "#27a243", "label": "Efficient"},
    {"band": "C", "min": 51, "max": 75, "color": "#8dce46", "label": "Fairly efficient"},
    {"band": "D", "min": 76, "max": 100, "color": "#ffd500", "label": "Average"},
    {"band": "E", "min": 101, "max": 125, "color": "#f4a335", "label": "Below average"},
    {"band": "F", "min": 126, "max": 150, "color": "#e8431e", "label": "Poor"},
    {"band": "G", "min": 151, "max": 999, "color": "#be1e2d", "label": "Very poor"},
]


class EnergyPerformanceCertificateTemplate:
    """
    Energy Performance Certificate report template.

    Renders EPC-style certificates with A-G rating bands including
    coloured visual bars, primary energy values, CO2 emissions,
    improvement recommendations, and assessor details across
    markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyPerformanceCertificateTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render energy performance certificate as Markdown.

        Args:
            data: EPC assessment data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_building_info(data),
            self._md_energy_rating(data),
            self._md_primary_energy(data),
            self._md_co2_emissions(data),
            self._md_recommendations(data),
            self._md_validity(data),
            self._md_assessor(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render energy performance certificate as self-contained HTML.

        Args:
            data: EPC assessment data from engine processing.

        Returns:
            Complete HTML string with inline CSS, coloured rating bars,
            and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_building_info(data),
            self._html_energy_rating(data),
            self._html_primary_energy(data),
            self._html_co2_emissions(data),
            self._html_recommendations(data),
            self._html_validity(data),
            self._html_assessor(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Performance Certificate</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render energy performance certificate as structured JSON.

        Args:
            data: EPC assessment data from engine processing.

        Returns:
            Dict with structured certificate sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "energy_performance_certificate",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "certificate_reference": data.get("certificate_reference", ""),
            "building": data.get("building", {}),
            "energy_rating": self._json_rating(data),
            "primary_energy": data.get("primary_energy", {}),
            "co2_emissions": data.get("co2_emissions", {}),
            "recommendations": data.get("recommendations", []),
            "validity": data.get("validity", {}),
            "assessor": data.get("assessor", {}),
            "bands": EPC_BANDS,
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render certificate header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Energy Performance Certificate\n\n"
            f"**Certificate Reference:** {data.get('certificate_reference', '-')}  \n"
            f"**Address:** {data.get('address', '-')}  \n"
            f"**Date of Assessment:** {data.get('assessment_date', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 EnergyPerformanceCertificateTemplate v35.0.0\n\n---"
        )

    def _md_building_info(self, data: Dict[str, Any]) -> str:
        """Render building information section."""
        b = data.get("building", {})
        return (
            "## 1. Building Information\n\n"
            "| Property | Value |\n|----------|-------|\n"
            f"| Type | {b.get('type', '-')} |\n"
            f"| Floor Area | {self._fmt(b.get('floor_area_sqm', 0), 0)} m2 |\n"
            f"| Year Built | {b.get('year_built', '-')} |\n"
            f"| Number of Floors | {b.get('num_floors', '-')} |\n"
            f"| Heating System | {b.get('heating_system', '-')} |\n"
            f"| Heating Fuel | {b.get('heating_fuel', '-')} |\n"
            f"| Wall Construction | {b.get('wall_construction', '-')} |\n"
            f"| Roof Insulation | {b.get('roof_insulation', '-')} |\n"
            f"| Glazing | {b.get('glazing', '-')} |\n"
            f"| Ventilation | {b.get('ventilation', '-')} |"
        )

    def _md_energy_rating(self, data: Dict[str, Any]) -> str:
        """Render energy rating section with A-G scale."""
        rating = data.get("energy_rating", {})
        current_score = rating.get("current_score", 0)
        current_band = rating.get("current_band", "G")
        potential_score = rating.get("potential_score", 0)
        potential_band = rating.get("potential_band", "G")
        lines = [
            "## 2. Energy Rating\n",
            f"**Current Rating:** **{current_band}** (Score: {current_score})  ",
            f"**Potential Rating:** **{potential_band}** (Score: {potential_score})\n",
            "### Rating Scale\n",
            "| Band | Score Range | Description | Current | Potential |",
            "|------|-----------|-------------|---------|-----------|",
        ]
        for b in EPC_BANDS:
            cur = " <<< " if b["band"] == current_band else ""
            pot = " <<< " if b["band"] == potential_band else ""
            lines.append(
                f"| **{b['band']}** | {b['min']}-{b['max']} "
                f"| {b['label']} | {cur} | {pot} |"
            )
        return "\n".join(lines)

    def _md_primary_energy(self, data: Dict[str, Any]) -> str:
        """Render primary energy value section."""
        pe = data.get("primary_energy", {})
        return (
            "## 3. Primary Energy\n\n"
            f"**Primary Energy Demand:** {self._fmt(pe.get('demand_kwh_m2_yr', 0))} kWh/m2/yr  \n"
            f"**Total Primary Energy:** {self._fmt(pe.get('total_kwh_yr', 0), 0)} kWh/yr  \n"
            f"**Primary Energy Factor:** {self._fmt(pe.get('pef', 0), 2)}  \n"
            f"**Renewable Energy Fraction:** {self._fmt(pe.get('renewable_fraction_pct', 0))}%  \n"
            f"**Non-Renewable Primary Energy:** "
            f"{self._fmt(pe.get('non_renewable_kwh_m2_yr', 0))} kWh/m2/yr"
        )

    def _md_co2_emissions(self, data: Dict[str, Any]) -> str:
        """Render CO2 emissions section."""
        co2 = data.get("co2_emissions", {})
        breakdown = co2.get("breakdown", [])
        lines = [
            "## 4. CO2 Emissions\n",
            f"**Total Emissions:** {self._fmt(co2.get('total_kg_co2_yr', 0), 0)} kg CO2/yr  ",
            f"**Emission Intensity:** {self._fmt(co2.get('kg_co2_m2_yr', 0))} kg CO2/m2/yr  ",
            f"**Environmental Impact Rating:** {co2.get('impact_rating', '-')}",
        ]
        if breakdown:
            lines.extend([
                "\n### Emissions by Source\n",
                "| Source | kg CO2/yr | Share (%) |",
                "|--------|----------|-----------|",
            ])
            for b in breakdown:
                lines.append(
                    f"| {b.get('source', '-')} "
                    f"| {self._fmt(b.get('kg_co2_yr', 0), 0)} "
                    f"| {self._fmt(b.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 5. Recommendations\n\n_No recommendations available._"
        lines = [
            "## 5. Recommendations\n",
            "| # | Measure | Typical Cost | Annual Saving | Rating Improvement |",
            "|---|---------|-------------|---------------|-------------------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('measure', '-')} "
                f"| {r.get('typical_cost', '-')} "
                f"| {r.get('annual_saving', '-')} "
                f"| {r.get('rating_improvement', '-')} |"
            )
        return "\n".join(lines)

    def _md_validity(self, data: Dict[str, Any]) -> str:
        """Render validity period section."""
        v = data.get("validity", {})
        return (
            "## 6. Validity\n\n"
            f"**Issue Date:** {v.get('issue_date', '-')}  \n"
            f"**Expiry Date:** {v.get('expiry_date', '-')}  \n"
            f"**Validity Period:** {v.get('validity_years', 10)} years  \n"
            f"**Status:** {v.get('status', 'Valid')}"
        )

    def _md_assessor(self, data: Dict[str, Any]) -> str:
        """Render assessor information section."""
        a = data.get("assessor", {})
        return (
            "## 7. Assessor Information\n\n"
            "| Field | Value |\n|-------|-------|\n"
            f"| Name | {a.get('name', '-')} |\n"
            f"| Accreditation Number | {a.get('accreditation_number', '-')} |\n"
            f"| Accreditation Scheme | {a.get('accreditation_scheme', '-')} |\n"
            f"| Company | {a.get('company', '-')} |\n"
            f"| Phone | {a.get('phone', '-')} |\n"
            f"| Email | {a.get('email', '-')} |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render certificate footer."""
        return "---\n\n*Generated by GreenLang PACK-035 Energy Benchmark Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML certificate header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Energy Performance Certificate</h1>\n'
            f'<p class="subtitle">Ref: {data.get("certificate_reference", "-")} | '
            f'Address: {data.get("address", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_building_info(self, data: Dict[str, Any]) -> str:
        """Render HTML building information table."""
        b = data.get("building", {})
        fields = [
            ("Type", b.get("type", "-")),
            ("Floor Area", f"{self._fmt(b.get('floor_area_sqm', 0), 0)} m2"),
            ("Year Built", b.get("year_built", "-")),
            ("Floors", b.get("num_floors", "-")),
            ("Heating System", b.get("heating_system", "-")),
            ("Heating Fuel", b.get("heating_fuel", "-")),
            ("Wall Construction", b.get("wall_construction", "-")),
            ("Glazing", b.get("glazing", "-")),
        ]
        rows = "".join(
            f'<tr><td>{label}</td><td>{val}</td></tr>\n' for label, val in fields
        )
        return (
            '<h2>Building Information</h2>\n'
            f'<table>\n<tr><th>Property</th><th>Value</th></tr>\n{rows}</table>'
        )

    def _html_energy_rating(self, data: Dict[str, Any]) -> str:
        """Render HTML energy rating with coloured A-G bars.

        Creates the distinctive EPC visual scale with arrows indicating
        the current and potential rating bands.
        """
        rating = data.get("energy_rating", {})
        current_band = rating.get("current_band", "G")
        current_score = rating.get("current_score", 0)
        potential_band = rating.get("potential_band", "G")
        potential_score = rating.get("potential_score", 0)

        bands_html = ""
        for idx, b in enumerate(EPC_BANDS):
            width = 92 - (idx * 8)
            is_current = b["band"] == current_band
            is_potential = b["band"] == potential_band
            marker = ""
            if is_current:
                marker += (
                    f'<span class="epc-marker epc-current">'
                    f'Current: {current_score}</span>'
                )
            if is_potential:
                marker += (
                    f'<span class="epc-marker epc-potential">'
                    f'Potential: {potential_score}</span>'
                )
            bands_html += (
                f'<div class="epc-band-row">'
                f'<div class="epc-band" style="background:{b["color"]};'
                f'width:{width}%;padding:8px 12px;color:#fff;font-weight:bold;">'
                f'{b["band"]} ({b["min"]}-{b["max"]}) - {b["label"]}</div>'
                f'{marker}</div>\n'
            )
        return (
            '<h2>Energy Rating</h2>\n'
            f'<div class="epc-scale">\n{bands_html}</div>'
        )

    def _html_primary_energy(self, data: Dict[str, Any]) -> str:
        """Render HTML primary energy section."""
        pe = data.get("primary_energy", {})
        return (
            '<h2>Primary Energy</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Primary Energy Demand</span>'
            f'<span class="value">{self._fmt(pe.get("demand_kwh_m2_yr", 0))}</span>'
            f'<span class="label">kWh/m2/yr</span></div>\n'
            f'  <div class="card"><span class="label">Total Primary Energy</span>'
            f'<span class="value">{self._fmt(pe.get("total_kwh_yr", 0), 0)}</span>'
            f'<span class="label">kWh/yr</span></div>\n'
            f'  <div class="card"><span class="label">Renewable Fraction</span>'
            f'<span class="value">{self._fmt(pe.get("renewable_fraction_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_co2_emissions(self, data: Dict[str, Any]) -> str:
        """Render HTML CO2 emissions section."""
        co2 = data.get("co2_emissions", {})
        breakdown = co2.get("breakdown", [])
        rows = "".join(
            f'<tr><td>{b.get("source", "-")}</td>'
            f'<td>{self._fmt(b.get("kg_co2_yr", 0), 0)}</td>'
            f'<td>{self._fmt(b.get("share_pct", 0))}%</td></tr>\n'
            for b in breakdown
        )
        return (
            '<h2>CO2 Emissions</h2>\n'
            f'<p>Total: <strong>{self._fmt(co2.get("total_kg_co2_yr", 0), 0)} kg CO2/yr</strong> | '
            f'Intensity: {self._fmt(co2.get("kg_co2_m2_yr", 0))} kg CO2/m2/yr</p>\n'
            '<table>\n<tr><th>Source</th><th>kg CO2/yr</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations table."""
        recs = data.get("recommendations", [])
        rows = ""
        for i, r in enumerate(recs, 1):
            rows += (
                f'<tr><td>{i}</td><td>{r.get("measure", "-")}</td>'
                f'<td>{r.get("typical_cost", "-")}</td>'
                f'<td>{r.get("annual_saving", "-")}</td>'
                f'<td>{r.get("rating_improvement", "-")}</td></tr>\n'
            )
        return (
            '<h2>Recommendations</h2>\n'
            '<table>\n<tr><th>#</th><th>Measure</th><th>Cost</th>'
            f'<th>Annual Saving</th><th>Rating Improvement</th></tr>\n{rows}</table>'
        )

    def _html_validity(self, data: Dict[str, Any]) -> str:
        """Render HTML validity section."""
        v = data.get("validity", {})
        return (
            '<h2>Validity</h2>\n'
            f'<div class="info-box">'
            f'<p><strong>Issue Date:</strong> {v.get("issue_date", "-")} | '
            f'<strong>Expiry Date:</strong> {v.get("expiry_date", "-")} | '
            f'<strong>Status:</strong> {v.get("status", "Valid")}</p></div>'
        )

    def _html_assessor(self, data: Dict[str, Any]) -> str:
        """Render HTML assessor information."""
        a = data.get("assessor", {})
        return (
            '<h2>Assessor Information</h2>\n'
            '<table>\n'
            f'<tr><td>Name</td><td>{a.get("name", "-")}</td></tr>\n'
            f'<tr><td>Accreditation</td><td>{a.get("accreditation_number", "-")}</td></tr>\n'
            f'<tr><td>Scheme</td><td>{a.get("accreditation_scheme", "-")}</td></tr>\n'
            f'<tr><td>Company</td><td>{a.get("company", "-")}</td></tr>\n'
            f'<tr><td>Assessment Date</td><td>{a.get("assessment_date", "-")}</td></tr>\n'
            '</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_rating(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON energy rating data."""
        rating = data.get("energy_rating", {})
        return {
            "current_score": rating.get("current_score", 0),
            "current_band": rating.get("current_band", "G"),
            "potential_score": rating.get("potential_score", 0),
            "potential_band": rating.get("potential_band", "G"),
            "bands": EPC_BANDS,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering with EPC-style visual scale."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".epc-scale{margin:20px 0;}"
            ".epc-band-row{display:flex;align-items:center;margin:3px 0;}"
            ".epc-band{border-radius:4px 0 0 4px;font-size:0.9em;min-height:28px;"
            "display:flex;align-items:center;}"
            ".epc-marker{margin-left:10px;padding:4px 10px;border-radius:4px;"
            "font-size:0.85em;font-weight:700;}"
            ".epc-current{background:#1a1a2e;color:#fff;}"
            ".epc-potential{background:#6c757d;color:#fff;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;"
            "text-align:center;min-width:150px;}"
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

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
