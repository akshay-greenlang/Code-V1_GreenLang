# -*- coding: utf-8 -*-
"""
EPCReportTemplate - Energy Performance Certificate report for PACK-032.

Generates Energy Performance Certificate (EPC) reports compliant with
the EU Energy Performance of Buildings Directive (EPBD). Includes
building details, EPC rating band (A-G) with visual representation,
energy use intensity, CO2 emissions, cost-effective and further
recommendations, estimated costs, Green Deal eligibility assessment,
and assessor details.

Sections:
    1. Header & Building Details
    2. EPC Rating (A-G Visual Band)
    3. Energy Use Breakdown
    4. CO2 Emissions
    5. Cost-Effective Recommendations
    6. Further Recommendations
    7. Estimated Costs & Savings
    8. Green Deal Eligibility
    9. Assessor Details
   10. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# EPC rating band definitions
EPC_BANDS: List[Dict[str, Any]] = [
    {"band": "A", "min": 0, "max": 25, "color": "#00a651"},
    {"band": "B", "min": 26, "max": 50, "color": "#50b848"},
    {"band": "C", "min": 51, "max": 75, "color": "#b2d235"},
    {"band": "D", "min": 76, "max": 100, "color": "#fff200"},
    {"band": "E", "min": 101, "max": 125, "color": "#f7941d"},
    {"band": "F", "min": 126, "max": 150, "color": "#ed1c24"},
    {"band": "G", "min": 151, "max": 999, "color": "#be1e2d"},
]


class EPCReportTemplate:
    """
    Energy Performance Certificate report template.

    Renders EPC reports with A-G rating bands, energy use breakdowns,
    CO2 emissions, improvement recommendations, cost estimates,
    Green Deal eligibility, and assessor information across markdown,
    HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    EPC_SECTIONS: List[str] = [
        "Building Details",
        "EPC Rating",
        "Energy Use",
        "CO2 Emissions",
        "Cost-Effective Recommendations",
        "Further Recommendations",
        "Estimated Costs",
        "Green Deal Eligibility",
        "Assessor Details",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EPCReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render EPC report as Markdown.

        Args:
            data: EPC assessment data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_building_details(data),
            self._md_epc_rating(data),
            self._md_energy_use(data),
            self._md_co2_emissions(data),
            self._md_cost_effective_recommendations(data),
            self._md_further_recommendations(data),
            self._md_estimated_costs(data),
            self._md_green_deal(data),
            self._md_assessor_details(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render EPC report as self-contained HTML.

        Args:
            data: EPC assessment data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_building_details(data),
            self._html_epc_rating(data),
            self._html_energy_use(data),
            self._html_co2_emissions(data),
            self._html_cost_effective_recommendations(data),
            self._html_further_recommendations(data),
            self._html_estimated_costs(data),
            self._html_green_deal(data),
            self._html_assessor_details(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Performance Certificate (EPC)</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render EPC report as structured JSON.

        Args:
            data: EPC assessment data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "epc_report",
            "version": "32.0.0",
            "standard": "EPBD",
            "generated_at": self.generated_at.isoformat(),
            "building": self._json_building(data),
            "epc_rating": self._json_epc_rating(data),
            "energy_use": self._json_energy_use(data),
            "co2_emissions": self._json_co2(data),
            "cost_effective_recommendations": data.get("cost_effective_recommendations", []),
            "further_recommendations": data.get("further_recommendations", []),
            "estimated_costs": self._json_estimated_costs(data),
            "green_deal": data.get("green_deal", {}),
            "assessor": data.get("assessor", {}),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with EPC metadata."""
        address = data.get("address", "")
        ref = data.get("certificate_reference", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        valid_until = data.get("valid_until", "")
        return (
            "# Energy Performance Certificate (EPC)\n\n"
            f"**Address:** {address}  \n"
            f"**Certificate Reference:** {ref}  \n"
            f"**Valid Until:** {valid_until}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 EPCReportTemplate v32.0.0\n\n---"
        )

    def _md_building_details(self, data: Dict[str, Any]) -> str:
        """Render building details section."""
        bld = data.get("building", {})
        return (
            "## 1. Building Details\n\n"
            f"| Property | Value |\n|----------|-------|\n"
            f"| Type | {bld.get('type', '-')} |\n"
            f"| Detachment | {bld.get('detachment', '-')} |\n"
            f"| Floor Area | {self._fmt(bld.get('floor_area_sqm', 0), 0)} m2 |\n"
            f"| Year Built | {bld.get('year_built', '-')} |\n"
            f"| Number of Floors | {bld.get('num_floors', '-')} |\n"
            f"| Number of Habitable Rooms | {bld.get('habitable_rooms', '-')} |\n"
            f"| Heating System | {bld.get('heating_system', '-')} |\n"
            f"| Heating Fuel | {bld.get('heating_fuel', '-')} |\n"
            f"| Wall Construction | {bld.get('wall_construction', '-')} |\n"
            f"| Roof Type | {bld.get('roof_type', '-')} |\n"
            f"| Glazing | {bld.get('glazing', '-')} |"
        )

    def _md_epc_rating(self, data: Dict[str, Any]) -> str:
        """Render EPC rating with A-G band visual."""
        rating = data.get("epc_rating", {})
        current_score = rating.get("current_score", 0)
        current_band = rating.get("current_band", "G")
        potential_score = rating.get("potential_score", 0)
        potential_band = rating.get("potential_band", "G")
        lines = [
            "## 2. Energy Performance Rating\n",
            f"**Current Rating:** {current_band} ({current_score})  ",
            f"**Potential Rating:** {potential_band} ({potential_score})\n",
            "### Rating Scale\n",
            "| Band | Range | Current | Potential |",
            "|------|-------|---------|-----------|",
        ]
        for b in EPC_BANDS:
            cur = " <-- " if b["band"] == current_band else ""
            pot = " <-- " if b["band"] == potential_band else ""
            lines.append(
                f"| **{b['band']}** | {b['min']}-{b['max']} | {cur} | {pot} |"
            )
        return "\n".join(lines)

    def _md_energy_use(self, data: Dict[str, Any]) -> str:
        """Render energy use breakdown section."""
        energy = data.get("energy_use", {})
        breakdown = energy.get("breakdown", [])
        lines = [
            "## 3. Energy Use\n",
            f"**Energy Use Intensity:** {self._fmt(energy.get('eui_kwh_m2', 0))} kWh/m2/yr  ",
            f"**Total Energy Use:** {self._fmt(energy.get('total_kwh', 0), 0)} kWh/yr  ",
            f"**Space Heating Demand:** {self._fmt(energy.get('heating_demand_kwh_m2', 0))} kWh/m2/yr  ",
            f"**Hot Water Demand:** {self._fmt(energy.get('hot_water_demand_kwh_m2', 0))} kWh/m2/yr  ",
            f"**Lighting Demand:** {self._fmt(energy.get('lighting_demand_kwh_m2', 0))} kWh/m2/yr",
        ]
        if breakdown:
            lines.extend([
                "\n### Energy Breakdown\n",
                "| End Use | kWh/yr | kWh/m2/yr | Share (%) |",
                "|---------|--------|-----------|-----------|",
            ])
            for b in breakdown:
                lines.append(
                    f"| {b.get('end_use', '-')} "
                    f"| {self._fmt(b.get('kwh_yr', 0), 0)} "
                    f"| {self._fmt(b.get('kwh_m2_yr', 0))} "
                    f"| {self._fmt(b.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_co2_emissions(self, data: Dict[str, Any]) -> str:
        """Render CO2 emissions section."""
        co2 = data.get("co2_emissions", {})
        breakdown = co2.get("breakdown", [])
        lines = [
            "## 4. CO2 Emissions\n",
            f"**Total CO2 Emissions:** {self._fmt(co2.get('total_kg_co2_yr', 0), 0)} kg CO2/yr  ",
            f"**CO2 Intensity:** {self._fmt(co2.get('kg_co2_m2_yr', 0))} kg CO2/m2/yr  ",
            f"**Current vs Typical:** {co2.get('vs_typical', '-')}",
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

    def _md_cost_effective_recommendations(self, data: Dict[str, Any]) -> str:
        """Render cost-effective recommendations section."""
        recs = data.get("cost_effective_recommendations", [])
        if not recs:
            return "## 5. Cost-Effective Recommendations\n\n_No cost-effective recommendations._"
        lines = [
            "## 5. Cost-Effective Recommendations\n",
            "These recommendations are considered cost-effective under current conditions.\n",
            "| # | Measure | Typical Cost | Annual Saving | Rating After |",
            "|---|---------|-------------|---------------|-------------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('measure', '-')} "
                f"| {r.get('typical_cost', '-')} "
                f"| {r.get('annual_saving', '-')} "
                f"| {r.get('rating_after', '-')} |"
            )
        return "\n".join(lines)

    def _md_further_recommendations(self, data: Dict[str, Any]) -> str:
        """Render further recommendations section."""
        recs = data.get("further_recommendations", [])
        if not recs:
            return "## 6. Further Recommendations\n\n_No further recommendations._"
        lines = [
            "## 6. Further Recommendations\n",
            "These recommendations may require higher upfront investment.\n",
            "| # | Measure | Typical Cost | Annual Saving | Rating After |",
            "|---|---------|-------------|---------------|-------------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('measure', '-')} "
                f"| {r.get('typical_cost', '-')} "
                f"| {r.get('annual_saving', '-')} "
                f"| {r.get('rating_after', '-')} |"
            )
        return "\n".join(lines)

    def _md_estimated_costs(self, data: Dict[str, Any]) -> str:
        """Render estimated energy costs section."""
        costs = data.get("estimated_costs", {})
        return (
            "## 7. Estimated Energy Costs\n\n"
            f"| Period | Lighting | Heating | Hot Water | Total |\n"
            f"|--------|----------|---------|-----------|-------|\n"
            f"| Current (GBP/yr) "
            f"| {costs.get('current_lighting', '-')} "
            f"| {costs.get('current_heating', '-')} "
            f"| {costs.get('current_hot_water', '-')} "
            f"| {costs.get('current_total', '-')} |\n"
            f"| Potential (GBP/yr) "
            f"| {costs.get('potential_lighting', '-')} "
            f"| {costs.get('potential_heating', '-')} "
            f"| {costs.get('potential_hot_water', '-')} "
            f"| {costs.get('potential_total', '-')} |\n"
            f"| Savings (GBP/yr) "
            f"| {costs.get('saving_lighting', '-')} "
            f"| {costs.get('saving_heating', '-')} "
            f"| {costs.get('saving_hot_water', '-')} "
            f"| {costs.get('saving_total', '-')} |"
        )

    def _md_green_deal(self, data: Dict[str, Any]) -> str:
        """Render Green Deal eligibility section."""
        gd = data.get("green_deal", {})
        eligible = gd.get("eligible", False)
        measures = gd.get("eligible_measures", [])
        status = "Eligible" if eligible else "Not Eligible"
        lines = [
            "## 8. Green Deal Eligibility\n",
            f"**Status:** {status}  ",
            f"**Estimated Finance Available:** {gd.get('finance_available', '-')}  ",
            f"**Golden Rule Met:** {'Yes' if gd.get('golden_rule_met', False) else 'No'}",
        ]
        if measures:
            lines.append("\n### Eligible Measures\n")
            for m in measures:
                lines.append(
                    f"- **{m.get('measure', '-')}**: "
                    f"Saving {m.get('annual_saving', '-')}/yr, "
                    f"Cost {m.get('install_cost', '-')}, "
                    f"Payback {m.get('payback_years', '-')} years"
                )
        return "\n".join(lines)

    def _md_assessor_details(self, data: Dict[str, Any]) -> str:
        """Render assessor details section."""
        assessor = data.get("assessor", {})
        return (
            "## 9. Assessor Details\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Name | {assessor.get('name', '-')} |\n"
            f"| Accreditation Number | {assessor.get('accreditation_number', '-')} |\n"
            f"| Accreditation Scheme | {assessor.get('accreditation_scheme', '-')} |\n"
            f"| Company | {assessor.get('company', '-')} |\n"
            f"| Phone | {assessor.get('phone', '-')} |\n"
            f"| Email | {assessor.get('email', '-')} |\n"
            f"| Assessment Date | {assessor.get('assessment_date', '-')} |\n"
            f"| Related Party Disclosure | {assessor.get('related_party', 'No')} |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 EPCReportTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        address = data.get("address", "")
        ref = data.get("certificate_reference", "")
        valid_until = data.get("valid_until", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Energy Performance Certificate (EPC)</h1>\n'
            f'<p class="subtitle">Address: {address} | Ref: {ref} | '
            f'Valid Until: {valid_until} | Generated: {ts}</p>'
        )

    def _html_building_details(self, data: Dict[str, Any]) -> str:
        """Render HTML building details table."""
        bld = data.get("building", {})
        rows = ""
        fields = [
            ("Type", bld.get("type", "-")),
            ("Floor Area", f"{self._fmt(bld.get('floor_area_sqm', 0), 0)} m2"),
            ("Year Built", bld.get("year_built", "-")),
            ("Floors", bld.get("num_floors", "-")),
            ("Heating System", bld.get("heating_system", "-")),
            ("Heating Fuel", bld.get("heating_fuel", "-")),
            ("Wall Construction", bld.get("wall_construction", "-")),
            ("Glazing", bld.get("glazing", "-")),
        ]
        for label, val in fields:
            rows += f'<tr><td>{label}</td><td>{val}</td></tr>\n'
        return (
            '<h2>Building Details</h2>\n'
            f'<table>\n<tr><th>Property</th><th>Value</th></tr>\n{rows}</table>'
        )

    def _html_epc_rating(self, data: Dict[str, Any]) -> str:
        """Render HTML EPC rating with color-coded bands."""
        rating = data.get("epc_rating", {})
        current_band = rating.get("current_band", "G")
        current_score = rating.get("current_score", 0)
        potential_band = rating.get("potential_band", "G")
        potential_score = rating.get("potential_score", 0)
        bands_html = ""
        for b in EPC_BANDS:
            is_current = b["band"] == current_band
            is_potential = b["band"] == potential_band
            marker = ""
            if is_current:
                marker += f' <strong>[Current: {current_score}]</strong>'
            if is_potential:
                marker += f' <em>[Potential: {potential_score}]</em>'
            width = max(30, 100 - (EPC_BANDS.index(b) * 8))
            bands_html += (
                f'<div class="epc-band" style="background:{b["color"]};'
                f'width:{width}%;padding:6px 12px;margin:2px 0;color:#fff;'
                f'font-weight:bold;">{b["band"]} ({b["min"]}-{b["max"]}){marker}</div>\n'
            )
        return (
            '<h2>Energy Performance Rating</h2>\n'
            f'<div class="epc-scale">\n{bands_html}</div>'
        )

    def _html_energy_use(self, data: Dict[str, Any]) -> str:
        """Render HTML energy use section."""
        energy = data.get("energy_use", {})
        breakdown = energy.get("breakdown", [])
        rows = ""
        for b in breakdown:
            rows += (
                f'<tr><td>{b.get("end_use", "-")}</td>'
                f'<td>{self._fmt(b.get("kwh_yr", 0), 0)}</td>'
                f'<td>{self._fmt(b.get("kwh_m2_yr", 0))}</td>'
                f'<td>{self._fmt(b.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Energy Use</h2>\n'
            f'<p>EUI: {self._fmt(energy.get("eui_kwh_m2", 0))} kWh/m2/yr | '
            f'Total: {self._fmt(energy.get("total_kwh", 0), 0)} kWh/yr</p>\n'
            '<table>\n<tr><th>End Use</th><th>kWh/yr</th><th>kWh/m2/yr</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_co2_emissions(self, data: Dict[str, Any]) -> str:
        """Render HTML CO2 emissions section."""
        co2 = data.get("co2_emissions", {})
        breakdown = co2.get("breakdown", [])
        rows = ""
        for b in breakdown:
            rows += (
                f'<tr><td>{b.get("source", "-")}</td>'
                f'<td>{self._fmt(b.get("kg_co2_yr", 0), 0)}</td>'
                f'<td>{self._fmt(b.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>CO2 Emissions</h2>\n'
            f'<p>Total: {self._fmt(co2.get("total_kg_co2_yr", 0), 0)} kg CO2/yr | '
            f'Intensity: {self._fmt(co2.get("kg_co2_m2_yr", 0))} kg CO2/m2/yr</p>\n'
            '<table>\n<tr><th>Source</th><th>kg CO2/yr</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_cost_effective_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML cost-effective recommendations."""
        recs = data.get("cost_effective_recommendations", [])
        rows = ""
        for i, r in enumerate(recs, 1):
            rows += (
                f'<tr><td>{i}</td><td>{r.get("measure", "-")}</td>'
                f'<td>{r.get("typical_cost", "-")}</td>'
                f'<td>{r.get("annual_saving", "-")}</td>'
                f'<td>{r.get("rating_after", "-")}</td></tr>\n'
            )
        return (
            '<h2>Cost-Effective Recommendations</h2>\n'
            '<table>\n<tr><th>#</th><th>Measure</th><th>Cost</th>'
            f'<th>Annual Saving</th><th>Rating After</th></tr>\n{rows}</table>'
        )

    def _html_further_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML further recommendations."""
        recs = data.get("further_recommendations", [])
        rows = ""
        for i, r in enumerate(recs, 1):
            rows += (
                f'<tr><td>{i}</td><td>{r.get("measure", "-")}</td>'
                f'<td>{r.get("typical_cost", "-")}</td>'
                f'<td>{r.get("annual_saving", "-")}</td>'
                f'<td>{r.get("rating_after", "-")}</td></tr>\n'
            )
        return (
            '<h2>Further Recommendations</h2>\n'
            '<table>\n<tr><th>#</th><th>Measure</th><th>Cost</th>'
            f'<th>Annual Saving</th><th>Rating After</th></tr>\n{rows}</table>'
        )

    def _html_estimated_costs(self, data: Dict[str, Any]) -> str:
        """Render HTML estimated costs."""
        costs = data.get("estimated_costs", {})
        return (
            '<h2>Estimated Energy Costs</h2>\n'
            '<table>\n<tr><th>Period</th><th>Lighting</th><th>Heating</th>'
            '<th>Hot Water</th><th>Total</th></tr>\n'
            f'<tr><td>Current</td><td>{costs.get("current_lighting", "-")}</td>'
            f'<td>{costs.get("current_heating", "-")}</td>'
            f'<td>{costs.get("current_hot_water", "-")}</td>'
            f'<td>{costs.get("current_total", "-")}</td></tr>\n'
            f'<tr><td>Potential</td><td>{costs.get("potential_lighting", "-")}</td>'
            f'<td>{costs.get("potential_heating", "-")}</td>'
            f'<td>{costs.get("potential_hot_water", "-")}</td>'
            f'<td>{costs.get("potential_total", "-")}</td></tr>\n'
            '</table>'
        )

    def _html_green_deal(self, data: Dict[str, Any]) -> str:
        """Render HTML Green Deal eligibility."""
        gd = data.get("green_deal", {})
        eligible = gd.get("eligible", False)
        status = "Eligible" if eligible else "Not Eligible"
        measures = gd.get("eligible_measures", [])
        items = ""
        for m in measures:
            items += (
                f'<li><strong>{m.get("measure", "-")}</strong>: '
                f'Saving {m.get("annual_saving", "-")}/yr, '
                f'Cost {m.get("install_cost", "-")}</li>\n'
            )
        return (
            '<h2>Green Deal Eligibility</h2>\n'
            f'<p>Status: <strong>{status}</strong> | '
            f'Golden Rule: {"Met" if gd.get("golden_rule_met", False) else "Not Met"}</p>\n'
            f'<ul>\n{items}</ul>'
        )

    def _html_assessor_details(self, data: Dict[str, Any]) -> str:
        """Render HTML assessor details."""
        assessor = data.get("assessor", {})
        return (
            '<h2>Assessor Details</h2>\n'
            '<table>\n'
            f'<tr><td>Name</td><td>{assessor.get("name", "-")}</td></tr>\n'
            f'<tr><td>Accreditation</td><td>{assessor.get("accreditation_number", "-")}</td></tr>\n'
            f'<tr><td>Scheme</td><td>{assessor.get("accreditation_scheme", "-")}</td></tr>\n'
            f'<tr><td>Company</td><td>{assessor.get("company", "-")}</td></tr>\n'
            f'<tr><td>Assessment Date</td><td>{assessor.get("assessment_date", "-")}</td></tr>\n'
            '</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_building(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON building details."""
        return data.get("building", {})

    def _json_epc_rating(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON EPC rating data."""
        rating = data.get("epc_rating", {})
        return {
            "current_score": rating.get("current_score", 0),
            "current_band": rating.get("current_band", "G"),
            "potential_score": rating.get("potential_score", 0),
            "potential_band": rating.get("potential_band", "G"),
            "bands": EPC_BANDS,
        }

    def _json_energy_use(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON energy use data."""
        return data.get("energy_use", {})

    def _json_co2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON CO2 emissions data."""
        return data.get("co2_emissions", {})

    def _json_estimated_costs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON estimated costs data."""
        return data.get("estimated_costs", {})

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
            ".epc-band{border-radius:4px;font-size:0.95em;}"
            ".epc-scale{margin:15px 0;}"
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
