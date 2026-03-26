# -*- coding: utf-8 -*-
"""
ESRSE1DisclosureTemplate - ESRS E1 Climate Change Disclosure for PACK-041.

Generates a comprehensive ESRS E1 Climate Change disclosure report covering
E1-1 transition plan summary, E1-4 GHG emission reduction targets, E1-5
energy consumption and mix, E1-6 Scope 1/2/3 GHG emissions in detailed
EFRAG format, biogenic CO2 separate reporting, and methodology notes per
ESRS requirements. JSON output includes XBRL-ready data structure.

Sections:
    1. E1-1: Transition Plan Summary
    2. E1-4: GHG Emission Reduction Targets
    3. E1-5: Energy Consumption and Mix
    4. E1-6: Scope 1, 2, 3 GHG Emissions
    5. Biogenic CO2 Emissions
    6. Methodology Notes (ESRS)
    7. Disclosure Requirements Checklist

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with XBRL-ready data)

Regulatory References:
    - ESRS E1 Climate Change (EFRAG, 2023)
    - Delegated Regulation (EU) 2023/2772
    - ESRS XBRL Taxonomy

Author: GreenLang Team
Version: 41.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "41.0.0"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M tCO2e"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K tCO2e"
    return f"{value:,.1f} tCO2e"


def _fmt_mwh(value: Optional[float]) -> str:
    """Format MWh with scale suffix."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M MWh"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K MWh"
    return f"{value:,.0f} MWh"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


class ESRSE1DisclosureTemplate:
    """
    ESRS E1 Climate Change disclosure template.

    Renders comprehensive ESRS E1 disclosures covering transition plan,
    emission reduction targets, energy consumption and mix, Scope 1/2/3
    emissions per EFRAG format, biogenic CO2, and methodology notes.
    JSON output includes XBRL-ready data structures. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = ESRSE1DisclosureTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSE1DisclosureTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ESRS E1 disclosure as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_e1_1_transition_plan(data),
            self._md_e1_4_reduction_targets(data),
            self._md_e1_5_energy_consumption(data),
            self._md_e1_6_ghg_emissions(data),
            self._md_biogenic_co2(data),
            self._md_methodology_notes(data),
            self._md_disclosure_checklist(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESRS E1 disclosure as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_e1_1_transition_plan(data),
            self._html_e1_4_reduction_targets(data),
            self._html_e1_5_energy_consumption(data),
            self._html_e1_6_ghg_emissions(data),
            self._html_biogenic_co2(data),
            self._html_methodology_notes(data),
            self._html_disclosure_checklist(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESRS E1 disclosure as JSON with XBRL-ready structure."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "esrs_e1_disclosure",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "e1_1_transition_plan": data.get("e1_1_transition_plan", {}),
            "e1_4_reduction_targets": data.get("e1_4_reduction_targets", []),
            "e1_5_energy": data.get("e1_5_energy", {}),
            "e1_6_emissions": data.get("e1_6_emissions", {}),
            "biogenic_co2": data.get("biogenic_co2", {}),
            "methodology": data.get("methodology", {}),
            "disclosure_checklist": data.get("disclosure_checklist", []),
            "xbrl_data": self._build_xbrl_data(data),
        }

    # ==================================================================
    # XBRL DATA BUILDER
    # ==================================================================

    def _build_xbrl_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build XBRL-ready data structure for E1 disclosure.

        Args:
            data: Full disclosure data.

        Returns:
            XBRL-tagged data dictionary.
        """
        e1_6 = data.get("e1_6_emissions", {})
        e1_5 = data.get("e1_5_energy", {})
        biogenic = data.get("biogenic_co2", {})
        return {
            "taxonomy": "ESRS_2023",
            "standard": "E1",
            "data_points": [
                {
                    "xbrl_tag": "esrs:GrossScope1GHGEmissions",
                    "value": e1_6.get("scope1_total_tco2e"),
                    "unit": "tCO2e",
                    "period": self._get_val(data, "reporting_year"),
                },
                {
                    "xbrl_tag": "esrs:GrossScope2LocationBasedGHGEmissions",
                    "value": e1_6.get("scope2_location_tco2e"),
                    "unit": "tCO2e",
                    "period": self._get_val(data, "reporting_year"),
                },
                {
                    "xbrl_tag": "esrs:GrossScope2MarketBasedGHGEmissions",
                    "value": e1_6.get("scope2_market_tco2e"),
                    "unit": "tCO2e",
                    "period": self._get_val(data, "reporting_year"),
                },
                {
                    "xbrl_tag": "esrs:TotalGHGEmissionsLocationBased",
                    "value": e1_6.get("total_location_tco2e"),
                    "unit": "tCO2e",
                    "period": self._get_val(data, "reporting_year"),
                },
                {
                    "xbrl_tag": "esrs:TotalGHGEmissionsMarketBased",
                    "value": e1_6.get("total_market_tco2e"),
                    "unit": "tCO2e",
                    "period": self._get_val(data, "reporting_year"),
                },
                {
                    "xbrl_tag": "esrs:BiogenicCO2EmissionsScope1",
                    "value": biogenic.get("scope1_biogenic_tco2"),
                    "unit": "tCO2",
                    "period": self._get_val(data, "reporting_year"),
                },
                {
                    "xbrl_tag": "esrs:TotalEnergyConsumption",
                    "value": e1_5.get("total_energy_mwh"),
                    "unit": "MWh",
                    "period": self._get_val(data, "reporting_year"),
                },
                {
                    "xbrl_tag": "esrs:ShareOfRenewableEnergy",
                    "value": e1_5.get("renewable_share_pct"),
                    "unit": "%",
                    "period": self._get_val(data, "reporting_year"),
                },
            ],
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# ESRS E1 Climate Change Disclosure - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Standard:** ESRS E1 (Climate Change) | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_e1_1_transition_plan(self, data: Dict[str, Any]) -> str:
        """Render Markdown E1-1 transition plan summary."""
        tp = data.get("e1_1_transition_plan", {})
        if not tp:
            return "## E1-1: Transition Plan\n\nNo transition plan data provided."
        lines = [
            "## E1-1: Transition Plan for Climate Change Mitigation",
            "",
            "| Aspect | Detail |",
            "|--------|--------|",
            f"| Has Transition Plan | {'Yes' if tp.get('has_plan') else 'No'} |",
            f"| Aligned With | {tp.get('alignment', 'Paris Agreement 1.5C')} |",
            f"| Approved By | {tp.get('approved_by', '-')} |",
            f"| Date Adopted | {tp.get('date_adopted', '-')} |",
            f"| Review Cycle | {tp.get('review_cycle', 'Annual')} |",
        ]
        summary = tp.get("summary", "")
        if summary:
            lines.append(f"\n**Summary:** {summary}")
        key_actions = tp.get("key_actions", [])
        if key_actions:
            lines.append("\n**Key Actions:**")
            for act in key_actions:
                timeline = act.get("timeline", "")
                lines.append(f"- {act.get('action', '')} ({timeline})")
        return "\n".join(lines)

    def _md_e1_4_reduction_targets(self, data: Dict[str, Any]) -> str:
        """Render Markdown E1-4 GHG emission reduction targets."""
        targets = data.get("e1_4_reduction_targets", [])
        if not targets:
            return "## E1-4: GHG Emission Reduction Targets\n\nNo reduction targets set."
        lines = [
            "## E1-4: GHG Emission Reduction Targets",
            "",
            "| Target | Scope | Base Year | Base Emissions | Target Year | Target % | Actual % | On Track |",
            "|--------|-------|----------|---------------|------------|---------|---------|---------|",
        ]
        for tgt in targets:
            name = tgt.get("target_name", "")
            scope = tgt.get("scope_coverage", "1+2")
            base_yr = tgt.get("base_year", "-")
            base_em = _fmt_tco2e(tgt.get("base_year_emissions"))
            tgt_yr = tgt.get("target_year", "-")
            tgt_pct = _fmt_pct(tgt.get("target_reduction_pct"))
            actual_pct = _fmt_pct(tgt.get("actual_reduction_pct"))
            on_track = "Yes" if tgt.get("on_track") else "No"
            lines.append(f"| {name} | {scope} | {base_yr} | {base_em} | {tgt_yr} | {tgt_pct} | {actual_pct} | **{on_track}** |")
        return "\n".join(lines)

    def _md_e1_5_energy_consumption(self, data: Dict[str, Any]) -> str:
        """Render Markdown E1-5 energy consumption and mix."""
        energy = data.get("e1_5_energy", {})
        if not energy:
            return "## E1-5: Energy Consumption and Mix\n\nNo energy data provided."
        total = energy.get("total_energy_mwh", 0.0)
        renewable = energy.get("renewable_mwh", 0.0)
        fossil = energy.get("fossil_mwh", 0.0)
        nuclear = energy.get("nuclear_mwh", 0.0)
        re_pct = energy.get("renewable_share_pct", 0.0)
        lines = [
            "## E1-5: Energy Consumption and Mix",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Energy Consumption | {_fmt_mwh(total)} |",
            f"| From Renewable Sources | {_fmt_mwh(renewable)} ({_fmt_pct(re_pct)}) |",
            f"| From Fossil Sources | {_fmt_mwh(fossil)} |",
            f"| From Nuclear Sources | {_fmt_mwh(nuclear)} |",
        ]
        # Energy mix breakdown
        mix = energy.get("energy_mix", [])
        if mix:
            lines.extend([
                "",
                "### Energy Mix Breakdown",
                "",
                "| Source | MWh | % of Total | Renewable |",
                "|--------|-----|-----------|-----------|",
            ])
            for src in mix:
                name = src.get("source", "")
                mwh = _fmt_mwh(src.get("mwh"))
                pct = _fmt_pct(src.get("pct_of_total"))
                is_re = "Yes" if src.get("is_renewable") else "No"
                lines.append(f"| {name} | {mwh} | {pct} | {is_re} |")
        # Intensity
        intensity = energy.get("energy_intensity", {})
        if intensity:
            lines.extend([
                "",
                f"**Energy Intensity:** {intensity.get('value', 'N/A')} {intensity.get('unit', 'MWh/EUR M revenue')}",
            ])
        return "\n".join(lines)

    def _md_e1_6_ghg_emissions(self, data: Dict[str, Any]) -> str:
        """Render Markdown E1-6 Scope 1/2/3 GHG emissions per EFRAG format."""
        e1_6 = data.get("e1_6_emissions", {})
        if not e1_6:
            return "## E1-6: Gross Scopes 1, 2, 3 GHG Emissions\n\nNo emissions data."
        s1 = e1_6.get("scope1_total_tco2e", 0.0)
        s2_loc = e1_6.get("scope2_location_tco2e", 0.0)
        s2_mkt = e1_6.get("scope2_market_tco2e", 0.0)
        s3 = e1_6.get("scope3_total_tco2e", 0.0)
        total_loc = e1_6.get("total_location_tco2e", s1 + s2_loc + s3)
        total_mkt = e1_6.get("total_market_tco2e", s1 + s2_mkt + s3)
        lines = [
            "## E1-6: Gross Scopes 1, 2, 3 and Total GHG Emissions",
            "",
            "### GHG Emissions Summary (EFRAG Format)",
            "",
            "| Scope | Method | Gross Emissions (tCO2e) | % of Total |",
            "|-------|--------|------------------------|-----------|",
            f"| Scope 1 | Direct | {_fmt_tco2e(s1)} | {_fmt_pct((s1/total_loc)*100 if total_loc else 0)} |",
            f"| Scope 2 | Location-Based | {_fmt_tco2e(s2_loc)} | {_fmt_pct((s2_loc/total_loc)*100 if total_loc else 0)} |",
            f"| Scope 2 | Market-Based | {_fmt_tco2e(s2_mkt)} | - |",
            f"| Scope 3 | Indirect Value Chain | {_fmt_tco2e(s3)} | {_fmt_pct((s3/total_loc)*100 if total_loc else 0)} |",
            f"| **Total** | **Location-Based** | **{_fmt_tco2e(total_loc)}** | **100%** |",
            f"| **Total** | **Market-Based** | **{_fmt_tco2e(total_mkt)}** | - |",
        ]
        # Scope 1 by category
        s1_categories = e1_6.get("scope1_by_category", [])
        if s1_categories:
            lines.extend([
                "",
                "### Scope 1 by Category",
                "",
                "| Category | tCO2e | % of Scope 1 |",
                "|----------|-------|-------------|",
            ])
            for cat in s1_categories:
                name = cat.get("category", "")
                em = _fmt_tco2e(cat.get("emissions_tco2e"))
                pct = _fmt_pct(cat.get("pct_of_scope1"))
                lines.append(f"| {name} | {em} | {pct} |")
        # Scope 3 by category
        s3_categories = e1_6.get("scope3_by_category", [])
        if s3_categories:
            lines.extend([
                "",
                "### Scope 3 by Category",
                "",
                "| Category | Name | tCO2e | Method | Relevant |",
                "|----------|------|-------|--------|---------|",
            ])
            for cat in s3_categories:
                num = cat.get("category_number", "")
                name = cat.get("category_name", "")
                em = _fmt_tco2e(cat.get("emissions_tco2e"))
                method = cat.get("calculation_method", "-")
                relevant = "Yes" if cat.get("is_relevant", True) else "No"
                lines.append(f"| {num} | {name} | {em} | {method} | {relevant} |")
        # GHG intensity
        intensity = e1_6.get("ghg_intensity", {})
        if intensity:
            lines.extend([
                "",
                f"**GHG Intensity:** {intensity.get('value', 'N/A')} {intensity.get('unit', 'tCO2e/EUR M')}",
            ])
        return "\n".join(lines)

    def _md_biogenic_co2(self, data: Dict[str, Any]) -> str:
        """Render Markdown biogenic CO2 emissions (reported separately per ESRS)."""
        bio = data.get("biogenic_co2", {})
        if not bio:
            return "## Biogenic CO2 Emissions\n\nNo biogenic CO2 data."
        lines = [
            "## Biogenic CO2 Emissions (Reported Separately)",
            "",
            "*Per ESRS E1-6 para 44(i), biogenic CO2 emissions are reported separately.*",
            "",
            "| Source | Scope | Biogenic CO2 (tCO2) | Notes |",
            "|--------|-------|-------------------|-------|",
        ]
        sources = bio.get("sources", [])
        for src in sources:
            name = src.get("source", "")
            scope = src.get("scope", "1")
            em = f"{src.get('biogenic_tco2', 0):,.1f} tCO2"
            notes = src.get("notes", "-")
            lines.append(f"| {name} | Scope {scope} | {em} | {notes} |")
        total = bio.get("total_biogenic_tco2", 0.0)
        lines.append(f"\n**Total Biogenic CO2:** {total:,.1f} tCO2")
        return "\n".join(lines)

    def _md_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology notes per ESRS requirements."""
        meth = data.get("methodology", {})
        if not meth:
            return "## Methodology Notes\n\nNo methodology notes."
        lines = [
            "## Methodology Notes (ESRS E1)",
            "",
            "| Aspect | Detail |",
            "|--------|--------|",
            f"| GHG Protocol Standard | {meth.get('ghg_protocol_standard', 'Corporate Standard')} |",
            f"| GWP Basis | {meth.get('gwp_basis', 'AR6 100-year')} |",
            f"| Consolidation Approach | {meth.get('consolidation_approach', 'Operational Control')} |",
            f"| Base Year | {meth.get('base_year', '-')} |",
            f"| Significant Changes | {meth.get('significant_changes', 'None')} |",
            f"| Scope 2 Primary Method | {meth.get('scope2_primary_method', 'Location-Based')} |",
            f"| Scope 3 Screening | {meth.get('scope3_screening_method', 'Spend-based + supplier-specific')} |",
        ]
        additional = meth.get("additional_notes", "")
        if additional:
            lines.append(f"\n{additional}")
        return "\n".join(lines)

    def _md_disclosure_checklist(self, data: Dict[str, Any]) -> str:
        """Render Markdown E1 disclosure requirements checklist."""
        checklist = data.get("disclosure_checklist", [])
        if not checklist:
            return ""
        lines = [
            "## Disclosure Requirements Checklist",
            "",
            "| DR | Requirement | Status | Reference |",
            "|----|-----------|--------|-----------|",
        ]
        for item in checklist:
            dr = item.get("disclosure_requirement", "")
            req = item.get("requirement_name", "")
            status = item.get("status", "N/A")
            ref = item.get("reference", "-")
            lines.append(f"| {dr} | {req} | **{status}** | {ref} |")
        met = sum(1 for i in checklist if i.get("status") == "Met")
        total = len(checklist)
        lines.append(f"\n**Completion:** {met}/{total} requirements met")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-041 Scope 1-2 Complete v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>ESRS E1 Disclosure - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #003f88;padding-bottom:0.5rem;}\n"
            "h2{color:#003f88;margin-top:2rem;border-bottom:2px solid #003f88;padding-bottom:0.3rem;}\n"
            "h3{color:#00509d;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#e8f0fe;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".total-row{font-weight:bold;background:#e0e8f0;}\n"
            ".met{color:#2a9d8f;font-weight:600;}\n"
            ".partial{color:#e9c46a;font-weight:600;}\n"
            ".not-met{color:#e63946;font-weight:600;}\n"
            ".esrs-tag{font-size:0.75rem;color:#003f88;background:#e8f0fe;padding:0.1rem 0.4rem;"
            "border-radius:3px;font-family:monospace;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>ESRS E1 &mdash; Climate Change Disclosure &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            "<strong>Standard:</strong> ESRS E1</p>\n<hr>\n</div>"
        )

    def _html_e1_1_transition_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML E1-1 transition plan."""
        tp = data.get("e1_1_transition_plan", {})
        if not tp:
            return ""
        has_plan = "Yes" if tp.get("has_plan") else "No"
        summary = tp.get("summary", "")
        actions = tp.get("key_actions", [])
        action_html = ""
        if actions:
            items = "".join(f"<li>{a.get('action', '')} ({a.get('timeline', '')})</li>" for a in actions)
            action_html = f"<h3>Key Actions</h3><ul>{items}</ul>"
        return (
            '<div class="section">\n'
            "<h2>E1-1: Transition Plan</h2>\n"
            "<table><thead><tr><th>Aspect</th><th>Detail</th></tr></thead>\n<tbody>"
            f"<tr><td>Has Plan</td><td>{has_plan}</td></tr>\n"
            f"<tr><td>Alignment</td><td>{tp.get('alignment', 'Paris 1.5C')}</td></tr>\n"
            f"<tr><td>Approved By</td><td>{tp.get('approved_by', '-')}</td></tr>\n"
            "</tbody></table>\n"
            f"<p>{summary}</p>\n{action_html}\n</div>"
        )

    def _html_e1_4_reduction_targets(self, data: Dict[str, Any]) -> str:
        """Render HTML E1-4 reduction targets."""
        targets = data.get("e1_4_reduction_targets", [])
        if not targets:
            return ""
        rows = ""
        for tgt in targets:
            on_track = tgt.get("on_track")
            css = "met" if on_track else "not-met"
            label = "On Track" if on_track else "Off Track"
            rows += (
                f"<tr><td>{tgt.get('target_name', '')}</td>"
                f"<td>{tgt.get('scope_coverage', '1+2')}</td>"
                f"<td>{tgt.get('base_year', '-')}</td>"
                f"<td>{tgt.get('target_year', '-')}</td>"
                f"<td>{_fmt_pct(tgt.get('target_reduction_pct'))}</td>"
                f"<td>{_fmt_pct(tgt.get('actual_reduction_pct'))}</td>"
                f'<td class="{css}">{label}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>E1-4: Reduction Targets</h2>\n'
            "<table><thead><tr><th>Target</th><th>Scope</th><th>Base</th>"
            "<th>Target Yr</th><th>Target %</th><th>Actual %</th>"
            "<th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_e1_5_energy_consumption(self, data: Dict[str, Any]) -> str:
        """Render HTML E1-5 energy consumption."""
        energy = data.get("e1_5_energy", {})
        if not energy:
            return ""
        total = energy.get("total_energy_mwh", 0.0)
        renewable = energy.get("renewable_mwh", 0.0)
        re_pct = energy.get("renewable_share_pct", 0.0)
        return (
            '<div class="section">\n<h2>E1-5: Energy Consumption</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>Total Energy</td><td>{_fmt_mwh(total)}</td></tr>\n"
            f"<tr><td>Renewable</td><td>{_fmt_mwh(renewable)} ({_fmt_pct(re_pct)})</td></tr>\n"
            f"<tr><td>Fossil</td><td>{_fmt_mwh(energy.get('fossil_mwh'))}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_e1_6_ghg_emissions(self, data: Dict[str, Any]) -> str:
        """Render HTML E1-6 GHG emissions."""
        e1_6 = data.get("e1_6_emissions", {})
        if not e1_6:
            return ""
        s1 = e1_6.get("scope1_total_tco2e", 0.0)
        s2_loc = e1_6.get("scope2_location_tco2e", 0.0)
        s2_mkt = e1_6.get("scope2_market_tco2e", 0.0)
        s3 = e1_6.get("scope3_total_tco2e", 0.0)
        total_loc = e1_6.get("total_location_tco2e", s1 + s2_loc + s3)
        total_mkt = e1_6.get("total_market_tco2e", s1 + s2_mkt + s3)
        return (
            '<div class="section">\n<h2>E1-6: GHG Emissions</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Method</th><th>tCO2e</th>"
            '<th>XBRL Tag</th></tr></thead>\n<tbody>'
            f"<tr><td>Scope 1</td><td>Direct</td><td>{_fmt_tco2e(s1)}</td>"
            f'<td><span class="esrs-tag">esrs:GrossScope1GHGEmissions</span></td></tr>\n'
            f"<tr><td>Scope 2</td><td>Location</td><td>{_fmt_tco2e(s2_loc)}</td>"
            f'<td><span class="esrs-tag">esrs:GrossScope2LocationBasedGHGEmissions</span></td></tr>\n'
            f"<tr><td>Scope 2</td><td>Market</td><td>{_fmt_tco2e(s2_mkt)}</td>"
            f'<td><span class="esrs-tag">esrs:GrossScope2MarketBasedGHGEmissions</span></td></tr>\n'
            f"<tr><td>Scope 3</td><td>Indirect</td><td>{_fmt_tco2e(s3)}</td>"
            f'<td><span class="esrs-tag">esrs:GrossScope3GHGEmissions</span></td></tr>\n'
            f'<tr class="total-row"><td>Total</td><td>Location</td><td>{_fmt_tco2e(total_loc)}</td>'
            f'<td><span class="esrs-tag">esrs:TotalGHGEmissionsLocationBased</span></td></tr>\n'
            f'<tr class="total-row"><td>Total</td><td>Market</td><td>{_fmt_tco2e(total_mkt)}</td>'
            f'<td><span class="esrs-tag">esrs:TotalGHGEmissionsMarketBased</span></td></tr>\n'
            "</tbody></table>\n</div>"
        )

    def _html_biogenic_co2(self, data: Dict[str, Any]) -> str:
        """Render HTML biogenic CO2."""
        bio = data.get("biogenic_co2", {})
        if not bio:
            return ""
        sources = bio.get("sources", [])
        rows = ""
        for src in sources:
            rows += (
                f"<tr><td>{src.get('source', '')}</td><td>Scope {src.get('scope', '1')}</td>"
                f"<td>{src.get('biogenic_tco2', 0):,.1f} tCO2</td>"
                f"<td>{src.get('notes', '-')}</td></tr>\n"
            )
        total = bio.get("total_biogenic_tco2", 0.0)
        return (
            '<div class="section">\n<h2>Biogenic CO2 (Reported Separately)</h2>\n'
            "<p><em>Per ESRS E1-6 para 44(i)</em></p>\n"
            "<table><thead><tr><th>Source</th><th>Scope</th>"
            "<th>tCO2</th><th>Notes</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n"
            f"<p><strong>Total:</strong> {total:,.1f} tCO2</p>\n</div>"
        )

    def _html_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology notes."""
        meth = data.get("methodology", {})
        if not meth:
            return ""
        return (
            '<div class="section">\n<h2>Methodology</h2>\n'
            "<table><thead><tr><th>Aspect</th><th>Detail</th></tr></thead>\n<tbody>"
            f"<tr><td>GHG Protocol</td><td>{meth.get('ghg_protocol_standard', 'Corporate Standard')}</td></tr>\n"
            f"<tr><td>GWP Basis</td><td>{meth.get('gwp_basis', 'AR6')}</td></tr>\n"
            f"<tr><td>Consolidation</td><td>{meth.get('consolidation_approach', 'Operational Control')}</td></tr>\n"
            f"<tr><td>Base Year</td><td>{meth.get('base_year', '-')}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_disclosure_checklist(self, data: Dict[str, Any]) -> str:
        """Render HTML disclosure checklist."""
        checklist = data.get("disclosure_checklist", [])
        if not checklist:
            return ""
        rows = ""
        for item in checklist:
            status = item.get("status", "N/A")
            css = "met" if status == "Met" else ("partial" if status == "Partial" else "not-met")
            rows += (
                f"<tr><td>{item.get('disclosure_requirement', '')}</td>"
                f"<td>{item.get('requirement_name', '')}</td>"
                f'<td class="{css}">{status}</td>'
                f"<td>{item.get('reference', '-')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>Disclosure Checklist</h2>\n'
            "<table><thead><tr><th>DR</th><th>Requirement</th>"
            "<th>Status</th><th>Reference</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-041 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
