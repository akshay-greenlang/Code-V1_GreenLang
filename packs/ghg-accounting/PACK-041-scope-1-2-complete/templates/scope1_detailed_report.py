# -*- coding: utf-8 -*-
"""
Scope1DetailedReportTemplate - Detailed Scope 1 Per-Category Report for PACK-041.

Generates a detailed Scope 1 emissions report with per-category tables covering
activity data, emission factors, and emissions per gas species. Includes
equipment/asset detail for stationary and mobile combustion and refrigerants,
process emission methodologies, fugitive estimation methods, agricultural
emission factors, cross-category reconciliation, and emission factor citations.

Sections:
    1. Scope 1 Overview
    2. Stationary Combustion Detail
    3. Mobile Combustion Detail
    4. Process Emissions Detail
    5. Fugitive Emissions Detail
    6. Refrigerant & F-Gas Detail
    7. Land Use Change Detail
    8. Waste Treatment Detail
    9. Agricultural Emissions Detail
    10. Cross-Category Reconciliation
    11. Emission Factor Citations

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with per-category detail)

Regulatory References:
    - GHG Protocol Corporate Standard Ch. 4-6
    - IPCC 2006/2019 Guidelines for National GHG Inventories
    - EPA Mandatory Reporting Rule (40 CFR Part 98)
    - ISO 14064-1:2018

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

SCOPE1_CATEGORY_ORDER = [
    "stationary_combustion",
    "mobile_combustion",
    "process_emissions",
    "fugitive_emissions",
    "refrigerant_fgas",
    "land_use_change",
    "waste_treatment",
    "agricultural",
]

CATEGORY_DISPLAY_NAMES = {
    "stationary_combustion": "Stationary Combustion",
    "mobile_combustion": "Mobile Combustion",
    "process_emissions": "Process Emissions",
    "fugitive_emissions": "Fugitive Emissions",
    "refrigerant_fgas": "Refrigerant & F-Gas",
    "land_use_change": "Land Use Change",
    "waste_treatment": "Waste Treatment",
    "agricultural": "Agricultural",
}


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    return f"{_fmt_num(value)} tCO2e"


def _pct_of(part: float, total: float) -> str:
    """Calculate and format percentage."""
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


class Scope1DetailedReportTemplate:
    """
    Detailed Scope 1 per-category emissions report template.

    Renders comprehensive per-category Scope 1 emissions reports with
    activity data, emission factors, emissions per gas species, equipment
    and asset detail, methodology descriptions, cross-category reconciliation,
    and emission factor citations. All outputs include SHA-256 provenance
    hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = Scope1DetailedReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope1DetailedReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def _total_scope1(self, data: Dict[str, Any]) -> float:
        """Sum all category totals for Scope 1."""
        categories = data.get("scope1_categories", {})
        total = 0.0
        for cat_key in SCOPE1_CATEGORY_ORDER:
            cat_data = categories.get(cat_key, {})
            total += cat_data.get("total_tco2e", 0.0)
        return total

    def _get_category(self, data: Dict[str, Any], cat_key: str) -> Dict[str, Any]:
        """Get category data by key."""
        return data.get("scope1_categories", {}).get(cat_key, {})

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render detailed Scope 1 report as Markdown.

        Args:
            data: Validated Scope 1 detail data dict.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_stationary_combustion(data),
            self._md_mobile_combustion(data),
            self._md_process_emissions(data),
            self._md_fugitive_emissions(data),
            self._md_refrigerant_fgas(data),
            self._md_land_use_change(data),
            self._md_waste_treatment(data),
            self._md_agricultural(data),
            self._md_cross_category_reconciliation(data),
            self._md_ef_citations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render detailed Scope 1 report as HTML.

        Args:
            data: Validated Scope 1 detail data dict.

        Returns:
            Self-contained HTML document string.
        """
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
            self._html_stationary_combustion(data),
            self._html_mobile_combustion(data),
            self._html_process_emissions(data),
            self._html_fugitive_emissions(data),
            self._html_refrigerant_fgas(data),
            self._html_land_use_change(data),
            self._html_waste_treatment(data),
            self._html_agricultural(data),
            self._html_cross_category_reconciliation(data),
            self._html_ef_citations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render detailed Scope 1 report as JSON-serializable dict.

        Args:
            data: Validated Scope 1 detail data dict.

        Returns:
            Structured dictionary for JSON serialization.
        """
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        categories_json = {}
        for cat_key in SCOPE1_CATEGORY_ORDER:
            cat_data = self._get_category(data, cat_key)
            if cat_data:
                categories_json[cat_key] = cat_data
        return {
            "template": "scope1_detailed_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "scope1_total_tco2e": self._total_scope1(data),
            "categories": categories_json,
            "cross_category_reconciliation": data.get("reconciliation", {}),
            "ef_citations": data.get("ef_citations", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 1 Detailed Emissions Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 1 overview table."""
        total = self._total_scope1(data)
        categories = data.get("scope1_categories", {})
        lines = [
            "## 1. Scope 1 Overview",
            "",
            "| Category | Total tCO2e | % of Scope 1 | Data Quality | Methodology Tier |",
            "|----------|------------|-------------|--------------|------------------|",
        ]
        for cat_key in SCOPE1_CATEGORY_ORDER:
            cat_data = categories.get(cat_key, {})
            display = CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key)
            cat_total = cat_data.get("total_tco2e", 0.0)
            pct = _pct_of(cat_total, total)
            quality = cat_data.get("data_quality", "-")
            tier = cat_data.get("methodology_tier", "-")
            lines.append(f"| {display} | {_fmt_tco2e(cat_total)} | {pct} | {quality} | {tier} |")
        lines.append(f"\n**Scope 1 Total:** {_fmt_tco2e(total)}")
        return "\n".join(lines)

    def _md_category_activity_table(self, cat_data: Dict[str, Any], section_num: str, title: str) -> str:
        """Render a standard per-category activity data table in Markdown."""
        activities = cat_data.get("activities", [])
        if not activities:
            return f"## {section_num}. {title}\n\nNo activity data available for this category."
        cat_total = cat_data.get("total_tco2e", 0.0)
        methodology = cat_data.get("methodology_description", "")
        lines = [
            f"## {section_num}. {title}",
            "",
        ]
        if methodology:
            lines.append(f"**Methodology:** {methodology}\n")
        lines.extend([
            "| Activity | Quantity | Unit | EF Value | EF Unit | EF Source | CO2 | CH4 | N2O | Total tCO2e |",
            "|----------|---------|------|----------|---------|----------|-----|-----|-----|------------|",
        ])
        for act in activities:
            name = act.get("activity_name", "")
            qty = _fmt_num(act.get("quantity"), 2)
            unit = act.get("unit", "")
            ef_val = f"{act.get('ef_value', 0):.6f}" if act.get("ef_value") is not None else "-"
            ef_unit = act.get("ef_unit", "")
            ef_src = act.get("ef_source", "-")
            co2 = _fmt_tco2e(act.get("co2_tco2e"))
            ch4 = _fmt_tco2e(act.get("ch4_tco2e"))
            n2o = _fmt_tco2e(act.get("n2o_tco2e"))
            total = _fmt_tco2e(act.get("total_tco2e", 0))
            lines.append(f"| {name} | {qty} | {unit} | {ef_val} | {ef_unit} | {ef_src} | {co2} | {ch4} | {n2o} | {total} |")
        lines.append(f"\n**Category Total:** {_fmt_tco2e(cat_total)}")
        return "\n".join(lines)

    def _md_equipment_table(self, equipment: List[Dict[str, Any]], title: str) -> str:
        """Render equipment/asset detail table in Markdown."""
        if not equipment:
            return ""
        lines = [
            f"### {title}",
            "",
            "| Equipment ID | Type | Fuel/Refrigerant | Capacity | Age (yr) | Annual Usage | tCO2e | Status |",
            "|-------------|------|-----------------|----------|----------|-------------|-------|--------|",
        ]
        for eq in equipment:
            eid = eq.get("equipment_id", "")
            etype = eq.get("equipment_type", "")
            fuel = eq.get("fuel_type", eq.get("refrigerant_type", "-"))
            capacity = eq.get("capacity", "-")
            age = str(eq.get("age_years", "-"))
            usage = eq.get("annual_usage", "-")
            em = _fmt_tco2e(eq.get("emissions_tco2e", 0))
            status = eq.get("status", "Active")
            lines.append(f"| {eid} | {etype} | {fuel} | {capacity} | {age} | {usage} | {em} | {status} |")
        return "\n".join(lines)

    def _md_stationary_combustion(self, data: Dict[str, Any]) -> str:
        """Render Markdown stationary combustion detail."""
        cat_data = self._get_category(data, "stationary_combustion")
        result = self._md_category_activity_table(cat_data, "2", "Stationary Combustion Detail")
        equipment = cat_data.get("equipment", [])
        if equipment:
            result += "\n\n" + self._md_equipment_table(equipment, "Stationary Equipment Inventory")
        return result

    def _md_mobile_combustion(self, data: Dict[str, Any]) -> str:
        """Render Markdown mobile combustion detail."""
        cat_data = self._get_category(data, "mobile_combustion")
        result = self._md_category_activity_table(cat_data, "3", "Mobile Combustion Detail")
        equipment = cat_data.get("equipment", [])
        if equipment:
            result += "\n\n" + self._md_equipment_table(equipment, "Fleet/Vehicle Inventory")
        return result

    def _md_process_emissions(self, data: Dict[str, Any]) -> str:
        """Render Markdown process emissions detail."""
        cat_data = self._get_category(data, "process_emissions")
        result = self._md_category_activity_table(cat_data, "4", "Process Emissions Detail")
        methodologies = cat_data.get("process_methodologies", [])
        if methodologies:
            lines = ["\n### Process-Specific Methodologies", ""]
            for meth in methodologies:
                name = meth.get("process_name", "")
                desc = meth.get("methodology_description", "")
                ref = meth.get("reference", "")
                lines.append(f"- **{name}:** {desc} (Ref: {ref})")
            result += "\n".join(lines)
        return result

    def _md_fugitive_emissions(self, data: Dict[str, Any]) -> str:
        """Render Markdown fugitive emissions detail."""
        cat_data = self._get_category(data, "fugitive_emissions")
        result = self._md_category_activity_table(cat_data, "5", "Fugitive Emissions Detail")
        methods = cat_data.get("estimation_methods", [])
        if methods:
            lines = ["\n### Fugitive Estimation Methods", ""]
            lines.append("| Source | Method | Detection Limit | Frequency | Notes |")
            lines.append("|--------|--------|----------------|-----------|-------|")
            for m in methods:
                source = m.get("source", "")
                method = m.get("method", "")
                limit = m.get("detection_limit", "-")
                freq = m.get("frequency", "-")
                notes = m.get("notes", "-")
                lines.append(f"| {source} | {method} | {limit} | {freq} | {notes} |")
            result += "\n" + "\n".join(lines)
        return result

    def _md_refrigerant_fgas(self, data: Dict[str, Any]) -> str:
        """Render Markdown refrigerant and F-gas detail."""
        cat_data = self._get_category(data, "refrigerant_fgas")
        result = self._md_category_activity_table(cat_data, "6", "Refrigerant & F-Gas Detail")
        equipment = cat_data.get("equipment", [])
        if equipment:
            result += "\n\n" + self._md_equipment_table(equipment, "Refrigeration Equipment Inventory")
        return result

    def _md_land_use_change(self, data: Dict[str, Any]) -> str:
        """Render Markdown land use change detail."""
        cat_data = self._get_category(data, "land_use_change")
        return self._md_category_activity_table(cat_data, "7", "Land Use Change Detail")

    def _md_waste_treatment(self, data: Dict[str, Any]) -> str:
        """Render Markdown waste treatment detail."""
        cat_data = self._get_category(data, "waste_treatment")
        return self._md_category_activity_table(cat_data, "8", "Waste Treatment Detail")

    def _md_agricultural(self, data: Dict[str, Any]) -> str:
        """Render Markdown agricultural emissions detail."""
        cat_data = self._get_category(data, "agricultural")
        result = self._md_category_activity_table(cat_data, "9", "Agricultural Emissions Detail")
        factors = cat_data.get("agricultural_factors", [])
        if factors:
            lines = ["\n### Agricultural Emission Factors", ""]
            lines.append("| Source | Factor | Unit | Region | Reference |")
            lines.append("|--------|--------|------|--------|-----------|")
            for f in factors:
                source = f.get("source_type", "")
                factor = f"{f.get('factor_value', 0):.6f}" if f.get("factor_value") is not None else "-"
                unit = f.get("unit", "")
                region = f.get("region", "Global")
                ref = f.get("reference", "-")
                lines.append(f"| {source} | {factor} | {unit} | {region} | {ref} |")
            result += "\n" + "\n".join(lines)
        return result

    def _md_cross_category_reconciliation(self, data: Dict[str, Any]) -> str:
        """Render Markdown cross-category reconciliation."""
        recon = data.get("reconciliation", {})
        if not recon:
            return "## 10. Cross-Category Reconciliation\n\nNo reconciliation data available."
        lines = [
            "## 10. Cross-Category Reconciliation",
            "",
        ]
        sum_categories = recon.get("sum_of_categories_tco2e", 0.0)
        reported_total = recon.get("reported_total_tco2e", 0.0)
        difference = recon.get("difference_tco2e", 0.0)
        status = recon.get("reconciliation_status", "PASS")
        lines.extend([
            "| Check | Value |",
            "|-------|-------|",
            f"| Sum of Category Totals | {_fmt_tco2e(sum_categories)} |",
            f"| Reported Scope 1 Total | {_fmt_tco2e(reported_total)} |",
            f"| Difference | {_fmt_tco2e(difference)} |",
            f"| Reconciliation Status | **{status}** |",
        ])
        notes = recon.get("notes", "")
        if notes:
            lines.append(f"\n{notes}")
        return "\n".join(lines)

    def _md_ef_citations(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission factor citations."""
        citations = data.get("ef_citations", [])
        if not citations:
            return "## 11. Emission Factor Citations\n\nNo emission factor citations provided."
        lines = [
            "## 11. Emission Factor Citations",
            "",
            "| Ref ID | Factor Name | Value | Unit | Source | Version | Geography | Provenance Hash |",
            "|--------|------------|-------|------|--------|---------|-----------|----------------|",
        ]
        for cit in citations:
            ref = cit.get("ref_id", "")
            name = cit.get("factor_name", "")
            value = f"{cit.get('value', 0):.6f}" if cit.get("value") is not None else "-"
            unit = cit.get("unit", "")
            source = cit.get("source", "")
            version = cit.get("version", "-")
            geo = cit.get("geography", "Global")
            phash = cit.get("provenance_hash", "-")
            if len(phash) > 12:
                phash = f"{phash[:12]}..."
            lines.append(f"| {ref} | {name} | {value} | {unit} | {source} | {version} | {geo} | `{phash}` |")
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
            f"<title>Scope 1 Detailed Report - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #e63946;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#415a77;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".cat-header{background:#e63946;color:#fff;padding:0.5rem 1rem;border-radius:4px 4px 0 0;}\n"
            ".total-row{font-weight:bold;background:#e8eef4;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".pass{color:#2a9d8f;font-weight:700;}\n"
            ".fail{color:#e63946;font-weight:700;}\n"
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
            f"<h1>Scope 1 Detailed Emissions Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview table."""
        total = self._total_scope1(data)
        categories = data.get("scope1_categories", {})
        rows = ""
        for cat_key in SCOPE1_CATEGORY_ORDER:
            cat_data = categories.get(cat_key, {})
            display = CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key)
            cat_total = cat_data.get("total_tco2e", 0.0)
            pct = _pct_of(cat_total, total)
            quality = cat_data.get("data_quality", "-")
            tier = cat_data.get("methodology_tier", "-")
            rows += f"<tr><td>{display}</td><td>{_fmt_tco2e(cat_total)}</td><td>{pct}</td><td>{quality}</td><td>{tier}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>1. Scope 1 Overview</h2>\n"
            "<table><thead><tr><th>Category</th><th>Total tCO2e</th>"
            "<th>% of S1</th><th>Quality</th><th>Tier</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n"
            f"<p><strong>Scope 1 Total:</strong> {_fmt_tco2e(total)}</p>\n</div>"
        )

    def _html_category_activity_table(self, cat_data: Dict[str, Any], section_num: str, title: str) -> str:
        """Render a standard per-category activity data table in HTML."""
        activities = cat_data.get("activities", [])
        if not activities:
            return (
                f'<div class="section">\n<h2>{section_num}. {title}</h2>\n'
                "<p>No activity data available.</p>\n</div>"
            )
        methodology = cat_data.get("methodology_description", "")
        meth_html = f"<p><strong>Methodology:</strong> {methodology}</p>\n" if methodology else ""
        rows = ""
        for act in activities:
            name = act.get("activity_name", "")
            qty = _fmt_num(act.get("quantity"), 2)
            unit = act.get("unit", "")
            ef_val = f"{act.get('ef_value', 0):.6f}" if act.get("ef_value") is not None else "-"
            ef_src = act.get("ef_source", "-")
            co2 = _fmt_tco2e(act.get("co2_tco2e"))
            ch4 = _fmt_tco2e(act.get("ch4_tco2e"))
            n2o = _fmt_tco2e(act.get("n2o_tco2e"))
            total = _fmt_tco2e(act.get("total_tco2e", 0))
            rows += (
                f"<tr><td>{name}</td><td>{qty}</td><td>{unit}</td>"
                f"<td>{ef_val}</td><td>{ef_src}</td><td>{co2}</td>"
                f"<td>{ch4}</td><td>{n2o}</td><td>{total}</td></tr>\n"
            )
        cat_total = cat_data.get("total_tco2e", 0.0)
        return (
            f'<div class="section">\n<h2>{section_num}. {title}</h2>\n'
            f"{meth_html}"
            "<table><thead><tr><th>Activity</th><th>Qty</th><th>Unit</th>"
            "<th>EF Value</th><th>EF Source</th><th>CO2</th><th>CH4</th>"
            f"<th>N2O</th><th>Total</th></tr></thead>\n<tbody>{rows}</tbody></table>\n"
            f"<p><strong>Category Total:</strong> {_fmt_tco2e(cat_total)}</p>\n</div>"
        )

    def _html_equipment_table(self, equipment: List[Dict[str, Any]], title: str) -> str:
        """Render equipment/asset detail table in HTML."""
        if not equipment:
            return ""
        rows = ""
        for eq in equipment:
            eid = eq.get("equipment_id", "")
            etype = eq.get("equipment_type", "")
            fuel = eq.get("fuel_type", eq.get("refrigerant_type", "-"))
            capacity = eq.get("capacity", "-")
            age = str(eq.get("age_years", "-"))
            usage = eq.get("annual_usage", "-")
            em = _fmt_tco2e(eq.get("emissions_tco2e", 0))
            status = eq.get("status", "Active")
            rows += (
                f"<tr><td>{eid}</td><td>{etype}</td><td>{fuel}</td>"
                f"<td>{capacity}</td><td>{age}</td><td>{usage}</td>"
                f"<td>{em}</td><td>{status}</td></tr>\n"
            )
        return (
            f"<h3>{title}</h3>\n"
            "<table><thead><tr><th>ID</th><th>Type</th><th>Fuel/Ref</th>"
            "<th>Capacity</th><th>Age</th><th>Usage</th><th>tCO2e</th>"
            f"<th>Status</th></tr></thead>\n<tbody>{rows}</tbody></table>"
        )

    def _html_stationary_combustion(self, data: Dict[str, Any]) -> str:
        """Render HTML stationary combustion."""
        cat_data = self._get_category(data, "stationary_combustion")
        result = self._html_category_activity_table(cat_data, "2", "Stationary Combustion Detail")
        equipment = cat_data.get("equipment", [])
        if equipment:
            result = result.replace("</div>", self._html_equipment_table(equipment, "Stationary Equipment") + "\n</div>")
        return result

    def _html_mobile_combustion(self, data: Dict[str, Any]) -> str:
        """Render HTML mobile combustion."""
        cat_data = self._get_category(data, "mobile_combustion")
        result = self._html_category_activity_table(cat_data, "3", "Mobile Combustion Detail")
        equipment = cat_data.get("equipment", [])
        if equipment:
            result = result.replace("</div>", self._html_equipment_table(equipment, "Fleet/Vehicle Inventory") + "\n</div>")
        return result

    def _html_process_emissions(self, data: Dict[str, Any]) -> str:
        """Render HTML process emissions."""
        cat_data = self._get_category(data, "process_emissions")
        return self._html_category_activity_table(cat_data, "4", "Process Emissions Detail")

    def _html_fugitive_emissions(self, data: Dict[str, Any]) -> str:
        """Render HTML fugitive emissions."""
        cat_data = self._get_category(data, "fugitive_emissions")
        return self._html_category_activity_table(cat_data, "5", "Fugitive Emissions Detail")

    def _html_refrigerant_fgas(self, data: Dict[str, Any]) -> str:
        """Render HTML refrigerant and F-gas."""
        cat_data = self._get_category(data, "refrigerant_fgas")
        result = self._html_category_activity_table(cat_data, "6", "Refrigerant & F-Gas Detail")
        equipment = cat_data.get("equipment", [])
        if equipment:
            result = result.replace("</div>", self._html_equipment_table(equipment, "Refrigeration Equipment") + "\n</div>")
        return result

    def _html_land_use_change(self, data: Dict[str, Any]) -> str:
        """Render HTML land use change."""
        cat_data = self._get_category(data, "land_use_change")
        return self._html_category_activity_table(cat_data, "7", "Land Use Change Detail")

    def _html_waste_treatment(self, data: Dict[str, Any]) -> str:
        """Render HTML waste treatment."""
        cat_data = self._get_category(data, "waste_treatment")
        return self._html_category_activity_table(cat_data, "8", "Waste Treatment Detail")

    def _html_agricultural(self, data: Dict[str, Any]) -> str:
        """Render HTML agricultural emissions."""
        cat_data = self._get_category(data, "agricultural")
        return self._html_category_activity_table(cat_data, "9", "Agricultural Emissions Detail")

    def _html_cross_category_reconciliation(self, data: Dict[str, Any]) -> str:
        """Render HTML cross-category reconciliation."""
        recon = data.get("reconciliation", {})
        if not recon:
            return ""
        sum_cat = recon.get("sum_of_categories_tco2e", 0.0)
        reported = recon.get("reported_total_tco2e", 0.0)
        diff = recon.get("difference_tco2e", 0.0)
        status = recon.get("reconciliation_status", "PASS")
        css = "pass" if status == "PASS" else "fail"
        return (
            '<div class="section">\n'
            "<h2>10. Cross-Category Reconciliation</h2>\n"
            "<table><thead><tr><th>Check</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>Sum of Categories</td><td>{_fmt_tco2e(sum_cat)}</td></tr>\n"
            f"<tr><td>Reported Total</td><td>{_fmt_tco2e(reported)}</td></tr>\n"
            f"<tr><td>Difference</td><td>{_fmt_tco2e(diff)}</td></tr>\n"
            f'<tr><td>Status</td><td class="{css}">{status}</td></tr>\n'
            "</tbody></table>\n</div>"
        )

    def _html_ef_citations(self, data: Dict[str, Any]) -> str:
        """Render HTML emission factor citations."""
        citations = data.get("ef_citations", [])
        if not citations:
            return ""
        rows = ""
        for cit in citations:
            ref = cit.get("ref_id", "")
            name = cit.get("factor_name", "")
            value = f"{cit.get('value', 0):.6f}" if cit.get("value") is not None else "-"
            unit = cit.get("unit", "")
            source = cit.get("source", "")
            version = cit.get("version", "-")
            geo = cit.get("geography", "Global")
            phash = cit.get("provenance_hash", "-")
            if len(phash) > 12:
                phash = f"{phash[:12]}..."
            rows += (
                f"<tr><td>{ref}</td><td>{name}</td><td>{value}</td>"
                f"<td>{unit}</td><td>{source}</td><td>{version}</td>"
                f"<td>{geo}</td><td><code>{phash}</code></td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>11. Emission Factor Citations</h2>\n"
            "<table><thead><tr><th>Ref</th><th>Factor</th><th>Value</th>"
            "<th>Unit</th><th>Source</th><th>Version</th>"
            "<th>Geography</th><th>Hash</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-041 Scope 1-2 Complete v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
