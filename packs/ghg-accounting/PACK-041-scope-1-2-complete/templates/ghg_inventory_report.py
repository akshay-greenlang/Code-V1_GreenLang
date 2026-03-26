# -*- coding: utf-8 -*-
"""
GHGInventoryReportTemplate - Complete GHG Inventory Report for PACK-041.

Generates a comprehensive GHG inventory report covering Scope 1 and Scope 2
emissions with full breakdowns by category, gas species, facility, and entity.
Includes executive summary, organizational boundary definition, Scope 1
eight-category detail, Scope 2 dual-method reporting, combined totals,
year-over-year comparison, uncertainty summary, methodology notes,
completeness statement, and data quality assessment.

Sections:
    1. Executive Summary
    2. Organizational Boundary
    3. Scope 1 Breakdown (8 categories, 7 gases, per facility, per entity)
    4. Scope 2 Dual Reporting (location-based + market-based)
    5. Combined Total
    6. Year-over-Year Comparison
    7. Uncertainty Summary
    8. Methodology Notes
    9. Completeness Statement
    10. Data Quality Assessment

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - ISO 14064-1:2018
    - ESRS E1 Climate Change
    - IPCC AR6 GWP values

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

# Seven greenhouse gases tracked per GHG Protocol
GHG_GASES = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]

# Eight Scope 1 categories
SCOPE1_CATEGORIES = [
    "Stationary Combustion",
    "Mobile Combustion",
    "Process Emissions",
    "Fugitive Emissions",
    "Refrigerant & F-Gas",
    "Land Use Change",
    "Waste Treatment",
    "Agricultural",
]


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format a numeric value with thousands separators."""
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


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _pct_of(part: float, total: float) -> str:
    """Percentage of total, formatted."""
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


def _safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


class GHGInventoryReportTemplate:
    """
    Complete GHG inventory report template for Scope 1 and Scope 2.

    Renders a full-featured GHG inventory report covering all eight Scope 1
    categories across seven greenhouse gases, dual Scope 2 methods (location
    and market-based), organizational boundary description, year-over-year
    comparison, uncertainty analysis, and data quality assessment. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = GHGInventoryReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GHGInventoryReportTemplate.

        Args:
            config: Optional configuration dict with overrides for
                    company_name, reporting_year, base_year, etc.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data.

        Args:
            data: Full report input data.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value from data with config override support.

        Args:
            data: Report data dict.
            key: Key to look up.
            default: Default if missing.

        Returns:
            Resolved value.
        """
        return self.config.get(key, data.get(key, default))

    def _scope1_total(self, data: Dict[str, Any]) -> float:
        """Calculate total Scope 1 emissions from breakdown data.

        Args:
            data: Report data dict.

        Returns:
            Total Scope 1 tCO2e.
        """
        breakdown = data.get("scope1_breakdown", [])
        return sum(item.get("emissions_tco2e", 0.0) for item in breakdown)

    def _scope2_location_total(self, data: Dict[str, Any]) -> float:
        """Calculate Scope 2 location-based total.

        Args:
            data: Report data dict.

        Returns:
            Location-based Scope 2 tCO2e.
        """
        scope2 = data.get("scope2_dual", {})
        return scope2.get("location_based_total_tco2e", 0.0)

    def _scope2_market_total(self, data: Dict[str, Any]) -> float:
        """Calculate Scope 2 market-based total.

        Args:
            data: Report data dict.

        Returns:
            Market-based Scope 2 tCO2e.
        """
        scope2 = data.get("scope2_dual", {})
        return scope2.get("market_based_total_tco2e", 0.0)

    def _combined_total(self, data: Dict[str, Any], method: str = "location") -> float:
        """Calculate combined Scope 1 + Scope 2 total.

        Args:
            data: Report data dict.
            method: 'location' or 'market' for Scope 2 selection.

        Returns:
            Combined total tCO2e.
        """
        s1 = self._scope1_total(data)
        if method == "market":
            return s1 + self._scope2_market_total(data)
        return s1 + self._scope2_location_total(data)

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render complete GHG inventory report as Markdown.

        Args:
            data: Validated inventory data dict.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_organizational_boundary(data),
            self._md_scope1_breakdown(data),
            self._md_scope1_by_facility(data),
            self._md_scope1_by_entity(data),
            self._md_scope2_dual(data),
            self._md_combined_total(data),
            self._md_yoy_comparison(data),
            self._md_uncertainty_summary(data),
            self._md_methodology_notes(data),
            self._md_completeness_statement(data),
            self._md_data_quality(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render complete GHG inventory report as HTML.

        Args:
            data: Validated inventory data dict.

        Returns:
            Self-contained HTML document string.
        """
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_organizational_boundary(data),
            self._html_scope1_breakdown(data),
            self._html_scope1_by_facility(data),
            self._html_scope2_dual(data),
            self._html_combined_total(data),
            self._html_yoy_comparison(data),
            self._html_uncertainty_summary(data),
            self._html_methodology_notes(data),
            self._html_completeness_statement(data),
            self._html_data_quality(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render complete GHG inventory report as JSON-serializable dict.

        Args:
            data: Validated inventory data dict.

        Returns:
            Structured dictionary for JSON serialization.
        """
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "ghg_inventory_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "base_year": self._get_val(data, "base_year"),
            "executive_summary": self._json_executive_summary(data),
            "organizational_boundary": self._json_organizational_boundary(data),
            "scope1": self._json_scope1(data),
            "scope2": self._json_scope2(data),
            "combined_totals": self._json_combined_totals(data),
            "yoy_comparison": self._json_yoy_comparison(data),
            "uncertainty": self._json_uncertainty(data),
            "methodology": data.get("methodology_notes", []),
            "completeness": self._json_completeness(data),
            "data_quality": self._json_data_quality(data),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown report header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        base = self._get_val(data, "base_year")
        base_str = f" | **Base Year:** {base}" if base else ""
        report_date = self._get_val(data, "report_date", datetime.utcnow().strftime("%Y-%m-%d"))
        return (
            f"# GHG Inventory Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {report_date}{base_str}\n\n"
            "---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown executive summary section."""
        s1 = self._scope1_total(data)
        s2_loc = self._scope2_location_total(data)
        s2_mkt = self._scope2_market_total(data)
        combined_loc = self._combined_total(data, "location")
        combined_mkt = self._combined_total(data, "market")
        yoy = data.get("yoy_change_pct")
        lines = [
            "## 1. Executive Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Scope 1 Emissions | {_fmt_tco2e(s1)} |",
            f"| Scope 2 (Location-Based) | {_fmt_tco2e(s2_loc)} |",
            f"| Scope 2 (Market-Based) | {_fmt_tco2e(s2_mkt)} |",
            f"| Combined Total (Location) | {_fmt_tco2e(combined_loc)} |",
            f"| Combined Total (Market) | {_fmt_tco2e(combined_mkt)} |",
        ]
        if yoy is not None:
            lines.append(f"| Year-over-Year Change | {_fmt_pct(yoy)} |")
        summary_text = data.get("executive_summary_text", "")
        if summary_text:
            lines.append(f"\n{summary_text}")
        return "\n".join(lines)

    def _md_organizational_boundary(self, data: Dict[str, Any]) -> str:
        """Render Markdown organizational boundary section."""
        boundary = data.get("organizational_boundary", {})
        approach = boundary.get("consolidation_approach", "Operational Control")
        entities = boundary.get("entities", [])
        lines = [
            "## 2. Organizational Boundary",
            "",
            f"**Consolidation Approach:** {approach}",
            "",
        ]
        if entities:
            lines.append("| Entity | Ownership % | Included | Notes |")
            lines.append("|--------|------------|----------|-------|")
            for ent in entities:
                name = ent.get("name", "")
                ownership = f"{ent.get('ownership_pct', 100):.0f}%"
                included = "Yes" if ent.get("included", True) else "No"
                notes = ent.get("notes", "-")
                lines.append(f"| {name} | {ownership} | {included} | {notes} |")
        else:
            lines.append("No entity data provided.")
        exclusions = boundary.get("exclusions", [])
        if exclusions:
            lines.append("\n**Exclusions:**")
            for exc in exclusions:
                lines.append(f"- {exc}")
        return "\n".join(lines)

    def _md_scope1_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 1 breakdown by category and gas."""
        breakdown = data.get("scope1_breakdown", [])
        s1_total = self._scope1_total(data)
        lines = [
            "## 3. Scope 1 Emissions Breakdown",
            "",
            "### 3.1 By Category and Gas Species",
            "",
            "| Category | Total tCO2e | % of S1 | CO2 | CH4 | N2O | HFCs | PFCs | SF6 | NF3 |",
            "|----------|------------|---------|-----|-----|-----|------|------|-----|-----|",
        ]
        for item in sorted(breakdown, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            cat = item.get("category", "Unknown")
            total = item.get("emissions_tco2e", 0.0)
            pct = _pct_of(total, s1_total)
            gas_values = []
            for gas in GHG_GASES:
                gv = item.get(f"{gas.lower()}_tco2e")
                gas_values.append(_fmt_tco2e(gv) if gv is not None else "-")
            gas_str = " | ".join(gas_values)
            lines.append(f"| {cat} | {_fmt_tco2e(total)} | {pct} | {gas_str} |")
        if not breakdown:
            lines.append("| - | No Scope 1 data available | - | - | - | - | - | - | - | - |")
        lines.append(f"\n**Scope 1 Total:** {_fmt_tco2e(s1_total)}")
        return "\n".join(lines)

    def _md_scope1_by_facility(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 1 by facility."""
        facilities = data.get("scope1_by_facility", [])
        if not facilities:
            return ""
        s1_total = self._scope1_total(data)
        lines = [
            "### 3.2 Scope 1 by Facility",
            "",
            "| Facility | Location | Emissions tCO2e | % of S1 | Top Category | Data Quality |",
            "|----------|----------|----------------|---------|--------------|--------------|",
        ]
        for fac in sorted(facilities, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = fac.get("facility_name", "Unknown")
            loc = fac.get("location", "-")
            em = fac.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s1_total)
            top = fac.get("top_category", "-")
            quality = fac.get("data_quality", "MEDIUM")
            lines.append(f"| {name} | {loc} | {_fmt_tco2e(em)} | {pct} | {top} | {quality} |")
        return "\n".join(lines)

    def _md_scope1_by_entity(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 1 by reporting entity."""
        entities = data.get("scope1_by_entity", [])
        if not entities:
            return ""
        s1_total = self._scope1_total(data)
        lines = [
            "### 3.3 Scope 1 by Entity",
            "",
            "| Entity | Emissions tCO2e | % of S1 | Ownership % | Reported tCO2e |",
            "|--------|----------------|---------|------------|----------------|",
        ]
        for ent in sorted(entities, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = ent.get("entity_name", "Unknown")
            em = ent.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s1_total)
            ownership = f"{ent.get('ownership_pct', 100):.0f}%"
            reported = _fmt_tco2e(ent.get("reported_tco2e", em))
            lines.append(f"| {name} | {_fmt_tco2e(em)} | {pct} | {ownership} | {reported} |")
        return "\n".join(lines)

    def _md_scope2_dual(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 2 dual-method section."""
        scope2 = data.get("scope2_dual", {})
        loc_total = self._scope2_location_total(data)
        mkt_total = self._scope2_market_total(data)
        loc_facilities = scope2.get("location_based_facilities", [])
        mkt_instruments = scope2.get("market_based_instruments", [])
        lines = [
            "## 4. Scope 2 Dual Reporting",
            "",
            "### 4.1 Location-Based Method",
            "",
            f"**Location-Based Total:** {_fmt_tco2e(loc_total)}",
            "",
        ]
        if loc_facilities:
            lines.append("| Facility | Grid Region | MWh Consumed | Grid EF | tCO2e |")
            lines.append("|----------|------------|-------------|---------|-------|")
            for fac in loc_facilities:
                name = fac.get("facility_name", "")
                region = fac.get("grid_region", "-")
                mwh = _fmt_num(fac.get("mwh_consumed"), 0)
                ef = f"{fac.get('grid_ef', 0):.4f}" if fac.get("grid_ef") else "-"
                em = _fmt_tco2e(fac.get("emissions_tco2e", 0))
                lines.append(f"| {name} | {region} | {mwh} | {ef} | {em} |")
        lines.extend([
            "",
            "### 4.2 Market-Based Method",
            "",
            f"**Market-Based Total:** {_fmt_tco2e(mkt_total)}",
            "",
        ])
        if mkt_instruments:
            lines.append("| Instrument | Type | MWh | Supplier | tCO2e |")
            lines.append("|------------|------|-----|----------|-------|")
            for inst in mkt_instruments:
                name = inst.get("instrument_name", "")
                itype = inst.get("type", "-")
                mwh = _fmt_num(inst.get("mwh", 0), 0)
                supplier = inst.get("supplier", "-")
                em = _fmt_tco2e(inst.get("emissions_tco2e", 0))
                lines.append(f"| {name} | {itype} | {mwh} | {supplier} | {em} |")
        # Variance
        if loc_total > 0 and mkt_total > 0:
            variance = mkt_total - loc_total
            variance_pct = (variance / loc_total) * 100
            lines.extend([
                "",
                f"**Location vs Market Variance:** {_fmt_tco2e(variance)} ({_fmt_pct(variance_pct)})",
            ])
        return "\n".join(lines)

    def _md_combined_total(self, data: Dict[str, Any]) -> str:
        """Render Markdown combined totals."""
        s1 = self._scope1_total(data)
        s2_loc = self._scope2_location_total(data)
        s2_mkt = self._scope2_market_total(data)
        comb_loc = s1 + s2_loc
        comb_mkt = s1 + s2_mkt
        lines = [
            "## 5. Combined Scope 1 + Scope 2 Totals",
            "",
            "| Component | Location Method | Market Method |",
            "|-----------|---------------|---------------|",
            f"| Scope 1 | {_fmt_tco2e(s1)} | {_fmt_tco2e(s1)} |",
            f"| Scope 2 | {_fmt_tco2e(s2_loc)} | {_fmt_tco2e(s2_mkt)} |",
            f"| **Combined Total** | **{_fmt_tco2e(comb_loc)}** | **{_fmt_tco2e(comb_mkt)}** |",
        ]
        return "\n".join(lines)

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown year-over-year comparison."""
        yoy = data.get("yoy_comparison", {})
        if not yoy:
            return "## 6. Year-over-Year Comparison\n\nNo prior year data available."
        prior_year = yoy.get("prior_year")
        lines = [
            "## 6. Year-over-Year Comparison",
            "",
            f"**Compared to:** {prior_year}",
            "",
            "| Metric | Current Year | Prior Year | Absolute Change | % Change |",
            "|--------|-------------|------------|----------------|----------|",
        ]
        for item in yoy.get("metrics", []):
            metric = item.get("metric_name", "")
            current = _fmt_tco2e(item.get("current_value"))
            prior = _fmt_tco2e(item.get("prior_value"))
            abs_change = _fmt_tco2e(item.get("absolute_change"))
            pct_change = _fmt_pct(item.get("pct_change"))
            lines.append(f"| {metric} | {current} | {prior} | {abs_change} | {pct_change} |")
        return "\n".join(lines)

    def _md_uncertainty_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown uncertainty summary."""
        uncertainty = data.get("uncertainty_summary", {})
        if not uncertainty:
            return "## 7. Uncertainty Summary\n\nNo uncertainty analysis performed."
        overall_pct = uncertainty.get("overall_uncertainty_pct")
        confidence = uncertainty.get("confidence_level", "95%")
        method = uncertainty.get("method", "Analytical propagation")
        lines = [
            "## 7. Uncertainty Summary",
            "",
            f"**Method:** {method} | **Confidence Level:** {confidence}",
            f"**Overall Uncertainty:** +/-{overall_pct:.1f}%" if overall_pct else "",
            "",
        ]
        contributors = uncertainty.get("top_contributors", [])
        if contributors:
            lines.append("| Source | Uncertainty % | Contribution |")
            lines.append("|--------|--------------|-------------|")
            for c in contributors:
                name = c.get("source", "")
                unc = f"+/-{c.get('uncertainty_pct', 0):.1f}%"
                contrib = f"{c.get('contribution_pct', 0):.1f}%"
                lines.append(f"| {name} | {unc} | {contrib} |")
        return "\n".join(s for s in lines if s)

    def _md_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology notes."""
        notes = data.get("methodology_notes", [])
        if not notes:
            return "## 8. Methodology Notes\n\nNo methodology notes provided."
        lines = [
            "## 8. Methodology Notes",
            "",
            "| Category | Methodology | Tier | EF Source | GWP Basis |",
            "|----------|------------|------|----------|-----------|",
        ]
        for note in notes:
            cat = note.get("category", "")
            meth = note.get("methodology", "")
            tier = note.get("tier", "-")
            ef_src = note.get("ef_source", "-")
            gwp = note.get("gwp_basis", "AR6")
            lines.append(f"| {cat} | {meth} | {tier} | {ef_src} | {gwp} |")
        return "\n".join(lines)

    def _md_completeness_statement(self, data: Dict[str, Any]) -> str:
        """Render Markdown completeness statement."""
        completeness = data.get("completeness", {})
        coverage = completeness.get("coverage_pct", 100.0)
        threshold = completeness.get("materiality_threshold_pct", 5.0)
        excluded = completeness.get("excluded_sources", [])
        lines = [
            "## 9. Completeness Statement",
            "",
            f"**Coverage:** {coverage:.1f}% of identified emission sources",
            f"**Materiality Threshold:** {threshold:.1f}%",
            "",
        ]
        if excluded:
            lines.append("**Excluded Sources:**")
            for src in excluded:
                reason = src.get("reason", "Below materiality threshold")
                lines.append(f"- {src.get('source', 'Unknown')}: {reason}")
        else:
            lines.append("All identified emission sources have been included.")
        statement = completeness.get("statement", "")
        if statement:
            lines.append(f"\n{statement}")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality assessment."""
        quality = data.get("data_quality", {})
        if not quality:
            return "## 10. Data Quality Assessment\n\nNo data quality assessment available."
        overall = quality.get("overall_score", "")
        lines = [
            "## 10. Data Quality Assessment",
            "",
            f"**Overall Data Quality Score:** {overall}",
            "",
        ]
        categories = quality.get("by_category", [])
        if categories:
            lines.append("| Category | Quality | Completeness | Source Type | Improvement Action |")
            lines.append("|----------|---------|-------------|------------|-------------------|")
            for cat in categories:
                name = cat.get("category", "")
                ql = cat.get("quality_level", "MEDIUM")
                comp = f"{cat.get('completeness_pct', 100):.0f}%"
                src_type = cat.get("source_type", "-")
                action = cat.get("improvement_action", "-")
                lines.append(f"| {name} | {ql} | {comp} | {src_type} | {action} |")
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
        """Wrap body content in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>GHG Inventory Report - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #1b263b;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#415a77;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".scope-1{border-left:4px solid #e63946;}\n"
            ".scope-2-loc{border-left:4px solid #457b9d;}\n"
            ".scope-2-mkt{border-left:4px solid #2a9d8f;}\n"
            ".total-row{font-weight:bold;background:#e8eef4;}\n"
            ".metric-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#1b263b;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".quality-high{color:#2a9d8f;font-weight:600;}\n"
            ".quality-medium{color:#e9c46a;font-weight:600;}\n"
            ".quality-low{color:#e76f51;font-weight:600;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML report header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        base = self._get_val(data, "base_year")
        base_str = f" | <strong>Base Year:</strong> {base}" if base else ""
        report_date = self._get_val(data, "report_date", datetime.utcnow().strftime("%Y-%m-%d"))
        return (
            '<div class="section">\n'
            f"<h1>GHG Inventory Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Report Date:</strong> {report_date}{base_str}</p>\n"
            "<hr>\n</div>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary with metric cards."""
        s1 = self._scope1_total(data)
        s2_loc = self._scope2_location_total(data)
        s2_mkt = self._scope2_market_total(data)
        comb_loc = self._combined_total(data, "location")
        cards = [
            ("Scope 1", _fmt_tco2e(s1)),
            ("Scope 2 (Location)", _fmt_tco2e(s2_loc)),
            ("Scope 2 (Market)", _fmt_tco2e(s2_mkt)),
            ("Combined (Location)", _fmt_tco2e(comb_loc)),
        ]
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        summary_text = data.get("executive_summary_text", "")
        text_html = f"<p>{summary_text}</p>" if summary_text else ""
        return (
            '<div class="section">\n'
            "<h2>1. Executive Summary</h2>\n"
            f"<div>{card_html}</div>\n"
            f"{text_html}\n</div>"
        )

    def _html_organizational_boundary(self, data: Dict[str, Any]) -> str:
        """Render HTML organizational boundary."""
        boundary = data.get("organizational_boundary", {})
        approach = boundary.get("consolidation_approach", "Operational Control")
        entities = boundary.get("entities", [])
        rows = ""
        for ent in entities:
            name = ent.get("name", "")
            ownership = f"{ent.get('ownership_pct', 100):.0f}%"
            included = "Yes" if ent.get("included", True) else "No"
            notes = ent.get("notes", "-")
            rows += f"<tr><td>{name}</td><td>{ownership}</td><td>{included}</td><td>{notes}</td></tr>\n"
        if not rows:
            rows = '<tr><td colspan="4">No entity data provided.</td></tr>'
        return (
            '<div class="section">\n'
            "<h2>2. Organizational Boundary</h2>\n"
            f"<p><strong>Consolidation Approach:</strong> {approach}</p>\n"
            "<table><thead><tr><th>Entity</th><th>Ownership %</th>"
            "<th>Included</th><th>Notes</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scope1_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 1 breakdown."""
        breakdown = data.get("scope1_breakdown", [])
        s1_total = self._scope1_total(data)
        rows = ""
        for item in sorted(breakdown, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            cat = item.get("category", "Unknown")
            total = item.get("emissions_tco2e", 0.0)
            pct = _pct_of(total, s1_total)
            gas_cells = ""
            for gas in GHG_GASES:
                gv = item.get(f"{gas.lower()}_tco2e")
                gas_cells += f"<td>{_fmt_tco2e(gv) if gv is not None else '-'}</td>"
            rows += f'<tr class="scope-1"><td>{cat}</td><td>{_fmt_tco2e(total)}</td><td>{pct}</td>{gas_cells}</tr>\n'
        if not rows:
            rows = f'<tr><td colspan="{3 + len(GHG_GASES)}">No Scope 1 data</td></tr>'
        gas_headers = "".join(f"<th>{g}</th>" for g in GHG_GASES)
        return (
            '<div class="section">\n'
            "<h2>3. Scope 1 Emissions Breakdown</h2>\n"
            "<h3>3.1 By Category and Gas Species</h3>\n"
            f"<table><thead><tr><th>Category</th><th>Total tCO2e</th><th>% of S1</th>{gas_headers}"
            f"</tr></thead>\n<tbody>{rows}</tbody></table>\n"
            f"<p><strong>Scope 1 Total:</strong> {_fmt_tco2e(s1_total)}</p>\n</div>"
        )

    def _html_scope1_by_facility(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 1 by facility."""
        facilities = data.get("scope1_by_facility", [])
        if not facilities:
            return ""
        s1_total = self._scope1_total(data)
        rows = ""
        for fac in sorted(facilities, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = fac.get("facility_name", "Unknown")
            loc = fac.get("location", "-")
            em = fac.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s1_total)
            top = fac.get("top_category", "-")
            quality = fac.get("data_quality", "MEDIUM")
            qclass = f"quality-{quality.lower()}"
            rows += (
                f"<tr><td>{name}</td><td>{loc}</td><td>{_fmt_tco2e(em)}</td>"
                f'<td>{pct}</td><td>{top}</td><td class="{qclass}">{quality}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h3>3.2 Scope 1 by Facility</h3>\n"
            "<table><thead><tr><th>Facility</th><th>Location</th><th>tCO2e</th>"
            "<th>% of S1</th><th>Top Category</th><th>Quality</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scope2_dual(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 2 dual reporting."""
        scope2 = data.get("scope2_dual", {})
        loc_total = self._scope2_location_total(data)
        mkt_total = self._scope2_market_total(data)
        loc_facilities = scope2.get("location_based_facilities", [])
        mkt_instruments = scope2.get("market_based_instruments", [])
        # Location table
        loc_rows = ""
        for fac in loc_facilities:
            name = fac.get("facility_name", "")
            region = fac.get("grid_region", "-")
            mwh = _fmt_num(fac.get("mwh_consumed"), 0)
            ef = f"{fac.get('grid_ef', 0):.4f}" if fac.get("grid_ef") else "-"
            em = _fmt_tco2e(fac.get("emissions_tco2e", 0))
            loc_rows += f'<tr class="scope-2-loc"><td>{name}</td><td>{region}</td><td>{mwh}</td><td>{ef}</td><td>{em}</td></tr>\n'
        if not loc_rows:
            loc_rows = '<tr><td colspan="5">No location-based data</td></tr>'
        # Market table
        mkt_rows = ""
        for inst in mkt_instruments:
            name = inst.get("instrument_name", "")
            itype = inst.get("type", "-")
            mwh = _fmt_num(inst.get("mwh", 0), 0)
            supplier = inst.get("supplier", "-")
            em = _fmt_tco2e(inst.get("emissions_tco2e", 0))
            mkt_rows += f'<tr class="scope-2-mkt"><td>{name}</td><td>{itype}</td><td>{mwh}</td><td>{supplier}</td><td>{em}</td></tr>\n'
        if not mkt_rows:
            mkt_rows = '<tr><td colspan="5">No market-based data</td></tr>'
        return (
            '<div class="section">\n'
            "<h2>4. Scope 2 Dual Reporting</h2>\n"
            "<h3>4.1 Location-Based Method</h3>\n"
            f"<p><strong>Total:</strong> {_fmt_tco2e(loc_total)}</p>\n"
            "<table><thead><tr><th>Facility</th><th>Grid Region</th><th>MWh</th>"
            f"<th>Grid EF</th><th>tCO2e</th></tr></thead>\n<tbody>{loc_rows}</tbody></table>\n"
            "<h3>4.2 Market-Based Method</h3>\n"
            f"<p><strong>Total:</strong> {_fmt_tco2e(mkt_total)}</p>\n"
            "<table><thead><tr><th>Instrument</th><th>Type</th><th>MWh</th>"
            f"<th>Supplier</th><th>tCO2e</th></tr></thead>\n<tbody>{mkt_rows}</tbody></table>\n"
            "</div>"
        )

    def _html_combined_total(self, data: Dict[str, Any]) -> str:
        """Render HTML combined totals."""
        s1 = self._scope1_total(data)
        s2_loc = self._scope2_location_total(data)
        s2_mkt = self._scope2_market_total(data)
        return (
            '<div class="section">\n'
            "<h2>5. Combined Scope 1 + Scope 2 Totals</h2>\n"
            "<table><thead><tr><th>Component</th><th>Location Method</th>"
            "<th>Market Method</th></tr></thead>\n<tbody>"
            f"<tr><td>Scope 1</td><td>{_fmt_tco2e(s1)}</td><td>{_fmt_tco2e(s1)}</td></tr>\n"
            f"<tr><td>Scope 2</td><td>{_fmt_tco2e(s2_loc)}</td><td>{_fmt_tco2e(s2_mkt)}</td></tr>\n"
            f'<tr class="total-row"><td>Combined Total</td>'
            f"<td>{_fmt_tco2e(s1 + s2_loc)}</td><td>{_fmt_tco2e(s1 + s2_mkt)}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year comparison."""
        yoy = data.get("yoy_comparison", {})
        if not yoy:
            return ""
        rows = ""
        for item in yoy.get("metrics", []):
            metric = item.get("metric_name", "")
            current = _fmt_tco2e(item.get("current_value"))
            prior = _fmt_tco2e(item.get("prior_value"))
            abs_c = _fmt_tco2e(item.get("absolute_change"))
            pct_c = _fmt_pct(item.get("pct_change"))
            rows += f"<tr><td>{metric}</td><td>{current}</td><td>{prior}</td><td>{abs_c}</td><td>{pct_c}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>6. Year-over-Year Comparison</h2>\n"
            f"<p><strong>Compared to:</strong> {yoy.get('prior_year', 'N/A')}</p>\n"
            "<table><thead><tr><th>Metric</th><th>Current</th><th>Prior</th>"
            f"<th>Abs Change</th><th>% Change</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_uncertainty_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty summary."""
        uncertainty = data.get("uncertainty_summary", {})
        if not uncertainty:
            return ""
        overall = uncertainty.get("overall_uncertainty_pct")
        method = uncertainty.get("method", "Analytical propagation")
        confidence = uncertainty.get("confidence_level", "95%")
        rows = ""
        for c in uncertainty.get("top_contributors", []):
            name = c.get("source", "")
            unc = f"+/-{c.get('uncertainty_pct', 0):.1f}%"
            contrib = f"{c.get('contribution_pct', 0):.1f}%"
            rows += f"<tr><td>{name}</td><td>{unc}</td><td>{contrib}</td></tr>\n"
        overall_str = f"+/-{overall:.1f}%" if overall else "N/A"
        return (
            '<div class="section">\n'
            "<h2>7. Uncertainty Summary</h2>\n"
            f"<p><strong>Method:</strong> {method} | <strong>Confidence:</strong> {confidence} | "
            f"<strong>Overall:</strong> {overall_str}</p>\n"
            "<table><thead><tr><th>Source</th><th>Uncertainty</th>"
            f"<th>Contribution</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology notes."""
        notes = data.get("methodology_notes", [])
        if not notes:
            return ""
        rows = ""
        for note in notes:
            cat = note.get("category", "")
            meth = note.get("methodology", "")
            tier = note.get("tier", "-")
            ef_src = note.get("ef_source", "-")
            gwp = note.get("gwp_basis", "AR6")
            rows += f"<tr><td>{cat}</td><td>{meth}</td><td>{tier}</td><td>{ef_src}</td><td>{gwp}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>8. Methodology Notes</h2>\n"
            "<table><thead><tr><th>Category</th><th>Methodology</th><th>Tier</th>"
            f"<th>EF Source</th><th>GWP Basis</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_completeness_statement(self, data: Dict[str, Any]) -> str:
        """Render HTML completeness statement."""
        completeness = data.get("completeness", {})
        coverage = completeness.get("coverage_pct", 100.0)
        threshold = completeness.get("materiality_threshold_pct", 5.0)
        statement = completeness.get("statement", "")
        return (
            '<div class="section">\n'
            "<h2>9. Completeness Statement</h2>\n"
            f"<p><strong>Coverage:</strong> {coverage:.1f}% | "
            f"<strong>Materiality Threshold:</strong> {threshold:.1f}%</p>\n"
            f"<p>{statement if statement else 'All identified emission sources have been included.'}</p>\n"
            "</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality assessment."""
        quality = data.get("data_quality", {})
        if not quality:
            return ""
        overall = quality.get("overall_score", "")
        categories = quality.get("by_category", [])
        rows = ""
        for cat in categories:
            name = cat.get("category", "")
            ql = cat.get("quality_level", "MEDIUM")
            qclass = f"quality-{ql.lower()}"
            comp = f"{cat.get('completeness_pct', 100):.0f}%"
            src_type = cat.get("source_type", "-")
            action = cat.get("improvement_action", "-")
            rows += (
                f'<tr><td>{name}</td><td class="{qclass}">{ql}</td>'
                f"<td>{comp}</td><td>{src_type}</td><td>{action}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>10. Data Quality Assessment</h2>\n"
            f"<p><strong>Overall Score:</strong> {overall}</p>\n"
            "<table><thead><tr><th>Category</th><th>Quality</th>"
            "<th>Completeness</th><th>Source Type</th><th>Improvement</th></tr></thead>\n"
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

    # ==================================================================
    # JSON SECTIONS
    # ==================================================================

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary section."""
        return {
            "scope1_total_tco2e": self._scope1_total(data),
            "scope2_location_total_tco2e": self._scope2_location_total(data),
            "scope2_market_total_tco2e": self._scope2_market_total(data),
            "combined_location_tco2e": self._combined_total(data, "location"),
            "combined_market_tco2e": self._combined_total(data, "market"),
            "yoy_change_pct": data.get("yoy_change_pct"),
            "summary_text": data.get("executive_summary_text", ""),
        }

    def _json_organizational_boundary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON organizational boundary section."""
        boundary = data.get("organizational_boundary", {})
        return {
            "consolidation_approach": boundary.get("consolidation_approach", "Operational Control"),
            "entities": boundary.get("entities", []),
            "exclusions": boundary.get("exclusions", []),
        }

    def _json_scope1(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON Scope 1 section."""
        return {
            "total_tco2e": self._scope1_total(data),
            "breakdown_by_category": data.get("scope1_breakdown", []),
            "by_facility": data.get("scope1_by_facility", []),
            "by_entity": data.get("scope1_by_entity", []),
        }

    def _json_scope2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON Scope 2 section."""
        scope2 = data.get("scope2_dual", {})
        return {
            "location_based_total_tco2e": self._scope2_location_total(data),
            "market_based_total_tco2e": self._scope2_market_total(data),
            "location_based_facilities": scope2.get("location_based_facilities", []),
            "market_based_instruments": scope2.get("market_based_instruments", []),
        }

    def _json_combined_totals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON combined totals."""
        return {
            "scope1_tco2e": self._scope1_total(data),
            "scope2_location_tco2e": self._scope2_location_total(data),
            "scope2_market_tco2e": self._scope2_market_total(data),
            "combined_location_tco2e": self._combined_total(data, "location"),
            "combined_market_tco2e": self._combined_total(data, "market"),
        }

    def _json_yoy_comparison(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build JSON YoY comparison."""
        yoy = data.get("yoy_comparison")
        if not yoy:
            return None
        return yoy

    def _json_uncertainty(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build JSON uncertainty section."""
        return data.get("uncertainty_summary")

    def _json_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON completeness section."""
        completeness = data.get("completeness", {})
        return {
            "coverage_pct": completeness.get("coverage_pct", 100.0),
            "materiality_threshold_pct": completeness.get("materiality_threshold_pct", 5.0),
            "excluded_sources": completeness.get("excluded_sources", []),
            "statement": completeness.get("statement", ""),
        }

    def _json_data_quality(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build JSON data quality section."""
        return data.get("data_quality")
