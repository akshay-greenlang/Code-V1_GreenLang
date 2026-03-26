# -*- coding: utf-8 -*-
"""
Scope2DualReportTemplate - Scope 2 Dual-Method Report for PACK-041.

Generates a comprehensive Scope 2 dual-method emissions report covering
location-based by facility with grid emission factors, market-based with
instrument allocation, steam/heat/cooling by supplier, instrument portfolio
detail (PPAs, RECs, GOs), residual mix application, location versus market
comparison with variance analysis, renewable energy procurement impact,
and quality criteria compliance.

Sections:
    1. Scope 2 Summary
    2. Location-Based by Facility
    3. Market-Based with Instrument Allocation
    4. Steam/Heat/Cooling by Supplier
    5. Instrument Portfolio Detail
    6. Residual Mix Application
    7. Location vs Market Comparison
    8. RE Procurement Impact
    9. Quality Criteria Compliance

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Regulatory References:
    - GHG Protocol Scope 2 Guidance (2015)
    - RE100 Technical Criteria
    - EU Guarantees of Origin (Directive 2018/2001)
    - ISO 14064-1:2018 Clause 5.2.4

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

INSTRUMENT_TYPES = [
    "PPA", "REC", "GO", "I-REC", "REGO", "Direct Line",
    "Green Tariff", "Residual Mix", "Supplier Specific",
]


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


def _fmt_mwh(value: Optional[float]) -> str:
    """Format MWh with scale suffix."""
    if value is None:
        return "N/A"
    return f"{_fmt_num(value, 0)} MWh"


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


class Scope2DualReportTemplate:
    """
    Scope 2 dual-method emissions report template.

    Renders comprehensive Scope 2 reports covering both location-based and
    market-based methods, with facility-level detail, instrument portfolio,
    residual mix application, variance analysis, renewable energy procurement
    impact, and quality criteria compliance. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = Scope2DualReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope2DualReportTemplate.

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

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render Scope 2 dual-method report as Markdown.

        Args:
            data: Validated Scope 2 data dict.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_location_based(data),
            self._md_market_based(data),
            self._md_steam_heat_cooling(data),
            self._md_instrument_portfolio(data),
            self._md_residual_mix(data),
            self._md_location_vs_market(data),
            self._md_re_procurement_impact(data),
            self._md_quality_criteria(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render Scope 2 dual-method report as HTML.

        Args:
            data: Validated Scope 2 data dict.

        Returns:
            Self-contained HTML document string.
        """
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_summary(data),
            self._html_location_based(data),
            self._html_market_based(data),
            self._html_steam_heat_cooling(data),
            self._html_instrument_portfolio(data),
            self._html_residual_mix(data),
            self._html_location_vs_market(data),
            self._html_re_procurement_impact(data),
            self._html_quality_criteria(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render Scope 2 dual-method report as JSON-serializable dict.

        Args:
            data: Validated Scope 2 data dict.

        Returns:
            Structured dictionary for JSON serialization.
        """
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        loc_total = data.get("location_based_total_tco2e", 0.0)
        mkt_total = data.get("market_based_total_tco2e", 0.0)
        variance = mkt_total - loc_total
        variance_pct = (variance / loc_total * 100) if loc_total > 0 else 0.0
        return {
            "template": "scope2_dual_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "summary": {
                "location_based_total_tco2e": loc_total,
                "market_based_total_tco2e": mkt_total,
                "total_electricity_mwh": data.get("total_electricity_mwh", 0.0),
                "total_steam_mwh": data.get("total_steam_mwh", 0.0),
                "total_cooling_mwh": data.get("total_cooling_mwh", 0.0),
                "variance_tco2e": variance,
                "variance_pct": variance_pct,
            },
            "location_based": data.get("location_based_facilities", []),
            "market_based": data.get("market_based_allocations", []),
            "steam_heat_cooling": data.get("steam_heat_cooling", []),
            "instrument_portfolio": data.get("instrument_portfolio", []),
            "residual_mix": data.get("residual_mix", {}),
            "re_procurement": data.get("re_procurement", {}),
            "quality_criteria": data.get("quality_criteria", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 2 Dual-Method Emissions Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 2 summary."""
        loc = data.get("location_based_total_tco2e", 0.0)
        mkt = data.get("market_based_total_tco2e", 0.0)
        elec = data.get("total_electricity_mwh", 0.0)
        steam = data.get("total_steam_mwh", 0.0)
        cooling = data.get("total_cooling_mwh", 0.0)
        variance = mkt - loc
        variance_pct = (variance / loc * 100) if loc > 0 else 0.0
        lines = [
            "## 1. Scope 2 Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Location-Based Total | {_fmt_tco2e(loc)} |",
            f"| Market-Based Total | {_fmt_tco2e(mkt)} |",
            f"| Total Electricity Consumed | {_fmt_mwh(elec)} |",
            f"| Total Steam/Heat Consumed | {_fmt_mwh(steam)} |",
            f"| Total Cooling Consumed | {_fmt_mwh(cooling)} |",
            f"| Location vs Market Variance | {_fmt_tco2e(variance)} ({_fmt_pct(variance_pct)}) |",
        ]
        return "\n".join(lines)

    def _md_location_based(self, data: Dict[str, Any]) -> str:
        """Render Markdown location-based by facility."""
        facilities = data.get("location_based_facilities", [])
        loc_total = data.get("location_based_total_tco2e", 0.0)
        lines = [
            "## 2. Location-Based Method by Facility",
            "",
            f"**Total:** {_fmt_tco2e(loc_total)}",
            "",
            "| Facility | Grid Region | MWh | Grid EF (tCO2e/MWh) | EF Source | EF Year | tCO2e | % of Total |",
            "|----------|------------|-----|---------------------|----------|---------|-------|-----------|",
        ]
        for fac in sorted(facilities, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = fac.get("facility_name", "")
            region = fac.get("grid_region", "-")
            mwh = _fmt_num(fac.get("mwh_consumed"), 0)
            ef = f"{fac.get('grid_ef', 0):.6f}" if fac.get("grid_ef") else "-"
            ef_src = fac.get("ef_source", "-")
            ef_yr = str(fac.get("ef_year", "-"))
            em = _fmt_tco2e(fac.get("emissions_tco2e", 0))
            pct = _pct_of(fac.get("emissions_tco2e", 0), loc_total)
            lines.append(f"| {name} | {region} | {mwh} | {ef} | {ef_src} | {ef_yr} | {em} | {pct} |")
        if not facilities:
            lines.append("| - | No facility data | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_market_based(self, data: Dict[str, Any]) -> str:
        """Render Markdown market-based with instrument allocation."""
        allocations = data.get("market_based_allocations", [])
        mkt_total = data.get("market_based_total_tco2e", 0.0)
        lines = [
            "## 3. Market-Based Method with Instrument Allocation",
            "",
            f"**Total:** {_fmt_tco2e(mkt_total)}",
            "",
            "| Facility | Instrument Type | MWh Covered | EF Applied | Supplier | tCO2e | % of Total |",
            "|----------|---------------|------------|-----------|----------|-------|-----------|",
        ]
        for alloc in sorted(allocations, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            fac = alloc.get("facility_name", "")
            inst = alloc.get("instrument_type", "-")
            mwh = _fmt_num(alloc.get("mwh_covered"), 0)
            ef = f"{alloc.get('ef_applied', 0):.6f}" if alloc.get("ef_applied") is not None else "-"
            supplier = alloc.get("supplier", "-")
            em = _fmt_tco2e(alloc.get("emissions_tco2e", 0))
            pct = _pct_of(alloc.get("emissions_tco2e", 0), mkt_total)
            lines.append(f"| {fac} | {inst} | {mwh} | {ef} | {supplier} | {em} | {pct} |")
        if not allocations:
            lines.append("| - | No allocation data | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_steam_heat_cooling(self, data: Dict[str, Any]) -> str:
        """Render Markdown steam/heat/cooling by supplier."""
        suppliers = data.get("steam_heat_cooling", [])
        if not suppliers:
            return "## 4. Steam/Heat/Cooling by Supplier\n\nNo steam, heat, or cooling data."
        lines = [
            "## 4. Steam/Heat/Cooling by Supplier",
            "",
            "| Supplier | Energy Type | MWh | EF (tCO2e/MWh) | tCO2e | Method |",
            "|----------|-----------|-----|----------------|-------|--------|",
        ]
        for sup in suppliers:
            name = sup.get("supplier_name", "")
            etype = sup.get("energy_type", "-")
            mwh = _fmt_num(sup.get("mwh_consumed"), 0)
            ef = f"{sup.get('ef_value', 0):.6f}" if sup.get("ef_value") is not None else "-"
            em = _fmt_tco2e(sup.get("emissions_tco2e", 0))
            method = sup.get("accounting_method", "-")
            lines.append(f"| {name} | {etype} | {mwh} | {ef} | {em} | {method} |")
        return "\n".join(lines)

    def _md_instrument_portfolio(self, data: Dict[str, Any]) -> str:
        """Render Markdown instrument portfolio detail."""
        instruments = data.get("instrument_portfolio", [])
        if not instruments:
            return "## 5. Instrument Portfolio Detail\n\nNo contractual instruments."
        lines = [
            "## 5. Instrument Portfolio Detail",
            "",
            "| Instrument | Type | Counterparty | Start Date | End Date | MWh/Year | Status | Additionality | Registry |",
            "|------------|------|-------------|-----------|---------|---------|--------|--------------|----------|",
        ]
        for inst in instruments:
            name = inst.get("instrument_name", "")
            itype = inst.get("type", "-")
            party = inst.get("counterparty", "-")
            start = inst.get("start_date", "-")
            end = inst.get("end_date", "-")
            mwh = _fmt_num(inst.get("annual_mwh"), 0)
            status = inst.get("status", "Active")
            additionality = inst.get("additionality", "-")
            registry = inst.get("registry", "-")
            lines.append(f"| {name} | {itype} | {party} | {start} | {end} | {mwh} | {status} | {additionality} | {registry} |")
        total_mwh = sum(inst.get("annual_mwh", 0) for inst in instruments)
        lines.append(f"\n**Total Contracted RE:** {_fmt_mwh(total_mwh)}")
        return "\n".join(lines)

    def _md_residual_mix(self, data: Dict[str, Any]) -> str:
        """Render Markdown residual mix application."""
        rmix = data.get("residual_mix", {})
        if not rmix:
            return "## 6. Residual Mix Application\n\nNo residual mix data."
        lines = [
            "## 6. Residual Mix Application",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Residual Mix EF | {rmix.get('residual_mix_ef', 'N/A')} tCO2e/MWh |",
            f"| Region | {rmix.get('region', 'N/A')} |",
            f"| Source | {rmix.get('source', 'N/A')} |",
            f"| Year | {rmix.get('year', 'N/A')} |",
            f"| Unmatched MWh | {_fmt_mwh(rmix.get('unmatched_mwh'))} |",
            f"| Residual Mix Emissions | {_fmt_tco2e(rmix.get('residual_emissions_tco2e'))} |",
        ]
        return "\n".join(lines)

    def _md_location_vs_market(self, data: Dict[str, Any]) -> str:
        """Render Markdown location vs market comparison."""
        loc = data.get("location_based_total_tco2e", 0.0)
        mkt = data.get("market_based_total_tco2e", 0.0)
        variance = mkt - loc
        variance_pct = (variance / loc * 100) if loc > 0 else 0.0
        comparison = data.get("location_vs_market_comparison", [])
        lines = [
            "## 7. Location vs Market Comparison",
            "",
            "| Metric | Location-Based | Market-Based | Variance | Variance % |",
            "|--------|---------------|-------------|---------|-----------|",
            f"| **Total Scope 2** | **{_fmt_tco2e(loc)}** | **{_fmt_tco2e(mkt)}** | **{_fmt_tco2e(variance)}** | **{_fmt_pct(variance_pct)}** |",
        ]
        if comparison:
            for item in comparison:
                fac = item.get("facility_name", "")
                loc_v = _fmt_tco2e(item.get("location_tco2e"))
                mkt_v = _fmt_tco2e(item.get("market_tco2e"))
                var_v = _fmt_tco2e(item.get("variance_tco2e"))
                var_p = _fmt_pct(item.get("variance_pct"))
                lines.append(f"| {fac} | {loc_v} | {mkt_v} | {var_v} | {var_p} |")
        explanation = data.get("variance_explanation", "")
        if explanation:
            lines.append(f"\n**Variance Explanation:** {explanation}")
        return "\n".join(lines)

    def _md_re_procurement_impact(self, data: Dict[str, Any]) -> str:
        """Render Markdown RE procurement impact."""
        re_data = data.get("re_procurement", {})
        if not re_data:
            return "## 8. RE Procurement Impact\n\nNo renewable energy procurement data."
        total_elec = re_data.get("total_electricity_mwh", 0.0)
        re_mwh = re_data.get("re_procured_mwh", 0.0)
        re_pct = (re_mwh / total_elec * 100) if total_elec > 0 else 0.0
        avoided = re_data.get("avoided_emissions_tco2e", 0.0)
        lines = [
            "## 8. RE Procurement Impact",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Electricity Consumed | {_fmt_mwh(total_elec)} |",
            f"| RE Procured | {_fmt_mwh(re_mwh)} ({re_pct:.1f}%) |",
            f"| Avoided Emissions (vs Grid) | {_fmt_tco2e(avoided)} |",
            f"| RE100 Target Progress | {re_data.get('re100_progress', 'N/A')} |",
        ]
        instruments_summary = re_data.get("instruments_summary", [])
        if instruments_summary:
            lines.extend([
                "",
                "### RE Procurement by Instrument Type",
                "",
                "| Instrument Type | MWh | % of RE | Cost Impact |",
                "|----------------|-----|---------|-------------|",
            ])
            for inst in instruments_summary:
                itype = inst.get("type", "-")
                mwh = _fmt_num(inst.get("mwh"), 0)
                pct = _pct_of(inst.get("mwh", 0), re_mwh) if re_mwh > 0 else "0.0%"
                cost = inst.get("cost_impact", "-")
                lines.append(f"| {itype} | {mwh} | {pct} | {cost} |")
        return "\n".join(lines)

    def _md_quality_criteria(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality criteria compliance."""
        criteria = data.get("quality_criteria", [])
        if not criteria:
            return "## 9. Quality Criteria Compliance\n\nNo quality criteria assessment performed."
        lines = [
            "## 9. Quality Criteria Compliance",
            "",
            "| Criterion | Requirement | Status | Evidence | Notes |",
            "|-----------|------------|--------|---------|-------|",
        ]
        for crit in criteria:
            name = crit.get("criterion", "")
            req = crit.get("requirement", "")
            status = crit.get("status", "N/A")
            evidence = crit.get("evidence", "-")
            notes = crit.get("notes", "-")
            lines.append(f"| {name} | {req} | **{status}** | {evidence} | {notes} |")
        passed = sum(1 for c in criteria if c.get("status") == "PASS")
        total = len(criteria)
        lines.append(f"\n**Compliance Score:** {passed}/{total} criteria met")
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
            f"<title>Scope 2 Dual Report - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #457b9d;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#415a77;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".loc-row{border-left:4px solid #457b9d;}\n"
            ".mkt-row{border-left:4px solid #2a9d8f;}\n"
            ".total-row{font-weight:bold;background:#e8eef4;}\n"
            ".metric-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#1b263b;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".pass{color:#2a9d8f;font-weight:600;}\n"
            ".fail{color:#e63946;font-weight:600;}\n"
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
            f"<h1>Scope 2 Dual-Method Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML summary with metric cards."""
        loc = data.get("location_based_total_tco2e", 0.0)
        mkt = data.get("market_based_total_tco2e", 0.0)
        elec = data.get("total_electricity_mwh", 0.0)
        cards = [
            ("Location-Based", _fmt_tco2e(loc)),
            ("Market-Based", _fmt_tco2e(mkt)),
            ("Electricity", _fmt_mwh(elec)),
        ]
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>1. Scope 2 Summary</h2>\n"
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_location_based(self, data: Dict[str, Any]) -> str:
        """Render HTML location-based by facility."""
        facilities = data.get("location_based_facilities", [])
        loc_total = data.get("location_based_total_tco2e", 0.0)
        rows = ""
        for fac in sorted(facilities, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            name = fac.get("facility_name", "")
            region = fac.get("grid_region", "-")
            mwh = _fmt_num(fac.get("mwh_consumed"), 0)
            ef = f"{fac.get('grid_ef', 0):.6f}" if fac.get("grid_ef") else "-"
            em = _fmt_tco2e(fac.get("emissions_tco2e", 0))
            pct = _pct_of(fac.get("emissions_tco2e", 0), loc_total)
            rows += f'<tr class="loc-row"><td>{name}</td><td>{region}</td><td>{mwh}</td><td>{ef}</td><td>{em}</td><td>{pct}</td></tr>\n'
        if not rows:
            rows = '<tr><td colspan="6">No facility data</td></tr>'
        return (
            '<div class="section">\n'
            "<h2>2. Location-Based by Facility</h2>\n"
            f"<p><strong>Total:</strong> {_fmt_tco2e(loc_total)}</p>\n"
            "<table><thead><tr><th>Facility</th><th>Grid Region</th><th>MWh</th>"
            "<th>Grid EF</th><th>tCO2e</th><th>%</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_market_based(self, data: Dict[str, Any]) -> str:
        """Render HTML market-based with instrument allocation."""
        allocations = data.get("market_based_allocations", [])
        mkt_total = data.get("market_based_total_tco2e", 0.0)
        rows = ""
        for alloc in sorted(allocations, key=lambda x: x.get("emissions_tco2e", 0), reverse=True):
            fac = alloc.get("facility_name", "")
            inst = alloc.get("instrument_type", "-")
            mwh = _fmt_num(alloc.get("mwh_covered"), 0)
            ef = f"{alloc.get('ef_applied', 0):.6f}" if alloc.get("ef_applied") is not None else "-"
            em = _fmt_tco2e(alloc.get("emissions_tco2e", 0))
            pct = _pct_of(alloc.get("emissions_tco2e", 0), mkt_total)
            rows += f'<tr class="mkt-row"><td>{fac}</td><td>{inst}</td><td>{mwh}</td><td>{ef}</td><td>{em}</td><td>{pct}</td></tr>\n'
        if not rows:
            rows = '<tr><td colspan="6">No allocation data</td></tr>'
        return (
            '<div class="section">\n'
            "<h2>3. Market-Based with Instruments</h2>\n"
            f"<p><strong>Total:</strong> {_fmt_tco2e(mkt_total)}</p>\n"
            "<table><thead><tr><th>Facility</th><th>Instrument</th><th>MWh</th>"
            "<th>EF</th><th>tCO2e</th><th>%</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_steam_heat_cooling(self, data: Dict[str, Any]) -> str:
        """Render HTML steam/heat/cooling by supplier."""
        suppliers = data.get("steam_heat_cooling", [])
        if not suppliers:
            return ""
        rows = ""
        for sup in suppliers:
            name = sup.get("supplier_name", "")
            etype = sup.get("energy_type", "-")
            mwh = _fmt_num(sup.get("mwh_consumed"), 0)
            ef = f"{sup.get('ef_value', 0):.6f}" if sup.get("ef_value") is not None else "-"
            em = _fmt_tco2e(sup.get("emissions_tco2e", 0))
            method = sup.get("accounting_method", "-")
            rows += f"<tr><td>{name}</td><td>{etype}</td><td>{mwh}</td><td>{ef}</td><td>{em}</td><td>{method}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>4. Steam/Heat/Cooling by Supplier</h2>\n"
            "<table><thead><tr><th>Supplier</th><th>Type</th><th>MWh</th>"
            "<th>EF</th><th>tCO2e</th><th>Method</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_instrument_portfolio(self, data: Dict[str, Any]) -> str:
        """Render HTML instrument portfolio."""
        instruments = data.get("instrument_portfolio", [])
        if not instruments:
            return ""
        rows = ""
        for inst in instruments:
            name = inst.get("instrument_name", "")
            itype = inst.get("type", "-")
            party = inst.get("counterparty", "-")
            mwh = _fmt_num(inst.get("annual_mwh"), 0)
            status = inst.get("status", "Active")
            rows += f"<tr><td>{name}</td><td>{itype}</td><td>{party}</td><td>{mwh}</td><td>{status}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>5. Instrument Portfolio</h2>\n"
            "<table><thead><tr><th>Instrument</th><th>Type</th><th>Counterparty</th>"
            "<th>MWh/Year</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_residual_mix(self, data: Dict[str, Any]) -> str:
        """Render HTML residual mix."""
        rmix = data.get("residual_mix", {})
        if not rmix:
            return ""
        return (
            '<div class="section">\n'
            "<h2>6. Residual Mix Application</h2>\n"
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>EF</td><td>{rmix.get('residual_mix_ef', 'N/A')} tCO2e/MWh</td></tr>\n"
            f"<tr><td>Region</td><td>{rmix.get('region', 'N/A')}</td></tr>\n"
            f"<tr><td>Unmatched MWh</td><td>{_fmt_mwh(rmix.get('unmatched_mwh'))}</td></tr>\n"
            f"<tr><td>Residual Emissions</td><td>{_fmt_tco2e(rmix.get('residual_emissions_tco2e'))}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_location_vs_market(self, data: Dict[str, Any]) -> str:
        """Render HTML location vs market comparison."""
        loc = data.get("location_based_total_tco2e", 0.0)
        mkt = data.get("market_based_total_tco2e", 0.0)
        variance = mkt - loc
        variance_pct = (variance / loc * 100) if loc > 0 else 0.0
        return (
            '<div class="section">\n'
            "<h2>7. Location vs Market Comparison</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Location</th><th>Market</th>"
            "<th>Variance</th></tr></thead>\n<tbody>"
            f'<tr class="total-row"><td>Total Scope 2</td><td>{_fmt_tco2e(loc)}</td>'
            f"<td>{_fmt_tco2e(mkt)}</td><td>{_fmt_tco2e(variance)} ({_fmt_pct(variance_pct)})</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_re_procurement_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML RE procurement impact."""
        re_data = data.get("re_procurement", {})
        if not re_data:
            return ""
        total_elec = re_data.get("total_electricity_mwh", 0.0)
        re_mwh = re_data.get("re_procured_mwh", 0.0)
        re_pct = (re_mwh / total_elec * 100) if total_elec > 0 else 0.0
        avoided = re_data.get("avoided_emissions_tco2e", 0.0)
        return (
            '<div class="section">\n'
            "<h2>8. RE Procurement Impact</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>Total Electricity</td><td>{_fmt_mwh(total_elec)}</td></tr>\n"
            f"<tr><td>RE Procured</td><td>{_fmt_mwh(re_mwh)} ({re_pct:.1f}%)</td></tr>\n"
            f"<tr><td>Avoided Emissions</td><td>{_fmt_tco2e(avoided)}</td></tr>\n"
            f"<tr><td>RE100 Progress</td><td>{re_data.get('re100_progress', 'N/A')}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_quality_criteria(self, data: Dict[str, Any]) -> str:
        """Render HTML quality criteria compliance."""
        criteria = data.get("quality_criteria", [])
        if not criteria:
            return ""
        rows = ""
        for crit in criteria:
            name = crit.get("criterion", "")
            req = crit.get("requirement", "")
            status = crit.get("status", "N/A")
            css = "pass" if status == "PASS" else "fail"
            evidence = crit.get("evidence", "-")
            rows += f'<tr><td>{name}</td><td>{req}</td><td class="{css}">{status}</td><td>{evidence}</td></tr>\n'
        return (
            '<div class="section">\n'
            "<h2>9. Quality Criteria Compliance</h2>\n"
            "<table><thead><tr><th>Criterion</th><th>Requirement</th>"
            "<th>Status</th><th>Evidence</th></tr></thead>\n"
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
