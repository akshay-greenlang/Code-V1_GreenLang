# -*- coding: utf-8 -*-
"""
Scope3InventoryReportTemplate - Full 15-Category Scope 3 Inventory for PACK-042.

Generates a comprehensive Scope 3 GHG inventory report covering all 15
value chain categories as defined by the GHG Protocol Corporate Value Chain
(Scope 3) Standard. Includes executive summary, methodology overview,
per-category breakdown table with tCO2e, percentage of total, methodology
tier, data quality rating, and year-over-year change. Also provides
upstream vs downstream split, gas-level breakdown, data quality summary,
uncertainty ranges, compliance status, and appendix with emission factor
sources and key assumptions.

Sections:
    1. Executive Summary
    2. Methodology Overview
    3. 15-Category Breakdown Table
    4. Upstream vs Downstream Split
    5. Gas-Level Breakdown
    6. Data Quality Summary
    7. Uncertainty Ranges
    8. Compliance Status
    9. Appendix - EF Sources
    10. Appendix - Assumptions

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, deep teal theme)
    - JSON (structured with chart-ready data)

Regulatory References:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions
    - ISO 14064-1:2018 Category 3-6
    - ESRS E1-6 para 44-46
    - IPCC AR6 GWP values

Author: GreenLang Team
Version: 42.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "42.0.0"

# GHG Protocol Scope 3 categories
SCOPE3_CATEGORIES = [
    {"number": 1, "name": "Purchased Goods and Services", "stream": "upstream"},
    {"number": 2, "name": "Capital Goods", "stream": "upstream"},
    {"number": 3, "name": "Fuel- and Energy-Related Activities", "stream": "upstream"},
    {"number": 4, "name": "Upstream Transportation and Distribution", "stream": "upstream"},
    {"number": 5, "name": "Waste Generated in Operations", "stream": "upstream"},
    {"number": 6, "name": "Business Travel", "stream": "upstream"},
    {"number": 7, "name": "Employee Commuting", "stream": "upstream"},
    {"number": 8, "name": "Upstream Leased Assets", "stream": "upstream"},
    {"number": 9, "name": "Downstream Transportation and Distribution", "stream": "downstream"},
    {"number": 10, "name": "Processing of Sold Products", "stream": "downstream"},
    {"number": 11, "name": "Use of Sold Products", "stream": "downstream"},
    {"number": 12, "name": "End-of-Life Treatment of Sold Products", "stream": "downstream"},
    {"number": 13, "name": "Downstream Leased Assets", "stream": "downstream"},
    {"number": 14, "name": "Franchises", "stream": "downstream"},
    {"number": 15, "name": "Investments", "stream": "downstream"},
]

GHG_GASES = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]


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


def _quality_label(score: Optional[float]) -> str:
    """Convert numeric DQR score to label."""
    if score is None:
        return "N/A"
    if score >= 4.0:
        return "HIGH"
    if score >= 2.5:
        return "MEDIUM"
    return "LOW"


class Scope3InventoryReportTemplate:
    """
    Full 15-category Scope 3 GHG inventory report template.

    Renders comprehensive Scope 3 inventory reports covering all 15 value
    chain categories per the GHG Protocol Scope 3 Standard. Reports include
    upstream/downstream splits, gas-level breakdowns, data quality summaries,
    uncertainty ranges, and multi-framework compliance status. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = Scope3InventoryReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope3InventoryReportTemplate.

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
        """Get value from data with config override support."""
        return self.config.get(key, data.get(key, default))

    def _scope3_total(self, data: Dict[str, Any]) -> float:
        """Calculate total Scope 3 emissions across all categories."""
        categories = data.get("scope3_categories", [])
        return sum(c.get("emissions_tco2e", 0.0) for c in categories)

    def _upstream_total(self, data: Dict[str, Any]) -> float:
        """Calculate total upstream (Cat 1-8) emissions."""
        categories = data.get("scope3_categories", [])
        return sum(
            c.get("emissions_tco2e", 0.0)
            for c in categories
            if c.get("category_number", 0) <= 8
        )

    def _downstream_total(self, data: Dict[str, Any]) -> float:
        """Calculate total downstream (Cat 9-15) emissions."""
        categories = data.get("scope3_categories", [])
        return sum(
            c.get("emissions_tco2e", 0.0)
            for c in categories
            if c.get("category_number", 0) >= 9
        )

    def _get_category_data(
        self, data: Dict[str, Any], cat_num: int
    ) -> Dict[str, Any]:
        """Get data for a specific category by number."""
        for c in data.get("scope3_categories", []):
            if c.get("category_number") == cat_num:
                return c
        return {}

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render full Scope 3 inventory report as Markdown.

        Args:
            data: Validated Scope 3 inventory data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_methodology_overview(data),
            self._md_category_breakdown(data),
            self._md_upstream_downstream(data),
            self._md_gas_breakdown(data),
            self._md_data_quality_summary(data),
            self._md_uncertainty_ranges(data),
            self._md_compliance_status(data),
            self._md_appendix_ef_sources(data),
            self._md_appendix_assumptions(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render full Scope 3 inventory report as HTML.

        Args:
            data: Validated Scope 3 inventory data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_methodology_overview(data),
            self._html_category_breakdown(data),
            self._html_upstream_downstream(data),
            self._html_gas_breakdown(data),
            self._html_data_quality_summary(data),
            self._html_uncertainty_ranges(data),
            self._html_compliance_status(data),
            self._html_appendix_ef_sources(data),
            self._html_appendix_assumptions(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render full Scope 3 inventory report as JSON-serializable dict.

        Args:
            data: Validated Scope 3 inventory data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        s3_total = self._scope3_total(data)
        return {
            "template": "scope3_inventory_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "base_year": self._get_val(data, "base_year"),
            "executive_summary": self._json_executive_summary(data),
            "methodology": self._json_methodology(data),
            "categories": self._json_categories(data),
            "upstream_downstream": {
                "upstream_tco2e": self._upstream_total(data),
                "downstream_tco2e": self._downstream_total(data),
                "upstream_pct": (self._upstream_total(data) / s3_total * 100)
                if s3_total > 0 else 0.0,
                "downstream_pct": (self._downstream_total(data) / s3_total * 100)
                if s3_total > 0 else 0.0,
            },
            "gas_breakdown": data.get("gas_breakdown", {}),
            "data_quality": self._json_data_quality(data),
            "uncertainty": data.get("uncertainty", {}),
            "compliance": data.get("compliance_status", {}),
            "ef_sources": data.get("ef_sources", []),
            "assumptions": data.get("assumptions", []),
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
        report_date = self._get_val(
            data, "report_date", datetime.utcnow().strftime("%Y-%m-%d")
        )
        return (
            f"# Scope 3 GHG Inventory Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {report_date}{base_str}\n\n"
            "---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown executive summary section."""
        s3_total = self._scope3_total(data)
        upstream = self._upstream_total(data)
        downstream = self._downstream_total(data)
        s1 = data.get("scope1_total_tco2e", 0.0)
        s2 = data.get("scope2_total_tco2e", 0.0)
        full_footprint = s1 + s2 + s3_total
        s3_pct = (s3_total / full_footprint * 100) if full_footprint > 0 else 0.0
        yoy = data.get("yoy_change_pct")
        top_categories = sorted(
            data.get("scope3_categories", []),
            key=lambda c: c.get("emissions_tco2e", 0),
            reverse=True,
        )[:5]
        lines = [
            "## 1. Executive Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Scope 3 Emissions | {_fmt_tco2e(s3_total)} |",
            f"| Upstream (Cat 1-8) | {_fmt_tco2e(upstream)} |",
            f"| Downstream (Cat 9-15) | {_fmt_tco2e(downstream)} |",
            f"| Scope 3 as % of Total Footprint | {s3_pct:.1f}% |",
            f"| Categories Reported | {len(data.get('scope3_categories', []))} of 15 |",
        ]
        if yoy is not None:
            lines.append(f"| Year-over-Year Change | {_fmt_pct(yoy)} |")
        if top_categories:
            lines.append("")
            lines.append("**Top 5 Categories by Emissions:**")
            lines.append("")
            for i, cat in enumerate(top_categories, 1):
                num = cat.get("category_number", "?")
                name = cat.get("category_name", "Unknown")
                em = _fmt_tco2e(cat.get("emissions_tco2e"))
                pct = _pct_of(cat.get("emissions_tco2e", 0), s3_total)
                lines.append(f"{i}. **Cat {num} - {name}:** {em} ({pct})")
        summary_text = data.get("executive_summary_text", "")
        if summary_text:
            lines.append(f"\n{summary_text}")
        return "\n".join(lines)

    def _md_methodology_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology overview section."""
        methodology = data.get("methodology_overview", {})
        if not methodology:
            return "## 2. Methodology Overview\n\nNo methodology overview provided."
        standard = methodology.get("standard", "GHG Protocol Scope 3 Standard")
        gwp = methodology.get("gwp_basis", "IPCC AR6")
        boundary = methodology.get("boundary_approach", "Operational Control")
        lines = [
            "## 2. Methodology Overview",
            "",
            f"**Standard:** {standard}",
            f"**GWP Basis:** {gwp}",
            f"**Boundary Approach:** {boundary}",
            "",
        ]
        methods = methodology.get("category_methods", [])
        if methods:
            lines.append(
                "| Category | Calculation Method | Tier | Key Data Sources |"
            )
            lines.append(
                "|----------|-------------------|------|-----------------|"
            )
            for m in methods:
                cat = f"Cat {m.get('category_number', '?')}"
                method = m.get("method", "-")
                tier = m.get("tier", "-")
                sources = m.get("data_sources", "-")
                lines.append(f"| {cat} | {method} | {tier} | {sources} |")
        notes = methodology.get("notes", "")
        if notes:
            lines.append(f"\n{notes}")
        return "\n".join(lines)

    def _md_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown 15-category breakdown table."""
        categories = data.get("scope3_categories", [])
        s3_total = self._scope3_total(data)
        lines = [
            "## 3. Scope 3 Category Breakdown",
            "",
            "| Cat # | Category | tCO2e | % of Total | Tier | DQ Rating | YoY Change |",
            "|-------|----------|-------|-----------|------|-----------|------------|",
        ]
        cat_lookup = {c.get("category_number"): c for c in categories}
        for ref in SCOPE3_CATEGORIES:
            num = ref["number"]
            cat_data = cat_lookup.get(num, {})
            name = cat_data.get("category_name", ref["name"])
            em = cat_data.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            tier = cat_data.get("methodology_tier", "-")
            dq = cat_data.get("data_quality_rating", "-")
            yoy = cat_data.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            em_str = _fmt_tco2e(em) if em > 0 else "Not reported"
            lines.append(
                f"| {num} | {name} | {em_str} | {pct} | {tier} | {dq} | {yoy_str} |"
            )
        lines.append(f"\n**Total Scope 3:** {_fmt_tco2e(s3_total)}")
        return "\n".join(lines)

    def _md_upstream_downstream(self, data: Dict[str, Any]) -> str:
        """Render Markdown upstream vs downstream split."""
        upstream = self._upstream_total(data)
        downstream = self._downstream_total(data)
        s3_total = self._scope3_total(data)
        lines = [
            "## 4. Upstream vs Downstream Split",
            "",
            "| Stream | Categories | tCO2e | % of Scope 3 |",
            "|--------|-----------|-------|-------------|",
            f"| Upstream | Cat 1-8 | {_fmt_tco2e(upstream)} | {_pct_of(upstream, s3_total)} |",
            f"| Downstream | Cat 9-15 | {_fmt_tco2e(downstream)} | {_pct_of(downstream, s3_total)} |",
            f"| **Total** | **All** | **{_fmt_tco2e(s3_total)}** | **100.0%** |",
        ]
        return "\n".join(lines)

    def _md_gas_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown gas-level breakdown."""
        gas_data = data.get("gas_breakdown", {})
        if not gas_data:
            return "## 5. Gas-Level Breakdown\n\nNo gas-level breakdown available."
        s3_total = self._scope3_total(data)
        lines = [
            "## 5. Gas-Level Breakdown",
            "",
            "| Gas | tCO2e | % of Scope 3 |",
            "|-----|-------|-------------|",
        ]
        for gas in GHG_GASES:
            val = gas_data.get(gas.lower() + "_tco2e", gas_data.get(gas, 0.0))
            if val and val > 0:
                lines.append(
                    f"| {gas} | {_fmt_tco2e(val)} | {_pct_of(val, s3_total)} |"
                )
        return "\n".join(lines)

    def _md_data_quality_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality summary."""
        quality = data.get("data_quality_summary", {})
        if not quality:
            return "## 6. Data Quality Summary\n\nNo data quality summary available."
        overall_dqr = quality.get("overall_dqr_score")
        lines = [
            "## 6. Data Quality Summary",
            "",
        ]
        if overall_dqr is not None:
            lines.append(
                f"**Overall DQR Score:** {overall_dqr:.1f} / 5.0 "
                f"({_quality_label(overall_dqr)})"
            )
            lines.append("")
        cat_quality = quality.get("by_category", [])
        if cat_quality:
            lines.append(
                "| Category | DQR Score | Quality Level | Primary Source | Coverage % |"
            )
            lines.append(
                "|----------|----------|---------------|---------------|-----------|"
            )
            for cq in cat_quality:
                cat_name = cq.get("category_name", "")
                dqr = cq.get("dqr_score")
                dqr_str = f"{dqr:.1f}" if dqr is not None else "-"
                level = cq.get("quality_level", "-")
                source = cq.get("primary_source", "-")
                coverage = cq.get("coverage_pct")
                cov_str = f"{coverage:.0f}%" if coverage is not None else "-"
                lines.append(
                    f"| {cat_name} | {dqr_str} | {level} | {source} | {cov_str} |"
                )
        return "\n".join(lines)

    def _md_uncertainty_ranges(self, data: Dict[str, Any]) -> str:
        """Render Markdown uncertainty ranges."""
        uncertainty = data.get("uncertainty", {})
        if not uncertainty:
            return "## 7. Uncertainty Ranges\n\nNo uncertainty analysis available."
        overall = uncertainty.get("overall_uncertainty_pct")
        method = uncertainty.get("method", "Combined analytical + Monte Carlo")
        confidence = uncertainty.get("confidence_level", "95%")
        lines = [
            "## 7. Uncertainty Ranges",
            "",
            f"**Method:** {method} | **Confidence Level:** {confidence}",
        ]
        if overall is not None:
            lines.append(f"**Overall Scope 3 Uncertainty:** +/-{overall:.1f}%")
        lines.append("")
        per_cat = uncertainty.get("by_category", [])
        if per_cat:
            lines.append(
                "| Category | Central tCO2e | Lower Bound | Upper Bound | Uncertainty % |"
            )
            lines.append(
                "|----------|-------------|-------------|-------------|--------------|"
            )
            for pc in per_cat:
                cat = pc.get("category_name", "")
                central = _fmt_tco2e(pc.get("central_tco2e"))
                lower = _fmt_tco2e(pc.get("lower_bound_tco2e"))
                upper = _fmt_tco2e(pc.get("upper_bound_tco2e"))
                unc_pct = pc.get("uncertainty_pct")
                unc_str = f"+/-{unc_pct:.1f}%" if unc_pct is not None else "-"
                lines.append(
                    f"| {cat} | {central} | {lower} | {upper} | {unc_str} |"
                )
        return "\n".join(lines)

    def _md_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render Markdown compliance status."""
        compliance = data.get("compliance_status", {})
        if not compliance:
            return "## 8. Compliance Status\n\nNo compliance assessment available."
        lines = [
            "## 8. Compliance Status",
            "",
            "| Framework | Status | Score | Key Gaps |",
            "|-----------|--------|-------|----------|",
        ]
        frameworks = compliance.get("frameworks", [])
        for fw in frameworks:
            name = fw.get("framework_name", "")
            status = fw.get("status", "-")
            score = fw.get("score")
            score_str = f"{score:.0f}%" if score is not None else "-"
            gaps = fw.get("key_gaps", "-")
            if isinstance(gaps, list):
                gaps = "; ".join(gaps)
            lines.append(f"| {name} | {status} | {score_str} | {gaps} |")
        return "\n".join(lines)

    def _md_appendix_ef_sources(self, data: Dict[str, Any]) -> str:
        """Render Markdown appendix with emission factor sources."""
        sources = data.get("ef_sources", [])
        if not sources:
            return "## Appendix A: Emission Factor Sources\n\nNo EF sources documented."
        lines = [
            "## Appendix A: Emission Factor Sources",
            "",
            "| Category | Factor Name | Value | Unit | Source | Version | Geography |",
            "|----------|-----------|-------|------|--------|---------|-----------|",
        ]
        for src in sources:
            cat = src.get("category", "-")
            name = src.get("factor_name", "")
            val = src.get("value")
            val_str = f"{val:.6f}" if val is not None else "-"
            unit = src.get("unit", "")
            source = src.get("source", "")
            version = src.get("version", "-")
            geo = src.get("geography", "Global")
            lines.append(
                f"| {cat} | {name} | {val_str} | {unit} | {source} | {version} | {geo} |"
            )
        return "\n".join(lines)

    def _md_appendix_assumptions(self, data: Dict[str, Any]) -> str:
        """Render Markdown appendix with assumptions."""
        assumptions = data.get("assumptions", [])
        if not assumptions:
            return "## Appendix B: Key Assumptions\n\nNo assumptions documented."
        lines = [
            "## Appendix B: Key Assumptions",
            "",
            "| # | Category | Assumption | Impact | Justification |",
            "|---|----------|-----------|--------|---------------|",
        ]
        for i, a in enumerate(assumptions, 1):
            cat = a.get("category", "General")
            assumption = a.get("assumption", "")
            impact = a.get("impact", "-")
            justification = a.get("justification", "-")
            lines.append(
                f"| {i} | {cat} | {assumption} | {impact} | {justification} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}*\n"
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
            f"<title>Scope 3 Inventory Report - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0D7377;border-bottom:3px solid #0D7377;padding-bottom:0.5rem;}\n"
            "h2{color:#0A5C5F;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}\n"
            "h3{color:#0D7377;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#e6f5f5;font-weight:600;color:#0A5C5F;}\n"
            "tr:nth-child(even){background:#f7fcfc;}\n"
            ".upstream{border-left:4px solid #0D7377;}\n"
            ".downstream{border-left:4px solid #E67E22;}\n"
            ".total-row{font-weight:bold;background:#d4eded;}\n"
            ".metric-card{display:inline-block;background:#e6f5f5;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;"
            "border-top:3px solid #0D7377;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#0A5C5F;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".quality-high{color:#0D7377;font-weight:600;}\n"
            ".quality-medium{color:#E67E22;font-weight:600;}\n"
            ".quality-low{color:#e74c3c;font-weight:600;}\n"
            ".status-pass{color:#0D7377;font-weight:700;}\n"
            ".status-fail{color:#e74c3c;font-weight:700;}\n"
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
        report_date = self._get_val(
            data, "report_date", datetime.utcnow().strftime("%Y-%m-%d")
        )
        return (
            '<div class="section">\n'
            f"<h1>Scope 3 GHG Inventory Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Report Date:</strong> {report_date}{base_str}</p>\n"
            "<hr>\n</div>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary with metric cards."""
        s3_total = self._scope3_total(data)
        upstream = self._upstream_total(data)
        downstream = self._downstream_total(data)
        s1 = data.get("scope1_total_tco2e", 0.0)
        s2 = data.get("scope2_total_tco2e", 0.0)
        full = s1 + s2 + s3_total
        s3_pct = (s3_total / full * 100) if full > 0 else 0.0
        cards = [
            ("Total Scope 3", _fmt_tco2e(s3_total)),
            ("Upstream (1-8)", _fmt_tco2e(upstream)),
            ("Downstream (9-15)", _fmt_tco2e(downstream)),
            ("% of Full Footprint", f"{s3_pct:.1f}%"),
        ]
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        top_cats = sorted(
            data.get("scope3_categories", []),
            key=lambda c: c.get("emissions_tco2e", 0),
            reverse=True,
        )[:5]
        top_list = ""
        if top_cats:
            top_list = "<h3>Top 5 Categories</h3>\n<ol>\n"
            for cat in top_cats:
                num = cat.get("category_number", "?")
                name = cat.get("category_name", "Unknown")
                em = _fmt_tco2e(cat.get("emissions_tco2e"))
                pct = _pct_of(cat.get("emissions_tco2e", 0), s3_total)
                top_list += f"<li><strong>Cat {num} - {name}:</strong> {em} ({pct})</li>\n"
            top_list += "</ol>\n"
        return (
            '<div class="section">\n'
            "<h2>1. Executive Summary</h2>\n"
            f"<div>{card_html}</div>\n"
            f"{top_list}</div>"
        )

    def _html_methodology_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology overview."""
        methodology = data.get("methodology_overview", {})
        if not methodology:
            return ""
        standard = methodology.get("standard", "GHG Protocol Scope 3 Standard")
        gwp = methodology.get("gwp_basis", "IPCC AR6")
        boundary = methodology.get("boundary_approach", "Operational Control")
        methods = methodology.get("category_methods", [])
        rows = ""
        for m in methods:
            cat = f"Cat {m.get('category_number', '?')}"
            method = m.get("method", "-")
            tier = m.get("tier", "-")
            sources = m.get("data_sources", "-")
            rows += f"<tr><td>{cat}</td><td>{method}</td><td>{tier}</td><td>{sources}</td></tr>\n"
        table = ""
        if rows:
            table = (
                "<table><thead><tr><th>Category</th><th>Method</th>"
                f"<th>Tier</th><th>Data Sources</th></tr></thead>\n<tbody>{rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Methodology Overview</h2>\n"
            f"<p><strong>Standard:</strong> {standard} | "
            f"<strong>GWP Basis:</strong> {gwp} | "
            f"<strong>Boundary:</strong> {boundary}</p>\n"
            f"{table}</div>"
        )

    def _html_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML 15-category breakdown table."""
        categories = data.get("scope3_categories", [])
        s3_total = self._scope3_total(data)
        cat_lookup = {c.get("category_number"): c for c in categories}
        rows = ""
        for ref in SCOPE3_CATEGORIES:
            num = ref["number"]
            cat_data = cat_lookup.get(num, {})
            name = cat_data.get("category_name", ref["name"])
            em = cat_data.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            tier = cat_data.get("methodology_tier", "-")
            dq = cat_data.get("data_quality_rating", "-")
            yoy = cat_data.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            em_str = _fmt_tco2e(em) if em > 0 else "Not reported"
            stream_class = "upstream" if ref["stream"] == "upstream" else "downstream"
            rows += (
                f'<tr class="{stream_class}"><td>{num}</td><td>{name}</td>'
                f"<td>{em_str}</td><td>{pct}</td><td>{tier}</td>"
                f"<td>{dq}</td><td>{yoy_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Scope 3 Category Breakdown</h2>\n"
            "<table><thead><tr><th>#</th><th>Category</th><th>tCO2e</th>"
            "<th>% of Total</th><th>Tier</th><th>DQ Rating</th>"
            f"<th>YoY</th></tr></thead>\n<tbody>{rows}"
            f'<tr class="total-row"><td></td><td>Total Scope 3</td>'
            f"<td>{_fmt_tco2e(s3_total)}</td><td>100.0%</td>"
            "<td></td><td></td><td></td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_upstream_downstream(self, data: Dict[str, Any]) -> str:
        """Render HTML upstream vs downstream split."""
        upstream = self._upstream_total(data)
        downstream = self._downstream_total(data)
        s3_total = self._scope3_total(data)
        return (
            '<div class="section">\n'
            "<h2>4. Upstream vs Downstream Split</h2>\n"
            "<table><thead><tr><th>Stream</th><th>Categories</th>"
            "<th>tCO2e</th><th>%</th></tr></thead>\n<tbody>"
            f'<tr class="upstream"><td>Upstream</td><td>Cat 1-8</td>'
            f"<td>{_fmt_tco2e(upstream)}</td><td>{_pct_of(upstream, s3_total)}</td></tr>\n"
            f'<tr class="downstream"><td>Downstream</td><td>Cat 9-15</td>'
            f"<td>{_fmt_tco2e(downstream)}</td><td>{_pct_of(downstream, s3_total)}</td></tr>\n"
            f'<tr class="total-row"><td>Total</td><td>All</td>'
            f"<td>{_fmt_tco2e(s3_total)}</td><td>100.0%</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_gas_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML gas-level breakdown."""
        gas_data = data.get("gas_breakdown", {})
        if not gas_data:
            return ""
        s3_total = self._scope3_total(data)
        rows = ""
        for gas in GHG_GASES:
            val = gas_data.get(gas.lower() + "_tco2e", gas_data.get(gas, 0.0))
            if val and val > 0:
                rows += (
                    f"<tr><td>{gas}</td><td>{_fmt_tco2e(val)}</td>"
                    f"<td>{_pct_of(val, s3_total)}</td></tr>\n"
                )
        if not rows:
            return ""
        return (
            '<div class="section">\n'
            "<h2>5. Gas-Level Breakdown</h2>\n"
            "<table><thead><tr><th>Gas</th><th>tCO2e</th>"
            f"<th>%</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_data_quality_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality summary."""
        quality = data.get("data_quality_summary", {})
        if not quality:
            return ""
        overall_dqr = quality.get("overall_dqr_score")
        overall_str = ""
        if overall_dqr is not None:
            label = _quality_label(overall_dqr)
            css = f"quality-{label.lower()}"
            overall_str = (
                f'<p><strong>Overall DQR Score:</strong> '
                f'<span class="{css}">{overall_dqr:.1f} / 5.0 ({label})</span></p>\n'
            )
        cat_quality = quality.get("by_category", [])
        rows = ""
        for cq in cat_quality:
            cat_name = cq.get("category_name", "")
            dqr = cq.get("dqr_score")
            dqr_str = f"{dqr:.1f}" if dqr is not None else "-"
            level = cq.get("quality_level", "-")
            css = f"quality-{level.lower()}" if level != "-" else ""
            source = cq.get("primary_source", "-")
            coverage = cq.get("coverage_pct")
            cov_str = f"{coverage:.0f}%" if coverage is not None else "-"
            rows += (
                f'<tr><td>{cat_name}</td><td>{dqr_str}</td>'
                f'<td class="{css}">{level}</td><td>{source}</td>'
                f"<td>{cov_str}</td></tr>\n"
            )
        table = ""
        if rows:
            table = (
                "<table><thead><tr><th>Category</th><th>DQR</th>"
                "<th>Level</th><th>Source</th><th>Coverage</th>"
                f"</tr></thead>\n<tbody>{rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Data Quality Summary</h2>\n"
            f"{overall_str}{table}</div>"
        )

    def _html_uncertainty_ranges(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty ranges."""
        uncertainty = data.get("uncertainty", {})
        if not uncertainty:
            return ""
        overall = uncertainty.get("overall_uncertainty_pct")
        method = uncertainty.get("method", "Combined analytical + Monte Carlo")
        confidence = uncertainty.get("confidence_level", "95%")
        overall_str = f"+/-{overall:.1f}%" if overall is not None else "N/A"
        per_cat = uncertainty.get("by_category", [])
        rows = ""
        for pc in per_cat:
            cat = pc.get("category_name", "")
            central = _fmt_tco2e(pc.get("central_tco2e"))
            lower = _fmt_tco2e(pc.get("lower_bound_tco2e"))
            upper = _fmt_tco2e(pc.get("upper_bound_tco2e"))
            unc = pc.get("uncertainty_pct")
            unc_str = f"+/-{unc:.1f}%" if unc is not None else "-"
            rows += (
                f"<tr><td>{cat}</td><td>{central}</td><td>{lower}</td>"
                f"<td>{upper}</td><td>{unc_str}</td></tr>\n"
            )
        table = ""
        if rows:
            table = (
                "<table><thead><tr><th>Category</th><th>Central</th>"
                "<th>Lower</th><th>Upper</th><th>Uncertainty</th>"
                f"</tr></thead>\n<tbody>{rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Uncertainty Ranges</h2>\n"
            f"<p><strong>Method:</strong> {method} | "
            f"<strong>Confidence:</strong> {confidence} | "
            f"<strong>Overall:</strong> {overall_str}</p>\n"
            f"{table}</div>"
        )

    def _html_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance status."""
        compliance = data.get("compliance_status", {})
        if not compliance:
            return ""
        frameworks = compliance.get("frameworks", [])
        rows = ""
        for fw in frameworks:
            name = fw.get("framework_name", "")
            status = fw.get("status", "-")
            score = fw.get("score")
            score_str = f"{score:.0f}%" if score is not None else "-"
            gaps = fw.get("key_gaps", "-")
            if isinstance(gaps, list):
                gaps = "; ".join(gaps)
            css = "status-pass" if status in ("Compliant", "Ready") else "status-fail"
            rows += (
                f'<tr><td>{name}</td><td class="{css}">{status}</td>'
                f"<td>{score_str}</td><td>{gaps}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>8. Compliance Status</h2>\n"
            "<table><thead><tr><th>Framework</th><th>Status</th>"
            f"<th>Score</th><th>Key Gaps</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_appendix_ef_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML appendix with emission factor sources."""
        sources = data.get("ef_sources", [])
        if not sources:
            return ""
        rows = ""
        for src in sources:
            cat = src.get("category", "-")
            name = src.get("factor_name", "")
            val = src.get("value")
            val_str = f"{val:.6f}" if val is not None else "-"
            unit = src.get("unit", "")
            source = src.get("source", "")
            version = src.get("version", "-")
            geo = src.get("geography", "Global")
            rows += (
                f"<tr><td>{cat}</td><td>{name}</td><td>{val_str}</td>"
                f"<td>{unit}</td><td>{source}</td><td>{version}</td>"
                f"<td>{geo}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>Appendix A: Emission Factor Sources</h2>\n"
            "<table><thead><tr><th>Category</th><th>Factor</th><th>Value</th>"
            "<th>Unit</th><th>Source</th><th>Version</th>"
            f"<th>Geography</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_appendix_assumptions(self, data: Dict[str, Any]) -> str:
        """Render HTML appendix with assumptions."""
        assumptions = data.get("assumptions", [])
        if not assumptions:
            return ""
        rows = ""
        for i, a in enumerate(assumptions, 1):
            cat = a.get("category", "General")
            assumption = a.get("assumption", "")
            impact = a.get("impact", "-")
            justification = a.get("justification", "-")
            rows += (
                f"<tr><td>{i}</td><td>{cat}</td><td>{assumption}</td>"
                f"<td>{impact}</td><td>{justification}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>Appendix B: Key Assumptions</h2>\n"
            "<table><thead><tr><th>#</th><th>Category</th><th>Assumption</th>"
            f"<th>Impact</th><th>Justification</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON SECTIONS
    # ==================================================================

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary section."""
        s3_total = self._scope3_total(data)
        s1 = data.get("scope1_total_tco2e", 0.0)
        s2 = data.get("scope2_total_tco2e", 0.0)
        full = s1 + s2 + s3_total
        top_cats = sorted(
            data.get("scope3_categories", []),
            key=lambda c: c.get("emissions_tco2e", 0),
            reverse=True,
        )[:5]
        return {
            "scope3_total_tco2e": s3_total,
            "upstream_tco2e": self._upstream_total(data),
            "downstream_tco2e": self._downstream_total(data),
            "scope3_pct_of_total": (s3_total / full * 100) if full > 0 else 0.0,
            "categories_reported": len(data.get("scope3_categories", [])),
            "yoy_change_pct": data.get("yoy_change_pct"),
            "top_5_categories": [
                {
                    "category_number": c.get("category_number"),
                    "category_name": c.get("category_name"),
                    "emissions_tco2e": c.get("emissions_tco2e"),
                    "pct_of_scope3": (c.get("emissions_tco2e", 0) / s3_total * 100)
                    if s3_total > 0 else 0.0,
                }
                for c in top_cats
            ],
        }

    def _json_methodology(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON methodology section."""
        methodology = data.get("methodology_overview", {})
        return {
            "standard": methodology.get("standard", "GHG Protocol Scope 3 Standard"),
            "gwp_basis": methodology.get("gwp_basis", "IPCC AR6"),
            "boundary_approach": methodology.get("boundary_approach", "Operational Control"),
            "category_methods": methodology.get("category_methods", []),
        }

    def _json_categories(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON categories array."""
        categories = data.get("scope3_categories", [])
        s3_total = self._scope3_total(data)
        result = []
        for c in categories:
            em = c.get("emissions_tco2e", 0.0)
            entry = {
                "category_number": c.get("category_number"),
                "category_name": c.get("category_name"),
                "emissions_tco2e": em,
                "pct_of_scope3": (em / s3_total * 100) if s3_total > 0 else 0.0,
                "methodology_tier": c.get("methodology_tier"),
                "data_quality_rating": c.get("data_quality_rating"),
                "yoy_change_pct": c.get("yoy_change_pct"),
                "stream": "upstream" if c.get("category_number", 0) <= 8 else "downstream",
            }
            result.append(entry)
        return result

    def _json_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON data quality section."""
        return data.get("data_quality_summary", {})
