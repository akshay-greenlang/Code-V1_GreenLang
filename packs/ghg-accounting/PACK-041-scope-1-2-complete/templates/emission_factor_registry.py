# -*- coding: utf-8 -*-
"""
EmissionFactorRegistryTemplate - Complete EF Registry Report for PACK-041.

Generates a complete emission factor registry report covering factor summary
statistics, per-source-category factor tables with full metadata (value, unit,
source, version, geography, year, provenance hash), GWP values used across
AR4/AR5/AR6, factor overrides with justification, factor consistency checks,
and data quality assessment of emission factor inputs.

Sections:
    1. Registry Overview & Statistics
    2. Emission Factors by Source Category
    3. GWP Values Used
    4. Factor Overrides & Justifications
    5. Factor Consistency Checks
    6. Data Quality Assessment

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with full factor metadata)

Regulatory References:
    - IPCC AR4, AR5, AR6 GWP tables
    - DEFRA Conversion Factors (annual)
    - EPA eGRID, GHG Emission Factors Hub
    - IEA Emission Factors (annual)

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

GWP_ASSESSMENT_REPORTS = {
    "AR4": {
        "report": "IPCC Fourth Assessment Report (2007)",
        "CO2": 1, "CH4": 25, "N2O": 298, "SF6": 22800, "NF3": 17200,
    },
    "AR5": {
        "report": "IPCC Fifth Assessment Report (2014)",
        "CO2": 1, "CH4": 28, "N2O": 265, "SF6": 23500, "NF3": 16100,
    },
    "AR6": {
        "report": "IPCC Sixth Assessment Report (2021)",
        "CO2": 1, "CH4": 27.9, "N2O": 273, "SF6": 25200, "NF3": 17400,
    },
}


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"


def _fmt_ef(value: Optional[float]) -> str:
    """Format emission factor value with appropriate precision."""
    if value is None:
        return "N/A"
    if abs(value) < 0.001:
        return f"{value:.8f}"
    if abs(value) < 1:
        return f"{value:.6f}"
    return f"{value:,.4f}"


def _truncate_hash(h: Optional[str], length: int = 12) -> str:
    """Truncate a hash string for display."""
    if not h:
        return "-"
    if len(h) > length:
        return f"{h[:length]}..."
    return h


class EmissionFactorRegistryTemplate:
    """
    Complete emission factor registry report template.

    Renders comprehensive emission factor registry reports with per-source
    factor tables, GWP value reference, override justifications, consistency
    checks, and data quality assessment. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = EmissionFactorRegistryTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EmissionFactorRegistryTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
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
        """Render emission factor registry report as Markdown.

        Args:
            data: Validated EF registry data dict.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_factors_by_category(data),
            self._md_gwp_values(data),
            self._md_factor_overrides(data),
            self._md_consistency_checks(data),
            self._md_data_quality(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render emission factor registry report as HTML.

        Args:
            data: Validated EF registry data dict.

        Returns:
            Self-contained HTML document string.
        """
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
            self._html_factors_by_category(data),
            self._html_gwp_values(data),
            self._html_factor_overrides(data),
            self._html_consistency_checks(data),
            self._html_data_quality(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render emission factor registry report as JSON-serializable dict.

        Args:
            data: Validated EF registry data dict.

        Returns:
            Structured dictionary for JSON serialization.
        """
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        factors = data.get("factors", [])
        return {
            "template": "emission_factor_registry",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "statistics": self._json_statistics(data),
            "factors": factors,
            "gwp_basis": self._get_val(data, "gwp_basis", "AR6"),
            "gwp_values": GWP_ASSESSMENT_REPORTS.get(
                self._get_val(data, "gwp_basis", "AR6"), {}
            ),
            "overrides": data.get("factor_overrides", []),
            "consistency_checks": data.get("consistency_checks", []),
            "data_quality": data.get("data_quality_assessment", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Emission Factor Registry - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown registry overview and statistics."""
        factors = data.get("factors", [])
        total_count = len(factors)
        sources = set(f.get("source", "") for f in factors)
        categories = set(f.get("category", "") for f in factors)
        geographies = set(f.get("geography", "") for f in factors)
        gwp_basis = self._get_val(data, "gwp_basis", "AR6")
        overrides = data.get("factor_overrides", [])
        lines = [
            "## 1. Registry Overview & Statistics",
            "",
            "| Statistic | Value |",
            "|-----------|-------|",
            f"| Total Emission Factors | {total_count} |",
            f"| Unique Sources | {len(sources)} |",
            f"| Source Categories | {len(categories)} |",
            f"| Geographic Regions | {len(geographies)} |",
            f"| GWP Basis | {gwp_basis} |",
            f"| Factor Overrides | {len(overrides)} |",
            "",
        ]
        # Source breakdown
        if sources:
            lines.append("### Factors by Source")
            lines.append("")
            lines.append("| Source | Count |")
            lines.append("|--------|-------|")
            source_counts: Dict[str, int] = {}
            for f in factors:
                src = f.get("source", "Unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {src} | {count} |")
        return "\n".join(lines)

    def _md_factors_by_category(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission factors by source category."""
        factors = data.get("factors", [])
        if not factors:
            return "## 2. Emission Factors by Source Category\n\nNo emission factors registered."
        # Group by category
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for f in factors:
            cat = f.get("category", "Uncategorized")
            by_category.setdefault(cat, []).append(f)
        lines = ["## 2. Emission Factors by Source Category"]
        for cat in sorted(by_category.keys()):
            cat_factors = by_category[cat]
            lines.extend([
                "",
                f"### {cat}",
                "",
                "| Factor Name | Value | Unit | Source | Version | Geography | Year | Provenance Hash |",
                "|------------|-------|------|--------|---------|-----------|------|----------------|",
            ])
            for f in sorted(cat_factors, key=lambda x: x.get("factor_name", "")):
                name = f.get("factor_name", "")
                value = _fmt_ef(f.get("value"))
                unit = f.get("unit", "")
                source = f.get("source", "-")
                version = f.get("version", "-")
                geo = f.get("geography", "Global")
                year = str(f.get("year", "-"))
                phash = _truncate_hash(f.get("provenance_hash"))
                lines.append(f"| {name} | {value} | {unit} | {source} | {version} | {geo} | {year} | `{phash}` |")
        return "\n".join(lines)

    def _md_gwp_values(self, data: Dict[str, Any]) -> str:
        """Render Markdown GWP values used."""
        gwp_basis = self._get_val(data, "gwp_basis", "AR6")
        custom_gwps = data.get("custom_gwp_values", {})
        lines = [
            "## 3. GWP Values Used",
            "",
            f"**Primary GWP Basis:** {gwp_basis}",
            "",
            "| Gas | AR4 (100yr) | AR5 (100yr) | AR6 (100yr) | Applied |",
            "|-----|-----------|-----------|-----------|---------|",
        ]
        gases = ["CO2", "CH4", "N2O", "SF6", "NF3"]
        for gas in gases:
            ar4 = GWP_ASSESSMENT_REPORTS["AR4"].get(gas, "-")
            ar5 = GWP_ASSESSMENT_REPORTS["AR5"].get(gas, "-")
            ar6 = GWP_ASSESSMENT_REPORTS["AR6"].get(gas, "-")
            applied = custom_gwps.get(gas, GWP_ASSESSMENT_REPORTS.get(gwp_basis, {}).get(gas, "-"))
            lines.append(f"| {gas} | {ar4} | {ar5} | {ar6} | **{applied}** |")
        # Additional HFC/PFC GWPs
        hfc_gwps = data.get("hfc_gwp_values", [])
        if hfc_gwps:
            lines.extend([
                "",
                "### HFC/PFC GWP Values",
                "",
                "| Refrigerant | GWP (100yr) | Source |",
                "|------------|-----------|--------|",
            ])
            for hfc in hfc_gwps:
                name = hfc.get("refrigerant", "")
                gwp = hfc.get("gwp_100yr", "-")
                source = hfc.get("source", gwp_basis)
                lines.append(f"| {name} | {gwp} | {source} |")
        return "\n".join(lines)

    def _md_factor_overrides(self, data: Dict[str, Any]) -> str:
        """Render Markdown factor overrides with justification."""
        overrides = data.get("factor_overrides", [])
        if not overrides:
            return "## 4. Factor Overrides & Justifications\n\nNo factor overrides applied."
        lines = [
            "## 4. Factor Overrides & Justifications",
            "",
            "| Factor Name | Default Value | Override Value | Unit | Justification | Approved By | Date |",
            "|------------|-------------|---------------|------|--------------|------------|------|",
        ]
        for ovr in overrides:
            name = ovr.get("factor_name", "")
            default = _fmt_ef(ovr.get("default_value"))
            override = _fmt_ef(ovr.get("override_value"))
            unit = ovr.get("unit", "")
            justification = ovr.get("justification", "-")
            approved = ovr.get("approved_by", "-")
            date = ovr.get("approval_date", "-")
            lines.append(f"| {name} | {default} | {override} | {unit} | {justification} | {approved} | {date} |")
        return "\n".join(lines)

    def _md_consistency_checks(self, data: Dict[str, Any]) -> str:
        """Render Markdown factor consistency checks."""
        checks = data.get("consistency_checks", [])
        if not checks:
            return "## 5. Factor Consistency Checks\n\nNo consistency checks performed."
        lines = [
            "## 5. Factor Consistency Checks",
            "",
            "| Check | Description | Status | Detail |",
            "|-------|-----------|--------|--------|",
        ]
        for chk in checks:
            name = chk.get("check_name", "")
            desc = chk.get("description", "")
            status = chk.get("status", "N/A")
            detail = chk.get("detail", "-")
            lines.append(f"| {name} | {desc} | **{status}** | {detail} |")
        passed = sum(1 for c in checks if c.get("status") == "PASS")
        lines.append(f"\n**Result:** {passed}/{len(checks)} checks passed")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality assessment."""
        dq = data.get("data_quality_assessment", {})
        if not dq:
            return "## 6. Data Quality Assessment\n\nNo data quality assessment available."
        overall = dq.get("overall_score", "N/A")
        lines = [
            "## 6. Data Quality Assessment",
            "",
            f"**Overall Quality Score:** {overall}",
            "",
        ]
        criteria = dq.get("criteria", [])
        if criteria:
            lines.append("| Criterion | Score | Weight | Notes |")
            lines.append("|-----------|-------|--------|-------|")
            for crit in criteria:
                name = crit.get("criterion", "")
                score = crit.get("score", "-")
                weight = crit.get("weight", "-")
                notes = crit.get("notes", "-")
                lines.append(f"| {name} | {score} | {weight} | {notes} |")
        recommendations = dq.get("recommendations", [])
        if recommendations:
            lines.append("\n**Recommendations:**")
            for rec in recommendations:
                lines.append(f"- {rec}")
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
            f"<title>Emission Factor Registry - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #6a4c93;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#6a4c93;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".pass{color:#2a9d8f;font-weight:600;}\n"
            ".fail{color:#e63946;font-weight:600;}\n"
            ".warning{color:#e9c46a;font-weight:600;}\n"
            ".override-row{background:#fff3cd;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".stat-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:140px;}\n"
            ".stat-value{font-size:1.4rem;font-weight:700;color:#6a4c93;}\n"
            ".stat-label{font-size:0.8rem;color:#555;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "code{background:#f0f0f0;padding:0.1rem 0.3rem;border-radius:3px;font-size:0.8rem;}\n"
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
            f"<h1>Emission Factor Registry &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview with stat cards."""
        factors = data.get("factors", [])
        total_count = len(factors)
        sources = set(f.get("source", "") for f in factors)
        categories = set(f.get("category", "") for f in factors)
        overrides = data.get("factor_overrides", [])
        cards = [
            ("Total Factors", str(total_count)),
            ("Sources", str(len(sources))),
            ("Categories", str(len(categories))),
            ("Overrides", str(len(overrides))),
        ]
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="stat-card">'
                f'<div class="stat-value">{val}</div>'
                f'<div class="stat-label">{label}</div></div>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>1. Registry Overview</h2>\n"
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_factors_by_category(self, data: Dict[str, Any]) -> str:
        """Render HTML emission factors by category."""
        factors = data.get("factors", [])
        if not factors:
            return '<div class="section"><h2>2. Emission Factors</h2><p>No factors registered.</p></div>'
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for f in factors:
            cat = f.get("category", "Uncategorized")
            by_category.setdefault(cat, []).append(f)
        parts = ['<div class="section">\n<h2>2. Emission Factors by Category</h2>']
        for cat in sorted(by_category.keys()):
            cat_factors = by_category[cat]
            rows = ""
            for f in sorted(cat_factors, key=lambda x: x.get("factor_name", "")):
                name = f.get("factor_name", "")
                value = _fmt_ef(f.get("value"))
                unit = f.get("unit", "")
                source = f.get("source", "-")
                version = f.get("version", "-")
                geo = f.get("geography", "Global")
                year = str(f.get("year", "-"))
                phash = _truncate_hash(f.get("provenance_hash"))
                rows += (
                    f"<tr><td>{name}</td><td>{value}</td><td>{unit}</td>"
                    f"<td>{source}</td><td>{version}</td><td>{geo}</td>"
                    f"<td>{year}</td><td><code>{phash}</code></td></tr>\n"
                )
            parts.append(
                f"<h3>{cat} ({len(cat_factors)} factors)</h3>\n"
                "<table><thead><tr><th>Factor</th><th>Value</th><th>Unit</th>"
                "<th>Source</th><th>Version</th><th>Geography</th>"
                f"<th>Year</th><th>Hash</th></tr></thead>\n<tbody>{rows}</tbody></table>"
            )
        parts.append("</div>")
        return "\n".join(parts)

    def _html_gwp_values(self, data: Dict[str, Any]) -> str:
        """Render HTML GWP values table."""
        gwp_basis = self._get_val(data, "gwp_basis", "AR6")
        gases = ["CO2", "CH4", "N2O", "SF6", "NF3"]
        rows = ""
        for gas in gases:
            ar4 = GWP_ASSESSMENT_REPORTS["AR4"].get(gas, "-")
            ar5 = GWP_ASSESSMENT_REPORTS["AR5"].get(gas, "-")
            ar6 = GWP_ASSESSMENT_REPORTS["AR6"].get(gas, "-")
            applied = GWP_ASSESSMENT_REPORTS.get(gwp_basis, {}).get(gas, "-")
            rows += f"<tr><td>{gas}</td><td>{ar4}</td><td>{ar5}</td><td>{ar6}</td><td><strong>{applied}</strong></td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>3. GWP Values Used</h2>\n"
            f"<p><strong>Primary Basis:</strong> {gwp_basis}</p>\n"
            "<table><thead><tr><th>Gas</th><th>AR4</th><th>AR5</th>"
            f"<th>AR6</th><th>Applied</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_factor_overrides(self, data: Dict[str, Any]) -> str:
        """Render HTML factor overrides."""
        overrides = data.get("factor_overrides", [])
        if not overrides:
            return ""
        rows = ""
        for ovr in overrides:
            name = ovr.get("factor_name", "")
            default = _fmt_ef(ovr.get("default_value"))
            override = _fmt_ef(ovr.get("override_value"))
            unit = ovr.get("unit", "")
            justification = ovr.get("justification", "-")
            approved = ovr.get("approved_by", "-")
            rows += (
                f'<tr class="override-row"><td>{name}</td><td>{default}</td>'
                f"<td>{override}</td><td>{unit}</td><td>{justification}</td>"
                f"<td>{approved}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Factor Overrides</h2>\n"
            "<table><thead><tr><th>Factor</th><th>Default</th><th>Override</th>"
            "<th>Unit</th><th>Justification</th><th>Approved By</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_consistency_checks(self, data: Dict[str, Any]) -> str:
        """Render HTML consistency checks."""
        checks = data.get("consistency_checks", [])
        if not checks:
            return ""
        rows = ""
        for chk in checks:
            name = chk.get("check_name", "")
            desc = chk.get("description", "")
            status = chk.get("status", "N/A")
            css = "pass" if status == "PASS" else ("fail" if status == "FAIL" else "warning")
            detail = chk.get("detail", "-")
            rows += f'<tr><td>{name}</td><td>{desc}</td><td class="{css}">{status}</td><td>{detail}</td></tr>\n'
        passed = sum(1 for c in checks if c.get("status") == "PASS")
        return (
            '<div class="section">\n'
            "<h2>5. Consistency Checks</h2>\n"
            "<table><thead><tr><th>Check</th><th>Description</th>"
            "<th>Status</th><th>Detail</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n"
            f"<p><strong>Result:</strong> {passed}/{len(checks)} passed</p>\n</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality assessment."""
        dq = data.get("data_quality_assessment", {})
        if not dq:
            return ""
        overall = dq.get("overall_score", "N/A")
        criteria = dq.get("criteria", [])
        rows = ""
        for crit in criteria:
            name = crit.get("criterion", "")
            score = crit.get("score", "-")
            weight = crit.get("weight", "-")
            notes = crit.get("notes", "-")
            rows += f"<tr><td>{name}</td><td>{score}</td><td>{weight}</td><td>{notes}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>6. Data Quality Assessment</h2>\n"
            f"<p><strong>Overall Score:</strong> {overall}</p>\n"
            "<table><thead><tr><th>Criterion</th><th>Score</th>"
            "<th>Weight</th><th>Notes</th></tr></thead>\n"
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
    # JSON HELPERS
    # ==================================================================

    def _json_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON statistics section."""
        factors = data.get("factors", [])
        sources = set(f.get("source", "") for f in factors)
        categories = set(f.get("category", "") for f in factors)
        geographies = set(f.get("geography", "") for f in factors)
        return {
            "total_factors": len(factors),
            "unique_sources": len(sources),
            "source_categories": len(categories),
            "geographic_regions": len(geographies),
            "gwp_basis": self._get_val(data, "gwp_basis", "AR6"),
            "override_count": len(data.get("factor_overrides", [])),
            "sources_list": sorted(sources),
            "categories_list": sorted(categories),
        }
