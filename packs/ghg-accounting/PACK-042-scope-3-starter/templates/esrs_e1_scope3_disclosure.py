# -*- coding: utf-8 -*-
"""
ESRSE1Scope3DisclosureTemplate - ESRS E1-6 Scope 3 Disclosure for PACK-042.

Generates ESRS E1-6 para 44-46 formatted Scope 3 disclosure covering
total and per-category emissions, methodology description per ESRS
requirements, data quality statement, phase-in compliance status
(2025 Cat 1-3, 2029 all), XBRL data points, significant Scope 3
categories rationale, and estimation methodology and assumptions.

Sections:
    1. E1-6 Scope 3 Total and Per-Category
    2. Methodology Description
    3. Data Quality Statement
    4. Phase-In Compliance Status
    5. XBRL Data Points
    6. Significant Categories Rationale
    7. Estimation Methodology and Assumptions

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, EU blue theme)
    - JSON (structured with XBRL-ready data)

Regulatory References:
    - ESRS E1 Climate Change, E1-6 para 44-46
    - EFRAG Implementation Guidance
    - EU Taxonomy Regulation
    - CSRD (Corporate Sustainability Reporting Directive)

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

# ESRS phase-in categories
PHASE_IN_2025 = [1, 2, 3]  # Required from 2025 reporting
PHASE_IN_2029 = list(range(1, 16))  # All 15 categories from 2029


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M tCO2e"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K tCO2e"
    return f"{value:,.1f} tCO2e"


def _pct_of(part: float, total: float) -> str:
    """Percentage of total, formatted."""
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


class ESRSE1Scope3DisclosureTemplate:
    """
    ESRS E1-6 para 44-46 formatted Scope 3 disclosure template.

    Renders ESRS-compliant Scope 3 disclosures with total and per-
    category emissions, methodology descriptions, data quality
    statements, phase-in compliance status, XBRL data points,
    significant category rationales, and estimation methodology.
    All outputs include SHA-256 provenance hashing for audit trails.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = ESRSE1Scope3DisclosureTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSE1Scope3DisclosureTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def _scope3_total(self, data: Dict[str, Any]) -> float:
        """Get total Scope 3 emissions."""
        categories = data.get("scope3_categories", [])
        return sum(c.get("emissions_tco2e", 0.0) for c in categories)

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render ESRS E1 Scope 3 disclosure as Markdown.

        Args:
            data: Validated ESRS disclosure data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_scope3_emissions(data),
            self._md_methodology(data),
            self._md_data_quality(data),
            self._md_phase_in(data),
            self._md_xbrl_datapoints(data),
            self._md_significant_categories(data),
            self._md_estimation_methodology(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render ESRS E1 Scope 3 disclosure as HTML.

        Args:
            data: Validated ESRS disclosure data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_scope3_emissions(data),
            self._html_methodology(data),
            self._html_data_quality(data),
            self._html_phase_in(data),
            self._html_xbrl_datapoints(data),
            self._html_significant_categories(data),
            self._html_estimation_methodology(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render ESRS E1 Scope 3 disclosure as JSON-serializable dict.

        Args:
            data: Validated ESRS disclosure data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary with XBRL-ready data points.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        s3_total = self._scope3_total(data)
        return {
            "template": "esrs_e1_scope3_disclosure",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "esrs_standard": "E1 Climate Change",
            "disclosure_requirement": "E1-6 para 44-46",
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "scope3_total_tco2e": s3_total,
            "scope3_categories": data.get("scope3_categories", []),
            "methodology": data.get("methodology", {}),
            "data_quality_statement": data.get("data_quality_statement", {}),
            "phase_in_compliance": self._json_phase_in(data),
            "xbrl_data_points": self._json_xbrl(data, s3_total),
            "significant_categories": data.get("significant_categories", []),
            "estimation_methodology": data.get("estimation_methodology", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# ESRS E1-6 Scope 3 GHG Emissions Disclosure - {company}\n\n"
            f"**Reporting Year:** {year} | "
            "**Standard:** ESRS E1 Climate Change | "
            "**Paragraph:** E1-6 para 44-46\n\n"
            "---"
        )

    def _md_scope3_emissions(self, data: Dict[str, Any]) -> str:
        """Render Markdown Scope 3 emissions per E1-6 format."""
        categories = data.get("scope3_categories", [])
        s3_total = self._scope3_total(data)
        biogenic = data.get("biogenic_co2_tco2e")
        lines = [
            "## 1. E1-6 Scope 3 GHG Emissions",
            "",
            f"**Total Scope 3 GHG Emissions:** {_fmt_tco2e(s3_total)}",
        ]
        if biogenic is not None:
            lines.append(f"**Biogenic CO2 (separate):** {_fmt_tco2e(biogenic)}")
        lines.extend([
            "",
            "| Cat # | Category | tCO2e | % of Scope 3 | Significant? | Method |",
            "|-------|----------|-------|-------------|-------------|--------|",
        ])
        for cat in categories:
            num = cat.get("category_number", "?")
            name = cat.get("category_name", "")
            em = cat.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            significant = "Yes" if cat.get("is_significant", False) else "No"
            method = cat.get("method", "-")
            em_str = _fmt_tco2e(em) if em > 0 else "Not reported"
            lines.append(
                f"| {num} | {name} | {em_str} | {pct} | {significant} | {method} |"
            )
        lines.append(f"\n**Total:** {_fmt_tco2e(s3_total)}")
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology description per ESRS."""
        methodology = data.get("methodology", {})
        if not methodology:
            return "## 2. Methodology Description\n\nNo methodology description."
        standard = methodology.get("standard", "GHG Protocol Scope 3 Standard")
        gwp = methodology.get("gwp_basis", "IPCC AR6")
        lines = [
            "## 2. Methodology Description (per ESRS E1 requirements)",
            "",
            f"**Standard Applied:** {standard}",
            f"**GWP Values:** {gwp}",
            "",
        ]
        description = methodology.get("description", "")
        if description:
            lines.append(description)
        category_methods = methodology.get("category_methods", [])
        if category_methods:
            lines.append("")
            lines.append("| Category | Calculation Approach | Data Sources |")
            lines.append("|----------|--------------------|--------------| ")
            for cm in category_methods:
                cat = cm.get("category_name", "")
                approach = cm.get("approach", "-")
                sources = cm.get("data_sources", "-")
                lines.append(f"| {cat} | {approach} | {sources} |")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality statement."""
        dq = data.get("data_quality_statement", {})
        if not dq:
            return "## 3. Data Quality Statement\n\nNo data quality statement."
        overall = dq.get("overall_dqr_score")
        primary_pct = dq.get("primary_data_pct")
        lines = [
            "## 3. Data Quality Statement",
            "",
        ]
        if overall is not None:
            lines.append(f"**Overall Data Quality Rating:** {overall:.1f} / 5.0")
        if primary_pct is not None:
            lines.append(f"**Primary Data Coverage:** {primary_pct:.0f}%")
        statement = dq.get("statement", "")
        if statement:
            lines.append(f"\n{statement}")
        limitations = dq.get("limitations", [])
        if limitations:
            lines.append("\n**Known Limitations:**")
            for lim in limitations:
                lines.append(f"- {lim}")
        return "\n".join(lines)

    def _md_phase_in(self, data: Dict[str, Any]) -> str:
        """Render Markdown phase-in compliance status."""
        phase_in = data.get("phase_in_compliance", {})
        reporting_year = self._get_val(data, "reporting_year")
        categories = data.get("scope3_categories", [])
        reported_nums = [c.get("category_number") for c in categories if c.get("emissions_tco2e", 0) > 0]
        lines = [
            "## 4. Phase-In Compliance Status",
            "",
            "| Phase | Required Categories | Deadline | Status |",
            "|-------|--------------------|----------|--------|",
        ]
        # Phase 1: 2025 - Categories 1, 2, 3
        phase1_required = set(PHASE_IN_2025)
        phase1_reported = phase1_required.intersection(set(reported_nums))
        phase1_status = "Compliant" if phase1_reported == phase1_required else "Non-compliant"
        lines.append(
            f"| Phase 1 (2025) | Cat 1, 2, 3 | FY 2025 | **{phase1_status}** |"
        )
        # Phase 2: 2029 - All 15 categories
        phase2_required = set(PHASE_IN_2029)
        phase2_reported = phase2_required.intersection(set(reported_nums))
        phase2_status = "Compliant" if phase2_reported == phase2_required else "In Progress"
        lines.append(
            f"| Phase 2 (2029) | Cat 1-15 | FY 2029 | **{phase2_status}** |"
        )
        lines.append(f"\n**Categories Currently Reported:** {len(reported_nums)} of 15")
        missing = sorted(phase2_required - set(reported_nums))
        if missing:
            lines.append(f"**Categories Not Yet Reported:** {', '.join(str(m) for m in missing)}")
        return "\n".join(lines)

    def _md_xbrl_datapoints(self, data: Dict[str, Any]) -> str:
        """Render Markdown XBRL data points."""
        xbrl = data.get("xbrl_data_points", [])
        s3_total = self._scope3_total(data)
        if not xbrl:
            # Generate standard XBRL data points
            xbrl = self._generate_xbrl_points(data, s3_total)
        lines = [
            "## 5. XBRL Data Points",
            "",
            "| XBRL Tag | Value | Unit | Notes |",
            "|----------|-------|------|-------|",
        ]
        for point in xbrl:
            tag = point.get("tag", "")
            value = point.get("value", "")
            unit = point.get("unit", "")
            notes = point.get("notes", "-")
            lines.append(f"| `{tag}` | {value} | {unit} | {notes} |")
        return "\n".join(lines)

    def _md_significant_categories(self, data: Dict[str, Any]) -> str:
        """Render Markdown significant categories rationale."""
        significant = data.get("significant_categories", [])
        if not significant:
            return "## 6. Significant Scope 3 Categories Rationale\n\nNo significance assessment."
        lines = [
            "## 6. Significant Scope 3 Categories Rationale",
            "",
            "| Category | tCO2e | % of Scope 3 | Rationale for Significance |",
            "|----------|-------|-------------|---------------------------|",
        ]
        s3_total = self._scope3_total(data)
        for cat in significant:
            name = cat.get("category_name", "")
            em = cat.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            rationale = cat.get("rationale", "-")
            lines.append(f"| {name} | {_fmt_tco2e(em)} | {pct} | {rationale} |")
        return "\n".join(lines)

    def _md_estimation_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown estimation methodology and assumptions."""
        est = data.get("estimation_methodology", {})
        if not est:
            return "## 7. Estimation Methodology and Assumptions\n\nNo estimation details."
        lines = [
            "## 7. Estimation Methodology and Assumptions",
            "",
        ]
        description = est.get("description", "")
        if description:
            lines.append(description)
            lines.append("")
        assumptions = est.get("key_assumptions", [])
        if assumptions:
            lines.append("**Key Assumptions:**")
            for a in assumptions:
                lines.append(f"- {a}")
            lines.append("")
        proxies = est.get("proxies_used", [])
        if proxies:
            lines.append("**Proxies Used:**")
            lines.append("")
            lines.append("| Proxy | Applied To | Justification |")
            lines.append("|-------|-----------|---------------|")
            for p in proxies:
                proxy = p.get("proxy", "")
                applied = p.get("applied_to", "-")
                justification = p.get("justification", "-")
                lines.append(f"| {proxy} | {applied} | {justification} |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}*\n"
            f"*ESRS E1-6 Scope 3 Disclosure*\n"
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
            f"<title>ESRS E1-6 Scope 3 Disclosure - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#003399;border-bottom:3px solid #003399;padding-bottom:0.5rem;}\n"
            "h2{color:#003399;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}\n"
            "h3{color:#0052CC;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#e6ecf5;font-weight:600;color:#003399;}\n"
            "tr:nth-child(even){background:#f5f7fb;}\n"
            ".eu-badge{display:inline-block;background:#003399;color:#fff;"
            "padding:0.2rem 0.6rem;border-radius:4px;font-size:0.8rem;}\n"
            ".metric-card{display:inline-block;background:#e6ecf5;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;"
            "border-top:3px solid #003399;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#003399;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".significant-yes{color:#003399;font-weight:700;}\n"
            ".significant-no{color:#95A5A6;}\n"
            ".compliant{color:#27AE60;font-weight:700;}\n"
            ".non-compliant{color:#E74C3C;font-weight:700;}\n"
            ".in-progress{color:#F39C12;font-weight:700;}\n"
            "code{background:#e6ecf5;padding:0.1rem 0.3rem;border-radius:3px;"
            "font-size:0.8rem;color:#003399;}\n"
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
            f"<h1>ESRS E1-6 Scope 3 GHG Emissions Disclosure &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            '<span class="eu-badge">ESRS E1 Climate Change</span> '
            '<span class="eu-badge">E1-6 para 44-46</span></p>\n'
            "<hr>\n</div>"
        )

    def _html_scope3_emissions(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 3 emissions table."""
        categories = data.get("scope3_categories", [])
        s3_total = self._scope3_total(data)
        biogenic = data.get("biogenic_co2_tco2e")
        cards = [("Total Scope 3", _fmt_tco2e(s3_total))]
        if biogenic is not None:
            cards.append(("Biogenic CO2", _fmt_tco2e(biogenic)))
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        rows = ""
        for cat in categories:
            num = cat.get("category_number", "?")
            name = cat.get("category_name", "")
            em = cat.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            significant = cat.get("is_significant", False)
            sig_str = "Yes" if significant else "No"
            sig_css = "significant-yes" if significant else "significant-no"
            method = cat.get("method", "-")
            em_str = _fmt_tco2e(em) if em > 0 else "Not reported"
            rows += (
                f"<tr><td>{num}</td><td>{name}</td><td>{em_str}</td>"
                f'<td>{pct}</td><td class="{sig_css}">{sig_str}</td>'
                f"<td>{method}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>1. E1-6 Scope 3 GHG Emissions</h2>\n"
            f"<div>{card_html}</div>\n"
            "<table><thead><tr><th>#</th><th>Category</th><th>tCO2e</th>"
            "<th>%</th><th>Significant?</th>"
            f"<th>Method</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology description."""
        methodology = data.get("methodology", {})
        if not methodology:
            return ""
        standard = methodology.get("standard", "GHG Protocol Scope 3 Standard")
        gwp = methodology.get("gwp_basis", "IPCC AR6")
        description = methodology.get("description", "")
        category_methods = methodology.get("category_methods", [])
        rows = ""
        for cm in category_methods:
            cat = cm.get("category_name", "")
            approach = cm.get("approach", "-")
            sources = cm.get("data_sources", "-")
            rows += f"<tr><td>{cat}</td><td>{approach}</td><td>{sources}</td></tr>\n"
        table = ""
        if rows:
            table = (
                "<table><thead><tr><th>Category</th><th>Approach</th>"
                f"<th>Data Sources</th></tr></thead>\n<tbody>{rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Methodology Description</h2>\n"
            f"<p><strong>Standard:</strong> {standard} | <strong>GWP:</strong> {gwp}</p>\n"
            + (f"<p>{description}</p>\n" if description else "")
            + f"{table}</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality statement."""
        dq = data.get("data_quality_statement", {})
        if not dq:
            return ""
        overall = dq.get("overall_dqr_score")
        primary = dq.get("primary_data_pct")
        statement = dq.get("statement", "")
        overall_str = f"<p><strong>Overall DQR:</strong> {overall:.1f} / 5.0</p>\n" if overall else ""
        primary_str = f"<p><strong>Primary Data:</strong> {primary:.0f}%</p>\n" if primary else ""
        return (
            '<div class="section">\n'
            "<h2>3. Data Quality Statement</h2>\n"
            f"{overall_str}{primary_str}"
            f"<p>{statement}</p>\n</div>"
        )

    def _html_phase_in(self, data: Dict[str, Any]) -> str:
        """Render HTML phase-in compliance status."""
        categories = data.get("scope3_categories", [])
        reported_nums = [c.get("category_number") for c in categories if c.get("emissions_tco2e", 0) > 0]
        phase1_ok = set(PHASE_IN_2025).issubset(set(reported_nums))
        phase2_ok = set(PHASE_IN_2029).issubset(set(reported_nums))
        p1_status = "Compliant" if phase1_ok else "Non-compliant"
        p1_css = "compliant" if phase1_ok else "non-compliant"
        p2_status = "Compliant" if phase2_ok else "In Progress"
        p2_css = "compliant" if phase2_ok else "in-progress"
        return (
            '<div class="section">\n'
            "<h2>4. Phase-In Compliance Status</h2>\n"
            "<table><thead><tr><th>Phase</th><th>Required</th><th>Deadline</th>"
            "<th>Status</th></tr></thead>\n<tbody>"
            f'<tr><td>Phase 1</td><td>Cat 1, 2, 3</td><td>FY 2025</td>'
            f'<td class="{p1_css}">{p1_status}</td></tr>\n'
            f'<tr><td>Phase 2</td><td>Cat 1-15</td><td>FY 2029</td>'
            f'<td class="{p2_css}">{p2_status}</td></tr>\n'
            "</tbody></table>\n"
            f"<p><strong>Categories Reported:</strong> {len(reported_nums)} of 15</p>\n</div>"
        )

    def _html_xbrl_datapoints(self, data: Dict[str, Any]) -> str:
        """Render HTML XBRL data points."""
        xbrl = data.get("xbrl_data_points", [])
        s3_total = self._scope3_total(data)
        if not xbrl:
            xbrl = self._generate_xbrl_points(data, s3_total)
        rows = ""
        for point in xbrl:
            tag = point.get("tag", "")
            value = point.get("value", "")
            unit = point.get("unit", "")
            notes = point.get("notes", "-")
            rows += f"<tr><td><code>{tag}</code></td><td>{value}</td><td>{unit}</td><td>{notes}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>5. XBRL Data Points</h2>\n"
            "<table><thead><tr><th>XBRL Tag</th><th>Value</th>"
            f"<th>Unit</th><th>Notes</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_significant_categories(self, data: Dict[str, Any]) -> str:
        """Render HTML significant categories rationale."""
        significant = data.get("significant_categories", [])
        if not significant:
            return ""
        s3_total = self._scope3_total(data)
        rows = ""
        for cat in significant:
            name = cat.get("category_name", "")
            em = cat.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3_total)
            rationale = cat.get("rationale", "-")
            rows += f"<tr><td>{name}</td><td>{_fmt_tco2e(em)}</td><td>{pct}</td><td>{rationale}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>6. Significant Categories Rationale</h2>\n"
            "<table><thead><tr><th>Category</th><th>tCO2e</th><th>%</th>"
            f"<th>Rationale</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_estimation_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML estimation methodology and assumptions."""
        est = data.get("estimation_methodology", {})
        if not est:
            return ""
        description = est.get("description", "")
        assumptions = est.get("key_assumptions", [])
        proxies = est.get("proxies_used", [])
        desc_html = f"<p>{description}</p>\n" if description else ""
        assumptions_html = ""
        if assumptions:
            assumptions_html = "<p><strong>Key Assumptions:</strong></p><ul>\n"
            for a in assumptions:
                assumptions_html += f"<li>{a}</li>\n"
            assumptions_html += "</ul>\n"
        proxies_html = ""
        if proxies:
            proxy_rows = ""
            for p in proxies:
                proxy = p.get("proxy", "")
                applied = p.get("applied_to", "-")
                justification = p.get("justification", "-")
                proxy_rows += f"<tr><td>{proxy}</td><td>{applied}</td><td>{justification}</td></tr>\n"
            proxies_html = (
                "<p><strong>Proxies Used:</strong></p>\n"
                "<table><thead><tr><th>Proxy</th><th>Applied To</th>"
                f"<th>Justification</th></tr></thead>\n<tbody>{proxy_rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Estimation Methodology and Assumptions</h2>\n"
            f"{desc_html}{assumptions_html}{proxies_html}</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}</p>\n"
            "<p>ESRS E1-6 Scope 3 Disclosure</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON / XBRL HELPERS
    # ==================================================================

    def _generate_xbrl_points(
        self, data: Dict[str, Any], s3_total: float
    ) -> List[Dict[str, Any]]:
        """Generate standard XBRL data points from data."""
        points = [
            {
                "tag": "esrs:Scope3GHGEmissions",
                "value": f"{s3_total:.2f}",
                "unit": "tCO2e",
                "notes": "Total Scope 3 indirect GHG emissions",
            },
        ]
        biogenic = data.get("biogenic_co2_tco2e")
        if biogenic is not None:
            points.append({
                "tag": "esrs:BiogenicCO2Scope3",
                "value": f"{biogenic:.2f}",
                "unit": "tCO2e",
                "notes": "Biogenic CO2 reported separately",
            })
        for cat in data.get("scope3_categories", []):
            num = cat.get("category_number", 0)
            em = cat.get("emissions_tco2e", 0.0)
            if em > 0:
                points.append({
                    "tag": f"esrs:Scope3Category{num}GHGEmissions",
                    "value": f"{em:.2f}",
                    "unit": "tCO2e",
                    "notes": cat.get("category_name", ""),
                })
        return points

    def _json_phase_in(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON phase-in compliance data."""
        categories = data.get("scope3_categories", [])
        reported = [c.get("category_number") for c in categories if c.get("emissions_tco2e", 0) > 0]
        return {
            "phase_1": {
                "required_categories": PHASE_IN_2025,
                "deadline": "FY 2025",
                "compliant": set(PHASE_IN_2025).issubset(set(reported)),
                "reported": sorted(set(PHASE_IN_2025).intersection(set(reported))),
                "missing": sorted(set(PHASE_IN_2025) - set(reported)),
            },
            "phase_2": {
                "required_categories": PHASE_IN_2029,
                "deadline": "FY 2029",
                "compliant": set(PHASE_IN_2029).issubset(set(reported)),
                "reported": sorted(set(PHASE_IN_2029).intersection(set(reported))),
                "missing": sorted(set(PHASE_IN_2029) - set(reported)),
            },
            "total_reported": len(reported),
            "total_required": 15,
        }

    def _json_xbrl(
        self, data: Dict[str, Any], s3_total: float
    ) -> List[Dict[str, Any]]:
        """Build JSON XBRL data points."""
        xbrl = data.get("xbrl_data_points", [])
        if not xbrl:
            xbrl = self._generate_xbrl_points(data, s3_total)
        return xbrl
