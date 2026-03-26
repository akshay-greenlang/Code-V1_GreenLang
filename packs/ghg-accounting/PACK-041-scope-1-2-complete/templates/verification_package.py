# -*- coding: utf-8 -*-
"""
VerificationPackageTemplate - ISO 14064-3 Verification Package for PACK-041.

Generates a comprehensive verification package aligned with ISO 14064-3
covering organization description, GHG inventory summary tables, methodology
per category (tier and calculation method), emission factor provenance with
SHA-256 hashes, activity data evidence (sources, collection methods),
uncertainty analysis, base year data and recalculations, completeness
statement, calculation audit trail with full SHA-256 chain, and quality
management procedures.

Sections:
    1. Organization Description
    2. GHG Inventory Summary
    3. Methodology per Category
    4. Emission Factor Provenance
    5. Activity Data Evidence
    6. Uncertainty Analysis
    7. Base Year & Recalculations
    8. Completeness Statement
    9. Calculation Audit Trail
    10. Quality Management Procedures

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with full audit chain)

Regulatory References:
    - ISO 14064-3:2019 (Verification & Validation)
    - ISO 14064-1:2018 (Quantification & Reporting)
    - ISAE 3410 (Assurance on GHG Statements)

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


def _truncate_hash(h: Optional[str], length: int = 16) -> str:
    """Truncate hash for display."""
    if not h:
        return "-"
    return h[:length] + "..." if len(h) > length else h


class VerificationPackageTemplate:
    """
    ISO 14064-3 verification package template.

    Renders comprehensive verification packages with organization description,
    inventory summaries, methodology detail, EF provenance with SHA-256 chains,
    activity data evidence, uncertainty analysis, completeness statements,
    and full calculation audit trails. All outputs include SHA-256 provenance
    hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = VerificationPackageTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize VerificationPackageTemplate."""
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
        """Render verification package as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_organization_description(data),
            self._md_inventory_summary(data),
            self._md_methodology_per_category(data),
            self._md_ef_provenance(data),
            self._md_activity_data_evidence(data),
            self._md_uncertainty(data),
            self._md_base_year(data),
            self._md_completeness(data),
            self._md_audit_trail(data),
            self._md_quality_management(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render verification package as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_organization_description(data),
            self._html_inventory_summary(data),
            self._html_methodology_per_category(data),
            self._html_ef_provenance(data),
            self._html_activity_data_evidence(data),
            self._html_uncertainty(data),
            self._html_base_year(data),
            self._html_completeness(data),
            self._html_audit_trail(data),
            self._html_quality_management(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render verification package as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "verification_package",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "organization": data.get("organization", {}),
            "inventory_summary": data.get("inventory_summary", {}),
            "methodology": data.get("methodology_per_category", []),
            "ef_provenance": data.get("ef_provenance", []),
            "activity_data_evidence": data.get("activity_data_evidence", []),
            "uncertainty": data.get("uncertainty", {}),
            "base_year": data.get("base_year_data", {}),
            "completeness": data.get("completeness", {}),
            "audit_trail": data.get("audit_trail", []),
            "quality_management": data.get("quality_management", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Verification Package (ISO 14064-3) - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))} | "
            f"**Assurance Level:** {self._get_val(data, 'assurance_level', 'Limited')}\n\n"
            "---"
        )

    def _md_organization_description(self, data: Dict[str, Any]) -> str:
        """Render Markdown organization description."""
        org = data.get("organization", {})
        lines = [
            "## 1. Organization Description",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Legal Name | {org.get('legal_name', '-')} |",
            f"| Sector | {org.get('sector', '-')} |",
            f"| Headquarters | {org.get('headquarters', '-')} |",
            f"| Consolidation Approach | {org.get('consolidation_approach', 'Operational Control')} |",
            f"| Number of Facilities | {org.get('facility_count', '-')} |",
            f"| Number of Employees | {org.get('employee_count', '-')} |",
            f"| Revenue | {org.get('revenue', '-')} |",
            f"| Reporting Period | {org.get('reporting_period', '-')} |",
        ]
        operations = org.get("operations_description", "")
        if operations:
            lines.append(f"\n**Operations:** {operations}")
        return "\n".join(lines)

    def _md_inventory_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown GHG inventory summary tables."""
        inv = data.get("inventory_summary", {})
        s1 = inv.get("scope1_total_tco2e", 0.0)
        s2_loc = inv.get("scope2_location_tco2e", 0.0)
        s2_mkt = inv.get("scope2_market_tco2e", 0.0)
        lines = [
            "## 2. GHG Inventory Summary",
            "",
            "| Scope | Method | tCO2e |",
            "|-------|--------|-------|",
            f"| Scope 1 | Direct | {_fmt_tco2e(s1)} |",
            f"| Scope 2 | Location-Based | {_fmt_tco2e(s2_loc)} |",
            f"| Scope 2 | Market-Based | {_fmt_tco2e(s2_mkt)} |",
            f"| **Combined** | **Location** | **{_fmt_tco2e(s1 + s2_loc)}** |",
            f"| **Combined** | **Market** | **{_fmt_tco2e(s1 + s2_mkt)}** |",
        ]
        # By category
        categories = inv.get("by_category", [])
        if categories:
            lines.extend([
                "",
                "### By Category",
                "",
                "| Category | tCO2e | % of Scope 1 |",
                "|----------|-------|-------------|",
            ])
            for cat in categories:
                name = cat.get("category", "")
                em = _fmt_tco2e(cat.get("emissions_tco2e"))
                pct = f"{cat.get('pct_of_scope1', 0):.1f}%"
                lines.append(f"| {name} | {em} | {pct} |")
        return "\n".join(lines)

    def _md_methodology_per_category(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology per category."""
        methods = data.get("methodology_per_category", [])
        if not methods:
            return "## 3. Methodology per Category\n\nNo methodology detail provided."
        lines = [
            "## 3. Methodology per Category",
            "",
            "| Category | Tier | Calculation Method | EF Source | GWP Basis | Reference |",
            "|----------|------|-------------------|----------|-----------|-----------|",
        ]
        for m in methods:
            cat = m.get("category", "")
            tier = m.get("tier", "-")
            method = m.get("calculation_method", "")
            ef_src = m.get("ef_source", "-")
            gwp = m.get("gwp_basis", "AR6")
            ref = m.get("reference", "-")
            lines.append(f"| {cat} | {tier} | {method} | {ef_src} | {gwp} | {ref} |")
        return "\n".join(lines)

    def _md_ef_provenance(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission factor provenance."""
        factors = data.get("ef_provenance", [])
        if not factors:
            return "## 4. Emission Factor Provenance\n\nNo EF provenance data."
        lines = [
            "## 4. Emission Factor Provenance",
            "",
            "| Factor | Value | Unit | Source | Geography | Year | SHA-256 Hash |",
            "|--------|-------|------|--------|-----------|------|-------------|",
        ]
        for f in factors:
            name = f.get("factor_name", "")
            value = f"{f.get('value', 0):.6f}" if f.get("value") is not None else "-"
            unit = f.get("unit", "")
            source = f.get("source", "-")
            geo = f.get("geography", "Global")
            year = str(f.get("year", "-"))
            phash = f.get("sha256_hash", "-")
            lines.append(f"| {name} | {value} | {unit} | {source} | {geo} | {year} | `{_truncate_hash(phash)}` |")
        return "\n".join(lines)

    def _md_activity_data_evidence(self, data: Dict[str, Any]) -> str:
        """Render Markdown activity data evidence."""
        evidence = data.get("activity_data_evidence", [])
        if not evidence:
            return "## 5. Activity Data Evidence\n\nNo activity data evidence."
        lines = [
            "## 5. Activity Data Evidence",
            "",
            "| Source Category | Data Source | Collection Method | Frequency | Quality | Verifiable |",
            "|---------------|-----------|------------------|-----------|---------|-----------|",
        ]
        for e in evidence:
            cat = e.get("source_category", "")
            source = e.get("data_source", "")
            method = e.get("collection_method", "")
            freq = e.get("frequency", "")
            quality = e.get("quality_level", "-")
            verifiable = "Yes" if e.get("verifiable", True) else "No"
            lines.append(f"| {cat} | {source} | {method} | {freq} | {quality} | {verifiable} |")
        return "\n".join(lines)

    def _md_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render Markdown uncertainty analysis for verification."""
        unc = data.get("uncertainty", {})
        if not unc:
            return "## 6. Uncertainty Analysis\n\nNo uncertainty analysis."
        lines = [
            "## 6. Uncertainty Analysis",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Method | {unc.get('method', 'IPCC Approach 1')} |",
            f"| Overall Uncertainty | +/-{unc.get('overall_pct', 0):.1f}% |",
            f"| Confidence Level | {unc.get('confidence_level', '95%')} |",
            f"| Scope 1 Uncertainty | +/-{unc.get('scope1_pct', 0):.1f}% |",
            f"| Scope 2 Uncertainty | +/-{unc.get('scope2_pct', 0):.1f}% |",
        ]
        return "\n".join(lines)

    def _md_base_year(self, data: Dict[str, Any]) -> str:
        """Render Markdown base year data and recalculations."""
        by = data.get("base_year_data", {})
        if not by:
            return "## 7. Base Year & Recalculations\n\nNo base year data."
        lines = [
            "## 7. Base Year & Recalculations",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Base Year | {by.get('base_year', '-')} |",
            f"| Base Year Scope 1 | {_fmt_tco2e(by.get('scope1_tco2e'))} |",
            f"| Base Year Scope 2 (Loc) | {_fmt_tco2e(by.get('scope2_location_tco2e'))} |",
            f"| Base Year Scope 2 (Mkt) | {_fmt_tco2e(by.get('scope2_market_tco2e'))} |",
            f"| Recalculation Policy | {by.get('recalculation_policy', 'Significant threshold')} |",
            f"| Significance Threshold | {by.get('significance_threshold', '5%')} |",
        ]
        recalcs = by.get("recalculations", [])
        if recalcs:
            lines.extend([
                "",
                "### Recalculation History",
                "",
                "| Date | Trigger | Original | Revised | Change % |",
                "|------|---------|---------|---------|---------|",
            ])
            for r in recalcs:
                lines.append(
                    f"| {r.get('date', '-')} | {r.get('trigger', '-')} | "
                    f"{_fmt_tco2e(r.get('original_tco2e'))} | "
                    f"{_fmt_tco2e(r.get('revised_tco2e'))} | "
                    f"{r.get('change_pct', 0):.1f}% |"
                )
        return "\n".join(lines)

    def _md_completeness(self, data: Dict[str, Any]) -> str:
        """Render Markdown completeness statement."""
        comp = data.get("completeness", {})
        lines = [
            "## 8. Completeness Statement",
            "",
            f"**Coverage:** {comp.get('coverage_pct', 100):.1f}%",
            f"**Materiality Threshold:** {comp.get('materiality_threshold_pct', 5):.1f}%",
            "",
        ]
        statement = comp.get("statement", "All identified emission sources have been included.")
        lines.append(statement)
        excluded = comp.get("excluded_sources", [])
        if excluded:
            lines.append("\n**Excluded Sources:**")
            for src in excluded:
                lines.append(f"- {src.get('source', '-')}: {src.get('reason', '-')}")
        return "\n".join(lines)

    def _md_audit_trail(self, data: Dict[str, Any]) -> str:
        """Render Markdown calculation audit trail with SHA-256 chain."""
        trail = data.get("audit_trail", [])
        if not trail:
            return "## 9. Calculation Audit Trail\n\nNo audit trail data."
        lines = [
            "## 9. Calculation Audit Trail",
            "",
            "| Step | Description | Input Hash | Output Hash | Timestamp | Agent |",
            "|------|-----------|-----------|------------|-----------|-------|",
        ]
        for step in trail:
            num = step.get("step_number", "")
            desc = step.get("description", "")
            in_hash = _truncate_hash(step.get("input_hash"))
            out_hash = _truncate_hash(step.get("output_hash"))
            ts = step.get("timestamp", "-")
            agent = step.get("agent_name", "-")
            lines.append(f"| {num} | {desc} | `{in_hash}` | `{out_hash}` | {ts} | {agent} |")
        lines.append(f"\n**Chain Valid:** {data.get('chain_valid', 'Not verified')}")
        return "\n".join(lines)

    def _md_quality_management(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality management procedures."""
        qm = data.get("quality_management", {})
        if not qm:
            return "## 10. Quality Management Procedures\n\nNo quality management data."
        procedures = qm.get("procedures", [])
        lines = [
            "## 10. Quality Management Procedures",
            "",
        ]
        if procedures:
            lines.extend([
                "| Procedure | Description | Frequency | Responsible | Last Review |",
                "|-----------|-----------|-----------|------------|------------|",
            ])
            for proc in procedures:
                name = proc.get("procedure_name", "")
                desc = proc.get("description", "")
                freq = proc.get("frequency", "-")
                resp = proc.get("responsible", "-")
                last = proc.get("last_review", "-")
                lines.append(f"| {name} | {desc} | {freq} | {resp} | {last} |")
        internal_controls = qm.get("internal_controls", "")
        if internal_controls:
            lines.append(f"\n**Internal Controls:** {internal_controls}")
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
            f"<title>Verification Package - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #1d3557;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#457b9d;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".audit-step{border-left:3px solid #1d3557;}\n"
            ".hash{font-family:monospace;font-size:0.8rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
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
        assurance = self._get_val(data, "assurance_level", "Limited")
        return (
            '<div class="section">\n'
            f"<h1>Verification Package (ISO 14064-3) &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Assurance:</strong> {assurance}</p>\n<hr>\n</div>"
        )

    def _html_organization_description(self, data: Dict[str, Any]) -> str:
        """Render HTML organization description."""
        org = data.get("organization", {})
        rows = ""
        for field, key in [("Legal Name", "legal_name"), ("Sector", "sector"),
                           ("Headquarters", "headquarters"), ("Consolidation", "consolidation_approach"),
                           ("Facilities", "facility_count"), ("Employees", "employee_count")]:
            rows += f"<tr><td>{field}</td><td>{org.get(key, '-')}</td></tr>\n"
        return (
            '<div class="section">\n<h2>1. Organization</h2>\n'
            "<table><thead><tr><th>Field</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_inventory_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML inventory summary."""
        inv = data.get("inventory_summary", {})
        s1 = inv.get("scope1_total_tco2e", 0.0)
        s2_loc = inv.get("scope2_location_tco2e", 0.0)
        s2_mkt = inv.get("scope2_market_tco2e", 0.0)
        return (
            '<div class="section">\n<h2>2. GHG Inventory Summary</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Method</th><th>tCO2e</th></tr></thead>\n<tbody>"
            f"<tr><td>Scope 1</td><td>Direct</td><td>{_fmt_tco2e(s1)}</td></tr>\n"
            f"<tr><td>Scope 2</td><td>Location</td><td>{_fmt_tco2e(s2_loc)}</td></tr>\n"
            f"<tr><td>Scope 2</td><td>Market</td><td>{_fmt_tco2e(s2_mkt)}</td></tr>\n"
            f"<tr style='font-weight:bold'><td>Combined</td><td>Location</td><td>{_fmt_tco2e(s1 + s2_loc)}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_methodology_per_category(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology per category."""
        methods = data.get("methodology_per_category", [])
        if not methods:
            return ""
        rows = ""
        for m in methods:
            rows += (
                f"<tr><td>{m.get('category', '')}</td><td>{m.get('tier', '-')}</td>"
                f"<td>{m.get('calculation_method', '')}</td><td>{m.get('ef_source', '-')}</td>"
                f"<td>{m.get('gwp_basis', 'AR6')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Methodology per Category</h2>\n'
            "<table><thead><tr><th>Category</th><th>Tier</th><th>Method</th>"
            "<th>EF Source</th><th>GWP</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_ef_provenance(self, data: Dict[str, Any]) -> str:
        """Render HTML EF provenance."""
        factors = data.get("ef_provenance", [])
        if not factors:
            return ""
        rows = ""
        for f in factors:
            name = f.get("factor_name", "")
            value = f"{f.get('value', 0):.6f}" if f.get("value") is not None else "-"
            source = f.get("source", "-")
            phash = _truncate_hash(f.get("sha256_hash"))
            rows += f'<tr><td>{name}</td><td>{value}</td><td>{f.get("unit", "")}</td><td>{source}</td><td class="hash">{phash}</td></tr>\n'
        return (
            '<div class="section">\n<h2>4. EF Provenance</h2>\n'
            "<table><thead><tr><th>Factor</th><th>Value</th><th>Unit</th>"
            "<th>Source</th><th>SHA-256</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_activity_data_evidence(self, data: Dict[str, Any]) -> str:
        """Render HTML activity data evidence."""
        evidence = data.get("activity_data_evidence", [])
        if not evidence:
            return ""
        rows = ""
        for e in evidence:
            rows += (
                f"<tr><td>{e.get('source_category', '')}</td><td>{e.get('data_source', '')}</td>"
                f"<td>{e.get('collection_method', '')}</td><td>{e.get('frequency', '')}</td>"
                f"<td>{e.get('quality_level', '-')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Activity Data Evidence</h2>\n'
            "<table><thead><tr><th>Category</th><th>Source</th><th>Method</th>"
            "<th>Freq</th><th>Quality</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty."""
        unc = data.get("uncertainty", {})
        if not unc:
            return ""
        return (
            '<div class="section">\n<h2>6. Uncertainty</h2>\n'
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>Method</td><td>{unc.get('method', 'IPCC Approach 1')}</td></tr>\n"
            f"<tr><td>Overall</td><td>+/-{unc.get('overall_pct', 0):.1f}%</td></tr>\n"
            f"<tr><td>Confidence</td><td>{unc.get('confidence_level', '95%')}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_base_year(self, data: Dict[str, Any]) -> str:
        """Render HTML base year."""
        by = data.get("base_year_data", {})
        if not by:
            return ""
        return (
            '<div class="section">\n<h2>7. Base Year</h2>\n'
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>Year</td><td>{by.get('base_year', '-')}</td></tr>\n"
            f"<tr><td>Scope 1</td><td>{_fmt_tco2e(by.get('scope1_tco2e'))}</td></tr>\n"
            f"<tr><td>Scope 2 (Loc)</td><td>{_fmt_tco2e(by.get('scope2_location_tco2e'))}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_completeness(self, data: Dict[str, Any]) -> str:
        """Render HTML completeness."""
        comp = data.get("completeness", {})
        return (
            '<div class="section">\n<h2>8. Completeness</h2>\n'
            f"<p><strong>Coverage:</strong> {comp.get('coverage_pct', 100):.1f}% | "
            f"<strong>Threshold:</strong> {comp.get('materiality_threshold_pct', 5):.1f}%</p>\n"
            f"<p>{comp.get('statement', 'All sources included.')}</p>\n</div>"
        )

    def _html_audit_trail(self, data: Dict[str, Any]) -> str:
        """Render HTML audit trail."""
        trail = data.get("audit_trail", [])
        if not trail:
            return ""
        rows = ""
        for step in trail:
            in_h = _truncate_hash(step.get("input_hash"))
            out_h = _truncate_hash(step.get("output_hash"))
            rows += (
                f'<tr class="audit-step"><td>{step.get("step_number", "")}</td>'
                f'<td>{step.get("description", "")}</td>'
                f'<td class="hash">{in_h}</td><td class="hash">{out_h}</td>'
                f'<td>{step.get("agent_name", "-")}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>9. Audit Trail</h2>\n'
            "<table><thead><tr><th>#</th><th>Step</th><th>Input Hash</th>"
            "<th>Output Hash</th><th>Agent</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_quality_management(self, data: Dict[str, Any]) -> str:
        """Render HTML quality management."""
        qm = data.get("quality_management", {})
        if not qm:
            return ""
        procedures = qm.get("procedures", [])
        rows = ""
        for proc in procedures:
            rows += (
                f"<tr><td>{proc.get('procedure_name', '')}</td>"
                f"<td>{proc.get('description', '')}</td>"
                f"<td>{proc.get('frequency', '-')}</td>"
                f"<td>{proc.get('responsible', '-')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>10. Quality Management</h2>\n'
            "<table><thead><tr><th>Procedure</th><th>Description</th>"
            "<th>Frequency</th><th>Responsible</th></tr></thead>\n"
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
