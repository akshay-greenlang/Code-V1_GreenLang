# -*- coding: utf-8 -*-
"""
Scope3VerificationPackageTemplate - ISO 14064-3 / ISAE 3410 Evidence for PACK-042.

Generates a verification-ready evidence package for third-party verifiers
aligned with ISO 14064-3:2019 and ISAE 3410. Includes methodology summary
per category, data source inventory, emission factor registry with SHA-256
provenance, calculation log, assumption register, materiality assessment,
completeness statement, data quality statement, uncertainty statement,
and organization boundary reference.

Sections:
    1. Methodology Summary per Category
    2. Data Source Inventory
    3. Emission Factor Registry with Provenance
    4. Calculation Log with SHA-256 Hashes
    5. Assumption Register
    6. Materiality Assessment
    7. Completeness Statement
    8. Data Quality Statement
    9. Uncertainty Statement
    10. Organization Boundary Reference

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, audit black theme)
    - JSON (structured with full audit chain)

Regulatory References:
    - ISO 14064-3:2019 (Verification & Validation)
    - ISAE 3410 (Assurance on GHG Statements)
    - GHG Protocol Scope 3 Standard

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


class Scope3VerificationPackageTemplate:
    """
    ISO 14064-3 / ISAE 3410 evidence bundle template for Scope 3.

    Renders verification-ready evidence packages with methodology
    summaries, data source inventories, emission factor registries
    with SHA-256 provenance chains, calculation logs, assumption
    registers, materiality assessments, and completeness/quality/
    uncertainty statements. Designed for third-party verifiers.
    All outputs include SHA-256 provenance hashing for audit trails.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = Scope3VerificationPackageTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope3VerificationPackageTemplate."""
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

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render verification package as Markdown.

        Args:
            data: Validated verification package data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_methodology_summary(data),
            self._md_data_sources(data),
            self._md_ef_registry(data),
            self._md_calculation_log(data),
            self._md_assumptions(data),
            self._md_materiality(data),
            self._md_completeness(data),
            self._md_data_quality(data),
            self._md_uncertainty(data),
            self._md_boundary(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render verification package as HTML.

        Args:
            data: Validated verification package data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_methodology_summary(data),
            self._html_data_sources(data),
            self._html_ef_registry(data),
            self._html_calculation_log(data),
            self._html_assumptions(data),
            self._html_materiality(data),
            self._html_completeness(data),
            self._html_data_quality(data),
            self._html_uncertainty(data),
            self._html_boundary(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render verification package as JSON-serializable dict.

        Args:
            data: Validated verification package data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "scope3_verification_package",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "methodology_summary": data.get("methodology_summary", []),
            "data_sources": data.get("data_sources", []),
            "ef_registry": data.get("ef_registry", []),
            "calculation_log": data.get("calculation_log", []),
            "assumptions": data.get("assumptions", []),
            "materiality": data.get("materiality", {}),
            "completeness": data.get("completeness", {}),
            "data_quality": data.get("data_quality_statement", {}),
            "uncertainty": data.get("uncertainty_statement", {}),
            "boundary": data.get("organization_boundary", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 3 Verification Package - {company}\n\n"
            f"**Reporting Year:** {year} | "
            "**Standard:** ISO 14064-3:2019 / ISAE 3410\n\n"
            "---"
        )

    def _md_methodology_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology summary per category."""
        methods = data.get("methodology_summary", [])
        if not methods:
            return "## 1. Methodology Summary per Category\n\nNo methodology data."
        lines = [
            "## 1. Methodology Summary per Category",
            "",
            "| Cat # | Category | Method | Tier | Data Type | EF Source | GWP |",
            "|-------|----------|--------|------|-----------|----------|-----|",
        ]
        for m in methods:
            num = m.get("category_number", "?")
            name = m.get("category_name", "")
            method = m.get("method", "-")
            tier = m.get("tier", "-")
            data_type = m.get("data_type", "-")
            ef_source = m.get("ef_source", "-")
            gwp = m.get("gwp_basis", "AR6")
            lines.append(
                f"| {num} | {name} | {method} | {tier} | {data_type} | {ef_source} | {gwp} |"
            )
        return "\n".join(lines)

    def _md_data_sources(self, data: Dict[str, Any]) -> str:
        """Render Markdown data source inventory."""
        sources = data.get("data_sources", [])
        if not sources:
            return "## 2. Data Source Inventory\n\nNo data sources documented."
        lines = [
            "## 2. Data Source Inventory",
            "",
            "| Source Name | Type | Category | Coverage | Frequency | Last Updated | Provenance |",
            "|-----------|------|----------|----------|-----------|-------------|-----------|",
        ]
        for s in sources:
            name = s.get("source_name", "")
            stype = s.get("source_type", "-")
            cat = s.get("category", "-")
            coverage = s.get("coverage", "-")
            freq = s.get("frequency", "-")
            updated = s.get("last_updated", "-")
            phash = _truncate_hash(s.get("provenance_hash"))
            lines.append(
                f"| {name} | {stype} | {cat} | {coverage} | {freq} | {updated} | `{phash}` |"
            )
        return "\n".join(lines)

    def _md_ef_registry(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission factor registry with provenance."""
        registry = data.get("ef_registry", [])
        if not registry:
            return "## 3. Emission Factor Registry\n\nNo emission factors registered."
        lines = [
            "## 3. Emission Factor Registry with Provenance",
            "",
            "| Factor ID | Category | Name | Value | Unit | Source | Version | Geography | SHA-256 |",
            "|-----------|----------|------|-------|------|--------|---------|-----------|---------|",
        ]
        for ef in registry:
            fid = ef.get("factor_id", "")
            cat = ef.get("category", "-")
            name = ef.get("factor_name", "")
            val = ef.get("value")
            val_str = f"{val:.6f}" if val is not None else "-"
            unit = ef.get("unit", "")
            source = ef.get("source", "")
            version = ef.get("version", "-")
            geo = ef.get("geography", "Global")
            phash = _truncate_hash(ef.get("provenance_hash"))
            lines.append(
                f"| {fid} | {cat} | {name} | {val_str} | {unit} | {source} | {version} | {geo} | `{phash}` |"
            )
        return "\n".join(lines)

    def _md_calculation_log(self, data: Dict[str, Any]) -> str:
        """Render Markdown calculation log with SHA-256 hashes."""
        log = data.get("calculation_log", [])
        if not log:
            return "## 4. Calculation Log\n\nNo calculation log entries."
        lines = [
            "## 4. Calculation Log with SHA-256 Hashes",
            "",
            "| Step | Category | Input | Calculation | Result tCO2e | Hash |",
            "|------|----------|-------|-------------|-------------|------|",
        ]
        for entry in log:
            step = entry.get("step", "")
            cat = entry.get("category", "-")
            input_desc = entry.get("input_description", "-")
            calc = entry.get("calculation", "-")
            result = _fmt_tco2e(entry.get("result_tco2e"))
            phash = _truncate_hash(entry.get("hash"))
            lines.append(
                f"| {step} | {cat} | {input_desc} | {calc} | {result} | `{phash}` |"
            )
        return "\n".join(lines)

    def _md_assumptions(self, data: Dict[str, Any]) -> str:
        """Render Markdown assumption register."""
        assumptions = data.get("assumptions", [])
        if not assumptions:
            return "## 5. Assumption Register\n\nNo assumptions registered."
        lines = [
            "## 5. Assumption Register",
            "",
            "| # | Category | Assumption | Basis | Impact | Sensitivity |",
            "|---|----------|-----------|-------|--------|-------------|",
        ]
        for i, a in enumerate(assumptions, 1):
            cat = a.get("category", "General")
            assumption = a.get("assumption", "")
            basis = a.get("basis", "-")
            impact = a.get("impact", "-")
            sensitivity = a.get("sensitivity", "-")
            lines.append(
                f"| {i} | {cat} | {assumption} | {basis} | {impact} | {sensitivity} |"
            )
        return "\n".join(lines)

    def _md_materiality(self, data: Dict[str, Any]) -> str:
        """Render Markdown materiality assessment."""
        mat = data.get("materiality", {})
        if not mat:
            return "## 6. Materiality Assessment\n\nNo materiality assessment."
        threshold = mat.get("threshold_pct", 5.0)
        lines = [
            "## 6. Materiality Assessment",
            "",
            f"**Materiality Threshold:** {threshold:.1f}%",
            "",
        ]
        categories = mat.get("categories", [])
        if categories:
            lines.append("| Category | tCO2e | % of Scope 3 | Material? | Rationale |")
            lines.append("|----------|-------|-------------|-----------|-----------|")
            for c in categories:
                name = c.get("category_name", "")
                em = _fmt_tco2e(c.get("emissions_tco2e"))
                pct = c.get("pct_of_scope3")
                pct_str = f"{pct:.1f}%" if pct is not None else "-"
                material = "Yes" if c.get("is_material", False) else "No"
                rationale = c.get("rationale", "-")
                lines.append(
                    f"| {name} | {em} | {pct_str} | {material} | {rationale} |"
                )
        return "\n".join(lines)

    def _md_completeness(self, data: Dict[str, Any]) -> str:
        """Render Markdown completeness statement."""
        comp = data.get("completeness", {})
        if not comp:
            return "## 7. Completeness Statement\n\nNo completeness statement."
        coverage = comp.get("coverage_pct", 100.0)
        categories_reported = comp.get("categories_reported", 0)
        categories_total = comp.get("categories_total", 15)
        statement = comp.get("statement", "")
        excluded = comp.get("excluded_categories", [])
        lines = [
            "## 7. Completeness Statement",
            "",
            f"**Coverage:** {coverage:.1f}%",
            f"**Categories Reported:** {categories_reported} of {categories_total}",
            "",
        ]
        if excluded:
            lines.append("**Excluded Categories:**")
            for exc in excluded:
                reason = exc.get("reason", "Not applicable")
                lines.append(f"- Cat {exc.get('category_number', '?')}: {reason}")
            lines.append("")
        if statement:
            lines.append(statement)
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality statement."""
        dq = data.get("data_quality_statement", {})
        if not dq:
            return "## 8. Data Quality Statement\n\nNo data quality statement."
        overall = dq.get("overall_dqr_score")
        lines = [
            "## 8. Data Quality Statement",
            "",
        ]
        if overall is not None:
            lines.append(f"**Overall DQR Score:** {overall:.1f} / 5.0")
            lines.append("")
        statement = dq.get("statement", "")
        if statement:
            lines.append(statement)
        return "\n".join(lines)

    def _md_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render Markdown uncertainty statement."""
        unc = data.get("uncertainty_statement", {})
        if not unc:
            return "## 9. Uncertainty Statement\n\nNo uncertainty statement."
        overall = unc.get("overall_uncertainty_pct")
        method = unc.get("method", "-")
        confidence = unc.get("confidence_level", "95%")
        lines = [
            "## 9. Uncertainty Statement",
            "",
            f"**Method:** {method} | **Confidence Level:** {confidence}",
        ]
        if overall is not None:
            lines.append(f"**Overall Uncertainty:** +/-{overall:.1f}%")
        statement = unc.get("statement", "")
        if statement:
            lines.append(f"\n{statement}")
        return "\n".join(lines)

    def _md_boundary(self, data: Dict[str, Any]) -> str:
        """Render Markdown organization boundary reference."""
        boundary = data.get("organization_boundary", {})
        if not boundary:
            return "## 10. Organization Boundary Reference\n\nNo boundary data."
        approach = boundary.get("consolidation_approach", "Operational Control")
        lines = [
            "## 10. Organization Boundary Reference",
            "",
            f"**Consolidation Approach:** {approach}",
            "",
        ]
        description = boundary.get("description", "")
        if description:
            lines.append(description)
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
            f"<title>Scope 3 Verification Package - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#2C3E50;border-bottom:3px solid #2C3E50;padding-bottom:0.5rem;}\n"
            "h2{color:#2C3E50;margin-top:2rem;border-bottom:1px solid #bbb;padding-bottom:0.3rem;}\n"
            "h3{color:#34495E;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #bbb;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#ecf0f1;font-weight:600;color:#2C3E50;}\n"
            "tr:nth-child(even){background:#f7f8f9;}\n"
            "code{background:#ecf0f1;padding:0.1rem 0.3rem;border-radius:3px;"
            "font-size:0.8rem;color:#2C3E50;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".material-yes{color:#27AE60;font-weight:700;}\n"
            ".material-no{color:#95A5A6;}\n"
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
            f"<h1>Scope 3 Verification Package &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            "<strong>Standard:</strong> ISO 14064-3:2019 / ISAE 3410</p>\n"
            "<hr>\n</div>"
        )

    def _html_methodology_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology summary."""
        methods = data.get("methodology_summary", [])
        if not methods:
            return ""
        rows = ""
        for m in methods:
            num = m.get("category_number", "?")
            name = m.get("category_name", "")
            method = m.get("method", "-")
            tier = m.get("tier", "-")
            data_type = m.get("data_type", "-")
            ef_source = m.get("ef_source", "-")
            gwp = m.get("gwp_basis", "AR6")
            rows += (
                f"<tr><td>{num}</td><td>{name}</td><td>{method}</td>"
                f"<td>{tier}</td><td>{data_type}</td><td>{ef_source}</td>"
                f"<td>{gwp}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>1. Methodology Summary per Category</h2>\n"
            "<table><thead><tr><th>#</th><th>Category</th><th>Method</th>"
            "<th>Tier</th><th>Data Type</th><th>EF Source</th>"
            f"<th>GWP</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_data_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML data source inventory."""
        sources = data.get("data_sources", [])
        if not sources:
            return ""
        rows = ""
        for s in sources:
            name = s.get("source_name", "")
            stype = s.get("source_type", "-")
            cat = s.get("category", "-")
            coverage = s.get("coverage", "-")
            freq = s.get("frequency", "-")
            updated = s.get("last_updated", "-")
            phash = _truncate_hash(s.get("provenance_hash"))
            rows += (
                f"<tr><td>{name}</td><td>{stype}</td><td>{cat}</td>"
                f"<td>{coverage}</td><td>{freq}</td><td>{updated}</td>"
                f"<td><code>{phash}</code></td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Data Source Inventory</h2>\n"
            "<table><thead><tr><th>Source</th><th>Type</th><th>Category</th>"
            "<th>Coverage</th><th>Frequency</th><th>Updated</th>"
            f"<th>Hash</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_ef_registry(self, data: Dict[str, Any]) -> str:
        """Render HTML emission factor registry."""
        registry = data.get("ef_registry", [])
        if not registry:
            return ""
        rows = ""
        for ef in registry:
            fid = ef.get("factor_id", "")
            cat = ef.get("category", "-")
            name = ef.get("factor_name", "")
            val = ef.get("value")
            val_str = f"{val:.6f}" if val is not None else "-"
            unit = ef.get("unit", "")
            source = ef.get("source", "")
            version = ef.get("version", "-")
            geo = ef.get("geography", "Global")
            phash = _truncate_hash(ef.get("provenance_hash"))
            rows += (
                f"<tr><td>{fid}</td><td>{cat}</td><td>{name}</td>"
                f"<td>{val_str}</td><td>{unit}</td><td>{source}</td>"
                f"<td>{version}</td><td>{geo}</td>"
                f"<td><code>{phash}</code></td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Emission Factor Registry</h2>\n"
            "<table><thead><tr><th>ID</th><th>Cat</th><th>Name</th>"
            "<th>Value</th><th>Unit</th><th>Source</th><th>Ver</th>"
            f"<th>Geo</th><th>SHA-256</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_calculation_log(self, data: Dict[str, Any]) -> str:
        """Render HTML calculation log."""
        log = data.get("calculation_log", [])
        if not log:
            return ""
        rows = ""
        for entry in log:
            step = entry.get("step", "")
            cat = entry.get("category", "-")
            input_desc = entry.get("input_description", "-")
            calc = entry.get("calculation", "-")
            result = _fmt_tco2e(entry.get("result_tco2e"))
            phash = _truncate_hash(entry.get("hash"))
            rows += (
                f"<tr><td>{step}</td><td>{cat}</td><td>{input_desc}</td>"
                f"<td>{calc}</td><td>{result}</td>"
                f"<td><code>{phash}</code></td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Calculation Log</h2>\n"
            "<table><thead><tr><th>Step</th><th>Cat</th><th>Input</th>"
            "<th>Calculation</th><th>Result</th>"
            f"<th>Hash</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_assumptions(self, data: Dict[str, Any]) -> str:
        """Render HTML assumption register."""
        assumptions = data.get("assumptions", [])
        if not assumptions:
            return ""
        rows = ""
        for i, a in enumerate(assumptions, 1):
            cat = a.get("category", "General")
            assumption = a.get("assumption", "")
            basis = a.get("basis", "-")
            impact = a.get("impact", "-")
            sensitivity = a.get("sensitivity", "-")
            rows += (
                f"<tr><td>{i}</td><td>{cat}</td><td>{assumption}</td>"
                f"<td>{basis}</td><td>{impact}</td><td>{sensitivity}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>5. Assumption Register</h2>\n"
            "<table><thead><tr><th>#</th><th>Category</th><th>Assumption</th>"
            f"<th>Basis</th><th>Impact</th><th>Sensitivity</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_materiality(self, data: Dict[str, Any]) -> str:
        """Render HTML materiality assessment."""
        mat = data.get("materiality", {})
        categories = mat.get("categories", [])
        if not categories:
            return ""
        threshold = mat.get("threshold_pct", 5.0)
        rows = ""
        for c in categories:
            name = c.get("category_name", "")
            em = _fmt_tco2e(c.get("emissions_tco2e"))
            pct = c.get("pct_of_scope3")
            pct_str = f"{pct:.1f}%" if pct is not None else "-"
            material = c.get("is_material", False)
            mat_str = "Yes" if material else "No"
            css = "material-yes" if material else "material-no"
            rationale = c.get("rationale", "-")
            rows += (
                f'<tr><td>{name}</td><td>{em}</td><td>{pct_str}</td>'
                f'<td class="{css}">{mat_str}</td><td>{rationale}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>6. Materiality Assessment</h2>\n"
            f"<p><strong>Materiality Threshold:</strong> {threshold:.1f}%</p>\n"
            "<table><thead><tr><th>Category</th><th>tCO2e</th><th>%</th>"
            f"<th>Material?</th><th>Rationale</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_completeness(self, data: Dict[str, Any]) -> str:
        """Render HTML completeness statement."""
        comp = data.get("completeness", {})
        if not comp:
            return ""
        coverage = comp.get("coverage_pct", 100.0)
        reported = comp.get("categories_reported", 0)
        total = comp.get("categories_total", 15)
        statement = comp.get("statement", "")
        return (
            '<div class="section">\n'
            "<h2>7. Completeness Statement</h2>\n"
            f"<p><strong>Coverage:</strong> {coverage:.1f}% | "
            f"<strong>Categories:</strong> {reported} of {total}</p>\n"
            f"<p>{statement if statement else 'All material categories have been included.'}</p>\n</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality statement."""
        dq = data.get("data_quality_statement", {})
        if not dq:
            return ""
        overall = dq.get("overall_dqr_score")
        statement = dq.get("statement", "")
        overall_str = f"<p><strong>Overall DQR:</strong> {overall:.1f} / 5.0</p>\n" if overall else ""
        return (
            '<div class="section">\n'
            "<h2>8. Data Quality Statement</h2>\n"
            f"{overall_str}"
            f"<p>{statement if statement else 'Data quality assessed per GHG Protocol guidance.'}</p>\n</div>"
        )

    def _html_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty statement."""
        unc = data.get("uncertainty_statement", {})
        if not unc:
            return ""
        overall = unc.get("overall_uncertainty_pct")
        method = unc.get("method", "-")
        confidence = unc.get("confidence_level", "95%")
        statement = unc.get("statement", "")
        overall_str = f"+/-{overall:.1f}%" if overall is not None else "N/A"
        return (
            '<div class="section">\n'
            "<h2>9. Uncertainty Statement</h2>\n"
            f"<p><strong>Method:</strong> {method} | "
            f"<strong>Confidence:</strong> {confidence} | "
            f"<strong>Overall:</strong> {overall_str}</p>\n"
            f"<p>{statement}</p>\n</div>"
        )

    def _html_boundary(self, data: Dict[str, Any]) -> str:
        """Render HTML organization boundary."""
        boundary = data.get("organization_boundary", {})
        if not boundary:
            return ""
        approach = boundary.get("consolidation_approach", "Operational Control")
        description = boundary.get("description", "")
        return (
            '<div class="section">\n'
            "<h2>10. Organization Boundary Reference</h2>\n"
            f"<p><strong>Consolidation Approach:</strong> {approach}</p>\n"
            f"<p>{description}</p>\n</div>"
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
