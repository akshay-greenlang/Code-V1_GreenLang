# -*- coding: utf-8 -*-
"""
AssurancePackageTemplate - ISAE 3410 Evidence Bundle for PACK-043.

Generates an assurance-ready evidence bundle aligned with ISAE 3410 for
third-party verifiers. Includes methodology summary per category, data
source inventory with provenance, calculation log with SHA-256 hash chain,
assumption register, emission factor registry, completeness statement,
uncertainty statement, assurance readiness score gauge, and verifier
query log.

Sections:
    1. Assurance Readiness Summary
    2. Methodology Summary per Category
    3. Data Source Inventory
    4. Calculation Log with SHA-256 Hash Chain
    5. Assumption Register
    6. Emission Factor Registry
    7. Completeness Statement
    8. Uncertainty Statement
    9. Verifier Query Log

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, audit charcoal #2C3E50 theme)
    - JSON (structured with full audit chain)

Regulatory References:
    - ISAE 3410 (Assurance on GHG Statements)
    - ISO 14064-3:2019 (Verification & Validation)
    - GHG Protocol Scope 3 Standard

Author: GreenLang Team
Version: 43.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "43.0.0"


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


def _readiness_label(score: Optional[float]) -> str:
    """Map readiness score to label."""
    if score is None:
        return "Not Assessed"
    if score >= 90:
        return "Ready for Assurance"
    if score >= 75:
        return "Nearly Ready"
    if score >= 50:
        return "Significant Gaps"
    return "Not Ready"


class AssurancePackageTemplate:
    """
    ISAE 3410 assurance evidence bundle template.

    Renders assurance-ready evidence packages with methodology summaries,
    data source inventories, SHA-256 calculation logs, assumption
    registers, emission factor registries, and completeness/uncertainty
    statements. All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = AssurancePackageTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AssurancePackageTemplate."""
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
        """Render assurance package as Markdown.

        Args:
            data: Validated assurance package data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_readiness_summary(data),
            self._md_methodology_summary(data),
            self._md_data_sources(data),
            self._md_calculation_log(data),
            self._md_assumptions(data),
            self._md_ef_registry(data),
            self._md_completeness(data),
            self._md_uncertainty(data),
            self._md_verifier_queries(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render assurance package as HTML.

        Args:
            data: Validated assurance package data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_readiness_summary(data),
            self._html_methodology_summary(data),
            self._html_data_sources(data),
            self._html_calculation_log(data),
            self._html_assumptions(data),
            self._html_ef_registry(data),
            self._html_completeness(data),
            self._html_uncertainty(data),
            self._html_verifier_queries(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render assurance package as JSON-serializable dict.

        Args:
            data: Validated assurance package data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "assurance_package",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "assurance_readiness": data.get("assurance_readiness", {}),
            "methodology_summary": data.get("methodology_summary", []),
            "data_sources": data.get("data_sources", []),
            "calculation_log": data.get("calculation_log", []),
            "assumptions": data.get("assumptions", []),
            "ef_registry": data.get("ef_registry", []),
            "completeness_statement": data.get("completeness_statement", {}),
            "uncertainty_statement": data.get("uncertainty_statement", {}),
            "verifier_queries": data.get("verifier_queries", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Assurance Evidence Package - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Standard:** ISAE 3410 / ISO 14064-3\n\n"
            "---"
        )

    def _md_readiness_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown assurance readiness summary."""
        readiness = data.get("assurance_readiness", {})
        if not readiness:
            return "## 1. Assurance Readiness\n\nNo readiness data available."
        score = readiness.get("overall_score")
        level = readiness.get("assurance_level", "-")
        lines = [
            "## 1. Assurance Readiness Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if score is not None:
            lines.append(
                f"| Readiness Score | {score:.0f}/100 ({_readiness_label(score)}) |"
            )
        lines.append(f"| Target Assurance Level | {level} |")
        dimensions = readiness.get("dimensions", [])
        if dimensions:
            lines.append("")
            lines.append("| Dimension | Score | Status |")
            lines.append("|-----------|-------|--------|")
            for dim in dimensions:
                name = dim.get("name", "-")
                dscore = dim.get("score")
                d_str = f"{dscore:.0f}/100" if dscore is not None else "-"
                status = dim.get("status", "-")
                lines.append(f"| {name} | {d_str} | {status} |")
        return "\n".join(lines)

    def _md_methodology_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology summary per category."""
        methods = data.get("methodology_summary", [])
        if not methods:
            return "## 2. Methodology Summary\n\nNo methodology data available."
        lines = [
            "## 2. Methodology Summary per Category",
            "",
            "| Category | Methodology | Data Tier | EF Source | Boundary |",
            "|----------|-----------|----------|----------|----------|",
        ]
        for m in methods:
            cat = f"Cat {m.get('category_number', '?')} - {m.get('category_name', 'Unknown')}"
            method = m.get("methodology", "-")
            tier = m.get("data_tier", "-")
            ef_source = m.get("ef_source", "-")
            boundary = m.get("boundary", "-")
            lines.append(f"| {cat} | {method} | {tier} | {ef_source} | {boundary} |")
        return "\n".join(lines)

    def _md_data_sources(self, data: Dict[str, Any]) -> str:
        """Render Markdown data source inventory."""
        sources = data.get("data_sources", [])
        if not sources:
            return "## 3. Data Source Inventory\n\nNo data source inventory available."
        lines = [
            "## 3. Data Source Inventory",
            "",
            "| Source | Type | Categories | Update Frequency | Provenance |",
            "|--------|------|-----------|-----------------|-----------|",
        ]
        for src in sources:
            name = src.get("source_name", "-")
            stype = src.get("source_type", "-")
            cats = src.get("categories_covered", "-")
            freq = src.get("update_frequency", "-")
            prov = _truncate_hash(src.get("provenance_hash"))
            lines.append(f"| {name} | {stype} | {cats} | {freq} | `{prov}` |")
        return "\n".join(lines)

    def _md_calculation_log(self, data: Dict[str, Any]) -> str:
        """Render Markdown calculation log with SHA-256 hashes."""
        log = data.get("calculation_log", [])
        if not log:
            return "## 4. Calculation Log\n\nNo calculation log available."
        lines = [
            "## 4. Calculation Log (SHA-256 Hash Chain)",
            "",
            "| Step | Category | Input Hash | Output Hash | Result (tCO2e) |",
            "|------|----------|-----------|------------|---------------|",
        ]
        for entry in log:
            step = entry.get("step", "-")
            cat = entry.get("category", "-")
            in_hash = _truncate_hash(entry.get("input_hash"))
            out_hash = _truncate_hash(entry.get("output_hash"))
            result = _fmt_tco2e(entry.get("result_tco2e"))
            lines.append(
                f"| {step} | {cat} | `{in_hash}` | `{out_hash}` | {result} |"
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
            "| ID | Assumption | Category | Impact | Justification |",
            "|----|-----------|----------|--------|---------------|",
        ]
        for a in assumptions:
            aid = a.get("assumption_id", "-")
            desc = a.get("description", "-")
            cat = a.get("category", "-")
            impact = a.get("impact_level", "-")
            just = a.get("justification", "-")
            lines.append(f"| {aid} | {desc} | {cat} | {impact} | {just} |")
        return "\n".join(lines)

    def _md_ef_registry(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission factor registry."""
        factors = data.get("ef_registry", [])
        if not factors:
            return "## 6. Emission Factor Registry\n\nNo EF registry available."
        lines = [
            "## 6. Emission Factor Registry",
            "",
            "| Factor ID | Name | Value | Unit | Source | Vintage | Hash |",
            "|-----------|------|-------|------|--------|---------|------|",
        ]
        for ef in factors:
            fid = ef.get("factor_id", "-")
            name = ef.get("name", "-")
            value = ef.get("value")
            val_str = f"{value:.6f}" if value is not None else "-"
            unit = ef.get("unit", "-")
            source = ef.get("source", "-")
            vintage = ef.get("vintage", "-")
            h = _truncate_hash(ef.get("hash"), 12)
            lines.append(
                f"| {fid} | {name} | {val_str} | {unit} | {source} | {vintage} | `{h}` |"
            )
        return "\n".join(lines)

    def _md_completeness(self, data: Dict[str, Any]) -> str:
        """Render Markdown completeness statement."""
        statement = data.get("completeness_statement", {})
        if not statement:
            return "## 7. Completeness Statement\n\nNo completeness statement available."
        cats_reported = statement.get("categories_reported", 0)
        cats_excluded = statement.get("categories_excluded", 0)
        coverage = statement.get("coverage_pct")
        lines = [
            "## 7. Completeness Statement",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Categories Reported | {cats_reported} |",
            f"| Categories Excluded | {cats_excluded} |",
        ]
        if coverage is not None:
            lines.append(f"| Emission Coverage | {coverage:.1f}% |")
        exclusions = statement.get("exclusion_rationale", [])
        if exclusions:
            lines.append("")
            lines.append("**Exclusion Rationale:**")
            for exc in exclusions:
                cat = exc.get("category", "-")
                reason = exc.get("reason", "-")
                lines.append(f"- **{cat}:** {reason}")
        narrative = statement.get("narrative")
        if narrative:
            lines.append(f"\n{narrative}")
        return "\n".join(lines)

    def _md_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render Markdown uncertainty statement."""
        statement = data.get("uncertainty_statement", {})
        if not statement:
            return "## 8. Uncertainty Statement\n\nNo uncertainty statement available."
        overall = statement.get("overall_uncertainty_pct")
        ci_low = statement.get("confidence_interval_low_tco2e")
        ci_high = statement.get("confidence_interval_high_tco2e")
        lines = [
            "## 8. Uncertainty Statement",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if overall is not None:
            lines.append(f"| Overall Uncertainty | +/- {overall:.0f}% |")
        if ci_low is not None and ci_high is not None:
            lines.append(
                f"| 95% Confidence Interval | {_fmt_tco2e(ci_low)} to {_fmt_tco2e(ci_high)} |"
            )
        methodology = statement.get("methodology")
        if methodology:
            lines.append(f"| Assessment Method | {methodology} |")
        narrative = statement.get("narrative")
        if narrative:
            lines.append(f"\n{narrative}")
        return "\n".join(lines)

    def _md_verifier_queries(self, data: Dict[str, Any]) -> str:
        """Render Markdown verifier query log."""
        queries = data.get("verifier_queries", [])
        if not queries:
            return "## 9. Verifier Query Log\n\nNo verifier queries logged."
        lines = [
            "## 9. Verifier Query Log",
            "",
            "| Query ID | Date | Query | Response | Status |",
            "|----------|------|-------|----------|--------|",
        ]
        for q in queries:
            qid = q.get("query_id", "-")
            date = q.get("date", "-")
            query = q.get("query", "-")
            response = q.get("response", "-")
            status = q.get("status", "-")
            lines.append(f"| {qid} | {date} | {query} | {response} | {status} |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Assurance Package - {company} ({year})</title>\n"
            "<style>\n"
            ":root{--primary:#2C3E50;--primary-light:#34495E;--accent:#5D6D7E;"
            "--bg:#EAECEE;--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
            "--border:#BDC3C7;--success:#10B981;--warning:#F59E0B;--danger:#EF4444;}\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:2rem;"
            "background:var(--bg);color:var(--text);line-height:1.6;}\n"
            ".container{max-width:1100px;margin:0 auto;}\n"
            "h1{color:var(--primary);border-bottom:3px solid var(--primary);"
            "padding-bottom:0.5rem;}\n"
            "h2{color:var(--primary);margin-top:2rem;"
            "border-bottom:1px solid var(--border);padding-bottom:0.3rem;}\n"
            "h3{color:var(--primary-light);}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid var(--border);padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:var(--primary);color:#fff;font-weight:600;}\n"
            "tr:nth-child(even){background:#EAECEE;}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".hash{font-family:'Courier New',monospace;font-size:0.8rem;color:#5D6D7E;}\n"
            ".gauge-bar{height:24px;border-radius:12px;background:#D5D8DC;overflow:hidden;}\n"
            ".gauge-fill{height:100%;border-radius:12px;}\n"
            ".ready{color:var(--success);font-weight:700;}\n"
            ".gap{color:var(--danger);font-weight:700;}\n"
            ".provenance{font-size:0.8rem;color:var(--text-muted);font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n<div class=\"container\">\n"
            f"{body}\n"
            "</div>\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Assurance Evidence Package &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Standard:</strong> ISAE 3410 / ISO 14064-3 | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_readiness_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML readiness gauge and summary."""
        readiness = data.get("assurance_readiness", {})
        if not readiness:
            return ""
        score = readiness.get("overall_score", 0)
        label = _readiness_label(score)
        color = "#10B981" if score >= 90 else "#F59E0B" if score >= 75 else "#EF4444"
        gauge_html = (
            f'<div class="gauge-bar">'
            f'<div class="gauge-fill" style="width:{score:.0f}%;background:{color};"></div>'
            f"</div>"
            f"<p><strong>{label}</strong> &mdash; {score:.0f}/100</p>"
        )
        dim_rows = ""
        dimensions = readiness.get("dimensions", [])
        for dim in dimensions:
            name = dim.get("name", "-")
            dscore = dim.get("score")
            d_str = f"{dscore:.0f}/100" if dscore is not None else "-"
            status = dim.get("status", "-")
            css = "ready" if status.lower() == "pass" else "gap"
            dim_rows += f'<tr><td>{name}</td><td>{d_str}</td><td class="{css}">{status}</td></tr>\n'
        dim_table = ""
        if dim_rows:
            dim_table = (
                "<table><thead><tr><th>Dimension</th><th>Score</th><th>Status</th></tr></thead>"
                f"<tbody>{dim_rows}</tbody></table>"
            )
        return (
            '<div class="section">\n'
            "<h2>1. Assurance Readiness Summary</h2>\n"
            f"{gauge_html}\n{dim_table}\n</div>"
        )

    def _html_methodology_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology summary."""
        methods = data.get("methodology_summary", [])
        if not methods:
            return ""
        rows = ""
        for m in methods:
            cat = f"Cat {m.get('category_number', '?')} - {m.get('category_name', 'Unknown')}"
            method = m.get("methodology", "-")
            tier = m.get("data_tier", "-")
            ef_source = m.get("ef_source", "-")
            boundary = m.get("boundary", "-")
            rows += (
                f"<tr><td>{cat}</td><td>{method}</td><td>{tier}</td>"
                f"<td>{ef_source}</td><td>{boundary}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Methodology Summary per Category</h2>\n"
            "<table><thead><tr><th>Category</th><th>Methodology</th>"
            "<th>Data Tier</th><th>EF Source</th><th>Boundary</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_data_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML data source inventory."""
        sources = data.get("data_sources", [])
        if not sources:
            return ""
        rows = ""
        for src in sources:
            name = src.get("source_name", "-")
            stype = src.get("source_type", "-")
            cats = src.get("categories_covered", "-")
            freq = src.get("update_frequency", "-")
            prov = _truncate_hash(src.get("provenance_hash"))
            rows += (
                f"<tr><td>{name}</td><td>{stype}</td><td>{cats}</td>"
                f'<td>{freq}</td><td class="hash">{prov}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>3. Data Source Inventory</h2>\n"
            "<table><thead><tr><th>Source</th><th>Type</th>"
            "<th>Categories</th><th>Frequency</th><th>Provenance</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_calculation_log(self, data: Dict[str, Any]) -> str:
        """Render HTML calculation log with hashes."""
        log = data.get("calculation_log", [])
        if not log:
            return ""
        rows = ""
        for entry in log:
            step = entry.get("step", "-")
            cat = entry.get("category", "-")
            in_hash = _truncate_hash(entry.get("input_hash"))
            out_hash = _truncate_hash(entry.get("output_hash"))
            result = _fmt_tco2e(entry.get("result_tco2e"))
            rows += (
                f"<tr><td>{step}</td><td>{cat}</td>"
                f'<td class="hash">{in_hash}</td>'
                f'<td class="hash">{out_hash}</td>'
                f"<td>{result}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Calculation Log (SHA-256)</h2>\n"
            "<table><thead><tr><th>Step</th><th>Category</th>"
            "<th>Input Hash</th><th>Output Hash</th><th>Result</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_assumptions(self, data: Dict[str, Any]) -> str:
        """Render HTML assumption register."""
        assumptions = data.get("assumptions", [])
        if not assumptions:
            return ""
        rows = ""
        for a in assumptions:
            aid = a.get("assumption_id", "-")
            desc = a.get("description", "-")
            cat = a.get("category", "-")
            impact = a.get("impact_level", "-")
            just = a.get("justification", "-")
            rows += (
                f"<tr><td>{aid}</td><td>{desc}</td><td>{cat}</td>"
                f"<td>{impact}</td><td>{just}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>5. Assumption Register</h2>\n"
            "<table><thead><tr><th>ID</th><th>Assumption</th>"
            "<th>Category</th><th>Impact</th><th>Justification</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_ef_registry(self, data: Dict[str, Any]) -> str:
        """Render HTML emission factor registry."""
        factors = data.get("ef_registry", [])
        if not factors:
            return ""
        rows = ""
        for ef in factors:
            fid = ef.get("factor_id", "-")
            name = ef.get("name", "-")
            value = ef.get("value")
            val_str = f"{value:.6f}" if value is not None else "-"
            unit = ef.get("unit", "-")
            source = ef.get("source", "-")
            vintage = ef.get("vintage", "-")
            h = _truncate_hash(ef.get("hash"), 12)
            rows += (
                f"<tr><td>{fid}</td><td>{name}</td><td>{val_str}</td>"
                f'<td>{unit}</td><td>{source}</td><td>{vintage}</td>'
                f'<td class="hash">{h}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>6. Emission Factor Registry</h2>\n"
            "<table><thead><tr><th>ID</th><th>Name</th><th>Value</th>"
            "<th>Unit</th><th>Source</th><th>Vintage</th><th>Hash</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_completeness(self, data: Dict[str, Any]) -> str:
        """Render HTML completeness statement."""
        statement = data.get("completeness_statement", {})
        if not statement:
            return ""
        cats_rep = statement.get("categories_reported", 0)
        cats_exc = statement.get("categories_excluded", 0)
        coverage = statement.get("coverage_pct")
        rows = (
            f"<tr><td>Categories Reported</td><td>{cats_rep}</td></tr>\n"
            f"<tr><td>Categories Excluded</td><td>{cats_exc}</td></tr>\n"
        )
        if coverage is not None:
            rows += f"<tr><td>Emission Coverage</td><td>{coverage:.1f}%</td></tr>\n"
        exclusions_html = ""
        exclusions = statement.get("exclusion_rationale", [])
        if exclusions:
            exc_items = ""
            for exc in exclusions:
                exc_items += f"<li><strong>{exc.get('category', '-')}:</strong> {exc.get('reason', '-')}</li>\n"
            exclusions_html = f"<h3>Exclusion Rationale</h3><ul>{exc_items}</ul>"
        narrative = statement.get("narrative", "")
        narr_html = f"<p>{narrative}</p>" if narrative else ""
        return (
            '<div class="section">\n'
            "<h2>7. Completeness Statement</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n"
            f"{exclusions_html}\n{narr_html}\n</div>"
        )

    def _html_uncertainty(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty statement."""
        statement = data.get("uncertainty_statement", {})
        if not statement:
            return ""
        overall = statement.get("overall_uncertainty_pct")
        ci_low = statement.get("confidence_interval_low_tco2e")
        ci_high = statement.get("confidence_interval_high_tco2e")
        rows = ""
        if overall is not None:
            rows += f"<tr><td>Overall Uncertainty</td><td>+/- {overall:.0f}%</td></tr>\n"
        if ci_low is not None and ci_high is not None:
            rows += (
                f"<tr><td>95% CI</td>"
                f"<td>{_fmt_tco2e(ci_low)} to {_fmt_tco2e(ci_high)}</td></tr>\n"
            )
        methodology = statement.get("methodology")
        if methodology:
            rows += f"<tr><td>Assessment Method</td><td>{methodology}</td></tr>\n"
        narrative = statement.get("narrative", "")
        narr_html = f"<p>{narrative}</p>" if narrative else ""
        return (
            '<div class="section">\n'
            "<h2>8. Uncertainty Statement</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n{narr_html}\n</div>"
        )

    def _html_verifier_queries(self, data: Dict[str, Any]) -> str:
        """Render HTML verifier query log."""
        queries = data.get("verifier_queries", [])
        if not queries:
            return ""
        rows = ""
        for q in queries:
            qid = q.get("query_id", "-")
            date = q.get("date", "-")
            query = q.get("query", "-")
            response = q.get("response", "-")
            status = q.get("status", "-")
            css = "ready" if status.lower() == "resolved" else "gap"
            rows += (
                f"<tr><td>{qid}</td><td>{date}</td><td>{query}</td>"
                f'<td>{response}</td><td class="{css}">{status}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>9. Verifier Query Log</h2>\n"
            "<table><thead><tr><th>ID</th><th>Date</th>"
            "<th>Query</th><th>Response</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
