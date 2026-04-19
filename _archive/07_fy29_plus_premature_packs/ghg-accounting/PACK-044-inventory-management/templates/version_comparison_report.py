# -*- coding: utf-8 -*-
"""
VersionComparisonReport - Version Diff Analysis for PACK-044.

Generates a version comparison report covering two inventory versions,
highlighting differences in emissions, methodology, data sources, and
emission factors.

Sections:
    1. Version Overview
    2. Emission Differences
    3. Methodology Changes
    4. Data Source Changes
    5. Emission Factor Changes

Author: GreenLang Team
Version: 44.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "44.0.0"


class VersionComparisonReport:
    """
    Version comparison report template for GHG inventory management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize VersionComparisonReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render version comparison as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_version_overview(data),
            self._md_emission_diffs(data),
            self._md_methodology_changes(data),
            self._md_data_source_changes(data),
            self._md_ef_changes(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render version comparison as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_version_overview(data),
            self._html_emission_diffs(data),
            self._html_methodology_changes(data),
            self._html_ef_changes(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render version comparison as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        return {
            "template": "version_comparison_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_provenance(data),
            "company_name": self._get_val(data, "company_name", ""),
            "version_a": data.get("version_a", {}),
            "version_b": data.get("version_b", {}),
            "emission_diffs": data.get("emission_diffs", []),
            "methodology_changes": data.get("methodology_changes", []),
            "data_source_changes": data.get("data_source_changes", []),
            "ef_changes": data.get("ef_changes", []),
        }

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        va = data.get("version_a", {}).get("version_id", "A")
        vb = data.get("version_b", {}).get("version_id", "B")
        return (
            f"# Version Comparison Report - {company}\n\n"
            f"**Comparing:** {va} vs {vb} | "
            f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"
        )

    def _md_version_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown version overview."""
        va = data.get("version_a", {})
        vb = data.get("version_b", {})
        return (
            "## 1. Version Overview\n\n"
            "| Attribute | Version A | Version B |\n"
            "|-----------|----------|----------|\n"
            f"| Version ID | {va.get('version_id', '-')} | {vb.get('version_id', '-')} |\n"
            f"| Created | {va.get('created_at', '-')} | {vb.get('created_at', '-')} |\n"
            f"| Created By | {va.get('created_by', '-')} | {vb.get('created_by', '-')} |\n"
            f"| Status | {va.get('status', '-')} | {vb.get('status', '-')} |\n"
            f"| Total tCO2e | {va.get('total_tco2e', 0):,.1f} | {vb.get('total_tco2e', 0):,.1f} |"
        )

    def _md_emission_diffs(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission differences."""
        diffs = data.get("emission_diffs", [])
        if not diffs:
            return "## 2. Emission Differences\n\nNo emission differences found."
        lines = [
            "## 2. Emission Differences",
            "",
            "| Scope | Category | Version A (tCO2e) | Version B (tCO2e) | Delta | % Change |",
            "|-------|---------|------------------|------------------|-------|----------|",
        ]
        for d in diffs:
            a_val = d.get("version_a_tco2e", 0.0)
            b_val = d.get("version_b_tco2e", 0.0)
            delta = b_val - a_val
            pct = (delta / a_val * 100) if a_val > 0 else 0.0
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {d.get('scope', '')} | {d.get('category', '')} | "
                f"{a_val:,.1f} | {b_val:,.1f} | {sign}{delta:,.1f} | {sign}{pct:.1f}% |"
            )
        return "\n".join(lines)

    def _md_methodology_changes(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology changes."""
        changes = data.get("methodology_changes", [])
        if not changes:
            return ""
        lines = [
            "## 3. Methodology Changes",
            "",
            "| Category | Attribute | Version A | Version B | Impact |",
            "|---------|----------|----------|----------|--------|",
        ]
        for c in changes:
            lines.append(
                f"| {c.get('category', '')} | {c.get('attribute', '')} | "
                f"{c.get('version_a_value', '')} | {c.get('version_b_value', '')} | {c.get('impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_data_source_changes(self, data: Dict[str, Any]) -> str:
        """Render Markdown data source changes."""
        changes = data.get("data_source_changes", [])
        if not changes:
            return ""
        lines = [
            "## 4. Data Source Changes",
            "",
            "| Source | Change Type | Details |",
            "|-------|-----------|---------|",
        ]
        for c in changes:
            lines.append(f"| {c.get('source', '')} | {c.get('change_type', '')} | {c.get('details', '')} |")
        return "\n".join(lines)

    def _md_ef_changes(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission factor changes."""
        changes = data.get("ef_changes", [])
        if not changes:
            return ""
        lines = [
            "## 5. Emission Factor Changes",
            "",
            "| Category | Factor | Version A | Version B | Unit | Source |",
            "|---------|--------|----------|----------|------|--------|",
        ]
        for c in changes:
            lines.append(
                f"| {c.get('category', '')} | {c.get('factor_name', '')} | "
                f"{c.get('version_a_value', '')} | {c.get('version_b_value', '')} | "
                f"{c.get('unit', '')} | {c.get('source', '')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f'<meta charset="UTF-8"><title>Version Comparison - {company}</title>\n'
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}"
            "h1{color:#0d1b2a;border-bottom:3px solid #457b9d;}h2{color:#1b263b;margin-top:2rem;}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}"
            "th{background:#f0f4f8;font-weight:600;}tr:nth-child(even){background:#fafbfc;}"
            ".increase{color:#e63946;}.decrease{color:#2a9d8f;}"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        return f'<div><h1>Version Comparison &mdash; {company}</h1><hr></div>'

    def _html_version_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML version overview."""
        va = data.get("version_a", {})
        vb = data.get("version_b", {})
        return (
            '<div><h2>1. Version Overview</h2>\n'
            "<table><thead><tr><th>Attribute</th><th>Version A</th><th>Version B</th></tr></thead>\n"
            f"<tbody><tr><td>Version ID</td><td>{va.get('version_id', '-')}</td><td>{vb.get('version_id', '-')}</td></tr>\n"
            f"<tr><td>Total tCO2e</td><td>{va.get('total_tco2e', 0):,.1f}</td><td>{vb.get('total_tco2e', 0):,.1f}</td></tr>"
            "</tbody></table></div>"
        )

    def _html_emission_diffs(self, data: Dict[str, Any]) -> str:
        """Render HTML emission diffs."""
        diffs = data.get("emission_diffs", [])
        if not diffs:
            return ""
        rows = ""
        for d in diffs:
            a_val = d.get("version_a_tco2e", 0.0)
            b_val = d.get("version_b_tco2e", 0.0)
            delta = b_val - a_val
            css = "increase" if delta > 0 else "decrease"
            rows += (
                f"<tr><td>{d.get('scope', '')}</td><td>{d.get('category', '')}</td>"
                f"<td>{a_val:,.1f}</td><td>{b_val:,.1f}</td>"
                f'<td class="{css}">{delta:+,.1f}</td></tr>\n'
            )
        return (
            '<div><h2>2. Emission Differences</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Category</th><th>Ver A</th><th>Ver B</th><th>Delta</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_methodology_changes(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology changes."""
        changes = data.get("methodology_changes", [])
        if not changes:
            return ""
        rows = ""
        for c in changes:
            rows += f"<tr><td>{c.get('category', '')}</td><td>{c.get('attribute', '')}</td><td>{c.get('version_a_value', '')}</td><td>{c.get('version_b_value', '')}</td></tr>\n"
        return (
            '<div><h2>3. Methodology Changes</h2>\n'
            "<table><thead><tr><th>Category</th><th>Attribute</th><th>Ver A</th><th>Ver B</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_ef_changes(self, data: Dict[str, Any]) -> str:
        """Render HTML EF changes."""
        changes = data.get("ef_changes", [])
        if not changes:
            return ""
        rows = ""
        for c in changes:
            rows += f"<tr><td>{c.get('category', '')}</td><td>{c.get('factor_name', '')}</td><td>{c.get('version_a_value', '')}</td><td>{c.get('version_b_value', '')}</td></tr>\n"
        return (
            '<div><h2>5. EF Changes</h2>\n'
            "<table><thead><tr><th>Category</th><th>Factor</th><th>Ver A</th><th>Ver B</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div style="font-size:0.85rem;color:#666;"><hr>'
            f"<p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p>"
            f'<p class="provenance">Provenance Hash: {provenance}</p></div>'
        )
