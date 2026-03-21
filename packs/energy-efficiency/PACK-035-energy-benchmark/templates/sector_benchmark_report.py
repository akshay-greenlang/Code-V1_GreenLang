# -*- coding: utf-8 -*-
"""
SectorBenchmarkReportTemplate - Multi-source sector benchmark report for PACK-035.

Generates sector benchmark comparison reports that evaluate a facility
against multiple benchmark databases including ENERGY STAR, CIBSE TM46,
DIN V 18599, and BPIE. Provides cross-source analysis to identify the
most relevant benchmark and highlight performance gaps.

Sections:
    1. Header & Building Type Classification
    2. Benchmark Sources Used
    3. Benchmark Comparison Table
    4. Facility vs Benchmarks
    5. Cross-Source Analysis
    6. Methodology Notes
    7. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Supported benchmark source definitions
BENCHMARK_SOURCES: List[Dict[str, str]] = [
    {"id": "energy_star", "name": "ENERGY STAR", "region": "US/Global", "unit": "kBtu/ft2/yr"},
    {"id": "cibse_tm46", "name": "CIBSE TM46", "region": "UK", "unit": "kWh/m2/yr"},
    {"id": "din_v18599", "name": "DIN V 18599", "region": "Germany/EU", "unit": "kWh/m2/yr"},
    {"id": "bpie", "name": "BPIE", "region": "EU", "unit": "kWh/m2/yr"},
]


class SectorBenchmarkReportTemplate:
    """
    Multi-source sector benchmark report template.

    Renders sector benchmark comparison reports with values from ENERGY
    STAR, CIBSE, DIN, and BPIE databases, cross-source analysis, and
    performance gap identification across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SectorBenchmarkReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render sector benchmark report as Markdown.

        Args:
            data: Sector benchmark data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_building_classification(data),
            self._md_benchmark_sources(data),
            self._md_benchmark_comparison(data),
            self._md_facility_vs_benchmarks(data),
            self._md_cross_source_analysis(data),
            self._md_methodology_notes(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render sector benchmark report as self-contained HTML.

        Args:
            data: Sector benchmark data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_building_classification(data),
            self._html_benchmark_sources(data),
            self._html_benchmark_comparison(data),
            self._html_facility_vs_benchmarks(data),
            self._html_cross_source_analysis(data),
            self._html_methodology_notes(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Sector Benchmark Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render sector benchmark report as structured JSON.

        Args:
            data: Sector benchmark data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "sector_benchmark_report",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "facility": data.get("facility", {}),
            "building_classification": data.get("building_classification", {}),
            "benchmark_sources": data.get("benchmark_sources", []),
            "benchmark_values": data.get("benchmark_values", []),
            "facility_vs_benchmarks": data.get("facility_vs_benchmarks", []),
            "cross_source_analysis": data.get("cross_source_analysis", {}),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Sector Benchmark Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Building Type:** {data.get('building_type', '-')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 SectorBenchmarkReportTemplate v35.0.0\n\n---"
        )

    def _md_building_classification(self, data: Dict[str, Any]) -> str:
        """Render building type classification section."""
        bc = data.get("building_classification", {})
        mappings = bc.get("source_mappings", [])
        lines = [
            "## 1. Building Type Classification\n",
            f"**Primary Type:** {bc.get('primary_type', '-')}  ",
            f"**Sub-Type:** {bc.get('sub_type', '-')}  ",
            f"**Confidence:** {self._fmt(bc.get('confidence_pct', 0))}%  ",
            f"**Classification Method:** {bc.get('method', 'User-defined')}",
        ]
        if mappings:
            lines.extend([
                "\n### Source-Specific Mapping\n",
                "| Source | Mapped Category | Match Confidence |",
                "|--------|----------------|-----------------|",
            ])
            for m in mappings:
                lines.append(
                    f"| {m.get('source', '-')} "
                    f"| {m.get('category', '-')} "
                    f"| {self._fmt(m.get('confidence_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_benchmark_sources(self, data: Dict[str, Any]) -> str:
        """Render benchmark sources section."""
        sources = data.get("benchmark_sources", BENCHMARK_SOURCES)
        lines = [
            "## 2. Benchmark Sources Used\n",
            "| Source | Region | Unit | Version | Available |",
            "|--------|--------|------|---------|-----------|",
        ]
        for s in sources:
            lines.append(
                f"| {s.get('name', '-')} "
                f"| {s.get('region', '-')} "
                f"| {s.get('unit', '-')} "
                f"| {s.get('version', '-')} "
                f"| {'Yes' if s.get('available', True) else 'No'} |"
            )
        return "\n".join(lines)

    def _md_benchmark_comparison(self, data: Dict[str, Any]) -> str:
        """Render benchmark comparison table section."""
        values = data.get("benchmark_values", [])
        if not values:
            return "## 3. Benchmark Comparison\n\n_No benchmark data available._"
        lines = [
            "## 3. Benchmark Comparison\n",
            "| Source | Typical (kWh/m2) | Good Practice | Best Practice | Notes |",
            "|--------|-----------------|---------------|---------------|-------|",
        ]
        for v in values:
            lines.append(
                f"| {v.get('source', '-')} "
                f"| {self._fmt(v.get('typical', 0))} "
                f"| {self._fmt(v.get('good_practice', 0))} "
                f"| {self._fmt(v.get('best_practice', 0))} "
                f"| {v.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_facility_vs_benchmarks(self, data: Dict[str, Any]) -> str:
        """Render facility vs benchmarks section."""
        comparisons = data.get("facility_vs_benchmarks", [])
        if not comparisons:
            return "## 4. Facility vs Benchmarks\n\n_No comparison data._"
        facility_eui = data.get("facility_eui", 0)
        lines = [
            "## 4. Facility vs Benchmarks\n",
            f"**Facility EUI:** {self._fmt(facility_eui)} kWh/m2/yr\n",
            "| Source | Benchmark | Facility | Gap (kWh/m2) | Gap (%) | Status |",
            "|--------|----------|---------|-------------|---------|--------|",
        ]
        for c in comparisons:
            lines.append(
                f"| {c.get('source', '-')} ({c.get('level', 'typical')}) "
                f"| {self._fmt(c.get('benchmark_eui', 0))} "
                f"| {self._fmt(c.get('facility_eui', 0))} "
                f"| {self._fmt(c.get('gap_kwh_m2', 0))} "
                f"| {self._fmt(c.get('gap_pct', 0))}% "
                f"| {c.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_cross_source_analysis(self, data: Dict[str, Any]) -> str:
        """Render cross-source analysis section."""
        csa = data.get("cross_source_analysis", {})
        findings = csa.get("findings", [])
        lines = [
            "## 5. Cross-Source Analysis\n",
            f"**Benchmark Spread:** {self._fmt(csa.get('min_benchmark', 0))} - "
            f"{self._fmt(csa.get('max_benchmark', 0))} kWh/m2/yr  ",
            f"**Coefficient of Variation:** {self._fmt(csa.get('cv_pct', 0))}%  ",
            f"**Recommended Benchmark:** {csa.get('recommended_source', '-')}  ",
            f"**Recommended Value:** {self._fmt(csa.get('recommended_value', 0))} kWh/m2/yr  ",
            f"**Rationale:** {csa.get('rationale', '-')}",
        ]
        if findings:
            lines.append("\n### Key Findings\n")
            for f in findings:
                lines.append(f"- {f}")
        return "\n".join(lines)

    def _md_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render methodology notes section."""
        notes = data.get("methodology_notes", [
            "ENERGY STAR values converted from kBtu/ft2/yr to kWh/m2/yr.",
            "CIBSE TM46 benchmarks use UK climate assumptions.",
            "DIN V 18599 uses German reference climate.",
            "BPIE values represent EU-wide averages.",
            "All values normalised to site energy unless noted.",
        ])
        lines = ["## 6. Methodology Notes\n"]
        for n in notes:
            lines.append(f"- {n}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-035 Energy Benchmark Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Sector Benchmark Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Type: {data.get("building_type", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_building_classification(self, data: Dict[str, Any]) -> str:
        """Render HTML building classification."""
        bc = data.get("building_classification", {})
        mappings = bc.get("source_mappings", [])
        rows = "".join(
            f'<tr><td>{m.get("source", "-")}</td>'
            f'<td>{m.get("category", "-")}</td>'
            f'<td>{self._fmt(m.get("confidence_pct", 0))}%</td></tr>\n'
            for m in mappings
        )
        return (
            '<h2>Building Type Classification</h2>\n'
            f'<div class="info-box">'
            f'<p><strong>Primary:</strong> {bc.get("primary_type", "-")} | '
            f'<strong>Sub-Type:</strong> {bc.get("sub_type", "-")} | '
            f'<strong>Confidence:</strong> {self._fmt(bc.get("confidence_pct", 0))}%</p>'
            '</div>\n'
            '<table>\n<tr><th>Source</th><th>Mapped Category</th>'
            f'<th>Confidence</th></tr>\n{rows}</table>'
        )

    def _html_benchmark_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML benchmark sources table."""
        sources = data.get("benchmark_sources", BENCHMARK_SOURCES)
        rows = "".join(
            f'<tr><td>{s.get("name", "-")}</td><td>{s.get("region", "-")}</td>'
            f'<td>{s.get("unit", "-")}</td>'
            f'<td>{"Yes" if s.get("available", True) else "No"}</td></tr>\n'
            for s in sources
        )
        return (
            '<h2>Benchmark Sources</h2>\n'
            '<table>\n<tr><th>Source</th><th>Region</th><th>Unit</th>'
            f'<th>Available</th></tr>\n{rows}</table>'
        )

    def _html_benchmark_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML benchmark comparison table."""
        values = data.get("benchmark_values", [])
        rows = ""
        for v in values:
            rows += (
                f'<tr><td>{v.get("source", "-")}</td>'
                f'<td>{self._fmt(v.get("typical", 0))}</td>'
                f'<td>{self._fmt(v.get("good_practice", 0))}</td>'
                f'<td>{self._fmt(v.get("best_practice", 0))}</td></tr>\n'
            )
        return (
            '<h2>Benchmark Comparison</h2>\n'
            '<table>\n<tr><th>Source</th><th>Typical</th>'
            '<th>Good Practice</th><th>Best Practice</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_facility_vs_benchmarks(self, data: Dict[str, Any]) -> str:
        """Render HTML facility vs benchmarks section."""
        comparisons = data.get("facility_vs_benchmarks", [])
        facility_eui = data.get("facility_eui", 0)
        rows = ""
        for c in comparisons:
            status = c.get("status", "")
            cls = ("status-pass" if status == "PASS"
                   else ("status-fail" if status == "FAIL" else ""))
            rows += (
                f'<tr><td>{c.get("source", "-")} ({c.get("level", "")})</td>'
                f'<td>{self._fmt(c.get("benchmark_eui", 0))}</td>'
                f'<td>{self._fmt(c.get("facility_eui", 0))}</td>'
                f'<td>{self._fmt(c.get("gap_pct", 0))}%</td>'
                f'<td class="{cls}">{status}</td></tr>\n'
            )
        return (
            '<h2>Facility vs Benchmarks</h2>\n'
            f'<p>Facility EUI: <strong>{self._fmt(facility_eui)} kWh/m2/yr</strong></p>\n'
            '<table>\n<tr><th>Source (Level)</th><th>Benchmark</th>'
            '<th>Facility</th><th>Gap</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_cross_source_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML cross-source analysis."""
        csa = data.get("cross_source_analysis", {})
        findings = csa.get("findings", [])
        items = "".join(f'<li>{f}</li>\n' for f in findings)
        return (
            '<h2>Cross-Source Analysis</h2>\n'
            f'<div class="info-box">'
            f'<p><strong>Recommended:</strong> {csa.get("recommended_source", "-")} = '
            f'{self._fmt(csa.get("recommended_value", 0))} kWh/m2/yr</p>'
            f'<p><strong>Spread:</strong> {self._fmt(csa.get("min_benchmark", 0))} - '
            f'{self._fmt(csa.get("max_benchmark", 0))} kWh/m2/yr | '
            f'<strong>CV:</strong> {self._fmt(csa.get("cv_pct", 0))}%</p>'
            f'<p>{csa.get("rationale", "")}</p>'
            '</div>\n'
            f'<ul>\n{items}</ul>'
        )

    def _html_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology notes."""
        notes = data.get("methodology_notes", [
            "ENERGY STAR values converted from kBtu/ft2/yr to kWh/m2/yr.",
            "CIBSE TM46 benchmarks use UK climate assumptions.",
            "DIN V 18599 uses German reference climate.",
        ])
        items = "".join(f'<li>{n}</li>\n' for n in notes)
        return f'<h2>Methodology Notes</h2>\n<ul>\n{items}</ul>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        values = data.get("benchmark_values", [])
        comparisons = data.get("facility_vs_benchmarks", [])
        return {
            "benchmark_bar": {
                "type": "grouped_bar",
                "labels": [v.get("source", "") for v in values],
                "series": {
                    "typical": [v.get("typical", 0) for v in values],
                    "good_practice": [v.get("good_practice", 0) for v in values],
                    "best_practice": [v.get("best_practice", 0) for v in values],
                },
                "facility_line": data.get("facility_eui", 0),
            },
            "gap_waterfall": {
                "type": "waterfall",
                "labels": [c.get("source", "") for c in comparisons],
                "values": [c.get("gap_kwh_m2", 0) for c in comparisons],
            },
            "source_radar": {
                "type": "radar",
                "labels": [v.get("source", "") for v in values],
                "series": {
                    "benchmark": [v.get("typical", 0) for v in values],
                    "facility": [data.get("facility_eui", 0)] * len(values),
                },
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".status-pass{color:#198754;font-weight:700;}"
            ".status-fail{color:#dc3545;font-weight:700;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
