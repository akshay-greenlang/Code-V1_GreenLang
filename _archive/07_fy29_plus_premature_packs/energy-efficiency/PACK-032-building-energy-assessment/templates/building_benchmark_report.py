# -*- coding: utf-8 -*-
"""
BuildingBenchmarkReportTemplate - Peer comparison benchmark report for PACK-032.

Generates building energy benchmark reports with performance summaries,
EUI analysis, DEC rating context, CRREM pathway compliance, Energy Star
scores, end-use breakdowns, weather normalization, peer comparison
chart data, and gap-to-best-practice analysis.

Sections:
    1. Performance Summary
    2. EUI Analysis
    3. DEC Rating Context
    4. CRREM Pathway Compliance
    5. Energy Star Score
    6. End-Use Breakdown
    7. Weather Normalization
    8. Peer Comparison Chart Data
    9. Gap to Best Practice
   10. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BuildingBenchmarkReportTemplate:
    """
    Building energy benchmark and peer comparison report template.

    Renders benchmark reports with EUI analysis, CRREM pathway checks,
    Energy Star scoring, weather normalization, and peer comparisons
    across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    BENCHMARK_SECTIONS: List[str] = [
        "Performance Summary",
        "EUI Analysis",
        "DEC Rating",
        "CRREM Pathway",
        "Energy Star Score",
        "End-Use Breakdown",
        "Weather Normalization",
        "Peer Comparison",
        "Gap to Best Practice",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BuildingBenchmarkReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render benchmark report as Markdown.

        Args:
            data: Benchmark analysis data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_performance_summary(data),
            self._md_eui_analysis(data),
            self._md_dec_rating(data),
            self._md_crrem_pathway(data),
            self._md_energy_star(data),
            self._md_end_use_breakdown(data),
            self._md_weather_normalization(data),
            self._md_peer_comparison(data),
            self._md_gap_analysis(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render benchmark report as self-contained HTML.

        Args:
            data: Benchmark analysis data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_performance_summary(data),
            self._html_eui_analysis(data),
            self._html_dec_rating(data),
            self._html_crrem_pathway(data),
            self._html_energy_star(data),
            self._html_end_use_breakdown(data),
            self._html_weather_normalization(data),
            self._html_peer_comparison(data),
            self._html_gap_analysis(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Building Energy Benchmark Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render benchmark report as structured JSON.

        Args:
            data: Benchmark analysis data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "building_benchmark_report",
            "version": "32.0.0",
            "generated_at": self.generated_at.isoformat(),
            "performance_summary": self._json_performance_summary(data),
            "eui_analysis": data.get("eui_analysis", {}),
            "dec_rating": data.get("dec_rating", {}),
            "crrem_pathway": data.get("crrem_pathway", {}),
            "energy_star": data.get("energy_star", {}),
            "end_use_breakdown": data.get("end_use_breakdown", []),
            "weather_normalization": data.get("weather_normalization", {}),
            "peer_comparison": data.get("peer_comparison", {}),
            "gap_analysis": data.get("gap_analysis", []),
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
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Building Energy Benchmark Report\n\n"
            f"**Building:** {name}  \n"
            f"**Address:** {data.get('address', '-')}  \n"
            f"**Sector:** {data.get('sector', '-')}  \n"
            f"**Benchmark Year:** {data.get('benchmark_year', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 BuildingBenchmarkReportTemplate v32.0.0\n\n---"
        )

    def _md_performance_summary(self, data: Dict[str, Any]) -> str:
        """Render performance summary section."""
        s = data.get("performance_summary", {})
        return (
            "## 1. Performance Summary\n\n"
            "| Metric | Value | Benchmark | Percentile |\n"
            "|--------|-------|-----------|------------|\n"
            f"| EUI (kWh/m2/yr) | {self._fmt(s.get('eui', 0))} "
            f"| {self._fmt(s.get('benchmark_eui', 0))} "
            f"| {s.get('eui_percentile', '-')}th |\n"
            f"| Carbon (kgCO2/m2/yr) | {self._fmt(s.get('carbon_intensity', 0))} "
            f"| {self._fmt(s.get('benchmark_carbon', 0))} "
            f"| {s.get('carbon_percentile', '-')}th |\n"
            f"| Energy Cost (per m2/yr) | {s.get('cost_per_m2', '-')} "
            f"| {s.get('benchmark_cost_m2', '-')} "
            f"| {s.get('cost_percentile', '-')}th |\n"
            f"| DEC Rating | {s.get('dec_score', '-')} "
            f"| {s.get('benchmark_dec', '-')} | - |\n"
            f"| Energy Star Score | {s.get('energy_star_score', '-')} "
            f"| 50 | - |"
        )

    def _md_eui_analysis(self, data: Dict[str, Any]) -> str:
        """Render EUI analysis section."""
        eui = data.get("eui_analysis", {})
        trend = eui.get("annual_trend", [])
        lines = [
            "## 2. EUI Analysis\n",
            f"**Current EUI:** {self._fmt(eui.get('current_eui', 0))} kWh/m2/yr  ",
            f"**Weather-Adjusted EUI:** {self._fmt(eui.get('weather_adjusted_eui', 0))} kWh/m2/yr  ",
            f"**Sector Median:** {self._fmt(eui.get('sector_median', 0))} kWh/m2/yr  ",
            f"**Best Practice:** {self._fmt(eui.get('best_practice', 0))} kWh/m2/yr  ",
            f"**Gap to Best Practice:** {self._fmt(eui.get('gap_to_best', 0))} kWh/m2/yr  ",
            f"**3-Year Trend:** {eui.get('trend_direction', '-')}",
        ]
        if trend:
            lines.extend([
                "\n### Annual EUI Trend\n",
                "| Year | EUI (kWh/m2) | vs Previous | vs Benchmark |",
                "|------|-------------|-------------|-------------|",
            ])
            for t in trend:
                lines.append(
                    f"| {t.get('year', '-')} "
                    f"| {self._fmt(t.get('eui', 0))} "
                    f"| {t.get('vs_previous', '-')} "
                    f"| {t.get('vs_benchmark', '-')} |"
                )
        return "\n".join(lines)

    def _md_dec_rating(self, data: Dict[str, Any]) -> str:
        """Render DEC rating context section."""
        dec = data.get("dec_rating", {})
        return (
            "## 3. DEC Rating Context\n\n"
            f"**DEC Score:** {dec.get('score', '-')}  \n"
            f"**DEC Band:** {dec.get('band', '-')}  \n"
            f"**Typical Score for Type:** {dec.get('typical_score', '-')}  \n"
            f"**vs Typical:** {dec.get('vs_typical', '-')}  \n"
            f"**Electricity Component:** {dec.get('electricity_score', '-')}  \n"
            f"**Heating Component:** {dec.get('heating_score', '-')}  \n"
            f"**Trend (3-Year):** {dec.get('trend', '-')}"
        )

    def _md_crrem_pathway(self, data: Dict[str, Any]) -> str:
        """Render CRREM pathway compliance section."""
        crrem = data.get("crrem_pathway", {})
        years = crrem.get("pathway_data", [])
        lines = [
            "## 4. CRREM Pathway Compliance\n",
            f"**CRREM Compliant:** {'Yes' if crrem.get('compliant', False) else 'No'}  ",
            f"**Stranding Year:** {crrem.get('stranding_year', '-')}  ",
            f"**Current Carbon Intensity:** {self._fmt(crrem.get('current_intensity', 0))} kgCO2/m2  ",
            f"**CRREM Target (2030):** {self._fmt(crrem.get('target_2030', 0))} kgCO2/m2  ",
            f"**CRREM Target (2050):** {self._fmt(crrem.get('target_2050', 0))} kgCO2/m2  ",
            f"**Reduction Required:** {self._fmt(crrem.get('reduction_required_pct', 0))}%",
        ]
        if years:
            lines.extend([
                "\n### CRREM Pathway Trajectory\n",
                "| Year | Building (kgCO2/m2) | CRREM Target | Status |",
                "|------|--------------------|-----------| -------|",
            ])
            for y in years:
                lines.append(
                    f"| {y.get('year', '-')} "
                    f"| {self._fmt(y.get('building_intensity', 0))} "
                    f"| {self._fmt(y.get('crrem_target', 0))} "
                    f"| {y.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_energy_star(self, data: Dict[str, Any]) -> str:
        """Render Energy Star score section."""
        es = data.get("energy_star", {})
        return (
            "## 5. Energy Star Score\n\n"
            f"**Score:** {es.get('score', '-')} / 100  \n"
            f"**Certification Eligible:** {'Yes' if es.get('certification_eligible', False) else 'No'}  \n"
            f"**Source EUI:** {self._fmt(es.get('source_eui', 0))} kBtu/ft2  \n"
            f"**Site EUI:** {self._fmt(es.get('site_eui', 0))} kBtu/ft2  \n"
            f"**National Median:** {self._fmt(es.get('national_median', 0))} kBtu/ft2  \n"
            f"**Weather-Normalized Source:** {self._fmt(es.get('weather_norm_source', 0))} kBtu/ft2  \n"
            f"**Points to Certification:** {es.get('points_to_75', '-')}"
        )

    def _md_end_use_breakdown(self, data: Dict[str, Any]) -> str:
        """Render end-use breakdown section."""
        end_uses = data.get("end_use_breakdown", [])
        if not end_uses:
            return "## 6. End-Use Breakdown\n\n_No end-use data available._"
        lines = [
            "## 6. End-Use Breakdown\n",
            "| End Use | kWh/m2/yr | Share (%) | vs Benchmark | Status |",
            "|---------|----------|-----------|-------------|--------|",
        ]
        for eu in end_uses:
            lines.append(
                f"| {eu.get('end_use', '-')} "
                f"| {self._fmt(eu.get('kwh_m2', 0))} "
                f"| {self._fmt(eu.get('share_pct', 0))}% "
                f"| {eu.get('vs_benchmark', '-')} "
                f"| {eu.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render weather normalization section."""
        wn = data.get("weather_normalization", {})
        monthly = wn.get("monthly", [])
        lines = [
            "## 7. Weather Normalization\n",
            f"**Method:** {wn.get('method', '-')}  ",
            f"**Baseline HDD:** {self._fmt(wn.get('baseline_hdd', 0), 0)}  ",
            f"**Baseline CDD:** {self._fmt(wn.get('baseline_cdd', 0), 0)}  ",
            f"**Actual HDD:** {self._fmt(wn.get('actual_hdd', 0), 0)}  ",
            f"**Actual CDD:** {self._fmt(wn.get('actual_cdd', 0), 0)}  ",
            f"**Normalized EUI:** {self._fmt(wn.get('normalized_eui', 0))} kWh/m2  ",
            f"**Raw EUI:** {self._fmt(wn.get('raw_eui', 0))} kWh/m2  ",
            f"**Weather Adjustment:** {self._fmt(wn.get('adjustment_pct', 0))}%",
        ]
        if monthly:
            lines.extend([
                "\n### Monthly Normalization\n",
                "| Month | Raw (kWh) | Normalized (kWh) | HDD | CDD |",
                "|-------|----------|-----------------|-----|-----|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('raw_kwh', 0), 0)} "
                    f"| {self._fmt(m.get('normalized_kwh', 0), 0)} "
                    f"| {self._fmt(m.get('hdd', 0), 0)} "
                    f"| {self._fmt(m.get('cdd', 0), 0)} |"
                )
        return "\n".join(lines)

    def _md_peer_comparison(self, data: Dict[str, Any]) -> str:
        """Render peer comparison section."""
        pc = data.get("peer_comparison", {})
        peers = pc.get("peers", [])
        lines = [
            "## 8. Peer Comparison\n",
            f"**Peer Group:** {pc.get('peer_group', '-')}  ",
            f"**Sample Size:** {pc.get('sample_size', 0)}  ",
            f"**Building Rank:** {pc.get('rank', '-')} of {pc.get('sample_size', 0)}",
        ]
        if peers:
            lines.extend([
                "\n### Peer Buildings\n",
                "| Building | EUI (kWh/m2) | DEC | Rank |",
                "|----------|-------------|-----|------|",
            ])
            for p in peers:
                lines.append(
                    f"| {p.get('name', '-')} "
                    f"| {self._fmt(p.get('eui', 0))} "
                    f"| {p.get('dec_score', '-')} "
                    f"| {p.get('rank', '-')} |"
                )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap to best practice analysis section."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return "## 9. Gap to Best Practice\n\n_No gap analysis data._"
        lines = [
            "## 9. Gap to Best Practice\n",
            "| Area | Current | Best Practice | Gap | Priority |",
            "|------|---------|-------------|-----|----------|",
        ]
        for g in gaps:
            lines.append(
                f"| {g.get('area', '-')} "
                f"| {g.get('current', '-')} "
                f"| {g.get('best_practice', '-')} "
                f"| {g.get('gap', '-')} "
                f"| {g.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 BuildingBenchmarkReportTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Building Energy Benchmark Report</h1>\n'
            f'<p class="subtitle">Building: {name} | Sector: {data.get("sector", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_performance_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML performance summary with KPI cards."""
        s = data.get("performance_summary", {})
        return (
            '<h2>Performance Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">EUI</span>'
            f'<span class="value">{self._fmt(s.get("eui", 0))}</span>'
            f'<span class="label">kWh/m2/yr</span></div>\n'
            f'<div class="card"><span class="label">Carbon</span>'
            f'<span class="value">{self._fmt(s.get("carbon_intensity", 0))}</span>'
            f'<span class="label">kgCO2/m2/yr</span></div>\n'
            f'<div class="card"><span class="label">DEC</span>'
            f'<span class="value">{s.get("dec_score", "-")}</span></div>\n'
            f'<div class="card"><span class="label">E-Star</span>'
            f'<span class="value">{s.get("energy_star_score", "-")}</span></div>\n'
            '</div>'
        )

    def _html_eui_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML EUI analysis."""
        eui = data.get("eui_analysis", {})
        trend = eui.get("annual_trend", [])
        rows = ""
        for t in trend:
            rows += (
                f'<tr><td>{t.get("year", "-")}</td>'
                f'<td>{self._fmt(t.get("eui", 0))}</td>'
                f'<td>{t.get("vs_benchmark", "-")}</td></tr>\n'
            )
        return (
            '<h2>EUI Analysis</h2>\n'
            f'<p>Current: {self._fmt(eui.get("current_eui", 0))} kWh/m2/yr | '
            f'Best Practice: {self._fmt(eui.get("best_practice", 0))} kWh/m2/yr</p>\n'
            '<table>\n<tr><th>Year</th><th>EUI</th><th>vs Benchmark</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_dec_rating(self, data: Dict[str, Any]) -> str:
        """Render HTML DEC rating."""
        dec = data.get("dec_rating", {})
        return (
            '<h2>DEC Rating</h2>\n'
            f'<p>Score: <strong>{dec.get("score", "-")}</strong> | '
            f'Band: <strong>{dec.get("band", "-")}</strong> | '
            f'Typical: {dec.get("typical_score", "-")}</p>'
        )

    def _html_crrem_pathway(self, data: Dict[str, Any]) -> str:
        """Render HTML CRREM pathway compliance."""
        crrem = data.get("crrem_pathway", {})
        years = crrem.get("pathway_data", [])
        rows = ""
        for y in years:
            rows += (
                f'<tr><td>{y.get("year", "-")}</td>'
                f'<td>{self._fmt(y.get("building_intensity", 0))}</td>'
                f'<td>{self._fmt(y.get("crrem_target", 0))}</td>'
                f'<td>{y.get("status", "-")}</td></tr>\n'
            )
        compliant = "Yes" if crrem.get("compliant", False) else "No"
        return (
            '<h2>CRREM Pathway Compliance</h2>\n'
            f'<p>Compliant: <strong>{compliant}</strong> | '
            f'Stranding Year: {crrem.get("stranding_year", "-")}</p>\n'
            '<table>\n<tr><th>Year</th><th>Building</th><th>CRREM Target</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_energy_star(self, data: Dict[str, Any]) -> str:
        """Render HTML Energy Star score."""
        es = data.get("energy_star", {})
        eligible = "Yes" if es.get("certification_eligible", False) else "No"
        return (
            '<h2>Energy Star Score</h2>\n'
            f'<p>Score: <strong>{es.get("score", "-")}/100</strong> | '
            f'Certification Eligible: {eligible}</p>'
        )

    def _html_end_use_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML end-use breakdown."""
        end_uses = data.get("end_use_breakdown", [])
        rows = ""
        for eu in end_uses:
            rows += (
                f'<tr><td>{eu.get("end_use", "-")}</td>'
                f'<td>{self._fmt(eu.get("kwh_m2", 0))}</td>'
                f'<td>{self._fmt(eu.get("share_pct", 0))}%</td>'
                f'<td>{eu.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>End-Use Breakdown</h2>\n'
            '<table>\n<tr><th>End Use</th><th>kWh/m2</th><th>Share</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_weather_normalization(self, data: Dict[str, Any]) -> str:
        """Render HTML weather normalization."""
        wn = data.get("weather_normalization", {})
        return (
            '<h2>Weather Normalization</h2>\n'
            f'<p>Method: {wn.get("method", "-")} | '
            f'Normalized EUI: {self._fmt(wn.get("normalized_eui", 0))} kWh/m2 | '
            f'Adjustment: {self._fmt(wn.get("adjustment_pct", 0))}%</p>'
        )

    def _html_peer_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML peer comparison."""
        pc = data.get("peer_comparison", {})
        peers = pc.get("peers", [])
        rows = ""
        for p in peers:
            rows += (
                f'<tr><td>{p.get("name", "-")}</td>'
                f'<td>{self._fmt(p.get("eui", 0))}</td>'
                f'<td>{p.get("dec_score", "-")}</td>'
                f'<td>{p.get("rank", "-")}</td></tr>\n'
            )
        return (
            '<h2>Peer Comparison</h2>\n'
            f'<p>Peer Group: {pc.get("peer_group", "-")} | '
            f'Rank: {pc.get("rank", "-")} of {pc.get("sample_size", 0)}</p>\n'
            '<table>\n<tr><th>Building</th><th>EUI</th><th>DEC</th>'
            f'<th>Rank</th></tr>\n{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis."""
        gaps = data.get("gap_analysis", [])
        rows = ""
        for g in gaps:
            rows += (
                f'<tr><td>{g.get("area", "-")}</td>'
                f'<td>{g.get("current", "-")}</td>'
                f'<td>{g.get("best_practice", "-")}</td>'
                f'<td>{g.get("gap", "-")}</td>'
                f'<td>{g.get("priority", "-")}</td></tr>\n'
            )
        return (
            '<h2>Gap to Best Practice</h2>\n'
            '<table>\n<tr><th>Area</th><th>Current</th><th>Best Practice</th>'
            f'<th>Gap</th><th>Priority</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_performance_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON performance summary."""
        s = data.get("performance_summary", {})
        return {
            "eui": s.get("eui", 0),
            "benchmark_eui": s.get("benchmark_eui", 0),
            "eui_percentile": s.get("eui_percentile", 0),
            "carbon_intensity": s.get("carbon_intensity", 0),
            "dec_score": s.get("dec_score", 0),
            "energy_star_score": s.get("energy_star_score", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        end_uses = data.get("end_use_breakdown", [])
        trend = data.get("eui_analysis", {}).get("annual_trend", [])
        peers = data.get("peer_comparison", {}).get("peers", [])
        return {
            "end_use_pie": {
                "type": "pie",
                "labels": [eu.get("end_use", "") for eu in end_uses],
                "values": [eu.get("kwh_m2", 0) for eu in end_uses],
            },
            "eui_trend_line": {
                "type": "line",
                "labels": [t.get("year", "") for t in trend],
                "values": [t.get("eui", 0) for t in trend],
            },
            "peer_bar": {
                "type": "bar",
                "labels": [p.get("name", "") for p in peers],
                "values": [p.get("eui", 0) for p in peers],
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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
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

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
