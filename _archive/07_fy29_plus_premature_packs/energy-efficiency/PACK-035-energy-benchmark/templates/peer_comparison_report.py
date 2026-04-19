# -*- coding: utf-8 -*-
"""
PeerComparisonReportTemplate - Peer group comparison report for PACK-035.

Generates peer comparison reports that rank a facility against a defined
peer group using percentile rankings, quartile analysis, and statistical
summaries. Helps facility managers understand their relative performance
position and distance to efficiency targets.

Sections:
    1. Header & Facility Overview
    2. Peer Group Definition
    3. Percentile Ranking
    4. Quartile Analysis
    5. Distance to Targets
    6. Peer Statistics Table
    7. Performance Distribution Chart Data
    8. Recommendations
    9. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PeerComparisonReportTemplate:
    """
    Peer group comparison report template.

    Renders peer comparison reports with percentile rankings, quartile
    analysis, distance to efficiency targets, and statistical summaries
    across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    QUARTILE_LABELS: Dict[int, str] = {
        1: "Top Performer (Q1)",
        2: "Above Average (Q2)",
        3: "Below Average (Q3)",
        4: "Underperformer (Q4)",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PeerComparisonReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render peer comparison report as Markdown.

        Args:
            data: Peer comparison data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_facility_overview(data),
            self._md_peer_group(data),
            self._md_percentile_ranking(data),
            self._md_quartile_analysis(data),
            self._md_distance_to_targets(data),
            self._md_peer_statistics(data),
            self._md_distribution(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render peer comparison report as self-contained HTML.

        Args:
            data: Peer comparison data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_facility_overview(data),
            self._html_peer_group(data),
            self._html_percentile_ranking(data),
            self._html_quartile_analysis(data),
            self._html_distance_to_targets(data),
            self._html_peer_statistics(data),
            self._html_recommendations(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Peer Comparison Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render peer comparison report as structured JSON.

        Args:
            data: Peer comparison data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "peer_comparison_report",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "facility": data.get("facility", {}),
            "peer_group": data.get("peer_group", {}),
            "percentile_ranking": data.get("percentile_ranking", {}),
            "quartile_analysis": data.get("quartile_analysis", {}),
            "distance_to_targets": data.get("distance_to_targets", []),
            "peer_statistics": data.get("peer_statistics", {}),
            "recommendations": data.get("recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with report metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Peer Comparison Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Peer Group:** {data.get('peer_group_name', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 PeerComparisonReportTemplate v35.0.0\n\n---"
        )

    def _md_facility_overview(self, data: Dict[str, Any]) -> str:
        """Render facility overview section."""
        f = data.get("facility", {})
        return (
            "## 1. Facility Overview\n\n"
            "| Property | Value |\n|----------|-------|\n"
            f"| Name | {f.get('name', '-')} |\n"
            f"| Building Type | {f.get('building_type', '-')} |\n"
            f"| Gross Floor Area | {self._fmt(f.get('gross_floor_area_sqm', 0), 0)} m2 |\n"
            f"| Climate Zone | {f.get('climate_zone', '-')} |\n"
            f"| Site EUI | {self._fmt(f.get('site_eui', 0))} kWh/m2/yr |\n"
            f"| Source EUI | {self._fmt(f.get('source_eui', 0))} kWh/m2/yr |"
        )

    def _md_peer_group(self, data: Dict[str, Any]) -> str:
        """Render peer group definition section."""
        pg = data.get("peer_group", {})
        criteria = pg.get("criteria", [])
        lines = [
            "## 2. Peer Group Definition\n",
            f"**Group Name:** {pg.get('name', '-')}  ",
            f"**Building Type Filter:** {pg.get('building_type', '-')}  ",
            f"**Climate Zone Filter:** {pg.get('climate_zone', '-')}  ",
            f"**Floor Area Range:** {self._fmt(pg.get('area_min_sqm', 0), 0)} - "
            f"{self._fmt(pg.get('area_max_sqm', 0), 0)} m2  ",
            f"**Peer Count:** {pg.get('peer_count', 0)} facilities  ",
            f"**Data Source:** {pg.get('data_source', '-')}",
        ]
        if criteria:
            lines.append("\n### Matching Criteria\n")
            for c in criteria:
                lines.append(f"- {c}")
        return "\n".join(lines)

    def _md_percentile_ranking(self, data: Dict[str, Any]) -> str:
        """Render percentile ranking section."""
        pr = data.get("percentile_ranking", {})
        rankings = pr.get("rankings", [])
        lines = [
            "## 3. Percentile Ranking\n",
            f"**Overall Percentile:** {self._fmt(pr.get('overall_percentile', 0), 0)}th  ",
            f"**Interpretation:** {pr.get('interpretation', '-')}\n",
        ]
        if rankings:
            lines.extend([
                "| Metric | Value | Percentile | Rank / Total |",
                "|--------|-------|-----------|-------------|",
            ])
            for r in rankings:
                lines.append(
                    f"| {r.get('metric', '-')} "
                    f"| {self._fmt(r.get('value', 0))} "
                    f"| {self._fmt(r.get('percentile', 0), 0)}th "
                    f"| {r.get('rank', '-')} / {r.get('total', '-')} |"
                )
        return "\n".join(lines)

    def _md_quartile_analysis(self, data: Dict[str, Any]) -> str:
        """Render quartile analysis section."""
        qa = data.get("quartile_analysis", {})
        current_q = qa.get("current_quartile", 0)
        label = self.QUARTILE_LABELS.get(current_q, "-")
        boundaries = qa.get("boundaries", {})
        lines = [
            "## 4. Quartile Analysis\n",
            f"**Current Quartile:** Q{current_q} - {label}  ",
            f"**EUI to reach Q1:** {self._fmt(qa.get('eui_to_q1', 0))} kWh/m2/yr  ",
            f"**EUI to reach Q2:** {self._fmt(qa.get('eui_to_q2', 0))} kWh/m2/yr\n",
            "| Quartile | EUI Range (kWh/m2/yr) | Facility Count |",
            "|----------|----------------------|---------------|",
        ]
        for q in range(1, 5):
            qkey = f"q{q}"
            b = boundaries.get(qkey, {})
            marker = " <--" if q == current_q else ""
            lines.append(
                f"| Q{q} | {self._fmt(b.get('min', 0))} - {self._fmt(b.get('max', 0))} "
                f"| {b.get('count', 0)}{marker} |"
            )
        return "\n".join(lines)

    def _md_distance_to_targets(self, data: Dict[str, Any]) -> str:
        """Render distance to targets section."""
        targets = data.get("distance_to_targets", [])
        if not targets:
            return "## 5. Distance to Targets\n\n_No targets defined._"
        lines = [
            "## 5. Distance to Targets\n",
            "| Target | Target EUI | Current EUI | Gap | Gap (%) | Savings Potential |",
            "|--------|-----------|------------|-----|---------|------------------|",
        ]
        for t in targets:
            lines.append(
                f"| {t.get('target_name', '-')} "
                f"| {self._fmt(t.get('target_eui', 0))} "
                f"| {self._fmt(t.get('current_eui', 0))} "
                f"| {self._fmt(t.get('gap_kwh_m2', 0))} "
                f"| {self._fmt(t.get('gap_pct', 0))}% "
                f"| {self._fmt(t.get('savings_potential_kwh', 0), 0)} kWh/yr |"
            )
        return "\n".join(lines)

    def _md_peer_statistics(self, data: Dict[str, Any]) -> str:
        """Render peer statistics table section."""
        ps = data.get("peer_statistics", {})
        return (
            "## 6. Peer Statistics\n\n"
            "| Statistic | Site EUI | Source EUI |\n"
            "|-----------|---------|----------|\n"
            f"| Mean | {self._fmt(ps.get('mean_site_eui', 0))} "
            f"| {self._fmt(ps.get('mean_source_eui', 0))} |\n"
            f"| Median | {self._fmt(ps.get('median_site_eui', 0))} "
            f"| {self._fmt(ps.get('median_source_eui', 0))} |\n"
            f"| Std Dev | {self._fmt(ps.get('std_site_eui', 0))} "
            f"| {self._fmt(ps.get('std_source_eui', 0))} |\n"
            f"| P10 (Best) | {self._fmt(ps.get('p10_site_eui', 0))} "
            f"| {self._fmt(ps.get('p10_source_eui', 0))} |\n"
            f"| P25 (Q1) | {self._fmt(ps.get('p25_site_eui', 0))} "
            f"| {self._fmt(ps.get('p25_source_eui', 0))} |\n"
            f"| P50 (Median) | {self._fmt(ps.get('p50_site_eui', 0))} "
            f"| {self._fmt(ps.get('p50_source_eui', 0))} |\n"
            f"| P75 (Q3) | {self._fmt(ps.get('p75_site_eui', 0))} "
            f"| {self._fmt(ps.get('p75_source_eui', 0))} |\n"
            f"| P90 (Worst) | {self._fmt(ps.get('p90_site_eui', 0))} "
            f"| {self._fmt(ps.get('p90_source_eui', 0))} |\n"
            f"| Min | {self._fmt(ps.get('min_site_eui', 0))} "
            f"| {self._fmt(ps.get('min_source_eui', 0))} |\n"
            f"| Max | {self._fmt(ps.get('max_site_eui', 0))} "
            f"| {self._fmt(ps.get('max_source_eui', 0))} |"
        )

    def _md_distribution(self, data: Dict[str, Any]) -> str:
        """Render performance distribution section."""
        dist = data.get("distribution", [])
        if not dist:
            return "## 7. Performance Distribution\n\n_No distribution data available._"
        lines = [
            "## 7. Performance Distribution\n",
            "| EUI Range (kWh/m2/yr) | Facility Count | Percentage |",
            "|----------------------|---------------|-----------|",
        ]
        for d in dist:
            marker = " <-- YOUR FACILITY" if d.get("contains_facility", False) else ""
            lines.append(
                f"| {d.get('range_label', '-')} "
                f"| {d.get('count', 0)} "
                f"| {self._fmt(d.get('pct', 0))}%{marker} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 8. Recommendations\n\n_No specific recommendations._"
        lines = [
            "## 8. Recommendations\n",
            "| # | Recommendation | Priority | Estimated Impact |",
            "|---|---------------|----------|-----------------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('recommendation', '-')} "
                f"| {r.get('priority', '-')} "
                f"| {r.get('estimated_impact', '-')} |"
            )
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
            f'<h1>Peer Comparison Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Peer Group: {data.get("peer_group_name", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_facility_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML facility overview."""
        f = data.get("facility", {})
        return (
            '<h2>Facility Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Building Type</span>'
            f'<span class="value">{f.get("building_type", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Floor Area</span>'
            f'<span class="value">{self._fmt(f.get("gross_floor_area_sqm", 0), 0)} m2</span></div>\n'
            f'  <div class="card"><span class="label">Site EUI</span>'
            f'<span class="value">{self._fmt(f.get("site_eui", 0))} kWh/m2</span></div>\n'
            f'  <div class="card"><span class="label">Source EUI</span>'
            f'<span class="value">{self._fmt(f.get("source_eui", 0))} kWh/m2</span></div>\n'
            '</div>'
        )

    def _html_peer_group(self, data: Dict[str, Any]) -> str:
        """Render HTML peer group definition."""
        pg = data.get("peer_group", {})
        return (
            '<h2>Peer Group Definition</h2>\n'
            f'<div class="info-box">'
            f'<p><strong>Group:</strong> {pg.get("name", "-")} | '
            f'<strong>Type:</strong> {pg.get("building_type", "-")} | '
            f'<strong>Climate:</strong> {pg.get("climate_zone", "-")} | '
            f'<strong>Peers:</strong> {pg.get("peer_count", 0)} facilities</p>'
            f'<p><strong>Area Range:</strong> {self._fmt(pg.get("area_min_sqm", 0), 0)} - '
            f'{self._fmt(pg.get("area_max_sqm", 0), 0)} m2 | '
            f'<strong>Source:</strong> {pg.get("data_source", "-")}</p>'
            '</div>'
        )

    def _html_percentile_ranking(self, data: Dict[str, Any]) -> str:
        """Render HTML percentile ranking with visual bar."""
        pr = data.get("percentile_ranking", {})
        pct = pr.get("overall_percentile", 50)
        bar_color = "#198754" if pct <= 25 else ("#ffc107" if pct <= 50 else "#dc3545")
        rankings = pr.get("rankings", [])
        rows = ""
        for r in rankings:
            rows += (
                f'<tr><td>{r.get("metric", "-")}</td>'
                f'<td>{self._fmt(r.get("value", 0))}</td>'
                f'<td>{self._fmt(r.get("percentile", 0), 0)}th</td>'
                f'<td>{r.get("rank", "-")} / {r.get("total", "-")}</td></tr>\n'
            )
        return (
            '<h2>Percentile Ranking</h2>\n'
            f'<div class="percentile-bar">'
            f'<div class="percentile-fill" style="width:{100 - pct}%;'
            f'background:{bar_color};"></div>'
            f'<span class="percentile-label">{pct}th Percentile</span></div>\n'
            f'<p>{pr.get("interpretation", "")}</p>\n'
            '<table>\n<tr><th>Metric</th><th>Value</th>'
            f'<th>Percentile</th><th>Rank</th></tr>\n{rows}</table>'
        )

    def _html_quartile_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML quartile analysis."""
        qa = data.get("quartile_analysis", {})
        current_q = qa.get("current_quartile", 0)
        boundaries = qa.get("boundaries", {})
        rows = ""
        for q in range(1, 5):
            b = boundaries.get(f"q{q}", {})
            cls = ' class="highlight-row"' if q == current_q else ""
            rows += (
                f'<tr{cls}><td>Q{q}</td>'
                f'<td>{self._fmt(b.get("min", 0))} - {self._fmt(b.get("max", 0))}</td>'
                f'<td>{b.get("count", 0)}</td></tr>\n'
            )
        label = self.QUARTILE_LABELS.get(current_q, "-")
        return (
            '<h2>Quartile Analysis</h2>\n'
            f'<p>Current Position: <strong>Q{current_q} - {label}</strong></p>\n'
            '<table>\n<tr><th>Quartile</th><th>EUI Range</th>'
            f'<th>Facilities</th></tr>\n{rows}</table>'
        )

    def _html_distance_to_targets(self, data: Dict[str, Any]) -> str:
        """Render HTML distance to targets."""
        targets = data.get("distance_to_targets", [])
        rows = ""
        for t in targets:
            gap_pct = t.get("gap_pct", 0)
            cls = "gap-positive" if gap_pct > 0 else "gap-negative"
            rows += (
                f'<tr><td>{t.get("target_name", "-")}</td>'
                f'<td>{self._fmt(t.get("target_eui", 0))}</td>'
                f'<td>{self._fmt(t.get("current_eui", 0))}</td>'
                f'<td class="{cls}">{self._fmt(gap_pct)}%</td>'
                f'<td>{self._fmt(t.get("savings_potential_kwh", 0), 0)} kWh/yr</td></tr>\n'
            )
        return (
            '<h2>Distance to Targets</h2>\n'
            '<table>\n<tr><th>Target</th><th>Target EUI</th><th>Current EUI</th>'
            f'<th>Gap (%)</th><th>Savings Potential</th></tr>\n{rows}</table>'
        )

    def _html_peer_statistics(self, data: Dict[str, Any]) -> str:
        """Render HTML peer statistics table."""
        ps = data.get("peer_statistics", {})
        stats = [
            ("Mean", ps.get("mean_site_eui", 0), ps.get("mean_source_eui", 0)),
            ("Median", ps.get("median_site_eui", 0), ps.get("median_source_eui", 0)),
            ("Std Dev", ps.get("std_site_eui", 0), ps.get("std_source_eui", 0)),
            ("P10 (Best)", ps.get("p10_site_eui", 0), ps.get("p10_source_eui", 0)),
            ("P90 (Worst)", ps.get("p90_site_eui", 0), ps.get("p90_source_eui", 0)),
        ]
        rows = "".join(
            f'<tr><td>{label}</td><td>{self._fmt(site)}</td>'
            f'<td>{self._fmt(source)}</td></tr>\n'
            for label, site, source in stats
        )
        return (
            '<h2>Peer Statistics</h2>\n'
            '<table>\n<tr><th>Statistic</th><th>Site EUI</th>'
            f'<th>Source EUI</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        items = "".join(
            f'<li><strong>[{r.get("priority", "-")}]</strong> '
            f'{r.get("recommendation", "-")} '
            f'(Impact: {r.get("estimated_impact", "-")})</li>\n'
            for r in recs
        )
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        dist = data.get("distribution", [])
        targets = data.get("distance_to_targets", [])
        return {
            "distribution_histogram": {
                "type": "histogram",
                "labels": [d.get("range_label", "") for d in dist],
                "values": [d.get("count", 0) for d in dist],
                "facility_bin": next(
                    (d.get("range_label", "") for d in dist if d.get("contains_facility")),
                    "",
                ),
            },
            "target_gap_bar": {
                "type": "bar",
                "labels": [t.get("target_name", "") for t in targets],
                "series": {
                    "current": [t.get("current_eui", 0) for t in targets],
                    "target": [t.get("target_eui", 0) for t in targets],
                },
            },
            "quartile_box": {
                "type": "box",
                "statistics": data.get("peer_statistics", {}),
                "facility_value": data.get("facility", {}).get("site_eui", 0),
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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".highlight-row{background:#fff3cd !important;}"
            ".percentile-bar{position:relative;height:30px;background:#e9ecef;"
            "border-radius:6px;margin:15px 0;overflow:hidden;}"
            ".percentile-fill{height:100%;border-radius:6px;}"
            ".percentile-label{position:absolute;top:5px;right:10px;font-weight:700;}"
            ".gap-positive{color:#dc3545;font-weight:600;}"
            ".gap-negative{color:#198754;font-weight:600;}"
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
