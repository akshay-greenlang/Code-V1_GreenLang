# -*- coding: utf-8 -*-
"""
ESRSDisclosureMapTemplate - ESRS disclosure mapping report for PACK-015.

Sections:
    1. Disclosure Mapping Overview
    2. Topic Coverage Summary
    3. Disclosure Requirements Table
    4. Gap Analysis
    5. Effort Estimates
    6. Implementation Timeline

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ESRSDisclosureMapTemplate:
    """
    ESRS disclosure mapping report template.

    Renders disclosure requirement coverage by topic, gap analysis
    with missing datapoints, effort estimates for gap closure,
    and an implementation timeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSDisclosureMapTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure map as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overview(data),
            self._md_topic_coverage(data),
            self._md_disclosure_table(data),
            self._md_gap_analysis(data),
            self._md_effort_estimates(data),
            self._md_timeline(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure map as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_disclosure_table(data),
            self._html_gap_analysis(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ESRS Disclosure Map</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESRS disclosure map as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "esrs_disclosure_map",
            "version": "15.0.0",
            "generated_at": self.generated_at.isoformat(),
            "material_topics_count": data.get("material_topics_count", 0),
            "total_drs_mapped": data.get("total_drs_mapped", 0),
            "coverage_pct": data.get("coverage_pct", 0),
            "gaps_identified": data.get("gaps_identified", []),
            "total_effort_weeks": data.get("total_effort_weeks", 0),
            "topic_coverage": data.get("topic_coverage", {}),
            "disclosure_requirements": data.get("disclosure_requirements", []),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"# ESRS Disclosure Mapping Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render mapping overview."""
        topics = data.get("material_topics_count", 0)
        drs = data.get("total_drs_mapped", 0)
        coverage = data.get("coverage_pct", 0)
        gaps = len(data.get("gaps_identified", []))
        effort = data.get("total_effort_weeks", 0)
        return (
            "## Disclosure Mapping Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Material Topics | {topics} |\n"
            f"| Disclosure Requirements Mapped | {drs} |\n"
            f"| Overall Coverage | {self._fmt(coverage)}% |\n"
            f"| Gaps Identified | {gaps} |\n"
            f"| Total Effort to Close Gaps | {self._fmt(effort)} weeks |"
        )

    def _md_topic_coverage(self, data: Dict[str, Any]) -> str:
        """Render per-topic coverage summary."""
        coverage = data.get("topic_coverage", {})
        if not coverage:
            return "## Topic Coverage\n\n_No topic coverage data available._"
        lines = [
            "## Topic Coverage", "",
            "| ESRS Topic | Coverage (%) | Status |",
            "|------------|-------------|--------|",
        ]
        for topic, pct in sorted(coverage.items()):
            if pct >= 80:
                status = "Good"
            elif pct >= 50:
                status = "Partial"
            else:
                status = "Gap"
            lines.append(f"| {topic} | {self._fmt(pct)}% | {status} |")
        return "\n".join(lines)

    def _md_disclosure_table(self, data: Dict[str, Any]) -> str:
        """Render disclosure requirements table."""
        drs = data.get("disclosure_requirements", [])
        if not drs:
            return "## Disclosure Requirements\n\n_No disclosure requirements mapped._"
        lines = [
            "## Disclosure Requirements", "",
            "| DR ID | Name | Topic | Required | Available | Status |",
            "|-------|------|-------|----------|-----------|--------|",
        ]
        for dr in drs:
            lines.append(
                f"| {dr.get('dr_id', '-')} | "
                f"{dr.get('dr_name', '-')} | "
                f"{dr.get('esrs_topic', '-')} | "
                f"{dr.get('datapoints_required', 0)} | "
                f"{dr.get('datapoints_available', 0)} | "
                f"**{dr.get('status', '-')}** |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis."""
        gaps = data.get("gaps_identified", [])
        if not gaps:
            return "## Gap Analysis\n\n_No disclosure gaps identified._"
        lines = [
            "## Gap Analysis", "",
            "| Priority | DR ID | DR Name | Topic | Missing DPs | Effort | Weeks |",
            "|----------|-------|---------|-------|-------------|--------|-------|",
        ]
        for g in gaps:
            lines.append(
                f"| {g.get('priority', '-')} | "
                f"{g.get('dr_id', '-')} | "
                f"{g.get('dr_name', '-')} | "
                f"{g.get('esrs_topic', '-')} | "
                f"{g.get('missing_datapoints', 0)} | "
                f"{g.get('effort_level', '-')} | "
                f"{self._fmt(g.get('estimated_weeks', 0))} |"
            )
        return "\n".join(lines)

    def _md_effort_estimates(self, data: Dict[str, Any]) -> str:
        """Render effort estimate summary."""
        gaps = data.get("gaps_identified", [])
        if not gaps:
            return "## Effort Estimates\n\n_No gaps to estimate._"

        effort_by_level: Dict[str, int] = {}
        for g in gaps:
            level = g.get("effort_level", "unknown")
            if isinstance(level, str):
                effort_by_level[level] = effort_by_level.get(level, 0) + 1
            else:
                effort_by_level[str(level)] = effort_by_level.get(str(level), 0) + 1

        lines = [
            "## Effort Summary by Level", "",
            "| Effort Level | Gap Count |",
            "|-------------|-----------|",
        ]
        for level, count in sorted(effort_by_level.items()):
            lines.append(f"| {level.replace('_', ' ').title()} | {count} |")
        total = data.get("total_effort_weeks", 0)
        lines.append(f"\n**Total Estimated Effort:** {self._fmt(total)} weeks")
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render implementation timeline."""
        gaps = data.get("gaps_identified", [])
        if not gaps:
            return "## Implementation Timeline\n\n_No gaps require remediation._"

        # Group by effort level
        quick_wins = [g for g in gaps if g.get("effort_level") in ("low", "LOW")]
        medium = [g for g in gaps if g.get("effort_level") in ("medium", "MEDIUM")]
        high_effort = [g for g in gaps if g.get("effort_level") in ("high", "HIGH", "very_high", "VERY_HIGH")]

        lines = [
            "## Implementation Timeline", "",
            f"- **Quick Wins (0-2 weeks):** {len(quick_wins)} gap(s)",
            f"- **Medium Term (2-8 weeks):** {len(medium)} gap(s)",
            f"- **Long Term (8+ weeks):** {len(high_effort)} gap(s)",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-015 Double Materiality Pack*"

    # -- HTML sections --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>ESRS Disclosure Mapping Report</h1>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview."""
        coverage = data.get("coverage_pct", 0)
        return f'<h2>Overview</h2>\n<p>Coverage: {self._fmt(coverage)}%</p>'

    def _html_disclosure_table(self, data: Dict[str, Any]) -> str:
        """Render HTML disclosure table."""
        drs = data.get("disclosure_requirements", [])
        rows = ""
        for dr in drs:
            rows += (
                f'<tr><td>{dr.get("dr_id", "-")}</td>'
                f'<td>{dr.get("dr_name", "-")}</td>'
                f'<td>{dr.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>Disclosure Requirements</h2>\n'
            f'<table><tr><th>DR ID</th><th>Name</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis."""
        gaps = len(data.get("gaps_identified", []))
        return f'<h2>Gap Analysis</h2>\n<p>{gaps} gap(s) identified</p>'

    # -- Helpers --

    def _css(self) -> str:
        """Build CSS."""
        return (
            "body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format numeric value."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
