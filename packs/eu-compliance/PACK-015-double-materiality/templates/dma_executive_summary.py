# -*- coding: utf-8 -*-
"""
DMAExecutiveSummaryTemplate - Executive-level DMA summary for PACK-015.

Sections:
    1. Executive Summary Header
    2. Key Metrics Dashboard
    3. Material Topics at a Glance
    4. Matrix Highlights
    5. Stakeholder Engagement Highlights
    6. Disclosure Readiness
    7. Recommendations
    8. Next Steps

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DMAExecutiveSummaryTemplate:
    """
    Executive-level DMA summary report template.

    Renders a concise board/management-ready overview of the double
    materiality assessment including key metrics, material topic
    highlights, stakeholder engagement summary, and disclosure readiness.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DMAExecutiveSummaryTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render DMA executive summary as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_key_metrics(data),
            self._md_material_topics(data),
            self._md_matrix_highlights(data),
            self._md_stakeholder_highlights(data),
            self._md_disclosure_readiness(data),
            self._md_recommendations(data),
            self._md_next_steps(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render DMA executive summary as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_key_metrics(data),
            self._html_material_topics(data),
            self._html_disclosure_readiness(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>DMA Executive Summary</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render DMA executive summary as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "dma_executive_summary",
            "version": "15.0.0",
            "generated_at": self.generated_at.isoformat(),
            "completeness": data.get("completeness", "draft"),
            "material_topics": data.get("material_topics", []),
            "non_material_topics": data.get("non_material_topics", []),
            "total_iros": data.get("total_iros", 0),
            "overall_coverage_pct": data.get("overall_coverage_pct", 0),
            "total_effort_weeks": data.get("total_effort_weeks", 0),
            "stakeholder_validation_passed": data.get("stakeholder_validation_passed", False),
            "topic_results": data.get("topic_results", []),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render executive summary header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        completeness = data.get("completeness", "draft")
        return (
            f"# Double Materiality Assessment - Executive Summary\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Assessment Status:** {completeness.upper()}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render key metrics dashboard."""
        material = len(data.get("material_topics", []))
        non_material = len(data.get("non_material_topics", []))
        total = material + non_material
        iros = data.get("total_iros", 0)
        coverage = data.get("overall_coverage_pct", 0)
        effort = data.get("total_effort_weeks", 0)
        sh_valid = data.get("stakeholder_validation_passed", False)
        return (
            "## Key Metrics\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| ESRS Topics Assessed | {total} |\n"
            f"| **Material Topics** | **{material}** |\n"
            f"| Non-Material Topics | {non_material} |\n"
            f"| IROs Identified | {iros} |\n"
            f"| Disclosure Coverage | {self._fmt(coverage)}% |\n"
            f"| Effort to Full Coverage | {self._fmt(effort)} weeks |\n"
            f"| Stakeholder Validation | {'PASS' if sh_valid else 'FAIL'} |"
        )

    def _md_material_topics(self, data: Dict[str, Any]) -> str:
        """Render material topics summary."""
        topics = data.get("material_topics", [])
        topic_results = data.get("topic_results", [])
        if not topics:
            return "## Material Topics\n\n_No material topics identified._"

        lines = ["## Material Topics at a Glance", ""]
        # Build lookup from topic_results
        result_lookup = {tr.get("topic_id", ""): tr for tr in topic_results}
        for topic_id in topics:
            tr = result_lookup.get(topic_id, {})
            name = tr.get("topic_name", topic_id)
            mat_type = tr.get("materiality_type", "material")
            label = mat_type.replace("_", " ").title()
            lines.append(f"- **{topic_id} - {name}** ({label})")
        return "\n".join(lines)

    def _md_matrix_highlights(self, data: Dict[str, Any]) -> str:
        """Render matrix highlights."""
        topic_results = data.get("topic_results", [])
        if not topic_results:
            return "## Matrix Highlights\n\n_No matrix data available._"

        double = [tr for tr in topic_results if tr.get("materiality_type") == "double_material"]
        impact_only = [tr for tr in topic_results if tr.get("materiality_type") == "impact_only"]
        financial_only = [tr for tr in topic_results if tr.get("materiality_type") == "financial_only"]

        lines = [
            "## Matrix Highlights", "",
            f"- **Double Material:** {len(double)} topic(s) - both impact and financial materiality",
            f"- **Impact Material Only:** {len(impact_only)} topic(s)",
            f"- **Financial Material Only:** {len(financial_only)} topic(s)",
        ]
        return "\n".join(lines)

    def _md_stakeholder_highlights(self, data: Dict[str, Any]) -> str:
        """Render stakeholder engagement highlights."""
        sh_result = data.get("stakeholder_result", {})
        if not sh_result:
            return "## Stakeholder Engagement\n\n_No stakeholder data available._"

        sh_count = sh_result.get("stakeholders_identified", 0)
        con_count = sh_result.get("consultations_recorded", 0)
        validated = sh_result.get("validation_passed", False)

        return (
            "## Stakeholder Engagement Highlights\n\n"
            f"- **Stakeholders Engaged:** {sh_count}\n"
            f"- **Consultations Conducted:** {con_count}\n"
            f"- **ESRS 1 Validation:** {'PASS' if validated else 'FAIL'}"
        )

    def _md_disclosure_readiness(self, data: Dict[str, Any]) -> str:
        """Render disclosure readiness summary."""
        coverage = data.get("overall_coverage_pct", 0)
        gaps = data.get("total_disclosure_gaps", 0)
        effort = data.get("total_effort_weeks", 0)

        if coverage >= 80:
            readiness = "HIGH"
        elif coverage >= 50:
            readiness = "MODERATE"
        else:
            readiness = "LOW"

        return (
            "## Disclosure Readiness\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Readiness Level | **{readiness}** |\n"
            f"| Datapoint Coverage | {self._fmt(coverage)}% |\n"
            f"| Disclosure Gaps | {gaps} |\n"
            f"| Effort to Close | {self._fmt(effort)} weeks |"
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render key recommendations."""
        recs = data.get("recommendations", [])
        if recs:
            lines = ["## Key Recommendations", ""]
            for i, r in enumerate(recs, 1):
                lines.append(f"{i}. {r}")
            return "\n".join(lines)

        # Generate default recommendations based on data
        lines = ["## Key Recommendations", ""]
        coverage = data.get("overall_coverage_pct", 0)
        if coverage < 50:
            lines.append("1. Prioritize closing high-impact disclosure gaps before reporting deadline")
        if not data.get("stakeholder_validation_passed", False):
            lines.append("2. Address stakeholder engagement validation failures")
        if data.get("total_disclosure_gaps", 0) > 10:
            lines.append("3. Establish dedicated workstreams for topic-specific gap closure")
        lines.append("4. Schedule annual DMA refresh to capture regulatory and business changes")
        lines.append("5. Ensure board oversight of material topic decisions")
        return "\n".join(lines)

    def _md_next_steps(self, data: Dict[str, Any]) -> str:
        """Render next steps."""
        return (
            "## Next Steps\n\n"
            "1. Present DMA results to management and board for approval\n"
            "2. Integrate material topics into CSRD reporting process\n"
            "3. Assign data owners for each disclosure requirement gap\n"
            "4. Set timeline for gap closure aligned with reporting deadline\n"
            "5. Schedule annual DMA refresh process"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-015 Double Materiality Pack*"

    # -- HTML sections --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>Double Materiality Assessment - Executive Summary</h1>'

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML key metrics."""
        material = len(data.get("material_topics", []))
        coverage = data.get("overall_coverage_pct", 0)
        return (
            f'<h2>Key Metrics</h2>\n'
            f'<p>Material Topics: {material} | Coverage: {self._fmt(coverage)}%</p>'
        )

    def _html_material_topics(self, data: Dict[str, Any]) -> str:
        """Render HTML material topics."""
        topics = data.get("material_topics", [])
        items = "".join(f'<li>{t}</li>\n' for t in topics)
        return f'<h2>Material Topics</h2>\n<ul>\n{items}</ul>'

    def _html_disclosure_readiness(self, data: Dict[str, Any]) -> str:
        """Render HTML disclosure readiness."""
        coverage = data.get("overall_coverage_pct", 0)
        return f'<h2>Disclosure Readiness</h2>\n<p>Coverage: {self._fmt(coverage)}%</p>'

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
