# -*- coding: utf-8 -*-
"""
ImpactMaterialityReportTemplate - Impact assessment report for PACK-015.

Sections:
    1. Impact Assessment Overview
    2. Severity Score Summary
    3. Topic Distribution
    4. Material Impacts Ranking
    5. Non-Material Impacts
    6. Methodology Notes

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ImpactMaterialityReportTemplate:
    """
    Impact materiality assessment report template.

    Renders severity score breakdowns, topic distributions, material
    impact rankings, and methodology notes for the inside-out
    dimension of double materiality per ESRS 1.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ImpactMaterialityReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render impact materiality report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overview(data),
            self._md_severity_summary(data),
            self._md_topic_distribution(data),
            self._md_material_impacts(data),
            self._md_non_material_impacts(data),
            self._md_methodology(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render impact materiality report as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_severity_summary(data),
            self._html_material_impacts(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Impact Materiality Report</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render impact materiality report as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "impact_materiality_report",
            "version": "15.0.0",
            "generated_at": self.generated_at.isoformat(),
            "matters_assessed": data.get("matters_assessed", 0),
            "material_impacts": data.get("material_impacts", 0),
            "non_material_impacts": data.get("non_material_impacts", 0),
            "severity_scores": data.get("severity_scores", []),
            "ranked_impacts": data.get("ranked_impacts", []),
            "topic_distribution": data.get("topic_distribution", {}),
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
            f"# Impact Materiality Assessment Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render assessment overview."""
        assessed = data.get("matters_assessed", 0)
        material = data.get("material_impacts", 0)
        non_material = data.get("non_material_impacts", 0)
        return (
            "## Impact Assessment Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Sustainability Matters Assessed | {assessed} |\n"
            f"| Material Impacts | {material} |\n"
            f"| Non-Material Impacts | {non_material} |\n"
            f"| Materiality Rate | {self._pct(material, assessed)} |"
        )

    def _md_severity_summary(self, data: Dict[str, Any]) -> str:
        """Render severity score summary table."""
        scores = data.get("severity_scores", [])
        if not scores:
            return "## Severity Score Summary\n\n_No severity scores available._"
        lines = [
            "## Severity Score Summary", "",
            "| Matter | Scale | Scope | Irremediability | Likelihood | Composite |",
            "|--------|-------|-------|-----------------|------------|-----------|",
        ]
        for s in scores[:20]:
            lines.append(
                f"| {s.get('matter_id', '-')} | "
                f"{self._fmt(s.get('scale_score', 0))} | "
                f"{self._fmt(s.get('scope_score', 0))} | "
                f"{self._fmt(s.get('irremediability_score', 0))} | "
                f"{self._fmt(s.get('likelihood_score', 0))} | "
                f"**{self._fmt(s.get('composite_score', 0))}** |"
            )
        return "\n".join(lines)

    def _md_topic_distribution(self, data: Dict[str, Any]) -> str:
        """Render ESRS topic distribution."""
        dist = data.get("topic_distribution", {})
        if not dist:
            return "## Topic Distribution\n\n_No topic data available._"
        lines = [
            "## ESRS Topic Distribution", "",
            "| ESRS Topic | Matter Count |",
            "|------------|-------------|",
        ]
        for topic, count in sorted(dist.items()):
            lines.append(f"| {topic} | {count} |")
        return "\n".join(lines)

    def _md_material_impacts(self, data: Dict[str, Any]) -> str:
        """Render material impacts ranking table."""
        ranked = data.get("ranked_impacts", [])
        material = [r for r in ranked if r.get("is_material", False)]
        if not material:
            return "## Material Impacts\n\n_No material impacts identified._"
        lines = [
            "## Material Impacts (Ranked)", "",
            "| Rank | Matter | ESRS Topic | Impact Type | Score |",
            "|------|--------|------------|-------------|-------|",
        ]
        for r in material:
            lines.append(
                f"| {r.get('rank', '-')} | {r.get('matter_name', '-')} | "
                f"{r.get('esrs_topic', '-')} | {r.get('impact_type', '-')} | "
                f"**{self._fmt(r.get('composite_score', 0))}** |"
            )
        return "\n".join(lines)

    def _md_non_material_impacts(self, data: Dict[str, Any]) -> str:
        """Render non-material impacts summary."""
        ranked = data.get("ranked_impacts", [])
        non_material = [r for r in ranked if not r.get("is_material", False)]
        if not non_material:
            return "## Non-Material Impacts\n\n_All assessed impacts are material._"
        lines = [
            "## Non-Material Impacts", "",
            "| Matter | ESRS Topic | Score | Threshold |",
            "|--------|------------|-------|-----------|",
        ]
        for r in non_material[:10]:
            lines.append(
                f"| {r.get('matter_name', '-')} | {r.get('esrs_topic', '-')} | "
                f"{self._fmt(r.get('composite_score', 0))} | "
                f"{self._fmt(r.get('threshold_applied', 0))} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render methodology notes."""
        return (
            "## Methodology\n\n"
            "This impact materiality assessment follows ESRS 1 Chapter 3 and "
            "EFRAG Implementation Guidance IG-1. Severity scoring uses four "
            "dimensions: scale, scope, irremediability (for all impacts), and "
            "likelihood (for potential impacts only). Composite scores are "
            "calculated as weighted averages. The materiality threshold is "
            "applied to filter material from non-material impacts."
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-015 Double Materiality Pack*"

    # -- HTML sections --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>Impact Materiality Assessment Report</h1>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview."""
        material = data.get("material_impacts", 0)
        total = data.get("matters_assessed", 0)
        return (
            f'<h2>Overview</h2>\n'
            f'<p>Matters Assessed: {total} | Material: {material}</p>'
        )

    def _html_severity_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML severity summary."""
        scores = data.get("severity_scores", [])
        rows = ""
        for s in scores[:20]:
            rows += (
                f'<tr><td>{s.get("matter_id", "-")}</td>'
                f'<td>{self._fmt(s.get("composite_score", 0))}</td></tr>\n'
            )
        return (
            f'<h2>Severity Scores</h2>\n'
            f'<table><tr><th>Matter</th><th>Composite Score</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_material_impacts(self, data: Dict[str, Any]) -> str:
        """Render HTML material impacts."""
        ranked = data.get("ranked_impacts", [])
        material = [r for r in ranked if r.get("is_material", False)]
        rows = ""
        for r in material:
            rows += (
                f'<tr><td>{r.get("rank", "-")}</td>'
                f'<td>{r.get("matter_name", "-")}</td>'
                f'<td>{self._fmt(r.get("composite_score", 0))}</td></tr>\n'
            )
        return (
            f'<h2>Material Impacts</h2>\n'
            f'<table><tr><th>Rank</th><th>Matter</th><th>Score</th></tr>\n'
            f'{rows}</table>'
        )

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

    def _pct(self, part: int, total: int) -> str:
        """Format percentage."""
        if total == 0:
            return "0.0%"
        return f"{part / total * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
