# -*- coding: utf-8 -*-
"""
MaterialityMatrixReportTemplate - Materiality matrix report for PACK-015.

Sections:
    1. Matrix Overview
    2. Quadrant Analysis
    3. Matrix Visualization Data
    4. Material Topics Detail
    5. Year-over-Year Comparison
    6. Threshold Configuration
    7. Methodology Notes

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MaterialityMatrixReportTemplate:
    """
    Double materiality matrix report template.

    Renders matrix visualization data, quadrant analysis, material
    topic details, year-over-year comparison, and threshold configuration
    for the double materiality assessment.
    """

    QUADRANT_LABELS: Dict[str, str] = {
        "double_material": "Double Material (Impact + Financial)",
        "impact_material_only": "Impact Material Only",
        "financial_material_only": "Financial Material Only",
        "not_material": "Not Material",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MaterialityMatrixReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render materiality matrix report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overview(data),
            self._md_quadrant_analysis(data),
            self._md_matrix_data(data),
            self._md_material_topics(data),
            self._md_yoy_comparison(data),
            self._md_thresholds(data),
            self._md_methodology(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render materiality matrix report as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_quadrant_analysis(data),
            self._html_matrix_data(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Double Materiality Matrix Report</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render materiality matrix report as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "materiality_matrix_report",
            "version": "15.0.0",
            "generated_at": self.generated_at.isoformat(),
            "topics_assessed": data.get("topics_assessed", 0),
            "matrix_entries": data.get("matrix_entries", []),
            "quadrant_distribution": data.get("quadrant_distribution", {}),
            "material_topics": data.get("material_topics", []),
            "non_material_topics": data.get("non_material_topics", []),
            "year_over_year_changes": data.get("year_over_year_changes", []),
            "threshold_results": data.get("threshold_results", []),
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
            f"# Double Materiality Matrix Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render matrix overview."""
        assessed = data.get("topics_assessed", 0)
        material = len(data.get("material_topics", []))
        non_material = len(data.get("non_material_topics", []))
        return (
            "## Matrix Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| ESRS Topics Assessed | {assessed} |\n"
            f"| Material Topics | {material} |\n"
            f"| Non-Material Topics | {non_material} |\n"
            f"| Materiality Rate | {self._pct(material, assessed)} |"
        )

    def _md_quadrant_analysis(self, data: Dict[str, Any]) -> str:
        """Render quadrant distribution analysis."""
        dist = data.get("quadrant_distribution", {})
        if not dist:
            return "## Quadrant Analysis\n\n_No quadrant data available._"
        lines = [
            "## Quadrant Analysis", "",
            "| Quadrant | Topics | Description |",
            "|----------|--------|-------------|",
        ]
        for quadrant, count in sorted(dist.items()):
            label = self.QUADRANT_LABELS.get(quadrant, quadrant)
            lines.append(f"| {label} | {count} | - |")
        return "\n".join(lines)

    def _md_matrix_data(self, data: Dict[str, Any]) -> str:
        """Render matrix position data for each topic."""
        entries = data.get("matrix_entries", [])
        if not entries:
            return "## Matrix Data\n\n_No matrix entries available._"
        lines = [
            "## Matrix Position Data", "",
            "| Topic | Impact Score | Financial Score | Quadrant | Material |",
            "|-------|------------|----------------|----------|----------|",
        ]
        for e in entries:
            is_mat = "Yes" if e.get("is_material", False) else "No"
            quadrant_label = e.get("quadrant", "not_material")
            if isinstance(quadrant_label, str):
                quadrant_label = quadrant_label.replace("_", " ").title()
            lines.append(
                f"| {e.get('topic_name', e.get('topic_id', '-'))} | "
                f"{self._fmt(e.get('impact_score', 0))} | "
                f"{self._fmt(e.get('financial_score', 0))} | "
                f"{quadrant_label} | {is_mat} |"
            )
        return "\n".join(lines)

    def _md_material_topics(self, data: Dict[str, Any]) -> str:
        """Render material topics detail."""
        material = data.get("material_topics", [])
        if not material:
            return "## Material Topics\n\n_No material topics identified._"
        lines = ["## Material Topics", ""]
        for i, topic in enumerate(material, 1):
            lines.append(f"{i}. **{topic}**")
        return "\n".join(lines)

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render year-over-year comparison."""
        changes = data.get("year_over_year_changes", [])
        if not changes:
            return "## Year-over-Year Comparison\n\n_No prior year data available for comparison._"
        lines = [
            "## Year-over-Year Comparison", "",
            "| Topic | Impact Delta | Financial Delta | Prior Quadrant | Current Quadrant | Changed |",
            "|-------|-------------|----------------|---------------|-----------------|---------|",
        ]
        for c in changes:
            changed = "Yes" if c.get("quadrant_changed", False) else "No"
            lines.append(
                f"| {c.get('topic_name', c.get('topic_id', '-'))} | "
                f"{c.get('impact_delta', 0):+.2f} | "
                f"{c.get('financial_delta', 0):+.2f} | "
                f"{c.get('prior_quadrant', '-')} | "
                f"{c.get('current_quadrant', '-')} | {changed} |"
            )
        return "\n".join(lines)

    def _md_thresholds(self, data: Dict[str, Any]) -> str:
        """Render threshold configuration."""
        thresholds = data.get("threshold_results", [])
        if not thresholds:
            return "## Threshold Configuration\n\n_No threshold data available._"
        lines = [
            "## Threshold Application Results", "",
            "| Topic | Impact Threshold | Financial Threshold | Impact Passes | Financial Passes | Result |",
            "|-------|-----------------|--------------------|--------------|--------------------|--------|",
        ]
        for t in thresholds:
            lines.append(
                f"| {t.get('topic_name', t.get('topic_id', '-'))} | "
                f"{self._fmt(t.get('impact_threshold', 0))} | "
                f"{self._fmt(t.get('financial_threshold', 0))} | "
                f"{'Yes' if t.get('impact_passes') else 'No'} | "
                f"{'Yes' if t.get('financial_passes') else 'No'} | "
                f"{t.get('final_materiality', '-')} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render methodology notes."""
        return (
            "## Methodology\n\n"
            "The double materiality matrix plots ESRS topics on two axes: "
            "impact materiality (inside-out, Y-axis) and financial materiality "
            "(outside-in, X-axis). Topics are classified into four quadrants "
            "based on configurable thresholds. Sector-specific adjustments "
            "can lower or raise thresholds for individual topics."
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-015 Double Materiality Pack*"

    # -- HTML sections --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>Double Materiality Matrix Report</h1>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview."""
        material = len(data.get("material_topics", []))
        total = data.get("topics_assessed", 0)
        return f'<h2>Overview</h2>\n<p>Material: {material}/{total}</p>'

    def _html_quadrant_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML quadrant analysis."""
        return '<h2>Quadrant Analysis</h2>'

    def _html_matrix_data(self, data: Dict[str, Any]) -> str:
        """Render HTML matrix data."""
        entries = data.get("matrix_entries", [])
        rows = ""
        for e in entries:
            rows += (
                f'<tr><td>{e.get("topic_name", "-")}</td>'
                f'<td>{self._fmt(e.get("impact_score", 0))}</td>'
                f'<td>{self._fmt(e.get("financial_score", 0))}</td></tr>\n'
            )
        return (
            f'<h2>Matrix Data</h2>\n'
            f'<table><tr><th>Topic</th><th>Impact</th><th>Financial</th></tr>\n'
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
