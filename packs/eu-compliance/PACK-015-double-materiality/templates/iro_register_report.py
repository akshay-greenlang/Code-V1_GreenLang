# -*- coding: utf-8 -*-
"""
IRORegisterReportTemplate - IRO register report for PACK-015.

Sections:
    1. IRO Register Overview
    2. Classification Summary
    3. IRO Register Table
    4. Topic Distribution
    5. Value Chain Coverage
    6. Prioritization Summary

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IRORegisterReportTemplate:
    """
    IRO (Impacts, Risks, Opportunities) register report template.

    Renders IRO classification summaries, detailed register tables,
    topic distributions, value chain coverage analysis, and
    prioritization results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IRORegisterReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render IRO register report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overview(data),
            self._md_classification(data),
            self._md_register_table(data),
            self._md_topic_distribution(data),
            self._md_value_chain(data),
            self._md_prioritization(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render IRO register report as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_register_table(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>IRO Register Report</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render IRO register report as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "iro_register_report",
            "version": "15.0.0",
            "generated_at": self.generated_at.isoformat(),
            "iros_identified": data.get("iros_identified", 0),
            "iros_by_type": data.get("iros_by_type", {}),
            "iros_by_topic": data.get("iros_by_topic", {}),
            "prioritized_iros": data.get("prioritized_iros", []),
            "value_chain_stages_mapped": data.get("value_chain_stages_mapped", 0),
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
            f"# IRO Register Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render IRO register overview."""
        total = data.get("iros_identified", 0)
        stages = data.get("value_chain_stages_mapped", 0)
        activities = data.get("activities_mapped", 0)
        return (
            "## IRO Register Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total IROs Identified | {total} |\n"
            f"| Value Chain Stages Mapped | {stages} |\n"
            f"| Activities Mapped | {activities} |"
        )

    def _md_classification(self, data: Dict[str, Any]) -> str:
        """Render IRO classification summary."""
        by_type = data.get("iros_by_type", {})
        if not by_type:
            return "## Classification Summary\n\n_No classification data available._"
        lines = [
            "## IRO Classification Summary", "",
            "| IRO Type | Count |",
            "|----------|-------|",
        ]
        for iro_type, count in sorted(by_type.items()):
            label = iro_type.replace("_", " ").title()
            lines.append(f"| {label} | {count} |")
        return "\n".join(lines)

    def _md_register_table(self, data: Dict[str, Any]) -> str:
        """Render the full IRO register table."""
        iros = data.get("prioritized_iros", [])
        if not iros:
            return "## IRO Register\n\n_No IROs in register._"
        lines = [
            "## IRO Register", "",
            "| Rank | Name | ESRS Topic | Type | Severity | Financial | Composite |",
            "|------|------|------------|------|----------|-----------|-----------|",
        ]
        for iro in iros[:30]:
            lines.append(
                f"| {iro.get('priority_rank', '-')} | "
                f"{iro.get('name', '-')} | "
                f"{iro.get('esrs_topic', '-')} | "
                f"{iro.get('iro_type', '-')} | "
                f"{self._fmt(iro.get('severity_score', 0))} | "
                f"{self._fmt(iro.get('financial_score', 0))} | "
                f"**{self._fmt(iro.get('composite_score', 0))}** |"
            )
        return "\n".join(lines)

    def _md_topic_distribution(self, data: Dict[str, Any]) -> str:
        """Render IRO distribution by ESRS topic."""
        by_topic = data.get("iros_by_topic", {})
        if not by_topic:
            return "## Topic Distribution\n\n_No topic data available._"
        lines = [
            "## IRO Distribution by ESRS Topic", "",
            "| ESRS Topic | IRO Count |",
            "|------------|----------|",
        ]
        for topic, count in sorted(by_topic.items()):
            lines.append(f"| {topic} | {count} |")
        return "\n".join(lines)

    def _md_value_chain(self, data: Dict[str, Any]) -> str:
        """Render value chain coverage."""
        stages = data.get("value_chain_stages_mapped", 0)
        total_stages = 7  # Number of ValueChainStage options
        return (
            "## Value Chain Coverage\n\n"
            f"**Stages Covered:** {stages} of {total_stages}\n\n"
            "Value chain stages: Raw Materials, Manufacturing, Distribution, "
            "Own Operations, Product Use, End of Life, Finance."
        )

    def _md_prioritization(self, data: Dict[str, Any]) -> str:
        """Render prioritization summary."""
        iros = data.get("prioritized_iros", [])
        if not iros:
            return "## Prioritization Summary\n\n_No prioritized IROs._"
        top_5 = iros[:5]
        lines = ["## Top 5 Priority IROs", ""]
        for i, iro in enumerate(top_5, 1):
            lines.append(
                f"{i}. **{iro.get('name', '-')}** "
                f"(Topic: {iro.get('esrs_topic', '-')}, "
                f"Score: {self._fmt(iro.get('composite_score', 0))})"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-015 Double Materiality Pack*"

    # -- HTML sections --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>IRO Register Report</h1>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview."""
        total = data.get("iros_identified", 0)
        return f'<h2>Overview</h2>\n<p>Total IROs: {total}</p>'

    def _html_register_table(self, data: Dict[str, Any]) -> str:
        """Render HTML register table."""
        iros = data.get("prioritized_iros", [])
        rows = ""
        for iro in iros[:20]:
            rows += (
                f'<tr><td>{iro.get("priority_rank", "-")}</td>'
                f'<td>{iro.get("name", "-")}</td>'
                f'<td>{iro.get("esrs_topic", "-")}</td>'
                f'<td>{self._fmt(iro.get("composite_score", 0))}</td></tr>\n'
            )
        return (
            f'<h2>IRO Register</h2>\n'
            f'<table><tr><th>Rank</th><th>Name</th><th>Topic</th><th>Score</th></tr>\n'
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

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
