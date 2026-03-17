# -*- coding: utf-8 -*-
"""
ESRSRetailDisclosureReportTemplate - ESRS disclosure report for PACK-014.

Renders retail-specific ESRS disclosures across E1 Climate, E5 Circular
Economy, S2 Value Chain Workers, and S4 Consumers, with materiality
assessment, datapoint completion, evidence trail, and audit readiness.

Example:
    >>> template = ESRSRetailDisclosureReportTemplate()
    >>> data = {"materiality": [...], "chapters": {...}}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ESRSRetailDisclosureReportTemplate:
    """
    ESRS disclosure report template for retail sector.

    Renders double materiality assessment, datapoint completion status,
    ESRS chapter content (E1/E5/S2/S4), evidence packaging for auditors,
    cross-reference to PACK-001/002/003, and audit readiness score.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RETAIL_ESRS_CHAPTERS: List[str] = [
        "E1 - Climate Change",
        "E5 - Circular Economy",
        "S2 - Workers in the Value Chain",
        "S4 - Consumers and End-users",
    ]

    MATERIALITY_LEVELS: List[str] = ["material", "moderately_material", "not_material"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSRetailDisclosureReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure report as Markdown.

        Args:
            data: Report data with materiality, chapters, datapoints,
                  evidence, audit_score.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_executive_summary(data))
        sections.append(self._md_materiality_assessment(data))
        sections.append(self._md_datapoint_completion(data))
        sections.append(self._md_chapter_summaries(data))
        sections.append(self._md_evidence_trail(data))
        sections.append(self._md_audit_readiness(data))
        sections.append(self._md_cross_references(data))
        sections.append(self._md_provenance(data))

        return "\n\n".join(sections)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure report as HTML.

        Args:
            data: Report data dict.

        Returns:
            Complete HTML string.
        """
        self.generated_at = datetime.utcnow()
        md = self.render_markdown(data)
        company = data.get("company_name", "Retail Company")
        period = data.get("period", "FY2025")

        html_parts: List[str] = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            f"<title>ESRS Retail Disclosure - {company} - {period}</title>",
            '<meta charset="utf-8">',
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1f2937; }",
            "h1 { color: #065f46; border-bottom: 3px solid #065f46; padding-bottom: 8px; }",
            "h2 { color: #047857; margin-top: 32px; }",
            "h3 { color: #059669; }",
            "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
            "th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }",
            "th { background: #f0fdf4; font-weight: 600; }",
            "tr:nth-child(even) { background: #f9fafb; }",
            ".material { color: #059669; font-weight: 700; }",
            ".not-material { color: #6b7280; }",
            ".provenance { background: #f3f4f6; padding: 12px; border-radius: 6px; font-size: 11px; }",
            "</style>",
            "</head>",
            "<body>",
        ]
        for line in md.split("\n"):
            if line.startswith("# "):
                html_parts.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_parts.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("| "):
                html_parts.append(self._md_table_row_to_html(line))
            elif line.startswith("- "):
                html_parts.append(f"<li>{line[2:]}</li>")
            elif line.strip():
                html_parts.append(f"<p>{line}</p>")
        html_parts.extend(["</body>", "</html>"])
        return "\n".join(html_parts)

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure report as JSON.

        Args:
            data: Report data dict.

        Returns:
            Pretty-printed JSON string.
        """
        self.generated_at = datetime.utcnow()
        output = {
            "template": "esrs_retail_disclosure_report",
            "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_hash(data),
            "retail_chapters": self.RETAIL_ESRS_CHAPTERS,
            "data": data,
        }
        return json.dumps(output, indent=2, default=str)

    # ------------------------------------------------------------------
    # Private section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        company = data.get("company_name", "Retail Company")
        period = data.get("period", "FY2025")
        return (
            f"# ESRS Retail Disclosure Report\n\n"
            f"**Company:** {company}  \n"
            f"**Period:** {period}  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 ESRS Retail Disclosure v14.0.0"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        return (
            f"## Executive Summary\n\n"
            f"- **Material Topics:** {data.get('material_topic_count', 0)}\n"
            f"- **Total Datapoints:** {data.get('total_datapoints', 0)}\n"
            f"- **Datapoints Complete:** {data.get('complete_datapoints', 0)} "
            f"({data.get('completion_pct', 0):.0f}%)\n"
            f"- **Evidence Documents:** {data.get('evidence_count', 0)}\n"
            f"- **Audit Readiness Score:** {data.get('audit_readiness_score', 0):.0f}/100"
        )

    def _md_materiality_assessment(self, data: Dict[str, Any]) -> str:
        materiality = data.get("materiality", [])
        if not materiality:
            return "## Materiality Assessment\n\nNo materiality data available."
        lines = [
            "## Double Materiality Assessment\n",
            "| ESRS Topic | Impact Materiality | Financial Materiality | Overall | Status |",
            "|------------|--------------------|-----------------------|---------|--------|",
        ]
        for m in materiality:
            lines.append(
                f"| {m.get('topic', 'N/A')} | {m.get('impact_score', 0):.0f}/5 "
                f"| {m.get('financial_score', 0):.0f}/5 | {m.get('overall', 'N/A')} "
                f"| {m.get('status', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_datapoint_completion(self, data: Dict[str, Any]) -> str:
        datapoints = data.get("datapoint_status", [])
        if not datapoints:
            return "## Datapoint Completion\n\nNo datapoint data available."
        lines = [
            "## Datapoint Completion by Chapter\n",
            "| Chapter | Required | Complete | In Progress | Missing | Completion (%) |",
            "|---------|----------|----------|-------------|---------|----------------|",
        ]
        for d in datapoints:
            total = d.get("required", 1)
            pct = (d.get("complete", 0) / total) * 100
            lines.append(
                f"| {d.get('chapter', 'N/A')} | {d.get('required', 0)} "
                f"| {d.get('complete', 0)} | {d.get('in_progress', 0)} "
                f"| {d.get('missing', 0)} | {pct:.0f}% |"
            )
        return "\n".join(lines)

    def _md_chapter_summaries(self, data: Dict[str, Any]) -> str:
        chapters = data.get("chapters", {})
        if not chapters:
            return "## Chapter Summaries\n\nNo chapter data available."
        lines = ["## ESRS Chapter Summaries\n"]
        for ch_name in self.RETAIL_ESRS_CHAPTERS:
            ch_key = ch_name.split(" - ")[0].lower().replace(" ", "_")
            ch_data = chapters.get(ch_key, {})
            lines.append(f"### {ch_name}\n")
            lines.append(f"- **Disclosures:** {ch_data.get('disclosure_count', 0)}")
            lines.append(f"- **Key Metric:** {ch_data.get('key_metric', 'N/A')}")
            lines.append(f"- **Narrative Status:** {ch_data.get('narrative_status', 'N/A')}")
            lines.append(f"- **Quality Score:** {ch_data.get('quality_score', 0):.0f}/100")
            lines.append("")
        return "\n".join(lines)

    def _md_evidence_trail(self, data: Dict[str, Any]) -> str:
        evidence = data.get("evidence", [])
        if not evidence:
            return "## Evidence Trail\n\nNo evidence data available."
        lines = [
            "## Evidence Trail\n",
            "| Document | Type | Chapter | Datapoints Covered | Upload Date |",
            "|----------|------|---------|--------------------|-----------:|",
        ]
        for e in evidence:
            lines.append(
                f"| {e.get('name', 'N/A')} | {e.get('type', 'N/A')} "
                f"| {e.get('chapter', 'N/A')} | {e.get('datapoints_covered', 0)} "
                f"| {e.get('upload_date', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_audit_readiness(self, data: Dict[str, Any]) -> str:
        audit = data.get("audit_readiness", {})
        if not audit:
            return "## Audit Readiness\n\nNo audit data available."
        return (
            f"## Audit Readiness\n\n"
            f"- **Overall Score:** {audit.get('score', 0):.0f}/100\n"
            f"- **Data Completeness:** {audit.get('completeness', 0):.0f}%\n"
            f"- **Evidence Coverage:** {audit.get('evidence_coverage', 0):.0f}%\n"
            f"- **Methodology Documentation:** {audit.get('methodology', 'N/A')}\n"
            f"- **Internal Review:** {audit.get('internal_review', 'N/A')}\n"
            f"- **Readiness Level:** {audit.get('level', 'N/A')}"
        )

    def _md_cross_references(self, data: Dict[str, Any]) -> str:
        refs = data.get("cross_references", [])
        if not refs:
            return "## Cross-References\n\nNo cross-reference data available."
        lines = [
            "## Cross-References to Base Packs\n",
            "| Source Pack | Chapter | Datapoints Imported | Status |",
            "|------------|---------|---------------------|---------| ",
        ]
        for r in refs:
            lines.append(
                f"| {r.get('pack', 'N/A')} | {r.get('chapter', 'N/A')} "
                f"| {r.get('datapoints', 0)} | {r.get('status', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        h = self._compute_hash(data)
        return (
            f"---\n\n"
            f"**Provenance:** SHA-256 `{h}`  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 ESRSRetailDisclosureReportTemplate v14.0.0"
        )

    @staticmethod
    def _compute_hash(data: Dict[str, Any]) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _md_table_row_to_html(line: str) -> str:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(c.startswith("-") for c in cells):
            return ""
        row = "".join(f"<td>{c}</td>" for c in cells)
        return f"<tr>{row}</tr>"
