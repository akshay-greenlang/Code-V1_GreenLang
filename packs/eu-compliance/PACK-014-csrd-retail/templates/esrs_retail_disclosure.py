# -*- coding: utf-8 -*-
"""
ESRSRetailDisclosureTemplate - ESRS chapter report for PACK-014.

Sections:
    1. ESRS Chapter Structure
    2. E1 Climate Change
    3. E5 Circular Economy
    4. S2 Value Chain Workers
    5. S4 Consumers
    6. G1 Business Conduct
    7. Data Quality Notes
    8. Audit Trail References

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ESRSRetailDisclosureTemplate:
    """
    ESRS retail disclosure chapter template.

    Renders structured ESRS disclosure content organized by topic,
    with chapter-level completeness tracking and audit trail references.
    """

    TOPIC_NAMES: Dict[str, str] = {
        "E1": "Climate Change",
        "E2": "Pollution",
        "E3": "Water and Marine Resources",
        "E4": "Biodiversity and Ecosystems",
        "E5": "Resource Use and Circular Economy",
        "S1": "Own Workforce",
        "S2": "Workers in the Value Chain",
        "S3": "Affected Communities",
        "S4": "Consumers and End-Users",
        "G1": "Business Conduct",
        "ESRS2": "General Disclosures",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSRetailDisclosureTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_chapter_structure(data),
            self._md_topic_e1(data),
            self._md_topic_e5(data),
            self._md_topic_s2(data),
            self._md_topic_s4(data),
            self._md_topic_g1(data),
            self._md_data_quality(data),
            self._md_audit_trail(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure report as HTML."""
        self.generated_at = datetime.utcnow()
        css = (
            "body{font-family:system-ui,sans-serif;padding:20px;}"
            ".report{max-width:1200px;margin:auto;}"
            "h1{color:#0d6efd;}h2{color:#198754;border-bottom:2px solid #198754;padding-bottom:8px;}"
            "h3{color:#495057;}table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px;text-align:left;}th{background:#f8f9fa;}"
            ".completeness{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:600;}"
            ".high{background:#d4edda;color:#155724;}"
            ".medium{background:#fff3cd;color:#856404;}"
            ".low{background:#f8d7da;color:#721c24;}"
        )
        body = f'<h1>ESRS Retail Disclosure Report</h1>\n<p>Overall Completeness: {data.get("completeness_pct", 0):.1f}%</p>'
        chapters = data.get("disclosure_chapters", [])
        for ch in chapters:
            pct = ch.get("completeness_pct", 0)
            cls = "high" if pct >= 80 else "medium" if pct >= 50 else "low"
            body += f'\n<h2>{ch.get("chapter_title", ch.get("topic", ""))}</h2>'
            body += f'\n<span class="completeness {cls}">{pct:.1f}% complete</span>'
        prov = self._provenance(body)
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head>\n<body><div class="report">{body}</div>\n<!-- Provenance: {prov} -->\n</body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESRS disclosure report as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "esrs_retail_disclosure", "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "completeness_pct": data.get("completeness_pct", 0),
            "material_topics": data.get("material_topics", []),
            "disclosure_chapters": data.get("disclosure_chapters", []),
            "audit_trail": data.get("audit_trail", []),
            "total_datapoints": data.get("total_datapoints", 0),
            "collected_datapoints": data.get("collected_datapoints", 0),
        }
        result["provenance_hash"] = self._provenance(json.dumps(result, default=str))
        return result

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "Retail Company")
        year = data.get("reporting_year", "")
        completeness = data.get("completeness_pct", 0)
        return (
            f"# ESRS Disclosure Report - {entity}\n\n"
            f"**Reporting Year:** {year}  \n**Generated:** {ts}  \n"
            f"**Overall Completeness:** {completeness:.1f}%\n\n---"
        )

    def _md_chapter_structure(self, data: Dict[str, Any]) -> str:
        chapters = data.get("disclosure_chapters", [])
        if not chapters:
            return "## Chapter Structure\n\n_No chapters generated._"
        lines = ["## ESRS Chapter Structure", "", "| Topic | Chapter | Completeness | Datapoints |", "|-------|---------|-------------|-----------|"]
        for ch in chapters:
            lines.append(
                f"| {ch.get('topic', '-')} | {ch.get('chapter_title', '-')} | "
                f"{ch.get('completeness_pct', 0):.1f}% | {ch.get('datapoints_used', 0)} |"
            )
        return "\n".join(lines)

    def _md_topic_section(self, data: Dict[str, Any], topic_code: str) -> str:
        """Render a single topic section."""
        chapters = data.get("disclosure_chapters", [])
        chapter = next((c for c in chapters if c.get("topic") == topic_code), None)
        topic_name = self.TOPIC_NAMES.get(topic_code, topic_code)
        if not chapter:
            return f"### ESRS {topic_code} - {topic_name}\n\n_Not material or no data collected._"
        sections = chapter.get("content_sections", [])
        lines = [f"### ESRS {topic_code} - {topic_name}", f"", f"**Completeness:** {chapter.get('completeness_pct', 0):.1f}%", ""]
        if sections:
            lines.extend(["| Disclosure Req | Datapoints | Status |", "|---------------|-----------|--------|"])
            for s in sections:
                lines.append(f"| {s.get('disclosure_requirement', '-')} | {s.get('datapoints_collected', 0)}/{s.get('datapoints_available', 0)} | {s.get('status', '-')} |")
        return "\n".join(lines)

    def _md_topic_e1(self, data: Dict[str, Any]) -> str:
        return self._md_topic_section(data, "E1")

    def _md_topic_e5(self, data: Dict[str, Any]) -> str:
        return self._md_topic_section(data, "E5")

    def _md_topic_s2(self, data: Dict[str, Any]) -> str:
        return self._md_topic_section(data, "S2")

    def _md_topic_s4(self, data: Dict[str, Any]) -> str:
        return self._md_topic_section(data, "S4")

    def _md_topic_g1(self, data: Dict[str, Any]) -> str:
        return self._md_topic_section(data, "G1")

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        total = data.get("total_datapoints", 0)
        collected = data.get("collected_datapoints", 0)
        return (
            "## Data Quality Notes\n\n"
            f"- **Total Datapoints Required:** {total}\n"
            f"- **Datapoints Collected:** {collected}\n"
            f"- **Collection Rate:** {(collected / max(total, 1) * 100):.1f}%"
        )

    def _md_audit_trail(self, data: Dict[str, Any]) -> str:
        audit = data.get("audit_trail", [])
        if not audit:
            return "## Audit Trail References\n\n_No audit trail data._"
        lines = ["## Audit Trail References", "", "| Topic | Evidence Items | Sources | Completeness |", "|-------|---------------|---------|-------------|"]
        for a in audit:
            lines.append(
                f"| {a.get('topic', '-')} | {len(a.get('evidence_items', []))} | "
                f"{len(a.get('data_sources', []))} | {a.get('completeness_pct', 0):.1f}% |"
            )
        return "\n".join(lines)

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
