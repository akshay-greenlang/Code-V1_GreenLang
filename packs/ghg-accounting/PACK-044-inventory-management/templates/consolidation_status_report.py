# -*- coding: utf-8 -*-
"""
ConsolidationStatusReport - Entity Consolidation Status for PACK-044.

Generates a consolidation status report covering entity submission
progress, entity-level emission totals, inter-company eliminations,
and consolidated inventory totals.

Sections:
    1. Consolidation Overview
    2. Entity Submission Status
    3. Entity Emission Totals
    4. Inter-Company Eliminations
    5. Consolidated Totals

Author: GreenLang Team
Version: 44.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "44.0.0"


class ConsolidationStatusReport:
    """
    Consolidation status report template for GHG inventory management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ConsolidationStatusReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render consolidation status as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_submission_status(data),
            self._md_entity_totals(data),
            self._md_eliminations(data),
            self._md_consolidated_totals(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render consolidation status as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
            self._html_submission_status(data),
            self._html_entity_totals(data),
            self._html_eliminations(data),
            self._html_consolidated_totals(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render consolidation status as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        return {
            "template": "consolidation_status_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_provenance(data),
            "company_name": self._get_val(data, "company_name", ""),
            "consolidation_approach": data.get("consolidation_approach", ""),
            "entity_submissions": data.get("entity_submissions", []),
            "entity_totals": data.get("entity_totals", []),
            "eliminations": data.get("eliminations", []),
            "consolidated_totals": data.get("consolidated_totals", {}),
        }

    def _md_header(self, data: Dict[str, Any]) -> str:
        company = self._get_val(data, "company_name", "Organization")
        approach = data.get("consolidation_approach", "Operational Control")
        return (
            f"# Consolidation Status Report - {company}\n\n"
            f"**Approach:** {approach} | **Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        entities = data.get("entity_submissions", [])
        total = len(entities)
        complete = sum(1 for e in entities if e.get("status") == "complete")
        return (
            "## 1. Consolidation Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Entities | {total} |\n"
            f"| Complete | {complete} |\n"
            f"| Pending | {total - complete} |\n"
            f"| Completion Rate | {(complete/total*100) if total > 0 else 0:.0f}% |"
        )

    def _md_submission_status(self, data: Dict[str, Any]) -> str:
        entities = data.get("entity_submissions", [])
        if not entities:
            return "## 2. Entity Submission Status\n\nNo entity data."
        lines = [
            "## 2. Entity Submission Status",
            "",
            "| Entity | Ownership % | Scope 1 | Scope 2 | Scope 3 | Status |",
            "|--------|-----------|---------|---------|---------|--------|",
        ]
        for e in entities:
            lines.append(
                f"| {e.get('entity_name', '')} | {e.get('ownership_pct', 100):.0f}% | "
                f"{'Done' if e.get('scope1_complete') else 'Pending'} | "
                f"{'Done' if e.get('scope2_complete') else 'Pending'} | "
                f"{'Done' if e.get('scope3_complete') else 'N/A'} | "
                f"**{e.get('status', 'pending')}** |"
            )
        return "\n".join(lines)

    def _md_entity_totals(self, data: Dict[str, Any]) -> str:
        totals = data.get("entity_totals", [])
        if not totals:
            return ""
        lines = [
            "## 3. Entity Emission Totals",
            "",
            "| Entity | Scope 1 (tCO2e) | Scope 2 (tCO2e) | Total (tCO2e) | % of Group |",
            "|--------|----------------|----------------|--------------|-----------|",
        ]
        grand = sum(t.get("total_tco2e", 0) for t in totals)
        for t in sorted(totals, key=lambda x: x.get("total_tco2e", 0), reverse=True):
            total = t.get("total_tco2e", 0)
            pct = (total / grand * 100) if grand > 0 else 0
            lines.append(
                f"| {t.get('entity_name', '')} | {t.get('scope1_tco2e', 0):,.1f} | "
                f"{t.get('scope2_tco2e', 0):,.1f} | {total:,.1f} | {pct:.1f}% |"
            )
        return "\n".join(lines)

    def _md_eliminations(self, data: Dict[str, Any]) -> str:
        elims = data.get("eliminations", [])
        if not elims:
            return ""
        lines = [
            "## 4. Inter-Company Eliminations",
            "",
            "| From Entity | To Entity | Category | Amount (tCO2e) | Reason |",
            "|------------|----------|----------|---------------|--------|",
        ]
        for e in elims:
            lines.append(
                f"| {e.get('from_entity', '')} | {e.get('to_entity', '')} | "
                f"{e.get('category', '')} | {e.get('amount_tco2e', 0):,.1f} | {e.get('reason', '')} |"
            )
        return "\n".join(lines)

    def _md_consolidated_totals(self, data: Dict[str, Any]) -> str:
        totals = data.get("consolidated_totals", {})
        if not totals:
            return ""
        return (
            "## 5. Consolidated Totals\n\n"
            "| Component | tCO2e |\n|-----------|-------|\n"
            f"| Scope 1 | {totals.get('scope1_tco2e', 0):,.1f} |\n"
            f"| Scope 2 (Location) | {totals.get('scope2_location_tco2e', 0):,.1f} |\n"
            f"| Scope 2 (Market) | {totals.get('scope2_market_tco2e', 0):,.1f} |\n"
            f"| Eliminations | -{totals.get('eliminations_tco2e', 0):,.1f} |\n"
            f"| **Consolidated Total** | **{totals.get('consolidated_total_tco2e', 0):,.1f}** |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f'<meta charset="UTF-8"><title>Consolidation Status - {company}</title>\n'
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;}h2{color:#1b263b;margin-top:2rem;}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}"
            "th{background:#f0f4f8;font-weight:600;}tr:nth-child(even){background:#fafbfc;}"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return f'<div><h1>Consolidation Status &mdash; {company}</h1><hr></div>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        entities = data.get("entity_submissions", [])
        total = len(entities)
        complete = sum(1 for e in entities if e.get("status") == "complete")
        return f'<div><h2>1. Overview</h2><p>Entities: {total} | Complete: {complete}</p></div>'

    def _html_submission_status(self, data: Dict[str, Any]) -> str:
        entities = data.get("entity_submissions", [])
        if not entities:
            return ""
        rows = ""
        for e in entities:
            rows += f"<tr><td>{e.get('entity_name', '')}</td><td>{e.get('ownership_pct', 100):.0f}%</td><td><strong>{e.get('status', 'pending')}</strong></td></tr>\n"
        return (
            '<div><h2>2. Entity Submissions</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Ownership</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_entity_totals(self, data: Dict[str, Any]) -> str:
        totals = data.get("entity_totals", [])
        if not totals:
            return ""
        rows = ""
        for t in totals:
            rows += f"<tr><td>{t.get('entity_name', '')}</td><td>{t.get('scope1_tco2e', 0):,.1f}</td><td>{t.get('scope2_tco2e', 0):,.1f}</td><td>{t.get('total_tco2e', 0):,.1f}</td></tr>\n"
        return (
            '<div><h2>3. Entity Totals</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Scope 1</th><th>Scope 2</th><th>Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_eliminations(self, data: Dict[str, Any]) -> str:
        elims = data.get("eliminations", [])
        if not elims:
            return ""
        rows = ""
        for e in elims:
            rows += f"<tr><td>{e.get('from_entity', '')}</td><td>{e.get('to_entity', '')}</td><td>{e.get('amount_tco2e', 0):,.1f}</td></tr>\n"
        return (
            '<div><h2>4. Eliminations</h2>\n'
            "<table><thead><tr><th>From</th><th>To</th><th>Amount (tCO2e)</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_consolidated_totals(self, data: Dict[str, Any]) -> str:
        totals = data.get("consolidated_totals", {})
        if not totals:
            return ""
        return (
            '<div><h2>5. Consolidated Totals</h2>\n'
            f"<p><strong>Total:</strong> {totals.get('consolidated_total_tco2e', 0):,.1f} tCO2e</p></div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div style="font-size:0.85rem;color:#666;"><hr>'
            f"<p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p>"
            f'<p class="provenance">Provenance Hash: {provenance}</p></div>'
        )
