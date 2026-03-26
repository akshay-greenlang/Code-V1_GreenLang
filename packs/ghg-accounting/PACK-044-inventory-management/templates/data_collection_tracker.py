# -*- coding: utf-8 -*-
"""
DataCollectionTracker - Data Submission Tracking for PACK-044.

Generates a data collection tracking report covering submission rates
per facility, overdue data submissions, coverage analysis by scope
and category, data freshness metrics, and collection reminders.

Sections:
    1. Collection Summary KPIs
    2. Submission Rates by Facility
    3. Overdue Submissions
    4. Coverage by Scope/Category
    5. Data Freshness

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

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


class DataCollectionTracker:
    """
    Data collection tracking template for GHG inventory management.

    Renders submission rate dashboards, overdue item lists, coverage
    analysis, and data freshness metrics. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = DataCollectionTracker()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DataCollectionTracker."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render data collection tracker as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_collection_summary(data),
            self._md_submission_rates(data),
            self._md_overdue(data),
            self._md_coverage(data),
            self._md_freshness(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render data collection tracker as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_collection_summary(data),
            self._html_submission_rates(data),
            self._html_overdue(data),
            self._html_coverage(data),
            self._html_freshness(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render data collection tracker as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "data_collection_tracker",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "collection_summary": data.get("collection_summary", {}),
            "submission_rates": data.get("submission_rates", []),
            "overdue_submissions": data.get("overdue_submissions", []),
            "coverage": data.get("coverage", []),
            "freshness": data.get("freshness", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Data Collection Tracker - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"
        )

    def _md_collection_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown collection summary."""
        summary = data.get("collection_summary", {})
        if not summary:
            return "## 1. Collection Summary\n\nNo summary data available."
        total = summary.get("total_expected", 0)
        received = summary.get("total_received", 0)
        overdue = summary.get("total_overdue", 0)
        rate = (received / total * 100) if total > 0 else 0.0
        return (
            "## 1. Collection Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Expected | {total} |\n"
            f"| Total Received | {received} |\n"
            f"| Submission Rate | {rate:.1f}% |\n"
            f"| Overdue | {overdue} |\n"
            f"| Pending | {total - received} |"
        )

    def _md_submission_rates(self, data: Dict[str, Any]) -> str:
        """Render Markdown submission rates by facility."""
        rates = data.get("submission_rates", [])
        if not rates:
            return "## 2. Submission Rates by Facility\n\nNo facility data."
        lines = [
            "## 2. Submission Rates by Facility",
            "",
            "| Facility | Expected | Received | Rate | Status |",
            "|----------|---------|----------|------|--------|",
        ]
        for r in sorted(rates, key=lambda x: x.get("rate_pct", 0)):
            fac = r.get("facility_name", "")
            exp = r.get("expected", 0)
            rec = r.get("received", 0)
            rate = r.get("rate_pct", 0.0)
            status = "Complete" if rate >= 100 else ("On Track" if rate >= 75 else "Behind")
            lines.append(f"| {fac} | {exp} | {rec} | {rate:.0f}% | **{status}** |")
        return "\n".join(lines)

    def _md_overdue(self, data: Dict[str, Any]) -> str:
        """Render Markdown overdue submissions."""
        overdue = data.get("overdue_submissions", [])
        if not overdue:
            return "## 3. Overdue Submissions\n\nNo overdue submissions."
        lines = [
            "## 3. Overdue Submissions",
            "",
            "| Facility | Data Type | Due Date | Days Overdue | Contact |",
            "|----------|----------|----------|-------------|---------|",
        ]
        for item in sorted(overdue, key=lambda x: x.get("days_overdue", 0), reverse=True):
            fac = item.get("facility_name", "")
            dtype = item.get("data_type", "")
            due = item.get("due_date", "-")
            days = item.get("days_overdue", 0)
            contact = item.get("contact", "-")
            lines.append(f"| {fac} | {dtype} | {due} | {days} | {contact} |")
        return "\n".join(lines)

    def _md_coverage(self, data: Dict[str, Any]) -> str:
        """Render Markdown coverage by scope/category."""
        coverage = data.get("coverage", [])
        if not coverage:
            return ""
        lines = [
            "## 4. Coverage by Scope/Category",
            "",
            "| Scope | Category | Facilities Covered | Coverage % |",
            "|-------|---------|-------------------|-----------|",
        ]
        for c in coverage:
            scope = c.get("scope", "")
            cat = c.get("category", "")
            fac_count = c.get("facilities_covered", 0)
            pct = c.get("coverage_pct", 0.0)
            lines.append(f"| {scope} | {cat} | {fac_count} | {pct:.0f}% |")
        return "\n".join(lines)

    def _md_freshness(self, data: Dict[str, Any]) -> str:
        """Render Markdown data freshness."""
        freshness = data.get("freshness", [])
        if not freshness:
            return ""
        lines = [
            "## 5. Data Freshness",
            "",
            "| Data Source | Last Updated | Age (Days) | Status |",
            "|-----------|-------------|-----------|--------|",
        ]
        for f in freshness:
            source = f.get("source", "")
            updated = f.get("last_updated", "-")
            age = f.get("age_days", 0)
            status = "Fresh" if age <= 30 else ("Stale" if age <= 90 else "Expired")
            lines.append(f"| {source} | {updated} | {age} | **{status}** |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            f"<title>Data Collection Tracker - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".overdue{background:#fff0f0;border-left:4px solid #e63946;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return f'<div><h1>Data Collection Tracker &mdash; {company}</h1>\n<p><strong>Period:</strong> {period}</p><hr></div>'

    def _html_collection_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML collection summary."""
        summary = data.get("collection_summary", {})
        if not summary:
            return ""
        total = summary.get("total_expected", 0)
        received = summary.get("total_received", 0)
        rate = (received / total * 100) if total > 0 else 0.0
        return (
            '<div><h2>1. Collection Summary</h2>\n'
            f"<p>Expected: {total} | Received: {received} | Rate: {rate:.1f}%</p></div>"
        )

    def _html_submission_rates(self, data: Dict[str, Any]) -> str:
        """Render HTML submission rates."""
        rates = data.get("submission_rates", [])
        if not rates:
            return ""
        rows = ""
        for r in rates:
            fac = r.get("facility_name", "")
            rate = r.get("rate_pct", 0.0)
            rows += f"<tr><td>{fac}</td><td>{r.get('expected', 0)}</td><td>{r.get('received', 0)}</td><td>{rate:.0f}%</td></tr>\n"
        return (
            '<div><h2>2. Submission Rates</h2>\n'
            "<table><thead><tr><th>Facility</th><th>Expected</th><th>Received</th><th>Rate</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_overdue(self, data: Dict[str, Any]) -> str:
        """Render HTML overdue submissions."""
        overdue = data.get("overdue_submissions", [])
        if not overdue:
            return ""
        rows = ""
        for item in overdue:
            rows += (
                f'<tr class="overdue"><td>{item.get("facility_name", "")}</td>'
                f'<td>{item.get("data_type", "")}</td>'
                f'<td>{item.get("due_date", "-")}</td>'
                f'<td>{item.get("days_overdue", 0)}</td></tr>\n'
            )
        return (
            '<div><h2>3. Overdue Submissions</h2>\n'
            "<table><thead><tr><th>Facility</th><th>Type</th><th>Due</th><th>Days Overdue</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_coverage(self, data: Dict[str, Any]) -> str:
        """Render HTML coverage."""
        coverage = data.get("coverage", [])
        if not coverage:
            return ""
        rows = ""
        for c in coverage:
            rows += f"<tr><td>{c.get('scope', '')}</td><td>{c.get('category', '')}</td><td>{c.get('coverage_pct', 0):.0f}%</td></tr>\n"
        return (
            '<div><h2>4. Coverage</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Category</th><th>Coverage</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_freshness(self, data: Dict[str, Any]) -> str:
        """Render HTML data freshness."""
        freshness = data.get("freshness", [])
        if not freshness:
            return ""
        rows = ""
        for f in freshness:
            age = f.get("age_days", 0)
            status = "Fresh" if age <= 30 else ("Stale" if age <= 90 else "Expired")
            rows += f"<tr><td>{f.get('source', '')}</td><td>{f.get('last_updated', '-')}</td><td>{age}</td><td>{status}</td></tr>\n"
        return (
            '<div><h2>5. Data Freshness</h2>\n'
            "<table><thead><tr><th>Source</th><th>Updated</th><th>Age</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div style="font-size:0.85rem;color:#666;"><hr>\n'
            f"<p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p></div>'
        )
