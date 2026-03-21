# -*- coding: utf-8 -*-
"""
ValidationReportTemplate - 42-criterion SBTi validation results for PACK-023.

Renders a comprehensive validation report covering all 42 SBTi criteria:
28 near-term criteria (C1-C28) and 14 net-zero criteria (NZ-C1 to NZ-C14),
with executive summary, pass/fail matrices, gap analysis with remediation
guidance, readiness scoring, and priority actions.

Sections:
    1. Executive Summary (overall pass/fail, score %)
    2. Near-Term Criteria Matrix (C1-C28 pass/fail grid)
    3. Net-Zero Criteria Matrix (NZ-C1 to NZ-C14)
    4. Gap Analysis (items with remediation)
    5. Readiness Score
    6. Priority Actions

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    """Format a value as a Decimal string with fixed decimal places."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 2) -> str:
    """Format a Decimal value with thousands separator."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)


def _pct(val: Any) -> str:
    """Format a value as percentage string."""
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)


def _status_icon(status: str) -> str:
    """Return a text-based status indicator for markdown."""
    s = str(status).upper()
    if s == "PASS":
        return "PASS"
    elif s == "FAIL":
        return "FAIL"
    elif s in ("WARNING", "WARN"):
        return "WARN"
    elif s in ("NA", "N/A"):
        return "N/A"
    return status


class ValidationReportTemplate:
    """
    SBTi 42-criterion validation report template.

    Renders the complete validation assessment for SBTi near-term (C1-C28)
    and net-zero (NZ-C1 to NZ-C14) criteria with pass/fail matrices,
    gap analysis, readiness scoring, and priority actions.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ValidationReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render validation report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_near_term_matrix(data),
            self._md_net_zero_matrix(data),
            self._md_gap_analysis(data),
            self._md_readiness_score(data),
            self._md_priority_actions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render validation report as self-contained HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_near_term_matrix(data),
            self._html_net_zero_matrix(data),
            self._html_gap_analysis(data),
            self._html_readiness_score(data),
            self._html_priority_actions(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>SBTi Validation Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render validation report as structured JSON."""
        self.generated_at = _utcnow()
        near_term = data.get("near_term_criteria", [])
        net_zero = data.get("net_zero_criteria", [])
        gaps = data.get("gaps", [])

        all_criteria = near_term + net_zero
        passed = len([c for c in all_criteria if str(c.get("status", "")).upper() == "PASS"])
        total = len(all_criteria)

        result: Dict[str, Any] = {
            "template": "validation_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "executive_summary": {
                "overall_status": data.get("overall_status", "FAIL"),
                "total_criteria": total,
                "criteria_passed": passed,
                "criteria_failed": total - passed,
                "score_pct": round(passed / total * 100, 1) if total > 0 else 0,
            },
            "near_term_criteria": near_term,
            "net_zero_criteria": net_zero,
            "gaps": gaps,
            "readiness_score": data.get("readiness_score", {}),
            "priority_actions": data.get("priority_actions", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# SBTi Validation Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        near_term = data.get("near_term_criteria", [])
        net_zero = data.get("net_zero_criteria", [])
        all_c = near_term + net_zero
        passed = len([c for c in all_c if str(c.get("status", "")).upper() == "PASS"])
        failed = len([c for c in all_c if str(c.get("status", "")).upper() == "FAIL"])
        warned = len([c for c in all_c if str(c.get("status", "")).upper() in ("WARNING", "WARN")])
        na = len([c for c in all_c if str(c.get("status", "")).upper() in ("NA", "N/A")])
        total = len(all_c)
        score = round(passed / total * 100, 1) if total > 0 else 0
        overall = data.get("overall_status", "FAIL")

        nt_pass = len([c for c in near_term if str(c.get("status", "")).upper() == "PASS"])
        nz_pass = len([c for c in net_zero if str(c.get("status", "")).upper() == "PASS"])

        return (
            f"## 1. Executive Summary\n\n"
            f"**Overall Status:** {overall}  \n"
            f"**Validation Score:** {_dec(score, 1)}%\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Criteria Assessed | {total} |\n"
            f"| Criteria Passed | {passed} |\n"
            f"| Criteria Failed | {failed} |\n"
            f"| Criteria Warnings | {warned} |\n"
            f"| Criteria N/A | {na} |\n"
            f"| Near-Term Pass Rate (C1-C28) | {nt_pass}/{len(near_term)} |\n"
            f"| Net-Zero Pass Rate (NZ-C1 to NZ-C14) | {nz_pass}/{len(net_zero)} |"
        )

    def _md_near_term_matrix(self, data: Dict[str, Any]) -> str:
        criteria = data.get("near_term_criteria", [])
        lines = [
            "## 2. Near-Term Criteria Matrix (C1-C28)\n",
            "| Criterion | Description | Status | Evidence | Notes |",
            "|-----------|-------------|:------:|----------|-------|",
        ]
        for c in criteria:
            lines.append(
                f"| {c.get('id', '-')} | {c.get('description', '-')} "
                f"| {_status_icon(c.get('status', 'N/A'))} "
                f"| {c.get('evidence', '-')} "
                f"| {c.get('notes', '-')} |"
            )
        if not criteria:
            lines.append("| - | _No near-term criteria assessed_ | - | - | - |")

        group_summary = self._criteria_group_summary(criteria)
        if group_summary:
            lines.append("")
            lines.append("### Criteria Group Summary\n")
            lines.append("| Group | Criteria Range | Passed | Total | Status |")
            lines.append("|-------|:--------------:|:------:|:-----:|--------|")
            for g in group_summary:
                lines.append(
                    f"| {g['group']} | {g['range']} "
                    f"| {g['passed']} | {g['total']} | {g['status']} |"
                )
        return "\n".join(lines)

    def _md_net_zero_matrix(self, data: Dict[str, Any]) -> str:
        criteria = data.get("net_zero_criteria", [])
        lines = [
            "## 3. Net-Zero Criteria Matrix (NZ-C1 to NZ-C14)\n",
            "| Criterion | Description | Status | Evidence | Notes |",
            "|-----------|-------------|:------:|----------|-------|",
        ]
        for c in criteria:
            lines.append(
                f"| {c.get('id', '-')} | {c.get('description', '-')} "
                f"| {_status_icon(c.get('status', 'N/A'))} "
                f"| {c.get('evidence', '-')} "
                f"| {c.get('notes', '-')} |"
            )
        if not criteria:
            lines.append("| - | _No net-zero criteria assessed_ | - | - | - |")
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        gaps = data.get("gaps", [])
        lines = [
            "## 4. Gap Analysis\n",
            f"**Total Gaps Identified:** {len(gaps)}\n",
            "| # | Criterion | Gap Description | Severity "
            "| Remediation Action | Effort | Timeline |",
            "|---|-----------|-----------------|:--------:"
            "|-------------------|:------:|:--------:|",
        ]
        for i, g in enumerate(gaps, 1):
            lines.append(
                f"| {i} | {g.get('criterion', '-')} "
                f"| {g.get('description', '-')} "
                f"| {g.get('severity', '-')} "
                f"| {g.get('remediation', '-')} "
                f"| {g.get('effort', '-')} "
                f"| {g.get('timeline', '-')} |"
            )
        if not gaps:
            lines.append(
                "| - | _No gaps identified_ | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_readiness_score(self, data: Dict[str, Any]) -> str:
        rs = data.get("readiness_score", {})
        dimensions = rs.get("dimensions", [])
        lines = [
            "## 5. Readiness Score\n",
            f"**Overall Readiness:** {_pct(rs.get('overall_pct', 0))}  \n"
            f"**Status:** {rs.get('status', 'N/A')}\n",
            "| Dimension | Score (%) | Weight | Weighted Score | Status |",
            "|-----------|:---------:|:------:|:--------------:|--------|",
        ]
        for d in dimensions:
            weighted = (
                float(d.get("score_pct", 0)) * float(d.get("weight", 0)) / 100
            )
            lines.append(
                f"| {d.get('name', '-')} "
                f"| {_pct(d.get('score_pct', 0))} "
                f"| {_dec(d.get('weight', 0), 0)} "
                f"| {_dec(weighted, 1)} "
                f"| {d.get('status', '-')} |"
            )
        if not dimensions:
            lines.append("| _No dimensions assessed_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_priority_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("priority_actions", [])
        lines = [
            "## 6. Priority Actions\n",
            "Actions ranked by impact and urgency.\n",
            "| Priority | Action | Criterion | Owner "
            "| Deadline | Status |",
            "|:--------:|--------|-----------|-------"
            "|:--------:|--------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('criterion', '-')} "
                f"| {a.get('owner', '-')} "
                f"| {a.get('deadline', '-')} "
                f"| {a.get('status', '-')} |"
            )
        if not actions:
            lines.append("| - | _No priority actions defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*Validation per SBTi Corporate Manual V5.3 (C1-C28) and "
            f"Net-Zero Standard V1.3 (NZ-C1 to NZ-C14).*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".badge-pass{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-fail{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-warn{display:inline-block;background:#ff9800;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-na{display:inline-block;background:#9e9e9e;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".severity-high{color:#d32f2f;font-weight:700;}"
            ".severity-medium{color:#f57c00;font-weight:600;}"
            ".severity-low{color:#388e3c;}"
            ".progress-bar{width:100%;height:20px;background:#e0e0e0;border-radius:10px;"
            "overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:10px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_status_badge(self, status: str) -> str:
        """Return an HTML badge for a criterion status."""
        s = str(status).upper()
        if s == "PASS":
            return '<span class="badge-pass">PASS</span>'
        elif s == "FAIL":
            return '<span class="badge-fail">FAIL</span>'
        elif s in ("WARNING", "WARN"):
            return '<span class="badge-warn">WARN</span>'
        elif s in ("NA", "N/A"):
            return '<span class="badge-na">N/A</span>'
        return status

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>SBTi Validation Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        near_term = data.get("near_term_criteria", [])
        net_zero = data.get("net_zero_criteria", [])
        all_c = near_term + net_zero
        passed = len([c for c in all_c if str(c.get("status", "")).upper() == "PASS"])
        failed = len([c for c in all_c if str(c.get("status", "")).upper() == "FAIL"])
        total = len(all_c)
        score = round(passed / total * 100, 1) if total > 0 else 0
        overall = data.get("overall_status", "FAIL")
        bar_color = "#43a047" if score >= 80 else "#ff9800" if score >= 50 else "#ef5350"

        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Overall Status</div>'
            f'<div class="card-value">{overall}</div></div>\n'
            f'  <div class="card"><div class="card-label">Score</div>'
            f'<div class="card-value">{_dec(score, 1)}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Passed</div>'
            f'<div class="card-value">{passed}/{total}</div></div>\n'
            f'  <div class="card"><div class="card-label">Failed</div>'
            f'<div class="card-value">{failed}</div></div>\n'
            f'  <div class="card"><div class="card-label">Near-Term</div>'
            f'<div class="card-value">'
            f'{len([c for c in near_term if str(c.get("status", "")).upper() == "PASS"])}'
            f'/{len(near_term)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Net-Zero</div>'
            f'<div class="card-value">'
            f'{len([c for c in net_zero if str(c.get("status", "")).upper() == "PASS"])}'
            f'/{len(net_zero)}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{score}%;background:{bar_color};"></div>'
            f'</div>'
        )

    def _html_near_term_matrix(self, data: Dict[str, Any]) -> str:
        criteria = data.get("near_term_criteria", [])
        rows = ""
        for c in criteria:
            rows += (
                f'<tr><td><strong>{c.get("id", "-")}</strong></td>'
                f'<td>{c.get("description", "-")}</td>'
                f'<td>{self._html_status_badge(c.get("status", "N/A"))}</td>'
                f'<td>{c.get("evidence", "-")}</td>'
                f'<td>{c.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Near-Term Criteria Matrix (C1-C28)</h2>\n'
            f'<table>\n'
            f'<tr><th>Criterion</th><th>Description</th><th>Status</th>'
            f'<th>Evidence</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_net_zero_matrix(self, data: Dict[str, Any]) -> str:
        criteria = data.get("net_zero_criteria", [])
        rows = ""
        for c in criteria:
            rows += (
                f'<tr><td><strong>{c.get("id", "-")}</strong></td>'
                f'<td>{c.get("description", "-")}</td>'
                f'<td>{self._html_status_badge(c.get("status", "N/A"))}</td>'
                f'<td>{c.get("evidence", "-")}</td>'
                f'<td>{c.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>3. Net-Zero Criteria Matrix (NZ-C1 to NZ-C14)</h2>\n'
            f'<table>\n'
            f'<tr><th>Criterion</th><th>Description</th><th>Status</th>'
            f'<th>Evidence</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        gaps = data.get("gaps", [])
        rows = ""
        for i, g in enumerate(gaps, 1):
            sev = str(g.get("severity", "")).lower()
            sev_cls = (
                "severity-high" if sev == "high"
                else "severity-medium" if sev == "medium"
                else "severity-low"
            )
            rows += (
                f'<tr><td>{i}</td><td><strong>{g.get("criterion", "-")}</strong></td>'
                f'<td>{g.get("description", "-")}</td>'
                f'<td class="{sev_cls}">{g.get("severity", "-")}</td>'
                f'<td>{g.get("remediation", "-")}</td>'
                f'<td>{g.get("effort", "-")}</td>'
                f'<td>{g.get("timeline", "-")}</td></tr>\n'
            )
        return (
            f'<h2>4. Gap Analysis</h2>\n'
            f'<p><strong>Total Gaps:</strong> {len(gaps)}</p>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Criterion</th><th>Gap</th><th>Severity</th>'
            f'<th>Remediation</th><th>Effort</th><th>Timeline</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_readiness_score(self, data: Dict[str, Any]) -> str:
        rs = data.get("readiness_score", {})
        dimensions = rs.get("dimensions", [])
        overall = rs.get("overall_pct", 0)
        bar_color = "#43a047" if overall >= 80 else "#ff9800" if overall >= 50 else "#ef5350"

        rows = ""
        for d in dimensions:
            weighted = (
                float(d.get("score_pct", 0)) * float(d.get("weight", 0)) / 100
            )
            rows += (
                f'<tr><td>{d.get("name", "-")}</td>'
                f'<td>{_pct(d.get("score_pct", 0))}</td>'
                f'<td>{_dec(d.get("weight", 0), 0)}</td>'
                f'<td>{_dec(weighted, 1)}</td>'
                f'<td>{d.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. Readiness Score</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Overall Readiness</div>'
            f'<div class="card-value">{_pct(overall)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value">{rs.get("status", "N/A")}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{overall}%;background:{bar_color};"></div>'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Dimension</th><th>Score</th><th>Weight</th>'
            f'<th>Weighted</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_priority_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("priority_actions", [])
        rows = ""
        for i, a in enumerate(actions, 1):
            rows += (
                f'<tr><td><strong>{i}</strong></td>'
                f'<td>{a.get("action", "-")}</td>'
                f'<td>{a.get("criterion", "-")}</td>'
                f'<td>{a.get("owner", "-")}</td>'
                f'<td>{a.get("deadline", "-")}</td>'
                f'<td>{a.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Priority Actions</h2>\n'
            f'<table>\n'
            f'<tr><th>Priority</th><th>Action</th><th>Criterion</th>'
            f'<th>Owner</th><th>Deadline</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'Validation per SBTi Corporate Manual V5.3 (C1-C28) and '
            f'Net-Zero Standard V1.3 (NZ-C1 to NZ-C14).</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _criteria_group_summary(
        self, criteria: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Compute summary by criteria group (C1-C4, C5-C8, etc.)."""
        groups = [
            ("Boundary & Coverage", "C1-C4", 1, 4),
            ("Base Year & Inventory", "C5-C8", 5, 8),
            ("Target Ambition", "C9-C12", 9, 12),
            ("Scope 2 Methodology", "C13-C16", 13, 16),
            ("Scope 3 Targets", "C17-C20", 17, 20),
            ("Timeframe & Review", "C21-C24", 21, 24),
            ("Reporting & Disclosure", "C25-C28", 25, 28),
        ]
        result = []
        for name, rng, start, end in groups:
            group_criteria = [
                c for c in criteria
                if self._criterion_number(c.get("id", "")) in range(start, end + 1)
            ]
            if group_criteria:
                passed = len([
                    c for c in group_criteria
                    if str(c.get("status", "")).upper() == "PASS"
                ])
                total = len(group_criteria)
                status = "PASS" if passed == total else "FAIL"
                result.append({
                    "group": name,
                    "range": rng,
                    "passed": str(passed),
                    "total": str(total),
                    "status": status,
                })
        return result

    def _criterion_number(self, criterion_id: str) -> int:
        """Extract numeric part from criterion ID like 'C5' or 'NZ-C3'."""
        try:
            clean = criterion_id.replace("NZ-", "").replace("C", "")
            return int(clean)
        except (ValueError, AttributeError):
            return 0

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
