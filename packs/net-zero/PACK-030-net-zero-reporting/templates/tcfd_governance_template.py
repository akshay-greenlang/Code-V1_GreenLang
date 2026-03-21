# -*- coding: utf-8 -*-
"""
TCFDGovernanceTemplate - TCFD Governance Pillar Template for PACK-030.

Renders TCFD Governance pillar disclosure covering board-level oversight
of climate-related risks and opportunities, management's role in
assessing and managing climate issues, committee structures, and
governance effectiveness assessment. Multi-format output (MD, HTML,
JSON, PDF) with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  Board Oversight - Structure & Composition
    3.  Board Oversight - Climate Agenda & Frequency
    4.  Board Oversight - Strategic Decisions
    5.  Management Role - Organizational Structure
    6.  Management Role - Responsibilities & Reporting
    7.  Management Role - Climate Competence
    8.  Governance Effectiveness Assessment
    9.  XBRL Tagging Summary
    10. Audit Trail & Provenance

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "tcfd_governance"

_PRIMARY = "#1a237e"
_SECONDARY = "#283593"
_ACCENT = "#42a5f5"
_LIGHT = "#e8eaf6"
_LIGHTER = "#f5f5ff"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

TCFD_GOVERNANCE_RECOMMENDATIONS = [
    {"id": "gov_a", "recommendation": "Describe the board's oversight of climate-related risks and opportunities",
     "sub_items": ["Board/committee responsibility", "Frequency of briefings", "How board considers climate in strategy"]},
    {"id": "gov_b", "recommendation": "Describe management's role in assessing and managing climate-related risks and opportunities",
     "sub_items": ["Management positions/committees", "Monitoring processes", "Reporting to board"]},
]

XBRL_TAGS: Dict[str, str] = {
    "board_oversight": "gl:TCFDGovBoardOversight",
    "board_committee": "gl:TCFDGovBoardCommittee",
    "board_frequency": "gl:TCFDGovBoardFrequency",
    "management_role": "gl:TCFDGovManagementRole",
    "governance_score": "gl:TCFDGovEffectivenessScore",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


class TCFDGovernanceTemplate:
    """
    TCFD Governance pillar template for PACK-030 Net Zero Reporting Pack.

    Generates TCFD-aligned governance disclosure covering board oversight
    and management's role in climate risk management. Supports MD, HTML,
    JSON, PDF.

    Example:
        >>> template = TCFDGovernanceTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "reporting_year": 2025,
        ...     "board_oversight": True,
        ...     "board_committee": "Sustainability & ESG Committee",
        ...     "board_frequency": "Quarterly",
        ...     "management_positions": [
        ...         {"title": "Chief Sustainability Officer", "reports_to": "CEO"},
        ...     ],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render TCFD governance report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_board_structure(data), self._md_board_agenda(data),
            self._md_board_decisions(data), self._md_mgmt_structure(data),
            self._md_mgmt_responsibilities(data), self._md_mgmt_competence(data),
            self._md_effectiveness(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render TCFD governance report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_board_structure(data), self._html_board_agenda(data),
            self._html_mgmt_structure(data), self._html_mgmt_responsibilities(data),
            self._html_effectiveness(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>TCFD Governance - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = _utcnow()
        effectiveness = self._calculate_effectiveness(data)
        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "framework": "TCFD",
            "pillar": "Governance",
            "board": {
                "oversight": data.get("board_oversight", False),
                "committee": data.get("board_committee", ""),
                "frequency": data.get("board_frequency", ""),
                "members": data.get("board_members", []),
                "climate_competence": data.get("board_competence", {}),
            },
            "management": {
                "positions": data.get("management_positions", []),
                "committees": data.get("management_committees", []),
                "reporting_lines": data.get("reporting_lines", []),
            },
            "effectiveness": effectiveness,
            "strategic_decisions": data.get("strategic_decisions", []),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"TCFD Governance - {data.get('org_name', '')}", "author": "GreenLang PACK-030", "framework": "TCFD"},
        }

    def _calculate_effectiveness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        criteria = {
            "board_oversight_exists": data.get("board_oversight", False),
            "dedicated_committee": bool(data.get("board_committee")),
            "quarterly_or_more": data.get("board_frequency", "") in ["Quarterly", "Monthly", "Bi-monthly"],
            "management_responsibility": len(data.get("management_positions", [])) > 0,
            "climate_competence": bool(data.get("board_competence", {}).get("has_competent_member")),
            "strategic_decisions_linked": len(data.get("strategic_decisions", [])) > 0,
            "reporting_lines_clear": len(data.get("reporting_lines", [])) > 0,
            "incentives_linked": data.get("incentives_linked", False),
        }
        passed = sum(1 for v in criteria.values() if v)
        total = len(criteria)
        return {"criteria": criteria, "passed": passed, "total": total, "score": round(passed / total * 100, 1)}

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# TCFD Governance Pillar Disclosure\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Framework:** TCFD (Task Force on Climate-related Financial Disclosures)  \n"
            f"**Pillar:** Governance  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        eff = self._calculate_effectiveness(data)
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Board Oversight | {'Yes' if data.get('board_oversight') else 'No'} |",
            f"| Dedicated Committee | {data.get('board_committee', 'None')} |",
            f"| Board Climate Briefing Frequency | {data.get('board_frequency', 'N/A')} |",
            f"| Management Positions Responsible | {len(data.get('management_positions', []))} |",
            f"| Management Committees | {len(data.get('management_committees', []))} |",
            f"| Governance Effectiveness Score | {eff['score']}% ({eff['passed']}/{eff['total']}) |",
        ]
        return "\n".join(lines)

    def _md_board_structure(self, data: Dict[str, Any]) -> str:
        members = data.get("board_members", [])
        lines = [
            "## 2. Board Oversight - Structure & Composition\n",
            f"**Board-level oversight exists:** {'Yes' if data.get('board_oversight') else 'No'}  \n"
            f"**Responsible committee:** {data.get('board_committee', 'Full Board')}\n",
        ]
        if members:
            lines.extend([
                "### Board Members with Climate Responsibilities\n",
                "| # | Name | Position | Climate Role | Since |",
                "|---|------|----------|-------------|-------|",
            ])
            for i, m in enumerate(members, 1):
                lines.append(
                    f"| {i} | {m.get('name', '')} | {m.get('position', '')} "
                    f"| {m.get('climate_role', '')} | {m.get('since', '')} |"
                )
        return "\n".join(lines)

    def _md_board_agenda(self, data: Dict[str, Any]) -> str:
        agenda_items = data.get("board_agenda_items", [])
        lines = [
            "## 3. Board Oversight - Climate Agenda & Frequency\n",
            f"**Briefing Frequency:** {data.get('board_frequency', 'Quarterly')}  \n"
            f"**Standing Agenda Item:** {data.get('standing_agenda', 'Yes')}\n",
        ]
        if agenda_items:
            lines.extend([
                "### Climate Agenda Items (Last 12 Months)\n",
                "| # | Date | Topic | Decision | Action |",
                "|---|------|-------|----------|--------|",
            ])
            for i, item in enumerate(agenda_items, 1):
                lines.append(
                    f"| {i} | {item.get('date', '')} | {item.get('topic', '')} "
                    f"| {item.get('decision', '')} | {item.get('action', '')} |"
                )
        return "\n".join(lines)

    def _md_board_decisions(self, data: Dict[str, Any]) -> str:
        decisions = data.get("strategic_decisions", [])
        lines = [
            "## 4. Board Oversight - Strategic Decisions\n",
            "| # | Decision | Climate Linkage | Impact | Date |",
            "|---|----------|----------------|--------|------|",
        ]
        for i, d in enumerate(decisions, 1):
            lines.append(
                f"| {i} | {d.get('decision', '')} | {d.get('climate_linkage', '')} "
                f"| {d.get('impact', '')} | {d.get('date', '')} |"
            )
        if not decisions:
            lines.append("| - | _No strategic decisions disclosed_ | - | - | - |")
        return "\n".join(lines)

    def _md_mgmt_structure(self, data: Dict[str, Any]) -> str:
        positions = data.get("management_positions", [])
        committees = data.get("management_committees", [])
        lines = [
            "## 5. Management Role - Organizational Structure\n",
            "### Key Positions\n",
            "| # | Title | Reports To | Scope | FTE Dedicated |",
            "|---|-------|-----------|-------|:-------------:|",
        ]
        for i, p in enumerate(positions, 1):
            lines.append(
                f"| {i} | {p.get('title', '')} | {p.get('reports_to', '')} "
                f"| {p.get('scope', '')} | {p.get('fte', '1.0')} |"
            )
        if committees:
            lines.extend([
                "\n### Management Committees\n",
                "| # | Committee | Chair | Members | Frequency |",
                "|---|-----------|-------|:-------:|-----------|",
            ])
            for i, c in enumerate(committees, 1):
                lines.append(
                    f"| {i} | {c.get('name', '')} | {c.get('chair', '')} "
                    f"| {c.get('members', 0)} | {c.get('frequency', '')} |"
                )
        return "\n".join(lines)

    def _md_mgmt_responsibilities(self, data: Dict[str, Any]) -> str:
        reporting_lines = data.get("reporting_lines", [])
        lines = [
            "## 6. Management Role - Responsibilities & Reporting\n",
            "| # | From | To | Content | Frequency |",
            "|---|------|-----|---------|-----------|",
        ]
        for i, r in enumerate(reporting_lines, 1):
            lines.append(
                f"| {i} | {r.get('from', '')} | {r.get('to', '')} "
                f"| {r.get('content', '')} | {r.get('frequency', '')} |"
            )
        if not reporting_lines:
            lines.append("| - | _No reporting lines disclosed_ | - | - | - |")
        return "\n".join(lines)

    def _md_mgmt_competence(self, data: Dict[str, Any]) -> str:
        comp = data.get("management_competence", {})
        lines = [
            "## 7. Management Role - Climate Competence\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Training Programs | {comp.get('training', 'Annual climate literacy')} |",
            f"| External Advisors | {comp.get('external_advisors', 'Retained')} |",
            f"| Industry Memberships | {comp.get('memberships', 'WBCSD, CLG')} |",
            f"| Certifications | {comp.get('certifications', '')} |",
        ]
        return "\n".join(lines)

    def _md_effectiveness(self, data: Dict[str, Any]) -> str:
        eff = self._calculate_effectiveness(data)
        lines = [
            "## 8. Governance Effectiveness Assessment\n",
            f"**Overall Score:** {eff['score']}% ({eff['passed']}/{eff['total']} criteria met)\n",
            "| # | Criterion | Status |",
            "|---|-----------|--------|",
        ]
        for i, (criterion, met) in enumerate(eff["criteria"].items(), 1):
            lines.append(f"| {i} | {criterion.replace('_', ' ').title()} | {'MET' if met else 'NOT MET'} |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 9. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
        ]
        tag_values = {
            "board_oversight": "Yes" if data.get("board_oversight") else "No",
            "board_committee": data.get("board_committee", ""),
            "board_frequency": data.get("board_frequency", ""),
            "management_role": str(len(data.get("management_positions", []))),
            "governance_score": str(self._calculate_effectiveness(data)["score"]),
        }
        for key, tag in XBRL_TAGS.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {tag} | {tag_values.get(key, '')} |")
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 10. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Data Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*TCFD Governance pillar disclosure.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c5cae9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f3f4fb;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>TCFD Governance Pillar</h1>\n<p><strong>{data.get("org_name", "")}</strong> | {data.get("reporting_year", "")} | {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        eff = self._calculate_effectiveness(data)
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Board Oversight</div><div class="card-value">{"Yes" if data.get("board_oversight") else "No"}</div></div>\n'
            f'<div class="card"><div class="card-label">Committee</div><div class="card-value">{data.get("board_committee", "N/A")}</div></div>\n'
            f'<div class="card"><div class="card-label">Frequency</div><div class="card-value">{data.get("board_frequency", "N/A")}</div></div>\n'
            f'<div class="card"><div class="card-label">Score</div><div class="card-value">{eff["score"]}%</div></div>\n'
            f'</div>'
        )

    def _html_board_structure(self, data: Dict[str, Any]) -> str:
        members = data.get("board_members", [])
        rows = ""
        for i, m in enumerate(members, 1):
            rows += f'<tr><td>{i}</td><td>{m.get("name", "")}</td><td>{m.get("position", "")}</td><td>{m.get("climate_role", "")}</td></tr>\n'
        return f'<h2>2. Board Structure</h2>\n<table>\n<tr><th>#</th><th>Name</th><th>Position</th><th>Climate Role</th></tr>\n{rows}</table>'

    def _html_board_agenda(self, data: Dict[str, Any]) -> str:
        items = data.get("board_agenda_items", [])
        rows = ""
        for i, item in enumerate(items, 1):
            rows += f'<tr><td>{i}</td><td>{item.get("date", "")}</td><td>{item.get("topic", "")}</td><td>{item.get("decision", "")}</td></tr>\n'
        return f'<h2>3. Board Agenda</h2>\n<table>\n<tr><th>#</th><th>Date</th><th>Topic</th><th>Decision</th></tr>\n{rows}</table>'

    def _html_mgmt_structure(self, data: Dict[str, Any]) -> str:
        positions = data.get("management_positions", [])
        rows = ""
        for i, p in enumerate(positions, 1):
            rows += f'<tr><td>{i}</td><td>{p.get("title", "")}</td><td>{p.get("reports_to", "")}</td><td>{p.get("scope", "")}</td></tr>\n'
        return f'<h2>4. Management Structure</h2>\n<table>\n<tr><th>#</th><th>Title</th><th>Reports To</th><th>Scope</th></tr>\n{rows}</table>'

    def _html_mgmt_responsibilities(self, data: Dict[str, Any]) -> str:
        lines = data.get("reporting_lines", [])
        rows = ""
        for i, r in enumerate(lines, 1):
            rows += f'<tr><td>{i}</td><td>{r.get("from", "")}</td><td>{r.get("to", "")}</td><td>{r.get("frequency", "")}</td></tr>\n'
        return f'<h2>5. Reporting Lines</h2>\n<table>\n<tr><th>#</th><th>From</th><th>To</th><th>Frequency</th></tr>\n{rows}</table>'

    def _html_effectiveness(self, data: Dict[str, Any]) -> str:
        eff = self._calculate_effectiveness(data)
        rows = ""
        for criterion, met in eff["criteria"].items():
            cls = "color:#2e7d32" if met else "color:#c62828"
            rows += f'<tr><td>{criterion.replace("_", " ").title()}</td><td style="{cls}">{"MET" if met else "NOT MET"}</td></tr>\n'
        return f'<h2>6. Effectiveness</h2>\n<p>Score: {eff["score"]}%</p>\n<table>\n<tr><th>Criterion</th><th>Status</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_", " ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>7. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>8. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - TCFD Governance</div>'
