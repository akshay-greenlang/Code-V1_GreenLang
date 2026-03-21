# -*- coding: utf-8 -*-
"""
CDPGovernanceTemplate - CDP C0-C2 Governance Template for PACK-030.

Renders CDP Climate Change questionnaire modules C0 (Introduction),
C1 (Governance), and C2 (Risks and Opportunities) with structured
responses, board-level governance disclosure, business strategy
integration, and risk/opportunity identification. Multi-format output
(MD, HTML, JSON, PDF) with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  C0 - Introduction & Organization Profile
    3.  C1.1 - Board-Level Governance
    4.  C1.2 - Management-Level Responsibility
    5.  C1.3 - Incentivization & Remuneration
    6.  C2.1 - Climate Risk & Opportunity Process
    7.  C2.2 - Risk Identification & Assessment
    8.  C2.3 - Climate-Related Risks Disclosed
    9.  C2.4 - Climate-Related Opportunities Disclosed
    10. Data Quality Assessment
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

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
_TEMPLATE_ID = "cdp_governance"

_PRIMARY = "#0d3b66"
_SECONDARY = "#1a6b8a"
_ACCENT = "#28a745"
_LIGHT = "#e3f0f7"
_LIGHTER = "#f4f9fc"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

CDP_GOVERNANCE_QUESTIONS = [
    {"id": "c1_1", "code": "C1.1", "question": "Is there board-level oversight of climate-related issues?"},
    {"id": "c1_1a", "code": "C1.1a", "question": "Identify the position(s) of the individual(s) on the board with oversight."},
    {"id": "c1_1b", "code": "C1.1b", "question": "Provide further details on the board oversight."},
    {"id": "c1_1d", "code": "C1.1d", "question": "Does your organization have at least one board member with climate competence?"},
    {"id": "c1_2", "code": "C1.2", "question": "Provide the highest management-level position(s) responsible for climate issues."},
    {"id": "c1_2a", "code": "C1.2a", "question": "Describe where in the organizational structure the position sits."},
    {"id": "c1_3", "code": "C1.3", "question": "Do you provide incentives for the management of climate-related issues?"},
    {"id": "c1_3a", "code": "C1.3a", "question": "Provide further details on the incentives."},
]

CDP_RISK_CATEGORIES = [
    "Current regulation", "Emerging regulation", "Technology",
    "Legal", "Market", "Reputation", "Acute physical", "Chronic physical",
]

CDP_OPPORTUNITY_CATEGORIES = [
    "Resource efficiency", "Energy source", "Products and services",
    "Markets", "Resilience",
]

XBRL_TAGS: Dict[str, str] = {
    "board_oversight": "gl:CDPBoardOversight",
    "management_responsibility": "gl:CDPManagementResponsibility",
    "climate_incentives": "gl:CDPClimateIncentives",
    "risks_identified": "gl:CDPRisksIdentified",
    "opportunities_identified": "gl:CDPOpportunitiesIdentified",
    "governance_score": "gl:CDPGovernanceScore",
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


def _dec_comma(val: Any, places: int = 2) -> str:
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


class CDPGovernanceTemplate:
    """
    CDP C0-C2 Governance template for PACK-030 Net Zero Reporting Pack.

    Generates CDP Climate Change questionnaire responses for modules C0
    (Introduction), C1 (Governance), and C2 (Risks and Opportunities).
    Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = CDPGovernanceTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "cdp_year": 2025,
        ...     "board_oversight": True,
        ...     "board_positions": [{"title": "Board Chair", "name": "J. Smith"}],
        ...     "risks": [{"category": "Emerging regulation", "description": "Carbon tax"}],
        ...     "opportunities": [{"category": "Energy source", "description": "Renewables"}],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render CDP governance report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_c0_introduction(data), self._md_c1_1_board(data),
            self._md_c1_2_management(data), self._md_c1_3_incentives(data),
            self._md_c2_1_process(data), self._md_c2_2_assessment(data),
            self._md_c2_3_risks(data), self._md_c2_4_opportunities(data),
            self._md_data_quality(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render CDP governance report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_c0_introduction(data), self._html_c1_1_board(data),
            self._html_c1_2_management(data), self._html_c1_3_incentives(data),
            self._html_c2_1_process(data), self._html_c2_2_assessment(data),
            self._html_c2_3_risks(data), self._html_c2_4_opportunities(data),
            self._html_data_quality(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>CDP Governance - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = _utcnow()
        risks = data.get("risks", [])
        opportunities = data.get("opportunities", [])
        governance_responses = data.get("governance_responses", {})
        answered = sum(1 for q in CDP_GOVERNANCE_QUESTIONS if q["id"] in governance_responses)
        total = len(CDP_GOVERNANCE_QUESTIONS)
        score = (answered / max(1, total)) * 100

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "cdp_year": data.get("cdp_year", ""),
            "board_oversight": data.get("board_oversight", False),
            "board_positions": data.get("board_positions", []),
            "management_positions": data.get("management_positions", []),
            "incentives_provided": data.get("incentives_provided", False),
            "risks": risks,
            "opportunities": opportunities,
            "governance_score": str(round(score, 1)),
            "questions_answered": answered,
            "questions_total": total,
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf",
            "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"CDP Governance - {data.get('org_name', '')}",
                "author": "GreenLang PACK-030", "framework": "CDP",
            },
        }

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# CDP Climate Change - Governance Report (C0-C2)\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**CDP Year:** {data.get('cdp_year', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 Net Zero Reporting Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        opps = data.get("opportunities", [])
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Board Oversight | {'Yes' if data.get('board_oversight', False) else 'No'} |",
            f"| Board Positions with Climate Oversight | {len(data.get('board_positions', []))} |",
            f"| Management Positions Responsible | {len(data.get('management_positions', []))} |",
            f"| Climate Incentives | {'Yes' if data.get('incentives_provided', False) else 'No'} |",
            f"| Risks Identified | {len(risks)} |",
            f"| Opportunities Identified | {len(opps)} |",
        ]
        return "\n".join(lines)

    def _md_c0_introduction(self, data: Dict[str, Any]) -> str:
        profile = data.get("organization_profile", {})
        lines = [
            "## 2. C0 - Introduction & Organization Profile\n",
            "| Field | Value |", "|-------|-------|",
            f"| Organization Name | {data.get('org_name', '')} |",
            f"| Country | {profile.get('country', '')} |",
            f"| Sector | {profile.get('sector', '')} |",
            f"| Primary Industry | {profile.get('industry', '')} |",
            f"| Number of Employees | {_dec_comma(profile.get('employees', 0), 0)} |",
            f"| Annual Revenue | {profile.get('revenue', '')} |",
            f"| Reporting Period | {profile.get('reporting_period', '')} |",
            f"| Reporting Boundary | {profile.get('boundary', 'Operational control')} |",
        ]
        return "\n".join(lines)

    def _md_c1_1_board(self, data: Dict[str, Any]) -> str:
        positions = data.get("board_positions", [])
        lines = [
            "## 3. C1.1 - Board-Level Governance\n",
            f"**C1.1: Is there board-level oversight of climate-related issues?** "
            f"{'Yes' if data.get('board_oversight', False) else 'No'}\n",
        ]
        if positions:
            lines.extend([
                "### C1.1a - Board Positions with Climate Oversight\n",
                "| # | Position | Individual | Committee | Frequency |",
                "|---|----------|-----------|-----------|-----------|",
            ])
            for i, pos in enumerate(positions, 1):
                lines.append(
                    f"| {i} | {pos.get('title', '')} | {pos.get('name', '')} "
                    f"| {pos.get('committee', '')} | {pos.get('frequency', 'Quarterly')} |"
                )
        board_detail = data.get("board_detail", {})
        if board_detail:
            lines.extend([
                "\n### C1.1b - Board Oversight Details\n",
                f"**Scheduled Agenda Item:** {board_detail.get('scheduled', 'Yes')}  \n"
                f"**Review of Targets:** {board_detail.get('reviews_targets', 'Yes')}  \n"
                f"**Approves Climate Strategy:** {board_detail.get('approves_strategy', 'Yes')}  \n"
                f"**Monitors Progress:** {board_detail.get('monitors_progress', 'Yes')}  \n"
                f"**Receives Reports:** {board_detail.get('receives_reports', 'Quarterly')}  ",
            ])
        competence = data.get("board_competence", {})
        if competence:
            lines.extend([
                "\n### C1.1d - Board Climate Competence\n",
                f"**Has Climate-Competent Member:** {competence.get('has_competent_member', 'Yes')}  \n"
                f"**Competence Criteria:** {competence.get('criteria', 'Climate science, energy transition, regulatory')}  ",
            ])
        return "\n".join(lines)

    def _md_c1_2_management(self, data: Dict[str, Any]) -> str:
        positions = data.get("management_positions", [])
        lines = [
            "## 4. C1.2 - Management-Level Responsibility\n",
            "| # | Position | Reports To | Scope of Responsibility |",
            "|---|----------|-----------|-------------------------|",
        ]
        for i, pos in enumerate(positions, 1):
            lines.append(
                f"| {i} | {pos.get('title', '')} | {pos.get('reports_to', '')} "
                f"| {pos.get('scope', '')} |"
            )
        if not positions:
            lines.append("| - | _No management positions disclosed_ | - | - |")
        return "\n".join(lines)

    def _md_c1_3_incentives(self, data: Dict[str, Any]) -> str:
        incentives = data.get("incentives", [])
        lines = [
            "## 5. C1.3 - Incentivization & Remuneration\n",
            f"**C1.3: Do you provide incentives for climate management?** "
            f"{'Yes' if data.get('incentives_provided', False) else 'No'}\n",
        ]
        if incentives:
            lines.extend([
                "### C1.3a - Incentive Details\n",
                "| # | Who | Type | Activity | Linked to Target |",
                "|---|-----|------|----------|:----------------:|",
            ])
            for i, inc in enumerate(incentives, 1):
                lines.append(
                    f"| {i} | {inc.get('who', '')} | {inc.get('type', '')} "
                    f"| {inc.get('activity', '')} | {inc.get('linked_to_target', 'Yes')} |"
                )
        return "\n".join(lines)

    def _md_c2_1_process(self, data: Dict[str, Any]) -> str:
        process = data.get("risk_process", {})
        lines = [
            "## 6. C2.1 - Climate Risk & Opportunity Process\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Process Exists | {process.get('exists', 'Yes')} |",
            f"| Frequency | {process.get('frequency', 'Annual')} |",
            f"| Time Horizons | {process.get('time_horizons', 'Short (<3yr), Medium (3-10yr), Long (>10yr)')} |",
            f"| Integration with ERM | {process.get('erm_integration', 'Fully integrated')} |",
            f"| Scenario Analysis Used | {process.get('scenario_analysis', 'Yes')} |",
            f"| Scenarios Considered | {process.get('scenarios', '1.5C, 2C, 4C')} |",
        ]
        return "\n".join(lines)

    def _md_c2_2_assessment(self, data: Dict[str, Any]) -> str:
        assessment = data.get("risk_assessment", {})
        lines = [
            "## 7. C2.2 - Risk Identification & Assessment\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Assessment Framework | {assessment.get('framework', 'TCFD-aligned')} |",
            f"| Materiality Threshold | {assessment.get('materiality_threshold', '>1% of revenue')} |",
            f"| Risk Categories | {', '.join(CDP_RISK_CATEGORIES)} |",
            f"| Opportunity Categories | {', '.join(CDP_OPPORTUNITY_CATEGORIES)} |",
            f"| Financial Impact Quantified | {assessment.get('financial_quantified', 'Yes')} |",
        ]
        return "\n".join(lines)

    def _md_c2_3_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 8. C2.3 - Climate-Related Risks Disclosed\n",
            "| # | Category | Type | Description | Time Horizon | Likelihood | Financial Impact | Mitigation |",
            "|---|----------|------|-------------|:------------:|------------|------------------|-----------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('category', '')} | {r.get('type', 'Transition')} "
                f"| {r.get('description', '')} | {r.get('time_horizon', 'Medium')} "
                f"| {r.get('likelihood', 'Likely')} | {r.get('financial_impact', '')} "
                f"| {r.get('mitigation', '')} |"
            )
        if not risks:
            lines.append("| - | _No risks disclosed_ | - | - | - | - | - | - |")
        lines.append(f"\n**Total Risks Identified:** {len(risks)}")
        return "\n".join(lines)

    def _md_c2_4_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("opportunities", [])
        lines = [
            "## 9. C2.4 - Climate-Related Opportunities Disclosed\n",
            "| # | Category | Description | Time Horizon | Financial Impact | Strategy |",
            "|---|----------|-------------|:------------:|------------------|----------|",
        ]
        for i, o in enumerate(opps, 1):
            lines.append(
                f"| {i} | {o.get('category', '')} | {o.get('description', '')} "
                f"| {o.get('time_horizon', 'Medium')} | {o.get('financial_impact', '')} "
                f"| {o.get('strategy', '')} |"
            )
        if not opps:
            lines.append("| - | _No opportunities disclosed_ | - | - | - | - |")
        lines.append(f"\n**Total Opportunities Identified:** {len(opps)}")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        governance_responses = data.get("governance_responses", {})
        answered = sum(1 for q in CDP_GOVERNANCE_QUESTIONS if q["id"] in governance_responses)
        total = len(CDP_GOVERNANCE_QUESTIONS)
        score = (answered / max(1, total)) * 100
        lines = [
            "## 10. Data Quality Assessment\n",
            f"**Governance Questions Answered:** {answered}/{total} ({_dec(score, 1)}%)\n",
            "| # | Code | Question | Status |",
            "|---|------|----------|--------|",
        ]
        for i, q in enumerate(CDP_GOVERNANCE_QUESTIONS, 1):
            status = "Answered" if q["id"] in governance_responses else "Pending"
            lines.append(f"| {i} | {q['code']} | {q['question']} | {status} |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 11. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
        ]
        for key, tag in XBRL_TAGS.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {tag} | - |")
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 12. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Data Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*CDP C0-C2 governance disclosure.*"

    # -- HTML sections --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #b3d4e6;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f0f7fb;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>CDP Climate Change - Governance (C0-C2)</h1>\n<p><strong>{data.get("org_name", "")}</strong> | CDP {data.get("cdp_year", "")} | {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        opps = data.get("opportunities", [])
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Board Oversight</div><div class="card-value">{"Yes" if data.get("board_oversight") else "No"}</div></div>\n'
            f'<div class="card"><div class="card-label">Board Positions</div><div class="card-value">{len(data.get("board_positions", []))}</div></div>\n'
            f'<div class="card"><div class="card-label">Risks</div><div class="card-value">{len(risks)}</div></div>\n'
            f'<div class="card"><div class="card-label">Opportunities</div><div class="card-value">{len(opps)}</div></div>\n'
            f'</div>'
        )

    def _html_c0_introduction(self, data: Dict[str, Any]) -> str:
        p = data.get("organization_profile", {})
        return (
            f'<h2>2. C0 - Introduction</h2>\n<table>\n<tr><th>Field</th><th>Value</th></tr>\n'
            f'<tr><td>Organization</td><td>{data.get("org_name", "")}</td></tr>\n'
            f'<tr><td>Sector</td><td>{p.get("sector", "")}</td></tr>\n'
            f'<tr><td>Country</td><td>{p.get("country", "")}</td></tr>\n'
            f'<tr><td>Employees</td><td>{_dec_comma(p.get("employees", 0), 0)}</td></tr>\n'
            f'</table>'
        )

    def _html_c1_1_board(self, data: Dict[str, Any]) -> str:
        positions = data.get("board_positions", [])
        rows = ""
        for i, pos in enumerate(positions, 1):
            rows += f'<tr><td>{i}</td><td>{pos.get("title", "")}</td><td>{pos.get("name", "")}</td><td>{pos.get("committee", "")}</td></tr>\n'
        return (
            f'<h2>3. C1.1 - Board Governance</h2>\n'
            f'<p><strong>Board oversight:</strong> {"Yes" if data.get("board_oversight") else "No"}</p>\n'
            f'<table>\n<tr><th>#</th><th>Position</th><th>Individual</th><th>Committee</th></tr>\n{rows}</table>'
        )

    def _html_c1_2_management(self, data: Dict[str, Any]) -> str:
        positions = data.get("management_positions", [])
        rows = ""
        for i, pos in enumerate(positions, 1):
            rows += f'<tr><td>{i}</td><td>{pos.get("title", "")}</td><td>{pos.get("reports_to", "")}</td><td>{pos.get("scope", "")}</td></tr>\n'
        return f'<h2>4. C1.2 - Management</h2>\n<table>\n<tr><th>#</th><th>Position</th><th>Reports To</th><th>Scope</th></tr>\n{rows}</table>'

    def _html_c1_3_incentives(self, data: Dict[str, Any]) -> str:
        incentives = data.get("incentives", [])
        rows = ""
        for i, inc in enumerate(incentives, 1):
            rows += f'<tr><td>{i}</td><td>{inc.get("who", "")}</td><td>{inc.get("type", "")}</td><td>{inc.get("activity", "")}</td></tr>\n'
        return f'<h2>5. C1.3 - Incentives</h2>\n<table>\n<tr><th>#</th><th>Who</th><th>Type</th><th>Activity</th></tr>\n{rows}</table>'

    def _html_c2_1_process(self, data: Dict[str, Any]) -> str:
        p = data.get("risk_process", {})
        return (
            f'<h2>6. C2.1 - Risk Process</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Process Exists</td><td>{p.get("exists", "Yes")}</td></tr>\n'
            f'<tr><td>Frequency</td><td>{p.get("frequency", "Annual")}</td></tr>\n'
            f'<tr><td>ERM Integration</td><td>{p.get("erm_integration", "Fully integrated")}</td></tr>\n'
            f'</table>'
        )

    def _html_c2_2_assessment(self, data: Dict[str, Any]) -> str:
        a = data.get("risk_assessment", {})
        return (
            f'<h2>7. C2.2 - Assessment</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Framework</td><td>{a.get("framework", "TCFD-aligned")}</td></tr>\n'
            f'<tr><td>Materiality</td><td>{a.get("materiality_threshold", ">1% revenue")}</td></tr>\n'
            f'</table>'
        )

    def _html_c2_3_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = ""
        for i, r in enumerate(risks, 1):
            rows += f'<tr><td>{i}</td><td>{r.get("category", "")}</td><td>{r.get("description", "")}</td><td>{r.get("likelihood", "")}</td><td>{r.get("financial_impact", "")}</td></tr>\n'
        return f'<h2>8. C2.3 - Risks</h2>\n<table>\n<tr><th>#</th><th>Category</th><th>Description</th><th>Likelihood</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_c2_4_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("opportunities", [])
        rows = ""
        for i, o in enumerate(opps, 1):
            rows += f'<tr><td>{i}</td><td>{o.get("category", "")}</td><td>{o.get("description", "")}</td><td>{o.get("financial_impact", "")}</td></tr>\n'
        return f'<h2>9. C2.4 - Opportunities</h2>\n<table>\n<tr><th>#</th><th>Category</th><th>Description</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        governance_responses = data.get("governance_responses", {})
        answered = sum(1 for q in CDP_GOVERNANCE_QUESTIONS if q["id"] in governance_responses)
        total = len(CDP_GOVERNANCE_QUESTIONS)
        score = (answered / max(1, total)) * 100
        return f'<h2>10. Data Quality</h2>\n<p>Questions answered: {answered}/{total} ({_dec(score, 1)}%)</p>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_", " ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>11. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>12. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - CDP Governance</div>'
