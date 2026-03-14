# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Board Governance Pack Template
==================================================

Board-level sustainability governance pack template aligned with
ESRS 2 GOV-1 through GOV-5 disclosure requirements. Includes
governance structure, KPI dashboards, risk overviews, compliance
status, target progress, and items requiring board decision.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 2.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class KPIStatus(str, Enum):
    """KPI tracking status."""
    ON_TRACK = "ON_TRACK"
    AT_RISK = "AT_RISK"
    OFF_TRACK = "OFF_TRACK"
    ACHIEVED = "ACHIEVED"


class KPITrend(str, Enum):
    """KPI trend direction."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"


class RiskCategory(str, Enum):
    """Risk category classification."""
    ENVIRONMENTAL = "ENVIRONMENTAL"
    SOCIAL = "SOCIAL"
    GOVERNANCE = "GOVERNANCE"
    REGULATORY = "REGULATORY"
    OPERATIONAL = "OPERATIONAL"


class RiskLikelihood(str, Enum):
    """Risk likelihood."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskImpact(str, Enum):
    """Risk impact severity."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MitigationStatus(str, Enum):
    """Risk mitigation status."""
    ACTIVE = "ACTIVE"
    PLANNED = "PLANNED"
    NOT_STARTED = "NOT_STARTED"
    COMPLETED = "COMPLETED"


class DecisionUrgency(str, Enum):
    """Decision urgency level."""
    IMMEDIATE = "IMMEDIATE"
    NEXT_QUARTER = "NEXT_QUARTER"
    THIS_YEAR = "THIS_YEAR"
    INFORMATIONAL = "INFORMATIONAL"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class GovernanceStructure(BaseModel):
    """Board sustainability governance structure."""
    board_composition: Dict[str, Any] = Field(
        default_factory=dict,
        description="Board composition details (members, diversity, etc.)",
    )
    sustainability_committee: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sustainability committee details",
    )
    delegation_matrix: Dict[str, str] = Field(
        default_factory=dict,
        description="Delegation matrix: responsibility -> delegate",
    )
    meeting_frequency: Optional[str] = Field(
        None, description="Committee meeting frequency"
    )
    last_review_date: Optional[date] = Field(
        None, description="Last governance review date"
    )


class KPIEntry(BaseModel):
    """Sustainability KPI entry."""
    name: str = Field(..., description="KPI name")
    current_value: float = Field(..., description="Current value")
    target_value: float = Field(..., description="Target value")
    unit: str = Field("", description="Value unit")
    trend: KPITrend = Field(KPITrend.STABLE, description="Trend direction")
    status: KPIStatus = Field(KPIStatus.ON_TRACK, description="Tracking status")
    esrs_reference: Optional[str] = Field(None, description="Related ESRS standard")
    yoy_change_pct: Optional[float] = Field(None, description="Year-over-year change %")


class RiskEntry(BaseModel):
    """Risk register entry for board oversight."""
    risk_name: str = Field(..., description="Risk name")
    category: RiskCategory = Field(..., description="Risk category")
    likelihood: RiskLikelihood = Field(..., description="Likelihood")
    impact: RiskImpact = Field(..., description="Impact severity")
    mitigation_status: MitigationStatus = Field(
        MitigationStatus.NOT_STARTED, description="Mitigation status"
    )
    owner: Optional[str] = Field(None, description="Risk owner")
    mitigation_summary: Optional[str] = Field(None, description="Mitigation summary")


class ComplianceStatus(BaseModel):
    """Overall compliance status."""
    overall_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Overall compliance %"
    )
    by_standard: Dict[str, float] = Field(
        default_factory=dict,
        description="Compliance % by ESRS standard",
    )
    key_gaps: List[str] = Field(
        default_factory=list, description="Key compliance gaps"
    )


class TargetProgress(BaseModel):
    """Target progress for board tracking."""
    target_name: str = Field(..., description="Target name")
    baseline: float = Field(..., description="Baseline value")
    target: float = Field(..., description="Target value")
    current: float = Field(..., description="Current value")
    unit: str = Field("", description="Value unit")
    status: KPIStatus = Field(KPIStatus.ON_TRACK, description="Status")
    target_year: int = Field(..., description="Target year")

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        total = self.baseline - self.target
        if total == 0:
            return 100.0
        achieved = self.baseline - self.current
        return min(100.0, max(0.0, (achieved / total) * 100.0))


class DecisionItem(BaseModel):
    """Item requiring board decision."""
    topic: str = Field(..., description="Decision topic")
    context: str = Field(..., description="Background context")
    options: List[str] = Field(
        default_factory=list, description="Available options"
    )
    recommendation: str = Field("", description="Management recommendation")
    urgency: DecisionUrgency = Field(
        DecisionUrgency.NEXT_QUARTER, description="Decision urgency"
    )
    financial_impact: Optional[str] = Field(
        None, description="Financial impact estimate"
    )


class BoardGovernancePackInput(BaseModel):
    """Complete input for board governance pack."""
    organization_name: str = Field(..., description="Organization name")
    board_meeting_date: date = Field(..., description="Board meeting date")
    reporting_year: int = Field(
        default_factory=lambda: date.today().year,
        ge=2020, le=2100,
        description="Reporting year",
    )
    governance_structure: GovernanceStructure = Field(
        default_factory=GovernanceStructure,
        description="Governance structure details",
    )
    gov_disclosures: Dict[str, str] = Field(
        default_factory=dict,
        description="GOV-1 through GOV-5 narrative disclosures",
    )
    sustainability_kpis: List[KPIEntry] = Field(
        default_factory=list, description="Sustainability KPIs"
    )
    risk_overview: List[RiskEntry] = Field(
        default_factory=list, description="Risk register entries"
    )
    compliance_status: ComplianceStatus = Field(
        default_factory=ComplianceStatus,
        description="Compliance status",
    )
    target_progress: List[TargetProgress] = Field(
        default_factory=list, description="Target progress items"
    )
    key_decisions_required: List[DecisionItem] = Field(
        default_factory=list, description="Items requiring board decision"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _fmt_number(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format numeric value."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def _kpi_badge(status: KPIStatus) -> str:
    """Badge for KPI status."""
    return f"[{status.value.replace('_', ' ')}]"


def _urgency_sort(urgency: DecisionUrgency) -> int:
    """Sort key for decision urgency."""
    return {
        DecisionUrgency.IMMEDIATE: 0,
        DecisionUrgency.NEXT_QUARTER: 1,
        DecisionUrgency.THIS_YEAR: 2,
        DecisionUrgency.INFORMATIONAL: 3,
    }.get(urgency, 99)


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class BoardGovernancePackTemplate:
    """Generate board-level sustainability governance pack.

    Sections:
        1. Executive Summary
        2. Governance Structure
        3. ESRS 2 GOV-1 to GOV-5 Disclosures
        4. Sustainability KPI Dashboard
        5. Risk Overview
        6. Compliance Status
        7. Target Progress
        8. Items Requiring Board Decision
        9. Next Steps

    Example:
        >>> template = BoardGovernancePackTemplate()
        >>> data = BoardGovernancePackInput(
        ...     organization_name="Acme", board_meeting_date=date(2025, 6, 15)
        ... )
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "board_governance_pack"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the board governance pack template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC RENDER METHODS
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: BoardGovernancePackInput) -> str:
        """Render as Markdown."""
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_governance_structure(data),
            self._md_gov_disclosures(data),
            self._md_kpi_dashboard(data),
            self._md_risk_overview(data),
            self._md_compliance_status(data),
            self._md_target_progress(data),
            self._md_decisions(data),
            self._md_next_steps(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: BoardGovernancePackInput) -> str:
        """Render as HTML document."""
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_governance_structure(data),
            self._html_gov_disclosures(data),
            self._html_kpi_dashboard(data),
            self._html_risk_overview(data),
            self._html_compliance_status(data),
            self._html_target_progress(data),
            self._html_decisions(data),
            self._html_next_steps(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, data.reporting_year, body)

    def render_json(self, data: BoardGovernancePackInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict."""
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "organization_name": data.organization_name,
            "board_meeting_date": data.board_meeting_date.isoformat(),
            "reporting_year": data.reporting_year,
            "governance_structure": data.governance_structure.model_dump(mode="json"),
            "gov_disclosures": data.gov_disclosures,
            "sustainability_kpis": [
                k.model_dump(mode="json") for k in data.sustainability_kpis
            ],
            "risk_overview": [
                r.model_dump(mode="json") for r in data.risk_overview
            ],
            "compliance_status": data.compliance_status.model_dump(mode="json"),
            "target_progress": [
                {**t.model_dump(mode="json"), "progress_pct": t.progress_pct}
                for t in data.target_progress
            ],
            "key_decisions_required": [
                d.model_dump(mode="json") for d in data.key_decisions_required
            ],
        }

    def _compute_provenance(self, data: BoardGovernancePackInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: BoardGovernancePackInput) -> str:
        return (
            f"# Board Sustainability Governance Pack - {data.organization_name}\n"
            f"**Board Meeting:** {data.board_meeting_date.isoformat()} | "
            f"**Reporting Year:** {data.reporting_year}\n\n"
            "**CONFIDENTIAL - FOR BOARD USE ONLY**\n\n---"
        )

    def _md_executive_summary(self, data: BoardGovernancePackInput) -> str:
        on_track = sum(1 for k in data.sustainability_kpis if k.status == KPIStatus.ON_TRACK)
        at_risk = sum(1 for k in data.sustainability_kpis if k.status == KPIStatus.AT_RISK)
        off_track = sum(1 for k in data.sustainability_kpis if k.status == KPIStatus.OFF_TRACK)
        high_risks = sum(
            1 for r in data.risk_overview
            if r.likelihood == RiskLikelihood.HIGH and r.impact == RiskImpact.HIGH
        )
        decisions = len(data.key_decisions_required)
        immediate = sum(
            1 for d in data.key_decisions_required
            if d.urgency == DecisionUrgency.IMMEDIATE
        )
        lines = [
            "## 1. Executive Summary",
            "",
            f"- **Overall Compliance:** {_fmt_pct(data.compliance_status.overall_pct)}",
            f"- **KPIs On Track:** {on_track} | At Risk: {at_risk} | Off Track: {off_track}",
            f"- **High-Priority Risks:** {high_risks}",
            f"- **Board Decisions Required:** {decisions} ({immediate} immediate)",
        ]
        return "\n".join(lines)

    def _md_governance_structure(self, data: BoardGovernancePackInput) -> str:
        gs = data.governance_structure
        lines = ["## 2. Governance Structure", ""]
        if gs.board_composition:
            lines.append("### Board Composition")
            for k, v in gs.board_composition.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")
        if gs.sustainability_committee:
            lines.append("### Sustainability Committee")
            for k, v in gs.sustainability_committee.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")
        if gs.delegation_matrix:
            lines.extend([
                "### Delegation Matrix",
                "",
                "| Responsibility | Delegate |",
                "|---------------|----------|",
            ])
            for resp, delegate in gs.delegation_matrix.items():
                lines.append(f"| {resp} | {delegate} |")
        if gs.meeting_frequency:
            lines.append(f"\n**Meeting Frequency:** {gs.meeting_frequency}")
        return "\n".join(lines)

    def _md_gov_disclosures(self, data: BoardGovernancePackInput) -> str:
        gov_labels = {
            "GOV-1": "Role of administrative, management and supervisory bodies",
            "GOV-2": "Information provided to and sustainability matters addressed by administrative, management and supervisory bodies",
            "GOV-3": "Integration of sustainability-related performance in incentive schemes",
            "GOV-4": "Statement on due diligence",
            "GOV-5": "Risk management and internal controls over sustainability reporting",
        }
        lines = ["## 3. ESRS 2 Governance Disclosures", ""]
        for gov_id in ["GOV-1", "GOV-2", "GOV-3", "GOV-4", "GOV-5"]:
            label = gov_labels.get(gov_id, gov_id)
            narrative = data.gov_disclosures.get(gov_id, "Not yet documented.")
            lines.extend([
                f"### {gov_id}: {label}",
                "",
                narrative,
                "",
            ])
        return "\n".join(lines)

    def _md_kpi_dashboard(self, data: BoardGovernancePackInput) -> str:
        if not data.sustainability_kpis:
            return "## 4. Sustainability KPI Dashboard\n\nNo KPIs defined."
        lines = [
            "## 4. Sustainability KPI Dashboard",
            "",
            "| KPI | Current | Target | Trend | Status | ESRS |",
            "|-----|---------|--------|-------|--------|------|",
        ]
        for k in data.sustainability_kpis:
            unit = f" {k.unit}" if k.unit else ""
            esrs = k.esrs_reference or "-"
            lines.append(
                f"| {k.name} | {_fmt_number(k.current_value, 1)}{unit} "
                f"| {_fmt_number(k.target_value, 1)}{unit} "
                f"| {k.trend.value} | {_kpi_badge(k.status)} | {esrs} |"
            )
        return "\n".join(lines)

    def _md_risk_overview(self, data: BoardGovernancePackInput) -> str:
        if not data.risk_overview:
            return "## 5. Risk Overview\n\nNo risks registered."
        lines = [
            "## 5. Risk Overview",
            "",
            "| Risk | Category | Likelihood | Impact | Mitigation | Owner |",
            "|------|----------|-----------|--------|------------|-------|",
        ]
        for r in data.risk_overview:
            owner = r.owner or "TBD"
            lines.append(
                f"| {r.risk_name} | {r.category.value} "
                f"| {r.likelihood.value} | {r.impact.value} "
                f"| {r.mitigation_status.value} | {owner} |"
            )
        return "\n".join(lines)

    def _md_compliance_status(self, data: BoardGovernancePackInput) -> str:
        cs = data.compliance_status
        lines = [
            "## 6. Compliance Status",
            "",
            f"**Overall Compliance:** {_fmt_pct(cs.overall_pct)}",
            "",
        ]
        if cs.by_standard:
            lines.extend([
                "| Standard | Compliance |",
                "|----------|-----------|",
            ])
            for std, pct in sorted(cs.by_standard.items()):
                lines.append(f"| {std} | {_fmt_pct(pct)} |")
        if cs.key_gaps:
            lines.extend(["", "**Key Gaps:**"])
            for gap in cs.key_gaps:
                lines.append(f"- {gap}")
        return "\n".join(lines)

    def _md_target_progress(self, data: BoardGovernancePackInput) -> str:
        if not data.target_progress:
            return "## 7. Target Progress\n\nNo targets defined."
        lines = [
            "## 7. Target Progress",
            "",
            "| Target | Current | Target Value | Target Year | Progress | Status |",
            "|--------|---------|-------------|-------------|----------|--------|",
        ]
        for t in data.target_progress:
            unit = f" {t.unit}" if t.unit else ""
            lines.append(
                f"| {t.target_name} | {_fmt_number(t.current, 1)}{unit} "
                f"| {_fmt_number(t.target, 1)}{unit} | {t.target_year} "
                f"| {t.progress_pct:.0f}% | {_kpi_badge(t.status)} |"
            )
        return "\n".join(lines)

    def _md_decisions(self, data: BoardGovernancePackInput) -> str:
        if not data.key_decisions_required:
            return "## 8. Items Requiring Board Decision\n\nNo decisions required."
        sorted_items = sorted(
            data.key_decisions_required,
            key=lambda d: _urgency_sort(d.urgency),
        )
        lines = ["## 8. Items Requiring Board Decision", ""]
        for i, item in enumerate(sorted_items, 1):
            impact = item.financial_impact or "Not quantified"
            lines.extend([
                f"### {i}. {item.topic} [{item.urgency.value}]",
                "",
                f"**Context:** {item.context}",
                "",
                f"**Financial Impact:** {impact}",
                "",
            ])
            if item.options:
                lines.append("**Options:**")
                for j, opt in enumerate(item.options, 1):
                    lines.append(f"  {j}. {opt}")
                lines.append("")
            if item.recommendation:
                lines.append(f"**Recommendation:** {item.recommendation}")
            lines.append("")
        return "\n".join(lines)

    def _md_next_steps(self, data: BoardGovernancePackInput) -> str:
        lines = ["## 9. Next Steps", ""]
        steps = []
        immediate_decisions = [
            d for d in data.key_decisions_required
            if d.urgency == DecisionUrgency.IMMEDIATE
        ]
        if immediate_decisions:
            steps.append(
                f"Resolve {len(immediate_decisions)} immediate decision(s) at this meeting."
            )
        off_track_kpis = [
            k for k in data.sustainability_kpis
            if k.status == KPIStatus.OFF_TRACK
        ]
        if off_track_kpis:
            steps.append(
                f"Review remediation plans for {len(off_track_kpis)} off-track KPI(s)."
            )
        if data.compliance_status.overall_pct < 80:
            steps.append(
                "Accelerate compliance gap closure to meet regulatory deadlines."
            )
        if not steps:
            steps.append(
                "Continue monitoring sustainability governance performance."
            )
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")
        return "\n".join(lines)

    def _md_footer(self, data: BoardGovernancePackInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, org: str, year: int, body: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Board Governance Pack - {org} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1200px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "h3{color:#533483;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".on-track{color:#1a7f37;font-weight:bold;}\n"
            ".at-risk{color:#b08800;font-weight:bold;}\n"
            ".off-track{color:#cf222e;font-weight:bold;}\n"
            ".confidential{background:#fef3c7;border:2px solid #b08800;"
            "padding:0.5rem;border-radius:4px;text-align:center;font-weight:bold;}\n"
            ".metric-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".metric-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".metric-label{font-size:0.85rem;color:#666;}\n"
            ".decision-box{background:#f8f9fa;border-left:4px solid #0f3460;"
            "padding:1rem;margin:1rem 0;border-radius:0 6px 6px 0;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: BoardGovernancePackInput) -> str:
        return (
            '<div class="section">\n'
            f"<h1>Board Sustainability Governance Pack &mdash; {data.organization_name}</h1>\n"
            f"<p><strong>Board Meeting:</strong> {data.board_meeting_date.isoformat()} | "
            f"<strong>Reporting Year:</strong> {data.reporting_year}</p>\n"
            '<p class="confidential">CONFIDENTIAL - FOR BOARD USE ONLY</p>\n'
            "<hr>\n</div>"
        )

    def _html_executive_summary(self, data: BoardGovernancePackInput) -> str:
        on_track = sum(1 for k in data.sustainability_kpis if k.status == KPIStatus.ON_TRACK)
        at_risk = sum(1 for k in data.sustainability_kpis if k.status == KPIStatus.AT_RISK)
        off_track = sum(1 for k in data.sustainability_kpis if k.status == KPIStatus.OFF_TRACK)
        decisions = len(data.key_decisions_required)
        cards = [
            (_fmt_pct(data.compliance_status.overall_pct), "Compliance"),
            (str(on_track), "On Track"),
            (str(at_risk), "At Risk"),
            (str(off_track), "Off Track"),
            (str(decisions), "Decisions"),
        ]
        card_html = "\n".join(
            f'<div class="metric-card"><div class="metric-value">{v}</div>'
            f'<div class="metric-label">{l}</div></div>'
            for v, l in cards
        )
        return (
            '<div class="section">\n<h2>1. Executive Summary</h2>\n'
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_governance_structure(self, data: BoardGovernancePackInput) -> str:
        gs = data.governance_structure
        parts = ['<div class="section">\n<h2>2. Governance Structure</h2>\n']
        if gs.board_composition:
            items = "".join(
                f"<li><strong>{k}:</strong> {v}</li>"
                for k, v in gs.board_composition.items()
            )
            parts.append(f"<h3>Board Composition</h3><ul>{items}</ul>\n")
        if gs.sustainability_committee:
            items = "".join(
                f"<li><strong>{k}:</strong> {v}</li>"
                for k, v in gs.sustainability_committee.items()
            )
            parts.append(f"<h3>Sustainability Committee</h3><ul>{items}</ul>\n")
        if gs.delegation_matrix:
            rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in gs.delegation_matrix.items()
            )
            parts.append(
                "<h3>Delegation Matrix</h3>\n"
                "<table><thead><tr><th>Responsibility</th><th>Delegate</th></tr></thead>\n"
                f"<tbody>{rows}</tbody></table>\n"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_gov_disclosures(self, data: BoardGovernancePackInput) -> str:
        gov_labels = {
            "GOV-1": "Role of administrative, management and supervisory bodies",
            "GOV-2": "Information provided to and sustainability matters addressed",
            "GOV-3": "Integration in incentive schemes",
            "GOV-4": "Statement on due diligence",
            "GOV-5": "Risk management and internal controls",
        }
        parts = ['<div class="section">\n<h2>3. ESRS 2 Governance Disclosures</h2>\n']
        for gov_id in ["GOV-1", "GOV-2", "GOV-3", "GOV-4", "GOV-5"]:
            label = gov_labels.get(gov_id, gov_id)
            narrative = data.gov_disclosures.get(gov_id, "Not yet documented.")
            parts.append(
                f"<h3>{gov_id}: {label}</h3>\n<p>{narrative}</p>\n"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_kpi_dashboard(self, data: BoardGovernancePackInput) -> str:
        if not data.sustainability_kpis:
            return (
                '<div class="section"><h2>4. KPI Dashboard</h2>'
                "<p>No KPIs defined.</p></div>"
            )
        rows = []
        for k in data.sustainability_kpis:
            css_map = {
                KPIStatus.ON_TRACK: "on-track",
                KPIStatus.ACHIEVED: "on-track",
                KPIStatus.AT_RISK: "at-risk",
                KPIStatus.OFF_TRACK: "off-track",
            }
            css = css_map.get(k.status, "")
            unit = f" {k.unit}" if k.unit else ""
            esrs = k.esrs_reference or "-"
            rows.append(
                f"<tr><td>{k.name}</td>"
                f"<td>{_fmt_number(k.current_value, 1)}{unit}</td>"
                f"<td>{_fmt_number(k.target_value, 1)}{unit}</td>"
                f"<td>{k.trend.value}</td>"
                f'<td class="{css}">{k.status.value.replace("_", " ")}</td>'
                f"<td>{esrs}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>4. Sustainability KPI Dashboard</h2>\n'
            "<table><thead><tr><th>KPI</th><th>Current</th><th>Target</th>"
            "<th>Trend</th><th>Status</th><th>ESRS</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_risk_overview(self, data: BoardGovernancePackInput) -> str:
        if not data.risk_overview:
            return (
                '<div class="section"><h2>5. Risk Overview</h2>'
                "<p>No risks registered.</p></div>"
            )
        rows = []
        for r in data.risk_overview:
            owner = r.owner or "TBD"
            rows.append(
                f"<tr><td>{r.risk_name}</td><td>{r.category.value}</td>"
                f"<td>{r.likelihood.value}</td><td>{r.impact.value}</td>"
                f"<td>{r.mitigation_status.value}</td><td>{owner}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>5. Risk Overview</h2>\n'
            "<table><thead><tr><th>Risk</th><th>Category</th>"
            "<th>Likelihood</th><th>Impact</th><th>Mitigation</th>"
            f"<th>Owner</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_compliance_status(self, data: BoardGovernancePackInput) -> str:
        cs = data.compliance_status
        parts = [
            '<div class="section">\n<h2>6. Compliance Status</h2>\n',
            f'<div class="metric-card"><div class="metric-value">'
            f'{_fmt_pct(cs.overall_pct)}</div>'
            f'<div class="metric-label">Overall Compliance</div></div>\n',
        ]
        if cs.by_standard:
            rows = "".join(
                f"<tr><td>{std}</td><td>{_fmt_pct(pct)}</td></tr>"
                for std, pct in sorted(cs.by_standard.items())
            )
            parts.append(
                "<table><thead><tr><th>Standard</th><th>Compliance</th></tr></thead>\n"
                f"<tbody>{rows}</tbody></table>\n"
            )
        if cs.key_gaps:
            items = "".join(f"<li>{g}</li>" for g in cs.key_gaps)
            parts.append(f"<p><strong>Key Gaps:</strong></p><ul>{items}</ul>\n")
        parts.append("</div>")
        return "".join(parts)

    def _html_target_progress(self, data: BoardGovernancePackInput) -> str:
        if not data.target_progress:
            return (
                '<div class="section"><h2>7. Target Progress</h2>'
                "<p>No targets defined.</p></div>"
            )
        rows = []
        for t in data.target_progress:
            css_map = {
                KPIStatus.ON_TRACK: "on-track",
                KPIStatus.ACHIEVED: "on-track",
                KPIStatus.AT_RISK: "at-risk",
                KPIStatus.OFF_TRACK: "off-track",
            }
            css = css_map.get(t.status, "")
            unit = f" {t.unit}" if t.unit else ""
            rows.append(
                f"<tr><td>{t.target_name}</td>"
                f"<td>{_fmt_number(t.current, 1)}{unit}</td>"
                f"<td>{_fmt_number(t.target, 1)}{unit}</td>"
                f"<td>{t.target_year}</td>"
                f"<td>{t.progress_pct:.0f}%</td>"
                f'<td class="{css}">{t.status.value.replace("_", " ")}</td></tr>'
            )
        return (
            '<div class="section">\n<h2>7. Target Progress</h2>\n'
            "<table><thead><tr><th>Target</th><th>Current</th><th>Target Value</th>"
            "<th>Year</th><th>Progress</th><th>Status</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_decisions(self, data: BoardGovernancePackInput) -> str:
        if not data.key_decisions_required:
            return (
                '<div class="section"><h2>8. Board Decisions</h2>'
                "<p>No decisions required.</p></div>"
            )
        sorted_items = sorted(
            data.key_decisions_required,
            key=lambda d: _urgency_sort(d.urgency),
        )
        parts = ['<div class="section">\n<h2>8. Items Requiring Board Decision</h2>\n']
        for i, item in enumerate(sorted_items, 1):
            impact = item.financial_impact or "Not quantified"
            options_html = ""
            if item.options:
                opt_items = "".join(f"<li>{o}</li>" for o in item.options)
                options_html = f"<p><strong>Options:</strong></p><ol>{opt_items}</ol>\n"
            rec_html = ""
            if item.recommendation:
                rec_html = (
                    f"<p><strong>Recommendation:</strong> {item.recommendation}</p>\n"
                )
            parts.append(
                f'<div class="decision-box">\n'
                f"<h3>{i}. {item.topic} [{item.urgency.value}]</h3>\n"
                f"<p>{item.context}</p>\n"
                f"<p><strong>Financial Impact:</strong> {impact}</p>\n"
                f"{options_html}{rec_html}</div>\n"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_next_steps(self, data: BoardGovernancePackInput) -> str:
        steps = []
        immediate = [
            d for d in data.key_decisions_required
            if d.urgency == DecisionUrgency.IMMEDIATE
        ]
        if immediate:
            steps.append(f"Resolve {len(immediate)} immediate decision(s).")
        off_track = [
            k for k in data.sustainability_kpis if k.status == KPIStatus.OFF_TRACK
        ]
        if off_track:
            steps.append(f"Review {len(off_track)} off-track KPI(s).")
        if data.compliance_status.overall_pct < 80:
            steps.append("Accelerate compliance gap closure.")
        if not steps:
            steps.append("Continue governance monitoring.")
        items = "".join(f"<li>{s}</li>" for s in steps)
        return (
            '<div class="section">\n<h2>9. Next Steps</h2>\n'
            f"<ol>{items}</ol>\n</div>"
        )

    def _html_footer(self, data: BoardGovernancePackInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
