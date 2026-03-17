"""
AuditReadinessScorecard - CBAM audit readiness assessment template.

This module implements the audit readiness scorecard for PACK-005 CBAM Complete.
It generates reports assessing audit readiness across 7 categories with an overall
score, evidence completeness tracking, unresolved findings, corrective action
progress, verifier engagement status, NCA correspondence, penalty exposure
estimates, and prioritized action items.

Example:
    >>> template = AuditReadinessScorecard()
    >>> data = AuditReadinessData(
    ...     overall_score=OverallScore(score=78, grade="B", ...),
    ...     category_scores=[CategoryScore(...)],
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class OverallScore(BaseModel):
    """Overall audit readiness score."""

    score: int = Field(0, ge=0, le=100, description="Overall readiness score 0-100")
    grade: str = Field("F", description="Grade: A (90+), B (80+), C (70+), D (60+), F (<60)")
    trend: str = Field("stable", description="Trend: improving, stable, declining")
    previous_score: Optional[int] = Field(None, ge=0, le=100, description="Previous assessment score")
    assessment_date: str = Field("", description="Assessment date")
    next_assessment_date: str = Field("", description="Next scheduled assessment")


class CategoryScore(BaseModel):
    """Score for a specific readiness category."""

    category: str = Field("", description="Category name")
    category_key: str = Field("", description="Category key identifier")
    score: int = Field(0, ge=0, le=100, description="Category score 0-100")
    weight: float = Field(0.0, ge=0.0, le=1.0, description="Weight in overall score")
    status: str = Field("adequate", description="excellent, adequate, needs_improvement, critical")
    findings_count: int = Field(0, ge=0, description="Number of open findings")
    notes: str = Field("", description="Assessment notes")


class EvidenceItem(BaseModel):
    """Evidence completeness tracking item."""

    evidence_id: str = Field("", description="Evidence identifier")
    category: str = Field("", description="Evidence category")
    description: str = Field("", description="Evidence description")
    required: bool = Field(True, description="Whether required for compliance")
    available: bool = Field(False, description="Whether currently available")
    quality: str = Field("pending", description="good, acceptable, poor, pending")
    last_updated: str = Field("", description="Last update date")
    owner: str = Field("", description="Responsible person/team")


class AuditFinding(BaseModel):
    """Unresolved audit finding."""

    finding_id: str = Field("", description="Finding identifier")
    category: str = Field("", description="Finding category")
    title: str = Field("", description="Finding title")
    severity: str = Field("low", description="low, medium, high, critical")
    description: str = Field("", description="Finding description")
    identified_date: str = Field("", description="Date identified")
    age_days: int = Field(0, ge=0, description="Age in days")
    assigned_owner: str = Field("", description="Assigned owner")
    deadline: str = Field("", description="Resolution deadline")
    status: str = Field("open", description="open, in_progress, overdue, resolved")


class CorrectiveAction(BaseModel):
    """Corrective action plan item."""

    action_id: str = Field("", description="Action identifier")
    finding_id: str = Field("", description="Related finding ID")
    description: str = Field("", description="Action description")
    assigned_to: str = Field("", description="Assigned person/team")
    due_date: str = Field("", description="Due date")
    status: str = Field("pending", description="pending, in_progress, completed, overdue")
    completion_pct: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")


class VerifierEngagement(BaseModel):
    """Verification engagement status."""

    verifier_name: str = Field("", description="Verifier organization name")
    verifier_accreditation: str = Field("", description="Accreditation details")
    engagement_status: str = Field("not_started", description="not_started, planning, active, completed")
    engagement_start_date: str = Field("", description="Engagement start date")
    next_verification_date: str = Field("", description="Next verification date")
    scope_of_verification: str = Field("", description="Scope description")
    estimated_cost_eur: float = Field(0.0, ge=0.0, description="Estimated verification cost")


class NCACorrespondence(BaseModel):
    """NCA correspondence record."""

    correspondence_id: str = Field("", description="Correspondence identifier")
    nca_name: str = Field("", description="NCA name")
    subject: str = Field("", description="Subject matter")
    direction: str = Field("inbound", description="inbound or outbound")
    date: str = Field("", description="Correspondence date")
    response_deadline: str = Field("", description="Response deadline if applicable")
    status: str = Field("open", description="open, responded, closed, overdue")
    pending_items: str = Field("", description="Pending items description")


class PenaltyExposure(BaseModel):
    """Penalty exposure assessment."""

    total_exposure_eur: float = Field(0.0, ge=0.0, description="Total penalty exposure")
    by_gap: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-gap: gap_description, penalty_basis, estimated_amount_eur, probability"
    )
    mitigating_factors: List[str] = Field(default_factory=list, description="Mitigating factors")
    aggravating_factors: List[str] = Field(default_factory=list, description="Aggravating factors")


class ActionItem(BaseModel):
    """Prioritized action item to improve readiness."""

    priority: int = Field(0, ge=1, description="Priority rank")
    action: str = Field("", description="Action description")
    category: str = Field("", description="Related category")
    impact_on_score: int = Field(0, ge=0, description="Expected score improvement")
    effort: str = Field("low", description="low, medium, high")
    deadline: str = Field("", description="Recommended deadline")
    assigned_to: str = Field("", description="Recommended assignee")


class AuditReadinessData(BaseModel):
    """Complete input data for audit readiness scorecard."""

    overall_score: OverallScore = Field(default_factory=OverallScore)
    category_scores: List[CategoryScore] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    audit_findings: List[AuditFinding] = Field(default_factory=list)
    corrective_actions: List[CorrectiveAction] = Field(default_factory=list)
    verifier_engagement: VerifierEngagement = Field(default_factory=VerifierEngagement)
    nca_correspondence: List[NCACorrespondence] = Field(default_factory=list)
    penalty_exposure: PenaltyExposure = Field(default_factory=PenaltyExposure)
    action_items: List[ActionItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class AuditReadinessScorecard:
    """
    CBAM audit readiness scorecard template.

    Generates audit readiness scorecards with overall and category-level scores,
    evidence completeness, open findings, corrective actions, verifier engagement
    status, NCA correspondence tracking, penalty exposure, and action items.

    Attributes:
        config: Optional configuration dictionary.
        pack_id: Pack identifier (PACK-005).
        template_name: Template name for metadata.
        version: Template version.

    Example:
        >>> template = AuditReadinessScorecard()
        >>> md = template.render_markdown(data)
        >>> assert "Overall Readiness Score" in md
    """

    PACK_ID = "PACK-005"
    TEMPLATE_NAME = "audit_readiness_scorecard"
    VERSION = "1.0"

    GRADE_THRESHOLDS = [
        (90, "A", "#2ecc71"),
        (80, "B", "#27ae60"),
        (70, "C", "#f1c40f"),
        (60, "D", "#f39c12"),
        (0, "F", "#e74c3c"),
    ]

    STATUS_LABELS = {
        "excellent": ("Excellent", "#2ecc71"),
        "adequate": ("Adequate", "#27ae60"),
        "needs_improvement": ("Needs Improvement", "#f39c12"),
        "critical": ("Critical", "#e74c3c"),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize AuditReadinessScorecard.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - currency (str): Currency code (default: EUR).
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the audit readiness scorecard as Markdown.

        Args:
            data: Report data dictionary matching AuditReadinessData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header())
        sections.append(self._md_overall_score(data))
        sections.append(self._md_category_scores(data))
        sections.append(self._md_evidence_completeness(data))
        sections.append(self._md_unresolved_findings(data))
        sections.append(self._md_corrective_actions(data))
        sections.append(self._md_verifier_engagement(data))
        sections.append(self._md_nca_correspondence(data))
        sections.append(self._md_penalty_exposure(data))
        sections.append(self._md_action_items(data))

        content = "\n\n".join(sections)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the audit readiness scorecard as self-contained HTML.

        Args:
            data: Report data dictionary matching AuditReadinessData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_header())
        sections.append(self._html_overall_score(data))
        sections.append(self._html_category_scores(data))
        sections.append(self._html_evidence_completeness(data))
        sections.append(self._html_unresolved_findings(data))
        sections.append(self._html_corrective_actions(data))
        sections.append(self._html_verifier_engagement(data))
        sections.append(self._html_nca_correspondence(data))
        sections.append(self._html_penalty_exposure(data))
        sections.append(self._html_action_items(data))

        body = "\n".join(sections)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="CBAM Audit Readiness Scorecard",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the audit readiness scorecard as structured JSON.

        Args:
            data: Report data dictionary matching AuditReadinessData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_audit_readiness_scorecard",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "overall_score": self._json_overall_score(data),
            "category_scores": self._json_category_scores(data),
            "evidence_completeness": self._json_evidence(data),
            "audit_findings": self._json_findings(data),
            "corrective_actions": self._json_corrective_actions(data),
            "verifier_engagement": self._json_verifier(data),
            "nca_correspondence": self._json_nca(data),
            "penalty_exposure": self._json_penalty(data),
            "action_items": self._json_action_items(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown report header."""
        return (
            "# CBAM Audit Readiness Scorecard\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_overall_score(self, data: Dict[str, Any]) -> str:
        """Build Markdown overall readiness score section."""
        os_data = data.get("overall_score", {})
        score = os_data.get("score", 0)
        grade = os_data.get("grade", "F")
        trend = os_data.get("trend", "stable")
        prev = os_data.get("previous_score")

        trend_arrow = {"improving": "^", "stable": "=", "declining": "v"}.get(trend, "=")
        prev_str = f" (previous: {prev})" if prev is not None else ""

        return (
            "## 1. Overall Readiness Score\n\n"
            f"```\n"
            f"  Score: {score}/100  |  Grade: {grade}  |  Trend: {trend_arrow} {trend}\n"
            f"```\n\n"
            f"**Score:** {score}/100{prev_str}\n\n"
            f"**Grade:** {grade}\n\n"
            f"**Trend:** {trend.capitalize()}\n\n"
            f"**Assessment Date:** {os_data.get('assessment_date', 'N/A')}\n\n"
            f"**Next Assessment:** {os_data.get('next_assessment_date', 'N/A')}"
        )

    def _md_category_scores(self, data: Dict[str, Any]) -> str:
        """Build Markdown category scores section."""
        categories = data.get("category_scores", [])

        header = (
            "## 2. Category Scores\n\n"
            "| Category | Score | Weight | Status | Findings | Notes |\n"
            "|----------|-------|--------|--------|----------|-------|\n"
        )

        rows: List[str] = []
        for c in categories:
            score = c.get("score", 0)
            status = c.get("status", "adequate")
            status_label = self.STATUS_LABELS.get(status, ("Unknown", "#95a5a6"))[0]
            bar = self._md_bar(score)
            notes = c.get("notes", "") or "-"
            rows.append(
                f"| {c.get('category', '')} | "
                f"{score}/100 {bar} | "
                f"{c.get('weight', 0.0):.0%} | "
                f"{status_label} | "
                f"{c.get('findings_count', 0)} | "
                f"{notes} |"
            )

        if not rows:
            return header + "| *No categories scored* | | | | | |"

        return header + "\n".join(rows)

    def _md_evidence_completeness(self, data: Dict[str, Any]) -> str:
        """Build Markdown evidence completeness section."""
        items = data.get("evidence_items", [])

        total = len(items)
        available = sum(1 for i in items if i.get("available", False))
        required = sum(1 for i in items if i.get("required", True))
        required_available = sum(
            1 for i in items if i.get("required", True) and i.get("available", False)
        )
        pct = (required_available / max(required, 1)) * 100

        summary = (
            "## 3. Evidence Completeness\n\n"
            f"**Required Evidence:** {required_available}/{required} available "
            f"({pct:.1f}%)\n\n"
            f"**Total Evidence:** {available}/{total} available\n\n"
        )

        # Missing required items
        missing = [i for i in items if i.get("required", True) and not i.get("available", False)]
        if missing:
            summary += "### Missing Required Evidence\n\n"
            summary += "| ID | Category | Description | Owner |\n"
            summary += "|----|----------|-------------|-------|\n"
            for m in missing:
                summary += (
                    f"| {m.get('evidence_id', '')} | "
                    f"{m.get('category', '')} | "
                    f"{m.get('description', '')} | "
                    f"{m.get('owner', 'Unassigned')} |\n"
                )
        else:
            summary += "*All required evidence is available.*"

        return summary

    def _md_unresolved_findings(self, data: Dict[str, Any]) -> str:
        """Build Markdown unresolved findings section."""
        findings = data.get("audit_findings", [])
        open_findings = [f for f in findings if f.get("status") != "resolved"]

        header = (
            "## 4. Unresolved Findings\n\n"
            f"**Open Findings:** {len(open_findings)}\n\n"
        )

        if not open_findings:
            return header + "*No unresolved findings.*"

        table = (
            "| ID | Title | Severity | Age (days) | Owner | Deadline | Status |\n"
            "|----|-------|----------|------------|-------|----------|--------|\n"
        )

        rows: List[str] = []
        for f in sorted(open_findings, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.get("severity", "low"), 4)):
            status = f.get("status", "open")
            severity = f.get("severity", "low").upper()
            status_upper = status.upper()
            rows.append(
                f"| {f.get('finding_id', '')} | "
                f"{f.get('title', '')} | "
                f"**{severity}** | "
                f"{f.get('age_days', 0)} | "
                f"{f.get('assigned_owner', 'Unassigned')} | "
                f"{self._fmt_date(f.get('deadline', ''))} | "
                f"{status_upper} |"
            )

        critical = sum(1 for f in open_findings if f.get("severity") == "critical")
        high = sum(1 for f in open_findings if f.get("severity") == "high")
        overdue = sum(1 for f in open_findings if f.get("status") == "overdue")

        summary = f"\n\n**Critical: {critical} | High: {high} | Overdue: {overdue}**"

        return header + table + "\n".join(rows) + summary

    def _md_corrective_actions(self, data: Dict[str, Any]) -> str:
        """Build Markdown corrective action status section."""
        actions = data.get("corrective_actions", [])

        header = "## 5. Corrective Action Status\n\n"

        if not actions:
            return header + "*No corrective actions in progress.*"

        table = (
            "| ID | Finding | Description | Assigned | Due Date | Status | Progress |\n"
            "|----|---------|-------------|----------|----------|--------|----------|\n"
        )

        rows: List[str] = []
        for a in actions:
            status = a.get("status", "pending").upper()
            pct = a.get("completion_pct", 0.0)
            bar = self._md_bar(pct)
            rows.append(
                f"| {a.get('action_id', '')} | "
                f"{a.get('finding_id', '')} | "
                f"{a.get('description', '')} | "
                f"{a.get('assigned_to', '')} | "
                f"{self._fmt_date(a.get('due_date', ''))} | "
                f"{status} | "
                f"{pct:.0f}% {bar} |"
            )

        completed = sum(1 for a in actions if a.get("status") == "completed")
        overdue = sum(1 for a in actions if a.get("status") == "overdue")
        total = len(actions)

        summary = (
            f"\n\n**Total: {total} | Completed: {completed} | "
            f"Overdue: {overdue}**"
        )

        return header + table + "\n".join(rows) + summary

    def _md_verifier_engagement(self, data: Dict[str, Any]) -> str:
        """Build Markdown verifier engagement section."""
        ve = data.get("verifier_engagement", {})
        cur = self._currency()

        return (
            "## 6. Verifier Engagement\n\n"
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| Verifier | {ve.get('verifier_name', 'N/A')} |\n"
            f"| Accreditation | {ve.get('verifier_accreditation', 'N/A')} |\n"
            f"| Status | {ve.get('engagement_status', 'not_started').replace('_', ' ').title()} |\n"
            f"| Start Date | {ve.get('engagement_start_date', 'N/A')} |\n"
            f"| Next Verification | {ve.get('next_verification_date', 'N/A')} |\n"
            f"| Scope | {ve.get('scope_of_verification', 'N/A')} |\n"
            f"| Estimated Cost | {self._fmt_cur(ve.get('estimated_cost_eur', 0.0), cur)} |"
        )

    def _md_nca_correspondence(self, data: Dict[str, Any]) -> str:
        """Build Markdown NCA correspondence section."""
        items = data.get("nca_correspondence", [])

        header = (
            "## 7. NCA Correspondence\n\n"
            "| ID | NCA | Subject | Direction | Date | Deadline | Status | Pending |\n"
            "|----|-----|---------|-----------|------|----------|--------|--------|\n"
        )

        rows: List[str] = []
        for item in items:
            status = item.get("status", "open").upper()
            deadline = item.get("response_deadline", "") or "-"
            pending = item.get("pending_items", "") or "-"
            direction = item.get("direction", "inbound").upper()
            rows.append(
                f"| {item.get('correspondence_id', '')} | "
                f"{item.get('nca_name', '')} | "
                f"{item.get('subject', '')} | "
                f"{direction} | "
                f"{self._fmt_date(item.get('date', ''))} | "
                f"{self._fmt_date(deadline)} | "
                f"{status} | "
                f"{pending} |"
            )

        if not rows:
            return header + "| *No correspondence* | | | | | | | |"

        overdue = sum(1 for i in items if i.get("status") == "overdue")
        if overdue > 0:
            rows.append(f"\n\n> **WARNING:** {overdue} overdue response(s) to NCA.")

        return header + "\n".join(rows)

    def _md_penalty_exposure(self, data: Dict[str, Any]) -> str:
        """Build Markdown penalty exposure section."""
        pe = data.get("penalty_exposure", {})
        cur = self._currency()
        gaps = pe.get("by_gap", [])

        header = (
            "## 8. Penalty Exposure\n\n"
            f"**Total Estimated Exposure:** "
            f"**{self._fmt_cur(pe.get('total_exposure_eur', 0.0), cur)}**\n\n"
        )

        if gaps:
            header += (
                "### By Gap\n\n"
                "| Gap | Penalty Basis | Estimated Amount | Probability |\n"
                "|----|---------------|-----------------|------------|\n"
            )
            for g in gaps:
                prob = g.get("probability", 0.0)
                header += (
                    f"| {g.get('gap_description', '')} | "
                    f"{g.get('penalty_basis', '')} | "
                    f"{self._fmt_cur(g.get('estimated_amount_eur', 0.0), cur)} | "
                    f"{prob:.0%} |\n"
                )

        mitigating = pe.get("mitigating_factors", [])
        aggravating = pe.get("aggravating_factors", [])

        if mitigating:
            header += "\n### Mitigating Factors\n\n"
            for m in mitigating:
                header += f"- {m}\n"

        if aggravating:
            header += "\n### Aggravating Factors\n\n"
            for a in aggravating:
                header += f"- {a}\n"

        return header

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Build Markdown prioritized action items section."""
        items = data.get("action_items", [])
        sorted_items = sorted(items, key=lambda x: x.get("priority", 0))

        header = "## 9. Action Items\n\n"

        if not sorted_items:
            return header + "*No action items required.*"

        table = (
            "| Priority | Action | Category | Score Impact | Effort | Deadline | Assignee |\n"
            "|----------|--------|----------|-------------|--------|----------|----------|\n"
        )

        rows: List[str] = []
        for item in sorted_items:
            effort = item.get("effort", "medium").upper()
            rows.append(
                f"| #{item.get('priority', 0)} | "
                f"{item.get('action', '')} | "
                f"{item.get('category', '')} | "
                f"+{item.get('impact_on_score', 0)} pts | "
                f"{effort} | "
                f"{self._fmt_date(item.get('deadline', ''))} | "
                f"{item.get('assigned_to', 'TBD')} |"
            )

        total_impact = sum(i.get("impact_on_score", 0) for i in sorted_items)
        summary = f"\n\n**Total potential score improvement: +{total_impact} points**"

        return header + table + "\n".join(rows) + summary

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Pack: {self.PACK_ID}*\n\n"
            f"*Provenance Hash: `{provenance_hash}`*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML report header."""
        return (
            '<div class="report-header">'
            '<h1>CBAM Audit Readiness Scorecard</h1>'
            f'<div class="meta-item">Pack: {self.PACK_ID} | '
            f'Template: {self.TEMPLATE_NAME} | Version: {self.VERSION}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_overall_score(self, data: Dict[str, Any]) -> str:
        """Build HTML overall readiness score section."""
        os_data = data.get("overall_score", {})
        score = os_data.get("score", 0)
        grade = os_data.get("grade", "F")
        trend = os_data.get("trend", "stable")
        prev = os_data.get("previous_score")

        grade_color = self._get_grade_color(score)
        trend_icons = {"improving": "&#9650;", "stable": "&#9654;", "declining": "&#9660;"}
        trend_colors = {"improving": "#2ecc71", "stable": "#f39c12", "declining": "#e74c3c"}
        trend_icon = trend_icons.get(trend, "&#9654;")
        trend_color = trend_colors.get(trend, "#f39c12")

        prev_html = ""
        if prev is not None:
            delta = score - prev
            delta_sign = "+" if delta >= 0 else ""
            prev_html = (
                f'<div style="font-size:14px;color:#7f8c8d;margin-top:4px">'
                f'Previous: {prev} ({delta_sign}{delta})</div>'
            )

        return (
            '<div class="section"><h2>1. Overall Readiness Score</h2>'
            '<div style="display:flex;align-items:center;gap:32px;margin-bottom:16px">'
            f'<div style="text-align:center">'
            f'<div style="font-size:72px;font-weight:700;color:{grade_color}">{score}</div>'
            f'<div style="font-size:14px;color:#7f8c8d">out of 100</div>{prev_html}</div>'
            f'<div style="text-align:center">'
            f'<div style="font-size:64px;font-weight:700;color:{grade_color}">{grade}</div>'
            f'<div style="font-size:14px;color:#7f8c8d">Grade</div></div>'
            f'<div style="text-align:center">'
            f'<div style="font-size:32px;color:{trend_color}">{trend_icon}</div>'
            f'<div style="font-size:14px;color:#7f8c8d">{trend.capitalize()}</div></div>'
            f'</div>'
            f'<div class="progress-bar" style="height:20px">'
            f'<div class="progress-fill" style="width:{score}%;'
            f'background:{grade_color}"></div></div>'
            f'<div style="margin-top:12px;font-size:13px;color:#7f8c8d">'
            f'Assessment: {os_data.get("assessment_date", "N/A")} | '
            f'Next: {os_data.get("next_assessment_date", "N/A")}</div></div>'
        )

    def _html_category_scores(self, data: Dict[str, Any]) -> str:
        """Build HTML category scores section."""
        categories = data.get("category_scores", [])

        rows_html = ""
        for c in categories:
            score = c.get("score", 0)
            status = c.get("status", "adequate")
            status_label, status_color = self.STATUS_LABELS.get(
                status, ("Unknown", "#95a5a6")
            )
            score_color = self._get_grade_color(score)
            notes = c.get("notes", "") or "-"

            rows_html += (
                f'<tr><td><strong>{c.get("category", "")}</strong></td>'
                f'<td class="num" style="color:{score_color};font-weight:bold">{score}/100</td>'
                f'<td><div class="progress-bar"><div class="progress-fill" '
                f'style="width:{score}%;background:{score_color}"></div></div></td>'
                f'<td class="num">{c.get("weight", 0.0):.0%}</td>'
                f'<td style="color:{status_color};font-weight:bold">{status_label}</td>'
                f'<td class="num">{c.get("findings_count", 0)}</td>'
                f'<td>{notes}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No categories scored</em></td></tr>'

        return (
            '<div class="section"><h2>2. Category Scores</h2>'
            '<table><thead><tr>'
            '<th>Category</th><th>Score</th><th>Progress</th>'
            '<th>Weight</th><th>Status</th><th>Findings</th><th>Notes</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_evidence_completeness(self, data: Dict[str, Any]) -> str:
        """Build HTML evidence completeness section."""
        items = data.get("evidence_items", [])

        total = len(items)
        available = sum(1 for i in items if i.get("available", False))
        required = sum(1 for i in items if i.get("required", True))
        required_available = sum(
            1 for i in items if i.get("required", True) and i.get("available", False)
        )
        pct = (required_available / max(required, 1)) * 100
        pct_color = "#2ecc71" if pct >= 90 else "#f39c12" if pct >= 70 else "#e74c3c"

        kpis = (
            f'<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">Required Evidence</div>'
            f'<div class="kpi-value" style="color:{pct_color}">{required_available}/{required}</div>'
            f'<div class="kpi-unit">{pct:.1f}% complete</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Total Evidence</div>'
            f'<div class="kpi-value">{available}/{total}</div></div>'
            f'</div>'
            f'<div class="progress-bar" style="height:16px">'
            f'<div class="progress-fill" style="width:{pct:.0f}%;'
            f'background:{pct_color}"></div></div>'
        )

        missing = [i for i in items if i.get("required", True) and not i.get("available", False)]
        missing_html = ""
        if missing:
            rows = ""
            for m in missing:
                rows += (
                    f'<tr><td>{m.get("evidence_id", "")}</td>'
                    f'<td>{m.get("category", "")}</td>'
                    f'<td>{m.get("description", "")}</td>'
                    f'<td>{m.get("owner", "Unassigned")}</td></tr>'
                )
            missing_html = (
                '<h3>Missing Required Evidence</h3>'
                '<table><thead><tr>'
                '<th>ID</th><th>Category</th><th>Description</th><th>Owner</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>'
            )
        else:
            missing_html = '<p style="color:#2ecc71;font-weight:bold">All required evidence is available.</p>'

        return (
            f'<div class="section"><h2>3. Evidence Completeness</h2>'
            f'{kpis}{missing_html}</div>'
        )

    def _html_unresolved_findings(self, data: Dict[str, Any]) -> str:
        """Build HTML unresolved findings section."""
        findings = data.get("audit_findings", [])
        open_findings = [f for f in findings if f.get("status") != "resolved"]

        severity_colors = {
            "critical": "#8e44ad", "high": "#e74c3c",
            "medium": "#f39c12", "low": "#2ecc71",
        }
        status_colors = {
            "open": "#3498db", "in_progress": "#f39c12",
            "overdue": "#e74c3c",
        }

        rows_html = ""
        sorted_findings = sorted(
            open_findings,
            key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                x.get("severity", "low"), 4
            ),
        )
        for f in sorted_findings:
            severity = f.get("severity", "low")
            status = f.get("status", "open")
            sev_color = severity_colors.get(severity, "#95a5a6")
            stat_color = status_colors.get(status, "#95a5a6")

            rows_html += (
                f'<tr><td>{f.get("finding_id", "")}</td>'
                f'<td>{f.get("title", "")}</td>'
                f'<td><span style="background:{sev_color};color:#fff;padding:2px 8px;'
                f'border-radius:10px;font-size:12px">{severity.upper()}</span></td>'
                f'<td class="num">{f.get("age_days", 0)}</td>'
                f'<td>{f.get("assigned_owner", "Unassigned")}</td>'
                f'<td>{self._fmt_date(f.get("deadline", ""))}</td>'
                f'<td style="color:{stat_color};font-weight:bold">{status.upper()}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No unresolved findings</em></td></tr>'

        critical = sum(1 for f in open_findings if f.get("severity") == "critical")
        high = sum(1 for f in open_findings if f.get("severity") == "high")
        overdue = sum(1 for f in open_findings if f.get("status") == "overdue")

        summary = (
            f'<div style="margin-top:12px;display:flex;gap:24px">'
            f'<div><strong>Open:</strong> {len(open_findings)}</div>'
            f'<div style="color:#8e44ad"><strong>Critical:</strong> {critical}</div>'
            f'<div style="color:#e74c3c"><strong>High:</strong> {high}</div>'
            f'<div style="color:#e74c3c"><strong>Overdue:</strong> {overdue}</div>'
            f'</div>'
        )

        return (
            '<div class="section"><h2>4. Unresolved Findings</h2>'
            '<table><thead><tr>'
            '<th>ID</th><th>Title</th><th>Severity</th><th>Age (days)</th>'
            '<th>Owner</th><th>Deadline</th><th>Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>{summary}</div>'
        )

    def _html_corrective_actions(self, data: Dict[str, Any]) -> str:
        """Build HTML corrective action status section."""
        actions = data.get("corrective_actions", [])

        status_colors = {
            "pending": "#3498db", "in_progress": "#f39c12",
            "completed": "#2ecc71", "overdue": "#e74c3c",
        }

        rows_html = ""
        for a in actions:
            status = a.get("status", "pending")
            color = status_colors.get(status, "#95a5a6")
            pct = a.get("completion_pct", 0.0)
            pct_color = "#2ecc71" if pct >= 80 else "#f39c12" if pct >= 40 else "#e74c3c"

            rows_html += (
                f'<tr><td>{a.get("action_id", "")}</td>'
                f'<td>{a.get("finding_id", "")}</td>'
                f'<td>{a.get("description", "")}</td>'
                f'<td>{a.get("assigned_to", "")}</td>'
                f'<td>{self._fmt_date(a.get("due_date", ""))}</td>'
                f'<td style="color:{color};font-weight:bold">{status.upper()}</td>'
                f'<td><div class="progress-bar">'
                f'<div class="progress-fill" style="width:{pct:.0f}%;'
                f'background:{pct_color}"></div></div>'
                f'<div style="font-size:11px;text-align:center">{pct:.0f}%</div></td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No corrective actions</em></td></tr>'

        return (
            '<div class="section"><h2>5. Corrective Action Status</h2>'
            '<table><thead><tr>'
            '<th>ID</th><th>Finding</th><th>Description</th><th>Assigned</th>'
            '<th>Due Date</th><th>Status</th><th>Progress</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_verifier_engagement(self, data: Dict[str, Any]) -> str:
        """Build HTML verifier engagement section."""
        ve = data.get("verifier_engagement", {})
        cur = self._currency()
        status = ve.get("engagement_status", "not_started")

        status_colors = {
            "not_started": "#95a5a6", "planning": "#3498db",
            "active": "#2ecc71", "completed": "#1a5276",
        }
        color = status_colors.get(status, "#95a5a6")

        fields = [
            ("Verifier", ve.get("verifier_name", "N/A")),
            ("Accreditation", ve.get("verifier_accreditation", "N/A")),
            ("Start Date", ve.get("engagement_start_date", "N/A")),
            ("Next Verification", ve.get("next_verification_date", "N/A")),
            ("Scope", ve.get("scope_of_verification", "N/A")),
            ("Estimated Cost", self._fmt_cur(ve.get("estimated_cost_eur", 0.0), cur)),
        ]

        rows = ""
        for label, val in fields:
            rows += f'<tr><td><strong>{label}</strong></td><td>{val}</td></tr>'

        rows += (
            f'<tr><td><strong>Status</strong></td>'
            f'<td style="color:{color};font-weight:bold;font-size:16px">'
            f'{status.replace("_", " ").title()}</td></tr>'
        )

        return (
            '<div class="section"><h2>6. Verifier Engagement</h2>'
            f'<table><tbody>{rows}</tbody></table></div>'
        )

    def _html_nca_correspondence(self, data: Dict[str, Any]) -> str:
        """Build HTML NCA correspondence section."""
        items = data.get("nca_correspondence", [])

        status_colors = {
            "open": "#3498db", "responded": "#2ecc71",
            "closed": "#95a5a6", "overdue": "#e74c3c",
        }

        rows_html = ""
        for item in items:
            status = item.get("status", "open")
            color = status_colors.get(status, "#95a5a6")
            deadline = item.get("response_deadline", "") or "-"
            pending = item.get("pending_items", "") or "-"

            rows_html += (
                f'<tr><td>{item.get("correspondence_id", "")}</td>'
                f'<td>{item.get("nca_name", "")}</td>'
                f'<td>{item.get("subject", "")}</td>'
                f'<td>{item.get("direction", "").upper()}</td>'
                f'<td>{self._fmt_date(item.get("date", ""))}</td>'
                f'<td>{self._fmt_date(deadline)}</td>'
                f'<td style="color:{color};font-weight:bold">{status.upper()}</td>'
                f'<td>{pending}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="8"><em>No correspondence</em></td></tr>'

        return (
            '<div class="section"><h2>7. NCA Correspondence</h2>'
            '<table><thead><tr>'
            '<th>ID</th><th>NCA</th><th>Subject</th><th>Direction</th>'
            '<th>Date</th><th>Deadline</th><th>Status</th><th>Pending</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_penalty_exposure(self, data: Dict[str, Any]) -> str:
        """Build HTML penalty exposure section."""
        pe = data.get("penalty_exposure", {})
        cur = self._currency()
        total = pe.get("total_exposure_eur", 0.0)
        gaps = pe.get("by_gap", [])

        exposure_color = "#2ecc71" if total == 0 else "#e74c3c"

        header_html = (
            f'<div style="text-align:center;margin-bottom:16px">'
            f'<div style="font-size:14px;color:#7f8c8d">Total Estimated Penalty Exposure</div>'
            f'<div style="font-size:36px;font-weight:700;color:{exposure_color}">'
            f'{self._fmt_cur(total, cur)}</div></div>'
        )

        gaps_html = ""
        if gaps:
            rows = ""
            for g in gaps:
                prob = g.get("probability", 0.0)
                rows += (
                    f'<tr><td>{g.get("gap_description", "")}</td>'
                    f'<td>{g.get("penalty_basis", "")}</td>'
                    f'<td class="num">{self._fmt_cur(g.get("estimated_amount_eur", 0.0), cur)}</td>'
                    f'<td class="num">{prob:.0%}</td></tr>'
                )
            gaps_html = (
                '<table><thead><tr>'
                '<th>Gap</th><th>Penalty Basis</th><th>Estimated Amount</th>'
                '<th>Probability</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>'
            )

        factors_html = ""
        mitigating = pe.get("mitigating_factors", [])
        aggravating = pe.get("aggravating_factors", [])

        if mitigating or aggravating:
            factors_html = '<div style="display:flex;gap:16px;margin-top:16px">'
            if mitigating:
                items = "".join(f'<li>{m}</li>' for m in mitigating)
                factors_html += (
                    f'<div style="flex:1;background:#f0fff0;padding:12px;border-radius:4px">'
                    f'<strong style="color:#2ecc71">Mitigating Factors</strong>'
                    f'<ul style="margin:8px 0;padding-left:20px">{items}</ul></div>'
                )
            if aggravating:
                items = "".join(f'<li>{a}</li>' for a in aggravating)
                factors_html += (
                    f'<div style="flex:1;background:#fff0f0;padding:12px;border-radius:4px">'
                    f'<strong style="color:#e74c3c">Aggravating Factors</strong>'
                    f'<ul style="margin:8px 0;padding-left:20px">{items}</ul></div>'
                )
            factors_html += '</div>'

        return (
            f'<div class="section"><h2>8. Penalty Exposure</h2>'
            f'{header_html}{gaps_html}{factors_html}</div>'
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Build HTML prioritized action items section."""
        items = data.get("action_items", [])
        sorted_items = sorted(items, key=lambda x: x.get("priority", 0))

        effort_colors = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}

        rows_html = ""
        for item in sorted_items:
            effort = item.get("effort", "medium")
            e_color = effort_colors.get(effort, "#95a5a6")
            impact = item.get("impact_on_score", 0)

            rows_html += (
                f'<tr><td><strong>#{item.get("priority", 0)}</strong></td>'
                f'<td>{item.get("action", "")}</td>'
                f'<td>{item.get("category", "")}</td>'
                f'<td class="num" style="color:#2ecc71;font-weight:bold">+{impact} pts</td>'
                f'<td style="color:{e_color};font-weight:bold">{effort.upper()}</td>'
                f'<td>{self._fmt_date(item.get("deadline", ""))}</td>'
                f'<td>{item.get("assigned_to", "TBD")}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7"><em>No action items required</em></td></tr>'

        total_impact = sum(i.get("impact_on_score", 0) for i in sorted_items)

        return (
            '<div class="section"><h2>9. Action Items</h2>'
            '<table><thead><tr>'
            '<th>Priority</th><th>Action</th><th>Category</th>'
            '<th>Score Impact</th><th>Effort</th><th>Deadline</th><th>Assignee</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
            f'<div style="margin-top:12px;padding:12px;background:#f8f9fa;'
            f'border-radius:4px;text-align:center">'
            f'<strong>Total potential score improvement: '
            f'<span style="color:#2ecc71">+{total_impact} points</span></strong></div></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_overall_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON overall score."""
        os_data = data.get("overall_score", {})
        return {
            "score": os_data.get("score", 0),
            "grade": os_data.get("grade", "F"),
            "trend": os_data.get("trend", "stable"),
            "previous_score": os_data.get("previous_score"),
            "assessment_date": os_data.get("assessment_date", ""),
            "next_assessment_date": os_data.get("next_assessment_date", ""),
        }

    def _json_category_scores(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON category scores."""
        return [
            {
                "category": c.get("category", ""),
                "category_key": c.get("category_key", ""),
                "score": c.get("score", 0),
                "weight": round(c.get("weight", 0.0), 2),
                "status": c.get("status", "adequate"),
                "findings_count": c.get("findings_count", 0),
                "notes": c.get("notes", ""),
            }
            for c in data.get("category_scores", [])
        ]

    def _json_evidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON evidence completeness."""
        items = data.get("evidence_items", [])
        total = len(items)
        required = sum(1 for i in items if i.get("required", True))
        required_available = sum(
            1 for i in items if i.get("required", True) and i.get("available", False)
        )

        return {
            "total_items": total,
            "required_items": required,
            "required_available": required_available,
            "completeness_pct": round(
                (required_available / max(required, 1)) * 100, 2
            ),
            "items": [
                {
                    "evidence_id": i.get("evidence_id", ""),
                    "category": i.get("category", ""),
                    "description": i.get("description", ""),
                    "required": i.get("required", True),
                    "available": i.get("available", False),
                    "quality": i.get("quality", "pending"),
                    "last_updated": i.get("last_updated", ""),
                    "owner": i.get("owner", ""),
                }
                for i in items
            ],
        }

    def _json_findings(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON audit findings."""
        return [
            {
                "finding_id": f.get("finding_id", ""),
                "category": f.get("category", ""),
                "title": f.get("title", ""),
                "severity": f.get("severity", "low"),
                "description": f.get("description", ""),
                "identified_date": f.get("identified_date", ""),
                "age_days": f.get("age_days", 0),
                "assigned_owner": f.get("assigned_owner", ""),
                "deadline": f.get("deadline", ""),
                "status": f.get("status", "open"),
            }
            for f in data.get("audit_findings", [])
        ]

    def _json_corrective_actions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON corrective actions."""
        return [
            {
                "action_id": a.get("action_id", ""),
                "finding_id": a.get("finding_id", ""),
                "description": a.get("description", ""),
                "assigned_to": a.get("assigned_to", ""),
                "due_date": a.get("due_date", ""),
                "status": a.get("status", "pending"),
                "completion_pct": round(a.get("completion_pct", 0.0), 1),
            }
            for a in data.get("corrective_actions", [])
        ]

    def _json_verifier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON verifier engagement."""
        ve = data.get("verifier_engagement", {})
        return {
            "verifier_name": ve.get("verifier_name", ""),
            "verifier_accreditation": ve.get("verifier_accreditation", ""),
            "engagement_status": ve.get("engagement_status", "not_started"),
            "engagement_start_date": ve.get("engagement_start_date", ""),
            "next_verification_date": ve.get("next_verification_date", ""),
            "scope_of_verification": ve.get("scope_of_verification", ""),
            "estimated_cost_eur": round(ve.get("estimated_cost_eur", 0.0), 2),
        }

    def _json_nca(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON NCA correspondence."""
        return [
            {
                "correspondence_id": i.get("correspondence_id", ""),
                "nca_name": i.get("nca_name", ""),
                "subject": i.get("subject", ""),
                "direction": i.get("direction", "inbound"),
                "date": i.get("date", ""),
                "response_deadline": i.get("response_deadline", ""),
                "status": i.get("status", "open"),
                "pending_items": i.get("pending_items", ""),
            }
            for i in data.get("nca_correspondence", [])
        ]

    def _json_penalty(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON penalty exposure."""
        pe = data.get("penalty_exposure", {})
        return {
            "total_exposure_eur": round(pe.get("total_exposure_eur", 0.0), 2),
            "by_gap": [
                {
                    "gap_description": g.get("gap_description", ""),
                    "penalty_basis": g.get("penalty_basis", ""),
                    "estimated_amount_eur": round(g.get("estimated_amount_eur", 0.0), 2),
                    "probability": round(g.get("probability", 0.0), 2),
                }
                for g in pe.get("by_gap", [])
            ],
            "mitigating_factors": pe.get("mitigating_factors", []),
            "aggravating_factors": pe.get("aggravating_factors", []),
        }

    def _json_action_items(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON action items."""
        return [
            {
                "priority": i.get("priority", 0),
                "action": i.get("action", ""),
                "category": i.get("category", ""),
                "impact_on_score": i.get("impact_on_score", 0),
                "effort": i.get("effort", "medium"),
                "deadline": i.get("deadline", ""),
                "assigned_to": i.get("assigned_to", ""),
            }
            for i in sorted(
                data.get("action_items", []),
                key=lambda x: x.get("priority", 0),
            )
        ]

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _compute_provenance_hash(self, content: str) -> str:
        """Compute SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _currency(self) -> str:
        """Get configured currency code."""
        return self.config.get("currency", "EUR")

    def _get_grade_color(self, score: int) -> str:
        """Get color for a given score."""
        for threshold, _, color in self.GRADE_THRESHOLDS:
            if score >= threshold:
                return color
        return "#e74c3c"

    def _md_bar(self, pct: float) -> str:
        """Create a simple text-based progress bar for Markdown."""
        filled = int(pct / 10)
        empty = 10 - filled
        return f"[{'#' * filled}{'.' * empty}]"

    def _fmt_int(self, value: Union[int, float, None]) -> str:
        """Format integer with thousand separators."""
        if value is None:
            return "0"
        return f"{int(value):,}"

    def _fmt_num(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format number with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _fmt_cur(self, value: Union[int, float], currency: str = "EUR") -> str:
        """Format currency value."""
        return f"{currency} {value:,.2f}"

    def _fmt_date(self, dt: Union[datetime, str]) -> str:
        """Format datetime to ISO date string."""
        if isinstance(dt, str):
            return dt[:10] if dt else ""
        return dt.strftime("%Y-%m-%d")

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = self._get_css()
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f'Pack: {self.PACK_ID} | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )

    def _get_css(self) -> str:
        """Return inline CSS for HTML reports."""
        return (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin-bottom:16px}"
            ".kpi-card{background:#fff;padding:20px;border-radius:8px;text-align:center;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:28px;font-weight:700;color:#1a5276}"
            ".kpi-unit{font-size:12px;color:#95a5a6;margin-top:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            ".section h3{margin:16px 0 8px 0;font-size:15px;color:#2c3e50}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;margin:4px 0}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
