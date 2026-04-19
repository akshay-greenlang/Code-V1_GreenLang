# -*- coding: utf-8 -*-
"""
CertificationScorecardTemplate - LEED/BREEAM/Energy Star scorecard for PACK-032.

Generates certification scorecard reports covering target certification
schemes (LEED, BREEAM, Energy Star, NABERS, HQE, DGNB), credit
category scores, prerequisite status, credits achieved vs available,
gap analysis, action items for next level, estimated cost to achieve,
and implementation timelines.

Sections:
    1. Target Certification Summary
    2. Credit Category Scores
    3. Prerequisites Status
    4. Credits Achieved vs Available
    5. Gap Analysis
    6. Action Items for Next Level
    7. Estimated Cost to Achieve
    8. Implementation Timeline
    9. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CertificationScorecardTemplate:
    """
    Green building certification scorecard report template.

    Renders certification scorecards for LEED, BREEAM, Energy Star,
    and other schemes with credit tracking, gap analysis, action
    items, and cost estimates across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    SCORECARD_SECTIONS: List[str] = [
        "Certification Summary",
        "Credit Category Scores",
        "Prerequisites",
        "Credits Achieved",
        "Gap Analysis",
        "Action Items",
        "Cost Estimate",
        "Timeline",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CertificationScorecardTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render certification scorecard as Markdown.

        Args:
            data: Certification assessment data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_certification_summary(data),
            self._md_credit_categories(data),
            self._md_prerequisites(data),
            self._md_credits_achieved(data),
            self._md_gap_analysis(data),
            self._md_action_items(data),
            self._md_cost_estimate(data),
            self._md_timeline(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render certification scorecard as self-contained HTML.

        Args:
            data: Certification assessment data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_certification_summary(data),
            self._html_credit_categories(data),
            self._html_prerequisites(data),
            self._html_credits_achieved(data),
            self._html_gap_analysis(data),
            self._html_action_items(data),
            self._html_cost_estimate(data),
            self._html_timeline(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Certification Scorecard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render certification scorecard as structured JSON.

        Args:
            data: Certification assessment data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "certification_scorecard",
            "version": "32.0.0",
            "generated_at": self.generated_at.isoformat(),
            "certification_summary": self._json_certification_summary(data),
            "credit_categories": data.get("credit_categories", []),
            "prerequisites": data.get("prerequisites", []),
            "credits_detail": data.get("credits_detail", []),
            "gap_analysis": data.get("gap_analysis", []),
            "action_items": data.get("action_items", []),
            "cost_estimate": data.get("cost_estimate", {}),
            "timeline": data.get("timeline", []),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        name = data.get("building_name", "Building")
        scheme = data.get("certification_scheme", "-")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Certification Scorecard\n\n"
            f"**Building:** {name}  \n"
            f"**Certification Scheme:** {scheme}  \n"
            f"**Target Level:** {data.get('target_level', '-')}  \n"
            f"**Assessment Date:** {data.get('assessment_date', '-')}  \n"
            f"**Assessor:** {data.get('assessor', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 CertificationScorecardTemplate v32.0.0\n\n---"
        )

    def _md_certification_summary(self, data: Dict[str, Any]) -> str:
        """Render certification summary section."""
        s = data.get("certification_summary", {})
        return (
            "## 1. Certification Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Scheme | {s.get('scheme', '-')} |\n"
            f"| Version | {s.get('version', '-')} |\n"
            f"| Current Score | {s.get('current_score', 0)} / {s.get('max_score', 0)} |\n"
            f"| Current Level | {s.get('current_level', '-')} |\n"
            f"| Target Level | {s.get('target_level', '-')} |\n"
            f"| Points Needed | {s.get('points_needed', 0)} |\n"
            f"| Prerequisites Met | {s.get('prerequisites_met', 0)} / "
            f"{s.get('prerequisites_total', 0)} |\n"
            f"| Confidence | {s.get('confidence', '-')} |\n"
            f"| Feasibility | {s.get('feasibility', '-')} |"
        )

    def _md_credit_categories(self, data: Dict[str, Any]) -> str:
        """Render credit category scores section."""
        categories = data.get("credit_categories", [])
        if not categories:
            return "## 2. Credit Category Scores\n\n_No category data available._"
        lines = [
            "## 2. Credit Category Scores\n",
            "| Category | Achieved | Available | Score (%) | Weight (%) |",
            "|----------|---------|-----------|-----------|-----------|",
        ]
        for c in categories:
            achieved = c.get("achieved", 0)
            available = c.get("available", 0)
            pct = (achieved / available * 100) if available > 0 else 0
            lines.append(
                f"| {c.get('category', '-')} "
                f"| {achieved} "
                f"| {available} "
                f"| {self._fmt(pct)}% "
                f"| {self._fmt(c.get('weight_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_prerequisites(self, data: Dict[str, Any]) -> str:
        """Render prerequisites status section."""
        prereqs = data.get("prerequisites", [])
        if not prereqs:
            return "## 3. Prerequisites Status\n\n_No prerequisites data._"
        lines = [
            "## 3. Prerequisites Status\n",
            "| Prerequisite | Category | Status | Notes |",
            "|-------------|----------|--------|-------|",
        ]
        for p in prereqs:
            status = p.get("status", "Not Met")
            lines.append(
                f"| {p.get('name', '-')} "
                f"| {p.get('category', '-')} "
                f"| {status} "
                f"| {p.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_credits_achieved(self, data: Dict[str, Any]) -> str:
        """Render credits achieved vs available section."""
        credits = data.get("credits_detail", [])
        if not credits:
            return "## 4. Credits Achieved vs Available\n\n_No credit details available._"
        lines = [
            "## 4. Credits Achieved vs Available\n",
            "| Credit | Category | Points | Max | Status | Difficulty |",
            "|--------|----------|--------|-----|--------|-----------|",
        ]
        for c in credits:
            lines.append(
                f"| {c.get('credit', '-')} "
                f"| {c.get('category', '-')} "
                f"| {c.get('points', 0)} "
                f"| {c.get('max_points', 0)} "
                f"| {c.get('status', '-')} "
                f"| {c.get('difficulty', '-')} |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis section."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return "## 5. Gap Analysis\n\n_No gaps identified._"
        lines = [
            "## 5. Gap Analysis\n",
            "| Credit | Points Gap | Effort | Cost Impact | Priority |",
            "|--------|-----------|--------|-------------|----------|",
        ]
        for g in gaps:
            lines.append(
                f"| {g.get('credit', '-')} "
                f"| {g.get('points_gap', 0)} "
                f"| {g.get('effort', '-')} "
                f"| {g.get('cost_impact', '-')} "
                f"| {g.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render action items for next level section."""
        actions = data.get("action_items", [])
        if not actions:
            return "## 6. Action Items for Next Level\n\n_No action items._"
        lines = [
            "## 6. Action Items for Next Level\n",
            "| # | Action | Credit | Points | Cost | Deadline | Owner |",
            "|---|--------|--------|--------|------|----------|-------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('credit', '-')} "
                f"| {a.get('points', 0)} "
                f"| {a.get('cost', '-')} "
                f"| {a.get('deadline', '-')} "
                f"| {a.get('owner', '-')} |"
            )
        return "\n".join(lines)

    def _md_cost_estimate(self, data: Dict[str, Any]) -> str:
        """Render estimated cost to achieve section."""
        cost = data.get("cost_estimate", {})
        breakdown = cost.get("breakdown", [])
        lines = [
            "## 7. Estimated Cost to Achieve\n",
            f"**Total Estimated Cost:** {cost.get('total_cost', '-')}  ",
            f"**Consultancy Fees:** {cost.get('consultancy_fees', '-')}  ",
            f"**Registration Fees:** {cost.get('registration_fees', '-')}  ",
            f"**Certification Fees:** {cost.get('certification_fees', '-')}  ",
            f"**Implementation Costs:** {cost.get('implementation_costs', '-')}  ",
            f"**Testing & Commissioning:** {cost.get('testing_costs', '-')}",
        ]
        if breakdown:
            lines.extend([
                "\n### Cost Breakdown by Category\n",
                "| Category | Cost | Notes |",
                "|----------|------|-------|",
            ])
            for b in breakdown:
                lines.append(
                    f"| {b.get('category', '-')} "
                    f"| {b.get('cost', '-')} "
                    f"| {b.get('notes', '-')} |"
                )
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render implementation timeline section."""
        phases = data.get("timeline", [])
        if not phases:
            return "## 8. Implementation Timeline\n\n_No timeline defined._"
        lines = [
            "## 8. Implementation Timeline\n",
            "| Phase | Start | End | Deliverables | Dependencies |",
            "|-------|-------|-----|-------------|-------------|",
        ]
        for p in phases:
            lines.append(
                f"| {p.get('phase', '-')} "
                f"| {p.get('start', '-')} "
                f"| {p.get('end', '-')} "
                f"| {p.get('deliverables', '-')} "
                f"| {p.get('dependencies', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 CertificationScorecardTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        name = data.get("building_name", "Building")
        scheme = data.get("certification_scheme", "-")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Certification Scorecard - {scheme}</h1>\n'
            f'<p class="subtitle">Building: {name} | Target: {data.get("target_level", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_certification_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML certification summary."""
        s = data.get("certification_summary", {})
        return (
            '<h2>Certification Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">Score</span>'
            f'<span class="value">{s.get("current_score", 0)}/{s.get("max_score", 0)}</span></div>\n'
            f'<div class="card"><span class="label">Current</span>'
            f'<span class="value">{s.get("current_level", "-")}</span></div>\n'
            f'<div class="card"><span class="label">Target</span>'
            f'<span class="value">{s.get("target_level", "-")}</span></div>\n'
            f'<div class="card"><span class="label">Points Needed</span>'
            f'<span class="value">{s.get("points_needed", 0)}</span></div>\n'
            '</div>'
        )

    def _html_credit_categories(self, data: Dict[str, Any]) -> str:
        """Render HTML credit category scores."""
        categories = data.get("credit_categories", [])
        rows = ""
        for c in categories:
            achieved = c.get("achieved", 0)
            available = c.get("available", 0)
            pct = (achieved / available * 100) if available > 0 else 0
            rows += (
                f'<tr><td>{c.get("category", "-")}</td>'
                f'<td>{achieved}</td><td>{available}</td>'
                f'<td>{self._fmt(pct)}%</td></tr>\n'
            )
        return (
            '<h2>Credit Category Scores</h2>\n'
            '<table>\n<tr><th>Category</th><th>Achieved</th><th>Available</th>'
            f'<th>Score</th></tr>\n{rows}</table>'
        )

    def _html_prerequisites(self, data: Dict[str, Any]) -> str:
        """Render HTML prerequisites status."""
        prereqs = data.get("prerequisites", [])
        rows = ""
        for p in prereqs:
            status = p.get("status", "Not Met")
            style = 'color:#198754' if status == "Met" else 'color:#dc3545'
            rows += (
                f'<tr><td>{p.get("name", "-")}</td>'
                f'<td>{p.get("category", "-")}</td>'
                f'<td style="{style};font-weight:bold">{status}</td></tr>\n'
            )
        return (
            '<h2>Prerequisites Status</h2>\n'
            '<table>\n<tr><th>Prerequisite</th><th>Category</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_credits_achieved(self, data: Dict[str, Any]) -> str:
        """Render HTML credits achieved table."""
        credits = data.get("credits_detail", [])
        rows = ""
        for c in credits:
            rows += (
                f'<tr><td>{c.get("credit", "-")}</td>'
                f'<td>{c.get("points", 0)}/{c.get("max_points", 0)}</td>'
                f'<td>{c.get("status", "-")}</td>'
                f'<td>{c.get("difficulty", "-")}</td></tr>\n'
            )
        return (
            '<h2>Credits Achieved vs Available</h2>\n'
            '<table>\n<tr><th>Credit</th><th>Points</th><th>Status</th>'
            f'<th>Difficulty</th></tr>\n{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis."""
        gaps = data.get("gap_analysis", [])
        rows = ""
        for g in gaps:
            rows += (
                f'<tr><td>{g.get("credit", "-")}</td>'
                f'<td>{g.get("points_gap", 0)}</td>'
                f'<td>{g.get("effort", "-")}</td>'
                f'<td>{g.get("priority", "-")}</td></tr>\n'
            )
        return (
            '<h2>Gap Analysis</h2>\n'
            '<table>\n<tr><th>Credit</th><th>Points Gap</th><th>Effort</th>'
            f'<th>Priority</th></tr>\n{rows}</table>'
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items."""
        actions = data.get("action_items", [])
        items = "".join(
            f'<li><strong>{a.get("action", "-")}</strong> '
            f'({a.get("credit", "-")}: {a.get("points", 0)} pts, '
            f'Cost: {a.get("cost", "-")})</li>\n'
            for a in actions
        )
        return f'<h2>Action Items for Next Level</h2>\n<ol>\n{items}</ol>'

    def _html_cost_estimate(self, data: Dict[str, Any]) -> str:
        """Render HTML cost estimate."""
        cost = data.get("cost_estimate", {})
        breakdown = cost.get("breakdown", [])
        rows = ""
        for b in breakdown:
            rows += (
                f'<tr><td>{b.get("category", "-")}</td>'
                f'<td>{b.get("cost", "-")}</td>'
                f'<td>{b.get("notes", "-")}</td></tr>\n'
            )
        return (
            '<h2>Estimated Cost to Achieve</h2>\n'
            f'<p>Total: <strong>{cost.get("total_cost", "-")}</strong></p>\n'
            '<table>\n<tr><th>Category</th><th>Cost</th>'
            f'<th>Notes</th></tr>\n{rows}</table>'
        )

    def _html_timeline(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation timeline."""
        phases = data.get("timeline", [])
        rows = ""
        for p in phases:
            rows += (
                f'<tr><td>{p.get("phase", "-")}</td>'
                f'<td>{p.get("start", "-")}</td>'
                f'<td>{p.get("end", "-")}</td>'
                f'<td>{p.get("deliverables", "-")}</td></tr>\n'
            )
        return (
            '<h2>Implementation Timeline</h2>\n'
            '<table>\n<tr><th>Phase</th><th>Start</th><th>End</th>'
            f'<th>Deliverables</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_certification_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON certification summary."""
        s = data.get("certification_summary", {})
        return {
            "scheme": s.get("scheme", ""),
            "version": s.get("version", ""),
            "current_score": s.get("current_score", 0),
            "max_score": s.get("max_score", 0),
            "current_level": s.get("current_level", ""),
            "target_level": s.get("target_level", ""),
            "points_needed": s.get("points_needed", 0),
            "prerequisites_met": s.get("prerequisites_met", 0),
            "prerequisites_total": s.get("prerequisites_total", 0),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
