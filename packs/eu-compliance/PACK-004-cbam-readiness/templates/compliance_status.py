"""
ComplianceStatusTemplate - CBAM overall compliance dashboard template.

This module implements the overall CBAM compliance status dashboard. It generates
formatted reports with regulatory timelines, compliance scoring, obligation
checklists, filing deadline calendars, risk indicators, action items, and
goods category coverage analysis.

Example:
    >>> template = ComplianceStatusTemplate()
    >>> data = {"compliance_score": 85, "obligations": [...], ...}
    >>> html = template.render_html(data)
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


# Key CBAM regulatory dates
CBAM_REGULATORY_TIMELINE: List[Dict[str, str]] = [
    {"date": "2023-10-01", "event": "CBAM transitional period begins", "phase": "transitional"},
    {"date": "2024-01-31", "event": "First quarterly report due (Q4 2023)", "phase": "transitional"},
    {"date": "2025-12-31", "event": "Transitional period ends", "phase": "transitional"},
    {"date": "2026-01-01", "event": "Definitive CBAM period begins", "phase": "definitive"},
    {"date": "2026-05-31", "event": "First certificate surrender deadline", "phase": "definitive"},
    {"date": "2026-12-31", "event": "Free allocation at 97.5%", "phase": "definitive"},
    {"date": "2027-12-31", "event": "Free allocation at 95.0%", "phase": "definitive"},
    {"date": "2034-01-01", "event": "Full CBAM (0% free allocation)", "phase": "definitive"},
]


class ComplianceStatusTemplate:
    """
    CBAM overall compliance status dashboard template.

    Generates formatted compliance dashboards with regulatory timelines,
    scoring, obligation checklists, deadline calendars, risk indicators,
    action items, and goods category coverage analysis.

    Attributes:
        config: Optional configuration dictionary.
        generated_at: Timestamp of report generation.
    """

    OBLIGATION_TYPES: List[str] = [
        "quarterly_reports",
        "annual_declaration",
        "certificates",
        "verification",
    ]

    RISK_CATEGORIES: List[str] = [
        "data_quality",
        "supplier",
        "cost_exposure",
        "regulatory_change",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize ComplianceStatusTemplate.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - score_threshold_pass (int): Minimum score for PASS (default: 80).
                - show_regulatory_timeline (bool): Whether to show timeline (default: True).
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the compliance dashboard as Markdown.

        Args:
            data: Dashboard data dictionary containing:
                - compliance_score (float): 0-100 overall score
                - obligations (list[dict]): obligation checklist items
                - deadlines (list[dict]): filing deadlines with countdown
                - risk_indicators (list[dict]): risk items by category
                - action_items (list[dict]): upcoming/pending/overdue actions
                - goods_coverage (list[dict]): coverage by goods category

        Returns:
            Formatted Markdown string.
        """
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_regulatory_timeline())
        sections.append(self._md_compliance_score(data))
        sections.append(self._md_obligation_checklist(data))
        sections.append(self._md_filing_deadlines(data))
        sections.append(self._md_risk_indicators(data))
        sections.append(self._md_action_items(data))
        sections.append(self._md_goods_coverage(data))
        sections.append(self._md_provenance_footer())

        content = "\n\n".join(sections)
        provenance_hash = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the compliance dashboard as self-contained HTML.

        Args:
            data: Dashboard data dictionary (same schema as render_markdown).

        Returns:
            Complete HTML document string with inline CSS.
        """
        sections: List[str] = []

        sections.append(self._html_header(data))
        sections.append(self._html_regulatory_timeline())
        sections.append(self._html_compliance_score(data))
        sections.append(self._html_obligation_checklist(data))
        sections.append(self._html_filing_deadlines(data))
        sections.append(self._html_risk_indicators(data))
        sections.append(self._html_action_items(data))
        sections.append(self._html_goods_coverage(data))

        body = "\n".join(sections)
        provenance_hash = self._generate_provenance_hash(body)

        return self._wrap_html(
            title="CBAM Compliance Status Dashboard",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the compliance dashboard as a structured dict.

        Args:
            data: Dashboard data dictionary (same schema as render_markdown).

        Returns:
            Dictionary with all dashboard sections and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_compliance_status",
            "generated_at": self.generated_at,
            "regulatory_timeline": CBAM_REGULATORY_TIMELINE,
            "compliance_score": self._json_compliance_score(data),
            "obligation_checklist": self._json_obligation_checklist(data),
            "filing_deadlines": self._json_filing_deadlines(data),
            "risk_indicators": self._json_risk_indicators(data),
            "action_items": self._json_action_items(data),
            "goods_coverage": self._json_goods_coverage(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown header."""
        score = data.get("compliance_score", 0.0)
        threshold = self.config.get("score_threshold_pass", 80)
        status = "COMPLIANT" if score >= threshold else "NON-COMPLIANT"

        return (
            f"# CBAM Compliance Status Dashboard\n\n"
            f"**Status:** {status}\n\n"
            f"**Compliance Score:** {self._format_number(score, 0)}/100\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_regulatory_timeline(self) -> str:
        """Build Markdown regulatory timeline section."""
        show = self.config.get("show_regulatory_timeline", True)
        if not show:
            return ""

        today = datetime.utcnow().strftime("%Y-%m-%d")

        header = (
            "## Regulatory Timeline\n\n"
            "| Date | Event | Phase | Status |\n"
            "|------|-------|-------|--------|\n"
        )

        rows: List[str] = []
        for entry in CBAM_REGULATORY_TIMELINE:
            date_str = entry["date"]
            status = "Past" if date_str <= today else "Upcoming"
            rows.append(
                f"| {date_str} | {entry['event']} | "
                f"{entry['phase'].capitalize()} | {status} |"
            )

        return header + "\n".join(rows)

    def _md_compliance_score(self, data: Dict[str, Any]) -> str:
        """Build Markdown compliance score gauge section."""
        score = data.get("compliance_score", 0.0)
        threshold = self.config.get("score_threshold_pass", 80)

        # ASCII gauge
        filled = int(score / 5)
        empty = 20 - filled
        gauge = "[" + "#" * filled + "-" * empty + "]"

        sub_scores: List[Dict[str, Any]] = data.get("sub_scores", [])

        section = (
            "## Compliance Score\n\n"
            f"```\n{gauge} {self._format_number(score, 0)}/100\n```\n\n"
            f"**Threshold:** {threshold}/100\n\n"
        )

        if sub_scores:
            section += (
                "| Component | Score | Weight |\n"
                "|-----------|-------|--------|\n"
            )
            for ss in sub_scores:
                section += (
                    f"| {ss.get('component', '')} | "
                    f"{self._format_number(ss.get('score', 0.0), 1)} | "
                    f"{self._format_percentage(ss.get('weight_pct', 0.0))} |\n"
                )

        return section

    def _md_obligation_checklist(self, data: Dict[str, Any]) -> str:
        """Build Markdown obligation checklist section."""
        obligations: List[Dict[str, Any]] = data.get("obligations", [])

        header = (
            "## Obligation Checklist\n\n"
            "| Obligation | Status | Due Date | Last Filed | Notes |\n"
            "|------------|--------|----------|------------|-------|\n"
        )

        rows: List[str] = []
        for obl in obligations:
            status = obl.get("status", "pending")
            check = "[x]" if status == "completed" else "[ ]"

            rows.append(
                f"| {check} {obl.get('name', '')} | "
                f"{status.upper()} | "
                f"{self._format_date(obl.get('due_date', 'N/A'))} | "
                f"{self._format_date(obl.get('last_filed', 'N/A'))} | "
                f"{obl.get('notes', '')} |"
            )

        return header + "\n".join(rows)

    def _md_filing_deadlines(self, data: Dict[str, Any]) -> str:
        """Build Markdown filing deadlines calendar section."""
        deadlines: List[Dict[str, Any]] = data.get("deadlines", [])

        header = (
            "## Filing Deadlines\n\n"
            "| Deadline | Description | Days Remaining | Priority |\n"
            "|----------|-------------|----------------|----------|\n"
        )

        rows: List[str] = []
        for dl in sorted(deadlines, key=lambda x: x.get("date", "")):
            days = dl.get("days_remaining", 0)
            priority = "URGENT" if days <= 7 else "HIGH" if days <= 30 else "NORMAL"

            rows.append(
                f"| {self._format_date(dl.get('date', ''))} | "
                f"{dl.get('description', '')} | "
                f"{days} days | "
                f"{priority} |"
            )

        return header + "\n".join(rows)

    def _md_risk_indicators(self, data: Dict[str, Any]) -> str:
        """Build Markdown risk indicators section."""
        risks: List[Dict[str, Any]] = data.get("risk_indicators", [])

        if not risks:
            return "## Risk Indicators\n\n*No risk indicators to display.*"

        section = "## Risk Indicators\n\n"

        for category in self.RISK_CATEGORIES:
            category_risks = [r for r in risks if r.get("category", "") == category]
            if not category_risks:
                continue

            section += (
                f"### {category.replace('_', ' ').title()} Risks\n\n"
                f"| Risk | Severity | Likelihood | Impact | Mitigation |\n"
                f"|------|----------|------------|--------|------------|\n"
            )

            for risk in category_risks:
                severity = risk.get("severity", "medium").upper()
                section += (
                    f"| {risk.get('description', '')} | "
                    f"{severity} | "
                    f"{risk.get('likelihood', 'N/A').capitalize()} | "
                    f"{risk.get('impact', 'N/A').capitalize()} | "
                    f"{risk.get('mitigation', 'N/A')} |\n"
                )

            section += "\n"

        return section.rstrip()

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Build Markdown action items section."""
        actions: List[Dict[str, Any]] = data.get("action_items", [])

        if not actions:
            return "## Action Items\n\n*No action items pending.*"

        # Group by status
        overdue = [a for a in actions if a.get("status", "") == "overdue"]
        pending = [a for a in actions if a.get("status", "") == "pending"]
        upcoming = [a for a in actions if a.get("status", "") == "upcoming"]

        section = "## Action Items\n\n"

        if overdue:
            section += "### OVERDUE\n\n"
            for a in overdue:
                section += (
                    f"- **{a.get('title', '')}** - Due: "
                    f"{self._format_date(a.get('due_date', 'N/A'))} "
                    f"({a.get('days_overdue', 0)} days overdue)\n"
                )
            section += "\n"

        if pending:
            section += "### Pending\n\n"
            for a in pending:
                section += (
                    f"- {a.get('title', '')} - Due: "
                    f"{self._format_date(a.get('due_date', 'N/A'))}\n"
                )
            section += "\n"

        if upcoming:
            section += "### Upcoming\n\n"
            for a in upcoming:
                section += (
                    f"- {a.get('title', '')} - Due: "
                    f"{self._format_date(a.get('due_date', 'N/A'))}\n"
                )

        return section.rstrip()

    def _md_goods_coverage(self, data: Dict[str, Any]) -> str:
        """Build Markdown goods category coverage section."""
        coverage: List[Dict[str, Any]] = data.get("goods_coverage", [])

        if not coverage:
            return "## Goods Category Coverage\n\n*No goods coverage data available.*"

        header = (
            "## Goods Category Coverage\n\n"
            "| Category | CN Codes Covered | Suppliers Mapped | "
            "Data Quality | Coverage Status |\n"
            "|----------|------------------|------------------|"
            "-------------|------------------|\n"
        )

        rows: List[str] = []
        for gc in coverage:
            status = gc.get("coverage_status", "partial")
            status_label = "FULL" if status == "full" else "GAP" if status == "gap" else "PARTIAL"

            rows.append(
                f"| {gc.get('category', '').capitalize()} | "
                f"{gc.get('cn_codes_covered', 0)}/{gc.get('cn_codes_total', 0)} | "
                f"{gc.get('suppliers_mapped', 0)} | "
                f"{gc.get('data_quality', 'N/A').capitalize()} | "
                f"{status_label} |"
            )

        return header + "\n".join(rows)

    def _md_provenance_footer(self) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: ComplianceStatusTemplate v1.0*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Build HTML header."""
        score = data.get("compliance_score", 0.0)
        threshold = self.config.get("score_threshold_pass", 80)
        compliant = score >= threshold
        status = "COMPLIANT" if compliant else "NON-COMPLIANT"
        color = "#2ecc71" if compliant else "#e74c3c"

        return (
            '<div class="report-header">'
            '<h1>CBAM Compliance Status Dashboard</h1>'
            '<div class="header-meta">'
            f'<div class="meta-item" style="background:{color}">'
            f'<strong>{status}</strong></div>'
            f'<div class="meta-item">Score: {self._format_number(score, 0)}/100</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div></div>'
        )

    def _html_regulatory_timeline(self) -> str:
        """Build HTML regulatory timeline."""
        show = self.config.get("show_regulatory_timeline", True)
        if not show:
            return ""

        today = datetime.utcnow().strftime("%Y-%m-%d")

        rows_html = ""
        for entry in CBAM_REGULATORY_TIMELINE:
            date_str = entry["date"]
            is_past = date_str <= today
            color = "#95a5a6" if is_past else "#1a5276"
            status = "Past" if is_past else "Upcoming"

            rows_html += (
                f'<tr style="color:{color}">'
                f'<td>{date_str}</td>'
                f'<td>{entry["event"]}</td>'
                f'<td>{entry["phase"].capitalize()}</td>'
                f'<td>{status}</td></tr>'
            )

        return (
            '<div class="section"><h2>Regulatory Timeline</h2>'
            '<table><thead><tr>'
            '<th>Date</th><th>Event</th><th>Phase</th><th>Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_compliance_score(self, data: Dict[str, Any]) -> str:
        """Build HTML compliance score gauge."""
        score = data.get("compliance_score", 0.0)
        threshold = self.config.get("score_threshold_pass", 80)
        color = "#2ecc71" if score >= threshold else "#e74c3c" if score < 50 else "#f39c12"

        sub_scores: List[Dict[str, Any]] = data.get("sub_scores", [])

        sub_html = ""
        for ss in sub_scores:
            ss_score = ss.get("score", 0.0)
            ss_color = "#2ecc71" if ss_score >= 80 else "#f39c12" if ss_score >= 50 else "#e74c3c"
            sub_html += (
                f'<div class="sub-score">'
                f'<div class="sub-label">{ss.get("component", "")}</div>'
                f'<div class="progress-bar">'
                f'<div class="progress-fill" style="width:{ss_score}%;background:{ss_color}"></div>'
                f'</div>'
                f'<div class="sub-value">{self._format_number(ss_score, 1)}</div>'
                f'</div>'
            )

        return (
            '<div class="section"><h2>Compliance Score</h2>'
            f'<div class="score-gauge">'
            f'<div class="gauge-circle" style="border-color:{color}">'
            f'<span class="gauge-value">{self._format_number(score, 0)}</span>'
            f'<span class="gauge-label">out of 100</span></div>'
            f'<div class="gauge-threshold">Pass threshold: {threshold}</div>'
            f'</div>'
            f'<div class="sub-scores">{sub_html}</div></div>'
        )

    def _html_obligation_checklist(self, data: Dict[str, Any]) -> str:
        """Build HTML obligation checklist."""
        obligations: List[Dict[str, Any]] = data.get("obligations", [])

        rows_html = ""
        for obl in obligations:
            status = obl.get("status", "pending")
            completed = status == "completed"
            color = "#2ecc71" if completed else "#f39c12"
            icon = "&#10003;" if completed else "&#9744;"

            rows_html += (
                f'<tr>'
                f'<td style="color:{color};font-size:18px">{icon}</td>'
                f'<td>{obl.get("name", "")}</td>'
                f'<td style="color:{color};font-weight:bold">{status.upper()}</td>'
                f'<td>{self._format_date(obl.get("due_date", "N/A"))}</td>'
                f'<td>{self._format_date(obl.get("last_filed", "N/A"))}</td>'
                f'<td>{obl.get("notes", "")}</td></tr>'
            )

        return (
            '<div class="section"><h2>Obligation Checklist</h2>'
            '<table><thead><tr>'
            '<th></th><th>Obligation</th><th>Status</th>'
            '<th>Due Date</th><th>Last Filed</th><th>Notes</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_filing_deadlines(self, data: Dict[str, Any]) -> str:
        """Build HTML filing deadlines calendar."""
        deadlines: List[Dict[str, Any]] = data.get("deadlines", [])

        rows_html = ""
        for dl in sorted(deadlines, key=lambda x: x.get("date", "")):
            days = dl.get("days_remaining", 0)
            if days <= 7:
                color = "#e74c3c"
                priority = "URGENT"
            elif days <= 30:
                color = "#f39c12"
                priority = "HIGH"
            else:
                color = "#2ecc71"
                priority = "NORMAL"

            rows_html += (
                f'<tr>'
                f'<td>{self._format_date(dl.get("date", ""))}</td>'
                f'<td>{dl.get("description", "")}</td>'
                f'<td class="num" style="color:{color};font-weight:bold">{days} days</td>'
                f'<td><span class="priority-badge" style="background:{color}">'
                f'{priority}</span></td></tr>'
            )

        return (
            '<div class="section"><h2>Filing Deadlines</h2>'
            '<table><thead><tr>'
            '<th>Deadline</th><th>Description</th>'
            '<th>Days Remaining</th><th>Priority</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_risk_indicators(self, data: Dict[str, Any]) -> str:
        """Build HTML risk indicators."""
        risks: List[Dict[str, Any]] = data.get("risk_indicators", [])

        if not risks:
            return (
                '<div class="section"><h2>Risk Indicators</h2>'
                '<p class="note">No risk indicators to display.</p></div>'
            )

        severity_colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}

        cards_html = ""
        for category in self.RISK_CATEGORIES:
            category_risks = [r for r in risks if r.get("category", "") == category]
            if not category_risks:
                continue

            items_html = ""
            for risk in category_risks:
                sev = risk.get("severity", "medium").lower()
                color = severity_colors.get(sev, "#95a5a6")
                items_html += (
                    f'<div class="risk-item">'
                    f'<span class="severity-badge" style="background:{color}">'
                    f'{sev.upper()}</span> '
                    f'{risk.get("description", "")}'
                    f'<div class="risk-detail">'
                    f'Likelihood: {risk.get("likelihood", "N/A").capitalize()} | '
                    f'Impact: {risk.get("impact", "N/A").capitalize()}</div>'
                    f'<div class="risk-mitigation">'
                    f'Mitigation: {risk.get("mitigation", "N/A")}</div>'
                    f'</div>'
                )

            cards_html += (
                f'<div class="risk-category">'
                f'<h3>{category.replace("_", " ").title()} Risks</h3>'
                f'{items_html}</div>'
            )

        return f'<div class="section"><h2>Risk Indicators</h2>{cards_html}</div>'

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Build HTML action items section."""
        actions: List[Dict[str, Any]] = data.get("action_items", [])

        if not actions:
            return (
                '<div class="section"><h2>Action Items</h2>'
                '<p class="note">No action items pending.</p></div>'
            )

        status_config = {
            "overdue": {"label": "Overdue", "color": "#e74c3c"},
            "pending": {"label": "Pending", "color": "#f39c12"},
            "upcoming": {"label": "Upcoming", "color": "#3498db"},
        }

        items_html = ""
        for status_key in ["overdue", "pending", "upcoming"]:
            status_actions = [a for a in actions if a.get("status", "") == status_key]
            if not status_actions:
                continue

            cfg = status_config.get(status_key, {"label": status_key, "color": "#95a5a6"})

            items_html += f'<h3 style="color:{cfg["color"]}">{cfg["label"]}</h3>'
            for a in status_actions:
                extra = ""
                if status_key == "overdue":
                    extra = f' ({a.get("days_overdue", 0)} days overdue)'

                items_html += (
                    f'<div class="action-item" style="border-left:3px solid {cfg["color"]}">'
                    f'<strong>{a.get("title", "")}</strong>'
                    f'<div class="action-due">Due: '
                    f'{self._format_date(a.get("due_date", "N/A"))}{extra}</div>'
                    f'</div>'
                )

        return f'<div class="section"><h2>Action Items</h2>{items_html}</div>'

    def _html_goods_coverage(self, data: Dict[str, Any]) -> str:
        """Build HTML goods category coverage section."""
        coverage: List[Dict[str, Any]] = data.get("goods_coverage", [])

        if not coverage:
            return (
                '<div class="section"><h2>Goods Category Coverage</h2>'
                '<p class="note">No goods coverage data available.</p></div>'
            )

        rows_html = ""
        for gc in coverage:
            status = gc.get("coverage_status", "partial")
            if status == "full":
                color = "#2ecc71"
                label = "FULL"
            elif status == "gap":
                color = "#e74c3c"
                label = "GAP"
            else:
                color = "#f39c12"
                label = "PARTIAL"

            cn_covered = gc.get("cn_codes_covered", 0)
            cn_total = gc.get("cn_codes_total", 0)
            pct = (cn_covered / cn_total * 100) if cn_total > 0 else 0.0

            rows_html += (
                f'<tr>'
                f'<td>{gc.get("category", "").capitalize()}</td>'
                f'<td class="num">{cn_covered}/{cn_total}</td>'
                f'<td class="num">{gc.get("suppliers_mapped", 0)}</td>'
                f'<td>{gc.get("data_quality", "N/A").capitalize()}</td>'
                f'<td><span class="coverage-badge" style="background:{color}">'
                f'{label}</span></td></tr>'
            )

        return (
            '<div class="section"><h2>Goods Category Coverage</h2>'
            '<table><thead><tr>'
            '<th>Category</th><th>CN Codes</th><th>Suppliers</th>'
            '<th>Data Quality</th><th>Coverage</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_compliance_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON compliance score."""
        score = data.get("compliance_score", 0.0)
        threshold = self.config.get("score_threshold_pass", 80)
        return {
            "score": round(score, 1),
            "threshold": threshold,
            "compliant": score >= threshold,
            "sub_scores": data.get("sub_scores", []),
        }

    def _json_obligation_checklist(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON obligation checklist."""
        obligations: List[Dict[str, Any]] = data.get("obligations", [])
        return [
            {
                "name": obl.get("name", ""),
                "status": obl.get("status", "pending"),
                "due_date": self._format_date(obl.get("due_date", "")),
                "last_filed": self._format_date(obl.get("last_filed", "")),
                "notes": obl.get("notes", ""),
            }
            for obl in obligations
        ]

    def _json_filing_deadlines(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON filing deadlines."""
        deadlines: List[Dict[str, Any]] = data.get("deadlines", [])
        return [
            {
                "date": self._format_date(dl.get("date", "")),
                "description": dl.get("description", ""),
                "days_remaining": dl.get("days_remaining", 0),
                "priority": "urgent" if dl.get("days_remaining", 0) <= 7
                else "high" if dl.get("days_remaining", 0) <= 30
                else "normal",
            }
            for dl in sorted(deadlines, key=lambda x: x.get("date", ""))
        ]

    def _json_risk_indicators(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON risk indicators."""
        risks: List[Dict[str, Any]] = data.get("risk_indicators", [])
        return [
            {
                "category": risk.get("category", ""),
                "description": risk.get("description", ""),
                "severity": risk.get("severity", "medium"),
                "likelihood": risk.get("likelihood", ""),
                "impact": risk.get("impact", ""),
                "mitigation": risk.get("mitigation", ""),
            }
            for risk in risks
        ]

    def _json_action_items(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Build JSON action items grouped by status."""
        actions: List[Dict[str, Any]] = data.get("action_items", [])
        result: Dict[str, List[Dict[str, Any]]] = {
            "overdue": [],
            "pending": [],
            "upcoming": [],
        }

        for a in actions:
            status = a.get("status", "pending")
            item = {
                "title": a.get("title", ""),
                "due_date": self._format_date(a.get("due_date", "")),
            }
            if status == "overdue":
                item["days_overdue"] = a.get("days_overdue", 0)
            if status in result:
                result[status].append(item)

        return result

    def _json_goods_coverage(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON goods coverage."""
        coverage: List[Dict[str, Any]] = data.get("goods_coverage", [])
        return [
            {
                "category": gc.get("category", ""),
                "cn_codes_covered": gc.get("cn_codes_covered", 0),
                "cn_codes_total": gc.get("cn_codes_total", 0),
                "suppliers_mapped": gc.get("suppliers_mapped", 0),
                "data_quality": gc.get("data_quality", ""),
                "coverage_status": gc.get("coverage_status", "partial"),
            }
            for gc in coverage
        ]

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _format_number(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format a numeric value with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _format_percentage(self, value: Union[int, float]) -> str:
        """Format a percentage value."""
        return f"{value:.2f}%"

    def _format_date(self, dt: Union[datetime, str]) -> str:
        """Format a datetime to ISO date string."""
        if isinstance(dt, str):
            return dt[:10] if len(dt) >= 10 else dt
        return dt.strftime("%Y-%m-%d")

    def _format_currency(self, value: Union[int, float], currency: str = "EUR") -> str:
        """Format a currency value."""
        return f"{currency} {value:,.2f}"

    def _format_cn_code(self, code: str) -> str:
        """Format a CN code to standard XXXX.XX format."""
        clean = code.replace(".", "").replace(" ", "")
        if len(clean) >= 6:
            return f"{clean[:4]}.{clean[4:6]}"
        elif len(clean) == 4:
            return f"{clean}.00"
        return code

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 12px 0;font-size:24px}"
            ".header-meta{display:flex;flex-wrap:wrap;gap:12px;font-size:14px}"
            ".meta-item{background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            ".section h3{margin:16px 0 8px 0;font-size:15px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".score-gauge{text-align:center;margin:24px 0}"
            ".gauge-circle{display:inline-flex;flex-direction:column;align-items:center;"
            "justify-content:center;width:120px;height:120px;border-radius:50%;"
            "border:6px solid;}"
            ".gauge-value{font-size:36px;font-weight:700;line-height:1}"
            ".gauge-label{font-size:12px;color:#7f8c8d}"
            ".gauge-threshold{font-size:13px;color:#7f8c8d;margin-top:8px}"
            ".sub-scores{margin-top:16px}"
            ".sub-score{display:flex;align-items:center;gap:12px;margin-bottom:6px}"
            ".sub-label{width:200px;font-size:13px}"
            ".sub-value{width:40px;font-size:13px;text-align:right}"
            ".progress-bar{flex:1;background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".priority-badge,.coverage-badge,.severity-badge{"
            "display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".risk-category{margin-bottom:16px}"
            ".risk-item{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".risk-detail{font-size:12px;color:#7f8c8d;margin-top:4px}"
            ".risk-mitigation{font-size:12px;color:#555;margin-top:2px}"
            ".action-item{padding:8px 12px;margin-bottom:8px;background:#f8f9fa;"
            "border-radius:4px}"
            ".action-due{font-size:12px;color:#7f8c8d;margin-top:2px}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )

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
            f'Template: ComplianceStatusTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
