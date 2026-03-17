"""
UnifiedGapAnalysisReportTemplate - Cross-framework gap inventory and remediation roadmap.

This module implements the UnifiedGapAnalysisReportTemplate for PACK-009
EU Climate Compliance Bundle. It renders a cross-framework gap inventory
sorted by multi-regulation impact score, gap severity breakdowns,
remediation roadmaps with timeline and cost estimates, and
multi-regulation impact matrices.

Example:
    >>> template = UnifiedGapAnalysisReportTemplate()
    >>> data = GapAnalysisData(
    ...     gaps=[...],
    ...     remediation_plan=[...],
    ...     impact_scores={...},
    ... )
    >>> md = template.render(data, fmt="markdown")
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
#  Pydantic data models
# ---------------------------------------------------------------------------

class ComplianceGap(BaseModel):
    """A single compliance gap across one or more regulations."""

    gap_id: str = Field(..., description="Unique gap identifier")
    title: str = Field(..., description="Short gap title")
    description: str = Field("", description="Detailed gap description")
    severity: str = Field("medium", description="critical, high, medium, low")
    category: str = Field("", description="Gap category e.g. Data, Process, Governance")
    regulations_affected: List[str] = Field(
        default_factory=list, description="List of affected regulation codes"
    )
    impact_score: float = Field(0.0, ge=0.0, le=100.0, description="Multi-regulation impact score")
    current_status: str = Field("open", description="open, in_progress, resolved, accepted")
    data_fields_affected: List[str] = Field(default_factory=list, description="Affected data fields")
    estimated_effort_hours: float = Field(0.0, ge=0.0, description="Estimated remediation effort in hours")
    estimated_cost_eur: float = Field(0.0, ge=0.0, description="Estimated cost in EUR")
    owner: str = Field("", description="Responsible team or person")
    due_date: str = Field("", description="Target remediation date ISO string")
    notes: str = Field("", description="Additional notes")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Ensure severity is one of the allowed values."""
        allowed = {"critical", "high", "medium", "low"}
        if v.lower() not in allowed:
            raise ValueError(f"severity must be one of {allowed}, got '{v}'")
        return v.lower()


class RemediationStep(BaseModel):
    """A step in the remediation roadmap."""

    step_id: str = Field(..., description="Step identifier")
    gap_id: str = Field(..., description="Related gap identifier")
    title: str = Field(..., description="Step title")
    description: str = Field("", description="Step description")
    phase: str = Field("", description="Remediation phase e.g. Phase 1, Phase 2")
    start_date: str = Field("", description="Planned start ISO date")
    end_date: str = Field("", description="Planned end ISO date")
    effort_hours: float = Field(0.0, ge=0.0, description="Effort in hours")
    cost_eur: float = Field(0.0, ge=0.0, description="Cost in EUR")
    status: str = Field("planned", description="planned, in_progress, completed")
    dependencies: List[str] = Field(default_factory=list, description="Dependent step IDs")
    owner: str = Field("", description="Responsible team or person")


class GapAnalysisConfig(BaseModel):
    """Configuration for the gap analysis template."""

    title: str = Field(
        "Unified Gap Analysis Report",
        description="Report title",
    )
    sort_by: str = Field("impact_score", description="Sort field: impact_score, severity, regulations_affected")
    show_resolved: bool = Field(False, description="Whether to show resolved gaps")
    severity_weights: Dict[str, float] = Field(
        default_factory=lambda: {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0},
        description="Severity weight multipliers for scoring",
    )


class GapAnalysisData(BaseModel):
    """Input data for the unified gap analysis report."""

    gaps: List[ComplianceGap] = Field(default_factory=list, description="List of compliance gaps")
    remediation_plan: List[RemediationStep] = Field(
        default_factory=list, description="Remediation roadmap steps"
    )
    impact_scores: Dict[str, float] = Field(
        default_factory=dict, description="Per-regulation overall impact scores"
    )
    per_regulation_gap_counts: Dict[str, int] = Field(
        default_factory=dict, description="Gap count per regulation"
    )
    per_regulation_compliance_pct: Dict[str, float] = Field(
        default_factory=dict, description="Compliance % per regulation after gaps"
    )
    total_estimated_cost_eur: float = Field(0.0, ge=0.0, description="Total remediation cost estimate")
    total_estimated_hours: float = Field(0.0, ge=0.0, description="Total remediation effort hours")
    reporting_period: str = Field("", description="Reporting period label")
    organization_name: str = Field("", description="Organization name")

    @field_validator("gaps")
    @classmethod
    def validate_gaps_present(cls, v: List[ComplianceGap]) -> List[ComplianceGap]:
        """Ensure at least one gap is provided."""
        if not v:
            raise ValueError("gaps must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
#  Template class
# ---------------------------------------------------------------------------

class UnifiedGapAnalysisReportTemplate:
    """
    Unified gap analysis report template for cross-regulation compliance gaps.

    Generates gap inventories sorted by multi-regulation impact, severity
    breakdowns, remediation roadmaps with cost/effort estimates, and
    regulation-level impact matrices.

    Attributes:
        config: Template configuration.
        generated_at: ISO timestamp of report generation.
    """

    SEVERITY_COLORS = {
        "critical": {"hex": "#c0392b", "label": "CRITICAL"},
        "high": {"hex": "#e74c3c", "label": "HIGH"},
        "medium": {"hex": "#f39c12", "label": "MEDIUM"},
        "low": {"hex": "#27ae60", "label": "LOW"},
    }

    STATUS_ICONS = {
        "open": "[O]",
        "in_progress": "[~]",
        "resolved": "[X]",
        "accepted": "[A]",
        "planned": "[P]",
        "completed": "[D]",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize UnifiedGapAnalysisReportTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        raw = config or {}
        self.config = GapAnalysisConfig(**raw) if raw else GapAnalysisConfig()
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def render(self, data: GapAnalysisData, fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render the gap analysis report in the specified format.

        Args:
            data: Validated GapAnalysisData input.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered output.

        Raises:
            ValueError: If fmt is unsupported.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    def render_markdown(self, data: GapAnalysisData) -> str:
        """Render as Markdown string."""
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_severity_breakdown(data),
            self._md_per_regulation_status(data),
            self._md_gap_inventory(data),
            self._md_impact_matrix(data),
            self._md_remediation_roadmap(data),
            self._md_cost_summary(data),
            self._md_footer(),
        ]
        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance} -->"
        return content

    def render_html(self, data: GapAnalysisData) -> str:
        """Render as self-contained HTML document."""
        sections: List[str] = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_severity_breakdown(data),
            self._html_per_regulation_status(data),
            self._html_gap_inventory(data),
            self._html_impact_matrix(data),
            self._html_remediation_roadmap(data),
            self._html_cost_summary(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(body)
        return self._wrap_html(body, provenance)

    def render_json(self, data: GapAnalysisData) -> Dict[str, Any]:
        """Render as structured dictionary."""
        report: Dict[str, Any] = {
            "report_type": "unified_gap_analysis",
            "template_version": "1.0",
            "generated_at": self.generated_at,
            "organization": data.organization_name,
            "reporting_period": data.reporting_period,
            "executive_summary": self._json_executive_summary(data),
            "severity_breakdown": self._json_severity_breakdown(data),
            "per_regulation_status": self._json_per_regulation_status(data),
            "gap_inventory": self._json_gap_inventory(data),
            "impact_matrix": self._json_impact_matrix(data),
            "remediation_roadmap": self._json_remediation_roadmap(data),
            "cost_summary": self._json_cost_summary(data),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Computation helpers
    # ------------------------------------------------------------------ #

    def _filtered_gaps(self, data: GapAnalysisData) -> List[ComplianceGap]:
        """Return gaps filtered and sorted according to config."""
        gaps = data.gaps
        if not self.config.show_resolved:
            gaps = [g for g in gaps if g.current_status != "resolved"]
        if self.config.sort_by == "impact_score":
            gaps = sorted(gaps, key=lambda g: g.impact_score, reverse=True)
        elif self.config.sort_by == "severity":
            order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            gaps = sorted(gaps, key=lambda g: order.get(g.severity, 4))
        elif self.config.sort_by == "regulations_affected":
            gaps = sorted(gaps, key=lambda g: len(g.regulations_affected), reverse=True)
        return gaps

    def _severity_counts(self, data: GapAnalysisData) -> Dict[str, int]:
        """Count gaps by severity."""
        counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for g in data.gaps:
            sev = g.severity.lower()
            if sev in counts:
                counts[sev] += 1
        return counts

    # ------------------------------------------------------------------ #
    #  Markdown builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: GapAnalysisData) -> str:
        """Build markdown header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        return (
            f"# {self.config.title}\n\n"
            f"**Organization:** {org}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_executive_summary(self, data: GapAnalysisData) -> str:
        """Build executive summary section."""
        total = len(data.gaps)
        counts = self._severity_counts(data)
        open_gaps = sum(1 for g in data.gaps if g.current_status == "open")
        in_progress = sum(1 for g in data.gaps if g.current_status == "in_progress")
        resolved = sum(1 for g in data.gaps if g.current_status == "resolved")
        regs_affected = len(set(r for g in data.gaps for r in g.regulations_affected))
        return (
            "## Executive Summary\n\n"
            f"- **Total Gaps Identified:** {total}\n"
            f"- **Critical:** {counts['critical']} | **High:** {counts['high']} | "
            f"**Medium:** {counts['medium']} | **Low:** {counts['low']}\n"
            f"- **Open:** {open_gaps} | **In Progress:** {in_progress} | "
            f"**Resolved:** {resolved}\n"
            f"- **Regulations Affected:** {regs_affected}\n"
            f"- **Total Estimated Cost:** EUR {data.total_estimated_cost_eur:,.0f}\n"
            f"- **Total Estimated Effort:** {data.total_estimated_hours:,.0f} hours"
        )

    def _md_severity_breakdown(self, data: GapAnalysisData) -> str:
        """Build severity breakdown section."""
        counts = self._severity_counts(data)
        total = len(data.gaps) or 1
        header = (
            "## Severity Breakdown\n\n"
            "| Severity | Count | Percentage | Bar |\n"
            "|----------|-------|------------|-----|\n"
        )
        rows: List[str] = []
        for sev in ["critical", "high", "medium", "low"]:
            count = counts[sev]
            pct = (count / total) * 100
            bar_len = int(pct / 5)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            rows.append(f"| {sev.upper()} | {count} | {pct:.1f}% | `[{bar}]` |")
        return header + "\n".join(rows)

    def _md_per_regulation_status(self, data: GapAnalysisData) -> str:
        """Build per-regulation gap status table."""
        if not data.per_regulation_gap_counts and not data.per_regulation_compliance_pct:
            return ""
        header = (
            "## Per-Regulation Status\n\n"
            "| Regulation | Gap Count | Compliance % | Impact Score |\n"
            "|------------|-----------|-------------|-------------|\n"
        )
        all_regs = sorted(
            set(list(data.per_regulation_gap_counts.keys())
                + list(data.per_regulation_compliance_pct.keys())
                + list(data.impact_scores.keys()))
        )
        rows: List[str] = []
        for reg in all_regs:
            gap_count = data.per_regulation_gap_counts.get(reg, 0)
            compliance = data.per_regulation_compliance_pct.get(reg, 0.0)
            impact = data.impact_scores.get(reg, 0.0)
            rows.append(
                f"| {reg} | {gap_count} | {compliance:.1f}% | {impact:.1f} |"
            )
        return header + "\n".join(rows)

    def _md_gap_inventory(self, data: GapAnalysisData) -> str:
        """Build detailed gap inventory table."""
        gaps = self._filtered_gaps(data)
        header = (
            "## Gap Inventory\n\n"
            "| ID | Title | Severity | Regulations | Impact | Status | Owner |\n"
            "|----|-------|----------|-------------|--------|--------|-------|\n"
        )
        rows: List[str] = []
        for g in gaps:
            regs = ", ".join(g.regulations_affected) if g.regulations_affected else "N/A"
            icon = self.STATUS_ICONS.get(g.current_status, "[?]")
            rows.append(
                f"| {g.gap_id} | {g.title} | {g.severity.upper()} | "
                f"{regs} | {g.impact_score:.1f} | "
                f"{icon} {g.current_status.replace('_', ' ').title()} | "
                f"{g.owner or 'Unassigned'} |"
            )
        return header + "\n".join(rows)

    def _md_impact_matrix(self, data: GapAnalysisData) -> str:
        """Build multi-regulation impact matrix."""
        all_regs = sorted(set(r for g in data.gaps for r in g.regulations_affected))
        if not all_regs:
            return ""
        header = "## Multi-Regulation Impact Matrix\n\n"
        header += "Gap ID | " + " | ".join(all_regs) + " |\n"
        header += "-------|" + "|".join("-------" for _ in all_regs) + "|\n"
        rows: List[str] = []
        gaps = self._filtered_gaps(data)
        for g in gaps[:20]:
            cells: List[str] = []
            for reg in all_regs:
                cells.append("X" if reg in g.regulations_affected else "-")
            rows.append(f"| {g.gap_id} | " + " | ".join(cells) + " |")
        return header + "\n".join(rows)

    def _md_remediation_roadmap(self, data: GapAnalysisData) -> str:
        """Build remediation roadmap section."""
        if not data.remediation_plan:
            return "## Remediation Roadmap\n\n*No remediation steps defined.*"
        header = (
            "## Remediation Roadmap\n\n"
            "| Phase | Step | Gap | Start | End | Effort (h) | Cost (EUR) | Status |\n"
            "|-------|------|-----|-------|-----|------------|------------|--------|\n"
        )
        rows: List[str] = []
        sorted_steps = sorted(data.remediation_plan, key=lambda s: (s.phase, s.start_date))
        for step in sorted_steps:
            icon = self.STATUS_ICONS.get(step.status, "[?]")
            rows.append(
                f"| {step.phase or 'N/A'} | {step.title} | {step.gap_id} | "
                f"{step.start_date[:10] if step.start_date else 'TBD'} | "
                f"{step.end_date[:10] if step.end_date else 'TBD'} | "
                f"{step.effort_hours:,.0f} | {step.cost_eur:,.0f} | "
                f"{icon} {step.status.replace('_', ' ').title()} |"
            )
        return header + "\n".join(rows)

    def _md_cost_summary(self, data: GapAnalysisData) -> str:
        """Build cost summary section."""
        total_cost = data.total_estimated_cost_eur
        total_hours = data.total_estimated_hours
        by_severity: Dict[str, float] = {"critical": 0.0, "high": 0.0, "medium": 0.0, "low": 0.0}
        for g in data.gaps:
            sev = g.severity.lower()
            if sev in by_severity:
                by_severity[sev] += g.estimated_cost_eur
        header = (
            "## Cost & Effort Summary\n\n"
            f"**Total Estimated Cost:** EUR {total_cost:,.0f}\n\n"
            f"**Total Estimated Effort:** {total_hours:,.0f} hours\n\n"
            "| Severity | Estimated Cost (EUR) | % of Total |\n"
            "|----------|---------------------|------------|\n"
        )
        rows: List[str] = []
        for sev in ["critical", "high", "medium", "low"]:
            cost = by_severity[sev]
            pct = (cost / total_cost * 100) if total_cost > 0 else 0.0
            rows.append(f"| {sev.upper()} | {cost:,.0f} | {pct:.1f}% |")
        return header + "\n".join(rows)

    def _md_footer(self) -> str:
        """Build markdown footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            "*Template: UnifiedGapAnalysisReportTemplate v1.0 | PACK-009*"
        )

    # ------------------------------------------------------------------ #
    #  HTML builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: GapAnalysisData) -> str:
        """Build HTML header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        return (
            '<div class="report-header">'
            f'<h1>{self.config.title}</h1>'
            '<div class="header-meta">'
            f'<div class="meta-item">Organization: {org}</div>'
            f'<div class="meta-item">Period: {period}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div></div>'
        )

    def _html_executive_summary(self, data: GapAnalysisData) -> str:
        """Build HTML executive summary."""
        total = len(data.gaps)
        counts = self._severity_counts(data)
        open_g = sum(1 for g in data.gaps if g.current_status == "open")
        in_prog = sum(1 for g in data.gaps if g.current_status == "in_progress")
        resolved = sum(1 for g in data.gaps if g.current_status == "resolved")
        cards = (
            f'<div class="stat-card"><span class="stat-val">{total}</span>'
            f'<span class="stat-lbl">Total Gaps</span></div>'
            f'<div class="stat-card" style="border-top:3px solid #c0392b">'
            f'<span class="stat-val">{counts["critical"]}</span>'
            f'<span class="stat-lbl">Critical</span></div>'
            f'<div class="stat-card" style="border-top:3px solid #e74c3c">'
            f'<span class="stat-val">{counts["high"]}</span>'
            f'<span class="stat-lbl">High</span></div>'
            f'<div class="stat-card"><span class="stat-val">{open_g}</span>'
            f'<span class="stat-lbl">Open</span></div>'
            f'<div class="stat-card"><span class="stat-val">{in_prog}</span>'
            f'<span class="stat-lbl">In Progress</span></div>'
            f'<div class="stat-card"><span class="stat-val">{resolved}</span>'
            f'<span class="stat-lbl">Resolved</span></div>'
            f'<div class="stat-card"><span class="stat-val">EUR {data.total_estimated_cost_eur:,.0f}</span>'
            f'<span class="stat-lbl">Est. Cost</span></div>'
        )
        return f'<div class="section"><h2>Executive Summary</h2><div class="stat-grid">{cards}</div></div>'

    def _html_severity_breakdown(self, data: GapAnalysisData) -> str:
        """Build HTML severity breakdown with progress bars."""
        counts = self._severity_counts(data)
        total = len(data.gaps) or 1
        bars = ""
        for sev in ["critical", "high", "medium", "low"]:
            count = counts[sev]
            pct = (count / total) * 100
            color = self.SEVERITY_COLORS[sev]["hex"]
            bars += (
                f'<div class="sev-row">'
                f'<span class="sev-label" style="color:{color}">{sev.upper()}</span>'
                f'<span class="sev-count">{count}</span>'
                f'<div class="progress-bar"><div class="progress-fill" '
                f'style="width:{pct:.0f}%;background:{color}"></div></div>'
                f'<span class="sev-pct">{pct:.1f}%</span>'
                f'</div>'
            )
        return f'<div class="section"><h2>Severity Breakdown</h2>{bars}</div>'

    def _html_per_regulation_status(self, data: GapAnalysisData) -> str:
        """Build HTML per-regulation status table."""
        all_regs = sorted(
            set(list(data.per_regulation_gap_counts.keys())
                + list(data.per_regulation_compliance_pct.keys())
                + list(data.impact_scores.keys()))
        )
        if not all_regs:
            return ""
        rows = ""
        for reg in all_regs:
            gap_count = data.per_regulation_gap_counts.get(reg, 0)
            compliance = data.per_regulation_compliance_pct.get(reg, 0.0)
            impact = data.impact_scores.get(reg, 0.0)
            comp_color = "#2ecc71" if compliance >= 80 else "#f39c12" if compliance >= 50 else "#e74c3c"
            rows += (
                f'<tr><td><strong>{reg}</strong></td>'
                f'<td class="num">{gap_count}</td>'
                f'<td class="num" style="color:{comp_color}">{compliance:.1f}%</td>'
                f'<td class="num">{impact:.1f}</td></tr>'
            )
        return (
            '<div class="section"><h2>Per-Regulation Status</h2>'
            '<table><thead><tr><th>Regulation</th><th>Gaps</th>'
            '<th>Compliance</th><th>Impact</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_gap_inventory(self, data: GapAnalysisData) -> str:
        """Build HTML gap inventory table."""
        gaps = self._filtered_gaps(data)
        rows = ""
        for g in gaps:
            color = self.SEVERITY_COLORS.get(g.severity, {"hex": "#95a5a6"})["hex"]
            regs = ", ".join(g.regulations_affected) if g.regulations_affected else "N/A"
            rows += (
                f'<tr>'
                f'<td>{g.gap_id}</td>'
                f'<td>{g.title}</td>'
                f'<td><span class="sev-badge" style="background:{color}">'
                f'{g.severity.upper()}</span></td>'
                f'<td>{regs}</td>'
                f'<td class="num">{g.impact_score:.1f}</td>'
                f'<td>{g.current_status.replace("_", " ").title()}</td>'
                f'<td>{g.owner or "Unassigned"}</td>'
                f'</tr>'
            )
        return (
            '<div class="section"><h2>Gap Inventory</h2>'
            '<table><thead><tr>'
            '<th>ID</th><th>Title</th><th>Severity</th>'
            '<th>Regulations</th><th>Impact</th><th>Status</th><th>Owner</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_impact_matrix(self, data: GapAnalysisData) -> str:
        """Build HTML multi-regulation impact matrix."""
        all_regs = sorted(set(r for g in data.gaps for r in g.regulations_affected))
        if not all_regs:
            return ""
        header_cells = "".join(f"<th>{r}</th>" for r in all_regs)
        rows = ""
        gaps = self._filtered_gaps(data)
        for g in gaps[:20]:
            cells = ""
            for reg in all_regs:
                if reg in g.regulations_affected:
                    cells += '<td style="text-align:center;background:#e8f8f5;font-weight:bold">X</td>'
                else:
                    cells += '<td style="text-align:center;color:#bdc3c7">-</td>'
            rows += f'<tr><td>{g.gap_id}</td><td>{g.title}</td>{cells}</tr>'
        return (
            '<div class="section"><h2>Multi-Regulation Impact Matrix</h2>'
            f'<table><thead><tr><th>ID</th><th>Gap</th>{header_cells}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )

    def _html_remediation_roadmap(self, data: GapAnalysisData) -> str:
        """Build HTML remediation roadmap."""
        if not data.remediation_plan:
            return (
                '<div class="section"><h2>Remediation Roadmap</h2>'
                '<p class="note">No remediation steps defined.</p></div>'
            )
        sorted_steps = sorted(data.remediation_plan, key=lambda s: (s.phase, s.start_date))
        rows = ""
        for step in sorted_steps:
            status_color = "#2ecc71" if step.status == "completed" else "#f39c12" if step.status == "in_progress" else "#3498db"
            rows += (
                f'<tr>'
                f'<td>{step.phase or "N/A"}</td>'
                f'<td>{step.title}</td>'
                f'<td>{step.gap_id}</td>'
                f'<td>{step.start_date[:10] if step.start_date else "TBD"}</td>'
                f'<td>{step.end_date[:10] if step.end_date else "TBD"}</td>'
                f'<td class="num">{step.effort_hours:,.0f}</td>'
                f'<td class="num">{step.cost_eur:,.0f}</td>'
                f'<td><span class="status-badge" style="background:{status_color}">'
                f'{step.status.replace("_", " ").title()}</span></td>'
                f'</tr>'
            )
        return (
            '<div class="section"><h2>Remediation Roadmap</h2>'
            '<table><thead><tr>'
            '<th>Phase</th><th>Step</th><th>Gap</th><th>Start</th>'
            '<th>End</th><th>Effort</th><th>Cost</th><th>Status</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_cost_summary(self, data: GapAnalysisData) -> str:
        """Build HTML cost summary."""
        total_cost = data.total_estimated_cost_eur or 1.0
        by_severity: Dict[str, float] = {"critical": 0.0, "high": 0.0, "medium": 0.0, "low": 0.0}
        for g in data.gaps:
            sev = g.severity.lower()
            if sev in by_severity:
                by_severity[sev] += g.estimated_cost_eur
        rows = ""
        for sev in ["critical", "high", "medium", "low"]:
            cost = by_severity[sev]
            pct = (cost / total_cost * 100) if total_cost > 0 else 0.0
            color = self.SEVERITY_COLORS[sev]["hex"]
            rows += (
                f'<tr><td style="color:{color};font-weight:bold">{sev.upper()}</td>'
                f'<td class="num">EUR {cost:,.0f}</td>'
                f'<td class="num">{pct:.1f}%</td></tr>'
            )
        return (
            '<div class="section"><h2>Cost &amp; Effort Summary</h2>'
            f'<p><strong>Total Estimated Cost:</strong> EUR {data.total_estimated_cost_eur:,.0f}</p>'
            f'<p><strong>Total Estimated Effort:</strong> {data.total_estimated_hours:,.0f} hours</p>'
            '<table><thead><tr><th>Severity</th><th>Est. Cost</th><th>% of Total</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON builders
    # ------------------------------------------------------------------ #

    def _json_executive_summary(self, data: GapAnalysisData) -> Dict[str, Any]:
        """Build JSON executive summary."""
        counts = self._severity_counts(data)
        return {
            "total_gaps": len(data.gaps),
            "severity_counts": counts,
            "open": sum(1 for g in data.gaps if g.current_status == "open"),
            "in_progress": sum(1 for g in data.gaps if g.current_status == "in_progress"),
            "resolved": sum(1 for g in data.gaps if g.current_status == "resolved"),
            "total_estimated_cost_eur": round(data.total_estimated_cost_eur, 2),
            "total_estimated_hours": round(data.total_estimated_hours, 2),
        }

    def _json_severity_breakdown(self, data: GapAnalysisData) -> Dict[str, Any]:
        """Build JSON severity breakdown."""
        counts = self._severity_counts(data)
        total = len(data.gaps) or 1
        return {
            sev: {"count": c, "percentage": round((c / total) * 100, 1)}
            for sev, c in counts.items()
        }

    def _json_per_regulation_status(self, data: GapAnalysisData) -> Dict[str, Dict[str, Any]]:
        """Build JSON per-regulation status."""
        all_regs = sorted(
            set(list(data.per_regulation_gap_counts.keys())
                + list(data.per_regulation_compliance_pct.keys())
                + list(data.impact_scores.keys()))
        )
        return {
            reg: {
                "gap_count": data.per_regulation_gap_counts.get(reg, 0),
                "compliance_pct": round(data.per_regulation_compliance_pct.get(reg, 0.0), 1),
                "impact_score": round(data.impact_scores.get(reg, 0.0), 1),
            }
            for reg in all_regs
        }

    def _json_gap_inventory(self, data: GapAnalysisData) -> List[Dict[str, Any]]:
        """Build JSON gap inventory."""
        gaps = self._filtered_gaps(data)
        return [
            {
                "gap_id": g.gap_id,
                "title": g.title,
                "description": g.description,
                "severity": g.severity,
                "category": g.category,
                "regulations_affected": g.regulations_affected,
                "impact_score": round(g.impact_score, 1),
                "current_status": g.current_status,
                "estimated_effort_hours": round(g.estimated_effort_hours, 1),
                "estimated_cost_eur": round(g.estimated_cost_eur, 2),
                "owner": g.owner,
                "due_date": g.due_date,
            }
            for g in gaps
        ]

    def _json_impact_matrix(self, data: GapAnalysisData) -> Dict[str, Any]:
        """Build JSON impact matrix."""
        all_regs = sorted(set(r for g in data.gaps for r in g.regulations_affected))
        matrix: Dict[str, Dict[str, bool]] = {}
        for g in data.gaps:
            matrix[g.gap_id] = {reg: reg in g.regulations_affected for reg in all_regs}
        return {"regulations": all_regs, "gap_regulation_map": matrix}

    def _json_remediation_roadmap(self, data: GapAnalysisData) -> List[Dict[str, Any]]:
        """Build JSON remediation roadmap."""
        sorted_steps = sorted(data.remediation_plan, key=lambda s: (s.phase, s.start_date))
        return [
            {
                "step_id": s.step_id,
                "gap_id": s.gap_id,
                "title": s.title,
                "phase": s.phase,
                "start_date": s.start_date,
                "end_date": s.end_date,
                "effort_hours": round(s.effort_hours, 1),
                "cost_eur": round(s.cost_eur, 2),
                "status": s.status,
                "dependencies": s.dependencies,
                "owner": s.owner,
            }
            for s in sorted_steps
        ]

    def _json_cost_summary(self, data: GapAnalysisData) -> Dict[str, Any]:
        """Build JSON cost summary."""
        by_severity: Dict[str, float] = {"critical": 0.0, "high": 0.0, "medium": 0.0, "low": 0.0}
        for g in data.gaps:
            sev = g.severity.lower()
            if sev in by_severity:
                by_severity[sev] += g.estimated_cost_eur
        return {
            "total_cost_eur": round(data.total_estimated_cost_eur, 2),
            "total_hours": round(data.total_estimated_hours, 2),
            "cost_by_severity": {k: round(v, 2) for k, v in by_severity.items()},
        }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, body: str, provenance_hash: str) -> str:
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
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".stat-grid{display:flex;flex-wrap:wrap;gap:12px}"
            ".stat-card{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;"
            "min-width:100px;flex:1}"
            ".stat-val{display:block;font-size:24px;font-weight:700;color:#1a5276}"
            ".stat-lbl{display:block;font-size:11px;color:#7f8c8d;margin-top:4px}"
            ".sev-badge,.status-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".sev-row{display:flex;align-items:center;gap:12px;margin-bottom:8px}"
            ".sev-label{width:80px;font-weight:600;font-size:13px}"
            ".sev-count{width:30px;text-align:right;font-size:14px;font-weight:600}"
            ".sev-pct{width:50px;text-align:right;font-size:13px;color:#7f8c8d}"
            ".progress-bar{flex:1;background:#ecf0f1;border-radius:4px;height:14px;"
            "overflow:hidden}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{self.config.title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: UnifiedGapAnalysisReportTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
