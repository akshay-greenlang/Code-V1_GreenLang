"""
RegulatoryCalendarReportTemplate - Timeline showing deadlines per regulation.

This module implements the RegulatoryCalendarReportTemplate for PACK-009
EU Climate Compliance Bundle. It renders a Gantt-style timeline of regulatory
deadlines, cross-regulation dependencies, urgency-sorted upcoming deadlines,
milestone tracking, and calendar conflict detection.

Example:
    >>> template = RegulatoryCalendarReportTemplate()
    >>> data = CalendarData(
    ...     events=[...],
    ...     dependencies=[...],
    ...     conflicts=[...],
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

class CalendarEvent(BaseModel):
    """A single regulatory calendar event."""

    event_id: str = Field(..., description="Unique event identifier")
    regulation: str = Field(..., description="Regulation code e.g. CSRD, CBAM")
    title: str = Field(..., description="Event title")
    description: str = Field("", description="Event description")
    date: str = Field(..., description="Event date ISO string YYYY-MM-DD")
    end_date: str = Field("", description="End date for multi-day events")
    event_type: str = Field("deadline", description="deadline, milestone, filing, review, audit")
    priority: str = Field("normal", description="urgent, high, normal, low")
    days_remaining: int = Field(0, description="Days until event")
    status: str = Field("upcoming", description="upcoming, imminent, overdue, completed")
    owner: str = Field("", description="Responsible team or person")
    notes: str = Field("", description="Additional notes")


class CalendarDependency(BaseModel):
    """A dependency between two calendar events."""

    source_event_id: str = Field(..., description="Predecessor event ID")
    target_event_id: str = Field(..., description="Dependent event ID")
    dependency_type: str = Field("finish_to_start", description="finish_to_start, start_to_start")
    lag_days: int = Field(0, description="Lag between events in days")
    description: str = Field("", description="Dependency description")
    is_cross_regulation: bool = Field(False, description="Whether this spans regulations")


class CalendarConflict(BaseModel):
    """A scheduling conflict between events."""

    conflict_id: str = Field(..., description="Conflict identifier")
    event_ids: List[str] = Field(default_factory=list, description="IDs of conflicting events")
    conflict_type: str = Field("overlap", description="overlap, resource, dependency_violation")
    severity: str = Field("medium", description="high, medium, low")
    description: str = Field("", description="Conflict description")
    resolution: str = Field("", description="Suggested resolution")
    resolved: bool = Field(False, description="Whether conflict has been resolved")


class MilestoneStatus(BaseModel):
    """Tracking status for a regulatory milestone."""

    regulation: str = Field(..., description="Regulation code")
    milestone: str = Field(..., description="Milestone name")
    target_date: str = Field(..., description="Target date ISO string")
    actual_date: str = Field("", description="Actual completion date if completed")
    status: str = Field("pending", description="pending, on_track, at_risk, completed, missed")
    completion_pct: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")


class CalendarConfig(BaseModel):
    """Configuration for the calendar template."""

    title: str = Field(
        "Regulatory Calendar Report",
        description="Report title",
    )
    gantt_width: int = Field(60, description="Width of Gantt chart in characters for markdown")
    show_completed: bool = Field(True, description="Whether to show completed events")
    urgency_threshold_days: int = Field(30, description="Days threshold for urgency highlighting")


class CalendarData(BaseModel):
    """Input data for the regulatory calendar report."""

    events: List[CalendarEvent] = Field(default_factory=list, description="Calendar events")
    dependencies: List[CalendarDependency] = Field(
        default_factory=list, description="Event dependencies"
    )
    conflicts: List[CalendarConflict] = Field(
        default_factory=list, description="Calendar conflicts"
    )
    milestones: List[MilestoneStatus] = Field(
        default_factory=list, description="Milestone tracking entries"
    )
    calendar_start: str = Field("", description="Calendar range start date")
    calendar_end: str = Field("", description="Calendar range end date")
    reporting_period: str = Field("", description="Reporting period label")
    organization_name: str = Field("", description="Organization name")

    @field_validator("events")
    @classmethod
    def validate_events_present(cls, v: List[CalendarEvent]) -> List[CalendarEvent]:
        """Ensure at least one event is present."""
        if not v:
            raise ValueError("events must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
#  Template class
# ---------------------------------------------------------------------------

class RegulatoryCalendarReportTemplate:
    """
    Regulatory calendar report template with Gantt-style timeline.

    Generates calendar reports showing per-regulation deadlines,
    cross-regulation dependencies, urgency-sorted upcoming items,
    milestone tracking, and conflict detection.

    Attributes:
        config: Template configuration.
        generated_at: ISO timestamp of report generation.
    """

    PRIORITY_MARKERS = {
        "urgent": {"md": "[!!!]", "color": "#c0392b"},
        "high": {"md": "[!!]", "color": "#e74c3c"},
        "normal": {"md": "[*]", "color": "#3498db"},
        "low": {"md": "[.]", "color": "#95a5a6"},
    }

    STATUS_MARKERS = {
        "upcoming": {"md": "[ ]", "color": "#3498db"},
        "imminent": {"md": "[!]", "color": "#f39c12"},
        "overdue": {"md": "[X]", "color": "#e74c3c"},
        "completed": {"md": "[v]", "color": "#2ecc71"},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize RegulatoryCalendarReportTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        raw = config or {}
        self.config = CalendarConfig(**raw) if raw else CalendarConfig()
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def render(self, data: CalendarData, fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render the calendar report in the specified format.

        Args:
            data: Validated CalendarData input.
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

    def render_markdown(self, data: CalendarData) -> str:
        """Render as Markdown string."""
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_gantt_chart(data),
            self._md_upcoming_deadlines(data),
            self._md_milestone_tracking(data),
            self._md_dependencies(data),
            self._md_conflicts(data),
            self._md_footer(),
        ]
        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance} -->"
        return content

    def render_html(self, data: CalendarData) -> str:
        """Render as self-contained HTML document."""
        sections: List[str] = [
            self._html_header(data),
            self._html_summary(data),
            self._html_gantt_chart(data),
            self._html_upcoming_deadlines(data),
            self._html_milestone_tracking(data),
            self._html_dependencies(data),
            self._html_conflicts(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(body)
        return self._wrap_html(body, provenance)

    def render_json(self, data: CalendarData) -> Dict[str, Any]:
        """Render as structured dictionary."""
        report: Dict[str, Any] = {
            "report_type": "regulatory_calendar",
            "template_version": "1.0",
            "generated_at": self.generated_at,
            "organization": data.organization_name,
            "reporting_period": data.reporting_period,
            "summary": self._json_summary(data),
            "events": self._json_events(data),
            "milestones": self._json_milestones(data),
            "dependencies": self._json_dependencies(data),
            "conflicts": self._json_conflicts(data),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Computation helpers
    # ------------------------------------------------------------------ #

    def _filtered_events(self, data: CalendarData) -> List[CalendarEvent]:
        """Return events filtered by config settings."""
        events = data.events
        if not self.config.show_completed:
            events = [e for e in events if e.status != "completed"]
        return sorted(events, key=lambda e: e.date)

    def _events_by_regulation(self, data: CalendarData) -> Dict[str, List[CalendarEvent]]:
        """Group events by regulation."""
        by_reg: Dict[str, List[CalendarEvent]] = {}
        for event in self._filtered_events(data):
            by_reg.setdefault(event.regulation, []).append(event)
        return by_reg

    def _event_lookup(self, data: CalendarData) -> Dict[str, CalendarEvent]:
        """Build an event lookup by ID."""
        return {e.event_id: e for e in data.events}

    # ------------------------------------------------------------------ #
    #  Markdown builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: CalendarData) -> str:
        """Build markdown header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        cal_range = ""
        if data.calendar_start and data.calendar_end:
            cal_range = f"\n\n**Calendar Range:** {data.calendar_start[:10]} to {data.calendar_end[:10]}"
        return (
            f"# {self.config.title}\n\n"
            f"**Organization:** {org}\n\n"
            f"**Reporting Period:** {period}"
            f"{cal_range}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_summary(self, data: CalendarData) -> str:
        """Build summary statistics."""
        total = len(data.events)
        overdue = sum(1 for e in data.events if e.status == "overdue")
        imminent = sum(1 for e in data.events if e.status == "imminent")
        upcoming = sum(1 for e in data.events if e.status == "upcoming")
        completed = sum(1 for e in data.events if e.status == "completed")
        regs = len(set(e.regulation for e in data.events))
        conflicts_open = sum(1 for c in data.conflicts if not c.resolved)
        return (
            "## Summary\n\n"
            f"- **Total Events:** {total}\n"
            f"- **Overdue:** {overdue}\n"
            f"- **Imminent (within {self.config.urgency_threshold_days} days):** {imminent}\n"
            f"- **Upcoming:** {upcoming}\n"
            f"- **Completed:** {completed}\n"
            f"- **Regulations:** {regs}\n"
            f"- **Dependencies:** {len(data.dependencies)}\n"
            f"- **Open Conflicts:** {conflicts_open}"
        )

    def _md_gantt_chart(self, data: CalendarData) -> str:
        """Build a text-based Gantt chart by regulation."""
        by_reg = self._events_by_regulation(data)
        if not by_reg:
            return ""
        all_events = self._filtered_events(data)
        if not all_events:
            return ""
        all_dates = [e.date[:10] for e in all_events]
        min_date = min(all_dates)
        max_date = max(all_dates)
        width = self.config.gantt_width
        section = "## Regulatory Timeline (Gantt)\n\n"
        section += f"Range: {min_date} to {max_date}\n\n"
        section += "```\n"
        reg_label_width = max(len(r) for r in by_reg.keys()) + 2
        for reg, events in sorted(by_reg.items()):
            label = reg.ljust(reg_label_width)
            bar = ["."] * width
            for event in events:
                pos = self._date_position(event.date[:10], min_date, max_date, width)
                if 0 <= pos < width:
                    if event.status == "overdue":
                        bar[pos] = "X"
                    elif event.status == "imminent":
                        bar[pos] = "!"
                    elif event.status == "completed":
                        bar[pos] = "v"
                    else:
                        bar[pos] = "|"
            section += f"{label}[{''.join(bar)}]\n"
        section += "```\n\n"
        section += "Legend: `|` = upcoming, `!` = imminent, `X` = overdue, `v` = completed, `.` = no event"
        return section

    def _md_upcoming_deadlines(self, data: CalendarData) -> str:
        """Build upcoming deadlines sorted by urgency."""
        events = [e for e in data.events if e.status in ("upcoming", "imminent", "overdue")]
        events = sorted(events, key=lambda e: e.days_remaining)
        if not events:
            return "## Upcoming Deadlines\n\n*No upcoming deadlines.*"
        header = (
            "## Upcoming Deadlines\n\n"
            "| Date | Regulation | Event | Days Left | Priority | Status |\n"
            "|------|------------|-------|-----------|----------|--------|\n"
        )
        rows: List[str] = []
        for e in events:
            pmark = self.PRIORITY_MARKERS.get(e.priority, self.PRIORITY_MARKERS["normal"])
            smark = self.STATUS_MARKERS.get(e.status, self.STATUS_MARKERS["upcoming"])
            rows.append(
                f"| {e.date[:10]} | {e.regulation} | {e.title} | "
                f"{e.days_remaining} | {pmark['md']} {e.priority.title()} | "
                f"{smark['md']} {e.status.replace('_', ' ').title()} |"
            )
        return header + "\n".join(rows)

    def _md_milestone_tracking(self, data: CalendarData) -> str:
        """Build milestone tracking table."""
        if not data.milestones:
            return ""
        header = (
            "## Milestone Tracking\n\n"
            "| Regulation | Milestone | Target | Actual | Status | Progress |\n"
            "|------------|-----------|--------|--------|--------|----------|\n"
        )
        rows: List[str] = []
        for ms in sorted(data.milestones, key=lambda m: m.target_date):
            bar_len = int(ms.completion_pct / 5)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            actual = ms.actual_date[:10] if ms.actual_date else "---"
            rows.append(
                f"| {ms.regulation} | {ms.milestone} | {ms.target_date[:10]} | "
                f"{actual} | {ms.status.replace('_', ' ').title()} | "
                f"`[{bar}]` {ms.completion_pct:.0f}% |"
            )
        return header + "\n".join(rows)

    def _md_dependencies(self, data: CalendarData) -> str:
        """Build dependencies section."""
        if not data.dependencies:
            return ""
        header = (
            "## Cross-Regulation Dependencies\n\n"
            "| Source Event | Target Event | Type | Lag (days) | Cross-Reg |\n"
            "|-------------|-------------|------|------------|----------|\n"
        )
        lookup = self._event_lookup(data)
        rows: List[str] = []
        for dep in data.dependencies:
            src = lookup.get(dep.source_event_id)
            tgt = lookup.get(dep.target_event_id)
            src_label = f"{src.regulation}: {src.title}" if src else dep.source_event_id
            tgt_label = f"{tgt.regulation}: {tgt.title}" if tgt else dep.target_event_id
            cross = "Yes" if dep.is_cross_regulation else "No"
            rows.append(
                f"| {src_label} | {tgt_label} | "
                f"{dep.dependency_type.replace('_', ' ').title()} | "
                f"{dep.lag_days} | {cross} |"
            )
        return header + "\n".join(rows)

    def _md_conflicts(self, data: CalendarData) -> str:
        """Build conflicts section."""
        if not data.conflicts:
            return ""
        header = (
            "## Calendar Conflicts\n\n"
            "| ID | Type | Severity | Events | Resolution | Resolved |\n"
            "|----|------|----------|--------|------------|----------|\n"
        )
        rows: List[str] = []
        for c in data.conflicts:
            event_ids = ", ".join(c.event_ids)
            resolved = "Yes" if c.resolved else "No"
            rows.append(
                f"| {c.conflict_id} | {c.conflict_type.replace('_', ' ').title()} | "
                f"{c.severity.upper()} | {event_ids} | "
                f"{c.resolution or 'Pending'} | {resolved} |"
            )
        return header + "\n".join(rows)

    def _md_footer(self) -> str:
        """Build markdown footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            "*Template: RegulatoryCalendarReportTemplate v1.0 | PACK-009*"
        )

    # ------------------------------------------------------------------ #
    #  HTML builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: CalendarData) -> str:
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

    def _html_summary(self, data: CalendarData) -> str:
        """Build HTML summary cards."""
        overdue = sum(1 for e in data.events if e.status == "overdue")
        imminent = sum(1 for e in data.events if e.status == "imminent")
        upcoming = sum(1 for e in data.events if e.status == "upcoming")
        conflicts_open = sum(1 for c in data.conflicts if not c.resolved)
        cards = (
            f'<div class="stat-card"><span class="stat-val">{len(data.events)}</span>'
            f'<span class="stat-lbl">Total Events</span></div>'
            f'<div class="stat-card" style="border-top:3px solid #e74c3c">'
            f'<span class="stat-val">{overdue}</span>'
            f'<span class="stat-lbl">Overdue</span></div>'
            f'<div class="stat-card" style="border-top:3px solid #f39c12">'
            f'<span class="stat-val">{imminent}</span>'
            f'<span class="stat-lbl">Imminent</span></div>'
            f'<div class="stat-card"><span class="stat-val">{upcoming}</span>'
            f'<span class="stat-lbl">Upcoming</span></div>'
            f'<div class="stat-card" style="border-top:3px solid #e74c3c">'
            f'<span class="stat-val">{conflicts_open}</span>'
            f'<span class="stat-lbl">Conflicts</span></div>'
        )
        return f'<div class="section"><h2>Summary</h2><div class="stat-grid">{cards}</div></div>'

    def _html_gantt_chart(self, data: CalendarData) -> str:
        """Build HTML Gantt chart."""
        by_reg = self._events_by_regulation(data)
        if not by_reg:
            return ""
        all_events = self._filtered_events(data)
        if not all_events:
            return ""
        all_dates = [e.date[:10] for e in all_events]
        min_date = min(all_dates)
        max_date = max(all_dates)
        status_colors = {
            "overdue": "#e74c3c",
            "imminent": "#f39c12",
            "completed": "#2ecc71",
            "upcoming": "#3498db",
        }
        rows_html = ""
        for reg in sorted(by_reg.keys()):
            events = by_reg[reg]
            markers = ""
            for event in events:
                pos_pct = self._date_position_pct(event.date[:10], min_date, max_date)
                color = status_colors.get(event.status, "#3498db")
                markers += (
                    f'<div class="gantt-marker" title="{event.title} ({event.date[:10]})" '
                    f'style="left:{pos_pct:.1f}%;background:{color}"></div>'
                )
            rows_html += (
                f'<div class="gantt-row">'
                f'<div class="gantt-label">{reg}</div>'
                f'<div class="gantt-bar-bg">{markers}</div>'
                f'</div>'
            )
        return (
            f'<div class="section"><h2>Regulatory Timeline</h2>'
            f'<div class="gantt-range">{min_date} to {max_date}</div>'
            f'<div class="gantt-chart">{rows_html}</div></div>'
        )

    def _html_upcoming_deadlines(self, data: CalendarData) -> str:
        """Build HTML upcoming deadlines table."""
        events = [e for e in data.events if e.status in ("upcoming", "imminent", "overdue")]
        events = sorted(events, key=lambda e: e.days_remaining)
        if not events:
            return (
                '<div class="section"><h2>Upcoming Deadlines</h2>'
                '<p class="note">No upcoming deadlines.</p></div>'
            )
        rows = ""
        for e in events:
            color = self.STATUS_MARKERS.get(e.status, self.STATUS_MARKERS["upcoming"])["color"]
            pcolor = self.PRIORITY_MARKERS.get(e.priority, self.PRIORITY_MARKERS["normal"])["color"]
            rows += (
                f'<tr>'
                f'<td>{e.date[:10]}</td>'
                f'<td>{e.regulation}</td>'
                f'<td>{e.title}</td>'
                f'<td class="num" style="color:{color};font-weight:bold">{e.days_remaining}</td>'
                f'<td><span class="priority-badge" style="background:{pcolor}">'
                f'{e.priority.title()}</span></td>'
                f'<td><span class="status-badge" style="background:{color}">'
                f'{e.status.replace("_", " ").title()}</span></td>'
                f'</tr>'
            )
        return (
            '<div class="section"><h2>Upcoming Deadlines</h2>'
            '<table><thead><tr>'
            '<th>Date</th><th>Regulation</th><th>Event</th>'
            '<th>Days Left</th><th>Priority</th><th>Status</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_milestone_tracking(self, data: CalendarData) -> str:
        """Build HTML milestone tracking."""
        if not data.milestones:
            return ""
        rows = ""
        status_colors = {
            "completed": "#2ecc71", "on_track": "#3498db",
            "at_risk": "#f39c12", "missed": "#e74c3c", "pending": "#95a5a6",
        }
        for ms in sorted(data.milestones, key=lambda m: m.target_date):
            color = status_colors.get(ms.status, "#95a5a6")
            actual = ms.actual_date[:10] if ms.actual_date else "---"
            rows += (
                f'<tr><td>{ms.regulation}</td><td>{ms.milestone}</td>'
                f'<td>{ms.target_date[:10]}</td><td>{actual}</td>'
                f'<td><span class="status-badge" style="background:{color}">'
                f'{ms.status.replace("_", " ").title()}</span></td>'
                f'<td><div class="progress-bar"><div class="progress-fill" '
                f'style="width:{ms.completion_pct:.0f}%;background:{color}"></div></div>'
                f'{ms.completion_pct:.0f}%</td></tr>'
            )
        return (
            '<div class="section"><h2>Milestone Tracking</h2>'
            '<table><thead><tr>'
            '<th>Regulation</th><th>Milestone</th><th>Target</th>'
            '<th>Actual</th><th>Status</th><th>Progress</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_dependencies(self, data: CalendarData) -> str:
        """Build HTML dependencies section."""
        if not data.dependencies:
            return ""
        lookup = self._event_lookup(data)
        rows = ""
        for dep in data.dependencies:
            src = lookup.get(dep.source_event_id)
            tgt = lookup.get(dep.target_event_id)
            src_label = f"{src.regulation}: {src.title}" if src else dep.source_event_id
            tgt_label = f"{tgt.regulation}: {tgt.title}" if tgt else dep.target_event_id
            cross_bg = "#e8f8f5" if dep.is_cross_regulation else "#fff"
            rows += (
                f'<tr style="background:{cross_bg}">'
                f'<td>{src_label}</td><td>{tgt_label}</td>'
                f'<td>{dep.dependency_type.replace("_", " ").title()}</td>'
                f'<td class="num">{dep.lag_days}</td>'
                f'<td>{"Yes" if dep.is_cross_regulation else "No"}</td></tr>'
            )
        return (
            '<div class="section"><h2>Cross-Regulation Dependencies</h2>'
            '<table><thead><tr>'
            '<th>Source</th><th>Target</th><th>Type</th><th>Lag</th><th>Cross-Reg</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_conflicts(self, data: CalendarData) -> str:
        """Build HTML conflicts section."""
        if not data.conflicts:
            return ""
        rows = ""
        for c in data.conflicts:
            sev_color = "#e74c3c" if c.severity == "high" else "#f39c12" if c.severity == "medium" else "#95a5a6"
            resolved_icon = '<span style="color:#2ecc71">Yes</span>' if c.resolved else '<span style="color:#e74c3c">No</span>'
            rows += (
                f'<tr><td>{c.conflict_id}</td>'
                f'<td>{c.conflict_type.replace("_", " ").title()}</td>'
                f'<td><span class="sev-badge" style="background:{sev_color}">'
                f'{c.severity.upper()}</span></td>'
                f'<td>{", ".join(c.event_ids)}</td>'
                f'<td>{c.resolution or "Pending"}</td>'
                f'<td>{resolved_icon}</td></tr>'
            )
        return (
            '<div class="section"><h2>Calendar Conflicts</h2>'
            '<table><thead><tr>'
            '<th>ID</th><th>Type</th><th>Severity</th>'
            '<th>Events</th><th>Resolution</th><th>Resolved</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON builders
    # ------------------------------------------------------------------ #

    def _json_summary(self, data: CalendarData) -> Dict[str, Any]:
        """Build JSON summary."""
        return {
            "total_events": len(data.events),
            "overdue": sum(1 for e in data.events if e.status == "overdue"),
            "imminent": sum(1 for e in data.events if e.status == "imminent"),
            "upcoming": sum(1 for e in data.events if e.status == "upcoming"),
            "completed": sum(1 for e in data.events if e.status == "completed"),
            "regulations": sorted(set(e.regulation for e in data.events)),
            "dependencies_count": len(data.dependencies),
            "open_conflicts": sum(1 for c in data.conflicts if not c.resolved),
            "calendar_start": data.calendar_start,
            "calendar_end": data.calendar_end,
        }

    def _json_events(self, data: CalendarData) -> List[Dict[str, Any]]:
        """Build JSON events list."""
        events = self._filtered_events(data)
        return [
            {
                "event_id": e.event_id,
                "regulation": e.regulation,
                "title": e.title,
                "description": e.description,
                "date": e.date,
                "end_date": e.end_date,
                "event_type": e.event_type,
                "priority": e.priority,
                "days_remaining": e.days_remaining,
                "status": e.status,
                "owner": e.owner,
            }
            for e in events
        ]

    def _json_milestones(self, data: CalendarData) -> List[Dict[str, Any]]:
        """Build JSON milestones list."""
        return [
            {
                "regulation": ms.regulation,
                "milestone": ms.milestone,
                "target_date": ms.target_date,
                "actual_date": ms.actual_date,
                "status": ms.status,
                "completion_pct": round(ms.completion_pct, 1),
            }
            for ms in sorted(data.milestones, key=lambda m: m.target_date)
        ]

    def _json_dependencies(self, data: CalendarData) -> List[Dict[str, Any]]:
        """Build JSON dependencies list."""
        return [
            {
                "source_event_id": d.source_event_id,
                "target_event_id": d.target_event_id,
                "dependency_type": d.dependency_type,
                "lag_days": d.lag_days,
                "is_cross_regulation": d.is_cross_regulation,
                "description": d.description,
            }
            for d in data.dependencies
        ]

    def _json_conflicts(self, data: CalendarData) -> List[Dict[str, Any]]:
        """Build JSON conflicts list."""
        return [
            {
                "conflict_id": c.conflict_id,
                "event_ids": c.event_ids,
                "conflict_type": c.conflict_type,
                "severity": c.severity,
                "description": c.description,
                "resolution": c.resolution,
                "resolved": c.resolved,
            }
            for c in data.conflicts
        ]

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _date_position(self, date_str: str, min_date: str, max_date: str, width: int) -> int:
        """Calculate position of a date within a character-width range."""
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            d_min = datetime.strptime(min_date, "%Y-%m-%d")
            d_max = datetime.strptime(max_date, "%Y-%m-%d")
            span = (d_max - d_min).days or 1
            offset = (d - d_min).days
            return int((offset / span) * (width - 1))
        except (ValueError, ZeroDivisionError):
            return 0

    def _date_position_pct(self, date_str: str, min_date: str, max_date: str) -> float:
        """Calculate position of a date as a percentage."""
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            d_min = datetime.strptime(min_date, "%Y-%m-%d")
            d_max = datetime.strptime(max_date, "%Y-%m-%d")
            span = (d_max - d_min).days or 1
            offset = (d - d_min).days
            return (offset / span) * 100.0
        except (ValueError, ZeroDivisionError):
            return 0.0

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
            ".priority-badge,.status-badge,.sev-badge{"
            "display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".gantt-chart{margin-top:12px}"
            ".gantt-range{font-size:13px;color:#7f8c8d;margin-bottom:8px}"
            ".gantt-row{display:flex;align-items:center;margin-bottom:4px}"
            ".gantt-label{width:120px;font-size:13px;font-weight:600}"
            ".gantt-bar-bg{flex:1;height:20px;background:#ecf0f1;border-radius:4px;"
            "position:relative;overflow:hidden}"
            ".gantt-marker{position:absolute;width:8px;height:20px;border-radius:2px;"
            "top:0}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;display:inline-block;width:80%}"
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
            f'Template: RegulatoryCalendarReportTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
