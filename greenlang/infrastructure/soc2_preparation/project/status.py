# -*- coding: utf-8 -*-
"""
Audit Status Reporter Module - SEC-009 Phase 8

Provides status reporting, executive summaries, and KPI calculation for
SOC 2 audit projects. Generates weekly reports, Gantt charts, and project
health metrics.

Classes:
    - ProjectKPIs: Key performance indicators model
    - WeeklyReport: Weekly status report model
    - AuditStatusReporter: Report generation class

Example:
    >>> reporter = AuditStatusReporter(timeline, task_manager)
    >>> weekly_report = await reporter.generate_weekly_report(project_id)
    >>> executive_summary = await reporter.generate_executive_summary(project_id)
    >>> gantt_svg = reporter.export_gantt(project_id)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ProjectKPIs(BaseModel):
    """Key performance indicators for an audit project.

    Attributes:
        project_id: Project identifier.
        on_track: Number of milestones on track.
        at_risk: Number of milestones at risk.
        delayed: Number of delayed milestones.
        tasks_completed: Number of completed tasks.
        tasks_in_progress: Number of tasks in progress.
        tasks_pending: Number of pending tasks.
        tasks_blocked: Number of blocked tasks.
        completion_percentage: Overall project completion percentage.
        schedule_variance_days: Days ahead (+) or behind (-) schedule.
        estimated_completion_date: Projected completion date.
        health_score: Overall project health score (0-100).
        health_status: Health status category (healthy, at_risk, critical).
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(..., description="Project identifier.")
    on_track: int = Field(default=0, description="Number of milestones on track.")
    at_risk: int = Field(default=0, description="Number of milestones at risk.")
    delayed: int = Field(default=0, description="Number of delayed milestones.")
    tasks_completed: int = Field(default=0, description="Number of completed tasks.")
    tasks_in_progress: int = Field(default=0, description="Number of tasks in progress.")
    tasks_pending: int = Field(default=0, description="Number of pending tasks.")
    tasks_blocked: int = Field(default=0, description="Number of blocked tasks.")
    completion_percentage: float = Field(
        default=0.0, description="Overall project completion percentage."
    )
    schedule_variance_days: int = Field(
        default=0, description="Days ahead (+) or behind (-) schedule."
    )
    estimated_completion_date: Optional[date] = Field(
        default=None, description="Projected completion date."
    )
    health_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall project health score."
    )
    health_status: str = Field(
        default="healthy", description="Health status: healthy, at_risk, critical."
    )


class WeeklyReport(BaseModel):
    """Weekly status report for an audit project.

    Attributes:
        project_id: Project identifier.
        project_name: Project name.
        report_date: Date of the report.
        report_period_start: Start of the reporting period.
        report_period_end: End of the reporting period.
        current_phase: Current audit phase.
        kpis: Project KPIs snapshot.
        accomplishments: List of accomplishments this week.
        upcoming_milestones: Milestones due in the next 2 weeks.
        risks_and_issues: Current risks and issues.
        next_week_priorities: Priorities for the coming week.
        generated_at: Report generation timestamp.
        generated_by: User who generated the report.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(..., description="Project identifier.")
    project_name: str = Field(..., description="Project name.")
    report_date: date = Field(..., description="Date of the report.")
    report_period_start: date = Field(..., description="Start of the reporting period.")
    report_period_end: date = Field(..., description="End of the reporting period.")
    current_phase: str = Field(..., description="Current audit phase.")
    kpis: ProjectKPIs = Field(..., description="Project KPIs snapshot.")
    accomplishments: List[str] = Field(
        default_factory=list, description="List of accomplishments this week."
    )
    upcoming_milestones: List[Dict[str, Any]] = Field(
        default_factory=list, description="Milestones due in the next 2 weeks."
    )
    risks_and_issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="Current risks and issues."
    )
    next_week_priorities: List[str] = Field(
        default_factory=list, description="Priorities for the coming week."
    )
    markdown_content: str = Field(
        default="", description="Full report in Markdown format."
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp.",
    )
    generated_by: str = Field(default="", description="User who generated the report.")


# ---------------------------------------------------------------------------
# Audit Status Reporter
# ---------------------------------------------------------------------------


class AuditStatusReporter:
    """Report generation and KPI calculation for audit projects.

    Generates weekly status reports, executive summaries, Gantt charts,
    and calculates project health metrics.

    Attributes:
        timeline: AuditTimeline instance for project/milestone data.
        task_manager: AuditTaskManager instance for task data.

    Example:
        >>> reporter = AuditStatusReporter(timeline, task_manager)
        >>> weekly_report = await reporter.generate_weekly_report(project_id)
    """

    def __init__(
        self,
        timeline: Any = None,
        task_manager: Any = None,
    ) -> None:
        """Initialize AuditStatusReporter.

        Args:
            timeline: AuditTimeline instance.
            task_manager: AuditTaskManager instance.
        """
        self.timeline = timeline
        self.task_manager = task_manager
        logger.info("AuditStatusReporter initialized")

    async def generate_weekly_report(
        self,
        project_id: str,
        generated_by: str = "",
    ) -> str:
        """Generate a weekly status report in Markdown format.

        Args:
            project_id: ID of the project.
            generated_by: User generating the report.

        Returns:
            Markdown-formatted weekly report string.

        Raises:
            ValueError: If project not found.
        """
        start_time = datetime.now(timezone.utc)

        # Get project data
        project = None
        if self.timeline:
            project = await self.timeline.get_project(project_id)

        if project is None:
            raise ValueError(f"Project '{project_id}' not found.")

        # Calculate KPIs
        kpis = self.calculate_kpis(project_id)

        # Get recent accomplishments (tasks completed this week)
        accomplishments = await self._get_weekly_accomplishments(project_id)

        # Get upcoming milestones (next 2 weeks)
        upcoming = self._get_upcoming_milestones(project, days=14)

        # Get risks and issues
        risks = self._get_risks_and_issues(project, kpis)

        # Determine next week priorities
        priorities = await self._determine_priorities(project_id)

        # Calculate reporting period
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)

        # Generate Markdown report
        report = self._format_weekly_report_markdown(
            project=project,
            kpis=kpis,
            accomplishments=accomplishments,
            upcoming=upcoming,
            risks=risks,
            priorities=priorities,
            week_start=week_start,
            week_end=week_end,
        )

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Weekly report generated: project=%s, elapsed=%.2fms",
            project_id,
            elapsed_ms,
        )

        return report

    async def generate_executive_summary(self, project_id: str) -> str:
        """Generate an executive summary for a project.

        Args:
            project_id: ID of the project.

        Returns:
            Markdown-formatted executive summary string.

        Raises:
            ValueError: If project not found.
        """
        project = None
        if self.timeline:
            project = await self.timeline.get_project(project_id)

        if project is None:
            raise ValueError(f"Project '{project_id}' not found.")

        kpis = self.calculate_kpis(project_id)

        # Format executive summary
        summary = f"""# Executive Summary: {project.project_name}

**Report Date:** {date.today().strftime('%B %d, %Y')}
**Current Phase:** {project.current_phase.value.title()}

## Project Health

| Metric | Value | Status |
|--------|-------|--------|
| Health Score | {kpis.health_score:.0f}/100 | {self._health_indicator(kpis.health_score)} |
| Overall Progress | {kpis.completion_percentage:.1f}% | - |
| Schedule Variance | {kpis.schedule_variance_days:+d} days | {self._variance_indicator(kpis.schedule_variance_days)} |
| Est. Completion | {kpis.estimated_completion_date or 'TBD'} | - |

## Milestone Status

| Status | Count |
|--------|-------|
| On Track | {kpis.on_track} |
| At Risk | {kpis.at_risk} |
| Delayed | {kpis.delayed} |

## Task Summary

| Status | Count |
|--------|-------|
| Completed | {kpis.tasks_completed} |
| In Progress | {kpis.tasks_in_progress} |
| Pending | {kpis.tasks_pending} |
| Blocked | {kpis.tasks_blocked} |

## Key Observations

"""
        # Add observations based on KPIs
        observations = self._generate_observations(kpis, project)
        for i, obs in enumerate(observations, 1):
            summary += f"{i}. {obs}\n"

        summary += f"""
## Recommended Actions

"""
        actions = self._generate_recommended_actions(kpis, project)
        for i, action in enumerate(actions, 1):
            summary += f"{i}. {action}\n"

        summary += f"""
---
*Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*
"""

        logger.info("Executive summary generated: project=%s", project_id)
        return summary

    def export_gantt(self, project_id: str) -> bytes:
        """Export a Gantt chart visualization as SVG.

        Args:
            project_id: ID of the project.

        Returns:
            SVG file contents as bytes.

        Raises:
            ValueError: If project not found.
        """
        if self.timeline is None:
            raise ValueError("Timeline not configured")

        project = self.timeline._projects.get(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' not found.")

        # Generate SVG Gantt chart
        svg = self._generate_gantt_svg(project)

        logger.info("Gantt chart exported: project=%s", project_id)
        return svg.encode("utf-8")

    def calculate_kpis(self, project_id: str) -> ProjectKPIs:
        """Calculate KPIs for a project.

        Args:
            project_id: ID of the project.

        Returns:
            ProjectKPIs with calculated metrics.
        """
        # Get project and milestone data
        project = None
        milestones = []
        if self.timeline:
            project = self.timeline._projects.get(project_id)
            if project:
                milestones = project.milestones

        # Count milestone statuses
        on_track = 0
        at_risk = 0
        delayed = 0

        today = date.today()
        for m in milestones:
            if m.status.value == "completed":
                on_track += 1
            elif m.is_overdue:
                delayed += 1
            elif m.days_until_due <= 7:
                at_risk += 1
            else:
                on_track += 1

        # Get task counts
        tasks_completed = 0
        tasks_in_progress = 0
        tasks_pending = 0
        tasks_blocked = 0

        if self.task_manager:
            for task in self.task_manager._tasks.values():
                if task.project_id != project_id:
                    continue
                if task.status.value == "done":
                    tasks_completed += 1
                elif task.status.value == "in_progress":
                    tasks_in_progress += 1
                elif task.status.value == "blocked":
                    tasks_blocked += 1
                elif task.status.value in ("todo", "in_review"):
                    tasks_pending += 1

        # Calculate completion percentage
        total_milestones = len(milestones)
        completed_milestones = sum(1 for m in milestones if m.status.value == "completed")
        completion_pct = (
            (completed_milestones / total_milestones * 100)
            if total_milestones > 0
            else 0.0
        )

        # Calculate schedule variance
        variance_days = 0
        if project and project.target_end_date:
            expected_progress = self._calculate_expected_progress(project)
            actual_progress = completion_pct
            variance_days = int((actual_progress - expected_progress) / 100 *
                               (project.target_end_date - project.start_date).days)

        # Calculate health score
        health_score = self._calculate_health_score(
            on_track=on_track,
            at_risk=at_risk,
            delayed=delayed,
            tasks_blocked=tasks_blocked,
            completion_pct=completion_pct,
        )

        # Determine health status
        if health_score >= 80:
            health_status = "healthy"
        elif health_score >= 60:
            health_status = "at_risk"
        else:
            health_status = "critical"

        # Estimate completion date
        est_completion = None
        if project and completion_pct > 0:
            days_elapsed = (today - project.start_date).days
            if days_elapsed > 0:
                velocity = completion_pct / days_elapsed
                if velocity > 0:
                    remaining_pct = 100 - completion_pct
                    remaining_days = int(remaining_pct / velocity)
                    est_completion = today + timedelta(days=remaining_days)

        return ProjectKPIs(
            project_id=project_id,
            on_track=on_track,
            at_risk=at_risk,
            delayed=delayed,
            tasks_completed=tasks_completed,
            tasks_in_progress=tasks_in_progress,
            tasks_pending=tasks_pending,
            tasks_blocked=tasks_blocked,
            completion_percentage=completion_pct,
            schedule_variance_days=variance_days,
            estimated_completion_date=est_completion,
            health_score=health_score,
            health_status=health_status,
        )

    # -----------------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------------

    async def _get_weekly_accomplishments(self, project_id: str) -> List[str]:
        """Get tasks completed in the past week."""
        accomplishments = []
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)

        if self.task_manager:
            for task in self.task_manager._tasks.values():
                if task.project_id != project_id:
                    continue
                if task.completed_at and task.completed_at >= week_ago:
                    accomplishments.append(f"Completed: {task.title}")

        return accomplishments

    def _get_upcoming_milestones(
        self,
        project: Any,
        days: int = 14,
    ) -> List[Dict[str, Any]]:
        """Get milestones due within the specified number of days."""
        upcoming = []
        today = date.today()
        cutoff = today + timedelta(days=days)

        for m in project.milestones:
            if m.status.value == "completed":
                continue
            if m.target_date <= cutoff:
                upcoming.append(
                    {
                        "name": m.name,
                        "target_date": m.target_date.isoformat(),
                        "days_until_due": (m.target_date - today).days,
                        "status": m.status.value,
                        "phase": m.phase.value,
                    }
                )

        upcoming.sort(key=lambda x: x["days_until_due"])
        return upcoming

    def _get_risks_and_issues(
        self,
        project: Any,
        kpis: ProjectKPIs,
    ) -> List[Dict[str, Any]]:
        """Identify risks and issues based on project status."""
        risks = []

        # Delayed milestones
        if kpis.delayed > 0:
            risks.append(
                {
                    "type": "issue",
                    "severity": "high",
                    "description": f"{kpis.delayed} milestone(s) are delayed",
                    "recommendation": "Review delayed milestones and create recovery plan",
                }
            )

        # At-risk milestones
        if kpis.at_risk > 0:
            risks.append(
                {
                    "type": "risk",
                    "severity": "medium",
                    "description": f"{kpis.at_risk} milestone(s) are at risk",
                    "recommendation": "Prioritize at-risk milestones to avoid delays",
                }
            )

        # Blocked tasks
        if kpis.tasks_blocked > 0:
            risks.append(
                {
                    "type": "issue",
                    "severity": "high",
                    "description": f"{kpis.tasks_blocked} task(s) are blocked",
                    "recommendation": "Resolve blockers to maintain velocity",
                }
            )

        # Schedule variance
        if kpis.schedule_variance_days < -7:
            risks.append(
                {
                    "type": "risk",
                    "severity": "high",
                    "description": f"Project is {abs(kpis.schedule_variance_days)} days behind schedule",
                    "recommendation": "Consider scope adjustment or resource addition",
                }
            )

        return risks

    async def _determine_priorities(self, project_id: str) -> List[str]:
        """Determine priorities for the coming week."""
        priorities = []

        if self.task_manager:
            # Get high-priority tasks
            tasks = await self.task_manager.list_tasks(project_id=project_id)
            critical = [t for t in tasks if t.priority.value == "critical" and t.status.value != "done"]
            high = [t for t in tasks if t.priority.value == "high" and t.status.value != "done"]

            for t in critical[:3]:
                priorities.append(f"[CRITICAL] {t.title}")
            for t in high[:3]:
                priorities.append(f"[HIGH] {t.title}")

        # Get upcoming milestones
        if self.timeline and priorities:
            project = self.timeline._projects.get(project_id)
            if project:
                upcoming = self._get_upcoming_milestones(project, days=7)
                for m in upcoming[:2]:
                    priorities.append(f"[MILESTONE] {m['name']} (due {m['target_date']})")

        if not priorities:
            priorities.append("No high-priority items identified")

        return priorities[:7]  # Limit to 7 priorities

    def _format_weekly_report_markdown(
        self,
        project: Any,
        kpis: ProjectKPIs,
        accomplishments: List[str],
        upcoming: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
        priorities: List[str],
        week_start: date,
        week_end: date,
    ) -> str:
        """Format the weekly report as Markdown."""
        report = f"""# Weekly Status Report: {project.project_name}

**Report Period:** {week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}
**Current Phase:** {project.current_phase.value.title()}

---

## Summary

| Metric | Value |
|--------|-------|
| Health Score | {kpis.health_score:.0f}/100 ({kpis.health_status}) |
| Overall Progress | {kpis.completion_percentage:.1f}% |
| Schedule Variance | {kpis.schedule_variance_days:+d} days |

## Milestone Status

| Status | Count |
|--------|-------|
| On Track | {kpis.on_track} |
| At Risk | {kpis.at_risk} |
| Delayed | {kpis.delayed} |

## Task Summary

| Status | Count |
|--------|-------|
| Completed | {kpis.tasks_completed} |
| In Progress | {kpis.tasks_in_progress} |
| Pending | {kpis.tasks_pending} |
| Blocked | {kpis.tasks_blocked} |

---

## Accomplishments This Week

"""
        if accomplishments:
            for item in accomplishments:
                report += f"- {item}\n"
        else:
            report += "- No tasks completed this week\n"

        report += """
## Upcoming Milestones (Next 2 Weeks)

"""
        if upcoming:
            report += "| Milestone | Due Date | Days | Status |\n"
            report += "|-----------|----------|------|--------|\n"
            for m in upcoming:
                report += f"| {m['name']} | {m['target_date']} | {m['days_until_due']} | {m['status']} |\n"
        else:
            report += "No milestones due in the next 2 weeks.\n"

        report += """
## Risks and Issues

"""
        if risks:
            for r in risks:
                severity_icon = "!" if r["severity"] == "high" else "?"
                report += f"- [{severity_icon}] **{r['type'].title()}**: {r['description']}\n"
                report += f"  - *Recommendation*: {r['recommendation']}\n"
        else:
            report += "No significant risks or issues identified.\n"

        report += """
## Next Week Priorities

"""
        for i, p in enumerate(priorities, 1):
            report += f"{i}. {p}\n"

        report += f"""
---

*Report generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*
"""

        return report

    def _calculate_expected_progress(self, project: Any) -> float:
        """Calculate expected progress based on elapsed time."""
        today = date.today()
        total_days = (project.target_end_date - project.start_date).days
        elapsed_days = (today - project.start_date).days

        if total_days <= 0:
            return 0.0

        return min(100.0, (elapsed_days / total_days) * 100)

    def _calculate_health_score(
        self,
        on_track: int,
        at_risk: int,
        delayed: int,
        tasks_blocked: int,
        completion_pct: float,
    ) -> float:
        """Calculate overall health score (0-100)."""
        total_milestones = on_track + at_risk + delayed
        if total_milestones == 0:
            return 100.0

        # Weight factors
        milestone_score = (on_track / total_milestones) * 40  # 40% weight
        at_risk_penalty = (at_risk / total_milestones) * 20  # Up to 20% penalty
        delay_penalty = (delayed / total_milestones) * 30  # Up to 30% penalty
        blocked_penalty = min(tasks_blocked * 2, 10)  # Up to 10% penalty

        health = 100 - at_risk_penalty - delay_penalty - blocked_penalty

        return max(0.0, min(100.0, health))

    def _health_indicator(self, score: float) -> str:
        """Return health indicator emoji/text."""
        if score >= 80:
            return "Healthy"
        elif score >= 60:
            return "At Risk"
        else:
            return "Critical"

    def _variance_indicator(self, days: int) -> str:
        """Return variance indicator."""
        if days >= 0:
            return "On/Ahead"
        elif days >= -7:
            return "Slightly Behind"
        else:
            return "Behind"

    def _generate_observations(
        self,
        kpis: ProjectKPIs,
        project: Any,
    ) -> List[str]:
        """Generate key observations based on KPIs."""
        observations = []

        if kpis.health_score >= 80:
            observations.append("Project is healthy and on track for completion.")
        elif kpis.health_score >= 60:
            observations.append("Project requires attention to avoid delays.")
        else:
            observations.append("Project is at critical status and requires immediate intervention.")

        if kpis.delayed > 0:
            observations.append(
                f"{kpis.delayed} milestones are delayed, impacting overall timeline."
            )

        if kpis.tasks_blocked > 0:
            observations.append(
                f"{kpis.tasks_blocked} tasks are blocked, affecting team velocity."
            )

        if kpis.schedule_variance_days > 0:
            observations.append(
                f"Project is {kpis.schedule_variance_days} days ahead of schedule."
            )
        elif kpis.schedule_variance_days < -7:
            observations.append(
                f"Project is {abs(kpis.schedule_variance_days)} days behind schedule."
            )

        return observations[:5]

    def _generate_recommended_actions(
        self,
        kpis: ProjectKPIs,
        project: Any,
    ) -> List[str]:
        """Generate recommended actions based on KPIs."""
        actions = []

        if kpis.delayed > 0:
            actions.append(
                "Review delayed milestones and create recovery plans with owners."
            )

        if kpis.tasks_blocked > 0:
            actions.append(
                "Schedule blocker resolution meeting to unblock tasks."
            )

        if kpis.at_risk > 0:
            actions.append(
                "Prioritize at-risk milestones to prevent them from becoming delayed."
            )

        if kpis.schedule_variance_days < -7:
            actions.append(
                "Consider requesting additional resources or adjusting scope."
            )

        if not actions:
            actions.append("Continue current execution pace to maintain healthy status.")

        return actions[:5]

    def _generate_gantt_svg(self, project: Any) -> str:
        """Generate an SVG Gantt chart for the project."""
        # SVG dimensions
        width = 1200
        height = max(400, 100 + len(project.milestones) * 30)
        margin_left = 200
        margin_top = 80
        chart_width = width - margin_left - 50
        row_height = 25

        # Calculate date range
        start_date = project.start_date
        end_date = project.target_end_date
        total_days = (end_date - start_date).days
        if total_days <= 0:
            total_days = 1

        # Start SVG
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; }}
    .label {{ font-family: Arial, sans-serif; font-size: 12px; }}
    .header {{ font-family: Arial, sans-serif; font-size: 10px; fill: #666; }}
    .bar-completed {{ fill: #4CAF50; }}
    .bar-in-progress {{ fill: #2196F3; }}
    .bar-pending {{ fill: #9E9E9E; }}
    .bar-delayed {{ fill: #F44336; }}
    .today-line {{ stroke: #FF5722; stroke-width: 2; stroke-dasharray: 5,5; }}
  </style>

  <!-- Title -->
  <text x="{width/2}" y="30" class="title" text-anchor="middle">{project.project_name} - Gantt Chart</text>

  <!-- Chart background -->
  <rect x="{margin_left}" y="{margin_top}" width="{chart_width}" height="{height - margin_top - 30}" fill="#f5f5f5" stroke="#ddd"/>
'''

        # Draw month headers
        current = start_date
        while current <= end_date:
            x = margin_left + ((current - start_date).days / total_days) * chart_width
            svg += f'  <text x="{x}" y="{margin_top - 5}" class="header">{current.strftime("%b %Y")}</text>\n'
            svg += f'  <line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - 30}" stroke="#ddd"/>\n'
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=current.month + 1, day=1)

        # Draw today line
        today = date.today()
        if start_date <= today <= end_date:
            today_x = margin_left + ((today - start_date).days / total_days) * chart_width
            svg += f'  <line x1="{today_x}" y1="{margin_top}" x2="{today_x}" y2="{height - 30}" class="today-line"/>\n'
            svg += f'  <text x="{today_x}" y="{height - 10}" class="header" text-anchor="middle">Today</text>\n'

        # Draw milestones
        for i, m in enumerate(project.milestones):
            y = margin_top + 10 + i * row_height

            # Label
            label = m.name[:25] + "..." if len(m.name) > 25 else m.name
            svg += f'  <text x="{margin_left - 10}" y="{y + 15}" class="label" text-anchor="end">{label}</text>\n'

            # Bar
            bar_start = (m.target_date - start_date).days - 7  # 7 days before target
            bar_end = (m.target_date - start_date).days
            bar_start = max(0, bar_start)

            bar_x = margin_left + (bar_start / total_days) * chart_width
            bar_width = max(5, ((bar_end - bar_start) / total_days) * chart_width)

            # Determine bar color
            if m.status.value == "completed":
                bar_class = "bar-completed"
            elif m.is_overdue:
                bar_class = "bar-delayed"
            elif m.status.value == "in_progress":
                bar_class = "bar-in-progress"
            else:
                bar_class = "bar-pending"

            svg += f'  <rect x="{bar_x}" y="{y}" width="{bar_width}" height="{row_height - 5}" class="{bar_class}" rx="3"/>\n'

            # Diamond marker for milestone date
            marker_x = margin_left + ((m.target_date - start_date).days / total_days) * chart_width
            svg += f'  <polygon points="{marker_x},{y+row_height/2-7} {marker_x+7},{y+row_height/2} {marker_x},{y+row_height/2+7} {marker_x-7},{y+row_height/2}" fill="#333"/>\n'

        # Legend
        legend_y = height - 25
        svg += f'''
  <!-- Legend -->
  <rect x="{margin_left}" y="{legend_y}" width="15" height="15" class="bar-completed"/>
  <text x="{margin_left + 20}" y="{legend_y + 12}" class="label">Completed</text>

  <rect x="{margin_left + 100}" y="{legend_y}" width="15" height="15" class="bar-in-progress"/>
  <text x="{margin_left + 120}" y="{legend_y + 12}" class="label">In Progress</text>

  <rect x="{margin_left + 220}" y="{legend_y}" width="15" height="15" class="bar-pending"/>
  <text x="{margin_left + 240}" y="{legend_y + 12}" class="label">Pending</text>

  <rect x="{margin_left + 320}" y="{legend_y}" width="15" height="15" class="bar-delayed"/>
  <text x="{margin_left + 340}" y="{legend_y + 12}" class="label">Delayed</text>
'''

        svg += "</svg>"
        return svg


__all__ = [
    "ProjectKPIs",
    "WeeklyReport",
    "AuditStatusReporter",
]
