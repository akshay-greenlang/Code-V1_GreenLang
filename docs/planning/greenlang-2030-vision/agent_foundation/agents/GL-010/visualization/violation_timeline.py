"""
GL-010 EMISSIONWATCH - Violation Timeline Visualization

Violation history visualization module for the EmissionsComplianceAgent.
Provides timeline view, severity analysis, and resolution tracking.

Author: GreenLang Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json


class ViolationSeverity(Enum):
    """Violation severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """Return numeric severity level."""
        return {"minor": 1, "moderate": 2, "major": 3, "critical": 4}.get(self.value, 1)

    @property
    def color(self) -> str:
        """Return color for severity."""
        colors = {
            "minor": "#3498DB",
            "moderate": "#F39C12",
            "major": "#E74C3C",
            "critical": "#8E44AD"
        }
        return colors.get(self.value, "#3498DB")

    @property
    def color_blind_safe(self) -> str:
        """Return color-blind safe color."""
        colors = {
            "minor": "#56B4E9",
            "moderate": "#E69F00",
            "major": "#D55E00",
            "critical": "#CC79A7"
        }
        return colors.get(self.value, "#56B4E9")


class ViolationStatus(Enum):
    """Violation resolution status."""
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    CORRECTIVE_ACTION = "corrective_action"
    RESOLVED = "resolved"
    REPORTED = "reported"
    CLOSED = "closed"

    @property
    def symbol(self) -> str:
        """Return marker symbol for status."""
        symbols = {
            "active": "circle",
            "under_review": "diamond",
            "corrective_action": "square",
            "resolved": "circle-open",
            "reported": "triangle-up",
            "closed": "star"
        }
        return symbols.get(self.value, "circle")


class ViolationType(Enum):
    """Types of compliance violations."""
    EMISSION_EXCEEDANCE = "emission_exceedance"
    OPACITY_VIOLATION = "opacity_violation"
    REPORTING_FAILURE = "reporting_failure"
    MONITORING_GAP = "monitoring_gap"
    PERMIT_DEVIATION = "permit_deviation"
    STARTUP_SHUTDOWN = "startup_shutdown"
    MALFUNCTION = "malfunction"
    CONTINUOUS_MONITORING = "continuous_monitoring"
    STACK_TEST_FAILURE = "stack_test_failure"

    @property
    def display_name(self) -> str:
        """Return human-readable name."""
        names = {
            "emission_exceedance": "Emission Exceedance",
            "opacity_violation": "Opacity Violation",
            "reporting_failure": "Reporting Failure",
            "monitoring_gap": "Monitoring Gap",
            "permit_deviation": "Permit Deviation",
            "startup_shutdown": "Startup/Shutdown",
            "malfunction": "Equipment Malfunction",
            "continuous_monitoring": "CEMS Violation",
            "stack_test_failure": "Stack Test Failure"
        }
        return names.get(self.value, self.value.replace("_", " ").title())


@dataclass
class RegulatoryResponse:
    """Regulatory response record."""
    response_type: str  # "notice", "warning", "citation", "penalty"
    agency: str
    date: str
    reference_number: str
    description: str
    penalty_amount: Optional[float] = None
    response_deadline: Optional[str] = None
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_type": self.response_type,
            "agency": self.agency,
            "date": self.date,
            "reference_number": self.reference_number,
            "description": self.description,
            "penalty_amount": self.penalty_amount,
            "response_deadline": self.response_deadline,
            "status": self.status
        }


@dataclass
class ViolationRecord:
    """Complete violation record."""
    violation_id: str
    violation_type: ViolationType
    pollutant: str
    pollutant_name: str
    start_time: str
    end_time: Optional[str]
    duration_minutes: int
    exceedance_value: float
    permit_limit: float
    unit: str
    exceedance_percent: float
    severity: ViolationSeverity
    status: ViolationStatus
    source_unit: str
    affected_limit: str  # e.g., "1-hour average", "24-hour rolling"
    root_cause: Optional[str] = None
    corrective_action: Optional[str] = None
    corrective_action_date: Optional[str] = None
    regulatory_responses: List[RegulatoryResponse] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type.value,
            "violation_type_display": self.violation_type.display_name,
            "pollutant": self.pollutant,
            "pollutant_name": self.pollutant_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": self.duration_minutes,
            "exceedance_value": self.exceedance_value,
            "permit_limit": self.permit_limit,
            "unit": self.unit,
            "exceedance_percent": self.exceedance_percent,
            "severity": self.severity.value,
            "status": self.status.value,
            "source_unit": self.source_unit,
            "affected_limit": self.affected_limit,
            "root_cause": self.root_cause,
            "corrective_action": self.corrective_action,
            "corrective_action_date": self.corrective_action_date,
            "regulatory_responses": [r.to_dict() for r in self.regulatory_responses],
            "notes": self.notes,
            "attachments": self.attachments
        }


@dataclass
class TimelineConfig:
    """Configuration for timeline visualization."""
    title: str = "Violation Timeline"
    show_duration_bars: bool = True
    show_severity_legend: bool = True
    show_status_filter: bool = True
    show_regulatory_markers: bool = True
    group_by_pollutant: bool = False
    color_blind_safe: bool = False
    interactive: bool = True
    export_enabled: bool = True


class ViolationTimelineChart:
    """Generate violation timeline visualization."""

    def __init__(
        self,
        violations: List[ViolationRecord],
        config: Optional[TimelineConfig] = None
    ):
        """
        Initialize timeline chart generator.

        Args:
            violations: List of violation records
            config: Timeline configuration
        """
        self.violations = sorted(violations, key=lambda v: v.start_time)
        self.config = config or TimelineConfig()

    def _get_severity_color(self, severity: ViolationSeverity) -> str:
        """Get color for severity level."""
        if self.config.color_blind_safe:
            return severity.color_blind_safe
        return severity.color

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string."""
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except ValueError:
            return datetime.now()

    def build_gantt_timeline(self) -> Dict[str, Any]:
        """
        Build Gantt-style timeline showing violation durations.

        Returns:
            Plotly chart dictionary
        """
        if not self.violations:
            return self._build_empty_chart("No violations to display")

        traces = []

        # Group violations by pollutant if configured
        if self.config.group_by_pollutant:
            grouped = {}
            for v in self.violations:
                if v.pollutant not in grouped:
                    grouped[v.pollutant] = []
                grouped[v.pollutant].append(v)

            y_labels = []
            for pollutant, violations in grouped.items():
                for idx, v in enumerate(violations):
                    label = f"{v.pollutant_name} - {v.violation_id}"
                    y_labels.append(label)

                    start = self._parse_datetime(v.start_time)
                    end = self._parse_datetime(v.end_time) if v.end_time else datetime.now()

                    traces.append(self._create_gantt_bar(v, label, start, end))

        else:
            for v in self.violations:
                label = v.violation_id
                start = self._parse_datetime(v.start_time)
                end = self._parse_datetime(v.end_time) if v.end_time else datetime.now()

                traces.append(self._create_gantt_bar(v, label, start, end))

        layout = {
            "title": {
                "text": self.config.title,
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {
                "title": "Time",
                "type": "date",
                "rangeslider": {"visible": True}
            },
            "yaxis": {
                "title": "Violations",
                "autorange": "reversed"
            },
            "showlegend": True,
            "legend": {
                "title": {"text": "Severity"},
                "orientation": "h",
                "y": 1.1
            },
            "hovermode": "closest",
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA",
            "height": max(400, len(self.violations) * 40 + 100)
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {"responsive": True, "displayModeBar": True}
        }

    def _create_gantt_bar(
        self,
        violation: ViolationRecord,
        label: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        """Create a Gantt bar trace for a violation."""
        color = self._get_severity_color(violation.severity)
        is_active = violation.status == ViolationStatus.ACTIVE

        hover_text = (
            f"<b>{violation.violation_id}</b><br>"
            f"Type: {violation.violation_type.display_name}<br>"
            f"Pollutant: {violation.pollutant_name}<br>"
            f"Start: {violation.start_time}<br>"
            f"End: {violation.end_time or 'Ongoing'}<br>"
            f"Duration: {violation.duration_minutes} min<br>"
            f"Value: {violation.exceedance_value:.2f} {violation.unit}<br>"
            f"Limit: {violation.permit_limit:.2f} {violation.unit}<br>"
            f"Exceedance: {violation.exceedance_percent:.1f}%<br>"
            f"Severity: {violation.severity.value.title()}<br>"
            f"Status: {violation.status.value.replace('_', ' ').title()}"
        )

        return {
            "type": "scatter",
            "mode": "lines",
            "name": violation.severity.value.title(),
            "x": [start, end],
            "y": [label, label],
            "line": {
                "color": color,
                "width": 20 if is_active else 15
            },
            "hovertemplate": hover_text + "<extra></extra>",
            "showlegend": True,
            "legendgroup": violation.severity.value
        }

    def build_scatter_timeline(self) -> Dict[str, Any]:
        """
        Build scatter plot timeline with violations as points.

        Returns:
            Plotly chart dictionary
        """
        if not self.violations:
            return self._build_empty_chart("No violations to display")

        traces = []

        # Group by severity for legend
        severity_groups: Dict[ViolationSeverity, List[ViolationRecord]] = {}
        for v in self.violations:
            if v.severity not in severity_groups:
                severity_groups[v.severity] = []
            severity_groups[v.severity].append(v)

        for severity, violations in severity_groups.items():
            x_vals = [v.start_time for v in violations]
            y_vals = [v.exceedance_percent for v in violations]
            sizes = [max(10, min(50, v.duration_minutes / 10)) for v in violations]

            hover_texts = []
            for v in violations:
                hover_texts.append(
                    f"<b>{v.violation_id}</b><br>"
                    f"Type: {v.violation_type.display_name}<br>"
                    f"Pollutant: {v.pollutant_name}<br>"
                    f"Start: {v.start_time}<br>"
                    f"Duration: {v.duration_minutes} min<br>"
                    f"Exceedance: {v.exceedance_percent:.1f}%"
                )

            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": severity.value.title(),
                "x": x_vals,
                "y": y_vals,
                "marker": {
                    "color": self._get_severity_color(severity),
                    "size": sizes,
                    "sizemode": "diameter",
                    "opacity": 0.7,
                    "line": {"color": "white", "width": 1}
                },
                "text": hover_texts,
                "hovertemplate": "%{text}<extra></extra>"
            })

        layout = {
            "title": {
                "text": self.config.title,
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {
                "title": "Time",
                "type": "date",
                "rangeslider": {"visible": True}
            },
            "yaxis": {
                "title": "Exceedance (%)",
                "rangemode": "tozero"
            },
            "showlegend": True,
            "legend": {
                "title": {"text": "Severity"},
                "orientation": "h",
                "y": 1.1
            },
            "hovermode": "closest",
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {"responsive": True, "displayModeBar": True}
        }

    def build_status_breakdown(self) -> Dict[str, Any]:
        """
        Build violation status breakdown chart.

        Returns:
            Plotly chart dictionary
        """
        status_counts: Dict[str, int] = {}
        for v in self.violations:
            status = v.status.value.replace("_", " ").title()
            status_counts[status] = status_counts.get(status, 0) + 1

        colors = {
            "Active": "#E74C3C",
            "Under Review": "#F39C12",
            "Corrective Action": "#3498DB",
            "Resolved": "#2ECC71",
            "Reported": "#9B59B6",
            "Closed": "#95A5A6"
        }

        trace = {
            "type": "pie",
            "labels": list(status_counts.keys()),
            "values": list(status_counts.values()),
            "hole": 0.4,
            "marker": {
                "colors": [colors.get(s, "#95A5A6") for s in status_counts.keys()]
            },
            "textinfo": "label+percent",
            "hovertemplate": "<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>"
        }

        layout = {
            "title": {
                "text": "Violation Status Distribution",
                "font": {"size": 16, "color": "#2C3E50"}
            },
            "showlegend": True,
            "annotations": [{
                "text": f"<b>{len(self.violations)}</b><br>Total",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 18},
                "showarrow": False
            }],
            "paper_bgcolor": "white"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_severity_distribution(self) -> Dict[str, Any]:
        """
        Build severity distribution chart.

        Returns:
            Plotly chart dictionary
        """
        severity_counts: Dict[str, int] = {}
        for v in self.violations:
            sev = v.severity.value.title()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        ordered = ["Minor", "Moderate", "Major", "Critical"]
        labels = [s for s in ordered if s in severity_counts]
        values = [severity_counts[s] for s in labels]
        colors = [
            self._get_severity_color(ViolationSeverity(s.lower()))
            for s in labels
        ]

        trace = {
            "type": "bar",
            "x": labels,
            "y": values,
            "marker": {"color": colors},
            "text": values,
            "textposition": "outside",
            "hovertemplate": "<b>%{x}</b><br>Count: %{y}<extra></extra>"
        }

        layout = {
            "title": {
                "text": "Violations by Severity",
                "font": {"size": 16, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Severity Level"},
            "yaxis": {"title": "Count", "rangemode": "tozero"},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_pollutant_breakdown(self) -> Dict[str, Any]:
        """
        Build violations by pollutant chart.

        Returns:
            Plotly chart dictionary
        """
        pollutant_counts: Dict[str, Dict[str, int]] = {}
        for v in self.violations:
            if v.pollutant_name not in pollutant_counts:
                pollutant_counts[v.pollutant_name] = {"total": 0, "active": 0}
            pollutant_counts[v.pollutant_name]["total"] += 1
            if v.status == ViolationStatus.ACTIVE:
                pollutant_counts[v.pollutant_name]["active"] += 1

        pollutants = list(pollutant_counts.keys())
        totals = [pollutant_counts[p]["total"] for p in pollutants]
        active = [pollutant_counts[p]["active"] for p in pollutants]

        traces = [
            {
                "type": "bar",
                "name": "Total Violations",
                "x": pollutants,
                "y": totals,
                "marker": {"color": "#3498DB"}
            },
            {
                "type": "bar",
                "name": "Active Violations",
                "x": pollutants,
                "y": active,
                "marker": {"color": "#E74C3C"}
            }
        ]

        layout = {
            "title": {
                "text": "Violations by Pollutant",
                "font": {"size": 16, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Pollutant"},
            "yaxis": {"title": "Count", "rangemode": "tozero"},
            "barmode": "group",
            "legend": {"orientation": "h", "y": 1.1},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_monthly_trend(self) -> Dict[str, Any]:
        """
        Build monthly violation trend chart.

        Returns:
            Plotly chart dictionary
        """
        monthly_counts: Dict[str, Dict[str, int]] = {}

        for v in self.violations:
            dt = self._parse_datetime(v.start_time)
            month_key = dt.strftime("%Y-%m")

            if month_key not in monthly_counts:
                monthly_counts[month_key] = {
                    "total": 0,
                    "minor": 0,
                    "moderate": 0,
                    "major": 0,
                    "critical": 0
                }

            monthly_counts[month_key]["total"] += 1
            monthly_counts[month_key][v.severity.value] += 1

        months = sorted(monthly_counts.keys())
        totals = [monthly_counts[m]["total"] for m in months]

        traces = [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Total Violations",
                "x": months,
                "y": totals,
                "line": {"color": "#2C3E50", "width": 3},
                "marker": {"size": 8}
            }
        ]

        # Add severity breakdown as stacked bars
        for severity in ["minor", "moderate", "major", "critical"]:
            values = [monthly_counts[m][severity] for m in months]
            if any(v > 0 for v in values):
                traces.append({
                    "type": "bar",
                    "name": severity.title(),
                    "x": months,
                    "y": values,
                    "marker": {
                        "color": self._get_severity_color(ViolationSeverity(severity))
                    }
                })

        layout = {
            "title": {
                "text": "Monthly Violation Trend",
                "font": {"size": 16, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Month", "tickangle": -45},
            "yaxis": {"title": "Count", "rangemode": "tozero"},
            "barmode": "stack",
            "legend": {"orientation": "h", "y": 1.15},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_duration_analysis(self) -> Dict[str, Any]:
        """
        Build violation duration analysis chart.

        Returns:
            Plotly chart dictionary
        """
        durations = [v.duration_minutes for v in self.violations]
        exceedances = [v.exceedance_percent for v in self.violations]
        severities = [v.severity for v in self.violations]

        hover_texts = [
            f"<b>{v.violation_id}</b><br>"
            f"Duration: {v.duration_minutes} min<br>"
            f"Exceedance: {v.exceedance_percent:.1f}%"
            for v in self.violations
        ]

        trace = {
            "type": "scatter",
            "mode": "markers",
            "x": durations,
            "y": exceedances,
            "marker": {
                "color": [self._get_severity_color(s) for s in severities],
                "size": 12,
                "opacity": 0.7
            },
            "text": hover_texts,
            "hovertemplate": "%{text}<extra></extra>"
        }

        layout = {
            "title": {
                "text": "Duration vs Exceedance Analysis",
                "font": {"size": 16, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Duration (minutes)", "type": "log"},
            "yaxis": {"title": "Exceedance (%)", "rangemode": "tozero"},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_regulatory_timeline(self) -> Dict[str, Any]:
        """
        Build timeline of regulatory responses.

        Returns:
            Plotly chart dictionary
        """
        regulatory_events = []
        for v in self.violations:
            for r in v.regulatory_responses:
                regulatory_events.append({
                    "violation_id": v.violation_id,
                    "date": r.date,
                    "type": r.response_type,
                    "agency": r.agency,
                    "penalty": r.penalty_amount,
                    "reference": r.reference_number
                })

        if not regulatory_events:
            return self._build_empty_chart("No regulatory responses recorded")

        events = sorted(regulatory_events, key=lambda e: e["date"])

        type_colors = {
            "notice": "#3498DB",
            "warning": "#F39C12",
            "citation": "#E74C3C",
            "penalty": "#8E44AD"
        }

        traces = []
        for event_type in set(e["type"] for e in events):
            type_events = [e for e in events if e["type"] == event_type]

            hover_texts = [
                f"<b>{e['reference']}</b><br>"
                f"Date: {e['date']}<br>"
                f"Agency: {e['agency']}<br>"
                f"Violation: {e['violation_id']}<br>"
                f"Penalty: ${e['penalty']:,.2f}" if e['penalty'] else ""
                for e in type_events
            ]

            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": event_type.title(),
                "x": [e["date"] for e in type_events],
                "y": [1] * len(type_events),
                "marker": {
                    "color": type_colors.get(event_type, "#95A5A6"),
                    "size": 15,
                    "symbol": "diamond"
                },
                "text": hover_texts,
                "hovertemplate": "%{text}<extra></extra>"
            })

        layout = {
            "title": {
                "text": "Regulatory Response Timeline",
                "font": {"size": 16, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Date", "type": "date"},
            "yaxis": {"visible": False, "range": [0, 2]},
            "showlegend": True,
            "legend": {"orientation": "h", "y": 1.1},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA",
            "height": 250
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {"responsive": True}
        }

    def _build_empty_chart(self, message: str) -> Dict[str, Any]:
        """Build empty chart with message."""
        return {
            "data": [],
            "layout": {
                "title": {"text": self.config.title},
                "annotations": [{
                    "text": message,
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16, "color": "#7F8C8D"}
                }],
                "paper_bgcolor": "white"
            },
            "config": {"responsive": True}
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for violations.

        Returns:
            Summary statistics dictionary
        """
        if not self.violations:
            return {
                "total_violations": 0,
                "active_violations": 0,
                "total_duration_hours": 0,
                "avg_duration_minutes": 0,
                "avg_exceedance_percent": 0,
                "by_severity": {},
                "by_status": {},
                "by_pollutant": {},
                "by_type": {}
            }

        durations = [v.duration_minutes for v in self.violations]
        exceedances = [v.exceedance_percent for v in self.violations]

        by_severity = {}
        by_status = {}
        by_pollutant = {}
        by_type = {}

        active_count = 0

        for v in self.violations:
            # Severity
            sev = v.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            # Status
            stat = v.status.value
            by_status[stat] = by_status.get(stat, 0) + 1
            if v.status == ViolationStatus.ACTIVE:
                active_count += 1

            # Pollutant
            poll = v.pollutant_name
            by_pollutant[poll] = by_pollutant.get(poll, 0) + 1

            # Type
            vtype = v.violation_type.display_name
            by_type[vtype] = by_type.get(vtype, 0) + 1

        return {
            "total_violations": len(self.violations),
            "active_violations": active_count,
            "total_duration_hours": sum(durations) / 60,
            "avg_duration_minutes": sum(durations) / len(durations),
            "avg_exceedance_percent": sum(exceedances) / len(exceedances),
            "max_exceedance_percent": max(exceedances),
            "by_severity": by_severity,
            "by_status": by_status,
            "by_pollutant": by_pollutant,
            "by_type": by_type
        }

    def to_plotly_json(self) -> str:
        """Export main timeline to Plotly JSON."""
        return json.dumps(self.build_gantt_timeline(), indent=2)

    def to_html(self) -> str:
        """
        Generate standalone HTML violation timeline.

        Returns:
            HTML string
        """
        charts = {
            "gantt": self.build_gantt_timeline(),
            "scatter": self.build_scatter_timeline(),
            "status": self.build_status_breakdown(),
            "severity": self.build_severity_distribution(),
            "pollutant": self.build_pollutant_breakdown(),
            "monthly": self.build_monthly_trend(),
            "duration": self.build_duration_analysis(),
            "regulatory": self.build_regulatory_timeline()
        }

        stats = self.get_summary_statistics()

        charts_json = json.dumps(charts)
        stats_json = json.dumps(stats)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Violation Timeline</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f5f6fa;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #7f8c8d;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-card.alert .value {{
            color: #e74c3c;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Violation Timeline Dashboard</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Violations</h3>
                <div class="value" id="stat-total">-</div>
            </div>
            <div class="stat-card alert">
                <h3>Active Violations</h3>
                <div class="value" id="stat-active">-</div>
            </div>
            <div class="stat-card">
                <h3>Total Duration</h3>
                <div class="value" id="stat-duration">-</div>
            </div>
            <div class="stat-card">
                <h3>Avg Exceedance</h3>
                <div class="value" id="stat-exceedance">-</div>
            </div>
        </div>

        <div class="chart-container full-width" id="gantt-chart"></div>

        <div class="charts-grid">
            <div class="chart-container" id="status-chart"></div>
            <div class="chart-container" id="severity-chart"></div>
            <div class="chart-container" id="pollutant-chart"></div>
            <div class="chart-container" id="monthly-chart"></div>
        </div>

        <h2>Detailed Analysis</h2>
        <div class="charts-grid">
            <div class="chart-container" id="scatter-chart"></div>
            <div class="chart-container" id="duration-chart"></div>
        </div>

        <div class="chart-container" id="regulatory-chart"></div>
    </div>

    <script>
        const charts = {charts_json};
        const stats = {stats_json};

        // Update stats
        document.getElementById('stat-total').textContent = stats.total_violations;
        document.getElementById('stat-active').textContent = stats.active_violations;
        document.getElementById('stat-duration').textContent = stats.total_duration_hours.toFixed(1) + 'h';
        document.getElementById('stat-exceedance').textContent = stats.avg_exceedance_percent.toFixed(1) + '%';

        // Render charts
        const chartConfigs = {{responsive: true, displayModeBar: true, displaylogo: false}};

        Plotly.newPlot('gantt-chart', charts.gantt.data, charts.gantt.layout, chartConfigs);
        Plotly.newPlot('status-chart', charts.status.data, charts.status.layout, chartConfigs);
        Plotly.newPlot('severity-chart', charts.severity.data, charts.severity.layout, chartConfigs);
        Plotly.newPlot('pollutant-chart', charts.pollutant.data, charts.pollutant.layout, chartConfigs);
        Plotly.newPlot('monthly-chart', charts.monthly.data, charts.monthly.layout, chartConfigs);
        Plotly.newPlot('scatter-chart', charts.scatter.data, charts.scatter.layout, chartConfigs);
        Plotly.newPlot('duration-chart', charts.duration.data, charts.duration.layout, chartConfigs);
        Plotly.newPlot('regulatory-chart', charts.regulatory.data, charts.regulatory.layout, chartConfigs);

        // Make responsive
        window.addEventListener('resize', () => {{
            ['gantt', 'status', 'severity', 'pollutant', 'monthly', 'scatter', 'duration', 'regulatory']
                .forEach(id => Plotly.Plots.resize(id + '-chart'));
        }});
    </script>
</body>
</html>"""

        return html

    def export_for_report(self) -> Dict[str, Any]:
        """
        Export violation data in format suitable for regulatory reports.

        Returns:
            Report-ready data structure
        """
        return {
            "summary": self.get_summary_statistics(),
            "violations": [v.to_dict() for v in self.violations],
            "charts": {
                "timeline": self.build_gantt_timeline(),
                "status": self.build_status_breakdown(),
                "severity": self.build_severity_distribution()
            },
            "generated_at": datetime.now().isoformat()
        }


def create_sample_violations(count: int = 20) -> List[ViolationRecord]:
    """
    Create sample violation records for testing.

    Args:
        count: Number of violations to generate

    Returns:
        List of sample violation records
    """
    import random
    random.seed(42)

    pollutants = [
        ("NOx", "Nitrogen Oxides", "lb/hr", 200.0),
        ("SO2", "Sulfur Dioxide", "lb/hr", 100.0),
        ("PM", "Particulate Matter", "lb/hr", 25.0),
        ("CO", "Carbon Monoxide", "ppm", 500.0)
    ]

    violation_types = list(ViolationType)
    severities = list(ViolationSeverity)
    statuses = list(ViolationStatus)

    violations = []
    base_date = datetime(2024, 1, 1)

    for i in range(count):
        pollutant = random.choice(pollutants)
        pol_id, pol_name, unit, limit = pollutant

        start = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
        duration = random.randint(15, 480)
        exceedance_pct = random.uniform(5, 50)

        severity = random.choice(severities)
        status = random.choice(statuses)

        v = ViolationRecord(
            violation_id=f"VIO-2024-{i+1:04d}",
            violation_type=random.choice(violation_types),
            pollutant=pol_id,
            pollutant_name=pol_name,
            start_time=start.isoformat() + "Z",
            end_time=(start + timedelta(minutes=duration)).isoformat() + "Z" if status != ViolationStatus.ACTIVE else None,
            duration_minutes=duration,
            exceedance_value=limit * (1 + exceedance_pct / 100),
            permit_limit=limit,
            unit=unit,
            exceedance_percent=exceedance_pct,
            severity=severity,
            status=status,
            source_unit=f"Unit-{random.randint(1, 5)}",
            affected_limit="1-hour rolling average",
            root_cause="Equipment malfunction" if random.random() > 0.5 else "Fuel quality",
            corrective_action="Maintenance performed" if status in [ViolationStatus.RESOLVED, ViolationStatus.CLOSED] else None
        )

        # Add regulatory responses for some violations
        if severity in [ViolationSeverity.MAJOR, ViolationSeverity.CRITICAL] and random.random() > 0.5:
            v.regulatory_responses.append(RegulatoryResponse(
                response_type=random.choice(["notice", "warning", "citation"]),
                agency="EPA Region 9",
                date=(start + timedelta(days=random.randint(5, 30))).strftime("%Y-%m-%d"),
                reference_number=f"EPA-{random.randint(1000, 9999)}",
                description="Notice of violation issued",
                penalty_amount=random.uniform(1000, 50000) if random.random() > 0.5 else None
            ))

        violations.append(v)

    return violations


if __name__ == "__main__":
    # Demo usage
    violations = create_sample_violations(25)
    config = TimelineConfig(
        title="Facility Violation Timeline",
        color_blind_safe=False
    )

    timeline = ViolationTimelineChart(violations, config)

    print("Timeline JSON (first 500 chars):")
    print(timeline.to_plotly_json()[:500])

    stats = timeline.get_summary_statistics()
    print(f"\nSummary Statistics:")
    print(f"  Total violations: {stats['total_violations']}")
    print(f"  Active violations: {stats['active_violations']}")
    print(f"  Total duration: {stats['total_duration_hours']:.1f} hours")
    print(f"  Avg exceedance: {stats['avg_exceedance_percent']:.1f}%")
