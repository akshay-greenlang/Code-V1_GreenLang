"""
GL-010 EMISSIONWATCH - Compliance Dashboard Visualization

Main compliance status visualization module for the EmissionsComplianceAgent.
Generates interactive compliance dashboards with Plotly-compatible JSON output.

Author: GreenLang Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from datetime import datetime, timedelta
import json
import math
from abc import ABC, abstractmethod


class ComplianceStatus(Enum):
    """Compliance status enumeration with severity levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    UNKNOWN = "unknown"
    PENDING = "pending"
    EXEMPT = "exempt"

    @property
    def severity(self) -> int:
        """Return severity level (higher = worse)."""
        severity_map = {
            "compliant": 0,
            "exempt": 0,
            "pending": 1,
            "unknown": 2,
            "warning": 3,
            "violation": 4
        }
        return severity_map.get(self.value, 2)

    @property
    def color(self) -> str:
        """Return color for status."""
        color_map = {
            "compliant": "#2ECC71",
            "warning": "#F39C12",
            "violation": "#E74C3C",
            "unknown": "#95A5A6",
            "pending": "#3498DB",
            "exempt": "#9B59B6"
        }
        return color_map.get(self.value, "#95A5A6")

    @property
    def color_blind_safe(self) -> str:
        """Return color-blind safe color for status."""
        color_map = {
            "compliant": "#009E73",
            "warning": "#E69F00",
            "violation": "#D55E00",
            "unknown": "#999999",
            "pending": "#0072B2",
            "exempt": "#CC79A7"
        }
        return color_map.get(self.value, "#999999")


class ViolationType(Enum):
    """Types of compliance violations."""
    EMISSION_EXCEEDANCE = "emission_exceedance"
    OPACITY_VIOLATION = "opacity_violation"
    REPORTING_FAILURE = "reporting_failure"
    MONITORING_GAP = "monitoring_gap"
    PERMIT_DEVIATION = "permit_deviation"
    STARTUP_SHUTDOWN = "startup_shutdown"
    MALFUNCTION = "malfunction"


@dataclass
class PollutantStatus:
    """Status information for a single pollutant."""
    pollutant_id: str
    pollutant_name: str
    current_value: float
    unit: str
    permit_limit: float
    averaging_period: str
    status: ComplianceStatus
    margin_percent: float
    trend: str  # "increasing", "decreasing", "stable"
    last_updated: str
    data_quality: float  # 0-100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pollutant_id": self.pollutant_id,
            "pollutant_name": self.pollutant_name,
            "current_value": self.current_value,
            "unit": self.unit,
            "permit_limit": self.permit_limit,
            "averaging_period": self.averaging_period,
            "status": self.status.value,
            "margin_percent": self.margin_percent,
            "trend": self.trend,
            "last_updated": self.last_updated,
            "data_quality": self.data_quality
        }


@dataclass
class Violation:
    """Violation record data structure."""
    violation_id: str
    violation_type: ViolationType
    pollutant: str
    start_time: str
    end_time: Optional[str]
    duration_minutes: int
    exceedance_value: float
    permit_limit: float
    exceedance_percent: float
    severity: str  # "minor", "moderate", "major", "critical"
    status: str  # "active", "resolved", "under_review", "reported"
    regulatory_action: Optional[str]
    root_cause: Optional[str]
    corrective_action: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type.value,
            "pollutant": self.pollutant,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": self.duration_minutes,
            "exceedance_value": self.exceedance_value,
            "permit_limit": self.permit_limit,
            "exceedance_percent": self.exceedance_percent,
            "severity": self.severity,
            "status": self.status,
            "regulatory_action": self.regulatory_action,
            "root_cause": self.root_cause,
            "corrective_action": self.corrective_action
        }


@dataclass
class ComplianceDashboardData:
    """Data structure for compliance dashboard."""
    timestamp: str
    facility_id: str
    facility_name: str
    jurisdiction: str
    permit_number: str
    pollutants: Dict[str, PollutantStatus]
    overall_status: ComplianceStatus
    active_violations: List[Violation]
    margin_to_limits: Dict[str, float]
    reporting_period: str
    data_completeness: float
    next_report_due: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "jurisdiction": self.jurisdiction,
            "permit_number": self.permit_number,
            "pollutants": {k: v.to_dict() for k, v in self.pollutants.items()},
            "overall_status": self.overall_status.value,
            "active_violations": [v.to_dict() for v in self.active_violations],
            "margin_to_limits": self.margin_to_limits,
            "reporting_period": self.reporting_period,
            "data_completeness": self.data_completeness,
            "next_report_due": self.next_report_due,
            "notes": self.notes
        }


class ColorScheme:
    """Color scheme management for visualizations."""

    COMPLIANCE_COLORS = {
        "compliant": "#2ECC71",
        "warning": "#F39C12",
        "violation": "#E74C3C",
        "unknown": "#95A5A6"
    }

    COMPLIANCE_COLORS_BLIND_SAFE = {
        "compliant": "#009E73",
        "warning": "#E69F00",
        "violation": "#D55E00",
        "unknown": "#999999"
    }

    POLLUTANT_COLORS = {
        "NOx": "#E74C3C",
        "SO2": "#9B59B6",
        "PM": "#3498DB",
        "CO": "#2ECC71",
        "VOC": "#F39C12",
        "Hg": "#1ABC9C",
        "HCl": "#E91E63",
        "CO2": "#607D8B"
    }

    GRADIENT_SCALES = {
        "compliance": [
            [0.0, "#2ECC71"],
            [0.5, "#F39C12"],
            [0.75, "#E74C3C"],
            [1.0, "#C0392B"]
        ],
        "severity": [
            [0.0, "#3498DB"],
            [0.33, "#F39C12"],
            [0.66, "#E74C3C"],
            [1.0, "#8E44AD"]
        ]
    }

    @classmethod
    def get_color(cls, status: ComplianceStatus, color_blind_safe: bool = False) -> str:
        """Get color for compliance status."""
        colors = cls.COMPLIANCE_COLORS_BLIND_SAFE if color_blind_safe else cls.COMPLIANCE_COLORS
        return colors.get(status.value, "#95A5A6")

    @classmethod
    def get_pollutant_color(cls, pollutant: str) -> str:
        """Get color for pollutant."""
        return cls.POLLUTANT_COLORS.get(pollutant, "#3498DB")


class ChartBase(ABC):
    """Base class for chart generation."""

    def __init__(self, title: str, color_blind_safe: bool = False):
        self.title = title
        self.color_blind_safe = color_blind_safe
        self._data: List[Dict] = []
        self._layout: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {
            "responsive": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False
        }

    @abstractmethod
    def build(self) -> Dict[str, Any]:
        """Build chart data structure."""
        pass

    def to_plotly_json(self) -> str:
        """Export to Plotly JSON format."""
        return json.dumps(self.build(), indent=2)

    def _get_base_layout(self) -> Dict[str, Any]:
        """Get base layout configuration."""
        return {
            "title": {
                "text": self.title,
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "font": {"family": "Arial, sans-serif"},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA",
            "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
            "hovermode": "closest"
        }


class StatusMatrixChart(ChartBase):
    """Multi-pollutant compliance status matrix visualization."""

    def __init__(
        self,
        data: ComplianceDashboardData,
        color_blind_safe: bool = False
    ):
        super().__init__("Compliance Status Matrix", color_blind_safe)
        self.dashboard_data = data

    def build(self) -> Dict[str, Any]:
        """Build status matrix chart."""
        pollutants = list(self.dashboard_data.pollutants.keys())
        status_values = []
        hover_texts = []
        colors = []

        for pollutant_id in pollutants:
            ps = self.dashboard_data.pollutants[pollutant_id]
            status_values.append(ps.status.severity)

            hover_text = (
                f"<b>{ps.pollutant_name}</b><br>"
                f"Status: {ps.status.value.title()}<br>"
                f"Value: {ps.current_value:.2f} {ps.unit}<br>"
                f"Limit: {ps.permit_limit:.2f} {ps.unit}<br>"
                f"Margin: {ps.margin_percent:.1f}%<br>"
                f"Trend: {ps.trend.title()}"
            )
            hover_texts.append(hover_text)

            if self.color_blind_safe:
                colors.append(ps.status.color_blind_safe)
            else:
                colors.append(ps.status.color)

        # Build heatmap trace
        trace = {
            "type": "heatmap",
            "z": [[status_values]],
            "x": pollutants,
            "y": [self.dashboard_data.facility_name],
            "text": [[hover_texts]],
            "hovertemplate": "%{text}<extra></extra>",
            "colorscale": ColorScheme.GRADIENT_SCALES["compliance"],
            "showscale": True,
            "colorbar": {
                "title": "Severity",
                "tickvals": [0, 1, 2, 3, 4],
                "ticktext": ["Compliant", "Pending", "Unknown", "Warning", "Violation"]
            }
        }

        layout = self._get_base_layout()
        layout.update({
            "xaxis": {
                "title": "Pollutants",
                "tickangle": -45
            },
            "yaxis": {
                "title": "Facility"
            }
        })

        return {
            "data": [trace],
            "layout": layout,
            "config": self._config
        }


class GaugeChart(ChartBase):
    """Gauge chart showing current value vs permit limit."""

    def __init__(
        self,
        pollutant_status: PollutantStatus,
        color_blind_safe: bool = False
    ):
        super().__init__(
            f"{pollutant_status.pollutant_name} Compliance Gauge",
            color_blind_safe
        )
        self.pollutant_status = ps = pollutant_status

    def build(self) -> Dict[str, Any]:
        """Build gauge chart."""
        ps = self.pollutant_status

        # Calculate percentage of limit
        percent_of_limit = (ps.current_value / ps.permit_limit) * 100 if ps.permit_limit > 0 else 0

        # Determine gauge ranges and colors
        if self.color_blind_safe:
            colors = ["#009E73", "#E69F00", "#D55E00"]
        else:
            colors = ["#2ECC71", "#F39C12", "#E74C3C"]

        trace = {
            "type": "indicator",
            "mode": "gauge+number+delta",
            "value": ps.current_value,
            "title": {
                "text": f"{ps.pollutant_name}<br><span style='font-size:0.8em;color:#666'>{ps.averaging_period}</span>"
            },
            "delta": {
                "reference": ps.permit_limit,
                "increasing": {"color": colors[2]},
                "decreasing": {"color": colors[0]}
            },
            "number": {
                "suffix": f" {ps.unit}",
                "font": {"size": 24}
            },
            "gauge": {
                "axis": {
                    "range": [0, ps.permit_limit * 1.5],
                    "tickwidth": 1,
                    "tickcolor": "#2C3E50"
                },
                "bar": {"color": "#2C3E50"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#BDC3C7",
                "steps": [
                    {"range": [0, ps.permit_limit * 0.7], "color": colors[0]},
                    {"range": [ps.permit_limit * 0.7, ps.permit_limit * 0.9], "color": colors[1]},
                    {"range": [ps.permit_limit * 0.9, ps.permit_limit * 1.5], "color": colors[2]}
                ],
                "threshold": {
                    "line": {"color": "#E74C3C", "width": 4},
                    "thickness": 0.75,
                    "value": ps.permit_limit
                }
            }
        }

        layout = self._get_base_layout()
        layout.update({
            "height": 300,
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20}
        })

        return {
            "data": [trace],
            "layout": layout,
            "config": self._config
        }


class TrendChart(ChartBase):
    """Time-series trend chart for compliance data."""

    def __init__(
        self,
        pollutant: str,
        historical_data: List[Dict[str, Any]],
        permit_limit: float,
        unit: str,
        color_blind_safe: bool = False
    ):
        super().__init__(f"{pollutant} Compliance Trend", color_blind_safe)
        self.pollutant = pollutant
        self.historical_data = historical_data
        self.permit_limit = permit_limit
        self.unit = unit

    def build(self) -> Dict[str, Any]:
        """Build trend chart."""
        timestamps = [d["timestamp"] for d in self.historical_data]
        values = [d["value"] for d in self.historical_data]

        # Main trend line
        trend_trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "name": self.pollutant,
            "x": timestamps,
            "y": values,
            "line": {
                "color": ColorScheme.get_pollutant_color(self.pollutant),
                "width": 2
            },
            "marker": {"size": 6},
            "hovertemplate": "<b>%{x}</b><br>Value: %{y:.2f} " + self.unit + "<extra></extra>"
        }

        # Permit limit line
        limit_trace = {
            "type": "scatter",
            "mode": "lines",
            "name": "Permit Limit",
            "x": timestamps,
            "y": [self.permit_limit] * len(timestamps),
            "line": {
                "color": "#E74C3C" if not self.color_blind_safe else "#D55E00",
                "width": 2,
                "dash": "dash"
            },
            "hovertemplate": "Permit Limit: %{y:.2f} " + self.unit + "<extra></extra>"
        }

        # Warning threshold line (90% of limit)
        warning_threshold = self.permit_limit * 0.9
        warning_trace = {
            "type": "scatter",
            "mode": "lines",
            "name": "Warning Threshold (90%)",
            "x": timestamps,
            "y": [warning_threshold] * len(timestamps),
            "line": {
                "color": "#F39C12" if not self.color_blind_safe else "#E69F00",
                "width": 1,
                "dash": "dot"
            },
            "hovertemplate": "Warning (90%%): %{y:.2f} " + self.unit + "<extra></extra>"
        }

        # Add exceedance highlighting
        exceedance_trace = self._build_exceedance_trace(timestamps, values)

        layout = self._get_base_layout()
        layout.update({
            "xaxis": {
                "title": "Time",
                "type": "date",
                "rangeslider": {"visible": True},
                "rangeselector": {
                    "buttons": [
                        {"count": 1, "label": "1h", "step": "hour", "stepmode": "backward"},
                        {"count": 24, "label": "24h", "step": "hour", "stepmode": "backward"},
                        {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
                        {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                        {"step": "all", "label": "All"}
                    ]
                }
            },
            "yaxis": {
                "title": f"{self.pollutant} ({self.unit})",
                "rangemode": "tozero"
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            },
            "shapes": self._build_exceedance_shapes(timestamps, values)
        })

        traces = [trend_trace, limit_trace, warning_trace]
        if exceedance_trace:
            traces.append(exceedance_trace)

        return {
            "data": traces,
            "layout": layout,
            "config": self._config
        }

    def _build_exceedance_trace(
        self,
        timestamps: List[str],
        values: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Build trace highlighting exceedances."""
        exceedance_x = []
        exceedance_y = []

        for i, (ts, val) in enumerate(zip(timestamps, values)):
            if val > self.permit_limit:
                exceedance_x.append(ts)
                exceedance_y.append(val)

        if not exceedance_x:
            return None

        return {
            "type": "scatter",
            "mode": "markers",
            "name": "Exceedances",
            "x": exceedance_x,
            "y": exceedance_y,
            "marker": {
                "color": "#E74C3C" if not self.color_blind_safe else "#D55E00",
                "size": 12,
                "symbol": "x"
            },
            "hovertemplate": "<b>EXCEEDANCE</b><br>%{x}<br>Value: %{y:.2f} " + self.unit + "<extra></extra>"
        }

    def _build_exceedance_shapes(
        self,
        timestamps: List[str],
        values: List[float]
    ) -> List[Dict[str, Any]]:
        """Build shapes to highlight exceedance periods."""
        shapes = []
        in_exceedance = False
        start_idx = 0

        for i, val in enumerate(values):
            if val > self.permit_limit and not in_exceedance:
                in_exceedance = True
                start_idx = i
            elif val <= self.permit_limit and in_exceedance:
                in_exceedance = False
                shapes.append({
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": timestamps[start_idx],
                    "y0": 0,
                    "x1": timestamps[i],
                    "y1": 1,
                    "fillcolor": "rgba(231, 76, 60, 0.1)",
                    "line": {"width": 0}
                })

        # Handle ongoing exceedance
        if in_exceedance:
            shapes.append({
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": timestamps[start_idx],
                "y0": 0,
                "x1": timestamps[-1],
                "y1": 1,
                "fillcolor": "rgba(231, 76, 60, 0.1)",
                "line": {"width": 0}
            })

        return shapes


class ViolationSummaryChart(ChartBase):
    """Summary visualization of active violations."""

    def __init__(
        self,
        violations: List[Violation],
        color_blind_safe: bool = False
    ):
        super().__init__("Active Violations Summary", color_blind_safe)
        self.violations = violations

    def build(self) -> Dict[str, Any]:
        """Build violation summary chart."""
        if not self.violations:
            return self._build_no_violations_chart()

        # Group by pollutant
        pollutant_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {"minor": 0, "moderate": 0, "major": 0, "critical": 0}

        for v in self.violations:
            pollutant_counts[v.pollutant] = pollutant_counts.get(v.pollutant, 0) + 1
            severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1

        # Build traces
        traces = []

        # Violations by pollutant
        traces.append({
            "type": "bar",
            "name": "By Pollutant",
            "x": list(pollutant_counts.keys()),
            "y": list(pollutant_counts.values()),
            "marker": {
                "color": [ColorScheme.get_pollutant_color(p) for p in pollutant_counts.keys()]
            },
            "xaxis": "x",
            "yaxis": "y"
        })

        # Violations by severity
        severity_colors = {
            "minor": "#3498DB",
            "moderate": "#F39C12",
            "major": "#E74C3C",
            "critical": "#8E44AD"
        }

        traces.append({
            "type": "pie",
            "name": "By Severity",
            "labels": list(severity_counts.keys()),
            "values": list(severity_counts.values()),
            "marker": {
                "colors": [severity_colors[s] for s in severity_counts.keys()]
            },
            "domain": {"x": [0.6, 1], "y": [0, 1]},
            "hole": 0.4,
            "textinfo": "label+percent"
        })

        layout = self._get_base_layout()
        layout.update({
            "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
            "xaxis": {
                "title": "Pollutant",
                "domain": [0, 0.5]
            },
            "yaxis": {
                "title": "Violation Count"
            },
            "annotations": [
                {
                    "text": f"<b>{len(self.violations)}</b><br>Active",
                    "x": 0.8,
                    "y": 0.5,
                    "font": {"size": 16},
                    "showarrow": False
                }
            ]
        })

        return {
            "data": traces,
            "layout": layout,
            "config": self._config
        }

    def _build_no_violations_chart(self) -> Dict[str, Any]:
        """Build chart indicating no violations."""
        trace = {
            "type": "indicator",
            "mode": "number+delta",
            "value": 0,
            "title": {"text": "Active Violations"},
            "number": {
                "font": {"color": "#2ECC71", "size": 72}
            }
        }

        layout = self._get_base_layout()
        layout.update({
            "annotations": [
                {
                    "text": "All Clear - No Active Violations",
                    "x": 0.5,
                    "y": 0.2,
                    "font": {"size": 18, "color": "#2ECC71"},
                    "showarrow": False
                }
            ]
        })

        return {
            "data": [trace],
            "layout": layout,
            "config": self._config
        }


class ComplianceMarginChart(ChartBase):
    """Chart showing margin to limits for all pollutants."""

    def __init__(
        self,
        margin_data: Dict[str, float],
        color_blind_safe: bool = False
    ):
        super().__init__("Margin to Compliance Limits", color_blind_safe)
        self.margin_data = margin_data

    def build(self) -> Dict[str, Any]:
        """Build margin chart."""
        pollutants = list(self.margin_data.keys())
        margins = list(self.margin_data.values())

        # Color based on margin
        colors = []
        for margin in margins:
            if margin >= 30:
                colors.append("#2ECC71" if not self.color_blind_safe else "#009E73")
            elif margin >= 10:
                colors.append("#F39C12" if not self.color_blind_safe else "#E69F00")
            else:
                colors.append("#E74C3C" if not self.color_blind_safe else "#D55E00")

        trace = {
            "type": "bar",
            "x": pollutants,
            "y": margins,
            "marker": {
                "color": colors,
                "line": {"color": "#2C3E50", "width": 1}
            },
            "text": [f"{m:.1f}%" for m in margins],
            "textposition": "outside",
            "hovertemplate": "<b>%{x}</b><br>Margin: %{y:.1f}%<extra></extra>"
        }

        # Add threshold lines
        shapes = [
            {
                "type": "line",
                "x0": -0.5,
                "x1": len(pollutants) - 0.5,
                "y0": 10,
                "y1": 10,
                "line": {"color": "#E74C3C", "width": 2, "dash": "dash"}
            },
            {
                "type": "line",
                "x0": -0.5,
                "x1": len(pollutants) - 0.5,
                "y0": 30,
                "y1": 30,
                "line": {"color": "#2ECC71", "width": 2, "dash": "dash"}
            }
        ]

        layout = self._get_base_layout()
        layout.update({
            "xaxis": {
                "title": "Pollutant",
                "tickangle": -45
            },
            "yaxis": {
                "title": "Margin to Limit (%)",
                "range": [min(0, min(margins) - 10), max(margins) + 20]
            },
            "shapes": shapes,
            "annotations": [
                {
                    "text": "Warning Zone",
                    "x": len(pollutants) - 0.3,
                    "y": 5,
                    "font": {"color": "#E74C3C", "size": 10},
                    "showarrow": False
                },
                {
                    "text": "Safe Zone",
                    "x": len(pollutants) - 0.3,
                    "y": 35,
                    "font": {"color": "#2ECC71", "size": 10},
                    "showarrow": False
                }
            ]
        })

        return {
            "data": [trace],
            "layout": layout,
            "config": self._config
        }


class ComplianceDashboard:
    """
    Generate interactive compliance status dashboard.
    Outputs Plotly-compatible JSON for web rendering.
    """

    def __init__(
        self,
        data: ComplianceDashboardData,
        color_blind_safe: bool = False,
        responsive: bool = True
    ):
        """
        Initialize compliance dashboard.

        Args:
            data: Dashboard data structure
            color_blind_safe: Use color-blind safe palette
            responsive: Enable responsive layout
        """
        self.data = data
        self.color_blind_safe = color_blind_safe
        self.responsive = responsive
        self._charts: Dict[str, ChartBase] = {}
        self._build_charts()

    def _build_charts(self) -> None:
        """Build all dashboard charts."""
        # Status matrix
        self._charts["status_matrix"] = StatusMatrixChart(
            self.data,
            self.color_blind_safe
        )

        # Gauge charts for each pollutant
        for pollutant_id, ps in self.data.pollutants.items():
            self._charts[f"gauge_{pollutant_id}"] = GaugeChart(
                ps,
                self.color_blind_safe
            )

        # Violation summary
        self._charts["violations"] = ViolationSummaryChart(
            self.data.active_violations,
            self.color_blind_safe
        )

        # Margin chart
        self._charts["margins"] = ComplianceMarginChart(
            self.data.margin_to_limits,
            self.color_blind_safe
        )

    def generate_status_matrix(self) -> Dict[str, Any]:
        """Generate multi-pollutant compliance status matrix."""
        return self._charts["status_matrix"].build()

    def generate_gauge_chart(self, pollutant_id: str) -> Dict[str, Any]:
        """
        Generate gauge chart showing current vs limit.

        Args:
            pollutant_id: Pollutant identifier

        Returns:
            Plotly chart dictionary
        """
        chart_key = f"gauge_{pollutant_id}"
        if chart_key not in self._charts:
            raise ValueError(f"Unknown pollutant: {pollutant_id}")
        return self._charts[chart_key].build()

    def generate_all_gauges(self) -> List[Dict[str, Any]]:
        """Generate gauge charts for all pollutants."""
        gauges = []
        for pollutant_id in self.data.pollutants.keys():
            gauges.append(self.generate_gauge_chart(pollutant_id))
        return gauges

    def generate_trend_chart(
        self,
        pollutant_id: str,
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate compliance trend over time.

        Args:
            pollutant_id: Pollutant identifier
            historical_data: List of {timestamp, value} dictionaries

        Returns:
            Plotly chart dictionary
        """
        if pollutant_id not in self.data.pollutants:
            raise ValueError(f"Unknown pollutant: {pollutant_id}")

        ps = self.data.pollutants[pollutant_id]
        chart = TrendChart(
            ps.pollutant_name,
            historical_data,
            ps.permit_limit,
            ps.unit,
            self.color_blind_safe
        )
        return chart.build()

    def generate_violation_summary(self) -> Dict[str, Any]:
        """Generate violation summary chart."""
        return self._charts["violations"].build()

    def generate_margin_chart(self) -> Dict[str, Any]:
        """Generate margin to limits chart."""
        return self._charts["margins"].build()

    def generate_overview_card(self) -> Dict[str, Any]:
        """Generate overview indicator card."""
        status = self.data.overall_status
        color = status.color_blind_safe if self.color_blind_safe else status.color

        trace = {
            "type": "indicator",
            "mode": "number+delta",
            "value": len(self.data.active_violations),
            "title": {
                "text": f"<b>{self.data.facility_name}</b><br>"
                        f"<span style='font-size:0.7em;color:{color}'>"
                        f"{status.value.upper()}</span>"
            },
            "number": {
                "suffix": " active violations",
                "font": {"size": 32}
            },
            "delta": {
                "reference": 0,
                "increasing": {"color": "#E74C3C"},
                "decreasing": {"color": "#2ECC71"}
            }
        }

        layout = {
            "title": {"text": ""},
            "paper_bgcolor": "white",
            "font": {"family": "Arial, sans-serif"},
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
            "height": 200
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True, "displayModeBar": False}
        }

    def to_plotly_json(self) -> str:
        """
        Export complete dashboard to Plotly JSON format.

        Returns:
            JSON string containing all dashboard charts
        """
        dashboard = {
            "metadata": {
                "timestamp": self.data.timestamp,
                "facility_id": self.data.facility_id,
                "facility_name": self.data.facility_name,
                "jurisdiction": self.data.jurisdiction,
                "overall_status": self.data.overall_status.value,
                "color_blind_safe": self.color_blind_safe
            },
            "charts": {
                "overview": self.generate_overview_card(),
                "status_matrix": self.generate_status_matrix(),
                "gauges": self.generate_all_gauges(),
                "violations": self.generate_violation_summary(),
                "margins": self.generate_margin_chart()
            },
            "data": self.data.to_dict()
        }

        return json.dumps(dashboard, indent=2)

    def to_html(self, include_plotly_js: bool = True) -> str:
        """
        Generate standalone HTML dashboard.

        Args:
            include_plotly_js: Include Plotly.js library

        Returns:
            HTML string
        """
        charts_json = self.to_plotly_json()

        plotly_cdn = ""
        if include_plotly_js:
            plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compliance Dashboard - {self.data.facility_name}</title>
    {plotly_cdn}
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Arial', sans-serif;
            background-color: #f5f6fa;
            color: #2c3e50;
            line-height: 1.6;
        }}
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 24px;
            font-weight: 600;
        }}
        .header .status-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-compliant {{ background-color: #2ecc71; }}
        .status-warning {{ background-color: #f39c12; }}
        .status-violation {{ background-color: #e74c3c; }}
        .status-unknown {{ background-color: #95a5a6; }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart-card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .chart-card.full-width {{
            grid-column: 1 / -1;
        }}
        .gauge-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .info-panel {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            padding: 10px;
            border-left: 3px solid #3498db;
        }}
        .info-item label {{
            display: block;
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .info-item value {{
            display: block;
            font-size: 16px;
            font-weight: 600;
            color: #2c3e50;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 12px;
        }}
        @media (max-width: 768px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
            .gauge-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        @media print {{
            .header {{
                background: #2c3e50 !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <div>
                <h1>{self.data.facility_name}</h1>
                <p>{self.data.jurisdiction} | Permit: {self.data.permit_number}</p>
            </div>
            <div class="status-badge status-{self.data.overall_status.value}">
                {self.data.overall_status.value.upper()}
            </div>
        </div>

        <div class="info-panel">
            <div class="info-grid">
                <div class="info-item">
                    <label>Report Period</label>
                    <value>{self.data.reporting_period}</value>
                </div>
                <div class="info-item">
                    <label>Data Completeness</label>
                    <value>{self.data.data_completeness:.1f}%</value>
                </div>
                <div class="info-item">
                    <label>Active Violations</label>
                    <value>{len(self.data.active_violations)}</value>
                </div>
                <div class="info-item">
                    <label>Last Updated</label>
                    <value>{self.data.timestamp}</value>
                </div>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-card full-width" id="status-matrix"></div>
            <div class="chart-card" id="violations-chart"></div>
            <div class="chart-card" id="margins-chart"></div>
        </div>

        <h2 style="margin: 30px 0 15px;">Pollutant Gauges</h2>
        <div class="gauge-grid" id="gauges-container"></div>

        <div class="footer">
            <p>Generated by GreenLang EMISSIONWATCH | GL-010</p>
            <p>Data timestamp: {self.data.timestamp}</p>
        </div>
    </div>

    <script>
        const dashboardData = {charts_json};

        // Render status matrix
        Plotly.newPlot('status-matrix',
            dashboardData.charts.status_matrix.data,
            dashboardData.charts.status_matrix.layout,
            dashboardData.charts.status_matrix.config
        );

        // Render violations chart
        Plotly.newPlot('violations-chart',
            dashboardData.charts.violations.data,
            dashboardData.charts.violations.layout,
            dashboardData.charts.violations.config
        );

        // Render margins chart
        Plotly.newPlot('margins-chart',
            dashboardData.charts.margins.data,
            dashboardData.charts.margins.layout,
            dashboardData.charts.margins.config
        );

        // Render gauge charts
        const gaugesContainer = document.getElementById('gauges-container');
        dashboardData.charts.gauges.forEach((gauge, index) => {{
            const gaugeDiv = document.createElement('div');
            gaugeDiv.className = 'chart-card';
            gaugeDiv.id = 'gauge-' + index;
            gaugesContainer.appendChild(gaugeDiv);

            Plotly.newPlot(gaugeDiv.id, gauge.data, gauge.layout, gauge.config);
        }});

        // Make charts responsive
        window.addEventListener('resize', () => {{
            Plotly.Plots.resize('status-matrix');
            Plotly.Plots.resize('violations-chart');
            Plotly.Plots.resize('margins-chart');
            dashboardData.charts.gauges.forEach((_, index) => {{
                Plotly.Plots.resize('gauge-' + index);
            }});
        }});
    </script>
</body>
</html>"""

        return html

    def to_d3_json(self) -> str:
        """
        Export data in D3.js-compatible format.

        Returns:
            JSON string with D3-friendly data structures
        """
        d3_data = {
            "metadata": {
                "timestamp": self.data.timestamp,
                "facility": {
                    "id": self.data.facility_id,
                    "name": self.data.facility_name,
                    "jurisdiction": self.data.jurisdiction,
                    "permit": self.data.permit_number
                },
                "status": self.data.overall_status.value
            },
            "pollutants": [
                {
                    "id": pid,
                    "name": ps.pollutant_name,
                    "value": ps.current_value,
                    "limit": ps.permit_limit,
                    "unit": ps.unit,
                    "margin": ps.margin_percent,
                    "status": ps.status.value,
                    "color": ps.status.color_blind_safe if self.color_blind_safe else ps.status.color
                }
                for pid, ps in self.data.pollutants.items()
            ],
            "violations": [v.to_dict() for v in self.data.active_violations],
            "margins": [
                {"pollutant": k, "margin": v}
                for k, v in self.data.margin_to_limits.items()
            ]
        }

        return json.dumps(d3_data, indent=2)


def create_sample_dashboard_data() -> ComplianceDashboardData:
    """Create sample dashboard data for testing."""
    pollutants = {
        "NOx": PollutantStatus(
            pollutant_id="NOx",
            pollutant_name="Nitrogen Oxides",
            current_value=145.5,
            unit="lb/hr",
            permit_limit=200.0,
            averaging_period="1-hour rolling",
            status=ComplianceStatus.COMPLIANT,
            margin_percent=27.25,
            trend="stable",
            last_updated="2024-01-15T14:30:00Z",
            data_quality=98.5
        ),
        "SO2": PollutantStatus(
            pollutant_id="SO2",
            pollutant_name="Sulfur Dioxide",
            current_value=85.2,
            unit="lb/hr",
            permit_limit=100.0,
            averaging_period="1-hour rolling",
            status=ComplianceStatus.WARNING,
            margin_percent=14.8,
            trend="increasing",
            last_updated="2024-01-15T14:30:00Z",
            data_quality=97.2
        ),
        "PM": PollutantStatus(
            pollutant_id="PM",
            pollutant_name="Particulate Matter",
            current_value=12.3,
            unit="lb/hr",
            permit_limit=25.0,
            averaging_period="6-hour rolling",
            status=ComplianceStatus.COMPLIANT,
            margin_percent=50.8,
            trend="decreasing",
            last_updated="2024-01-15T14:30:00Z",
            data_quality=95.0
        )
    }

    violations = [
        Violation(
            violation_id="VIO-2024-001",
            violation_type=ViolationType.EMISSION_EXCEEDANCE,
            pollutant="SO2",
            start_time="2024-01-10T08:15:00Z",
            end_time="2024-01-10T09:45:00Z",
            duration_minutes=90,
            exceedance_value=115.3,
            permit_limit=100.0,
            exceedance_percent=15.3,
            severity="moderate",
            status="under_review",
            regulatory_action=None,
            root_cause="Fuel quality variation",
            corrective_action="Fuel supplier notified"
        )
    ]

    return ComplianceDashboardData(
        timestamp="2024-01-15T14:30:00Z",
        facility_id="FAC-001",
        facility_name="GreenPower Plant Alpha",
        jurisdiction="California - SCAQMD",
        permit_number="SCAQMD-12345",
        pollutants=pollutants,
        overall_status=ComplianceStatus.WARNING,
        active_violations=violations,
        margin_to_limits={"NOx": 27.25, "SO2": 14.8, "PM": 50.8},
        reporting_period="Q1 2024",
        data_completeness=97.5,
        next_report_due="2024-04-15",
        notes=["Fuel switch planned for Q2", "Stack test scheduled March 2024"]
    )


if __name__ == "__main__":
    # Demo usage
    sample_data = create_sample_dashboard_data()
    dashboard = ComplianceDashboard(sample_data, color_blind_safe=False)

    # Generate and print JSON
    print("Dashboard JSON (first 500 chars):")
    print(dashboard.to_plotly_json()[:500])

    # Save HTML
    html_output = dashboard.to_html()
    print(f"\nGenerated HTML dashboard ({len(html_output)} characters)")
