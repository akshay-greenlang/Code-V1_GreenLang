"""
GL-010 EMISSIONWATCH - Regulatory Compliance Heatmap Visualization

Multi-jurisdiction compliance heatmap module for the EmissionsComplianceAgent.
Provides grid views, drill-down capability, and time-based animation.

Author: GreenLang Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json


class ComplianceLevel(Enum):
    """Compliance level enumeration."""
    EXCELLENT = "excellent"       # >30% margin
    GOOD = "good"                 # 20-30% margin
    ADEQUATE = "adequate"         # 10-20% margin
    MARGINAL = "marginal"         # 5-10% margin
    WARNING = "warning"           # 0-5% margin
    VIOLATION = "violation"       # <0% margin (exceedance)
    UNKNOWN = "unknown"

    @property
    def color(self) -> str:
        """Return color for compliance level."""
        colors = {
            "excellent": "#27AE60",
            "good": "#2ECC71",
            "adequate": "#F1C40F",
            "marginal": "#F39C12",
            "warning": "#E67E22",
            "violation": "#E74C3C",
            "unknown": "#95A5A6"
        }
        return colors.get(self.value, "#95A5A6")

    @property
    def color_blind_safe(self) -> str:
        """Return color-blind safe color."""
        colors = {
            "excellent": "#009E73",
            "good": "#56B4E9",
            "adequate": "#F0E442",
            "marginal": "#E69F00",
            "warning": "#D55E00",
            "violation": "#CC79A7",
            "unknown": "#999999"
        }
        return colors.get(self.value, "#999999")

    @property
    def numeric_value(self) -> int:
        """Return numeric value for sorting/comparison."""
        values = {
            "excellent": 0,
            "good": 1,
            "adequate": 2,
            "marginal": 3,
            "warning": 4,
            "violation": 5,
            "unknown": 6
        }
        return values.get(self.value, 6)

    @classmethod
    def from_margin(cls, margin_percent: float) -> 'ComplianceLevel':
        """Determine compliance level from margin percentage."""
        if margin_percent is None:
            return cls.UNKNOWN
        if margin_percent > 30:
            return cls.EXCELLENT
        if margin_percent > 20:
            return cls.GOOD
        if margin_percent > 10:
            return cls.ADEQUATE
        if margin_percent > 5:
            return cls.MARGINAL
        if margin_percent >= 0:
            return cls.WARNING
        return cls.VIOLATION


@dataclass
class JurisdictionInfo:
    """Jurisdiction information."""
    jurisdiction_id: str
    name: str
    abbreviation: str
    country: str
    region: Optional[str] = None
    regulatory_agency: Optional[str] = None
    contact_info: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jurisdiction_id": self.jurisdiction_id,
            "name": self.name,
            "abbreviation": self.abbreviation,
            "country": self.country,
            "region": self.region,
            "regulatory_agency": self.regulatory_agency,
            "contact_info": self.contact_info
        }


@dataclass
class PollutantLimit:
    """Pollutant limit information for a jurisdiction."""
    pollutant: str
    pollutant_name: str
    limit_value: float
    unit: str
    averaging_period: str
    limit_type: str  # "emission_rate", "concentration", "mass"
    effective_date: str
    source: str  # regulation reference

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pollutant": self.pollutant,
            "pollutant_name": self.pollutant_name,
            "limit_value": self.limit_value,
            "unit": self.unit,
            "averaging_period": self.averaging_period,
            "limit_type": self.limit_type,
            "effective_date": self.effective_date,
            "source": self.source
        }


@dataclass
class ComplianceCell:
    """Single cell in compliance heatmap."""
    jurisdiction_id: str
    pollutant: str
    current_value: float
    limit_value: float
    unit: str
    margin_percent: float
    compliance_level: ComplianceLevel
    timestamp: str
    data_quality: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jurisdiction_id": self.jurisdiction_id,
            "pollutant": self.pollutant,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "unit": self.unit,
            "margin_percent": self.margin_percent,
            "compliance_level": self.compliance_level.value,
            "timestamp": self.timestamp,
            "data_quality": self.data_quality,
            "notes": self.notes
        }


@dataclass
class HeatmapConfig:
    """Configuration for heatmap visualization."""
    title: str = "Multi-Jurisdiction Compliance Heatmap"
    color_blind_safe: bool = False
    show_values: bool = True
    show_margins: bool = True
    animate: bool = False
    animation_frame_duration: int = 1000  # milliseconds
    interactive: bool = True
    drill_down_enabled: bool = True
    export_enabled: bool = True


class RegulatoryHeatmap:
    """Generate multi-jurisdiction regulatory compliance heatmaps."""

    # Color scales
    COMPLIANCE_SCALE = [
        [0.0, "#27AE60"],   # Excellent
        [0.2, "#2ECC71"],   # Good
        [0.4, "#F1C40F"],   # Adequate
        [0.6, "#F39C12"],   # Marginal
        [0.8, "#E67E22"],   # Warning
        [1.0, "#E74C3C"]    # Violation
    ]

    COMPLIANCE_SCALE_BLIND_SAFE = [
        [0.0, "#009E73"],
        [0.2, "#56B4E9"],
        [0.4, "#F0E442"],
        [0.6, "#E69F00"],
        [0.8, "#D55E00"],
        [1.0, "#CC79A7"]
    ]

    def __init__(
        self,
        jurisdictions: List[JurisdictionInfo],
        pollutants: List[str],
        config: Optional[HeatmapConfig] = None
    ):
        """
        Initialize heatmap generator.

        Args:
            jurisdictions: List of jurisdictions
            pollutants: List of pollutant identifiers
            config: Heatmap configuration
        """
        self.jurisdictions = jurisdictions
        self.pollutants = pollutants
        self.config = config or HeatmapConfig()
        self._data: Dict[str, Dict[str, ComplianceCell]] = {}
        self._time_series: List[Dict[str, Dict[str, ComplianceCell]]] = []

    def set_compliance_data(
        self,
        data: List[ComplianceCell]
    ) -> None:
        """
        Set compliance data for heatmap.

        Args:
            data: List of compliance cells
        """
        self._data = {}
        for cell in data:
            if cell.jurisdiction_id not in self._data:
                self._data[cell.jurisdiction_id] = {}
            self._data[cell.jurisdiction_id][cell.pollutant] = cell

    def add_time_frame(
        self,
        timestamp: str,
        data: List[ComplianceCell]
    ) -> None:
        """
        Add time frame for animation.

        Args:
            timestamp: Frame timestamp
            data: Compliance data for this frame
        """
        frame_data = {}
        for cell in data:
            if cell.jurisdiction_id not in frame_data:
                frame_data[cell.jurisdiction_id] = {}
            frame_data[cell.jurisdiction_id][cell.pollutant] = cell

        self._time_series.append({
            "timestamp": timestamp,
            "data": frame_data
        })

    def _get_color_scale(self) -> List[List]:
        """Get appropriate color scale."""
        return self.COMPLIANCE_SCALE_BLIND_SAFE if self.config.color_blind_safe \
            else self.COMPLIANCE_SCALE

    def _get_level_color(self, level: ComplianceLevel) -> str:
        """Get color for compliance level."""
        return level.color_blind_safe if self.config.color_blind_safe else level.color

    def build_heatmap(self) -> Dict[str, Any]:
        """
        Build main compliance heatmap.

        Returns:
            Plotly chart dictionary
        """
        if not self._data:
            return self._build_empty_chart("No compliance data available")

        # Build matrix data
        jurisdiction_names = [j.name for j in self.jurisdictions]
        jurisdiction_ids = [j.jurisdiction_id for j in self.jurisdictions]

        z_values = []
        text_matrix = []
        hover_text = []

        for jid, jname in zip(jurisdiction_ids, jurisdiction_names):
            row_values = []
            row_text = []
            row_hover = []

            for pollutant in self.pollutants:
                cell = self._data.get(jid, {}).get(pollutant)

                if cell:
                    # Convert compliance level to numeric (0=best, 5=worst)
                    row_values.append(cell.compliance_level.numeric_value)

                    # Display text
                    if self.config.show_margins:
                        row_text.append(f"{cell.margin_percent:.1f}%")
                    elif self.config.show_values:
                        row_text.append(f"{cell.current_value:.1f}")
                    else:
                        row_text.append("")

                    # Hover text
                    hover = (
                        f"<b>{jname}</b><br>"
                        f"<b>{pollutant}</b><br>"
                        f"Current: {cell.current_value:.2f} {cell.unit}<br>"
                        f"Limit: {cell.limit_value:.2f} {cell.unit}<br>"
                        f"Margin: {cell.margin_percent:.1f}%<br>"
                        f"Status: {cell.compliance_level.value.title()}<br>"
                        f"Data Quality: {cell.data_quality:.0f}%"
                    )
                    row_hover.append(hover)
                else:
                    row_values.append(6)  # Unknown
                    row_text.append("N/A")
                    row_hover.append(f"<b>{jname}</b><br><b>{pollutant}</b><br>No data available")

            z_values.append(row_values)
            text_matrix.append(row_text)
            hover_text.append(row_hover)

        trace = {
            "type": "heatmap",
            "z": z_values,
            "x": self.pollutants,
            "y": jurisdiction_names,
            "text": text_matrix,
            "texttemplate": "%{text}",
            "textfont": {"size": 10},
            "hovertext": hover_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "colorscale": self._get_color_scale(),
            "showscale": True,
            "colorbar": {
                "title": "Compliance Status",
                "tickvals": [0, 1, 2, 3, 4, 5],
                "ticktext": ["Excellent", "Good", "Adequate", "Marginal", "Warning", "Violation"],
                "len": 0.8
            },
            "zmin": 0,
            "zmax": 5
        }

        layout = {
            "title": {
                "text": self.config.title,
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {
                "title": "Pollutants",
                "tickangle": -45,
                "side": "bottom"
            },
            "yaxis": {
                "title": "Jurisdictions",
                "autorange": "reversed"
            },
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA",
            "margin": {"l": 150, "r": 100, "t": 80, "b": 100}
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True, "displayModeBar": True}
        }

    def build_animated_heatmap(self) -> Dict[str, Any]:
        """
        Build animated heatmap showing compliance over time.

        Returns:
            Plotly chart dictionary with animation frames
        """
        if not self._time_series:
            return self.build_heatmap()

        jurisdiction_names = [j.name for j in self.jurisdictions]
        jurisdiction_ids = [j.jurisdiction_id for j in self.jurisdictions]

        frames = []
        timestamps = []

        for frame_data in self._time_series:
            timestamp = frame_data["timestamp"]
            data = frame_data["data"]
            timestamps.append(timestamp)

            z_values = []
            text_matrix = []

            for jid in jurisdiction_ids:
                row_values = []
                row_text = []

                for pollutant in self.pollutants:
                    cell = data.get(jid, {}).get(pollutant)
                    if cell:
                        row_values.append(cell.compliance_level.numeric_value)
                        row_text.append(f"{cell.margin_percent:.1f}%")
                    else:
                        row_values.append(6)
                        row_text.append("N/A")

                z_values.append(row_values)
                text_matrix.append(row_text)

            frames.append({
                "name": timestamp,
                "data": [{
                    "z": z_values,
                    "text": text_matrix
                }]
            })

        # Initial frame
        initial_trace = {
            "type": "heatmap",
            "z": frames[0]["data"][0]["z"] if frames else [],
            "x": self.pollutants,
            "y": jurisdiction_names,
            "text": frames[0]["data"][0]["text"] if frames else [],
            "texttemplate": "%{text}",
            "colorscale": self._get_color_scale(),
            "showscale": True,
            "colorbar": {
                "title": "Status",
                "tickvals": [0, 1, 2, 3, 4, 5],
                "ticktext": ["Excellent", "Good", "Adequate", "Marginal", "Warning", "Violation"]
            },
            "zmin": 0,
            "zmax": 5
        }

        layout = {
            "title": {
                "text": self.config.title,
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Pollutants", "tickangle": -45},
            "yaxis": {"title": "Jurisdictions", "autorange": "reversed"},
            "updatemenus": [{
                "type": "buttons",
                "showactive": False,
                "y": 1.15,
                "x": 0.1,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": self.config.animation_frame_duration},
                            "fromcurrent": True,
                            "transition": {"duration": 300}
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    }
                ]
            }],
            "sliders": [{
                "active": 0,
                "steps": [
                    {"label": ts, "method": "animate", "args": [[ts]]}
                    for ts in timestamps
                ],
                "x": 0.1,
                "len": 0.8,
                "y": -0.1,
                "currentvalue": {
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "center"
                }
            }],
            "paper_bgcolor": "white",
            "margin": {"l": 150, "r": 100, "t": 100, "b": 150}
        }

        return {
            "data": [initial_trace],
            "layout": layout,
            "frames": frames,
            "config": {"responsive": True}
        }

    def build_jurisdiction_detail(
        self,
        jurisdiction_id: str
    ) -> Dict[str, Any]:
        """
        Build detailed view for single jurisdiction.

        Args:
            jurisdiction_id: Jurisdiction to display

        Returns:
            Plotly chart dictionary
        """
        jurisdiction = next(
            (j for j in self.jurisdictions if j.jurisdiction_id == jurisdiction_id),
            None
        )

        if not jurisdiction or jurisdiction_id not in self._data:
            return self._build_empty_chart("Jurisdiction not found")

        cells = self._data[jurisdiction_id]
        pollutants = list(cells.keys())
        margins = [cells[p].margin_percent for p in pollutants]
        levels = [cells[p].compliance_level for p in pollutants]
        colors = [self._get_level_color(l) for l in levels]

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
            "hovertemplate": (
                "<b>%{x}</b><br>"
                "Margin: %{y:.1f}%<extra></extra>"
            )
        }

        # Add threshold lines
        shapes = [
            {"type": "line", "y0": 0, "y1": 0, "x0": -0.5, "x1": len(pollutants) - 0.5,
             "line": {"color": "#E74C3C", "width": 2, "dash": "dash"}},
            {"type": "line", "y0": 10, "y1": 10, "x0": -0.5, "x1": len(pollutants) - 0.5,
             "line": {"color": "#F39C12", "width": 1, "dash": "dot"}},
            {"type": "line", "y0": 30, "y1": 30, "x0": -0.5, "x1": len(pollutants) - 0.5,
             "line": {"color": "#27AE60", "width": 1, "dash": "dot"}}
        ]

        layout = {
            "title": {
                "text": f"Compliance Detail - {jurisdiction.name}",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Pollutant", "tickangle": -45},
            "yaxis": {
                "title": "Margin to Limit (%)",
                "range": [min(-10, min(margins) - 10), max(50, max(margins) + 10)]
            },
            "shapes": shapes,
            "annotations": [
                {"text": "Violation Zone", "x": len(pollutants) - 0.3, "y": -5,
                 "font": {"color": "#E74C3C", "size": 10}, "showarrow": False},
                {"text": "Warning Zone", "x": len(pollutants) - 0.3, "y": 5,
                 "font": {"color": "#F39C12", "size": 10}, "showarrow": False},
                {"text": "Safe Zone", "x": len(pollutants) - 0.3, "y": 35,
                 "font": {"color": "#27AE60", "size": 10}, "showarrow": False}
            ],
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_pollutant_comparison(
        self,
        pollutant: str
    ) -> Dict[str, Any]:
        """
        Build comparison of single pollutant across jurisdictions.

        Args:
            pollutant: Pollutant to compare

        Returns:
            Plotly chart dictionary
        """
        jurisdiction_data = []

        for j in self.jurisdictions:
            cell = self._data.get(j.jurisdiction_id, {}).get(pollutant)
            if cell:
                jurisdiction_data.append({
                    "name": j.name,
                    "margin": cell.margin_percent,
                    "value": cell.current_value,
                    "limit": cell.limit_value,
                    "level": cell.compliance_level
                })

        if not jurisdiction_data:
            return self._build_empty_chart(f"No data for {pollutant}")

        # Sort by margin
        jurisdiction_data.sort(key=lambda x: x["margin"], reverse=True)

        names = [d["name"] for d in jurisdiction_data]
        margins = [d["margin"] for d in jurisdiction_data]
        colors = [self._get_level_color(d["level"]) for d in jurisdiction_data]

        trace = {
            "type": "bar",
            "y": names,
            "x": margins,
            "orientation": "h",
            "marker": {
                "color": colors,
                "line": {"color": "#2C3E50", "width": 1}
            },
            "text": [f"{m:.1f}%" for m in margins],
            "textposition": "outside",
            "hovertemplate": (
                "<b>%{y}</b><br>"
                f"{pollutant} Margin: %{{x:.1f}}%<extra></extra>"
            )
        }

        layout = {
            "title": {
                "text": f"{pollutant} Compliance by Jurisdiction",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Margin to Limit (%)"},
            "yaxis": {"title": "", "autorange": "reversed"},
            "shapes": [
                {"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": len(names) - 0.5,
                 "line": {"color": "#E74C3C", "width": 2, "dash": "dash"}}
            ],
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA",
            "margin": {"l": 150}
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_summary_indicators(self) -> Dict[str, Any]:
        """
        Build summary indicator cards.

        Returns:
            Plotly chart dictionary
        """
        # Calculate summary stats
        total_cells = 0
        violation_count = 0
        warning_count = 0
        compliant_count = 0
        worst_margin = float('inf')
        best_margin = float('-inf')

        for jid, cells in self._data.items():
            for pollutant, cell in cells.items():
                total_cells += 1
                if cell.compliance_level == ComplianceLevel.VIOLATION:
                    violation_count += 1
                elif cell.compliance_level == ComplianceLevel.WARNING:
                    warning_count += 1
                else:
                    compliant_count += 1

                if cell.margin_percent < worst_margin:
                    worst_margin = cell.margin_percent
                if cell.margin_percent > best_margin:
                    best_margin = cell.margin_percent

        compliance_rate = (compliant_count / total_cells * 100) if total_cells > 0 else 0

        traces = [
            {
                "type": "indicator",
                "mode": "number+delta",
                "value": compliance_rate,
                "title": {"text": "Compliance Rate (%)"},
                "number": {"suffix": "%", "font": {"color": "#27AE60" if compliance_rate >= 90 else "#E74C3C"}},
                "delta": {"reference": 95, "suffix": "%"},
                "domain": {"row": 0, "column": 0}
            },
            {
                "type": "indicator",
                "mode": "number",
                "value": violation_count,
                "title": {"text": "Active Violations"},
                "number": {"font": {"color": "#E74C3C" if violation_count > 0 else "#27AE60"}},
                "domain": {"row": 0, "column": 1}
            },
            {
                "type": "indicator",
                "mode": "number",
                "value": warning_count,
                "title": {"text": "Warnings"},
                "number": {"font": {"color": "#F39C12"}},
                "domain": {"row": 0, "column": 2}
            },
            {
                "type": "indicator",
                "mode": "number",
                "value": worst_margin if worst_margin != float('inf') else 0,
                "title": {"text": "Worst Margin (%)"},
                "number": {"suffix": "%", "font": {"color": "#E74C3C" if worst_margin < 0 else "#2C3E50"}},
                "domain": {"row": 0, "column": 3}
            }
        ]

        layout = {
            "title": {"text": "Compliance Summary", "font": {"size": 16}},
            "grid": {"rows": 1, "columns": 4, "pattern": "independent"},
            "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
            "paper_bgcolor": "white"
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_geographic_map(
        self,
        coordinates: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Build geographic map showing compliance by location.

        Args:
            coordinates: Dict of jurisdiction_id -> (lat, lon)

        Returns:
            Plotly chart dictionary
        """
        # Calculate overall compliance level per jurisdiction
        jurisdiction_levels: Dict[str, ComplianceLevel] = {}

        for jid, cells in self._data.items():
            worst_level = ComplianceLevel.EXCELLENT
            for cell in cells.values():
                if cell.compliance_level.numeric_value > worst_level.numeric_value:
                    worst_level = cell.compliance_level
            jurisdiction_levels[jid] = worst_level

        lats = []
        lons = []
        texts = []
        colors = []
        sizes = []

        for j in self.jurisdictions:
            if j.jurisdiction_id in coordinates:
                lat, lon = coordinates[j.jurisdiction_id]
                level = jurisdiction_levels.get(j.jurisdiction_id, ComplianceLevel.UNKNOWN)

                lats.append(lat)
                lons.append(lon)
                texts.append(f"<b>{j.name}</b><br>Status: {level.value.title()}")
                colors.append(self._get_level_color(level))
                sizes.append(30 if level == ComplianceLevel.VIOLATION else 20)

        trace = {
            "type": "scattergeo",
            "lat": lats,
            "lon": lons,
            "mode": "markers",
            "marker": {
                "size": sizes,
                "color": colors,
                "line": {"color": "white", "width": 2}
            },
            "text": texts,
            "hovertemplate": "%{text}<extra></extra>"
        }

        layout = {
            "title": {
                "text": "Geographic Compliance Overview",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "geo": {
                "scope": "usa",  # Adjust as needed
                "projection": {"type": "albers usa"},
                "showland": True,
                "landcolor": "#FAFAFA",
                "showlakes": True,
                "lakecolor": "#E3F2FD",
                "showcountries": True,
                "countrycolor": "#BDC3C7"
            },
            "paper_bgcolor": "white"
        }

        return {
            "data": [trace],
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
            }
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics.

        Returns:
            Summary statistics dictionary
        """
        stats = {
            "total_jurisdictions": len(self.jurisdictions),
            "total_pollutants": len(self.pollutants),
            "total_cells": 0,
            "by_level": {},
            "by_jurisdiction": {},
            "by_pollutant": {},
            "worst_cases": [],
            "overall_compliance_rate": 0
        }

        all_cells = []

        for jid, cells in self._data.items():
            j_stats = {"total": 0, "violations": 0, "warnings": 0, "compliant": 0}

            for pollutant, cell in cells.items():
                stats["total_cells"] += 1
                all_cells.append(cell)

                # By level
                level = cell.compliance_level.value
                stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

                # By jurisdiction
                j_stats["total"] += 1
                if cell.compliance_level == ComplianceLevel.VIOLATION:
                    j_stats["violations"] += 1
                elif cell.compliance_level in [ComplianceLevel.WARNING, ComplianceLevel.MARGINAL]:
                    j_stats["warnings"] += 1
                else:
                    j_stats["compliant"] += 1

                # By pollutant
                if pollutant not in stats["by_pollutant"]:
                    stats["by_pollutant"][pollutant] = {"total": 0, "avg_margin": 0, "margins": []}
                stats["by_pollutant"][pollutant]["total"] += 1
                stats["by_pollutant"][pollutant]["margins"].append(cell.margin_percent)

            stats["by_jurisdiction"][jid] = j_stats

        # Calculate pollutant averages
        for pollutant in stats["by_pollutant"]:
            margins = stats["by_pollutant"][pollutant]["margins"]
            stats["by_pollutant"][pollutant]["avg_margin"] = sum(margins) / len(margins) if margins else 0
            del stats["by_pollutant"][pollutant]["margins"]

        # Worst cases
        sorted_cells = sorted(all_cells, key=lambda c: c.margin_percent)
        stats["worst_cases"] = [c.to_dict() for c in sorted_cells[:5]]

        # Overall compliance rate
        compliant = stats["by_level"].get("excellent", 0) + stats["by_level"].get("good", 0) + \
                   stats["by_level"].get("adequate", 0)
        stats["overall_compliance_rate"] = (compliant / stats["total_cells"] * 100) \
            if stats["total_cells"] > 0 else 0

        return stats

    def to_plotly_json(self) -> str:
        """Export heatmap to Plotly JSON."""
        return json.dumps(self.build_heatmap(), indent=2)

    def to_html(self) -> str:
        """
        Generate standalone HTML heatmap dashboard.

        Returns:
            HTML string
        """
        charts = {
            "heatmap": self.build_animated_heatmap() if self._time_series else self.build_heatmap(),
            "summary": self.build_summary_indicators()
        }

        # Add jurisdiction details
        for j in self.jurisdictions[:3]:  # First 3
            charts[f"detail_{j.jurisdiction_id}"] = self.build_jurisdiction_detail(j.jurisdiction_id)

        # Add pollutant comparisons
        for p in self.pollutants[:3]:  # First 3
            charts[f"pollutant_{p}"] = self.build_pollutant_comparison(p)

        stats = self.get_summary_statistics()

        charts_json = json.dumps(charts)
        stats_json = json.dumps(stats)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Regulatory Compliance Heatmap</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f5f6fa;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        h1, h2 {{
            color: #2c3e50;
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
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Jurisdiction Compliance Heatmap</h1>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #27AE60;"></div>
                <span>Excellent (>30%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2ECC71;"></div>
                <span>Good (20-30%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #F1C40F;"></div>
                <span>Adequate (10-20%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #F39C12;"></div>
                <span>Marginal (5-10%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #E67E22;"></div>
                <span>Warning (0-5%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #E74C3C;"></div>
                <span>Violation (<0%)</span>
            </div>
        </div>

        <div class="chart-container" id="summary-chart"></div>
        <div class="chart-container full-width" id="heatmap-chart"></div>

        <h2>Jurisdiction Details</h2>
        <div class="charts-grid" id="details-container"></div>

        <h2>Pollutant Analysis</h2>
        <div class="charts-grid" id="pollutants-container"></div>
    </div>

    <script>
        const charts = {charts_json};
        const stats = {stats_json};
        const config = {{responsive: true, displayModeBar: true, displaylogo: false}};

        // Render summary
        Plotly.newPlot('summary-chart', charts.summary.data, charts.summary.layout, config);

        // Render heatmap (with potential animation)
        if (charts.heatmap.frames) {{
            Plotly.newPlot('heatmap-chart', charts.heatmap.data, charts.heatmap.layout, config)
                .then(gd => Plotly.addFrames(gd, charts.heatmap.frames));
        }} else {{
            Plotly.newPlot('heatmap-chart', charts.heatmap.data, charts.heatmap.layout, config);
        }}

        // Render jurisdiction details
        const detailsContainer = document.getElementById('details-container');
        Object.entries(charts).forEach(([key, chart]) => {{
            if (key.startsWith('detail_')) {{
                const div = document.createElement('div');
                div.className = 'chart-container';
                div.id = key + '-chart';
                detailsContainer.appendChild(div);
                Plotly.newPlot(div.id, chart.data, chart.layout, config);
            }}
        }});

        // Render pollutant comparisons
        const pollutantsContainer = document.getElementById('pollutants-container');
        Object.entries(charts).forEach(([key, chart]) => {{
            if (key.startsWith('pollutant_')) {{
                const div = document.createElement('div');
                div.className = 'chart-container';
                div.id = key + '-chart';
                pollutantsContainer.appendChild(div);
                Plotly.newPlot(div.id, chart.data, chart.layout, config);
            }}
        }});

        // Make responsive
        window.addEventListener('resize', () => {{
            document.querySelectorAll('.chart-container').forEach(el => {{
                if (el.id) Plotly.Plots.resize(el.id);
            }});
        }});
    </script>
</body>
</html>"""

        return html

    def export_for_report(self) -> Dict[str, Any]:
        """
        Export data in format suitable for regulatory reports.

        Returns:
            Report-ready data structure
        """
        return {
            "summary": self.get_summary_statistics(),
            "jurisdictions": [j.to_dict() for j in self.jurisdictions],
            "pollutants": self.pollutants,
            "compliance_data": {
                jid: {p: c.to_dict() for p, c in cells.items()}
                for jid, cells in self._data.items()
            },
            "charts": {
                "heatmap": self.build_heatmap(),
                "summary": self.build_summary_indicators()
            },
            "generated_at": datetime.now().isoformat()
        }


def create_sample_heatmap_data(
    num_jurisdictions: int = 8,
    num_pollutants: int = 6
) -> Tuple[List[JurisdictionInfo], List[str], List[ComplianceCell]]:
    """
    Create sample heatmap data for testing.

    Args:
        num_jurisdictions: Number of jurisdictions
        num_pollutants: Number of pollutants

    Returns:
        Tuple of (jurisdictions, pollutants, cells)
    """
    import random
    random.seed(42)

    jurisdictions = [
        JurisdictionInfo("CA_SCAQMD", "South Coast AQMD", "SCAQMD", "USA", "California", "South Coast Air Quality Management District"),
        JurisdictionInfo("CA_BAAQMD", "Bay Area AQMD", "BAAQMD", "USA", "California", "Bay Area Air Quality Management District"),
        JurisdictionInfo("TX_TCEQ", "Texas Commission", "TCEQ", "USA", "Texas", "Texas Commission on Environmental Quality"),
        JurisdictionInfo("PA_DEP", "Pennsylvania DEP", "PA DEP", "USA", "Pennsylvania", "Department of Environmental Protection"),
        JurisdictionInfo("OH_EPA", "Ohio EPA", "OH EPA", "USA", "Ohio", "Ohio Environmental Protection Agency"),
        JurisdictionInfo("IL_EPA", "Illinois EPA", "IL EPA", "USA", "Illinois", "Illinois Environmental Protection Agency"),
        JurisdictionInfo("NY_DEC", "New York DEC", "NY DEC", "USA", "New York", "Department of Environmental Conservation"),
        JurisdictionInfo("FL_DEP", "Florida DEP", "FL DEP", "USA", "Florida", "Department of Environmental Protection")
    ][:num_jurisdictions]

    pollutants = ["NOx", "SO2", "PM2.5", "PM10", "CO", "VOC", "Hg", "HCl"][:num_pollutants]

    cells = []
    timestamp = datetime.now().isoformat()

    for j in jurisdictions:
        for p in pollutants:
            # Generate random but realistic values
            limit = random.uniform(50, 200)
            margin = random.gauss(20, 15)  # Mean 20%, std 15%
            current = limit * (1 - margin / 100)
            level = ComplianceLevel.from_margin(margin)

            cells.append(ComplianceCell(
                jurisdiction_id=j.jurisdiction_id,
                pollutant=p,
                current_value=max(0, current),
                limit_value=limit,
                unit="lb/hr",
                margin_percent=margin,
                compliance_level=level,
                timestamp=timestamp,
                data_quality=random.uniform(90, 100)
            ))

    return jurisdictions, pollutants, cells


if __name__ == "__main__":
    # Demo usage
    jurisdictions, pollutants, cells = create_sample_heatmap_data()

    config = HeatmapConfig(
        title="Multi-State Compliance Heatmap",
        color_blind_safe=False
    )

    heatmap = RegulatoryHeatmap(jurisdictions, pollutants, config)
    heatmap.set_compliance_data(cells)

    print("Heatmap JSON (first 500 chars):")
    print(heatmap.to_plotly_json()[:500])

    stats = heatmap.get_summary_statistics()
    print(f"\nSummary Statistics:")
    print(f"  Total jurisdictions: {stats['total_jurisdictions']}")
    print(f"  Total pollutants: {stats['total_pollutants']}")
    print(f"  Total cells: {stats['total_cells']}")
    print(f"  Overall compliance rate: {stats['overall_compliance_rate']:.1f}%")
