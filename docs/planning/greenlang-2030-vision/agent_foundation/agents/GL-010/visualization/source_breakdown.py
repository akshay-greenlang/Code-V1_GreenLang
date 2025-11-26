"""
GL-010 EMISSIONWATCH - Emissions Source Breakdown Visualization

Emissions source analysis module for the EmissionsComplianceAgent.
Provides pie/donut charts, stack contributions, fuel analysis, and Sankey diagrams.

Author: GreenLang Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import json


class SourceType(Enum):
    """Types of emission sources."""
    COMBUSTION = "combustion"
    PROCESS = "process"
    FUGITIVE = "fugitive"
    STORAGE = "storage"
    FLARE = "flare"
    COOLING_TOWER = "cooling_tower"
    WASTEWATER = "wastewater"
    OTHER = "other"

    @property
    def display_name(self) -> str:
        """Return human-readable name."""
        return self.value.replace("_", " ").title()

    @property
    def color(self) -> str:
        """Return default color for source type."""
        colors = {
            "combustion": "#E74C3C",
            "process": "#3498DB",
            "fugitive": "#2ECC71",
            "storage": "#F39C12",
            "flare": "#9B59B6",
            "cooling_tower": "#1ABC9C",
            "wastewater": "#34495E",
            "other": "#95A5A6"
        }
        return colors.get(self.value, "#95A5A6")


class FuelType(Enum):
    """Types of fuels."""
    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    FUEL_OIL = "fuel_oil"
    DIESEL = "diesel"
    BIOMASS = "biomass"
    WASTE = "waste"
    HYDROGEN = "hydrogen"
    MIXED = "mixed"

    @property
    def display_name(self) -> str:
        """Return human-readable name."""
        return self.value.replace("_", " ").title()

    @property
    def color(self) -> str:
        """Return default color for fuel type."""
        colors = {
            "natural_gas": "#3498DB",
            "coal": "#2C3E50",
            "fuel_oil": "#E74C3C",
            "diesel": "#F39C12",
            "biomass": "#27AE60",
            "waste": "#9B59B6",
            "hydrogen": "#1ABC9C",
            "mixed": "#95A5A6"
        }
        return colors.get(self.value, "#95A5A6")


@dataclass
class EmissionSource:
    """Individual emission source data."""
    source_id: str
    source_name: str
    source_type: SourceType
    unit_id: str
    unit_name: str
    fuel_type: Optional[FuelType] = None
    emissions_by_pollutant: Dict[str, float] = field(default_factory=dict)
    total_emissions: float = 0.0
    unit: str = "tons/year"
    operating_hours: float = 0.0
    capacity_factor: float = 0.0
    permit_limit: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    stack_height: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "unit_id": self.unit_id,
            "unit_name": self.unit_name,
            "fuel_type": self.fuel_type.value if self.fuel_type else None,
            "emissions_by_pollutant": self.emissions_by_pollutant,
            "total_emissions": self.total_emissions,
            "unit": self.unit,
            "operating_hours": self.operating_hours,
            "capacity_factor": self.capacity_factor,
            "permit_limit": self.permit_limit,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "stack_height": self.stack_height,
            "notes": self.notes
        }


@dataclass
class ProcessUnit:
    """Process unit containing multiple emission sources."""
    unit_id: str
    unit_name: str
    description: str
    sources: List[EmissionSource] = field(default_factory=list)
    total_emissions: float = 0.0
    emissions_by_pollutant: Dict[str, float] = field(default_factory=dict)

    def calculate_totals(self) -> None:
        """Calculate total emissions from sources."""
        self.total_emissions = sum(s.total_emissions for s in self.sources)
        self.emissions_by_pollutant = {}
        for source in self.sources:
            for pollutant, value in source.emissions_by_pollutant.items():
                self.emissions_by_pollutant[pollutant] = \
                    self.emissions_by_pollutant.get(pollutant, 0) + value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "unit_id": self.unit_id,
            "unit_name": self.unit_name,
            "description": self.description,
            "sources": [s.to_dict() for s in self.sources],
            "total_emissions": self.total_emissions,
            "emissions_by_pollutant": self.emissions_by_pollutant
        }


@dataclass
class SourceBreakdownConfig:
    """Configuration for source breakdown visualization."""
    title: str = "Emissions Source Breakdown"
    pollutant_filter: Optional[str] = None
    color_blind_safe: bool = False
    show_percentages: bool = True
    min_percent_label: float = 5.0  # Minimum % to show label
    sort_by_value: bool = True
    interactive: bool = True


class SourceBreakdownChart:
    """Generate emissions source breakdown visualizations."""

    # Color palettes
    COLORS_STANDARD = [
        "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
        "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#27AE60"
    ]

    COLORS_BLIND_SAFE = [
        "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7",
        "#56B4E9", "#F0E442", "#999999", "#000000", "#666666"
    ]

    def __init__(
        self,
        sources: List[EmissionSource],
        config: Optional[SourceBreakdownConfig] = None
    ):
        """
        Initialize source breakdown chart generator.

        Args:
            sources: List of emission sources
            config: Visualization configuration
        """
        self.sources = sources
        self.config = config or SourceBreakdownConfig()
        self._colors = self.COLORS_BLIND_SAFE if config and config.color_blind_safe else self.COLORS_STANDARD

    def _get_color(self, index: int) -> str:
        """Get color for index."""
        return self._colors[index % len(self._colors)]

    def build_pie_chart(
        self,
        group_by: str = "source",
        pollutant: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build pie/donut chart for emissions breakdown.

        Args:
            group_by: Grouping method - "source", "type", "fuel", "unit"
            pollutant: Optional pollutant filter

        Returns:
            Plotly chart dictionary
        """
        data = self._aggregate_data(group_by, pollutant)

        if not data:
            return self._build_empty_chart("No emission data available")

        labels = list(data.keys())
        values = list(data.values())
        total = sum(values)

        # Calculate percentages
        percentages = [(v / total * 100) if total > 0 else 0 for v in values]

        # Format text labels
        text_labels = []
        for label, value, pct in zip(labels, values, percentages):
            if pct >= self.config.min_percent_label:
                text_labels.append(f"{label}<br>{value:,.1f}<br>({pct:.1f}%)")
            else:
                text_labels.append("")

        trace = {
            "type": "pie",
            "labels": labels,
            "values": values,
            "hole": 0.4,  # Donut chart
            "textinfo": "label+percent" if self.config.show_percentages else "label",
            "textposition": "auto",
            "marker": {
                "colors": [self._get_color(i) for i in range(len(labels))],
                "line": {"color": "white", "width": 2}
            },
            "hovertemplate": (
                "<b>%{label}</b><br>"
                "Emissions: %{value:,.2f}<br>"
                "Percentage: %{percent}<extra></extra>"
            ),
            "sort": self.config.sort_by_value
        }

        title_suffix = f" - {pollutant}" if pollutant else ""
        layout = {
            "title": {
                "text": f"{self.config.title}{title_suffix}",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "showlegend": True,
            "legend": {
                "orientation": "v",
                "x": 1.05,
                "y": 0.5
            },
            "annotations": [{
                "text": f"<b>{total:,.0f}</b><br>Total",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 16},
                "showarrow": False
            }],
            "paper_bgcolor": "white"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True, "displayModeBar": True}
        }

    def build_bar_chart(
        self,
        group_by: str = "source",
        pollutant: Optional[str] = None,
        horizontal: bool = False
    ) -> Dict[str, Any]:
        """
        Build bar chart for emissions comparison.

        Args:
            group_by: Grouping method
            pollutant: Optional pollutant filter
            horizontal: Use horizontal bars

        Returns:
            Plotly chart dictionary
        """
        data = self._aggregate_data(group_by, pollutant)

        if not data:
            return self._build_empty_chart("No emission data available")

        # Sort by value if configured
        if self.config.sort_by_value:
            data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

        labels = list(data.keys())
        values = list(data.values())

        trace = {
            "type": "bar",
            "x": values if horizontal else labels,
            "y": labels if horizontal else values,
            "orientation": "h" if horizontal else "v",
            "marker": {
                "color": [self._get_color(i) for i in range(len(labels))],
                "line": {"color": "#2C3E50", "width": 1}
            },
            "text": [f"{v:,.1f}" for v in values],
            "textposition": "outside" if not horizontal else "auto",
            "hovertemplate": "<b>%{" + ("y" if horizontal else "x") + "}</b><br>Emissions: %{" + ("x" if horizontal else "y") + ":,.2f}<extra></extra>"
        }

        layout = {
            "title": {
                "text": self.config.title,
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {
                "title": "Emissions" if horizontal else "Source",
                "tickangle": 0 if horizontal else -45
            },
            "yaxis": {
                "title": "Source" if horizontal else "Emissions",
                "autorange": "reversed" if horizontal else True
            },
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA",
            "margin": {"b": 100} if not horizontal else {"l": 150}
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_stacked_bar_chart(
        self,
        group_by: str = "unit",
        stack_by: str = "pollutant"
    ) -> Dict[str, Any]:
        """
        Build stacked bar chart showing contribution breakdown.

        Args:
            group_by: Primary grouping (x-axis)
            stack_by: Secondary grouping (stacked)

        Returns:
            Plotly chart dictionary
        """
        # Aggregate data with two-level grouping
        data: Dict[str, Dict[str, float]] = {}

        for source in self.sources:
            # Determine group key
            if group_by == "unit":
                group_key = source.unit_name
            elif group_by == "type":
                group_key = source.source_type.display_name
            elif group_by == "fuel":
                group_key = source.fuel_type.display_name if source.fuel_type else "Unknown"
            else:
                group_key = source.source_name

            if group_key not in data:
                data[group_key] = {}

            # Add stack values
            if stack_by == "pollutant":
                for pollutant, value in source.emissions_by_pollutant.items():
                    data[group_key][pollutant] = data[group_key].get(pollutant, 0) + value
            elif stack_by == "type":
                stack_key = source.source_type.display_name
                data[group_key][stack_key] = data[group_key].get(stack_key, 0) + source.total_emissions
            elif stack_by == "fuel":
                stack_key = source.fuel_type.display_name if source.fuel_type else "Unknown"
                data[group_key][stack_key] = data[group_key].get(stack_key, 0) + source.total_emissions

        if not data:
            return self._build_empty_chart("No emission data available")

        # Get all unique stack keys
        all_stack_keys = set()
        for group_data in data.values():
            all_stack_keys.update(group_data.keys())

        groups = list(data.keys())
        traces = []

        for idx, stack_key in enumerate(sorted(all_stack_keys)):
            values = [data[g].get(stack_key, 0) for g in groups]
            traces.append({
                "type": "bar",
                "name": stack_key,
                "x": groups,
                "y": values,
                "marker": {"color": self._get_color(idx)},
                "hovertemplate": f"<b>{stack_key}</b><br>%{{x}}: %{{y:,.2f}}<extra></extra>"
            })

        layout = {
            "title": {
                "text": f"Emissions by {group_by.title()} and {stack_by.title()}",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {"title": group_by.title(), "tickangle": -45},
            "yaxis": {"title": "Emissions", "rangemode": "tozero"},
            "barmode": "stack",
            "legend": {"orientation": "h", "y": 1.1},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_treemap(self, hierarchy: List[str] = None) -> Dict[str, Any]:
        """
        Build treemap visualization for hierarchical breakdown.

        Args:
            hierarchy: List of grouping levels (e.g., ["unit", "type", "source"])

        Returns:
            Plotly chart dictionary
        """
        hierarchy = hierarchy or ["unit", "source"]

        ids = []
        labels = []
        parents = []
        values = []
        colors = []

        # Root level
        total = sum(s.total_emissions for s in self.sources)
        ids.append("Total")
        labels.append("Total Emissions")
        parents.append("")
        values.append(total)
        colors.append("#2C3E50")

        # Build hierarchy
        if "unit" in hierarchy:
            units: Dict[str, float] = {}
            for source in self.sources:
                units[source.unit_name] = units.get(source.unit_name, 0) + source.total_emissions

            for idx, (unit, value) in enumerate(units.items()):
                ids.append(unit)
                labels.append(unit)
                parents.append("Total")
                values.append(value)
                colors.append(self._get_color(idx))

                if "type" in hierarchy:
                    # Add source types under each unit
                    type_data: Dict[str, float] = {}
                    for source in self.sources:
                        if source.unit_name == unit:
                            st = source.source_type.display_name
                            type_data[st] = type_data.get(st, 0) + source.total_emissions

                    for st, st_value in type_data.items():
                        type_id = f"{unit}/{st}"
                        ids.append(type_id)
                        labels.append(st)
                        parents.append(unit)
                        values.append(st_value)
                        colors.append(source.source_type.color if hasattr(source.source_type, 'color') else self._get_color(len(colors)))

                        if "source" in hierarchy:
                            # Add individual sources
                            for source in self.sources:
                                if source.unit_name == unit and source.source_type.display_name == st:
                                    source_id = f"{type_id}/{source.source_id}"
                                    ids.append(source_id)
                                    labels.append(source.source_name)
                                    parents.append(type_id)
                                    values.append(source.total_emissions)
                                    colors.append(self._get_color(len(colors)))

        trace = {
            "type": "treemap",
            "ids": ids,
            "labels": labels,
            "parents": parents,
            "values": values,
            "marker": {"colors": colors},
            "textinfo": "label+value+percent entry",
            "hovertemplate": "<b>%{label}</b><br>Emissions: %{value:,.2f}<br>Percentage: %{percentEntry:.1%}<extra></extra>",
            "branchvalues": "total"
        }

        layout = {
            "title": {
                "text": "Emissions Source Hierarchy",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "paper_bgcolor": "white"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_sankey_diagram(self) -> Dict[str, Any]:
        """
        Build Sankey diagram showing emissions flow.
        Sources -> Pollutants -> (Atmosphere)

        Returns:
            Plotly chart dictionary
        """
        # Build node list
        nodes = []
        node_colors = []
        node_indices: Dict[str, int] = {}

        # Source type nodes
        source_types = set(s.source_type for s in self.sources)
        for st in source_types:
            idx = len(nodes)
            node_indices[f"type_{st.value}"] = idx
            nodes.append(st.display_name)
            node_colors.append(st.color)

        # Pollutant nodes
        pollutants = set()
        for source in self.sources:
            pollutants.update(source.emissions_by_pollutant.keys())

        pollutant_colors = {
            "NOx": "#E74C3C",
            "SO2": "#9B59B6",
            "PM": "#3498DB",
            "CO": "#2ECC71",
            "VOC": "#F39C12",
            "CO2": "#607D8B",
            "Hg": "#1ABC9C",
            "HCl": "#E91E63"
        }

        for pollutant in sorted(pollutants):
            idx = len(nodes)
            node_indices[f"pollutant_{pollutant}"] = idx
            nodes.append(pollutant)
            node_colors.append(pollutant_colors.get(pollutant, "#95A5A6"))

        # Atmosphere node
        atmosphere_idx = len(nodes)
        nodes.append("Atmosphere")
        node_colors.append("#87CEEB")

        # Build links
        sources_list = []
        targets = []
        values = []
        link_colors = []

        # Source type -> Pollutant links
        for source in self.sources:
            type_idx = node_indices[f"type_{source.source_type.value}"]
            for pollutant, value in source.emissions_by_pollutant.items():
                if value > 0:
                    pollutant_idx = node_indices[f"pollutant_{pollutant}"]
                    sources_list.append(type_idx)
                    targets.append(pollutant_idx)
                    values.append(value)
                    link_colors.append(f"rgba{self._hex_to_rgba(source.source_type.color, 0.5)}")

        # Pollutant -> Atmosphere links
        pollutant_totals: Dict[str, float] = {}
        for source in self.sources:
            for pollutant, value in source.emissions_by_pollutant.items():
                pollutant_totals[pollutant] = pollutant_totals.get(pollutant, 0) + value

        for pollutant, total in pollutant_totals.items():
            if total > 0:
                sources_list.append(node_indices[f"pollutant_{pollutant}"])
                targets.append(atmosphere_idx)
                values.append(total)
                link_colors.append(f"rgba{self._hex_to_rgba(pollutant_colors.get(pollutant, '#95A5A6'), 0.5)}")

        trace = {
            "type": "sankey",
            "node": {
                "pad": 15,
                "thickness": 20,
                "line": {"color": "black", "width": 0.5},
                "label": nodes,
                "color": node_colors
            },
            "link": {
                "source": sources_list,
                "target": targets,
                "value": values,
                "color": link_colors
            }
        }

        layout = {
            "title": {
                "text": "Emissions Flow Diagram",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "font": {"size": 12},
            "paper_bgcolor": "white"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_fuel_breakdown(self) -> Dict[str, Any]:
        """
        Build fuel type breakdown chart.

        Returns:
            Plotly chart dictionary
        """
        fuel_data: Dict[str, Dict[str, float]] = {}

        for source in self.sources:
            if source.fuel_type:
                fuel_name = source.fuel_type.display_name
                if fuel_name not in fuel_data:
                    fuel_data[fuel_name] = {"total": 0, "pollutants": {}}

                fuel_data[fuel_name]["total"] += source.total_emissions
                for pollutant, value in source.emissions_by_pollutant.items():
                    fuel_data[fuel_name]["pollutants"][pollutant] = \
                        fuel_data[fuel_name]["pollutants"].get(pollutant, 0) + value

        if not fuel_data:
            return self._build_empty_chart("No fuel data available")

        fuels = list(fuel_data.keys())
        totals = [fuel_data[f]["total"] for f in fuels]

        # Main pie chart
        trace = {
            "type": "pie",
            "labels": fuels,
            "values": totals,
            "hole": 0.4,
            "marker": {
                "colors": [
                    FuelType(f.lower().replace(" ", "_")).color
                    if f.lower().replace(" ", "_") in [ft.value for ft in FuelType]
                    else self._get_color(i)
                    for i, f in enumerate(fuels)
                ]
            },
            "textinfo": "label+percent",
            "hovertemplate": "<b>%{label}</b><br>Emissions: %{value:,.2f}<br>Percentage: %{percent}<extra></extra>"
        }

        layout = {
            "title": {
                "text": "Emissions by Fuel Type",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "showlegend": True,
            "annotations": [{
                "text": f"<b>{sum(totals):,.0f}</b><br>Total",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 16},
                "showarrow": False
            }],
            "paper_bgcolor": "white"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_process_unit_comparison(self) -> Dict[str, Any]:
        """
        Build process unit comparison chart.

        Returns:
            Plotly chart dictionary
        """
        unit_data: Dict[str, Dict[str, Any]] = {}

        for source in self.sources:
            if source.unit_name not in unit_data:
                unit_data[source.unit_name] = {
                    "total": 0,
                    "sources": 0,
                    "pollutants": {}
                }

            unit_data[source.unit_name]["total"] += source.total_emissions
            unit_data[source.unit_name]["sources"] += 1
            for pollutant, value in source.emissions_by_pollutant.items():
                unit_data[source.unit_name]["pollutants"][pollutant] = \
                    unit_data[source.unit_name]["pollutants"].get(pollutant, 0) + value

        if not unit_data:
            return self._build_empty_chart("No unit data available")

        # Sort by total emissions
        sorted_units = sorted(unit_data.items(), key=lambda x: x[1]["total"], reverse=True)
        units = [u[0] for u in sorted_units]
        totals = [u[1]["total"] for u in sorted_units]
        source_counts = [u[1]["sources"] for u in sorted_units]

        traces = [
            {
                "type": "bar",
                "name": "Total Emissions",
                "x": units,
                "y": totals,
                "marker": {"color": "#3498DB"},
                "yaxis": "y"
            },
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Source Count",
                "x": units,
                "y": source_counts,
                "marker": {"color": "#E74C3C", "size": 10},
                "line": {"width": 2},
                "yaxis": "y2"
            }
        ]

        layout = {
            "title": {
                "text": "Process Unit Comparison",
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "xaxis": {"title": "Process Unit", "tickangle": -45},
            "yaxis": {"title": "Total Emissions", "side": "left"},
            "yaxis2": {
                "title": "Number of Sources",
                "side": "right",
                "overlaying": "y"
            },
            "legend": {"orientation": "h", "y": 1.1},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFAFA"
        }

        return {
            "data": traces,
            "layout": layout,
            "config": {"responsive": True}
        }

    def build_pollutant_profile(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Build radar/spider chart for pollutant profile.

        Args:
            source_id: Optional specific source, or aggregate all

        Returns:
            Plotly chart dictionary
        """
        if source_id:
            sources = [s for s in self.sources if s.source_id == source_id]
            title = f"Pollutant Profile - {source_id}"
        else:
            sources = self.sources
            title = "Facility Pollutant Profile"

        pollutant_totals: Dict[str, float] = {}
        for source in sources:
            for pollutant, value in source.emissions_by_pollutant.items():
                pollutant_totals[pollutant] = pollutant_totals.get(pollutant, 0) + value

        if not pollutant_totals:
            return self._build_empty_chart("No pollutant data available")

        pollutants = list(pollutant_totals.keys())
        values = list(pollutant_totals.values())

        # Normalize to 0-100 scale
        max_val = max(values) if values else 1
        normalized = [v / max_val * 100 for v in values]

        trace = {
            "type": "scatterpolar",
            "r": normalized + [normalized[0]],  # Close the polygon
            "theta": pollutants + [pollutants[0]],
            "fill": "toself",
            "fillcolor": "rgba(52, 152, 219, 0.3)",
            "line": {"color": "#3498DB", "width": 2},
            "marker": {"size": 8},
            "hovertemplate": "<b>%{theta}</b><br>Value: %{r:.1f}<br>(Normalized)<extra></extra>"
        }

        layout = {
            "title": {
                "text": title,
                "font": {"size": 18, "color": "#2C3E50"}
            },
            "polar": {
                "radialaxis": {
                    "visible": True,
                    "range": [0, 100]
                }
            },
            "showlegend": False,
            "paper_bgcolor": "white"
        }

        return {
            "data": [trace],
            "layout": layout,
            "config": {"responsive": True}
        }

    def _aggregate_data(
        self,
        group_by: str,
        pollutant: Optional[str] = None
    ) -> Dict[str, float]:
        """Aggregate emission data by specified grouping."""
        data: Dict[str, float] = {}

        for source in self.sources:
            # Determine group key
            if group_by == "source":
                key = source.source_name
            elif group_by == "type":
                key = source.source_type.display_name
            elif group_by == "fuel":
                key = source.fuel_type.display_name if source.fuel_type else "Unknown"
            elif group_by == "unit":
                key = source.unit_name
            else:
                key = source.source_name

            # Get value
            if pollutant:
                value = source.emissions_by_pollutant.get(pollutant, 0)
            else:
                value = source.total_emissions

            data[key] = data.get(key, 0) + value

        return data

    def _hex_to_rgba(self, hex_color: str, alpha: float) -> Tuple[int, int, int, float]:
        """Convert hex color to RGBA tuple."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)

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
        Calculate summary statistics for sources.

        Returns:
            Summary statistics dictionary
        """
        if not self.sources:
            return {
                "total_sources": 0,
                "total_emissions": 0,
                "by_type": {},
                "by_fuel": {},
                "by_pollutant": {},
                "top_sources": []
            }

        by_type: Dict[str, Dict[str, Any]] = {}
        by_fuel: Dict[str, Dict[str, Any]] = {}
        by_pollutant: Dict[str, float] = {}

        for source in self.sources:
            # By type
            st = source.source_type.value
            if st not in by_type:
                by_type[st] = {"count": 0, "emissions": 0}
            by_type[st]["count"] += 1
            by_type[st]["emissions"] += source.total_emissions

            # By fuel
            if source.fuel_type:
                ft = source.fuel_type.value
                if ft not in by_fuel:
                    by_fuel[ft] = {"count": 0, "emissions": 0}
                by_fuel[ft]["count"] += 1
                by_fuel[ft]["emissions"] += source.total_emissions

            # By pollutant
            for pollutant, value in source.emissions_by_pollutant.items():
                by_pollutant[pollutant] = by_pollutant.get(pollutant, 0) + value

        # Top sources
        sorted_sources = sorted(self.sources, key=lambda s: s.total_emissions, reverse=True)
        top_sources = [
            {"source_id": s.source_id, "name": s.source_name, "emissions": s.total_emissions}
            for s in sorted_sources[:10]
        ]

        return {
            "total_sources": len(self.sources),
            "total_emissions": sum(s.total_emissions for s in self.sources),
            "by_type": by_type,
            "by_fuel": by_fuel,
            "by_pollutant": by_pollutant,
            "top_sources": top_sources
        }

    def to_plotly_json(self) -> str:
        """Export pie chart to Plotly JSON."""
        return json.dumps(self.build_pie_chart(), indent=2)

    def to_html(self) -> str:
        """
        Generate standalone HTML source breakdown dashboard.

        Returns:
            HTML string
        """
        charts = {
            "pie": self.build_pie_chart(),
            "pie_by_type": self.build_pie_chart(group_by="type"),
            "bar": self.build_bar_chart(horizontal=True),
            "stacked": self.build_stacked_bar_chart(),
            "treemap": self.build_treemap(),
            "sankey": self.build_sankey_diagram(),
            "fuel": self.build_fuel_breakdown(),
            "units": self.build_process_unit_comparison(),
            "profile": self.build_pollutant_profile()
        }

        stats = self.get_summary_statistics()

        charts_json = json.dumps(charts)
        stats_json = json.dumps(stats)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emissions Source Breakdown</title>
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
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Emissions Source Breakdown</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Sources</h3>
                <div class="value" id="stat-sources">-</div>
            </div>
            <div class="stat-card">
                <h3>Total Emissions</h3>
                <div class="value" id="stat-emissions">-</div>
            </div>
            <div class="stat-card">
                <h3>Source Types</h3>
                <div class="value" id="stat-types">-</div>
            </div>
            <div class="stat-card">
                <h3>Pollutants</h3>
                <div class="value" id="stat-pollutants">-</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container" id="pie-chart"></div>
            <div class="chart-container" id="type-chart"></div>
        </div>

        <div class="chart-container full-width" id="sankey-chart"></div>

        <div class="charts-grid">
            <div class="chart-container" id="bar-chart"></div>
            <div class="chart-container" id="stacked-chart"></div>
        </div>

        <div class="chart-container full-width" id="treemap-chart"></div>

        <h2>Additional Analysis</h2>
        <div class="charts-grid">
            <div class="chart-container" id="fuel-chart"></div>
            <div class="chart-container" id="units-chart"></div>
            <div class="chart-container" id="profile-chart"></div>
        </div>
    </div>

    <script>
        const charts = {charts_json};
        const stats = {stats_json};

        // Update stats
        document.getElementById('stat-sources').textContent = stats.total_sources;
        document.getElementById('stat-emissions').textContent = stats.total_emissions.toLocaleString(undefined, {{maximumFractionDigits: 0}});
        document.getElementById('stat-types').textContent = Object.keys(stats.by_type).length;
        document.getElementById('stat-pollutants').textContent = Object.keys(stats.by_pollutant).length;

        // Render charts
        const config = {{responsive: true, displayModeBar: true, displaylogo: false}};

        Plotly.newPlot('pie-chart', charts.pie.data, charts.pie.layout, config);
        Plotly.newPlot('type-chart', charts.pie_by_type.data, charts.pie_by_type.layout, config);
        Plotly.newPlot('sankey-chart', charts.sankey.data, charts.sankey.layout, config);
        Plotly.newPlot('bar-chart', charts.bar.data, charts.bar.layout, config);
        Plotly.newPlot('stacked-chart', charts.stacked.data, charts.stacked.layout, config);
        Plotly.newPlot('treemap-chart', charts.treemap.data, charts.treemap.layout, config);
        Plotly.newPlot('fuel-chart', charts.fuel.data, charts.fuel.layout, config);
        Plotly.newPlot('units-chart', charts.units.data, charts.units.layout, config);
        Plotly.newPlot('profile-chart', charts.profile.data, charts.profile.layout, config);

        // Make responsive
        window.addEventListener('resize', () => {{
            ['pie', 'type', 'sankey', 'bar', 'stacked', 'treemap', 'fuel', 'units', 'profile']
                .forEach(id => Plotly.Plots.resize(id + '-chart'));
        }});
    </script>
</body>
</html>"""

        return html


def create_sample_sources(count: int = 15) -> List[EmissionSource]:
    """
    Create sample emission sources for testing.

    Args:
        count: Number of sources to generate

    Returns:
        List of sample emission sources
    """
    import random
    random.seed(42)

    units = [
        ("UNIT-1", "Boiler Unit 1", "Main steam boiler"),
        ("UNIT-2", "Boiler Unit 2", "Auxiliary boiler"),
        ("UNIT-3", "Gas Turbine", "Combined cycle turbine"),
        ("UNIT-4", "Process Heater", "Refinery process heater"),
        ("UNIT-5", "Cooling System", "Cooling tower complex")
    ]

    source_types = list(SourceType)
    fuel_types = list(FuelType)

    pollutants = ["NOx", "SO2", "PM", "CO", "VOC", "CO2"]

    sources = []

    for i in range(count):
        unit = random.choice(units)
        source_type = random.choice(source_types[:5])  # Exclude less common types
        fuel_type = random.choice(fuel_types[:5]) if source_type == SourceType.COMBUSTION else None

        emissions = {p: random.uniform(10, 500) for p in random.sample(pollutants, random.randint(3, 6))}
        total = sum(emissions.values())

        source = EmissionSource(
            source_id=f"SRC-{i+1:03d}",
            source_name=f"{source_type.display_name} {i+1}",
            source_type=source_type,
            unit_id=unit[0],
            unit_name=unit[1],
            fuel_type=fuel_type,
            emissions_by_pollutant=emissions,
            total_emissions=total,
            unit="tons/year",
            operating_hours=random.uniform(4000, 8760),
            capacity_factor=random.uniform(0.5, 0.95),
            permit_limit=total * random.uniform(1.1, 1.5),
            stack_height=random.uniform(50, 200)
        )

        sources.append(source)

    return sources


if __name__ == "__main__":
    # Demo usage
    sources = create_sample_sources(20)
    config = SourceBreakdownConfig(
        title="Facility Emissions Breakdown",
        color_blind_safe=False
    )

    chart = SourceBreakdownChart(sources, config)

    print("Pie chart JSON (first 500 chars):")
    print(chart.to_plotly_json()[:500])

    stats = chart.get_summary_statistics()
    print(f"\nSummary Statistics:")
    print(f"  Total sources: {stats['total_sources']}")
    print(f"  Total emissions: {stats['total_emissions']:,.0f}")
    print(f"  Source types: {len(stats['by_type'])}")
    print(f"  Top source: {stats['top_sources'][0]['name'] if stats['top_sources'] else 'N/A'}")
