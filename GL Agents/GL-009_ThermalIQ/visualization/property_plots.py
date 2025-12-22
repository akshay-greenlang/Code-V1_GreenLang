"""
ThermalIQ Fluid Property Plotter

Generates interactive plots for comparing thermophysical properties
of heat transfer fluids across temperature ranges.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PropertyType(Enum):
    """Types of thermophysical properties."""
    SPECIFIC_HEAT = "specific_heat"
    VISCOSITY = "viscosity"
    DENSITY = "density"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    PRANDTL_NUMBER = "prandtl_number"
    VAPOR_PRESSURE = "vapor_pressure"
    SURFACE_TENSION = "surface_tension"
    ENTHALPY = "enthalpy"


@dataclass
class PropertyData:
    """Container for fluid property data."""
    fluid_name: str
    temperatures: np.ndarray  # Temperature values (C or K)
    values: np.ndarray  # Property values
    unit: str
    property_type: PropertyType
    temperature_unit: str = "C"
    source: str = ""
    uncertainty: Optional[float] = None  # Relative uncertainty (%)


@dataclass
class ComparisonChart:
    """Container for property comparison chart data."""
    property_type: PropertyType
    fluids: List[str]
    temperature_range: Tuple[float, float]
    data: Dict[str, PropertyData]
    figure: Optional[go.Figure] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FluidPropertyPlotter:
    """
    Generates interactive plots for fluid thermophysical properties.

    Supports plotting and comparison of:
    - Specific heat capacity (Cp)
    - Dynamic viscosity
    - Density
    - Thermal conductivity
    - Prandtl number
    - And other properties
    """

    # Property metadata
    PROPERTY_INFO = {
        PropertyType.SPECIFIC_HEAT: {
            "name": "Specific Heat Capacity",
            "symbol": "Cp",
            "default_unit": "kJ/kg-K",
            "axis_title": "Specific Heat Capacity (kJ/kg-K)",
            "color": "#e74c3c"
        },
        PropertyType.VISCOSITY: {
            "name": "Dynamic Viscosity",
            "symbol": "mu",
            "default_unit": "mPa-s",
            "axis_title": "Dynamic Viscosity (mPa-s)",
            "color": "#3498db"
        },
        PropertyType.DENSITY: {
            "name": "Density",
            "symbol": "rho",
            "default_unit": "kg/m3",
            "axis_title": "Density (kg/m^3)",
            "color": "#27ae60"
        },
        PropertyType.THERMAL_CONDUCTIVITY: {
            "name": "Thermal Conductivity",
            "symbol": "k",
            "default_unit": "W/m-K",
            "axis_title": "Thermal Conductivity (W/m-K)",
            "color": "#f39c12"
        },
        PropertyType.PRANDTL_NUMBER: {
            "name": "Prandtl Number",
            "symbol": "Pr",
            "default_unit": "-",
            "axis_title": "Prandtl Number (-)",
            "color": "#9b59b6"
        },
        PropertyType.VAPOR_PRESSURE: {
            "name": "Vapor Pressure",
            "symbol": "Pv",
            "default_unit": "kPa",
            "axis_title": "Vapor Pressure (kPa)",
            "color": "#1abc9c"
        },
        PropertyType.SURFACE_TENSION: {
            "name": "Surface Tension",
            "symbol": "sigma",
            "default_unit": "mN/m",
            "axis_title": "Surface Tension (mN/m)",
            "color": "#e67e22"
        },
        PropertyType.ENTHALPY: {
            "name": "Specific Enthalpy",
            "symbol": "h",
            "default_unit": "kJ/kg",
            "axis_title": "Specific Enthalpy (kJ/kg)",
            "color": "#c0392b"
        }
    }

    # Default color palette for multiple fluids
    COLOR_PALETTE = [
        "#e74c3c",  # Red
        "#3498db",  # Blue
        "#27ae60",  # Green
        "#f39c12",  # Orange
        "#9b59b6",  # Purple
        "#1abc9c",  # Teal
        "#e67e22",  # Dark orange
        "#2c3e50",  # Dark blue
        "#16a085",  # Dark teal
        "#c0392b",  # Dark red
    ]

    def __init__(
        self,
        property_calculator: Optional[Callable] = None,
        default_n_points: int = 100
    ):
        """
        Initialize the property plotter.

        Args:
            property_calculator: Optional function to calculate properties
                                Signature: (fluid_name, property_type, temperature) -> value
            default_n_points: Default number of points for temperature range
        """
        self.property_calculator = property_calculator
        self.default_n_points = default_n_points
        self._cached_data: Dict[str, PropertyData] = {}

    def plot_Cp_vs_temperature(
        self,
        fluids: Union[List[str], Dict[str, Callable]],
        T_range: Tuple[float, float],
        n_points: int = None,
        temperature_unit: str = "C",
        title: str = None
    ) -> go.Figure:
        """
        Plot specific heat capacity vs temperature for multiple fluids.

        Args:
            fluids: List of fluid names or dict of {name: Cp_function}
            T_range: Temperature range (T_min, T_max)
            n_points: Number of temperature points
            temperature_unit: "C" or "K"
            title: Optional custom title

        Returns:
            Plotly figure with Cp vs T plot
        """
        return self._plot_property_vs_temperature(
            fluids=fluids,
            property_type=PropertyType.SPECIFIC_HEAT,
            T_range=T_range,
            n_points=n_points,
            temperature_unit=temperature_unit,
            title=title or "Specific Heat Capacity vs Temperature"
        )

    def plot_viscosity_vs_temperature(
        self,
        fluids: Union[List[str], Dict[str, Callable]],
        T_range: Tuple[float, float],
        n_points: int = None,
        temperature_unit: str = "C",
        log_scale: bool = True,
        title: str = None
    ) -> go.Figure:
        """
        Plot dynamic viscosity vs temperature for multiple fluids.

        Args:
            fluids: List of fluid names or dict of {name: viscosity_function}
            T_range: Temperature range (T_min, T_max)
            n_points: Number of temperature points
            temperature_unit: "C" or "K"
            log_scale: Whether to use log scale for viscosity axis
            title: Optional custom title

        Returns:
            Plotly figure with viscosity vs T plot
        """
        fig = self._plot_property_vs_temperature(
            fluids=fluids,
            property_type=PropertyType.VISCOSITY,
            T_range=T_range,
            n_points=n_points,
            temperature_unit=temperature_unit,
            title=title or "Dynamic Viscosity vs Temperature"
        )

        if log_scale:
            fig.update_yaxes(type="log")

        return fig

    def plot_density_vs_temperature(
        self,
        fluids: Union[List[str], Dict[str, Callable]],
        T_range: Tuple[float, float],
        n_points: int = None,
        temperature_unit: str = "C",
        title: str = None
    ) -> go.Figure:
        """
        Plot density vs temperature for multiple fluids.

        Args:
            fluids: List of fluid names or dict of {name: density_function}
            T_range: Temperature range (T_min, T_max)
            n_points: Number of temperature points
            temperature_unit: "C" or "K"
            title: Optional custom title

        Returns:
            Plotly figure with density vs T plot
        """
        return self._plot_property_vs_temperature(
            fluids=fluids,
            property_type=PropertyType.DENSITY,
            T_range=T_range,
            n_points=n_points,
            temperature_unit=temperature_unit,
            title=title or "Density vs Temperature"
        )

    def plot_conductivity_vs_temperature(
        self,
        fluids: Union[List[str], Dict[str, Callable]],
        T_range: Tuple[float, float],
        n_points: int = None,
        temperature_unit: str = "C",
        title: str = None
    ) -> go.Figure:
        """
        Plot thermal conductivity vs temperature for multiple fluids.

        Args:
            fluids: List of fluid names or dict of {name: conductivity_function}
            T_range: Temperature range (T_min, T_max)
            n_points: Number of temperature points
            temperature_unit: "C" or "K"
            title: Optional custom title

        Returns:
            Plotly figure with thermal conductivity vs T plot
        """
        return self._plot_property_vs_temperature(
            fluids=fluids,
            property_type=PropertyType.THERMAL_CONDUCTIVITY,
            T_range=T_range,
            n_points=n_points,
            temperature_unit=temperature_unit,
            title=title or "Thermal Conductivity vs Temperature"
        )

    def compare_fluids(
        self,
        fluids: Union[List[str], Dict[str, Dict[str, Callable]]],
        property_type: PropertyType,
        T_range: Tuple[float, float],
        n_points: int = None,
        temperature_unit: str = "C",
        show_uncertainty: bool = False
    ) -> ComparisonChart:
        """
        Compare a specific property across multiple fluids.

        Args:
            fluids: List of fluid names or dict of {name: {property_type: function}}
            property_type: Property to compare
            T_range: Temperature range
            n_points: Number of temperature points
            temperature_unit: "C" or "K"
            show_uncertainty: Whether to show uncertainty bands

        Returns:
            ComparisonChart with comparison data and figure
        """
        n_points = n_points or self.default_n_points
        temperatures = np.linspace(T_range[0], T_range[1], n_points)

        property_info = self.PROPERTY_INFO[property_type]
        data = {}

        # Get fluid list
        if isinstance(fluids, dict):
            fluid_names = list(fluids.keys())
        else:
            fluid_names = fluids

        # Calculate property for each fluid
        for i, fluid_name in enumerate(fluid_names):
            if isinstance(fluids, dict) and fluid_name in fluids:
                # Use provided function
                prop_func = fluids[fluid_name]
                if isinstance(prop_func, dict):
                    prop_func = prop_func.get(property_type.value)

                if prop_func:
                    values = np.array([prop_func(T) for T in temperatures])
                else:
                    values = np.zeros_like(temperatures)
            elif self.property_calculator:
                # Use default calculator
                values = np.array([
                    self.property_calculator(fluid_name, property_type, T)
                    for T in temperatures
                ])
            else:
                # Generate placeholder data
                values = self._generate_placeholder_data(
                    property_type, temperatures, seed=i
                )

            data[fluid_name] = PropertyData(
                fluid_name=fluid_name,
                temperatures=temperatures,
                values=values,
                unit=property_info["default_unit"],
                property_type=property_type,
                temperature_unit=temperature_unit
            )

        # Generate figure
        figure = self._create_comparison_figure(
            data, property_type, temperature_unit, show_uncertainty
        )

        return ComparisonChart(
            property_type=property_type,
            fluids=fluid_names,
            temperature_range=T_range,
            data=data,
            figure=figure,
            metadata={
                "n_points": n_points,
                "show_uncertainty": show_uncertainty
            }
        )

    def export_comparison_chart(
        self,
        chart: ComparisonChart,
        path: Union[str, Path],
        format: str = "png",
        width: int = 1000,
        height: int = 600
    ) -> None:
        """
        Export comparison chart to file.

        Args:
            chart: ComparisonChart to export
            path: Output file path
            format: "png", "svg", "pdf", or "html"
            width: Image width in pixels
            height: Image height in pixels
        """
        if chart.figure is None:
            raise ValueError("Chart has no figure to export")

        path = Path(path)

        if format == "html":
            chart.figure.write_html(str(path))
        else:
            chart.figure.write_image(
                str(path),
                format=format,
                width=width,
                height=height,
                scale=2.0
            )

    def create_property_dashboard(
        self,
        fluids: Union[List[str], Dict[str, Dict[str, Callable]]],
        T_range: Tuple[float, float],
        properties: List[PropertyType] = None,
        temperature_unit: str = "C"
    ) -> go.Figure:
        """
        Create a dashboard with multiple property plots.

        Args:
            fluids: List of fluid names or dict of property functions
            T_range: Temperature range
            properties: List of properties to include (default: all main properties)
            temperature_unit: "C" or "K"

        Returns:
            Plotly figure with multi-property dashboard
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        properties = properties or [
            PropertyType.SPECIFIC_HEAT,
            PropertyType.VISCOSITY,
            PropertyType.DENSITY,
            PropertyType.THERMAL_CONDUCTIVITY
        ]

        n_props = len(properties)
        n_cols = 2
        n_rows = (n_props + 1) // 2

        # Create subplot titles
        subplot_titles = [
            self.PROPERTY_INFO[p]["name"] for p in properties
        ]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Get fluid list
        if isinstance(fluids, dict):
            fluid_names = list(fluids.keys())
        else:
            fluid_names = fluids

        n_points = self.default_n_points
        temperatures = np.linspace(T_range[0], T_range[1], n_points)

        # Add traces for each property
        for prop_idx, prop_type in enumerate(properties):
            row = prop_idx // n_cols + 1
            col = prop_idx % n_cols + 1

            property_info = self.PROPERTY_INFO[prop_type]

            for fluid_idx, fluid_name in enumerate(fluid_names):
                color = self.COLOR_PALETTE[fluid_idx % len(self.COLOR_PALETTE)]

                # Get property values
                if isinstance(fluids, dict) and fluid_name in fluids:
                    prop_funcs = fluids[fluid_name]
                    if isinstance(prop_funcs, dict):
                        prop_func = prop_funcs.get(prop_type.value)
                    else:
                        prop_func = None

                    if prop_func:
                        values = np.array([prop_func(T) for T in temperatures])
                    else:
                        values = self._generate_placeholder_data(
                            prop_type, temperatures, seed=fluid_idx
                        )
                elif self.property_calculator:
                    values = np.array([
                        self.property_calculator(fluid_name, prop_type, T)
                        for T in temperatures
                    ])
                else:
                    values = self._generate_placeholder_data(
                        prop_type, temperatures, seed=fluid_idx
                    )

                # Show legend only on first subplot
                show_legend = (prop_idx == 0)

                fig.add_trace(
                    go.Scatter(
                        x=temperatures,
                        y=values,
                        mode='lines',
                        name=fluid_name,
                        line=dict(color=color, width=2),
                        legendgroup=fluid_name,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"<b>{fluid_name}</b><br>" +
                            f"T: %{{x:.1f}} {temperature_unit}<br>" +
                            f"{property_info['symbol']}: %{{y:.3f}} {property_info['default_unit']}<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=row,
                    col=col
                )

            # Update axes
            fig.update_xaxes(
                title_text=f"Temperature ({temperature_unit})",
                row=row,
                col=col
            )
            fig.update_yaxes(
                title_text=f"{property_info['symbol']} ({property_info['default_unit']})",
                row=row,
                col=col
            )

            # Log scale for viscosity
            if prop_type == PropertyType.VISCOSITY:
                fig.update_yaxes(type="log", row=row, col=col)

        fig.update_layout(
            title=dict(
                text="Fluid Properties Comparison Dashboard",
                font=dict(size=18)
            ),
            height=400 * n_rows,
            width=1000,
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )

        return fig

    def _plot_property_vs_temperature(
        self,
        fluids: Union[List[str], Dict[str, Callable]],
        property_type: PropertyType,
        T_range: Tuple[float, float],
        n_points: int = None,
        temperature_unit: str = "C",
        title: str = None
    ) -> go.Figure:
        """Generate property vs temperature plot."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        n_points = n_points or self.default_n_points
        temperatures = np.linspace(T_range[0], T_range[1], n_points)

        property_info = self.PROPERTY_INFO[property_type]

        # Get fluid list
        if isinstance(fluids, dict):
            fluid_names = list(fluids.keys())
            fluid_funcs = fluids
        else:
            fluid_names = fluids
            fluid_funcs = {}

        fig = go.Figure()

        for i, fluid_name in enumerate(fluid_names):
            color = self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)]

            # Get property values
            if fluid_name in fluid_funcs:
                prop_func = fluid_funcs[fluid_name]
                values = np.array([prop_func(T) for T in temperatures])
            elif self.property_calculator:
                values = np.array([
                    self.property_calculator(fluid_name, property_type, T)
                    for T in temperatures
                ])
            else:
                values = self._generate_placeholder_data(
                    property_type, temperatures, seed=i
                )

            fig.add_trace(go.Scatter(
                x=temperatures,
                y=values,
                mode='lines',
                name=fluid_name,
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{fluid_name}</b><br>" +
                    f"Temperature: %{{x:.1f}} {temperature_unit}<br>" +
                    f"{property_info['name']}: %{{y:.4f}} {property_info['default_unit']}<br>" +
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title=dict(
                text=title or property_info["name"],
                font=dict(size=18)
            ),
            xaxis_title=f"Temperature ({temperature_unit})",
            yaxis_title=property_info["axis_title"],
            template="plotly_white",
            height=500,
            width=800,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            hovermode="x unified"
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

        return fig

    def _create_comparison_figure(
        self,
        data: Dict[str, PropertyData],
        property_type: PropertyType,
        temperature_unit: str,
        show_uncertainty: bool
    ) -> go.Figure:
        """Create comparison figure from property data."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        property_info = self.PROPERTY_INFO[property_type]

        fig = go.Figure()

        for i, (fluid_name, prop_data) in enumerate(data.items()):
            color = self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)]

            # Main line
            fig.add_trace(go.Scatter(
                x=prop_data.temperatures,
                y=prop_data.values,
                mode='lines',
                name=fluid_name,
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"<b>{fluid_name}</b><br>" +
                    f"T: %{{x:.1f}} {temperature_unit}<br>" +
                    f"{property_info['symbol']}: %{{y:.4f}} {prop_data.unit}<br>" +
                    "<extra></extra>"
                )
            ))

            # Uncertainty band
            if show_uncertainty and prop_data.uncertainty:
                uncertainty_factor = prop_data.uncertainty / 100
                upper = prop_data.values * (1 + uncertainty_factor)
                lower = prop_data.values * (1 - uncertainty_factor)

                fig.add_trace(go.Scatter(
                    x=np.concatenate([prop_data.temperatures, prop_data.temperatures[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor=color.replace('1)', '0.2)') if 'rgba' in color else f"rgba(127,127,127,0.2)",
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))

        fig.update_layout(
            title=dict(
                text=f"{property_info['name']} Comparison",
                font=dict(size=18)
            ),
            xaxis_title=f"Temperature ({temperature_unit})",
            yaxis_title=property_info["axis_title"],
            template="plotly_white",
            height=500,
            width=900,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        return fig

    def _generate_placeholder_data(
        self,
        property_type: PropertyType,
        temperatures: np.ndarray,
        seed: int = 0
    ) -> np.ndarray:
        """Generate placeholder property data for demonstration."""
        np.random.seed(seed)
        T = temperatures

        if property_type == PropertyType.SPECIFIC_HEAT:
            # Typical Cp: 1.5-4.5 kJ/kg-K, slight increase with T
            base = 2.0 + seed * 0.5
            return base + 0.002 * (T - T.min()) + np.random.normal(0, 0.01, len(T))

        elif property_type == PropertyType.VISCOSITY:
            # Typical viscosity: decreases exponentially with T
            base = 50 + seed * 20
            return base * np.exp(-0.03 * (T - T.min())) + np.random.normal(0, 0.1, len(T))

        elif property_type == PropertyType.DENSITY:
            # Typical density: 800-1200 kg/m3, slight decrease with T
            base = 1000 - seed * 50
            return base - 0.5 * (T - T.min()) + np.random.normal(0, 0.5, len(T))

        elif property_type == PropertyType.THERMAL_CONDUCTIVITY:
            # Typical k: 0.1-0.7 W/m-K
            base = 0.15 + seed * 0.05
            return base + 0.0002 * (T - T.min()) + np.random.normal(0, 0.001, len(T))

        elif property_type == PropertyType.PRANDTL_NUMBER:
            # Prandtl number: decreases with T (due to viscosity)
            base = 100 + seed * 50
            return base * np.exp(-0.02 * (T - T.min()))

        else:
            # Generic linear relationship
            return 1.0 + 0.01 * T + np.random.normal(0, 0.01, len(T))

    def create_radar_comparison(
        self,
        fluids: Dict[str, Dict[str, float]],
        properties: List[str] = None,
        title: str = "Fluid Property Comparison"
    ) -> go.Figure:
        """
        Create radar chart comparing normalized fluid properties.

        Args:
            fluids: Dict of {fluid_name: {property_name: value}}
            properties: Properties to include (uses all if None)
            title: Chart title

        Returns:
            Plotly figure with radar chart
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization")

        if not fluids:
            raise ValueError("At least one fluid is required")

        # Get properties to plot
        if properties is None:
            properties = list(list(fluids.values())[0].keys())

        # Normalize values (0-1 scale)
        normalized = {}
        for prop in properties:
            values = [fluids[f].get(prop, 0) for f in fluids.keys()]
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val if max_val > min_val else 1

            for fluid_name in fluids.keys():
                if fluid_name not in normalized:
                    normalized[fluid_name] = {}
                raw_val = fluids[fluid_name].get(prop, 0)
                normalized[fluid_name][prop] = (raw_val - min_val) / range_val

        fig = go.Figure()

        for i, fluid_name in enumerate(fluids.keys()):
            color = self.COLOR_PALETTE[i % len(self.COLOR_PALETTE)]
            values = [normalized[fluid_name][p] for p in properties]
            values.append(values[0])  # Close the polygon

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=properties + [properties[0]],
                fill='toself',
                name=fluid_name,
                line=dict(color=color),
                fillcolor=color.replace('0.8)', '0.2)') if 'rgba' in color else f"rgba(127,127,127,0.2)"
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=dict(text=title, font=dict(size=16)),
            height=500,
            width=600
        )

        return fig
