# -*- coding: utf-8 -*-
"""
Chart Generator
GL-VCCI Scope 3 Platform

Generates charts and visualizations for sustainability reports.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import io
import base64

# Import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, charts will not be generated")

# Import seaborn for better styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import numpy as np

from ..models import ChartInfo, EmissionsData
from ..config import CHART_CONFIG, ChartType
from ..exceptions import ChartGenerationError

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generates charts and visualizations for sustainability reports.

    Features:
    - Scope 1, 2, 3 pie chart
    - Category breakdown bar chart
    - Year-over-year trends
    - Data quality heatmaps
    - Pareto charts
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize chart generator.

        Args:
            config: Optional chart configuration
        """
        if not HAS_MATPLOTLIB:
            raise ChartGenerationError("matplotlib is required for chart generation")

        self.config = config or CHART_CONFIG
        self._setup_style()

    def _setup_style(self):
        """Setup matplotlib style."""
        if HAS_SEABORN:
            sns.set_style("whitegrid")
        else:
            plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Set font sizes
        plt.rcParams.update({
            'font.size': self.config.get("font", {}).get("size", 11),
            'axes.titlesize': self.config.get("font", {}).get("title_size", 14),
            'axes.labelsize': 11,
            'legend.fontsize': 10,
        })

    def generate_scope_pie_chart(
        self,
        emissions_data: EmissionsData,
        output_path: Optional[str] = None,
    ) -> ChartInfo:
        """
        Generate pie chart showing Scope 1, 2, 3 breakdown.

        Args:
            emissions_data: Emissions data
            output_path: Optional output file path

        Returns:
            ChartInfo with generated chart
        """
        logger.info("Generating Scope 1, 2, 3 pie chart")

        fig, ax = plt.subplots(figsize=self.config.get("figure_size", (12, 8)))

        # Data
        scopes = ["Scope 1", "Scope 2\n(Location)", "Scope 3"]
        values = [
            emissions_data.scope1_tco2e,
            emissions_data.scope2_location_tco2e,
            emissions_data.scope3_tco2e,
        ]

        # Colors
        colors = [
            self.config.get("colors", {}).get("scope1", "#FF6B6B"),
            self.config.get("colors", {}).get("scope2", "#4ECDC4"),
            self.config.get("colors", {}).get("scope3", "#45B7D1"),
        ]

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=scopes,
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({pct * sum(values) / 100:.0f} tCO2e)',
            startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'},
        )

        # Make percentage text white
        for autotext in autotexts:
            autotext.set_color('white')

        ax.set_title(
            f"GHG Emissions by Scope\nTotal: {sum(values):,.0f} tCO2e",
            fontsize=14,
            fontweight='bold',
            pad=20,
        )

        plt.tight_layout()

        # Save or return
        if output_path:
            plt.savefig(output_path, dpi=self.config.get("dpi", 300), bbox_inches='tight')
            logger.info(f"Scope pie chart saved to {output_path}")
        else:
            output_path = self._save_to_temp("scope_pie_chart.png", fig)

        plt.close(fig)

        return ChartInfo(
            chart_id="scope_pie_chart",
            chart_type=ChartType.PIE,
            title="GHG Emissions by Scope",
            image_path=output_path,
            width=1200,
            height=800,
        )

    def generate_category_bar_chart(
        self,
        emissions_data: EmissionsData,
        output_path: Optional[str] = None,
    ) -> ChartInfo:
        """
        Generate bar chart showing Scope 3 category breakdown.

        Args:
            emissions_data: Emissions data
            output_path: Optional output file path

        Returns:
            ChartInfo with generated chart
        """
        logger.info("Generating Scope 3 category bar chart")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Sort categories by emissions
        categories = sorted(
            emissions_data.scope3_categories.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        category_names = [f"Cat {cat}" for cat, _ in categories]
        values = [val for _, val in categories]

        # Create bar chart
        bars = ax.barh(category_names, values, color=self.config.get("colors", {}).get("scope3", "#45B7D1"))

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f' {val:,.0f} tCO2e ({val / sum(values) * 100:.1f}%)',
                ha='left',
                va='center',
                fontsize=10,
            )

        ax.set_xlabel('Emissions (tCO2e)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Scope 3 Emissions by Category\nTotal: {sum(values):,.0f} tCO2e',
            fontsize=14,
            fontweight='bold',
            pad=20,
        )

        # Grid
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save
        if output_path:
            plt.savefig(output_path, dpi=self.config.get("dpi", 300), bbox_inches='tight')
        else:
            output_path = self._save_to_temp("category_bar_chart.png", fig)

        plt.close(fig)

        return ChartInfo(
            chart_id="category_bar_chart",
            chart_type=ChartType.BAR,
            title="Scope 3 Emissions by Category",
            image_path=output_path,
            width=1400,
            height=800,
        )

    def generate_yoy_trend_chart(
        self,
        emissions_data: EmissionsData,
        output_path: Optional[str] = None,
    ) -> ChartInfo:
        """
        Generate year-over-year trend line chart.

        Args:
            emissions_data: Emissions data
            output_path: Optional output file path

        Returns:
            ChartInfo with generated chart
        """
        logger.info("Generating year-over-year trend chart")

        if not emissions_data.prior_year_emissions:
            raise ChartGenerationError("Prior year emissions data required for YoY chart")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Data
        years = [emissions_data.reporting_year - 1, emissions_data.reporting_year]
        prior_total = emissions_data.prior_year_emissions.get("total_tco2e", 0)
        current_total = (
            emissions_data.scope1_tco2e
            + emissions_data.scope2_location_tco2e
            + emissions_data.scope3_tco2e
        )

        scope1_prior = emissions_data.prior_year_emissions.get("scope1_tco2e", 0)
        scope2_prior = emissions_data.prior_year_emissions.get("scope2_tco2e", 0)
        scope3_prior = emissions_data.prior_year_emissions.get("scope3_tco2e", 0)

        # Plot stacked area
        scope1_values = [scope1_prior, emissions_data.scope1_tco2e]
        scope2_values = [scope2_prior, emissions_data.scope2_location_tco2e]
        scope3_values = [scope3_prior, emissions_data.scope3_tco2e]

        ax.plot(years, scope1_values, marker='o', linewidth=2.5, label='Scope 1', color=self.config.get("colors", {}).get("scope1", "#FF6B6B"))
        ax.plot(years, scope2_values, marker='s', linewidth=2.5, label='Scope 2', color=self.config.get("colors", {}).get("scope2", "#4ECDC4"))
        ax.plot(years, scope3_values, marker='^', linewidth=2.5, label='Scope 3', color=self.config.get("colors", {}).get("scope3", "#45B7D1"))

        # Total line
        total_values = [prior_total, current_total]
        ax.plot(years, total_values, marker='D', linewidth=3, label='Total', color='#2C3E50', linestyle='--')

        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Emissions (tCO2e)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Year-over-Year Emissions Trend\nChange: {emissions_data.yoy_change_pct:+.1f}%',
            fontsize=14,
            fontweight='bold',
            pad=20,
        )

        ax.legend(loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        plt.tight_layout()

        # Save
        if output_path:
            plt.savefig(output_path, dpi=self.config.get("dpi", 300), bbox_inches='tight')
        else:
            output_path = self._save_to_temp("yoy_trend_chart.png", fig)

        plt.close(fig)

        return ChartInfo(
            chart_id="yoy_trend_chart",
            chart_type=ChartType.LINE,
            title="Year-over-Year Emissions Trend",
            image_path=output_path,
            width=1200,
            height=800,
        )

    def generate_intensity_chart(
        self,
        intensity_metrics: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> ChartInfo:
        """
        Generate intensity metrics bar chart.

        Args:
            intensity_metrics: Intensity metrics
            output_path: Optional output file path

        Returns:
            ChartInfo with generated chart
        """
        logger.info("Generating intensity metrics chart")

        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_names = []
        metrics_values = []

        if "tco2e_per_million_usd" in intensity_metrics:
            metrics_names.append("tCO2e per\n$M Revenue")
            metrics_values.append(intensity_metrics["tco2e_per_million_usd"])

        if "tco2e_per_fte" in intensity_metrics:
            metrics_names.append("tCO2e per\nEmployee")
            metrics_values.append(intensity_metrics["tco2e_per_fte"])

        if not metrics_values:
            raise ChartGenerationError("No intensity metrics available")

        bars = ax.bar(metrics_names, metrics_values, color='#3498DB', alpha=0.8)

        # Add values on bars
        for bar, val in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
            )

        ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
        ax.set_title('Carbon Intensity Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save
        if output_path:
            plt.savefig(output_path, dpi=self.config.get("dpi", 300), bbox_inches='tight')
        else:
            output_path = self._save_to_temp("intensity_chart.png", fig)

        plt.close(fig)

        return ChartInfo(
            chart_id="intensity_chart",
            chart_type=ChartType.BAR,
            title="Carbon Intensity Metrics",
            image_path=output_path,
            width=1000,
            height=600,
        )

    def generate_data_quality_heatmap(
        self,
        dqi_by_scope: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> ChartInfo:
        """
        Generate data quality heatmap.

        Args:
            dqi_by_scope: DQI scores by scope
            output_path: Optional output file path

        Returns:
            ChartInfo with generated chart
        """
        logger.info("Generating data quality heatmap")

        fig, ax = plt.subplots(figsize=(10, 4))

        scopes = list(dqi_by_scope.keys())
        values = list(dqi_by_scope.values())

        # Create heatmap data
        data = np.array([values])

        # Color map
        cmap = plt.cm.RdYlGn
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(range(len(scopes)))
        ax.set_xticklabels(scopes)
        ax.set_yticks([])

        # Add text annotations
        for i, (scope, val) in enumerate(zip(scopes, values)):
            text_color = 'white' if val < 50 else 'black'
            ax.text(i, 0, f'{val:.1f}', ha='center', va='center', fontsize=12, fontweight='bold', color=text_color)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Data Quality Index (DQI)', fontsize=11)

        ax.set_title('Data Quality by Scope', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save
        if output_path:
            plt.savefig(output_path, dpi=self.config.get("dpi", 300), bbox_inches='tight')
        else:
            output_path = self._save_to_temp("dqi_heatmap.png", fig)

        plt.close(fig)

        return ChartInfo(
            chart_id="dqi_heatmap",
            chart_type=ChartType.HEATMAP,
            title="Data Quality by Scope",
            image_path=output_path,
            width=1000,
            height=400,
        )

    def _save_to_temp(self, filename: str, fig) -> str:
        """Save figure to temporary file."""
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "vcci_reports"
        temp_dir.mkdir(exist_ok=True)

        output_path = temp_dir / filename
        fig.savefig(str(output_path), dpi=self.config.get("dpi", 300), bbox_inches='tight')

        return str(output_path)

    def generate_all_charts(
        self,
        emissions_data: EmissionsData,
        intensity_metrics: Optional[Dict[str, float]] = None,
        output_dir: Optional[str] = None,
    ) -> List[ChartInfo]:
        """
        Generate all standard charts for report.

        Args:
            emissions_data: Emissions data
            intensity_metrics: Optional intensity metrics
            output_dir: Optional output directory

        Returns:
            List of ChartInfo objects
        """
        logger.info("Generating all charts")

        charts = []

        # Scope pie chart
        try:
            chart = self.generate_scope_pie_chart(emissions_data, output_dir)
            charts.append(chart)
        except Exception as e:
            logger.error(f"Failed to generate scope pie chart: {e}")

        # Category bar chart
        try:
            chart = self.generate_category_bar_chart(emissions_data, output_dir)
            charts.append(chart)
        except Exception as e:
            logger.error(f"Failed to generate category bar chart: {e}")

        # YoY trend (if data available)
        if emissions_data.prior_year_emissions:
            try:
                chart = self.generate_yoy_trend_chart(emissions_data, output_dir)
                charts.append(chart)
            except Exception as e:
                logger.error(f"Failed to generate YoY chart: {e}")

        # Intensity chart (if data available)
        if intensity_metrics:
            try:
                chart = self.generate_intensity_chart(intensity_metrics, output_dir)
                charts.append(chart)
            except Exception as e:
                logger.error(f"Failed to generate intensity chart: {e}")

        # Data quality heatmap
        if emissions_data.data_quality_by_scope:
            try:
                chart = self.generate_data_quality_heatmap(emissions_data.data_quality_by_scope, output_dir)
                charts.append(chart)
            except Exception as e:
                logger.error(f"Failed to generate DQI heatmap: {e}")

        logger.info(f"Generated {len(charts)} charts")
        return charts


__all__ = ["ChartGenerator"]
