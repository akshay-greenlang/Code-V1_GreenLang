"""GL-009 THERMALIQ Visualization Examples.

Comprehensive examples demonstrating all visualization capabilities:
- Sankey diagrams for energy flow
- Waterfall charts for heat balance
- Efficiency trends over time
- Loss breakdown charts
- Multi-chart dashboards

Run this file to generate example visualizations.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from sankey_engine import SankeyEngine, ColorScheme
from waterfall_chart import WaterfallChart
from efficiency_trends import EfficiencyTrends
from loss_breakdown import LossBreakdown
from export import (
    export_to_html,
    export_to_json,
    export_dashboard,
    ExportConfig
)


def example_boiler_sankey() -> Dict:
    """Generate example Sankey diagram for industrial boiler."""
    print("Generating boiler Sankey diagram...")

    # Example boiler data
    energy_inputs = {
        "natural_gas": 5000.0,  # kW
        "electricity": 150.0
    }

    useful_outputs = {
        "steam": 4200.0,
        "hot_water": 300.0
    }

    losses = {
        "flue_gas": 350.0,
        "radiation": 120.0,
        "blowdown": 100.0,
        "convection": 80.0
    }

    engine = SankeyEngine(color_scheme=ColorScheme.ENERGY_TYPE)
    diagram = engine.generate_from_efficiency_result(
        energy_inputs=energy_inputs,
        useful_outputs=useful_outputs,
        losses=losses,
        title="Industrial Boiler Energy Flow",
        process_name="Steam Boiler",
        metadata={
            "facility": "Manufacturing Plant A",
            "boiler_id": "BOILER-001",
            "rated_capacity": "5 MW"
        }
    )

    print(f"  Efficiency: {diagram.efficiency_percent:.1f}%")
    print(f"  Total Input: {diagram.total_input_kw:.0f} kW")
    print(f"  Useful Output: {diagram.total_output_kw:.0f} kW")
    print(f"  Total Losses: {diagram.total_losses_kw:.0f} kW")

    return diagram.to_plotly_json()


def example_multi_stage_sankey() -> Dict:
    """Generate example multi-stage Sankey for cogeneration system."""
    print("\nGenerating multi-stage cogeneration Sankey...")

    stages = [
        {
            "name": "Combustion",
            "inputs": {"natural_gas": 10000.0},
            "outputs": {"hot_gas": 9200.0},
            "losses": {"unburned": 200.0, "radiation": 600.0}
        },
        {
            "name": "Turbine",
            "inputs": {"hot_gas": 9200.0},
            "outputs": {"mechanical": 3500.0, "exhaust_gas": 5200.0},
            "losses": {"friction": 300.0, "heat": 200.0}
        },
        {
            "name": "Generator",
            "inputs": {"mechanical": 3500.0},
            "outputs": {"electricity": 3300.0},
            "losses": {"electrical": 200.0}
        },
        {
            "name": "Heat Recovery",
            "inputs": {"exhaust_gas": 5200.0},
            "outputs": {"process_heat": 4500.0},
            "losses": {"stack": 700.0}
        }
    ]

    engine = SankeyEngine()
    diagram = engine.generate_multi_stage(
        stages=stages,
        title="Cogeneration System - Multi-Stage Energy Flow"
    )

    print(f"  Overall Efficiency: {diagram.efficiency_percent:.1f}%")

    return diagram.to_plotly_json()


def example_waterfall_chart() -> Dict:
    """Generate example waterfall chart for heat balance."""
    print("\nGenerating heat balance waterfall chart...")

    input_energy = {
        "fuel_input": 5150.0
    }

    losses = {
        "flue_gas": 350.0,
        "radiation": 120.0,
        "convection": 80.0,
        "blowdown": 100.0,
        "unburned": 50.0
    }

    useful_output = {
        "steam_output": 4450.0
    }

    chart = WaterfallChart()
    waterfall = chart.generate_from_heat_balance(
        input_energy=input_energy,
        losses=losses,
        useful_output=useful_output,
        title="Boiler Heat Balance Breakdown"
    )

    print(f"  Start: {waterfall.start_value:.0f} kW")
    print(f"  End: {waterfall.end_value:.0f} kW")
    print(f"  Total Losses: {waterfall.total_losses:.0f} kW")

    return waterfall.to_plotly_json()


def example_detailed_waterfall() -> Dict:
    """Generate detailed waterfall with process stages."""
    print("\nGenerating detailed waterfall chart...")

    input_energy = {"total_input": 5150.0}

    process_losses = {
        "combustion": 50.0,
        "heat_transfer": 200.0,
        "flue_gas": 350.0
    }

    distribution_losses = {
        "pipe_radiation": 80.0,
        "valve_leakage": 20.0
    }

    useful_output = {"delivered_steam": 4450.0}

    chart = WaterfallChart()
    waterfall = chart.generate_detailed_breakdown(
        input_energy=input_energy,
        process_losses=process_losses,
        distribution_losses=distribution_losses,
        useful_output=useful_output,
        title="Detailed Heat Balance: Process & Distribution"
    )

    return waterfall.to_plotly_json()


def example_efficiency_trend() -> Dict:
    """Generate example efficiency trend over time."""
    print("\nGenerating efficiency trend chart...")

    # Generate sample data for 30 days
    base_date = datetime(2024, 1, 1)
    efficiency_data = []

    for day in range(30):
        date = base_date + timedelta(days=day)
        # Simulate efficiency with some variation
        base_efficiency = 87.0
        variation = (day % 7 - 3) * 0.5  # Weekly pattern
        noise = (hash(str(date)) % 100) / 100.0 - 0.5
        efficiency = base_efficiency + variation + noise
        efficiency_data.append((date, efficiency))

    trends = EfficiencyTrends()
    trend = trends.generate_efficiency_trend(
        efficiency_data=efficiency_data,
        title="30-Day Thermal Efficiency Trend",
        benchmark_efficiency=88.0,
        moving_average_days=7
    )

    print(f"  Average Efficiency: {trend.avg_value:.2f}%")
    print(f"  Min: {trend.min_value:.2f}%, Max: {trend.max_value:.2f}%")
    print(f"  Std Dev: {trend.std_dev:.2f}%")

    return trend.to_plotly_json()


def example_loss_trends() -> Dict:
    """Generate loss trends over time."""
    print("\nGenerating loss trends...")

    base_date = datetime(2024, 1, 1)
    loss_data = []

    for day in range(30):
        date = base_date + timedelta(days=day)
        losses = {
            "flue_gas": 350.0 + (day % 5) * 10.0,
            "radiation": 120.0 + (day % 3) * 5.0,
            "convection": 80.0 + (day % 4) * 3.0
        }
        loss_data.append((date, losses))

    trends = EfficiencyTrends()
    loss_trends = trends.generate_loss_trend(
        loss_data=loss_data,
        title="Heat Loss Trends",
        loss_categories=["flue_gas", "radiation", "convection"]
    )

    print(f"  Generated trends for {len(loss_trends)} loss categories")

    # Return multi-metric chart
    metrics = {}
    for category, trend in loss_trends.items():
        metrics[category] = [(p.timestamp, p.value) for p in trend.points]

    return trends.generate_multi_metric_trend(
        metrics=metrics,
        title="Heat Loss Trends - All Categories",
        y_axis_label="Heat Loss",
        unit="kW"
    )


def example_loss_pie_chart() -> Dict:
    """Generate loss breakdown pie chart."""
    print("\nGenerating loss breakdown pie chart...")

    losses = {
        "flue_gas": 350.0,
        "radiation": 120.0,
        "convection": 80.0,
        "blowdown": 100.0,
        "unburned": 50.0
    }

    breakdown = LossBreakdown()
    chart = breakdown.generate_pie_chart(
        losses=losses,
        title="Heat Loss Distribution",
        total_input=5150.0
    )

    print(f"  Total Losses: {chart.total_value:.0f} kW")
    print(f"  Categories: {len(chart.categories)}")

    return chart.to_plotly_json()


def example_loss_donut_chart() -> Dict:
    """Generate loss breakdown donut chart."""
    print("\nGenerating loss breakdown donut chart...")

    losses = {
        "flue_gas": 350.0,
        "radiation": 120.0,
        "convection": 80.0,
        "blowdown": 100.0,
        "unburned": 50.0
    }

    breakdown = LossBreakdown()
    chart = breakdown.generate_donut_chart(
        losses=losses,
        title="Heat Loss Breakdown (Donut)",
        total_input=5150.0
    )

    return chart.to_plotly_json()


def example_loss_comparison() -> Dict:
    """Generate baseline vs current loss comparison."""
    print("\nGenerating loss comparison chart...")

    baseline_losses = {
        "flue_gas": 400.0,
        "radiation": 150.0,
        "convection": 100.0,
        "blowdown": 120.0,
        "unburned": 80.0
    }

    current_losses = {
        "flue_gas": 350.0,
        "radiation": 120.0,
        "convection": 80.0,
        "blowdown": 100.0,
        "unburned": 50.0
    }

    breakdown = LossBreakdown()
    chart = breakdown.generate_comparison_chart(
        baseline_losses=baseline_losses,
        current_losses=current_losses,
        title="Loss Reduction Analysis: Baseline vs Current"
    )

    return chart


def example_performance_comparison() -> Dict:
    """Generate baseline vs current performance comparison."""
    print("\nGenerating performance comparison...")

    base_date = datetime(2024, 1, 1)

    # Baseline data (previous period)
    baseline_data = []
    for day in range(30):
        date = base_date + timedelta(days=day)
        efficiency = 85.0 + (day % 7 - 3) * 0.3
        baseline_data.append((date, efficiency))

    # Current data (improved performance)
    current_data = []
    for day in range(30):
        date = base_date + timedelta(days=day)
        efficiency = 87.5 + (day % 7 - 3) * 0.3
        current_data.append((date, efficiency))

    trends = EfficiencyTrends()
    chart = trends.generate_comparison_chart(
        baseline_data=baseline_data,
        current_data=current_data,
        title="Performance Improvement: Before & After Optimization",
        y_axis_label="Thermal Efficiency",
        unit="%"
    )

    return chart


def generate_all_examples(output_dir: Path):
    """Generate all example visualizations.

    Args:
        output_dir: Directory to save example files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GL-009 THERMALIQ Visualization Examples")
    print(f"{'='*60}")

    # Configure export
    config = ExportConfig(width=1200, height=800)

    # 1. Boiler Sankey
    fig1 = example_boiler_sankey()
    export_to_html(
        fig1,
        output_dir / "01_boiler_sankey.html",
        title="Industrial Boiler Energy Flow",
        config=config
    )
    export_to_json(fig1, output_dir / "01_boiler_sankey.json", config=config)

    # 2. Multi-stage Sankey
    fig2 = example_multi_stage_sankey()
    export_to_html(
        fig2,
        output_dir / "02_cogeneration_sankey.html",
        title="Cogeneration System - Multi-Stage",
        config=config
    )

    # 3. Waterfall Chart
    fig3 = example_waterfall_chart()
    export_to_html(
        fig3,
        output_dir / "03_heat_balance_waterfall.html",
        title="Heat Balance Breakdown",
        config=config
    )

    # 4. Detailed Waterfall
    fig4 = example_detailed_waterfall()
    export_to_html(
        fig4,
        output_dir / "04_detailed_waterfall.html",
        title="Detailed Heat Balance",
        config=config
    )

    # 5. Efficiency Trend
    fig5 = example_efficiency_trend()
    export_to_html(
        fig5,
        output_dir / "05_efficiency_trend.html",
        title="30-Day Efficiency Trend",
        config=config
    )

    # 6. Loss Trends
    fig6 = example_loss_trends()
    export_to_html(
        fig6,
        output_dir / "06_loss_trends.html",
        title="Heat Loss Trends",
        config=config
    )

    # 7. Loss Pie Chart
    fig7 = example_loss_pie_chart()
    export_to_html(
        fig7,
        output_dir / "07_loss_pie_chart.html",
        title="Loss Distribution (Pie)",
        config=config
    )

    # 8. Loss Donut Chart
    fig8 = example_loss_donut_chart()
    export_to_html(
        fig8,
        output_dir / "08_loss_donut_chart.html",
        title="Loss Distribution (Donut)",
        config=config
    )

    # 9. Loss Comparison
    fig9 = example_loss_comparison()
    export_to_html(
        fig9,
        output_dir / "09_loss_comparison.html",
        title="Loss Reduction Analysis",
        config=config
    )

    # 10. Performance Comparison
    fig10 = example_performance_comparison()
    export_to_html(
        fig10,
        output_dir / "10_performance_comparison.html",
        title="Performance Improvement",
        config=config
    )

    # Create comprehensive dashboard
    print("\nGenerating comprehensive dashboard...")
    dashboard_figures = [fig1, fig3, fig5, fig7]
    export_dashboard(
        dashboard_figures,
        output_dir / "dashboard_grid.html",
        title="THERMALIQ Energy Analysis Dashboard - Grid Layout",
        layout="grid",
        config=config
    )

    export_dashboard(
        dashboard_figures,
        output_dir / "dashboard_tabs.html",
        title="THERMALIQ Energy Analysis Dashboard - Tabbed",
        layout="tabs",
        config=config
    )

    print(f"\n{'='*60}")
    print(f"Generated {10} individual visualizations")
    print(f"Generated 2 dashboards (grid + tabs)")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Generate all examples in 'examples_output' directory
    output_path = Path(__file__).parent / "examples_output"
    generate_all_examples(output_path)

    print("\nExample Usage:")
    print("-" * 60)
    print("""
from visualization import (
    SankeyEngine,
    WaterfallChart,
    EfficiencyTrends,
    LossBreakdown,
    export_to_html
)

# Generate Sankey diagram
engine = SankeyEngine()
diagram = engine.generate_from_efficiency_result(
    energy_inputs={"natural_gas": 5000},
    useful_outputs={"steam": 4200},
    losses={"flue_gas": 350, "radiation": 120}
)

# Export to HTML
export_to_html(
    diagram.to_plotly_json(),
    "energy_flow.html",
    title="My Energy Flow Analysis"
)
    """)
