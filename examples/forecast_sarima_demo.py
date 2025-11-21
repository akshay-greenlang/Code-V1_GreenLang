# -*- coding: utf-8 -*-
"""Interactive SARIMA Forecasting Demo.

This demo showcases the SARIMA Forecast Agent with three real-world scenarios:
1. Energy Consumption Forecasting (monthly, seasonal peaks)
2. Temperature Prediction (daily, strong seasonality)
3. Emissions Trend Forecasting (monthly, declining trend with seasonality)

Each scenario demonstrates:
- Data preprocessing and cleaning
- Automatic seasonality detection
- Model fitting and parameter tuning
- Forecast generation with confidence intervals
- AI-generated insights and recommendations
- Performance benchmarking

Run:
    python examples/forecast_sarima_demo.py

Author: GreenLang Framework Team
Date: October 2025
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore')

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for better output...")
    os.system("pip install rich")
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

from greenlang.agents.forecast_agent_sarima import SARIMAForecastAgent


console = Console()


def print_header():
    """Print demo header."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]SARIMA Forecasting Agent Demo[/bold cyan]\n"
        "[dim]AI-Powered Time-Series Forecasting for Climate & Energy[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def generate_energy_consumption_data() -> pd.DataFrame:
    """Generate synthetic monthly energy consumption data.

    Pattern:
    - 3 years of history
    - Seasonal peaks in summer (cooling) and winter (heating)
    - Gradual upward trend
    - Random noise

    Returns:
        DataFrame with monthly energy consumption
    """
    np.random.seed(42)

    # 36 months of historical data
    dates = pd.date_range('2022-01-01', periods=36, freq='M')
    t = np.arange(36)

    # Components
    trend = 10000 + 200 * t  # Increasing consumption
    seasonal = 2000 * np.sin(2 * np.pi * t / 12)  # Annual cycle
    summer_peak = 1500 * np.maximum(0, np.sin(2 * np.pi * (t - 5) / 12))  # Summer AC
    winter_peak = 1000 * np.maximum(0, np.sin(2 * np.pi * (t - 11) / 12))  # Winter heating
    noise = np.random.normal(0, 500, 36)

    energy_kwh = trend + seasonal + summer_peak + winter_peak + noise

    # Add some missing values (realistic)
    energy_kwh[10] = np.nan
    energy_kwh[25] = np.nan

    # Add temperature as exogenous variable
    temperature = 65 + 20 * np.sin(2 * np.pi * (t - 3) / 12) + np.random.normal(0, 3, 36)

    df = pd.DataFrame({
        'energy_kwh': energy_kwh,
        'temperature_f': temperature,
    }, index=dates)

    return df


def generate_temperature_data() -> pd.DataFrame:
    """Generate synthetic daily temperature data.

    Pattern:
    - 2 years of history
    - Strong annual seasonality
    - Weekly micro-patterns
    - Weather noise

    Returns:
        DataFrame with daily temperature
    """
    np.random.seed(43)

    # 730 days (2 years)
    dates = pd.date_range('2023-01-01', periods=730, freq='D')
    t = np.arange(730)

    # Components
    annual_cycle = 60 + 25 * np.sin(2 * np.pi * (t - 80) / 365)  # Annual temperature cycle
    weekly_variation = 3 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
    noise = np.random.normal(0, 5, 730)

    temperature_c = annual_cycle + weekly_variation + noise

    df = pd.DataFrame({
        'temperature_c': temperature_c,
    }, index=dates)

    return df


def generate_emissions_data() -> pd.DataFrame:
    """Generate synthetic monthly emissions data.

    Pattern:
    - 5 years of history
    - Declining trend (decarbonization)
    - Seasonal variation (heating/cooling)
    - Occasional spikes (events)

    Returns:
        DataFrame with monthly CO2 emissions
    """
    np.random.seed(44)

    # 60 months (5 years)
    dates = pd.date_range('2020-01-01', periods=60, freq='M')
    t = np.arange(60)

    # Components
    declining_trend = 50000 - 400 * t  # Decarbonization trend
    seasonal = 8000 * np.sin(2 * np.pi * t / 12)  # Seasonal heating/cooling
    noise = np.random.normal(0, 2000, 60)

    emissions_kg = declining_trend + seasonal + noise

    # Add some outliers (operational events)
    emissions_kg[15] *= 1.3  # Spike
    emissions_kg[42] *= 0.7  # Reduction

    df = pd.DataFrame({
        'co2_emissions_kg': emissions_kg,
    }, index=dates)

    return df


def display_data_summary(df: pd.DataFrame, title: str, target_col: str):
    """Display data summary table."""
    table = Table(title=f"{title} - Data Summary", box=box.ROUNDED)

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    series = df[target_col]

    table.add_row("Total Observations", str(len(df)))
    table.add_row("Date Range", f"{df.index[0].date()} to {df.index[-1].date()}")
    table.add_row("Frequency", str(df.index.freq) if df.index.freq else "Irregular")
    table.add_row("Missing Values", str(series.isnull().sum()))
    table.add_row("Mean", f"{series.mean():.2f}")
    table.add_row("Std Dev", f"{series.std():.2f}")
    table.add_row("Min", f"{series.min():.2f}")
    table.add_row("Max", f"{series.max():.2f}")

    console.print(table)
    console.print()


def display_forecast_results(
    result: Dict[str, Any],
    scenario_name: str,
    target_col: str,
    unit: str,
):
    """Display forecast results in formatted tables."""
    # Forecast table
    table = Table(
        title=f"{scenario_name} - Forecast Results",
        box=box.ROUNDED,
        show_header=True,
    )

    table.add_column("Period", style="cyan", no_wrap=True)
    table.add_column("Date", style="blue")
    table.add_column(f"Forecast ({unit})", style="green", justify="right")
    table.add_column("Lower 95%", style="yellow", justify="right")
    table.add_column("Upper 95%", style="yellow", justify="right")

    forecast = result["forecast"]
    lower = result["lower_bound"]
    upper = result["upper_bound"]
    dates = result.get("forecast_dates", [])

    # Show first 12 periods
    display_count = min(12, len(forecast))

    for i in range(display_count):
        date_str = pd.Timestamp(dates[i]).strftime("%Y-%m-%d") if dates else f"T+{i+1}"

        table.add_row(
            f"T+{i+1}",
            date_str,
            f"{forecast[i]:,.1f}",
            f"{lower[i]:,.1f}",
            f"{upper[i]:,.1f}",
        )

    console.print(table)
    console.print()

    # Model performance table
    if "metrics" in result:
        metrics = result["metrics"]

        table2 = Table(title="Model Performance Metrics", box=box.SIMPLE)
        table2.add_column("Metric", style="cyan")
        table2.add_column("Value", style="green", justify="right")

        table2.add_row("RMSE", f"{metrics.get('rmse', 0):.2f}")
        table2.add_row("MAE", f"{metrics.get('mae', 0):.2f}")
        table2.add_row("MAPE", f"{metrics.get('mape', 0):.2f}%")
        table2.add_row("Training Size", str(metrics.get('train_size', 0)))
        table2.add_row("Validation Size", str(metrics.get('test_size', 0)))

        console.print(table2)
        console.print()

    # Model parameters
    if "model_params" in result:
        params = result["model_params"]

        table3 = Table(title="SARIMA Model Parameters", box=box.SIMPLE)
        table3.add_column("Parameter", style="cyan")
        table3.add_column("Value", style="green")

        order = params.get("order", (0, 0, 0))
        seasonal_order = params.get("seasonal_order", (0, 0, 0, 0))

        table3.add_row("Order (p, d, q)", str(order))
        table3.add_row("Seasonal Order (P, D, Q, s)", str(seasonal_order))
        table3.add_row("AIC", f"{params.get('aic', 0):.2f}")
        table3.add_row("BIC", f"{params.get('bic', 0):.2f}")
        table3.add_row("Auto-tuned", "Yes" if params.get('auto_tuned') else "No")

        console.print(table3)
        console.print()

    # AI Explanation
    if "explanation" in result:
        console.print(Panel(
            result["explanation"],
            title="[bold]AI Analysis[/bold]",
            border_style="magenta",
            padding=(1, 2),
        ))
        console.print()


def run_scenario_1():
    """Scenario 1: Energy Consumption Forecasting."""
    console.print("[bold cyan]Scenario 1: Energy Consumption Forecasting[/bold cyan]")
    console.print("[dim]Monthly electricity usage with seasonal peaks[/dim]\n")

    # Generate data
    console.print("[yellow]Generating synthetic energy consumption data...[/yellow]")
    df = generate_energy_consumption_data()

    display_data_summary(df, "Energy Consumption", "energy_kwh")

    # Create agent
    console.print("[yellow]Initializing SARIMA Forecast Agent...[/yellow]")
    agent = SARIMAForecastAgent(
        budget_usd=1.0,
        enable_explanations=True,
        enable_recommendations=True,
        enable_auto_tune=True,
    )

    # Run forecast
    console.print("[yellow]Running forecast analysis...[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Forecasting...", total=None)

        start_time = time.time()

        input_data = {
            "data": df,
            "target_column": "energy_kwh",
            "forecast_horizon": 12,
            "seasonal_period": 12,
            "confidence_level": 0.95,
        }

        result = agent.run(input_data)
        duration = time.time() - start_time

        progress.update(task, completed=True)

    if result.success:
        console.print(f"[green]✓ Forecast completed in {duration:.2f}s[/green]\n")
        display_forecast_results(
            result.data,
            "Energy Consumption",
            "energy_kwh",
            "kWh",
        )

        # Performance summary
        perf = agent.get_performance_summary()
        console.print(f"[dim]AI Cost: ${perf['ai_metrics']['total_cost_usd']:.4f} | "
                     f"Tool Calls: {perf['ai_metrics']['tool_call_count']}[/dim]\n")
    else:
        console.print(f"[red]✗ Forecast failed: {result.error}[/red]\n")

    console.print("[bold]" + "=" * 80 + "[/bold]\n")


def run_scenario_2():
    """Scenario 2: Temperature Prediction."""
    console.print("[bold cyan]Scenario 2: Temperature Prediction[/bold cyan]")
    console.print("[dim]Daily temperature forecasting with strong seasonality[/dim]\n")

    # Generate data
    console.print("[yellow]Generating synthetic temperature data...[/yellow]")
    df = generate_temperature_data()

    display_data_summary(df, "Temperature", "temperature_c")

    # Create agent
    console.print("[yellow]Initializing SARIMA Forecast Agent...[/yellow]")
    agent = SARIMAForecastAgent(
        budget_usd=1.0,
        enable_explanations=True,
        enable_recommendations=True,
        enable_auto_tune=True,
    )

    # Run forecast
    console.print("[yellow]Running forecast analysis...[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Forecasting...", total=None)

        start_time = time.time()

        input_data = {
            "data": df,
            "target_column": "temperature_c",
            "forecast_horizon": 90,  # 90 days ahead
            "seasonal_period": 7,  # Weekly seasonality
            "confidence_level": 0.95,
        }

        result = agent.run(input_data)
        duration = time.time() - start_time

        progress.update(task, completed=True)

    if result.success:
        console.print(f"[green]✓ Forecast completed in {duration:.2f}s[/green]\n")
        display_forecast_results(
            result.data,
            "Temperature",
            "temperature_c",
            "°C",
        )

        # Performance summary
        perf = agent.get_performance_summary()
        console.print(f"[dim]AI Cost: ${perf['ai_metrics']['total_cost_usd']:.4f} | "
                     f"Tool Calls: {perf['ai_metrics']['tool_call_count']}[/dim]\n")
    else:
        console.print(f"[red]✗ Forecast failed: {result.error}[/red]\n")

    console.print("[bold]" + "=" * 80 + "[/bold]\n")


def run_scenario_3():
    """Scenario 3: Emissions Trend Forecasting."""
    console.print("[bold cyan]Scenario 3: Emissions Trend Forecasting[/bold cyan]")
    console.print("[dim]Monthly CO2 emissions with declining trend and seasonality[/dim]\n")

    # Generate data
    console.print("[yellow]Generating synthetic emissions data...[/yellow]")
    df = generate_emissions_data()

    display_data_summary(df, "CO2 Emissions", "co2_emissions_kg")

    # Create agent
    console.print("[yellow]Initializing SARIMA Forecast Agent...[/yellow]")
    agent = SARIMAForecastAgent(
        budget_usd=1.0,
        enable_explanations=True,
        enable_recommendations=True,
        enable_auto_tune=True,
    )

    # Run forecast
    console.print("[yellow]Running forecast analysis...[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Forecasting...", total=None)

        start_time = time.time()

        input_data = {
            "data": df,
            "target_column": "co2_emissions_kg",
            "forecast_horizon": 24,  # 24 months ahead
            "seasonal_period": 12,
            "confidence_level": 0.95,
        }

        result = agent.run(input_data)
        duration = time.time() - start_time

        progress.update(task, completed=True)

    if result.success:
        console.print(f"[green]✓ Forecast completed in {duration:.2f}s[/green]\n")
        display_forecast_results(
            result.data,
            "CO2 Emissions",
            "co2_emissions_kg",
            "kg",
        )

        # Performance summary
        perf = agent.get_performance_summary()
        console.print(f"[dim]AI Cost: ${perf['ai_metrics']['total_cost_usd']:.4f} | "
                     f"Tool Calls: {perf['ai_metrics']['tool_call_count']}[/dim]\n")
    else:
        console.print(f"[red]✗ Forecast failed: {result.error}[/red]\n")

    console.print("[bold]" + "=" * 80 + "[/bold]\n")


def run_benchmark():
    """Run performance benchmarks."""
    console.print("[bold cyan]Performance Benchmarks[/bold cyan]\n")

    # Generate test data
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    t = np.arange(100)
    values = 1000 + 50 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 10, 100)
    df = pd.DataFrame({'value': values}, index=dates)

    agent = SARIMAForecastAgent(
        budget_usd=1.0,
        enable_explanations=False,  # Disable for pure performance
        enable_recommendations=False,
    )

    # Benchmark different configurations
    configs = [
        {"horizon": 6, "auto_tune": False, "desc": "6-period, manual params"},
        {"horizon": 12, "auto_tune": False, "desc": "12-period, manual params"},
        {"horizon": 12, "auto_tune": True, "desc": "12-period, auto-tuned"},
        {"horizon": 24, "auto_tune": True, "desc": "24-period, auto-tuned"},
    ]

    table = Table(title="Performance Benchmarks", box=box.ROUNDED)
    table.add_column("Configuration", style="cyan")
    table.add_column("Time (s)", style="green", justify="right")
    table.add_column("Tool Calls", style="yellow", justify="right")

    for config in configs:
        # Reset agent state
        agent._tool_call_count = 0

        start = time.time()

        input_data = {
            "data": df,
            "target_column": "value",
            "forecast_horizon": config["horizon"],
            "seasonal_period": 12,
            "auto_tune": config["auto_tune"],
        }

        result = agent.run(input_data)
        duration = time.time() - start

        if result.success:
            table.add_row(
                config["desc"],
                f"{duration:.3f}",
                str(agent._tool_call_count),
            )

    console.print(table)
    console.print()


def main():
    """Run the complete demo."""
    print_header()

    # Menu
    console.print("[bold]Choose a demo scenario:[/bold]")
    console.print("  [cyan]1[/cyan] - Energy Consumption Forecasting")
    console.print("  [cyan]2[/cyan] - Temperature Prediction")
    console.print("  [cyan]3[/cyan] - Emissions Trend Forecasting")
    console.print("  [cyan]4[/cyan] - Run All Scenarios")
    console.print("  [cyan]5[/cyan] - Performance Benchmarks")
    console.print("  [cyan]0[/cyan] - Exit\n")

    try:
        choice = console.input("[bold]Enter choice (0-5): [/bold]")

        console.print()

        if choice == "1":
            run_scenario_1()
        elif choice == "2":
            run_scenario_2()
        elif choice == "3":
            run_scenario_3()
        elif choice == "4":
            run_scenario_1()
            run_scenario_2()
            run_scenario_3()
            run_benchmark()
        elif choice == "5":
            run_benchmark()
        elif choice == "0":
            console.print("[yellow]Exiting demo.[/yellow]")
            return
        else:
            console.print("[red]Invalid choice. Please run again.[/red]")
            return

        # Final summary
        console.print(Panel.fit(
            "[bold green]Demo Complete![/bold green]\n\n"
            "Key Takeaways:\n"
            "• SARIMA handles seasonal patterns automatically\n"
            "• Confidence intervals quantify uncertainty\n"
            "• Auto-tuning finds optimal parameters\n"
            "• AI explains patterns and provides insights\n"
            "• Production-ready with full provenance",
            title="Summary",
            border_style="green",
            padding=(1, 2),
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
