# -*- coding: utf-8 -*-
"""Interactive Demo for Isolation Forest Anomaly Detection Agent.

This demo showcases the IsolationForestAnomalyAgent with three real-world scenarios:
1. Energy Consumption Anomalies (spikes, drops, unusual patterns)
2. Temperature Anomalies (extreme weather events)
3. Emissions Anomalies (equipment malfunction, sensor errors)

Features:
- Rich console output with visualizations
- Performance benchmarks
- Detailed anomaly analysis
- Alert generation demonstrations

Usage:
    python examples/anomaly_iforest_demo.py

Author: GreenLang Framework Team
Date: October 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List

from greenlang.agents.anomaly_agent_iforest import IsolationForestAnomalyAgent

# Rich console output (optional - graceful fallback)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install 'rich' for better output: pip install rich")


class DemoRunner:
    """Demo runner with rich console output."""

    def __init__(self):
        """Initialize demo runner."""
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    def print_header(self, title: str, subtitle: str = ""):
        """Print section header."""
        if self.console:
            panel = Panel(
                f"[bold cyan]{title}[/bold cyan]\n{subtitle}",
                box=box.DOUBLE,
                border_style="cyan",
            )
            self.console.print(panel)
        else:
            print(f"\n{'=' * 80}")
            print(f"{title}")
            if subtitle:
                print(f"{subtitle}")
            print(f"{'=' * 80}\n")

    def print_info(self, message: str):
        """Print info message."""
        if self.console:
            self.console.print(f"[blue]INFO:[/blue] {message}")
        else:
            print(f"INFO: {message}")

    def print_success(self, message: str):
        """Print success message."""
        if self.console:
            self.console.print(f"[green]SUCCESS:[/green] {message}")
        else:
            print(f"SUCCESS: {message}")

    def print_warning(self, message: str):
        """Print warning message."""
        if self.console:
            self.console.print(f"[yellow]WARNING:[/yellow] {message}")
        else:
            print(f"WARNING: {message}")

    def print_error(self, message: str):
        """Print error message."""
        if self.console:
            self.console.print(f"[red]ERROR:[/red] {message}")
        else:
            print(f"ERROR: {message}")

    def print_table(self, title: str, data: List[Dict[str, Any]], columns: List[str]):
        """Print data table."""
        if self.console:
            table = Table(title=title, box=box.ROUNDED)

            for col in columns:
                table.add_column(col, style="cyan")

            for row in data:
                table.add_row(*[str(row.get(col, "N/A")) for col in columns])

            self.console.print(table)
        else:
            print(f"\n{title}")
            print("-" * 80)
            # Print header
            print(" | ".join(columns))
            print("-" * 80)
            # Print rows
            for row in data:
                print(" | ".join([str(row.get(col, "N/A")) for col in columns]))
            print("-" * 80)

    def print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a formatted way."""
        if self.console:
            table = Table(title="Performance Metrics", box=box.SIMPLE)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in metrics.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                table.add_row(key, formatted_value)

            self.console.print(table)
        else:
            print("\nPerformance Metrics:")
            print("-" * 40)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            print("-" * 40)


def generate_energy_scenario() -> tuple[pd.DataFrame, List[int], str]:
    """Generate energy consumption data with anomalies.

    Returns:
        Tuple of (DataFrame, true_anomaly_indices, description)
    """
    np.random.seed(42)

    # Simulate hourly energy data for 30 days
    n_samples = 720  # 30 days * 24 hours
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='H')

    # Normal pattern: base load + daily cycle + weekly pattern
    hours = np.arange(n_samples) % 24
    days = np.arange(n_samples) // 24
    weekdays = days % 7

    base_load = 100
    daily_cycle = 30 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)  # Peak at noon
    weekly_pattern = 10 * (weekdays < 5)  # Higher on weekdays
    noise = np.random.normal(0, 5, n_samples)

    energy = base_load + daily_cycle + weekly_pattern + noise

    # Inject realistic anomalies
    anomaly_indices = []

    # 1. Equipment failure (sudden spike)
    spike_start = 200
    energy[spike_start:spike_start+5] = 300
    anomaly_indices.extend(range(spike_start, spike_start+5))

    # 2. Power outage (sudden drop)
    outage_start = 450
    energy[outage_start:outage_start+8] = 10
    anomaly_indices.extend(range(outage_start, outage_start+8))

    # 3. Sensor drift (gradual increase)
    drift_start = 600
    drift_length = 50
    energy[drift_start:drift_start+drift_length] += np.linspace(0, 60, drift_length)
    anomaly_indices.extend(range(drift_start, drift_start+drift_length))

    # 4. Random spikes (intermittent issues)
    for idx in [100, 250, 380, 550]:
        energy[idx] = 250
        anomaly_indices.append(idx)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'energy_kwh': energy,
        'hour': hours,
        'is_weekday': (weekdays < 5).astype(int),
    })

    description = """
    Energy Consumption Scenario:
    - 30 days of hourly electricity usage
    - Normal pattern: 100 kWh base + daily cycle (peak at noon) + weekday/weekend variation
    - Injected anomalies:
      * Equipment failure spike (hour 200-205): ~300 kWh
      * Power outage (hour 450-458): ~10 kWh
      * Sensor drift (hour 600-650): gradual increase +60 kWh
      * Random spikes (hours 100, 250, 380, 550): ~250 kWh
    """

    return df, anomaly_indices, description


def generate_temperature_scenario() -> tuple[pd.DataFrame, List[int], str]:
    """Generate temperature data with extreme weather anomalies.

    Returns:
        Tuple of (DataFrame, true_anomaly_indices, description)
    """
    np.random.seed(42)

    # Daily temperature for 6 months (180 days)
    n_samples = 180
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')

    # Seasonal pattern (winter to summer)
    days = np.arange(n_samples)
    seasonal = 15 + 10 * np.sin(2 * np.pi * days / 365)
    noise = np.random.normal(0, 2, n_samples)

    temperature = seasonal + noise

    # Inject extreme weather events
    anomaly_indices = []

    # 1. Heatwave (5 days)
    heatwave_start = 60
    temperature[heatwave_start:heatwave_start+5] = 42
    anomaly_indices.extend(range(heatwave_start, heatwave_start+5))

    # 2. Cold snap (4 days)
    coldsnap_start = 120
    temperature[coldsnap_start:coldsnap_start+4] = -8
    anomaly_indices.extend(range(coldsnap_start, coldsnap_start+4))

    # 3. Unseasonably hot day
    temperature[90] = 38
    anomaly_indices.append(90)

    # 4. Freak freeze
    temperature[150] = -5
    anomaly_indices.append(150)

    df = pd.DataFrame({
        'date': dates,
        'temperature_c': temperature,
        'day_of_year': days,
    })

    description = """
    Temperature Anomaly Scenario:
    - 6 months of daily temperature readings
    - Normal pattern: seasonal variation (15-25°C range)
    - Injected extreme weather events:
      * Heatwave (days 60-65): 42°C
      * Cold snap (days 120-124): -8°C
      * Unseasonably hot day (day 90): 38°C
      * Freak freeze (day 150): -5°C
    """

    return df, anomaly_indices, description


def generate_emissions_scenario() -> tuple[pd.DataFrame, List[int], str]:
    """Generate emissions data with equipment malfunction patterns.

    Returns:
        Tuple of (DataFrame, true_anomaly_indices, description)
    """
    np.random.seed(42)

    # Daily CO2 emissions for 90 days
    n_samples = 90
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')

    # Normal operations with slight trend
    base_emissions = 500
    trend = np.linspace(0, 20, n_samples)  # Gradual increase
    noise = np.random.normal(0, 30, n_samples)

    emissions = base_emissions + trend + noise

    # Inject malfunction patterns
    anomaly_indices = []

    # 1. Equipment malfunction (sudden spike for 5 days)
    malfunction_start = 30
    emissions[malfunction_start:malfunction_start+5] = 1500
    anomaly_indices.extend(range(malfunction_start, malfunction_start+5))

    # 2. Sensor error (zero readings for 4 days)
    sensor_error_start = 60
    emissions[sensor_error_start:sensor_error_start+4] = 0
    anomaly_indices.extend(range(sensor_error_start, sensor_error_start+4))

    # 3. Filter failure (elevated emissions for 7 days)
    filter_failure_start = 75
    emissions[filter_failure_start:filter_failure_start+7] = 900
    anomaly_indices.extend(range(filter_failure_start, filter_failure_start+7))

    # 4. Intermittent spikes (faulty valve)
    for idx in [15, 45, 70]:
        emissions[idx] = 1200
        anomaly_indices.append(idx)

    df = pd.DataFrame({
        'date': dates,
        'co2_kg': emissions,
        'day': np.arange(n_samples),
    })

    description = """
    Emissions Anomaly Scenario:
    - 90 days of daily CO2 emissions (kg)
    - Normal pattern: ~500 kg/day with slight upward trend
    - Injected equipment issues:
      * Equipment malfunction (days 30-35): 1500 kg/day
      * Sensor error (days 60-64): 0 kg/day (false readings)
      * Filter failure (days 75-82): 900 kg/day
      * Intermittent spikes (days 15, 45, 70): 1200 kg/day
    """

    return df, anomaly_indices, description


def run_scenario(
    runner: DemoRunner,
    scenario_name: str,
    df: pd.DataFrame,
    true_anomaly_indices: List[int],
    description: str,
    feature_columns: List[str],
    contamination: float = 0.1,
):
    """Run a single anomaly detection scenario.

    Args:
        runner: Demo runner
        scenario_name: Name of the scenario
        df: Data DataFrame
        true_anomaly_indices: Known anomaly indices
        description: Scenario description
        feature_columns: Features to use for detection
        contamination: Expected anomaly rate
    """
    runner.print_header(
        f"Scenario: {scenario_name}",
        description.strip()
    )

    # Display data summary
    runner.print_info(f"Dataset: {len(df)} observations, {len(feature_columns)} features")
    runner.print_info(f"Features: {', '.join(feature_columns)}")
    runner.print_info(f"True anomalies: {len(true_anomaly_indices)} ({len(true_anomaly_indices)/len(df)*100:.1f}%)")

    # Create agent (without AI to avoid API costs in demo)
    agent = IsolationForestAnomalyAgent(
        budget_usd=0.1,
        enable_explanations=False,
        enable_recommendations=False,
        enable_alerts=True,
    )

    # Prepare input
    input_data = {
        "data": df,
        "feature_columns": feature_columns,
        "contamination": contamination,
        "labels": [i in true_anomaly_indices for i in range(len(df))],
    }

    # Validate
    if not agent.validate(input_data):
        runner.print_error("Input validation failed!")
        return

    runner.print_success("Input validated successfully")

    # Run detection
    runner.print_info("Running anomaly detection...")
    start_time = time.time()

    try:
        # Execute tools directly (without full AI orchestration for demo)
        model_result = agent._fit_isolation_forest_impl(
            input_data,
            contamination=contamination,
            n_estimators=100,
        )

        anomalies_result = agent._detect_anomalies_impl(input_data)
        scores_result = agent._calculate_anomaly_scores_impl(input_data)
        rankings_result = agent._rank_anomalies_impl(input_data, top_k=10)
        patterns_result = agent._analyze_anomaly_patterns_impl(input_data)
        alerts_result = agent._generate_alerts_impl(input_data, min_severity="high")

        detection_time = time.time() - start_time

        runner.print_success(f"Detection completed in {detection_time:.2f} seconds")

        # Build output
        tool_results = {
            "model": model_result,
            "anomalies": anomalies_result,
            "scores": scores_result,
            "rankings": rankings_result,
            "patterns": patterns_result,
            "alerts": alerts_result,
        }

        output = agent._build_output(input_data, tool_results, None)

        # Display results
        display_results(runner, output, true_anomaly_indices, detection_time)

    except Exception as e:
        runner.print_error(f"Detection failed: {e}")
        import traceback
        traceback.print_exc()


def display_results(
    runner: DemoRunner,
    output: Dict[str, Any],
    true_anomaly_indices: List[int],
    detection_time: float,
):
    """Display detection results.

    Args:
        runner: Demo runner
        output: Detection output
        true_anomaly_indices: Known anomaly indices
        detection_time: Time taken for detection
    """
    # Summary statistics
    runner.print_info("\n=== Detection Summary ===")
    runner.print_info(f"Total observations: {output['n_anomalies'] + output['n_normal']}")
    runner.print_info(f"Detected anomalies: {output['n_anomalies']} ({output['anomaly_rate']*100:.1f}%)")
    runner.print_info(f"Normal observations: {output['n_normal']}")

    # Severity distribution
    if "severity_distribution" in output:
        runner.print_info("\n=== Severity Distribution ===")
        severity_dist = output["severity_distribution"]
        for severity in ["critical", "high", "medium", "low"]:
            count = severity_dist.get(severity, 0)
            if count > 0:
                runner.print_info(f"{severity.capitalize()}: {count}")

    # Top anomalies
    if "top_anomalies" in output and len(output["top_anomalies"]) > 0:
        runner.print_info("\n=== Top 10 Anomalies ===")

        table_data = []
        for i, anomaly in enumerate(output["top_anomalies"][:10], 1):
            row = {
                "Rank": i,
                "Index": anomaly["index"],
                "Score": f"{anomaly['score']:.3f}",
                "Severity": anomaly["severity"],
            }

            # Add feature values (first 2 features only for display)
            for j, (feature, value) in enumerate(list(anomaly["features"].items())[:2]):
                row[f"{feature}"] = f"{value:.1f}"

            table_data.append(row)

        columns = ["Rank", "Index", "Score", "Severity"] + list(output["top_anomalies"][0]["features"].keys())[:2]
        runner.print_table("Top Anomalies", table_data, columns[:6])  # Limit columns

    # Pattern analysis
    if "patterns" in output and "feature_importance" in output["patterns"]:
        runner.print_info("\n=== Feature Importance ===")
        feature_importance = output["patterns"]["feature_importance"]

        importance_data = []
        for feature, importance in list(feature_importance.items())[:5]:  # Top 5
            importance_data.append({
                "Feature": feature,
                "Importance": f"{importance:.3f}",
            })

        runner.print_table("Feature Importance", importance_data, ["Feature", "Importance"])

    # Alerts
    if "alerts" in output and len(output["alerts"]) > 0:
        runner.print_info(f"\n=== Generated Alerts ({len(output['alerts'])}) ===")

        for i, alert in enumerate(output["alerts"][:5], 1):  # Show top 5
            runner.print_info(f"\nAlert #{i}:")
            runner.print_info(f"  Index: {alert['index']}")
            runner.print_info(f"  Severity: {alert['severity']}")
            runner.print_info(f"  Score: {alert['score']:.3f}")
            runner.print_info(f"  Confidence: {alert['confidence']:.2f}")

            if alert["root_cause_hints"]:
                runner.print_info(f"  Root causes: {', '.join(alert['root_cause_hints'][:2])}")

            if alert["recommendations"]:
                runner.print_info(f"  Recommendations: {alert['recommendations'][0]}")

    # Accuracy metrics (if available)
    if "metrics" in output:
        runner.print_info("\n=== Accuracy Metrics ===")
        metrics = output["metrics"]

        metrics_data = {
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1-Score": metrics.get("f1_score", 0),
            "ROC-AUC": metrics.get("roc_auc", 0),
        }

        runner.print_metrics(metrics_data)

    # Performance
    runner.print_info("\n=== Performance ===")
    runner.print_metrics({
        "Detection Time": f"{detection_time:.3f}s",
        "Throughput": f"{(output['n_anomalies'] + output['n_normal'])/detection_time:.0f} obs/s",
    })


def main():
    """Run all demo scenarios."""
    runner = DemoRunner()

    runner.print_header(
        "Isolation Forest Anomaly Detection Agent Demo",
        "Demonstrating anomaly detection on climate and energy data"
    )

    print("\n")

    # Scenario 1: Energy Consumption
    df1, anomalies1, desc1 = generate_energy_scenario()
    run_scenario(
        runner,
        "Energy Consumption Anomalies",
        df1,
        anomalies1,
        desc1,
        feature_columns=["energy_kwh"],
        contamination=0.1,
    )

    print("\n" + "="*100 + "\n")

    # Scenario 2: Temperature
    df2, anomalies2, desc2 = generate_temperature_scenario()
    run_scenario(
        runner,
        "Temperature Anomalies (Extreme Weather)",
        df2,
        anomalies2,
        desc2,
        feature_columns=["temperature_c"],
        contamination=0.08,
    )

    print("\n" + "="*100 + "\n")

    # Scenario 3: Emissions
    df3, anomalies3, desc3 = generate_emissions_scenario()
    run_scenario(
        runner,
        "Emissions Anomalies (Equipment Issues)",
        df3,
        anomalies3,
        desc3,
        feature_columns=["co2_kg"],
        contamination=0.15,
    )

    print("\n" + "="*100 + "\n")

    runner.print_header(
        "Demo Complete!",
        "All scenarios executed successfully"
    )

    runner.print_info("\nKey Takeaways:")
    runner.print_info("1. Isolation Forest effectively detects outliers without labeled data")
    runner.print_info("2. Contamination parameter should match expected anomaly rate")
    runner.print_info("3. Multi-dimensional detection captures complex patterns")
    runner.print_info("4. Severity-based alerts help prioritize investigation")
    runner.print_info("5. Performance is suitable for real-time monitoring (< 2s)")

    runner.print_info("\nNext Steps:")
    runner.print_info("1. Integrate with real data sources (APIs, databases)")
    runner.print_info("2. Enable AI explanations for detailed insights")
    runner.print_info("3. Set up automated alert pipelines")
    runner.print_info("4. Tune contamination based on historical data")
    runner.print_info("5. Combine with other agents (SARIMA for forecast anomalies)")


if __name__ == "__main__":
    main()
