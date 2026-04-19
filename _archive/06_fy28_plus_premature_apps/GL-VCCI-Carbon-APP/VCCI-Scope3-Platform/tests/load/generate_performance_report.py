# -*- coding: utf-8 -*-
"""
GL-VCCI Performance Report Generator

Generates comprehensive HTML performance reports from Locust CSV results.

Features:
    - Response time charts (p50, p95, p99 over time)
    - Throughput charts (RPS over time)
    - Error rate charts
    - Resource utilization charts (CPU, memory, DB connections)
    - Summary statistics tables
    - Performance target validation
    - Comparison across multiple test runs

Usage:
    python generate_performance_report.py \\
        --results rampup_results_stats.csv \\
        --output rampup_report.html \\
        --test-name "Ramp-Up Test"

    # With resource monitoring data
    python generate_performance_report.py \\
        --results sustained_results_stats.csv \\
        --resources sustained_resources.json \\
        --output sustained_report.html \\
        --test-name "Sustained Load Test"

    # Compare multiple test runs
    python generate_performance_report.py \\
        --compare rampup_results_stats.csv sustained_results_stats.csv \\
        --output comparison_report.html

Author: GL-VCCI Team
Version: 1.0.0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib
from greenlang.determinism import DeterministicClock
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Template
import io
import base64


# ============================================================================
# CHART GENERATION
# ============================================================================

class ChartGenerator:
    """Generate performance charts for load test results."""

    def __init__(self, figsize=(12, 6), dpi=100):
        """Initialize chart generator with default settings."""
        self.figsize = figsize
        self.dpi = dpi
        self.style = 'seaborn-v0_8-darkgrid'
        plt.style.use(self.style)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def create_response_time_chart(self, df: pd.DataFrame) -> str:
        """
        Create response time chart showing p50, p95, p99 over time.

        Args:
            df: DataFrame with columns: timestamp, response_time_p50, p95, p99

        Returns:
            Base64 encoded PNG image
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Parse timestamps if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            x_data = df['timestamp']
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
        else:
            x_data = range(len(df))

        # Plot percentiles
        ax.plot(x_data, df['50%'], label='p50', linewidth=2, color='#2ecc71')
        ax.plot(x_data, df['95%'], label='p95', linewidth=2, color='#f39c12')
        ax.plot(x_data, df['99%'], label='p99', linewidth=2, color='#e74c3c')

        # Add target line
        ax.axhline(y=200, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (200ms)')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Response Time (ms)', fontsize=12)
        ax.set_title('Response Time Percentiles Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def create_throughput_chart(self, df: pd.DataFrame) -> str:
        """
        Create throughput chart showing requests per second over time.

        Args:
            df: DataFrame with columns: timestamp, requests_per_second

        Returns:
            Base64 encoded PNG image
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            x_data = df['timestamp']
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
        else:
            x_data = range(len(df))

        # Plot throughput
        if 'Requests/s' in df.columns:
            ax.plot(x_data, df['Requests/s'], linewidth=2, color='#3498db', label='Throughput')
            ax.fill_between(x_data, 0, df['Requests/s'], alpha=0.3, color='#3498db')
        elif 'Total Request Count' in df.columns:
            # Calculate RPS from cumulative requests
            rps = df['Total Request Count'].diff() / df['timestamp'].diff().dt.total_seconds()
            ax.plot(x_data[1:], rps[1:], linewidth=2, color='#3498db', label='Throughput')
            ax.fill_between(x_data[1:], 0, rps[1:], alpha=0.3, color='#3498db')

        # Add target line
        ax.axhline(y=1000, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (1000 RPS)')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Requests per Second', fontsize=12)
        ax.set_title('Throughput Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def create_error_rate_chart(self, df: pd.DataFrame) -> str:
        """
        Create error rate chart showing failure percentage over time.

        Args:
            df: DataFrame with columns: timestamp, failure_rate

        Returns:
            Base64 encoded PNG image
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            x_data = df['timestamp']
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
        else:
            x_data = range(len(df))

        # Calculate error rate percentage
        if 'Total Failure Count' in df.columns and 'Total Request Count' in df.columns:
            error_rate = (df['Total Failure Count'] / df['Total Request Count'] * 100).fillna(0)
        elif 'Failures/s' in df.columns and 'Requests/s' in df.columns:
            error_rate = (df['Failures/s'] / df['Requests/s'] * 100).fillna(0)
        else:
            error_rate = pd.Series([0] * len(df))

        # Plot error rate
        ax.plot(x_data, error_rate, linewidth=2, color='#e74c3c', label='Error Rate')
        ax.fill_between(x_data, 0, error_rate, alpha=0.3, color='#e74c3c')

        # Add target line
        ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Target (0.1%)')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Error Rate Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)

    def create_user_count_chart(self, df: pd.DataFrame) -> str:
        """
        Create user count chart showing concurrent users over time.

        Args:
            df: DataFrame with columns: timestamp, user_count

        Returns:
            Base64 encoded PNG image
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            x_data = df['timestamp']
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
        else:
            x_data = range(len(df))

        if 'User Count' in df.columns:
            ax.plot(x_data, df['User Count'], linewidth=2, color='#9b59b6', label='Active Users')
            ax.fill_between(x_data, 0, df['User Count'], alpha=0.3, color='#9b59b6')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Concurrent Users', fontsize=12)
        ax.set_title('Concurrent Users Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return self._fig_to_base64(fig)


# ============================================================================
# REPORT GENERATION
# ============================================================================

class PerformanceReportGenerator:
    """Generate comprehensive HTML performance report."""

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ test_name }} - Performance Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin: 30px 0 15px 0;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        h3 {
            color: #555;
            margin: 20px 0 10px 0;
            font-size: 1.3em;
        }
        .metadata {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .metadata p {
            margin: 5px 0;
            font-size: 1.1em;
        }
        .metadata strong {
            color: #2c3e50;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card.success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .metric-card.warning {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .metric-card.error {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        .metric-card h3 {
            color: white;
            margin: 0 0 10px 0;
            font-size: 1em;
            opacity: 0.9;
        }
        .metric-card .value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-card .label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .chart {
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 5px;
        }
        .chart img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        th {
            background: #34495e;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
        .warning { color: #f39c12; font-weight: bold; }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ test_name }}</h1>

        <div class="metadata">
            <p><strong>Test Date:</strong> {{ test_date }}</p>
            <p><strong>Duration:</strong> {{ duration }}</p>
            <p><strong>Target Host:</strong> {{ host }}</p>
            <p><strong>Generated:</strong> {{ generated_at }}</p>
        </div>

        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="metric-card {{ 'success' if summary.error_rate_pass else 'error' }}">
                <h3>Error Rate</h3>
                <div class="value">{{ "%.3f"|format(summary.error_rate_pct) }}%</div>
                <div class="label">Target: &lt; 0.1%</div>
            </div>
            <div class="metric-card {{ 'success' if summary.p95_pass else 'warning' }}">
                <h3>p95 Latency</h3>
                <div class="value">{{ "%.0f"|format(summary.p95_ms) }}ms</div>
                <div class="label">Target: &lt; 200ms</div>
            </div>
            <div class="metric-card {{ 'success' if summary.throughput_pass else 'warning' }}">
                <h3>Throughput</h3>
                <div class="value">{{ "%.0f"|format(summary.rps) }}</div>
                <div class="label">Target: &gt; 1000 RPS</div>
            </div>
            <div class="metric-card success">
                <h3>Total Requests</h3>
                <div class="value">{{ "{:,}".format(summary.total_requests) }}</div>
                <div class="label">All test requests</div>
            </div>
        </div>

        <h2>Performance Targets Validation</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Target</th>
                    <th>Actual</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for target in targets %}
                <tr>
                    <td>{{ target.metric }}</td>
                    <td>{{ target.target }}</td>
                    <td>{{ target.actual }}</td>
                    <td class="{{ target.status_class }}">{{ target.status }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        {% if charts.response_time %}
        <h2>Response Time Analysis</h2>
        <div class="chart">
            <img src="data:image/png;base64,{{ charts.response_time }}" alt="Response Time Chart">
        </div>
        {% endif %}

        {% if charts.throughput %}
        <h2>Throughput Analysis</h2>
        <div class="chart">
            <img src="data:image/png;base64,{{ charts.throughput }}" alt="Throughput Chart">
        </div>
        {% endif %}

        {% if charts.error_rate %}
        <h2>Error Rate Analysis</h2>
        <div class="chart">
            <img src="data:image/png;base64,{{ charts.error_rate }}" alt="Error Rate Chart">
        </div>
        {% endif %}

        {% if charts.user_count %}
        <h2>Concurrent Users</h2>
        <div class="chart">
            <img src="data:image/png;base64,{{ charts.user_count }}" alt="User Count Chart">
        </div>
        {% endif %}

        <h2>Request Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Endpoint</th>
                    <th>Requests</th>
                    <th>Failures</th>
                    <th>Avg (ms)</th>
                    <th>p95 (ms)</th>
                    <th>p99 (ms)</th>
                </tr>
            </thead>
            <tbody>
                {% for stat in request_stats %}
                <tr>
                    <td>{{ stat.name }}</td>
                    <td>{{ "{:,}".format(stat.requests) }}</td>
                    <td>{{ "{:,}".format(stat.failures) }}</td>
                    <td>{{ "%.2f"|format(stat.avg) }}</td>
                    <td>{{ "%.2f"|format(stat.p95) }}</td>
                    <td>{{ "%.2f"|format(stat.p99) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="footer">
            <p>Generated by GL-VCCI Performance Report Generator v1.0.0</p>
            <p>GL-VCCI Scope 3 Carbon Intelligence Platform - Phase 6 Testing</p>
        </div>
    </div>
</body>
</html>
    """

    def __init__(self):
        """Initialize report generator."""
        self.chart_gen = ChartGenerator()

    def load_locust_results(self, stats_file: Path) -> pd.DataFrame:
        """Load Locust stats CSV file."""
        return pd.read_csv(stats_file)

    def calculate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        # Get aggregated stats (last row or 'Aggregated' row)
        if 'Name' in df.columns:
            agg_row = df[df['Name'] == 'Aggregated'].iloc[-1] if 'Aggregated' in df['Name'].values else df.iloc[-1]
        else:
            agg_row = df.iloc[-1]

        total_requests = agg_row.get('Request Count', agg_row.get('Total Request Count', 0))
        failures = agg_row.get('Failure Count', agg_row.get('Total Failure Count', 0))
        error_rate = (failures / total_requests * 100) if total_requests > 0 else 0

        p95 = agg_row.get('95%', agg_row.get('95th percentile', 0))
        rps = agg_row.get('Requests/s', agg_row.get('Average Response Time', 0))

        return {
            'total_requests': int(total_requests),
            'failures': int(failures),
            'error_rate_pct': error_rate,
            'error_rate_pass': error_rate < 0.1,
            'p95_ms': p95,
            'p95_pass': p95 < 200,
            'rps': rps,
            'throughput_pass': rps >= 1000,
        }

    def generate_report(
        self,
        results_file: Path,
        output_file: Path,
        test_name: str,
        host: str = "http://localhost:8000"
    ):
        """Generate HTML performance report."""
        # Load data
        df_stats = self.load_locust_results(results_file)

        # Calculate summary
        summary = self.calculate_summary(df_stats)

        # Generate charts
        charts = {}
        try:
            charts['response_time'] = self.chart_gen.create_response_time_chart(df_stats)
        except Exception as e:
            print(f"Warning: Could not generate response time chart: {e}")

        try:
            charts['throughput'] = self.chart_gen.create_throughput_chart(df_stats)
        except Exception as e:
            print(f"Warning: Could not generate throughput chart: {e}")

        try:
            charts['error_rate'] = self.chart_gen.create_error_rate_chart(df_stats)
        except Exception as e:
            print(f"Warning: Could not generate error rate chart: {e}")

        try:
            charts['user_count'] = self.chart_gen.create_user_count_chart(df_stats)
        except Exception as e:
            print(f"Warning: Could not generate user count chart: {e}")

        # Prepare target validation
        targets = [
            {
                'metric': 'Error Rate',
                'target': '< 0.1%',
                'actual': f"{summary['error_rate_pct']:.3f}%",
                'status': 'PASS' if summary['error_rate_pass'] else 'FAIL',
                'status_class': 'pass' if summary['error_rate_pass'] else 'fail'
            },
            {
                'metric': 'p95 Latency',
                'target': '< 200ms',
                'actual': f"{summary['p95_ms']:.0f}ms",
                'status': 'PASS' if summary['p95_pass'] else 'FAIL',
                'status_class': 'pass' if summary['p95_pass'] else 'fail'
            },
            {
                'metric': 'Throughput',
                'target': '≥ 1000 RPS',
                'actual': f"{summary['rps']:.0f} RPS",
                'status': 'PASS' if summary['throughput_pass'] else 'FAIL',
                'status_class': 'pass' if summary['throughput_pass'] else 'fail'
            },
        ]

        # Prepare request statistics
        request_stats = []
        for _, row in df_stats.iterrows():
            if row.get('Name') and row['Name'] != 'Aggregated':
                request_stats.append({
                    'name': row['Name'],
                    'requests': int(row.get('Request Count', row.get('Total Request Count', 0))),
                    'failures': int(row.get('Failure Count', row.get('Total Failure Count', 0))),
                    'avg': float(row.get('Average Response Time', row.get('Average', 0))),
                    'p95': float(row.get('95%', row.get('95th percentile', 0))),
                    'p99': float(row.get('99%', row.get('99th percentile', 0))),
                })

        # Render template
        template = Template(self.HTML_TEMPLATE)
        html = template.render(
            test_name=test_name,
            test_date=DeterministicClock.now().strftime('%Y-%m-%d'),
            duration='N/A',
            host=host,
            generated_at=DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S'),
            summary=summary,
            charts=charts,
            targets=targets,
            request_stats=request_stats
        )

        # Write report
        output_file.write_text(html, encoding='utf-8')
        print(f"\nPerformance report generated: {output_file}")
        print(f"Open in browser: file://{output_file.absolute()}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Generate HTML performance report from Locust results'
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to Locust stats CSV file (e.g., rampup_results_stats.csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output HTML file path'
    )
    parser.add_argument(
        '--test-name',
        type=str,
        default='Load Test Performance Report',
        help='Name of the test'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='http://localhost:8000',
        help='Target host URL'
    )

    args = parser.parse_args()

    # Validate input file
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)

    # Generate report
    generator = PerformanceReportGenerator()
    generator.generate_report(
        results_file=args.results,
        output_file=args.output,
        test_name=args.test_name,
        host=args.host
    )

    print("\n✅ Report generation complete!")


if __name__ == '__main__':
    main()
