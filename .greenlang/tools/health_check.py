#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infrastructure Health Checker

Automated health check for codebases.
Identifies infrastructure usage, anti-patterns, and calculates IUM score.
"""

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
from greenlang.determinism import DeterministicClock


@dataclass
class HealthMetrics:
    """Health check metrics."""
    total_files: int = 0
    python_files: int = 0
    lines_of_code: int = 0

    # Infrastructure usage
    using_chat_session: int = 0
    using_base_agent: int = 0
    using_cache_manager: int = 0
    using_validation: int = 0
    using_logger: int = 0
    using_config_manager: int = 0

    # Anti-patterns
    direct_openai_calls: int = 0
    direct_anthropic_calls: int = 0
    custom_caching: int = 0
    print_statements: int = 0
    raw_requests: int = 0
    manual_env_loading: int = 0

    # Code quality
    has_tests: bool = False
    test_coverage: float = 0.0
    has_documentation: bool = False

    # IUM Score components
    infrastructure_adoption_score: float = 0.0
    code_quality_score: float = 0.0
    anti_pattern_penalty: float = 0.0
    ium_score: float = 0.0


@dataclass
class IssueReport:
    """Issue found during health check."""
    file_path: str
    line_number: int
    severity: str  # "critical", "warning", "info"
    category: str
    message: str
    suggestion: str


@dataclass
class HealthReport:
    """Complete health check report."""
    timestamp: str
    directory: str
    metrics: HealthMetrics
    issues: List[IssueReport] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    file_details: Dict[str, Any] = field(default_factory=dict)


class CodeScanner:
    """Scan code for infrastructure usage and issues."""

    def __init__(self):
        self.metrics = HealthMetrics()
        self.issues: List[IssueReport] = []
        self.file_details: Dict[str, Any] = {}

    def scan_file(self, file_path: str) -> Dict[str, Any]:
        """Scan a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            details = {
                'path': file_path,
                'lines': len(content.split('\n')),
                'infrastructure_used': [],
                'anti_patterns': [],
                'issues': []
            }

            self.metrics.lines_of_code += details['lines']

            # Detect infrastructure usage
            if 'from shared.infrastructure.llm import ChatSession' in content or 'ChatSession' in content:
                self.metrics.using_chat_session += 1
                details['infrastructure_used'].append('ChatSession')

            if 'from shared.infrastructure.agents import BaseAgent' in content or 'BaseAgent' in content:
                self.metrics.using_base_agent += 1
                details['infrastructure_used'].append('BaseAgent')

            if 'from shared.infrastructure.cache import CacheManager' in content or 'CacheManager' in content:
                self.metrics.using_cache_manager += 1
                details['infrastructure_used'].append('CacheManager')

            if 'from shared.infrastructure.validation import' in content:
                self.metrics.using_validation += 1
                details['infrastructure_used'].append('ValidationFramework')

            if 'from shared.infrastructure.logging import Logger' in content:
                self.metrics.using_logger += 1
                details['infrastructure_used'].append('Logger')

            if 'from shared.infrastructure.config import ConfigManager' in content:
                self.metrics.using_config_manager += 1
                details['infrastructure_used'].append('ConfigManager')

            # Detect anti-patterns
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Direct OpenAI
                if re.search(r'openai\.(ChatCompletion|Completion)', line):
                    self.metrics.direct_openai_calls += 1
                    details['anti_patterns'].append('direct_openai')
                    self.issues.append(IssueReport(
                        file_path=file_path,
                        line_number=i,
                        severity="warning",
                        category="LLM",
                        message="Direct OpenAI API usage detected",
                        suggestion="Use ChatSession from shared.infrastructure.llm"
                    ))

                # Direct Anthropic
                if 'anthropic.' in line or 'from anthropic import' in line:
                    self.metrics.direct_anthropic_calls += 1
                    details['anti_patterns'].append('direct_anthropic')
                    self.issues.append(IssueReport(
                        file_path=file_path,
                        line_number=i,
                        severity="warning",
                        category="LLM",
                        message="Direct Anthropic API usage detected",
                        suggestion="Use ChatSession from shared.infrastructure.llm"
                    ))

                # Custom caching
                if re.search(r'cache\s*=\s*\{\}|@lru_cache|redis\.Redis\(', line):
                    self.metrics.custom_caching += 1
                    details['anti_patterns'].append('custom_caching')
                    self.issues.append(IssueReport(
                        file_path=file_path,
                        line_number=i,
                        severity="info",
                        category="Caching",
                        message="Custom caching implementation",
                        suggestion="Use CacheManager from shared.infrastructure.cache"
                    ))

                # Print statements
                if re.search(r'\bprint\s*\(', line):
                    self.metrics.print_statements += 1
                    details['anti_patterns'].append('print_statement')

                # Raw requests
                if re.search(r'requests\.(get|post|put|delete)', line):
                    self.metrics.raw_requests += 1
                    details['anti_patterns'].append('raw_requests')

                # Manual env loading
                if re.search(r'os\.getenv\(|os\.environ\[', line):
                    self.metrics.manual_env_loading += 1
                    details['anti_patterns'].append('manual_env')

            return details

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            return {}

    def scan_directory(self, directory: str) -> HealthMetrics:
        """Scan entire directory."""
        print(f"Scanning {directory}...")

        for root, dirs, files in os.walk(directory):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in [
                '.git', '__pycache__', 'venv', 'node_modules', '.venv',
                'dist', 'build', '.tox', 'htmlcov'
            ]]

            for file in files:
                if file.endswith('.py'):
                    self.metrics.python_files += 1
                    file_path = os.path.join(root, file)
                    details = self.scan_file(file_path)
                    if details:
                        self.file_details[file_path] = details

                self.metrics.total_files += 1

        # Check for tests
        test_files = [f for f in self.file_details.keys() if 'test' in f.lower()]
        self.metrics.has_tests = len(test_files) > 0

        # Check for documentation
        for root, dirs, files in os.walk(directory):
            if any(f.lower() in ['readme.md', 'readme.rst', 'readme.txt'] for f in files):
                self.metrics.has_documentation = True
                break

        return self.metrics


class IUMCalculator:
    """Calculate Infrastructure Usage Maturity (IUM) score."""

    @staticmethod
    def calculate(metrics: HealthMetrics) -> float:
        """
        Calculate IUM score (0-100).

        Components:
        - Infrastructure Adoption (50%): Usage of infrastructure components
        - Code Quality (30%): Tests, documentation, best practices
        - Anti-pattern Penalty (20%): Deductions for anti-patterns
        """

        if metrics.python_files == 0:
            return 0.0

        # 1. Infrastructure Adoption Score (0-50)
        infrastructure_components = [
            metrics.using_chat_session,
            metrics.using_base_agent,
            metrics.using_cache_manager,
            metrics.using_validation,
            metrics.using_logger,
            metrics.using_config_manager,
        ]

        # Normalize by number of Python files
        adoption_rate = sum(infrastructure_components) / (metrics.python_files * 6)  # 6 components
        adoption_score = min(50, adoption_rate * 100)

        # 2. Code Quality Score (0-30)
        quality_score = 0

        if metrics.has_tests:
            quality_score += 15

        if metrics.has_documentation:
            quality_score += 10

        # Bonus for using Logger instead of print
        if metrics.using_logger > 0 and metrics.print_statements == 0:
            quality_score += 5

        # 3. Anti-pattern Penalty (0-20 deduction)
        anti_pattern_count = (
            metrics.direct_openai_calls +
            metrics.direct_anthropic_calls +
            metrics.custom_caching +
            metrics.raw_requests +
            (min(metrics.print_statements, 10))  # Cap at 10
        )

        penalty = min(20, anti_pattern_count * 2)

        # Total IUM Score
        ium_score = adoption_score + quality_score - penalty
        ium_score = max(0, min(100, ium_score))

        # Store components
        metrics.infrastructure_adoption_score = adoption_score
        metrics.code_quality_score = quality_score
        metrics.anti_pattern_penalty = penalty
        metrics.ium_score = ium_score

        return ium_score


class ReportGenerator:
    """Generate health check reports."""

    @staticmethod
    def generate_recommendations(metrics: HealthMetrics) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Infrastructure adoption
        if metrics.using_chat_session == 0 and (metrics.direct_openai_calls > 0 or metrics.direct_anthropic_calls > 0):
            recommendations.append(
                "Migrate LLM calls to ChatSession for provider abstraction and better monitoring"
            )

        if metrics.using_base_agent == 0:
            recommendations.append(
                "Consider using BaseAgent for standardized agent interface and built-in features"
            )

        if metrics.using_cache_manager == 0 and metrics.custom_caching > 0:
            recommendations.append(
                "Replace custom caching with CacheManager for distributed caching and TTL management"
            )

        if metrics.using_logger == 0 and metrics.print_statements > 10:
            recommendations.append(
                "Replace print() statements with structured logging using Logger"
            )

        if metrics.using_config_manager == 0 and metrics.manual_env_loading > 5:
            recommendations.append(
                "Use ConfigManager for centralized configuration management"
            )

        # Code quality
        if not metrics.has_tests:
            recommendations.append(
                "Add unit tests to improve code reliability and enable CI/CD"
            )

        if not metrics.has_documentation:
            recommendations.append(
                "Add README.md with project overview, setup instructions, and usage examples"
            )

        # Anti-patterns
        if metrics.direct_openai_calls > 0:
            recommendations.append(
                f"Found {metrics.direct_openai_calls} direct OpenAI API calls - migrate to ChatSession"
            )

        if metrics.raw_requests > 5:
            recommendations.append(
                f"Found {metrics.raw_requests} raw requests calls - use APIClient for retry and error handling"
            )

        # IUM Score improvement
        if metrics.ium_score < 50:
            recommendations.append(
                "IUM Score is below 50 - focus on adopting core infrastructure components"
            )
        elif metrics.ium_score < 75:
            recommendations.append(
                "IUM Score is good - consider adding tests and eliminating remaining anti-patterns"
            )
        else:
            recommendations.append(
                "Excellent IUM Score - continue monitoring and maintain best practices"
            )

        return recommendations

    @staticmethod
    def generate_text_report(report: HealthReport) -> str:
        """Generate text report."""
        m = report.metrics

        output = []
        output.append("=" * 80)
        output.append("INFRASTRUCTURE HEALTH CHECK REPORT")
        output.append("=" * 80)
        output.append(f"\nDirectory: {report.directory}")
        output.append(f"Timestamp: {report.timestamp}")
        output.append(f"\n{'=' * 80}\n")

        # IUM Score
        output.append("INFRASTRUCTURE USAGE MATURITY (IUM) SCORE")
        output.append("-" * 80)
        output.append(f"\nOverall IUM Score: {m.ium_score:.1f}/100")

        # Score bar
        bar_length = 50
        filled = int((m.ium_score / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        output.append(f"[{bar}]")

        # Rating
        if m.ium_score >= 90:
            rating = "EXCELLENT"
        elif m.ium_score >= 75:
            rating = "GOOD"
        elif m.ium_score >= 50:
            rating = "FAIR"
        elif m.ium_score >= 25:
            rating = "POOR"
        else:
            rating = "CRITICAL"

        output.append(f"Rating: {rating}\n")

        # Score breakdown
        output.append("Score Breakdown:")
        output.append(f"  Infrastructure Adoption: {m.infrastructure_adoption_score:.1f}/50")
        output.append(f"  Code Quality:           {m.code_quality_score:.1f}/30")
        output.append(f"  Anti-pattern Penalty:   -{m.anti_pattern_penalty:.1f}/20")

        # Metrics
        output.append(f"\n{'=' * 80}\n")
        output.append("METRICS")
        output.append("-" * 80)
        output.append(f"\nCode Statistics:")
        output.append(f"  Total Files:       {m.total_files}")
        output.append(f"  Python Files:      {m.python_files}")
        output.append(f"  Lines of Code:     {m.lines_of_code:,}")

        output.append(f"\nInfrastructure Usage:")
        output.append(f"  ChatSession:       {m.using_chat_session} files")
        output.append(f"  BaseAgent:         {m.using_base_agent} files")
        output.append(f"  CacheManager:      {m.using_cache_manager} files")
        output.append(f"  ValidationFramework: {m.using_validation} files")
        output.append(f"  Logger:            {m.using_logger} files")
        output.append(f"  ConfigManager:     {m.using_config_manager} files")

        output.append(f"\nAnti-patterns Detected:")
        output.append(f"  Direct OpenAI calls:   {m.direct_openai_calls}")
        output.append(f"  Direct Anthropic calls: {m.direct_anthropic_calls}")
        output.append(f"  Custom caching:        {m.custom_caching}")
        output.append(f"  Print statements:      {m.print_statements}")
        output.append(f"  Raw requests:          {m.raw_requests}")
        output.append(f"  Manual env loading:    {m.manual_env_loading}")

        output.append(f"\nCode Quality:")
        output.append(f"  Has Tests:         {'✓' if m.has_tests else '✗'}")
        output.append(f"  Has Documentation: {'✓' if m.has_documentation else '✗'}")

        # Issues
        if report.issues:
            output.append(f"\n{'=' * 80}\n")
            output.append(f"ISSUES FOUND ({len(report.issues)})")
            output.append("-" * 80)

            # Group by severity
            by_severity = defaultdict(list)
            for issue in report.issues[:20]:  # Limit to first 20
                by_severity[issue.severity].append(issue)

            for severity in ['critical', 'warning', 'info']:
                issues = by_severity.get(severity, [])
                if issues:
                    output.append(f"\n{severity.upper()} ({len(issues)}):")
                    for issue in issues[:10]:  # Limit to 10 per severity
                        output.append(f"\n  {issue.file_path}:{issue.line_number}")
                        output.append(f"  {issue.message}")
                        output.append(f"  → {issue.suggestion}")

        # Recommendations
        output.append(f"\n{'=' * 80}\n")
        output.append("RECOMMENDATIONS")
        output.append("-" * 80)

        for i, rec in enumerate(report.recommendations, 1):
            output.append(f"\n{i}. {rec}")

        output.append(f"\n{'=' * 80}\n")

        return "\n".join(output)

    @staticmethod
    def generate_json_report(report: HealthReport) -> str:
        """Generate JSON report."""
        data = {
            "timestamp": report.timestamp,
            "directory": report.directory,
            "metrics": asdict(report.metrics),
            "issues": [asdict(i) for i in report.issues],
            "recommendations": report.recommendations,
            "file_count": len(report.file_details)
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def generate_html_report(report: HealthReport) -> str:
        """Generate HTML report."""
        m = report.metrics

        # IUM score color
        if m.ium_score >= 75:
            score_color = "#27ae60"
        elif m.ium_score >= 50:
            score_color = "#f39c12"
        else:
            score_color = "#e74c3c"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Infrastructure Health Check Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .score-container {{ text-align: center; margin: 30px 0; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
        .score {{ font-size: 72px; font-weight: bold; margin: 20px 0; }}
        .score-label {{ font-size: 24px; opacity: 0.9; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; font-size: 16px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
        .issue {{ border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; background: #fff5f5; }}
        .issue.warning {{ border-left-color: #f39c12; background: #fffbf0; }}
        .issue.info {{ border-left-color: #3498db; background: #f0f8ff; }}
        .recommendation {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        .progress-bar {{ width: 100%; height: 30px; background: #ecf0f1; border-radius: 15px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; background: {score_color}; transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Infrastructure Health Check Report</h1>
        <p><strong>Directory:</strong> {report.directory}</p>
        <p><strong>Generated:</strong> {report.timestamp}</p>

        <div class="score-container">
            <div class="score-label">Infrastructure Usage Maturity (IUM) Score</div>
            <div class="score" style="color: {score_color}">{m.ium_score:.1f}</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {m.ium_score}%; background: {score_color}"></div>
            </div>
        </div>

        <h2>Score Breakdown</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Infrastructure Adoption</h3>
                <div class="metric-value">{m.infrastructure_adoption_score:.1f}</div>
                <div class="metric-label">out of 50</div>
            </div>
            <div class="metric-card">
                <h3>Code Quality</h3>
                <div class="metric-value">{m.code_quality_score:.1f}</div>
                <div class="metric-label">out of 30</div>
            </div>
            <div class="metric-card">
                <h3>Anti-pattern Penalty</h3>
                <div class="metric-value">-{m.anti_pattern_penalty:.1f}</div>
                <div class="metric-label">out of 20</div>
            </div>
        </div>

        <h2>Code Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Files</td><td>{m.total_files}</td></tr>
            <tr><td>Python Files</td><td>{m.python_files}</td></tr>
            <tr><td>Lines of Code</td><td>{m.lines_of_code:,}</td></tr>
        </table>

        <h2>Infrastructure Usage</h2>
        <table>
            <tr><th>Component</th><th>Files Using</th></tr>
            <tr><td>ChatSession</td><td>{m.using_chat_session}</td></tr>
            <tr><td>BaseAgent</td><td>{m.using_base_agent}</td></tr>
            <tr><td>CacheManager</td><td>{m.using_cache_manager}</td></tr>
            <tr><td>ValidationFramework</td><td>{m.using_validation}</td></tr>
            <tr><td>Logger</td><td>{m.using_logger}</td></tr>
            <tr><td>ConfigManager</td><td>{m.using_config_manager}</td></tr>
        </table>

        <h2>Anti-patterns Detected</h2>
        <table>
            <tr><th>Anti-pattern</th><th>Count</th></tr>
            <tr><td>Direct OpenAI Calls</td><td>{m.direct_openai_calls}</td></tr>
            <tr><td>Direct Anthropic Calls</td><td>{m.direct_anthropic_calls}</td></tr>
            <tr><td>Custom Caching</td><td>{m.custom_caching}</td></tr>
            <tr><td>Print Statements</td><td>{m.print_statements}</td></tr>
            <tr><td>Raw Requests</td><td>{m.raw_requests}</td></tr>
            <tr><td>Manual Env Loading</td><td>{m.manual_env_loading}</td></tr>
        </table>

        <h2>Recommendations ({len(report.recommendations)})</h2>
"""

        for i, rec in enumerate(report.recommendations, 1):
            html += f'<div class="recommendation">{i}. {rec}</div>\n'

        html += """
    </div>
</body>
</html>
"""

        return html


class HealthChecker:
    """Main health checker."""

    def __init__(self):
        self.scanner = CodeScanner()
        self.calculator = IUMCalculator()
        self.reporter = ReportGenerator()

    def check(self, directory: str) -> HealthReport:
        """Run health check."""
        # Scan directory
        metrics = self.scanner.scan_directory(directory)

        # Calculate IUM score
        self.calculator.calculate(metrics)

        # Generate recommendations
        recommendations = self.reporter.generate_recommendations(metrics)

        # Create report
        report = HealthReport(
            timestamp=DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S"),
            directory=directory,
            metrics=metrics,
            issues=self.scanner.issues,
            recommendations=recommendations,
            file_details=self.scanner.file_details
        )

        return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Infrastructure health checker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current directory
  greenlang health-check

  # Check specific app
  greenlang health-check --directory GL-CBAM-APP

  # Generate HTML report
  greenlang health-check --format html --output health-report.html

  # JSON output
  greenlang health-check --format json
        """
    )

    parser.add_argument('--directory', default='.', help='Directory to check')
    parser.add_argument('--format', choices=['text', 'json', 'html'], default='text', help='Output format')
    parser.add_argument('--output', help='Output file')

    args = parser.parse_args()

    # Run health check
    print("Running infrastructure health check...")
    checker = HealthChecker()
    report = checker.check(args.directory)

    # Generate report
    if args.format == 'json':
        output = checker.reporter.generate_json_report(report)
    elif args.format == 'html':
        output = checker.reporter.generate_html_report(report)
    else:
        output = checker.reporter.generate_text_report(report)

    # Save or print
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\nReport saved to: {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
