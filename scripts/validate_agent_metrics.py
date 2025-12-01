#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Agent Metrics Validator
==================================

Validates that all GreenLang agents implement the required 50+ baseline metrics
according to the StandardAgentMetrics specification.

This script:
1. Scans agent directories for metric implementations
2. Validates metric count against 50+ baseline requirement
3. Checks metric naming conventions
4. Generates compliance reports
5. Identifies agents needing metric updates

Usage:
    python scripts/validate_agent_metrics.py

    # Generate detailed report
    python scripts/validate_agent_metrics.py --detailed

    # Check specific agent
    python scripts/validate_agent_metrics.py --agent GL-001

    # Generate JSON report
    python scripts/validate_agent_metrics.py --format json

Exit codes:
    0 - All agents compliant (50+ metrics)
    1 - One or more agents non-compliant
    2 - Script error

Author: GreenLang Team
License: Proprietary
"""

import os
import sys
import ast
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re


@dataclass
class MetricInfo:
    """Information about a single metric."""
    name: str
    metric_type: str  # Counter, Gauge, Histogram, Summary, Info
    description: str
    labels: List[str] = field(default_factory=list)
    category: str = "uncategorized"


@dataclass
class AgentMetricsReport:
    """Metrics compliance report for an agent."""
    agent_id: str
    agent_name: str
    total_metrics: int
    baseline_metrics: int
    agent_specific_metrics: int
    compliant: bool  # True if >= 50 metrics
    compliance_percentage: float
    metrics_by_category: Dict[str, int] = field(default_factory=dict)
    missing_categories: List[str] = field(default_factory=list)
    metric_details: List[MetricInfo] = field(default_factory=list)
    file_path: str = ""
    uses_standard_metrics: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MetricsValidator:
    """Validates agent metrics implementations."""

    # Minimum required metrics count
    MIN_METRICS_REQUIRED = 50

    # Standard metric categories and their minimum counts
    REQUIRED_CATEGORIES = {
        "agent_info": 3,      # At least 3 agent info metrics
        "request": 5,         # At least 5 request metrics
        "calculation": 4,     # At least 4 calculation metrics
        "validation": 3,      # At least 3 validation metrics
        "error": 3,           # At least 3 error metrics
        "performance": 3,     # At least 3 performance metrics
        "resource": 3,        # At least 3 resource metrics
        "integration": 3,     # At least 3 integration metrics
    }

    # Prometheus metric types
    METRIC_TYPES = {"Counter", "Gauge", "Histogram", "Summary", "Info"}

    def __init__(self, greenlang_root: Path):
        """Initialize validator."""
        self.greenlang_root = greenlang_root
        self.agents_dir = greenlang_root / "docs" / "planning" / "greenlang-2030-vision" / "agent_foundation" / "agents"
        self.reports: List[AgentMetricsReport] = []

    def find_agent_directories(self) -> List[Path]:
        """Find all agent directories (GL-XXX pattern)."""
        if not self.agents_dir.exists():
            print(f"Warning: Agents directory not found: {self.agents_dir}")
            return []

        agent_dirs = []
        for item in self.agents_dir.iterdir():
            if item.is_dir() and re.match(r"GL-\d+", item.name):
                agent_dirs.append(item)

        return sorted(agent_dirs)

    def find_metrics_file(self, agent_dir: Path) -> Optional[Path]:
        """Find metrics.py file in agent directory."""
        # Check standard location: monitoring/metrics.py
        metrics_file = agent_dir / "monitoring" / "metrics.py"
        if metrics_file.exists():
            return metrics_file

        # Check alternative locations
        alt_locations = [
            agent_dir / "metrics.py",
            agent_dir / "monitoring" / "prometheus_metrics.py",
        ]

        for location in alt_locations:
            if location.exists():
                return location

        return None

    def parse_metrics_file(self, metrics_file: Path) -> List[MetricInfo]:
        """Parse metrics.py file to extract metric definitions."""
        metrics = []

        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)

            # Look for metric assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    metric_info = self._extract_metric_from_assignment(node, content)
                    if metric_info:
                        metrics.append(metric_info)

        except Exception as e:
            print(f"Warning: Failed to parse {metrics_file}: {e}")

        return metrics

    def _extract_metric_from_assignment(self, node: ast.Assign, content: str) -> Optional[MetricInfo]:
        """Extract metric information from an assignment node."""
        # Check if it's a metric assignment (Counter, Gauge, etc.)
        if not isinstance(node.value, ast.Call):
            return None

        if not isinstance(node.value.func, ast.Name):
            return None

        metric_type = node.value.func.id
        if metric_type not in self.METRIC_TYPES:
            return None

        # Get metric name (first argument)
        if len(node.value.args) < 1:
            return None

        metric_name = None
        if isinstance(node.value.args[0], ast.Constant):
            metric_name = node.value.args[0].value
        elif isinstance(node.value.args[0], ast.Str):  # Python 3.7 compatibility
            metric_name = node.value.args[0].s

        if not metric_name:
            return None

        # Get description (second argument)
        description = "No description"
        if len(node.value.args) >= 2:
            if isinstance(node.value.args[1], ast.Constant):
                description = node.value.args[1].value
            elif isinstance(node.value.args[1], ast.Str):
                description = node.value.args[1].s

        # Get labels (third argument, if it's a list)
        labels = []
        if len(node.value.args) >= 3:
            labels_arg = node.value.args[2]
            if isinstance(labels_arg, ast.List):
                for elt in labels_arg.elts:
                    if isinstance(elt, ast.Constant):
                        labels.append(elt.value)
                    elif isinstance(elt, ast.Str):
                        labels.append(elt.s)

        # Categorize metric
        category = self._categorize_metric(metric_name, description)

        return MetricInfo(
            name=metric_name,
            metric_type=metric_type,
            description=description,
            labels=labels,
            category=category
        )

    def _categorize_metric(self, name: str, description: str) -> str:
        """Categorize metric based on name and description."""
        name_lower = name.lower()
        desc_lower = description.lower()

        if "agent" in name_lower and ("info" in name_lower or "health" in name_lower or "uptime" in name_lower):
            return "agent_info"
        elif "http" in name_lower or "request" in name_lower or "response" in name_lower:
            return "request"
        elif "calculation" in name_lower or "compute" in name_lower:
            return "calculation"
        elif "validation" in name_lower or "validate" in name_lower:
            return "validation"
        elif "error" in name_lower or "exception" in name_lower:
            return "error"
        elif "duration" in name_lower or "latency" in name_lower or "throughput" in name_lower:
            return "performance"
        elif "memory" in name_lower or "cpu" in name_lower or "thread" in name_lower:
            return "resource"
        elif "integration" in name_lower or "scada" in name_lower or "erp" in name_lower:
            return "integration"
        elif "cache" in name_lower:
            return "cache"
        elif "savings" in desc_lower or "cost" in desc_lower or "roi" in desc_lower:
            return "business"
        elif "provenance" in name_lower or "determinism" in name_lower:
            return "provenance"
        else:
            return "agent_specific"

    def check_standard_metrics_usage(self, metrics_file: Path) -> bool:
        """Check if agent uses StandardAgentMetrics."""
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for StandardAgentMetrics import or inheritance
            patterns = [
                r"from.*standard_metrics.*import.*StandardAgentMetrics",
                r"import.*standard_metrics",
                r"class.*\(StandardAgentMetrics\)",
            ]

            for pattern in patterns:
                if re.search(pattern, content):
                    return True

        except Exception as e:
            print(f"Warning: Failed to check StandardAgentMetrics usage: {e}")

        return False

    def validate_agent(self, agent_dir: Path) -> AgentMetricsReport:
        """Validate metrics for a single agent."""
        agent_id = agent_dir.name
        agent_name = agent_id  # Default to agent_id

        # Find metrics file
        metrics_file = self.find_metrics_file(agent_dir)

        if not metrics_file:
            return AgentMetricsReport(
                agent_id=agent_id,
                agent_name=agent_name,
                total_metrics=0,
                baseline_metrics=0,
                agent_specific_metrics=0,
                compliant=False,
                compliance_percentage=0.0,
                file_path="NOT FOUND"
            )

        # Parse metrics
        metrics = self.parse_metrics_file(metrics_file)

        # Check if using StandardAgentMetrics
        uses_standard = self.check_standard_metrics_usage(metrics_file)

        # Categorize metrics
        metrics_by_category = {}
        for metric in metrics:
            category = metric.category
            metrics_by_category[category] = metrics_by_category.get(category, 0) + 1

        # Count baseline vs agent-specific
        baseline_count = sum(
            count for cat, count in metrics_by_category.items()
            if cat != "agent_specific"
        )
        agent_specific_count = metrics_by_category.get("agent_specific", 0)

        # Check compliance
        total_metrics = len(metrics)
        compliant = total_metrics >= self.MIN_METRICS_REQUIRED
        compliance_percentage = (total_metrics / self.MIN_METRICS_REQUIRED) * 100

        # Check for missing required categories
        missing_categories = []
        for category, min_count in self.REQUIRED_CATEGORIES.items():
            if metrics_by_category.get(category, 0) < min_count:
                missing_categories.append(f"{category} (need {min_count}, have {metrics_by_category.get(category, 0)})")

        return AgentMetricsReport(
            agent_id=agent_id,
            agent_name=agent_name,
            total_metrics=total_metrics,
            baseline_metrics=baseline_count,
            agent_specific_metrics=agent_specific_count,
            compliant=compliant,
            compliance_percentage=compliance_percentage,
            metrics_by_category=metrics_by_category,
            missing_categories=missing_categories,
            metric_details=metrics,
            file_path=str(metrics_file.relative_to(self.greenlang_root)),
            uses_standard_metrics=uses_standard
        )

    def validate_all_agents(self) -> List[AgentMetricsReport]:
        """Validate all agents."""
        agent_dirs = self.find_agent_directories()

        print(f"Found {len(agent_dirs)} agent directories to validate...")

        reports = []
        for agent_dir in agent_dirs:
            print(f"  Validating {agent_dir.name}...")
            report = self.validate_agent(agent_dir)
            reports.append(report)

        self.reports = reports
        return reports

    def generate_summary_report(self) -> str:
        """Generate summary report."""
        if not self.reports:
            return "No agents validated."

        compliant_agents = [r for r in self.reports if r.compliant]
        non_compliant_agents = [r for r in self.reports if not r.compliant]
        using_standard = [r for r in self.reports if r.uses_standard_metrics]

        lines = [
            "=" * 80,
            "GreenLang Agent Metrics Compliance Report",
            "=" * 80,
            f"Validation Date: {datetime.utcnow().isoformat()}",
            f"Minimum Required Metrics: {self.MIN_METRICS_REQUIRED}",
            "",
            "Summary:",
            f"  Total Agents Validated: {len(self.reports)}",
            f"  Compliant (50+ metrics): {len(compliant_agents)} ({len(compliant_agents)/len(self.reports)*100:.1f}%)",
            f"  Non-Compliant (<50 metrics): {len(non_compliant_agents)} ({len(non_compliant_agents)/len(self.reports)*100:.1f}%)",
            f"  Using StandardAgentMetrics: {len(using_standard)} ({len(using_standard)/len(self.reports)*100:.1f}%)",
            "",
            "-" * 80,
            "Compliant Agents (50+ metrics):",
            "-" * 80,
        ]

        for report in sorted(compliant_agents, key=lambda r: r.total_metrics, reverse=True):
            status = "✓ STANDARD" if report.uses_standard_metrics else "✓ CUSTOM"
            lines.append(
                f"  {status} {report.agent_id:10} - {report.total_metrics:3} metrics "
                f"({report.baseline_metrics} baseline + {report.agent_specific_metrics} specific) "
                f"[{report.compliance_percentage:.0f}% of requirement]"
            )

        if non_compliant_agents:
            lines.extend([
                "",
                "-" * 80,
                "Non-Compliant Agents (<50 metrics) - ACTION REQUIRED:",
                "-" * 80,
            ])

            for report in sorted(non_compliant_agents, key=lambda r: r.total_metrics):
                status = "✗ FAIL"
                lines.append(
                    f"  {status} {report.agent_id:10} - {report.total_metrics:3} metrics "
                    f"({self.MIN_METRICS_REQUIRED - report.total_metrics} short) "
                    f"[{report.compliance_percentage:.0f}% of requirement]"
                )

                if report.missing_categories:
                    lines.append(f"         Missing: {', '.join(report.missing_categories)}")

        lines.extend([
            "",
            "=" * 80,
            f"Overall Compliance: {len(compliant_agents)}/{len(self.reports)} agents compliant",
            "=" * 80,
        ])

        return "\n".join(lines)

    def generate_detailed_report(self, agent_id: Optional[str] = None) -> str:
        """Generate detailed report for specific agent or all agents."""
        reports_to_show = self.reports

        if agent_id:
            reports_to_show = [r for r in self.reports if r.agent_id == agent_id]

        if not reports_to_show:
            return f"No reports found for agent: {agent_id}"

        lines = []

        for report in reports_to_show:
            lines.extend([
                "=" * 80,
                f"Agent: {report.agent_id} - {report.agent_name}",
                "=" * 80,
                f"Total Metrics: {report.total_metrics}",
                f"Baseline Metrics: {report.baseline_metrics}",
                f"Agent-Specific Metrics: {report.agent_specific_metrics}",
                f"Compliant: {'YES ✓' if report.compliant else 'NO ✗'}",
                f"Compliance: {report.compliance_percentage:.1f}%",
                f"Uses StandardAgentMetrics: {'Yes' if report.uses_standard_metrics else 'No'}",
                f"File: {report.file_path}",
                "",
                "Metrics by Category:",
            ])

            for category, count in sorted(report.metrics_by_category.items()):
                required = self.REQUIRED_CATEGORIES.get(category, 0)
                status = "✓" if count >= required or category == "agent_specific" else "✗"
                lines.append(f"  {status} {category:20} - {count:3} metrics (required: {required})")

            if report.missing_categories:
                lines.extend([
                    "",
                    "Missing/Insufficient Categories:",
                ])
                for missing in report.missing_categories:
                    lines.append(f"  ✗ {missing}")

            lines.append("")

        return "\n".join(lines)

    def export_json(self, output_file: Path):
        """Export reports to JSON."""
        data = {
            "validation_date": datetime.utcnow().isoformat(),
            "min_required_metrics": self.MIN_METRICS_REQUIRED,
            "total_agents": len(self.reports),
            "compliant_agents": len([r for r in self.reports if r.compliant]),
            "non_compliant_agents": len([r for r in self.reports if not r.compliant]),
            "agents": [
                {
                    **asdict(report),
                    "metric_details": [
                        asdict(m) for m in report.metric_details
                    ]
                }
                for report in self.reports
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"JSON report exported to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GreenLang agent metrics compliance"
    )
    parser.add_argument(
        "--agent",
        help="Validate specific agent (e.g., GL-001)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed report"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for JSON report"
    )

    args = parser.parse_args()

    # Find GreenLang root
    script_dir = Path(__file__).parent
    greenlang_root = script_dir.parent

    # Create validator
    validator = MetricsValidator(greenlang_root)

    # Validate agents
    if args.agent:
        agent_dir = validator.agents_dir / args.agent
        if not agent_dir.exists():
            print(f"Error: Agent directory not found: {agent_dir}")
            return 2

        report = validator.validate_agent(agent_dir)
        validator.reports = [report]
    else:
        validator.validate_all_agents()

    # Generate report
    if args.format == "json":
        output_file = args.output or greenlang_root / "metrics_compliance_report.json"
        validator.export_json(output_file)
    else:
        if args.detailed:
            print(validator.generate_detailed_report(args.agent))
        else:
            print(validator.generate_summary_report())

    # Exit with appropriate code
    non_compliant = [r for r in validator.reports if not r.compliant]
    if non_compliant:
        print(f"\n⚠ Warning: {len(non_compliant)} agent(s) do not meet the 50+ metrics requirement")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
