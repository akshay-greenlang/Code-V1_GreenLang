#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GreenLang Agent Exit Bar Validation Script.

This script automates the validation of GreenLang AI agents against the
12-dimension exit bar criteria defined in GL_agent_requirement.md.

Features:
    - Automated validation of all exit bar criteria
    - Support for single agent or batch validation
    - Multiple output formats (markdown, HTML, JSON, YAML)
    - Detailed scoring and blocker identification
    - Production readiness assessment
    - Timeline estimation for production

Usage:
    # Validate single agent
    python scripts/validate_exit_bar.py --agent carbon_agent

    # Validate all agents
    python scripts/validate_exit_bar.py --all-agents

    # Generate HTML report
    python scripts/validate_exit_bar.py --agent carbon_agent --format html

    # Watch mode (continuous validation)
    python scripts/validate_exit_bar.py --agent carbon_agent --watch

    # Specify custom checklist
    python scripts/validate_exit_bar.py --agent carbon_agent --checklist custom.yaml

Author: GreenLang Framework Team
Date: 2025-10-16
Version: 1.0.0
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from greenlang.determinism import DeterministicClock


class ExitBarValidator:
    """Validates GreenLang agents against exit bar criteria."""

    def __init__(
        self,
        checklist_path: str = "templates/exit_bar_checklist.yaml",
        verbose: bool = False
    ):
        """Initialize the validator.

        Args:
            checklist_path: Path to exit bar checklist YAML
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.checklist_path = Path(checklist_path)
        self.checklist = self._load_checklist()
        self.results: Dict[str, Any] = {}

    def _load_checklist(self) -> Dict[str, Any]:
        """Load exit bar checklist from YAML."""
        if not self.checklist_path.exists():
            raise FileNotFoundError(
                f"Checklist not found: {self.checklist_path}"
            )

        with open(self.checklist_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _log(self, message: str, level: str = "INFO") -> None:
        """Log message if verbose mode enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def _resolve_path_template(
        self, template: str, agent_name: str, **kwargs
    ) -> str:
        """Resolve path template with agent name."""
        return template.format(agent_name=agent_name, **kwargs)

    def _file_exists(self, config: Dict[str, Any], agent_name: str) -> Tuple[bool, str]:
        """Validate file existence.

        Args:
            config: Validation configuration
            agent_name: Agent name

        Returns:
            Tuple of (success, message)
        """
        if "path_template" in config:
            try:
                path = self._resolve_path_template(config["path_template"], agent_name)
                file_path = Path(path)

                if file_path.exists():
                    return True, f"File exists: {file_path}"
                else:
                    return False, f"File not found: {file_path}"
            except KeyError as e:
                # Template variable not provided (e.g., {domain})
                return False, f"Cannot resolve path template: missing {e}"

        elif "path_options" in config:
            for path_template in config["path_options"]:
                try:
                    path = self._resolve_path_template(path_template, agent_name)
                    file_path = Path(path)

                    if file_path.exists():
                        return True, f"File exists: {file_path}"
                except KeyError:
                    # Template variable not provided, skip this option
                    continue

            return False, "None of the expected files found"

        return False, "Invalid file_exists configuration"

    def _yaml_sections(self, config: Dict[str, Any], spec_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate YAML sections exist.

        Args:
            config: Validation configuration
            spec_data: Agent specification data

        Returns:
            Tuple of (success, message)
        """
        required_sections = config.get("required_sections", [])
        missing_sections = []

        for section in required_sections:
            # Navigate nested sections using dot notation
            parts = section.split(".")
            current = spec_data

            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    missing_sections.append(section)
                    break

        if missing_sections:
            return False, f"Missing sections: {', '.join(missing_sections)}"

        return True, f"All {len(required_sections)} sections present"

    def _yaml_value(self, config: Dict[str, Any], spec_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate YAML value.

        Args:
            config: Validation configuration
            spec_data: Agent specification data

        Returns:
            Tuple of (success, message)
        """
        yaml_path = config.get("yaml_path", "")
        parts = yaml_path.split(".")
        current = spec_data

        # Navigate to the value
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False, f"Path not found: {yaml_path}"

        # Check expected value
        if "expected_value" in config:
            expected = config["expected_value"]
            if config.get("exact_match", False):
                if current == expected:
                    return True, f"Value matches: {current}"
                else:
                    return False, f"Expected {expected}, got {current}"

        # Check min count (for lists)
        if "min_count" in config:
            min_count = config["min_count"]
            if isinstance(current, list):
                actual_count = len(current)
                if actual_count >= min_count:
                    return True, f"Count OK: {actual_count} >= {min_count}"
                else:
                    return False, f"Count too low: {actual_count} < {min_count}"

        # Check allowed values
        if "allowed_values" in config:
            allowed = config["allowed_values"]
            if current in allowed:
                return True, f"Value allowed: {current}"
            else:
                return False, f"Value not allowed: {current} (expected one of {allowed})"

        return True, f"Value OK: {current}"

    def _code_pattern(self, config: Dict[str, Any], agent_name: str) -> Tuple[bool, str]:
        """Validate code pattern exists.

        Args:
            config: Validation configuration
            agent_name: Agent name

        Returns:
            Tuple of (success, message)
        """
        try:
            file_path_template = config.get("file_path", "")
            file_path = Path(self._resolve_path_template(file_path_template, agent_name))

            if not file_path.exists():
                return False, f"File not found: {file_path}"
        except KeyError as e:
            return False, f"Cannot resolve path template: missing {e}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = config.get("pattern", "")
        multiline = config.get("multiline", False)

        if multiline:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        else:
            matches = re.findall(pattern, content)

        match_count = len(matches)
        min_count = config.get("min_count", 1)
        max_count = config.get("max_count", float("inf"))
        invert = config.get("invert", False)

        if invert:
            # For inverted checks (e.g., ensure pattern does NOT exist)
            if match_count > max_count:
                return False, f"Pattern should not exist, but found {match_count} matches"
            else:
                return True, f"Pattern correctly absent (0 matches)"
        else:
            if match_count >= min_count and match_count <= max_count:
                return True, f"Pattern found {match_count} times (min: {min_count})"
            else:
                return False, f"Pattern found {match_count} times (expected: {min_count}+)"

    def _test_count(self, config: Dict[str, Any], agent_name: str) -> Tuple[bool, str]:
        """Count tests in test file.

        Args:
            config: Validation configuration
            agent_name: Agent name

        Returns:
            Tuple of (success, message)
        """
        try:
            file_path_template = config.get("file_path", "")
            file_path = Path(self._resolve_path_template(file_path_template, agent_name))

            if not file_path.exists():
                return False, f"Test file not found: {file_path}"
        except KeyError as e:
            return False, f"Cannot resolve path template: missing {e}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = config.get("test_pattern", "def test_")
        matches = re.findall(pattern, content)
        test_count = len(matches)
        min_count = config.get("min_count", 1)

        if test_count >= min_count:
            return True, f"Found {test_count} tests (min: {min_count})"
        else:
            return False, f"Only {test_count} tests found (need {min_count}+)"

    def _command(self, config: Dict[str, Any], agent_name: str) -> Tuple[bool, str]:
        """Execute validation command.

        Args:
            config: Validation configuration
            agent_name: Agent name

        Returns:
            Tuple of (success, message)
        """
        try:
            command = config.get("command", "")
            command = self._resolve_path_template(command, agent_name)
            timeout = config.get("timeout_seconds", 30)
            success_pattern = config.get("success_pattern", "")
            error_pattern = config.get("error_pattern", "")
        except KeyError as e:
            return False, f"Cannot resolve command template: missing {e}"

        try:
            self._log(f"Running command: {command}", "DEBUG")

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout + result.stderr

            # Check for success pattern
            if success_pattern:
                if re.search(success_pattern, output):
                    return True, f"Command passed: {success_pattern} found"

            # Check for error pattern
            if error_pattern:
                if re.search(error_pattern, output):
                    return False, f"Command failed: {error_pattern} found"

            # Default to return code
            if result.returncode == 0:
                return True, "Command executed successfully"
            else:
                return False, f"Command failed with code {result.returncode}"

        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, f"Command failed: {str(e)}"

    def _docstring_coverage(self, config: Dict[str, Any], agent_name: str) -> Tuple[bool, str]:
        """Calculate docstring coverage.

        Args:
            config: Validation configuration
            agent_name: Agent name

        Returns:
            Tuple of (success, message)
        """
        try:
            file_path_template = config.get("file_path", "")
            file_path = Path(self._resolve_path_template(file_path_template, agent_name))

            if not file_path.exists():
                return False, f"File not found: {file_path}"
        except KeyError as e:
            return False, f"Cannot resolve path template: missing {e}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Count functions/methods
        functions = re.findall(r"^\s*def\s+\w+\s*\(", content, re.MULTILINE)
        function_count = len(functions)

        # Count docstrings (look for """ or ''' after def)
        docstrings = re.findall(
            r"^\s*def\s+\w+\s*\(.*?\).*?:\s*(?:\n\s*)?['\"]{{3}}",
            content,
            re.MULTILINE | re.DOTALL
        )
        docstring_count = len(docstrings)

        if function_count == 0:
            return True, "No functions found (N/A)"

        coverage = docstring_count / function_count
        min_coverage = config.get("min_coverage", 0.90)

        if coverage >= min_coverage:
            return True, f"Docstring coverage: {coverage:.1%} (min: {min_coverage:.0%})"
        else:
            return False, f"Docstring coverage: {coverage:.1%} (need {min_coverage:.0%})"

    def validate_criterion(
        self,
        criterion: Dict[str, Any],
        agent_name: str,
        spec_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a single criterion.

        Args:
            criterion: Criterion configuration
            agent_name: Agent name
            spec_data: Agent specification data

        Returns:
            Validation result dictionary
        """
        criterion_id = criterion.get("id", "Unknown")
        name = criterion.get("name", "Unnamed criterion")
        validation_type = criterion.get("validation_type", "unknown")
        config = criterion.get("config", {})
        required = criterion.get("required", False)
        points = criterion.get("points", 0)

        self._log(f"Validating {criterion_id}: {name}", "DEBUG")

        # Dispatch to appropriate validation method
        if validation_type == "file_exists":
            success, message = self._file_exists(config, agent_name)
        elif validation_type == "yaml_sections":
            if spec_data is None:
                success, message = False, "Spec data not available"
            else:
                success, message = self._yaml_sections(config, spec_data)
        elif validation_type == "yaml_value":
            if spec_data is None:
                success, message = False, "Spec data not available"
            else:
                success, message = self._yaml_value(config, spec_data)
        elif validation_type == "code_pattern":
            success, message = self._code_pattern(config, agent_name)
        elif validation_type == "test_count":
            success, message = self._test_count(config, agent_name)
        elif validation_type == "command":
            success, message = self._command(config, agent_name)
        elif validation_type == "docstring_coverage":
            success, message = self._docstring_coverage(config, agent_name)
        else:
            success, message = False, f"Unknown validation type: {validation_type}"

        # Determine status
        if success:
            status = "PASS"
            points_earned = points
        else:
            if required:
                status = "FAIL"
            else:
                status = "WARN"
            points_earned = 0

        return {
            "id": criterion_id,
            "name": name,
            "status": status,
            "required": required,
            "points_possible": points,
            "points_earned": points_earned,
            "message": message,
            "validation_type": validation_type
        }

    def validate_dimension(
        self,
        dimension_key: str,
        agent_name: str,
        spec_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a dimension.

        Args:
            dimension_key: Dimension key (e.g., "d1_specification")
            agent_name: Agent name
            spec_data: Agent specification data

        Returns:
            Dimension validation results
        """
        dimension = self.checklist.get(dimension_key, {})
        criteria = dimension.get("criteria", [])
        max_score = dimension.get("max_score", 0)

        self._log(f"Validating dimension: {dimension_key}", "INFO")

        results = []
        total_points = 0
        points_earned = 0
        blockers = []

        for criterion in criteria:
            result = self.validate_criterion(criterion, agent_name, spec_data)
            results.append(result)

            total_points += result["points_possible"]
            points_earned += result["points_earned"]

            if result["status"] == "FAIL" and result["required"]:
                blockers.append({
                    "id": result["id"],
                    "name": result["name"],
                    "message": result["message"]
                })

        # Calculate score
        if total_points > 0:
            score = points_earned
        else:
            score = 0

        # Determine status
        if blockers:
            status = "FAIL"
        elif score == max_score:
            status = "PASS"
        else:
            status = "PARTIAL"

        return {
            "dimension": dimension_key,
            "status": status,
            "score": score,
            "max_score": max_score,
            "criteria_results": results,
            "blockers": blockers
        }

    def validate_agent(
        self,
        agent_name: str,
        spec_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate an agent against all exit bar criteria.

        Args:
            agent_name: Agent name (e.g., "carbon_agent")
            spec_path: Optional path to agent spec YAML

        Returns:
            Complete validation results
        """
        self._log(f"Starting validation for agent: {agent_name}", "INFO")

        # Load agent specification if available
        spec_data = None
        if spec_path:
            spec_file = Path(spec_path)
            if spec_file.exists():
                with open(spec_file, "r", encoding="utf-8") as f:
                    spec_data = yaml.safe_load(f)
                    self._log(f"Loaded spec: {spec_path}", "INFO")

        # Validate all dimensions
        dimension_results = {}
        all_blockers = []
        total_score = 0
        max_total_score = 0

        for dimension_key in [
            "d1_specification",
            "d2_implementation",
            "d3_test_coverage",
            "d4_deterministic_ai",
            "d5_documentation",
            "d6_compliance",
            "d7_deployment",
            "d8_exit_bar",
            "d9_integration",
            "d10_business_impact",
            "d11_operations",
            "d12_improvement"
        ]:
            result = self.validate_dimension(dimension_key, agent_name, spec_data)
            dimension_results[dimension_key] = result

            total_score += result["score"]
            max_total_score += result["max_score"]

            if result["blockers"]:
                all_blockers.extend([
                    {**blocker, "dimension": dimension_key}
                    for blocker in result["blockers"]
                ])

        # Calculate overall score
        if max_total_score > 0:
            overall_score = (total_score / max_total_score) * 100
        else:
            overall_score = 0

        # Determine readiness status
        thresholds = self.checklist.get("score_thresholds", {})
        if overall_score >= thresholds.get("production_ready", 95):
            readiness = "PRODUCTION READY"
        elif overall_score >= thresholds.get("pre_production", 80):
            readiness = "PRE-PRODUCTION"
        elif overall_score >= thresholds.get("development", 60):
            readiness = "DEVELOPMENT"
        else:
            readiness = "EARLY DEVELOPMENT"

        return {
            "agent_name": agent_name,
            "validation_date": DeterministicClock.now().isoformat(),
            "overall_score": round(overall_score, 2),
            "total_points_earned": total_score,
            "total_points_possible": max_total_score,
            "readiness_status": readiness,
            "blockers": all_blockers,
            "dimensions": dimension_results
        }

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown validation report.

        Args:
            results: Validation results

        Returns:
            Markdown report string
        """
        agent_name = results["agent_name"]
        date = DeterministicClock.now().strftime("%Y-%m-%d")
        score = results["overall_score"]
        status = results["readiness_status"]
        blockers = results["blockers"]
        dimensions = results["dimensions"]

        report = f"""# Exit Bar Validation Report
## Agent: {agent_name}
## Date: {date}
## Overall Score: {score}/100 ({status})

---

### Executive Summary

**Validation Date:** {results["validation_date"]}
**Total Score:** {results["total_points_earned"]}/{results["total_points_possible"]} points ({score}%)
**Readiness Status:** {status}
**Production Ready:** {"✅ YES" if score >= 95 else "❌ NO"}
**Blockers:** {len(blockers)}

---

### Dimension Breakdown

| Dimension | Score | Max | Status | Blockers |
|-----------|-------|-----|--------|----------|
"""

        # Add dimension rows
        dimension_names = {
            "d1_specification": "D1: Specification",
            "d2_implementation": "D2: Implementation",
            "d3_test_coverage": "D3: Test Coverage",
            "d4_deterministic_ai": "D4: Deterministic AI",
            "d5_documentation": "D5: Documentation",
            "d6_compliance": "D6: Compliance",
            "d7_deployment": "D7: Deployment",
            "d8_exit_bar": "D8: Exit Bar",
            "d9_integration": "D9: Integration",
            "d10_business_impact": "D10: Business Impact",
            "d11_operations": "D11: Operations",
            "d12_improvement": "D12: Improvement"
        }

        for dim_key, dim_name in dimension_names.items():
            dim_result = dimensions.get(dim_key, {})
            dim_score = dim_result.get("score", 0)
            dim_max = dim_result.get("max_score", 0)
            dim_status = dim_result.get("status", "UNKNOWN")
            dim_blockers = len(dim_result.get("blockers", []))

            status_emoji = {
                "PASS": "✅",
                "PARTIAL": "⚠️",
                "FAIL": "❌"
            }.get(dim_status, "❓")

            report += f"| {dim_name} | {dim_score} | {dim_max} | {status_emoji} {dim_status} | {dim_blockers} |\n"

        # Add blockers section
        report += "\n---\n\n### Blockers to Production\n\n"

        if blockers:
            report += f"**Total Blockers:** {len(blockers)}\n\n"
            for i, blocker in enumerate(blockers, 1):
                dim = blocker.get("dimension", "Unknown")
                blocker_id = blocker.get("id", "Unknown")
                name = blocker.get("name", "Unknown")
                message = blocker.get("message", "No details")

                report += f"{i}. **{blocker_id}** ({dim}): {name}\n"
                report += f"   - Issue: {message}\n\n"
        else:
            report += "✅ **No blockers detected!**\n\n"

        # Add recommended actions
        report += "---\n\n### Recommended Actions\n\n"

        if score < 95:
            report += "To reach production readiness (95%):\n\n"

            # Identify top priority fixes
            for dim_key, dim_result in dimensions.items():
                if dim_result.get("blockers"):
                    dim_name = dimension_names.get(dim_key, dim_key)
                    report += f"**{dim_name}:**\n"

                    for blocker in dim_result["blockers"]:
                        report += f"- {blocker['name']}: {blocker['message']}\n"

                    report += "\n"
        else:
            report += "✅ Agent is production ready! No critical actions needed.\n\n"

        # Add timeline estimate
        report += "---\n\n### Timeline to Production\n\n"

        if score >= 95:
            report += "**Status:** READY FOR PRODUCTION\n"
            report += "**Estimated time:** 0 days (ready now)\n"
        elif score >= 80:
            report += "**Status:** Pre-production (minor gaps)\n"
            report += f"**Blockers:** {len(blockers)}\n"
            report += "**Estimated time:** 1-2 weeks\n"
        elif score >= 60:
            report += "**Status:** Development (major gaps)\n"
            report += f"**Blockers:** {len(blockers)}\n"
            report += "**Estimated time:** 3-4 weeks\n"
        else:
            report += "**Status:** Early development\n"
            report += f"**Blockers:** {len(blockers)}\n"
            report += "**Estimated time:** 6-8 weeks\n"

        # Add detailed results
        report += "\n---\n\n### Detailed Validation Results\n\n"

        for dim_key, dim_name in dimension_names.items():
            dim_result = dimensions.get(dim_key, {})
            criteria_results = dim_result.get("criteria_results", [])

            report += f"#### {dim_name}\n\n"
            report += f"**Score:** {dim_result.get('score', 0)}/{dim_result.get('max_score', 0)}\n"
            report += f"**Status:** {dim_result.get('status', 'UNKNOWN')}\n\n"

            if criteria_results:
                report += "| ID | Criterion | Status | Points | Message |\n"
                report += "|----|-----------|--------|--------|----------|\n"

                for criterion in criteria_results:
                    crit_id = criterion.get("id", "")
                    crit_name = criterion.get("name", "")
                    crit_status = criterion.get("status", "")
                    crit_points = f"{criterion.get('points_earned', 0)}/{criterion.get('points_possible', 0)}"
                    crit_message = criterion.get("message", "")

                    status_emoji = {
                        "PASS": "✅",
                        "WARN": "⚠️",
                        "FAIL": "❌"
                    }.get(crit_status, "❓")

                    report += f"| {crit_id} | {crit_name} | {status_emoji} {crit_status} | {crit_points} | {crit_message} |\n"

            report += "\n"

        return report

    def generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate JSON validation report.

        Args:
            results: Validation results

        Returns:
            JSON report string
        """
        return json.dumps(results, indent=2)

    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML validation report.

        Args:
            results: Validation results

        Returns:
            HTML report string
        """
        # Simple HTML wrapper for markdown
        markdown_report = self.generate_markdown_report(results)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Exit Bar Validation - {results['agent_name']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        h3 {{ color: #7f8c8d; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
        .warn {{ color: #f39c12; font-weight: bold; }}
        .score {{
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }}
        .production {{ color: #27ae60; }}
        .pre-production {{ color: #f39c12; }}
        .development {{ color: #e67e22; }}
    </style>
</head>
<body>
    <pre>{markdown_report}</pre>
</body>
</html>
"""
        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GreenLang agents against exit bar criteria"
    )

    parser.add_argument(
        "--agent",
        help="Agent name to validate (e.g., carbon_agent)"
    )

    parser.add_argument(
        "--spec-path",
        help="Path to agent specification YAML"
    )

    parser.add_argument(
        "--all-agents",
        action="store_true",
        help="Validate all agents"
    )

    parser.add_argument(
        "--checklist",
        default="templates/exit_bar_checklist.yaml",
        help="Path to exit bar checklist YAML"
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "json", "html", "yaml"],
        default="markdown",
        help="Output format"
    )

    parser.add_argument(
        "--output",
        help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - continuous validation"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.agent and not args.all_agents:
        parser.error("Either --agent or --all-agents is required")

    # Create validator
    validator = ExitBarValidator(
        checklist_path=args.checklist,
        verbose=args.verbose
    )

    # Run validation
    if args.all_agents:
        # TODO: Implement batch validation
        print("Batch validation not yet implemented", file=sys.stderr)
        sys.exit(1)
    else:
        if args.watch:
            # Watch mode
            print(f"Watching {args.agent} for changes (Ctrl+C to stop)...\n")
            try:
                while True:
                    results = validator.validate_agent(args.agent, args.spec_path)

                    # Generate report
                    if args.format == "markdown":
                        report = validator.generate_markdown_report(results)
                    elif args.format == "json":
                        report = validator.generate_json_report(results)
                    elif args.format == "html":
                        report = validator.generate_html_report(results)
                    else:
                        report = yaml.dump(results, default_flow_style=False)

                    # Clear screen
                    print("\033[2J\033[H")  # ANSI clear screen
                    print(report)
                    print(f"\n[Last updated: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}]")

                    time.sleep(5)  # Wait 5 seconds
            except KeyboardInterrupt:
                print("\nWatch mode stopped.")
                sys.exit(0)
        else:
            # Single validation
            results = validator.validate_agent(args.agent, args.spec_path)

            # Generate report
            if args.format == "markdown":
                report = validator.generate_markdown_report(results)
            elif args.format == "json":
                report = validator.generate_json_report(results)
            elif args.format == "html":
                report = validator.generate_html_report(results)
            else:
                report = yaml.dump(results, default_flow_style=False)

            # Output report
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"Report written to: {args.output}")
            else:
                print(report)

            # Exit code based on results
            if results["overall_score"] >= 95:
                sys.exit(0)  # Production ready
            else:
                sys.exit(1)  # Not production ready


if __name__ == "__main__":
    main()
