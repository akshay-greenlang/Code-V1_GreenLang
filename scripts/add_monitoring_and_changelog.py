#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add monitoring and changelog to existing GreenLang agents.

This script automatically integrates operational monitoring and changelog
templates into existing GreenLang AI agents.

Features:
- Adds OperationalMonitoringMixin to agent classes
- Creates CHANGELOG.md for agent versioning
- Updates agent code to call monitoring methods
- Verifies integration completeness
- Generates integration reports

Usage:
    python scripts/add_monitoring_and_changelog.py --agent carbon_agent
    python scripts/add_monitoring_and_changelog.py --all-agents
    python scripts/add_monitoring_and_changelog.py --agent fuel_agent --dry-run
    python scripts/add_monitoring_and_changelog.py --agent boiler_agent --version 1.0.0

Author: GreenLang Framework Team
Date: October 2025
Version: 1.0.0
"""

import argparse
import sys
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from greenlang.determinism import DeterministicClock


class MonitoringIntegrator:
    """Integrates monitoring and changelog into GreenLang agents."""

    def __init__(self, workspace_root: Path):
        """Initialize the integrator.

        Args:
            workspace_root: Root directory of the GreenLang workspace
        """
        self.workspace_root = workspace_root
        self.agents_dir = workspace_root / "greenlang" / "agents"
        self.templates_dir = workspace_root / "templates"
        self.monitoring_template = self.templates_dir / "agent_monitoring.py"
        self.changelog_template = self.templates_dir / "CHANGELOG_TEMPLATE.md"

        # Verify templates exist
        if not self.monitoring_template.exists():
            raise FileNotFoundError(f"Monitoring template not found: {self.monitoring_template}")

        if not self.changelog_template.exists():
            raise FileNotFoundError(f"Changelog template not found: {self.changelog_template}")

    def find_all_agents(self) -> List[str]:
        """Find all agent files in the agents directory.

        Returns:
            List of agent names (without _agent.py suffix)
        """
        if not self.agents_dir.exists():
            return []

        agents = []
        for file in self.agents_dir.glob("*_agent.py"):
            # Skip base, types, and mock
            if file.stem in ["base", "types", "mock"]:
                continue

            agent_name = file.stem.replace("_agent", "")
            agents.append(agent_name)

        return sorted(agents)

    def add_monitoring_to_agent(
        self,
        agent_name: str,
        dry_run: bool = False
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Add OperationalMonitoringMixin to an agent.

        Args:
            agent_name: Name of the agent (e.g., "carbon", "fuel", "carbon_agent_ai")
            dry_run: If True, only simulate changes

        Returns:
            Tuple of (success, message, changes_dict)
        """
        # Try standard pattern first
        agent_file = self.agents_dir / f"{agent_name}_agent.py"

        # If not found, try AI agent pattern
        if not agent_file.exists():
            agent_file = self.agents_dir / f"{agent_name}_agent_ai.py"

        # If still not found, maybe it's already complete (e.g., passed "carbon_agent")
        if not agent_file.exists():
            agent_file = self.agents_dir / f"{agent_name}.py"

        if not agent_file.exists():
            return False, f"Agent file not found: {agent_name} (tried multiple patterns)", {}

        # Read agent file
        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()

        changes = {
            "import_added": False,
            "mixin_added": False,
            "setup_monitoring_added": False,
            "tracking_added": False,
            "backup_created": False
        }

        # Check if already integrated
        if "OperationalMonitoringMixin" in content:
            return True, f"Agent {agent_name} already has monitoring integrated", changes

        # Parse the file
        lines = content.split('\n')
        new_lines = []

        # Track state
        import_section_done = False
        class_found = False
        class_name = None
        init_found = False
        execute_found = False
        indent = ""

        for i, line in enumerate(lines):
            # Step 1: Add import after base imports (handle both patterns)
            if not import_section_done:
                # Pattern 1: from greenlang.agents.base import BaseAgent
                if line.startswith("from greenlang.agents.base"):
                    new_lines.append(line)
                    new_lines.append("from greenlang.templates.agent_monitoring import OperationalMonitoringMixin")
                    changes["import_added"] = True
                    import_section_done = True
                    continue
                # Pattern 2: from ..types import Agent
                elif line.startswith("from ..types import Agent"):
                    new_lines.append(line)
                    new_lines.append("from greenlang.templates.agent_monitoring import OperationalMonitoringMixin")
                    changes["import_added"] = True
                    import_section_done = True
                    continue

            # Step 2: Find class definition and add mixin (handle both patterns)
            if not class_found:
                # Pattern 1: class Name(BaseAgent):
                if re.match(r'^class (\w+)\(BaseAgent\):', line):
                    class_match = re.match(r'^class (\w+)\(BaseAgent\):', line)
                    class_name = class_match.group(1)
                    new_line = line.replace("(BaseAgent)", "(OperationalMonitoringMixin, BaseAgent)")
                    new_lines.append(new_line)
                    changes["mixin_added"] = True
                    class_found = True
                    continue
                # Pattern 2: class Name(Agent[InputType, OutputType]):
                elif re.match(r'^class (\w+)\(Agent\[', line):
                    class_match = re.match(r'^class (\w+)\(Agent\[', line)
                    class_name = class_match.group(1)
                    # Insert OperationalMonitoringMixin before Agent
                    new_line = line.replace("(Agent[", "(OperationalMonitoringMixin, Agent[")
                    new_lines.append(new_line)
                    changes["mixin_added"] = True
                    class_found = True
                    continue

            # Step 3: Add setup_monitoring() call in __init__
            if class_found and not init_found and re.match(r'\s+def __init__\(', line):
                init_found = True
                new_lines.append(line)

                # Get indentation
                indent_match = re.match(r'(\s+)', line)
                if indent_match:
                    indent = indent_match.group(1)

                # Look ahead for super().__init__() call or end of __init__
                j = i + 1
                found_super = False
                init_end_line = None

                while j < len(lines):
                    next_line = lines[j]

                    # Check if we've reached the end of __init__ (next method)
                    if next_line.strip() and not next_line.strip().startswith('#') and \
                       re.match(r'\s+def ', next_line) and j > i + 1:
                        init_end_line = j
                        break

                    new_lines.append(next_line)

                    if "super().__init__" in next_line:
                        # Add setup_monitoring after super().__init__()
                        actual_agent_name = agent_file.stem  # Get actual filename without .py
                        setup_call = f"{indent}    self.setup_monitoring(agent_name=\"{actual_agent_name}\")"
                        new_lines.append(setup_call)
                        changes["setup_monitoring_added"] = True
                        found_super = True
                        break

                    j += 1

                # If no super().__init__() found, add at the end of __init__
                if not found_super and init_end_line is not None:
                    # Add setup_monitoring before the next method
                    actual_agent_name = agent_file.stem
                    setup_call = f"{indent}    self.setup_monitoring(agent_name=\"{actual_agent_name}\")"
                    new_lines.insert(len(new_lines) - 1, setup_call)  # Insert before last line (next method)
                    new_lines.insert(len(new_lines) - 1, "")  # Add blank line
                    changes["setup_monitoring_added"] = True

                # Skip lines we already processed
                while i < j:
                    i += 1
                    if i < len(lines):
                        line = lines[i]

                continue

            # Step 4: Wrap execute() or run() method with tracking
            if class_found and not execute_found and (re.match(r'\s+def execute\(', line) or re.match(r'\s+def run\(', line)):
                execute_found = True
                new_lines.append(line)

                # Extract parameter name from method signature
                param_match = re.search(r'def (?:execute|run)\(self,\s*(\w+)', line)
                param_name = param_match.group(1) if param_match else "input_data"

                # Get indentation
                indent_match = re.match(r'(\s+)', line)
                if indent_match:
                    method_indent = indent_match.group(1)

                # Look for the method body
                j = i + 1
                method_body_start = j

                # Find first line of actual method body (skip docstring)
                in_docstring = False
                while j < len(lines):
                    next_line = lines[j]

                    if '"""' in next_line or "'''" in next_line:
                        in_docstring = not in_docstring
                        new_lines.append(next_line)
                        j += 1
                        continue

                    if not in_docstring and next_line.strip() and not next_line.strip().startswith('#'):
                        # Found first real line of method
                        # Add tracking wrapper with actual parameter name
                        new_lines.append(f"{method_indent}    with self.track_execution({param_name}) as tracker:")

                        # Indent the rest of the method body
                        method_end = self._find_method_end(lines, j, method_indent)

                        for k in range(j, method_end):
                            method_line = lines[k]
                            # Add extra indent
                            if method_line.strip():
                                new_lines.append(f"    {method_line}")
                            else:
                                new_lines.append(method_line)

                        changes["tracking_added"] = True

                        # Skip to end of method
                        i = method_end - 1
                        break

                    new_lines.append(next_line)
                    j += 1

                continue

            new_lines.append(line)

        # Create modified content
        modified_content = '\n'.join(new_lines)

        if dry_run:
            return True, f"[DRY RUN] Would modify {agent_file}", changes

        # Create backup
        backup_file = agent_file.with_suffix('.py.backup')
        shutil.copy2(agent_file, backup_file)
        changes["backup_created"] = True

        # Write modified content
        with open(agent_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)

        return True, f"Successfully added monitoring to {agent_name}_agent", changes

    def _find_method_end(self, lines: List[str], start_idx: int, method_indent: str) -> int:
        """Find the end of a method by tracking indentation.

        Args:
            lines: All lines in the file
            start_idx: Starting index
            method_indent: Base indentation of the method

        Returns:
            Index of the line after the method ends
        """
        method_indent_level = len(method_indent)

        for i in range(start_idx, len(lines)):
            line = lines[i]

            # Empty lines don't count
            if not line.strip():
                continue

            # Check if we've de-dented back to class level or less
            if line.strip() and not line.startswith(' ' * (method_indent_level + 4)):
                return i

        return len(lines)

    def create_changelog(
        self,
        agent_name: str,
        version: str = "1.0.0",
        dry_run: bool = False
    ) -> Tuple[bool, str]:
        """Create CHANGELOG.md for an agent.

        Args:
            agent_name: Name of the agent
            version: Initial version number
            dry_run: If True, only simulate creation

        Returns:
            Tuple of (success, message)
        """
        agent_dir = self.agents_dir

        # Determine actual agent file name
        agent_file_name = None
        for pattern in [f"{agent_name}_agent.py", f"{agent_name}_agent_ai.py", f"{agent_name}.py"]:
            if (self.agents_dir / pattern).exists():
                agent_file_name = pattern[:-3]  # Remove .py
                break

        if not agent_file_name:
            agent_file_name = f"{agent_name}_agent"

        changelog_file = agent_dir / f"CHANGELOG_{agent_file_name}.md"

        # Check if changelog already exists
        if changelog_file.exists():
            return True, f"Changelog already exists: {changelog_file}"

        # Read template
        with open(self.changelog_template, 'r', encoding='utf-8') as f:
            template_content = f.read()

        # Customize template
        agent_display_name = agent_file_name.replace("_", " ").title()
        today = DeterministicClock.now().strftime("%Y-%m-%d")

        # Replace placeholders (if any specific to agent)
        customized_content = template_content

        # Add agent-specific header
        header = f"# {agent_display_name} - Changelog\n\n"
        header += f"**Agent:** {agent_file_name}\n"
        header += f"**Initial Version:** {version}\n"
        header += f"**Created:** {today}\n\n"
        header += "---\n\n"

        customized_content = header + customized_content

        if dry_run:
            return True, f"[DRY RUN] Would create {changelog_file}"

        # Write changelog
        with open(changelog_file, 'w', encoding='utf-8') as f:
            f.write(customized_content)

        return True, f"Created changelog: {changelog_file}"

    def verify_integration(self, agent_name: str) -> Dict[str, Any]:
        """Verify that monitoring and changelog are properly integrated.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with verification results
        """
        # Try to find the agent file with multiple patterns
        agent_file = None
        agent_file_name = None
        for pattern in [f"{agent_name}_agent.py", f"{agent_name}_agent_ai.py", f"{agent_name}.py"]:
            test_file = self.agents_dir / pattern
            if test_file.exists():
                agent_file = test_file
                agent_file_name = pattern[:-3]  # Remove .py
                break

        if not agent_file:
            agent_file_name = f"{agent_name}_agent"
            agent_file = self.agents_dir / f"{agent_file_name}.py"

        changelog_file = self.agents_dir / f"CHANGELOG_{agent_file_name}.md"

        results = {
            "agent_exists": agent_file.exists() if agent_file else False,
            "monitoring_imported": False,
            "mixin_inherited": False,
            "setup_called": False,
            "tracking_used": False,
            "changelog_exists": changelog_file.exists(),
            "all_checks_passed": False
        }

        if not results["agent_exists"]:
            return results

        # Read agent file
        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check monitoring integration
        results["monitoring_imported"] = "OperationalMonitoringMixin" in content
        results["mixin_inherited"] = "OperationalMonitoringMixin" in content and "BaseAgent" in content
        results["setup_called"] = "setup_monitoring" in content
        results["tracking_used"] = "track_execution" in content

        # Overall status
        results["all_checks_passed"] = all([
            results["agent_exists"],
            results["monitoring_imported"],
            results["mixin_inherited"],
            results["setup_called"],
            results["tracking_used"],
            results["changelog_exists"]
        ])

        return results

    def generate_integration_report(
        self,
        agent_name: str,
        changes: Dict[str, Any],
        verification: Dict[str, Any]
    ) -> str:
        """Generate an integration report.

        Args:
            agent_name: Name of the agent
            changes: Changes made during integration
            verification: Verification results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append(f"INTEGRATION REPORT: {agent_name}_agent")
        report.append("=" * 70)
        report.append("")

        report.append("CHANGES APPLIED:")
        for change, applied in changes.items():
            status = "[OK]" if applied else "[FAIL]"
            report.append(f"  {status} {change.replace('_', ' ').title()}")

        report.append("")
        report.append("VERIFICATION:")
        for check, passed in verification.items():
            if check == "all_checks_passed":
                continue
            status = "[OK]" if passed else "[FAIL]"
            report.append(f"  {status} {check.replace('_', ' ').title()}")

        report.append("")
        overall = "PASSED" if verification.get("all_checks_passed") else "FAILED"
        report.append(f"OVERALL STATUS: {overall}")
        report.append("=" * 70)

        return "\n".join(report)

    def integrate_agent(
        self,
        agent_name: str,
        version: str = "1.0.0",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Perform complete integration (monitoring + changelog).

        Args:
            agent_name: Name of the agent
            version: Version number for changelog
            dry_run: If True, simulate changes only

        Returns:
            Dictionary with integration results
        """
        results = {
            "agent": agent_name,
            "success": False,
            "monitoring_integration": {},
            "changelog_creation": {},
            "verification": {},
            "report": ""
        }

        # Add monitoring
        success, message, changes = self.add_monitoring_to_agent(agent_name, dry_run)
        results["monitoring_integration"] = {
            "success": success,
            "message": message,
            "changes": changes
        }

        # Create changelog
        success, message = self.create_changelog(agent_name, version, dry_run)
        results["changelog_creation"] = {
            "success": success,
            "message": message
        }

        # Verify integration (skip in dry-run)
        if not dry_run:
            verification = self.verify_integration(agent_name)
            results["verification"] = verification
            results["success"] = verification.get("all_checks_passed", False)

            # Generate report
            results["report"] = self.generate_integration_report(
                agent_name,
                changes,
                verification
            )
        else:
            results["success"] = True
            results["report"] = f"[DRY RUN] Integration simulation for {agent_name}_agent"

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add monitoring and changelog to GreenLang agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Integrate monitoring for a single agent
  python scripts/add_monitoring_and_changelog.py --agent carbon

  # Integrate all agents
  python scripts/add_monitoring_and_changelog.py --all-agents

  # Dry run (simulate changes)
  python scripts/add_monitoring_and_changelog.py --agent fuel --dry-run

  # Specify version
  python scripts/add_monitoring_and_changelog.py --agent boiler --version 1.2.0

  # Generate report only
  python scripts/add_monitoring_and_changelog.py --agent grid_factor --verify-only
        """
    )

    parser.add_argument(
        "--agent",
        help="Agent name (e.g., carbon, fuel, boiler)"
    )

    parser.add_argument(
        "--all-agents",
        action="store_true",
        help="Integrate all agents"
    )

    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Version number for changelog (default: 1.0.0)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate changes without modifying files"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify integration status"
    )

    parser.add_argument(
        "--output-json",
        help="Output results as JSON to specified file"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.agent and not args.all_agents:
        parser.error("Must specify --agent or --all-agents")

    if args.agent and args.all_agents:
        parser.error("Cannot specify both --agent and --all-agents")

    # Find workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent

    # Initialize integrator
    try:
        integrator = MonitoringIntegrator(workspace_root)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Determine which agents to process
    if args.all_agents:
        agents = integrator.find_all_agents()
        if not agents:
            print("No agents found!", file=sys.stderr)
            return 1
        print(f"Found {len(agents)} agents: {', '.join(agents)}")
    else:
        agents = [args.agent]

    # Process each agent
    all_results = []

    for agent_name in agents:
        print(f"\nProcessing {agent_name}_agent...")

        if args.verify_only:
            # Verification only
            verification = integrator.verify_integration(agent_name)
            print(f"\nVerification Results for {agent_name}_agent:")
            for check, passed in verification.items():
                status = "[OK]" if passed else "[FAIL]"
                print(f"  {status} {check.replace('_', ' ').title()}")

            all_results.append({
                "agent": agent_name,
                "verification": verification
            })
        else:
            # Full integration
            results = integrator.integrate_agent(agent_name, args.version, args.dry_run)
            all_results.append(results)

            # Print report
            print(results["report"])

    # Output JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if args.verify_only:
        passed = sum(1 for r in all_results if r["verification"].get("all_checks_passed"))
        print(f"Agents verified: {len(all_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {len(all_results) - passed}")
    else:
        successful = sum(1 for r in all_results if r["success"])
        print(f"Agents processed: {len(all_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(all_results) - successful}")

    # Return exit code
    if args.verify_only:
        return 0 if all(r["verification"].get("all_checks_passed") for r in all_results) else 1
    else:
        return 0 if all(r["success"] for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
