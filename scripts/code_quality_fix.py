#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Code Quality Automation Script

This script automates the critical code quality fixes for GreenLang:
1. Auto-formats code with black
2. Removes unused imports with autoflake
3. Identifies hardcoded Unix paths
4. Finds bare except clauses
5. Generates a report of issues found and fixed

Usage:
    python scripts/code_quality_fix.py [--fix] [--report-only]

    --fix: Apply automatic fixes where safe
    --report-only: Only generate reports, don't modify files
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import argparse
import re
from greenlang.determinism import DeterministicClock


class CodeQualityFixer:
    def __init__(self, project_root: Path, fix_mode: bool = False):
        self.project_root = project_root
        self.fix_mode = fix_mode
        self.report = {
            "timestamp": None,
            "issues_found": {},
            "fixes_applied": {},
            "summary": {}
        }

    def run_black_formatter(self) -> Dict[str, Any]:
        """Run black code formatter on greenlang/ and core/ directories."""
        result = {"status": "success", "files_formatted": 0, "errors": []}

        try:
            if self.fix_mode:
                # Run black formatter
                cmd = ["black", "greenlang/", "core/"]
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )

                if process.returncode == 0:
                    # Count reformatted files from output
                    output_lines = process.stdout.split('\n')
                    formatted_count = len([line for line in output_lines if "reformatted" in line])
                    result["files_formatted"] = formatted_count
                else:
                    result["status"] = "error"
                    result["errors"].append(process.stderr)
            else:
                # Check mode only
                cmd = ["black", "--check", "--diff", "greenlang/", "core/"]
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )

                if process.returncode != 0:
                    result["needs_formatting"] = True
                    result["diff"] = process.stdout
                else:
                    result["needs_formatting"] = False

        except FileNotFoundError:
            result["status"] = "error"
            result["errors"].append("Black not installed. Run: pip install black")

        return result

    def run_autoflake(self) -> Dict[str, Any]:
        """Run autoflake to remove unused imports."""
        result = {"status": "success", "changes_made": False, "errors": []}

        try:
            if self.fix_mode:
                cmd = [
                    "autoflake",
                    "--remove-all-unused-imports",
                    "--in-place",
                    "-r",
                    "greenlang/",
                    "core/"
                ]
            else:
                cmd = [
                    "autoflake",
                    "--remove-all-unused-imports",
                    "--check-diff",
                    "-r",
                    "greenlang/",
                    "core/"
                ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if process.stdout.strip():
                result["changes_made"] = True
                result["output"] = process.stdout

        except FileNotFoundError:
            result["status"] = "error"
            result["errors"].append("Autoflake not installed. Run: pip install autoflake")

        return result

    def find_hardcoded_unix_paths(self) -> Dict[str, Any]:
        """Find hardcoded Unix paths that need cross-platform fixes."""
        result = {"status": "success", "issues": [], "errors": []}

        # Patterns to search for
        patterns = [
            r'Path\("/etc/[^"]*"\)',
            r'Path\("/tmp/[^"]*"\)',
            r'"/etc/[^"]*"',
            r'"/tmp/[^"]*"',
            r"'/etc/[^']*'",
            r"'/tmp/[^']*'"
        ]

        # Directories to search
        search_dirs = ["greenlang", "core"]

        for search_dir in search_dirs:
            search_path = self.project_root / search_dir
            if not search_path.exists():
                continue

            for py_file in search_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    lines = content.split('\n')

                    for line_num, line in enumerate(lines, 1):
                        for pattern in patterns:
                            matches = re.finditer(pattern, line)
                            for match in matches:
                                result["issues"].append({
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": line_num,
                                    "content": line.strip(),
                                    "match": match.group(),
                                    "type": "hardcoded_unix_path"
                                })

                except Exception as e:
                    result["errors"].append(f"Error reading {py_file}: {e}")

        return result

    def find_bare_except_clauses(self) -> Dict[str, Any]:
        """Find bare except clauses that should be more specific."""
        result = {"status": "success", "issues": [], "errors": []}

        # Pattern for bare except
        pattern = r'^\s*except\s*:'

        # Directories to search
        search_dirs = ["greenlang", "core"]

        for search_dir in search_dirs:
            search_path = self.project_root / search_dir
            if not search_path.exists():
                continue

            for py_file in search_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    lines = content.split('\n')

                    for line_num, line in enumerate(lines, 1):
                        if re.match(pattern, line):
                            result["issues"].append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": line_num,
                                "content": line.strip(),
                                "type": "bare_except"
                            })

                except Exception as e:
                    result["errors"].append(f"Error reading {py_file}: {e}")

        return result

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        import datetime

        self.report["timestamp"] = datetime.DeterministicClock.now().isoformat()

        # Run all checks
        print("Running black formatter...")
        self.report["issues_found"]["black"] = self.run_black_formatter()

        print("Running autoflake...")
        self.report["issues_found"]["autoflake"] = self.run_autoflake()

        print("Checking for hardcoded Unix paths...")
        self.report["issues_found"]["unix_paths"] = self.find_hardcoded_unix_paths()

        print("Checking for bare except clauses...")
        self.report["issues_found"]["bare_except"] = self.find_bare_except_clauses()

        # Generate summary
        self.report["summary"] = {
            "total_hardcoded_paths": len(self.report["issues_found"]["unix_paths"]["issues"]),
            "total_bare_except": len(self.report["issues_found"]["bare_except"]["issues"]),
            "black_formatting_needed": self.report["issues_found"]["black"].get("needs_formatting", False),
            "autoflake_changes_needed": self.report["issues_found"]["autoflake"]["changes_made"]
        }

        return self.report

    def save_report(self, output_file: Path):
        """Save report to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2)

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "="*60)
        print("CODE QUALITY REPORT SUMMARY")
        print("="*60)

        summary = self.report["summary"]

        print(f"Hardcoded Unix paths found: {summary['total_hardcoded_paths']}")
        print(f"Bare except clauses found: {summary['total_bare_except']}")
        print(f"Black formatting needed: {'Yes' if summary['black_formatting_needed'] else 'No'}")
        print(f"Autoflake changes needed: {'Yes' if summary['autoflake_changes_needed'] else 'No'}")

        if summary["total_hardcoded_paths"] > 0:
            print("\nHardcoded Unix paths:")
            for issue in self.report["issues_found"]["unix_paths"]["issues"][:5]:
                print(f"  {issue['file']}:{issue['line']} - {issue['match']}")
            if summary["total_hardcoded_paths"] > 5:
                print(f"  ... and {summary['total_hardcoded_paths'] - 5} more")

        if summary["total_bare_except"] > 0:
            print("\nBare except clauses:")
            for issue in self.report["issues_found"]["bare_except"]["issues"][:5]:
                print(f"  {issue['file']}:{issue['line']} - {issue['content']}")
            if summary["total_bare_except"] > 5:
                print(f"  ... and {summary['total_bare_except'] - 5} more")

        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="GreenLang Code Quality Automation")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply automatic fixes where safe"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate reports, don't modify files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="code_quality_report.json",
        help="Output file for JSON report"
    )

    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).parent.parent

    # Create fixer instance
    fix_mode = args.fix and not args.report_only
    fixer = CodeQualityFixer(project_root, fix_mode=fix_mode)

    print(f"GreenLang Code Quality Checker")
    print(f"Project root: {project_root}")
    print(f"Mode: {'Fix' if fix_mode else 'Report Only'}")
    print("-" * 60)

    # Generate report
    report = fixer.generate_report()

    # Save report
    output_path = project_root / args.output
    fixer.save_report(output_path)
    print(f"\nDetailed report saved to: {output_path}")

    # Print summary
    fixer.print_summary()

    # Return appropriate exit code
    summary = report["summary"]
    issues_found = (
        summary["total_hardcoded_paths"] > 0 or
        summary["total_bare_except"] > 0 or
        summary["black_formatting_needed"] or
        summary["autoflake_changes_needed"]
    )

    if issues_found and not fix_mode:
        print("\nRecommendation: Run with --fix to apply automatic fixes")
        return 1
    elif issues_found and fix_mode:
        print("\nFixes applied! Re-run to verify.")
        return 0
    else:
        print("\nCode quality looks good!")
        return 0


if __name__ == "__main__":
    sys.exit(main())