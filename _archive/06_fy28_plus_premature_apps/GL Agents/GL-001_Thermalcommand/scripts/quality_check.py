#!/usr/bin/env python3
"""
GL-001 ThermalCommand - Local Quality Check Script

This script runs all quality checks locally, matching the CI pipeline.
Use this before committing to ensure your code passes all quality gates.

Usage:
    python scripts/quality_check.py           # Run all checks
    python scripts/quality_check.py --quick   # Run quick checks only
    python scripts/quality_check.py --fix     # Run with auto-fix enabled
    python scripts/quality_check.py --check black ruff  # Run specific checks

Requirements:
    pip install black ruff mypy pytest pytest-cov bandit

Author: GreenLang DevOps Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


class CheckResult(Enum):
    """Result of a quality check."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    WARNING = "WARNING"


@dataclass
class CheckOutput:
    """Output from a quality check."""

    name: str
    result: CheckResult
    duration: float
    message: str = ""
    details: str = ""


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_check_start(name: str) -> None:
    """Print check start message."""
    print(f"{Colors.CYAN}[RUNNING]{Colors.END} {name}...", end=" ", flush=True)


def print_check_result(output: CheckOutput) -> None:
    """Print check result."""
    if output.result == CheckResult.PASSED:
        status = f"{Colors.GREEN}[PASSED]{Colors.END}"
    elif output.result == CheckResult.FAILED:
        status = f"{Colors.RED}[FAILED]{Colors.END}"
    elif output.result == CheckResult.WARNING:
        status = f"{Colors.YELLOW}[WARNING]{Colors.END}"
    else:
        status = f"{Colors.YELLOW}[SKIPPED]{Colors.END}"

    print(f"\r{status} {output.name} ({output.duration:.2f}s)")

    if output.message:
        print(f"         {output.message}")

    if output.details and output.result == CheckResult.FAILED:
        print(f"\n{Colors.RED}Details:{Colors.END}")
        # Limit output to avoid flooding terminal
        lines = output.details.strip().split("\n")
        if len(lines) > 50:
            for line in lines[:25]:
                print(f"  {line}")
            print(f"  ... ({len(lines) - 50} more lines)")
            for line in lines[-25:]:
                print(f"  {line}")
        else:
            for line in lines:
                print(f"  {line}")


def run_command(
    cmd: list[str],
    check_name: str,
    capture_output: bool = True,
) -> CheckOutput:
    """Run a command and return the result."""
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            return CheckOutput(
                name=check_name,
                result=CheckResult.PASSED,
                duration=duration,
            )
        else:
            return CheckOutput(
                name=check_name,
                result=CheckResult.FAILED,
                duration=duration,
                details=result.stdout + result.stderr,
            )

    except subprocess.TimeoutExpired:
        return CheckOutput(
            name=check_name,
            result=CheckResult.FAILED,
            duration=300.0,
            message="Timeout after 5 minutes",
        )
    except FileNotFoundError:
        return CheckOutput(
            name=check_name,
            result=CheckResult.SKIPPED,
            duration=0.0,
            message=f"Tool not found: {cmd[0]}",
        )
    except Exception as e:
        return CheckOutput(
            name=check_name,
            result=CheckResult.FAILED,
            duration=time.time() - start_time,
            message=str(e),
        )


def check_black(fix: bool = False) -> CheckOutput:
    """Run Black formatting check."""
    cmd = ["python", "-m", "black"]
    if not fix:
        cmd.extend(["--check", "--diff"])
    cmd.extend(["--config", "pyproject.toml", "."])

    return run_command(cmd, "Black (Formatting)")


def check_ruff(fix: bool = False) -> CheckOutput:
    """Run Ruff linting check."""
    cmd = ["python", "-m", "ruff", "check"]
    if fix:
        cmd.append("--fix")
    cmd.extend(["--config", "pyproject.toml", "."])

    return run_command(cmd, "Ruff (Linting)")


def check_ruff_format(fix: bool = False) -> CheckOutput:
    """Run Ruff format check."""
    cmd = ["python", "-m", "ruff", "format"]
    if not fix:
        cmd.append("--check")
    cmd.extend(["--config", "pyproject.toml", "."])

    return run_command(cmd, "Ruff (Format Check)")


def check_mypy() -> CheckOutput:
    """Run MyPy type checking."""
    cmd = [
        "python",
        "-m",
        "mypy",
        "--config-file",
        "pyproject.toml",
        "--install-types",
        "--non-interactive",
        "--exclude",
        "tests/",
        "--exclude",
        "deployment/",
        ".",
    ]

    return run_command(cmd, "MyPy (Type Checking)")


def check_pytest(quick: bool = False) -> CheckOutput:
    """Run Pytest with coverage."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-x",
    ]

    if quick:
        cmd.extend(["-m", "not slow and not integration and not requires_hardware"])
        cmd.extend(["--timeout=30"])
    else:
        cmd.extend(["--cov=.", "--cov-config=pyproject.toml"])
        cmd.extend(["--cov-report=term-missing"])
        cmd.extend(["--cov-fail-under=85"])

    return run_command(cmd, "Pytest (Tests + Coverage)")


def check_bandit() -> CheckOutput:
    """Run Bandit security check."""
    cmd = [
        "python",
        "-m",
        "bandit",
        "-c",
        "pyproject.toml",
        "-r",
        ".",
        "--exclude",
        "./tests,./deployment,./.venv",
        "-ll",
    ]

    return run_command(cmd, "Bandit (Security)")


def check_interrogate() -> CheckOutput:
    """Run Interrogate docstring coverage check."""
    cmd = [
        "python",
        "-m",
        "interrogate",
        "--config",
        "pyproject.toml",
        "--fail-under=80",
        ".",
    ]

    return run_command(cmd, "Interrogate (Docstring Coverage)")


# Available checks
CHECKS: dict[str, Callable[..., CheckOutput]] = {
    "black": check_black,
    "ruff": check_ruff,
    "ruff-format": check_ruff_format,
    "mypy": check_mypy,
    "pytest": check_pytest,
    "bandit": check_bandit,
    "interrogate": check_interrogate,
}

QUICK_CHECKS = ["black", "ruff", "ruff-format"]
FULL_CHECKS = ["black", "ruff", "ruff-format", "mypy", "pytest", "bandit"]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GL-001 ThermalCommand Quality Check Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quality_check.py              # Run all checks
  python scripts/quality_check.py --quick      # Run quick checks only
  python scripts/quality_check.py --fix        # Run with auto-fix enabled
  python scripts/quality_check.py black ruff   # Run specific checks
        """,
    )

    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick checks only (black, ruff)",
    )
    parser.add_argument(
        "--fix",
        "-f",
        action="store_true",
        help="Enable auto-fix mode for applicable tools",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "checks",
        nargs="*",
        choices=list(CHECKS.keys()),
        help="Specific checks to run (default: all)",
    )

    args = parser.parse_args()

    # Determine which checks to run
    if args.checks:
        checks_to_run = args.checks
    elif args.quick:
        checks_to_run = QUICK_CHECKS
    else:
        checks_to_run = FULL_CHECKS

    print_header("GL-001 ThermalCommand Quality Checks")
    print(f"Running checks: {', '.join(checks_to_run)}")
    print(f"Fix mode: {'enabled' if args.fix else 'disabled'}")
    print()

    # Run checks
    results: list[CheckOutput] = []
    total_start = time.time()

    for check_name in checks_to_run:
        check_func = CHECKS[check_name]
        print_check_start(check_name)

        # Pass fix argument to applicable checks
        if check_name in ["black", "ruff", "ruff-format"]:
            output = check_func(fix=args.fix)
        elif check_name == "pytest":
            output = check_func(quick=args.quick)
        else:
            output = check_func()

        print_check_result(output)
        results.append(output)

    total_duration = time.time() - total_start

    # Summary
    print_header("Summary")

    passed = sum(1 for r in results if r.result == CheckResult.PASSED)
    failed = sum(1 for r in results if r.result == CheckResult.FAILED)
    skipped = sum(1 for r in results if r.result == CheckResult.SKIPPED)
    warnings = sum(1 for r in results if r.result == CheckResult.WARNING)

    print(f"Total checks: {len(results)}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.END}")
    if skipped > 0:
        print(f"{Colors.YELLOW}Skipped: {skipped}{Colors.END}")
    if warnings > 0:
        print(f"{Colors.YELLOW}Warnings: {warnings}{Colors.END}")
    print(f"Total time: {total_duration:.2f}s")

    # Exit code
    if failed > 0:
        print(f"\n{Colors.RED}{Colors.BOLD}Quality gate FAILED{Colors.END}")
        return 1
    elif skipped > 0:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Quality gate passed with warnings{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}Quality gate PASSED{Colors.END}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
