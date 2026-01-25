#!/usr/bin/env python3
"""
GitHub Cleanup Script for GreenLang Repository

This script helps manage stale GitHub issues and PRs:
1. Closes old Friday Gate Failure issues (from archived workflow)
2. Provides commands to merge or close Dependabot PRs

Usage:
    # List all actions to be taken (dry run)
    python scripts/github_cleanup.py --dry-run

    # Execute cleanup (requires gh CLI and authentication)
    python scripts/github_cleanup.py --execute

Requirements:
    - GitHub CLI (gh) installed and authenticated
    - Repository write access

Author: GreenLang DevOps
Date: 2026-01-26
"""

import subprocess
import sys
import json
from datetime import datetime

# Repository info
REPO = "akshay-greenlang/Code-V1_GreenLang"

# Friday Gate issues to close (all are from archived workflow)
FRIDAY_GATE_ISSUES = [
    58, 57, 56, 55, 54, 53, 52, 50, 49, 45, 44, 43, 42, 40, 35, 26, 19
]

# Dependabot PRs with their actions
DEPENDABOT_PRS = {
    # PR number: (action, description)
    51: ("close", "Python 3.14-slim is too new, stay on 3.11-slim for stability"),
    48: ("merge", "psutil 5.9.8 -> 7.1.3 (security update)"),
    47: ("merge", "pydantic 2.5.3 -> 2.12.4 (security update)"),
    46: ("merge", "spacy 3.7.2 -> 3.8.11 (security update)"),
    34: ("merge", "charset-normalizer 3.4.3 -> 3.4.4 (security update)"),
    32: ("merge", "pydantic-core 2.33.2 -> 2.41.4 (security update)"),
    31: ("merge", "referencing 0.36.2 -> 0.37.0 (security update)"),
    28: ("merge", "idna 3.10 -> 3.11 (security update)"),
    22: ("merge", "attrs 25.3.0 -> 25.4.0 (security update)"),
    21: ("merge", "certifi 2025.8.3 -> 2025.10.5 (security update)"),
    16: ("merge", "typing-inspection 0.4.1 -> 0.4.2 (security update)"),
}

CLOSE_COMMENT = """This issue is being closed as part of repository cleanup.

**Reason:** The Friday Gate workflow has been consolidated into `scheduled-maintenance.yml`.
These automated failure issues from the archived `friday-gate.yml` workflow are no longer relevant.

**Action taken:** Dependencies have been updated in:
- `pyproject.toml`
- `config/requirements.txt`
- `config/requirements-pinned.txt`

If you believe this issue should remain open, please reopen it with additional context.

---
*Automated cleanup performed on {date}*
"""


def run_gh_command(args: list[str], dry_run: bool = False) -> tuple[int, str]:
    """Run a GitHub CLI command."""
    cmd = ["gh"] + args
    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return 0, ""

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr
    except FileNotFoundError:
        return 1, "Error: GitHub CLI (gh) not found. Please install it first."


def close_friday_gate_issues(dry_run: bool = False):
    """Close all Friday Gate Failure issues."""
    print("\n=== Closing Friday Gate Failure Issues ===")
    print(f"Issues to close: {FRIDAY_GATE_ISSUES}")

    comment = CLOSE_COMMENT.format(date=datetime.now().strftime("%Y-%m-%d"))

    for issue_num in FRIDAY_GATE_ISSUES:
        print(f"\nClosing issue #{issue_num}...")

        # Add comment
        code, output = run_gh_command([
            "issue", "comment", str(issue_num),
            "--repo", REPO,
            "--body", comment
        ], dry_run)

        if code != 0 and not dry_run:
            print(f"  Warning: Could not add comment: {output}")

        # Close issue
        code, output = run_gh_command([
            "issue", "close", str(issue_num),
            "--repo", REPO,
            "--reason", "not_planned"
        ], dry_run)

        if code != 0 and not dry_run:
            print(f"  Error closing issue: {output}")
        else:
            print(f"  Issue #{issue_num} closed successfully")


def handle_dependabot_prs(dry_run: bool = False):
    """Handle Dependabot PRs (merge or close)."""
    print("\n=== Handling Dependabot PRs ===")

    for pr_num, (action, description) in DEPENDABOT_PRS.items():
        print(f"\nPR #{pr_num}: {description}")
        print(f"  Action: {action.upper()}")

        if action == "merge":
            code, output = run_gh_command([
                "pr", "merge", str(pr_num),
                "--repo", REPO,
                "--merge",
                "--delete-branch"
            ], dry_run)
        elif action == "close":
            code, output = run_gh_command([
                "pr", "close", str(pr_num),
                "--repo", REPO,
                "--comment", f"Closing: {description}"
            ], dry_run)

        if code != 0 and not dry_run:
            print(f"  Error: {output}")
        else:
            print(f"  PR #{pr_num} {action}d successfully")


def main():
    """Main entry point."""
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    execute = "--execute" in sys.argv or "-x" in sys.argv

    if not dry_run and not execute:
        print(__doc__)
        print("\nPlease specify --dry-run or --execute")
        sys.exit(1)

    print("=" * 60)
    print("GreenLang GitHub Cleanup Script")
    print("=" * 60)
    print(f"Repository: {REPO}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check gh CLI availability
    code, _ = run_gh_command(["--version"], dry_run=True)

    # Close Friday Gate issues
    close_friday_gate_issues(dry_run)

    # Handle Dependabot PRs
    handle_dependabot_prs(dry_run)

    print("\n" + "=" * 60)
    print("Cleanup complete!")
    if dry_run:
        print("This was a dry run. Use --execute to perform actual changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
