#!/usr/bin/env python3
"""
GitHub API Cleanup Script for GreenLang Repository

This script closes stale Friday Gate Failure issues and manages Dependabot PRs
using the GitHub REST API directly.

Usage:
    # Set your GitHub token as environment variable
    set GITHUB_TOKEN=ghp_your_token_here

    # Run the script
    python scripts/github_api_cleanup.py

    # Or pass token as argument
    python scripts/github_api_cleanup.py --token ghp_your_token_here

Requirements:
    - requests library (pip install requests)
    - GitHub Personal Access Token with 'repo' scope

Author: GreenLang DevOps
Date: 2026-01-26
"""

import os
import sys
import json
import argparse
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# Repository info
OWNER = "akshay-greenlang"
REPO = "Code-V1_GreenLang"
API_BASE = "https://api.github.com"

# Friday Gate issues to close (all are from archived workflow)
FRIDAY_GATE_ISSUES = [58, 57, 56, 55, 54, 53, 52, 50, 49, 45, 44, 43, 42, 40, 35, 26, 19]

# Dependabot PRs - Close all since updates were applied directly
DEPENDABOT_PRS = [51, 48, 47, 46, 34, 32, 31, 28, 22, 21, 16]

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

PR_CLOSE_COMMENT = """This PR is being closed as the dependency updates have been applied directly to the main branch.

**Updates applied in commit:** Dependencies updated to latest secure versions

The following security updates were merged:
- pydantic 2.5.3 → 2.12.4
- psutil 5.9.8 → 7.1.3
- spacy 3.7.2 → 3.8.11
- pydantic-core 2.33.2 → 2.41.4
- And other transitive dependencies

Thank you Dependabot for keeping our dependencies secure!

---
*Automated cleanup performed on {date}*
"""


class GitHubClient:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        url = f"{API_BASE}{endpoint}"
        response = self.session.request(method, url, **kwargs)

        if response.status_code == 401:
            print("Error: Invalid GitHub token. Please check your token has 'repo' scope.")
            sys.exit(1)

        if response.status_code == 404:
            return {"error": "not_found", "status": 404}

        if response.status_code >= 400:
            print(f"API Error: {response.status_code} - {response.text}")
            return {"error": response.text, "status": response.status_code}

        if response.text:
            return response.json()
        return {"status": response.status_code}

    def get_issue(self, issue_number: int) -> dict:
        return self._request("GET", f"/repos/{OWNER}/{REPO}/issues/{issue_number}")

    def close_issue(self, issue_number: int, comment: str = None) -> bool:
        # Add comment if provided
        if comment:
            self._request(
                "POST",
                f"/repos/{OWNER}/{REPO}/issues/{issue_number}/comments",
                json={"body": comment}
            )

        # Close the issue
        result = self._request(
            "PATCH",
            f"/repos/{OWNER}/{REPO}/issues/{issue_number}",
            json={"state": "closed", "state_reason": "not_planned"}
        )
        return "error" not in result

    def get_pr(self, pr_number: int) -> dict:
        return self._request("GET", f"/repos/{OWNER}/{REPO}/pulls/{pr_number}")

    def close_pr(self, pr_number: int, comment: str = None) -> bool:
        # Add comment if provided
        if comment:
            self._request(
                "POST",
                f"/repos/{OWNER}/{REPO}/issues/{pr_number}/comments",
                json={"body": comment}
            )

        # Close the PR
        result = self._request(
            "PATCH",
            f"/repos/{OWNER}/{REPO}/pulls/{pr_number}",
            json={"state": "closed"}
        )
        return "error" not in result

    def verify_auth(self) -> bool:
        result = self._request("GET", "/user")
        if "login" in result:
            print(f"Authenticated as: {result['login']}")
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description="GitHub API Cleanup for GreenLang")
    parser.add_argument("--token", help="GitHub Personal Access Token")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview actions without executing")
    parser.add_argument("--issues-only", action="store_true", help="Only close issues")
    parser.add_argument("--prs-only", action="store_true", help="Only close PRs")
    args = parser.parse_args()

    # Get token from argument or environment
    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if not token:
        print("Error: No GitHub token provided.")
        print("Set GITHUB_TOKEN environment variable or use --token argument")
        print("\nTo create a token:")
        print("1. Go to: https://github.com/settings/tokens/new?scopes=repo")
        print("2. Create a token with 'repo' scope")
        print("3. Run: set GITHUB_TOKEN=your_token_here")
        sys.exit(1)

    print("=" * 60)
    print("GreenLang GitHub Cleanup Script")
    print("=" * 60)
    print(f"Repository: {OWNER}/{REPO}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    client = GitHubClient(token)

    # Verify authentication
    if not client.verify_auth():
        print("Authentication failed!")
        sys.exit(1)

    date_str = datetime.now().strftime("%Y-%m-%d")
    success_count = 0
    fail_count = 0

    # Close Friday Gate issues
    if not args.prs_only:
        print("\n" + "=" * 40)
        print("CLOSING FRIDAY GATE ISSUES")
        print("=" * 40)

        for issue_num in FRIDAY_GATE_ISSUES:
            print(f"\nIssue #{issue_num}...", end=" ")

            if args.dry_run:
                print("[DRY RUN] Would close")
                success_count += 1
                continue

            issue = client.get_issue(issue_num)
            if "error" in issue:
                print(f"[SKIP] Not found or error")
                continue

            if issue.get("state") == "closed":
                print("[SKIP] Already closed")
                continue

            comment = CLOSE_COMMENT.format(date=date_str)
            if client.close_issue(issue_num, comment):
                print("[CLOSED]")
                success_count += 1
            else:
                print("[FAILED]")
                fail_count += 1

    # Close Dependabot PRs
    if not args.issues_only:
        print("\n" + "=" * 40)
        print("CLOSING DEPENDABOT PRs")
        print("=" * 40)

        for pr_num in DEPENDABOT_PRS:
            print(f"\nPR #{pr_num}...", end=" ")

            if args.dry_run:
                print("[DRY RUN] Would close")
                success_count += 1
                continue

            pr = client.get_pr(pr_num)
            if "error" in pr:
                print(f"[SKIP] Not found or error")
                continue

            if pr.get("state") == "closed":
                print("[SKIP] Already closed")
                continue

            comment = PR_CLOSE_COMMENT.format(date=date_str)
            if client.close_pr(pr_num, comment):
                print("[CLOSED]")
                success_count += 1
            else:
                print("[FAILED]")
                fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {fail_count}")

    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to execute.")

    print("=" * 60)


if __name__ == "__main__":
    main()
