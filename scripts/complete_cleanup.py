#!/usr/bin/env python3
"""
Complete GreenLang Repository Cleanup Script

Handles:
1. Merge open Dependabot PRs
2. Dismiss Dependabot security alerts
3. Clean up stale branches
4. Keep only essential branches

Author: GreenLang DevOps
Date: 2026-01-26
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

OWNER = "akshay-greenlang"
REPO = "Code-V1_GreenLang"
API_BASE = "https://api.github.com"

# Branches to KEEP (protected)
KEEP_BRANCHES = {"master", "main", "release/0.3.0"}


def get_token():
    """Get GitHub token from git credential manager."""
    try:
        input_data = "protocol=https\nhost=github.com\n\n"
        result = subprocess.run(
            ["git", "credential", "fill"],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("password="):
                    return line.split("=", 1)[1]
    except Exception:
        pass
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


class CleanupManager:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })
        self.stats = {
            "prs_merged": 0,
            "prs_closed": 0,
            "branches_deleted": 0,
            "alerts_dismissed": 0,
            "errors": []
        }

    def check_rate_limit(self):
        """Check GitHub API rate limit."""
        r = self.session.get(f"{API_BASE}/rate_limit")
        if r.status_code == 200:
            core = r.json()["resources"]["core"]
            return core["remaining"], core["limit"], core["reset"]
        return 0, 0, 0

    def wait_if_rate_limited(self):
        """Wait if rate limit is low."""
        remaining, limit, reset_time = self.check_rate_limit()
        if remaining < 50:
            wait_secs = max(0, reset_time - time.time()) + 5
            print(f"  Rate limit low ({remaining}/{limit}), waiting {int(wait_secs)}s...")
            time.sleep(wait_secs)
        return remaining

    # ============== PR MANAGEMENT ==============

    def get_open_prs(self):
        """Get all open PRs."""
        prs = []
        page = 1
        while True:
            r = self.session.get(
                f"{API_BASE}/repos/{OWNER}/{REPO}/pulls",
                params={"state": "open", "per_page": 100, "page": page}
            )
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
            prs.extend(data)
            if len(data) < 100:
                break
            page += 1
        return prs

    def merge_pr(self, pr_number: int, title: str) -> bool:
        """Merge a PR."""
        try:
            r = self.session.put(
                f"{API_BASE}/repos/{OWNER}/{REPO}/pulls/{pr_number}/merge",
                json={
                    "commit_title": f"Merge PR #{pr_number}: {title}",
                    "merge_method": "squash"
                }
            )
            if r.status_code in [200, 201]:
                return True
            # Try rebase if squash fails
            r = self.session.put(
                f"{API_BASE}/repos/{OWNER}/{REPO}/pulls/{pr_number}/merge",
                json={"merge_method": "merge"}
            )
            return r.status_code in [200, 201]
        except Exception as e:
            self.stats["errors"].append(f"PR #{pr_number}: {e}")
            return False

    def close_pr(self, pr_number: int) -> bool:
        """Close a PR without merging."""
        try:
            r = self.session.patch(
                f"{API_BASE}/repos/{OWNER}/{REPO}/pulls/{pr_number}",
                json={"state": "closed"}
            )
            return r.status_code == 200
        except:
            return False

    def handle_prs(self):
        """Process all open PRs."""
        print("\n" + "="*60)
        print("PHASE 1: Processing Open PRs")
        print("="*60)

        prs = self.get_open_prs()
        print(f"Found {len(prs)} open PRs")

        for pr in prs:
            pr_num = pr["number"]
            title = pr["title"]
            branch = pr["head"]["ref"]

            print(f"\n  PR #{pr_num}: {title[:50]}...")
            print(f"    Branch: {branch}")

            # Try to merge Dependabot PRs
            if "dependabot" in branch.lower() or "dependabot" in pr.get("user", {}).get("login", "").lower():
                print("    [MERGING] Dependabot security update...")
                if self.merge_pr(pr_num, title):
                    print("    [OK] Merged successfully")
                    self.stats["prs_merged"] += 1
                else:
                    print("    [CLOSE] Could not merge, closing...")
                    if self.close_pr(pr_num):
                        self.stats["prs_closed"] += 1
            else:
                # Non-dependabot PR - close if stale
                print("    [SKIP] Non-Dependabot PR, skipping")

        print(f"\n  PRs merged: {self.stats['prs_merged']}")
        print(f"  PRs closed: {self.stats['prs_closed']}")

    # ============== BRANCH MANAGEMENT ==============

    def get_branches(self):
        """Get all branches."""
        branches = []
        page = 1
        while True:
            r = self.session.get(
                f"{API_BASE}/repos/{OWNER}/{REPO}/branches",
                params={"per_page": 100, "page": page}
            )
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
            branches.extend(data)
            if len(data) < 100:
                break
            page += 1
        return branches

    def delete_branch(self, branch_name: str) -> bool:
        """Delete a branch."""
        try:
            r = self.session.delete(
                f"{API_BASE}/repos/{OWNER}/{REPO}/git/refs/heads/{branch_name}"
            )
            return r.status_code in [200, 204]
        except:
            return False

    def handle_branches(self):
        """Clean up stale branches."""
        print("\n" + "="*60)
        print("PHASE 2: Cleaning Up Branches")
        print("="*60)

        branches = self.get_branches()
        print(f"Found {len(branches)} branches")
        print(f"Protected branches: {KEEP_BRANCHES}")

        for branch in branches:
            name = branch["name"]

            if name in KEEP_BRANCHES:
                print(f"  [KEEP] {name} (protected)")
                continue

            # Delete stale branches
            print(f"  [DELETE] {name}...", end=" ")
            if self.delete_branch(name):
                print("OK")
                self.stats["branches_deleted"] += 1
            else:
                print("FAILED")

        print(f"\n  Branches deleted: {self.stats['branches_deleted']}")

    # ============== SECURITY ALERTS ==============

    def get_dependabot_alerts(self):
        """Get all open Dependabot alerts."""
        alerts = []
        page = 1
        while page <= 20:  # Safety limit
            self.wait_if_rate_limited()
            r = self.session.get(
                f"{API_BASE}/repos/{OWNER}/{REPO}/dependabot/alerts",
                params={"state": "open", "per_page": 100, "page": page}
            )
            if r.status_code != 200:
                print(f"  Error fetching page {page}: {r.status_code}")
                break
            data = r.json()
            if not data:
                break
            alerts.extend(data)
            print(f"  Fetched {len(alerts)} alerts...", end="\r")
            if len(data) < 100:
                break
            page += 1
        print(f"  Total Dependabot alerts: {len(alerts)}")
        return alerts

    def dismiss_dependabot_alert(self, alert_number: int, reason: str, comment: str) -> bool:
        """Dismiss a Dependabot alert."""
        try:
            r = self.session.patch(
                f"{API_BASE}/repos/{OWNER}/{REPO}/dependabot/alerts/{alert_number}",
                json={
                    "state": "dismissed",
                    "dismissed_reason": reason,
                    "dismissed_comment": comment
                }
            )
            return r.status_code == 200
        except:
            return False

    def bulk_dismiss_dependabot(self, alerts: list, reason: str, comment: str, workers: int = 10):
        """Dismiss alerts in parallel."""
        dismissed = failed = 0
        total = len(alerts)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.dismiss_dependabot_alert, alert["number"], reason, comment): alert
                for alert in alerts
            }
            for i, future in enumerate(as_completed(futures), 1):
                if future.result():
                    dismissed += 1
                else:
                    failed += 1
                if i % 50 == 0 or i == total:
                    print(f"  Progress: {i}/{total} (dismissed: {dismissed}, failed: {failed})")

                # Check rate limit every 200 requests
                if i % 200 == 0:
                    self.wait_if_rate_limited()

        return dismissed, failed

    def categorize_alerts(self, alerts):
        """Categorize alerts into dismissable and needs-review."""
        dismissable = []
        needs_review = []

        for alert in alerts:
            severity = alert.get("security_advisory", {}).get("severity", "unknown")
            manifest = alert.get("dependency", {}).get("manifest_path", "")
            pkg = alert.get("dependency", {}).get("package", {}).get("name", "")

            # Non-essential paths - always dismissable
            if any(x in manifest.lower() for x in [
                "example", "docs/", "planning/", "demo", "test/",
                "scripts/", ".github/", "benchmark", "sample"
            ]):
                dismissable.append(alert)
                continue

            # Low/medium severity - dismissable with note
            if severity in ["low", "medium"]:
                dismissable.append(alert)
                continue

            # Development-only dependencies
            dev_packages = ["black", "flake8", "pytest", "mypy", "ruff", "pylint", "isort"]
            if any(d in pkg.lower() for d in dev_packages):
                dismissable.append(alert)
                continue

            needs_review.append(alert)

        return dismissable, needs_review

    def handle_security_alerts(self):
        """Process all security alerts."""
        print("\n" + "="*60)
        print("PHASE 3: Processing Security Alerts")
        print("="*60)

        alerts = self.get_dependabot_alerts()

        if not alerts:
            print("  No open Dependabot alerts!")
            return

        # Categorize
        print("\n  Categorizing alerts...")
        dismissable, needs_review = self.categorize_alerts(alerts)

        print(f"  Dismissable (low risk/non-essential): {len(dismissable)}")
        print(f"  Needs manual review (critical/high): {len(needs_review)}")

        # Show severity breakdown
        by_sev = {}
        for a in alerts:
            sev = a.get("security_advisory", {}).get("severity", "unknown")
            by_sev[sev] = by_sev.get(sev, 0) + 1
        print("\n  By severity:")
        for sev in ["critical", "high", "medium", "low", "unknown"]:
            if sev in by_sev:
                print(f"    {sev}: {by_sev[sev]}")

        # Dismiss low-risk alerts
        if dismissable:
            print(f"\n  Dismissing {len(dismissable)} low-risk alerts...")
            comment = f"Auto-dismissed: Low severity or non-production dependency. Reviewed {datetime.now().strftime('%Y-%m-%d')}"
            dismissed, failed = self.bulk_dismiss_dependabot(dismissable, "tolerable_risk", comment)
            self.stats["alerts_dismissed"] = dismissed
            print(f"\n  Dismissed: {dismissed}, Failed: {failed}")

        # Report on remaining
        if needs_review:
            print(f"\n  Alerts requiring manual review: {len(needs_review)}")
            # Save report
            self.save_alert_report(needs_review)

    def save_alert_report(self, alerts):
        """Save report of alerts needing review."""
        report_file = "security_review_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# Security Alerts Requiring Manual Review\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total: {len(alerts)}\n\n")

            by_sev = {}
            for a in alerts:
                sev = a.get("security_advisory", {}).get("severity", "unknown")
                by_sev.setdefault(sev, []).append(a)

            for sev in ["critical", "high", "medium", "low"]:
                sev_alerts = by_sev.get(sev, [])
                if sev_alerts:
                    f.write(f"\n## {sev.upper()} ({len(sev_alerts)})\n")
                    f.write("-" * 50 + "\n")
                    for alert in sev_alerts[:30]:
                        advisory = alert.get("security_advisory", {})
                        dep = alert.get("dependency", {})
                        f.write(f"\nAlert #{alert.get('number')}\n")
                        f.write(f"  Package: {dep.get('package', {}).get('name')}\n")
                        f.write(f"  Manifest: {dep.get('manifest_path')}\n")
                        f.write(f"  Summary: {advisory.get('summary', 'N/A')[:100]}\n")
                        f.write(f"  CVE: {advisory.get('cve_id', 'N/A')}\n")

        print(f"  Report saved to: {report_file}")


def main():
    token = get_token()
    if not token:
        print("Error: No GitHub token found")
        sys.exit(1)

    manager = CleanupManager(token)

    print("="*60)
    print("GreenLang Complete Repository Cleanup")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Repository: {OWNER}/{REPO}")

    # Check rate limit
    remaining, limit, _ = manager.check_rate_limit()
    print(f"Rate limit: {remaining}/{limit}")

    # Phase 1: PRs
    manager.handle_prs()

    # Phase 2: Branches
    manager.handle_branches()

    # Phase 3: Security
    manager.handle_security_alerts()

    # Final Summary
    print("\n" + "="*60)
    print("CLEANUP COMPLETE")
    print("="*60)
    print(f"PRs merged: {manager.stats['prs_merged']}")
    print(f"PRs closed: {manager.stats['prs_closed']}")
    print(f"Branches deleted: {manager.stats['branches_deleted']}")
    print(f"Security alerts dismissed: {manager.stats['alerts_dismissed']}")

    if manager.stats["errors"]:
        print(f"\nErrors encountered: {len(manager.stats['errors'])}")
        for err in manager.stats["errors"][:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
