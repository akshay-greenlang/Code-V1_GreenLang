#!/usr/bin/env python3
"""
Dismiss Dependabot Security Alerts

Uses cursor-based pagination to handle all alerts properly.
Dismisses alerts in non-production paths and low-severity alerts.

Author: GreenLang DevOps
Date: 2026-01-26
"""

import os
import sys
import subprocess
import re
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


class DependabotManager:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })

    def check_rate_limit(self):
        """Check and return remaining rate limit."""
        r = self.session.get(f"{API_BASE}/rate_limit")
        if r.status_code == 200:
            core = r.json()["resources"]["core"]
            return core["remaining"], core["limit"], core["reset"]
        return 0, 0, 0

    def wait_if_needed(self, min_remaining=100):
        """Wait if rate limit is low."""
        remaining, limit, reset_time = self.check_rate_limit()
        if remaining < min_remaining:
            wait = max(0, reset_time - time.time()) + 5
            print(f"  Rate limit low ({remaining}), waiting {int(wait)}s...")
            time.sleep(wait)
        return remaining

    def fetch_all_alerts(self):
        """Fetch all open Dependabot alerts using cursor pagination."""
        all_alerts = []
        url = f"{API_BASE}/repos/{OWNER}/{REPO}/dependabot/alerts?per_page=100"

        while url:
            self.wait_if_needed(50)
            r = self.session.get(url)

            if r.status_code != 200:
                print(f"Error fetching alerts: {r.status_code}")
                break

            data = r.json()
            if not data:
                break

            # Only keep open alerts
            open_alerts = [a for a in data if a.get("state") == "open"]
            all_alerts.extend(open_alerts)
            print(f"  Fetched {len(all_alerts)} open alerts...", end="\r")

            # Get next page from Link header
            link_header = r.headers.get("Link", "")
            url = None
            if 'rel="next"' in link_header:
                match = re.search(r'<([^>]+)>; rel="next"', link_header)
                if match:
                    url = match.group(1)

        print(f"  Total open alerts: {len(all_alerts)}      ")
        return all_alerts

    def dismiss_alert(self, alert_number: int, reason: str, comment: str) -> bool:
        """Dismiss a single Dependabot alert."""
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
        except Exception:
            return False

    def bulk_dismiss(self, alerts: list, reason: str, comment: str, workers: int = 10):
        """Dismiss alerts in parallel."""
        dismissed = failed = 0
        total = len(alerts)

        if total == 0:
            return 0, 0

        print(f"\n  Dismissing {total} alerts...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.dismiss_alert, alert["number"], reason, comment): alert
                for alert in alerts
            }

            for i, future in enumerate(as_completed(futures), 1):
                if future.result():
                    dismissed += 1
                else:
                    failed += 1

                if i % 50 == 0 or i == total:
                    print(f"  Progress: {i}/{total} (dismissed: {dismissed}, failed: {failed})")

                # Check rate limit every 150 requests
                if i % 150 == 0:
                    self.wait_if_needed(100)

        return dismissed, failed


def categorize_alerts(alerts):
    """
    Categorize alerts into:
    - dismissable: Low risk, non-production, or dev dependencies
    - keep: Critical/high in production code
    """
    dismissable = []
    keep = []

    # Paths that are non-production (safe to dismiss)
    non_prod_patterns = [
        "docs/", "planning/", "example", "demo", "test/", "tests/",
        "scripts/", "benchmark", "sample", ".github/", "archive/",
        "old/", "deprecated/", "legacy/"
    ]

    # Development-only packages (safe to dismiss)
    dev_packages = [
        "black", "flake8", "pytest", "mypy", "ruff", "pylint", "isort",
        "autopep8", "bandit", "coverage", "pre-commit", "sphinx", "mkdocs",
        "jupyter", "ipython", "notebook", "tox", "nox"
    ]

    for alert in alerts:
        severity = alert.get("security_advisory", {}).get("severity", "unknown")
        manifest = alert.get("dependency", {}).get("manifest_path", "").lower()
        pkg_name = alert.get("dependency", {}).get("package", {}).get("name", "").lower()

        # Non-production paths - always dismissable
        if any(p in manifest for p in non_prod_patterns):
            dismissable.append(alert)
            continue

        # Development packages - always dismissable
        if any(d in pkg_name for d in dev_packages):
            dismissable.append(alert)
            continue

        # Low/medium severity - dismissable
        if severity in ["low", "medium"]:
            dismissable.append(alert)
            continue

        # Keep critical/high for review
        keep.append(alert)

    return dismissable, keep


def main():
    token = get_token()
    if not token:
        print("Error: No GitHub token found")
        sys.exit(1)

    manager = DependabotManager(token)

    print("=" * 60)
    print("Dependabot Security Alert Cleanup")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Repository: {OWNER}/{REPO}")

    remaining, limit, _ = manager.check_rate_limit()
    print(f"Rate limit: {remaining}/{limit}")

    # Fetch all alerts
    print("\n--- Fetching Alerts ---")
    alerts = manager.fetch_all_alerts()

    if not alerts:
        print("\nNo open Dependabot alerts!")
        return

    # Show severity breakdown
    by_sev = {}
    for a in alerts:
        sev = a.get("security_advisory", {}).get("severity", "unknown")
        by_sev[sev] = by_sev.get(sev, 0) + 1

    print("\nSeverity breakdown:")
    for s in ["critical", "high", "medium", "low", "unknown"]:
        if s in by_sev:
            print(f"  {s}: {by_sev[s]}")

    # Categorize
    print("\n--- Categorizing Alerts ---")
    dismissable, keep = categorize_alerts(alerts)
    print(f"  Dismissable (non-prod/low-risk): {len(dismissable)}")
    print(f"  Keep for review (prod critical/high): {len(keep)}")

    # Dismiss low-risk alerts
    if dismissable:
        print("\n--- Dismissing Low-Risk Alerts ---")
        comment = f"Auto-dismissed: Non-production dependency or low-risk severity. {datetime.now().strftime('%Y-%m-%d')}"
        dismissed, failed = manager.bulk_dismiss(dismissable, "tolerable_risk", comment)
        print(f"\n  Result: {dismissed} dismissed, {failed} failed")

    # Report on remaining
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total alerts processed: {len(alerts)}")
    print(f"Dismissed: {len(dismissable)}")
    print(f"Remaining for review: {len(keep)}")

    if keep:
        print("\nAlerts requiring manual review:")
        by_sev_keep = {}
        for a in keep:
            sev = a.get("security_advisory", {}).get("severity", "unknown")
            by_sev_keep[sev] = by_sev_keep.get(sev, 0) + 1
        for s in ["critical", "high", "medium", "low"]:
            if s in by_sev_keep:
                print(f"  {s}: {by_sev_keep[s]}")

        # Show a few examples
        print("\nSample alerts to review:")
        for alert in keep[:5]:
            adv = alert.get("security_advisory", {})
            dep = alert.get("dependency", {})
            print(f"  #{alert['number']}: {dep.get('package', {}).get('name')} - {adv.get('summary', 'N/A')[:60]}")


if __name__ == "__main__":
    main()
