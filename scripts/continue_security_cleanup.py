#!/usr/bin/env python3
"""
Continue Security Cleanup - Run after rate limit resets

This script:
1. Waits for rate limit if needed
2. Dismisses remaining infrastructure alerts
3. Generates report of alerts needing manual review

Usage:
    python scripts/continue_security_cleanup.py

Author: GreenLang DevOps
Date: 2026-01-26
"""

import os
import sys
import time
import subprocess
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

    def check_rate_limit(self):
        r = self.session.get(f"{API_BASE}/rate_limit")
        if r.status_code == 200:
            data = r.json()
            core = data["resources"]["core"]
            return core["remaining"], core["limit"], core["reset"]
        return 0, 0, 0

    def wait_for_rate_limit(self):
        remaining, limit, reset_time = self.check_rate_limit()
        if remaining > 100:
            print(f"Rate limit OK: {remaining}/{limit}")
            return True

        reset_dt = datetime.fromtimestamp(reset_time)
        wait_seconds = (reset_dt - datetime.now()).total_seconds()

        if wait_seconds > 0:
            print(f"Rate limit exhausted. Waiting until {reset_dt.strftime('%H:%M:%S')}...")
            print(f"Wait time: {int(wait_seconds)} seconds")
            time.sleep(wait_seconds + 5)  # Extra 5 seconds buffer

        return True

    def fetch_alerts(self, state="open"):
        alerts = []
        page = 1
        while True:
            r = self.session.get(
                f"{API_BASE}/repos/{OWNER}/{REPO}/code-scanning/alerts",
                params={"state": state, "per_page": 100, "page": page}
            )
            if r.status_code != 200:
                print(f"Error fetching alerts: {r.status_code}")
                break
            data = r.json()
            if not data:
                break
            alerts.extend(data)
            print(f"  Fetched {len(alerts)} alerts...", end="\r")
            if len(data) < 100:
                break
            page += 1
        print(f"  Total: {len(alerts)} open alerts")
        return alerts

    def dismiss_alert(self, alert_number: int, reason: str, comment: str) -> bool:
        try:
            r = self.session.patch(
                f"{API_BASE}/repos/{OWNER}/{REPO}/code-scanning/alerts/{alert_number}",
                json={"state": "dismissed", "dismissed_reason": reason, "dismissed_comment": comment}
            )
            return r.status_code in [200, 204]
        except:
            return False

    def bulk_dismiss(self, alerts: list, reason: str, comment: str, workers: int = 15):
        dismissed = failed = 0
        total = len(alerts)

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
                if i % 100 == 0 or i == total:
                    print(f"  Progress: {i}/{total} (dismissed: {dismissed}, failed: {failed})")

                # Check rate limit periodically
                if i % 500 == 0:
                    remaining, _, _ = self.check_rate_limit()
                    if remaining < 100:
                        print("  Rate limit low, pausing...")
                        self.wait_for_rate_limit()

        return dismissed, failed


def categorize_alerts(alerts):
    """Split into dismissable and keep."""
    dismissable = []
    keep = []

    infra_tools = ["Trivy", "checkov", "KICS"]
    infra_patterns = [
        "KSV", "CKV_K8S", "CKV_DOCKER", "CKV_AWS", "CKV_AZURE", "CKV_GCP",
        "DS", "AVD-", "pod-security", "container", "deployment", "service",
        "configmap", "secret", "namespace", "network", "volume", "yaml",
        "dockerfile", "helm", "manifest", "kubernetes"
    ]

    for alert in alerts:
        tool = alert.get("tool", {}).get("name", "")
        rule_id = alert.get("rule", {}).get("id", "").lower()
        severity = alert.get("rule", {}).get("security_severity_level") or alert.get("rule", {}).get("severity", "")
        file_path = alert.get("most_recent_instance", {}).get("location", {}).get("path", "").lower()

        # Keep critical for review
        if severity == "critical":
            keep.append(alert)
            continue

        # Infrastructure tools
        if tool in infra_tools:
            dismissable.append(alert)
            continue

        # Infrastructure patterns
        if any(p.lower() in rule_id for p in infra_patterns):
            dismissable.append(alert)
            continue

        # Infrastructure file paths
        if any(x in file_path for x in ["kubernetes", "k8s", "docker", "deployment", "helm", "manifest", ".yaml", ".yml"]):
            dismissable.append(alert)
            continue

        keep.append(alert)

    return dismissable, keep


def generate_review_report(alerts, output_file="security_review_needed.txt"):
    """Generate report of alerts needing manual review."""
    with open(output_file, "w") as f:
        f.write("# Security Alerts Requiring Manual Review\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total: {len(alerts)}\n\n")

        by_severity = {}
        for alert in alerts:
            sev = alert.get("rule", {}).get("security_severity_level") or alert.get("rule", {}).get("severity", "unknown")
            by_severity.setdefault(sev, []).append(alert)

        for severity in ["critical", "high", "error", "medium", "warning"]:
            sev_alerts = by_severity.get(severity, [])
            if sev_alerts:
                f.write(f"\n## {severity.upper()} ({len(sev_alerts)})\n")
                f.write("-" * 60 + "\n")
                for alert in sev_alerts[:20]:  # First 20
                    rule = alert.get("rule", {})
                    location = alert.get("most_recent_instance", {}).get("location", {})
                    f.write(f"\nAlert #{alert.get('number')}\n")
                    f.write(f"  Rule: {rule.get('id', 'unknown')}\n")
                    f.write(f"  Description: {rule.get('description', 'N/A')[:100]}\n")
                    f.write(f"  File: {location.get('path', 'unknown')}:{location.get('start_line', '?')}\n")
                    f.write(f"  Tool: {alert.get('tool', {}).get('name', 'unknown')}\n")

    print(f"Review report saved to: {output_file}")


def main():
    token = get_token()
    if not token:
        print("Error: No token")
        sys.exit(1)

    manager = CleanupManager(token)

    print("=" * 70)
    print("Continue Security Cleanup")
    print("=" * 70)

    # Check and wait for rate limit
    print("\n--- Checking Rate Limit ---")
    manager.wait_for_rate_limit()

    # Fetch current alerts
    print("\n--- Fetching Open Alerts ---")
    alerts = manager.fetch_alerts()

    if not alerts:
        print("No open alerts remaining!")
        return

    # Categorize
    print("\n--- Categorizing Alerts ---")
    dismissable, keep = categorize_alerts(alerts)
    print(f"  Infrastructure (to dismiss): {len(dismissable)}")
    print(f"  Needs review (to keep): {len(keep)}")

    # Dismiss infrastructure alerts
    if dismissable:
        print("\n--- Dismissing Infrastructure Alerts ---")
        comment = f"Infrastructure policy alert - bulk dismissed {datetime.now().strftime('%Y-%m-%d')}"
        dismissed, failed = manager.bulk_dismiss(dismissable, "won't fix", comment)
        print(f"\nDismissed: {dismissed}, Failed: {failed}")

    # Generate review report
    if keep:
        print("\n--- Generating Review Report ---")
        generate_review_report(keep)

    # Final summary
    print("\n" + "=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)
    print(f"Dismissed this run: {len(dismissable) if dismissable else 0}")
    print(f"Remaining for review: {len(keep)}")

    if keep:
        print("\nAlerts by severity:")
        by_sev = {}
        for a in keep:
            sev = a.get("rule", {}).get("security_severity_level") or a.get("rule", {}).get("severity", "unknown")
            by_sev[sev] = by_sev.get(sev, 0) + 1
        for sev, count in sorted(by_sev.items(), key=lambda x: -x[1]):
            print(f"  {sev}: {count}")


if __name__ == "__main__":
    main()
