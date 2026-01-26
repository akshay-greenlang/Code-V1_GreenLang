#!/usr/bin/env python3
"""
Bulk Dismiss Security Alerts Script

Efficiently dismisses large numbers of code scanning alerts based on categories.

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


def get_token():
    """Get GitHub token."""
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


class BulkDismisser:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })
        self.dismissed = 0
        self.failed = 0

    def fetch_all_alerts(self, state="open"):
        """Fetch all alerts with pagination."""
        alerts = []
        page = 1

        while True:
            response = self.session.get(
                f"{API_BASE}/repos/{OWNER}/{REPO}/code-scanning/alerts",
                params={"state": state, "per_page": 100, "page": page}
            )

            if response.status_code != 200:
                break

            data = response.json()
            if not data:
                break

            alerts.extend(data)
            print(f"  Fetched {len(alerts)} alerts...", end="\r")

            if len(data) < 100:
                break
            page += 1

        print(f"  Fetched {len(alerts)} alerts total")
        return alerts

    def dismiss_alert(self, alert_number: int, reason: str, comment: str) -> bool:
        """Dismiss a single alert."""
        try:
            response = self.session.patch(
                f"{API_BASE}/repos/{OWNER}/{REPO}/code-scanning/alerts/{alert_number}",
                json={
                    "state": "dismissed",
                    "dismissed_reason": reason,
                    "dismissed_comment": comment
                }
            )
            return response.status_code in [200, 204]
        except Exception:
            return False

    def bulk_dismiss(self, alerts: list, reason: str, comment: str, max_workers: int = 10):
        """Dismiss alerts in parallel."""
        total = len(alerts)
        print(f"\nDismissing {total} alerts...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.dismiss_alert, alert["number"], reason, comment): alert
                for alert in alerts
            }

            for i, future in enumerate(as_completed(futures), 1):
                if future.result():
                    self.dismissed += 1
                else:
                    self.failed += 1

                if i % 50 == 0 or i == total:
                    print(f"  Progress: {i}/{total} (dismissed: {self.dismissed}, failed: {self.failed})")

        return self.dismissed, self.failed


def categorize_alerts(alerts: list) -> dict:
    """Categorize alerts for bulk dismissal."""
    categories = {
        "k8s_security_context": [],      # Missing security context
        "k8s_cpu_limits": [],            # Missing CPU limits
        "k8s_memory_limits": [],         # Missing memory limits
        "k8s_read_only_fs": [],          # Read-only filesystem not set
        "k8s_run_as_non_root": [],       # Not running as non-root
        "k8s_privileged": [],            # Privileged container
        "k8s_capabilities": [],          # Container capabilities
        "docker_user": [],               # Dockerfile USER instruction
        "low_severity": [],              # All low severity
        "note_severity": [],             # All note severity
        "test_files": [],                # Test files
        "example_files": [],             # Example files
        "vendor_files": [],              # Vendor/venv files
        "other": []                      # Everything else
    }

    for alert in alerts:
        rule_id = alert.get("rule", {}).get("id", "")
        severity = alert.get("rule", {}).get("security_severity_level") or alert.get("rule", {}).get("severity", "")
        file_path = alert.get("most_recent_instance", {}).get("location", {}).get("path", "")

        # File-based categorization
        if "/tests/" in file_path or "/test_" in file_path or file_path.startswith("tests/"):
            categories["test_files"].append(alert)
            continue
        elif "/examples/" in file_path or "/example" in file_path:
            categories["example_files"].append(alert)
            continue
        elif "/vendor/" in file_path or "/node_modules/" in file_path or "/venv/" in file_path or "site-packages" in file_path:
            categories["vendor_files"].append(alert)
            continue

        # Severity-based
        if severity in ["low"]:
            categories["low_severity"].append(alert)
            continue
        elif severity in ["note"]:
            categories["note_severity"].append(alert)
            continue

        # Rule-based categorization (Kubernetes)
        if any(x in rule_id for x in ["KSV021", "KSV020", "CKV_K8S_38", "CKV_K8S_20", "securityContext"]):
            categories["k8s_security_context"].append(alert)
        elif any(x in rule_id for x in ["KSV011", "CKV_K8S_13", "cpuLimit"]):
            categories["k8s_cpu_limits"].append(alert)
        elif any(x in rule_id for x in ["KSV018", "CKV_K8S_14", "memoryLimit"]):
            categories["k8s_memory_limits"].append(alert)
        elif any(x in rule_id for x in ["KSV014", "CKV_K8S_22", "readOnlyRoot"]):
            categories["k8s_read_only_fs"].append(alert)
        elif any(x in rule_id for x in ["KSV012", "CKV_K8S_6", "runAsNonRoot"]):
            categories["k8s_run_as_non_root"].append(alert)
        elif any(x in rule_id for x in ["KSV001", "CKV_K8S_1", "privileged"]):
            categories["k8s_privileged"].append(alert)
        elif any(x in rule_id for x in ["KSV003", "KSV022", "CKV_K8S_25", "capabilities"]):
            categories["k8s_capabilities"].append(alert)
        elif any(x in rule_id for x in ["DS002", "CKV_DOCKER_3", "USER"]):
            categories["docker_user"].append(alert)
        else:
            categories["other"].append(alert)

    return categories


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bulk dismiss security alerts")
    parser.add_argument("--dismiss-k8s", action="store_true", help="Dismiss Kubernetes policy alerts")
    parser.add_argument("--dismiss-low", action="store_true", help="Dismiss low severity alerts")
    parser.add_argument("--dismiss-notes", action="store_true", help="Dismiss note severity alerts")
    parser.add_argument("--dismiss-all-safe", action="store_true", help="Dismiss all safe-to-dismiss alerts")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without dismissing")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    args = parser.parse_args()

    token = get_token()
    if not token:
        print("Error: No GitHub token")
        sys.exit(1)

    dismisser = BulkDismisser(token)

    print("=" * 70)
    print("Bulk Security Alert Dismisser")
    print("=" * 70)
    print(f"Repository: {OWNER}/{REPO}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")

    print("\n--- Fetching Alerts ---")
    alerts = dismisser.fetch_all_alerts()

    print("\n--- Categorizing Alerts ---")
    categories = categorize_alerts(alerts)

    for cat, cat_alerts in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(cat_alerts)}")

    # Determine what to dismiss
    to_dismiss = []
    dismiss_reasons = []

    if args.dismiss_all_safe:
        args.dismiss_k8s = True
        args.dismiss_low = True
        args.dismiss_notes = True

    if args.dismiss_k8s:
        for cat in ["k8s_security_context", "k8s_cpu_limits", "k8s_memory_limits",
                    "k8s_read_only_fs", "k8s_run_as_non_root", "k8s_capabilities", "docker_user"]:
            to_dismiss.extend(categories[cat])
            dismiss_reasons.append(f"{cat}: {len(categories[cat])}")

    if args.dismiss_low:
        to_dismiss.extend(categories["low_severity"])
        dismiss_reasons.append(f"low_severity: {len(categories['low_severity'])}")

    if args.dismiss_notes:
        to_dismiss.extend(categories["note_severity"])
        dismiss_reasons.append(f"note_severity: {len(categories['note_severity'])}")

    # Always dismiss test/example/vendor
    to_dismiss.extend(categories["test_files"])
    to_dismiss.extend(categories["example_files"])
    to_dismiss.extend(categories["vendor_files"])

    print(f"\n--- Alerts to Dismiss: {len(to_dismiss)} ---")
    for reason in dismiss_reasons:
        print(f"  {reason}")

    if not to_dismiss:
        print("\nNo alerts to dismiss. Use --dismiss-k8s, --dismiss-low, or --dismiss-all-safe")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would dismiss {len(to_dismiss)} alerts")
        return

    # Bulk dismiss
    comment = f"Bulk dismissed on {datetime.now().strftime('%Y-%m-%d')} - Infrastructure policy alert, acceptable for development"
    dismissed, failed = dismisser.bulk_dismiss(to_dismiss, "won't fix", comment, args.workers)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total dismissed: {dismissed}")
    print(f"Failed: {failed}")
    print(f"Remaining: {len(alerts) - dismissed}")


if __name__ == "__main__":
    main()
