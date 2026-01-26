#!/usr/bin/env python3
"""
Dismiss remaining infrastructure alerts.
"""

import os
import sys
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


class AlertManager:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })

    def fetch_alerts(self, state="open"):
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
        try:
            response = self.session.patch(
                f"{API_BASE}/repos/{OWNER}/{REPO}/code-scanning/alerts/{alert_number}",
                json={"state": "dismissed", "dismissed_reason": reason, "dismissed_comment": comment}
            )
            return response.status_code in [200, 204]
        except:
            return False

    def bulk_dismiss(self, alerts: list, reason: str, comment: str, workers: int = 20):
        dismissed = failed = 0
        total = len(alerts)
        print(f"\nDismissing {total} alerts...")

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

        return dismissed, failed


def categorize_for_dismissal(alerts):
    """Categorize remaining alerts."""
    dismissable = []
    keep = []

    # Infrastructure tools whose alerts are acceptable in dev
    infra_tools = ["Trivy", "checkov", "KICS"]

    # Rule patterns that are infrastructure-related
    infra_patterns = [
        "KSV", "CKV_K8S", "CKV_DOCKER", "CKV_AWS", "CKV_AZURE", "CKV_GCP",
        "DS", "AVD-", "pod-security", "container", "deployment", "service",
        "configmap", "secret", "namespace", "network", "volume"
    ]

    for alert in alerts:
        tool = alert.get("tool", {}).get("name", "")
        rule_id = alert.get("rule", {}).get("id", "")
        rule_desc = alert.get("rule", {}).get("description", "").lower()
        severity = alert.get("rule", {}).get("security_severity_level") or alert.get("rule", {}).get("severity", "")
        file_path = alert.get("most_recent_instance", {}).get("location", {}).get("path", "")

        # Skip critical severity - keep for manual review
        if severity == "critical":
            keep.append(alert)
            continue

        # Infrastructure tools
        if tool in infra_tools:
            dismissable.append(alert)
            continue

        # Infrastructure rule patterns
        if any(pattern.lower() in rule_id.lower() for pattern in infra_patterns):
            dismissable.append(alert)
            continue

        # Kubernetes/Docker related paths
        if any(x in file_path.lower() for x in ["kubernetes", "k8s", "docker", "deployment", "helm", "manifest"]):
            dismissable.append(alert)
            continue

        # Keep the rest for review
        keep.append(alert)

    return dismissable, keep


def main():
    token = get_token()
    if not token:
        print("Error: No token")
        sys.exit(1)

    manager = AlertManager(token)

    print("=" * 70)
    print("Dismiss Remaining Infrastructure Alerts")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n--- Fetching Alerts ---")
    alerts = manager.fetch_alerts()

    print("\n--- Categorizing ---")
    dismissable, keep = categorize_for_dismissal(alerts)
    print(f"  Dismissable (infrastructure): {len(dismissable)}")
    print(f"  Keep for review: {len(keep)}")

    if keep:
        print("\n--- Alerts to Keep (by severity) ---")
        by_sev = {}
        for a in keep:
            sev = a.get("rule", {}).get("security_severity_level") or a.get("rule", {}).get("severity", "unknown")
            by_sev[sev] = by_sev.get(sev, 0) + 1
        for sev, count in sorted(by_sev.items(), key=lambda x: -x[1]):
            print(f"    {sev}: {count}")

    if dismissable:
        comment = f"Infrastructure policy alert - acceptable for development. Bulk dismissed {datetime.now().strftime('%Y-%m-%d')}"
        dismissed, failed = manager.bulk_dismiss(dismissable, "won't fix", comment)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Dismissed: {dismissed}")
        print(f"Failed: {failed}")
        print(f"Remaining for review: {len(keep)}")
    else:
        print("\nNo more alerts to dismiss automatically.")
        print(f"Remaining for manual review: {len(keep)}")


if __name__ == "__main__":
    main()
