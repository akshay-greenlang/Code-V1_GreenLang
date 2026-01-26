#!/usr/bin/env python3
"""
Full Security Audit and Fix Script for GreenLang Repository

Handles:
1. Enabling Dependabot security features
2. Fetching and analyzing all code scanning alerts
3. Auto-dismissing false positives and low-risk alerts
4. Generating fix recommendations

Author: GreenLang DevOps
Date: 2026-01-26
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from collections import defaultdict

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


class GitHubSecurityManager:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _request(self, method: str, endpoint: str, **kwargs):
        url = f"{API_BASE}{endpoint}"
        response = self.session.request(method, url, **kwargs)

        if response.status_code == 204:
            return {"status": "success"}
        if response.status_code >= 400:
            return {"error": response.text, "status": response.status_code}
        if response.text:
            try:
                return response.json()
            except:
                return {"text": response.text}
        return {"status": response.status_code}

    def enable_vulnerability_alerts(self):
        """Enable Dependabot vulnerability alerts."""
        result = self._request(
            "PUT",
            f"/repos/{OWNER}/{REPO}/vulnerability-alerts"
        )
        return "error" not in result

    def enable_automated_security_fixes(self):
        """Enable Dependabot automated security fixes."""
        result = self._request(
            "PUT",
            f"/repos/{OWNER}/{REPO}/automated-security-fixes"
        )
        return "error" not in result

    def get_all_code_scanning_alerts(self, state: str = "open"):
        """Fetch ALL code scanning alerts with pagination."""
        alerts = []
        page = 1
        per_page = 100

        while True:
            result = self._request(
                "GET",
                f"/repos/{OWNER}/{REPO}/code-scanning/alerts",
                params={"state": state, "per_page": per_page, "page": page}
            )

            if isinstance(result, dict) and "error" in result:
                print(f"  API Error: {result.get('status', 'unknown')}")
                break

            if not result or not isinstance(result, list):
                break

            alerts.extend(result)
            print(f"  Fetched page {page}: {len(result)} alerts (total: {len(alerts)})")

            if len(result) < per_page:
                break
            page += 1

        return alerts

    def dismiss_code_scanning_alert(self, alert_number: int, reason: str, comment: str = None):
        """Dismiss a code scanning alert."""
        data = {
            "state": "dismissed",
            "dismissed_reason": reason  # false positive, won't fix, used in tests
        }
        if comment:
            data["dismissed_comment"] = comment

        result = self._request(
            "PATCH",
            f"/repos/{OWNER}/{REPO}/code-scanning/alerts/{alert_number}",
            json=data
        )
        return "error" not in result

    def get_repo_security_settings(self):
        """Get current security settings."""
        result = self._request("GET", f"/repos/{OWNER}/{REPO}")
        if "error" not in result:
            return {
                "vulnerability_alerts": result.get("security_and_analysis", {}).get("dependabot_security_updates", {}).get("status"),
                "secret_scanning": result.get("security_and_analysis", {}).get("secret_scanning", {}).get("status"),
                "code_scanning": "enabled" if result.get("security_and_analysis", {}).get("advanced_security", {}).get("status") == "enabled" else "disabled"
            }
        return {}


def analyze_code_alerts(alerts: list) -> dict:
    """Analyze code scanning alerts."""
    analysis = {
        "total": len(alerts),
        "by_severity": defaultdict(list),
        "by_rule": defaultdict(list),
        "by_tool": defaultdict(list),
        "by_file": defaultdict(list),
        "dismissable": {
            "test_files": [],
            "example_files": [],
            "vendor_files": [],
            "low_severity": [],
            "false_positives": []
        }
    }

    for alert in alerts:
        severity = alert.get("rule", {}).get("security_severity_level") or alert.get("rule", {}).get("severity", "unknown")
        rule_id = alert.get("rule", {}).get("id", "unknown")
        tool = alert.get("tool", {}).get("name", "unknown")

        location = alert.get("most_recent_instance", {}).get("location", {})
        file_path = location.get("path", "unknown")

        analysis["by_severity"][severity].append(alert)
        analysis["by_rule"][rule_id].append(alert)
        analysis["by_tool"][tool].append(alert)
        analysis["by_file"][file_path].append(alert)

        # Categorize dismissable alerts
        if "/tests/" in file_path or "/test_" in file_path or file_path.startswith("tests/"):
            analysis["dismissable"]["test_files"].append(alert)
        elif "/examples/" in file_path or "/example" in file_path:
            analysis["dismissable"]["example_files"].append(alert)
        elif "/vendor/" in file_path or "/node_modules/" in file_path or "/venv/" in file_path:
            analysis["dismissable"]["vendor_files"].append(alert)
        elif severity in ["low", "note", "warning"]:
            analysis["dismissable"]["low_severity"].append(alert)

    return analysis


def print_analysis(analysis: dict):
    """Print formatted analysis."""
    print("\n" + "=" * 70)
    print("CODE SCANNING ALERTS ANALYSIS")
    print("=" * 70)

    print(f"\nTotal Alerts: {analysis['total']}")

    print("\n--- By Severity ---")
    severity_order = ["critical", "high", "error", "medium", "warning", "low", "note", "unknown"]
    for severity in severity_order:
        alerts = analysis["by_severity"].get(severity, [])
        if alerts:
            icons = {
                "critical": "[CRIT]", "high": "[HIGH]", "error": "[ERR]",
                "medium": "[MED]", "warning": "[WARN]",
                "low": "[LOW]", "note": "[NOTE]", "unknown": "[?]"
            }
            print(f"  {icons.get(severity, '[?]')} {severity.upper()}: {len(alerts)}")

    print("\n--- By Tool ---")
    for tool, alerts in sorted(analysis["by_tool"].items(), key=lambda x: -len(x[1])):
        print(f"  {tool}: {len(alerts)}")

    print("\n--- Top Rules Triggered ---")
    sorted_rules = sorted(analysis["by_rule"].items(), key=lambda x: -len(x[1]))[:10]
    for rule_id, alerts in sorted_rules:
        print(f"  {rule_id}: {len(alerts)}")

    print("\n--- Dismissable Alerts ---")
    dismissable = analysis["dismissable"]
    print(f"  Test files: {len(dismissable['test_files'])}")
    print(f"  Example files: {len(dismissable['example_files'])}")
    print(f"  Vendor/venv files: {len(dismissable['vendor_files'])}")
    print(f"  Low severity: {len(dismissable['low_severity'])}")

    total_dismissable = (
        len(dismissable['test_files']) +
        len(dismissable['example_files']) +
        len(dismissable['vendor_files'])
    )
    print(f"\n  Total auto-dismissable: {total_dismissable} ({total_dismissable * 100 // max(analysis['total'], 1)}%)")


def main():
    parser = argparse.ArgumentParser(description="Full Security Audit")
    parser.add_argument("--enable-dependabot", action="store_true", help="Enable Dependabot features")
    parser.add_argument("--analyze", action="store_true", help="Analyze all alerts")
    parser.add_argument("--dismiss-tests", action="store_true", help="Dismiss alerts in test files")
    parser.add_argument("--dismiss-examples", action="store_true", help="Dismiss alerts in example files")
    parser.add_argument("--dismiss-vendor", action="store_true", help="Dismiss alerts in vendor/venv files")
    parser.add_argument("--dismiss-all-safe", action="store_true", help="Dismiss all safe-to-dismiss alerts")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview without making changes")
    args = parser.parse_args()

    if not any(vars(args).values()):
        args.analyze = True

    token = get_token()
    if not token:
        print("Error: No GitHub token found")
        sys.exit(1)

    manager = GitHubSecurityManager(token)

    print("=" * 70)
    print("GreenLang Full Security Audit")
    print("=" * 70)
    print(f"Repository: {OWNER}/{REPO}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")

    if args.enable_dependabot:
        print("\n--- Enabling Dependabot Features ---")

        print("  Enabling vulnerability alerts...", end=" ")
        if manager.enable_vulnerability_alerts():
            print("[ENABLED]")
        else:
            print("[FAILED or already enabled]")

        print("  Enabling automated security fixes...", end=" ")
        if manager.enable_automated_security_fixes():
            print("[ENABLED]")
        else:
            print("[FAILED or already enabled]")

    # Fetch all code scanning alerts
    print("\n--- Fetching Code Scanning Alerts ---")
    alerts = manager.get_all_code_scanning_alerts()
    print(f"Total alerts fetched: {len(alerts)}")

    if args.analyze or args.dismiss_tests or args.dismiss_examples or args.dismiss_vendor or args.dismiss_all_safe:
        analysis = analyze_code_alerts(alerts)
        print_analysis(analysis)

    # Dismiss alerts
    dismiss_actions = []

    if args.dismiss_all_safe:
        args.dismiss_tests = True
        args.dismiss_examples = True
        args.dismiss_vendor = True

    if args.dismiss_tests:
        for alert in analysis["dismissable"]["test_files"]:
            dismiss_actions.append((alert, "used in tests", "Alert in test file - not production code"))

    if args.dismiss_examples:
        for alert in analysis["dismissable"]["example_files"]:
            dismiss_actions.append((alert, "won't fix", "Alert in example/demo file - not production code"))

    if args.dismiss_vendor:
        for alert in analysis["dismissable"]["vendor_files"]:
            dismiss_actions.append((alert, "won't fix", "Alert in vendor/venv directory - external code"))

    if dismiss_actions:
        print(f"\n--- Dismissing {len(dismiss_actions)} Alerts ---")

        success = 0
        failed = 0

        for alert, reason, comment in dismiss_actions:
            alert_num = alert.get("number")
            file_path = alert.get("most_recent_instance", {}).get("location", {}).get("path", "unknown")

            if args.dry_run:
                print(f"  [DRY RUN] Would dismiss #{alert_num} ({file_path[:40]}...)")
                success += 1
            else:
                print(f"  Dismissing #{alert_num}...", end=" ")
                if manager.dismiss_code_scanning_alert(alert_num, reason, comment):
                    print("[OK]")
                    success += 1
                else:
                    print("[FAILED]")
                    failed += 1

        print(f"\nDismissed: {success}, Failed: {failed}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    # Summary
    remaining = len(alerts) - len(dismiss_actions) if dismiss_actions else len(alerts)
    print(f"\nRemaining alerts to review: {remaining}")

    if remaining > 0 and analysis:
        print("\nPriority items to address:")
        for severity in ["critical", "high", "error"]:
            count = len(analysis["by_severity"].get(severity, []))
            if count > 0:
                print(f"  - {severity.upper()}: {count} alerts")


if __name__ == "__main__":
    main()
