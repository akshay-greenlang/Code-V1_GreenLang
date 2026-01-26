#!/usr/bin/env python3
"""
Security Audit Script for GreenLang Repository

Fetches and analyzes Dependabot security alerts from GitHub API.
Provides recommendations and can auto-fix vulnerabilities.

Usage:
    python scripts/security_audit.py --analyze
    python scripts/security_audit.py --fix
    python scripts/security_audit.py --dismiss-low

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

# Repository info
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


class SecurityAuditor:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.alerts = []

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        url = f"{API_BASE}{endpoint}"
        response = self.session.request(method, url, **kwargs)

        if response.status_code == 401:
            print("Error: Invalid token or insufficient permissions")
            return {"error": "unauthorized"}

        if response.status_code == 404:
            return {"error": "not_found"}

        if response.status_code >= 400:
            return {"error": response.text, "status": response.status_code}

        if response.text:
            return response.json()
        return {"status": response.status_code}

    def get_dependabot_alerts(self, state: str = "open") -> list:
        """Fetch all Dependabot alerts."""
        alerts = []
        page = 1
        per_page = 100

        while True:
            result = self._request(
                "GET",
                f"/repos/{OWNER}/{REPO}/dependabot/alerts",
                params={"state": state, "per_page": per_page, "page": page}
            )

            if isinstance(result, dict) and "error" in result:
                print(f"Error fetching alerts: {result}")
                break

            if not result:
                break

            alerts.extend(result)

            if len(result) < per_page:
                break
            page += 1

        self.alerts = alerts
        return alerts

    def get_code_scanning_alerts(self, state: str = "open") -> list:
        """Fetch code scanning alerts."""
        result = self._request(
            "GET",
            f"/repos/{OWNER}/{REPO}/code-scanning/alerts",
            params={"state": state, "per_page": 100}
        )

        if isinstance(result, dict) and "error" in result:
            return []
        return result if isinstance(result, list) else []

    def get_secret_scanning_alerts(self, state: str = "open") -> list:
        """Fetch secret scanning alerts."""
        result = self._request(
            "GET",
            f"/repos/{OWNER}/{REPO}/secret-scanning/alerts",
            params={"state": state, "per_page": 100}
        )

        if isinstance(result, dict) and "error" in result:
            return []
        return result if isinstance(result, list) else []

    def analyze_alerts(self) -> dict:
        """Analyze alerts and categorize by severity and package."""
        analysis = {
            "total": len(self.alerts),
            "by_severity": defaultdict(list),
            "by_package": defaultdict(list),
            "by_ecosystem": defaultdict(list),
            "fixable": [],
            "needs_review": []
        }

        for alert in self.alerts:
            severity = alert.get("security_advisory", {}).get("severity", "unknown")
            package = alert.get("security_vulnerability", {}).get("package", {}).get("name", "unknown")
            ecosystem = alert.get("security_vulnerability", {}).get("package", {}).get("ecosystem", "unknown")

            analysis["by_severity"][severity].append(alert)
            analysis["by_package"][package].append(alert)
            analysis["by_ecosystem"][ecosystem].append(alert)

            # Check if auto-fixable
            if alert.get("security_vulnerability", {}).get("first_patched_version"):
                analysis["fixable"].append(alert)
            else:
                analysis["needs_review"].append(alert)

        return analysis

    def dismiss_alert(self, alert_number: int, reason: str, comment: str = None) -> bool:
        """Dismiss a Dependabot alert."""
        data = {
            "state": "dismissed",
            "dismissed_reason": reason,  # fix_started, inaccurate, no_bandwidth, not_used, tolerable_risk
        }
        if comment:
            data["dismissed_comment"] = comment

        result = self._request(
            "PATCH",
            f"/repos/{OWNER}/{REPO}/dependabot/alerts/{alert_number}",
            json=data
        )
        return "error" not in result

    def get_dependency_graph(self) -> dict:
        """Get dependency graph to understand what's actually used."""
        result = self._request(
            "GET",
            f"/repos/{OWNER}/{REPO}/dependency-graph/sbom"
        )
        return result


def print_analysis(analysis: dict):
    """Print formatted analysis report."""
    print("\n" + "=" * 60)
    print("SECURITY ALERTS ANALYSIS")
    print("=" * 60)

    print(f"\nTotal Open Alerts: {analysis['total']}")

    print("\n--- By Severity ---")
    for severity in ["critical", "high", "medium", "low"]:
        count = len(analysis["by_severity"].get(severity, []))
        if count > 0:
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
            print(f"  {icon} {severity.upper()}: {count}")

    print("\n--- By Ecosystem ---")
    for ecosystem, alerts in sorted(analysis["by_ecosystem"].items(), key=lambda x: -len(x[1])):
        print(f"  {ecosystem}: {len(alerts)}")

    print("\n--- Top Vulnerable Packages ---")
    sorted_packages = sorted(analysis["by_package"].items(), key=lambda x: -len(x[1]))[:10]
    for package, alerts in sorted_packages:
        severities = [a.get("security_advisory", {}).get("severity", "?") for a in alerts]
        print(f"  {package}: {len(alerts)} alerts ({', '.join(set(severities))})")

    print(f"\n--- Fixability ---")
    print(f"  Auto-fixable (patched version available): {len(analysis['fixable'])}")
    print(f"  Needs manual review: {len(analysis['needs_review'])}")


def generate_fix_recommendations(analysis: dict) -> list:
    """Generate fix recommendations."""
    recommendations = []

    # Critical and high severity first
    for severity in ["critical", "high"]:
        for alert in analysis["by_severity"].get(severity, []):
            vuln = alert.get("security_vulnerability", {})
            package = vuln.get("package", {}).get("name", "unknown")
            ecosystem = vuln.get("package", {}).get("ecosystem", "unknown")
            patched = vuln.get("first_patched_version", {}).get("identifier")
            current = vuln.get("vulnerable_version_range", "")

            if patched:
                recommendations.append({
                    "priority": "HIGH" if severity == "critical" else "MEDIUM",
                    "action": "upgrade",
                    "package": package,
                    "ecosystem": ecosystem,
                    "from": current,
                    "to": patched,
                    "alert_number": alert.get("number")
                })
            else:
                recommendations.append({
                    "priority": "HIGH" if severity == "critical" else "MEDIUM",
                    "action": "review",
                    "package": package,
                    "ecosystem": ecosystem,
                    "reason": "No patched version available",
                    "alert_number": alert.get("number")
                })

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Security Audit for GreenLang")
    parser.add_argument("--analyze", action="store_true", help="Analyze security alerts")
    parser.add_argument("--fix", action="store_true", help="Generate fix script")
    parser.add_argument("--dismiss-low", action="store_true", help="Dismiss low severity alerts")
    parser.add_argument("--dismiss-dev", action="store_true", help="Dismiss alerts in dev dependencies")
    parser.add_argument("--full-report", action="store_true", help="Generate full report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not any([args.analyze, args.fix, args.dismiss_low, args.dismiss_dev, args.full_report]):
        args.analyze = True  # Default action

    token = get_token()
    if not token:
        print("Error: No GitHub token found")
        print("Set GITHUB_TOKEN or authenticate with: gh auth login")
        sys.exit(1)

    print("=" * 60)
    print("GreenLang Security Audit")
    print("=" * 60)
    print(f"Repository: {OWNER}/{REPO}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    auditor = SecurityAuditor(token)

    # Fetch all alerts
    print("\nFetching Dependabot alerts...")
    dependabot_alerts = auditor.get_dependabot_alerts()
    print(f"  Found: {len(dependabot_alerts)} Dependabot alerts")

    print("\nFetching code scanning alerts...")
    code_alerts = auditor.get_code_scanning_alerts()
    print(f"  Found: {len(code_alerts)} code scanning alerts")

    print("\nFetching secret scanning alerts...")
    secret_alerts = auditor.get_secret_scanning_alerts()
    print(f"  Found: {len(secret_alerts)} secret scanning alerts")

    total_alerts = len(dependabot_alerts) + len(code_alerts) + len(secret_alerts)
    print(f"\nTotal Security Alerts: {total_alerts}")

    if args.analyze or args.full_report:
        if dependabot_alerts:
            analysis = auditor.analyze_alerts()

            if args.json:
                print(json.dumps(analysis, indent=2, default=str))
            else:
                print_analysis(analysis)

    if args.fix:
        analysis = auditor.analyze_alerts()
        recommendations = generate_fix_recommendations(analysis)

        print("\n" + "=" * 60)
        print("FIX RECOMMENDATIONS")
        print("=" * 60)

        for rec in recommendations[:20]:  # Top 20
            print(f"\n[{rec['priority']}] {rec['package']} ({rec['ecosystem']})")
            if rec['action'] == 'upgrade':
                print(f"  Action: Upgrade from {rec['from']} to {rec['to']}")
            else:
                print(f"  Action: {rec['reason']}")
            print(f"  Alert #: {rec['alert_number']}")

    if args.dismiss_low:
        print("\n" + "=" * 60)
        print("DISMISSING LOW SEVERITY ALERTS")
        print("=" * 60)

        analysis = auditor.analyze_alerts()
        low_alerts = analysis["by_severity"].get("low", [])

        for alert in low_alerts:
            alert_num = alert.get("number")
            package = alert.get("security_vulnerability", {}).get("package", {}).get("name", "unknown")
            print(f"  Dismissing #{alert_num} ({package})...", end=" ")

            if auditor.dismiss_alert(alert_num, "tolerable_risk", "Low severity - acceptable risk"):
                print("[DISMISSED]")
            else:
                print("[FAILED]")

    if args.dismiss_dev:
        print("\n" + "=" * 60)
        print("DISMISSING DEV DEPENDENCY ALERTS")
        print("=" * 60)

        # This would require checking if the package is a dev dependency
        # which needs parsing requirements-dev.txt or pyproject.toml [dev] section
        print("  Note: Manual review recommended for dev dependencies")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
