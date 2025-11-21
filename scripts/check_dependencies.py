#!/usr/bin/env python3
"""
Check for outdated dependencies and security vulnerabilities
GreenLang Security Tool - Dependency Checker
"""

import subprocess
import sys
import json
from typing import List, Dict, Tuple
from datetime import datetime
import requests
from packaging import version

def check_outdated_packages() -> List[Dict[str, str]]:
    """Check for outdated packages."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True
        )
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error checking outdated packages: {e}")
        return []

def check_security_vulnerabilities() -> List[Dict[str, any]]:
    """Check for known security vulnerabilities using pip-audit."""
    vulnerabilities = []

    try:
        # Try pip-audit if installed
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--format", "json"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            audit_data = json.loads(result.stdout)
            if 'vulnerabilities' in audit_data:
                vulnerabilities = audit_data['vulnerabilities']
    except:
        print("Warning: pip-audit not installed. Install with: pip install pip-audit")

    # Check critical packages manually
    critical_packages = {
        'cryptography': '46.0.3',  # Latest secure version
        'pyjwt': '2.8.0',
        'requests': '2.31.0',
        'httpx': '0.26.0',
        'pyyaml': '6.0.1',
        'lxml': '5.1.0',
        'jinja2': '3.1.3',
        'sqlalchemy': '2.0.25'
    }

    installed = get_installed_packages()

    for package, min_version in critical_packages.items():
        if package in installed:
            current = version.parse(installed[package])
            minimum = version.parse(min_version)

            if current < minimum:
                vulnerabilities.append({
                    'package': package,
                    'installed_version': str(current),
                    'fixed_version': min_version,
                    'severity': 'CRITICAL',
                    'description': f'Version {current} has known vulnerabilities. Update to {min_version} or later.'
                })

    return vulnerabilities

def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages with their versions."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True,
        text=True
    )
    packages = json.loads(result.stdout)
    return {pkg["name"].lower(): pkg["version"] for pkg in packages}

def check_license_compliance() -> List[Dict[str, str]]:
    """Check for license compliance issues."""
    problematic_licenses = []

    # Licenses that may have compatibility issues
    restricted_licenses = ['GPL', 'LGPL', 'AGPL', 'SSPL', 'Commons Clause']

    try:
        # Try pip-licenses if installed
        result = subprocess.run(
            [sys.executable, "-m", "piplicenses", "--format=json"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            licenses = json.loads(result.stdout)
            for pkg in licenses:
                for restricted in restricted_licenses:
                    if restricted.lower() in pkg.get('License', '').lower():
                        problematic_licenses.append({
                            'package': pkg['Name'],
                            'version': pkg['Version'],
                            'license': pkg['License'],
                            'issue': f'Potentially restrictive license: {restricted}'
                        })
    except:
        print("Note: pip-licenses not installed. Install with: pip install pip-licenses")

    return problematic_licenses

def verify_package_integrity() -> Dict[str, any]:
    """Verify installed package integrity."""
    print("Verifying package integrity...")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True
    )

    integrity_ok = result.returncode == 0
    issues = result.stdout if not integrity_ok else None

    return {
        'integrity_ok': integrity_ok,
        'issues': issues
    }

def generate_security_report() -> Dict[str, any]:
    """Generate comprehensive security report."""

    print("GreenLang Dependency Security Check")
    print("=" * 50)
    print(f"Scan Date: {datetime.now().isoformat()}")
    print(f"Python Version: {sys.version}")
    print()

    # Check outdated packages
    print("Checking for outdated packages...")
    outdated = check_outdated_packages()

    # Check security vulnerabilities
    print("Checking for security vulnerabilities...")
    vulnerabilities = check_security_vulnerabilities()

    # Check license compliance
    print("Checking license compliance...")
    license_issues = check_license_compliance()

    # Verify integrity
    print("Verifying package integrity...")
    integrity = verify_package_integrity()

    # Generate report
    report = {
        'scan_date': datetime.now().isoformat(),
        'python_version': sys.version,
        'outdated_packages': outdated,
        'vulnerabilities': vulnerabilities,
        'license_issues': license_issues,
        'integrity': integrity,
        'summary': {
            'total_outdated': len(outdated),
            'total_vulnerabilities': len(vulnerabilities),
            'critical_vulnerabilities': len([v for v in vulnerabilities if v.get('severity') == 'CRITICAL']),
            'license_issues': len(license_issues),
            'integrity_ok': integrity['integrity_ok']
        }
    }

    return report

def print_report(report: Dict[str, any]):
    """Print formatted security report."""

    print("\n" + "=" * 50)
    print("SECURITY REPORT SUMMARY")
    print("=" * 50)

    summary = report['summary']

    # Overall status
    if summary['critical_vulnerabilities'] > 0:
        status = "❌ CRITICAL ISSUES FOUND"
        status_color = "\033[91m"  # Red
    elif summary['total_vulnerabilities'] > 0 or not summary['integrity_ok']:
        status = "⚠️  WARNINGS FOUND"
        status_color = "\033[93m"  # Yellow
    else:
        status = "✅ ALL CHECKS PASSED"
        status_color = "\033[92m"  # Green

    print(f"\n{status_color}Status: {status}\033[0m")

    # Summary statistics
    print("\nStatistics:")
    print(f"  Outdated packages: {summary['total_outdated']}")
    print(f"  Security vulnerabilities: {summary['total_vulnerabilities']}")
    print(f"  Critical vulnerabilities: {summary['critical_vulnerabilities']}")
    print(f"  License issues: {summary['license_issues']}")
    print(f"  Package integrity: {'✅ OK' if summary['integrity_ok'] else '❌ FAILED'}")

    # Critical vulnerabilities
    if report['vulnerabilities']:
        print("\n⚠️  SECURITY VULNERABILITIES:")
        for vuln in report['vulnerabilities']:
            severity = vuln.get('severity', 'UNKNOWN')
            print(f"\n  [{severity}] {vuln['package']}:")
            print(f"    Installed: {vuln.get('installed_version', 'unknown')}")
            print(f"    Fixed in: {vuln.get('fixed_version', 'unknown')}")
            print(f"    {vuln.get('description', 'No description available')}")

    # Outdated packages (top 10)
    if report['outdated_packages']:
        print("\n📦 OUTDATED PACKAGES (Top 10):")
        for pkg in report['outdated_packages'][:10]:
            print(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")

    # License issues
    if report['license_issues']:
        print("\n⚖️  LICENSE COMPLIANCE ISSUES:")
        for issue in report['license_issues']:
            print(f"  {issue['package']} ({issue['version']}): {issue['issue']}")

    # Integrity issues
    if not summary['integrity_ok']:
        print("\n❌ PACKAGE INTEGRITY ISSUES:")
        print(report['integrity']['issues'])

    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    if summary['critical_vulnerabilities'] > 0:
        print("  1. ⚠️  IMMEDIATELY update packages with critical vulnerabilities")
    if summary['total_outdated'] > 0:
        print("  2. Review and update outdated packages")
        print("     Run: python scripts/update_dependencies.py")
    if summary['license_issues'] > 0:
        print("  3. Review license compliance issues")
    print("  4. Regenerate pinned requirements:")
    print("     Run: python scripts/generate_pinned_requirements.py")
    print("  5. Monitor for new vulnerabilities:")
    print("     Enable Dependabot in GitHub repository")

    print("\n" + "=" * 50)

def save_report(report: Dict[str, any], filename: str = "dependency_security_report.json"):
    """Save report to JSON file."""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check dependency security and compliance")
    parser.add_argument(
        "--save",
        help="Save report to file",
        default="dependency_security_report.json"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format"
    )

    args = parser.parse_args()

    # Generate report
    report = generate_security_report()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    if args.save:
        save_report(report, args.save)