# -*- coding: utf-8 -*-
"""
Automated Security Scanning Pipeline for CSRD Reporting Platform
=================================================================

Integrates multiple security scanning tools:
- Bandit: Python security linter
- Safety: Dependency vulnerability scanner
- Semgrep: Advanced SAST scanning
- Secrets detection: Custom patterns for sensitive data

Author: GreenLang Security Team
Date: 2025-10-20
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import re
from greenlang.determinism import DeterministicClock


class SecurityScanner:
    """Orchestrates multiple security scanning tools."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "timestamp": DeterministicClock.now().isoformat(),
            "project": "CSRD-Reporting-Platform",
            "scans": {},
            "summary": {}
        }
        self.exit_code = 0

    def run_bandit(self) -> Dict[str, Any]:
        """Run Bandit Python security scanner."""
        print("\n" + "="*80)
        print("RUNNING BANDIT - Python Security Scanner")
        print("="*80)

        try:
            # Run Bandit with JSON output
            cmd = [
                "bandit",
                "-r", str(self.project_root / "agents"),
                "-r", str(self.project_root / "utils"),
                "-f", "json",
                "-o", str(self.project_root / "bandit_report.json"),
                "-ll",  # Only report medium and high severity
                "--exclude", "*/tests/*,*/venv/*,*/.venv/*"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            # Load results
            report_path = self.project_root / "bandit_report.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    bandit_results = json.load(f)

                # Parse results
                results = bandit_results.get("results", [])
                metrics = bandit_results.get("metrics", {})

                high_severity = [r for r in results if r.get("issue_severity") == "HIGH"]
                medium_severity = [r for r in results if r.get("issue_severity") == "MEDIUM"]

                scan_result = {
                    "status": "completed",
                    "total_issues": len(results),
                    "high_severity": len(high_severity),
                    "medium_severity": len(medium_severity),
                    "low_severity": len(results) - len(high_severity) - len(medium_severity),
                    "files_scanned": metrics.get("_totals", {}).get("loc", 0),
                    "issues": results[:10],  # Top 10 issues
                    "report_file": "bandit_report.json"
                }

                # Print summary
                print(f"✓ Bandit scan completed")
                print(f"  Files scanned: {scan_result['files_scanned']}")
                print(f"  Total issues: {scan_result['total_issues']}")
                print(f"  HIGH severity: {scan_result['high_severity']}")
                print(f"  MEDIUM severity: {scan_result['medium_severity']}")

                if scan_result['high_severity'] > 0:
                    print("\n⚠️  WARNING: High severity issues found!")
                    self.exit_code = 1
                    for issue in high_severity[:5]:
                        print(f"  - {issue.get('test_id')}: {issue.get('issue_text')}")
                        print(f"    File: {issue.get('filename')}:{issue.get('line_number')}")

                return scan_result
            else:
                return {
                    "status": "error",
                    "message": "Bandit report not generated"
                }

        except FileNotFoundError:
            print("❌ Bandit not installed. Install with: pip install bandit")
            return {
                "status": "not_installed",
                "message": "Bandit not found"
            }
        except Exception as e:
            print(f"❌ Bandit scan failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def run_safety(self) -> Dict[str, Any]:
        """Run Safety dependency vulnerability scanner."""
        print("\n" + "="*80)
        print("RUNNING SAFETY - Dependency Vulnerability Scanner")
        print("="*80)

        try:
            # Check for requirements.txt
            req_file = self.project_root / "requirements.txt"
            if not req_file.exists():
                print("⚠️  requirements.txt not found")
                return {
                    "status": "skipped",
                    "message": "requirements.txt not found"
                }

            # Run Safety with JSON output
            cmd = [
                "safety",
                "check",
                "--file", str(req_file),
                "--json",
                "--output", str(self.project_root / "safety_report.json")
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            # Parse output (Safety may write to stdout or file)
            report_path = self.project_root / "safety_report.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    safety_results = json.load(f)
            elif result.stdout:
                safety_results = json.loads(result.stdout)
            else:
                safety_results = []

            # Parse vulnerabilities
            vulnerabilities = safety_results if isinstance(safety_results, list) else []

            critical = [v for v in vulnerabilities if v.get("severity") == "critical"]
            high = [v for v in vulnerabilities if v.get("severity") == "high"]
            medium = [v for v in vulnerabilities if v.get("severity") == "medium"]

            scan_result = {
                "status": "completed",
                "total_vulnerabilities": len(vulnerabilities),
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(vulnerabilities) - len(critical) - len(high) - len(medium),
                "vulnerabilities": vulnerabilities[:10],  # Top 10
                "report_file": "safety_report.json"
            }

            # Print summary
            print(f"✓ Safety scan completed")
            print(f"  Total vulnerabilities: {scan_result['total_vulnerabilities']}")
            print(f"  CRITICAL: {scan_result['critical']}")
            print(f"  HIGH: {scan_result['high']}")
            print(f"  MEDIUM: {scan_result['medium']}")

            if scan_result['critical'] > 0 or scan_result['high'] > 0:
                print("\n⚠️  WARNING: Critical/High vulnerabilities found!")
                self.exit_code = 1
                for vuln in (critical + high)[:5]:
                    print(f"  - {vuln.get('package')}: {vuln.get('vulnerability')}")
                    print(f"    Installed: {vuln.get('installed_version')} | Fixed: {vuln.get('fixed_version', 'N/A')}")

            return scan_result

        except FileNotFoundError:
            print("❌ Safety not installed. Install with: pip install safety")
            return {
                "status": "not_installed",
                "message": "Safety not found"
            }
        except Exception as e:
            print(f"❌ Safety scan failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def run_semgrep(self) -> Dict[str, Any]:
        """Run Semgrep advanced SAST scanner."""
        print("\n" + "="*80)
        print("RUNNING SEMGREP - Advanced SAST Scanner")
        print("="*80)

        try:
            # Run Semgrep with auto config
            cmd = [
                "semgrep",
                "--config=auto",
                "--json",
                "--output", str(self.project_root / "semgrep_report.json"),
                str(self.project_root / "agents"),
                str(self.project_root / "utils")
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            # Load results
            report_path = self.project_root / "semgrep_report.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    semgrep_results = json.load(f)

                results = semgrep_results.get("results", [])

                # Categorize by severity
                error = [r for r in results if r.get("extra", {}).get("severity") == "ERROR"]
                warning = [r for r in results if r.get("extra", {}).get("severity") == "WARNING"]

                scan_result = {
                    "status": "completed",
                    "total_findings": len(results),
                    "errors": len(error),
                    "warnings": len(warning),
                    "info": len(results) - len(error) - len(warning),
                    "findings": results[:10],  # Top 10
                    "report_file": "semgrep_report.json"
                }

                # Print summary
                print(f"✓ Semgrep scan completed")
                print(f"  Total findings: {scan_result['total_findings']}")
                print(f"  ERRORS: {scan_result['errors']}")
                print(f"  WARNINGS: {scan_result['warnings']}")

                if scan_result['errors'] > 0:
                    print("\n⚠️  WARNING: Semgrep errors found!")
                    self.exit_code = 1
                    for finding in error[:5]:
                        print(f"  - {finding.get('check_id')}: {finding.get('extra', {}).get('message')}")
                        print(f"    File: {finding.get('path')}:{finding.get('start', {}).get('line')}")

                return scan_result
            else:
                return {
                    "status": "error",
                    "message": "Semgrep report not generated"
                }

        except FileNotFoundError:
            print("❌ Semgrep not installed. Install with: pip install semgrep")
            return {
                "status": "not_installed",
                "message": "Semgrep not found"
            }
        except Exception as e:
            print(f"❌ Semgrep scan failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def scan_secrets(self) -> Dict[str, Any]:
        """Scan for hardcoded secrets and sensitive data."""
        print("\n" + "="*80)
        print("RUNNING SECRETS DETECTION")
        print("="*80)

        # Patterns for common secrets
        patterns = {
            "AWS Access Key": r"AKIA[0-9A-Z]{16}",
            "AWS Secret Key": r"aws_secret_access_key\s*=\s*['\"]([^'\"]+)['\"]",
            "API Key": r"api[_-]?key\s*=\s*['\"]([^'\"]+)['\"]",
            "Password": r"password\s*=\s*['\"]([^'\"]+)['\"]",
            "Secret": r"secret\s*=\s*['\"]([^'\"]+)['\"]",
            "Token": r"token\s*=\s*['\"]([^'\"]+)['\"]",
            "Private Key": r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----",
        }

        findings = []
        excluded_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', 'tests'}

        try:
            # Scan Python files
            for py_file in self.project_root.rglob("*.py"):
                # Skip excluded directories
                if any(excluded in py_file.parts for excluded in excluded_dirs):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    for secret_type, pattern in patterns.items():
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Skip test files and comments
                            line_num = content[:match.start()].count('\n') + 1
                            line = content.split('\n')[line_num - 1]

                            if '#' not in line and 'test' not in py_file.name.lower():
                                findings.append({
                                    "type": secret_type,
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": line_num,
                                    "match": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0)
                                })
                except Exception as e:
                    print(f"Warning: Could not scan {py_file}: {e}")

            scan_result = {
                "status": "completed",
                "total_findings": len(findings),
                "findings": findings
            }

            # Print summary
            print(f"✓ Secrets scan completed")
            print(f"  Total potential secrets: {scan_result['total_findings']}")

            if scan_result['total_findings'] > 0:
                print("\n⚠️  WARNING: Potential secrets found!")
                self.exit_code = 1
                for finding in findings[:10]:
                    print(f"  - {finding['type']}: {finding['file']}:{finding['line']}")

            return scan_result

        except Exception as e:
            print(f"❌ Secrets scan failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def generate_summary(self) -> None:
        """Generate overall security summary."""
        print("\n" + "="*80)
        print("SECURITY SCAN SUMMARY")
        print("="*80)

        # Aggregate results
        total_critical = 0
        total_high = 0
        total_medium = 0

        for scan_name, scan_data in self.results["scans"].items():
            if scan_data.get("status") == "completed":
                if scan_name == "bandit":
                    total_high += scan_data.get("high_severity", 0)
                    total_medium += scan_data.get("medium_severity", 0)
                elif scan_name == "safety":
                    total_critical += scan_data.get("critical", 0)
                    total_high += scan_data.get("high", 0)
                    total_medium += scan_data.get("medium", 0)
                elif scan_name == "semgrep":
                    total_high += scan_data.get("errors", 0)
                    total_medium += scan_data.get("warnings", 0)
                elif scan_name == "secrets":
                    total_high += scan_data.get("total_findings", 0)

        self.results["summary"] = {
            "critical_issues": total_critical,
            "high_issues": total_high,
            "medium_issues": total_medium,
            "overall_status": "PASS" if self.exit_code == 0 else "FAIL"
        }

        # Print summary
        print(f"\n📊 Overall Results:")
        print(f"  CRITICAL: {total_critical}")
        print(f"  HIGH: {total_high}")
        print(f"  MEDIUM: {total_medium}")
        print(f"\n  Overall Status: {self.results['summary']['overall_status']}")

        if self.exit_code == 0:
            print("\n✅ All security scans passed!")
        else:
            print("\n❌ Security issues found - review reports for details")

        # Save summary
        summary_path = self.project_root / "security_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Full report saved to: {summary_path}")

    def run_all(self) -> int:
        """Run all security scans."""
        print("\n" + "="*80)
        print("CSRD REPORTING PLATFORM - AUTOMATED SECURITY SCAN")
        print("="*80)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Project: {self.project_root}")
        print("="*80)

        # Run all scans
        self.results["scans"]["bandit"] = self.run_bandit()
        self.results["scans"]["safety"] = self.run_safety()
        self.results["scans"]["semgrep"] = self.run_semgrep()
        self.results["scans"]["secrets"] = self.scan_secrets()

        # Generate summary
        self.generate_summary()

        return self.exit_code


def main():
    """Main entry point."""
    # Get project root
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = Path(__file__).parent

    # Run scanner
    scanner = SecurityScanner(project_root)
    exit_code = scanner.run_all()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
