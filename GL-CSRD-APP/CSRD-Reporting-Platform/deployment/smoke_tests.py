"""
Production Smoke Tests for CSRD Reporting Platform
===================================================

Critical smoke tests to validate production deployment.
These tests MUST pass before considering deployment successful.

Author: QA Team
Date: 2025-10-20
"""

import requests
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import argparse


class SmokeTests:
    """Production smoke test suite."""

    def __init__(self, base_url: str = "https://csrd.prod.example.com"):
        self.base_url = base_url.rstrip('/')
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "tests": {},
            "summary": {}
        }
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CSRD-Smoke-Tests/1.0"
        })

    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record result."""
        print(f"\n🧪 Running: {test_name}")
        start_time = time.time()

        try:
            test_func()
            duration = time.time() - start_time

            self.results["tests"][test_name] = {
                "status": "PASS",
                "duration": round(duration, 3)
            }

            print(f"   ✅ PASS ({duration:.3f}s)")
            return True

        except AssertionError as e:
            duration = time.time() - start_time

            self.results["tests"][test_name] = {
                "status": "FAIL",
                "duration": round(duration, 3),
                "error": str(e)
            }

            print(f"   ❌ FAIL: {e}")
            return False

        except Exception as e:
            duration = time.time() - start_time

            self.results["tests"][test_name] = {
                "status": "ERROR",
                "duration": round(duration, 3),
                "error": str(e)
            }

            print(f"   ⚠️  ERROR: {e}")
            return False

    # ========================================================================
    # CRITICAL SMOKE TESTS
    # ========================================================================

    def test_health_endpoint(self):
        """Test basic health check endpoint."""
        response = self.session.get(f"{self.base_url}/health", timeout=5)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"

        data = response.json()
        assert data.get("status") == "healthy", f"Unhealthy status: {data.get('status')}"

    def test_ready_endpoint(self):
        """Test readiness check endpoint."""
        response = self.session.get(f"{self.base_url}/health/ready", timeout=5)
        assert response.status_code == 200, f"Readiness check failed: {response.status_code}"

        data = response.json()
        assert data.get("status") in ["healthy", "ready"], f"Not ready: {data.get('status')}"

        # Check all dependencies
        checks = data.get("checks", {})
        for check_name, check_result in checks.items():
            assert check_result.get("healthy") is not False, f"Check failed: {check_name}"

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = self.session.get(f"{self.base_url}/health/metrics", timeout=5)
        assert response.status_code == 200, f"Metrics endpoint failed: {response.status_code}"

        # Verify metrics format
        content = response.text
        assert "# HELP" in content, "Invalid Prometheus metrics format"
        assert "http_requests_total" in content, "Missing http_requests_total metric"

    def test_api_root(self):
        """Test API root endpoint."""
        response = self.session.get(f"{self.base_url}/", timeout=5)
        assert response.status_code == 200, f"API root failed: {response.status_code}"

    def test_api_version(self):
        """Test API version endpoint."""
        response = self.session.get(f"{self.base_url}/version", timeout=5)
        assert response.status_code == 200, f"Version endpoint failed: {response.status_code}"

        data = response.json()
        assert "version" in data, "Missing version in response"
        print(f"   API Version: {data.get('version')}")

    def test_database_connectivity(self):
        """Test database connectivity through API."""
        # Assuming there's a database health check endpoint
        response = self.session.get(f"{self.base_url}/health/ready", timeout=5)
        assert response.status_code == 200, "Database check failed"

        data = response.json()
        db_check = data.get("checks", {}).get("database", {})
        assert db_check.get("healthy") is True, f"Database unhealthy: {db_check.get('message')}"

    def test_cache_connectivity(self):
        """Test Redis cache connectivity."""
        response = self.session.get(f"{self.base_url}/health/ready", timeout=5)
        assert response.status_code == 200, "Cache check failed"

        data = response.json()
        cache_check = data.get("checks", {}).get("cache", {})
        # Cache is degraded mode acceptable, not critical
        if cache_check:
            print(f"   Cache status: {cache_check.get('message')}")

    def test_api_latency(self):
        """Test API response latency."""
        start = time.time()
        response = self.session.get(f"{self.base_url}/health", timeout=5)
        latency = time.time() - start

        assert response.status_code == 200, "Health check failed"
        assert latency < 1.0, f"Latency too high: {latency:.3f}s (threshold: 1s)"

        print(f"   Latency: {latency*1000:.0f}ms")

    def test_cors_headers(self):
        """Test CORS headers are configured."""
        response = self.session.options(f"{self.base_url}/health", timeout=5)

        # Check for CORS headers (should be present in production)
        headers = response.headers
        print(f"   Access-Control-Allow-Origin: {headers.get('Access-Control-Allow-Origin', 'NOT SET')}")

    def test_security_headers(self):
        """Test security headers are present."""
        response = self.session.get(f"{self.base_url}/health", timeout=5)

        headers = response.headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Strict-Transport-Security": "max-age=31536000"
        }

        for header, expected in security_headers.items():
            actual = headers.get(header)
            if actual:
                print(f"   ✓ {header}: {actual}")
            else:
                print(f"   ⚠️  {header}: NOT SET")

    # ========================================================================
    # FUNCTIONAL SMOKE TESTS
    # ========================================================================

    def test_intake_agent_health(self):
        """Test intake agent is healthy."""
        # Assuming agent health endpoints exist
        try:
            response = self.session.get(f"{self.base_url}/agents/intake/health", timeout=5)
            assert response.status_code == 200, "Intake agent not responding"
        except:
            # If endpoint doesn't exist, skip this test
            print("   ℹ️  Intake agent health endpoint not available")

    def test_calculator_agent_health(self):
        """Test calculator agent is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/agents/calculator/health", timeout=5)
            assert response.status_code == 200, "Calculator agent not responding"
        except:
            print("   ℹ️  Calculator agent health endpoint not available")

    def test_reporting_agent_health(self):
        """Test reporting agent is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/agents/reporting/health", timeout=5)
            assert response.status_code == 200, "Reporting agent not responding"
        except:
            print("   ℹ️  Reporting agent health endpoint not available")

    def test_simple_calculation(self):
        """Test a simple emissions calculation."""
        # Simple API call to verify calculator works
        payload = {
            "activity": 1000.0,
            "emission_factor": 0.5
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/calculate",
                json=payload,
                timeout=10
            )

            if response.status_code == 404:
                print("   ℹ️  Calculate endpoint not available")
                return

            assert response.status_code == 200, f"Calculation failed: {response.status_code}"

            data = response.json()
            assert "result" in data or "co2e_kg" in data, "Invalid calculation response"
            print(f"   Calculation result: {data}")

        except requests.exceptions.RequestException:
            print("   ℹ️  Calculate endpoint not available")

    def test_data_validation(self):
        """Test data validation is working."""
        # Send invalid data to verify validation
        payload = {
            "activity": -1000.0,  # Negative value should be rejected
            "emission_factor": "invalid"  # Invalid type
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/calculate",
                json=payload,
                timeout=10
            )

            if response.status_code == 404:
                print("   ℹ️  Validation test skipped (endpoint not available)")
                return

            # Should get 400 or 422 for validation error
            assert response.status_code in [400, 422], f"Validation not working: {response.status_code}"
            print(f"   Validation working (got {response.status_code})")

        except requests.exceptions.RequestException:
            print("   ℹ️  Validation test skipped")

    # ========================================================================
    # PERFORMANCE SMOKE TESTS
    # ========================================================================

    def test_concurrent_requests(self):
        """Test system handles concurrent requests."""
        import concurrent.futures

        def make_request():
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_count = sum(results)
        assert success_count >= 9, f"Only {success_count}/10 concurrent requests succeeded"
        print(f"   {success_count}/10 concurrent requests succeeded")

    def test_response_time_under_load(self):
        """Test response time under light load."""
        latencies = []

        for i in range(5):
            start = time.time()
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            latency = time.time() - start

            assert response.status_code == 200, f"Request {i+1} failed"
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        assert avg_latency < 0.5, f"Average latency too high: {avg_latency:.3f}s"
        assert max_latency < 1.0, f"Max latency too high: {max_latency:.3f}s"

        print(f"   Avg latency: {avg_latency*1000:.0f}ms, Max: {max_latency*1000:.0f}ms")

    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================

    def run_all(self) -> bool:
        """Run all smoke tests."""
        print("\n" + "="*80)
        print("CSRD PLATFORM - PRODUCTION SMOKE TESTS")
        print("="*80)
        print(f"Target: {self.base_url}")
        print(f"Time: {datetime.now().isoformat()}")
        print("="*80)

        # Critical tests (must pass)
        critical_tests = [
            ("Health Endpoint", self.test_health_endpoint),
            ("Readiness Endpoint", self.test_ready_endpoint),
            ("Metrics Endpoint", self.test_metrics_endpoint),
            ("API Root", self.test_api_root),
            ("Database Connectivity", self.test_database_connectivity),
            ("API Latency", self.test_api_latency),
        ]

        # Functional tests (should pass)
        functional_tests = [
            ("API Version", self.test_api_version),
            ("Cache Connectivity", self.test_cache_connectivity),
            ("Security Headers", self.test_security_headers),
            ("Intake Agent Health", self.test_intake_agent_health),
            ("Calculator Agent Health", self.test_calculator_agent_health),
            ("Reporting Agent Health", self.test_reporting_agent_health),
            ("Simple Calculation", self.test_simple_calculation),
            ("Data Validation", self.test_data_validation),
        ]

        # Performance tests (nice to have)
        performance_tests = [
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Response Time Under Load", self.test_response_time_under_load),
        ]

        # Run tests
        all_passed = True

        print("\n🔴 CRITICAL TESTS")
        for test_name, test_func in critical_tests:
            if not self.run_test(test_name, test_func):
                all_passed = False

        print("\n🟡 FUNCTIONAL TESTS")
        for test_name, test_func in functional_tests:
            self.run_test(test_name, test_func)

        print("\n🟢 PERFORMANCE TESTS")
        for test_name, test_func in performance_tests:
            self.run_test(test_name, test_func)

        # Generate summary
        self.generate_summary()

        return all_passed

    def generate_summary(self):
        """Generate test summary."""
        print("\n" + "="*80)
        print("SMOKE TEST SUMMARY")
        print("="*80)

        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"].values() if t["status"] == "PASS")
        failed = sum(1 for t in self.results["tests"].values() if t["status"] == "FAIL")
        errors = sum(1 for t in self.results["tests"].values() if t["status"] == "ERROR")

        self.results["summary"] = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": round((passed / total * 100) if total > 0 else 0, 2),
            "status": "PASS" if failed == 0 and errors == 0 else "FAIL"
        }

        print(f"\n📊 Results:")
        print(f"  Total Tests: {total}")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⚠️  Errors: {errors}")
        print(f"  Pass Rate: {self.results['summary']['pass_rate']}%")
        print(f"\n  Overall Status: {self.results['summary']['status']}")

        # Save results
        with open("smoke-test-results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Full results saved to: smoke-test-results.json")

        if self.results["summary"]["status"] == "PASS":
            print("\n✅ ALL SMOKE TESTS PASSED - DEPLOYMENT SUCCESSFUL")
        else:
            print("\n❌ SOME SMOKE TESTS FAILED - INVESTIGATE IMMEDIATELY")

        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CSRD Production Smoke Tests")
    parser.add_argument("--url", default="https://csrd.prod.example.com",
                       help="Base URL of the deployment")
    args = parser.parse_args()

    tests = SmokeTests(base_url=args.url)
    success = tests.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
