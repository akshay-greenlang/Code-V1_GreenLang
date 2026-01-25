# -*- coding: utf-8 -*-
"""
GL-VCCI Load Testing - Ramp-Up Scenario

Simulates gradual user ramp-up from 0 to 1,000 users over 10 minutes.
Tests system stability during load increase and validates performance targets.

Scenario: Ramp-Up Test
    - Duration: 10 minutes
    - Users: 0 → 1,000 (spawn rate: 1.67 users/second)
    - Pattern: Linear increase
    - Focus: System stability during scaling

Performance Targets:
    - All requests complete successfully (0% error rate)
    - API p95 latency < 200ms throughout ramp-up
    - Database connections stable (no pool exhaustion)
    - Memory usage linear growth (no leaks)

Usage:
    locust -f locustfile_rampup.py --host=http://localhost:8000 \\
           --users=1000 --spawn-rate=1.67 --run-time=10m

    # With web UI
    locust -f locustfile_rampup.py --host=http://localhost:8000

    # Headless with CSV output
    locust -f locustfile_rampup.py --host=http://localhost:8000 \\
           --users=1000 --spawn-rate=1.67 --run-time=10m \\
           --headless --csv=rampup_results

Author: GL-VCCI Team
Version: 1.0.0
"""

import random
import json
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner
from io import BytesIO
import time

from greenlang.determinism import deterministic_random
from load_test_utils import (
    generate_csv_data,
    generate_realistic_procurement_data,
    monitor_system_resources,
)


# ============================================================================
# RAMP-UP USER CLASS
# ============================================================================

class RampUpUser(HttpUser):
    """
    Simulates gradual user ramp-up from 0 to 1,000 users over 10 minutes.

    Task Distribution (weighted):
        - Dashboard queries: 30% (weight=3)
        - Supplier queries: 20% (weight=2)
        - Emissions queries: 20% (weight=2)
        - Create calculations: 10% (weight=1)
        - File uploads: 10% (weight=1)
        - Generate reports: 10% (weight=1)

    User Behavior:
        - Wait 1-5 seconds between tasks (think time)
        - Realistic authentication flow
        - Error handling and retries
        - Correlation IDs for tracking
    """

    # Wait time between tasks (simulates user think time)
    wait_time = between(1, 5)

    # Track user state
    auth_token = None
    headers = {}
    user_id = None
    correlation_id = None

    def on_start(self):
        """
        Called when a user starts.

        Performs:
            - User authentication
            - Token retrieval
            - Header setup
            - Initial state setup
        """
        self.user_id = f"loadtest_{deterministic_random().randint(1, 10000)}"
        self.correlation_id = f"rampup_{int(time.time() * 1000)}_{deterministic_random().randint(1000, 9999)}"

        # Authenticate
        with self.client.post(
            "/api/auth/login",
            json={
                "email": f"{self.user_id}@example.com",
                "password": "LoadTest123!"
            },
            catch_response=True,
            name="AUTH: Login"
        ) as response:
            if response.status_code == 200:
                try:
                    self.auth_token = response.json()["access_token"]
                    self.headers = {
                        "Authorization": f"Bearer {self.auth_token}",
                        "X-Correlation-ID": self.correlation_id,
                        "X-User-ID": self.user_id
                    }
                    response.success()
                except (KeyError, json.JSONDecodeError) as e:
                    response.failure(f"Auth failed: {e}")
            else:
                response.failure(f"Login failed with status {response.status_code}")

    def on_stop(self):
        """Called when a user stops - cleanup if needed."""
        pass

    # ========================================================================
    # READ-HEAVY OPERATIONS (70% of traffic)
    # ========================================================================

    @task(3)
    def get_dashboard(self):
        """
        GET /api/dashboard - Fetch emissions dashboard

        Weight: 30% of all requests
        Expected latency: p95 < 150ms
        """
        with self.client.get(
            "/api/dashboard",
            headers=self.headers,
            catch_response=True,
            name="GET Dashboard"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate response structure
                    required_fields = ["total_emissions", "by_category", "by_supplier"]
                    if all(field in data for field in required_fields):
                        response.success()
                    else:
                        response.failure("Missing required fields in dashboard response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in dashboard response")
            else:
                response.failure(f"Dashboard request failed: {response.status_code}")

    @task(2)
    def get_suppliers(self):
        """
        GET /api/suppliers - List suppliers

        Weight: 20% of all requests
        Expected latency: p95 < 150ms
        Pagination: 100 items per page
        """
        page = deterministic_random().randint(1, 10)
        limit = 100

        with self.client.get(
            f"/api/suppliers?page={page}&limit={limit}",
            headers=self.headers,
            catch_response=True,
            name="GET Suppliers List"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate pagination
                    if "items" in data and "total" in data:
                        response.success()
                    else:
                        response.failure("Invalid supplier list response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in supplier list")
            else:
                response.failure(f"Supplier list failed: {response.status_code}")

    @task(2)
    def get_emissions(self):
        """
        GET /api/emissions - Query emissions data

        Weight: 20% of all requests
        Expected latency: p95 < 200ms (aggregate query)
        Filters: category, date range, limit
        """
        category = deterministic_random().randint(1, 15)  # Scope 3 categories 1-15
        limit = deterministic_random().choice([100, 500, 1000])

        # Random date range (last 90 days)
        days_ago = deterministic_random().randint(1, 90)

        with self.client.get(
            f"/api/emissions?category={category}&limit={limit}&days_ago={days_ago}",
            headers=self.headers,
            catch_response=True,
            name="GET Emissions Data"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "emissions" in data:
                        response.success()
                    else:
                        response.failure("Invalid emissions response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in emissions response")
            else:
                response.failure(f"Emissions query failed: {response.status_code}")

    @task(1)
    def get_supplier_detail(self):
        """
        GET /api/suppliers/{id} - Get specific supplier details

        Weight: 10% of all requests (combined with other tasks)
        Expected latency: p95 < 150ms
        """
        supplier_id = f"SUP-{deterministic_random().randint(1, 10000)}"

        with self.client.get(
            f"/api/suppliers/{supplier_id}",
            headers=self.headers,
            catch_response=True,
            name="GET Supplier Detail"
        ) as response:
            if response.status_code in [200, 404]:
                # 404 is acceptable (supplier might not exist)
                response.success()
            else:
                response.failure(f"Supplier detail failed: {response.status_code}")

    # ========================================================================
    # WRITE OPERATIONS (30% of traffic)
    # ========================================================================

    @task(1)
    def create_calculation(self):
        """
        POST /api/calculations - Create new calculation

        Weight: 10% of all requests
        Expected latency: p95 < 500ms (compute-intensive)
        """
        # Generate realistic calculation request
        categories = [1, 4, 6]  # Purchased goods, Transportation, Business travel
        category = deterministic_random().choice(categories)

        payload = {
            "category": category,
            "tier": deterministic_random().choice([1, 2, 3]),
            "supplier_id": f"SUP-{deterministic_random().randint(1, 10000)}",
            "product": deterministic_random().choice([
                "Steel plate", "Plastic resin", "Electronic components",
                "Chemical compounds", "Textile materials"
            ]),
            "quantity": round(random.uniform(100, 10000), 2),
            "unit": deterministic_random().choice(["kg", "ton", "lb", "unit"]),
            "spend_usd": round(random.uniform(1000, 100000), 2),
            "year": 2024
        }

        with self.client.post(
            "/api/calculations",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="POST Create Calculation"
        ) as response:
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    # Validate calculation result
                    if "co2e_kg" in data and "uncertainty" in data:
                        response.success()
                    else:
                        response.failure("Invalid calculation response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in calculation response")
            else:
                response.failure(f"Calculation failed: {response.status_code}")

    @task(1)
    def upload_file(self):
        """
        POST /api/intake/upload - Upload CSV file

        Weight: 10% of all requests
        Expected latency: p95 < 2s (I/O intensive)
        File size: 100-500 rows
        """
        # Generate CSV data (100-500 rows)
        num_rows = deterministic_random().randint(100, 500)
        csv_data = generate_csv_data(num_rows)

        # Create file-like object
        files = {
            "file": (
                f"procurement_{self.user_id}_{int(time.time())}.csv",
                BytesIO(csv_data.encode('utf-8')),
                "text/csv"
            )
        }

        with self.client.post(
            "/api/intake/upload",
            headers=self.headers,
            files=files,
            catch_response=True,
            name="POST Upload File"
        ) as response:
            if response.status_code in [200, 201, 202]:
                try:
                    data = response.json()
                    # Validate upload response
                    if "job_id" in data or "records_processed" in data:
                        response.success()
                    else:
                        response.failure("Invalid upload response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in upload response")
            else:
                response.failure(f"Upload failed: {response.status_code}")

    @task(1)
    def generate_report(self):
        """
        POST /api/reports/generate - Generate ESRS report

        Weight: 10% of all requests
        Expected latency: p95 < 3s (report generation)
        """
        report_types = ["esrs_e1", "cdp", "ifrs_s2"]
        payload = {
            "report_type": deterministic_random().choice(report_types),
            "year": deterministic_random().choice([2023, 2024]),
            "format": deterministic_random().choice(["pdf", "json", "xlsx"])
        }

        with self.client.post(
            "/api/reports/generate",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="POST Generate Report"
        ) as response:
            if response.status_code in [200, 201, 202]:
                try:
                    data = response.json()
                    # Validate report generation response
                    if "report_id" in data or "download_url" in data:
                        response.success()
                    else:
                        response.failure("Invalid report generation response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in report response")
            else:
                response.failure(f"Report generation failed: {response.status_code}")


# ============================================================================
# EVENT HANDLERS
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 80)
    print("GL-VCCI RAMP-UP LOAD TEST STARTING")
    print("=" * 80)
    print(f"Target: 0 → 1,000 users over 10 minutes")
    print(f"Host: {environment.host}")
    print(f"Spawn rate: 1.67 users/second")
    print("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 80)
    print("GL-VCCI RAMP-UP LOAD TEST COMPLETED")
    print("=" * 80)

    # Get statistics
    stats = environment.stats
    total_rps = stats.total.current_rps
    total_fail_ratio = stats.total.fail_ratio

    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average RPS: {total_rps:.2f}")
    print(f"Failure rate: {total_fail_ratio * 100:.2f}%")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {stats.total.min_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time:.2f}ms")

    # Check against targets
    p95_latency = stats.total.get_response_time_percentile(0.95)
    p99_latency = stats.total.get_response_time_percentile(0.99)

    print("\nPerformance Targets:")
    print(f"  p95 latency: {p95_latency:.2f}ms (target: <200ms) - {'✅ PASS' if p95_latency < 200 else '❌ FAIL'}")
    print(f"  p99 latency: {p99_latency:.2f}ms (target: <500ms) - {'✅ PASS' if p99_latency < 500 else '❌ FAIL'}")
    print(f"  Error rate: {total_fail_ratio * 100:.2f}% (target: <0.1%) - {'✅ PASS' if total_fail_ratio < 0.001 else '❌ FAIL'}")

    print("=" * 80)


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test environment."""
    if not isinstance(environment.runner, WorkerRunner):
        print("\nInitializing Ramp-Up Load Test...")
        print("Performance targets:")
        print("  - p95 latency: <200ms")
        print("  - Error rate: <0.1%")
        print("  - Memory: stable growth")
        print("  - Database: stable connections")
        print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import os
    import sys

    # Run Locust programmatically for testing
    print("To run this load test, use:")
    print("\n  locust -f locustfile_rampup.py --host=http://localhost:8000 \\")
    print("         --users=1000 --spawn-rate=1.67 --run-time=10m\n")
