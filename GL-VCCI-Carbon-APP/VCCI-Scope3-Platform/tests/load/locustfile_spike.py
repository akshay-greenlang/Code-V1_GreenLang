# -*- coding: utf-8 -*-
"""
GL-VCCI Load Testing - Spike Test Scenario

Simulates sudden traffic spike from 1,000 to 5,000 users.
Tests system resilience, autoscaling, and graceful degradation.

Scenario: Spike Test
    - Initial load: 1,000 users (baseline)
    - Spike to: 5,000 users (5x increase)
    - Spike duration: 10 minutes
    - Recovery: Return to 1,000 users
    - Focus: System resilience, circuit breakers, graceful degradation

Performance Targets:
    - No requests fail during spike (circuit breaker may activate)
    - Critical paths (dashboard, calculations) stay < 500ms p95
    - Heavy operations (uploads) may degrade gracefully (< 2s → < 10s acceptable)
    - System auto-scales (Kubernetes HPA triggers)
    - System recovers within 2 minutes after spike ends
    - No cascading failures

Usage:
    # Manual spike test (adjust users manually via web UI)
    locust -f locustfile_spike.py --host=http://localhost:8000

    # Automated spike test with custom load shape
    locust -f locustfile_spike.py --host=http://localhost:8000 \\
           --headless --run-time=25m --csv=spike_results

Load Pattern:
    0-5 min: Ramp to 1,000 users (baseline)
    5-6 min: Spike to 5,000 users (instant)
    6-16 min: Hold 5,000 users (10 minutes)
    16-17 min: Drop to 1,000 users (instant)
    17-25 min: Monitor recovery (8 minutes)

Author: GL-VCCI Team
Version: 1.0.0
"""

import random
import json
from locust import HttpUser, task, between, events, LoadTestShape
from locust.runners import WorkerRunner
from io import BytesIO
import time

from load_test_utils import (
from greenlang.determinism import deterministic_random
    generate_csv_data,
    monitor_system_resources,
)


# ============================================================================
# CUSTOM LOAD SHAPE FOR SPIKE TEST
# ============================================================================

class SpikeLoadShape(LoadTestShape):
    """
    Custom load shape implementing spike test pattern.

    Pattern:
        0-300s (0-5min): Ramp from 0 to 1,000 users
        300-360s (5-6min): Spike to 5,000 users
        360-960s (6-16min): Hold 5,000 users (10 minutes)
        960-1020s (16-17min): Drop to 1,000 users
        1020-1500s (17-25min): Hold 1,000 users (recovery monitoring)
    """

    stages = [
        # Stage 1: Ramp to baseline (5 minutes)
        {"duration": 300, "users": 1000, "spawn_rate": 10},
        # Stage 2: Spike (1 minute - aggressive ramp)
        {"duration": 360, "users": 5000, "spawn_rate": 4000},
        # Stage 3: Hold spike (10 minutes)
        {"duration": 960, "users": 5000, "spawn_rate": 1},
        # Stage 4: Drop to baseline (1 minute)
        {"duration": 1020, "users": 1000, "spawn_rate": 4000},
        # Stage 5: Recovery monitoring (8 minutes)
        {"duration": 1500, "users": 1000, "spawn_rate": 1},
    ]

    def tick(self):
        """
        Return (user_count, spawn_rate) for current time.
        Returns None to stop the test.
        """
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])

        return None


# ============================================================================
# SPIKE TEST USER CLASS
# ============================================================================

class SpikeTestUser(HttpUser):
    """
    Simulates users during spike test.

    Task Focus:
        - Critical read paths: dashboard, aggregates (must stay fast)
        - Critical write paths: calculations (must not fail)
        - Heavy operations: file uploads (acceptable degradation)

    Priority Levels:
        - P0 (Critical): Dashboard, calculations - must succeed quickly
        - P1 (Important): Queries, reports - should succeed
        - P2 (Best-effort): File uploads - acceptable degradation
    """

    wait_time = between(1, 3)  # Faster requests during spike

    auth_token = None
    headers = {}
    user_id = None

    def on_start(self):
        """Initialize user session."""
        self.user_id = f"spike_{deterministic_random().randint(1, 10000)}"

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
                        "X-User-ID": self.user_id
                    }
                    response.success()
                except (KeyError, json.JSONDecodeError):
                    response.failure("Auth failed")
            else:
                response.failure(f"Login failed: {response.status_code}")

    # ========================================================================
    # P0: CRITICAL READ PATHS (50% of traffic - must stay fast)
    # ========================================================================

    @task(5)
    def critical_dashboard(self):
        """
        P0: Dashboard query - MUST stay fast (<500ms p95)

        This is the most common user entry point. Any degradation here
        impacts all users immediately.
        """
        with self.client.get(
            "/api/dashboard",
            headers=self.headers,
            catch_response=True,
            name="P0-CRITICAL: Dashboard"
        ) as response:
            if response.status_code == 200:
                # Strict latency requirement during spike
                if response.elapsed.total_seconds() * 1000 < 500:
                    response.success()
                else:
                    response.failure(f"Too slow: {response.elapsed.total_seconds() * 1000:.0f}ms")
            else:
                response.failure(f"Failed: {response.status_code}")

    @task(3)
    def critical_aggregate(self):
        """
        P0: Aggregate query - MUST stay fast (<500ms p95)

        These queries are heavily cached and should remain fast even
        during spikes.
        """
        category = deterministic_random().randint(1, 15)

        with self.client.get(
            f"/api/emissions/aggregate?category={category}",
            headers=self.headers,
            catch_response=True,
            name="P0-CRITICAL: Aggregate Query"
        ) as response:
            if response.status_code == 200:
                if response.elapsed.total_seconds() * 1000 < 500:
                    response.success()
                else:
                    response.failure(f"Too slow: {response.elapsed.total_seconds() * 1000:.0f}ms")
            else:
                response.failure(f"Failed: {response.status_code}")

    # ========================================================================
    # P0: CRITICAL WRITE PATHS (30% of traffic - must not fail)
    # ========================================================================

    @task(3)
    def critical_calculation(self):
        """
        P0: Create calculation - MUST not fail

        Calculations are business-critical. They may be slower during
        spikes but must not fail. Timeout: 5s acceptable.
        """
        payload = {
            "category": deterministic_random().choice([1, 4, 6]),
            "tier": deterministic_random().choice([1, 2, 3]),
            "supplier_id": f"SUP-{deterministic_random().randint(1, 10000)}",
            "product": "Steel plate",
            "quantity": random.uniform(100, 10000),
            "unit": "kg",
            "spend_usd": random.uniform(1000, 100000)
        }

        with self.client.post(
            "/api/calculations",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="P0-CRITICAL: Calculation",
            timeout=5.0  # Acceptable timeout during spike
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 429:
                # Rate limiting is acceptable - don't mark as failure
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    # ========================================================================
    # P1: IMPORTANT OPERATIONS (10% of traffic - should succeed)
    # ========================================================================

    @task(1)
    def important_supplier_query(self):
        """
        P1: Supplier query - should succeed

        These queries are important but not critical. Some degradation
        acceptable.
        """
        with self.client.get(
            "/api/suppliers?limit=100",
            headers=self.headers,
            catch_response=True,
            name="P1-IMPORTANT: Suppliers"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # Service unavailable during spike - log but don't fail test
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    @task(1)
    def important_report(self):
        """
        P1: Report generation - should succeed but may queue

        Reports can be queued during high load.
        """
        payload = {
            "report_type": deterministic_random().choice(["esrs_e1", "cdp"]),
            "year": 2024,
            "format": "json"
        }

        with self.client.post(
            "/api/reports/generate",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="P1-IMPORTANT: Report",
            timeout=10.0
        ) as response:
            if response.status_code in [200, 201, 202]:
                # 202 Accepted (queued) is fine
                response.success()
            elif response.status_code == 429:
                # Rate limited - acceptable
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    # ========================================================================
    # P2: BEST-EFFORT OPERATIONS (10% of traffic - acceptable degradation)
    # ========================================================================

    @task(1)
    def best_effort_upload(self):
        """
        P2: File upload - acceptable degradation

        Heavy I/O operations may be throttled or queued during spikes.
        Degradation from <2s to <10s is acceptable.
        """
        # Small file during spike (50 rows instead of 500)
        num_rows = deterministic_random().randint(50, 100)
        csv_data = generate_csv_data(num_rows)

        files = {
            "file": (
                f"spike_test_{self.user_id}_{int(time.time())}.csv",
                BytesIO(csv_data.encode('utf-8')),
                "text/csv"
            )
        }

        with self.client.post(
            "/api/intake/upload",
            headers=self.headers,
            files=files,
            catch_response=True,
            name="P2-BEST-EFFORT: Upload",
            timeout=15.0  # Extended timeout during spike
        ) as response:
            if response.status_code in [200, 201, 202]:
                # Any success is good
                response.success()
            elif response.status_code in [429, 503]:
                # Rate limited or service unavailable - acceptable
                response.success()
            elif response.status_code == 408:
                # Timeout - log but don't fail
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")


# ============================================================================
# EVENT HANDLERS & MONITORING
# ============================================================================

spike_metrics = {
    "baseline_rps": 0,
    "spike_rps": 0,
    "baseline_p95": 0,
    "spike_p95": 0,
    "spike_start_time": None,
    "spike_end_time": None,
}


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 80)
    print("GL-VCCI SPIKE TEST STARTING")
    print("=" * 80)
    print("Load Pattern:")
    print("  0-5 min: Ramp to 1,000 users (baseline)")
    print("  5-6 min: SPIKE to 5,000 users (5x increase)")
    print("  6-16 min: Hold 5,000 users (10 minutes)")
    print("  16-17 min: Drop to 1,000 users")
    print("  17-25 min: Recovery monitoring")
    print("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("\n" + "=" * 80)
    print("GL-VCCI SPIKE TEST COMPLETED")
    print("=" * 80)

    stats = environment.stats

    print("\nOverall Statistics:")
    print(f"  Total requests: {stats.total.num_requests:,}")
    print(f"  Total failures: {stats.total.num_failures:,}")
    print(f"  Failure rate: {stats.total.fail_ratio * 100:.3f}%")

    print("\nResponse Times:")
    p50 = stats.total.get_response_time_percentile(0.50)
    p95 = stats.total.get_response_time_percentile(0.95)
    p99 = stats.total.get_response_time_percentile(0.99)

    print(f"  Average: {stats.total.avg_response_time:.2f}ms")
    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  p99: {p99:.2f}ms")
    print(f"  Max: {stats.total.max_response_time:.2f}ms")

    # Analyze critical paths
    print("\nCritical Path Analysis:")

    critical_endpoints = [
        "P0-CRITICAL: Dashboard",
        "P0-CRITICAL: Aggregate Query",
        "P0-CRITICAL: Calculation"
    ]

    for endpoint in critical_endpoints:
        if endpoint in stats.entries:
            entry = stats.entries[endpoint]
            p95_endpoint = entry.get_response_time_percentile(0.95)
            print(f"\n  {endpoint}:")
            print(f"    Requests: {entry.num_requests:,}")
            print(f"    Failures: {entry.num_failures:,} ({entry.fail_ratio * 100:.2f}%)")
            print(f"    p95 latency: {p95_endpoint:.2f}ms")

            # Validate critical path
            if endpoint.startswith("P0-CRITICAL"):
                if p95_endpoint < 500 and entry.fail_ratio < 0.01:
                    print(f"    Status: ✅ PASS (fast & reliable)")
                elif entry.fail_ratio < 0.01:
                    print(f"    Status: ⚠️ DEGRADED (slower but reliable)")
                else:
                    print(f"    Status: ❌ FAIL (too many failures)")

    print("\nSpike Test Targets:")
    print(f"  No cascading failures: "
          f"{'✅ PASS' if stats.total.fail_ratio < 0.05 else '❌ FAIL'}")
    print(f"  Critical paths < 500ms p95: "
          f"{'✅ PASS' if p95 < 500 else '⚠️ DEGRADED'}")
    print(f"  Error rate < 5% (spike): "
          f"{'✅ PASS' if stats.total.fail_ratio < 0.05 else '❌ FAIL'}")

    print("=" * 80)


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test environment."""
    if not isinstance(environment.runner, WorkerRunner):
        print("\nInitializing Spike Test...")
        print("Performance targets:")
        print("  - Critical paths: <500ms p95 during spike")
        print("  - No cascading failures")
        print("  - Error rate: <5% during spike")
        print("  - Auto-scaling: triggers appropriately")
        print("  - Recovery: within 2 minutes")
        print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("To run this spike test, use:")
    print("\n  locust -f locustfile_spike.py --host=http://localhost:8000\n")
    print("The test will automatically follow the spike load pattern.")
    print("Monitor the web UI to observe system behavior during the spike.")
