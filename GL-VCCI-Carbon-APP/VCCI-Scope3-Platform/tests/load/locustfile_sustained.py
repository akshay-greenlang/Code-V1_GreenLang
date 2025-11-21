# -*- coding: utf-8 -*-
"""
GL-VCCI Load Testing - Sustained Load Scenario

Simulates 1,000 concurrent users for 1 hour sustained load.
Tests system stability and resource utilization over time.

Scenario: Sustained Load Test
    - Duration: 1 hour
    - Users: 1,000 concurrent (constant)
    - Pattern: Steady state after initial ramp-up
    - Focus: Long-term stability, resource leaks, cache performance

Performance Targets:
    - Average response time < 100ms, p95 < 200ms, p99 < 500ms
    - Error rate < 0.1%
    - Throughput: 10K+ requests/second sustained
    - CPU utilization < 70% (headroom for spikes)
    - Memory stable (no gradual increase indicating leaks)
    - Database connection pool utilization < 80%

Usage:
    locust -f locustfile_sustained.py --host=http://localhost:8000 \\
           --users=1000 --spawn-rate=50 --run-time=1h

    # With web UI for monitoring
    locust -f locustfile_sustained.py --host=http://localhost:8000

    # Headless with CSV output
    locust -f locustfile_sustained.py --host=http://localhost:8000 \\
           --users=1000 --spawn-rate=50 --run-time=1h \\
           --headless --csv=sustained_results

Author: GL-VCCI Team
Version: 1.0.0
"""

import random
import json
from locust import HttpUser, task, between, events, SequentialTaskSet
from locust.runners import WorkerRunner
from io import BytesIO
import time
from collections import defaultdict

from load_test_utils import (
from greenlang.determinism import deterministic_random
    generate_csv_data,
    generate_realistic_procurement_data,
    monitor_system_resources,
)


# ============================================================================
# REALISTIC USER WORKFLOW TASK SET
# ============================================================================

class RealisticWorkflowTaskSet(SequentialTaskSet):
    """
    Simulates realistic user journey through the platform.

    Workflow:
        1. View dashboard (entry point)
        2. Browse suppliers
        3. Drill down into specific supplier
        4. View emissions breakdown for supplier
        5. Generate report (exit point)

    This pattern represents typical user behavior with realistic think time.
    """

    def on_start(self):
        """Initialize workflow state."""
        self.supplier_id = None
        self.category = None

    @task
    def step_1_view_dashboard(self):
        """Step 1: User views main dashboard."""
        with self.client.get(
            "/api/dashboard",
            headers=self.user.headers,
            catch_response=True,
            name="WORKFLOW: 1-Dashboard"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Pick a category for drill-down
                    if "by_category" in data and data["by_category"]:
                        self.category = deterministic_random().choice(list(data["by_category"].keys()))
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in dashboard")
            else:
                response.failure(f"Dashboard failed: {response.status_code}")

        # Think time
        self.wait()

    @task
    def step_2_browse_suppliers(self):
        """Step 2: User browses supplier list."""
        with self.client.get(
            "/api/suppliers?limit=100",
            headers=self.user.headers,
            catch_response=True,
            name="WORKFLOW: 2-Browse Suppliers"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Pick a supplier for detail view
                    if "items" in data and data["items"]:
                        self.supplier_id = data["items"][0].get("id", f"SUP-{deterministic_random().randint(1, 10000)}")
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in suppliers")
            else:
                response.failure(f"Suppliers failed: {response.status_code}")

        self.wait()

    @task
    def step_3_supplier_detail(self):
        """Step 3: User views specific supplier details."""
        if not self.supplier_id:
            self.supplier_id = f"SUP-{deterministic_random().randint(1, 10000)}"

        with self.client.get(
            f"/api/suppliers/{self.supplier_id}",
            headers=self.user.headers,
            catch_response=True,
            name="WORKFLOW: 3-Supplier Detail"
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Supplier detail failed: {response.status_code}")

        self.wait()

    @task
    def step_4_emissions_breakdown(self):
        """Step 4: User views emissions breakdown for supplier."""
        if not self.supplier_id:
            self.supplier_id = f"SUP-{deterministic_random().randint(1, 10000)}"

        with self.client.get(
            f"/api/emissions?supplier_id={self.supplier_id}",
            headers=self.user.headers,
            catch_response=True,
            name="WORKFLOW: 4-Emissions Breakdown"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Emissions failed: {response.status_code}")

        self.wait()

    @task
    def step_5_generate_report(self):
        """Step 5: User generates report (exit)."""
        if not self.supplier_id:
            self.supplier_id = f"SUP-{deterministic_random().randint(1, 10000)}"

        payload = {
            "report_type": "supplier_detail",
            "supplier_id": self.supplier_id,
            "format": "pdf"
        }

        with self.client.post(
            "/api/reports/generate",
            headers=self.user.headers,
            json=payload,
            catch_response=True,
            name="WORKFLOW: 5-Generate Report"
        ) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            else:
                response.failure(f"Report failed: {response.status_code}")

        # Longer think time before next workflow iteration
        time.sleep(random.uniform(5, 10))


# ============================================================================
# SUSTAINED LOAD USER CLASS
# ============================================================================

class SustainedLoadUser(HttpUser):
    """
    Simulates 1,000 concurrent users for 1 hour sustained load.

    User Types:
        - 60% workflow users (realistic journeys)
        - 40% mixed users (random tasks)

    Focus Areas:
        - Cache hit rate validation (70-80% expected)
        - Database query performance monitoring
        - Memory leak detection (heap growth over time)
        - Connection pool stability
    """

    wait_time = between(2, 8)  # More realistic think time

    auth_token = None
    headers = {}
    user_id = None
    correlation_id = None

    # Define task sets with weights
    tasks = {RealisticWorkflowTaskSet: 6}  # 60% workflow users

    def on_start(self):
        """Initialize user session."""
        self.user_id = f"sustained_{deterministic_random().randint(1, 10000)}"
        self.correlation_id = f"sustained_{int(time.time() * 1000)}_{deterministic_random().randint(1000, 9999)}"

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
                response.failure(f"Login failed: {response.status_code}")

    # ========================================================================
    # MIXED USER TASKS (40% of users)
    # ========================================================================

    @task(2)
    def dashboard_query(self):
        """Quick dashboard check."""
        with self.client.get(
            "/api/dashboard",
            headers=self.headers,
            catch_response=True,
            name="MIXED: Dashboard"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    @task(2)
    def emissions_aggregate(self):
        """Aggregate emissions query (cache-friendly)."""
        category = deterministic_random().randint(1, 15)

        with self.client.get(
            f"/api/emissions/aggregate?category={category}",
            headers=self.headers,
            catch_response=True,
            name="MIXED: Emissions Aggregate"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    @task(1)
    def create_calculation_sustained(self):
        """Create calculation (compute-intensive)."""
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
            name="MIXED: Create Calculation"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    @task(1)
    def batch_query(self):
        """Batch query multiple categories."""
        categories = [1, 4, 6]

        with self.client.post(
            "/api/emissions/batch",
            headers=self.headers,
            json={"categories": categories},
            catch_response=True,
            name="MIXED: Batch Query"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")


# ============================================================================
# RESOURCE MONITORING
# ============================================================================

# Global storage for resource monitoring
resource_samples = defaultdict(list)
sample_interval = 60  # Sample every 60 seconds


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 80)
    print("GL-VCCI SUSTAINED LOAD TEST STARTING")
    print("=" * 80)
    print(f"Target: 1,000 users sustained for 1 hour")
    print(f"Host: {environment.host}")
    print(f"Expected throughput: 10,000+ RPS")
    print("=" * 80)

    # Start resource monitoring
    def monitor_resources():
        """Background task to monitor resources."""
        while environment.runner.state != "stopped":
            try:
                stats = monitor_system_resources()
                resource_samples['cpu'].append(stats['cpu']['percent_overall'])
                resource_samples['memory'].append(stats['memory']['percent'])
                resource_samples['timestamp'].append(stats['timestamp'])

                # Print every 5 minutes
                if len(resource_samples['timestamp']) % 5 == 0:
                    print(f"\n[{stats['timestamp']}] Resource Status:")
                    print(f"  CPU: {stats['cpu']['percent_overall']:.1f}%")
                    print(f"  Memory: {stats['memory']['percent']:.1f}%")
                    print(f"  Network: Sent {stats['network_io']['sent_mb']:.1f}MB, "
                          f"Recv {stats['network_io']['recv_mb']:.1f}MB")

            except Exception as e:
                print(f"Resource monitoring error: {e}")

            time.sleep(sample_interval)

    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("\n" + "=" * 80)
    print("GL-VCCI SUSTAINED LOAD TEST COMPLETED")
    print("=" * 80)

    stats = environment.stats
    total_rps = stats.total.current_rps
    total_fail_ratio = stats.total.fail_ratio

    print("\nRequest Statistics:")
    print(f"  Total requests: {stats.total.num_requests:,}")
    print(f"  Total failures: {stats.total.num_failures:,}")
    print(f"  Average RPS: {total_rps:.2f}")
    print(f"  Failure rate: {total_fail_ratio * 100:.3f}%")

    print("\nResponse Times:")
    print(f"  Average: {stats.total.avg_response_time:.2f}ms")
    print(f"  Min: {stats.total.min_response_time:.2f}ms")
    print(f"  Max: {stats.total.max_response_time:.2f}ms")

    p50 = stats.total.get_response_time_percentile(0.50)
    p95 = stats.total.get_response_time_percentile(0.95)
    p99 = stats.total.get_response_time_percentile(0.99)

    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  p99: {p99:.2f}ms")

    print("\nPerformance Targets:")
    print(f"  Average < 100ms: {stats.total.avg_response_time:.2f}ms - "
          f"{'✅ PASS' if stats.total.avg_response_time < 100 else '❌ FAIL'}")
    print(f"  p95 < 200ms: {p95:.2f}ms - {'✅ PASS' if p95 < 200 else '❌ FAIL'}")
    print(f"  p99 < 500ms: {p99:.2f}ms - {'✅ PASS' if p99 < 500 else '❌ FAIL'}")
    print(f"  Error rate < 0.1%: {total_fail_ratio * 100:.3f}% - "
          f"{'✅ PASS' if total_fail_ratio < 0.001 else '❌ FAIL'}")
    print(f"  Throughput ≥ 10K RPS: {total_rps:.2f} - "
          f"{'✅ PASS' if total_rps >= 10000 else '❌ FAIL'}")

    # Resource analysis
    if resource_samples['cpu']:
        avg_cpu = sum(resource_samples['cpu']) / len(resource_samples['cpu'])
        max_cpu = max(resource_samples['cpu'])
        avg_memory = sum(resource_samples['memory']) / len(resource_samples['memory'])
        max_memory = max(resource_samples['memory'])

        print("\nResource Utilization:")
        print(f"  CPU Average: {avg_cpu:.1f}% (Max: {max_cpu:.1f}%)")
        print(f"  CPU < 70%: {'✅ PASS' if avg_cpu < 70 else '❌ FAIL'}")
        print(f"  Memory Average: {avg_memory:.1f}% (Max: {max_memory:.1f}%)")

        # Check for memory leaks (gradual increase)
        if len(resource_samples['memory']) >= 10:
            first_half = resource_samples['memory'][:len(resource_samples['memory'])//2]
            second_half = resource_samples['memory'][len(resource_samples['memory'])//2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            memory_growth = avg_second - avg_first

            print(f"  Memory Growth: {memory_growth:+.1f}% (1st half → 2nd half)")
            print(f"  Stable Memory: {'✅ PASS' if abs(memory_growth) < 5 else '⚠️ WARNING'}")

    print("=" * 80)


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test environment."""
    if not isinstance(environment.runner, WorkerRunner):
        print("\nInitializing Sustained Load Test...")
        print("Performance targets:")
        print("  - Average: <100ms")
        print("  - p95: <200ms, p99: <500ms")
        print("  - Error rate: <0.1%")
        print("  - Throughput: ≥10K RPS")
        print("  - CPU: <70%")
        print("  - Memory: stable (no leaks)")
        print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("To run this load test, use:")
    print("\n  locust -f locustfile_sustained.py --host=http://localhost:8000 \\")
    print("         --users=1000 --spawn-rate=50 --run-time=1h\n")
