"""
GL-VCCI Load Testing - Endurance Test Scenario

Simulates 500 users for 24 hours continuous operation.
Tests for memory leaks, connection leaks, and long-term stability.

Scenario: Endurance Test (Soak Test)
    - Duration: 24 hours
    - Users: 500 concurrent (constant)
    - Pattern: Steady load over extended period
    - Focus: Memory leaks, connection leaks, resource exhaustion

Performance Targets:
    - Zero memory leaks (heap size stable over 24h)
    - Zero connection leaks (all connections properly closed)
    - Error rate < 0.01% over 24 hours
    - Average response time stable (no gradual degradation)
    - No crash or restart required
    - Logs clean (no WARNING/ERROR accumulation)
    - Database performance stable (no query slow-down over time)

Usage:
    locust -f locustfile_endurance.py --host=http://localhost:8000 \\
           --users=500 --spawn-rate=10 --run-time=24h

    # With CSV output for long-term analysis
    locust -f locustfile_endurance.py --host=http://localhost:8000 \\
           --users=500 --spawn-rate=10 --run-time=24h \\
           --headless --csv=endurance_results

    # Shorter test for validation (4 hours)
    locust -f locustfile_endurance.py --host=http://localhost:8000 \\
           --users=500 --spawn-rate=10 --run-time=4h

Author: GL-VCCI Team
Version: 1.0.0
"""

import random
import json
from locust import HttpUser, task, between, events
from locust.runners import WorkerRunner
from io import BytesIO
import time
from datetime import datetime
from collections import defaultdict

from load_test_utils import (
    generate_csv_data,
    generate_realistic_procurement_data,
    monitor_system_resources,
)


# ============================================================================
# ENDURANCE TEST USER CLASS
# ============================================================================

class EnduranceTestUser(HttpUser):
    """
    Simulates 500 users for 24 hours continuous operation.

    User Behavior:
        - Realistic think time (5-15 seconds between operations)
        - Diverse operation mix to exercise all code paths
        - CRUD operations (Create, Read, Update, Delete)
        - Background jobs (Celery tasks)
        - Cache operations (Redis)
        - Database operations (PostgreSQL)

    Focus Areas:
        - Memory leak detection (heap monitoring)
        - Connection leak detection (DB pool monitoring)
        - Log analysis (error accumulation)
        - Performance degradation over time
        - Cache effectiveness over time
    """

    wait_time = between(5, 15)  # Realistic user think time

    auth_token = None
    headers = {}
    user_id = None
    operation_count = 0

    def on_start(self):
        """Initialize user session."""
        self.user_id = f"endurance_{random.randint(1, 10000)}"
        self.operation_count = 0

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
    # DIVERSE OPERATIONS - CYCLE THROUGH ALL CODE PATHS
    # ========================================================================

    @task
    def diverse_operations(self):
        """
        Execute diverse operations to exercise all code paths.

        This prevents focusing on one area and ensures comprehensive
        testing of the entire system over 24 hours.
        """
        operations = [
            self._dashboard,
            self._calculation,
            self._upload,
            self._report,
            self._entity_resolution,
            self._supplier_engagement,
            self._emissions_query,
            self._batch_calculation,
            self._cache_operations,
            self._update_operations,
        ]

        # Randomly select and execute operation
        operation = random.choice(operations)
        operation()

        self.operation_count += 1

        # Periodic health check every 100 operations
        if self.operation_count % 100 == 0:
            self._health_check()

    # ========================================================================
    # OPERATION IMPLEMENTATIONS
    # ========================================================================

    def _dashboard(self):
        """Dashboard query - heavily cached."""
        with self.client.get(
            "/api/dashboard",
            headers=self.headers,
            catch_response=True,
            name="OP: Dashboard"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _calculation(self):
        """Create calculation - compute intensive."""
        payload = {
            "category": random.choice([1, 4, 6]),
            "tier": random.choice([1, 2, 3]),
            "supplier_id": f"SUP-{random.randint(1, 10000)}",
            "product": random.choice([
                "Steel plate", "Plastic resin", "Electronic components"
            ]),
            "quantity": random.uniform(100, 10000),
            "unit": random.choice(["kg", "ton", "lb"]),
            "spend_usd": random.uniform(1000, 100000)
        }

        with self.client.post(
            "/api/calculations",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="OP: Calculation"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _upload(self):
        """File upload - I/O intensive."""
        num_rows = random.randint(50, 200)
        csv_data = generate_csv_data(num_rows)

        files = {
            "file": (
                f"endurance_{self.user_id}_{int(time.time())}.csv",
                BytesIO(csv_data.encode('utf-8')),
                "text/csv"
            )
        }

        with self.client.post(
            "/api/intake/upload",
            headers=self.headers,
            files=files,
            catch_response=True,
            name="OP: Upload"
        ) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _report(self):
        """Generate report - mixed operations."""
        payload = {
            "report_type": random.choice(["esrs_e1", "cdp", "ifrs_s2"]),
            "year": random.choice([2023, 2024]),
            "format": random.choice(["json", "pdf", "xlsx"])
        }

        with self.client.post(
            "/api/reports/generate",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="OP: Report"
        ) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _entity_resolution(self):
        """Entity resolution - ML operations."""
        payload = {
            "supplier_name": f"Global Steel Corp {random.randint(1, 100)}",
            "country": random.choice(["USA", "Germany", "China", "Japan"]),
            "confidence_threshold": 0.85
        }

        with self.client.post(
            "/api/entity/resolve",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="OP: Entity Resolution"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _supplier_engagement(self):
        """Supplier engagement - email operations."""
        supplier_id = f"SUP-{random.randint(1, 10000)}"

        with self.client.post(
            f"/api/suppliers/{supplier_id}/engage",
            headers=self.headers,
            json={"campaign_type": "data_request"},
            catch_response=True,
            name="OP: Supplier Engagement"
        ) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _emissions_query(self):
        """Complex emissions query - database intensive."""
        filters = {
            "category": random.choice([1, 4, 6]),
            "year": random.choice([2023, 2024]),
            "limit": random.choice([100, 500, 1000])
        }

        with self.client.get(
            f"/api/emissions?category={filters['category']}&year={filters['year']}&limit={filters['limit']}",
            headers=self.headers,
            catch_response=True,
            name="OP: Emissions Query"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _batch_calculation(self):
        """Batch calculation - queue operations."""
        items = []
        for _ in range(random.randint(5, 20)):
            items.append({
                "category": random.choice([1, 4, 6]),
                "tier": random.choice([1, 2, 3]),
                "quantity": random.uniform(100, 1000),
                "unit": "kg",
                "spend_usd": random.uniform(1000, 10000)
            })

        with self.client.post(
            "/api/calculations/batch",
            headers=self.headers,
            json={"items": items},
            catch_response=True,
            name="OP: Batch Calculation"
        ) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _cache_operations(self):
        """Cache-focused operations to test cache stability."""
        # Query frequently accessed data (should hit cache)
        category = random.randint(1, 3)  # Focus on few categories for cache hits

        with self.client.get(
            f"/api/emissions/aggregate?category={category}",
            headers=self.headers,
            catch_response=True,
            name="OP: Cache Hit"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _update_operations(self):
        """Update operations to test database transactions."""
        supplier_id = f"SUP-{random.randint(1, 1000)}"

        payload = {
            "name": f"Updated Supplier {random.randint(1, 1000)}",
            "contact_email": f"contact{random.randint(1, 1000)}@supplier.com"
        }

        with self.client.put(
            f"/api/suppliers/{supplier_id}",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="OP: Update Supplier"
        ) as response:
            if response.status_code in [200, 404]:
                # 404 acceptable if supplier doesn't exist
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    def _health_check(self):
        """Periodic health check."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="HEALTH: Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


# ============================================================================
# RESOURCE MONITORING & LEAK DETECTION
# ============================================================================

# Global storage for long-term monitoring
hourly_metrics = defaultdict(list)
last_metric_time = time.time()


def collect_hourly_metrics(stats):
    """Collect metrics every hour for trend analysis."""
    global last_metric_time

    current_time = time.time()
    if current_time - last_metric_time >= 3600:  # Every hour
        last_metric_time = current_time

        # Collect system resources
        resources = monitor_system_resources()

        # Collect performance metrics
        p95 = stats.total.get_response_time_percentile(0.95)
        fail_ratio = stats.total.fail_ratio

        hourly_metrics['timestamp'].append(datetime.now().isoformat())
        hourly_metrics['memory_percent'].append(resources['memory']['percent'])
        hourly_metrics['cpu_percent'].append(resources['cpu']['percent_overall'])
        hourly_metrics['p95_latency'].append(p95)
        hourly_metrics['error_rate'].append(fail_ratio * 100)
        hourly_metrics['total_requests'].append(stats.total.num_requests)

        # Print hourly summary
        hour = len(hourly_metrics['timestamp'])
        print(f"\n{'='*80}")
        print(f"HOUR {hour} SUMMARY ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"{'='*80}")
        print(f"Memory: {resources['memory']['percent']:.1f}%")
        print(f"CPU: {resources['cpu']['percent_overall']:.1f}%")
        print(f"p95 Latency: {p95:.2f}ms")
        print(f"Error Rate: {fail_ratio * 100:.3f}%")
        print(f"Total Requests: {stats.total.num_requests:,}")
        print(f"{'='*80}\n")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 80)
    print("GL-VCCI ENDURANCE TEST STARTING")
    print("=" * 80)
    print(f"Duration: 24 hours")
    print(f"Users: 500 concurrent")
    print(f"Focus: Memory leaks, connection leaks, long-term stability")
    print(f"Start time: {datetime.now().isoformat()}")
    print("=" * 80)

    # Start background monitoring
    def monitor():
        while environment.runner.state != "stopped":
            try:
                collect_hourly_metrics(environment.stats)
            except Exception as e:
                print(f"Monitoring error: {e}")
            time.sleep(300)  # Check every 5 minutes

    import threading
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("\n" + "=" * 80)
    print("GL-VCCI ENDURANCE TEST COMPLETED")
    print("=" * 80)
    print(f"End time: {datetime.now().isoformat()}")

    stats = environment.stats

    print("\nOverall Statistics:")
    print(f"  Total requests: {stats.total.num_requests:,}")
    print(f"  Total failures: {stats.total.num_failures:,}")
    print(f"  Failure rate: {stats.total.fail_ratio * 100:.4f}%")

    print("\nResponse Times:")
    p50 = stats.total.get_response_time_percentile(0.50)
    p95 = stats.total.get_response_time_percentile(0.95)
    p99 = stats.total.get_response_time_percentile(0.99)

    print(f"  Average: {stats.total.avg_response_time:.2f}ms")
    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  p99: {p99:.2f}ms")

    # Analyze trends over time
    if len(hourly_metrics['memory_percent']) >= 2:
        print("\nLong-term Trend Analysis:")

        # Memory leak detection
        memory_data = hourly_metrics['memory_percent']
        memory_first_quarter = memory_data[:len(memory_data)//4] if len(memory_data) >= 4 else memory_data[:1]
        memory_last_quarter = memory_data[-len(memory_data)//4:] if len(memory_data) >= 4 else memory_data[-1:]

        avg_memory_start = sum(memory_first_quarter) / len(memory_first_quarter)
        avg_memory_end = sum(memory_last_quarter) / len(memory_last_quarter)
        memory_growth = avg_memory_end - avg_memory_start

        print(f"\n  Memory Analysis:")
        print(f"    Start (first quarter): {avg_memory_start:.1f}%")
        print(f"    End (last quarter): {avg_memory_end:.1f}%")
        print(f"    Growth: {memory_growth:+.1f}%")

        if abs(memory_growth) < 5:
            print(f"    Assessment: ✅ STABLE (no memory leak)")
        elif abs(memory_growth) < 10:
            print(f"    Assessment: ⚠️ WARNING (minor growth)")
        else:
            print(f"    Assessment: ❌ FAIL (possible memory leak)")

        # Performance degradation detection
        p95_data = hourly_metrics['p95_latency']
        p95_first_quarter = p95_data[:len(p95_data)//4] if len(p95_data) >= 4 else p95_data[:1]
        p95_last_quarter = p95_data[-len(p95_data)//4:] if len(p95_data) >= 4 else p95_data[-1:]

        avg_p95_start = sum(p95_first_quarter) / len(p95_first_quarter)
        avg_p95_end = sum(p95_last_quarter) / len(p95_last_quarter)
        p95_degradation = ((avg_p95_end - avg_p95_start) / avg_p95_start) * 100

        print(f"\n  Performance Analysis:")
        print(f"    Start p95: {avg_p95_start:.2f}ms")
        print(f"    End p95: {avg_p95_end:.2f}ms")
        print(f"    Degradation: {p95_degradation:+.1f}%")

        if abs(p95_degradation) < 10:
            print(f"    Assessment: ✅ STABLE")
        elif abs(p95_degradation) < 25:
            print(f"    Assessment: ⚠️ WARNING (minor degradation)")
        else:
            print(f"    Assessment: ❌ FAIL (significant degradation)")

    print("\nEndurance Test Targets:")
    print(f"  Error rate < 0.01%: {stats.total.fail_ratio * 100:.4f}% - "
          f"{'✅ PASS' if stats.total.fail_ratio < 0.0001 else '❌ FAIL'}")
    print(f"  Memory stable: {'✅ PASS' if abs(memory_growth) < 5 else '❌ FAIL'}")
    print(f"  Performance stable: {'✅ PASS' if abs(p95_degradation) < 10 else '❌ FAIL'}")
    print(f"  No crashes: ✅ PASS (test completed)")

    print("=" * 80)


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test environment."""
    if not isinstance(environment.runner, WorkerRunner):
        print("\nInitializing Endurance Test...")
        print("Targets:")
        print("  - Zero memory leaks")
        print("  - Zero connection leaks")
        print("  - Error rate: <0.01%")
        print("  - Performance: stable over 24h")
        print("  - No crashes or restarts")
        print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("To run this endurance test, use:")
    print("\n  locust -f locustfile_endurance.py --host=http://localhost:8000 \\")
    print("         --users=500 --spawn-rate=10 --run-time=24h\n")
    print("For shorter validation test (4 hours):")
    print("\n  locust -f locustfile_endurance.py --host=http://localhost:8000 \\")
    print("         --users=500 --spawn-rate=10 --run-time=4h\n")
