# -*- coding: utf-8 -*-
"""
GreenLang Load Testing Scenarios
================================

Load testing scenarios using Locust for all major GreenLang services.

Usage:
    # Run sustained load test
    locust -f locustfile.py --users 100 --spawn-rate 10 --run-time 1h

    # Run spike test
    locust -f locustfile.py --users 1000 --spawn-rate 100 --run-time 5m

    # Run capacity test
    locust -f locustfile.py --users 2000 --spawn-rate 50

    # Web UI
    locust -f locustfile.py --host=http://localhost:8000

Author: Performance Engineering Team
Date: 2025-11-09
"""

import random
import time
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask
from greenlang.determinism import deterministic_random


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

def generate_shipment():
    """Generate realistic CBAM shipment data."""
    return {
        "shipment_id": f"SHIP-{deterministic_random().randint(100000, 999999)}",
        "origin_country": deterministic_random().choice(["CN", "IN", "US", "JP", "KR"]),
        "destination_country": deterministic_random().choice(["DE", "FR", "IT", "ES", "NL"]),
        "goods_category": deterministic_random().choice(["Steel", "Aluminum", "Cement", "Fertilizer"]),
        "cn_code": deterministic_random().choice(["7208", "7601", "2523", "3102"]),
        "quantity": deterministic_random().randint(100, 10000),
        "unit": "kg",
        "transport_mode": deterministic_random().choice(["Sea", "Air", "Rail", "Road"]),
        "invoice_value": deterministic_random().randint(1000, 100000),
        "currency": "EUR"
    }


def generate_company():
    """Generate CSRD company data."""
    return {
        "company_id": f"COMP-{deterministic_random().randint(10000, 99999)}",
        "name": f"Company {deterministic_random().randint(1, 1000)}",
        "sector": deterministic_random().choice(["Manufacturing", "Energy", "Transport", "Agriculture"]),
        "employees": deterministic_random().randint(100, 10000),
        "revenue": deterministic_random().randint(1000000, 1000000000),
        "countries": deterministic_random().sample(["DE", "FR", "IT", "ES", "NL", "PL"], k=deterministic_random().randint(1, 3)),
        "reporting_period": "2024"
    }


def generate_supplier():
    """Generate VCCI supplier data."""
    return {
        "supplier_id": f"SUPP-{deterministic_random().randint(100000, 999999)}",
        "name": f"Supplier {deterministic_random().randint(1, 10000)}",
        "country": deterministic_random().choice(["CN", "US", "DE", "IN", "JP", "KR", "TW"]),
        "sector": deterministic_random().choice(["Manufacturing", "Services", "Energy", "Agriculture"]),
        "spend": deterministic_random().randint(10000, 1000000),
        "currency": "USD"
    }


# ============================================================================
# SCENARIO 1: SUSTAINED LOAD (Normal Operations)
# ============================================================================

class CBOMSustainedLoad(HttpUser):
    """
    Scenario 1: Sustained Load
    - 100 users
    - 1 hour duration
    - Measure: Throughput, latency, error rate
    """

    wait_time = between(1, 3)  # 1-3 seconds between requests

    @task(3)  # Weight: 3 (most common operation)
    def process_single_shipment(self):
        """Process a single shipment."""
        shipment = generate_shipment()

        with self.client.post(
            "/api/cbam/shipments",
            json=shipment,
            name="Process Single Shipment",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # Validate response time
                if response.elapsed.total_seconds() > 1.0:
                    response.failure("Response time > 1 second")
                else:
                    response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)  # Weight: 2
    def get_emission_factor(self):
        """Retrieve emission factor."""
        cn_code = deterministic_random().choice(["7208", "7601", "2523", "3102"])
        country = deterministic_random().choice(["CN", "IN", "US", "DE"])

        with self.client.get(
            f"/api/factors/emission/{cn_code}",
            params={"country": country},
            name="Get Emission Factor",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                if response.elapsed.total_seconds() > 0.1:
                    response.failure("Response time > 100ms")
                else:
                    response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)  # Weight: 1 (less common)
    def process_batch_shipments(self):
        """Process batch of shipments."""
        batch = [generate_shipment() for _ in range(100)]

        with self.client.post(
            "/api/cbam/shipments/batch",
            json={"shipments": batch},
            name="Process Batch Shipments",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                # Batch should be fast (> 1000 records/sec)
                if response.elapsed.total_seconds() > 0.1:  # 100ms for 100 records
                    response.failure("Batch processing too slow")
                else:
                    response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# ============================================================================
# SCENARIO 2: SPIKE LOAD (Traffic Spike)
# ============================================================================

class SpikeLoadTest(HttpUser):
    """
    Scenario 2: Spike Load
    - 0 → 1000 users in 1 minute
    - 5 minute duration
    - Measure: Recovery time, failures
    """

    wait_time = between(0.5, 2)

    @task
    def spike_request(self):
        """Send requests during spike."""
        shipment = generate_shipment()

        with self.client.post(
            "/api/cbam/shipments",
            json=shipment,
            name="Spike Request",
            catch_response=True
        ) as response:
            if response.status_code in [200, 429]:  # 429 = rate limited (acceptable)
                response.success()
            elif response.status_code == 503:  # Service unavailable
                response.failure("Service overloaded")
            else:
                response.failure(f"Unexpected status: {response.status_code}")


# ============================================================================
# SCENARIO 3: CAPACITY TEST (Breaking Point)
# ============================================================================

class CapacityTest(HttpUser):
    """
    Scenario 3: Capacity Test
    - Increase users until failure
    - Measure: Breaking point, bottlenecks
    """

    wait_time = between(0.1, 0.5)

    @task
    def capacity_request(self):
        """Send requests to find capacity limit."""
        shipment = generate_shipment()

        with self.client.post(
            "/api/cbam/shipments",
            json=shipment,
            name="Capacity Request",
            catch_response=True
        ) as response:
            # Track when system starts failing
            if response.status_code != 200:
                response.failure(f"System at capacity: {response.status_code}")
            else:
                response.success()


# ============================================================================
# SCENARIO 4: ENDURANCE TEST (Memory Leaks)
# ============================================================================

class EnduranceTest(HttpUser):
    """
    Scenario 4: Endurance Test
    - 50 users
    - 24 hours duration
    - Measure: Memory leaks, degradation
    """

    wait_time = between(2, 5)

    @task
    def endurance_request(self):
        """Send requests over long period."""
        shipment = generate_shipment()

        # Track response time degradation
        start = time.time()

        with self.client.post(
            "/api/cbam/shipments",
            json=shipment,
            name="Endurance Request",
            catch_response=True
        ) as response:
            duration = time.time() - start

            if response.status_code == 200:
                # Check for performance degradation
                if duration > 2.0:  # Degraded
                    response.failure(f"Performance degraded: {duration:.2f}s")
                else:
                    response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# ============================================================================
# MIXED WORKLOAD (Realistic Usage)
# ============================================================================

class MixedWorkload(HttpUser):
    """
    Mixed workload simulating realistic usage patterns.
    """

    wait_time = between(1, 3)

    @task(5)  # CBAM shipment processing (50%)
    def cbam_shipment(self):
        """CBAM shipment processing."""
        shipment = generate_shipment()
        self.client.post("/api/cbam/shipments", json=shipment)

    @task(3)  # CSRD assessment (30%)
    def csrd_assessment(self):
        """CSRD materiality assessment."""
        company = generate_company()
        self.client.post("/api/csrd/materiality", json=company)

    @task(2)  # VCCI calculation (20%)
    def vcci_calculation(self):
        """VCCI Scope 3 calculation."""
        suppliers = [generate_supplier() for _ in range(10)]
        self.client.post("/api/vcci/scope3", json={"suppliers": suppliers})


# ============================================================================
# CUSTOM EVENT LISTENERS
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment."""
    print("=" * 80)
    print("GREENLANG LOAD TEST STARTING")
    print("=" * 80)
    print(f"Target: {environment.host}")
    print(f"Scenario: {environment.user_classes}")
    print("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test report."""
    print("\n" + "=" * 80)
    print("LOAD TEST COMPLETE")
    print("=" * 80)

    stats = environment.stats

    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Error Rate: {stats.total.fail_ratio:.2%}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"P95 Response Time: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"P99 Response Time: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Requests/sec: {stats.total.current_rps:.2f}")

    print("\n" + "=" * 80)
    print("TOP 10 SLOWEST ENDPOINTS")
    print("=" * 80)

    # Sort by response time
    sorted_stats = sorted(
        stats.entries.values(),
        key=lambda x: x.avg_response_time,
        reverse=True
    )

    for stat in sorted_stats[:10]:
        print(f"{stat.name:<40} {stat.avg_response_time:>10.2f}ms (P95: {stat.get_response_time_percentile(0.95):.2f}ms)")

    # Check SLO compliance
    print("\n" + "=" * 80)
    print("SLO COMPLIANCE")
    print("=" * 80)

    slo_violations = []

    if stats.total.fail_ratio > 0.01:  # Error rate > 1%
        slo_violations.append(f"Error rate: {stats.total.fail_ratio:.2%} (SLO: < 1%)")

    if stats.total.get_response_time_percentile(0.95) > 1000:  # P95 > 1s
        slo_violations.append(f"P95 latency: {stats.total.get_response_time_percentile(0.95):.2f}ms (SLO: < 1000ms)")

    if stats.total.current_rps < 50:  # Throughput < 50 req/sec
        slo_violations.append(f"Throughput: {stats.total.current_rps:.2f}/sec (SLO: > 50/sec)")

    if slo_violations:
        print("⚠️  SLO VIOLATIONS:")
        for violation in slo_violations:
            print(f"  - {violation}")
    else:
        print("✓ All SLOs met")

    print("\n" + "=" * 80)


# ============================================================================
# CUSTOM SHAPES (LOAD PATTERNS)
# ============================================================================

from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    Step load pattern: gradually increase load in steps.

    Steps:
    - 0-300s: 50 users
    - 300-600s: 100 users
    - 600-900s: 200 users
    - 900-1200s: 400 users
    """

    step_time = 300  # 5 minutes per step
    step_load = 50
    spawn_rate = 10
    time_limit = 1200  # 20 minutes total

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        current_step = run_time // self.step_time
        return (int((current_step + 1) * self.step_load), self.spawn_rate)


class SpikeShape(LoadTestShape):
    """
    Spike load pattern: sudden spike then back to normal.

    Pattern:
    - 0-60s: 50 users (baseline)
    - 60-180s: 1000 users (spike)
    - 180-300s: 50 users (recovery)
    """

    def tick(self):
        run_time = self.get_run_time()

        if run_time < 60:
            return (50, 10)
        elif run_time < 180:
            return (1000, 100)
        elif run_time < 300:
            return (50, 10)
        else:
            return None


# ============================================================================
# ACCEPTANCE CRITERIA
# ============================================================================

"""
ACCEPTANCE CRITERIA FOR LOAD TESTS

Sustained Load (Scenario 1):
- Error rate < 1%
- P95 latency < 1 second
- Throughput > 100 requests/sec
- CPU utilization < 80%
- Memory growth < 10% per hour

Spike Load (Scenario 2):
- Recovery time < 5 minutes
- Error rate during spike < 5%
- No service crashes
- Circuit breakers activate properly

Capacity Test (Scenario 3):
- Identify breaking point (users/sec)
- System degrades gracefully
- No data corruption
- Clear bottleneck identification

Endurance Test (Scenario 4):
- Memory growth < 5% over 24 hours
- No memory leaks detected
- Performance stable (< 10% degradation)
- No connection pool exhaustion

Mixed Workload:
- All endpoints within SLO
- Resource utilization balanced
- Cache hit rate > 50%
- Database connection pool < 80%
"""
