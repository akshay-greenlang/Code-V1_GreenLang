# -*- coding: utf-8 -*-
"""
Load Testing Suite - GL-VCCI Scope 3 Platform
Performance Optimization Team

Comprehensive load testing using Locust framework.

Test Scenarios:
1. Single calculation API (1000 req/s target)
2. Batch upload (10 concurrent batches of 10K records)
3. Report generation (100 concurrent requests)
4. Mixed workload (realistic traffic pattern)

Performance Targets:
- P50 latency: <100ms
- P95 latency: <500ms
- P99 latency: <1000ms
- Throughput: 5000 req/s
- Error rate: <0.1%

Installation:
    pip install locust

Run Tests:
    locust -f tests/performance/load_test.py --host=http://localhost:8000

Version: 1.0.0
Team: Performance Optimization Team
Date: 2025-11-09
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_random


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

class TestDataGenerator:
    """Generate realistic test data for load testing"""

    @staticmethod
    def generate_emission_record() -> Dict[str, Any]:
        """Generate single emission record"""
        return {
            "supplier_id": f"SUP-{deterministic_random().randint(1, 1000):04d}",
            "supplier_name": f"Supplier {deterministic_random().randint(1, 1000)}",
            "scope3_category": deterministic_random().choice([
                "CATEGORY_1",
                "CATEGORY_2",
                "CATEGORY_3",
                "CATEGORY_4",
                "CATEGORY_5",
                "CATEGORY_6"
            ]),
            "activity_name": "Electricity consumption",
            "quantity": round(random.uniform(100, 10000), 2),
            "unit": "kWh",
            "transaction_date": (
                DeterministicClock.now() - timedelta(days=deterministic_random().randint(1, 365))
            ).isoformat(),
            "region": deterministic_random().choice(["US", "EU", "APAC", "LATAM"]),
            "data_quality": deterministic_random().choice(["primary", "secondary", "proxy"])
        }

    @staticmethod
    def generate_batch_records(count: int = 1000) -> List[Dict[str, Any]]:
        """Generate batch of emission records"""
        return [
            TestDataGenerator.generate_emission_record()
            for _ in range(count)
        ]

    @staticmethod
    def generate_supplier_record() -> Dict[str, Any]:
        """Generate supplier record for intake"""
        return {
            "supplier_name": f"Test Supplier {deterministic_random().randint(1, 10000)}",
            "duns_number": f"{deterministic_random().randint(100000000, 999999999)}",
            "country": deterministic_random().choice(["USA", "CHN", "DEU", "GBR", "JPN"]),
            "annual_spend_usd": round(random.uniform(10000, 10000000), 2),
            "industry_code": f"NAICS-{deterministic_random().randint(11, 99)}"
        }


# ============================================================================
# LOAD TEST SCENARIOS
# ============================================================================

class EmissionsCalculationUser(FastHttpUser):
    """
    Load test for emissions calculation API.

    Simulates users submitting individual calculations.
    """
    wait_time = between(1, 3)  # 1-3 seconds between requests
    host = "http://localhost:8000"

    @task(3)
    def calculate_single_emission(self):
        """Single emission calculation"""
        payload = TestDataGenerator.generate_emission_record()

        with self.client.post(
            "/api/v1/calculator/calculate",
            json=payload,
            catch_response=True,
            name="/api/v1/calculator/calculate [single]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(1)
    def get_emission_factors(self):
        """Retrieve emission factors (cached)"""
        category = deterministic_random().choice([
            "electricity",
            "natural_gas",
            "diesel",
            "gasoline"
        ])

        with self.client.get(
            f"/api/v1/factors/{category}",
            catch_response=True,
            name="/api/v1/factors/:category"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")


class BatchProcessingUser(FastHttpUser):
    """
    Load test for batch processing.

    Simulates concurrent batch uploads.
    """
    wait_time = between(5, 10)  # 5-10 seconds between batches
    host = "http://localhost:8000"

    @task
    def upload_batch(self):
        """Upload batch of emissions"""
        # Generate batch (1000 records)
        batch = TestDataGenerator.generate_batch_records(1000)

        with self.client.post(
            "/api/v1/calculator/batch",
            json={"records": batch},
            catch_response=True,
            name="/api/v1/calculator/batch [1000 records]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")


class SupplierIntakeUser(FastHttpUser):
    """
    Load test for supplier data intake.

    Simulates supplier data ingestion.
    """
    wait_time = between(2, 5)
    host = "http://localhost:8000"

    @task(2)
    def ingest_supplier(self):
        """Ingest single supplier"""
        payload = TestDataGenerator.generate_supplier_record()

        with self.client.post(
            "/api/v1/intake/suppliers",
            json=payload,
            catch_response=True,
            name="/api/v1/intake/suppliers [single]"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(1)
    def get_suppliers(self):
        """List suppliers with pagination"""
        cursor = None  # Start from beginning

        with self.client.get(
            "/api/v1/intake/suppliers",
            params={"limit": 100, "cursor": cursor},
            catch_response=True,
            name="/api/v1/intake/suppliers [list]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")


class ReportingUser(FastHttpUser):
    """
    Load test for reporting endpoints.

    Simulates report generation requests.
    """
    wait_time = between(10, 20)  # Reports are less frequent
    host = "http://localhost:8000"

    @task
    def generate_emissions_report(self):
        """Generate emissions report"""
        start_date = (DeterministicClock.now() - timedelta(days=365)).isoformat()
        end_date = DeterministicClock.now().isoformat()

        with self.client.post(
            "/api/v1/reporting/emissions",
            json={
                "start_date": start_date,
                "end_date": end_date,
                "scope3_categories": ["CATEGORY_1", "CATEGORY_2"],
                "format": "json"
            },
            catch_response=True,
            name="/api/v1/reporting/emissions"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")


class MixedWorkloadUser(FastHttpUser):
    """
    Mixed workload simulating realistic usage.

    Combination of:
    - 60% calculations
    - 20% data retrieval
    - 10% batch processing
    - 10% reporting
    """
    wait_time = between(1, 5)
    host = "http://localhost:8000"

    @task(60)
    def calculate_emission(self):
        """Emission calculation (60% of traffic)"""
        payload = TestDataGenerator.generate_emission_record()

        self.client.post(
            "/api/v1/calculator/calculate",
            json=payload,
            name="/api/v1/calculator/calculate"
        )

    @task(20)
    def get_data(self):
        """Data retrieval (20% of traffic)"""
        endpoint = deterministic_random().choice([
            "/api/v1/intake/suppliers?limit=100",
            "/api/v1/factors/electricity",
            "/api/v1/calculator/history?limit=50"
        ])

        self.client.get(endpoint, name="[GET] Data Retrieval")

    @task(10)
    def batch_processing(self):
        """Batch processing (10% of traffic)"""
        batch = TestDataGenerator.generate_batch_records(100)

        self.client.post(
            "/api/v1/calculator/batch",
            json={"records": batch},
            name="/api/v1/calculator/batch [100 records]"
        )

    @task(10)
    def generate_report(self):
        """Report generation (10% of traffic)"""
        start_date = (DeterministicClock.now() - timedelta(days=30)).isoformat()
        end_date = DeterministicClock.now().isoformat()

        self.client.post(
            "/api/v1/reporting/emissions",
            json={
                "start_date": start_date,
                "end_date": end_date,
                "format": "json"
            },
            name="/api/v1/reporting/emissions"
        )


# ============================================================================
# CUSTOM EVENT HANDLERS
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    print("\n" + "=" * 80)
    print("GL-VCCI SCOPE 3 PLATFORM - LOAD TEST")
    print("=" * 80)
    print(f"Target host: {environment.host}")
    print(f"Start time: {DeterministicClock.now().isoformat()}")
    print("=" * 80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    print("\n" + "=" * 80)
    print("LOAD TEST COMPLETED")
    print("=" * 80)
    print(f"End time: {DeterministicClock.now().isoformat()}")
    print("\nFinal Statistics:")
    print(f"  Total requests: {environment.stats.total.num_requests}")
    print(f"  Total failures: {environment.stats.total.num_failures}")
    print(f"  Failure rate: {environment.stats.total.fail_ratio:.2%}")
    print(f"  Avg response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"  P50 response time: {environment.stats.total.get_response_time_percentile(0.5):.2f}ms")
    print(f"  P95 response time: {environment.stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"  P99 response time: {environment.stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"  RPS: {environment.stats.total.total_rps:.2f}")
    print("=" * 80 + "\n")


# ============================================================================
# LOAD TEST EXECUTION GUIDE
# ============================================================================

LOAD_TEST_GUIDE = """
# ============================================================================
# LOAD TEST EXECUTION GUIDE
# ============================================================================

# 1. Basic Load Test (Web UI)
# ----------------------------------------------------------------------------
locust -f tests/performance/load_test.py --host=http://localhost:8000

# Then open http://localhost:8089 in browser
# Configure:
#   - Number of users: 100
#   - Spawn rate: 10 users/second
#   - Click "Start swarming"


# 2. Headless Load Test (CLI)
# ----------------------------------------------------------------------------
# Run with specific parameters
locust -f tests/performance/load_test.py \\
    --host=http://localhost:8000 \\
    --users=100 \\
    --spawn-rate=10 \\
    --run-time=5m \\
    --headless

# Generate HTML report
locust -f tests/performance/load_test.py \\
    --host=http://localhost:8000 \\
    --users=100 \\
    --spawn-rate=10 \\
    --run-time=5m \\
    --headless \\
    --html=reports/load_test_report.html


# 3. Specific User Class
# ----------------------------------------------------------------------------
# Test only emissions calculation
locust -f tests/performance/load_test.py \\
    EmissionsCalculationUser \\
    --host=http://localhost:8000 \\
    --users=50 \\
    --spawn-rate=10 \\
    --run-time=2m \\
    --headless

# Test only batch processing
locust -f tests/performance/load_test.py \\
    BatchProcessingUser \\
    --host=http://localhost:8000 \\
    --users=10 \\
    --spawn-rate=2 \\
    --run-time=5m \\
    --headless


# 4. Stress Test (Find Breaking Point)
# ----------------------------------------------------------------------------
locust -f tests/performance/load_test.py \\
    --host=http://localhost:8000 \\
    --users=1000 \\
    --spawn-rate=100 \\
    --run-time=10m \\
    --headless


# 5. Distributed Load Test (Multiple Workers)
# ----------------------------------------------------------------------------
# Start master
locust -f tests/performance/load_test.py \\
    --master \\
    --expect-workers=4

# Start workers (on same or different machines)
locust -f tests/performance/load_test.py --worker --master-host=localhost
locust -f tests/performance/load_test.py --worker --master-host=localhost
locust -f tests/performance/load_test.py --worker --master-host=localhost
locust -f tests/performance/load_test.py --worker --master-host=localhost


# 6. Performance Target Validation
# ----------------------------------------------------------------------------
# Target: 5000 req/s, P95 < 500ms
locust -f tests/performance/load_test.py \\
    MixedWorkloadUser \\
    --host=http://localhost:8000 \\
    --users=500 \\
    --spawn-rate=50 \\
    --run-time=10m \\
    --headless \\
    --html=reports/performance_validation.html

# Success criteria:
# - Total RPS >= 5000
# - P95 latency < 500ms
# - P99 latency < 1000ms
# - Error rate < 0.1%


# 7. Monitoring During Load Test
# ----------------------------------------------------------------------------
# In separate terminal, monitor metrics:
watch -n 1 'curl -s http://localhost:8000/metrics | grep greenlang'

# Or use Prometheus + Grafana for real-time dashboards


# ============================================================================
# INTERPRETING RESULTS
# ============================================================================

# Good Performance Indicators:
# - P50 latency < 100ms
# - P95 latency < 500ms
# - P99 latency < 1000ms
# - Error rate < 0.1%
# - Throughput >= 5000 req/s
# - Cache hit rate > 85%
# - Database connection pool utilization 60-80%

# Warning Signs:
# - P95 latency > 1000ms (investigate slow endpoints)
# - Error rate > 1% (check logs for errors)
# - Throughput declining over time (memory leak?)
# - Cache hit rate < 50% (cache configuration issue)
# - Pool utilization > 90% (increase pool size)

# Critical Issues:
# - P99 latency > 5000ms (severe performance problem)
# - Error rate > 10% (system unstable)
# - Throughput drops to zero (system crash)
# - OOM errors (memory leak or insufficient resources)
"""

if __name__ == "__main__":
    print(LOAD_TEST_GUIDE)
