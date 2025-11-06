"""
GL-VCCI Load Testing Suite - Main Orchestrator
===============================================

Main entry point for load testing the GL-VCCI Scope 3 Carbon Intelligence Platform.
This orchestrator coordinates all load test scenarios and provides a unified interface.

Performance Targets:
- Ingestion: 100K transactions/hour sustained
- Calculations: 10K/second sustained
- API p95 latency: <200ms on aggregates
- API p99 latency: <500ms
- Concurrent users: 1,000 users stable
- Error rate: <0.1%
- CPU usage: <80% under normal load
- Memory: No leaks over 24 hours
- Database connections: Pool utilization <80%

Usage:
    # Run with web UI
    locust -f locustfile.py --host=http://localhost:8000

    # Run headless (automated)
    locust -f locustfile.py --host=http://localhost:8000 --users=1000 --spawn-rate=50 --run-time=1h --headless

    # Run specific scenario
    locust -f locust/ingestion_tests.py --host=http://localhost:8000

    # Distributed mode (master)
    locust -f locustfile.py --master --host=http://localhost:8000

    # Distributed mode (worker)
    locust -f locustfile.py --worker --master-host=<master-ip>

Author: GL-VCCI Team
Phase: Phase 6 - Testing & Validation
Version: 2.0
"""

import os
import logging
from typing import Dict, Any
from locust import HttpUser, task, between, events
from locust.env import Environment

# Import all test scenarios
from locust.ingestion_tests import (
    IngestionSustainedUser,
    IngestionBurstUser,
    IngestionRampUpUser,
    IngestionWithFailuresUser,
    IngestionMultiTenantUser
)

from locust.api_tests import (
    APIConcurrentUser,
    APISustainedUser,
    APISpikeUser,
    APILatencyUser,
    APIReadWriteUser
)

from locust.calculation_tests import (
    CalculationSustainedUser,
    CalculationMonteCarloUser,
    CalculationBatchUser,
    CalculationRealtimeUser
)

from locust.database_tests import (
    DatabaseConnectionPoolUser,
    CacheHitRateUser,
    DatabaseQueryUser
)

from locust.endurance_tests import (
    EnduranceSoakUser,
    EnduranceMemoryLeakUser,
    EnduranceDegradationUser
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Main Orchestrator User - Mixed Workload
# ============================================================================

class VCCIPlatformUser(HttpUser):
    """
    Main orchestrator user that simulates realistic platform usage.

    This user performs a mix of operations representing typical user behavior:
    - Browse dashboards and reports (40%)
    - Query emissions data (30%)
    - Run calculations (15%)
    - Upload data (10%)
    - Generate reports (5%)
    """

    # Wait between 1-5 seconds between tasks (realistic think time)
    wait_time = between(1, 5)

    # Authentication token
    token = None

    def on_start(self):
        """
        Called when a user starts.
        Authenticate and get access token.
        """
        try:
            response = self.client.post("/api/auth/login", json={
                "email": f"loadtest_{self.environment.runner.user_count}@example.com",
                "password": "LoadTest123!"
            })

            if response.status_code == 200:
                self.token = response.json().get("access_token")
                logger.info(f"User authenticated successfully")
            else:
                logger.error(f"Authentication failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error during authentication: {e}")

    @task(40)
    def view_dashboard(self):
        """View main dashboard - 40% of traffic"""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        with self.client.get(
            "/api/dashboard",
            headers=headers,
            catch_response=True,
            name="Dashboard: View"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(30)
    def query_emissions(self):
        """Query emissions data - 30% of traffic"""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        with self.client.get(
            "/api/emissions",
            params={
                "category": "1",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31"
            },
            headers=headers,
            catch_response=True,
            name="Emissions: Query"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(30)
    def query_suppliers(self):
        """Query supplier data - 30% of traffic"""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        with self.client.get(
            "/api/suppliers",
            params={"limit": 100, "offset": 0},
            headers=headers,
            catch_response=True,
            name="Suppliers: Query"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(15)
    def run_calculation(self):
        """Run emission calculation - 15% of traffic"""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        calculation_data = {
            "category": 1,
            "supplier_name": "Test Supplier Ltd",
            "product_name": "Steel Rebar",
            "quantity": 1000,
            "unit": "kg",
            "spend_usd": 50000
        }

        with self.client.post(
            "/api/calculations",
            json=calculation_data,
            headers=headers,
            catch_response=True,
            name="Calculations: Create"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(10)
    def upload_data(self):
        """Upload procurement data - 10% of traffic"""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        # Generate small CSV data
        csv_data = """supplier_name,product_name,quantity,unit,spend_usd,date
Test Supplier Ltd,Steel Rebar,1000,kg,50000,2024-01-15
Another Supplier Inc,Aluminum Sheets,500,kg,25000,2024-01-16"""

        files = {"file": ("test_data.csv", csv_data, "text/csv")}

        with self.client.post(
            "/api/intake/upload",
            files=files,
            headers=headers,
            catch_response=True,
            name="Intake: Upload CSV"
        ) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(5)
    def generate_report(self):
        """Generate ESRS report - 5% of traffic"""
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        report_data = {
            "report_type": "esrs_e1",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "format": "pdf"
        }

        with self.client.post(
            "/api/reports/generate",
            json=report_data,
            headers=headers,
            catch_response=True,
            name="Reports: Generate"
        ) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")


# ============================================================================
# Event Handlers - Performance Monitoring
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Called when load test starts.
    Initialize monitoring and logging.
    """
    logger.info("="*80)
    logger.info("GL-VCCI Load Test Starting")
    logger.info("="*80)
    logger.info(f"Host: {environment.host}")
    logger.info(f"Users: {getattr(environment.runner, 'target_user_count', 'N/A')}")
    logger.info(f"Spawn Rate: {getattr(environment.runner, 'spawn_rate', 'N/A')}")
    logger.info("="*80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when load test stops.
    Print summary statistics.
    """
    stats = environment.stats

    logger.info("="*80)
    logger.info("GL-VCCI Load Test Completed")
    logger.info("="*80)
    logger.info(f"Total Requests: {stats.total.num_requests}")
    logger.info(f"Total Failures: {stats.total.num_failures}")
    logger.info(f"Error Rate: {stats.total.fail_ratio * 100:.2f}%")
    logger.info(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"p50: {stats.total.get_response_time_percentile(0.50):.2f}ms")
    logger.info(f"p95: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    logger.info(f"p99: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    logger.info(f"RPS: {stats.total.total_rps:.2f}")
    logger.info("="*80)

    # Validate performance targets
    p95 = stats.total.get_response_time_percentile(0.95)
    error_rate = stats.total.fail_ratio

    targets_met = True

    if p95 > 200:
        logger.warning(f"FAILED: p95 latency {p95:.2f}ms exceeds target 200ms")
        targets_met = False
    else:
        logger.info(f"PASSED: p95 latency {p95:.2f}ms within target 200ms")

    if error_rate > 0.001:  # 0.1%
        logger.warning(f"FAILED: Error rate {error_rate*100:.2f}% exceeds target 0.1%")
        targets_met = False
    else:
        logger.info(f"PASSED: Error rate {error_rate*100:.2f}% within target 0.1%")

    if targets_met:
        logger.info("="*80)
        logger.info("ALL PERFORMANCE TARGETS MET!")
        logger.info("="*80)
    else:
        logger.warning("="*80)
        logger.warning("SOME PERFORMANCE TARGETS NOT MET")
        logger.warning("="*80)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """
    Called on every request.
    Can be used for custom metrics collection.
    """
    # Log slow requests
    if response_time > 1000:  # > 1 second
        logger.warning(f"SLOW REQUEST: {name} took {response_time:.2f}ms")

    # Log errors
    if exception:
        logger.error(f"REQUEST ERROR: {name} - {exception}")


# ============================================================================
# Custom Load Shapes
# ============================================================================

from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """
    Step load shape that increases load in stages.

    Useful for identifying breaking points:
    - Stage 1: 100 users for 2 minutes
    - Stage 2: 300 users for 2 minutes
    - Stage 3: 500 users for 2 minutes
    - Stage 4: 1000 users for 2 minutes
    - Stage 5: 2000 users for 2 minutes
    """

    stages = [
        {"duration": 120, "users": 100, "spawn_rate": 10},
        {"duration": 240, "users": 300, "spawn_rate": 20},
        {"duration": 360, "users": 500, "spawn_rate": 20},
        {"duration": 480, "users": 1000, "spawn_rate": 50},
        {"duration": 600, "users": 2000, "spawn_rate": 100},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])

        return None  # Stop test


class WaveLoadShape(LoadTestShape):
    """
    Wave load shape that simulates periodic traffic patterns.

    Simulates daily traffic pattern:
    - Low traffic: 100 users
    - Peak traffic: 1000 users
    - Wave period: 10 minutes
    """

    def tick(self):
        run_time = self.get_run_time()

        # Calculate wave position (0 to 1)
        wave_period = 600  # 10 minutes
        wave_position = (run_time % wave_period) / wave_period

        # Calculate user count using sine wave
        import math
        min_users = 100
        max_users = 1000
        user_range = max_users - min_users

        current_users = int(min_users + (user_range * (math.sin(wave_position * 2 * math.pi) + 1) / 2))
        spawn_rate = 20

        return (current_users, spawn_rate)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    This section is not executed when running via 'locust' command.
    It's here for reference and documentation.
    """

    print("GL-VCCI Load Testing Suite")
    print("="*80)
    print()
    print("To run load tests, use one of the following commands:")
    print()
    print("1. Web UI mode (interactive):")
    print("   locust -f locustfile.py --host=http://localhost:8000")
    print()
    print("2. Headless mode (automated):")
    print("   locust -f locustfile.py --host=http://localhost:8000 \\")
    print("          --users=1000 --spawn-rate=50 --run-time=1h --headless")
    print()
    print("3. Specific scenario:")
    print("   locust -f locust/ingestion_tests.py --host=http://localhost:8000")
    print()
    print("4. Distributed mode (master):")
    print("   locust -f locustfile.py --master --host=http://localhost:8000")
    print()
    print("5. Distributed mode (worker):")
    print("   locust -f locustfile.py --worker --master-host=<master-ip>")
    print()
    print("="*80)
    print()
    print("For more information, see README.md")
    print()
