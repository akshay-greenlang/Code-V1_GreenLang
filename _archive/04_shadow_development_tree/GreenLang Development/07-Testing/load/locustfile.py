# -*- coding: utf-8 -*-
"""
Enterprise Load Testing Suite for GreenLang

This module provides comprehensive load testing scenarios using Locust
for API endpoints, pipeline execution, and registry operations.
"""

from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import json
import random
import time
import os
from typing import Dict, Any, Optional
from greenlang.determinism import deterministic_random


# Setup logging
setup_logging("INFO", None)


class GreenLangUser(HttpUser):
    """
    Base user class for GreenLang load testing.
    
    Simulates realistic user behavior with authentication,
    pack operations, and pipeline execution.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Initialize user session with authentication"""
        self.token = None
        self.tenant_id = None
        self.headers = {"Content-Type": "application/json"}
        
        # Authenticate user
        self.authenticate()
    
    def authenticate(self):
        """Authenticate and obtain JWT token"""
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": f"user_{deterministic_random().randint(1, 100)}",
                "password": "test_password",
                "tenant": "test_tenant"
            },
            catch_response=True
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("token")
            self.tenant_id = data.get("tenant_id")
            self.headers["Authorization"] = f"Bearer {self.token}"
            response.success()
        else:
            response.failure(f"Authentication failed: {response.status_code}")
    
    @task(10)
    def list_packs(self):
        """List available packs - most common operation"""
        with self.client.get(
            "/api/v1/packs",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to list packs: {response.status_code}")
    
    @task(5)
    def search_packs(self):
        """Search for packs with various filters"""
        queries = ["carbon", "energy", "emissions", "solar", "wind"]
        query = deterministic_random().choice(queries)
        
        with self.client.get(
            f"/api/v1/packs/search?q={query}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Pack search failed: {response.status_code}")
    
    @task(3)
    def get_pack_details(self):
        """Get detailed information about a specific pack"""
        pack_names = ["boiler-solar", "emissions-core", "carbon-tracker"]
        pack = deterministic_random().choice(pack_names)
        
        with self.client.get(
            f"/api/v1/packs/{pack}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                response.success()  # Pack not found is valid
            else:
                response.failure(f"Failed to get pack details: {response.status_code}")
    
    @task(2)
    def execute_pipeline(self):
        """Execute a pipeline - resource intensive operation"""
        pipeline_data = {
            "pipeline": {
                "name": f"test_pipeline_{deterministic_random().randint(1, 1000)}",
                "steps": [
                    {
                        "name": "validate",
                        "agent": "InputValidatorAgent",
                        "inputs": {"data": "test_data"}
                    },
                    {
                        "name": "calculate",
                        "agent": "CarbonAgent",
                        "inputs": {"fuel_type": "natural_gas", "amount": 100}
                    }
                ]
            },
            "inputs": {
                "building_data": {
                    "area": deterministic_random().randint(1000, 10000),
                    "type": deterministic_random().choice(["office", "retail", "industrial"])
                }
            }
        }
        
        with self.client.post(
            "/api/v1/pipelines/execute",
            json=pipeline_data,
            headers=self.headers,
            timeout=30,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201, 202]:
                response.success()
            else:
                response.failure(f"Pipeline execution failed: {response.status_code}")
    
    @task(1)
    def upload_pack(self):
        """Upload a new pack - write operation"""
        pack_data = {
            "name": f"test_pack_{deterministic_random().randint(1, 10000)}",
            "version": "1.0.0",
            "description": "Load test pack",
            "content": "test content data"
        }
        
        with self.client.post(
            "/api/v1/packs/upload",
            json=pack_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 409:
                response.success()  # Pack already exists
            else:
                response.failure(f"Pack upload failed: {response.status_code}")
    
    @task(8)
    def check_health(self):
        """Check system health - lightweight operation"""
        with self.client.get(
            "/health",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(4)
    def get_metrics(self):
        """Get system metrics"""
        with self.client.get(
            "/metrics",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics retrieval failed: {response.status_code}")


class AdminUser(HttpUser):
    """
    Admin user class for testing administrative operations.
    """
    
    wait_time = between(2, 5)
    
    def on_start(self):
        """Initialize admin session"""
        self.headers = {
            "Content-Type": "application/json",
            "X-Admin-Key": os.getenv("ADMIN_API_KEY", "admin_test_key")
        }
    
    @task(5)
    def list_tenants(self):
        """List all tenants"""
        with self.client.get(
            "/api/v1/admin/tenants",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to list tenants: {response.status_code}")
    
    @task(3)
    def get_audit_logs(self):
        """Retrieve audit logs"""
        with self.client.get(
            "/api/v1/admin/audit/logs?limit=100",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get audit logs: {response.status_code}")
    
    @task(2)
    def update_quota(self):
        """Update tenant quota"""
        tenant_id = f"tenant_{deterministic_random().randint(1, 10)}"
        quota_data = {
            "cpu_cores": deterministic_random().randint(2, 8),
            "memory_gb": deterministic_random().randint(4, 32),
            "storage_gb": deterministic_random().randint(10, 100)
        }
        
        with self.client.patch(
            f"/api/v1/admin/tenants/{tenant_id}/quota",
            json=quota_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Failed to update quota: {response.status_code}")
    
    @task(1)
    def create_tenant(self):
        """Create a new tenant"""
        tenant_data = {
            "name": f"load_test_tenant_{deterministic_random().randint(1, 10000)}",
            "email": f"test_{deterministic_random().randint(1, 10000)}@example.com",
            "subscription_tier": deterministic_random().choice(["FREE", "STARTER", "PROFESSIONAL"])
        }
        
        with self.client.post(
            "/api/v1/admin/tenants",
            json=tenant_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201, 409]:
                response.success()
            else:
                response.failure(f"Failed to create tenant: {response.status_code}")


class RegistryUser(HttpUser):
    """
    User class for testing registry operations.
    """
    
    wait_time = between(1, 2)
    host = os.getenv("REGISTRY_HOST", "https://registry.greenlang.io")
    
    def on_start(self):
        """Initialize registry session"""
        self.headers = {"Accept": "application/json"}
        self.authenticate_registry()
    
    def authenticate_registry(self):
        """Authenticate with registry"""
        # Registry authentication logic
        pass
    
    @task(10)
    def pull_manifest(self):
        """Pull pack manifest from registry"""
        pack = deterministic_random().choice(["boiler-solar", "emissions-core", "carbon-tracker"])
        version = deterministic_random().choice(["1.0.0", "1.1.0", "latest"])
        
        with self.client.get(
            f"/v2/greenlang/{pack}/manifests/{version}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                response.success()  # Not found is valid
            else:
                response.failure(f"Failed to pull manifest: {response.status_code}")
    
    @task(5)
    def list_tags(self):
        """List available tags for a pack"""
        pack = deterministic_random().choice(["boiler-solar", "emissions-core", "carbon-tracker"])
        
        with self.client.get(
            f"/v2/greenlang/{pack}/tags/list",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to list tags: {response.status_code}")
    
    @task(2)
    def pull_blob(self):
        """Pull pack blob from registry"""
        digest = f"sha256:{random.randbytes(32).hex()}"
        
        with self.client.get(
            f"/v2/greenlang/test-pack/blobs/{digest}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Failed to pull blob: {response.status_code}")


class StressTestUser(HttpUser):
    """
    User class for stress testing with aggressive patterns.
    """
    
    wait_time = between(0.1, 0.5)  # Minimal wait time
    
    def on_start(self):
        """Initialize stress test session"""
        self.headers = {"Content-Type": "application/json"}
    
    @task
    def rapid_fire_requests(self):
        """Send rapid requests to stress the system"""
        endpoints = [
            "/health",
            "/api/v1/packs",
            "/api/v1/pipelines",
            "/metrics"
        ]
        
        endpoint = deterministic_random().choice(endpoints)
        
        with self.client.get(
            endpoint,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code < 500:
                response.success()
            else:
                response.failure(f"Server error: {response.status_code}")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """Initialize custom metrics on test start"""
    print("Load test starting...")
    print(f"Target host: {environment.host}")
    print(f"Total users: {environment.parsed_options.num_users}")
    print(f"Spawn rate: {environment.parsed_options.spawn_rate}")


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Generate test report on completion"""
    print("\nLoad test completed!")
    print("\nTest Statistics:")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"RPS: {environment.stats.total.current_rps:.2f}")
    
    # Calculate percentiles
    if environment.stats.total.num_requests > 0:
        print(f"\nResponse Time Percentiles:")
        for percentile in [50, 90, 95, 99]:
            value = environment.stats.total.get_response_time_percentile(percentile)
            if value is not None:
                print(f"  P{percentile}: {value:.2f}ms")


# Configuration for different test scenarios
class LoadTestScenarios:
    """
    Predefined load test scenarios for different testing needs.
    """
    
    @staticmethod
    def baseline_load():
        """Baseline load test - normal usage pattern"""
        return {
            "users": 100,
            "spawn_rate": 10,
            "run_time": "5m",
            "user_classes": [GreenLangUser]
        }
    
    @staticmethod
    def stress_test():
        """Stress test - push system to limits"""
        return {
            "users": 1000,
            "spawn_rate": 50,
            "run_time": "10m",
            "user_classes": [GreenLangUser, StressTestUser]
        }
    
    @staticmethod
    def spike_test():
        """Spike test - sudden traffic increase"""
        return {
            "users": 500,
            "spawn_rate": 100,
            "run_time": "3m",
            "user_classes": [StressTestUser]
        }
    
    @staticmethod
    def endurance_test():
        """Endurance test - sustained load over time"""
        return {
            "users": 200,
            "spawn_rate": 5,
            "run_time": "1h",
            "user_classes": [GreenLangUser, AdminUser]
        }
    
    @staticmethod
    def registry_test():
        """Registry-specific load test"""
        return {
            "users": 150,
            "spawn_rate": 15,
            "run_time": "10m",
            "user_classes": [RegistryUser]
        }


if __name__ == "__main__":
    # Example: Run baseline load test
    print("GreenLang Enterprise Load Testing Suite")
    print("========================================")
    print("Available scenarios:")
    print("1. Baseline Load Test")
    print("2. Stress Test")
    print("3. Spike Test")
    print("4. Endurance Test")
    print("5. Registry Test")
    print("\nConfigure and run with:")
    print("locust -f locustfile.py --host=http://localhost:8000")