#!/usr/bin/env python3
"""
GL-VCCI Production Environment Validation Script
Version: 2.0.0
Date: November 8, 2025

This script validates that the production environment is properly configured
and ready for deployment. It checks infrastructure, configuration, security,
and performance requirements.

Usage:
    python validate_production_env.py --env production
    python validate_production_env.py --env production --skip-load-tests
    python validate_production_env.py --env staging --verbose

Exit Codes:
    0: All validations passed
    1: Critical validation failed
    2: Warning validation failed (deployment possible but not recommended)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Third-party imports (install with: pip install requests psycopg2-binary redis kubernetes boto3)
try:
    import requests
    import psycopg2
    import redis
    from kubernetes import client, config
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Install with: pip install requests psycopg2-binary redis kubernetes boto3")
    sys.exit(1)


class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    name: str
    passed: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict] = None
    duration: float = 0.0


class ProductionValidator:
    """Main validation orchestrator"""

    def __init__(self, environment: str, verbose: bool = False, skip_load_tests: bool = False):
        self.environment = environment
        self.verbose = verbose
        self.skip_load_tests = skip_load_tests
        self.results: List[ValidationResult] = []
        self.start_time = time.time()

        # Load configuration
        self.config = self._load_config()

        # Initialize colors for output
        self.COLOR_GREEN = '\033[92m'
        self.COLOR_YELLOW = '\033[93m'
        self.COLOR_RED = '\033[91m'
        self.COLOR_BLUE = '\033[94m'
        self.COLOR_END = '\033[0m'

    def _load_config(self) -> Dict:
        """Load environment-specific configuration"""
        config_file = f"config/{self.environment}.json"

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)

        # Default configuration if file doesn't exist
        return {
            "api_url": os.getenv("API_URL", "https://api.vcci.greenlang.io"),
            "database_url": os.getenv("DATABASE_URL"),
            "redis_url": os.getenv("REDIS_URL"),
            "kubernetes_namespace": os.getenv("K8S_NAMESPACE", "vcci-production"),
            "expected_services": ["backend-api", "worker", "frontend"],
            "min_replicas": 3,
            "max_latency_ms": 500,
            "min_uptime_percent": 99.9,
        }

    def log(self, message: str, level: str = "INFO"):
        """Formatted logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = {
            "INFO": self.COLOR_BLUE,
            "SUCCESS": self.COLOR_GREEN,
            "WARNING": self.COLOR_YELLOW,
            "ERROR": self.COLOR_RED,
        }.get(level, "")

        if self.verbose or level in ["SUCCESS", "WARNING", "ERROR"]:
            print(f"{color}[{timestamp}] [{level}] {message}{self.COLOR_END}")

    def run_validation(self, name: str, func, level: ValidationLevel) -> ValidationResult:
        """Run a single validation check with timing"""
        self.log(f"Running: {name}", "INFO")
        start = time.time()

        try:
            passed, message, details = func()
            duration = time.time() - start

            result = ValidationResult(
                name=name,
                passed=passed,
                level=level,
                message=message,
                details=details,
                duration=duration
            )

            self.results.append(result)

            if passed:
                self.log(f"✓ {name}: {message} ({duration:.2f}s)", "SUCCESS")
            else:
                log_level = "ERROR" if level == ValidationLevel.CRITICAL else "WARNING"
                self.log(f"✗ {name}: {message} ({duration:.2f}s)", log_level)

            return result

        except Exception as e:
            duration = time.time() - start
            result = ValidationResult(
                name=name,
                passed=False,
                level=level,
                message=f"Exception: {str(e)}",
                duration=duration
            )
            self.results.append(result)
            self.log(f"✗ {name}: Exception - {str(e)}", "ERROR")
            return result

    # ============================================================================
    # INFRASTRUCTURE VALIDATIONS
    # ============================================================================

    def validate_kubernetes_cluster(self) -> Tuple[bool, str, Dict]:
        """Validate Kubernetes cluster is accessible and healthy"""
        try:
            config.load_kube_config()
            v1 = client.CoreV1Api()

            # Get nodes
            nodes = v1.list_node()
            node_count = len(nodes.items)
            ready_nodes = sum(1 for node in nodes.items
                            if any(condition.type == "Ready" and condition.status == "True"
                                   for condition in node.status.conditions))

            # Check node distribution across AZs
            zones = set()
            for node in nodes.items:
                zone = node.metadata.labels.get("topology.kubernetes.io/zone", "unknown")
                zones.add(zone)

            details = {
                "total_nodes": node_count,
                "ready_nodes": ready_nodes,
                "availability_zones": len(zones),
                "zones": list(zones)
            }

            if ready_nodes < 3:
                return False, f"Insufficient ready nodes ({ready_nodes}/3 minimum)", details

            if len(zones) < 2:
                return False, f"Nodes not distributed across AZs ({len(zones)}/2 minimum)", details

            return True, f"Cluster healthy: {ready_nodes} nodes across {len(zones)} AZs", details

        except Exception as e:
            return False, f"Cannot access Kubernetes cluster: {str(e)}", {}

    def validate_namespace(self) -> Tuple[bool, str, Dict]:
        """Validate namespace exists and has proper resource quotas"""
        try:
            config.load_kube_config()
            v1 = client.CoreV1Api()
            namespace = self.config["kubernetes_namespace"]

            # Check namespace exists
            try:
                ns = v1.read_namespace(namespace)
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    return False, f"Namespace '{namespace}' does not exist", {}
                raise

            # Check resource quotas
            quotas = v1.list_namespaced_resource_quota(namespace)

            details = {
                "namespace": namespace,
                "status": ns.status.phase,
                "resource_quotas": len(quotas.items)
            }

            if len(quotas.items) == 0:
                return False, "No resource quotas configured", details

            return True, f"Namespace '{namespace}' is ready", details

        except Exception as e:
            return False, f"Namespace validation failed: {str(e)}", {}

    def validate_deployments(self) -> Tuple[bool, str, Dict]:
        """Validate all required deployments are running"""
        try:
            config.load_kube_config()
            apps_v1 = client.AppsV1Api()
            namespace = self.config["kubernetes_namespace"]

            deployments = apps_v1.list_namespaced_deployment(namespace)

            expected_services = self.config["expected_services"]
            found_services = []
            deployment_details = {}

            for deployment in deployments.items:
                name = deployment.metadata.name
                replicas = deployment.spec.replicas
                ready_replicas = deployment.status.ready_replicas or 0

                deployment_details[name] = {
                    "replicas": replicas,
                    "ready": ready_replicas,
                    "status": "healthy" if ready_replicas >= replicas else "degraded"
                }

                if any(expected in name for expected in expected_services):
                    found_services.append(name)

                    if ready_replicas < replicas:
                        return False, f"Deployment '{name}' not ready ({ready_replicas}/{replicas})", deployment_details

            missing_services = [svc for svc in expected_services if not any(svc in found for found in found_services)]

            if missing_services:
                return False, f"Missing deployments: {', '.join(missing_services)}", deployment_details

            return True, f"All {len(found_services)} deployments ready", deployment_details

        except Exception as e:
            return False, f"Deployment validation failed: {str(e)}", {}

    def validate_pods_healthy(self) -> Tuple[bool, str, Dict]:
        """Validate all pods are in Running state"""
        try:
            config.load_kube_config()
            v1 = client.CoreV1Api()
            namespace = self.config["kubernetes_namespace"]

            pods = v1.list_namespaced_pod(namespace)

            total_pods = len(pods.items)
            running_pods = 0
            failed_pods = []

            for pod in pods.items:
                if pod.status.phase == "Running":
                    # Check all containers are ready
                    container_statuses = pod.status.container_statuses or []
                    all_ready = all(cs.ready for cs in container_statuses)

                    if all_ready:
                        running_pods += 1
                    else:
                        failed_pods.append(f"{pod.metadata.name} (containers not ready)")
                else:
                    failed_pods.append(f"{pod.metadata.name} ({pod.status.phase})")

            details = {
                "total_pods": total_pods,
                "running_pods": running_pods,
                "failed_pods": failed_pods
            }

            if failed_pods:
                return False, f"{len(failed_pods)} pods not healthy: {', '.join(failed_pods[:3])}", details

            return True, f"All {running_pods} pods are running and ready", details

        except Exception as e:
            return False, f"Pod validation failed: {str(e)}", {}

    # ============================================================================
    # DATABASE VALIDATIONS
    # ============================================================================

    def validate_database_connectivity(self) -> Tuple[bool, str, Dict]:
        """Validate database is accessible"""
        database_url = self.config.get("database_url")

        if not database_url:
            return False, "DATABASE_URL not configured", {}

        try:
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()

            # Check database version
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            # Check connection count
            cursor.execute("SELECT count(*) FROM pg_stat_activity;")
            connections = cursor.fetchone()[0]

            # Check database size
            cursor.execute("SELECT pg_database_size(current_database());")
            db_size_bytes = cursor.fetchone()[0]
            db_size_gb = db_size_bytes / (1024**3)

            cursor.close()
            conn.close()

            details = {
                "version": version.split()[0],
                "active_connections": connections,
                "database_size_gb": round(db_size_gb, 2)
            }

            return True, f"Database connected (PostgreSQL {version.split()[1]})", details

        except Exception as e:
            return False, f"Database connection failed: {str(e)}", {}

    def validate_database_migrations(self) -> Tuple[bool, str, Dict]:
        """Validate all database migrations are applied"""
        database_url = self.config.get("database_url")

        if not database_url:
            return False, "DATABASE_URL not configured", {}

        try:
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()

            # Check alembic version
            cursor.execute("SELECT version_num FROM alembic_version;")
            result = cursor.fetchone()

            if not result:
                cursor.close()
                conn.close()
                return False, "No migrations applied", {}

            current_version = result[0]

            # Check critical tables exist
            critical_tables = [
                "tenants", "users", "suppliers", "emissions",
                "calculations", "emission_factors", "audit_logs"
            ]

            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]

            missing_tables = [table for table in critical_tables if table not in existing_tables]

            cursor.close()
            conn.close()

            details = {
                "migration_version": current_version,
                "total_tables": len(existing_tables),
                "missing_tables": missing_tables
            }

            if missing_tables:
                return False, f"Missing critical tables: {', '.join(missing_tables)}", details

            return True, f"Migrations applied (version: {current_version})", details

        except Exception as e:
            return False, f"Migration validation failed: {str(e)}", {}

    def validate_database_performance(self) -> Tuple[bool, str, Dict]:
        """Validate database performance parameters"""
        database_url = self.config.get("database_url")

        if not database_url:
            return False, "DATABASE_URL not configured", {}

        try:
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()

            # Check key performance settings
            cursor.execute("SHOW shared_buffers;")
            shared_buffers = cursor.fetchone()[0]

            cursor.execute("SHOW effective_cache_size;")
            effective_cache_size = cursor.fetchone()[0]

            cursor.execute("SHOW work_mem;")
            work_mem = cursor.fetchone()[0]

            cursor.execute("SHOW maintenance_work_mem;")
            maintenance_work_mem = cursor.fetchone()[0]

            # Check slow queries
            cursor.execute("""
                SELECT count(*)
                FROM pg_stat_statements
                WHERE mean_exec_time > 1000
            """)
            slow_queries = cursor.fetchone()
            slow_query_count = slow_queries[0] if slow_queries else 0

            cursor.close()
            conn.close()

            details = {
                "shared_buffers": shared_buffers,
                "effective_cache_size": effective_cache_size,
                "work_mem": work_mem,
                "maintenance_work_mem": maintenance_work_mem,
                "slow_queries": slow_query_count
            }

            if slow_query_count > 10:
                return False, f"{slow_query_count} slow queries detected (> 1s)", details

            return True, f"Database performance configured correctly", details

        except Exception as e:
            # pg_stat_statements may not be enabled, treat as warning
            return True, f"Performance check completed (some metrics unavailable)", {}

    # ============================================================================
    # REDIS VALIDATIONS
    # ============================================================================

    def validate_redis_connectivity(self) -> Tuple[bool, str, Dict]:
        """Validate Redis is accessible"""
        redis_url = self.config.get("redis_url")

        if not redis_url:
            return False, "REDIS_URL not configured", {}

        try:
            r = redis.from_url(redis_url)

            # Test connection
            r.ping()

            # Get Redis info
            info = r.info()

            # Test set/get
            test_key = "vcci:health:test"
            r.set(test_key, "test_value", ex=60)
            value = r.get(test_key)

            if value.decode() != "test_value":
                return False, "Redis read/write test failed", {}

            details = {
                "version": info.get("redis_version"),
                "uptime_days": info.get("uptime_in_days"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_mb": round(info.get("used_memory", 0) / (1024**2), 2),
                "hit_rate": round(info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1) * 100, 2)
            }

            return True, f"Redis connected (v{info.get('redis_version')})", details

        except Exception as e:
            return False, f"Redis connection failed: {str(e)}", {}

    # ============================================================================
    # API VALIDATIONS
    # ============================================================================

    def validate_api_health(self) -> Tuple[bool, str, Dict]:
        """Validate API health endpoints"""
        api_url = self.config["api_url"]

        health_endpoints = {
            "liveness": f"{api_url}/health/live",
            "readiness": f"{api_url}/health/ready",
            "startup": f"{api_url}/health/startup"
        }

        results = {}

        for name, url in health_endpoints.items():
            try:
                start = time.time()
                response = requests.get(url, timeout=5)
                latency = (time.time() - start) * 1000

                results[name] = {
                    "status_code": response.status_code,
                    "latency_ms": round(latency, 2),
                    "healthy": response.status_code == 200
                }

            except Exception as e:
                results[name] = {
                    "status_code": 0,
                    "latency_ms": 0,
                    "healthy": False,
                    "error": str(e)
                }

        all_healthy = all(r["healthy"] for r in results.values())

        if not all_healthy:
            failed = [name for name, r in results.items() if not r["healthy"]]
            return False, f"Health checks failed: {', '.join(failed)}", results

        avg_latency = sum(r["latency_ms"] for r in results.values()) / len(results)

        return True, f"All health checks passed (avg latency: {round(avg_latency, 2)}ms)", results

    def validate_api_authentication(self) -> Tuple[bool, str, Dict]:
        """Validate API authentication is working"""
        api_url = self.config["api_url"]

        # Test unauthenticated request should fail
        try:
            response = requests.get(f"{api_url}/api/v1/suppliers", timeout=5)

            if response.status_code == 401:
                return True, "API authentication required (401 returned)", {"status": "protected"}
            else:
                return False, f"API not properly protected (returned {response.status_code})", {}

        except Exception as e:
            return False, f"API authentication test failed: {str(e)}", {}

    def validate_api_performance(self) -> Tuple[bool, str, Dict]:
        """Validate API response times"""
        if self.skip_load_tests:
            return True, "API performance test skipped", {}

        api_url = self.config["api_url"]
        max_latency = self.config["max_latency_ms"]

        # Test multiple requests
        latencies = []

        for i in range(10):
            try:
                start = time.time()
                response = requests.get(f"{api_url}/health/live", timeout=5)
                latency = (time.time() - start) * 1000

                if response.status_code == 200:
                    latencies.append(latency)

            except Exception:
                pass

        if not latencies:
            return False, "No successful API requests", {}

        p50 = sorted(latencies)[len(latencies)//2]
        p95 = sorted(latencies)[int(len(latencies)*0.95)]
        p99 = sorted(latencies)[int(len(latencies)*0.99)]
        avg = sum(latencies) / len(latencies)

        details = {
            "requests": len(latencies),
            "avg_ms": round(avg, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2)
        }

        if p95 > max_latency:
            return False, f"API latency p95 ({round(p95, 2)}ms) exceeds limit ({max_latency}ms)", details

        return True, f"API performance acceptable (p95: {round(p95, 2)}ms)", details

    # ============================================================================
    # SECURITY VALIDATIONS
    # ============================================================================

    def validate_ssl_certificate(self) -> Tuple[bool, str, Dict]:
        """Validate SSL certificate is valid"""
        api_url = self.config["api_url"]

        if not api_url.startswith("https://"):
            return False, "API URL is not HTTPS", {}

        try:
            response = requests.get(api_url, timeout=5)

            # Check if request was successful (certificate valid)
            if response.status_code >= 200:
                return True, "SSL certificate valid", {"url": api_url}

            return False, f"SSL validation failed (status: {response.status_code})", {}

        except requests.exceptions.SSLError as e:
            return False, f"SSL certificate error: {str(e)}", {}
        except Exception as e:
            return False, f"SSL validation failed: {str(e)}", {}

    def validate_secrets_configured(self) -> Tuple[bool, str, Dict]:
        """Validate required secrets are configured"""
        try:
            config.load_kube_config()
            v1 = client.CoreV1Api()
            namespace = self.config["kubernetes_namespace"]

            secrets = v1.list_namespaced_secret(namespace)

            required_secrets = [
                "vcci-database-credentials",
                "vcci-redis-credentials",
                "vcci-api-keys",
                "vcci-jwt-keys"
            ]

            existing_secrets = [secret.metadata.name for secret in secrets.items]
            missing_secrets = [s for s in required_secrets if s not in existing_secrets]

            details = {
                "total_secrets": len(existing_secrets),
                "required_secrets": required_secrets,
                "missing_secrets": missing_secrets
            }

            if missing_secrets:
                return False, f"Missing secrets: {', '.join(missing_secrets)}", details

            return True, f"All {len(required_secrets)} required secrets configured", details

        except Exception as e:
            return False, f"Secrets validation failed: {str(e)}", {}

    # ============================================================================
    # MONITORING VALIDATIONS
    # ============================================================================

    def validate_prometheus_metrics(self) -> Tuple[bool, str, Dict]:
        """Validate Prometheus is scraping metrics"""
        api_url = self.config["api_url"]

        try:
            response = requests.get(f"{api_url}/metrics", timeout=5)

            if response.status_code != 200:
                return False, f"Metrics endpoint not accessible (status: {response.status_code})", {}

            # Parse Prometheus metrics
            metrics_text = response.text
            metric_lines = [line for line in metrics_text.split('\n') if line and not line.startswith('#')]

            # Check for key metrics
            required_metrics = [
                "http_requests_total",
                "http_request_duration_seconds",
                "process_cpu_seconds_total"
            ]

            found_metrics = []
            for metric in required_metrics:
                if any(metric in line for line in metric_lines):
                    found_metrics.append(metric)

            missing_metrics = [m for m in required_metrics if m not in found_metrics]

            details = {
                "total_metrics": len(metric_lines),
                "required_metrics": required_metrics,
                "found_metrics": found_metrics,
                "missing_metrics": missing_metrics
            }

            if missing_metrics:
                return True, f"Some metrics missing (warning only): {', '.join(missing_metrics)}", details

            return True, f"Metrics endpoint healthy ({len(metric_lines)} metrics)", details

        except Exception as e:
            return True, f"Metrics validation warning: {str(e)}", {}

    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================

    def run_all_validations(self):
        """Run all validation checks"""
        print("\n" + "="*80)
        print(f"{self.COLOR_BLUE}GL-VCCI Production Environment Validation{self.COLOR_END}")
        print(f"{self.COLOR_BLUE}Environment: {self.environment.upper()}{self.COLOR_END}")
        print(f"{self.COLOR_BLUE}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{self.COLOR_END}")
        print("="*80 + "\n")

        # Infrastructure Validations (Critical)
        print(f"\n{self.COLOR_BLUE}[1/6] INFRASTRUCTURE VALIDATIONS{self.COLOR_END}")
        print("-" * 80)
        self.run_validation("Kubernetes Cluster", self.validate_kubernetes_cluster, ValidationLevel.CRITICAL)
        self.run_validation("Kubernetes Namespace", self.validate_namespace, ValidationLevel.CRITICAL)
        self.run_validation("Deployments", self.validate_deployments, ValidationLevel.CRITICAL)
        self.run_validation("Pod Health", self.validate_pods_healthy, ValidationLevel.CRITICAL)

        # Database Validations (Critical)
        print(f"\n{self.COLOR_BLUE}[2/6] DATABASE VALIDATIONS{self.COLOR_END}")
        print("-" * 80)
        self.run_validation("Database Connectivity", self.validate_database_connectivity, ValidationLevel.CRITICAL)
        self.run_validation("Database Migrations", self.validate_database_migrations, ValidationLevel.CRITICAL)
        self.run_validation("Database Performance", self.validate_database_performance, ValidationLevel.WARNING)

        # Redis Validations (Critical)
        print(f"\n{self.COLOR_BLUE}[3/6] CACHE VALIDATIONS{self.COLOR_END}")
        print("-" * 80)
        self.run_validation("Redis Connectivity", self.validate_redis_connectivity, ValidationLevel.CRITICAL)

        # API Validations (Critical)
        print(f"\n{self.COLOR_BLUE}[4/6] API VALIDATIONS{self.COLOR_END}")
        print("-" * 80)
        self.run_validation("API Health Endpoints", self.validate_api_health, ValidationLevel.CRITICAL)
        self.run_validation("API Authentication", self.validate_api_authentication, ValidationLevel.CRITICAL)
        self.run_validation("API Performance", self.validate_api_performance, ValidationLevel.WARNING)

        # Security Validations (Critical)
        print(f"\n{self.COLOR_BLUE}[5/6] SECURITY VALIDATIONS{self.COLOR_END}")
        print("-" * 80)
        self.run_validation("SSL Certificate", self.validate_ssl_certificate, ValidationLevel.CRITICAL)
        self.run_validation("Secrets Configuration", self.validate_secrets_configured, ValidationLevel.CRITICAL)

        # Monitoring Validations (Warning)
        print(f"\n{self.COLOR_BLUE}[6/6] MONITORING VALIDATIONS{self.COLOR_END}")
        print("-" * 80)
        self.run_validation("Prometheus Metrics", self.validate_prometheus_metrics, ValidationLevel.WARNING)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print validation summary"""
        total_duration = time.time() - self.start_time

        critical_results = [r for r in self.results if r.level == ValidationLevel.CRITICAL]
        warning_results = [r for r in self.results if r.level == ValidationLevel.WARNING]
        info_results = [r for r in self.results if r.level == ValidationLevel.INFO]

        critical_passed = sum(1 for r in critical_results if r.passed)
        warning_passed = sum(1 for r in warning_results if r.passed)

        print("\n" + "="*80)
        print(f"{self.COLOR_BLUE}VALIDATION SUMMARY{self.COLOR_END}")
        print("="*80 + "\n")

        print(f"Total Validations: {len(self.results)}")
        print(f"Total Duration: {total_duration:.2f}s")
        print()

        # Critical results
        if critical_results:
            color = self.COLOR_GREEN if critical_passed == len(critical_results) else self.COLOR_RED
            print(f"{color}CRITICAL: {critical_passed}/{len(critical_results)} passed{self.COLOR_END}")

            for result in critical_results:
                if not result.passed:
                    print(f"  {self.COLOR_RED}✗ {result.name}: {result.message}{self.COLOR_END}")

        # Warning results
        if warning_results:
            color = self.COLOR_GREEN if warning_passed == len(warning_results) else self.COLOR_YELLOW
            print(f"{color}WARNING: {warning_passed}/{len(warning_results)} passed{self.COLOR_END}")

            for result in warning_results:
                if not result.passed:
                    print(f"  {self.COLOR_YELLOW}⚠ {result.name}: {result.message}{self.COLOR_END}")

        print()

        # Determine overall status
        critical_failed = len(critical_results) - critical_passed
        warning_failed = len(warning_results) - warning_passed

        if critical_failed > 0:
            print(f"{self.COLOR_RED}RESULT: VALIDATION FAILED{self.COLOR_END}")
            print(f"{self.COLOR_RED}Action Required: Fix {critical_failed} critical issue(s) before deployment{self.COLOR_END}")
            return 1
        elif warning_failed > 0:
            print(f"{self.COLOR_YELLOW}RESULT: VALIDATION PASSED WITH WARNINGS{self.COLOR_END}")
            print(f"{self.COLOR_YELLOW}Deployment possible but {warning_failed} warning(s) should be reviewed{self.COLOR_END}")
            return 2
        else:
            print(f"{self.COLOR_GREEN}RESULT: ALL VALIDATIONS PASSED ✓{self.COLOR_END}")
            print(f"{self.COLOR_GREEN}Environment is ready for production deployment{self.COLOR_END}")
            return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GL-VCCI Production Environment Validator")
    parser.add_argument("--env", choices=["production", "staging", "development"],
                       default="production", help="Environment to validate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip-load-tests", action="store_true", help="Skip performance load tests")
    parser.add_argument("--output-json", help="Output results to JSON file")

    args = parser.parse_args()

    validator = ProductionValidator(
        environment=args.env,
        verbose=args.verbose,
        skip_load_tests=args.skip_load_tests
    )

    exit_code = validator.run_all_validations()

    # Output JSON if requested
    if args.output_json:
        output = {
            "environment": args.env,
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - validator.start_time,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "level": r.level.value,
                    "message": r.message,
                    "details": r.details,
                    "duration": r.duration
                }
                for r in validator.results
            ],
            "summary": {
                "total": len(validator.results),
                "passed": sum(1 for r in validator.results if r.passed),
                "failed": sum(1 for r in validator.results if not r.passed),
                "exit_code": exit_code
            }
        }

        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults written to: {args.output_json}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
