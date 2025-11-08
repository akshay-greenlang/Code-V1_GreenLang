"""
CBAM Importer Copilot - Health Check Endpoints

Production-grade health checks for monitoring system availability,
readiness, and liveness. Compatible with Kubernetes health probes
and monitoring systems.

Health Check Levels:
1. /health - Basic health (always returns 200 if service is running)
2. /health/ready - Readiness check (checks dependencies are available)
3. /health/live - Liveness check (checks application is functioning)

Version: 1.0.0
Author: GreenLang CBAM Team (Team A3: Monitoring & Observability)
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ============================================================================
# HEALTH CHECK MODELS
# ============================================================================

class HealthStatus:
    """Health check status constants."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheckResult:
    """Result of a health check operation."""

    def __init__(
        self,
        status: str,
        message: str = "",
        details: Dict[str, Any] = None,
        duration_ms: float = 0
    ):
        self.status = status
        self.message = message
        self.details = details or {}
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2)
        }


# ============================================================================
# DEPENDENCY HEALTH CHECKS
# ============================================================================

class DependencyHealthChecker:
    """Check health of system dependencies."""

    def __init__(self, base_path: str = None):
        """
        Initialize dependency health checker.

        Args:
            base_path: Base path for CBAM application
        """
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent

    def check_file_system(self) -> HealthCheckResult:
        """Check if required directories are accessible."""
        start_time = time.time()

        try:
            required_dirs = [
                self.base_path / "data",
                self.base_path / "rules",
                self.base_path / "schemas"
            ]

            missing_dirs = []
            for directory in required_dirs:
                if not directory.exists():
                    missing_dirs.append(str(directory))

            duration_ms = (time.time() - start_time) * 1000

            if missing_dirs:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message=f"Missing {len(missing_dirs)} required directories",
                    details={"missing_directories": missing_dirs},
                    duration_ms=duration_ms
                )

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="All required directories accessible",
                details={"directories_checked": len(required_dirs)},
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {str(e)}",
                duration_ms=duration_ms
            )

    def check_reference_data(self) -> HealthCheckResult:
        """Check if required reference data files are present."""
        start_time = time.time()

        try:
            required_files = {
                "cn_codes": self.base_path / "data" / "cn_codes.json",
                "cbam_rules": self.base_path / "rules" / "cbam_rules.yaml",
                "emission_factors": self.base_path / "data" / "emission_factors.py"
            }

            missing_files = []
            file_sizes = {}

            for name, file_path in required_files.items():
                if not file_path.exists():
                    missing_files.append(name)
                else:
                    file_sizes[name] = file_path.stat().st_size

            duration_ms = (time.time() - start_time) * 1000

            if missing_files:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Missing {len(missing_files)} required reference data files",
                    details={"missing_files": missing_files},
                    duration_ms=duration_ms
                )

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="All reference data files present",
                details={"file_sizes": file_sizes},
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Reference data check failed: {str(e)}",
                duration_ms=duration_ms
            )

    def check_python_dependencies(self) -> HealthCheckResult:
        """Check if required Python packages are installed."""
        start_time = time.time()

        try:
            required_packages = [
                "pandas",
                "pydantic",
                "jsonschema",
                "yaml",
                "openpyxl"
            ]

            missing_packages = []
            package_versions = {}

            for package in required_packages:
                try:
                    if package == "yaml":
                        import yaml
                        package_versions[package] = getattr(yaml, "__version__", "unknown")
                    else:
                        module = __import__(package)
                        package_versions[package] = getattr(module, "__version__", "unknown")
                except ImportError:
                    missing_packages.append(package)

            duration_ms = (time.time() - start_time) * 1000

            if missing_packages:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Missing {len(missing_packages)} required Python packages",
                    details={"missing_packages": missing_packages},
                    duration_ms=duration_ms
                )

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="All required Python packages installed",
                details={"packages": package_versions},
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Python dependencies check failed: {str(e)}",
                duration_ms=duration_ms
            )


# ============================================================================
# APPLICATION HEALTH CHECKER
# ============================================================================

class CBAMHealthChecker:
    """
    Comprehensive health checker for CBAM Importer Copilot.

    Provides three levels of health checks:
    1. Basic health - Service is running
    2. Readiness - Dependencies are available
    3. Liveness - Application is functioning correctly
    """

    def __init__(self, base_path: str = None):
        """
        Initialize CBAM health checker.

        Args:
            base_path: Base path for CBAM application
        """
        self.base_path = base_path or str(Path(__file__).parent.parent)
        self.dependency_checker = DependencyHealthChecker(self.base_path)
        self.start_time = datetime.now()

    def basic_health(self) -> Dict[str, Any]:
        """
        Basic health check - just confirms service is running.

        This should always return 200 if the service is up.
        Used for: Basic monitoring, uptime checks

        Returns:
            Health status dictionary
        """
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "cbam-importer-copilot",
            "version": "1.0.0",
            "uptime_seconds": round(uptime_seconds, 2),
            "uptime_human": self._format_uptime(uptime_seconds)
        }

    def readiness_check(self) -> Tuple[Dict[str, Any], int]:
        """
        Readiness check - verifies all dependencies are available.

        This checks if the service is ready to accept requests.
        Used for: Kubernetes readiness probes, load balancer health checks

        Returns:
            Tuple of (health status dictionary, HTTP status code)
        """
        start_time = time.time()

        # Run all dependency checks
        checks = {
            "file_system": self.dependency_checker.check_file_system(),
            "reference_data": self.dependency_checker.check_reference_data(),
            "python_dependencies": self.dependency_checker.check_python_dependencies()
        }

        # Aggregate results
        all_healthy = all(
            check.status == HealthStatus.HEALTHY
            for check in checks.values()
        )

        any_unhealthy = any(
            check.status == HealthStatus.UNHEALTHY
            for check in checks.values()
        )

        # Determine overall status
        if all_healthy:
            overall_status = HealthStatus.HEALTHY
            http_status = 200
        elif any_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
            http_status = 503  # Service Unavailable
        else:
            overall_status = HealthStatus.DEGRADED
            http_status = 200  # Still accepting requests but degraded

        total_duration_ms = (time.time() - start_time) * 1000

        result = {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "service": "cbam-importer-copilot",
            "checks": {
                name: check.to_dict()
                for name, check in checks.items()
            },
            "duration_ms": round(total_duration_ms, 2)
        }

        return result, http_status

    def liveness_check(self) -> Tuple[Dict[str, Any], int]:
        """
        Liveness check - verifies application is functioning correctly.

        This performs a lightweight functional test to ensure the app
        hasn't deadlocked or entered an unrecoverable state.

        Used for: Kubernetes liveness probes (restart on failure)

        Returns:
            Tuple of (health status dictionary, HTTP status code)
        """
        start_time = time.time()

        checks = {}

        # Check 1: Can we import core modules?
        try:
            from agents.shipment_intake_agent import ShipmentIntakeAgent
            from agents.emissions_calculator_agent import EmissionsCalculatorAgent
            from agents.reporting_packager_agent import ReportingPackagerAgent

            checks["module_import"] = HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Core modules importable",
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            checks["module_import"] = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to import core modules: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Check 2: Can we perform basic data operations?
        try:
            import pandas as pd
            test_df = pd.DataFrame({"test": [1, 2, 3]})
            assert len(test_df) == 3

            checks["data_operations"] = HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Basic data operations functional",
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            checks["data_operations"] = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Data operations failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )

        # Determine overall status
        any_unhealthy = any(
            check.status == HealthStatus.UNHEALTHY
            for check in checks.values()
        )

        overall_status = HealthStatus.UNHEALTHY if any_unhealthy else HealthStatus.HEALTHY
        http_status = 503 if any_unhealthy else 200

        total_duration_ms = (time.time() - start_time) * 1000

        result = {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "service": "cbam-importer-copilot",
            "checks": {
                name: check.to_dict()
                for name, check in checks.items()
            },
            "duration_ms": round(total_duration_ms, 2)
        }

        return result, http_status

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime in human-readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)


# ============================================================================
# FASTAPI INTEGRATION (for future web deployment)
# ============================================================================

def create_health_endpoints():
    """
    Create FastAPI health check endpoints.

    This function can be imported by a FastAPI app to add health endpoints.

    Example:
        from fastapi import FastAPI
        from backend.health import create_health_endpoints

        app = FastAPI()
        health_checker, routes = create_health_endpoints()

        for route in routes:
            app.add_api_route(**route)
    """
    try:
        from fastapi import APIRouter, Response
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.warning("FastAPI not installed. Health endpoints will not be available for web deployment.")
        return None, []

    health_checker = CBAMHealthChecker()
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("")
    async def health():
        """Basic health check endpoint."""
        result = health_checker.basic_health()
        return JSONResponse(content=result, status_code=200)

    @router.get("/ready")
    async def readiness():
        """Readiness check endpoint."""
        result, status_code = health_checker.readiness_check()
        return JSONResponse(content=result, status_code=status_code)

    @router.get("/live")
    async def liveness():
        """Liveness check endpoint."""
        result, status_code = health_checker.liveness_check()
        return JSONResponse(content=result, status_code=status_code)

    routes = [
        {
            "path": "/health",
            "endpoint": health,
            "methods": ["GET"],
            "tags": ["health"]
        },
        {
            "path": "/health/ready",
            "endpoint": readiness,
            "methods": ["GET"],
            "tags": ["health"]
        },
        {
            "path": "/health/live",
            "endpoint": liveness,
            "methods": ["GET"],
            "tags": ["health"]
        }
    ]

    return health_checker, routes


# ============================================================================
# CLI INTERFACE (for testing health checks)
# ============================================================================

def main():
    """CLI interface for testing health checks."""
    import argparse

    parser = argparse.ArgumentParser(description="CBAM Health Check CLI")
    parser.add_argument(
        "--check",
        choices=["basic", "ready", "live", "all"],
        default="all",
        help="Type of health check to run"
    )
    parser.add_argument(
        "--base-path",
        help="Base path for CBAM application"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    checker = CBAMHealthChecker(args.base_path)

    results = {}

    if args.check in ["basic", "all"]:
        results["basic"] = checker.basic_health()

    if args.check in ["ready", "all"]:
        result, status = checker.readiness_check()
        results["ready"] = result

    if args.check in ["live", "all"]:
        result, status = checker.liveness_check()
        results["live"] = result

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for check_type, result in results.items():
            print(f"\n{'='*60}")
            print(f"{check_type.upper()} HEALTH CHECK")
            print(f"{'='*60}")
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
