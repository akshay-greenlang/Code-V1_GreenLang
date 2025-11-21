#!/usr/bin/env python3
"""
GreenLang Agent Configuration Validator

Validates agent-specific configurations and ensures they meet
operational requirements.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any


class AgentConfigValidator:
    """Validator for GreenLang agent configurations."""

    REQUIRED_RUNTIME_CONFIGS = [
        "max_memory_mb",
        "max_cpu_percent",
        "timeout_seconds",
    ]

    REQUIRED_API_CONFIGS = [
        "endpoints",
        "rate_limit",
        "authentication",
    ]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Validate the agent configuration."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                self.errors.append("Configuration file is empty")
                return False

            # Validate runtime configuration
            self._validate_runtime_config(data)

            # Validate API configuration
            self._validate_api_config(data)

            # Validate security settings
            self._validate_security_settings(data)

            # Validate resource limits
            self._validate_resource_limits(data)

            # Validate monitoring configuration
            self._validate_monitoring_config(data)

            return len(self.errors) == 0

        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"File not found: {self.file_path}")
            return False
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
            return False

    def _validate_runtime_config(self, data: Dict[str, Any]) -> None:
        """Validate runtime configuration."""
        if "runtime" not in data:
            self.warnings.append("No runtime configuration found")
            return

        runtime = data["runtime"]

        # Check max memory
        if "max_memory_mb" in runtime:
            memory = runtime["max_memory_mb"]
            if not isinstance(memory, int) or memory < 128 or memory > 8192:
                self.errors.append(
                    "max_memory_mb must be between 128 and 8192"
                )

        # Check max CPU
        if "max_cpu_percent" in runtime:
            cpu = runtime["max_cpu_percent"]
            if not isinstance(cpu, int) or cpu < 1 or cpu > 100:
                self.errors.append(
                    "max_cpu_percent must be between 1 and 100"
                )

        # Check timeout
        if "timeout_seconds" in runtime:
            timeout = runtime["timeout_seconds"]
            if not isinstance(timeout, int) or timeout < 1 or timeout > 3600:
                self.errors.append(
                    "timeout_seconds must be between 1 and 3600"
                )

    def _validate_api_config(self, data: Dict[str, Any]) -> None:
        """Validate API configuration."""
        if "api" not in data:
            self.warnings.append("No API configuration found")
            return

        api = data["api"]

        # Validate endpoints
        if "endpoints" in api:
            endpoints = api["endpoints"]
            if not isinstance(endpoints, list):
                self.errors.append("api.endpoints must be a list")
            else:
                for endpoint in endpoints:
                    if not isinstance(endpoint, dict):
                        self.errors.append("Each endpoint must be a dictionary")
                    else:
                        if "path" not in endpoint:
                            self.errors.append("Endpoint missing 'path'")
                        if "method" not in endpoint:
                            self.errors.append("Endpoint missing 'method'")
                        elif endpoint["method"] not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                            self.errors.append(
                                f"Invalid HTTP method: {endpoint['method']}"
                            )

        # Validate rate limiting
        if "rate_limit" in api:
            rate_limit = api["rate_limit"]
            if not isinstance(rate_limit, dict):
                self.errors.append("api.rate_limit must be a dictionary")
            else:
                if "requests_per_minute" in rate_limit:
                    rpm = rate_limit["requests_per_minute"]
                    if not isinstance(rpm, int) or rpm < 1:
                        self.errors.append(
                            "rate_limit.requests_per_minute must be positive integer"
                        )

        # Validate authentication
        if "authentication" in api:
            auth = api["authentication"]
            if not isinstance(auth, dict):
                self.errors.append("api.authentication must be a dictionary")
            else:
                if "type" in auth:
                    valid_types = ["none", "api_key", "bearer", "oauth2", "jwt"]
                    if auth["type"] not in valid_types:
                        self.errors.append(
                            f"Invalid authentication type: {auth['type']}"
                        )

    def _validate_security_settings(self, data: Dict[str, Any]) -> None:
        """Validate security settings."""
        if "security" in data:
            security = data["security"]

            # Check HTTPS requirement
            if "require_https" in security:
                if not isinstance(security["require_https"], bool):
                    self.errors.append("security.require_https must be boolean")
                elif not security["require_https"]:
                    self.warnings.append(
                        "HTTPS is not required - consider enabling for production"
                    )

            # Check CORS settings
            if "cors" in security:
                cors = security["cors"]
                if "allowed_origins" in cors:
                    if "*" in cors["allowed_origins"]:
                        self.warnings.append(
                            "CORS allows all origins - restrict for production"
                        )

            # Check API key validation
            if "validate_api_keys" in security:
                if not security["validate_api_keys"]:
                    self.errors.append(
                        "API key validation should be enabled for security"
                    )

    def _validate_resource_limits(self, data: Dict[str, Any]) -> None:
        """Validate resource limits."""
        if "resources" in data:
            resources = data["resources"]

            # Database connections
            if "database_connections" in resources:
                db_conns = resources["database_connections"]
                if not isinstance(db_conns, dict):
                    self.errors.append("resources.database_connections must be dictionary")
                else:
                    if "max_pool_size" in db_conns:
                        pool_size = db_conns["max_pool_size"]
                        if not isinstance(pool_size, int) or pool_size < 1 or pool_size > 100:
                            self.errors.append(
                                "max_pool_size must be between 1 and 100"
                            )

            # Redis connections
            if "redis_connections" in resources:
                redis_conns = resources["redis_connections"]
                if not isinstance(redis_conns, dict):
                    self.errors.append("resources.redis_connections must be dictionary")

    def _validate_monitoring_config(self, data: Dict[str, Any]) -> None:
        """Validate monitoring configuration."""
        if "monitoring" not in data:
            self.warnings.append(
                "No monitoring configuration - consider adding metrics"
            )
            return

        monitoring = data["monitoring"]

        # Check metrics endpoint
        if "metrics_enabled" in monitoring:
            if monitoring["metrics_enabled"]:
                if "metrics_port" not in monitoring:
                    self.warnings.append(
                        "Metrics enabled but no metrics_port specified"
                    )

        # Check health check
        if "health_check" in monitoring:
            health = monitoring["health_check"]
            if not isinstance(health, dict):
                self.errors.append("monitoring.health_check must be dictionary")
            else:
                if "endpoint" not in health:
                    self.warnings.append(
                        "Health check configuration missing endpoint"
                    )
                if "interval_seconds" in health:
                    interval = health["interval_seconds"]
                    if not isinstance(interval, int) or interval < 5 or interval > 300:
                        self.errors.append(
                            "health_check.interval_seconds must be between 5 and 300"
                        )

        # Check logging configuration
        if "logging" in monitoring:
            logging = monitoring["logging"]
            if "level" in logging:
                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                if logging["level"] not in valid_levels:
                    self.errors.append(
                        f"Invalid logging level: {logging['level']}"
                    )

    def print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            print(f"\n{'='*80}")
            print(f"VALIDATION FAILED: {self.file_path}")
            print(f"{'='*80}")
            for error in self.errors:
                print(f"ERROR: {error}")

        if self.warnings:
            print(f"\n{'='*80}")
            print(f"WARNINGS: {self.file_path}")
            print(f"{'='*80}")
            for warning in self.warnings:
                print(f"WARNING: {warning}")

        if not self.errors and not self.warnings:
            print(f"✓ {self.file_path} - Valid")


def main() -> int:
    """Main entry point for the hook."""
    if len(sys.argv) < 2:
        print("Usage: validate-agent-config.py <pack.yaml> [pack.yaml ...]")
        return 1

    all_valid = True
    for file_path in sys.argv[1:]:
        validator = AgentConfigValidator(file_path)
        is_valid = validator.validate()
        validator.print_results()

        if not is_valid:
            all_valid = False

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
