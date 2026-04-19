"""
Dimension 12: Production Readiness Verification

This dimension verifies that agents are ready for production deployment
including logging, error handling, and health checks.

Checks:
    - Logging configured
    - Error handling complete
    - Health checks implemented
    - Configuration management

Example:
    >>> dimension = ProductionReadinessDimension()
    >>> result = dimension.evaluate(agent_path, agent, sample_input)
    >>> assert result.status == DimensionStatus.PASS
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDimension, CheckResult, DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class ProductionReadinessDimension(BaseDimension):
    """
    Production Readiness Dimension Evaluator (D12).

    Verifies that agents are ready for production deployment.

    Configuration:
        require_health_check: Require health check endpoint (default: True)
        require_structured_logging: Require structured logging (default: True)
    """

    DIMENSION_ID = "D12"
    DIMENSION_NAME = "Production Readiness"
    DESCRIPTION = "Verifies logging configured, error handling complete, health checks implemented"
    WEIGHT = 1.0
    REQUIRED_FOR_CERTIFICATION = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize production readiness dimension evaluator."""
        super().__init__(config)

        self.require_health_check = self.config.get("require_health_check", True)
        self.require_structured_logging = self.config.get("require_structured_logging", True)

    def evaluate(
        self,
        agent_path: Path,
        agent: Optional[Any] = None,
        sample_input: Optional[Any] = None,
    ) -> DimensionResult:
        """
        Evaluate production readiness for the given agent.

        Args:
            agent_path: Path to agent directory
            agent: Optional agent instance
            sample_input: Optional sample input

        Returns:
            DimensionResult with production readiness evaluation
        """
        start_time = datetime.utcnow()
        self._reset_checks()

        logger.info("Starting production readiness evaluation")

        # Get all Python files
        python_files = list(agent_path.glob("**/*.py"))
        if not python_files:
            self._add_check(
                name="python_files_exist",
                passed=False,
                message="No Python files found",
                severity="error",
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return self._create_result(execution_time)

        # Combine all source code
        all_source = ""
        for py_file in python_files:
            try:
                all_source += py_file.read_text(encoding="utf-8") + "\n"
            except Exception:
                pass

        # Check 1: Logging configured
        logging_check = self._check_logging(all_source)
        self._add_check(
            name="logging_configured",
            passed=logging_check["has_logging"],
            message="Logging is configured"
            if logging_check["has_logging"]
            else "No logging configuration found",
            severity="error",
            details=logging_check,
        )

        # Check 2: Structured logging
        if self.require_structured_logging:
            structured_check = self._check_structured_logging(all_source)
            self._add_check(
                name="structured_logging",
                passed=structured_check["is_structured"],
                message="Uses structured logging"
                if structured_check["is_structured"]
                else "Consider using structured logging",
                severity="warning",
                details=structured_check,
            )

        # Check 3: Error handling
        error_handling = self._check_error_handling(all_source)
        self._add_check(
            name="error_handling",
            passed=error_handling["has_handling"],
            message="Error handling is implemented"
            if error_handling["has_handling"]
            else "Missing error handling",
            severity="error",
            details=error_handling,
        )

        # Check 4: Health check
        health_check = self._check_health_check(all_source, agent_path)
        self._add_check(
            name="health_check",
            passed=health_check["has_health_check"],
            message="Health check is implemented"
            if health_check["has_health_check"]
            else "No health check found",
            severity="warning" if self.require_health_check else "info",
            details=health_check,
        )

        # Check 5: Configuration management
        config_check = self._check_configuration_management(all_source, agent_path)
        self._add_check(
            name="configuration_management",
            passed=config_check["has_config"],
            message="Configuration management present"
            if config_check["has_config"]
            else "No configuration management found",
            severity="warning",
            details=config_check,
        )

        # Check 6: Version information
        version_check = self._check_version_info(all_source, agent_path)
        self._add_check(
            name="version_info",
            passed=version_check["has_version"],
            message=f"Version info: {version_check.get('version', 'found')}"
            if version_check["has_version"]
            else "No version information found",
            severity="warning",
            details=version_check,
        )

        # Check 7: Graceful shutdown
        shutdown_check = self._check_graceful_shutdown(all_source)
        self._add_check(
            name="graceful_shutdown",
            passed=shutdown_check["has_shutdown"],
            message="Graceful shutdown handling present"
            if shutdown_check["has_shutdown"]
            else "No graceful shutdown handling",
            severity="warning",
            details=shutdown_check,
        )

        # Check 8: Retry logic
        retry_check = self._check_retry_logic(all_source)
        self._add_check(
            name="retry_logic",
            passed=True,  # Optional but good
            message="Retry logic present"
            if retry_check["has_retry"]
            else "No retry logic (optional for pure calculations)",
            severity="info",
            details=retry_check,
        )

        # Check 9: Metrics/monitoring
        metrics_check = self._check_metrics(all_source)
        self._add_check(
            name="metrics_monitoring",
            passed=True,  # Optional
            message="Metrics/monitoring present"
            if metrics_check["has_metrics"]
            else "No metrics collection (optional)",
            severity="info",
            details=metrics_check,
        )

        # Check 10: Docker/deployment config
        deployment_check = self._check_deployment_config(agent_path)
        self._add_check(
            name="deployment_config",
            passed=deployment_check["has_config"],
            message="Deployment configuration present"
            if deployment_check["has_config"]
            else "No deployment configuration found",
            severity="warning",
            details=deployment_check,
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return self._create_result(
            execution_time,
            details={
                "has_logging": logging_check["has_logging"],
                "has_error_handling": error_handling["has_handling"],
                "has_health_check": health_check["has_health_check"],
                "has_deployment_config": deployment_check["has_config"],
            },
        )

    def _check_logging(self, source_code: str) -> Dict[str, Any]:
        """
        Check for logging configuration.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with logging check results
        """
        result = {
            "has_logging": False,
            "logging_patterns": [],
        }

        patterns = [
            (r"import\s+logging", "logging import"),
            (r"logging\.getLogger", "getLogger"),
            (r"logger\s*=", "logger assignment"),
            (r"logger\.(info|debug|warning|error|critical)", "logger calls"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, source_code):
                result["has_logging"] = True
                result["logging_patterns"].append(description)

        return result

    def _check_structured_logging(self, source_code: str) -> Dict[str, Any]:
        """
        Check for structured logging.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with structured logging check results
        """
        result = {
            "is_structured": False,
            "structured_patterns": [],
        }

        patterns = [
            (r"structlog", "structlog library"),
            (r"json_format|JSONFormatter", "JSON formatter"),
            (r"extra\s*=\s*\{", "extra fields"),
            (r"logger\.\w+\([^)]*\{[^}]+\}", "dict in log call"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, source_code):
                result["is_structured"] = True
                result["structured_patterns"].append(description)

        return result

    def _check_error_handling(self, source_code: str) -> Dict[str, Any]:
        """
        Check for error handling.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with error handling check results
        """
        result = {
            "has_handling": False,
            "handling_patterns": [],
        }

        patterns = [
            (r"try\s*:", "try blocks"),
            (r"except\s+\w+", "specific exceptions"),
            (r"raise\s+\w+Error", "custom exceptions"),
            (r"finally\s*:", "finally blocks"),
            (r"logger\.error|logger\.exception", "error logging"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, source_code):
                result["has_handling"] = True
                result["handling_patterns"].append(description)

        return result

    def _check_health_check(
        self,
        source_code: str,
        agent_path: Path,
    ) -> Dict[str, Any]:
        """
        Check for health check implementation.

        Args:
            source_code: Combined source code
            agent_path: Path to agent directory

        Returns:
            Dictionary with health check results
        """
        result = {
            "has_health_check": False,
            "health_patterns": [],
        }

        patterns = [
            (r"def\s+health|def\s+healthcheck", "health method"),
            (r"/health|/healthz|/ready|/live", "health endpoint"),
            (r"HealthCheck|health_check", "health check class/func"),
            (r"is_healthy|check_health", "health check method"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_health_check"] = True
                result["health_patterns"].append(description)

        # Check for health check in pack.yaml
        pack_file = agent_path / "pack.yaml"
        if pack_file.exists():
            try:
                content = pack_file.read_text(encoding="utf-8")
                if "health" in content.lower():
                    result["has_health_check"] = True
                    result["health_patterns"].append("pack.yaml health config")
            except Exception:
                pass

        return result

    def _check_configuration_management(
        self,
        source_code: str,
        agent_path: Path,
    ) -> Dict[str, Any]:
        """
        Check for configuration management.

        Args:
            source_code: Combined source code
            agent_path: Path to agent directory

        Returns:
            Dictionary with configuration check results
        """
        result = {
            "has_config": False,
            "config_patterns": [],
        }

        patterns = [
            (r"os\.environ|os\.getenv", "environment variables"),
            (r"config\.yaml|config\.json", "config file"),
            (r"pydantic.*Settings|BaseSettings", "Pydantic settings"),
            (r"configparser|ConfigParser", "ConfigParser"),
            (r"dotenv|load_dotenv", "dotenv"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, source_code):
                result["has_config"] = True
                result["config_patterns"].append(description)

        # Check for config files
        config_files = ["config.yaml", "config.json", "settings.py", ".env.example"]
        for config_file in config_files:
            if (agent_path / config_file).exists():
                result["has_config"] = True
                result["config_patterns"].append(f"{config_file} file")

        return result

    def _check_version_info(
        self,
        source_code: str,
        agent_path: Path,
    ) -> Dict[str, Any]:
        """
        Check for version information.

        Args:
            source_code: Combined source code
            agent_path: Path to agent directory

        Returns:
            Dictionary with version check results
        """
        result = {
            "has_version": False,
            "version": None,
        }

        # Check in source code
        version_pattern = re.compile(
            r"(?:__version__|VERSION)\s*=\s*['\"]([^'\"]+)['\"]",
            re.IGNORECASE,
        )

        match = version_pattern.search(source_code)
        if match:
            result["has_version"] = True
            result["version"] = match.group(1)

        # Check in pack.yaml
        pack_file = agent_path / "pack.yaml"
        if pack_file.exists():
            try:
                import yaml

                with open(pack_file, "r", encoding="utf-8") as f:
                    pack_spec = yaml.safe_load(f)

                version = pack_spec.get("pack", {}).get("version")
                if version:
                    result["has_version"] = True
                    result["version"] = version

            except Exception:
                pass

        return result

    def _check_graceful_shutdown(self, source_code: str) -> Dict[str, Any]:
        """
        Check for graceful shutdown handling.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with shutdown check results
        """
        result = {
            "has_shutdown": False,
            "shutdown_patterns": [],
        }

        patterns = [
            (r"signal\.signal|signal\.SIGTERM", "signal handling"),
            (r"atexit\.register", "atexit handler"),
            (r"KeyboardInterrupt", "keyboard interrupt handling"),
            (r"__del__|cleanup|shutdown", "cleanup method"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, source_code):
                result["has_shutdown"] = True
                result["shutdown_patterns"].append(description)

        return result

    def _check_retry_logic(self, source_code: str) -> Dict[str, Any]:
        """
        Check for retry logic.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with retry check results
        """
        result = {
            "has_retry": False,
            "retry_patterns": [],
        }

        patterns = [
            (r"@retry|@backoff", "retry decorator"),
            (r"tenacity|backoff|retrying", "retry library"),
            (r"for\s+\w+\s+in\s+range.*try", "manual retry loop"),
            (r"max_retries|retry_count", "retry configuration"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_retry"] = True
                result["retry_patterns"].append(description)

        return result

    def _check_metrics(self, source_code: str) -> Dict[str, Any]:
        """
        Check for metrics/monitoring.

        Args:
            source_code: Combined source code

        Returns:
            Dictionary with metrics check results
        """
        result = {
            "has_metrics": False,
            "metrics_patterns": [],
        }

        patterns = [
            (r"prometheus|Counter|Gauge|Histogram", "Prometheus metrics"),
            (r"statsd|datadog", "StatsD/Datadog"),
            (r"opentelemetry|otel", "OpenTelemetry"),
            (r"execution_time|processing_time", "timing metrics"),
        ]

        for pattern, description in patterns:
            if re.search(pattern, source_code, re.IGNORECASE):
                result["has_metrics"] = True
                result["metrics_patterns"].append(description)

        return result

    def _check_deployment_config(self, agent_path: Path) -> Dict[str, Any]:
        """
        Check for deployment configuration.

        Args:
            agent_path: Path to agent directory

        Returns:
            Dictionary with deployment config check results
        """
        result = {
            "has_config": False,
            "config_files": [],
        }

        deployment_files = [
            "Dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            "kubernetes.yaml",
            "k8s.yaml",
            "deployment.yaml",
            ".github/workflows",
            "Makefile",
        ]

        for filename in deployment_files:
            file_path = agent_path / filename
            if file_path.exists():
                result["has_config"] = True
                result["config_files"].append(filename)

        return result

    def _get_check_remediation(self, check: CheckResult) -> Optional[str]:
        """Get remediation for failed checks."""
        remediation_map = {
            "python_files_exist": (
                "Create Python files for the agent implementation."
            ),
            "logging_configured": (
                "Add logging configuration:\n"
                "  import logging\n"
                "  logger = logging.getLogger(__name__)\n"
                "\n"
                "  # In methods:\n"
                "  logger.info('Processing...', extra={'input': input})"
            ),
            "structured_logging": (
                "Use structured logging:\n"
                "  import structlog\n"
                "  logger = structlog.get_logger()\n"
                "  logger.info('event', key='value')"
            ),
            "error_handling": (
                "Add error handling:\n"
                "  try:\n"
                "      result = calculate(input)\n"
                "  except ValueError as e:\n"
                "      logger.error('Validation failed', error=str(e))\n"
                "      raise\n"
                "  except Exception as e:\n"
                "      logger.exception('Unexpected error')\n"
                "      raise"
            ),
            "health_check": (
                "Add health check:\n"
                "  def health_check(self) -> dict:\n"
                "      return {\n"
                "          'status': 'healthy',\n"
                "          'version': self.VERSION,\n"
                "          'timestamp': datetime.utcnow().isoformat()\n"
                "      }"
            ),
            "configuration_management": (
                "Add configuration management:\n"
                "  from pydantic_settings import BaseSettings\n"
                "\n"
                "  class Settings(BaseSettings):\n"
                "      api_key: str\n"
                "      debug: bool = False\n"
                "\n"
                "      class Config:\n"
                "          env_file = '.env'"
            ),
            "version_info": (
                "Add version information:\n"
                "  __version__ = '1.0.0'\n"
                "  VERSION = '1.0.0'\n"
                "\n"
                "Or in pack.yaml:\n"
                "  pack:\n"
                "    version: '1.0.0'"
            ),
            "graceful_shutdown": (
                "Add graceful shutdown:\n"
                "  import atexit\n"
                "\n"
                "  def cleanup():\n"
                "      logger.info('Shutting down...')\n"
                "\n"
                "  atexit.register(cleanup)"
            ),
            "deployment_config": (
                "Add deployment configuration:\n"
                "  - Create Dockerfile\n"
                "  - Create docker-compose.yml\n"
                "  - Add CI/CD workflow in .github/workflows/"
            ),
        }

        return remediation_map.get(check.name)
