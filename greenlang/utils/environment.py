# -*- coding: utf-8 -*-
"""
Environment and Production Safety Module
=========================================

Provides environment detection and production safety guards for GreenLang.
Ensures proper configuration and security in different deployment environments.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvironmentDetector:
    """
    Detects and validates the current runtime environment
    """

    PRODUCTION_ENVS = ["production", "prod", "staging"]
    DEVELOPMENT_ENVS = ["development", "dev", "local", "test"]
    CI_ENVS = ["ci", "ci/cd", "github", "jenkins", "gitlab"]

    def __init__(self):
        """Initialize environment detector"""
        self.env = self._detect_environment()
        self.is_production = self._is_production()
        self.is_ci = self._is_ci()
        self.safety_checks = []

    def _detect_environment(self) -> str:
        """
        Detect the current environment from various sources

        Returns:
            Environment name (production, development, ci, etc.)
        """
        # Check GL_ENV first (GreenLang specific)
        env = os.getenv("GL_ENV", "").lower()
        if env:
            return env

        # Check standard environment variables
        env_vars = [
            "ENVIRONMENT",
            "ENV",
            "NODE_ENV",
            "PYTHON_ENV",
            "APP_ENV",
            "DEPLOYMENT_ENV"
        ]

        for var in env_vars:
            env = os.getenv(var, "").lower()
            if env:
                return env

        # Check CI environment markers
        ci_markers = [
            "CI",
            "CONTINUOUS_INTEGRATION",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "JENKINS_URL",
            "CIRCLECI",
            "TRAVIS"
        ]

        for marker in ci_markers:
            if os.getenv(marker):
                return "ci"

        # Check for production indicators
        if os.path.exists("/etc/production") or os.path.exists("/var/run/production"):
            return "production"

        # Check for containerization
        if os.path.exists("/.dockerenv") or os.getenv("KUBERNETES_SERVICE_HOST"):
            # In container, default to production unless specified
            return "production"

        # Default to development
        return "development"

    def _is_production(self) -> bool:
        """Check if running in production environment"""
        return self.env in self.PRODUCTION_ENVS

    def _is_ci(self) -> bool:
        """Check if running in CI environment"""
        return self.env in self.CI_ENVS or bool(os.getenv("CI"))

    def require_production_config(self) -> bool:
        """
        Validate that production configuration is present

        Returns:
            True if valid, raises exception otherwise
        """
        if not self.is_production:
            return True

        required_configs = [
            "GL_SIGNING_KEY",
            "GL_VERIFY_SIGNATURES",
            "GL_TELEMETRY_ENDPOINT",
            "GL_LOG_LEVEL"
        ]

        missing = []
        for config in required_configs:
            if not os.getenv(config):
                missing.append(config)

        if missing:
            raise EnvironmentError(
                f"Production environment detected but required configs missing: {', '.join(missing)}"
            )

        return True

    def get_safety_report(self) -> Dict[str, Any]:
        """
        Generate a safety report for the current environment

        Returns:
            Dictionary with safety check results
        """
        report = {
            "environment": self.env,
            "is_production": self.is_production,
            "is_ci": self.is_ci,
            "checks": []
        }

        # Check signature verification
        sig_verify = os.getenv("GL_VERIFY_SIGNATURES", "true" if self.is_production else "false")
        report["checks"].append({
            "name": "signature_verification",
            "enabled": sig_verify.lower() == "true",
            "required_in_prod": True,
            "status": "pass" if sig_verify.lower() == "true" or not self.is_production else "fail"
        })

        # Check dev mode
        dev_mode = os.getenv("GREENLANG_DEV_MODE", "false")
        report["checks"].append({
            "name": "dev_mode_disabled",
            "enabled": dev_mode.lower() != "true",
            "required_in_prod": True,
            "status": "pass" if dev_mode.lower() != "true" or not self.is_production else "fail"
        })

        # Check sandbox
        sandbox_enabled = os.getenv("GL_SANDBOX_ENABLED", "true" if self.is_production else "false")
        report["checks"].append({
            "name": "sandbox_enabled",
            "enabled": sandbox_enabled.lower() == "true",
            "required_in_prod": True,
            "status": "pass" if sandbox_enabled.lower() == "true" or not self.is_production else "warn"
        })

        # Check telemetry
        telemetry = os.getenv("GL_TELEMETRY_ENABLED", "true" if self.is_production else "false")
        report["checks"].append({
            "name": "telemetry_enabled",
            "enabled": telemetry.lower() == "true",
            "required_in_prod": False,
            "status": "pass" if telemetry.lower() == "true" or not self.is_production else "info"
        })

        # Check debug mode
        debug = os.getenv("GL_DEBUG", "false")
        report["checks"].append({
            "name": "debug_disabled",
            "enabled": debug.lower() != "true",
            "required_in_prod": True,
            "status": "pass" if debug.lower() != "true" or not self.is_production else "fail"
        })

        # Count failures
        failures = [c for c in report["checks"] if c["status"] == "fail"]
        warnings = [c for c in report["checks"] if c["status"] == "warn"]

        report["summary"] = {
            "total_checks": len(report["checks"]),
            "passed": len([c for c in report["checks"] if c["status"] == "pass"]),
            "failed": len(failures),
            "warnings": len(warnings),
            "safe_for_production": len(failures) == 0
        }

        return report


class ProductionGuard:
    """
    Guards against unsafe operations in production
    """

    def __init__(self):
        """Initialize production guard"""
        self.detector = EnvironmentDetector()
        self.blocked_operations = []

    def check_operation(self, operation: str, details: Dict[str, Any] = None) -> bool:
        """
        Check if an operation is allowed in current environment

        Args:
            operation: Operation name
            details: Operation details

        Returns:
            True if allowed, False otherwise
        """
        if not self.detector.is_production:
            return True

        # Operations that are NEVER allowed in production
        forbidden_ops = [
            "delete_all_data",
            "reset_database",
            "disable_authentication",
            "bypass_signature_verification",
            "enable_debug_mode",
            "expose_secrets",
            "disable_rate_limiting"
        ]

        if operation in forbidden_ops:
            logger.error(f"Operation '{operation}' blocked in production environment")
            self.blocked_operations.append({
                "operation": operation,
                "details": details,
                "environment": self.detector.env
            })
            return False

        # Operations that require additional checks
        restricted_ops = {
            "modify_configuration": self._check_config_modification,
            "execute_shell_command": self._check_shell_execution,
            "access_filesystem": self._check_filesystem_access,
            "network_request": self._check_network_request
        }

        if operation in restricted_ops:
            checker = restricted_ops[operation]
            allowed = checker(details or {})
            if not allowed:
                logger.warning(f"Operation '{operation}' restricted in production: {details}")
                self.blocked_operations.append({
                    "operation": operation,
                    "details": details,
                    "environment": self.detector.env
                })
            return allowed

        return True

    def _check_config_modification(self, details: Dict[str, Any]) -> bool:
        """Check if configuration modification is allowed"""
        # In production, only allow specific config changes
        allowed_configs = [
            "log_level",
            "telemetry_sampling_rate",
            "cache_ttl"
        ]

        config_key = details.get("key", "")
        return config_key.lower() in allowed_configs

    def _check_shell_execution(self, details: Dict[str, Any]) -> bool:
        """Check if shell command execution is allowed"""
        # In production, only allow whitelisted commands
        allowed_commands = [
            "ps", "top", "df", "du", "free",
            "netstat", "ss", "ip", "hostname",
            "date", "uptime", "whoami"
        ]

        command = details.get("command", "").split()[0]
        return command in allowed_commands

    def _check_filesystem_access(self, details: Dict[str, Any]) -> bool:
        """Check if filesystem access is allowed"""
        # In production, restrict to specific paths
        path = details.get("path", "")
        allowed_prefixes = [
            "/tmp/",
            "/var/log/greenlang/",
            "/opt/greenlang/data/"
        ]

        return any(path.startswith(prefix) for prefix in allowed_prefixes)

    def _check_network_request(self, details: Dict[str, Any]) -> bool:
        """Check if network request is allowed"""
        # In production, validate against allowlist
        url = details.get("url", "")
        allowed_domains = [
            "api.greenlang.ai",
            "telemetry.greenlang.ai",
            "hub.greenlang.ai",
            "pypi.org",
            "registry.npmjs.org"
        ]

        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc in allowed_domains

    def enforce_production_policies(self) -> None:
        """
        Enforce all production policies

        Raises:
            EnvironmentError: If production policies are violated
        """
        if not self.detector.is_production:
            return

        # Check for dev mode
        if os.getenv("GREENLANG_DEV_MODE", "false").lower() == "true":
            raise EnvironmentError(
                "Dev mode is not allowed in production environment"
            )

        # Check for debug mode
        if os.getenv("GL_DEBUG", "false").lower() == "true":
            raise EnvironmentError(
                "Debug mode is not allowed in production environment"
            )

        # Check signature verification
        if os.getenv("GL_VERIFY_SIGNATURES", "true").lower() != "true":
            raise EnvironmentError(
                "Signature verification must be enabled in production"
            )

        # Check for insecure settings
        insecure_vars = [
            "GL_ALLOW_INSECURE",
            "GL_SKIP_TLS_VERIFY",
            "GL_DISABLE_AUTH"
        ]

        for var in insecure_vars:
            if os.getenv(var, "false").lower() == "true":
                raise EnvironmentError(
                    f"Insecure setting {var} is not allowed in production"
                )

        logger.info("Production policies enforced successfully")


def get_environment() -> str:
    """Get the current environment name"""
    detector = EnvironmentDetector()
    return detector.env


def is_production() -> bool:
    """Check if running in production"""
    detector = EnvironmentDetector()
    return detector.is_production


def is_safe_for_production() -> bool:
    """Check if current configuration is safe for production"""
    detector = EnvironmentDetector()
    report = detector.get_safety_report()
    return report["summary"]["safe_for_production"]


def enforce_production_safety():
    """Enforce production safety policies"""
    guard = ProductionGuard()
    guard.enforce_production_policies()


def check_operation_allowed(operation: str, details: Dict[str, Any] = None) -> bool:
    """Check if an operation is allowed in current environment"""
    guard = ProductionGuard()
    return guard.check_operation(operation, details)


# Auto-enforce in production when module is imported
if __name__ != "__main__":
    detector = EnvironmentDetector()
    if detector.is_production:
        logger.info(f"Production environment detected: {detector.env}")
        try:
            guard = ProductionGuard()
            guard.enforce_production_policies()
        except EnvironmentError as e:
            logger.critical(f"Production safety violation: {e}")
            # In production, fail fast on safety violations
            sys.exit(1)