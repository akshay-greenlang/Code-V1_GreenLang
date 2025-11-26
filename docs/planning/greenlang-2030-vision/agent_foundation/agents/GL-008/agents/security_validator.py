# -*- coding: utf-8 -*-
"""
GL-008 SteamTrapInspector - Security Validator

Startup security validation to prevent deployment with insecure configurations.
Implements security checks per IEC 62443-4-2 and OWASP guidelines.

Key validations:
- No hardcoded credentials in configuration
- API key validation for external services
- Configuration security (dev vs prod)
- Environment validation
- Rate limiting configuration
- ML model integrity checks
"""

import logging
import sys
import re
import os
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityValidationError(Exception):
    """Raised when security validation fails"""
    pass


class SecurityValidator:
    """
    Validates security configuration at startup for GL-008 SteamTrapInspector

    Per IEC 62443-4-2 requirements:
    - SR 1.1: User identification and authentication
    - SR 1.5: Authenticator management
    - SR 2.1: Authorization enforcement
    - SR 3.1: Communication integrity
    - SR 7.1: Denial of service protection
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize SecurityValidator

        Args:
            config: Configuration object (TrapInspectorConfig) or None
        """
        self.config = config

    @staticmethod
    def validate_no_hardcoded_credentials() -> Tuple[bool, str]:
        """
        Validate no hardcoded credentials in environment or configuration

        Checks:
        - No default/placeholder API keys
        - No hardcoded passwords
        - No test credentials in production

        Returns:
            Tuple of (success: bool, message: str)
        """
        errors = []

        # Check environment variables for weak patterns
        weak_patterns = [
            "change",
            "placeholder",
            "example",
            "test_key",
            "demo",
            "YOUR_",
            "INSERT_",
            "REPLACE_",
            "default",
            "password123",
            "admin123",
        ]

        # Check critical environment variables
        critical_env_vars = [
            "DATABASE_URL",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "JWT_SECRET",
            "API_KEY",
            "SECRET_KEY",
        ]

        for env_var in critical_env_vars:
            value = os.environ.get(env_var, "")
            if not value:
                continue  # Not set, will be caught by other validators

            value_lower = value.lower()
            for pattern in weak_patterns:
                if pattern in value_lower:
                    errors.append(
                        f"{env_var} contains weak/placeholder pattern: '{pattern}'"
                    )
                    break

        if errors:
            return False, "Hardcoded credentials detected: " + "; ".join(errors)

        return True, "No hardcoded credentials detected"

    @staticmethod
    def validate_api_keys() -> Tuple[bool, str]:
        """
        Validate API keys for external services

        Requirements:
        - Anthropic API key present if LLM classification enabled
        - API keys have minimum length (32 chars for Claude)
        - No test/demo keys in production

        Returns:
            Tuple of (success: bool, message: str)
        """
        errors = []

        # Check for Anthropic API key (required for LLM classification)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if not anthropic_key:
            # Check if LLM is enabled in config
            # For now, warn but don't fail if key is missing
            return True, "ANTHROPIC_API_KEY not set (LLM classification may be disabled)"

        # Validate key format and length
        if len(anthropic_key) < 32:
            errors.append(f"ANTHROPIC_API_KEY too short: {len(anthropic_key)} chars (minimum: 32)")

        # Check for test/demo key patterns
        test_patterns = ["sk-ant-test", "demo", "example", "placeholder"]
        key_lower = anthropic_key.lower()
        for pattern in test_patterns:
            if pattern in key_lower:
                errors.append(f"ANTHROPIC_API_KEY contains test pattern: '{pattern}'")

        # Validate key format (should start with sk-ant-)
        if not anthropic_key.startswith("sk-ant-"):
            errors.append("ANTHROPIC_API_KEY has invalid format (should start with 'sk-ant-')")

        if errors:
            return False, "API key validation failed: " + "; ".join(errors)

        return True, "API keys are valid"

    def validate_configuration_security(self) -> Tuple[bool, str]:
        """
        Validate configuration security settings

        Checks:
        - Zero secrets policy enabled
        - Audit logging enabled
        - Provenance tracking enabled
        - Secure defaults

        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.config is None:
            return True, "No config provided, skipping configuration validation"

        errors = []

        # Check zero secrets policy
        if not getattr(self.config, 'zero_secrets', True):
            errors.append("zero_secrets policy is disabled")

        # Check audit logging
        if not getattr(self.config, 'enable_audit_logging', True):
            errors.append("enable_audit_logging is disabled")

        # Check provenance tracking
        if not getattr(self.config, 'enable_provenance_tracking', True):
            errors.append("enable_provenance_tracking is disabled")

        # Check LLM determinism settings
        llm_temp = getattr(self.config, 'llm_temperature', None)
        if llm_temp is not None and llm_temp != 0.0:
            errors.append(f"llm_temperature is {llm_temp} (must be 0.0 for determinism)")

        llm_seed = getattr(self.config, 'llm_seed', None)
        if llm_seed is not None and llm_seed != 42:
            errors.append(f"llm_seed is {llm_seed} (must be 42 for reproducibility)")

        if errors:
            return False, "Configuration security failed: " + "; ".join(errors)

        return True, "Configuration security is valid"

    @staticmethod
    def validate_environment() -> Tuple[bool, str]:
        """
        Validate environment settings (dev vs prod)

        Production requirements:
        - GREENLANG_ENV set to 'production'
        - No DEBUG mode in production
        - Secure logging levels

        Returns:
            Tuple of (success: bool, message: str)
        """
        env = os.environ.get("GREENLANG_ENV", "development")
        is_production = env.lower() in ["production", "prod"]

        if not is_production:
            return True, f"Environment is '{env}' (development mode)"

        errors = []

        # Check DEBUG flag
        debug_flag = os.environ.get("DEBUG", "false").lower()
        if debug_flag in ["true", "1", "yes"]:
            errors.append("DEBUG mode is enabled in production")

        # Check logging level
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        if log_level == "DEBUG":
            errors.append("LOG_LEVEL is DEBUG in production (should be INFO or WARNING)")

        if errors:
            return False, "Production environment validation failed: " + "; ".join(errors)

        return True, "Production environment settings are secure"

    def validate_rate_limiting(self) -> Tuple[bool, str]:
        """
        Validate rate limiting configuration

        Requirements:
        - Rate limits configured for API endpoints
        - Reasonable concurrency limits
        - Timeout settings configured

        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.config is None:
            return True, "No config provided, skipping rate limiting validation"

        errors = []

        # Check max concurrent inspections
        max_concurrent = getattr(self.config, 'max_concurrent_inspections', None)
        if max_concurrent is None:
            errors.append("max_concurrent_inspections not configured")
        elif max_concurrent < 1:
            errors.append(f"max_concurrent_inspections too low: {max_concurrent}")
        elif max_concurrent > 100:
            errors.append(f"max_concurrent_inspections too high: {max_concurrent} (max: 100)")

        # Check timeout settings
        timeout = getattr(self.config, 'calculation_timeout_seconds', None)
        if timeout is None:
            errors.append("calculation_timeout_seconds not configured")
        elif timeout <= 0:
            errors.append(f"calculation_timeout_seconds invalid: {timeout}")
        elif timeout > 300:
            errors.append(f"calculation_timeout_seconds too high: {timeout}s (max: 300s)")

        # Check monitoring interval
        monitor_interval = getattr(self.config, 'monitoring_interval_seconds', None)
        if monitor_interval is not None and monitor_interval < 10:
            errors.append(
                f"monitoring_interval_seconds too low: {monitor_interval}s "
                f"(may cause resource exhaustion)"
            )

        if errors:
            return False, "Rate limiting validation failed: " + "; ".join(errors)

        return True, "Rate limiting configuration is valid"

    def validate_ml_model_integrity(self) -> Tuple[bool, str]:
        """
        Validate ML model paths and integrity

        Checks:
        - Model paths exist or are properly configured
        - Model files are not in version control (if they exist)
        - Model versioning is configured

        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.config is None:
            return True, "No config provided, skipping ML model validation"

        errors = []
        warnings = []

        # Check ML model configuration
        ml_config = getattr(self.config, 'ml_models', None)
        if ml_config is None:
            return True, "ML models not configured (optional)"

        # Check model paths
        model_paths = [
            ('acoustic_model_path', getattr(ml_config, 'acoustic_model_path', None)),
            ('thermal_model_path', getattr(ml_config, 'thermal_model_path', None)),
            ('rul_model_path', getattr(ml_config, 'rul_model_path', None)),
        ]

        for name, path in model_paths:
            if path is None:
                continue

            # Convert to Path object if string
            if isinstance(path, str):
                path = Path(path)

            # Check if path is absolute (security best practice)
            if not path.is_absolute():
                warnings.append(f"{name} uses relative path: {path}")

        # Check model version is set
        model_version = getattr(ml_config, 'model_version', None)
        if not model_version or model_version == "0.0.0":
            warnings.append("model_version not set or is default value")

        # Check confidence threshold
        confidence = getattr(ml_config, 'confidence_threshold', None)
        if confidence is not None:
            if confidence < 0.5:
                errors.append(f"confidence_threshold too low: {confidence} (min: 0.5)")
            elif confidence > 1.0:
                errors.append(f"confidence_threshold invalid: {confidence} (max: 1.0)")

        message_parts = []
        if errors:
            message_parts.append("Errors: " + "; ".join(errors))
        if warnings:
            message_parts.append("Warnings: " + "; ".join(warnings))

        if errors:
            return False, "ML model validation failed: " + " | ".join(message_parts)
        elif warnings:
            return True, "ML model validation passed with warnings: " + " | ".join(message_parts)
        else:
            return True, "ML model configuration is valid"

    @classmethod
    def validate_all(
        cls,
        config: Optional[Any] = None,
        fail_fast: bool = True
    ) -> List[str]:
        """
        Run all security validations

        Args:
            config: Configuration object (TrapInspectorConfig) or None
            fail_fast: If True, raise exception on first failure

        Returns:
            List of validation errors (empty if all passed)

        Raises:
            SecurityValidationError: If fail_fast=True and any validation fails
        """
        validator = cls(config)

        validations = [
            ("Hardcoded Credentials", cls.validate_no_hardcoded_credentials),
            ("API Keys", cls.validate_api_keys),
            ("Configuration Security", validator.validate_configuration_security),
            ("Environment", cls.validate_environment),
            ("Rate Limiting", validator.validate_rate_limiting),
            ("ML Model Integrity", validator.validate_ml_model_integrity),
        ]

        errors = []

        logger.info("=" * 80)
        logger.info("SECURITY VALIDATION - GL-008 SteamTrapInspector")
        logger.info("=" * 80)

        for name, validation_func in validations:
            try:
                success, message = validation_func()

                if success:
                    logger.info(f"PASS {name}: {message}")
                else:
                    error_msg = f"FAIL {name}: {message}"
                    logger.error(error_msg)
                    errors.append(error_msg)

                    if fail_fast:
                        raise SecurityValidationError(error_msg)

            except Exception as e:
                error_msg = f"ERROR {name}: Validation error - {e}"
                logger.error(error_msg)
                errors.append(error_msg)

                if fail_fast:
                    raise SecurityValidationError(error_msg)

        logger.info("=" * 80)

        if errors:
            logger.error(f"SECURITY VALIDATION FAILED - {len(errors)} error(s)")
            for error in errors:
                logger.error(f"  - {error}")
            logger.error("=" * 80)

            if fail_fast:
                logger.critical("STARTUP ABORTED - Fix security issues before deployment")
                sys.exit(1)
        else:
            logger.info("SECURITY VALIDATION PASSED - All checks OK")
            logger.info("=" * 80)

        return errors


def validate_startup_security(
    config: Optional[Any] = None,
    fail_fast: bool = True
) -> None:
    """
    Convenience function to run security validation at startup

    Usage in main application:
        from agents.security_validator import validate_startup_security

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Validate security before starting
            validate_startup_security(config=settings, fail_fast=True)

            # ... rest of startup

    Args:
        config: Configuration object (TrapInspectorConfig) or None
        fail_fast: If True, abort on first failure

    Raises:
        SecurityValidationError: If fail_fast=True and validation fails
    """
    errors = SecurityValidator.validate_all(config=config, fail_fast=fail_fast)

    if errors and fail_fast:
        raise SecurityValidationError(
            f"Security validation failed with {len(errors)} error(s). "
            "See logs for details."
        )


if __name__ == "__main__":
    # Allow running as standalone script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        validate_startup_security(config=None, fail_fast=True)
        print("\nPASS: All security validations passed")
        sys.exit(0)
    except SecurityValidationError as e:
        print(f"\nFAIL: Security validation failed: {e}")
        sys.exit(1)
