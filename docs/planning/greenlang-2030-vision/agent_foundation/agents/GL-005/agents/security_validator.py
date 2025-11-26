# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Security Validator

Startup security validation to prevent deployment with insecure configurations.
Implements security checks per IEC 62443-4-2 and OWASP guidelines.
"""

import logging
import sys
import re
from typing import List, Tuple
from urllib.parse import urlparse

from .config import settings

logger = logging.getLogger(__name__)


class SecurityValidationError(Exception):
    """Raised when security validation fails"""
    pass


class SecurityValidator:
    """
    Validates security configuration at startup

    Per IEC 62443-4-2 requirements:
    - SR 1.1: User identification and authentication
    - SR 1.5: Authenticator management
    - SR 2.1: Authorization enforcement
    """

    @staticmethod
    def validate_jwt_secret() -> Tuple[bool, str]:
        """
        Validate JWT secret meets security requirements

        Requirements:
        - Minimum length: 32 characters
        - No weak/default values
        - Sufficient entropy
        """
        secret = settings.JWT_SECRET

        # Length check
        if len(secret) < 32:
            return False, f"JWT_SECRET too short: {len(secret)} chars (minimum: 32)"

        # Recommended length for production
        if settings.is_production() and len(secret) < 48:
            return False, f"JWT_SECRET too short for production: {len(secret)} chars (minimum: 48)"

        # Check for weak patterns
        weak_patterns = [
            "change",
            "placeholder",
            "example",
            "test",
            "dev",
            "secret",
            "password",
            "your-",
            "my-",
            "default",
        ]

        secret_lower = secret.lower()
        for pattern in weak_patterns:
            if pattern in secret_lower:
                return False, f"JWT_SECRET contains weak pattern: '{pattern}'"

        # Check for sufficient entropy (simple heuristic)
        unique_chars = len(set(secret))
        if unique_chars < 16:
            return False, f"JWT_SECRET has low entropy: only {unique_chars} unique characters"

        # Check it's not all same character type
        has_digit = any(c.isdigit() for c in secret)
        has_alpha = any(c.isalpha() for c in secret)
        has_special = any(not c.isalnum() for c in secret)

        char_type_count = sum([has_digit, has_alpha, has_special])
        if char_type_count < 2:
            return False, "JWT_SECRET should contain multiple character types (letters, digits, symbols)"

        return True, "JWT_SECRET is secure"

    @staticmethod
    def validate_database_url() -> Tuple[bool, str]:
        """
        Validate database URL doesn't contain weak credentials

        Checks:
        - No default usernames/passwords
        - No embedded credentials in version control
        - Proper connection string format
        """
        db_url = settings.DATABASE_URL

        # Parse URL
        try:
            parsed = urlparse(db_url)
        except Exception as e:
            return False, f"Invalid DATABASE_URL format: {e}"

        # Check for weak credentials
        weak_credentials = [
            ("user", "password"),
            ("user", "pass"),
            ("admin", "admin"),
            ("postgres", "postgres"),
            ("root", "root"),
            ("test", "test"),
            ("guest", "guest"),
        ]

        username = parsed.username
        password = parsed.password

        if username and password:
            credential_pair = (username.lower(), password.lower())

            for weak_user, weak_pass in weak_credentials:
                if credential_pair == (weak_user, weak_pass):
                    return False, f"DATABASE_URL uses default credentials: {weak_user}:{weak_pass}"

                # Check if password is too simple
                if len(password) < 12:
                    return False, f"DATABASE_URL password too short: {len(password)} chars (minimum: 12)"

                # Check for username in password (common weak pattern)
                if username.lower() in password.lower():
                    return False, "DATABASE_URL password contains username"

        return True, "DATABASE_URL credentials are acceptable"

    @staticmethod
    def validate_production_settings() -> Tuple[bool, str]:
        """
        Validate production-specific security settings

        Production requirements:
        - Debug mode disabled
        - Proper environment setting
        - Security features enabled
        """
        if not settings.is_production():
            return True, "Not production environment, skipping production checks"

        errors = []

        # Debug must be off in production
        if settings.DEBUG:
            errors.append("DEBUG mode is enabled in production")

        # Safety interlocks must be enabled
        if not settings.SAFETY_INTERLOCKS_ENABLED:
            errors.append("SAFETY_INTERLOCKS_ENABLED is disabled in production")

        # Flame detection must be required
        if not settings.FLAME_DETECTION_REQUIRED:
            errors.append("FLAME_DETECTION_REQUIRED is disabled in production")

        # Emergency shutdown must be enabled
        if not settings.EMERGENCY_SHUTDOWN_ENABLED:
            errors.append("EMERGENCY_SHUTDOWN_ENABLED is disabled in production")

        if errors:
            return False, "Production validation failed: " + "; ".join(errors)

        return True, "Production settings are secure"

    @staticmethod
    def validate_control_parameters() -> Tuple[bool, str]:
        """
        Validate control parameters are within safe ranges

        Safety validation per IEC 62443:
        - Verify limits are sensible
        - Check for configuration errors
        """
        errors = []

        # Check temperature limits are sensible
        if settings.MAX_FLAME_TEMPERATURE_C < settings.MAX_FURNACE_TEMPERATURE_C:
            errors.append(
                f"MAX_FLAME_TEMPERATURE_C ({settings.MAX_FLAME_TEMPERATURE_C}) "
                f"< MAX_FURNACE_TEMPERATURE_C ({settings.MAX_FURNACE_TEMPERATURE_C})"
            )

        # Check heat output range is valid
        if settings.HEAT_OUTPUT_MIN_KW >= settings.HEAT_OUTPUT_MAX_KW:
            errors.append(
                f"HEAT_OUTPUT_MIN_KW ({settings.HEAT_OUTPUT_MIN_KW}) "
                f">= HEAT_OUTPUT_MAX_KW ({settings.HEAT_OUTPUT_MAX_KW})"
            )

        # Check target is within range
        if not (settings.HEAT_OUTPUT_MIN_KW <= settings.HEAT_OUTPUT_TARGET_KW <= settings.HEAT_OUTPUT_MAX_KW):
            errors.append(
                f"HEAT_OUTPUT_TARGET_KW ({settings.HEAT_OUTPUT_TARGET_KW}) "
                f"not in valid range [{settings.HEAT_OUTPUT_MIN_KW}, {settings.HEAT_OUTPUT_MAX_KW}]"
            )

        # Check control loop interval is reasonable for industrial control
        if settings.CONTROL_LOOP_INTERVAL_MS > 1000:
            errors.append(
                f"CONTROL_LOOP_INTERVAL_MS ({settings.CONTROL_LOOP_INTERVAL_MS}ms) "
                f"too slow for real-time control (should be <1000ms)"
            )

        if errors:
            return False, "Control parameter validation failed: " + "; ".join(errors)

        return True, "Control parameters are valid"

    @classmethod
    def validate_all(cls, fail_fast: bool = True) -> List[str]:
        """
        Run all security validations

        Args:
            fail_fast: If True, raise exception on first failure

        Returns:
            List of validation errors (empty if all passed)

        Raises:
            SecurityValidationError: If fail_fast=True and any validation fails
        """
        validations = [
            ("JWT Secret", cls.validate_jwt_secret),
            ("Database URL", cls.validate_database_url),
            ("Production Settings", cls.validate_production_settings),
            ("Control Parameters", cls.validate_control_parameters),
        ]

        errors = []

        logger.info("=" * 80)
        logger.info("SECURITY VALIDATION - GL-005 CombustionControlAgent")
        logger.info("=" * 80)

        for name, validator in validations:
            try:
                success, message = validator()

                if success:
                    logger.info(f"✓ {name}: {message}")
                else:
                    error_msg = f"✗ {name}: {message}"
                    logger.error(error_msg)
                    errors.append(error_msg)

                    if fail_fast:
                        raise SecurityValidationError(error_msg)

            except Exception as e:
                error_msg = f"✗ {name}: Validation error - {e}"
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


def validate_startup_security(fail_fast: bool = True) -> None:
    """
    Convenience function to run security validation at startup

    Usage in main.py lifespan:
        from security_validator import validate_startup_security

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Validate security before starting
            validate_startup_security(fail_fast=True)

            # ... rest of startup
    """
    errors = SecurityValidator.validate_all(fail_fast=fail_fast)

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
        validate_startup_security(fail_fast=True)
        print("\n✓ All security validations passed")
        sys.exit(0)
    except SecurityValidationError as e:
        print(f"\n✗ Security validation failed: {e}")
        sys.exit(1)
