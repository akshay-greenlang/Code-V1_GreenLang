"""
Environment Configuration Validator for GreenLang

This module validates environment variables to ensure:
1. All required variables are set
2. Variables have valid formats
3. No common misconfigurations exist
4. Helpful error messages guide fixes

Example:
    >>> python config/env_validator.py
    >>> python config/env_validator.py --env-file .env.production
"""

import os
import re
import sys
import json
import base64
import argparse
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "CRITICAL"  # Must be fixed
    WARNING = "WARNING"    # Should be fixed
    INFO = "INFO"         # Nice to have


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    required: bool
    level: ValidationLevel
    validator: Optional[callable] = None
    default: Optional[str] = None
    description: str = ""
    example: str = ""
    group: str = "General"


class EnvValidator:
    """Environment configuration validator."""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize validator.

        Args:
            env_file: Path to .env file (optional)
        """
        self.env_file = env_file
        self.env_vars: Dict[str, str] = {}
        self.errors: List[Tuple[ValidationLevel, str]] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.validated_count = 0
        self.total_count = 0

    def load_env(self) -> None:
        """Load environment variables from file or system."""
        if self.env_file:
            self._load_from_file(self.env_file)
        else:
            self.env_vars = dict(os.environ)

    def _load_from_file(self, filepath: str) -> None:
        """Load environment variables from .env file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Environment file not found: {filepath}")

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove inline comments
                        if '#' in value:
                            value = value.split('#')[0].strip()
                        self.env_vars[key.strip()] = value.strip()

    # Validation Functions
    @staticmethod
    def validate_url(value: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(value)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @staticmethod
    def validate_port(value: str) -> bool:
        """Validate port number."""
        try:
            port = int(value)
            return 1 <= port <= 65535
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_boolean(value: str) -> bool:
        """Validate boolean value."""
        return value.lower() in ['true', 'false', '1', '0', 'yes', 'no']

    @staticmethod
    def validate_email(value: str) -> bool:
        """Validate email address."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))

    @staticmethod
    def validate_jwt_secret(value: str) -> bool:
        """Validate JWT secret strength."""
        return len(value) >= 32

    @staticmethod
    def validate_hex_key(value: str, length: int = 32) -> bool:
        """Validate hexadecimal key."""
        pattern = f'^[a-fA-F0-9]{{{length * 2}}}$'
        return bool(re.match(pattern, value))

    @staticmethod
    def validate_base64(value: str) -> bool:
        """Validate base64 encoded string."""
        try:
            base64.b64decode(value)
            return True
        except Exception:
            return False

    @staticmethod
    def validate_aws_key(value: str) -> bool:
        """Validate AWS access key format."""
        return bool(re.match(r'^AKIA[0-9A-Z]{16}$', value))

    @staticmethod
    def validate_aws_secret(value: str) -> bool:
        """Validate AWS secret key format."""
        return len(value) == 40

    @staticmethod
    def validate_azure_guid(value: str) -> bool:
        """Validate Azure GUID format."""
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, value, re.IGNORECASE))

    @staticmethod
    def validate_cron(value: str) -> bool:
        """Validate cron expression."""
        parts = value.split()
        if len(parts) not in [5, 6]:
            return False
        # Basic validation - can be expanded
        return True

    @staticmethod
    def validate_log_level(value: str) -> bool:
        """Validate log level."""
        return value.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    @staticmethod
    def validate_environment(value: str) -> bool:
        """Validate environment name."""
        return value.lower() in ['development', 'staging', 'production', 'test']

    @staticmethod
    def validate_region(value: str) -> bool:
        """Validate cloud region."""
        # AWS regions pattern
        aws_pattern = r'^[a-z]{2}-[a-z]+-\d{1}$'
        # Azure regions pattern
        azure_pattern = r'^[a-z]+[a-z0-9]*$'
        # GCP regions pattern
        gcp_pattern = r'^[a-z]+-[a-z]+\d{1}$'

        return any([
            re.match(aws_pattern, value.lower()),
            re.match(azure_pattern, value.lower()),
            re.match(gcp_pattern, value.lower())
        ])

    def get_validation_rules(self) -> List[ValidationRule]:
        """Get all validation rules."""
        return [
            # Database Configuration
            ValidationRule(
                name="POSTGRES_HOST",
                required=True,
                level=ValidationLevel.CRITICAL,
                description="PostgreSQL host address",
                example="localhost or db.example.com",
                group="Database"
            ),
            ValidationRule(
                name="POSTGRES_PORT",
                required=True,
                level=ValidationLevel.CRITICAL,
                validator=self.validate_port,
                default="5432",
                description="PostgreSQL port",
                group="Database"
            ),
            ValidationRule(
                name="POSTGRES_DB",
                required=True,
                level=ValidationLevel.CRITICAL,
                description="PostgreSQL database name",
                example="greenlang_db",
                group="Database"
            ),
            ValidationRule(
                name="POSTGRES_USER",
                required=True,
                level=ValidationLevel.CRITICAL,
                description="PostgreSQL username",
                group="Database"
            ),
            ValidationRule(
                name="POSTGRES_PASSWORD",
                required=True,
                level=ValidationLevel.CRITICAL,
                description="PostgreSQL password",
                group="Database"
            ),

            # Redis Configuration
            ValidationRule(
                name="REDIS_HOST",
                required=True,
                level=ValidationLevel.CRITICAL,
                description="Redis host address",
                example="localhost or redis.example.com",
                group="Cache"
            ),
            ValidationRule(
                name="REDIS_PORT",
                required=True,
                level=ValidationLevel.CRITICAL,
                validator=self.validate_port,
                default="6379",
                description="Redis port",
                group="Cache"
            ),

            # Security Settings
            ValidationRule(
                name="JWT_SECRET_KEY",
                required=True,
                level=ValidationLevel.CRITICAL,
                validator=self.validate_jwt_secret,
                description="JWT secret key (minimum 32 characters)",
                example="Generate with: openssl rand -hex 32",
                group="Security"
            ),
            ValidationRule(
                name="ENCRYPTION_KEY",
                required=True,
                level=ValidationLevel.CRITICAL,
                validator=lambda v: self.validate_hex_key(v, 32),
                description="Encryption key (32 bytes hex)",
                example="Generate with: openssl rand -hex 32",
                group="Security"
            ),

            # External APIs (Optional but validated if present)
            ValidationRule(
                name="GLEIF_API_URL",
                required=False,
                level=ValidationLevel.INFO,
                validator=self.validate_url,
                default="https://api.gleif.org/api/v1",
                description="GLEIF API endpoint",
                group="External APIs"
            ),
            ValidationRule(
                name="OPENAI_API_KEY",
                required=False,
                level=ValidationLevel.INFO,
                validator=lambda v: v.startswith('sk-'),
                description="OpenAI API key",
                group="External APIs"
            ),

            # AWS Configuration (if using AWS)
            ValidationRule(
                name="AWS_ACCESS_KEY_ID",
                required=False,
                level=ValidationLevel.WARNING,
                validator=self.validate_aws_key,
                description="AWS access key ID",
                group="Cloud Providers"
            ),
            ValidationRule(
                name="AWS_SECRET_ACCESS_KEY",
                required=False,
                level=ValidationLevel.WARNING,
                validator=self.validate_aws_secret,
                description="AWS secret access key",
                group="Cloud Providers"
            ),
            ValidationRule(
                name="AWS_REGION",
                required=False,
                level=ValidationLevel.WARNING,
                validator=self.validate_region,
                description="AWS region",
                example="us-east-1",
                group="Cloud Providers"
            ),

            # Azure Configuration (if using Azure)
            ValidationRule(
                name="AZURE_TENANT_ID",
                required=False,
                level=ValidationLevel.WARNING,
                validator=self.validate_azure_guid,
                description="Azure tenant ID",
                group="Cloud Providers"
            ),

            # Email Configuration
            ValidationRule(
                name="SMTP_HOST",
                required=False,
                level=ValidationLevel.WARNING,
                description="SMTP server host",
                example="smtp.gmail.com",
                group="Email"
            ),
            ValidationRule(
                name="EMAIL_FROM_ADDRESS",
                required=False,
                level=ValidationLevel.WARNING,
                validator=self.validate_email,
                description="From email address",
                group="Email"
            ),

            # Feature Flags
            ValidationRule(
                name="FEATURE_ZERO_HALLUCINATION",
                required=False,
                level=ValidationLevel.INFO,
                validator=self.validate_boolean,
                default="true",
                description="Enable zero-hallucination mode",
                group="Features"
            ),
            ValidationRule(
                name="FEATURE_PROVENANCE_TRACKING",
                required=False,
                level=ValidationLevel.INFO,
                validator=self.validate_boolean,
                default="true",
                description="Enable provenance tracking",
                group="Features"
            ),

            # Logging Configuration
            ValidationRule(
                name="LOG_LEVEL",
                required=False,
                level=ValidationLevel.INFO,
                validator=self.validate_log_level,
                default="INFO",
                description="Log level",
                group="Logging"
            ),

            # Environment Settings
            ValidationRule(
                name="ENVIRONMENT",
                required=True,
                level=ValidationLevel.CRITICAL,
                validator=self.validate_environment,
                description="Environment name",
                example="development, staging, production",
                group="Environment"
            ),
            ValidationRule(
                name="DEBUG",
                required=False,
                level=ValidationLevel.WARNING,
                validator=self.validate_boolean,
                default="false",
                description="Debug mode",
                group="Environment"
            ),

            # Performance Settings
            ValidationRule(
                name="APP_WORKERS",
                required=False,
                level=ValidationLevel.INFO,
                validator=lambda v: v.isdigit() and int(v) > 0,
                default="4",
                description="Number of worker processes",
                group="Performance"
            ),
            ValidationRule(
                name="BATCH_SIZE_DEFAULT",
                required=False,
                level=ValidationLevel.INFO,
                validator=lambda v: v.isdigit() and int(v) > 0,
                default="1000",
                description="Default batch size",
                group="Performance"
            ),

            # Backup Configuration
            ValidationRule(
                name="BACKUP_SCHEDULE",
                required=False,
                level=ValidationLevel.INFO,
                validator=self.validate_cron,
                description="Backup schedule (cron expression)",
                example="0 2 * * *",
                group="Backup"
            ),
        ]

    def validate_rule(self, rule: ValidationRule) -> None:
        """Validate a single rule."""
        self.total_count += 1
        value = self.env_vars.get(rule.name)

        # Check if required variable is missing
        if rule.required and not value:
            if rule.default:
                self.info.append(f"{rule.name}: Using default value '{rule.default}'")
                self.validated_count += 1
            else:
                error_msg = f"{rule.name}: Required variable not set"
                if rule.description:
                    error_msg += f" - {rule.description}"
                if rule.example:
                    error_msg += f" (Example: {rule.example})"
                self.errors.append((rule.level, error_msg))
            return

        # Skip validation if not set and not required
        if not value and not rule.required:
            self.validated_count += 1
            return

        # Run validator if provided
        if value and rule.validator:
            try:
                if not rule.validator(value):
                    error_msg = f"{rule.name}: Invalid value '{value}'"
                    if rule.description:
                        error_msg += f" - {rule.description}"
                    if rule.example:
                        error_msg += f" (Example: {rule.example})"

                    if rule.level == ValidationLevel.CRITICAL:
                        self.errors.append((rule.level, error_msg))
                    else:
                        self.warnings.append(error_msg)
                else:
                    self.validated_count += 1
            except Exception as e:
                error_msg = f"{rule.name}: Validation error - {str(e)}"
                self.errors.append((ValidationLevel.WARNING, error_msg))
        else:
            self.validated_count += 1

    def check_common_misconfigurations(self) -> None:
        """Check for common misconfigurations."""

        # Check for production with debug enabled
        if self.env_vars.get('ENVIRONMENT') == 'production':
            if self.env_vars.get('DEBUG', '').lower() == 'true':
                self.warnings.append(
                    "DEBUG is enabled in production environment - this is a security risk!"
                )

            # Check for default/weak secrets
            jwt_secret = self.env_vars.get('JWT_SECRET_KEY', '')
            if 'CHANGE_THIS' in jwt_secret or 'changeme' in jwt_secret.lower():
                self.errors.append((
                    ValidationLevel.CRITICAL,
                    "JWT_SECRET_KEY contains default value - MUST be changed for production!"
                ))

        # Check for conflicting settings
        if self.env_vars.get('REDIS_SSL', '').lower() == 'true':
            redis_host = self.env_vars.get('REDIS_HOST', '')
            if redis_host in ['localhost', '127.0.0.1']:
                self.warnings.append(
                    "REDIS_SSL is enabled but REDIS_HOST is localhost - this is unusual"
                )

        # Check for missing related variables
        if self.env_vars.get('SMTP_HOST'):
            required_smtp = ['SMTP_PORT', 'SMTP_USER', 'SMTP_PASSWORD']
            missing_smtp = [v for v in required_smtp if not self.env_vars.get(v)]
            if missing_smtp:
                self.warnings.append(
                    f"SMTP_HOST is set but missing: {', '.join(missing_smtp)}"
                )

        # Check for insecure settings
        if self.env_vars.get('SESSION_COOKIE_SECURE', '').lower() == 'false':
            if self.env_vars.get('ENVIRONMENT') == 'production':
                self.warnings.append(
                    "SESSION_COOKIE_SECURE is false in production - cookies will be sent over HTTP"
                )

        # Check database SSL
        ssl_mode = self.env_vars.get('POSTGRES_SSL_MODE', '')
        if self.env_vars.get('ENVIRONMENT') == 'production' and ssl_mode == 'disable':
            self.warnings.append(
                "POSTGRES_SSL_MODE is 'disable' in production - database connection is not encrypted"
            )

    def validate_dependencies(self) -> None:
        """Validate dependent variables."""

        # If using AWS KMS, check for AWS credentials
        if self.env_vars.get('AWS_KMS_KEY_ID'):
            if not self.env_vars.get('AWS_ACCESS_KEY_ID'):
                self.errors.append((
                    ValidationLevel.CRITICAL,
                    "AWS_KMS_KEY_ID is set but AWS_ACCESS_KEY_ID is missing"
                ))

        # If using feature flags, check for required dependencies
        if self.env_vars.get('FEATURE_LLM_CLASSIFICATION', '').lower() == 'true':
            has_llm = any([
                self.env_vars.get('OPENAI_API_KEY'),
                self.env_vars.get('ANTHROPIC_API_KEY'),
                self.env_vars.get('AZURE_OPENAI_API_KEY')
            ])
            if not has_llm:
                self.warnings.append(
                    "FEATURE_LLM_CLASSIFICATION is enabled but no LLM API key is configured"
                )

    def generate_report(self) -> str:
        """Generate validation report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("GREENLANG ENVIRONMENT CONFIGURATION VALIDATION REPORT")
        report.append("=" * 80)

        if self.env_file:
            report.append(f"Environment File: {self.env_file}")
        else:
            report.append("Environment: System Environment Variables")

        report.append(f"\nValidation Summary:")
        report.append(f"  Total Variables: {self.total_count}")
        report.append(f"  Validated: {self.validated_count}")
        report.append(f"  Errors: {len(self.errors)}")
        report.append(f"  Warnings: {len(self.warnings)}")
        report.append(f"  Info: {len(self.info)}")

        # Critical Errors
        critical_errors = [e for level, e in self.errors if level == ValidationLevel.CRITICAL]
        if critical_errors:
            report.append(f"\n[X] CRITICAL ERRORS (Must Fix):")
            for error in critical_errors:
                report.append(f"  - {error}")

        # Warnings
        if self.warnings:
            report.append(f"\n[!] WARNINGS (Should Fix):")
            for warning in self.warnings:
                report.append(f"  - {warning}")

        # Info
        if self.info:
            report.append(f"\n[i] INFO:")
            for info in self.info:
                report.append(f"  - {info}")

        # Status
        report.append("\n" + "-" * 80)
        if critical_errors:
            report.append(f"[X] VALIDATION FAILED - Fix critical errors before deployment")
            report.append("\nNext Steps:")
            report.append("1. Copy .env.example to .env")
            report.append("2. Fill in all required values")
            report.append("3. Run validation again: python config/env_validator.py")
        elif self.warnings:
            report.append(f"[!] VALIDATION PASSED WITH WARNINGS")
            report.append("Consider addressing warnings for production deployment")
        else:
            report.append(f"[OK] VALIDATION PASSED")
            report.append("Environment configuration is valid!")

        report.append("=" * 80 + "\n")

        return "\n".join(report)

    def run(self) -> bool:
        """
        Run validation.

        Returns:
            True if validation passed (no critical errors)
        """
        try:
            # Load environment variables
            self.load_env()

            # Run validation rules
            rules = self.get_validation_rules()
            for rule in rules:
                self.validate_rule(rule)

            # Check common misconfigurations
            self.check_common_misconfigurations()

            # Check dependencies
            self.validate_dependencies()

            # Generate and print report
            report = self.generate_report()
            print(report)

            # Return success if no critical errors
            critical_errors = [e for level, e in self.errors if level == ValidationLevel.CRITICAL]
            return len(critical_errors) == 0

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GreenLang environment configuration"
    )
    parser.add_argument(
        '--env-file',
        type=str,
        help='Path to .env file (default: use system environment)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )

    args = parser.parse_args()

    # Run validation
    validator = EnvValidator(env_file=args.env_file)

    # Handle JSON output if requested
    if args.json:
        # Suppress normal output for JSON mode
        import io
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            success = validator.run()
    else:
        success = validator.run()

    # Handle JSON output if requested
    if args.json:
        result = {
            'success': success,
            'total_vars': validator.total_count,
            'validated': validator.validated_count,
            'errors': [{'level': level.value, 'message': msg} for level, msg in validator.errors],
            'warnings': validator.warnings,
            'info': validator.info
        }
        print(json.dumps(result, indent=2))

    # Exit with appropriate code
    if not success or (args.strict and validator.warnings):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()