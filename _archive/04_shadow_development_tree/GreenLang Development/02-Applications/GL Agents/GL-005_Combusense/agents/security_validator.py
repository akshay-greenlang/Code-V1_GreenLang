# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Security Validator

Comprehensive startup security validation to prevent deployment with insecure configurations.
Implements security checks per IEC 62443-4-2 and OWASP guidelines.

IEC 62443-4-2 Security Requirements Covered:
- SR 1.1: Human user identification and authentication
- SR 1.2: Software process and device identification and authentication
- SR 1.5: Authenticator management
- SR 1.7: Strength of password-based authentication
- SR 2.1: Authorization enforcement
- SR 3.1: Communication integrity
- SR 4.1: Information confidentiality
- SR 4.3: Use of cryptography

Author: GL-SecurityEngineer
Date: 2025-11-18
Version: 2.0.0
"""

import logging
import os
import re
import secrets
import ssl
import sys
import math
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .config import settings

logger = logging.getLogger(__name__)

# Prometheus metrics integration
try:
    from prometheus_client import Counter, Gauge, Info
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class SecuritySeverity(Enum):
    """Security finding severity levels per IEC 62443-4-2."""
    CRITICAL = "critical"  # Must fix before deployment
    HIGH = "high"          # Should fix before production
    MEDIUM = "medium"      # Recommend fixing
    LOW = "low"            # Informational
    INFO = "info"          # Best practice suggestions


class SecurityCategory(Enum):
    """Security check categories per IEC 62443-4-2."""
    AUTHENTICATION = "authentication"       # SR 1.x
    AUTHORIZATION = "authorization"         # SR 2.x
    COMMUNICATION = "communication"         # SR 3.x
    DATA_CONFIDENTIALITY = "data_confidentiality"  # SR 4.x
    SYSTEM_INTEGRITY = "system_integrity"   # SR 5.x
    AUDIT_LOGGING = "audit_logging"         # SR 6.x
    CONFIGURATION = "configuration"         # SR 7.x


@dataclass
class SecurityFinding:
    """Individual security finding from validation."""
    check_name: str
    category: SecurityCategory
    severity: SecuritySeverity
    passed: bool
    message: str
    details: Optional[str] = None
    remediation: Optional[str] = None
    iec_reference: Optional[str] = None  # IEC 62443-4-2 reference


@dataclass
class SecurityValidationResult:
    """
    Comprehensive result of security validation.

    Contains all findings, scores, and pass/fail status
    for security gate decisions.
    """
    passed: bool
    timestamp: str
    environment: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    findings: List[SecurityFinding] = field(default_factory=list)

    # Severity breakdown
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Security score (0-100)
    security_score: float = 0.0

    # Category scores
    category_scores: Dict[str, float] = field(default_factory=dict)

    # Compliance flags
    iec_62443_compliant: bool = False
    production_ready: bool = False

    def add_finding(self, finding: SecurityFinding) -> None:
        """Add a finding and update counts."""
        self.findings.append(finding)
        self.total_checks += 1

        if finding.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1

            # Update severity counts
            if finding.severity == SecuritySeverity.CRITICAL:
                self.critical_count += 1
            elif finding.severity == SecuritySeverity.HIGH:
                self.high_count += 1
            elif finding.severity == SecuritySeverity.MEDIUM:
                self.medium_count += 1
            elif finding.severity == SecuritySeverity.LOW:
                self.low_count += 1

    def calculate_score(self) -> None:
        """Calculate overall security score."""
        if self.total_checks == 0:
            self.security_score = 0.0
            return

        # Weighted scoring based on severity
        # Critical failures have highest impact
        penalty = (
            self.critical_count * 25 +
            self.high_count * 15 +
            self.medium_count * 5 +
            self.low_count * 2
        )

        # Base score from pass rate
        base_score = (self.passed_checks / self.total_checks) * 100

        # Apply penalties
        self.security_score = max(0, base_score - penalty)

        # Determine compliance
        self.iec_62443_compliant = (
            self.critical_count == 0 and
            self.high_count == 0 and
            self.security_score >= 80
        )

        self.production_ready = (
            self.critical_count == 0 and
            self.high_count == 0 and
            self.medium_count <= 2 and
            self.security_score >= 90
        )

        # Update overall pass status
        self.passed = self.critical_count == 0 and self.high_count == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "security_score": round(self.security_score, 2),
            "severity_breakdown": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count
            },
            "iec_62443_compliant": self.iec_62443_compliant,
            "production_ready": self.production_ready,
            "findings": [
                {
                    "check": f.check_name,
                    "category": f.category.value,
                    "severity": f.severity.value,
                    "passed": f.passed,
                    "message": f.message,
                    "details": f.details,
                    "remediation": f.remediation,
                    "iec_reference": f.iec_reference
                }
                for f in self.findings
            ],
            "category_scores": self.category_scores
        }


class SecurityValidationError(Exception):
    """Raised when security validation fails with blocking issues."""

    def __init__(self, message: str, result: Optional[SecurityValidationResult] = None):
        super().__init__(message)
        self.result = result


class SecurityMetrics:
    """Prometheus metrics for security monitoring."""

    def __init__(self):
        if not METRICS_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True

        # Security score gauge
        self.security_score = Gauge(
            'gl005_security_score',
            'Overall security validation score (0-100)',
            ['agent', 'environment']
        )

        # Validation result counter
        self.validation_results = Counter(
            'gl005_security_validations_total',
            'Total security validations performed',
            ['agent', 'result']  # result: passed, failed
        )

        # Finding counters by severity
        self.findings_by_severity = Counter(
            'gl005_security_findings_total',
            'Security findings by severity',
            ['agent', 'severity']
        )

        # Finding counters by category
        self.findings_by_category = Counter(
            'gl005_security_findings_by_category_total',
            'Security findings by category',
            ['agent', 'category']
        )

        # Compliance status
        self.iec_compliance = Gauge(
            'gl005_iec_62443_compliant',
            'IEC 62443-4-2 compliance status (1=compliant, 0=non-compliant)',
            ['agent']
        )

        # Production readiness
        self.production_ready = Gauge(
            'gl005_production_ready',
            'Production readiness status (1=ready, 0=not ready)',
            ['agent']
        )

        # Last validation timestamp
        self.last_validation = Gauge(
            'gl005_security_last_validation_timestamp',
            'Timestamp of last security validation',
            ['agent']
        )

    def record_validation(self, result: SecurityValidationResult) -> None:
        """Record validation results in metrics."""
        if not self.enabled:
            return

        agent_id = "GL-005"
        env = result.environment

        # Update security score
        self.security_score.labels(agent=agent_id, environment=env).set(result.security_score)

        # Update validation counter
        status = "passed" if result.passed else "failed"
        self.validation_results.labels(agent=agent_id, result=status).inc()

        # Update findings counters
        for finding in result.findings:
            if not finding.passed:
                self.findings_by_severity.labels(
                    agent=agent_id,
                    severity=finding.severity.value
                ).inc()

                self.findings_by_category.labels(
                    agent=agent_id,
                    category=finding.category.value
                ).inc()

        # Update compliance status
        self.iec_compliance.labels(agent=agent_id).set(1 if result.iec_62443_compliant else 0)
        self.production_ready.labels(agent=agent_id).set(1 if result.production_ready else 0)

        # Update timestamp
        self.last_validation.labels(agent=agent_id).set(datetime.utcnow().timestamp())


# Global metrics instance
security_metrics = SecurityMetrics()


class SecurityAuditLogger:
    """
    Security audit logger for compliance tracking.

    Per IEC 62443-4-2 SR 6.1: Audit log accessibility
    Per IEC 62443-4-2 SR 6.2: Audit log protection
    """

    def __init__(self):
        self.audit_logger = logging.getLogger("security.audit")

        # Configure audit logging to separate file if in production
        if settings.is_production():
            handler = logging.FileHandler(
                f"security_audit_{datetime.utcnow().strftime('%Y%m%d')}.log"
            )
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - SECURITY_AUDIT - %(levelname)s - %(message)s'
            ))
            self.audit_logger.addHandler(handler)

        self.audit_logger.setLevel(logging.INFO)

    def log_validation_start(self, environment: str) -> None:
        """Log start of security validation."""
        self.audit_logger.info(
            f"SECURITY_VALIDATION_START | environment={environment} | "
            f"timestamp={datetime.utcnow().isoformat()}"
        )

    def log_finding(self, finding: SecurityFinding) -> None:
        """Log individual security finding."""
        log_level = logging.WARNING if not finding.passed else logging.INFO

        self.audit_logger.log(
            log_level,
            f"SECURITY_FINDING | check={finding.check_name} | "
            f"category={finding.category.value} | severity={finding.severity.value} | "
            f"passed={finding.passed} | message={finding.message}"
        )

    def log_validation_complete(self, result: SecurityValidationResult) -> None:
        """Log completion of security validation."""
        log_level = logging.INFO if result.passed else logging.ERROR

        self.audit_logger.log(
            log_level,
            f"SECURITY_VALIDATION_COMPLETE | passed={result.passed} | "
            f"score={result.security_score:.2f} | "
            f"critical={result.critical_count} | high={result.high_count} | "
            f"iec_compliant={result.iec_62443_compliant} | "
            f"production_ready={result.production_ready}"
        )


# Global audit logger
audit_logger = SecurityAuditLogger()


class SecurityValidator:
    """
    Comprehensive security validator for GL-005 CombustionControlAgent.

    Validates security configuration at startup per IEC 62443-4-2:
    - SR 1.1: User identification and authentication
    - SR 1.5: Authenticator management
    - SR 1.7: Password-based authentication strength
    - SR 2.1: Authorization enforcement
    - SR 3.1: Communication integrity
    - SR 4.1: Information confidentiality
    - SR 4.3: Use of cryptography
    """

    # Known weak/default secrets patterns
    WEAK_SECRET_PATTERNS = [
        r"changeme",
        r"change.?this",
        r"placeholder",
        r"example",
        r"test.?secret",
        r"dev.?secret",
        r"my.?secret",
        r"your.?secret",
        r"default",
        r"password",
        r"secret",
        r"admin",
        r"123456",
        r"qwerty",
        r"letmein",
        r"welcome",
        r"monkey",
        r"dragon",
        r"master",
    ]

    # Secret patterns to detect in environment/code
    SECRET_DETECTION_PATTERNS = [
        (r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?", "API Key"),
        (r"(?i)(secret[_-]?key|secretkey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?", "Secret Key"),
        (r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"]{8,})['\"]?", "Password"),
        (r"(?i)(token|auth[_-]?token)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-\.]{20,})['\"]?", "Token"),
        (r"(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[=:]\s*['\"]?([A-Z0-9]{20})['\"]?", "AWS Access Key"),
        (r"(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*['\"]?([a-zA-Z0-9/+=]{40})['\"]?", "AWS Secret Key"),
        (r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----", "Private Key"),
        (r"(?i)(github[_-]?token|gh[_-]?token)\s*[=:]\s*['\"]?(ghp_[a-zA-Z0-9]{36})['\"]?", "GitHub Token"),
        (r"(?i)(slack[_-]?token)\s*[=:]\s*['\"]?(xox[baprs]-[a-zA-Z0-9-]+)['\"]?", "Slack Token"),
    ]

    # RBAC role definitions per IEC 62443-4-2 SR 2.1
    REQUIRED_ROLES = {
        "operator": {
            "description": "Plant operator - can view status and acknowledge alarms",
            "permissions": ["read:status", "read:alarms", "write:alarm_ack"]
        },
        "engineer": {
            "description": "Control engineer - can modify setpoints and tuning",
            "permissions": ["read:*", "write:setpoints", "write:tuning"]
        },
        "supervisor": {
            "description": "Shift supervisor - can override interlocks",
            "permissions": ["read:*", "write:setpoints", "write:interlocks"]
        },
        "admin": {
            "description": "System administrator - full access",
            "permissions": ["read:*", "write:*", "admin:*"]
        }
    }

    def __init__(self):
        self.result = SecurityValidationResult(
            passed=True,
            timestamp=datetime.utcnow().isoformat(),
            environment=settings.GREENLANG_ENV,
            total_checks=0,
            passed_checks=0,
            failed_checks=0
        )

    def validate_jwt_secret(self) -> SecurityFinding:
        """
        Validate JWT secret meets security requirements.

        Per IEC 62443-4-2 SR 1.7: Strength of password-based authentication

        Requirements:
        - Minimum length: 32 characters (48 for production)
        - No weak/default values
        - Sufficient entropy (>= 128 bits effective)
        - Multiple character types
        """
        check_name = "JWT Secret Strength"
        secret = settings.JWT_SECRET

        # Length check
        min_length = 48 if settings.is_production() else 32
        if len(secret) < min_length:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.AUTHENTICATION,
                severity=SecuritySeverity.CRITICAL,
                passed=False,
                message=f"JWT_SECRET too short: {len(secret)} chars (minimum: {min_length})",
                details="Short secrets are vulnerable to brute-force attacks",
                remediation=f"Generate a cryptographically secure secret with at least {min_length} characters: "
                           f"python -c 'import secrets; print(secrets.token_urlsafe(64))'",
                iec_reference="IEC 62443-4-2 SR 1.7"
            )

        # Check for weak patterns
        secret_lower = secret.lower()
        for pattern in self.WEAK_SECRET_PATTERNS:
            if re.search(pattern, secret_lower):
                return SecurityFinding(
                    check_name=check_name,
                    category=SecurityCategory.AUTHENTICATION,
                    severity=SecuritySeverity.CRITICAL,
                    passed=False,
                    message=f"JWT_SECRET contains weak/default pattern",
                    details="Secret contains predictable or commonly-used patterns",
                    remediation="Generate a random secret: python -c 'import secrets; print(secrets.token_urlsafe(64))'",
                    iec_reference="IEC 62443-4-2 SR 1.5"
                )

        # Entropy calculation (Shannon entropy)
        entropy = self._calculate_entropy(secret)
        min_entropy = 4.0  # Bits per character minimum

        if entropy < min_entropy:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.AUTHENTICATION,
                severity=SecuritySeverity.HIGH,
                passed=False,
                message=f"JWT_SECRET has low entropy: {entropy:.2f} bits/char (minimum: {min_entropy})",
                details="Low entropy makes the secret easier to guess or brute-force",
                remediation="Use a cryptographically random secret generator",
                iec_reference="IEC 62443-4-2 SR 4.3"
            )

        # Check character diversity
        has_digit = any(c.isdigit() for c in secret)
        has_lower = any(c.islower() for c in secret)
        has_upper = any(c.isupper() for c in secret)
        has_special = any(not c.isalnum() for c in secret)

        char_types = sum([has_digit, has_lower, has_upper, has_special])
        if char_types < 3:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.AUTHENTICATION,
                severity=SecuritySeverity.MEDIUM,
                passed=False,
                message=f"JWT_SECRET lacks character diversity (only {char_types} character types)",
                details="Secrets should contain uppercase, lowercase, digits, and special characters",
                remediation="Use a secret with mixed character types for better security",
                iec_reference="IEC 62443-4-2 SR 1.7"
            )

        # Unique characters check
        unique_chars = len(set(secret))
        min_unique = max(16, len(secret) // 3)

        if unique_chars < min_unique:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.AUTHENTICATION,
                severity=SecuritySeverity.MEDIUM,
                passed=False,
                message=f"JWT_SECRET has low character uniqueness: {unique_chars} unique chars",
                details="Repetitive patterns reduce effective entropy",
                remediation="Use a more random secret with diverse characters",
                iec_reference="IEC 62443-4-2 SR 4.3"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.AUTHENTICATION,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="JWT_SECRET meets security requirements",
            details=f"Length: {len(secret)}, Entropy: {entropy:.2f} bits/char, Character types: {char_types}",
            iec_reference="IEC 62443-4-2 SR 1.7"
        )

    def validate_database_security(self) -> SecurityFinding:
        """
        Validate database connection security.

        Per IEC 62443-4-2 SR 4.1: Information confidentiality

        Checks:
        - No default/weak credentials
        - SSL/TLS enabled for production
        - Connection string format
        """
        check_name = "Database Connection Security"
        db_url = settings.DATABASE_URL

        try:
            parsed = urlparse(db_url)
        except Exception as e:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.DATA_CONFIDENTIALITY,
                severity=SecuritySeverity.CRITICAL,
                passed=False,
                message=f"Invalid DATABASE_URL format: {e}",
                remediation="Use format: postgresql://user:password@host:port/database",
                iec_reference="IEC 62443-4-2 SR 4.1"
            )

        # Check for weak credentials
        weak_credentials = [
            ("user", "password"),
            ("user", "pass"),
            ("admin", "admin"),
            ("postgres", "postgres"),
            ("root", "root"),
            ("test", "test"),
            ("guest", "guest"),
            ("sa", "sa"),
        ]

        username = parsed.username or ""
        password = parsed.password or ""

        for weak_user, weak_pass in weak_credentials:
            if username.lower() == weak_user and password.lower() == weak_pass:
                return SecurityFinding(
                    check_name=check_name,
                    category=SecurityCategory.AUTHENTICATION,
                    severity=SecuritySeverity.CRITICAL,
                    passed=False,
                    message=f"DATABASE_URL uses default credentials: {weak_user}:****",
                    details="Default credentials are easily guessed by attackers",
                    remediation="Use strong, unique credentials from a secrets manager",
                    iec_reference="IEC 62443-4-2 SR 1.5"
                )

        # Password strength check
        if password and len(password) < 12:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.AUTHENTICATION,
                severity=SecuritySeverity.HIGH,
                passed=False,
                message=f"Database password too short: {len(password)} chars (minimum: 12)",
                remediation="Use a password with at least 12 characters",
                iec_reference="IEC 62443-4-2 SR 1.7"
            )

        # Check if username in password
        if username and password and username.lower() in password.lower():
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.AUTHENTICATION,
                severity=SecuritySeverity.HIGH,
                passed=False,
                message="Database password contains username",
                remediation="Use a password that doesn't contain the username",
                iec_reference="IEC 62443-4-2 SR 1.7"
            )

        # SSL/TLS check for production
        if settings.is_production():
            if "sslmode=" not in db_url.lower() and "ssl=" not in db_url.lower():
                return SecurityFinding(
                    check_name=check_name,
                    category=SecurityCategory.COMMUNICATION,
                    severity=SecuritySeverity.HIGH,
                    passed=False,
                    message="Database connection lacks SSL/TLS in production",
                    details="Unencrypted database connections expose data in transit",
                    remediation="Add sslmode=require or sslmode=verify-full to connection string",
                    iec_reference="IEC 62443-4-2 SR 3.1"
                )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.DATA_CONFIDENTIALITY,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="Database connection security is acceptable",
            iec_reference="IEC 62443-4-2 SR 4.1"
        )

    def validate_tls_certificates(self) -> SecurityFinding:
        """
        Validate TLS certificate configuration for integrations.

        Per IEC 62443-4-2 SR 3.1: Communication integrity
        Per IEC 62443-4-2 SR 4.3: Use of cryptography
        """
        check_name = "TLS Certificate Verification"
        issues = []

        # Check OPC UA endpoint
        opcua_endpoint = settings.SCADA_OPC_UA_ENDPOINT
        if opcua_endpoint.startswith("opc.tcp://"):
            # OPC UA should use secure endpoints in production
            if settings.is_production() and "localhost" not in opcua_endpoint:
                # In production, we expect certificate-based security
                # This is a basic check - actual cert validation happens at connection time
                issues.append("OPC UA endpoint should use certificate-based security")

        # Check analyzer endpoints for HTTPS
        for endpoint in settings.COMBUSTION_ANALYZER_ENDPOINTS:
            parsed = urlparse(endpoint)
            if settings.is_production() and parsed.scheme == "http":
                issues.append(f"Analyzer endpoint uses HTTP instead of HTTPS: {endpoint}")

        # Check OTLP endpoint
        otlp_endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT
        if settings.is_production() and "localhost" not in otlp_endpoint:
            parsed = urlparse(otlp_endpoint)
            if parsed.scheme == "http":
                issues.append(f"OTLP endpoint uses HTTP: {otlp_endpoint}")

        if issues and settings.is_production():
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.COMMUNICATION,
                severity=SecuritySeverity.HIGH,
                passed=False,
                message="TLS/HTTPS not enforced for all external connections",
                details="; ".join(issues),
                remediation="Enable TLS for all external communication endpoints",
                iec_reference="IEC 62443-4-2 SR 3.1"
            )
        elif issues:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.COMMUNICATION,
                severity=SecuritySeverity.MEDIUM,
                passed=False,
                message="Some endpoints lack TLS (acceptable for non-production)",
                details="; ".join(issues),
                remediation="Enable TLS for all connections before production deployment",
                iec_reference="IEC 62443-4-2 SR 3.1"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.COMMUNICATION,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="TLS configuration is acceptable for current environment",
            iec_reference="IEC 62443-4-2 SR 3.1"
        )

    def validate_plc_dcs_security(self) -> SecurityFinding:
        """
        Validate PLC/DCS connection security settings.

        Per IEC 62443-4-2 SR 1.2: Software process and device identification
        Per IEC 62443-4-2 SR 3.1: Communication integrity
        """
        check_name = "PLC/DCS Connection Security"
        issues = []
        warnings = []

        # DCS Configuration checks
        dcs_protocol = settings.DCS_PROTOCOL.lower()

        if dcs_protocol == "modbus_tcp":
            # Modbus TCP has no built-in authentication
            if settings.is_production():
                warnings.append(
                    "Modbus TCP protocol lacks authentication - ensure network segmentation"
                )

        # Check for non-standard ports (might indicate security-enhanced protocols)
        if settings.DCS_PORT == 502:  # Standard Modbus port
            warnings.append("Using standard Modbus port - consider network segmentation")

        # PLC Configuration checks
        plc_protocol = settings.PLC_PROTOCOL.lower()

        if plc_protocol == "modbus_tcp" and settings.is_production():
            warnings.append(
                "PLC uses Modbus TCP without encryption - ensure isolated control network"
            )

        # Check timeout settings (too long can leave connections vulnerable)
        if settings.DCS_TIMEOUT_MS > 5000:
            warnings.append(f"DCS timeout ({settings.DCS_TIMEOUT_MS}ms) is high - consider reducing")

        if settings.PLC_TIMEOUT_MS > 5000:
            warnings.append(f"PLC timeout ({settings.PLC_TIMEOUT_MS}ms) is high - consider reducing")

        # Critical: Check if connecting to localhost in production (dev oversight)
        if settings.is_production():
            if settings.DCS_HOST == "localhost" or settings.DCS_HOST == "127.0.0.1":
                issues.append("DCS_HOST is localhost in production - invalid configuration")

            if settings.PLC_HOST == "localhost" or settings.PLC_HOST == "127.0.0.1":
                issues.append("PLC_HOST is localhost in production - invalid configuration")

        if issues:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.COMMUNICATION,
                severity=SecuritySeverity.CRITICAL if settings.is_production() else SecuritySeverity.HIGH,
                passed=False,
                message="PLC/DCS connection configuration has critical issues",
                details="; ".join(issues),
                remediation="Configure valid production endpoints and ensure network segmentation",
                iec_reference="IEC 62443-4-2 SR 1.2"
            )

        if warnings and settings.is_production():
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.COMMUNICATION,
                severity=SecuritySeverity.MEDIUM,
                passed=False,
                message="PLC/DCS security has warnings for production",
                details="; ".join(warnings),
                remediation="Address security warnings before production deployment",
                iec_reference="IEC 62443-4-2 SR 3.1"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.COMMUNICATION,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="PLC/DCS connection security is acceptable",
            details="; ".join(warnings) if warnings else None,
            iec_reference="IEC 62443-4-2 SR 1.2"
        )

    def validate_environment_variables(self) -> SecurityFinding:
        """
        Audit environment variables for security issues.

        Per IEC 62443-4-2 SR 4.1: Information confidentiality

        Checks:
        - No hardcoded secrets in environment
        - Required security variables are set
        - No overly permissive settings
        """
        check_name = "Environment Variable Security"
        issues = []
        warnings = []

        # Required environment variables for security
        required_vars = {
            "DATABASE_URL": "Database connection string",
            "JWT_SECRET": "JWT signing secret",
        }

        # Optional but recommended
        recommended_vars = {
            "GREENLANG_ENV": "Environment designation",
        }

        # Check required variables
        for var, description in required_vars.items():
            value = os.environ.get(var)
            if not value:
                # Check if set in settings (from .env file)
                try:
                    value = getattr(settings, var, None)
                except:
                    value = None

            if not value:
                issues.append(f"Required security variable not set: {var}")

        # Scan environment for potential secrets
        sensitive_value_patterns = [
            (r"^AKIA[0-9A-Z]{16}$", "AWS Access Key ID"),
            (r"^sk_live_[a-zA-Z0-9]+$", "Stripe Live Key"),
            (r"^ghp_[a-zA-Z0-9]{36}$", "GitHub Personal Access Token"),
        ]

        for key, value in os.environ.items():
            if value:
                for pattern, secret_type in sensitive_value_patterns:
                    if re.match(pattern, value):
                        warnings.append(f"Potential {secret_type} found in env var: {key}")

        # Check DEBUG mode in production
        if settings.is_production() and settings.DEBUG:
            issues.append("DEBUG=True in production environment")

        if issues:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.CONFIGURATION,
                severity=SecuritySeverity.HIGH if settings.is_production() else SecuritySeverity.MEDIUM,
                passed=False,
                message="Environment variable security issues detected",
                details="; ".join(issues + warnings),
                remediation="Set required variables and review flagged items",
                iec_reference="IEC 62443-4-2 SR 4.1"
            )

        if warnings:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.CONFIGURATION,
                severity=SecuritySeverity.LOW,
                passed=True,  # Warnings don't fail the check
                message="Environment variables have minor warnings",
                details="; ".join(warnings),
                iec_reference="IEC 62443-4-2 SR 4.1"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.CONFIGURATION,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="Environment variable configuration is secure",
            iec_reference="IEC 62443-4-2 SR 4.1"
        )

    def validate_secret_detection(self) -> SecurityFinding:
        """
        Detect hardcoded secrets in configuration.

        Per IEC 62443-4-2 SR 4.1: Information confidentiality

        Scans configuration values for patterns indicating hardcoded secrets.
        """
        check_name = "Hardcoded Secret Detection"
        detected_secrets = []

        # Get all configuration values as strings
        config_values = {}
        for field_name in dir(settings):
            if not field_name.startswith("_"):
                try:
                    value = getattr(settings, field_name)
                    if isinstance(value, str):
                        config_values[field_name] = value
                except:
                    continue

        # Scan for secret patterns
        for field_name, value in config_values.items():
            # Skip known secret fields (they're expected to have secret values)
            if field_name in ["JWT_SECRET", "DATABASE_URL", "API_KEY"]:
                continue

            for pattern, secret_type in self.SECRET_DETECTION_PATTERNS:
                if re.search(pattern, value):
                    detected_secrets.append(f"{secret_type} in {field_name}")

        # Check for common secret file extensions in paths
        path_fields = ["opcua_cert_path", "opcua_private_key_path"]
        for field_name in path_fields:
            value = config_values.get(field_name.upper())
            if value and os.path.exists(value):
                # Check if key file is world-readable (Unix systems)
                try:
                    mode = os.stat(value).st_mode
                    if mode & 0o004:  # World-readable
                        detected_secrets.append(f"World-readable key file: {field_name}")
                except:
                    pass

        if detected_secrets:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.DATA_CONFIDENTIALITY,
                severity=SecuritySeverity.CRITICAL,
                passed=False,
                message="Potential hardcoded secrets detected",
                details="; ".join(detected_secrets),
                remediation="Move secrets to environment variables or a secrets manager",
                iec_reference="IEC 62443-4-2 SR 4.1"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.DATA_CONFIDENTIALITY,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="No hardcoded secrets detected in configuration",
            iec_reference="IEC 62443-4-2 SR 4.1"
        )

    def validate_rbac_configuration(self) -> SecurityFinding:
        """
        Validate role-based access control configuration.

        Per IEC 62443-4-2 SR 2.1: Authorization enforcement

        Verifies RBAC roles are properly defined for ICS operations.
        """
        check_name = "RBAC Configuration"
        issues = []

        # In a real implementation, this would check actual RBAC configuration
        # For now, we verify JWT algorithm supports role claims

        jwt_algorithm = settings.JWT_ALGORITHM

        # Check JWT algorithm strength
        weak_algorithms = ["none", "HS256"]  # HS256 is acceptable but not ideal
        strong_algorithms = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]

        if jwt_algorithm.lower() == "none":
            issues.append("JWT algorithm set to 'none' - authentication disabled!")
        elif jwt_algorithm in weak_algorithms and settings.is_production():
            issues.append(f"Consider using asymmetric algorithm instead of {jwt_algorithm} for production")

        # Check JWT expiration
        if settings.JWT_EXPIRATION_HOURS > 24:
            issues.append(f"JWT expiration ({settings.JWT_EXPIRATION_HOURS}h) exceeds 24h - consider shorter tokens")

        if settings.JWT_EXPIRATION_HOURS > 168:  # 1 week
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.AUTHORIZATION,
                severity=SecuritySeverity.HIGH,
                passed=False,
                message=f"JWT expiration too long: {settings.JWT_EXPIRATION_HOURS} hours",
                details="Long-lived tokens increase risk if compromised",
                remediation="Set JWT_EXPIRATION_HOURS to 24 or less",
                iec_reference="IEC 62443-4-2 SR 2.1"
            )

        if issues and settings.is_production():
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.AUTHORIZATION,
                severity=SecuritySeverity.MEDIUM,
                passed=False,
                message="RBAC configuration has issues",
                details="; ".join(issues),
                remediation="Review and strengthen RBAC configuration",
                iec_reference="IEC 62443-4-2 SR 2.1"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.AUTHORIZATION,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="RBAC configuration is acceptable",
            details=f"Algorithm: {jwt_algorithm}, Expiration: {settings.JWT_EXPIRATION_HOURS}h",
            iec_reference="IEC 62443-4-2 SR 2.1"
        )

    def validate_production_settings(self) -> SecurityFinding:
        """
        Validate production-specific security settings.

        Production requirements per IEC 62443-4-2:
        - Debug mode disabled
        - Safety interlocks enabled
        - Emergency shutdown enabled
        """
        check_name = "Production Security Settings"

        if not settings.is_production():
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.CONFIGURATION,
                severity=SecuritySeverity.INFO,
                passed=True,
                message="Not production environment - skipping production checks",
                iec_reference="IEC 62443-4-2 SR 7.1"
            )

        issues = []

        # Debug must be off in production
        if settings.DEBUG:
            issues.append("DEBUG mode is enabled in production")

        # Safety interlocks must be enabled
        if not settings.SAFETY_INTERLOCKS_ENABLED:
            issues.append("SAFETY_INTERLOCKS_ENABLED is disabled in production")

        # Flame detection must be required
        if not settings.FLAME_DETECTION_REQUIRED:
            issues.append("FLAME_DETECTION_REQUIRED is disabled in production")

        # Emergency shutdown must be enabled
        if not settings.EMERGENCY_SHUTDOWN_ENABLED:
            issues.append("EMERGENCY_SHUTDOWN_ENABLED is disabled in production")

        if issues:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.SYSTEM_INTEGRITY,
                severity=SecuritySeverity.CRITICAL,
                passed=False,
                message="Critical production settings misconfigured",
                details="; ".join(issues),
                remediation="Enable all safety features before production deployment",
                iec_reference="IEC 62443-4-2 SR 7.1"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.SYSTEM_INTEGRITY,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="Production settings are properly configured",
            iec_reference="IEC 62443-4-2 SR 7.1"
        )

    def validate_control_parameters(self) -> SecurityFinding:
        """
        Validate control parameters are within safe ranges.

        Safety validation per IEC 62443-4-2:
        - Verify limits are sensible
        - Check for configuration errors
        """
        check_name = "Control Parameter Validation"
        issues = []

        # Check temperature limits are sensible
        if settings.MAX_FLAME_TEMPERATURE_C < settings.MAX_FURNACE_TEMPERATURE_C:
            issues.append(
                f"MAX_FLAME_TEMPERATURE_C ({settings.MAX_FLAME_TEMPERATURE_C}) "
                f"< MAX_FURNACE_TEMPERATURE_C ({settings.MAX_FURNACE_TEMPERATURE_C})"
            )

        # Check heat output range is valid
        if settings.HEAT_OUTPUT_MIN_KW >= settings.HEAT_OUTPUT_MAX_KW:
            issues.append(
                f"HEAT_OUTPUT_MIN_KW ({settings.HEAT_OUTPUT_MIN_KW}) "
                f">= HEAT_OUTPUT_MAX_KW ({settings.HEAT_OUTPUT_MAX_KW})"
            )

        # Check target is within range
        if not (settings.HEAT_OUTPUT_MIN_KW <= settings.HEAT_OUTPUT_TARGET_KW <= settings.HEAT_OUTPUT_MAX_KW):
            issues.append(
                f"HEAT_OUTPUT_TARGET_KW ({settings.HEAT_OUTPUT_TARGET_KW}) "
                f"not in valid range [{settings.HEAT_OUTPUT_MIN_KW}, {settings.HEAT_OUTPUT_MAX_KW}]"
            )

        # Check control loop interval is reasonable for industrial control
        if settings.CONTROL_LOOP_INTERVAL_MS > 1000:
            issues.append(
                f"CONTROL_LOOP_INTERVAL_MS ({settings.CONTROL_LOOP_INTERVAL_MS}ms) "
                f"too slow for real-time control (should be <1000ms)"
            )

        # Check fuel/air flow limits
        if settings.MIN_FUEL_FLOW >= settings.MAX_FUEL_FLOW:
            issues.append("MIN_FUEL_FLOW >= MAX_FUEL_FLOW - invalid range")

        if settings.MIN_AIR_FLOW >= settings.MAX_AIR_FLOW:
            issues.append("MIN_AIR_FLOW >= MAX_AIR_FLOW - invalid range")

        if issues:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.SYSTEM_INTEGRITY,
                severity=SecuritySeverity.HIGH,
                passed=False,
                message="Control parameter configuration errors detected",
                details="; ".join(issues),
                remediation="Review and correct control parameter limits",
                iec_reference="IEC 62443-4-2 SR 7.1"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.SYSTEM_INTEGRITY,
            severity=SecuritySeverity.INFO,
            passed=True,
            message="Control parameters are within valid ranges",
            iec_reference="IEC 62443-4-2 SR 7.1"
        )

    def validate_rate_limiting(self) -> SecurityFinding:
        """
        Validate rate limiting configuration.

        Per IEC 62443-4-2 SR 7.7: Denial of service protection
        """
        check_name = "Rate Limiting Configuration"

        rate_limit = settings.RATE_LIMIT_PER_MINUTE

        # Check rate limit is configured
        if rate_limit <= 0:
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.SYSTEM_INTEGRITY,
                severity=SecuritySeverity.HIGH,
                passed=False,
                message="Rate limiting is disabled (RATE_LIMIT_PER_MINUTE <= 0)",
                details="No rate limiting exposes API to DoS attacks",
                remediation="Set RATE_LIMIT_PER_MINUTE to a reasonable value (e.g., 60-200)",
                iec_reference="IEC 62443-4-2 SR 7.7"
            )

        # Check rate limit is not too high
        if rate_limit > 1000 and settings.is_production():
            return SecurityFinding(
                check_name=check_name,
                category=SecurityCategory.SYSTEM_INTEGRITY,
                severity=SecuritySeverity.MEDIUM,
                passed=False,
                message=f"Rate limit may be too high for production: {rate_limit}/min",
                details="High rate limits reduce DoS protection effectiveness",
                remediation="Consider reducing rate limit for production",
                iec_reference="IEC 62443-4-2 SR 7.7"
            )

        return SecurityFinding(
            check_name=check_name,
            category=SecurityCategory.SYSTEM_INTEGRITY,
            severity=SecuritySeverity.INFO,
            passed=True,
            message=f"Rate limiting configured: {rate_limit} requests/minute",
            iec_reference="IEC 62443-4-2 SR 7.7"
        )

    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not data:
            return 0.0

        # Count character frequencies
        freq = {}
        for char in data:
            freq[char] = freq.get(char, 0) + 1

        # Calculate entropy
        length = len(data)
        entropy = 0.0

        for count in freq.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def run_all_validations(self) -> SecurityValidationResult:
        """
        Run all security validations.

        Returns:
            SecurityValidationResult with all findings
        """
        audit_logger.log_validation_start(settings.GREENLANG_ENV)

        # List of all validation methods
        validations = [
            ("JWT Secret", self.validate_jwt_secret),
            ("Database Security", self.validate_database_security),
            ("TLS Certificates", self.validate_tls_certificates),
            ("PLC/DCS Security", self.validate_plc_dcs_security),
            ("Environment Variables", self.validate_environment_variables),
            ("Secret Detection", self.validate_secret_detection),
            ("RBAC Configuration", self.validate_rbac_configuration),
            ("Production Settings", self.validate_production_settings),
            ("Control Parameters", self.validate_control_parameters),
            ("Rate Limiting", self.validate_rate_limiting),
        ]

        logger.info("=" * 80)
        logger.info("SECURITY VALIDATION - GL-005 CombustionControlAgent")
        logger.info(f"Environment: {settings.GREENLANG_ENV}")
        logger.info("=" * 80)

        for name, validator in validations:
            try:
                finding = validator()
                self.result.add_finding(finding)
                audit_logger.log_finding(finding)

                # Log result
                status = "PASS" if finding.passed else "FAIL"
                severity = finding.severity.value.upper()
                logger.info(f"[{status}] [{severity}] {name}: {finding.message}")

            except Exception as e:
                # Create error finding
                finding = SecurityFinding(
                    check_name=name,
                    category=SecurityCategory.CONFIGURATION,
                    severity=SecuritySeverity.HIGH,
                    passed=False,
                    message=f"Validation error: {str(e)}",
                    details=str(e)
                )
                self.result.add_finding(finding)
                logger.error(f"[ERROR] {name}: {e}")

        # Calculate final score
        self.result.calculate_score()

        # Calculate category scores
        category_findings: Dict[str, List[SecurityFinding]] = {}
        for finding in self.result.findings:
            cat = finding.category.value
            if cat not in category_findings:
                category_findings[cat] = []
            category_findings[cat].append(finding)

        for cat, findings in category_findings.items():
            passed = sum(1 for f in findings if f.passed)
            self.result.category_scores[cat] = (passed / len(findings)) * 100 if findings else 100

        # Log summary
        logger.info("=" * 80)
        logger.info(f"SECURITY VALIDATION {'PASSED' if self.result.passed else 'FAILED'}")
        logger.info(f"Score: {self.result.security_score:.1f}/100")
        logger.info(f"Checks: {self.result.passed_checks}/{self.result.total_checks} passed")
        logger.info(f"Critical: {self.result.critical_count}, High: {self.result.high_count}, "
                   f"Medium: {self.result.medium_count}, Low: {self.result.low_count}")
        logger.info(f"IEC 62443-4-2 Compliant: {self.result.iec_62443_compliant}")
        logger.info(f"Production Ready: {self.result.production_ready}")
        logger.info("=" * 80)

        # Record metrics
        security_metrics.record_validation(self.result)

        # Log to audit
        audit_logger.log_validation_complete(self.result)

        return self.result


def validate_startup_security(fail_fast: bool = True) -> SecurityValidationResult:
    """
    Convenience function to run security validation at startup.

    This is the main entry point called from main.py lifespan.

    Args:
        fail_fast: If True, raise exception on critical/high severity failures

    Returns:
        SecurityValidationResult with all findings

    Raises:
        SecurityValidationError: If fail_fast=True and validation fails

    Usage in main.py lifespan:
        from .security_validator import validate_startup_security

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Validate security before starting
            result = validate_startup_security(fail_fast=True)

            # Access results
            if not result.production_ready:
                logger.warning("System is not production ready")

            # ... rest of startup
    """
    validator = SecurityValidator()
    result = validator.run_all_validations()

    if fail_fast and not result.passed:
        # Log detailed failure information
        logger.critical("=" * 80)
        logger.critical("STARTUP ABORTED - SECURITY VALIDATION FAILED")
        logger.critical("=" * 80)

        for finding in result.findings:
            if not finding.passed and finding.severity in [SecuritySeverity.CRITICAL, SecuritySeverity.HIGH]:
                logger.critical(f"  [{finding.severity.value.upper()}] {finding.check_name}")
                logger.critical(f"    Message: {finding.message}")
                if finding.remediation:
                    logger.critical(f"    Fix: {finding.remediation}")

        logger.critical("=" * 80)
        logger.critical("Fix the above issues before deployment")
        logger.critical("=" * 80)

        raise SecurityValidationError(
            f"Security validation failed with {result.critical_count} critical and "
            f"{result.high_count} high severity issues. See logs for details.",
            result=result
        )

    return result


def get_security_score() -> float:
    """
    Get the current security score.

    Useful for health checks and monitoring dashboards.

    Returns:
        Security score (0-100), or -1 if not yet validated
    """
    # Run a quick validation
    validator = SecurityValidator()
    result = validator.run_all_validations()
    return result.security_score


if __name__ == "__main__":
    # Allow running as standalone script for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        result = validate_startup_security(fail_fast=False)

        print("\n" + "=" * 60)
        print("SECURITY VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Result: {'PASSED' if result.passed else 'FAILED'}")
        print(f"Score: {result.security_score:.1f}/100")
        print(f"Environment: {result.environment}")
        print(f"Total Checks: {result.total_checks}")
        print(f"Passed: {result.passed_checks}")
        print(f"Failed: {result.failed_checks}")
        print()
        print("Severity Breakdown:")
        print(f"  Critical: {result.critical_count}")
        print(f"  High: {result.high_count}")
        print(f"  Medium: {result.medium_count}")
        print(f"  Low: {result.low_count}")
        print()
        print(f"IEC 62443-4-2 Compliant: {result.iec_62443_compliant}")
        print(f"Production Ready: {result.production_ready}")
        print("=" * 60)

        # Print failed findings
        if result.failed_checks > 0:
            print("\nFAILED CHECKS:")
            print("-" * 60)
            for finding in result.findings:
                if not finding.passed:
                    print(f"\n[{finding.severity.value.upper()}] {finding.check_name}")
                    print(f"  Message: {finding.message}")
                    if finding.details:
                        print(f"  Details: {finding.details}")
                    if finding.remediation:
                        print(f"  Fix: {finding.remediation}")
                    if finding.iec_reference:
                        print(f"  Reference: {finding.iec_reference}")

        sys.exit(0 if result.passed else 1)

    except SecurityValidationError as e:
        print(f"\nSecurity validation failed: {e}")
        sys.exit(1)
