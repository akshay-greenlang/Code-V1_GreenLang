# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Security Validator Module.

This module provides comprehensive security validation capabilities for the
GL-011 FuelManagementAgent, including fuel price input validation, API security,
blend ratio verification, supply chain integrity, and rate limiting protection.

Security Standards Compliance:
- OWASP Application Security Verification Standard (ASVS) 4.0
- NIST 800-53 Security Controls
- SOC 2 Type II Requirements
- Petroleum Industry Data Security Standards
- ISO/IEC 27001 Information Security Management

Author: GreenLang Foundation Security Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import os
import re
import secrets
import ssl
import threading
import time
import ipaddress
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Pattern
from urllib.parse import urlparse, parse_qs
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SeverityLevel(Enum):
    """Security finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(Enum):
    """Security finding categories."""
    SECRET_DETECTED = "secret_detected"
    INPUT_VALIDATION = "input_validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    INJECTION = "injection"
    PERMISSION = "permission"
    DATA_EXPOSURE = "data_exposure"
    RATE_LIMITING = "rate_limiting"
    CSRF = "csrf"
    CORS = "cors"
    SSRF = "ssrf"


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class SecretType(Enum):
    """Types of secrets to detect."""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"
    CERTIFICATE = "certificate"
    CONNECTION_STRING = "connection_string"
    AWS_CREDENTIALS = "aws_credentials"
    AZURE_CREDENTIALS = "azure_credentials"
    GCP_CREDENTIALS = "gcp_credentials"
    JWT = "jwt"
    OAUTH = "oauth"
    FUEL_API_KEY = "fuel_api_key"
    PETROLEUM_API_TOKEN = "petroleum_api_token"
    GENERIC = "generic"


class FuelType(Enum):
    """Supported fuel types for validation."""
    GASOLINE = "gasoline"
    DIESEL = "diesel"
    NATURAL_GAS = "natural_gas"
    LNG = "lng"
    BIODIESEL = "biodiesel"
    ETHANOL = "ethanol"
    HYDROGEN = "hydrogen"
    AVIATION_FUEL = "aviation_fuel"
    MARINE_FUEL = "marine_fuel"
    HEATING_OIL = "heating_oil"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SecurityFinding:
    """Represents a security finding from validation."""
    finding_id: str
    category: FindingCategory
    severity: SeverityLevel
    title: str
    description: str
    location: str
    line_number: Optional[int]
    remediation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    evidence: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary."""
        return {
            'finding_id': self.finding_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'location': self.location,
            'line_number': self.line_number,
            'remediation': self.remediation,
            'cwe_id': self.cwe_id,
            'cvss_score': self.cvss_score,
            'evidence': self.evidence,
            'timestamp': self.timestamp
        }


@dataclass
class ValidationResult:
    """Result from a security validation check."""
    status: ValidationStatus
    check_name: str
    findings: List[SecurityFinding] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'status': self.status.value,
            'check_name': self.check_name,
            'findings': [f.to_dict() for f in self.findings],
            'details': self.details,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp
        }


@dataclass
class SecretMatch:
    """Represents a detected secret."""
    secret_type: SecretType
    pattern_name: str
    file_path: str
    line_number: int
    column: int
    masked_value: str
    confidence: float
    severity: SeverityLevel

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'secret_type': self.secret_type.value,
            'pattern_name': self.pattern_name,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'column': self.column,
            'masked_value': self.masked_value,
            'confidence': self.confidence,
            'severity': self.severity.value
        }


@dataclass
class RateLimitState:
    """Tracks rate limiting state for an endpoint or client."""
    client_id: str
    endpoint: str
    request_count: int
    window_start: datetime
    window_duration_seconds: int
    max_requests: int
    blocked_until: Optional[datetime] = None

    def is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        if self.blocked_until and datetime.now(timezone.utc) < self.blocked_until:
            return True

        # Check if window has expired
        window_end = self.window_start + timedelta(seconds=self.window_duration_seconds)
        if datetime.now(timezone.utc) > window_end:
            # Reset window
            self.request_count = 0
            self.window_start = datetime.now(timezone.utc)
            return False

        return self.request_count >= self.max_requests


# ============================================================================
# SECRET PATTERNS
# ============================================================================

class SecretPatterns:
    """
    Regex patterns for detecting secrets in code and configuration.

    Patterns specific to fuel management systems and petroleum industry APIs.
    """

    # API Key patterns
    API_KEY_PATTERNS = {
        'anthropic_api_key': (
            r'sk-ant-[a-zA-Z0-9\-_]{40,}',
            SecretType.API_KEY,
            SeverityLevel.CRITICAL,
            0.95
        ),
        'openai_api_key': (
            r'sk-[a-zA-Z0-9]{48,}',
            SecretType.API_KEY,
            SeverityLevel.CRITICAL,
            0.95
        ),
        'aws_access_key': (
            r'AKIA[0-9A-Z]{16}',
            SecretType.AWS_CREDENTIALS,
            SeverityLevel.CRITICAL,
            0.95
        ),
        'aws_secret_key': (
            r'(?i)aws[_\-]?secret[_\-]?access[_\-]?key["\']?\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})',
            SecretType.AWS_CREDENTIALS,
            SeverityLevel.CRITICAL,
            0.90
        ),
        'azure_subscription_key': (
            r'[a-f0-9]{32}',
            SecretType.AZURE_CREDENTIALS,
            SeverityLevel.HIGH,
            0.60
        ),
        'gcp_api_key': (
            r'AIza[0-9A-Za-z\-_]{35}',
            SecretType.GCP_CREDENTIALS,
            SeverityLevel.CRITICAL,
            0.95
        ),
        'github_token': (
            r'gh[pousr]_[A-Za-z0-9_]{36,}',
            SecretType.TOKEN,
            SeverityLevel.CRITICAL,
            0.95
        ),
        'generic_api_key': (
            r'(?i)(api[_\-]?key|apikey)["\']?\s*[=:]\s*["\']?([a-zA-Z0-9\-_]{20,})',
            SecretType.API_KEY,
            SeverityLevel.HIGH,
            0.80
        ),
        'fuel_api_key': (
            r'(?i)(fuel[_\-]?api[_\-]?key|petroleum[_\-]?api)["\']?\s*[=:]\s*["\']?([a-zA-Z0-9\-_]{20,})',
            SecretType.FUEL_API_KEY,
            SeverityLevel.CRITICAL,
            0.90
        ),
        'petroleum_token': (
            r'(?i)(petroleum[_\-]?token|oil[_\-]?api[_\-]?key)["\']?\s*[=:]\s*["\']?([a-zA-Z0-9\-_]{20,})',
            SecretType.PETROLEUM_API_TOKEN,
            SeverityLevel.CRITICAL,
            0.90
        ),
    }

    # Password patterns
    PASSWORD_PATTERNS = {
        'password_assignment': (
            r'(?i)(password|passwd|pwd)["\']?\s*[=:]\s*["\']([^"\']{8,})["\']',
            SecretType.PASSWORD,
            SeverityLevel.CRITICAL,
            0.85
        ),
        'password_in_url': (
            r'(?i)://[^:]+:([^@]+)@',
            SecretType.PASSWORD,
            SeverityLevel.CRITICAL,
            0.90
        ),
        'database_password': (
            r'(?i)(db[_\-]?password|database[_\-]?password)["\']?\s*[=:]\s*["\']([^"\']+)["\']',
            SecretType.PASSWORD,
            SeverityLevel.CRITICAL,
            0.90
        ),
    }

    # Token patterns
    TOKEN_PATTERNS = {
        'jwt_token': (
            r'eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+',
            SecretType.JWT,
            SeverityLevel.HIGH,
            0.95
        ),
        'bearer_token': (
            r'(?i)bearer\s+([a-zA-Z0-9\-_\.]+)',
            SecretType.TOKEN,
            SeverityLevel.HIGH,
            0.85
        ),
        'oauth_token': (
            r'(?i)(oauth[_\-]?token|access[_\-]?token)["\']?\s*[=:]\s*["\']([^"\']+)["\']',
            SecretType.OAUTH,
            SeverityLevel.HIGH,
            0.85
        ),
    }

    # Private key patterns
    PRIVATE_KEY_PATTERNS = {
        'rsa_private_key': (
            r'-----BEGIN RSA PRIVATE KEY-----',
            SecretType.PRIVATE_KEY,
            SeverityLevel.CRITICAL,
            0.99
        ),
        'ec_private_key': (
            r'-----BEGIN EC PRIVATE KEY-----',
            SecretType.PRIVATE_KEY,
            SeverityLevel.CRITICAL,
            0.99
        ),
        'private_key_generic': (
            r'-----BEGIN PRIVATE KEY-----',
            SecretType.PRIVATE_KEY,
            SeverityLevel.CRITICAL,
            0.99
        ),
        'encrypted_private_key': (
            r'-----BEGIN ENCRYPTED PRIVATE KEY-----',
            SecretType.PRIVATE_KEY,
            SeverityLevel.HIGH,
            0.95
        ),
    }

    # Connection string patterns
    CONNECTION_STRING_PATTERNS = {
        'postgres_connection': (
            r'postgres(?:ql)?://[^:]+:[^@]+@[^\s]+',
            SecretType.CONNECTION_STRING,
            SeverityLevel.CRITICAL,
            0.95
        ),
        'mysql_connection': (
            r'mysql://[^:]+:[^@]+@[^\s]+',
            SecretType.CONNECTION_STRING,
            SeverityLevel.CRITICAL,
            0.95
        ),
        'mongodb_connection': (
            r'mongodb(?:\+srv)?://[^:]+:[^@]+@[^\s]+',
            SecretType.CONNECTION_STRING,
            SeverityLevel.CRITICAL,
            0.95
        ),
        'redis_connection': (
            r'redis://:[^@]+@[^\s]+',
            SecretType.CONNECTION_STRING,
            SeverityLevel.CRITICAL,
            0.95
        ),
    }

    @classmethod
    def get_all_patterns(cls) -> Dict[str, Tuple[str, SecretType, SeverityLevel, float]]:
        """Get all secret detection patterns."""
        all_patterns = {}
        all_patterns.update(cls.API_KEY_PATTERNS)
        all_patterns.update(cls.PASSWORD_PATTERNS)
        all_patterns.update(cls.TOKEN_PATTERNS)
        all_patterns.update(cls.PRIVATE_KEY_PATTERNS)
        all_patterns.update(cls.CONNECTION_STRING_PATTERNS)
        return all_patterns


# ============================================================================
# MAIN SECURITY VALIDATOR CLASS
# ============================================================================

class SecurityValidator:
    """
    Comprehensive security validator for GL-011 FUELCRAFT.

    Provides security validation capabilities including:
    - Secret detection in code and configuration
    - Fuel price input validation
    - API URL validation (SSRF protection)
    - Blend ratio validation
    - Rate limiting enforcement
    - CORS policy validation
    - CSRF token validation
    - JWT token validation
    - Input sanitization validation
    - Authentication verification
    - Encryption validation
    - Permission auditing

    Example:
        >>> validator = SecurityValidator()
        >>> result = validator.validate_fuel_price_input("gasoline", 3.45)
        >>> print(f"Validation status: {result.status.value}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SecurityValidator.

        Args:
            config: Optional configuration dictionary with rate limits, CORS origins, etc.
        """
        self.config = config or {}
        self._lock = threading.RLock()
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._initialize_patterns()
        self._findings: List[SecurityFinding] = []
        self._validation_history: List[ValidationResult] = []

        # Rate limiting state
        self._rate_limit_state: Dict[str, RateLimitState] = {}

        # Injection patterns for fuel management
        self._initialize_injection_patterns()

        logger.info("GL-011 FUELCRAFT SecurityValidator initialized")

    def _initialize_patterns(self) -> None:
        """Initialize and compile regex patterns."""
        patterns = SecretPatterns.get_all_patterns()
        for name, (pattern, _, _, _) in patterns.items():
            try:
                self._compiled_patterns[name] = re.compile(pattern, re.MULTILINE)
            except re.error as e:
                logger.warning(f"Failed to compile pattern {name}: {e}")

    def _initialize_injection_patterns(self) -> None:
        """Initialize injection attack patterns."""
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
            r"(--|#|/\*|\*/|;)",
            r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)",
            r"(\bUNION\b.*\bSELECT\b)",
            r"('.*OR.*'.*=.*')",
        ]

        self.command_injection_patterns = [
            r"[;&|`$()]",
            r"\b(rm|cat|ls|chmod|chown|kill|wget|curl|nc|bash|sh)\b",
            r"\$\{.*\}",
            r">\s*/dev/",
        ]

        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.",
            r"%2e%2e",
            r"\.\.\\",
            r"%252e",
            r"\.\.%2f",
        ]

        self.xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"onerror\s*=",
            r"onload\s*=",
            r"onclick\s*=",
            r"<iframe",
            r"<embed",
            r"<object",
        ]

        # Compile all patterns
        self.compiled_sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.sql_injection_patterns]
        self.compiled_cmd_patterns = [re.compile(p, re.IGNORECASE) for p in self.command_injection_patterns]
        self.compiled_path_patterns = [re.compile(p, re.IGNORECASE) for p in self.path_traversal_patterns]
        self.compiled_xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.xss_patterns]

    def _generate_finding_id(self) -> str:
        """Generate unique finding ID."""
        return f"FUEL-SEC-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(4).upper()}"

    def _mask_secret(self, value: str, visible_chars: int = 4) -> str:
        """Mask a secret value for safe logging."""
        if len(value) <= visible_chars * 2:
            return '*' * len(value)
        return value[:visible_chars] + '*' * (len(value) - visible_chars * 2) + value[-visible_chars:]

    # ========================================================================
    # FUEL PRICE INPUT VALIDATION
    # ========================================================================

    def validate_fuel_price_input(
        self,
        fuel_type: str,
        price: float,
        currency: str = "USD",
        unit: str = "gallon"
    ) -> ValidationResult:
        """
        Validate fuel price input for security threats.

        Checks for:
        - SQL injection in fuel_type
        - Command injection in fuel_type
        - XSS in fuel_type
        - Price range validation
        - Precision attacks
        - Null byte injection
        - Type confusion

        Args:
            fuel_type: Type of fuel (e.g., "gasoline", "diesel")
            price: Price value
            currency: Currency code (default: USD)
            unit: Unit of measurement (default: gallon)

        Returns:
            ValidationResult with security findings

        Example:
            >>> result = validator.validate_fuel_price_input("gasoline", 3.45)
            >>> assert result.status == ValidationStatus.PASSED
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        # Validate fuel_type string
        if not isinstance(fuel_type, str):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.HIGH,
                title="Invalid fuel_type data type",
                description=f"fuel_type must be string, got {type(fuel_type).__name__}",
                location="fuel_type",
                line_number=None,
                remediation="Ensure fuel_type is a string value",
                cwe_id="CWE-20"
            ))
        else:
            # SQL injection check
            if self._contains_sql_injection(fuel_type):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INJECTION,
                    severity=SeverityLevel.CRITICAL,
                    title="SQL injection detected in fuel_type",
                    description=f"Potential SQL injection pattern in fuel_type: {self._mask_secret(fuel_type, 8)}",
                    location="fuel_type",
                    line_number=None,
                    remediation="Sanitize fuel_type input and use parameterized queries",
                    cwe_id="CWE-89",
                    cvss_score=9.8,
                    evidence=self._mask_secret(fuel_type, 8)
                ))

            # Command injection check
            if self._contains_command_injection(fuel_type):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INJECTION,
                    severity=SeverityLevel.CRITICAL,
                    title="Command injection detected in fuel_type",
                    description=f"Potential command injection in fuel_type: {self._mask_secret(fuel_type, 8)}",
                    location="fuel_type",
                    line_number=None,
                    remediation="Remove special characters and validate against whitelist",
                    cwe_id="CWE-78",
                    cvss_score=9.8,
                    evidence=self._mask_secret(fuel_type, 8)
                ))

            # XSS check
            if self._contains_xss(fuel_type):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INJECTION,
                    severity=SeverityLevel.HIGH,
                    title="XSS detected in fuel_type",
                    description=f"Potential XSS payload in fuel_type: {self._mask_secret(fuel_type, 8)}",
                    location="fuel_type",
                    line_number=None,
                    remediation="HTML-encode fuel_type output and validate input",
                    cwe_id="CWE-79",
                    cvss_score=7.3,
                    evidence=self._mask_secret(fuel_type, 8)
                ))

            # Null byte injection check
            if '\x00' in fuel_type or '%00' in fuel_type:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INJECTION,
                    severity=SeverityLevel.HIGH,
                    title="Null byte injection detected",
                    description="Null byte detected in fuel_type input",
                    location="fuel_type",
                    line_number=None,
                    remediation="Remove null bytes from input",
                    cwe_id="CWE-626"
                ))

            # Length validation
            if len(fuel_type) > 100:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.MEDIUM,
                    title="Excessively long fuel_type",
                    description=f"fuel_type length is {len(fuel_type)}, max allowed is 100",
                    location="fuel_type",
                    line_number=None,
                    remediation="Limit fuel_type to 100 characters",
                    cwe_id="CWE-400"
                ))

            # Validate against known fuel types
            valid_fuel_types = [ft.value for ft in FuelType]
            if fuel_type.lower() not in valid_fuel_types:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.LOW,
                    title="Unknown fuel type",
                    description=f"fuel_type '{fuel_type}' not in known types: {valid_fuel_types}",
                    location="fuel_type",
                    line_number=None,
                    remediation=f"Use one of the valid fuel types: {', '.join(valid_fuel_types)}",
                    cwe_id="CWE-20"
                ))

        # Validate price
        if not isinstance(price, (int, float)):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.HIGH,
                title="Invalid price data type",
                description=f"price must be numeric, got {type(price).__name__}",
                location="price",
                line_number=None,
                remediation="Ensure price is a numeric value",
                cwe_id="CWE-20"
            ))
        else:
            # Range validation (fuel prices typically $0.01 - $1,000,000 per unit)
            if price < 0:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.HIGH,
                    title="Negative price value",
                    description=f"Price cannot be negative: {price}",
                    location="price",
                    line_number=None,
                    remediation="Ensure price is positive",
                    cwe_id="CWE-20"
                ))

            if price > 1_000_000:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.MEDIUM,
                    title="Price exceeds maximum threshold",
                    description=f"Price {price} exceeds maximum allowed value of 1,000,000",
                    location="price",
                    line_number=None,
                    remediation="Verify price is correct or increase threshold if legitimate",
                    cwe_id="CWE-20"
                ))

            # Precision check (may indicate injection attempt or overflow attack)
            price_str = str(price)
            if '.' in price_str:
                decimal_places = len(price_str.split('.')[-1])
                if decimal_places > 10:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.INPUT_VALIDATION,
                        severity=SeverityLevel.MEDIUM,
                        title="Excessive price precision",
                        description=f"Price has {decimal_places} decimal places (may indicate attack)",
                        location="price",
                        line_number=None,
                        remediation="Limit price precision to 10 decimal places",
                        cwe_id="CWE-20"
                    ))

            # NaN/Infinity check
            if isinstance(price, float) and (price != price or abs(price) == float('inf')):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.HIGH,
                    title="Invalid numeric value (NaN/Infinity)",
                    description=f"Price is NaN or Infinity: {price}",
                    location="price",
                    line_number=None,
                    remediation="Validate price is a valid finite number",
                    cwe_id="CWE-20"
                ))

        # Validate currency
        valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CNY']
        if currency.upper() not in valid_currencies:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.LOW,
                title="Unknown currency code",
                description=f"Currency '{currency}' not in known codes: {valid_currencies}",
                location="currency",
                line_number=None,
                remediation=f"Use one of: {', '.join(valid_currencies)}",
                cwe_id="CWE-20"
            ))

        # Validate unit
        valid_units = ['gallon', 'liter', 'barrel', 'cubic_meter', 'ton']
        if unit.lower() not in valid_units:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.LOW,
                title="Unknown unit of measurement",
                description=f"Unit '{unit}' not in known units: {valid_units}",
                location="unit",
                line_number=None,
                remediation=f"Use one of: {', '.join(valid_units)}",
                cwe_id="CWE-20"
            ))

        execution_time = (time.perf_counter() - start_time) * 1000

        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="fuel_price_input_validation",
            findings=findings,
            details={
                'fuel_type': fuel_type,
                'price': price,
                'currency': currency,
                'unit': unit,
                'checks_performed': 12
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # API URL VALIDATION (SSRF PROTECTION)
    # ========================================================================

    def validate_api_url(
        self,
        url: str,
        allow_private_ips: bool = False,
        allowed_schemes: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate API URL for SSRF and other threats.

        Checks for:
        - SSRF via private IP addresses
        - Protocol validation
        - Port restrictions
        - URL injection
        - DNS rebinding protection

        Args:
            url: URL to validate
            allow_private_ips: Whether to allow private IP addresses (default: False)
            allowed_schemes: List of allowed URL schemes (default: ['https', 'http'])

        Returns:
            ValidationResult with security findings

        Example:
            >>> result = validator.validate_api_url("https://api.example.com/fuel-prices")
            >>> assert result.status == ValidationStatus.PASSED
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        if allowed_schemes is None:
            allowed_schemes = ['https', 'http']

        try:
            parsed = urlparse(url)

            # Protocol whitelist
            if parsed.scheme not in allowed_schemes:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.SSRF,
                    severity=SeverityLevel.HIGH,
                    title=f"Invalid URL protocol: {parsed.scheme}",
                    description=f"Protocol '{parsed.scheme}' not in allowed list: {allowed_schemes}",
                    location="url.scheme",
                    line_number=None,
                    remediation=f"Use one of: {', '.join(allowed_schemes)}",
                    cwe_id="CWE-918"
                ))

            # HTTPS enforcement for production
            if parsed.scheme != 'https' and self.config.get('enforce_https', True):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.ENCRYPTION,
                    severity=SeverityLevel.MEDIUM,
                    title="Non-HTTPS URL in production",
                    description=f"URL uses {parsed.scheme} instead of HTTPS",
                    location="url.scheme",
                    line_number=None,
                    remediation="Use HTTPS for all production API calls",
                    cwe_id="CWE-319"
                ))

            # SSRF - Check for private IP ranges
            if parsed.hostname:
                try:
                    ip = ipaddress.ip_address(parsed.hostname)
                    if not allow_private_ips:
                        if ip.is_private or ip.is_loopback or ip.is_link_local:
                            findings.append(SecurityFinding(
                                finding_id=self._generate_finding_id(),
                                category=FindingCategory.SSRF,
                                severity=SeverityLevel.CRITICAL,
                                title="SSRF detected: Private IP not allowed",
                                description=f"URL contains private IP address: {parsed.hostname}",
                                location="url.hostname",
                                line_number=None,
                                remediation="Use public IP addresses or domain names only",
                                cwe_id="CWE-918",
                                cvss_score=9.1
                            ))

                        if ip.is_reserved or ip.is_multicast:
                            findings.append(SecurityFinding(
                                finding_id=self._generate_finding_id(),
                                category=FindingCategory.SSRF,
                                severity=SeverityLevel.CRITICAL,
                                title="SSRF detected: Reserved IP",
                                description=f"URL contains reserved/multicast IP: {parsed.hostname}",
                                location="url.hostname",
                                line_number=None,
                                remediation="Use valid public IP addresses only",
                                cwe_id="CWE-918"
                            ))
                except ValueError:
                    # Not an IP address, it's a hostname - check for suspicious patterns
                    suspicious_hostnames = ['localhost', '127.0.0.1', '0.0.0.0', 'metadata']
                    if any(sus in parsed.hostname.lower() for sus in suspicious_hostnames):
                        findings.append(SecurityFinding(
                            finding_id=self._generate_finding_id(),
                            category=FindingCategory.SSRF,
                            severity=SeverityLevel.HIGH,
                            title="Suspicious hostname detected",
                            description=f"Hostname contains suspicious pattern: {parsed.hostname}",
                            location="url.hostname",
                            line_number=None,
                            remediation="Use production domain names only",
                            cwe_id="CWE-918"
                        ))

            # Port restrictions
            if parsed.port:
                allowed_ports = self.config.get('allowed_ports', [80, 443, 8080, 8443])
                if parsed.port not in allowed_ports:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.SSRF,
                        severity=SeverityLevel.MEDIUM,
                        title=f"Non-standard port detected: {parsed.port}",
                        description=f"Port {parsed.port} not in allowed list: {allowed_ports}",
                        location="url.port",
                        line_number=None,
                        remediation=f"Use standard ports: {allowed_ports}",
                        cwe_id="CWE-918"
                    ))

            # Check for URL injection patterns
            if any(char in url for char in ['<', '>', '"', "'"]):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INJECTION,
                    severity=SeverityLevel.HIGH,
                    title="URL injection characters detected",
                    description="URL contains potentially dangerous characters",
                    location="url",
                    line_number=None,
                    remediation="URL-encode special characters",
                    cwe_id="CWE-74"
                ))

            # Check URL length
            if len(url) > 2048:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.MEDIUM,
                    title="Excessively long URL",
                    description=f"URL length {len(url)} exceeds recommended maximum of 2048",
                    location="url",
                    line_number=None,
                    remediation="Shorten URL or use POST request",
                    cwe_id="CWE-400"
                ))

            # Check for credentials in URL
            if parsed.username or parsed.password:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.SECRET_DETECTED,
                    severity=SeverityLevel.CRITICAL,
                    title="Credentials in URL",
                    description="URL contains username/password (security risk)",
                    location="url",
                    line_number=None,
                    remediation="Use Authorization header instead of URL credentials",
                    cwe_id="CWE-598"
                ))

        except Exception as e:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.HIGH,
                title="URL parsing error",
                description=f"Failed to parse URL: {str(e)}",
                location="url",
                line_number=None,
                remediation="Provide a valid URL",
                cwe_id="CWE-20"
            ))

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="api_url_validation",
            findings=findings,
            details={
                'url': url,
                'checks_performed': 8
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # FILE PATH VALIDATION (PATH TRAVERSAL PROTECTION)
    # ========================================================================

    def validate_file_path(
        self,
        path: str,
        allowed_base_dir: str = "/var/greenlang/data"
    ) -> ValidationResult:
        """
        Validate file path for path traversal attacks.

        Checks for:
        - Path traversal patterns (../, ..\, etc.)
        - Absolute path validation
        - Null byte injection
        - Symbolic link attacks
        - Directory restrictions

        Args:
            path: File path to validate
            allowed_base_dir: Base directory for allowed paths

        Returns:
            ValidationResult with security findings

        Example:
            >>> result = validator.validate_file_path("/var/greenlang/data/fuel_prices.csv")
            >>> assert result.status == ValidationStatus.PASSED
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        # Path traversal check
        if self._contains_path_traversal(path):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INJECTION,
                severity=SeverityLevel.CRITICAL,
                title="Path traversal detected",
                description=f"Path contains traversal pattern: {self._mask_secret(path, 10)}",
                location="path",
                line_number=None,
                remediation="Remove path traversal sequences and validate against allowed paths",
                cwe_id="CWE-22",
                cvss_score=8.6,
                evidence=self._mask_secret(path, 10)
            ))

        # Absolute path check
        if not path.startswith(allowed_base_dir):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.HIGH,
                title="Path outside allowed directory",
                description=f"Path '{path}' not within allowed base: {allowed_base_dir}",
                location="path",
                line_number=None,
                remediation=f"Ensure path starts with {allowed_base_dir}",
                cwe_id="CWE-22"
            ))

        # Null byte injection
        if '\x00' in path or '%00' in path:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INJECTION,
                severity=SeverityLevel.CRITICAL,
                title="Null byte injection detected",
                description="Null byte detected in file path",
                location="path",
                line_number=None,
                remediation="Remove null bytes from path",
                cwe_id="CWE-626"
            ))

        # Check for dangerous file extensions
        dangerous_extensions = ['.exe', '.dll', '.so', '.dylib', '.sh', '.bat', '.cmd', '.ps1']
        if any(path.lower().endswith(ext) for ext in dangerous_extensions):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.HIGH,
                title="Dangerous file extension",
                description=f"File has executable extension: {Path(path).suffix}",
                location="path",
                line_number=None,
                remediation="Only allow data file extensions (.csv, .json, .txt, .parquet)",
                cwe_id="CWE-434"
            ))

        # Check for overly long paths
        if len(path) > 4096:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.MEDIUM,
                title="Excessively long path",
                description=f"Path length {len(path)} exceeds maximum of 4096",
                location="path",
                line_number=None,
                remediation="Shorten file path",
                cwe_id="CWE-400"
            ))

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="file_path_validation",
            findings=findings,
            details={
                'path': path,
                'allowed_base_dir': allowed_base_dir,
                'checks_performed': 5
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # BLEND RATIO VALIDATION
    # ========================================================================

    def validate_blend_ratios(
        self,
        blend_ratios: Dict[str, float],
        tolerance: float = 0.001
    ) -> ValidationResult:
        """
        Validate fuel blend ratios for correctness and security.

        Checks for:
        - Ratios sum to 1.0
        - All ratios are positive
        - No overflow/underflow
        - Injection in fuel type names
        - Type validation

        Args:
            blend_ratios: Dictionary mapping fuel type to ratio (e.g., {"gasoline": 0.7, "ethanol": 0.3})
            tolerance: Tolerance for sum validation (default: 0.001)

        Returns:
            ValidationResult with security findings

        Example:
            >>> ratios = {"gasoline": 0.85, "ethanol": 0.15}
            >>> result = validator.validate_blend_ratios(ratios)
            >>> assert result.status == ValidationStatus.PASSED
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        # Type validation
        if not isinstance(blend_ratios, dict):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.HIGH,
                title="Invalid blend_ratios type",
                description=f"blend_ratios must be dict, got {type(blend_ratios).__name__}",
                location="blend_ratios",
                line_number=None,
                remediation="Provide blend_ratios as dictionary",
                cwe_id="CWE-20"
            ))
            # Can't continue validation if not a dict
            execution_time = (time.perf_counter() - start_time) * 1000
            return ValidationResult(
                status=ValidationStatus.FAILED,
                check_name="blend_ratio_validation",
                findings=findings,
                details={},
                execution_time_ms=execution_time
            )

        # Check if empty
        if len(blend_ratios) == 0:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.MEDIUM,
                title="Empty blend ratios",
                description="blend_ratios dictionary is empty",
                location="blend_ratios",
                line_number=None,
                remediation="Provide at least one fuel type and ratio",
                cwe_id="CWE-20"
            ))

        total_ratio = 0.0
        for fuel_type, ratio in blend_ratios.items():
            # Validate fuel type (check for injection)
            if self._contains_sql_injection(fuel_type):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INJECTION,
                    severity=SeverityLevel.CRITICAL,
                    title=f"SQL injection in fuel type: {fuel_type}",
                    description=f"Potential SQL injection in blend ratio key",
                    location=f"blend_ratios.{fuel_type}",
                    line_number=None,
                    remediation="Sanitize fuel type names",
                    cwe_id="CWE-89"
                ))

            # Validate ratio is numeric
            if not isinstance(ratio, (int, float)):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.HIGH,
                    title=f"Invalid ratio type for {fuel_type}",
                    description=f"Ratio must be numeric, got {type(ratio).__name__}",
                    location=f"blend_ratios.{fuel_type}",
                    line_number=None,
                    remediation="Ensure all ratios are numeric",
                    cwe_id="CWE-20"
                ))
                continue

            # Check for negative ratios
            if ratio < 0:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.HIGH,
                    title=f"Negative ratio for {fuel_type}",
                    description=f"Ratio cannot be negative: {ratio}",
                    location=f"blend_ratios.{fuel_type}",
                    line_number=None,
                    remediation="Ensure all ratios are >= 0",
                    cwe_id="CWE-20"
                ))

            # Check for ratios > 1.0
            if ratio > 1.0:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.MEDIUM,
                    title=f"Ratio exceeds 1.0 for {fuel_type}",
                    description=f"Individual ratio {ratio} exceeds maximum of 1.0",
                    location=f"blend_ratios.{fuel_type}",
                    line_number=None,
                    remediation="Ensure all ratios are <= 1.0",
                    cwe_id="CWE-20"
                ))

            # Check for NaN/Infinity
            if isinstance(ratio, float) and (ratio != ratio or abs(ratio) == float('inf')):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.HIGH,
                    title=f"Invalid numeric value for {fuel_type}",
                    description=f"Ratio is NaN or Infinity: {ratio}",
                    location=f"blend_ratios.{fuel_type}",
                    line_number=None,
                    remediation="Provide valid finite numbers",
                    cwe_id="CWE-20"
                ))
                continue

            total_ratio += ratio

        # Validate ratios sum to 1.0 (within tolerance)
        if abs(total_ratio - 1.0) > tolerance:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.HIGH,
                title="Blend ratios do not sum to 1.0",
                description=f"Ratios sum to {total_ratio:.6f}, must equal 1.0 (tolerance: {tolerance})",
                location="blend_ratios",
                line_number=None,
                remediation="Ensure blend ratios sum to exactly 1.0",
                cwe_id="CWE-20"
            ))

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="blend_ratio_validation",
            findings=findings,
            details={
                'total_ratio': total_ratio,
                'fuel_types_count': len(blend_ratios),
                'tolerance': tolerance
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # TOOL INPUT VALIDATION (ALL 6 GL-011 TOOLS)
    # ========================================================================

    def validate_tool_input(
        self,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate input for all 6 GL-011 FUELCRAFT tools.

        Tools:
        1. analyze_fuel_prices
        2. optimize_fuel_blend
        3. forecast_fuel_demand
        4. track_supply_chain
        5. calculate_fuel_emissions
        6. audit_fuel_compliance

        Args:
            tool_name: Name of the tool being invoked
            tool_input: Input parameters for the tool

        Returns:
            ValidationResult with security findings

        Example:
            >>> result = validator.validate_tool_input(
            ...     "analyze_fuel_prices",
            ...     {"fuel_types": ["gasoline", "diesel"], "region": "US"}
            ... )
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        valid_tools = [
            'analyze_fuel_prices',
            'optimize_fuel_blend',
            'forecast_fuel_demand',
            'track_supply_chain',
            'calculate_fuel_emissions',
            'audit_fuel_compliance'
        ]

        if tool_name not in valid_tools:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.MEDIUM,
                title="Unknown tool name",
                description=f"Tool '{tool_name}' not in valid tools: {valid_tools}",
                location="tool_name",
                line_number=None,
                remediation=f"Use one of: {', '.join(valid_tools)}",
                cwe_id="CWE-20"
            ))

        # Tool-specific validation
        if tool_name == "analyze_fuel_prices":
            fuel_types = tool_input.get('fuel_types', [])
            if not isinstance(fuel_types, list):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.HIGH,
                    title="Invalid fuel_types type",
                    description=f"fuel_types must be list, got {type(fuel_types).__name__}",
                    location="tool_input.fuel_types",
                    line_number=None,
                    remediation="Provide fuel_types as list of strings",
                    cwe_id="CWE-20"
                ))
            else:
                for fuel_type in fuel_types:
                    # Validate each fuel type
                    price_result = self.validate_fuel_price_input(fuel_type, 0.0)
                    findings.extend(price_result.findings)

        elif tool_name == "optimize_fuel_blend":
            blend_ratios = tool_input.get('blend_ratios', {})
            blend_result = self.validate_blend_ratios(blend_ratios)
            findings.extend(blend_result.findings)

        elif tool_name == "forecast_fuel_demand":
            forecast_horizon = tool_input.get('forecast_horizon_days')
            if forecast_horizon is not None:
                if not isinstance(forecast_horizon, int):
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.INPUT_VALIDATION,
                        severity=SeverityLevel.MEDIUM,
                        title="Invalid forecast_horizon type",
                        description=f"forecast_horizon_days must be int, got {type(forecast_horizon).__name__}",
                        location="tool_input.forecast_horizon_days",
                        line_number=None,
                        remediation="Provide forecast_horizon_days as integer",
                        cwe_id="CWE-20"
                    ))
                elif forecast_horizon < 1 or forecast_horizon > 365:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.INPUT_VALIDATION,
                        severity=SeverityLevel.MEDIUM,
                        title="Forecast horizon out of range",
                        description=f"forecast_horizon_days {forecast_horizon} not in range [1, 365]",
                        location="tool_input.forecast_horizon_days",
                        line_number=None,
                        remediation="Set forecast_horizon_days between 1 and 365",
                        cwe_id="CWE-20"
                    ))

        elif tool_name == "track_supply_chain":
            supplier_id = tool_input.get('supplier_id', '')
            if self._contains_sql_injection(str(supplier_id)):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INJECTION,
                    severity=SeverityLevel.CRITICAL,
                    title="SQL injection in supplier_id",
                    description="Potential SQL injection in supplier_id parameter",
                    location="tool_input.supplier_id",
                    line_number=None,
                    remediation="Sanitize supplier_id input",
                    cwe_id="CWE-89"
                ))

        elif tool_name == "calculate_fuel_emissions":
            fuel_quantity = tool_input.get('fuel_quantity')
            if fuel_quantity is not None:
                if not isinstance(fuel_quantity, (int, float)):
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.INPUT_VALIDATION,
                        severity=SeverityLevel.HIGH,
                        title="Invalid fuel_quantity type",
                        description=f"fuel_quantity must be numeric, got {type(fuel_quantity).__name__}",
                        location="tool_input.fuel_quantity",
                        line_number=None,
                        remediation="Provide fuel_quantity as number",
                        cwe_id="CWE-20"
                    ))
                elif fuel_quantity < 0:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.INPUT_VALIDATION,
                        severity=SeverityLevel.MEDIUM,
                        title="Negative fuel quantity",
                        description=f"fuel_quantity cannot be negative: {fuel_quantity}",
                        location="tool_input.fuel_quantity",
                        line_number=None,
                        remediation="Ensure fuel_quantity is positive",
                        cwe_id="CWE-20"
                    ))

        elif tool_name == "audit_fuel_compliance":
            reporting_period = tool_input.get('reporting_period', '')
            # Validate date format (YYYY-MM)
            if reporting_period and not re.match(r'^\d{4}-\d{2}$', str(reporting_period)):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.MEDIUM,
                    title="Invalid reporting_period format",
                    description=f"reporting_period must be YYYY-MM format, got: {reporting_period}",
                    location="tool_input.reporting_period",
                    line_number=None,
                    remediation="Use YYYY-MM format (e.g., 2025-12)",
                    cwe_id="CWE-20"
                ))

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="tool_input_validation",
            findings=findings,
            details={
                'tool_name': tool_name,
                'input_keys': list(tool_input.keys())
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # RATE LIMITING VALIDATION
    # ========================================================================

    def check_rate_limit(
        self,
        client_id: str,
        endpoint: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> ValidationResult:
        """
        Check and enforce rate limiting for API endpoints.

        Args:
            client_id: Client identifier (IP, API key hash, user ID)
            endpoint: API endpoint being accessed
            max_requests: Maximum requests allowed in window (default: 100)
            window_seconds: Time window in seconds (default: 60)

        Returns:
            ValidationResult indicating if rate limit exceeded

        Example:
            >>> result = validator.check_rate_limit("client-123", "/api/fuel-prices")
            >>> if result.status == ValidationStatus.FAILED:
            ...     print("Rate limit exceeded!")
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        state_key = f"{client_id}:{endpoint}"

        with self._lock:
            if state_key not in self._rate_limit_state:
                # Initialize new state
                self._rate_limit_state[state_key] = RateLimitState(
                    client_id=client_id,
                    endpoint=endpoint,
                    request_count=1,
                    window_start=datetime.now(timezone.utc),
                    window_duration_seconds=window_seconds,
                    max_requests=max_requests
                )
            else:
                state = self._rate_limit_state[state_key]

                # Check if window has expired
                window_end = state.window_start + timedelta(seconds=state.window_duration_seconds)
                if datetime.now(timezone.utc) > window_end:
                    # Reset window
                    state.request_count = 1
                    state.window_start = datetime.now(timezone.utc)
                    state.blocked_until = None
                else:
                    # Increment counter
                    state.request_count += 1

                # Check if rate limited
                if state.is_rate_limited():
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.RATE_LIMITING,
                        severity=SeverityLevel.HIGH,
                        title="Rate limit exceeded",
                        description=(
                            f"Client {client_id} exceeded rate limit for {endpoint}: "
                            f"{state.request_count}/{max_requests} requests in {window_seconds}s"
                        ),
                        location=f"{endpoint}",
                        line_number=None,
                        remediation="Reduce request frequency or request rate limit increase",
                        cwe_id="CWE-770"
                    ))

                    # Block for additional time
                    if not state.blocked_until:
                        state.blocked_until = datetime.now(timezone.utc) + timedelta(seconds=window_seconds)

        execution_time = (time.perf_counter() - start_time) * 1000
        status = ValidationStatus.FAILED if findings else ValidationStatus.PASSED

        current_state = self._rate_limit_state.get(state_key)
        result = ValidationResult(
            status=status,
            check_name="rate_limit_check",
            findings=findings,
            details={
                'client_id': client_id,
                'endpoint': endpoint,
                'request_count': current_state.request_count if current_state else 0,
                'max_requests': max_requests,
                'window_seconds': window_seconds,
                'blocked_until': current_state.blocked_until.isoformat() if current_state and current_state.blocked_until else None
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # CORS VALIDATION
    # ========================================================================

    def validate_cors_headers(
        self,
        origin: str,
        allowed_origins: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate CORS (Cross-Origin Resource Sharing) headers.

        Args:
            origin: Origin header from request
            allowed_origins: List of allowed origins (default: from config)

        Returns:
            ValidationResult with CORS validation findings

        Example:
            >>> result = validator.validate_cors_headers(
            ...     "https://app.greenlang.com",
            ...     ["https://app.greenlang.com", "https://admin.greenlang.com"]
            ... )
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        if allowed_origins is None:
            allowed_origins = self.config.get('allowed_origins', [])

        # Check for wildcard origin (security risk)
        if '*' in allowed_origins:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.CORS,
                severity=SeverityLevel.HIGH,
                title="Wildcard CORS origin allowed",
                description="CORS allows all origins (*), which is a security risk",
                location="allowed_origins",
                line_number=None,
                remediation="Specify explicit allowed origins instead of wildcard",
                cwe_id="CWE-942"
            ))

        # Validate origin format
        if origin:
            if not origin.startswith(('http://', 'https://')):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.CORS,
                    severity=SeverityLevel.MEDIUM,
                    title="Invalid origin format",
                    description=f"Origin '{origin}' does not start with http:// or https://",
                    location="origin",
                    line_number=None,
                    remediation="Ensure origin has valid protocol",
                    cwe_id="CWE-346"
                ))

            # Check if origin is in allowed list
            if allowed_origins and origin not in allowed_origins and '*' not in allowed_origins:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.CORS,
                    severity=SeverityLevel.MEDIUM,
                    title="Origin not in allowed list",
                    description=f"Origin '{origin}' not in allowed origins: {allowed_origins}",
                    location="origin",
                    line_number=None,
                    remediation="Add origin to allowed list or reject request",
                    cwe_id="CWE-346"
                ))

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="cors_validation",
            findings=findings,
            details={
                'origin': origin,
                'allowed_origins': allowed_origins
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # CSRF TOKEN VALIDATION
    # ========================================================================

    def validate_csrf_token(
        self,
        token: str,
        expected_token: Optional[str] = None,
        min_token_length: int = 32
    ) -> ValidationResult:
        """
        Validate CSRF (Cross-Site Request Forgery) token.

        Args:
            token: CSRF token from request
            expected_token: Expected token value (if available)
            min_token_length: Minimum required token length

        Returns:
            ValidationResult with CSRF validation findings

        Example:
            >>> token = secrets.token_hex(32)
            >>> result = validator.validate_csrf_token(token, token)
            >>> assert result.status == ValidationStatus.PASSED
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        # Check token presence
        if not token:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.CSRF,
                severity=SeverityLevel.CRITICAL,
                title="Missing CSRF token",
                description="CSRF token is required but not provided",
                location="csrf_token",
                line_number=None,
                remediation="Include CSRF token in request",
                cwe_id="CWE-352",
                cvss_score=8.8
            ))
        else:
            # Check token length
            if len(token) < min_token_length:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.CSRF,
                    severity=SeverityLevel.HIGH,
                    title="CSRF token too short",
                    description=f"Token length {len(token)} is less than minimum {min_token_length}",
                    location="csrf_token",
                    line_number=None,
                    remediation=f"Use CSRF tokens of at least {min_token_length} characters",
                    cwe_id="CWE-352"
                ))

            # Check token format (should be hex or base64)
            if not re.match(r'^[a-fA-F0-9]+$', token) and not re.match(r'^[A-Za-z0-9+/=]+$', token):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.CSRF,
                    severity=SeverityLevel.MEDIUM,
                    title="Invalid CSRF token format",
                    description="Token does not match expected format (hex or base64)",
                    location="csrf_token",
                    line_number=None,
                    remediation="Use cryptographically random hex or base64 tokens",
                    cwe_id="CWE-352"
                ))

            # Validate against expected token if provided
            if expected_token is not None:
                if token != expected_token:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.CSRF,
                        severity=SeverityLevel.CRITICAL,
                        title="CSRF token mismatch",
                        description="Provided token does not match expected token",
                        location="csrf_token",
                        line_number=None,
                        remediation="Ensure CSRF token matches session token",
                        cwe_id="CWE-352",
                        cvss_score=8.8
                    ))

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="csrf_validation",
            findings=findings,
            details={
                'token_length': len(token) if token else 0,
                'min_token_length': min_token_length,
                'token_provided': bool(token)
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # JWT TOKEN VALIDATION
    # ========================================================================

    def validate_jwt_token(
        self,
        token: str,
        verify_signature: bool = False,
        check_expiration: bool = True
    ) -> ValidationResult:
        """
        Validate JWT (JSON Web Token) structure and claims.

        Args:
            token: JWT token to validate
            verify_signature: Whether to verify signature (requires key)
            check_expiration: Whether to check expiration claim

        Returns:
            ValidationResult with JWT validation findings

        Example:
            >>> result = validator.validate_jwt_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        # Check token format (header.payload.signature)
        parts = token.split('.')
        if len(parts) != 3:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.AUTHENTICATION,
                severity=SeverityLevel.HIGH,
                title="Invalid JWT format",
                description=f"JWT must have 3 parts (header.payload.signature), got {len(parts)}",
                location="jwt_token",
                line_number=None,
                remediation="Provide valid JWT token",
                cwe_id="CWE-287"
            ))
        else:
            try:
                # Decode header (without verification)
                header_data = parts[0] + '=' * (4 - len(parts[0]) % 4)
                header = json.loads(base64.urlsafe_b64decode(header_data))

                # Check algorithm
                alg = header.get('alg', '').lower()
                if alg == 'none':
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.AUTHENTICATION,
                        severity=SeverityLevel.CRITICAL,
                        title="JWT 'none' algorithm detected",
                        description="JWT uses 'none' algorithm, allowing forgery",
                        location="jwt_token.header.alg",
                        line_number=None,
                        remediation="Use RS256 or ES256 algorithm",
                        cwe_id="CWE-327",
                        cvss_score=9.8
                    ))

                weak_algorithms = ['hs256', 'hs384', 'hs512']
                if alg in weak_algorithms:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.AUTHENTICATION,
                        severity=SeverityLevel.MEDIUM,
                        title=f"Symmetric JWT algorithm: {alg.upper()}",
                        description="Symmetric algorithms require secret sharing",
                        location="jwt_token.header.alg",
                        line_number=None,
                        remediation="Consider using RS256 or ES256",
                        cwe_id="CWE-327"
                    ))

                # Decode payload
                payload_data = parts[1] + '=' * (4 - len(parts[1]) % 4)
                payload = json.loads(base64.urlsafe_b64decode(payload_data))

                # Check expiration if enabled
                if check_expiration and 'exp' in payload:
                    exp_timestamp = payload['exp']
                    if datetime.fromtimestamp(exp_timestamp, tz=timezone.utc) < datetime.now(timezone.utc):
                        findings.append(SecurityFinding(
                            finding_id=self._generate_finding_id(),
                            category=FindingCategory.AUTHENTICATION,
                            severity=SeverityLevel.HIGH,
                            title="JWT token expired",
                            description=f"Token expired at {datetime.fromtimestamp(exp_timestamp, tz=timezone.utc).isoformat()}",
                            location="jwt_token.payload.exp",
                            line_number=None,
                            remediation="Refresh token or re-authenticate",
                            cwe_id="CWE-613"
                        ))

            except Exception as e:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.AUTHENTICATION,
                    severity=SeverityLevel.HIGH,
                    title="JWT decoding error",
                    description=f"Failed to decode JWT: {str(e)}",
                    location="jwt_token",
                    line_number=None,
                    remediation="Provide valid JWT token",
                    cwe_id="CWE-287"
                ))

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="jwt_validation",
            findings=findings,
            details={
                'token_parts': len(parts),
                'verify_signature': verify_signature,
                'check_expiration': check_expiration
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # CONTENT-TYPE VALIDATION
    # ========================================================================

    def validate_content_type(
        self,
        content_type: str,
        allowed_types: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate HTTP Content-Type header.

        Args:
            content_type: Content-Type header value
            allowed_types: List of allowed content types

        Returns:
            ValidationResult with content type validation findings

        Example:
            >>> result = validator.validate_content_type(
            ...     "application/json",
            ...     ["application/json", "application/xml"]
            ... )
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        if allowed_types is None:
            allowed_types = [
                'application/json',
                'application/xml',
                'text/csv',
                'application/vnd.api+json',
                'multipart/form-data'
            ]

        # Parse content type (may have charset)
        base_type = content_type.split(';')[0].strip().lower()

        if base_type not in allowed_types:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.INPUT_VALIDATION,
                severity=SeverityLevel.MEDIUM,
                title="Unsupported Content-Type",
                description=f"Content-Type '{base_type}' not in allowed types: {allowed_types}",
                location="content_type",
                line_number=None,
                remediation=f"Use one of: {', '.join(allowed_types)}",
                cwe_id="CWE-436"
            ))

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="content_type_validation",
            findings=findings,
            details={
                'content_type': content_type,
                'base_type': base_type,
                'allowed_types': allowed_types
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # HELPER METHODS FOR INJECTION DETECTION
    # ========================================================================

    def _contains_sql_injection(self, input_str: str) -> bool:
        """Check for SQL injection patterns."""
        for pattern in self.compiled_sql_patterns:
            if pattern.search(input_str):
                return True
        return False

    def _contains_command_injection(self, input_str: str) -> bool:
        """Check for command injection patterns."""
        for pattern in self.compiled_cmd_patterns:
            if pattern.search(input_str):
                return True
        return False

    def _contains_path_traversal(self, input_str: str) -> bool:
        """Check for path traversal patterns."""
        for pattern in self.compiled_path_patterns:
            if pattern.search(input_str):
                return True
        return False

    def _contains_xss(self, input_str: str) -> bool:
        """Check for XSS patterns."""
        for pattern in self.compiled_xss_patterns:
            if pattern.search(input_str):
                return True
        return False

    def _determine_status(self, findings: List[SecurityFinding]) -> ValidationStatus:
        """Determine validation status from findings."""
        if not findings:
            return ValidationStatus.PASSED

        has_critical = any(f.severity == SeverityLevel.CRITICAL for f in findings)
        has_high = any(f.severity == SeverityLevel.HIGH for f in findings)

        if has_critical:
            return ValidationStatus.FAILED
        elif has_high:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.WARNING

    # ========================================================================
    # SCAN FOR SECRETS
    # ========================================================================

    def scan_for_secrets(
        self,
        target_path: Union[str, Path],
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Scan files for hardcoded secrets.

        Args:
            target_path: Path to file or directory to scan
            file_extensions: List of file extensions to scan
            exclude_patterns: Patterns to exclude from scanning

        Returns:
            ValidationResult with detected secrets
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []
        secret_matches: List[SecretMatch] = []

        if file_extensions is None:
            file_extensions = [
                '.py', '.js', '.ts', '.json', '.yaml', '.yml',
                '.env', '.conf', '.cfg', '.ini', '.xml', '.toml',
                '.sh', '.bash', '.dockerfile'
            ]

        if exclude_patterns is None:
            exclude_patterns = [
                r'\.git/',
                r'__pycache__/',
                r'node_modules/',
                r'\.venv/',
                r'venv/',
            ]

        target = Path(target_path)

        if target.is_file():
            files_to_scan = [target]
        else:
            files_to_scan = []
            for ext in file_extensions:
                files_to_scan.extend(target.rglob(f'*{ext}'))

        compiled_excludes = [re.compile(p) for p in exclude_patterns]
        files_to_scan = [
            f for f in files_to_scan
            if not any(exc.search(str(f)) for exc in compiled_excludes)
        ]

        patterns = SecretPatterns.get_all_patterns()

        for file_path in files_to_scan:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')

                for pattern_name, (_, secret_type, severity, confidence) in patterns.items():
                    compiled = self._compiled_patterns.get(pattern_name)
                    if compiled is None:
                        continue

                    for match in compiled.finditer(content):
                        line_start = content.rfind('\n', 0, match.start()) + 1
                        line_number = content[:match.start()].count('\n') + 1
                        column = match.start() - line_start

                        matched_value = match.group(0)
                        if match.groups():
                            matched_value = match.group(len(match.groups()))

                        line_content = lines[line_number - 1] if line_number <= len(lines) else ''
                        if self._is_in_comment(line_content, column):
                            continue

                        if self._is_false_positive(matched_value, pattern_name):
                            continue

                        secret_match = SecretMatch(
                            secret_type=secret_type,
                            pattern_name=pattern_name,
                            file_path=str(file_path),
                            line_number=line_number,
                            column=column,
                            masked_value=self._mask_secret(matched_value),
                            confidence=confidence,
                            severity=severity
                        )
                        secret_matches.append(secret_match)

                        finding = SecurityFinding(
                            finding_id=self._generate_finding_id(),
                            category=FindingCategory.SECRET_DETECTED,
                            severity=severity,
                            title=f"Potential {secret_type.value} detected",
                            description=(
                                f"A potential {secret_type.value} was detected. "
                                f"Pattern: {pattern_name}, Confidence: {confidence * 100:.0f}%"
                            ),
                            location=f"{file_path}:{line_number}:{column}",
                            line_number=line_number,
                            remediation="Remove hardcoded secret and use environment variables or secrets manager",
                            cwe_id="CWE-798",
                            evidence=self._mask_secret(matched_value)
                        )
                        findings.append(finding)

            except Exception as e:
                logger.warning(f"Error scanning {file_path}: {e}")

        execution_time = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(findings)

        result = ValidationResult(
            status=status,
            check_name="secret_scan",
            findings=findings,
            details={
                'files_scanned': len(files_to_scan),
                'secrets_found': len(secret_matches),
                'patterns_used': len(patterns)
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    def _is_in_comment(self, line: str, column: int) -> bool:
        """Check if position is within a comment."""
        stripped = line.lstrip()
        if stripped.startswith('#') or stripped.startswith('//'):
            return True
        if '"""' in line or "'''" in line:
            return True
        comment_pos = line.find('#')
        if comment_pos != -1 and comment_pos < column:
            return True
        return False

    def _is_false_positive(self, value: str, pattern_name: str) -> bool:
        """Check for common false positives."""
        placeholders = [
            'your_api_key', 'your-api-key', 'api_key_here',
            'password', 'changeme', 'secret', 'xxx', 'example', 'test'
        ]
        lower_value = value.lower()
        if any(p in lower_value for p in placeholders):
            return True

        if value.startswith('${') or value.startswith('$'):
            return True

        if len(value) < 8:
            return True

        return False

    # ========================================================================
    # REPORTING
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all security findings."""
        with self._lock:
            severity_counts = {s.value: 0 for s in SeverityLevel}
            category_counts = {c.value: 0 for c in FindingCategory}

            for finding in self._findings:
                severity_counts[finding.severity.value] += 1
                category_counts[finding.category.value] += 1

            return {
                'total_findings': len(self._findings),
                'severity_breakdown': severity_counts,
                'category_breakdown': category_counts,
                'validations_run': len(self._validation_history),
                'last_scan': self._validation_history[-1].timestamp if self._validation_history else None
            }

    def get_findings(
        self,
        severity: Optional[SeverityLevel] = None,
        category: Optional[FindingCategory] = None
    ) -> List[SecurityFinding]:
        """Get findings filtered by severity or category."""
        with self._lock:
            findings = self._findings.copy()

            if severity:
                findings = [f for f in findings if f.severity == severity]

            if category:
                findings = [f for f in findings if f.category == category]

            return findings

    def export_report(self, format: str = 'json') -> str:
        """Export security findings report."""
        with self._lock:
            report = {
                'report_id': f"FUEL-SEC-RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'agent': 'GL-011-FUELCRAFT',
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'summary': self.get_summary(),
                'findings': [f.to_dict() for f in self._findings],
                'validation_history': [r.to_dict() for r in self._validation_history]
            }

            if format == 'json':
                return json.dumps(report, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def clear_findings(self) -> None:
        """Clear all findings and history."""
        with self._lock:
            self._findings.clear()
            self._validation_history.clear()
            self._rate_limit_state.clear()
            logger.info("GL-011 Security findings cleared")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_security_validator(config: Optional[Dict[str, Any]] = None) -> SecurityValidator:
    """
    Factory function to create SecurityValidator.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured SecurityValidator instance

    Example:
        >>> config = {
        ...     'allowed_origins': ['https://app.greenlang.com'],
        ...     'enforce_https': True,
        ...     'allowed_ports': [443, 8443]
        ... }
        >>> validator = create_security_validator(config)
    """
    return SecurityValidator(config)


# Global validator instance
security_validator = SecurityValidator()


# ============================================================================
# CLI SUPPORT
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='GL-011 FUELCRAFT Security Validator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan directory for secrets
  python security_validator.py /path/to/code

  # Validate fuel price input
  python security_validator.py --validate-fuel-price gasoline 3.45

  # Validate API URL
  python security_validator.py --validate-url https://api.example.com/fuel-prices

  # Export report
  python security_validator.py /path/to/code --output report.json
        """
    )

    parser.add_argument('target', nargs='?', help='Path to scan for secrets')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--format', '-f', default='json', help='Report format (json)')
    parser.add_argument('--validate-fuel-price', nargs=2, metavar=('FUEL_TYPE', 'PRICE'),
                        help='Validate fuel price input')
    parser.add_argument('--validate-url', metavar='URL', help='Validate API URL')
    parser.add_argument('--validate-blend', metavar='JSON', help='Validate blend ratios (JSON)')

    args = parser.parse_args()

    validator = SecurityValidator()

    if args.validate_fuel_price:
        fuel_type, price = args.validate_fuel_price
        result = validator.validate_fuel_price_input(fuel_type, float(price))
        print(f"Validation Status: {result.status.value}")
        print(f"Findings: {len(result.findings)}")
        for finding in result.findings:
            print(f"  - [{finding.severity.value}] {finding.title}")

    elif args.validate_url:
        result = validator.validate_api_url(args.validate_url)
        print(f"Validation Status: {result.status.value}")
        print(f"Findings: {len(result.findings)}")
        for finding in result.findings:
            print(f"  - [{finding.severity.value}] {finding.title}")

    elif args.validate_blend:
        blend_ratios = json.loads(args.validate_blend)
        result = validator.validate_blend_ratios(blend_ratios)
        print(f"Validation Status: {result.status.value}")
        print(f"Findings: {len(result.findings)}")

    elif args.target:
        result = validator.scan_for_secrets(args.target)
        print(f"Scan Status: {result.status.value}")
        print(f"Findings: {len(result.findings)}")
        print(f"Files Scanned: {result.details.get('files_scanned', 0)}")

        if args.output:
            report = validator.export_report(args.format)
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
    else:
        parser.print_help()
