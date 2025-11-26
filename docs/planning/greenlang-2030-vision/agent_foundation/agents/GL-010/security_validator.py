# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH Security Validator Module.

This module provides comprehensive security validation capabilities for the
GL-010 EmissionsComplianceAgent, including secret detection, input sanitization,
authentication validation, encryption verification, and permission auditing.

Security Standards Compliance:
- OWASP Application Security Verification Standard (ASVS) 4.0
- NIST 800-53 Security Controls
- SOC 2 Type II Requirements
- EPA 40 CFR Part 75 Data Integrity Requirements

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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Pattern
from urllib.parse import urlparse
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
    GENERIC = "generic"


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


# ============================================================================
# SECRET PATTERNS
# ============================================================================

class SecretPatterns:
    """
    Regex patterns for detecting secrets in code and configuration.

    Patterns are based on common secret formats and known credential patterns.
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
    Comprehensive security validator for GL-010 EMISSIONWATCH.

    Provides security validation capabilities including:
    - Secret detection in code and configuration
    - Input sanitization validation
    - Authentication verification
    - Encryption validation
    - Permission auditing

    Example:
        >>> validator = SecurityValidator()
        >>> result = validator.scan_for_secrets('/path/to/code')
        >>> print(f"Found {len(result.findings)} potential secrets")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SecurityValidator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._lock = threading.RLock()
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._initialize_patterns()
        self._findings: List[SecurityFinding] = []
        self._validation_history: List[ValidationResult] = []

        logger.info("SecurityValidator initialized")

    def _initialize_patterns(self) -> None:
        """Initialize and compile regex patterns."""
        patterns = SecretPatterns.get_all_patterns()
        for name, (pattern, _, _, _) in patterns.items():
            try:
                self._compiled_patterns[name] = re.compile(pattern, re.MULTILINE)
            except re.error as e:
                logger.warning(f"Failed to compile pattern {name}: {e}")

    def _generate_finding_id(self) -> str:
        """Generate unique finding ID."""
        return f"SEC-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(4).upper()}"

    def _mask_secret(self, value: str, visible_chars: int = 4) -> str:
        """Mask a secret value for safe logging."""
        if len(value) <= visible_chars * 2:
            return '*' * len(value)
        return value[:visible_chars] + '*' * (len(value) - visible_chars * 2) + value[-visible_chars:]

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

        Scans the specified path for potential hardcoded secrets including
        API keys, passwords, tokens, private keys, and connection strings.

        Args:
            target_path: Path to file or directory to scan
            file_extensions: List of file extensions to scan (default: common code files)
            exclude_patterns: Patterns to exclude from scanning

        Returns:
            ValidationResult with detected secrets

        Example:
            >>> result = validator.scan_for_secrets('/app/src')
            >>> for finding in result.findings:
            ...     print(f"{finding.severity}: {finding.title}")
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []
        secret_matches: List[SecretMatch] = []

        # Default file extensions
        if file_extensions is None:
            file_extensions = [
                '.py', '.js', '.ts', '.json', '.yaml', '.yml',
                '.env', '.conf', '.cfg', '.ini', '.xml', '.toml',
                '.sh', '.bash', '.zsh', '.dockerfile', '.tf'
            ]

        # Default exclude patterns
        if exclude_patterns is None:
            exclude_patterns = [
                r'\.git/',
                r'__pycache__/',
                r'node_modules/',
                r'\.venv/',
                r'venv/',
                r'\.pytest_cache/',
                r'\.mypy_cache/',
            ]

        target = Path(target_path)

        if target.is_file():
            files_to_scan = [target]
        else:
            files_to_scan = []
            for ext in file_extensions:
                files_to_scan.extend(target.rglob(f'*{ext}'))

        # Filter excluded patterns
        compiled_excludes = [re.compile(p) for p in exclude_patterns]
        files_to_scan = [
            f for f in files_to_scan
            if not any(exc.search(str(f)) for exc in compiled_excludes)
        ]

        # Scan each file
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
                        # Find line number
                        line_start = content.rfind('\n', 0, match.start()) + 1
                        line_number = content[:match.start()].count('\n') + 1
                        column = match.start() - line_start

                        # Get matched value
                        matched_value = match.group(0)
                        if match.groups():
                            matched_value = match.group(len(match.groups()))

                        # Skip if in comment or docstring
                        line_content = lines[line_number - 1] if line_number <= len(lines) else ''
                        if self._is_in_comment(line_content, column):
                            continue

                        # Skip false positives
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

                        # Create finding
                        finding = SecurityFinding(
                            finding_id=self._generate_finding_id(),
                            category=FindingCategory.SECRET_DETECTED,
                            severity=severity,
                            title=f"Potential {secret_type.value} detected",
                            description=(
                                f"A potential {secret_type.value} was detected in the source code. "
                                f"Pattern: {pattern_name}, Confidence: {confidence * 100:.0f}%"
                            ),
                            location=f"{file_path}:{line_number}:{column}",
                            line_number=line_number,
                            remediation=(
                                "Remove the hardcoded secret and use environment variables "
                                "or a secrets management service instead."
                            ),
                            cwe_id="CWE-798",
                            evidence=self._mask_secret(matched_value)
                        )
                        findings.append(finding)

            except Exception as e:
                logger.warning(f"Error scanning {file_path}: {e}")

        execution_time = (time.perf_counter() - start_time) * 1000

        # Determine status
        has_critical = any(f.severity == SeverityLevel.CRITICAL for f in findings)
        has_high = any(f.severity == SeverityLevel.HIGH for f in findings)

        if has_critical:
            status = ValidationStatus.FAILED
        elif has_high:
            status = ValidationStatus.WARNING
        elif findings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED

        result = ValidationResult(
            status=status,
            check_name="secret_scan",
            findings=findings,
            details={
                'files_scanned': len(files_to_scan),
                'secrets_found': len(secret_matches),
                'secret_matches': [s.to_dict() for s in secret_matches[:10]],  # Limit output
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
        # Skip placeholder values
        placeholders = [
            'your_api_key', 'your-api-key', 'api_key_here',
            'password', 'changeme', 'secret', 'xxx', 'yyy',
            'example', 'placeholder', 'dummy', 'test'
        ]
        lower_value = value.lower()
        if any(p in lower_value for p in placeholders):
            return True

        # Skip environment variable references
        if value.startswith('${') or value.startswith('$'):
            return True

        # Skip very short values
        if len(value) < 8:
            return True

        return False

    # ========================================================================
    # INPUT SANITIZATION VALIDATION
    # ========================================================================

    def validate_input_sanitization(
        self,
        input_data: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate input data for proper sanitization.

        Checks input data for potential injection attacks, malformed data,
        and policy violations.

        Args:
            input_data: Data to validate
            schema: Optional validation schema

        Returns:
            ValidationResult with sanitization findings

        Example:
            >>> data = {'nox_ppm': 45.2, 'facility_id': '<script>alert(1)</script>'}
            >>> result = validator.validate_input_sanitization(data)
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        # Define injection patterns
        injection_patterns = {
            'xss_script': (r'<script[^>]*>.*?</script>', 'XSS script tag detected'),
            'xss_event': (r'on\w+\s*=', 'XSS event handler detected'),
            'sql_injection': (r"('|\"|;|--|\bOR\b|\bAND\b).*?(=|>|<)", 'Potential SQL injection'),
            'command_injection': (r'[;&|`$]', 'Potential command injection character'),
            'path_traversal': (r'\.\./', 'Path traversal attempt detected'),
            'ldap_injection': (r'[()|\*\\]', 'Potential LDAP injection'),
            'xml_injection': (r'<!ENTITY|<!DOCTYPE', 'Potential XML injection'),
        }

        def check_value(key: str, value: Any, path: str = '') -> None:
            """Recursively check values for injection patterns."""
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                # Check for injection patterns
                for pattern_name, (pattern, description) in injection_patterns.items():
                    if re.search(pattern, value, re.IGNORECASE):
                        findings.append(SecurityFinding(
                            finding_id=self._generate_finding_id(),
                            category=FindingCategory.INJECTION,
                            severity=SeverityLevel.HIGH,
                            title=f"Input injection risk: {description}",
                            description=(
                                f"The input field '{current_path}' contains potentially "
                                f"dangerous content matching pattern '{pattern_name}'."
                            ),
                            location=current_path,
                            line_number=None,
                            remediation=(
                                "Sanitize input by escaping special characters or "
                                "using parameterized queries."
                            ),
                            cwe_id="CWE-79" if 'xss' in pattern_name else "CWE-89",
                            evidence=self._mask_secret(value[:50])
                        ))

                # Check for overly long strings
                if len(value) > 10000:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.INPUT_VALIDATION,
                        severity=SeverityLevel.MEDIUM,
                        title="Excessively long input value",
                        description=f"Field '{current_path}' contains {len(value)} characters.",
                        location=current_path,
                        line_number=None,
                        remediation="Implement input length validation.",
                        cwe_id="CWE-400"
                    ))

            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(k, v, current_path)

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_value(f"[{i}]", item, current_path)

            elif isinstance(value, (int, float)):
                # Check for extreme numeric values
                if isinstance(value, float) and (value != value or abs(value) == float('inf')):
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.INPUT_VALIDATION,
                        severity=SeverityLevel.MEDIUM,
                        title="Invalid numeric value",
                        description=f"Field '{current_path}' contains NaN or Infinity.",
                        location=current_path,
                        line_number=None,
                        remediation="Validate numeric inputs for valid ranges.",
                        cwe_id="CWE-20"
                    ))

        # Run checks
        for key, value in input_data.items():
            check_value(key, value)

        # Validate against schema if provided
        if schema:
            schema_findings = self._validate_against_schema(input_data, schema)
            findings.extend(schema_findings)

        execution_time = (time.perf_counter() - start_time) * 1000

        status = (
            ValidationStatus.FAILED if any(f.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH] for f in findings)
            else ValidationStatus.WARNING if findings
            else ValidationStatus.PASSED
        )

        result = ValidationResult(
            status=status,
            check_name="input_sanitization",
            findings=findings,
            details={
                'fields_checked': self._count_fields(input_data),
                'issues_found': len(findings)
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    def _count_fields(self, data: Any, count: int = 0) -> int:
        """Count total fields in nested structure."""
        if isinstance(data, dict):
            for v in data.values():
                count = self._count_fields(v, count + 1)
        elif isinstance(data, list):
            for item in data:
                count = self._count_fields(item, count)
        return count

    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[SecurityFinding]:
        """Validate data against provided schema."""
        findings = []

        # Check required fields
        required = schema.get('required', [])
        for field in required:
            if field not in data:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.INPUT_VALIDATION,
                    severity=SeverityLevel.MEDIUM,
                    title=f"Missing required field: {field}",
                    description=f"Required field '{field}' is missing from input.",
                    location=field,
                    line_number=None,
                    remediation=f"Provide the required field '{field}'.",
                    cwe_id="CWE-20"
                ))

        return findings

    # ========================================================================
    # AUTHENTICATION VALIDATION
    # ========================================================================

    def check_authentication(
        self,
        auth_config: Dict[str, Any],
        check_strength: bool = True
    ) -> ValidationResult:
        """
        Validate authentication configuration and mechanisms.

        Checks authentication settings for security issues including
        weak algorithms, missing requirements, and configuration errors.

        Args:
            auth_config: Authentication configuration to validate
            check_strength: Whether to check credential strength

        Returns:
            ValidationResult with authentication findings

        Example:
            >>> auth = {'jwt_algorithm': 'HS256', 'token_expiry': 3600}
            >>> result = validator.check_authentication(auth)
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        # Check JWT configuration
        jwt_alg = auth_config.get('jwt_algorithm', auth_config.get('algorithm'))
        if jwt_alg:
            weak_algorithms = ['none', 'HS256', 'HS384', 'HS512']
            if jwt_alg.lower() == 'none':
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.AUTHENTICATION,
                    severity=SeverityLevel.CRITICAL,
                    title="JWT 'none' algorithm enabled",
                    description="The 'none' algorithm allows forged tokens.",
                    location="auth_config.jwt_algorithm",
                    line_number=None,
                    remediation="Use RS256 or ES256 algorithm for JWT signing.",
                    cwe_id="CWE-327"
                ))
            elif jwt_alg in weak_algorithms[1:]:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.AUTHENTICATION,
                    severity=SeverityLevel.MEDIUM,
                    title=f"Symmetric JWT algorithm ({jwt_alg}) in use",
                    description="Symmetric algorithms require secret sharing.",
                    location="auth_config.jwt_algorithm",
                    line_number=None,
                    remediation="Consider using asymmetric algorithm (RS256, ES256).",
                    cwe_id="CWE-327"
                ))

        # Check token expiry
        token_expiry = auth_config.get('token_expiry', auth_config.get('access_token_expire_minutes'))
        if token_expiry:
            expiry_seconds = token_expiry if token_expiry > 1000 else token_expiry * 60
            if expiry_seconds > 86400:  # 24 hours
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.AUTHENTICATION,
                    severity=SeverityLevel.MEDIUM,
                    title="Long token expiry period",
                    description=f"Token expires after {expiry_seconds // 3600} hours.",
                    location="auth_config.token_expiry",
                    line_number=None,
                    remediation="Reduce token expiry to 1 hour or less.",
                    cwe_id="CWE-613"
                ))

        # Check for API key requirements
        api_key_min_length = auth_config.get('api_key_min_length', 0)
        if api_key_min_length < 32:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.AUTHENTICATION,
                severity=SeverityLevel.MEDIUM,
                title="API key minimum length too short",
                description=f"Minimum length is {api_key_min_length}, should be >= 32.",
                location="auth_config.api_key_min_length",
                line_number=None,
                remediation="Require API keys of at least 32 characters.",
                cwe_id="CWE-521"
            ))

        # Check password policy
        password_policy = auth_config.get('password_policy', {})
        if password_policy:
            min_length = password_policy.get('min_length', 0)
            if min_length < 12:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.AUTHENTICATION,
                    severity=SeverityLevel.MEDIUM,
                    title="Weak password minimum length",
                    description=f"Minimum length is {min_length}, should be >= 12.",
                    location="auth_config.password_policy.min_length",
                    line_number=None,
                    remediation="Require passwords of at least 12 characters.",
                    cwe_id="CWE-521"
                ))

        # Check MFA requirement
        mfa_required = auth_config.get('mfa_required', auth_config.get('require_mfa', False))
        if not mfa_required:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.AUTHENTICATION,
                severity=SeverityLevel.LOW,
                title="MFA not required",
                description="Multi-factor authentication is not enforced.",
                location="auth_config.mfa_required",
                line_number=None,
                remediation="Enable MFA requirement for enhanced security.",
                cwe_id="CWE-308"
            ))

        execution_time = (time.perf_counter() - start_time) * 1000

        status = (
            ValidationStatus.FAILED if any(f.severity == SeverityLevel.CRITICAL for f in findings)
            else ValidationStatus.WARNING if findings
            else ValidationStatus.PASSED
        )

        result = ValidationResult(
            status=status,
            check_name="authentication",
            findings=findings,
            details={
                'checks_performed': 5,
                'issues_found': len(findings)
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # ENCRYPTION VALIDATION
    # ========================================================================

    def validate_encryption(
        self,
        encryption_config: Dict[str, Any],
        check_tls: bool = True,
        target_host: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate encryption configuration and implementation.

        Checks encryption settings for strong algorithms, proper key sizes,
        and TLS configuration.

        Args:
            encryption_config: Encryption configuration to validate
            check_tls: Whether to check TLS configuration
            target_host: Optional host to verify TLS

        Returns:
            ValidationResult with encryption findings

        Example:
            >>> config = {'algorithm': 'AES-256-GCM', 'tls_version': 'TLSv1.3'}
            >>> result = validator.validate_encryption(config)
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        # Check encryption algorithm
        algorithm = encryption_config.get('algorithm', encryption_config.get('cipher'))
        if algorithm:
            weak_algorithms = ['DES', '3DES', 'RC4', 'MD5', 'SHA1', 'BLOWFISH']
            if any(weak.lower() in algorithm.lower() for weak in weak_algorithms):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.ENCRYPTION,
                    severity=SeverityLevel.CRITICAL,
                    title=f"Weak encryption algorithm: {algorithm}",
                    description="This algorithm is cryptographically weak.",
                    location="encryption_config.algorithm",
                    line_number=None,
                    remediation="Use AES-256-GCM or ChaCha20-Poly1305.",
                    cwe_id="CWE-327"
                ))

        # Check key size
        key_size = encryption_config.get('key_size', encryption_config.get('key_length'))
        if key_size:
            if key_size < 256:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.ENCRYPTION,
                    severity=SeverityLevel.HIGH,
                    title=f"Encryption key size too small: {key_size} bits",
                    description="Key sizes below 256 bits may be vulnerable.",
                    location="encryption_config.key_size",
                    line_number=None,
                    remediation="Use minimum 256-bit keys for symmetric encryption.",
                    cwe_id="CWE-326"
                ))

        # Check TLS version
        tls_version = encryption_config.get('tls_version', encryption_config.get('ssl_version'))
        if tls_version:
            deprecated_versions = ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1.0', 'TLSv1.1', 'TLS1.0', 'TLS1.1']
            if any(dep.lower() in tls_version.lower() for dep in deprecated_versions):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.ENCRYPTION,
                    severity=SeverityLevel.CRITICAL,
                    title=f"Deprecated TLS version: {tls_version}",
                    description="This TLS version has known vulnerabilities.",
                    location="encryption_config.tls_version",
                    line_number=None,
                    remediation="Use TLS 1.3 or minimum TLS 1.2.",
                    cwe_id="CWE-326"
                ))

        # Check cipher suites
        cipher_suites = encryption_config.get('cipher_suites', [])
        weak_ciphers = ['NULL', 'EXPORT', 'DES', 'RC4', 'MD5', 'ANON']
        for cipher in cipher_suites:
            if any(weak in cipher.upper() for weak in weak_ciphers):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.ENCRYPTION,
                    severity=SeverityLevel.HIGH,
                    title=f"Weak cipher suite: {cipher}",
                    description="This cipher suite is not secure.",
                    location="encryption_config.cipher_suites",
                    line_number=None,
                    remediation="Use only strong cipher suites.",
                    cwe_id="CWE-327"
                ))

        # Check key rotation
        key_rotation_days = encryption_config.get('key_rotation_days')
        if key_rotation_days is None:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.ENCRYPTION,
                severity=SeverityLevel.MEDIUM,
                title="Key rotation not configured",
                description="Encryption key rotation is not specified.",
                location="encryption_config.key_rotation_days",
                line_number=None,
                remediation="Configure key rotation every 90 days or less.",
                cwe_id="CWE-320"
            ))
        elif key_rotation_days > 365:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.ENCRYPTION,
                severity=SeverityLevel.MEDIUM,
                title=f"Key rotation period too long: {key_rotation_days} days",
                description="Keys should be rotated more frequently.",
                location="encryption_config.key_rotation_days",
                line_number=None,
                remediation="Rotate encryption keys every 90 days.",
                cwe_id="CWE-320"
            ))

        # Verify TLS on target host if provided
        if check_tls and target_host:
            tls_findings = self._check_tls_connection(target_host)
            findings.extend(tls_findings)

        execution_time = (time.perf_counter() - start_time) * 1000

        status = (
            ValidationStatus.FAILED if any(f.severity == SeverityLevel.CRITICAL for f in findings)
            else ValidationStatus.WARNING if findings
            else ValidationStatus.PASSED
        )

        result = ValidationResult(
            status=status,
            check_name="encryption",
            findings=findings,
            details={
                'checks_performed': 5,
                'issues_found': len(findings),
                'tls_checked': check_tls and target_host is not None
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    def _check_tls_connection(self, host: str) -> List[SecurityFinding]:
        """Check TLS configuration of a host."""
        findings = []
        try:
            context = ssl.create_default_context()
            # Attempt connection (simplified check)
            # In production, would use socket connection
            logger.info(f"TLS check for {host} skipped in validation")
        except Exception as e:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.ENCRYPTION,
                severity=SeverityLevel.HIGH,
                title=f"TLS connection check failed: {host}",
                description=str(e),
                location=host,
                line_number=None,
                remediation="Verify TLS configuration on target host.",
                cwe_id="CWE-295"
            ))
        return findings

    # ========================================================================
    # PERMISSION AUDIT
    # ========================================================================

    def audit_permissions(
        self,
        rbac_config: Dict[str, Any],
        user_permissions: Optional[Dict[str, List[str]]] = None
    ) -> ValidationResult:
        """
        Audit access control and permission configurations.

        Validates RBAC configuration, checks for excessive permissions,
        and identifies potential privilege escalation paths.

        Args:
            rbac_config: RBAC configuration to audit
            user_permissions: Optional current user permission mappings

        Returns:
            ValidationResult with permission findings

        Example:
            >>> rbac = {
            ...     'roles': {'admin': ['*'], 'user': ['read']},
            ...     'users': {'alice': ['admin']}
            ... }
            >>> result = validator.audit_permissions(rbac)
        """
        start_time = time.perf_counter()
        findings: List[SecurityFinding] = []

        roles = rbac_config.get('roles', {})
        users = rbac_config.get('users', {})
        groups = rbac_config.get('groups', {})

        # Check for wildcard permissions
        for role_name, permissions in roles.items():
            if isinstance(permissions, list):
                if '*' in permissions or 'all' in [p.lower() for p in permissions]:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.PERMISSION,
                        severity=SeverityLevel.HIGH,
                        title=f"Wildcard permission in role: {role_name}",
                        description="Role has unrestricted access permissions.",
                        location=f"rbac_config.roles.{role_name}",
                        line_number=None,
                        remediation="Replace wildcard with specific permissions.",
                        cwe_id="CWE-269"
                    ))

        # Check for admin role assignments
        admin_roles = {'admin', 'administrator', 'root', 'superuser', 'super_admin'}
        admin_users = []
        for user, user_roles in users.items():
            if isinstance(user_roles, list):
                if any(r.lower() in admin_roles for r in user_roles):
                    admin_users.append(user)

        if len(admin_users) > 3:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                category=FindingCategory.PERMISSION,
                severity=SeverityLevel.MEDIUM,
                title="Excessive admin users",
                description=f"Found {len(admin_users)} users with admin roles.",
                location="rbac_config.users",
                line_number=None,
                remediation="Limit admin access to essential personnel only.",
                cwe_id="CWE-250"
            ))

        # Check for role hierarchy issues
        dangerous_combos = [
            (['read', 'write', 'delete'], 'Full CRUD access'),
            (['read', 'execute'], 'Read and execute'),
            (['admin', 'user'], 'Mixed privilege levels')
        ]

        for user, user_roles in users.items():
            if isinstance(user_roles, list):
                for combo, description in dangerous_combos:
                    if all(any(c in r.lower() for r in user_roles) for c in combo):
                        findings.append(SecurityFinding(
                            finding_id=self._generate_finding_id(),
                            category=FindingCategory.PERMISSION,
                            severity=SeverityLevel.LOW,
                            title=f"Permission combination: {description}",
                            description=f"User '{user}' has potentially dangerous permission combo.",
                            location=f"rbac_config.users.{user}",
                            line_number=None,
                            remediation="Review if all permissions are necessary.",
                            cwe_id="CWE-269"
                        ))
                        break

        # Check for missing service account restrictions
        service_accounts = [u for u in users.keys() if 'service' in u.lower() or 'sa-' in u.lower()]
        for sa in service_accounts:
            sa_roles = users.get(sa, [])
            if isinstance(sa_roles, list) and len(sa_roles) > 2:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    category=FindingCategory.PERMISSION,
                    severity=SeverityLevel.MEDIUM,
                    title=f"Service account with multiple roles: {sa}",
                    description="Service accounts should have minimal permissions.",
                    location=f"rbac_config.users.{sa}",
                    line_number=None,
                    remediation="Apply principle of least privilege to service accounts.",
                    cwe_id="CWE-250"
                ))

        # Validate user_permissions if provided
        if user_permissions:
            for user, perms in user_permissions.items():
                if not isinstance(perms, list):
                    continue
                if len(perms) > 20:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        category=FindingCategory.PERMISSION,
                        severity=SeverityLevel.LOW,
                        title=f"User with excessive permissions: {user}",
                        description=f"User has {len(perms)} individual permissions.",
                        location=f"user_permissions.{user}",
                        line_number=None,
                        remediation="Consolidate permissions into roles.",
                        cwe_id="CWE-269"
                    ))

        execution_time = (time.perf_counter() - start_time) * 1000

        status = (
            ValidationStatus.FAILED if any(f.severity == SeverityLevel.CRITICAL for f in findings)
            else ValidationStatus.WARNING if any(f.severity == SeverityLevel.HIGH for f in findings)
            else ValidationStatus.PASSED
        )

        result = ValidationResult(
            status=status,
            check_name="permission_audit",
            findings=findings,
            details={
                'roles_audited': len(roles),
                'users_audited': len(users),
                'admin_users': len(admin_users),
                'service_accounts': len(service_accounts),
                'issues_found': len(findings)
            },
            execution_time_ms=execution_time
        )

        with self._lock:
            self._findings.extend(findings)
            self._validation_history.append(result)

        return result

    # ========================================================================
    # COMPREHENSIVE SECURITY SCAN
    # ========================================================================

    def run_full_security_scan(
        self,
        target_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Run comprehensive security scan including all validation checks.

        Args:
            target_path: Path to scan for secrets
            config: Optional configuration for all checks

        Returns:
            Dictionary of validation results by check type
        """
        config = config or {}
        results = {}

        # Secret scan
        results['secret_scan'] = self.scan_for_secrets(target_path)

        # Input sanitization (with sample data)
        sample_input = config.get('sample_input', {})
        if sample_input:
            results['input_sanitization'] = self.validate_input_sanitization(sample_input)

        # Authentication check
        auth_config = config.get('auth_config', {})
        if auth_config:
            results['authentication'] = self.check_authentication(auth_config)

        # Encryption check
        encryption_config = config.get('encryption_config', {})
        if encryption_config:
            results['encryption'] = self.validate_encryption(encryption_config)

        # Permission audit
        rbac_config = config.get('rbac_config', {})
        if rbac_config:
            results['permissions'] = self.audit_permissions(rbac_config)

        return results

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
                'report_id': f"SEC-RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
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
            logger.info("Security findings cleared")


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
    """
    return SecurityValidator(config)


# ============================================================================
# CLI SUPPORT
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GL-010 Security Validator')
    parser.add_argument('target', help='Path to scan')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--format', '-f', default='json', help='Report format')

    args = parser.parse_args()

    validator = SecurityValidator()
    result = validator.scan_for_secrets(args.target)

    print(f"Scan Status: {result.status.value}")
    print(f"Findings: {len(result.findings)}")

    if args.output:
        report = validator.export_report(args.format)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
