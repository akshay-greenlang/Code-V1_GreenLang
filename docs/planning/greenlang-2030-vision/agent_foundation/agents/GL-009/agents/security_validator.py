# -*- coding: utf-8 -*-
"""
GL-009 THERMALIQ - Security Validator Module.

Validates security configurations and enforces zero-secrets policy for
GL-009 THERMALIQ ThermalEfficiencyCalculator agent.

Standards Compliance:
- IEC 62443-4-2 (Industrial Automation Security)
- OWASP Top 10 (2023)
- NIST Cybersecurity Framework
- CIS Benchmarks

Security Features:
- Secret pattern detection (API keys, passwords, tokens)
- Configuration security validation
- JWT token validation
- Input sanitization
- Rate limit enforcement
- TLS/SSL validation
- RBAC enforcement checks

Author: GreenLang Foundation Security Team
Version: 1.0.0
Last Updated: 2025-11-26
"""

import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union
from urllib.parse import urlparse

try:
    import jwt
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
except ImportError:
    jwt = None
    hashes = None
    PBKDF2HMAC = None
    x509 = None
    default_backend = None

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SecurityLevel(Enum):
    """Security severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class SecretType(Enum):
    """Types of secrets to detect."""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"
    AWS_KEY = "aws_key"
    DATABASE_URL = "database_url"
    JWT = "jwt"
    GENERIC_SECRET = "generic_secret"


# ============================================================================
# RESULT DATA CLASSES
# ============================================================================

@dataclass
class SecurityFinding:
    """Security validation finding."""
    finding_id: str
    severity: SecurityLevel
    category: str
    title: str
    description: str
    affected_resource: str
    line_number: Optional[int] = None
    remediation: str = ""
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'finding_id': self.finding_id,
            'severity': self.severity.value,
            'category': self.category,
            'title': self.title,
            'description': self.description,
            'affected_resource': self.affected_resource,
            'line_number': self.line_number,
            'remediation': self.remediation,
            'cve_id': self.cve_id,
            'cvss_score': self.cvss_score,
            'references': self.references
        }


@dataclass
class SecurityValidationResult:
    """Security validation result."""
    is_valid: bool
    status: ValidationStatus
    score: int  # 0-100
    findings: List[SecurityFinding]
    warnings: List[str]
    timestamp: str
    validator_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'status': self.status.value,
            'score': self.score,
            'findings': [f.to_dict() for f in self.findings],
            'warnings': self.warnings,
            'timestamp': self.timestamp,
            'validator_version': self.validator_version
        }


@dataclass
class SecretScanResult:
    """Secret scanning result."""
    secrets_found: int
    secret_locations: List[Dict[str, Any]]
    scan_time_ms: float
    files_scanned: int
    is_clean: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'secrets_found': self.secrets_found,
            'secret_locations': self.secret_locations,
            'scan_time_ms': self.scan_time_ms,
            'files_scanned': self.files_scanned,
            'is_clean': self.is_clean
        }


# ============================================================================
# SECURITY VALIDATOR CLASS
# ============================================================================

class SecurityValidator:
    """
    Security validator for GL-009 THERMALIQ.

    Validates security configurations and enforces zero-secrets policy.

    Features:
    - Secret pattern detection
    - Configuration security validation
    - JWT token validation
    - Input sanitization
    - Rate limit checking
    - TLS/SSL validation

    Example:
        >>> validator = SecurityValidator()
        >>> result = validator.validate_no_secrets(code_content)
        >>> if not result.is_valid:
        ...     print(f"Security issues found: {result.findings}")
    """

    # Patterns for detecting various types of secrets
    SECRET_PATTERNS: Dict[SecretType, List[Pattern]] = {
        SecretType.API_KEY: [
            re.compile(r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})'),
            re.compile(r'(?i)(anthropic|openai)[_-]?api[_-]?key\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})'),
            re.compile(r'(sk-[a-zA-Z0-9]{20,})'),  # OpenAI/Anthropic style keys
            re.compile(r'(gl_sk_[a-zA-Z0-9_]{20,})'),  # GreenLang API keys
        ],
        SecretType.PASSWORD: [
            re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\'\s]{8,})'),
            re.compile(r'(?i)(db_password|database_password)\s*[=:]\s*["\']([^"\'\s]{8,})'),
        ],
        SecretType.TOKEN: [
            re.compile(r'(?i)(access[_-]?token|auth[_-]?token|bearer[_-]?token)\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{20,})'),
            re.compile(r'(?i)bearer\s+([a-zA-Z0-9_\-\.]{20,})'),
            re.compile(r'(ghp_[a-zA-Z0-9]{36})'),  # GitHub personal access token
        ],
        SecretType.PRIVATE_KEY: [
            re.compile(r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'),
            re.compile(r'-----BEGIN ENCRYPTED PRIVATE KEY-----'),
        ],
        SecretType.AWS_KEY: [
            re.compile(r'(AKIA[0-9A-Z]{16})'),  # AWS Access Key ID
            re.compile(r'(?i)(aws[_-]?access[_-]?key|aws[_-]?secret)\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})'),
        ],
        SecretType.DATABASE_URL: [
            re.compile(r'(postgres|mysql|mongodb):\/\/[^:]+:[^@]+@'),
            re.compile(r'(?i)(database_url|db_url)\s*[=:]\s*["\']([^"\']+:[^"\']+@)'),
        ],
        SecretType.JWT: [
            re.compile(r'eyJ[a-zA-Z0-9_\-]*\.eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*'),
        ],
        SecretType.GENERIC_SECRET: [
            re.compile(r'(?i)(secret|credentials?|creds)\s*[=:]\s*["\']([a-zA-Z0-9_\-\.]{20,})'),
        ],
    }

    # Patterns that are false positives (safe to ignore)
    FALSE_POSITIVE_PATTERNS: List[Pattern] = [
        re.compile(r'example|sample|test|demo|placeholder|your[_-]?api[_-]?key', re.IGNORECASE),
        re.compile(r'xxx+|yyy+|zzz+|aaa+', re.IGNORECASE),
        re.compile(r'<api[_-]?key>|<password>|<token>', re.IGNORECASE),
        re.compile(r'\{\{.*?\}\}|\$\{.*?\}'),  # Template variables
    ]

    def __init__(
        self,
        config: Optional[Any] = None,
        enable_strict_mode: bool = True
    ):
        """
        Initialize SecurityValidator.

        Args:
            config: Optional configuration object
            enable_strict_mode: Enable strict validation (fail on warnings)
        """
        self.config = config
        self.enable_strict_mode = enable_strict_mode
        self.findings: List[SecurityFinding] = []

        logger.info("SecurityValidator initialized (strict_mode=%s)", enable_strict_mode)

    # ========================================================================
    # SECRET DETECTION
    # ========================================================================

    def validate_no_secrets(
        self,
        code: str,
        filename: Optional[str] = None
    ) -> SecurityValidationResult:
        """
        Validate that code contains no hardcoded secrets.

        Scans code for patterns matching API keys, passwords, tokens,
        private keys, and other sensitive credentials.

        Args:
            code: Code content to scan
            filename: Optional filename for context

        Returns:
            SecurityValidationResult with findings
        """
        findings = []
        warnings = []

        # Scan for each secret type
        for secret_type, patterns in self.SECRET_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(code):
                    # Check if it's a false positive
                    matched_text = match.group(0)
                    if self._is_false_positive(matched_text):
                        continue

                    # Calculate line number
                    line_number = code[:match.start()].count('\n') + 1

                    # Mask the secret value for logging
                    secret_value = match.group(1) if match.groups() else match.group(0)
                    masked_value = self._mask_secret(secret_value)

                    finding = SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        severity=SecurityLevel.CRITICAL,
                        category="secrets",
                        title=f"Hardcoded {secret_type.value.replace('_', ' ').title()} Detected",
                        description=f"Potential {secret_type.value} found: {masked_value}",
                        affected_resource=filename or "code",
                        line_number=line_number,
                        remediation=(
                            f"Remove hardcoded {secret_type.value} and use environment variables "
                            "or Kubernetes Secrets with External Secrets Operator. "
                            "See SECURITY_POLICY.md for approved secret storage methods."
                        ),
                        references=[
                            "https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html",
                            "https://kubernetes.io/docs/concepts/configuration/secret/",
                        ]
                    )
                    findings.append(finding)

        # Calculate validation score
        if findings:
            score = 0  # Zero tolerance for secrets
            is_valid = False
            status = ValidationStatus.FAILED
        else:
            score = 100
            is_valid = True
            status = ValidationStatus.PASSED

        return SecurityValidationResult(
            is_valid=is_valid,
            status=status,
            score=score,
            findings=findings,
            warnings=warnings,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def scan_directory_for_secrets(
        self,
        directory_path: Union[str, Path],
        file_patterns: List[str] = None
    ) -> SecretScanResult:
        """
        Scan directory recursively for secrets.

        Args:
            directory_path: Path to directory to scan
            file_patterns: Glob patterns for files to scan (default: *.py, *.yaml, *.json)

        Returns:
            SecretScanResult with scan summary
        """
        start_time = datetime.now()
        directory = Path(directory_path)

        if file_patterns is None:
            file_patterns = ['*.py', '*.yaml', '*.yml', '*.json', '*.env', '*.conf']

        secret_locations = []
        files_scanned = 0

        for pattern in file_patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        result = self.validate_no_secrets(content, str(file_path))
                        files_scanned += 1

                        if not result.is_valid:
                            for finding in result.findings:
                                secret_locations.append({
                                    'file': str(file_path),
                                    'line': finding.line_number,
                                    'type': finding.title,
                                    'severity': finding.severity.value
                                })
                    except Exception as e:
                        logger.warning(f"Failed to scan {file_path}: {e}")

        scan_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return SecretScanResult(
            secrets_found=len(secret_locations),
            secret_locations=secret_locations,
            scan_time_ms=scan_time_ms,
            files_scanned=files_scanned,
            is_clean=len(secret_locations) == 0
        )

    def _is_false_positive(self, text: str) -> bool:
        """Check if detected secret is a false positive."""
        for pattern in self.FALSE_POSITIVE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _mask_secret(self, secret: str, visible_chars: int = 4) -> str:
        """Mask secret value for safe logging."""
        if len(secret) <= visible_chars * 2:
            return '*' * len(secret)
        return f"{secret[:visible_chars]}...{secret[-visible_chars:]}"

    # ========================================================================
    # CONFIGURATION SECURITY VALIDATION
    # ========================================================================

    def validate_config_security(
        self,
        config: Dict[str, Any]
    ) -> SecurityValidationResult:
        """
        Validate configuration security settings.

        Checks:
        - TLS enabled
        - Authentication required
        - Strong encryption
        - Secure defaults
        - No credentials in URLs

        Args:
            config: Configuration dictionary

        Returns:
            SecurityValidationResult with findings
        """
        findings = []
        warnings = []

        # Check TLS configuration
        if not config.get('tls_enabled', False):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                severity=SecurityLevel.CRITICAL,
                category="configuration",
                title="TLS Not Enabled",
                description="TLS/SSL encryption is not enabled for network communication",
                affected_resource="tls_enabled",
                remediation="Enable TLS 1.3 for all network communication. Set tls_enabled=true.",
                references=["https://www.rfc-editor.org/rfc/rfc8446"]
            ))

        # Check authentication requirement
        if not config.get('auth_required', True):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                severity=SecurityLevel.CRITICAL,
                category="configuration",
                title="Authentication Not Required",
                description="API endpoints do not require authentication",
                affected_resource="auth_required",
                remediation="Enable authentication for all API endpoints. Set auth_required=true.",
                references=["https://owasp.org/www-project-api-security/"]
            ))

        # Check rate limiting
        if not config.get('rate_limiting_enabled', False):
            warnings.append(
                "Rate limiting not enabled. Consider enabling to prevent DDoS attacks."
            )

        # Check audit logging
        if not config.get('enable_audit_logging', True):
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                severity=SecurityLevel.HIGH,
                category="configuration",
                title="Audit Logging Disabled",
                description="Audit logging is disabled, preventing security monitoring",
                affected_resource="enable_audit_logging",
                remediation="Enable audit logging for compliance and security monitoring.",
                references=["https://www.iec62443.com/"]
            ))

        # Check for credentials in URLs
        for key, value in config.items():
            if isinstance(value, str) and ('://' in value):
                parsed = urlparse(value)
                if parsed.username or parsed.password:
                    findings.append(SecurityFinding(
                        finding_id=self._generate_finding_id(),
                        severity=SecurityLevel.CRITICAL,
                        category="secrets",
                        title="Credentials in URL",
                        description=f"URL '{key}' contains embedded credentials",
                        affected_resource=key,
                        remediation=(
                            "Remove credentials from URL. Use environment variables or "
                            "External Secrets Operator for credential management."
                        ),
                        references=["https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html"]
                    ))

        # Calculate score
        critical_count = sum(1 for f in findings if f.severity == SecurityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SecurityLevel.HIGH)

        score = 100 - (critical_count * 30) - (high_count * 15) - (len(warnings) * 5)
        score = max(0, score)

        is_valid = len([f for f in findings if f.severity in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]]) == 0
        status = ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED

        return SecurityValidationResult(
            is_valid=is_valid,
            status=status,
            score=score,
            findings=findings,
            warnings=warnings,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    # ========================================================================
    # JWT TOKEN VALIDATION
    # ========================================================================

    def validate_jwt_token(
        self,
        token: str,
        secret_key: str,
        algorithm: str = "HS256",
        audience: Optional[str] = None,
        issuer: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate JWT token.

        Args:
            token: JWT token string
            secret_key: Secret key for validation
            algorithm: JWT algorithm (default: HS256)
            audience: Expected audience claim
            issuer: Expected issuer claim

        Returns:
            Tuple of (is_valid, payload, error_message)
        """
        if jwt is None:
            return False, None, "PyJWT library not available"

        try:
            # Validate token
            payload = jwt.decode(
                token,
                secret_key,
                algorithms=[algorithm],
                audience=audience,
                issuer=issuer,
                options={
                    'verify_signature': True,
                    'verify_exp': True,
                    'verify_nbf': True,
                    'verify_iat': True,
                    'verify_aud': audience is not None,
                    'verify_iss': issuer is not None,
                }
            )

            # Additional validation
            if 'exp' not in payload:
                return False, None, "Token missing expiration claim"

            if 'iat' not in payload:
                return False, None, "Token missing issued-at claim"

            # Check if token is expired
            exp_timestamp = payload['exp']
            if datetime.fromtimestamp(exp_timestamp, tz=timezone.utc) < datetime.now(timezone.utc):
                return False, None, "Token has expired"

            return True, payload, None

        except jwt.ExpiredSignatureError:
            return False, None, "Token has expired"
        except jwt.InvalidAudienceError:
            return False, None, "Invalid audience"
        except jwt.InvalidIssuerError:
            return False, None, "Invalid issuer"
        except jwt.InvalidSignatureError:
            return False, None, "Invalid signature"
        except jwt.DecodeError as e:
            return False, None, f"Token decode error: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"

    # ========================================================================
    # INPUT SANITIZATION
    # ========================================================================

    def sanitize_input(
        self,
        user_input: str,
        max_length: int = 1000,
        allow_html: bool = False,
        allow_sql: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Sanitize user input to prevent injection attacks.

        Args:
            user_input: Raw user input
            max_length: Maximum allowed length
            allow_html: Allow HTML tags
            allow_sql: Allow SQL keywords (dangerous!)

        Returns:
            Tuple of (sanitized_input, warnings)
        """
        warnings = []
        sanitized = user_input

        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            warnings.append(f"Input truncated to {max_length} characters")

        # SQL injection prevention
        if not allow_sql:
            sql_keywords = [
                'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
                'ALTER', 'EXEC', 'EXECUTE', 'UNION', 'DECLARE', '--', ';'
            ]
            for keyword in sql_keywords:
                if keyword.upper() in sanitized.upper():
                    warnings.append(f"Potential SQL injection attempt detected: {keyword}")
                    # Remove SQL keywords
                    sanitized = re.sub(
                        rf'\b{re.escape(keyword)}\b',
                        '',
                        sanitized,
                        flags=re.IGNORECASE
                    )

        # XSS prevention
        if not allow_html:
            # Remove HTML tags
            sanitized = re.sub(r'<[^>]+>', '', sanitized)

            # Remove JavaScript event handlers
            sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)

            # Remove script tags
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)

        # Command injection prevention
        shell_chars = ['|', ';', '&', '$', '`', '\n', '\r']
        for char in shell_chars:
            if char in sanitized:
                warnings.append(f"Potential command injection character detected: {repr(char)}")
                sanitized = sanitized.replace(char, '')

        # Path traversal prevention
        if '../' in sanitized or '..\\' in sanitized:
            warnings.append("Potential path traversal attempt detected")
            sanitized = sanitized.replace('../', '').replace('..\\', '')

        return sanitized, warnings

    # ========================================================================
    # RATE LIMITING
    # ========================================================================

    def check_rate_limit(
        self,
        client_id: str,
        limit: int,
        window_seconds: int,
        request_cache: Optional[Dict[str, List[datetime]]] = None
    ) -> Tuple[bool, int]:
        """
        Check if client exceeds rate limit.

        Args:
            client_id: Client identifier (IP, API key, etc.)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            request_cache: Optional cache of recent requests

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        if request_cache is None:
            # In-memory cache (for demo; use Redis in production)
            request_cache = {}

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=window_seconds)

        # Get recent requests for this client
        if client_id not in request_cache:
            request_cache[client_id] = []

        # Filter to requests within the window
        recent_requests = [
            req_time for req_time in request_cache[client_id]
            if req_time > window_start
        ]

        # Check if limit exceeded
        if len(recent_requests) >= limit:
            return False, 0

        # Add current request
        recent_requests.append(now)
        request_cache[client_id] = recent_requests

        remaining = limit - len(recent_requests)
        return True, remaining

    # ========================================================================
    # TLS/SSL VALIDATION
    # ========================================================================

    def validate_tls_certificate(
        self,
        cert_path: Union[str, Path],
        min_key_size: int = 2048,
        max_age_days: int = 90
    ) -> SecurityValidationResult:
        """
        Validate TLS certificate security.

        Args:
            cert_path: Path to certificate file
            min_key_size: Minimum key size in bits
            max_age_days: Maximum certificate age before renewal

        Returns:
            SecurityValidationResult with findings
        """
        findings = []
        warnings = []

        if x509 is None or default_backend is None:
            return SecurityValidationResult(
                is_valid=False,
                status=ValidationStatus.FAILED,
                score=0,
                findings=[SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    severity=SecurityLevel.CRITICAL,
                    category="configuration",
                    title="Cryptography Library Not Available",
                    description="Cannot validate TLS certificate without cryptography library",
                    affected_resource=str(cert_path),
                    remediation="Install cryptography library: pip install cryptography"
                )],
                warnings=[],
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        try:
            # Read certificate
            cert_path = Path(cert_path)
            cert_data = cert_path.read_bytes()
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())

            # Check expiration
            now = datetime.now(timezone.utc)
            if cert.not_valid_after < now:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    severity=SecurityLevel.CRITICAL,
                    category="tls",
                    title="Certificate Expired",
                    description=f"Certificate expired on {cert.not_valid_after}",
                    affected_resource=str(cert_path),
                    remediation="Renew TLS certificate immediately"
                ))
            elif cert.not_valid_after < now + timedelta(days=max_age_days):
                days_remaining = (cert.not_valid_after - now).days
                warnings.append(
                    f"Certificate expires in {days_remaining} days. Consider renewal."
                )

            # Check key size
            public_key = cert.public_key()
            key_size = public_key.key_size
            if key_size < min_key_size:
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    severity=SecurityLevel.HIGH,
                    category="tls",
                    title="Weak Certificate Key Size",
                    description=f"Certificate key size ({key_size} bits) is below minimum ({min_key_size} bits)",
                    affected_resource=str(cert_path),
                    remediation=f"Use at least {min_key_size}-bit RSA or 256-bit ECC keys"
                ))

            # Check signature algorithm
            sig_alg = cert.signature_algorithm_oid._name
            weak_algorithms = ['sha1', 'md5']
            if any(alg in sig_alg.lower() for alg in weak_algorithms):
                findings.append(SecurityFinding(
                    finding_id=self._generate_finding_id(),
                    severity=SecurityLevel.HIGH,
                    category="tls",
                    title="Weak Signature Algorithm",
                    description=f"Certificate uses weak signature algorithm: {sig_alg}",
                    affected_resource=str(cert_path),
                    remediation="Use SHA-256 or stronger signature algorithm"
                ))

        except Exception as e:
            findings.append(SecurityFinding(
                finding_id=self._generate_finding_id(),
                severity=SecurityLevel.HIGH,
                category="tls",
                title="Certificate Validation Failed",
                description=f"Failed to validate certificate: {str(e)}",
                affected_resource=str(cert_path),
                remediation="Ensure certificate is valid PEM format"
            ))

        # Calculate score
        critical_count = sum(1 for f in findings if f.severity == SecurityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SecurityLevel.HIGH)

        score = 100 - (critical_count * 40) - (high_count * 20)
        score = max(0, score)

        is_valid = critical_count == 0 and high_count == 0
        status = ValidationStatus.PASSED if is_valid else ValidationStatus.FAILED

        return SecurityValidationResult(
            is_valid=is_valid,
            status=status,
            score=score,
            findings=findings,
            warnings=warnings,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    # ========================================================================
    # RBAC VALIDATION
    # ========================================================================

    def validate_rbac_permissions(
        self,
        user_role: str,
        required_permission: str,
        role_permissions: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[bool, str]:
        """
        Validate RBAC permissions.

        Args:
            user_role: User's role (viewer, operator, analyst, admin)
            required_permission: Required permission for operation
            role_permissions: Optional custom role-permission mapping

        Returns:
            Tuple of (has_permission, reason)
        """
        if role_permissions is None:
            # Default GL-009 RBAC permissions
            role_permissions = {
                'viewer': ['read', 'health', 'metrics'],
                'operator': ['read', 'health', 'metrics', 'calculate', 'analyze'],
                'analyst': ['read', 'health', 'metrics', 'calculate', 'analyze', 'benchmark', 'optimize'],
                'admin': ['*'],  # All permissions
            }

        # Admin has all permissions
        if user_role == 'admin':
            return True, "Admin role has all permissions"

        # Check if role exists
        if user_role not in role_permissions:
            return False, f"Unknown role: {user_role}"

        # Check if role has required permission
        permissions = role_permissions[user_role]
        if '*' in permissions or required_permission in permissions:
            return True, f"Role {user_role} has permission {required_permission}"

        return False, f"Role {user_role} lacks permission {required_permission}"

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _generate_finding_id(self) -> str:
        """Generate unique finding ID."""
        return f"GL009-SEC-{secrets.token_hex(4).upper()}"

    def generate_api_key(
        self,
        prefix: str = "gl_sk_th",
        environment: str = "dev",
        length: int = 32
    ) -> str:
        """
        Generate cryptographically secure API key.

        Args:
            prefix: Key prefix (default: gl_sk_th)
            environment: Environment (dev, staging, prod)
            length: Random portion length in bytes

        Returns:
            API key string
        """
        random_part = secrets.token_urlsafe(length)
        return f"{prefix}_{environment}_{random_part}"

    def hash_api_key(self, api_key: str) -> str:
        """
        Hash API key for secure storage (bcrypt-style).

        Args:
            api_key: Plain API key

        Returns:
            Hashed API key (SHA-256)
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    def verify_ip_allowlist(
        self,
        client_ip: str,
        allowlist: List[str]
    ) -> Tuple[bool, str]:
        """
        Verify client IP against allowlist.

        Args:
            client_ip: Client IP address
            allowlist: List of allowed IP addresses/CIDRs

        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            client_ip_obj = ipaddress.ip_address(client_ip)

            for allowed in allowlist:
                # Check if CIDR notation
                if '/' in allowed:
                    network = ipaddress.ip_network(allowed, strict=False)
                    if client_ip_obj in network:
                        return True, f"IP {client_ip} in allowed network {allowed}"
                else:
                    # Single IP
                    if client_ip == allowed:
                        return True, f"IP {client_ip} matches allowed IP"

            return False, f"IP {client_ip} not in allowlist"

        except ValueError as e:
            return False, f"Invalid IP address: {str(e)}"


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_security_validator(
    config: Optional[Any] = None,
    enable_strict_mode: bool = True
) -> SecurityValidator:
    """
    Factory function to create SecurityValidator.

    Args:
        config: Optional configuration object
        enable_strict_mode: Enable strict validation

    Returns:
        SecurityValidator instance
    """
    return SecurityValidator(config=config, enable_strict_mode=enable_strict_mode)
