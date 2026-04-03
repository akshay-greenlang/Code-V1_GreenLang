"""
GreenLang Security Exceptions - Authentication, Authorization, and Security Errors

This module provides exception classes for security-related errors across
the GreenLang platform including JWT, RBAC, encryption, secrets management,
PII handling, and certificate operations.

Features:
- Authentication failures (JWT, API key, token expiry)
- Authorization failures (RBAC, permissions, roles)
- Encryption and decryption errors
- Secret access failures (Vault integration)
- PII violation detection and blocking
- Egress and network security policy enforcement
- Certificate validation errors

Author: GreenLang Team
Date: 2026-04-02
"""

from typing import Any, Dict, List, Optional

from greenlang.exceptions.base import GreenLangException


class SecurityException(GreenLangException):
    """Base exception for security-related errors.

    Raised when a security operation fails including authentication,
    authorization, encryption, or policy enforcement.
    """
    ERROR_PREFIX = "GL_SECURITY"


class AuthenticationError(SecurityException):
    """Authentication failed.

    Raised when user or service authentication fails due to invalid
    credentials, expired tokens, or missing authentication headers.

    Example:
        >>> raise AuthenticationError(
        ...     message="JWT token expired",
        ...     auth_method="jwt",
        ...     context={"token_expiry": "2026-04-01T12:00:00Z"}
        ... )
    """

    def __init__(
        self,
        message: str,
        auth_method: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize authentication error.

        Args:
            message: Error message
            auth_method: Authentication method (jwt, api_key, oauth2, mTLS)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if auth_method:
            context["auth_method"] = auth_method
        super().__init__(message, agent_name=agent_name, context=context)


class AuthorizationError(SecurityException):
    """Authorization check failed.

    Raised when an authenticated user or service lacks the required
    permissions or role to perform the requested operation.

    Example:
        >>> raise AuthorizationError(
        ...     message="Insufficient permissions for CBAM report export",
        ...     required_permission="cbam:report:export",
        ...     user_role="viewer"
        ... )
    """

    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        user_role: Optional[str] = None,
        resource: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize authorization error.

        Args:
            message: Error message
            required_permission: Permission that was required
            user_role: Role of the user who was denied
            resource: Resource that was being accessed
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if required_permission:
            context["required_permission"] = required_permission
        if user_role:
            context["user_role"] = user_role
        if resource:
            context["resource"] = resource
        super().__init__(message, agent_name=agent_name, context=context)


class EncryptionError(SecurityException):
    """Encryption or decryption operation failed.

    Raised when AES-256-GCM encryption, decryption, or key derivation
    operations fail.

    Example:
        >>> raise EncryptionError(
        ...     message="Decryption failed: invalid ciphertext",
        ...     operation="decrypt",
        ...     algorithm="AES-256-GCM"
        ... )
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        algorithm: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize encryption error.

        Args:
            message: Error message
            operation: Operation that failed (encrypt, decrypt, key_derive)
            algorithm: Encryption algorithm used
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if operation:
            context["operation"] = operation
        if algorithm:
            context["algorithm"] = algorithm
        super().__init__(message, agent_name=agent_name, context=context)


class SecretAccessError(SecurityException):
    """Secret retrieval or storage failed.

    Raised when accessing secrets from Vault or other secret management
    backends fails.

    Example:
        >>> raise SecretAccessError(
        ...     message="Failed to retrieve secret from Vault",
        ...     secret_path="secret/data/greenlang/api-keys",
        ...     backend="vault"
        ... )
    """

    def __init__(
        self,
        message: str,
        secret_path: Optional[str] = None,
        backend: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize secret access error.

        Args:
            message: Error message
            secret_path: Path to the secret that could not be accessed
            backend: Secret management backend (vault, env, aws_ssm)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if secret_path:
            context["secret_path"] = secret_path
        if backend:
            context["backend"] = backend
        super().__init__(message, agent_name=agent_name, context=context)


class PIIViolationError(SecurityException):
    """PII detected in data that should be clean.

    Raised when personally identifiable information is detected in
    output data, reports, or logs where it must not appear.

    Example:
        >>> raise PIIViolationError(
        ...     message="PII detected in emissions report output",
        ...     pii_types=["email", "phone_number"],
        ...     field_name="supplier_notes"
        ... )
    """

    def __init__(
        self,
        message: str,
        pii_types: Optional[List[str]] = None,
        field_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize PII violation error.

        Args:
            message: Error message
            pii_types: Types of PII detected (email, phone, ssn, etc.)
            field_name: Field where PII was detected
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if pii_types:
            context["pii_types"] = pii_types
        if field_name:
            context["field_name"] = field_name
        super().__init__(message, agent_name=agent_name, context=context)


class EgressBlockedError(SecurityException):
    """Outbound network request blocked by egress policy.

    Raised when an outbound request is denied by the network egress
    policy, domain allowlist, or firewall rules.

    Example:
        >>> raise EgressBlockedError(
        ...     message="Egress blocked: domain not in allowlist",
        ...     blocked_domain="malicious-site.example.com",
        ...     policy_name="egress_allowlist"
        ... )
    """

    def __init__(
        self,
        message: str,
        blocked_domain: Optional[str] = None,
        policy_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize egress blocked error.

        Args:
            message: Error message
            blocked_domain: Domain that was blocked
            policy_name: Name of the policy that blocked the request
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if blocked_domain:
            context["blocked_domain"] = blocked_domain
        if policy_name:
            context["policy_name"] = policy_name
        super().__init__(message, agent_name=agent_name, context=context)


class CertificateError(SecurityException):
    """TLS certificate validation failed.

    Raised when TLS certificate verification, rotation, or renewal fails.

    Example:
        >>> raise CertificateError(
        ...     message="TLS certificate expired",
        ...     certificate_subject="*.greenlang.io",
        ...     expiry_date="2026-03-30T00:00:00Z"
        ... )
    """

    def __init__(
        self,
        message: str,
        certificate_subject: Optional[str] = None,
        expiry_date: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize certificate error.

        Args:
            message: Error message
            certificate_subject: Certificate subject or CN
            expiry_date: Certificate expiry date (ISO 8601)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if certificate_subject:
            context["certificate_subject"] = certificate_subject
        if expiry_date:
            context["expiry_date"] = expiry_date
        super().__init__(message, agent_name=agent_name, context=context)


__all__ = [
    'SecurityException',
    'AuthenticationError',
    'AuthorizationError',
    'EncryptionError',
    'SecretAccessError',
    'PIIViolationError',
    'EgressBlockedError',
    'CertificateError',
]
