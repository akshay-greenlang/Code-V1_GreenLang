# -*- coding: utf-8 -*-
"""
Unit Tests for Enterprise Authentication Providers

Tests for SAML, OAuth/OIDC, LDAP, MFA, and SCIM providers.
"""

import pytest
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import authentication providers
from greenlang.auth.saml_provider import (
    SAMLProvider, SAMLConfig, SAMLError,
    SAMLCertificateManager, SAMLAttributeMapper
)
from greenlang.auth.oauth_provider import (
    OAuthProvider, OAuthConfig, OAuthError,
    PKCEHelper, JWTValidator, OIDCDiscovery
)
from greenlang.auth.ldap_provider import (
    LDAPProvider, LDAPConfig, LDAPError,
    LDAPSearchHelper, create_active_directory_config
)
from greenlang.auth.mfa import (
    MFAManager, MFAConfig, MFAMethod, MFAError,
    TOTPProvider, BackupCodeGenerator
)
from greenlang.auth.scim_provider import (
    SCIMProvider, SCIMConfig, SCIMError,
    SCIMFilter, SCIMUser, SCIMGroup
)


# ============================================================================
# SAML Tests
# ============================================================================

class TestSAMLProvider:
    """Tests for SAML provider"""

    @pytest.fixture
    def saml_config(self):
        return SAMLConfig(
            sp_entity_id="https://app.greenlang.io",
            sp_acs_url="https://app.greenlang.io/auth/saml/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_x509_cert="fake-cert",
            strict=False  # Disable strict mode for testing
        )

    def test_saml_config_creation(self, saml_config):
        assert saml_config.sp_entity_id == "https://app.greenlang.io"
        assert saml_config.want_assertions_signed is True
        assert saml_config.authn_requests_signed is True

    def test_certificate_generation(self):
        cert_pem, key_pem = SAMLCertificateManager.generate_self_signed_cert()
        assert "BEGIN CERTIFICATE" in cert_pem
        assert "BEGIN RSA PRIVATE KEY" in key_pem

    def test_attribute_mapper(self):
        mapper = SAMLAttributeMapper()
        saml_attrs = {
            "email": ["user@example.com"],
            "givenName": ["John"],
            "surname": ["Doe"]
        }

        mapped = mapper.map_attributes(saml_attrs)
        assert mapped["email"] == "user@example.com"
        assert mapped["first_name"] == "John"
        assert mapped["last_name"] == "Doe"

    def test_attribute_mapper_okta(self):
        mapper = SAMLAttributeMapper(
            SAMLAttributeMapper.get_idp_mapping("okta")
        )
        assert "email" in mapper.mapping
        assert "firstName" in mapper.mapping


class TestSAMLCertificateManager:
    """Tests for SAML certificate management"""

    def test_self_signed_cert_generation(self):
        cert_pem, key_pem = SAMLCertificateManager.generate_self_signed_cert(
            common_name="test-sp",
            validity_days=365
        )

        assert cert_pem.startswith("-----BEGIN CERTIFICATE-----")
        assert key_pem.startswith("-----BEGIN RSA PRIVATE KEY-----")

    def test_certificate_validation(self):
        cert_pem, _ = SAMLCertificateManager.generate_self_signed_cert()
        is_valid = SAMLCertificateManager.validate_certificate(cert_pem)
        assert is_valid is True

    def test_certificate_info_extraction(self):
        cert_pem, _ = SAMLCertificateManager.generate_self_signed_cert(
            common_name="test-cert"
        )
        info = SAMLCertificateManager.extract_cert_info(cert_pem)

        assert "subject" in info
        assert "issuer" in info
        assert "fingerprint" in info


# ============================================================================
# OAuth/OIDC Tests
# ============================================================================

class TestOAuthProvider:
    """Tests for OAuth/OIDC provider"""

    @pytest.fixture
    def oauth_config(self):
        return OAuthConfig(
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="https://app.greenlang.io/callback",
            authorization_endpoint="https://provider.com/authorize",
            token_endpoint="https://provider.com/token",
            userinfo_endpoint="https://provider.com/userinfo",
        )

    @pytest.fixture
    def oauth_provider(self, oauth_config):
        return OAuthProvider(oauth_config)

    def test_oauth_config_creation(self, oauth_config):
        assert oauth_config.client_id == "test-client-id"
        assert oauth_config.use_pkce is True

    def test_authorization_url_generation(self, oauth_provider):
        auth_url, state, code_verifier, nonce = oauth_provider.get_authorization_url()

        assert "https://provider.com/authorize" in auth_url
        assert "client_id=test-client-id" in auth_url
        assert state is not None
        assert len(state) > 20

    def test_pkce_code_generation(self):
        verifier = PKCEHelper.generate_code_verifier(128)
        assert len(verifier) == 128

        challenge = PKCEHelper.generate_code_challenge(verifier, "S256")
        assert len(challenge) > 0
        assert challenge != verifier

    def test_pkce_plain_method(self):
        verifier = PKCEHelper.generate_code_verifier()
        challenge = PKCEHelper.generate_code_challenge(verifier, "plain")
        assert challenge == verifier


class TestPKCEHelper:
    """Tests for PKCE helper"""

    def test_verifier_length_validation(self):
        with pytest.raises(ValueError):
            PKCEHelper.generate_code_verifier(length=42)  # Too short

        with pytest.raises(ValueError):
            PKCEHelper.generate_code_verifier(length=129)  # Too long

    def test_verifier_generation(self):
        verifier = PKCEHelper.generate_code_verifier(length=64)
        assert len(verifier) == 64
        assert isinstance(verifier, str)

    def test_challenge_s256(self):
        verifier = PKCEHelper.generate_code_verifier()
        challenge = PKCEHelper.generate_code_challenge(verifier, "S256")

        # Challenge should be different from verifier
        assert challenge != verifier
        # Challenge should be base64url encoded
        assert all(c.isalnum() or c in "-_" for c in challenge)


# ============================================================================
# LDAP Tests
# ============================================================================

class TestLDAPProvider:
    """Tests for LDAP provider"""

    def test_ldap_config_creation(self):
        config = LDAPConfig(
            server_uri="ldaps://ldap.example.com:636",
            base_dn="dc=example,dc=com",
            bind_dn="cn=admin,dc=example,dc=com",
            bind_password="password"
        )

        assert config.server_uri == "ldaps://ldap.example.com:636"
        assert config.use_ssl is True

    def test_active_directory_config(self):
        config = create_active_directory_config(
            server_uri="ldaps://dc.example.com:636",
            base_dn="dc=example,dc=com",
            bind_dn="cn=service,dc=example,dc=com",
            bind_password="password",
            domain="EXAMPLE"
        )

        assert config.is_active_directory is True
        assert config.ad_domain == "EXAMPLE"
        assert config.user_object_class == "user"

    def test_ldap_filter_sanitization(self):
        unsafe_input = "admin*)(uid=*"
        safe_output = LDAPSearchHelper.sanitize_filter(unsafe_input)

        # Should not contain unescaped special characters
        assert "*" not in safe_output or "\\2a" in safe_output
        assert "(" not in safe_output or "\\28" in safe_output

    def test_dn_parsing(self):
        dn = "cn=John Doe,ou=Users,dc=example,dc=com"
        parsed = LDAPSearchHelper.parse_dn(dn)

        assert "cn" in parsed
        assert parsed["cn"][0] == "John Doe"
        assert "ou" in parsed
        assert "dc" in parsed
        assert len(parsed["dc"]) == 2

    def test_cn_extraction(self):
        dn = "cn=John Doe,ou=Users,dc=example,dc=com"
        cn = LDAPSearchHelper.get_cn_from_dn(dn)
        assert cn == "John Doe"


class TestLDAPSearchHelper:
    """Tests for LDAP search helper"""

    def test_user_filter_building(self):
        config = LDAPConfig(
            server_uri="ldap://localhost",
            base_dn="dc=example,dc=com",
            user_search_filter="(uid={username})"
        )

        filter_str = LDAPSearchHelper.build_user_filter(config, "testuser")
        assert "(uid=testuser)" in filter_str

    def test_group_filter_building(self):
        config = LDAPConfig(
            server_uri="ldap://localhost",
            base_dn="dc=example,dc=com",
            group_search_filter="(objectClass=groupOfNames)",
            group_name_attribute="cn"
        )

        filter_str = LDAPSearchHelper.build_group_filter(config, "admins")
        assert "(objectClass=groupOfNames)" in filter_str
        assert "(cn=admins)" in filter_str


# ============================================================================
# MFA Tests
# ============================================================================

class TestMFAManager:
    """Tests for MFA manager"""

    @pytest.fixture
    def mfa_config(self):
        return MFAConfig(
            totp_issuer="GreenLang Test",
            sms_enabled=False,
            email_enabled=False
        )

    @pytest.fixture
    def mfa_manager(self, mfa_config):
        return MFAManager(mfa_config)

    def test_totp_provider_initialization(self, mfa_config):
        provider = TOTPProvider(mfa_config)
        assert provider.config.totp_issuer == "GreenLang Test"

    def test_totp_secret_generation(self, mfa_config):
        provider = TOTPProvider(mfa_config)
        secret = provider.generate_secret()

        assert len(secret) > 0
        assert isinstance(secret, str)
        # Base32 characters
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in secret)

    def test_totp_code_verification(self, mfa_config):
        provider = TOTPProvider(mfa_config)
        secret = provider.generate_secret()

        # Get current code
        code = provider.get_current_code(secret)

        # Verify it
        is_valid = provider.verify_code(secret, code)
        assert is_valid is True

        # Invalid code should fail
        is_valid = provider.verify_code(secret, "000000")
        assert is_valid is False

    def test_backup_code_generation(self):
        codes = BackupCodeGenerator.generate_codes(count=10, length=8)

        assert len(codes) == 10
        for code in codes:
            assert len(code.code) >= 8
            assert code.used is False
            assert code.code_hash is not None

    def test_backup_code_verification(self):
        codes = BackupCodeGenerator.generate_codes(count=5)
        test_code = codes[0].code

        # Should verify successfully
        verified_code = BackupCodeGenerator.verify_code(test_code, codes)
        assert verified_code is not None
        assert verified_code.code_hash == codes[0].code_hash

        # Mark as used
        verified_code.used = True

        # Should not verify again
        verified_code = BackupCodeGenerator.verify_code(test_code, codes)
        assert verified_code is None

    def test_mfa_enrollment_totp(self, mfa_manager):
        user_id = "test-user"

        # Enroll user
        device_id, secret, qr_code = mfa_manager.enroll_totp(user_id, "My Device")

        assert device_id is not None
        assert secret is not None
        assert qr_code is not None  # QR code image bytes

        # Get enrollment
        enrollment = mfa_manager.get_enrollment(user_id)
        assert enrollment is not None
        assert len(enrollment.totp_devices) == 1

    def test_mfa_rate_limiting(self, mfa_manager):
        from greenlang.auth.mfa import RateLimiter

        limiter = RateLimiter(max_attempts=3, window_seconds=60)

        # Should allow first 3 attempts
        for i in range(3):
            allowed, remaining = limiter.check_rate_limit("test-user")
            assert allowed is True
            limiter.record_attempt("test-user")

        # 4th attempt should be blocked
        allowed, remaining = limiter.check_rate_limit("test-user")
        assert allowed is False
        assert remaining == 0


class TestBackupCodeGenerator:
    """Tests for backup code generator"""

    def test_code_generation(self):
        codes = BackupCodeGenerator.generate_codes(count=10, length=8)
        assert len(codes) == 10

        # All codes should be unique
        code_values = [c.code for c in codes]
        assert len(set(code_values)) == 10

    def test_code_format(self):
        codes = BackupCodeGenerator.generate_codes(count=1, length=8)
        code = codes[0].code

        # Should have dash separator for length >= 8
        assert "-" in code

    def test_code_hashing(self):
        codes = BackupCodeGenerator.generate_codes(count=2)

        # Different codes should have different hashes
        assert codes[0].code_hash != codes[1].code_hash


# ============================================================================
# SCIM Tests
# ============================================================================

class TestSCIMProvider:
    """Tests for SCIM provider"""

    @pytest.fixture
    def scim_config(self):
        return SCIMConfig(
            base_url="https://api.greenlang.io/scim/v2",
            bearer_token="test-token"
        )

    @pytest.fixture
    def scim_provider(self, scim_config):
        return SCIMProvider(scim_config)

    def test_scim_user_creation(self, scim_provider):
        user_data = {
            "userName": "testuser",
            "name": {
                "givenName": "Test",
                "familyName": "User"
            },
            "emails": [
                {"value": "test@example.com", "primary": True}
            ],
            "active": True
        }

        user = scim_provider.create_user(user_data)

        assert user.id is not None
        assert user.userName == "testuser"
        assert user.name.givenName == "Test"
        assert user.name.familyName == "User"
        assert len(user.emails) == 1
        assert user.active is True

    def test_scim_user_retrieval(self, scim_provider):
        # Create user
        user_data = {"userName": "testuser", "active": True}
        created_user = scim_provider.create_user(user_data)

        # Retrieve user
        retrieved_user = scim_provider.get_user(created_user.id)

        assert retrieved_user is not None
        assert retrieved_user.id == created_user.id
        assert retrieved_user.userName == "testuser"

    def test_scim_user_update(self, scim_provider):
        # Create user
        user_data = {"userName": "testuser", "active": True}
        user = scim_provider.create_user(user_data)

        # Update user
        updated_data = {
            "userName": "testuser",
            "active": False,
            "displayName": "Test User Updated"
        }
        updated_user = scim_provider.update_user(user.id, updated_data)

        assert updated_user.active is False
        assert updated_user.displayName == "Test User Updated"

    def test_scim_user_deletion(self, scim_provider):
        # Create user
        user_data = {"userName": "testuser", "active": True}
        user = scim_provider.create_user(user_data)

        # Delete user
        success = scim_provider.delete_user(user.id)
        assert success is True

        # User should not exist
        deleted_user = scim_provider.get_user(user.id)
        assert deleted_user is None

    def test_scim_user_search(self, scim_provider):
        # Create multiple users
        for i in range(5):
            user_data = {"userName": f"user{i}", "active": True}
            scim_provider.create_user(user_data)

        # Search all users
        result = scim_provider.search_users()

        assert result.totalResults == 5
        assert len(result.Resources) == 5

    def test_scim_user_filtering(self, scim_provider):
        # Create users
        scim_provider.create_user({"userName": "alice", "active": True})
        scim_provider.create_user({"userName": "bob", "active": True})
        scim_provider.create_user({"userName": "charlie", "active": False})

        # Filter by userName
        result = scim_provider.search_users(filter_expr='userName eq "alice"')

        assert result.totalResults == 1
        assert result.Resources[0]["userName"] == "alice"

    def test_scim_group_creation(self, scim_provider):
        group_data = {
            "displayName": "Developers",
            "members": []
        }

        group = scim_provider.create_group(group_data)

        assert group.id is not None
        assert group.displayName == "Developers"

    def test_scim_filter_parsing(self):
        # Test simple equality filter
        filter_expr = 'userName eq "testuser"'
        parsed = SCIMFilter.parse(filter_expr)

        assert parsed["attribute"] == "userName"
        assert parsed["op"] == "eq"
        assert parsed["value"] == "testuser"

    def test_scim_filter_evaluation(self):
        resource = {"userName": "testuser", "active": True}

        # Test equality
        filter_expr = {"attribute": "userName", "op": "eq", "value": "testuser"}
        result = SCIMFilter.evaluate(resource, filter_expr)
        assert result is True

        # Test contains
        filter_expr = {"attribute": "userName", "op": "co", "value": "test"}
        result = SCIMFilter.evaluate(resource, filter_expr)
        assert result is True

    def test_scim_service_provider_config(self, scim_provider):
        config = scim_provider.get_service_provider_config()

        assert "schemas" in config
        assert config["patch"]["supported"] is True
        assert config["bulk"]["supported"] is True
        assert config["filter"]["supported"] is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestAuthProviderIntegration:
    """Integration tests for authentication providers"""

    def test_oauth_with_mfa_flow(self):
        """Test OAuth authentication followed by MFA challenge"""

        # Setup OAuth
        oauth_config = OAuthConfig(
            client_id="test-client",
            client_secret="test-secret",
            redirect_uri="https://app.greenlang.io/callback",
            authorization_endpoint="https://provider.com/authorize",
            token_endpoint="https://provider.com/token",
        )
        oauth_provider = OAuthProvider(oauth_config)

        # Setup MFA
        mfa_config = MFAConfig(totp_issuer="Test")
        mfa_manager = MFAManager(mfa_config)

        # Step 1: Get OAuth authorization URL
        auth_url, state, _, _ = oauth_provider.get_authorization_url()
        assert "https://provider.com/authorize" in auth_url

        # Step 2: Enroll user in TOTP MFA
        user_id = "test-user"
        device_id, secret, qr_code = mfa_manager.enroll_totp(user_id)

        # Step 3: Verify TOTP enrollment
        totp = TOTPProvider(mfa_config)
        code = totp.get_current_code(secret)
        verified = mfa_manager.verify_totp_enrollment(user_id, device_id, code)
        assert verified is True

    def test_scim_user_provisioning_with_groups(self):
        """Test SCIM user and group provisioning"""

        scim_config = SCIMConfig(base_url="https://api.test.com/scim/v2")
        scim_provider = SCIMProvider(scim_config)

        # Create users
        user1 = scim_provider.create_user({
            "userName": "alice",
            "active": True
        })
        user2 = scim_provider.create_user({
            "userName": "bob",
            "active": True
        })

        # Create group with members
        group = scim_provider.create_group({
            "displayName": "Developers",
            "members": [
                {"value": user1.id, "$ref": f"Users/{user1.id}"},
                {"value": user2.id, "$ref": f"Users/{user2.id}"}
            ]
        })

        # Verify group membership
        assert len(group.members) == 2

        # Verify users have group reference
        updated_user1 = scim_provider.get_user(user1.id)
        assert len(updated_user1.groups) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
