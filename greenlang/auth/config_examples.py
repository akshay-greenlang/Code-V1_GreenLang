"""
Configuration Examples for Enterprise Authentication

This module provides ready-to-use configuration examples for all authentication providers.
"""

from .saml_provider import SAMLConfig, create_okta_config, create_azure_config
from .oauth_provider import OAuthConfig, create_google_config, create_github_config
from .ldap_provider import LDAPConfig, create_active_directory_config, create_openldap_config
from .mfa import MFAConfig
from .scim_provider import SCIMConfig


# ============================================================================
# SAML Configuration Examples
# ============================================================================

def example_okta_saml_config():
    """Example Okta SAML configuration"""
    return create_okta_config(
        sp_entity_id="https://app.greenlang.io",
        sp_acs_url="https://app.greenlang.io/auth/saml/acs",
        okta_domain="your-domain.okta.com",
        okta_app_id="your-app-id",
        idp_cert="""-----BEGIN CERTIFICATE-----
MIIDpDCCAoygAwIBAgIGAXoqSTEKMA0GCSqGSIb3DQEBCwUAMIGSMQswCQYDVQQG
... (your IdP certificate)
-----END CERTIFICATE-----""",
        sp_sls_url="https://app.greenlang.io/auth/saml/sls"
    )


def example_azure_saml_config():
    """Example Azure AD SAML configuration"""
    return create_azure_config(
        sp_entity_id="https://app.greenlang.io",
        sp_acs_url="https://app.greenlang.io/auth/saml/acs",
        tenant_id="your-tenant-id",
        app_id="your-application-id",
        idp_cert="""-----BEGIN CERTIFICATE-----
... (your Azure AD certificate)
-----END CERTIFICATE-----"""
    )


def example_generic_saml_config():
    """Example generic SAML configuration"""
    return SAMLConfig(
        sp_entity_id="https://app.greenlang.io",
        sp_acs_url="https://app.greenlang.io/auth/saml/acs",
        sp_sls_url="https://app.greenlang.io/auth/saml/sls",
        idp_entity_id="https://idp.example.com/entity",
        idp_sso_url="https://idp.example.com/sso",
        idp_slo_url="https://idp.example.com/slo",
        idp_x509_cert="""-----BEGIN CERTIFICATE-----
... (your IdP certificate)
-----END CERTIFICATE-----""",
        want_assertions_signed=True,
        want_messages_signed=True,
        authn_requests_signed=True,
        attribute_mapping={
            "email": "email",
            "firstName": "first_name",
            "lastName": "last_name",
            "groups": "groups"
        }
    )


# ============================================================================
# OAuth/OIDC Configuration Examples
# ============================================================================

def example_google_oauth_config():
    """Example Google OAuth configuration"""
    return create_google_config(
        client_id="your-client-id.apps.googleusercontent.com",
        client_secret="your-client-secret",
        redirect_uri="https://app.greenlang.io/auth/oauth/callback"
    )


def example_github_oauth_config():
    """Example GitHub OAuth configuration"""
    return create_github_config(
        client_id="your-github-client-id",
        client_secret="your-github-client-secret",
        redirect_uri="https://app.greenlang.io/auth/oauth/callback"
    )


def example_azure_oauth_config():
    """Example Azure AD OAuth configuration"""
    from .oauth_provider import create_azure_config

    return create_azure_config(
        tenant_id="your-tenant-id",
        client_id="your-client-id",
        client_secret="your-client-secret",
        redirect_uri="https://app.greenlang.io/auth/oauth/callback"
    )


def example_generic_oauth_config():
    """Example generic OAuth/OIDC configuration"""
    return OAuthConfig(
        client_id="your-client-id",
        client_secret="your-client-secret",
        redirect_uri="https://app.greenlang.io/auth/oauth/callback",
        authorization_endpoint="https://provider.com/oauth/authorize",
        token_endpoint="https://provider.com/oauth/token",
        userinfo_endpoint="https://provider.com/oauth/userinfo",
        jwks_uri="https://provider.com/.well-known/jwks.json",
        issuer="https://provider.com",
        scope=["openid", "profile", "email"],
        use_pkce=True
    )


# ============================================================================
# LDAP Configuration Examples
# ============================================================================

def example_active_directory_config():
    """Example Active Directory configuration"""
    return create_active_directory_config(
        server_uri="ldaps://dc.example.com:636",
        base_dn="dc=example,dc=com",
        bind_dn="cn=greenlang-service,cn=Users,dc=example,dc=com",
        bind_password="your-service-account-password",
        domain="EXAMPLE",
        use_ssl=True,
        validate_cert=True
    )


def example_openldap_config():
    """Example OpenLDAP configuration"""
    return create_openldap_config(
        server_uri="ldaps://ldap.example.com:636",
        base_dn="dc=example,dc=com",
        bind_dn="cn=admin,dc=example,dc=com",
        bind_password="your-admin-password",
        use_ssl=True
    )


def example_ldap_with_custom_attributes():
    """Example LDAP with custom attribute mapping"""
    return LDAPConfig(
        server_uri="ldaps://ldap.example.com:636",
        base_dn="dc=example,dc=com",
        bind_dn="cn=admin,dc=example,dc=com",
        bind_password="password",
        user_search_filter="(|(uid={username})(mail={username}))",
        user_id_attribute="uid",
        user_email_attribute="mail",
        user_name_attribute="cn",
        user_first_name_attribute="givenName",
        user_last_name_attribute="sn",
        group_search_filter="(objectClass=groupOfNames)",
        group_member_attribute="member",
        use_ssl=True,
        pool_size=20,
        connection_timeout=15
    )


# ============================================================================
# MFA Configuration Examples
# ============================================================================

def example_mfa_config_with_totp_only():
    """Example MFA config with TOTP only"""
    return MFAConfig(
        totp_issuer="GreenLang",
        totp_digits=6,
        totp_interval=30,
        sms_enabled=False,
        email_enabled=False,
        backup_codes_count=10,
        max_attempts=5,
        lockout_duration=900
    )


def example_mfa_config_with_sms():
    """Example MFA config with SMS (Twilio)"""
    return MFAConfig(
        totp_issuer="GreenLang",
        sms_enabled=True,
        twilio_account_sid="your-twilio-account-sid",
        twilio_auth_token="your-twilio-auth-token",
        twilio_phone_number="+1234567890",
        backup_codes_count=10,
        max_attempts=5,
        lockout_duration=900
    )


def example_mfa_config_with_enforcement():
    """Example MFA config with role-based enforcement"""
    return MFAConfig(
        totp_issuer="GreenLang",
        sms_enabled=True,
        twilio_account_sid="your-twilio-account-sid",
        twilio_auth_token="your-twilio-auth-token",
        twilio_phone_number="+1234567890",
        require_mfa=False,  # Not required for all users
        require_mfa_for_roles=["admin", "developer", "operator"],  # Required for these roles
        grace_period_days=7,  # 7 days to set up MFA
        backup_codes_count=10
    )


def example_mfa_config_full():
    """Example MFA config with all features enabled"""
    return MFAConfig(
        # TOTP
        totp_issuer="GreenLang Production",
        totp_digits=6,
        totp_interval=30,
        totp_window=1,

        # SMS
        sms_enabled=True,
        twilio_account_sid="your-twilio-account-sid",
        twilio_auth_token="your-twilio-auth-token",
        twilio_phone_number="+1234567890",

        # Email
        email_enabled=True,
        email_sender="security@greenlang.io",
        email_subject="GreenLang Security Code",

        # Backup codes
        backup_codes_count=10,
        backup_code_length=8,

        # Rate limiting
        max_attempts=5,
        lockout_duration=900,

        # Enforcement
        require_mfa=True,
        require_mfa_for_roles=["admin"],
        grace_period_days=7
    )


# ============================================================================
# SCIM Configuration Examples
# ============================================================================

def example_scim_config():
    """Example SCIM configuration"""
    return SCIMConfig(
        base_url="https://api.greenlang.io/scim/v2",
        bearer_token="your-bearer-token",
        support_bulk=True,
        support_patch=True,
        support_filter=True,
        max_results=100,
        max_bulk_operations=1000
    )


def example_scim_config_with_webhooks():
    """Example SCIM configuration with webhooks"""
    return SCIMConfig(
        base_url="https://api.greenlang.io/scim/v2",
        bearer_token="your-bearer-token",
        support_bulk=True,
        support_patch=True,
        support_filter=True,
        webhook_enabled=True,
        webhook_url="https://your-app.com/webhooks/scim",
        webhook_secret="your-webhook-secret"
    )


# ============================================================================
# Combined Configuration Examples
# ============================================================================

def example_enterprise_auth_stack():
    """
    Example complete enterprise authentication stack

    This shows how to configure all authentication methods together
    """
    return {
        "saml": {
            "okta": example_okta_saml_config(),
            "azure": example_azure_saml_config(),
        },
        "oauth": {
            "google": example_google_oauth_config(),
            "github": example_github_oauth_config(),
        },
        "ldap": example_active_directory_config(),
        "mfa": example_mfa_config_full(),
        "scim": example_scim_config_with_webhooks()
    }


def example_startup_auth_stack():
    """
    Example minimal authentication stack for startups

    OAuth with Google/GitHub + MFA (TOTP only)
    """
    return {
        "oauth": {
            "google": example_google_oauth_config(),
            "github": example_github_oauth_config(),
        },
        "mfa": example_mfa_config_with_totp_only()
    }


def example_hybrid_auth_stack():
    """
    Example hybrid authentication stack

    SAML for enterprise customers + OAuth for individual users + LDAP for internal
    """
    return {
        "saml": {
            "okta": example_okta_saml_config(),
        },
        "oauth": {
            "google": example_google_oauth_config(),
        },
        "ldap": example_active_directory_config(),
        "mfa": example_mfa_config_with_sms(),
        "scim": example_scim_config()
    }


# ============================================================================
# Environment-based Configuration
# ============================================================================

def get_auth_config_from_env():
    """
    Load authentication configuration from environment variables

    Environment variables expected:

    SAML:
    - SAML_SP_ENTITY_ID
    - SAML_SP_ACS_URL
    - SAML_IDP_ENTITY_ID
    - SAML_IDP_SSO_URL
    - SAML_IDP_CERT

    OAuth:
    - OAUTH_CLIENT_ID
    - OAUTH_CLIENT_SECRET
    - OAUTH_REDIRECT_URI
    - OAUTH_PROVIDER (google, github, azure, generic)

    LDAP:
    - LDAP_SERVER_URI
    - LDAP_BASE_DN
    - LDAP_BIND_DN
    - LDAP_BIND_PASSWORD

    MFA:
    - MFA_TOTP_ISSUER
    - MFA_SMS_ENABLED (true/false)
    - TWILIO_ACCOUNT_SID
    - TWILIO_AUTH_TOKEN
    - TWILIO_PHONE_NUMBER

    SCIM:
    - SCIM_BASE_URL
    - SCIM_BEARER_TOKEN
    """
    import os

    config = {}

    # SAML
    if os.getenv("SAML_SP_ENTITY_ID"):
        config["saml"] = SAMLConfig(
            sp_entity_id=os.getenv("SAML_SP_ENTITY_ID"),
            sp_acs_url=os.getenv("SAML_SP_ACS_URL"),
            idp_entity_id=os.getenv("SAML_IDP_ENTITY_ID"),
            idp_sso_url=os.getenv("SAML_IDP_SSO_URL"),
            idp_x509_cert=os.getenv("SAML_IDP_CERT"),
        )

    # OAuth
    if os.getenv("OAUTH_CLIENT_ID"):
        provider = os.getenv("OAUTH_PROVIDER", "generic").lower()
        client_id = os.getenv("OAUTH_CLIENT_ID")
        client_secret = os.getenv("OAUTH_CLIENT_SECRET")
        redirect_uri = os.getenv("OAUTH_REDIRECT_URI")

        if provider == "google":
            config["oauth"] = create_google_config(client_id, client_secret, redirect_uri)
        elif provider == "github":
            config["oauth"] = create_github_config(client_id, client_secret, redirect_uri)
        else:
            config["oauth"] = OAuthConfig(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                authorization_endpoint=os.getenv("OAUTH_AUTH_ENDPOINT"),
                token_endpoint=os.getenv("OAUTH_TOKEN_ENDPOINT"),
                userinfo_endpoint=os.getenv("OAUTH_USERINFO_ENDPOINT"),
            )

    # LDAP
    if os.getenv("LDAP_SERVER_URI"):
        config["ldap"] = LDAPConfig(
            server_uri=os.getenv("LDAP_SERVER_URI"),
            base_dn=os.getenv("LDAP_BASE_DN"),
            bind_dn=os.getenv("LDAP_BIND_DN"),
            bind_password=os.getenv("LDAP_BIND_PASSWORD"),
        )

    # MFA
    config["mfa"] = MFAConfig(
        totp_issuer=os.getenv("MFA_TOTP_ISSUER", "GreenLang"),
        sms_enabled=os.getenv("MFA_SMS_ENABLED", "false").lower() == "true",
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        twilio_phone_number=os.getenv("TWILIO_PHONE_NUMBER"),
    )

    # SCIM
    if os.getenv("SCIM_BASE_URL"):
        config["scim"] = SCIMConfig(
            base_url=os.getenv("SCIM_BASE_URL"),
            bearer_token=os.getenv("SCIM_BEARER_TOKEN"),
        )

    return config


__all__ = [
    # SAML examples
    "example_okta_saml_config",
    "example_azure_saml_config",
    "example_generic_saml_config",

    # OAuth examples
    "example_google_oauth_config",
    "example_github_oauth_config",
    "example_azure_oauth_config",
    "example_generic_oauth_config",

    # LDAP examples
    "example_active_directory_config",
    "example_openldap_config",
    "example_ldap_with_custom_attributes",

    # MFA examples
    "example_mfa_config_with_totp_only",
    "example_mfa_config_with_sms",
    "example_mfa_config_with_enforcement",
    "example_mfa_config_full",

    # SCIM examples
    "example_scim_config",
    "example_scim_config_with_webhooks",

    # Combined stacks
    "example_enterprise_auth_stack",
    "example_startup_auth_stack",
    "example_hybrid_auth_stack",

    # Environment-based
    "get_auth_config_from_env",
]
