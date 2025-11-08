# GreenLang Enterprise Authentication Security Documentation

## Table of Contents

1. [Overview](#overview)
2. [Authentication Methods](#authentication-methods)
3. [Security Architecture](#security-architecture)
4. [SAML 2.0 Authentication](#saml-20-authentication)
5. [OAuth 2.0 / OIDC Authentication](#oauth-20--oidc-authentication)
6. [LDAP/Active Directory](#ldapactive-directory)
7. [Multi-Factor Authentication (MFA)](#multi-factor-authentication-mfa)
8. [SCIM 2.0 User Provisioning](#scim-20-user-provisioning)
9. [Security Best Practices](#security-best-practices)
10. [Deployment Guide](#deployment-guide)
11. [Troubleshooting](#troubleshooting)

---

## Overview

GreenLang Phase 4 implements comprehensive enterprise authentication with support for:

- **SAML 2.0**: Enterprise SSO with Okta, Azure AD, OneLogin
- **OAuth 2.0 / OIDC**: Modern authentication with Google, GitHub, Azure
- **LDAP/AD**: Directory integration for user authentication
- **MFA**: Multi-factor authentication (TOTP, SMS, backup codes)
- **SCIM 2.0**: Automated user provisioning

### Security Highlights

- Industry-standard protocols (SAML 2.0, OAuth 2.0, OIDC)
- Certificate-based authentication
- Token signature verification
- Encrypted credential storage
- Rate limiting and brute force protection
- Comprehensive audit logging
- PKCE support for OAuth
- Session management with automatic expiration

---

## Authentication Methods

### Supported Providers

#### SAML 2.0
- Okta
- Azure AD
- OneLogin
- Generic SAML 2.0 IdPs

#### OAuth 2.0 / OIDC
- Google Workspace
- GitHub
- Microsoft Azure AD
- Generic OAuth/OIDC providers

#### LDAP
- OpenLDAP
- Active Directory
- FreeIPA
- Generic LDAP servers

### Authentication Flow Diagrams

#### SAML Authentication Flow

```
User                Browser              GreenLang SP         Identity Provider
 |                     |                      |                      |
 |---(1) Access App--->|                      |                      |
 |                     |---(2) Redirect------>|                      |
 |                     |                      |                      |
 |                     |<--(3) SAML Request---|                      |
 |                     |                      |                      |
 |                     |---(4) Forward Request------------------>    |
 |                     |                      |                      |
 |<---(5) Login Page-------------------------|                       |
 |                     |                      |                      |
 |---(6) Credentials------------------------>|                       |
 |                     |                      |                      |
 |<---(7) SAML Response----------------------|                       |
 |                     |                      |                      |
 |---(8) Post Response----------------->     |                       |
 |                     |                      |                      |
 |                     |<--(9) Validate Assertion                    |
 |                     |                      |                      |
 |<--(10) Create Session & Redirect----------|                      |
 |                     |                      |                      |
```

#### OAuth 2.0 / OIDC Flow

```
User                Browser              GreenLang              OAuth Provider
 |                     |                      |                      |
 |---(1) Login-------->|                      |                      |
 |                     |---(2) Request Auth-->|                      |
 |                     |                      |                      |
 |                     |<--(3) Auth URL-------|                      |
 |                     |                      |                      |
 |                     |---(4) Redirect to OAuth Provider-------> |
 |                     |                      |                      |
 |<---(5) Login Page-------------------------|                       |
 |                     |                      |                      |
 |---(6) Credentials------------------------>|                       |
 |                     |                      |                      |
 |<---(7) Auth Code--------------------------                        |
 |                     |                      |                      |
 |---(8) Send Code---------------->          |                       |
 |                     |                      |                      |
 |                     |         (9) Exchange Code for Tokens----->  |
 |                     |                      |                      |
 |                     |         <--(10) Access Token + ID Token---- |
 |                     |                      |                      |
 |<--(11) Create Session & Redirect----------|                      |
 |                     |                      |                      |
```

#### LDAP Authentication Flow

```
User                GreenLang            LDAP Server
 |                      |                      |
 |---(1) Login--------->|                      |
 |  (username/password) |                      |
 |                      |                      |
 |                      |---(2) Search User--->|
 |                      |  (using bind account)|
 |                      |                      |
 |                      |<--(3) User DN--------|
 |                      |                      |
 |                      |---(4) Bind as User-->|
 |                      |  (user DN + password)|
 |                      |                      |
 |                      |<--(5) Auth Result----|
 |                      |                      |
 |                      |---(6) Get Groups---->|
 |                      |                      |
 |                      |<--(7) Group List-----|
 |                      |                      |
 |<--(8) Create Session-|                      |
 |                      |                      |
```

#### MFA Flow (TOTP)

```
User                GreenLang            Authenticator App
 |                      |                      |
 |---(1) Login--------->|                      |
 |  (username/password) |                      |
 |                      |                      |
 |<--(2) MFA Required---|                      |
 |                      |                      |
 |---(3) Open App------>|                      |
 |                      |                      |
 |<--(4) Display Code---|                      |
 |                      |                      |
 |---(5) Enter Code---->|                      |
 |                      |                      |
 |                      |---(6) Verify Code    |
 |                      | (check TOTP secret)  |
 |                      |                      |
 |<--(7) Access Granted-|                      |
 |                      |                      |
```

---

## SAML 2.0 Authentication

### Features

- **Assertion Validation**: XML signature verification using xmlsec1
- **Multiple IdP Support**: Okta, Azure AD, OneLogin, generic SAML
- **Attribute Mapping**: Flexible mapping from SAML to internal user model
- **Encryption**: Support for encrypted assertions
- **Single Logout (SLO)**: Coordinated logout across IdP and SP
- **Metadata Exchange**: Automatic SP metadata generation

### Security Features

1. **XML Signature Validation**
   - Validates IdP signatures on assertions
   - Supports RSA-SHA256 and RSA-SHA1 algorithms
   - Certificate chain verification

2. **Replay Attack Prevention**
   - Request ID caching with TTL
   - Assertion ID tracking
   - Timestamp validation (NotBefore/NotOnOrAfter)

3. **Certificate Management**
   - Self-signed certificate generation
   - Certificate rotation support
   - Expiration monitoring

### Configuration Example

```python
from greenlang.auth import SAMLProvider, create_okta_config

# Okta SAML configuration
config = create_okta_config(
    sp_entity_id="https://app.greenlang.io",
    sp_acs_url="https://app.greenlang.io/auth/saml/acs",
    okta_domain="your-domain.okta.com",
    okta_app_id="your-app-id",
    idp_cert="<your-okta-certificate>"
)

# Initialize provider
saml_provider = SAMLProvider(config)

# Generate authentication request
auth_url, request_id = saml_provider.get_auth_request_url()

# Process SAML response
user = saml_provider.process_response(saml_response, request_id)
```

### Security Considerations

- **Always use HTTPS** for ACS and SLS endpoints
- **Enable assertion signing** in production
- **Rotate certificates** annually
- **Monitor for suspicious assertion patterns**
- **Implement rate limiting** on SAML endpoints

---

## OAuth 2.0 / OIDC Authentication

### Features

- **OAuth 2.0 Flows**: Authorization code, client credentials, refresh token
- **OIDC Support**: ID token validation, UserInfo endpoint
- **PKCE**: Proof Key for Code Exchange for mobile/SPA security
- **JWT Validation**: Signature verification with JWKS
- **Token Refresh**: Automatic access token refresh
- **Discovery**: OIDC discovery document support

### Security Features

1. **PKCE (Proof Key for Code Exchange)**
   - Protects against authorization code interception
   - SHA256 code challenge
   - Required for public clients

2. **State Parameter**
   - CSRF protection
   - Random state generation
   - State validation on callback

3. **Nonce Validation**
   - Prevents replay attacks in OIDC
   - Cryptographically random nonce
   - Verified in ID token

4. **JWT Signature Verification**
   - JWKS-based key retrieval
   - RS256 algorithm support
   - Issuer and audience validation

### Configuration Example

```python
from greenlang.auth import OAuthProvider, create_google_config

# Google OAuth configuration
config = create_google_config(
    client_id="your-client-id.apps.googleusercontent.com",
    client_secret="your-client-secret",
    redirect_uri="https://app.greenlang.io/auth/oauth/callback"
)

# Initialize provider
oauth_provider = OAuthProvider(config)

# Get authorization URL
auth_url, state, code_verifier, nonce = oauth_provider.get_authorization_url()

# Exchange code for tokens
tokens = oauth_provider.exchange_code_for_tokens(
    code=authorization_code,
    state=state,
    code_verifier=code_verifier
)

# Create user from tokens
user = oauth_provider.create_user_from_tokens(tokens)
```

### Security Considerations

- **Always use PKCE** for public clients
- **Validate state parameter** on callback
- **Verify ID token signatures** using JWKS
- **Use short-lived access tokens**
- **Implement token refresh** for long-running sessions
- **Store tokens securely** (encrypted at rest)

---

## LDAP/Active Directory

### Features

- **Connection Pooling**: Thread-safe connection pool
- **User Authentication**: Bind-based authentication
- **Group Synchronization**: Recursive group membership
- **Active Directory Support**: Nested groups, primary group
- **TLS/SSL**: Encrypted connections
- **Incremental Sync**: Delta updates for large directories

### Security Features

1. **Encrypted Connections**
   - LDAPS (LDAP over SSL)
   - StartTLS support
   - Certificate validation

2. **Input Sanitization**
   - LDAP injection prevention
   - Filter escaping
   - DN validation

3. **Credential Protection**
   - Secure bind credentials storage
   - Connection pooling with timeout
   - Automatic reconnection

### Configuration Example

```python
from greenlang.auth import LDAPProvider, create_active_directory_config

# Active Directory configuration
config = create_active_directory_config(
    server_uri="ldaps://dc.example.com:636",
    base_dn="dc=example,dc=com",
    bind_dn="cn=greenlang-service,cn=Users,dc=example,dc=com",
    bind_password="service-account-password",
    domain="EXAMPLE"
)

# Initialize provider
ldap_provider = LDAPProvider(config)

# Authenticate user
user = ldap_provider.authenticate("username", "password")

# Sync users and groups
stats = ldap_provider.sync_users_and_groups()
```

### Security Considerations

- **Always use LDAPS or StartTLS**
- **Use dedicated service account** with minimal privileges
- **Implement connection pooling** to prevent DOS
- **Sanitize all user input** to prevent LDAP injection
- **Monitor for failed bind attempts**
- **Implement account lockout** after repeated failures

---

## Multi-Factor Authentication (MFA)

### Features

- **TOTP**: Time-based OTP (Google Authenticator, Authy)
- **SMS OTP**: SMS delivery via Twilio
- **Email OTP**: Email-based verification codes
- **Backup Codes**: Recovery codes for lost devices
- **QR Code Generation**: Easy TOTP enrollment
- **Rate Limiting**: Brute force protection

### Security Features

1. **TOTP Security**
   - 30-second time window
   - SHA1/SHA256 HMAC
   - 6-digit codes
   - Time drift tolerance (±1 window)

2. **SMS Security**
   - 6-digit numeric codes
   - 10-minute expiration
   - Rate limiting (5 attempts per 15 minutes)
   - Phone number verification

3. **Backup Codes**
   - One-time use codes
   - Cryptographically secure generation
   - Hashed storage (SHA256)
   - 10 codes per user

### Configuration Example

```python
from greenlang.auth import MFAManager, MFAConfig

# MFA configuration
config = MFAConfig(
    totp_issuer="GreenLang",
    sms_enabled=True,
    twilio_account_sid="your-twilio-sid",
    twilio_auth_token="your-twilio-token",
    twilio_phone_number="+1234567890",
    backup_codes_count=10,
    max_attempts=5,
    lockout_duration=900  # 15 minutes
)

# Initialize MFA manager
mfa_manager = MFAManager(config)

# Enroll user in TOTP
device_id, secret, qr_code = mfa_manager.enroll_totp(
    user_id="user123",
    device_name="My Phone"
)

# Verify enrollment
verified = mfa_manager.verify_totp_enrollment(
    user_id="user123",
    device_id=device_id,
    code="123456"
)

# Verify MFA during login
success = mfa_manager.verify_mfa(
    user_id="user123",
    method=MFAMethod.TOTP,
    code="123456",
    device_id=device_id
)
```

### Security Considerations

- **Enforce MFA for privileged accounts**
- **Provide backup codes** for device loss
- **Implement rate limiting** to prevent brute force
- **Use secure random generators** for codes
- **Monitor for suspicious MFA patterns**
- **Support multiple devices** per user

---

## SCIM 2.0 User Provisioning

### Features

- **User Provisioning**: Create, read, update, delete users
- **Group Provisioning**: Group management and membership
- **Bulk Operations**: Batch user/group operations
- **Filtering**: SCIM filter expressions
- **Pagination**: Efficient large dataset handling
- **Webhooks**: Real-time provisioning notifications

### Security Features

1. **Bearer Token Authentication**
   - OAuth 2.0 bearer tokens
   - Token validation on each request
   - Token rotation support

2. **Input Validation**
   - Schema validation
   - Attribute type checking
   - Required field enforcement

3. **Audit Logging**
   - All provisioning events logged
   - User/group change tracking
   - Webhook event history

### Configuration Example

```python
from greenlang.auth import SCIMProvider, SCIMConfig

# SCIM configuration
config = SCIMConfig(
    base_url="https://api.greenlang.io/scim/v2",
    bearer_token="your-bearer-token",
    support_bulk=True,
    support_patch=True,
    webhook_enabled=True,
    webhook_url="https://your-app.com/webhooks/scim"
)

# Initialize provider
scim_provider = SCIMProvider(config)

# Create user
user = scim_provider.create_user({
    "userName": "john.doe",
    "name": {
        "givenName": "John",
        "familyName": "Doe"
    },
    "emails": [
        {"value": "john@example.com", "primary": True}
    ],
    "active": True
})

# Search users
results = scim_provider.search_users(
    filter_expr='userName eq "john.doe"',
    start_index=1,
    count=10
)
```

### Security Considerations

- **Use strong bearer tokens** (min 32 bytes)
- **Implement token rotation** every 90 days
- **Validate all input** against SCIM schema
- **Rate limit SCIM endpoints**
- **Monitor for bulk operation abuse**
- **Audit all provisioning changes**

---

## Security Best Practices

### 1. Credential Management

**DO:**
- Store credentials in environment variables or secrets manager
- Use separate credentials for each environment
- Rotate credentials regularly (90 days)
- Use strong, unique passwords (min 16 characters)

**DON'T:**
- Hardcode credentials in source code
- Commit credentials to version control
- Share credentials across services
- Use default or weak passwords

### 2. Session Management

**DO:**
- Use secure, random session IDs (min 32 bytes)
- Set session timeout (recommend 1 hour)
- Implement absolute session timeout (recommend 8 hours)
- Regenerate session ID after authentication
- Clear session on logout

**DON'T:**
- Use predictable session IDs
- Allow indefinite sessions
- Store sensitive data in sessions
- Reuse session IDs

### 3. Transport Security

**DO:**
- Use TLS 1.2 or higher for all connections
- Validate SSL/TLS certificates
- Use HSTS headers
- Implement certificate pinning for mobile apps

**DON'T:**
- Use plain HTTP for authentication
- Disable certificate validation
- Use self-signed certificates in production
- Support outdated TLS versions (< 1.2)

### 4. Error Handling

**DO:**
- Log authentication failures
- Return generic error messages to users
- Implement rate limiting
- Monitor for attack patterns

**DON'T:**
- Expose detailed error messages
- Reveal whether username exists
- Log passwords or tokens
- Allow unlimited authentication attempts

### 5. Audit Logging

**DO:**
- Log all authentication events
- Include timestamps, IP addresses, user agents
- Store logs securely with retention policy
- Implement log monitoring and alerting

**DON'T:**
- Log sensitive data (passwords, tokens)
- Allow log tampering
- Expose logs to unauthorized users
- Ignore failed authentication patterns

---

## Deployment Guide

### Prerequisites

```bash
# Install required packages
pip install python3-saml authlib ldap3 pyotp twilio qrcode[pil] cryptography PyJWT
```

### Environment Variables

```bash
# SAML
export SAML_SP_ENTITY_ID="https://app.greenlang.io"
export SAML_SP_ACS_URL="https://app.greenlang.io/auth/saml/acs"
export SAML_IDP_ENTITY_ID="https://idp.example.com"
export SAML_IDP_SSO_URL="https://idp.example.com/sso"
export SAML_IDP_CERT="<certificate-content>"

# OAuth
export OAUTH_CLIENT_ID="your-client-id"
export OAUTH_CLIENT_SECRET="your-client-secret"
export OAUTH_REDIRECT_URI="https://app.greenlang.io/callback"
export OAUTH_PROVIDER="google"  # google, github, azure, generic

# LDAP
export LDAP_SERVER_URI="ldaps://ldap.example.com:636"
export LDAP_BASE_DN="dc=example,dc=com"
export LDAP_BIND_DN="cn=service,dc=example,dc=com"
export LDAP_BIND_PASSWORD="service-password"

# MFA
export MFA_TOTP_ISSUER="GreenLang"
export MFA_SMS_ENABLED="true"
export TWILIO_ACCOUNT_SID="your-twilio-sid"
export TWILIO_AUTH_TOKEN="your-twilio-token"
export TWILIO_PHONE_NUMBER="+1234567890"

# SCIM
export SCIM_BASE_URL="https://api.greenlang.io/scim/v2"
export SCIM_BEARER_TOKEN="your-bearer-token"
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY greenlang/ ./greenlang/

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "greenlang.cli.main_new"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-auth
spec:
  replicas: 3
  selector:
    matchLabels:
      app: greenlang-auth
  template:
    metadata:
      labels:
        app: greenlang-auth
    spec:
      containers:
      - name: greenlang
        image: greenlang/auth:latest
        env:
        - name: SAML_SP_ENTITY_ID
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: saml-entity-id
        - name: OAUTH_CLIENT_SECRET
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: oauth-client-secret
        ports:
        - containerPort: 8000
```

---

## Troubleshooting

### SAML Issues

**Problem**: "Invalid SAML response signature"
- **Solution**: Verify IdP certificate is correct and not expired
- Check clock synchronization between SP and IdP
- Ensure assertion signing is enabled in IdP

**Problem**: "User not found after SAML authentication"
- **Solution**: Check attribute mapping configuration
- Verify SAML assertions contain expected attributes
- Enable debug logging to see raw assertions

### OAuth Issues

**Problem**: "Invalid state parameter"
- **Solution**: Ensure state is stored correctly in session
- Check for cookie domain/path issues
- Verify state TTL is not too short

**Problem**: "ID token validation failed"
- **Solution**: Check JWKS URI is accessible
- Verify issuer and audience claims
- Ensure system clock is synchronized

### LDAP Issues

**Problem**: "LDAP bind failed"
- **Solution**: Verify bind DN and password
- Check LDAP server is accessible
- Ensure SSL/TLS certificates are valid

**Problem**: "User groups not syncing"
- **Solution**: Verify group search base and filter
- Check group member attribute name
- For AD, verify nested group support is enabled

### MFA Issues

**Problem**: "TOTP codes always failing"
- **Solution**: Check system clock synchronization
- Verify TOTP secret is stored correctly
- Increase time window tolerance

**Problem**: "SMS not sending"
- **Solution**: Verify Twilio credentials
- Check phone number format (+1XXXXXXXXXX)
- Review Twilio account balance and limits

### SCIM Issues

**Problem**: "User provisioning failing"
- **Solution**: Verify bearer token is valid
- Check SCIM endpoint is accessible
- Review SCIM logs for specific errors

**Problem**: "Groups not updating"
- **Solution**: Verify group member updates are sent
- Check SCIM PATCH support is enabled
- Review webhook events for errors

---

## Support and Resources

### Documentation
- [SAML 2.0 Specification](http://docs.oasis-open.org/security/saml/v2.0/)
- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [OpenID Connect Specification](https://openid.net/specs/openid-connect-core-1_0.html)
- [SCIM 2.0 RFC 7644](https://tools.ietf.org/html/rfc7644)
- [TOTP RFC 6238](https://tools.ietf.org/html/rfc6238)

### Security Contacts
- Security Issues: security@greenlang.io
- Bug Reports: bugs@greenlang.io
- General Support: support@greenlang.io

### Version Information
- GreenLang Version: 1.0.0
- Phase: 4 (Enterprise Authentication)
- Last Updated: 2025-11-08

---

**Note**: This documentation is for GreenLang Phase 4 Enterprise Authentication. For production deployments, always follow your organization's security policies and compliance requirements.
