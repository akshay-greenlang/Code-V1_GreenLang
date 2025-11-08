# GreenLang Enterprise Authentication - Quick Start Guide

Get started with GreenLang's enterprise authentication in 5 minutes!

## Installation

```bash
# Install core dependencies
pip install python3-saml authlib ldap3 pyotp twilio qrcode[pil] cryptography PyJWT
```

## Quick Examples

### 1. SAML 2.0 with Okta (Most Common)

```python
from greenlang.auth import SAMLProvider, create_okta_config

# Configure
config = create_okta_config(
    sp_entity_id="https://app.greenlang.io",
    sp_acs_url="https://app.greenlang.io/auth/saml/acs",
    okta_domain="your-company.okta.com",
    okta_app_id="0oa1234567890abcde",
    idp_cert="""-----BEGIN CERTIFICATE-----
MIIDpDCCAoygAwIBAgIGAXoqSTEKMA0GCS...
-----END CERTIFICATE-----"""
)

# Initialize provider
saml = SAMLProvider(config)

# Step 1: Generate login URL
auth_url, request_id = saml.get_auth_request_url()
# Redirect user to auth_url

# Step 2: Handle callback
user = saml.process_response(saml_response, request_id)
print(f"Authenticated: {user.email}")
```

### 2. OAuth with Google (Fastest Setup)

```python
from greenlang.auth import OAuthProvider, create_google_config

# Configure
config = create_google_config(
    client_id="123456789.apps.googleusercontent.com",
    client_secret="GOCSPX-xyz123",
    redirect_uri="https://app.greenlang.io/callback"
)

# Initialize provider
oauth = OAuthProvider(config)

# Step 1: Generate login URL
auth_url, state, verifier, nonce = oauth.get_authorization_url()
# Redirect user to auth_url

# Step 2: Handle callback
tokens = oauth.exchange_code_for_tokens(code, state, verifier)
user = oauth.create_user_from_tokens(tokens)
print(f"Authenticated: {user.email}")
```

### 3. LDAP/Active Directory

```python
from greenlang.auth import LDAPProvider, create_active_directory_config

# Configure
config = create_active_directory_config(
    server_uri="ldaps://dc.company.com:636",
    base_dn="dc=company,dc=com",
    bind_dn="cn=greenlang,cn=Users,dc=company,dc=com",
    bind_password="service-password",
    domain="COMPANY"
)

# Initialize provider
ldap = LDAPProvider(config)

# Authenticate user
user = ldap.authenticate("john.doe", "user-password")
if user:
    print(f"Authenticated: {user.email}")
    print(f"Groups: {user.groups}")
```

### 4. Multi-Factor Authentication (MFA)

```python
from greenlang.auth import MFAManager, MFAConfig, MFAMethod

# Configure
config = MFAConfig(
    totp_issuer="GreenLang",
    sms_enabled=True,
    twilio_account_sid="AC1234567890",
    twilio_auth_token="your-token",
    twilio_phone_number="+15551234567"
)

# Initialize manager
mfa = MFAManager(config)

# Enroll user in TOTP
device_id, secret, qr_code = mfa.enroll_totp("user-123", "My Phone")
# Display qr_code to user for scanning

# Verify enrollment
if mfa.verify_totp_enrollment("user-123", device_id, "123456"):
    print("MFA enrolled!")

# Later: Verify during login
if mfa.verify_mfa("user-123", MFAMethod.TOTP, "654321", device_id):
    print("MFA verified!")
```

### 5. SCIM User Provisioning

```python
from greenlang.auth import SCIMProvider, SCIMConfig

# Configure
config = SCIMConfig(
    base_url="https://api.greenlang.io/scim/v2",
    bearer_token="scim-token-xyz"
)

# Initialize provider
scim = SCIMProvider(config)

# Create user
user = scim.create_user({
    "userName": "alice",
    "name": {"givenName": "Alice", "familyName": "Smith"},
    "emails": [{"value": "alice@company.com", "primary": True}],
    "active": True
})

# Search users
results = scim.search_users(filter_expr='userName eq "alice"')
print(f"Found {results.totalResults} users")
```

## Environment Variables (Recommended)

Create a `.env` file:

```bash
# SAML
SAML_SP_ENTITY_ID=https://app.greenlang.io
SAML_SP_ACS_URL=https://app.greenlang.io/auth/saml/acs
SAML_IDP_ENTITY_ID=https://idp.example.com
SAML_IDP_SSO_URL=https://idp.example.com/sso
SAML_IDP_CERT=<paste-certificate>

# OAuth
OAUTH_CLIENT_ID=your-client-id
OAUTH_CLIENT_SECRET=your-client-secret
OAUTH_REDIRECT_URI=https://app.greenlang.io/callback
OAUTH_PROVIDER=google

# LDAP
LDAP_SERVER_URI=ldaps://ldap.company.com:636
LDAP_BASE_DN=dc=company,dc=com
LDAP_BIND_DN=cn=service,dc=company,dc=com
LDAP_BIND_PASSWORD=password

# MFA
MFA_TOTP_ISSUER=GreenLang
MFA_SMS_ENABLED=true
TWILIO_ACCOUNT_SID=AC1234567890
TWILIO_AUTH_TOKEN=your-token
TWILIO_PHONE_NUMBER=+15551234567

# SCIM
SCIM_BASE_URL=https://api.greenlang.io/scim/v2
SCIM_BEARER_TOKEN=scim-token
```

Then load config:

```python
from greenlang.auth.config_examples import get_auth_config_from_env

config = get_auth_config_from_env()
# Returns dict with all configured providers
```

## Common Patterns

### Pattern 1: OAuth + MFA

```python
from greenlang.auth import OAuthProvider, MFAManager, MFAMethod

# Step 1: OAuth authentication
oauth = OAuthProvider(oauth_config)
auth_url, state, verifier, nonce = oauth.get_authorization_url()
# ... user authenticates ...
tokens = oauth.exchange_code_for_tokens(code, state, verifier)
user = oauth.create_user_from_tokens(tokens)

# Step 2: MFA challenge
mfa = MFAManager(mfa_config)
if mfa.is_mfa_required(user.user_id, user.roles):
    # Prompt for MFA code
    if mfa.verify_mfa(user.user_id, MFAMethod.TOTP, code):
        # Grant access
        pass
```

### Pattern 2: SAML with Auto-Provisioning (SCIM)

```python
from greenlang.auth import SAMLProvider, SCIMProvider

saml = SAMLProvider(saml_config)
scim = SCIMProvider(scim_config)

# SAML authentication
user = saml.process_response(saml_response, request_id)

# Auto-provision user via SCIM
scim_user = scim.create_user({
    "userName": user.username,
    "name": {
        "givenName": user.first_name,
        "familyName": user.last_name
    },
    "emails": [{"value": user.email, "primary": True}],
    "active": True
})
```

### Pattern 3: LDAP with Group-Based RBAC

```python
from greenlang.auth import LDAPProvider, RBACManager

ldap = LDAPProvider(ldap_config)
rbac = RBACManager()

# Authenticate user
user = ldap.authenticate(username, password)

# Map LDAP groups to RBAC roles
role_mapping = {
    "CN=Admins,OU=Groups,DC=company,DC=com": "admin",
    "CN=Developers,OU=Groups,DC=company,DC=com": "developer",
    "CN=Users,OU=Groups,DC=company,DC=com": "user"
}

for group_dn in user.group_dns:
    if group_dn in role_mapping:
        rbac.assign_role(user.user_id, role_mapping[group_dn])
```

## Testing

```bash
# Run all tests
pytest tests/test_auth_providers.py -v

# Run specific test
pytest tests/test_auth_providers.py::TestSAMLProvider -v

# With coverage
pytest tests/test_auth_providers.py --cov=greenlang.auth --cov-report=html
```

## Docker Quick Start

```bash
# Build
docker build -t greenlang-auth .

# Run with environment file
docker run --env-file .env -p 8000:8000 greenlang-auth
```

## Troubleshooting

### SAML: "Invalid signature"
- ✅ Verify IdP certificate is correct
- ✅ Check clock synchronization
- ✅ Enable `debug=True` in config

### OAuth: "State mismatch"
- ✅ Check session/cookie settings
- ✅ Verify redirect URI matches exactly
- ✅ Use HTTPS in production

### LDAP: "Bind failed"
- ✅ Test connection: `ldap.test_connection()`
- ✅ Verify bind DN and password
- ✅ Check firewall/network

### MFA: "Invalid code"
- ✅ Check system time is synchronized
- ✅ Increase time window: `totp_window=2`
- ✅ Verify secret is stored correctly

## Next Steps

1. **Read full documentation**: `SECURITY_AUTH.md`
2. **Review examples**: `greenlang/auth/config_examples.py`
3. **Run tests**: `pytest tests/test_auth_providers.py`
4. **Deploy**: See `SECURITY_AUTH.md` deployment section

## Support

- 📧 Email: support@greenlang.io
- 📚 Docs: `SECURITY_AUTH.md`
- 🐛 Bugs: bugs@greenlang.io
- 🔒 Security: security@greenlang.io

Happy authenticating! 🚀
