# Authentication Guide

## GL-VCCI Scope 3 Platform Authentication

This guide covers authentication and authorization for the GL-VCCI API, including OAuth 2.0 flows, API key management, and best practices for secure access.

---

## Table of Contents

- [Overview](#overview)
- [Authentication Methods](#authentication-methods)
- [OAuth 2.0 Flows](#oauth-20-flows)
- [API Key Authentication](#api-key-authentication)
- [Multi-Tenant Access](#multi-tenant-access)
- [Token Management](#token-management)
- [Security Best Practices](#security-best-practices)
- [Code Examples](#code-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The GL-VCCI platform supports two primary authentication methods:

1. **OAuth 2.0** - Recommended for user-facing applications and interactive sessions
2. **API Keys** - Recommended for service-to-service communication and backend integrations

All API requests must include valid authentication credentials. Unauthenticated requests will receive a `401 Unauthorized` response.

---

## Authentication Methods

### When to Use OAuth 2.0

Use OAuth 2.0 for:
- Web applications with user login
- Mobile applications
- Single-page applications (SPAs)
- Desktop applications
- Any scenario requiring user-level permissions

**Advantages:**
- User-specific access control
- Token expiration and refresh
- Standard industry protocol
- Granular permission scopes

### When to Use API Keys

Use API keys for:
- Service-to-service communication
- Backend batch jobs
- CI/CD pipelines
- Scheduled data syncs
- Long-running processes

**Advantages:**
- Simpler implementation
- No token refresh needed
- Persistent authentication
- Machine-to-machine communication

---

## OAuth 2.0 Flows

The GL-VCCI platform supports multiple OAuth 2.0 flows to accommodate different application architectures.

### Supported Flows

| Flow | Use Case | Security |
|------|----------|----------|
| **Client Credentials** | Service-to-service | High |
| **Authorization Code** | Web/mobile apps | Highest |
| **Refresh Token** | Token renewal | High |

### 1. Client Credentials Flow

**Best for:** Backend services, APIs, batch jobs

This flow exchanges client credentials directly for an access token. No user interaction is required.

#### Request

```http
POST /v2/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
&client_id=vcci_client_abc123
&client_secret=sk_live_abc123xyz789
&scope=read write
```

#### Response

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "read write"
}
```

#### Python Example

```python
import requests

# Client credentials
CLIENT_ID = "vcci_client_abc123"
CLIENT_SECRET = "sk_live_abc123xyz789"
TOKEN_URL = "https://api.vcci.greenlang.io/v2/auth/token"

# Request access token
response = requests.post(
    TOKEN_URL,
    data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "read write"
    }
)

token_data = response.json()
access_token = token_data["access_token"]

# Use access token for API calls
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# Example: List suppliers
suppliers_response = requests.get(
    "https://api.vcci.greenlang.io/v2/suppliers",
    headers=headers
)
print(suppliers_response.json())
```

#### JavaScript Example

```javascript
const axios = require('axios');

const CLIENT_ID = 'vcci_client_abc123';
const CLIENT_SECRET = 'sk_live_abc123xyz789';
const TOKEN_URL = 'https://api.vcci.greenlang.io/v2/auth/token';

async function getAccessToken() {
  const response = await axios.post(TOKEN_URL, new URLSearchParams({
    grant_type: 'client_credentials',
    client_id: CLIENT_ID,
    client_secret: CLIENT_SECRET,
    scope: 'read write'
  }), {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    }
  });

  return response.data.access_token;
}

async function listSuppliers() {
  const token = await getAccessToken();

  const response = await axios.get(
    'https://api.vcci.greenlang.io/v2/suppliers',
    {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    }
  );

  return response.data;
}

// Usage
listSuppliers().then(data => console.log(data));
```

---

### 2. Authorization Code Flow

**Best for:** Web applications, mobile apps, SPAs with user login

This is the most secure flow for user-facing applications. It requires user interaction to authorize the application.

#### Step 1: Authorization Request

Redirect the user to the authorization endpoint:

```
GET https://api.vcci.greenlang.io/v2/auth/authorize?
  response_type=code
  &client_id=vcci_client_abc123
  &redirect_uri=https://yourapp.com/callback
  &scope=read write
  &state=random_state_string
```

**Parameters:**
- `response_type`: Must be `code`
- `client_id`: Your application's client ID
- `redirect_uri`: URL to redirect after authorization (must be pre-registered)
- `scope`: Space-separated list of permissions
- `state`: Random string to prevent CSRF attacks

#### Step 2: User Authorization

The user will be presented with a login screen and asked to authorize your application. After authorization, they are redirected to your `redirect_uri` with an authorization code:

```
https://yourapp.com/callback?code=AUTH_CODE_HERE&state=random_state_string
```

#### Step 3: Exchange Code for Token

Exchange the authorization code for an access token:

```http
POST /v2/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code
&client_id=vcci_client_abc123
&client_secret=sk_live_abc123xyz789
&code=AUTH_CODE_HERE
&redirect_uri=https://yourapp.com/callback
```

#### Response

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "scope": "read write"
}
```

#### Python Example (Flask)

```python
from flask import Flask, request, redirect, session
import requests

app = Flask(__name__)
app.secret_key = 'your-secret-key'

CLIENT_ID = "vcci_client_abc123"
CLIENT_SECRET = "sk_live_abc123xyz789"
REDIRECT_URI = "http://localhost:5000/callback"
AUTH_URL = "https://api.vcci.greenlang.io/v2/auth/authorize"
TOKEN_URL = "https://api.vcci.greenlang.io/v2/auth/token"

@app.route('/login')
def login():
    # Generate random state for CSRF protection
    import secrets
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state

    # Redirect to authorization URL
    auth_params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'read write',
        'state': state
    }
    auth_url = f"{AUTH_URL}?{'&'.join([f'{k}={v}' for k, v in auth_params.items()])}"
    return redirect(auth_url)

@app.route('/callback')
def callback():
    # Verify state to prevent CSRF
    if request.args.get('state') != session.get('oauth_state'):
        return "Invalid state parameter", 400

    # Exchange authorization code for access token
    code = request.args.get('code')
    response = requests.post(TOKEN_URL, data={
        'grant_type': 'authorization_code',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'redirect_uri': REDIRECT_URI
    })

    token_data = response.json()
    session['access_token'] = token_data['access_token']
    session['refresh_token'] = token_data['refresh_token']

    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    access_token = session.get('access_token')
    if not access_token:
        return redirect('/login')

    # Use access token to fetch data
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(
        'https://api.vcci.greenlang.io/v2/suppliers',
        headers=headers
    )

    return f"Suppliers: {response.json()}"
```

---

### 3. Refresh Token Flow

**Best for:** Renewing expired access tokens without user interaction

Access tokens expire after 1 hour. Use the refresh token to obtain a new access token without requiring the user to log in again.

#### Request

```http
POST /v2/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token
&client_id=vcci_client_abc123
&client_secret=sk_live_abc123xyz789
&refresh_token=REFRESH_TOKEN_HERE
```

#### Response

```json
{
  "access_token": "NEW_ACCESS_TOKEN",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "NEW_REFRESH_TOKEN",
  "scope": "read write"
}
```

#### Python Example

```python
def refresh_access_token(refresh_token):
    response = requests.post(
        "https://api.vcci.greenlang.io/v2/auth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": refresh_token
        }
    )

    token_data = response.json()
    return token_data["access_token"], token_data["refresh_token"]

# Automatic token refresh wrapper
class VCCIClient:
    def __init__(self, client_id, client_secret, refresh_token):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.refresh_token = refresh_token
        self.token_expiry = 0
        self.refresh_access_token()

    def refresh_access_token(self):
        import time
        response = requests.post(
            "https://api.vcci.greenlang.io/v2/auth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token
            }
        )
        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data["refresh_token"]
        self.token_expiry = time.time() + token_data["expires_in"]

    def get_headers(self):
        import time
        # Refresh token if expired (with 5 minute buffer)
        if time.time() >= (self.token_expiry - 300):
            self.refresh_access_token()

        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def get(self, endpoint):
        response = requests.get(
            f"https://api.vcci.greenlang.io/v2{endpoint}",
            headers=self.get_headers()
        )
        return response.json()

# Usage
client = VCCIClient(CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN)
suppliers = client.get("/suppliers")  # Automatically handles token refresh
```

---

## API Key Authentication

API keys provide a simpler authentication method for service-to-service communication.

### Creating an API Key

#### Request

```http
POST /v2/auth/api-keys
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json

{
  "name": "Production Integration Key",
  "scopes": ["read", "write:procurement"],
  "expires_at": "2025-12-31T23:59:59Z"
}
```

#### Response

```json
{
  "id": "key_abc123",
  "name": "Production Integration Key",
  "key": "sk_live_abc123xyz789def456ghi789jkl012",
  "prefix": "sk_live_abc",
  "scopes": ["read", "write:procurement"],
  "created_at": "2024-11-06T10:00:00Z",
  "expires_at": "2025-12-31T23:59:59Z"
}
```

**Important:** The full API key is only shown once during creation. Store it securely immediately.

### Using an API Key

Include the API key in the `X-API-Key` header:

```http
GET /v2/suppliers
X-API-Key: sk_live_abc123xyz789def456ghi789jkl012
```

#### Python Example

```python
import requests

API_KEY = "sk_live_abc123xyz789def456ghi789jkl012"
BASE_URL = "https://api.vcci.greenlang.io/v2"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# List suppliers
response = requests.get(f"{BASE_URL}/suppliers", headers=headers)
suppliers = response.json()

# Create transaction
transaction_data = {
    "transaction_type": "purchase_order",
    "transaction_id": "PO-2024-12345",
    "transaction_date": "2024-11-01",
    "supplier_name": "Acme Corporation",
    "product_name": "Steel beams",
    "quantity": 1000,
    "unit": "kg",
    "spend_usd": 5000.00,
    "currency": "USD"
}

response = requests.post(
    f"{BASE_URL}/procurement/transactions",
    headers=headers,
    json=transaction_data
)
transaction = response.json()
```

#### cURL Example

```bash
# List suppliers
curl -X GET "https://api.vcci.greenlang.io/v2/suppliers" \
  -H "X-API-Key: sk_live_abc123xyz789def456ghi789jkl012"

# Calculate emissions
curl -X POST "https://api.vcci.greenlang.io/v2/emissions/calculate" \
  -H "X-API-Key: sk_live_abc123xyz789def456ghi789jkl012" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"transaction_id": "txn_abc123"}
    ],
    "options": {
      "gwp_standard": "AR6",
      "uncertainty_method": "monte_carlo"
    }
  }'
```

### Managing API Keys

#### List All API Keys

```http
GET /v2/auth/api-keys
Authorization: Bearer YOUR_ACCESS_TOKEN
```

#### Revoke an API Key

```http
DELETE /v2/auth/api-keys/key_abc123
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Note:** Revoked API keys cannot be restored. Applications using the revoked key will immediately lose access.

---

## Multi-Tenant Access

The GL-VCCI platform is multi-tenant. Each organization (tenant) has its own isolated data namespace.

### Tenant Isolation

- **Data isolation:** Each tenant can only access their own data
- **Namespace isolation:** API keys and OAuth clients are scoped to a tenant
- **User isolation:** Users belong to a single tenant

### Accessing Multiple Tenants

If you need to access multiple tenants (e.g., for a service provider managing multiple customers):

1. **Option 1:** Create separate API keys for each tenant
2. **Option 2:** Use OAuth with user-level permissions (if the user has access to multiple tenants)

#### Example: Multi-Tenant Client

```python
class MultiTenantVCCIClient:
    def __init__(self, tenant_credentials):
        """
        tenant_credentials: dict mapping tenant_id to (client_id, client_secret)
        """
        self.clients = {}
        for tenant_id, (client_id, client_secret) in tenant_credentials.items():
            self.clients[tenant_id] = VCCIClient(client_id, client_secret)

    def get_suppliers(self, tenant_id):
        if tenant_id not in self.clients:
            raise ValueError(f"No credentials for tenant {tenant_id}")
        return self.clients[tenant_id].get("/suppliers")

# Usage
client = MultiTenantVCCIClient({
    "tenant_abc": ("client_1", "secret_1"),
    "tenant_def": ("client_2", "secret_2")
})

suppliers_abc = client.get_suppliers("tenant_abc")
suppliers_def = client.get_suppliers("tenant_def")
```

---

## Token Management

### Token Expiration

| Token Type | Default Lifetime | Can Refresh? |
|------------|------------------|--------------|
| Access Token | 1 hour | Yes (with refresh token) |
| Refresh Token | 30 days | No (get new via re-auth) |
| API Key | 1 year (configurable) | N/A |

### Token Storage Best Practices

#### Web Applications

```javascript
// Store tokens in memory (most secure)
let accessToken = null;
let refreshToken = null;

// OR use sessionStorage (cleared on tab close)
sessionStorage.setItem('access_token', token);

// AVOID localStorage for sensitive tokens (XSS risk)
// localStorage.setItem('access_token', token); // DON'T DO THIS
```

#### Backend Services

```python
# Store tokens in environment variables or secure vault
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

CLIENT_ID = os.getenv('VCCI_CLIENT_ID')
CLIENT_SECRET = os.getenv('VCCI_CLIENT_SECRET')
API_KEY = os.getenv('VCCI_API_KEY')
```

**Never:**
- Commit credentials to version control
- Log credentials in application logs
- Expose credentials in client-side code
- Share credentials across environments (dev/staging/prod)

---

## Security Best Practices

### 1. Use HTTPS Only

All API requests must use HTTPS. HTTP requests are not supported and will fail.

```python
# Good
BASE_URL = "https://api.vcci.greenlang.io/v2"

# Bad - will fail
BASE_URL = "http://api.vcci.greenlang.io/v2"
```

### 2. Rotate Credentials Regularly

- **API Keys:** Rotate every 90 days
- **Client Secrets:** Rotate every 180 days
- **Access Tokens:** Automatically expire after 1 hour

#### API Key Rotation Workflow

```python
# 1. Create new API key
new_key = create_api_key("Production Key v2")

# 2. Update applications to use new key
update_application_config(new_key)

# 3. Monitor for 24 hours
time.sleep(86400)

# 4. Revoke old key
revoke_api_key(old_key_id)
```

### 3. Use Least Privilege Scopes

Only request the scopes your application needs:

```python
# Good - specific scopes
scopes = "read write:procurement"

# Bad - overly broad
scopes = "read write admin"
```

### Available Scopes

| Scope | Description |
|-------|-------------|
| `read` | Read-only access to all resources |
| `write` | Write access to data resources |
| `write:procurement` | Write access to procurement data only |
| `write:suppliers` | Write access to supplier data only |
| `write:policies` | Write access to calculation policies |
| `admin` | Full administrative access |

### 4. Implement Token Refresh Logic

Always implement automatic token refresh to prevent service disruptions:

```python
import time
import requests
from functools import wraps

def require_auth(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if token is expired or about to expire (5 min buffer)
        if time.time() >= (self.token_expiry - 300):
            self.refresh_token()
        return func(self, *args, **kwargs)
    return wrapper

class VCCIClient:
    @require_auth
    def get_suppliers(self):
        # Token automatically refreshed if needed
        return self._make_request("GET", "/suppliers")
```

### 5. Handle Authentication Errors

```python
import requests
from requests.exceptions import HTTPError

def make_authenticated_request(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except HTTPError as e:
            if e.response.status_code == 401:
                # Authentication failed - refresh token and retry
                headers = refresh_and_get_headers()
                continue
            elif e.response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(e.response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
                continue
            else:
                raise

    raise Exception("Max retries exceeded")
```

### 6. Secure API Key Storage

#### Environment Variables (.env file)

```bash
# .env file (add to .gitignore!)
VCCI_API_KEY=sk_live_abc123xyz789def456ghi789jkl012
VCCI_CLIENT_ID=vcci_client_abc123
VCCI_CLIENT_SECRET=sk_live_secret123
```

```python
# Load from .env
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('VCCI_API_KEY')
```

#### AWS Secrets Manager

```python
import boto3
import json

def get_vcci_credentials():
    client = boto3.client('secretsmanager', region_name='us-west-2')
    secret = client.get_secret_value(SecretId='vcci/production')
    return json.loads(secret['SecretString'])

creds = get_vcci_credentials()
api_key = creds['api_key']
```

#### HashiCorp Vault

```python
import hvac

client = hvac.Client(url='http://vault:8200', token='vault-token')
secret = client.secrets.kv.v2.read_secret_version(path='vcci/production')
api_key = secret['data']['data']['api_key']
```

---

## Code Examples

### Complete Integration Example (Python)

```python
import os
import time
import requests
from typing import Optional, Dict, Any

class VCCIAPIClient:
    """
    Complete GL-VCCI API client with automatic token refresh,
    error handling, and retry logic.
    """

    def __init__(self, auth_method: str = "api_key"):
        self.base_url = "https://api.vcci.greenlang.io/v2"
        self.auth_method = auth_method

        if auth_method == "api_key":
            self.api_key = os.getenv('VCCI_API_KEY')
            if not self.api_key:
                raise ValueError("VCCI_API_KEY environment variable not set")
        elif auth_method == "oauth":
            self.client_id = os.getenv('VCCI_CLIENT_ID')
            self.client_secret = os.getenv('VCCI_CLIENT_SECRET')
            self.access_token = None
            self.refresh_token = os.getenv('VCCI_REFRESH_TOKEN')
            self.token_expiry = 0
            self._refresh_oauth_token()
        else:
            raise ValueError("auth_method must be 'api_key' or 'oauth'")

    def _refresh_oauth_token(self):
        """Refresh OAuth access token"""
        grant_type = "refresh_token" if self.refresh_token else "client_credentials"

        data = {
            "grant_type": grant_type,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        if grant_type == "refresh_token":
            data["refresh_token"] = self.refresh_token
        else:
            data["scope"] = "read write"

        response = requests.post(f"{self.base_url}/auth/token", data=data)
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.refresh_token = token_data.get("refresh_token", self.refresh_token)
        self.token_expiry = time.time() + token_data["expires_in"]

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if self.auth_method == "api_key":
            return {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
        else:
            # Check if token needs refresh (5 min buffer)
            if time.time() >= (self.token_expiry - 300):
                self._refresh_oauth_token()

            return {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Make authenticated API request with retry logic"""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                headers = self._get_headers()

                if method == "GET":
                    response = requests.get(url, headers=headers, params=params)
                elif method == "POST":
                    response = requests.post(url, headers=headers, json=data)
                elif method == "PATCH":
                    response = requests.patch(url, headers=headers, json=data)
                elif method == "DELETE":
                    response = requests.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()

                # Return empty dict for 204 No Content
                if response.status_code == 204:
                    return {}

                return response.json()

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401 and self.auth_method == "oauth":
                    # Token expired - refresh and retry
                    self._refresh_oauth_token()
                    continue
                elif e.response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                else:
                    # Other error - raise
                    print(f"API Error: {e.response.status_code} - {e.response.text}")
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Request failed. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        raise Exception("Max retries exceeded")

    # Supplier methods
    def list_suppliers(self, limit: int = 100, offset: int = 0, **filters):
        """List suppliers with optional filters"""
        params = {"limit": limit, "offset": offset, **filters}
        return self._make_request("GET", "/suppliers", params=params)

    def get_supplier(self, supplier_id: str):
        """Get supplier details"""
        return self._make_request("GET", f"/suppliers/{supplier_id}")

    def resolve_supplier(self, supplier_name: str, **hints):
        """Resolve supplier entity"""
        data = {"supplier_name": supplier_name, "hints": hints}
        return self._make_request("POST", "/suppliers/resolve", data=data)

    # Procurement methods
    def create_transaction(self, transaction_data: Dict[str, Any]):
        """Create procurement transaction"""
        return self._make_request("POST", "/procurement/transactions", data=transaction_data)

    def batch_create_transactions(self, transactions: list, async_mode: bool = False):
        """Batch create transactions"""
        data = {"transactions": transactions, "async": async_mode}
        return self._make_request("POST", "/procurement/transactions/batch", data=data)

    # Emissions methods
    def calculate_emissions(self, transaction_ids: list, **options):
        """Calculate emissions for transactions"""
        data = {
            "transactions": [{"transaction_id": tid} for tid in transaction_ids],
            "options": options
        }
        return self._make_request("POST", "/emissions/calculate", data=data)

    def get_aggregated_emissions(self, date_from: str, date_to: str, **filters):
        """Get aggregated emissions"""
        params = {"date_from": date_from, "date_to": date_to, **filters}
        return self._make_request("GET", "/emissions/aggregate", params=params)

    # Reports methods
    def generate_report(
        self,
        report_type: str,
        start_date: str,
        end_date: str,
        **options
    ):
        """Generate standards-compliant report"""
        data = {
            "report_type": report_type,
            "reporting_period": {
                "start_date": start_date,
                "end_date": end_date
            },
            **options
        }
        return self._make_request("POST", "/reports/generate", data=data)

    def get_report(self, report_id: str):
        """Get report status and download URL"""
        return self._make_request("GET", f"/reports/{report_id}")


# Usage Example
if __name__ == "__main__":
    # Initialize client with API key
    client = VCCIAPIClient(auth_method="api_key")

    # List suppliers
    suppliers = client.list_suppliers(limit=10, has_lei=True)
    print(f"Found {len(suppliers['data'])} suppliers")

    # Create transaction
    transaction = client.create_transaction({
        "transaction_type": "purchase_order",
        "transaction_id": "PO-2024-12345",
        "transaction_date": "2024-11-01",
        "supplier_name": "Acme Corporation",
        "product_name": "Steel beams",
        "quantity": 1000,
        "unit": "kg",
        "spend_usd": 5000.00
    })
    print(f"Created transaction: {transaction['id']}")

    # Calculate emissions
    emissions = client.calculate_emissions(
        [transaction['id']],
        gwp_standard="AR6",
        uncertainty_method="monte_carlo"
    )
    print(f"Emissions: {emissions['summary']['total_emissions_kg_co2e']} kg CO2e")

    # Generate ESRS report
    report = client.generate_report(
        "esrs_e1",
        "2024-01-01",
        "2024-12-31",
        output_format="pdf"
    )
    print(f"Report generating: {report['report_id']}")
```

---

## Troubleshooting

### Common Authentication Errors

#### 401 Unauthorized

**Cause:** Invalid or expired credentials

**Solutions:**
1. Verify API key is correct and not expired
2. Check that OAuth token hasn't expired
3. Ensure you're using the correct authentication method
4. Verify the `Authorization` or `X-API-Key` header is properly formatted

```python
# Check token expiry
import jwt

token = "YOUR_ACCESS_TOKEN"
decoded = jwt.decode(token, options={"verify_signature": False})
print(f"Token expires at: {decoded['exp']}")
```

#### 403 Forbidden

**Cause:** Insufficient permissions for the requested resource

**Solutions:**
1. Check that your API key/token has the required scopes
2. Verify you have access to the tenant/resource
3. Contact support to review your permissions

#### 429 Too Many Requests

**Cause:** Rate limit exceeded

**Solutions:**
1. Implement exponential backoff with retry logic
2. Use batch endpoints where available
3. Contact support to increase rate limits for your plan

```python
# Handle rate limiting
retry_after = int(response.headers.get('Retry-After', 60))
time.sleep(retry_after)
```

### Testing Authentication

```bash
# Test API key
curl -X GET "https://api.vcci.greenlang.io/v2/health" \
  -H "X-API-Key: YOUR_API_KEY"

# Test OAuth token
curl -X GET "https://api.vcci.greenlang.io/v2/suppliers" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Getting Help

If you continue to experience authentication issues:

1. **Check API Status:** https://status.vcci.greenlang.io
2. **Review Documentation:** https://docs.vcci.greenlang.io
3. **Contact Support:** support@greenlang.io
4. **Report Issues:** Include request ID from `X-Request-ID` response header

---

## Next Steps

- [API Reference Guide](./API_REFERENCE.md) - Detailed endpoint documentation
- [Rate Limits & Best Practices](./RATE_LIMITS.md) - Optimize API usage
- [Quickstart Guide](./integrations/QUICKSTART.md) - Get started in 15 minutes
- [Python SDK](./integrations/PYTHON_SDK.md) - Official Python SDK usage
- [JavaScript SDK](./integrations/JAVASCRIPT_SDK.md) - Official JS/Node SDK usage

---

**Last Updated:** November 6, 2025
**Version:** 2.0.0
