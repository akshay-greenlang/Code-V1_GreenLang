# Security Architecture

## 1. Zero-Trust Network Design

### Core Principles
- Never trust, always verify
- Least privilege access
- Assume breach mindset
- Continuous verification

### Network Architecture

```yaml
zones:
  dmz:
    description: "Public-facing services"
    components:
      - WAF (Cloudflare/AWS WAF)
      - API Gateway
      - Load Balancers
    security_controls:
      - DDoS protection
      - Rate limiting
      - Geo-blocking
      - Bot protection

  application:
    description: "Application tier"
    components:
      - Kubernetes clusters
      - Application servers
      - Service mesh (Istio)
    security_controls:
      - mTLS between services
      - Network policies
      - Service-to-service authentication
      - Runtime protection (Falco)

  data:
    description: "Data tier"
    components:
      - Databases (PostgreSQL, MongoDB)
      - Object storage (S3)
      - Message queues (Kafka)
    security_controls:
      - Encryption at rest (AES-256)
      - Database activity monitoring
      - Access logging
      - Data loss prevention

  management:
    description: "Management plane"
    components:
      - CI/CD systems
      - Monitoring tools
      - Security tools
    security_controls:
      - Privileged access management
      - Jump servers/bastion hosts
      - Session recording
      - MFA enforcement
```

## 2. Identity and Access Management (IAM)

### Architecture Components

```yaml
identity_provider:
  primary: "Okta/Auth0"
  protocols:
    - SAML 2.0
    - OAuth 2.0
    - OpenID Connect
  features:
    - Single Sign-On (SSO)
    - Multi-Factor Authentication (MFA)
    - Adaptive authentication
    - Risk-based access

rbac_model:
  roles:
    super_admin:
      permissions: ["all"]
      mfa_required: true
      session_timeout: 30

    platform_admin:
      permissions:
        - "platform:manage"
        - "users:manage"
        - "reports:all"
      mfa_required: true
      session_timeout: 60

    data_analyst:
      permissions:
        - "reports:read"
        - "data:read"
        - "dashboards:manage"
      mfa_required: false
      session_timeout: 480

    auditor:
      permissions:
        - "audit:read"
        - "reports:read"
        - "compliance:read"
      mfa_required: true
      session_timeout: 120

attribute_based_access:
  policies:
    - name: "data_residency"
      condition: "user.location == resource.region"
    - name: "time_based"
      condition: "current_time in user.access_window"
    - name: "ip_restriction"
      condition: "request.ip in user.allowed_ips"
```

## 3. Secret Management (HashiCorp Vault)

### Vault Configuration

```hcl
# vault-config.hcl
storage "consul" {
  address = "127.0.0.1:8500"
  path    = "vault/"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 0
  tls_cert_file = "/opt/vault/tls/cert.pem"
  tls_key_file  = "/opt/vault/tls/key.pem"
}

seal "awskms" {
  region     = "us-west-2"
  kms_key_id = "alias/vault-unseal"
}

api_addr = "https://vault.greenlang.io:8200"
cluster_addr = "https://vault.greenlang.io:8201"

ui = true
disable_mlock = false

telemetry {
  prometheus_retention_time = "0s"
  disable_hostname = false
}
```

### Secret Rotation Policy

```yaml
rotation_policies:
  database_credentials:
    rotation_period: "30d"
    rotation_window: "24h"
    notification_lead_time: "7d"

  api_keys:
    rotation_period: "90d"
    rotation_window: "48h"
    notification_lead_time: "14d"

  certificates:
    rotation_period: "365d"
    rotation_window: "30d"
    notification_lead_time: "60d"

  encryption_keys:
    rotation_period: "180d"
    rotation_window: "7d"
    notification_lead_time: "30d"
```

## 4. Encryption Standards

### Encryption at Rest

```yaml
encryption_at_rest:
  databases:
    algorithm: "AES-256-GCM"
    key_management: "AWS KMS / Azure Key Vault"
    key_rotation: "90 days"

  file_storage:
    algorithm: "AES-256-CBC"
    key_management: "HashiCorp Vault"
    key_rotation: "180 days"

  backups:
    algorithm: "AES-256-GCM"
    key_management: "Dedicated HSM"
    key_rotation: "365 days"
    additional: "Client-side encryption before backup"
```

### Encryption in Transit

```yaml
encryption_in_transit:
  external_traffic:
    protocol: "TLS 1.3"
    cipher_suites:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_AES_128_GCM_SHA256"
      - "TLS_CHACHA20_POLY1305_SHA256"
    certificate_authority: "Let's Encrypt / DigiCert"
    hsts_enabled: true
    hsts_max_age: 31536000

  internal_traffic:
    protocol: "mTLS"
    service_mesh: "Istio"
    certificate_rotation: "30 days"
    root_ca: "Internal PKI"

  database_connections:
    protocol: "TLS 1.2+"
    certificate_validation: "required"
    connection_string_encryption: true
```

## 5. API Security

### OAuth2 Configuration

```yaml
oauth2_config:
  authorization_server:
    issuer: "https://auth.greenlang.io"
    authorization_endpoint: "/oauth2/authorize"
    token_endpoint: "/oauth2/token"
    userinfo_endpoint: "/oauth2/userinfo"
    jwks_uri: "/oauth2/.well-known/jwks.json"

  supported_flows:
    - "authorization_code"
    - "client_credentials"
    - "refresh_token"

  token_settings:
    access_token_lifetime: 3600
    refresh_token_lifetime: 2592000
    id_token_lifetime: 3600

  security_features:
    pkce_required: true
    state_parameter_required: true
    nonce_required: true
```

### JWT Security

```javascript
// JWT Configuration
const jwtConfig = {
  algorithm: 'RS256',
  issuer: 'https://auth.greenlang.io',
  audience: 'https://api.greenlang.io',
  expiresIn: '1h',
  notBefore: '0s',
  clockTolerance: 30,

  claims: {
    required: ['sub', 'iat', 'exp', 'aud', 'iss'],
    custom: ['permissions', 'tenant_id', 'role']
  },

  validation: {
    algorithms: ['RS256', 'ES256'],
    issuer: true,
    audience: true,
    expiration: true,
    notBefore: true,
    signature: true
  }
};
```

### API Gateway Security

```yaml
api_gateway:
  rate_limiting:
    global:
      requests_per_second: 1000
      burst: 2000
    per_user:
      requests_per_minute: 100
      requests_per_hour: 1000
      requests_per_day: 10000

  authentication:
    methods:
      - bearer_token
      - api_key
      - mutual_tls
    token_validation:
      introspection_endpoint: "/oauth2/introspect"
      cache_ttl: 300

  authorization:
    type: "policy_based"
    engine: "Open Policy Agent"
    policies_refresh: 60

  security_headers:
    X-Content-Type-Options: "nosniff"
    X-Frame-Options: "DENY"
    X-XSS-Protection: "1; mode=block"
    Content-Security-Policy: "default-src 'self'"
    Strict-Transport-Security: "max-age=31536000; includeSubDomains"
```

## 6. Network Segmentation

### Microsegmentation Strategy

```yaml
network_policies:
  - name: "frontend-to-api"
    source:
      selector: "app=frontend"
    destination:
      selector: "app=api"
      ports: [443, 8080]
    action: "allow"

  - name: "api-to-database"
    source:
      selector: "app=api"
    destination:
      selector: "app=database"
      ports: [5432, 27017]
    action: "allow"

  - name: "deny-all-default"
    source:
      selector: "*"
    destination:
      selector: "*"
    action: "deny"

  - name: "monitoring-access"
    source:
      selector: "app=monitoring"
    destination:
      selector: "*"
      ports: [9090, 3000]
    action: "allow"
```

### VLAN Configuration

```yaml
vlans:
  - id: 10
    name: "dmz"
    subnet: "10.1.0.0/24"
    gateway: "10.1.0.1"

  - id: 20
    name: "application"
    subnet: "10.2.0.0/24"
    gateway: "10.2.0.1"

  - id: 30
    name: "database"
    subnet: "10.3.0.0/24"
    gateway: "10.3.0.1"

  - id: 40
    name: "management"
    subnet: "10.4.0.0/24"
    gateway: "10.4.0.1"

firewall_rules:
  - rule: "allow dmz to application on 443"
  - rule: "allow application to database on 5432"
  - rule: "allow management to all for administration"
  - rule: "deny all other traffic"
```

## 7. DDoS Protection

### CloudFlare Configuration

```yaml
ddos_protection:
  provider: "CloudFlare Enterprise"

  rate_limiting_rules:
    - threshold: 50
      period: 10
      action: "challenge"

    - threshold: 100
      period: 10
      action: "block"

  under_attack_mode:
    sensitivity: "high"
    challenge_passage: 30

  firewall_rules:
    - expression: "cf.threat_score > 30"
      action: "challenge"

    - expression: "cf.bot_score < 30"
      action: "block"

  origin_protection:
    - only_allow_cloudflare_ips: true
    - origin_certificates: true
    - authenticated_origin_pulls: true
```

### AWS Shield Advanced

```yaml
aws_shield:
  protection_groups:
    - name: "api-protection"
      resources:
        - "arn:aws:elasticloadbalancing:*:*:loadbalancer/app/*"
        - "arn:aws:cloudfront::*:distribution/*"

  detection_settings:
    - resource_type: "ELB"
      detection_threshold: 1000

    - resource_type: "CloudFront"
      detection_threshold: 5000

  response_team:
    enabled: true
    escalation_contacts:
      - "security@greenlang.io"
      - "oncall@greenlang.io"
```