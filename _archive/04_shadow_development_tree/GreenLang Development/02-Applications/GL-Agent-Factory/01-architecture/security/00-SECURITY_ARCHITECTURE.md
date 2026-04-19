# GreenLang Agent Factory: Security Architecture

**Version:** 1.0.0
**Date:** December 3, 2025
**Status:** ARCHITECTURE SPECIFICATION
**Classification:** Technical Architecture Document

---

## Overview

This document defines the comprehensive security architecture for the GreenLang Agent Factory, including authentication, authorization, tenant isolation, data security, and agent sandboxing.

---

## 1. Security Architecture Overview

### 1.1 Defense in Depth Model

```
+===========================================================================+
|                        Defense in Depth Layers                             |
+===========================================================================+

     Layer 1: Perimeter Security
     +------------------------------------------------------------------+
     |  - WAF (Web Application Firewall)                                 |
     |  - DDoS Protection (AWS Shield / CloudFlare)                      |
     |  - IP Allowlisting / Blocklisting                                 |
     |  - Rate Limiting (Global)                                         |
     +------------------------------------------------------------------+
                                    |
     Layer 2: Network Security
     +------------------------------------------------------------------+
     |  - TLS 1.3 Encryption (External)                                  |
     |  - mTLS (Internal Service-to-Service)                             |
     |  - Network Policies (Kubernetes)                                  |
     |  - VPC Isolation / Security Groups                                |
     +------------------------------------------------------------------+
                                    |
     Layer 3: Application Security
     +------------------------------------------------------------------+
     |  - Authentication (JWT / OAuth 2.0)                               |
     |  - Authorization (RBAC / ABAC)                                    |
     |  - Input Validation (Whitelist-based)                             |
     |  - API Rate Limiting (Per-tenant)                                 |
     +------------------------------------------------------------------+
                                    |
     Layer 4: Data Security
     +------------------------------------------------------------------+
     |  - Encryption at Rest (AES-256)                                   |
     |  - Tenant Data Isolation (Separate DBs)                           |
     |  - Row-Level Security (RLS)                                       |
     |  - Data Masking / Tokenization                                    |
     +------------------------------------------------------------------+
                                    |
     Layer 5: Agent Sandboxing
     +------------------------------------------------------------------+
     |  - Container Isolation (gVisor / Kata)                            |
     |  - Resource Limits (CPU, Memory, Network)                         |
     |  - Capability Dropping                                            |
     |  - Read-Only Filesystems                                          |
     +------------------------------------------------------------------+
                                    |
     Layer 6: Monitoring & Audit
     +------------------------------------------------------------------+
     |  - Security Event Logging                                         |
     |  - Intrusion Detection                                            |
     |  - Compliance Auditing                                            |
     |  - Anomaly Detection                                              |
     +------------------------------------------------------------------+
```

### 1.2 Security Zones

```
+===========================================================================+
|                        Security Zone Architecture                          |
+===========================================================================+

                        INTERNET (Untrusted)
                               |
                        +------v------+
                        |    DMZ      |  Zone: PUBLIC
                        | - WAF       |  Trust: None
                        | - LB        |  Access: Anonymous
                        +------+------+
                               |
                        +------v------+
                        | API Gateway |  Zone: EDGE
                        | - Auth      |  Trust: Authenticated
                        | - Rate Limit|  Access: API Key/JWT
                        +------+------+
                               |
        +----------------------+----------------------+
        |                      |                      |
+-------v-------+      +-------v-------+      +-------v-------+
| Application   |      | Data Services |      | Agent Runtime |
| Zone          |      | Zone          |      | Zone          |
| Trust: High   |      | Trust: Highest|      | Trust: Medium |
| - Factory     |      | - PostgreSQL  |      | - Sandboxed   |
| - Registry    |      | - Redis       |      | - Isolated    |
| - Workers     |      | - Kafka       |      | - Monitored   |
+---------------+      +---------------+      +---------------+
```

---

## 2. Authentication & Authorization

### 2.1 Authentication Flow

```
+===========================================================================+
|                        Authentication Flow                                 |
+===========================================================================+

     Client                 API Gateway              Auth Service
        |                       |                         |
        | 1. Request + Creds    |                         |
        |---------------------->|                         |
        |                       |                         |
        |                       | 2. Validate Token       |
        |                       |------------------------>|
        |                       |                         |
        |                       |                         | 3. Verify JWT
        |                       |                         |    - Signature
        |                       |                         |    - Expiry
        |                       |                         |    - Claims
        |                       |                         |
        |                       | 4. Token Valid          |
        |                       |<------------------------|
        |                       |    + User Context       |
        |                       |                         |
        | 5. Authorized Request |                         |
        |<----------------------|                         |
        |                       |                         |
```

### 2.2 JWT Token Structure

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT",
    "kid": "key-id-2025-12"
  },
  "payload": {
    "iss": "https://auth.greenlang.ai",
    "sub": "user-uuid-12345",
    "aud": "greenlang-api",
    "exp": 1733313600,
    "iat": 1733227200,
    "jti": "unique-token-id",

    "tenant_id": "tenant-uuid-67890",
    "org_id": "org-uuid-11111",
    "roles": ["agent_creator", "viewer"],
    "permissions": [
      "agent:create",
      "agent:read",
      "agent:execute"
    ],
    "tier": "enterprise",
    "quotas": {
      "agents": 1000,
      "api_calls_per_minute": 10000
    }
  },
  "signature": "..."
}
```

### 2.3 Authentication Methods

```yaml
# Authentication Configuration
authentication:
  methods:
    jwt:
      enabled: true
      algorithm: "RS256"
      issuer: "https://auth.greenlang.ai"
      audience: "greenlang-api"
      jwks_uri: "https://auth.greenlang.ai/.well-known/jwks.json"
      token_expiry: 3600  # 1 hour
      refresh_token_expiry: 604800  # 7 days

    api_key:
      enabled: true
      prefix: "glk_"
      hash_algorithm: "sha256"
      rotation_days: 90
      max_keys_per_user: 5

    oauth2:
      enabled: true
      providers:
        - name: "google"
          client_id: "${GOOGLE_CLIENT_ID}"
          scopes: ["openid", "email", "profile"]
        - name: "microsoft"
          client_id: "${MICROSOFT_CLIENT_ID}"
          scopes: ["openid", "email", "profile"]
        - name: "github"
          client_id: "${GITHUB_CLIENT_ID}"
          scopes: ["user:email"]

    saml:
      enabled: true
      enterprise_only: true
      metadata_url: "/saml/metadata"
      assertion_consumer_service: "/saml/acs"

    mfa:
      required_for:
        - "admin"
        - "enterprise"
      methods:
        - "totp"
        - "sms"
        - "email"
```

### 2.4 Role-Based Access Control (RBAC)

```yaml
# RBAC Configuration
rbac:
  roles:
    # Platform Roles
    platform_admin:
      description: "Full platform access"
      permissions:
        - "*:*"

    # Organization Roles
    org_admin:
      description: "Organization administrator"
      permissions:
        - "org:*"
        - "tenant:*"
        - "user:*"
        - "agent:*"
        - "billing:read"

    org_member:
      description: "Organization member"
      permissions:
        - "org:read"
        - "agent:read"
        - "agent:execute"

    # Agent Roles
    agent_creator:
      description: "Can create and manage agents"
      permissions:
        - "agent:create"
        - "agent:read"
        - "agent:update"
        - "agent:delete"
        - "agent:execute"
        - "agent:deploy"

    agent_executor:
      description: "Can execute agents"
      permissions:
        - "agent:read"
        - "agent:execute"

    agent_viewer:
      description: "Read-only agent access"
      permissions:
        - "agent:read"
        - "execution:read"

    # Billing Roles
    billing_admin:
      description: "Billing management"
      permissions:
        - "billing:*"
        - "usage:read"

  # Permission Hierarchy
  permissions:
    agent:
      - create
      - read
      - update
      - delete
      - execute
      - deploy
      - rollback

    execution:
      - create
      - read
      - cancel

    org:
      - create
      - read
      - update
      - delete
      - invite
      - remove

    tenant:
      - create
      - read
      - update
      - suspend
      - delete

    billing:
      - read
      - update
      - admin
```

### 2.5 Authorization Middleware

```python
# Authorization Middleware Implementation
from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader

class AuthorizationMiddleware:
    """
    RBAC/ABAC authorization middleware.
    Validates permissions before request processing.
    """

    def __init__(self):
        self.permission_cache = {}

    async def check_permission(
        self,
        user: User,
        resource: str,
        action: str,
        resource_id: Optional[str] = None
    ) -> bool:
        """
        Check if user has permission to perform action on resource.

        Args:
            user: Authenticated user context
            resource: Resource type (e.g., "agent", "execution")
            action: Action (e.g., "create", "read", "execute")
            resource_id: Optional specific resource ID

        Returns:
            True if authorized, raises HTTPException otherwise
        """
        required_permission = f"{resource}:{action}"

        # Check role-based permissions
        user_permissions = self._get_user_permissions(user)
        if required_permission not in user_permissions:
            if "*:*" not in user_permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {required_permission}"
                )

        # Check resource-level access (ABAC)
        if resource_id:
            if not await self._check_resource_access(user, resource, resource_id):
                raise HTTPException(
                    status_code=403,
                    detail="Access denied to this resource"
                )

        return True

    async def _check_resource_access(
        self,
        user: User,
        resource: str,
        resource_id: str
    ) -> bool:
        """Check tenant-scoped resource access."""
        # Verify resource belongs to user's tenant
        resource_tenant = await self._get_resource_tenant(resource, resource_id)
        return resource_tenant == user.tenant_id
```

---

## 3. Tenant Isolation

### 3.1 Multi-Layer Tenant Isolation

```
+===========================================================================+
|                    Multi-Layer Tenant Isolation                            |
+===========================================================================+

Layer 1: Database Isolation (Complete Separation)
+------------------------------------------------------------------+
|                                                                    |
|   Tenant A                    Tenant B                            |
|   +------------------+        +------------------+                |
|   | greenlang_       |        | greenlang_       |                |
|   | tenant_aaaa1111  |        | tenant_bbbb2222  |                |
|   +------------------+        +------------------+                |
|   | - agents         |        | - agents         |                |
|   | - executions     |        | - executions     |                |
|   | - audit_logs     |        | - audit_logs     |                |
|   | - configs        |        | - configs        |                |
|   +------------------+        +------------------+                |
|                                                                    |
+------------------------------------------------------------------+

Layer 2: Row-Level Security (Additional Protection)
+------------------------------------------------------------------+
|                                                                    |
|   CREATE POLICY tenant_isolation ON agents                        |
|   FOR ALL                                                          |
|   USING (tenant_id = current_setting('app.current_tenant')::uuid) |
|   WITH CHECK (tenant_id = current_setting('app.current_tenant')::uuid);
|                                                                    |
+------------------------------------------------------------------+

Layer 3: Application-Level Checks
+------------------------------------------------------------------+
|                                                                    |
|   class TenantContext:                                             |
|       def __init__(self, tenant_id: str):                         |
|           self.tenant_id = tenant_id                               |
|                                                                    |
|       def validate_access(self, resource_tenant_id: str) -> bool: |
|           if resource_tenant_id != self.tenant_id:                |
|               raise TenantAccessDenied()                          |
|           return True                                              |
|                                                                    |
+------------------------------------------------------------------+

Layer 4: Network Isolation (Kubernetes)
+------------------------------------------------------------------+
|                                                                    |
|   NetworkPolicy: Restrict cross-tenant pod communication          |
|   Namespace: Optional per-tenant namespace                        |
|   Service Mesh: mTLS with tenant identity                         |
|                                                                    |
+------------------------------------------------------------------+
```

### 3.2 Tenant Database Configuration

```python
# Tenant Manager Implementation
class TenantManager:
    """
    Production-grade multi-tenancy with complete database isolation.
    Resolves CWE-639 (Data Leakage Between Tenants).
    """

    async def create_tenant(
        self,
        slug: str,
        metadata: TenantMetadata,
        tier: TenantTier = TenantTier.STARTER
    ) -> Tenant:
        """
        Create new tenant with isolated database.

        Creates:
        1. Tenant record in master database
        2. Dedicated PostgreSQL database
        3. Default tables and indexes
        4. Row-level security policies
        """
        tenant_id = str(uuid.uuid4())
        database_name = f"greenlang_tenant_{tenant_id.replace('-', '')[:12]}"

        # Create tenant database
        await self._create_tenant_database(database_name)

        # Apply schema migrations
        await self._apply_tenant_schema(database_name)

        # Configure RLS policies
        await self._configure_rls(database_name)

        # Generate API key
        api_key = self._generate_api_key()
        api_key_hash = self._hash_api_key(api_key)

        # Create tenant record
        tenant = Tenant(
            id=tenant_id,
            slug=slug,
            database_name=database_name,
            api_key_hash=api_key_hash,
            tier=tier,
            quotas=TIER_QUOTAS[tier],
            metadata=metadata
        )

        await self._save_tenant(tenant)

        return tenant, api_key  # Return unhashed key once

    async def get_tenant_connection(self, tenant_id: str) -> Connection:
        """Get database connection for specific tenant."""
        tenant = await self.get_tenant(tenant_id)

        # Return pooled connection to tenant database
        return await self._get_connection(tenant.database_name)

    async def execute_query(
        self,
        tenant_id: str,
        query: str,
        params: List[Any]
    ) -> List[Dict]:
        """Execute query in tenant's database context."""
        conn = await self.get_tenant_connection(tenant_id)

        # Set tenant context for RLS
        await conn.execute(
            f"SET app.current_tenant = '{tenant_id}'"
        )

        # Execute query
        return await conn.fetch(query, *params)
```

### 3.3 Tenant Tier Quotas

```yaml
# Tier-Based Resource Quotas
tier_quotas:
  free:
    agents: 10
    users: 1
    api_calls_per_minute: 100
    storage_gb: 1
    llm_tokens_per_day: 10000
    executions_per_day: 1000
    retention_days: 30

  starter:
    agents: 100
    users: 10
    api_calls_per_minute: 1000
    storage_gb: 10
    llm_tokens_per_day: 100000
    executions_per_day: 10000
    retention_days: 90

  professional:
    agents: 1000
    users: 100
    api_calls_per_minute: 10000
    storage_gb: 100
    llm_tokens_per_day: 1000000
    executions_per_day: 100000
    retention_days: 365

  enterprise:
    agents: 10000
    users: 1000
    api_calls_per_minute: 100000
    storage_gb: 10000
    llm_tokens_per_day: 10000000
    executions_per_day: 1000000
    retention_days: 2555  # 7 years

# Quota Enforcement
quota_enforcement:
  soft_limit_warning: 80  # Warn at 80%
  hard_limit_action: "reject"  # Reject or throttle
  grace_period_minutes: 5
  overage_allowed_percent: 10  # For enterprise only
```

---

## 4. Data Security

### 4.1 Encryption Architecture

```
+===========================================================================+
|                        Encryption Architecture                             |
+===========================================================================+

Data in Transit:
+------------------------------------------------------------------+
|                                                                    |
|   External Traffic                                                 |
|   [Client] <--TLS 1.3--> [Load Balancer] <--TLS 1.3--> [Gateway]  |
|                                                                    |
|   Internal Traffic                                                 |
|   [Service A] <--mTLS--> [Service B] <--mTLS--> [Database]       |
|                                                                    |
|   Configuration:                                                   |
|   - TLS Version: 1.3 (minimum 1.2)                                |
|   - Cipher Suites: TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305 |
|   - Certificate: Let's Encrypt (auto-renewed)                      |
|   - HSTS: Enabled (max-age=31536000)                              |
|                                                                    |
+------------------------------------------------------------------+

Data at Rest:
+------------------------------------------------------------------+
|                                                                    |
|   Database Encryption                                              |
|   +------------------+                                             |
|   | PostgreSQL      |  AES-256 (aws:kms)                          |
|   | - Data files    |  Key: per-tenant CMK                        |
|   | - WAL logs      |  Rotation: 365 days                         |
|   +------------------+                                             |
|                                                                    |
|   Cache Encryption                                                 |
|   +------------------+                                             |
|   | Redis           |  AES-256-CBC                                 |
|   | - Memory        |  Key: cluster key                           |
|   | - RDB/AOF       |  Rotation: 90 days                          |
|   +------------------+                                             |
|                                                                    |
|   Object Storage                                                   |
|   +------------------+                                             |
|   | S3              |  SSE-KMS                                     |
|   | - Agent packs   |  Key: per-bucket CMK                        |
|   | - Audit logs    |  Rotation: 365 days                         |
|   +------------------+                                             |
|                                                                    |
+------------------------------------------------------------------+

Sensitive Field Encryption:
+------------------------------------------------------------------+
|                                                                    |
|   Application-Level Encryption (ALE)                              |
|   +----------------------------------+                             |
|   | Field          | Encryption     |                             |
|   +----------------------------------+                             |
|   | API Keys       | SHA-256 Hash   |                             |
|   | Passwords      | Argon2id       |                             |
|   | PII Fields     | AES-256-GCM    |                             |
|   | Secrets        | Vault Transit  |                             |
|   +----------------------------------+                             |
|                                                                    |
+------------------------------------------------------------------+
```

### 4.2 Key Management

```yaml
# Key Management Configuration
key_management:
  provider: "aws-kms"  # or hashicorp-vault

  master_keys:
    platform:
      key_id: "alias/greenlang-platform"
      rotation_days: 365
      usage: "platform-wide encryption"

    tenant:
      key_id: "alias/greenlang-tenant-{tenant_id}"
      rotation_days: 365
      usage: "per-tenant encryption"

    api:
      key_id: "alias/greenlang-api"
      rotation_days: 90
      usage: "api key hashing"

  data_keys:
    generation: "envelope-encryption"
    caching: true
    cache_ttl: 300

  vault_config:
    enabled: true
    address: "https://vault.greenlang.ai:8200"
    auth_method: "kubernetes"
    role: "greenlang-api"
    secrets_path: "secret/data/greenlang"
    transit_path: "transit"
```

### 4.3 Data Classification

```yaml
# Data Classification Schema
data_classification:
  levels:
    public:
      description: "Publicly available information"
      encryption: "none"
      retention: "indefinite"
      examples:
        - "Documentation"
        - "Public API schemas"

    internal:
      description: "Internal business data"
      encryption: "in-transit"
      retention: "5 years"
      examples:
        - "Agent configurations"
        - "Usage metrics"

    confidential:
      description: "Sensitive business data"
      encryption: "at-rest + in-transit"
      retention: "7 years"
      examples:
        - "Tenant data"
        - "Calculation results"
        - "Audit logs"

    restricted:
      description: "Highly sensitive data"
      encryption: "field-level + at-rest + in-transit"
      retention: "as required"
      access: "need-to-know"
      examples:
        - "API keys"
        - "Credentials"
        - "PII"
        - "Financial data"

  handling_rules:
    restricted:
      - "Never log in plaintext"
      - "Mask in UI displays"
      - "Encrypt before storage"
      - "Audit all access"
      - "Tokenize where possible"
```

---

## 5. Agent Sandboxing

### 5.1 Container Isolation Architecture

```
+===========================================================================+
|                        Agent Sandbox Architecture                          |
+===========================================================================+

                    Kubernetes Node
+------------------------------------------------------------------+
|                                                                    |
|   +------------------+     +------------------+                    |
|   | Agent Container  |     | Agent Container  |                    |
|   | (gVisor/Kata)    |     | (gVisor/Kata)    |                    |
|   +------------------+     +------------------+                    |
|   | - Read-only FS   |     | - Read-only FS   |                    |
|   | - No privileged  |     | - No privileged  |                    |
|   | - CPU limit      |     | - CPU limit      |                    |
|   | - Memory limit   |     | - Memory limit   |                    |
|   | - No host net    |     | - No host net    |                    |
|   | - seccomp profile|     | - seccomp profile|                    |
|   | - AppArmor       |     | - AppArmor       |                    |
|   +------------------+     +------------------+                    |
|          |                        |                                |
|          v                        v                                |
|   +------------------------------------------+                     |
|   |           Network Policy                  |                     |
|   | - Deny inter-pod (default)               |                     |
|   | - Allow to API only                      |                     |
|   | - Egress whitelist                       |                     |
|   +------------------------------------------+                     |
|                                                                    |
+------------------------------------------------------------------+
```

### 5.2 Pod Security Configuration

```yaml
# Pod Security Policy for Agent Sandbox
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: agent-sandbox-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false

  # Drop all capabilities
  requiredDropCapabilities:
    - ALL

  # Volume restrictions
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'

  hostNetwork: false
  hostIPC: false
  hostPID: false

  runAsUser:
    rule: 'MustRunAsNonRoot'

  runAsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535

  seLinux:
    rule: 'RunAsAny'

  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535

  readOnlyRootFilesystem: true

  # Seccomp profile
  seccompProfiles:
    - 'runtime/default'

---
# Security Context for Agent Pods
apiVersion: v1
kind: Pod
metadata:
  name: agent-instance
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault

  containers:
    - name: agent
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop:
            - ALL
        seccompProfile:
          type: RuntimeDefault

      resources:
        limits:
          cpu: "500m"
          memory: "512Mi"
          ephemeral-storage: "100Mi"
        requests:
          cpu: "100m"
          memory: "128Mi"

      volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /home/agent/.cache

  volumes:
    - name: tmp
      emptyDir:
        sizeLimit: "50Mi"
    - name: cache
      emptyDir:
        sizeLimit: "50Mi"
```

### 5.3 Runtime Security Controls

```yaml
# Runtime Security Configuration
runtime_security:
  gvisor:
    enabled: true
    runtime_class: "gvisor"
    platform: "kvm"  # or ptrace

  seccomp:
    enabled: true
    profile: "runtime/default"
    custom_profile: |
      {
        "defaultAction": "SCMP_ACT_ERRNO",
        "architectures": ["SCMP_ARCH_X86_64"],
        "syscalls": [
          {
            "names": ["read", "write", "close", "fstat", "mmap", "mprotect", "munmap", "brk", "rt_sigaction", "rt_sigprocmask", "ioctl", "access", "clone", "execve", "wait4", "exit_group", "arch_prctl", "futex", "set_tid_address", "set_robust_list", "prlimit64", "openat", "newfstatat", "getdents64", "lseek", "pipe2", "dup2", "fcntl", "getrandom"],
            "action": "SCMP_ACT_ALLOW"
          }
        ]
      }

  apparmor:
    enabled: true
    profile: "greenlang-agent"

  capabilities:
    drop_all: true
    add: []  # No capabilities added

  filesystem:
    read_only_root: true
    allowed_writable:
      - "/tmp"
      - "/home/agent/.cache"

  network:
    host_network: false
    dns_policy: "ClusterFirst"
    egress_whitelist:
      - "api.greenlang.ai:443"
      - "*.amazonaws.com:443"
```

### 5.4 Resource Limits and Quotas

```yaml
# Agent Resource Limits
resource_limits:
  per_agent:
    cpu:
      request: "100m"
      limit: "500m"
    memory:
      request: "128Mi"
      limit: "512Mi"
    ephemeral_storage:
      limit: "100Mi"
    network:
      bandwidth_limit: "10Mbps"
      connections_limit: 100

  per_tenant:
    concurrent_agents: 100  # Enterprise: 1000
    total_cpu: "10"
    total_memory: "20Gi"
    total_storage: "100Gi"

  timeouts:
    execution_timeout: 300  # 5 minutes
    idle_timeout: 60  # 1 minute
    startup_timeout: 30  # 30 seconds

# Resource Quota (Kubernetes)
apiVersion: v1
kind: ResourceQuota
metadata:
  name: agent-quota
  namespace: greenlang-agents
spec:
  hard:
    requests.cpu: "50"
    requests.memory: "100Gi"
    limits.cpu: "100"
    limits.memory: "200Gi"
    pods: "1000"
    persistentvolumeclaims: "100"
```

---

## 6. Input Validation & Security

### 6.1 Input Validation Framework

```python
# Input Validation Implementation
from pydantic import BaseModel, validator, Field
import re

class InputValidator:
    """
    Whitelist-based input validation framework.
    Prevents SQL injection, command injection, XSS, SSRF.
    """

    # Allowed patterns (whitelist)
    PATTERNS = {
        "alphanumeric": r"^[a-zA-Z0-9_-]+$",
        "email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "path": r"^[a-zA-Z0-9/_.-]+$",
        "url": r"^https://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9._-]*)*$",
    }

    # Blocked patterns (blacklist for detection)
    BLOCKED_PATTERNS = {
        "sql_injection": r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
        "command_injection": r"[;&|`$]",
        "path_traversal": r"\.\./",
        "xss": r"<script|javascript:|on\w+=",
    }

    @classmethod
    def validate_alphanumeric(
        cls,
        value: str,
        field_name: str,
        min_length: int = 1,
        max_length: int = 255
    ) -> str:
        """Validate alphanumeric input."""
        if not value or len(value) < min_length or len(value) > max_length:
            raise ValueError(f"{field_name} must be {min_length}-{max_length} characters")

        if not re.match(cls.PATTERNS["alphanumeric"], value):
            raise ValueError(f"{field_name} contains invalid characters")

        # Check for blocked patterns
        for pattern_name, pattern in cls.BLOCKED_PATTERNS.items():
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(f"{field_name} contains blocked pattern: {pattern_name}")

        return value

    @classmethod
    def validate_sql_param(cls, value: str, field_name: str) -> str:
        """Validate SQL parameter to prevent injection."""
        if re.search(cls.BLOCKED_PATTERNS["sql_injection"], value, re.IGNORECASE):
            raise ValueError(f"{field_name} contains potential SQL injection")
        return value

    @classmethod
    def validate_path(cls, value: str, field_name: str) -> str:
        """Validate file path to prevent traversal."""
        if re.search(cls.BLOCKED_PATTERNS["path_traversal"], value):
            raise ValueError(f"{field_name} contains path traversal attempt")

        if not re.match(cls.PATTERNS["path"], value):
            raise ValueError(f"{field_name} contains invalid path characters")

        return value


# Pydantic Models with Validation
class AgentCreateRequest(BaseModel):
    """Validated agent creation request."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=1000)
    pack_yaml: str = Field(..., max_length=100000)

    @validator("name")
    def validate_name(cls, v):
        return InputValidator.validate_alphanumeric(v, "name")

    @validator("description")
    def validate_description(cls, v):
        # Check for XSS patterns
        if re.search(InputValidator.BLOCKED_PATTERNS["xss"], v, re.IGNORECASE):
            raise ValueError("description contains blocked content")
        return v
```

### 6.2 API Security Headers

```yaml
# Security Headers Configuration
security_headers:
  strict_transport_security: "max-age=31536000; includeSubDomains; preload"
  content_security_policy: "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
  x_content_type_options: "nosniff"
  x_frame_options: "DENY"
  x_xss_protection: "1; mode=block"
  referrer_policy: "strict-origin-when-cross-origin"
  permissions_policy: "geolocation=(), microphone=(), camera=()"
  cache_control: "no-store, no-cache, must-revalidate"
```

---

## 7. Audit & Compliance

### 7.1 Audit Logging

```yaml
# Audit Logging Configuration
audit_logging:
  enabled: true
  retention_days: 2555  # 7 years for compliance

  events:
    authentication:
      - "login_success"
      - "login_failure"
      - "logout"
      - "token_refresh"
      - "mfa_challenge"

    authorization:
      - "permission_granted"
      - "permission_denied"
      - "role_assigned"
      - "role_revoked"

    data_access:
      - "agent_created"
      - "agent_updated"
      - "agent_deleted"
      - "agent_executed"
      - "data_exported"
      - "data_deleted"

    admin:
      - "tenant_created"
      - "tenant_suspended"
      - "user_invited"
      - "settings_changed"

  log_format:
    timestamp: "ISO8601"
    fields:
      - "event_type"
      - "actor_id"
      - "actor_type"
      - "tenant_id"
      - "resource_type"
      - "resource_id"
      - "action"
      - "result"
      - "ip_address"
      - "user_agent"
      - "request_id"
      - "details"

  storage:
    primary: "postgresql"  # Structured queries
    archive: "s3"  # Long-term retention
    index: "elasticsearch"  # Fast search
```

### 7.2 Compliance Frameworks

```yaml
# Compliance Configuration
compliance:
  frameworks:
    soc2:
      enabled: true
      controls:
        - "CC1: Control Environment"
        - "CC2: Communication and Information"
        - "CC3: Risk Assessment"
        - "CC4: Monitoring Activities"
        - "CC5: Control Activities"
        - "CC6: Logical and Physical Access"
        - "CC7: System Operations"
        - "CC8: Change Management"
        - "CC9: Risk Mitigation"

    iso27001:
      enabled: true
      controls:
        - "A.5: Information Security Policies"
        - "A.6: Organization of Information Security"
        - "A.7: Human Resource Security"
        - "A.8: Asset Management"
        - "A.9: Access Control"
        - "A.10: Cryptography"
        - "A.12: Operations Security"
        - "A.13: Communications Security"

    gdpr:
      enabled: true
      requirements:
        - "Right to Access (Art. 15)"
        - "Right to Rectification (Art. 16)"
        - "Right to Erasure (Art. 17)"
        - "Right to Data Portability (Art. 20)"
        - "Data Breach Notification (Art. 33)"
        - "Privacy by Design (Art. 25)"

  automated_checks:
    schedule: "0 6 * * *"  # Daily at 6 AM
    checks:
      - "encryption_at_rest"
      - "encryption_in_transit"
      - "access_logging"
      - "password_policy"
      - "mfa_enforcement"
      - "key_rotation"
      - "backup_verification"
```

### 7.3 Security Monitoring

```yaml
# Security Monitoring Configuration
security_monitoring:
  siem_integration:
    enabled: true
    provider: "datadog"  # or splunk, elastic
    event_types:
      - "authentication"
      - "authorization"
      - "data_access"
      - "api_errors"
      - "rate_limit_exceeded"

  anomaly_detection:
    enabled: true
    rules:
      - name: "unusual_login_location"
        condition: "login from new country"
        action: "alert + mfa_required"

      - name: "brute_force_attempt"
        condition: "5+ failed logins in 5 minutes"
        action: "block_ip + alert"

      - name: "data_exfiltration"
        condition: "unusual data export volume"
        action: "alert + rate_limit"

      - name: "privilege_escalation"
        condition: "role change to admin"
        action: "alert + require_approval"

  alerting:
    channels:
      - type: "slack"
        webhook: "${SECURITY_SLACK_WEBHOOK}"
        severity: ["critical", "high"]

      - type: "pagerduty"
        integration_key: "${PAGERDUTY_KEY}"
        severity: ["critical"]

      - type: "email"
        recipients: ["security@greenlang.ai"]
        severity: ["critical", "high", "medium"]
```

---

## 8. Security Testing & Validation

### 8.1 Security Testing Schedule

```yaml
# Security Testing Schedule
security_testing:
  automated:
    sast:
      tool: "semgrep"
      schedule: "on_commit"
      fail_on: "high"

    dast:
      tool: "zap"
      schedule: "weekly"
      target: "staging"

    dependency_scan:
      tool: "snyk"
      schedule: "daily"
      fail_on: "high"

    container_scan:
      tool: "trivy"
      schedule: "on_build"
      fail_on: "critical"

    secrets_scan:
      tool: "gitleaks"
      schedule: "on_commit"
      fail_on: "any"

  manual:
    penetration_test:
      frequency: "quarterly"
      provider: "external"
      scope: "full_platform"

    security_audit:
      frequency: "annually"
      provider: "Big4"
      scope: "SOC2 + ISO27001"

    red_team:
      frequency: "annually"
      scope: "targeted"
```

### 8.2 Security Checklist

```markdown
# Pre-Deployment Security Checklist

## Authentication & Authorization
- [ ] JWT tokens properly validated
- [ ] Token expiration enforced
- [ ] RBAC permissions configured
- [ ] API keys hashed (SHA-256)
- [ ] MFA enabled for admin accounts

## Data Security
- [ ] Encryption at rest enabled
- [ ] TLS 1.3 for all traffic
- [ ] Tenant isolation verified
- [ ] PII fields encrypted
- [ ] Backup encryption configured

## Agent Sandbox
- [ ] gVisor/Kata runtime enabled
- [ ] Resource limits configured
- [ ] Capabilities dropped
- [ ] Read-only filesystem
- [ ] Network policies applied

## Monitoring
- [ ] Audit logging enabled
- [ ] Security alerts configured
- [ ] Anomaly detection active
- [ ] Log retention configured
- [ ] SIEM integration verified

## Compliance
- [ ] SOC2 controls implemented
- [ ] GDPR requirements met
- [ ] Data classification applied
- [ ] Retention policies configured
- [ ] Right to erasure functional
```

---

## 9. Incident Response

### 9.1 Incident Response Plan

```yaml
# Incident Response Configuration
incident_response:
  severity_levels:
    critical:
      description: "Active breach, data exfiltration"
      response_time: "15 minutes"
      escalation: "immediate"
      team: ["security_lead", "cto", "legal"]

    high:
      description: "Vulnerability actively exploited"
      response_time: "1 hour"
      escalation: "within 30 minutes"
      team: ["security_team", "engineering_lead"]

    medium:
      description: "Potential vulnerability, no exploitation"
      response_time: "4 hours"
      escalation: "within 2 hours"
      team: ["security_team"]

    low:
      description: "Security improvement needed"
      response_time: "24 hours"
      escalation: "as needed"
      team: ["security_team"]

  playbooks:
    data_breach:
      - "Isolate affected systems"
      - "Preserve evidence"
      - "Assess scope of breach"
      - "Notify affected parties (72 hours GDPR)"
      - "Implement remediation"
      - "Post-incident review"

    credential_compromise:
      - "Revoke compromised credentials"
      - "Force password reset"
      - "Review access logs"
      - "Enable MFA"
      - "Notify affected users"

    ddos_attack:
      - "Enable DDoS protection"
      - "Increase rate limits"
      - "Scale infrastructure"
      - "Coordinate with provider"
      - "Monitor and adapt"
```

---

## Related Documents

| Document | Location | Description |
|----------|----------|-------------|
| Architecture Overview | `../system-design/00-ARCHITECTURE_OVERVIEW.md` | High-level system view |
| Layer Architecture | `../system-design/01-LAYER_ARCHITECTURE.md` | Layer specifications |
| Data Flow Patterns | `../data-flows/00-DATA_FLOW_PATTERNS.md` | Data flow documentation |
| Infrastructure Requirements | `../infrastructure/00-INFRASTRUCTURE_REQUIREMENTS.md` | Compute/storage/network |
| Input Validation Guide | Existing codebase | Security input validation |
| Tenant Management | Existing codebase | Multi-tenancy implementation |

---

**Document Owner:** GL-AppArchitect
**Last Updated:** December 3, 2025
**Review Cycle:** Quarterly
**Security Classification:** CONFIDENTIAL
