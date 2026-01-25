# Enterprise Architecture Phase 4 Implementation Plan
## GreenLang Process Heat Agents - Remaining 18 Tasks

**Document Version:** 1.0.0
**Created:** December 5, 2025
**Status:** Phase 4 at 64% (32/50 tasks complete)
**Target:** Production-Ready Enterprise Deployment

---

## Executive Summary

This document provides a comprehensive implementation plan for completing Phase 4 of the GreenLang Process Heat Agents engineering initiative. The analysis covers 18 remaining tasks across four areas:

| Category | Tasks | Priority | Estimated Effort |
|----------|-------|----------|------------------|
| Event-Driven Architecture | 6 tasks (124-130) | HIGH | 3 weeks |
| API Design | 4 tasks (133-136) | MEDIUM | 2 weeks |
| Scalability & Resilience | 1 task (149) | HIGH | 1 week |
| Security Architecture | 7 tasks (151-160) | CRITICAL | 4 weeks |

**Total Estimated Timeline:** 8-10 weeks for complete implementation

---

## Gap Analysis Summary

### Existing Infrastructure (Completed)

Based on codebase analysis, the following is already implemented:

**Protocol Layer (100% Complete):**
- OPC-UA server/client: `greenlang/infrastructure/protocols/opcua_*.py`
- MQTT integration: `greenlang/infrastructure/protocols/mqtt_client.py`
- Kafka producer/consumer: `greenlang/infrastructure/protocols/kafka_*.py`
- Modbus TCP/RTU gateway: `greenlang/infrastructure/protocols/modbus_gateway.py`

**Event Framework (70% Complete):**
- Event schemas (Avro): `greenlang/infrastructure/events/event_schema.py`
- Event producer/consumer: `greenlang/infrastructure/events/event_*.py`
- Event sourcing: `greenlang/infrastructure/events/event_sourcing.py`
- Dead Letter Queue: `greenlang/infrastructure/events/dead_letter_queue.py` (EXISTS but marked incomplete)
- Saga Orchestrator: `greenlang/infrastructure/events/saga_orchestrator.py` (EXISTS but marked incomplete)

**API Layer (75% Complete):**
- OpenAPI 3.0 specs: `greenlang/infrastructure/api/openapi_generator.py`
- REST router with versioning: `greenlang/infrastructure/api/rest_router.py`
- Rate limiting: `greenlang/infrastructure/api/rate_limiter.py`
- GraphQL schema builder: `greenlang/infrastructure/api/graphql_schema.py` (EXISTS but marked incomplete)
- gRPC services: `greenlang/infrastructure/api/grpc_services.py` (EXISTS but marked incomplete)
- Webhooks: `greenlang/infrastructure/api/webhooks.py` (EXISTS but marked incomplete)
- SSE streaming: `greenlang/infrastructure/api/sse_stream.py` (EXISTS but marked incomplete)

**Resilience Patterns (100% Complete):**
- Circuit breaker: `greenlang/infrastructure/resilience/circuit_breaker.py`
- Retry policies: `greenlang/infrastructure/resilience/retry_policy.py`
- Bulkhead: `greenlang/infrastructure/resilience/bulkhead.py`
- Health checks: `greenlang/infrastructure/resilience/health_check.py`

**Kubernetes Infrastructure (80% Complete):**
- Helm charts: `helm/greenlang-agents/` (values, deployments, services, HPA)
- Network policies: `infrastructure/k8s/production/network-policies.yaml`
- Pod security: `infrastructure/k8s/production/pod-security-standards.yaml`
- ArgoCD applications: `infrastructure/argocd/applications/`

**Terraform (90% Complete):**
- EKS cluster: `terraform/modules/eks/main.tf`
- VPC: `terraform/modules/vpc/`
- RDS: `terraform/modules/rds/`
- ElastiCache: `terraform/modules/elasticache/`
- S3: `terraform/modules/s3/`
- IAM: `terraform/modules/iam/`

**CI/CD (85% Complete):**
- Production deployment: `.github/workflows/deploy-production.yml`
- Blue-green/canary deployment strategies implemented
- Rollback mechanisms in place

---

## Priority-Ranked Implementation Plan

### Priority 1: CRITICAL - Security Architecture (Tasks 151-156, 160)

Security is the foundation for enterprise deployment. These tasks must be completed first.

#### TASK-151: Configure Istio mTLS (STRICT mode)
**Effort:** 3 days
**Dependencies:** Kubernetes cluster, Istio installed

**Implementation:**

```yaml
# File: infrastructure/k8s/istio/peer-authentication.yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: greenlang-mtls-strict
  namespace: greenlang-agents
spec:
  mtls:
    mode: STRICT

---
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default-mtls-strict
  namespace: istio-system
spec:
  mtls:
    mode: STRICT

---
# Destination Rule for mTLS
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: greenlang-mtls
  namespace: greenlang-agents
spec:
  host: "*.greenlang-agents.svc.cluster.local"
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
```

**Helm Values Update:**
```yaml
# helm/greenlang-agents/values.yaml - Add to existing
istio:
  enabled: true
  mtls:
    mode: STRICT
  sidecar:
    inject: true
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 200m
        memory: 256Mi
```

#### TASK-152: Implement OAuth2/OIDC (Keycloak)
**Effort:** 5 days
**Dependencies:** Keycloak deployment, PostgreSQL

**Terraform Module:**
```hcl
# File: terraform/modules/keycloak/main.tf
resource "helm_release" "keycloak" {
  name       = "keycloak"
  repository = "https://codecentric.github.io/helm-charts"
  chart      = "keycloak"
  version    = "18.4.4"
  namespace  = "keycloak"

  values = [
    yamlencode({
      replicas = 2

      postgresql = {
        enabled = true
        persistence = {
          enabled = true
          size    = "10Gi"
        }
      }

      extraEnv = [
        {
          name  = "KEYCLOAK_ADMIN"
          value = "admin"
        },
        {
          name = "KEYCLOAK_ADMIN_PASSWORD"
          valueFrom = {
            secretKeyRef = {
              name = "keycloak-admin-secret"
              key  = "password"
            }
          }
        },
        {
          name  = "KC_PROXY"
          value = "edge"
        }
      ]

      ingress = {
        enabled = true
        annotations = {
          "kubernetes.io/ingress.class"                = "nginx"
          "cert-manager.io/cluster-issuer"             = "letsencrypt-prod"
          "nginx.ingress.kubernetes.io/ssl-redirect"   = "true"
        }
        rules = [{
          host  = "auth.greenlang.io"
          paths = [{
            path     = "/"
            pathType = "Prefix"
          }]
        }]
        tls = [{
          secretName = "keycloak-tls"
          hosts      = ["auth.greenlang.io"]
        }]
      }
    })
  ]
}

# Keycloak Realm Configuration
resource "keycloak_realm" "greenlang" {
  realm   = "greenlang"
  enabled = true

  login_theme = "keycloak"

  access_token_lifespan = "5m"

  security_defenses {
    brute_force_detection {
      permanent_lockout                = false
      max_login_failures               = 5
      wait_increment_seconds           = 60
      max_failure_wait_seconds         = 900
      failure_reset_time_seconds       = 43200
    }
  }
}

# Client for GreenLang API
resource "keycloak_openid_client" "greenlang_api" {
  realm_id                     = keycloak_realm.greenlang.id
  client_id                    = "greenlang-api"
  name                         = "GreenLang API"
  enabled                      = true
  access_type                  = "CONFIDENTIAL"
  standard_flow_enabled        = true
  direct_access_grants_enabled = true
  service_accounts_enabled     = true

  valid_redirect_uris = [
    "https://api.greenlang.io/*",
    "https://app.greenlang.io/*"
  ]
}
```

**Python Integration:**
```python
# File: greenlang/infrastructure/security/oauth2_client.py
"""
OAuth2/OIDC Integration for GreenLang

Provides Keycloak integration for authentication and authorization.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from jose import JWTError, jwt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class OAuth2Config:
    """OAuth2/OIDC configuration."""
    issuer_url: str
    client_id: str
    client_secret: str
    audience: str = "greenlang-api"
    algorithms: List[str] = None

    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ["RS256"]


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str = Field(..., description="Subject (user ID)")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    iss: str = Field(..., description="Issuer")
    aud: str = Field(..., description="Audience")
    realm_access: Optional[Dict[str, List[str]]] = None
    resource_access: Optional[Dict[str, Any]] = None
    scope: Optional[str] = None
    email: Optional[str] = None
    preferred_username: Optional[str] = None


class OAuth2Client:
    """
    OAuth2/OIDC client for Keycloak.

    Handles token validation, introspection, and user info retrieval.
    """

    def __init__(self, config: OAuth2Config):
        self.config = config
        self._jwks: Optional[Dict] = None
        self._jwks_uri: Optional[str] = None
        self._http_client = httpx.AsyncClient()

    async def initialize(self) -> None:
        """Initialize by fetching OIDC configuration."""
        well_known_url = f"{self.config.issuer_url}/.well-known/openid-configuration"

        response = await self._http_client.get(well_known_url)
        response.raise_for_status()

        oidc_config = response.json()
        self._jwks_uri = oidc_config["jwks_uri"]

        # Fetch JWKS
        jwks_response = await self._http_client.get(self._jwks_uri)
        jwks_response.raise_for_status()
        self._jwks = jwks_response.json()

        logger.info(f"OAuth2Client initialized with issuer: {self.config.issuer_url}")

    async def validate_token(self, token: str) -> TokenPayload:
        """
        Validate a JWT token.

        Args:
            token: JWT access token

        Returns:
            Decoded token payload

        Raises:
            JWTError: If token is invalid
        """
        if not self._jwks:
            await self.initialize()

        try:
            payload = jwt.decode(
                token,
                self._jwks,
                algorithms=self.config.algorithms,
                audience=self.config.audience,
                issuer=self.config.issuer_url,
            )
            return TokenPayload(**payload)

        except JWTError as e:
            logger.warning(f"Token validation failed: {e}")
            raise

    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user info from token."""
        userinfo_url = f"{self.config.issuer_url}/protocol/openid-connect/userinfo"

        response = await self._http_client.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        return response.json()

    async def introspect_token(self, token: str) -> Dict[str, Any]:
        """Introspect a token."""
        introspect_url = f"{self.config.issuer_url}/protocol/openid-connect/token/introspect"

        response = await self._http_client.post(
            introspect_url,
            data={
                "token": token,
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
            }
        )
        response.raise_for_status()
        return response.json()

    def get_user_roles(self, payload: TokenPayload) -> List[str]:
        """Extract user roles from token payload."""
        roles = []

        # Realm roles
        if payload.realm_access and "roles" in payload.realm_access:
            roles.extend(payload.realm_access["roles"])

        # Client roles
        if payload.resource_access:
            client_access = payload.resource_access.get(self.config.client_id, {})
            if "roles" in client_access:
                roles.extend(client_access["roles"])

        return list(set(roles))

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http_client.aclose()
```

#### TASK-153: Build RBAC Policies
**Effort:** 3 days

```python
# File: greenlang/infrastructure/security/rbac.py
"""
Role-Based Access Control (RBAC) for GreenLang

Provides hierarchical role management and permission checking.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class Permission(str, Enum):
    """GreenLang permissions."""
    # Agent permissions
    AGENT_READ = "agent:read"
    AGENT_WRITE = "agent:write"
    AGENT_EXECUTE = "agent:execute"
    AGENT_ADMIN = "agent:admin"

    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # Emission permissions
    EMISSION_READ = "emission:read"
    EMISSION_WRITE = "emission:write"
    EMISSION_SUBMIT = "emission:submit"
    EMISSION_APPROVE = "emission:approve"

    # Report permissions
    REPORT_READ = "report:read"
    REPORT_GENERATE = "report:generate"
    REPORT_SUBMIT = "report:submit"

    # Admin permissions
    USER_MANAGE = "user:manage"
    ROLE_MANAGE = "role:manage"
    TENANT_MANAGE = "tenant:manage"
    SYSTEM_ADMIN = "system:admin"


class Role(str, Enum):
    """GreenLang roles."""
    VIEWER = "viewer"
    OPERATOR = "operator"
    ANALYST = "analyst"
    MANAGER = "manager"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


# Role hierarchy - higher roles inherit lower role permissions
ROLE_HIERARCHY: Dict[Role, List[Role]] = {
    Role.SUPER_ADMIN: [Role.ADMIN, Role.MANAGER, Role.ANALYST, Role.OPERATOR, Role.VIEWER],
    Role.ADMIN: [Role.MANAGER, Role.ANALYST, Role.OPERATOR, Role.VIEWER],
    Role.MANAGER: [Role.ANALYST, Role.OPERATOR, Role.VIEWER],
    Role.ANALYST: [Role.OPERATOR, Role.VIEWER],
    Role.OPERATOR: [Role.VIEWER],
    Role.VIEWER: [],
}

# Role to permission mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.AGENT_READ,
        Permission.DATA_READ,
        Permission.EMISSION_READ,
        Permission.REPORT_READ,
    },
    Role.OPERATOR: {
        Permission.AGENT_EXECUTE,
        Permission.DATA_WRITE,
        Permission.EMISSION_WRITE,
    },
    Role.ANALYST: {
        Permission.DATA_EXPORT,
        Permission.REPORT_GENERATE,
        Permission.EMISSION_SUBMIT,
    },
    Role.MANAGER: {
        Permission.AGENT_WRITE,
        Permission.EMISSION_APPROVE,
        Permission.REPORT_SUBMIT,
    },
    Role.ADMIN: {
        Permission.AGENT_ADMIN,
        Permission.DATA_DELETE,
        Permission.USER_MANAGE,
        Permission.ROLE_MANAGE,
    },
    Role.SUPER_ADMIN: {
        Permission.TENANT_MANAGE,
        Permission.SYSTEM_ADMIN,
    },
}


class RBACPolicy(BaseModel):
    """RBAC policy definition."""
    policy_id: str = Field(..., description="Policy identifier")
    name: str = Field(..., description="Policy name")
    description: Optional[str] = None
    roles: List[Role] = Field(default_factory=list)
    permissions: List[Permission] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list, description="Resource patterns")
    conditions: Dict[str, str] = Field(default_factory=dict)


class RBACEngine:
    """
    RBAC engine for permission checking.

    Supports role hierarchy and resource-based access control.
    """

    def __init__(self):
        self._policies: Dict[str, RBACPolicy] = {}

    def get_effective_permissions(self, roles: List[Role]) -> Set[Permission]:
        """
        Get all effective permissions for a set of roles.

        Includes inherited permissions from role hierarchy.
        """
        permissions: Set[Permission] = set()

        for role in roles:
            # Direct permissions
            permissions.update(ROLE_PERMISSIONS.get(role, set()))

            # Inherited permissions
            for inherited_role in ROLE_HIERARCHY.get(role, []):
                permissions.update(ROLE_PERMISSIONS.get(inherited_role, set()))

        return permissions

    def has_permission(
        self,
        user_roles: List[Role],
        required_permission: Permission
    ) -> bool:
        """Check if user has required permission."""
        effective_permissions = self.get_effective_permissions(user_roles)
        return required_permission in effective_permissions

    def has_any_permission(
        self,
        user_roles: List[Role],
        required_permissions: List[Permission]
    ) -> bool:
        """Check if user has any of the required permissions."""
        effective_permissions = self.get_effective_permissions(user_roles)
        return bool(effective_permissions.intersection(set(required_permissions)))

    def has_all_permissions(
        self,
        user_roles: List[Role],
        required_permissions: List[Permission]
    ) -> bool:
        """Check if user has all required permissions."""
        effective_permissions = self.get_effective_permissions(user_roles)
        return set(required_permissions).issubset(effective_permissions)

    def add_policy(self, policy: RBACPolicy) -> None:
        """Add a custom RBAC policy."""
        self._policies[policy.policy_id] = policy

    def check_resource_access(
        self,
        user_roles: List[Role],
        resource: str,
        action: Permission
    ) -> bool:
        """
        Check access to a specific resource.

        Evaluates custom policies in addition to role permissions.
        """
        # Check base permission
        if not self.has_permission(user_roles, action):
            return False

        # Check custom policies
        for policy in self._policies.values():
            if self._matches_policy(user_roles, resource, action, policy):
                return True

        return True  # Default allow if base permission exists

    def _matches_policy(
        self,
        user_roles: List[Role],
        resource: str,
        action: Permission,
        policy: RBACPolicy
    ) -> bool:
        """Check if access matches a policy."""
        # Check role match
        if policy.roles and not any(r in policy.roles for r in user_roles):
            return False

        # Check permission match
        if policy.permissions and action not in policy.permissions:
            return False

        # Check resource pattern match
        if policy.resources:
            import fnmatch
            if not any(fnmatch.fnmatch(resource, pattern) for pattern in policy.resources):
                return False

        return True
```

#### TASK-154: Create ABAC for Contextual Auth
**Effort:** 4 days

```python
# File: greenlang/infrastructure/security/abac.py
"""
Attribute-Based Access Control (ABAC) for GreenLang

Provides fine-grained, contextual authorization based on attributes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AttributeType(str, Enum):
    """Attribute types for ABAC."""
    SUBJECT = "subject"      # User/service attributes
    RESOURCE = "resource"    # Resource being accessed
    ACTION = "action"        # Action being performed
    ENVIRONMENT = "environment"  # Contextual attributes


class Operator(str, Enum):
    """Comparison operators."""
    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    CONTAINS = "contains"
    IN = "in"
    NOT_IN = "not_in"
    MATCHES = "matches"  # Regex match
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


@dataclass
class Attribute:
    """Attribute definition."""
    attribute_type: AttributeType
    key: str
    value: Any


class Condition(BaseModel):
    """ABAC condition."""
    attribute_type: AttributeType
    key: str
    operator: Operator
    value: Any

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        # Get attribute value from context
        attributes = context.get(self.attribute_type.value, {})
        actual_value = attributes.get(self.key)

        if actual_value is None:
            return False

        # Evaluate based on operator
        if self.operator == Operator.EQUALS:
            return actual_value == self.value
        elif self.operator == Operator.NOT_EQUALS:
            return actual_value != self.value
        elif self.operator == Operator.GREATER_THAN:
            return actual_value > self.value
        elif self.operator == Operator.LESS_THAN:
            return actual_value < self.value
        elif self.operator == Operator.CONTAINS:
            return self.value in actual_value
        elif self.operator == Operator.IN:
            return actual_value in self.value
        elif self.operator == Operator.NOT_IN:
            return actual_value not in self.value
        elif self.operator == Operator.MATCHES:
            import re
            return bool(re.match(self.value, str(actual_value)))
        elif self.operator == Operator.STARTS_WITH:
            return str(actual_value).startswith(self.value)
        elif self.operator == Operator.ENDS_WITH:
            return str(actual_value).endswith(self.value)

        return False


class Effect(str, Enum):
    """Policy effect."""
    ALLOW = "allow"
    DENY = "deny"


class ABACPolicy(BaseModel):
    """ABAC policy definition."""
    policy_id: str = Field(..., description="Policy identifier")
    name: str = Field(..., description="Policy name")
    description: Optional[str] = None
    effect: Effect = Field(default=Effect.ALLOW)
    conditions: List[Condition] = Field(default_factory=list)
    priority: int = Field(default=0, description="Higher priority evaluated first")
    enabled: bool = Field(default=True)

    def evaluate(self, context: Dict[str, Any]) -> Optional[bool]:
        """
        Evaluate policy against context.

        Returns:
            True if all conditions match, False if any fail, None if not applicable
        """
        if not self.enabled:
            return None

        for condition in self.conditions:
            if not condition.evaluate(context):
                return None  # Condition not met, policy not applicable

        # All conditions met
        return self.effect == Effect.ALLOW


class ABACEngine:
    """
    ABAC engine for contextual authorization.

    Evaluates policies based on subject, resource, action, and environment attributes.
    """

    def __init__(self):
        self._policies: List[ABACPolicy] = []
        self._context_enrichers: List[Callable] = []

    def add_policy(self, policy: ABACPolicy) -> None:
        """Add a policy to the engine."""
        self._policies.append(policy)
        # Sort by priority (descending)
        self._policies.sort(key=lambda p: p.priority, reverse=True)
        logger.debug(f"Added ABAC policy: {policy.name}")

    def add_context_enricher(self, enricher: Callable[[Dict], Dict]) -> None:
        """Add a context enricher function."""
        self._context_enrichers.append(enricher)

    def evaluate(
        self,
        subject: Dict[str, Any],
        resource: Dict[str, Any],
        action: str,
        environment: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Evaluate access request.

        Args:
            subject: Subject attributes (user, service)
            resource: Resource attributes (type, id, owner)
            action: Action being performed
            environment: Environmental attributes (time, location, ip)

        Returns:
            True if access is allowed
        """
        # Build context
        context = {
            AttributeType.SUBJECT.value: subject,
            AttributeType.RESOURCE.value: resource,
            AttributeType.ACTION.value: {"name": action},
            AttributeType.ENVIRONMENT.value: environment or {},
        }

        # Add default environment attributes
        context[AttributeType.ENVIRONMENT.value].update({
            "current_time": datetime.utcnow().isoformat(),
            "current_hour": datetime.utcnow().hour,
            "current_day": datetime.utcnow().strftime("%A"),
        })

        # Apply context enrichers
        for enricher in self._context_enrichers:
            context = enricher(context)

        # Evaluate policies
        for policy in self._policies:
            result = policy.evaluate(context)
            if result is not None:
                logger.debug(
                    f"ABAC policy '{policy.name}' returned {result} "
                    f"for action '{action}' on resource {resource}"
                )
                return result

        # Default deny
        logger.warning(f"No matching ABAC policy for action '{action}'")
        return False

    def create_context(
        self,
        user_id: str,
        user_roles: List[str],
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        resource_owner: Optional[str] = None,
        action: str = "read",
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Convenience method to evaluate common access patterns.
        """
        subject = {
            "user_id": user_id,
            "roles": user_roles,
            "tenant_id": tenant_id,
        }

        resource = {
            "type": resource_type,
            "id": resource_id,
            "owner": resource_owner,
            "tenant_id": tenant_id,
        }

        environment = {
            "ip_address": ip_address,
        }

        return self.evaluate(subject, resource, action, environment)


# Pre-defined enterprise policies
ENTERPRISE_POLICIES = [
    # Tenant isolation - users can only access their tenant's resources
    ABACPolicy(
        policy_id="tenant-isolation",
        name="Tenant Isolation",
        description="Ensure users can only access resources in their tenant",
        effect=Effect.DENY,
        priority=100,
        conditions=[
            Condition(
                attribute_type=AttributeType.SUBJECT,
                key="tenant_id",
                operator=Operator.NOT_EQUALS,
                value="${resource.tenant_id}"  # Dynamic value
            )
        ]
    ),

    # Business hours restriction for sensitive operations
    ABACPolicy(
        policy_id="business-hours-sensitive",
        name="Business Hours for Sensitive Operations",
        description="Restrict sensitive operations to business hours",
        effect=Effect.DENY,
        priority=90,
        conditions=[
            Condition(
                attribute_type=AttributeType.ACTION,
                key="name",
                operator=Operator.IN,
                value=["emission:approve", "report:submit", "data:delete"]
            ),
            Condition(
                attribute_type=AttributeType.ENVIRONMENT,
                key="current_hour",
                operator=Operator.NOT_IN,
                value=list(range(9, 18))  # 9 AM to 6 PM
            )
        ]
    ),

    # Owner-based access
    ABACPolicy(
        policy_id="owner-full-access",
        name="Owner Full Access",
        description="Resource owners have full access",
        effect=Effect.ALLOW,
        priority=80,
        conditions=[
            Condition(
                attribute_type=AttributeType.SUBJECT,
                key="user_id",
                operator=Operator.EQUALS,
                value="${resource.owner}"
            )
        ]
    ),
]
```

#### TASK-155: Implement API Key Management (Vault)
**Effort:** 4 days

```hcl
# File: terraform/modules/vault/main.tf
resource "helm_release" "vault" {
  name       = "vault"
  repository = "https://helm.releases.hashicorp.com"
  chart      = "vault"
  version    = "0.27.0"
  namespace  = "vault"
  create_namespace = true

  values = [
    yamlencode({
      global = {
        enabled = true
      }

      server = {
        ha = {
          enabled  = true
          replicas = 3
          raft = {
            enabled = true
            config  = <<-EOF
              ui = true

              listener "tcp" {
                tls_disable = 1
                address     = "[::]:8200"
                cluster_address = "[::]:8201"
              }

              storage "raft" {
                path = "/vault/data"
              }

              service_registration "kubernetes" {}
            EOF
          }
        }

        ingress = {
          enabled = true
          annotations = {
            "kubernetes.io/ingress.class"              = "nginx"
            "cert-manager.io/cluster-issuer"           = "letsencrypt-prod"
            "nginx.ingress.kubernetes.io/ssl-redirect" = "true"
          }
          hosts = [{
            host  = "vault.greenlang.io"
            paths = ["/"]
          }]
          tls = [{
            secretName = "vault-tls"
            hosts      = ["vault.greenlang.io"]
          }]
        }

        resources = {
          requests = {
            memory = "256Mi"
            cpu    = "250m"
          }
          limits = {
            memory = "512Mi"
            cpu    = "500m"
          }
        }
      }

      injector = {
        enabled = true
        resources = {
          requests = {
            memory = "64Mi"
            cpu    = "50m"
          }
          limits = {
            memory = "128Mi"
            cpu    = "100m"
          }
        }
      }
    })
  ]
}

# Vault Kubernetes auth
resource "vault_auth_backend" "kubernetes" {
  type = "kubernetes"
  path = "kubernetes"
}

resource "vault_kubernetes_auth_backend_config" "config" {
  backend                = vault_auth_backend.kubernetes.path
  kubernetes_host        = data.aws_eks_cluster.cluster.endpoint
  kubernetes_ca_cert     = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  issuer                 = "https://kubernetes.default.svc.cluster.local"
}

# Secrets engine for API keys
resource "vault_mount" "api_keys" {
  path        = "greenlang/api-keys"
  type        = "kv"
  options     = { version = "2" }
  description = "GreenLang API Keys"
}

# Policy for GreenLang services
resource "vault_policy" "greenlang_read" {
  name = "greenlang-read"

  policy = <<-EOT
    path "greenlang/api-keys/data/*" {
      capabilities = ["read"]
    }

    path "greenlang/api-keys/metadata/*" {
      capabilities = ["read", "list"]
    }

    path "greenlang/database/*" {
      capabilities = ["read"]
    }
  EOT
}

# Kubernetes role for GreenLang agents
resource "vault_kubernetes_auth_backend_role" "greenlang_agents" {
  backend                          = vault_auth_backend.kubernetes.path
  role_name                        = "greenlang-agents"
  bound_service_account_names      = ["greenlang-agents-sa"]
  bound_service_account_namespaces = ["greenlang-agents"]
  token_ttl                        = 3600
  token_policies                   = ["greenlang-read"]
}
```

```python
# File: greenlang/infrastructure/security/vault_client.py
"""
HashiCorp Vault Client for GreenLang

Provides secret management and API key storage/retrieval.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import hvac
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class VaultConfig:
    """Vault configuration."""
    url: str = "http://vault:8200"
    auth_method: str = "kubernetes"  # kubernetes, token, approle
    role: str = "greenlang-agents"
    token: Optional[str] = None
    namespace: Optional[str] = None
    mount_point: str = "greenlang/api-keys"


class APIKey(BaseModel):
    """API Key model."""
    key_id: str = Field(..., description="Key identifier")
    key_hash: str = Field(..., description="Hashed key value")
    name: str = Field(..., description="Key name")
    description: Optional[str] = None
    tenant_id: str = Field(..., description="Tenant ID")
    scopes: List[str] = Field(default_factory=list)
    rate_limit: int = Field(default=1000, description="Requests per minute")
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    enabled: bool = Field(default=True)


class VaultClient:
    """
    Vault client for secret management.

    Supports Kubernetes auth for service accounts and
    provides API key management operations.
    """

    def __init__(self, config: VaultConfig):
        self.config = config
        self._client: Optional[hvac.Client] = None
        self._token_expires_at: Optional[datetime] = None

    async def initialize(self) -> None:
        """Initialize Vault client and authenticate."""
        self._client = hvac.Client(
            url=self.config.url,
            namespace=self.config.namespace
        )

        if self.config.auth_method == "kubernetes":
            await self._auth_kubernetes()
        elif self.config.auth_method == "token":
            self._client.token = self.config.token
        elif self.config.auth_method == "approle":
            await self._auth_approle()

        logger.info("Vault client initialized")

    async def _auth_kubernetes(self) -> None:
        """Authenticate using Kubernetes service account."""
        # Read service account token
        with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as f:
            jwt = f.read()

        response = self._client.auth.kubernetes.login(
            role=self.config.role,
            jwt=jwt
        )

        self._token_expires_at = datetime.utcnow() + timedelta(
            seconds=response["auth"]["lease_duration"]
        )

        logger.info(f"Authenticated with Kubernetes role: {self.config.role}")

    async def _auth_approle(self) -> None:
        """Authenticate using AppRole."""
        # Implementation for AppRole auth
        pass

    async def _ensure_authenticated(self) -> None:
        """Ensure token is valid, refresh if needed."""
        if self._token_expires_at and datetime.utcnow() >= self._token_expires_at:
            await self._auth_kubernetes()

    async def create_api_key(
        self,
        name: str,
        tenant_id: str,
        scopes: List[str],
        description: Optional[str] = None,
        rate_limit: int = 1000,
        expires_in_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Returns:
            Tuple of (raw_key, api_key_metadata)
        """
        await self._ensure_authenticated()

        import secrets
        import hashlib

        # Generate secure key
        raw_key = f"gl_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = secrets.token_urlsafe(16)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            description=description,
            tenant_id=tenant_id,
            scopes=scopes,
            rate_limit=rate_limit,
            expires_at=expires_at,
        )

        # Store in Vault
        self._client.secrets.kv.v2.create_or_update_secret(
            path=f"keys/{tenant_id}/{key_id}",
            secret=api_key.dict(),
            mount_point=self.config.mount_point
        )

        logger.info(f"Created API key: {key_id} for tenant: {tenant_id}")

        return raw_key, api_key

    async def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Returns:
            APIKey if valid, None if invalid
        """
        await self._ensure_authenticated()

        import hashlib
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Search for key (in production, use a lookup index)
        try:
            keys = self._client.secrets.kv.v2.list_secrets(
                path="keys",
                mount_point=self.config.mount_point
            )

            for tenant_id in keys.get("data", {}).get("keys", []):
                tenant_keys = self._client.secrets.kv.v2.list_secrets(
                    path=f"keys/{tenant_id}",
                    mount_point=self.config.mount_point
                )

                for key_id in tenant_keys.get("data", {}).get("keys", []):
                    response = self._client.secrets.kv.v2.read_secret_version(
                        path=f"keys/{tenant_id}/{key_id}",
                        mount_point=self.config.mount_point
                    )

                    data = response.get("data", {}).get("data", {})
                    if data.get("key_hash") == key_hash:
                        api_key = APIKey(**data)

                        # Check if enabled and not expired
                        if not api_key.enabled:
                            return None
                        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                            return None

                        # Update last used
                        await self._update_last_used(tenant_id, key_id)

                        return api_key

            return None

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    async def _update_last_used(self, tenant_id: str, key_id: str) -> None:
        """Update last used timestamp."""
        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=f"keys/{tenant_id}/{key_id}",
                mount_point=self.config.mount_point
            )

            data = response.get("data", {}).get("data", {})
            data["last_used_at"] = datetime.utcnow().isoformat()

            self._client.secrets.kv.v2.create_or_update_secret(
                path=f"keys/{tenant_id}/{key_id}",
                secret=data,
                mount_point=self.config.mount_point
            )
        except Exception as e:
            logger.warning(f"Failed to update last_used_at: {e}")

    async def revoke_api_key(self, tenant_id: str, key_id: str) -> bool:
        """Revoke an API key."""
        await self._ensure_authenticated()

        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=f"keys/{tenant_id}/{key_id}",
                mount_point=self.config.mount_point
            )

            data = response.get("data", {}).get("data", {})
            data["enabled"] = False

            self._client.secrets.kv.v2.create_or_update_secret(
                path=f"keys/{tenant_id}/{key_id}",
                secret=data,
                mount_point=self.config.mount_point
            )

            logger.info(f"Revoked API key: {key_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False

    async def get_secret(self, path: str) -> Dict[str, Any]:
        """Get a secret from Vault."""
        await self._ensure_authenticated()

        response = self._client.secrets.kv.v2.read_secret_version(
            path=path,
            mount_point=self.config.mount_point
        )

        return response.get("data", {}).get("data", {})
```

#### TASK-156: Build Secrets Rotation
**Effort:** 3 days

```python
# File: greenlang/infrastructure/security/secrets_rotation.py
"""
Automated Secrets Rotation for GreenLang

Provides automatic rotation of database credentials, API keys, and certificates.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets."""
    DATABASE_CREDENTIAL = "database_credential"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    ENCRYPTION_KEY = "encryption_key"
    SERVICE_ACCOUNT = "service_account"


class RotationStatus(str, Enum):
    """Rotation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class RotationPolicy(BaseModel):
    """Secret rotation policy."""
    policy_id: str
    secret_type: SecretType
    secret_path: str
    rotation_interval_days: int = Field(default=90)
    notification_days_before: int = Field(default=14)
    auto_rotate: bool = Field(default=True)
    retain_versions: int = Field(default=3)
    rotation_function: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RotationEvent(BaseModel):
    """Secret rotation event."""
    event_id: str
    policy_id: str
    secret_path: str
    status: RotationStatus
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    old_version: Optional[int] = None
    new_version: Optional[int] = None


@dataclass
class SecretsRotationConfig:
    """Configuration for secrets rotation manager."""
    vault_url: str = "http://vault:8200"
    check_interval_hours: int = 6
    notification_webhook: Optional[str] = None
    slack_webhook: Optional[str] = None


class SecretsRotationManager:
    """
    Manages automatic secrets rotation.

    Supports database credentials, API keys, and certificates.
    """

    def __init__(self, config: SecretsRotationConfig):
        self.config = config
        self._policies: Dict[str, RotationPolicy] = {}
        self._rotation_functions: Dict[str, Callable] = {}
        self._running = False
        self._rotation_history: List[RotationEvent] = []

    def register_policy(self, policy: RotationPolicy) -> None:
        """Register a rotation policy."""
        self._policies[policy.policy_id] = policy
        logger.info(f"Registered rotation policy: {policy.policy_id}")

    def register_rotation_function(
        self,
        secret_type: SecretType,
        func: Callable
    ) -> None:
        """Register a rotation function for a secret type."""
        self._rotation_functions[secret_type.value] = func
        logger.info(f"Registered rotation function for: {secret_type.value}")

    async def start(self) -> None:
        """Start the rotation manager."""
        self._running = True
        logger.info("Secrets rotation manager started")

        while self._running:
            try:
                await self._check_rotation_due()
            except Exception as e:
                logger.error(f"Rotation check error: {e}")

            await asyncio.sleep(self.config.check_interval_hours * 3600)

    async def stop(self) -> None:
        """Stop the rotation manager."""
        self._running = False
        logger.info("Secrets rotation manager stopped")

    async def _check_rotation_due(self) -> None:
        """Check all policies for rotation due."""
        for policy in self._policies.values():
            if await self._is_rotation_due(policy):
                if policy.auto_rotate:
                    await self.rotate_secret(policy)
                else:
                    await self._send_rotation_notification(policy)

    async def _is_rotation_due(self, policy: RotationPolicy) -> bool:
        """Check if secret rotation is due."""
        # Get last rotation time from Vault metadata
        try:
            # This would check Vault secret metadata
            last_rotated = await self._get_last_rotation_time(policy.secret_path)

            if last_rotated is None:
                return True

            rotation_due = last_rotated + timedelta(days=policy.rotation_interval_days)
            notification_due = rotation_due - timedelta(days=policy.notification_days_before)

            now = datetime.utcnow()

            if now >= rotation_due:
                return True

            if now >= notification_due:
                await self._send_rotation_notification(policy)

            return False

        except Exception as e:
            logger.error(f"Error checking rotation due: {e}")
            return False

    async def _get_last_rotation_time(self, secret_path: str) -> Optional[datetime]:
        """Get last rotation time from Vault."""
        # Implementation would read Vault metadata
        pass

    async def rotate_secret(self, policy: RotationPolicy) -> RotationEvent:
        """Rotate a secret based on policy."""
        from uuid import uuid4

        event = RotationEvent(
            event_id=str(uuid4()),
            policy_id=policy.policy_id,
            secret_path=policy.secret_path,
            status=RotationStatus.IN_PROGRESS,
        )

        try:
            logger.info(f"Starting rotation for: {policy.secret_path}")

            # Get rotation function
            rotate_func = self._rotation_functions.get(policy.secret_type.value)
            if not rotate_func:
                raise ValueError(f"No rotation function for: {policy.secret_type}")

            # Execute rotation
            if asyncio.iscoroutinefunction(rotate_func):
                result = await rotate_func(policy)
            else:
                result = rotate_func(policy)

            event.status = RotationStatus.COMPLETED
            event.completed_at = datetime.utcnow()
            event.new_version = result.get("version")

            logger.info(f"Rotation completed for: {policy.secret_path}")

            # Send success notification
            await self._send_rotation_complete_notification(policy, event)

        except Exception as e:
            event.status = RotationStatus.FAILED
            event.error = str(e)
            event.completed_at = datetime.utcnow()

            logger.error(f"Rotation failed for {policy.secret_path}: {e}")

            # Send failure notification
            await self._send_rotation_failure_notification(policy, event, e)

        self._rotation_history.append(event)
        return event

    async def _send_rotation_notification(self, policy: RotationPolicy) -> None:
        """Send rotation due notification."""
        message = (
            f"Secret rotation due for: {policy.secret_path}\n"
            f"Type: {policy.secret_type.value}\n"
            f"Auto-rotate: {policy.auto_rotate}"
        )

        if self.config.slack_webhook:
            await self._send_slack_notification(message)

    async def _send_rotation_complete_notification(
        self,
        policy: RotationPolicy,
        event: RotationEvent
    ) -> None:
        """Send rotation complete notification."""
        message = (
            f"Secret rotation completed for: {policy.secret_path}\n"
            f"New version: {event.new_version}\n"
            f"Duration: {(event.completed_at - event.started_at).total_seconds()}s"
        )

        if self.config.slack_webhook:
            await self._send_slack_notification(message)

    async def _send_rotation_failure_notification(
        self,
        policy: RotationPolicy,
        event: RotationEvent,
        error: Exception
    ) -> None:
        """Send rotation failure notification."""
        message = (
            f"SECRET ROTATION FAILED: {policy.secret_path}\n"
            f"Error: {str(error)}\n"
            f"Manual intervention required!"
        )

        if self.config.slack_webhook:
            await self._send_slack_notification(message, is_error=True)

    async def _send_slack_notification(
        self,
        message: str,
        is_error: bool = False
    ) -> None:
        """Send Slack notification."""
        import httpx

        color = "#FF0000" if is_error else "#36A64F"

        payload = {
            "attachments": [{
                "color": color,
                "text": message,
                "footer": "GreenLang Secrets Manager",
            }]
        }

        async with httpx.AsyncClient() as client:
            await client.post(self.config.slack_webhook, json=payload)


# Database credential rotation function
async def rotate_database_credential(policy: RotationPolicy) -> Dict[str, Any]:
    """Rotate database credential."""
    import secrets

    # Generate new password
    new_password = secrets.token_urlsafe(32)

    # Update database user password
    # This would connect to the database and ALTER USER

    # Update Vault secret
    # This would update the secret in Vault

    return {"version": 1, "password_updated": True}


# API key rotation function
async def rotate_api_key(policy: RotationPolicy) -> Dict[str, Any]:
    """Rotate API key."""
    import secrets

    # Generate new key
    new_key = f"gl_{secrets.token_urlsafe(32)}"

    # Store new key, mark old as deprecated
    # Grace period allows both keys to work

    return {"version": 1, "key_rotated": True}
```

#### TASK-160: Create Vulnerability Scanning
**Effort:** 3 days

```yaml
# File: .github/workflows/security-scanning.yml
name: Security Scanning Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: greenlang/agents

jobs:
  # ============================================
  # Static Application Security Testing (SAST)
  # ============================================
  sast:
    name: SAST - Code Analysis
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      # Semgrep SAST
      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/python
            p/security-audit
            p/owasp-top-ten
            p/secrets
          generateSarif: "1"

      - name: Upload Semgrep SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: semgrep.sarif
          category: semgrep

      # Bandit Python Security
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Bandit
        run: pip install bandit[toml]

      - name: Run Bandit
        run: |
          bandit -r greenlang/ -f json -o bandit-results.json || true
          bandit -r greenlang/ -f txt

      - name: Upload Bandit results
        uses: actions/upload-artifact@v4
        with:
          name: bandit-results
          path: bandit-results.json

  # ============================================
  # Dependency Scanning
  # ============================================
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Safety Check
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Safety
        run: pip install safety

      - name: Run Safety Check
        run: |
          pip install -r requirements.txt
          safety check --full-report --json > safety-results.json || true
          safety check --full-report

      # Snyk Dependency Scan
      - name: Run Snyk
        uses: snyk/actions/python@master
        continue-on-error: true
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high --json-file-output=snyk-results.json

      - name: Upload dependency scan results
        uses: actions/upload-artifact@v4
        with:
          name: dependency-scan-results
          path: |
            safety-results.json
            snyk-results.json

  # ============================================
  # Container Image Scanning
  # ============================================
  container-scan:
    name: Container Vulnerability Scan
    runs-on: ubuntu-latest
    needs: build-image
    steps:
      - uses: actions/checkout@v4

      # Trivy Container Scan
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
          vuln-type: 'os,library'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
          category: trivy

      # Grype Container Scan
      - name: Run Grype scanner
        uses: anchore/scan-action@v3
        id: grype
        with:
          image: '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}'
          fail-build: true
          severity-cutoff: high
          output-format: sarif

      - name: Upload Grype scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: ${{ steps.grype.outputs.sarif }}
          category: grype

  # ============================================
  # Infrastructure as Code Scanning
  # ============================================
  iac-scan:
    name: IaC Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Checkov IaC Scan
      - name: Run Checkov
        uses: bridgecrewio/checkov-action@v12
        with:
          directory: .
          framework: terraform,kubernetes,helm
          output_format: sarif
          output_file_path: checkov-results.sarif
          soft_fail: true
          skip_check: CKV_K8S_40,CKV_K8S_37  # Skip specific checks if needed

      - name: Upload Checkov results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: checkov-results.sarif
          category: checkov

      # KICS IaC Scan
      - name: Run KICS
        uses: checkmarx/kics-github-action@v1.7.0
        with:
          path: 'terraform,infrastructure,helm'
          output_path: kics-results/
          output_formats: 'sarif'
          fail_on: high

      - name: Upload KICS results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: kics-results/results.sarif
          category: kics

  # ============================================
  # Secret Scanning
  # ============================================
  secret-scan:
    name: Secret Scanning
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      # TruffleHog Secret Scanner
      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.pull_request.base.sha }}
          head: ${{ github.event.pull_request.head.sha }}
          extra_args: --only-verified

      # Gitleaks Secret Scanner
      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_LICENSE: ${{ secrets.GITLEAKS_LICENSE }}

  # ============================================
  # DAST - Dynamic Testing (on staging)
  # ============================================
  dast:
    name: DAST - Dynamic Security Testing
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    needs: [sast, dependency-scan, container-scan]
    steps:
      - uses: actions/checkout@v4

      # OWASP ZAP Scan
      - name: Run OWASP ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.10.0
        with:
          target: 'https://staging-api.greenlang.io'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'

      # Nuclei Vulnerability Scanner
      - name: Run Nuclei
        uses: projectdiscovery/nuclei-action@main
        with:
          target: 'https://staging-api.greenlang.io'
          templates: 'cves,vulnerabilities,misconfiguration'
          output: nuclei-results.txt

      - name: Upload DAST results
        uses: actions/upload-artifact@v4
        with:
          name: dast-results
          path: |
            nuclei-results.txt

  # ============================================
  # Security Report Generation
  # ============================================
  security-report:
    name: Generate Security Report
    runs-on: ubuntu-latest
    needs: [sast, dependency-scan, container-scan, iac-scan, secret-scan]
    if: always()
    steps:
      - uses: actions/checkout@v4

      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: security-artifacts

      - name: Generate consolidated report
        run: |
          echo "# Security Scan Report" > security-report.md
          echo "" >> security-report.md
          echo "**Date:** $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> security-report.md
          echo "**Commit:** ${{ github.sha }}" >> security-report.md
          echo "**Branch:** ${{ github.ref_name }}" >> security-report.md
          echo "" >> security-report.md

          echo "## Scan Results" >> security-report.md
          echo "" >> security-report.md
          echo "| Scanner | Status |" >> security-report.md
          echo "|---------|--------|" >> security-report.md
          echo "| SAST (Semgrep/Bandit) | ${{ needs.sast.result }} |" >> security-report.md
          echo "| Dependency Scan | ${{ needs.dependency-scan.result }} |" >> security-report.md
          echo "| Container Scan | ${{ needs.container-scan.result }} |" >> security-report.md
          echo "| IaC Scan | ${{ needs.iac-scan.result }} |" >> security-report.md
          echo "| Secret Scan | ${{ needs.secret-scan.result }} |" >> security-report.md

      - name: Upload security report
        uses: actions/upload-artifact@v4
        with:
          name: security-report
          path: security-report.md

      - name: Notify on failures
        if: failure()
        uses: slackapi/slack-github-action@v1.25.0
        with:
          payload: |
            {
              "text": "Security Scan Failed",
              "attachments": [{
                "color": "#FF0000",
                "text": "Security vulnerabilities detected in ${{ github.repository }}\nBranch: ${{ github.ref_name }}\n<${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View Results>"
              }]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SECURITY_SLACK_WEBHOOK }}

  # ============================================
  # Build Image Job (dependency for container scan)
  # ============================================
  build-image:
    name: Build Container Image
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

### Priority 2: HIGH - Event-Driven Architecture (Tasks 124-130)

#### TASK-124: Implement Dead Letter Queue Handling (ENHANCEMENT)
**Status:** Base implementation exists, needs production enhancements
**Effort:** 2 days

The existing `dead_letter_queue.py` has the core functionality. Enhancements needed:

```python
# File: greenlang/infrastructure/events/dead_letter_queue.py
# Add Redis backend for production persistence

class RedisDLQStorage(DLQStorageBackend):
    """Redis-based DLQ storage for production."""

    def __init__(self, redis_url: str):
        import redis.asyncio as redis
        self._redis = redis.from_url(redis_url)
        self._prefix = "greenlang:dlq:"

    async def save(self, entry: DLQEntry) -> None:
        key = f"{self._prefix}{entry.entry_id}"
        await self._redis.setex(
            key,
            timedelta(days=30),  # TTL
            entry.json()
        )
        # Add to pending set
        await self._redis.zadd(
            f"{self._prefix}pending",
            {entry.entry_id: entry.created_at.timestamp()}
        )

    async def get(self, entry_id: str) -> Optional[DLQEntry]:
        key = f"{self._prefix}{entry_id}"
        data = await self._redis.get(key)
        if data:
            return DLQEntry.parse_raw(data)
        return None

    async def list_pending(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[DLQEntry]:
        entry_ids = await self._redis.zrange(
            f"{self._prefix}pending",
            offset,
            offset + limit - 1
        )
        entries = []
        for eid in entry_ids:
            entry = await self.get(eid.decode())
            if entry:
                entries.append(entry)
        return entries
```

#### TASK-125: Build Event Replay Mechanism
**Effort:** 3 days

```python
# File: greenlang/infrastructure/events/event_replay.py
"""
Event Replay System for GreenLang

Enables replaying historical events for recovery, debugging, and testing.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReplayStatus(str, Enum):
    """Replay session status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReplayMode(str, Enum):
    """Replay execution mode."""
    SEQUENTIAL = "sequential"     # One event at a time
    PARALLEL = "parallel"         # Concurrent replay
    REALTIME = "realtime"         # Preserve original timing
    FAST_FORWARD = "fast_forward" # As fast as possible


@dataclass
class EventReplayConfig:
    """Configuration for event replay."""
    kafka_bootstrap_servers: str = "localhost:9092"
    schema_registry_url: str = "http://localhost:8081"
    batch_size: int = 100
    max_parallel_events: int = 10
    checkpoint_interval: int = 1000
    enable_dry_run: bool = False


class ReplayFilter(BaseModel):
    """Filter criteria for event replay."""
    event_types: List[str] = Field(default_factory=list)
    agent_ids: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    correlation_ids: List[str] = Field(default_factory=list)
    custom_filter: Optional[str] = None  # JMESPath expression


class ReplaySession(BaseModel):
    """Replay session state."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Session name")
    source_topic: str = Field(..., description="Source topic to replay from")
    target_topic: Optional[str] = Field(default=None, description="Target topic (if different)")
    filter: ReplayFilter = Field(default_factory=ReplayFilter)
    mode: ReplayMode = Field(default=ReplayMode.SEQUENTIAL)
    status: ReplayStatus = Field(default=ReplayStatus.PENDING)

    # Progress tracking
    total_events: int = Field(default=0)
    processed_events: int = Field(default=0)
    failed_events: int = Field(default=0)
    skipped_events: int = Field(default=0)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Checkpointing
    last_checkpoint: Optional[str] = None
    last_offset: Optional[int] = None


class EventReplayManager:
    """
    Manages event replay sessions.

    Supports replaying events from Kafka topics with filtering,
    rate limiting, and checkpoint/resume capabilities.
    """

    def __init__(self, config: EventReplayConfig):
        self.config = config
        self._sessions: Dict[str, ReplaySession] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._event_handlers: Dict[str, Callable] = {}

    def register_handler(
        self,
        event_type: str,
        handler: Callable
    ) -> None:
        """Register event handler for replay."""
        self._event_handlers[event_type] = handler

    async def create_session(
        self,
        name: str,
        source_topic: str,
        filter: Optional[ReplayFilter] = None,
        mode: ReplayMode = ReplayMode.SEQUENTIAL,
        target_topic: Optional[str] = None
    ) -> ReplaySession:
        """Create a new replay session."""
        session = ReplaySession(
            name=name,
            source_topic=source_topic,
            target_topic=target_topic,
            filter=filter or ReplayFilter(),
            mode=mode,
        )

        # Calculate total events
        session.total_events = await self._count_matching_events(
            source_topic,
            session.filter
        )

        self._sessions[session.session_id] = session
        logger.info(
            f"Created replay session {session.session_id}: "
            f"{session.total_events} events to replay"
        )

        return session

    async def start_replay(self, session_id: str) -> None:
        """Start or resume a replay session."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.status == ReplayStatus.RUNNING:
            logger.warning(f"Session {session_id} is already running")
            return

        session.status = ReplayStatus.RUNNING
        session.started_at = datetime.utcnow()

        # Start replay task
        task = asyncio.create_task(
            self._replay_events(session)
        )
        self._active_tasks[session_id] = task

        logger.info(f"Started replay session: {session_id}")

    async def pause_replay(self, session_id: str) -> None:
        """Pause a replay session."""
        session = self._sessions.get(session_id)
        if not session or session.status != ReplayStatus.RUNNING:
            return

        session.status = ReplayStatus.PAUSED

        # Cancel task
        task = self._active_tasks.get(session_id)
        if task:
            task.cancel()

        logger.info(f"Paused replay session: {session_id}")

    async def _replay_events(self, session: ReplaySession) -> None:
        """Execute event replay."""
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

        consumer = AIOKafkaConsumer(
            session.source_topic,
            bootstrap_servers=self.config.kafka_bootstrap_servers,
            auto_offset_reset='earliest' if not session.last_offset else 'none',
            enable_auto_commit=False,
        )

        producer = None
        if session.target_topic:
            producer = AIOKafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers
            )
            await producer.start()

        await consumer.start()

        try:
            # Seek to checkpoint if resuming
            if session.last_offset:
                # Seek to last offset
                pass

            async for message in consumer:
                if session.status != ReplayStatus.RUNNING:
                    break

                # Deserialize event
                event = self._deserialize_event(message.value)

                # Apply filter
                if not self._matches_filter(event, session.filter):
                    session.skipped_events += 1
                    continue

                # Process event
                try:
                    if self.config.enable_dry_run:
                        logger.debug(f"Dry run: {event}")
                    else:
                        await self._process_event(event, session, producer)

                    session.processed_events += 1

                except Exception as e:
                    session.failed_events += 1
                    logger.error(f"Event replay failed: {e}")

                # Checkpoint
                if session.processed_events % self.config.checkpoint_interval == 0:
                    session.last_offset = message.offset
                    await self._save_checkpoint(session)

                # Check completion
                if session.processed_events >= session.total_events:
                    break

                # Rate limiting for realtime mode
                if session.mode == ReplayMode.REALTIME:
                    # Calculate delay based on original event timing
                    await asyncio.sleep(0.1)  # Simplified

            session.status = ReplayStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            logger.info(f"Replay session completed: {session.session_id}")

        except asyncio.CancelledError:
            logger.info(f"Replay session cancelled: {session.session_id}")
        except Exception as e:
            session.status = ReplayStatus.FAILED
            logger.error(f"Replay session failed: {e}")
        finally:
            await consumer.stop()
            if producer:
                await producer.stop()

    async def _process_event(
        self,
        event: Dict[str, Any],
        session: ReplaySession,
        producer: Optional[Any]
    ) -> None:
        """Process a single replayed event."""
        event_type = event.get("event_type")

        # Call registered handler
        handler = self._event_handlers.get(event_type)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

        # Forward to target topic
        if producer and session.target_topic:
            await producer.send_and_wait(
                session.target_topic,
                value=self._serialize_event(event)
            )

    def _matches_filter(
        self,
        event: Dict[str, Any],
        filter: ReplayFilter
    ) -> bool:
        """Check if event matches filter criteria."""
        # Event type filter
        if filter.event_types:
            if event.get("event_type") not in filter.event_types:
                return False

        # Agent ID filter
        if filter.agent_ids:
            if event.get("agent_id") not in filter.agent_ids:
                return False

        # Time range filter
        event_time = event.get("timestamp")
        if event_time:
            if filter.start_time and event_time < filter.start_time:
                return False
            if filter.end_time and event_time > filter.end_time:
                return False

        # Correlation ID filter
        if filter.correlation_ids:
            if event.get("correlation_id") not in filter.correlation_ids:
                return False

        return True

    async def _count_matching_events(
        self,
        topic: str,
        filter: ReplayFilter
    ) -> int:
        """Count events matching filter."""
        # This would query Kafka to count matching events
        # Simplified implementation
        return 0

    async def _save_checkpoint(self, session: ReplaySession) -> None:
        """Save session checkpoint."""
        logger.debug(f"Checkpoint saved: {session.session_id} at offset {session.last_offset}")

    def _deserialize_event(self, data: bytes) -> Dict[str, Any]:
        """Deserialize event from bytes."""
        import json
        return json.loads(data.decode())

    def _serialize_event(self, event: Dict[str, Any]) -> bytes:
        """Serialize event to bytes."""
        import json
        return json.dumps(event).encode()

    def get_session(self, session_id: str) -> Optional[ReplaySession]:
        """Get replay session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[ReplaySession]:
        """List all replay sessions."""
        return list(self._sessions.values())
```

#### TASK-127 & TASK-128: Saga Orchestration & Compensation (ENHANCEMENT)
**Status:** Base implementation exists, needs compensation transaction patterns
**Effort:** 3 days

The existing `saga_orchestrator.py` is well-implemented. Add compensation patterns:

```python
# File: greenlang/infrastructure/events/compensation.py
"""
Compensation Transaction Framework for GreenLang

Provides compensation patterns for saga rollback and recovery.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class CompensationType(str, Enum):
    """Types of compensation."""
    UNDO = "undo"           # Reverse the action
    RETRY = "retry"         # Retry the action
    SKIP = "skip"           # Skip and continue
    MANUAL = "manual"       # Require manual intervention
    FORWARD = "forward"     # Forward recovery (complete remaining)


@dataclass
class CompensationAction:
    """Compensation action definition."""
    step_name: str
    compensation_type: CompensationType
    handler: Callable
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: int = 5
    require_confirmation: bool = False


class CompensationManager:
    """
    Manages compensation transactions for sagas.

    Supports various compensation strategies including
    semantic compensation, retry, and forward recovery.
    """

    def __init__(self):
        self._compensations: Dict[str, CompensationAction] = {}
        self._compensation_log: List[Dict[str, Any]] = []

    def register_compensation(
        self,
        step_name: str,
        handler: Callable,
        compensation_type: CompensationType = CompensationType.UNDO,
        **kwargs
    ) -> None:
        """Register a compensation handler."""
        action = CompensationAction(
            step_name=step_name,
            compensation_type=compensation_type,
            handler=handler,
            **kwargs
        )
        self._compensations[step_name] = action
        logger.debug(f"Registered compensation for: {step_name}")

    async def compensate(
        self,
        step_name: str,
        context: Dict[str, Any],
        step_output: Dict[str, Any]
    ) -> bool:
        """
        Execute compensation for a step.

        Args:
            step_name: Name of step to compensate
            context: Saga context
            step_output: Original step output

        Returns:
            True if compensation succeeded
        """
        action = self._compensations.get(step_name)
        if not action:
            logger.warning(f"No compensation registered for: {step_name}")
            return False

        for attempt in range(action.max_retries):
            try:
                logger.info(f"Executing compensation for: {step_name} (attempt {attempt + 1})")

                if asyncio.iscoroutinefunction(action.handler):
                    result = await asyncio.wait_for(
                        action.handler(context, step_output),
                        timeout=action.timeout_seconds
                    )
                else:
                    result = action.handler(context, step_output)

                # Log compensation
                self._compensation_log.append({
                    "step_name": step_name,
                    "attempt": attempt + 1,
                    "success": True,
                    "result": result,
                })

                logger.info(f"Compensation succeeded for: {step_name}")
                return True

            except asyncio.TimeoutError:
                logger.warning(f"Compensation timeout for: {step_name}")
                if attempt < action.max_retries - 1:
                    await asyncio.sleep(action.retry_delay_seconds)

            except Exception as e:
                logger.error(f"Compensation failed for {step_name}: {e}")
                if attempt < action.max_retries - 1:
                    await asyncio.sleep(action.retry_delay_seconds)

        # Log failure
        self._compensation_log.append({
            "step_name": step_name,
            "attempt": action.max_retries,
            "success": False,
            "error": "Max retries exceeded",
        })

        return False


# Common compensation handlers
async def compensate_database_insert(
    context: Dict[str, Any],
    step_output: Dict[str, Any]
) -> bool:
    """Compensate database insert by deleting the record."""
    record_id = step_output.get("record_id")
    if record_id:
        # Execute DELETE
        logger.info(f"Deleting record: {record_id}")
        return True
    return False


async def compensate_external_api_call(
    context: Dict[str, Any],
    step_output: Dict[str, Any]
) -> bool:
    """Compensate external API call by calling reversal endpoint."""
    transaction_id = step_output.get("transaction_id")
    if transaction_id:
        # Call reversal API
        logger.info(f"Reversing transaction: {transaction_id}")
        return True
    return False


async def compensate_message_publish(
    context: Dict[str, Any],
    step_output: Dict[str, Any]
) -> bool:
    """Compensate message publish by publishing cancellation."""
    message_id = step_output.get("message_id")
    if message_id:
        # Publish cancellation message
        logger.info(f"Publishing cancellation for: {message_id}")
        return True
    return False
```

#### TASK-129: Create Event Monitoring Dashboard
**Effort:** 2 days

```yaml
# File: infrastructure/monitoring/grafana/dashboards/event-monitoring.json
{
  "dashboard": {
    "title": "GreenLang Event Monitoring",
    "uid": "greenlang-events",
    "tags": ["greenlang", "events", "kafka"],
    "timezone": "UTC",
    "refresh": "10s",
    "panels": [
      {
        "title": "Event Throughput",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(kafka_consumer_messages_total{topic=~\"greenlang.*\"}[5m])) by (topic)",
            "legendFormat": "{{topic}}"
          }
        ]
      },
      {
        "title": "Event Processing Latency",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(event_processing_duration_seconds_bucket[5m])) by (le, event_type))",
            "legendFormat": "p95 - {{event_type}}"
          }
        ]
      },
      {
        "title": "Dead Letter Queue Size",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "sum(dlq_entries_total{status=\"pending\"})",
            "legendFormat": "Pending"
          }
        ],
        "options": {
          "colorMode": "value",
          "graphMode": "area"
        },
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 100}
              ]
            }
          }
        }
      },
      {
        "title": "Active Sagas",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
        "targets": [
          {
            "expr": "sum(saga_active_total)",
            "legendFormat": "Active"
          }
        ]
      },
      {
        "title": "Saga Success Rate",
        "type": "gauge",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "sum(rate(saga_completed_total[1h])) / sum(rate(saga_started_total[1h])) * 100",
            "legendFormat": "Success Rate"
          }
        ],
        "options": {
          "showThresholdMarkers": true
        },
        "fieldConfig": {
          "defaults": {
            "max": 100,
            "min": 0,
            "unit": "percent",
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 90},
                {"color": "green", "value": 99}
              ]
            }
          }
        }
      },
      {
        "title": "Event Failures by Type",
        "type": "piechart",
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 12},
        "targets": [
          {
            "expr": "sum(dlq_entries_total) by (failure_reason)",
            "legendFormat": "{{failure_reason}}"
          }
        ]
      },
      {
        "title": "Consumer Lag",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 16, "x": 8, "y": 12},
        "targets": [
          {
            "expr": "sum(kafka_consumer_group_lag) by (consumer_group, topic)",
            "legendFormat": "{{consumer_group}} - {{topic}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "custom": {
              "fillOpacity": 10
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 1000},
                {"color": "red", "value": 10000}
              ]
            }
          }
        }
      }
    ]
  }
}
```

#### TASK-130: Implement Event Versioning
**Effort:** 2 days

```python
# File: greenlang/infrastructure/events/event_versioning.py
"""
Event Schema Versioning for GreenLang

Provides schema evolution and compatibility management for events.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CompatibilityMode(str, Enum):
    """Schema compatibility modes."""
    BACKWARD = "backward"         # New schema can read old data
    BACKWARD_TRANSITIVE = "backward_transitive"
    FORWARD = "forward"           # Old schema can read new data
    FORWARD_TRANSITIVE = "forward_transitive"
    FULL = "full"                 # Both backward and forward
    FULL_TRANSITIVE = "full_transitive"
    NONE = "none"                 # No compatibility checking


class SchemaVersion(BaseModel):
    """Schema version metadata."""
    version: int = Field(..., ge=1)
    schema_id: str
    schema_definition: Dict[str, Any]
    created_at: str
    description: Optional[str] = None
    deprecated: bool = False
    deprecation_notice: Optional[str] = None


class VersionedEvent(BaseModel):
    """Base class for versioned events."""
    event_type: str = Field(..., description="Event type identifier")
    schema_version: int = Field(default=1, description="Schema version")

    class Config:
        extra = "allow"  # Allow unknown fields for forward compatibility


class EventSchemaRegistry:
    """
    Schema registry for event versioning.

    Manages schema versions and provides compatibility checking.
    """

    def __init__(
        self,
        compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARD
    ):
        self.compatibility_mode = compatibility_mode
        self._schemas: Dict[str, Dict[int, SchemaVersion]] = {}
        self._upgraders: Dict[str, Dict[int, Callable]] = {}
        self._downgraders: Dict[str, Dict[int, Callable]] = {}

    def register_schema(
        self,
        event_type: str,
        version: int,
        schema_definition: Dict[str, Any],
        description: Optional[str] = None
    ) -> SchemaVersion:
        """Register a new schema version."""
        from datetime import datetime
        import hashlib
        import json

        # Generate schema ID
        schema_str = json.dumps(schema_definition, sort_keys=True)
        schema_id = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        schema_version = SchemaVersion(
            version=version,
            schema_id=schema_id,
            schema_definition=schema_definition,
            created_at=datetime.utcnow().isoformat(),
            description=description,
        )

        # Check compatibility if not first version
        if event_type in self._schemas:
            current_version = max(self._schemas[event_type].keys())
            current_schema = self._schemas[event_type][current_version]

            if not self._check_compatibility(
                current_schema.schema_definition,
                schema_definition
            ):
                raise ValueError(
                    f"Schema version {version} is not compatible with "
                    f"version {current_version} under {self.compatibility_mode.value} mode"
                )

        if event_type not in self._schemas:
            self._schemas[event_type] = {}

        self._schemas[event_type][version] = schema_version

        logger.info(f"Registered schema: {event_type} v{version}")
        return schema_version

    def register_upgrader(
        self,
        event_type: str,
        from_version: int,
        to_version: int,
        upgrader: Callable[[Dict], Dict]
    ) -> None:
        """Register a schema upgrade function."""
        if event_type not in self._upgraders:
            self._upgraders[event_type] = {}

        key = f"{from_version}_{to_version}"
        self._upgraders[event_type][key] = upgrader

        logger.debug(f"Registered upgrader: {event_type} v{from_version} -> v{to_version}")

    def register_downgrader(
        self,
        event_type: str,
        from_version: int,
        to_version: int,
        downgrader: Callable[[Dict], Dict]
    ) -> None:
        """Register a schema downgrade function."""
        if event_type not in self._downgraders:
            self._downgraders[event_type] = {}

        key = f"{from_version}_{to_version}"
        self._downgraders[event_type][key] = downgrader

        logger.debug(f"Registered downgrader: {event_type} v{from_version} -> v{to_version}")

    def upgrade_event(
        self,
        event: Dict[str, Any],
        target_version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Upgrade an event to target version.

        If target_version is None, upgrades to latest version.
        """
        event_type = event.get("event_type")
        current_version = event.get("schema_version", 1)

        if not event_type or event_type not in self._schemas:
            return event

        if target_version is None:
            target_version = max(self._schemas[event_type].keys())

        if current_version >= target_version:
            return event

        # Apply upgraders sequentially
        result = dict(event)
        for v in range(current_version, target_version):
            key = f"{v}_{v + 1}"
            upgrader = self._upgraders.get(event_type, {}).get(key)

            if upgrader:
                result = upgrader(result)
                result["schema_version"] = v + 1
            else:
                logger.warning(
                    f"No upgrader for {event_type} v{v} -> v{v + 1}, "
                    f"attempting auto-upgrade"
                )
                result["schema_version"] = v + 1

        return result

    def downgrade_event(
        self,
        event: Dict[str, Any],
        target_version: int
    ) -> Dict[str, Any]:
        """Downgrade an event to target version."""
        event_type = event.get("event_type")
        current_version = event.get("schema_version", 1)

        if not event_type or current_version <= target_version:
            return event

        # Apply downgraders sequentially
        result = dict(event)
        for v in range(current_version, target_version, -1):
            key = f"{v}_{v - 1}"
            downgrader = self._downgraders.get(event_type, {}).get(key)

            if downgrader:
                result = downgrader(result)
                result["schema_version"] = v - 1
            else:
                raise ValueError(
                    f"No downgrader for {event_type} v{v} -> v{v - 1}"
                )

        return result

    def _check_compatibility(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any]
    ) -> bool:
        """Check schema compatibility."""
        if self.compatibility_mode == CompatibilityMode.NONE:
            return True

        old_fields = set(old_schema.get("properties", {}).keys())
        new_fields = set(new_schema.get("properties", {}).keys())
        old_required = set(old_schema.get("required", []))
        new_required = set(new_schema.get("required", []))

        if self.compatibility_mode in [
            CompatibilityMode.BACKWARD,
            CompatibilityMode.BACKWARD_TRANSITIVE,
            CompatibilityMode.FULL,
            CompatibilityMode.FULL_TRANSITIVE
        ]:
            # New schema must not add required fields
            added_required = new_required - old_required
            if added_required:
                logger.warning(f"Backward incompatible: added required fields {added_required}")
                return False

        if self.compatibility_mode in [
            CompatibilityMode.FORWARD,
            CompatibilityMode.FORWARD_TRANSITIVE,
            CompatibilityMode.FULL,
            CompatibilityMode.FULL_TRANSITIVE
        ]:
            # New schema must not remove fields
            removed_fields = old_fields - new_fields
            if removed_fields:
                logger.warning(f"Forward incompatible: removed fields {removed_fields}")
                return False

        return True

    def get_latest_version(self, event_type: str) -> Optional[int]:
        """Get latest schema version for event type."""
        if event_type not in self._schemas:
            return None
        return max(self._schemas[event_type].keys())

    def get_schema(
        self,
        event_type: str,
        version: Optional[int] = None
    ) -> Optional[SchemaVersion]:
        """Get schema for event type and version."""
        if event_type not in self._schemas:
            return None

        if version is None:
            version = max(self._schemas[event_type].keys())

        return self._schemas[event_type].get(version)

    def deprecate_version(
        self,
        event_type: str,
        version: int,
        notice: str
    ) -> None:
        """Mark a schema version as deprecated."""
        schema = self.get_schema(event_type, version)
        if schema:
            schema.deprecated = True
            schema.deprecation_notice = notice
            logger.info(f"Deprecated schema: {event_type} v{version}")


# Example usage
def create_emission_event_schemas():
    """Create emission event schema versions."""
    registry = EventSchemaRegistry(CompatibilityMode.BACKWARD)

    # Version 1
    registry.register_schema(
        event_type="emission.calculated",
        version=1,
        schema_definition={
            "properties": {
                "emission_id": {"type": "string"},
                "co2_value": {"type": "number"},
                "unit": {"type": "string"},
                "timestamp": {"type": "string"},
            },
            "required": ["emission_id", "co2_value", "unit", "timestamp"]
        }
    )

    # Version 2 - Added optional fields
    registry.register_schema(
        event_type="emission.calculated",
        version=2,
        schema_definition={
            "properties": {
                "emission_id": {"type": "string"},
                "co2_value": {"type": "number"},
                "unit": {"type": "string"},
                "timestamp": {"type": "string"},
                "source": {"type": "string"},     # New optional field
                "scope": {"type": "integer"},     # New optional field
            },
            "required": ["emission_id", "co2_value", "unit", "timestamp"]
        }
    )

    # Register upgrader
    def upgrade_v1_to_v2(event: Dict) -> Dict:
        event["source"] = event.get("source", "unknown")
        event["scope"] = event.get("scope", 1)
        return event

    registry.register_upgrader("emission.calculated", 1, 2, upgrade_v1_to_v2)

    return registry
```

---

### Priority 3: MEDIUM - API Design (Tasks 133-136)

These tasks have base implementations. Focus on production hardening.

#### TASK-133 & TASK-134: GraphQL & gRPC (ENHANCEMENT)
**Status:** Base implementations exist
**Effort:** 3 days for production hardening

Add to existing implementations:
- Authentication middleware
- Rate limiting
- Tracing integration
- Error handling improvements

#### TASK-135 & TASK-136: Webhooks & SSE (ENHANCEMENT)
**Status:** Base implementations exist
**Effort:** 2 days for production hardening

### Priority 4: HIGH - Chaos Engineering (Task 149)

```yaml
# File: infrastructure/chaos/litmus-experiments.yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosExperiment
metadata:
  name: greenlang-pod-kill
  namespace: greenlang-agents
spec:
  definition:
    scope: Namespaced
    permissions:
      - apiGroups: [""]
        resources: ["pods"]
        verbs: ["delete", "list", "get"]
    image: litmuschaos/go-runner:latest
    args:
      - -c
      - ./experiments -name pod-delete
    command:
      - /bin/bash
    env:
      - name: TOTAL_CHAOS_DURATION
        value: "30"
      - name: CHAOS_INTERVAL
        value: "10"
      - name: FORCE
        value: "false"
      - name: TARGET_PODS
        value: ""
      - name: PODS_AFFECTED_PERC
        value: "50"
    labels:
      name: pod-delete

---
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosExperiment
metadata:
  name: greenlang-network-chaos
  namespace: greenlang-agents
spec:
  definition:
    scope: Namespaced
    permissions:
      - apiGroups: [""]
        resources: ["pods"]
        verbs: ["get", "list"]
    image: litmuschaos/go-runner:latest
    args:
      - -c
      - ./experiments -name pod-network-latency
    command:
      - /bin/bash
    env:
      - name: TOTAL_CHAOS_DURATION
        value: "60"
      - name: NETWORK_LATENCY
        value: "200"
      - name: TARGET_PODS
        value: ""
      - name: CONTAINER_RUNTIME
        value: "containerd"
    labels:
      name: pod-network-latency

---
# Chaos Engine to run experiments
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: greenlang-chaos-engine
  namespace: greenlang-agents
spec:
  appinfo:
    appns: greenlang-agents
    applabel: app=greenlang-api
    appkind: deployment
  engineState: active
  chaosServiceAccount: litmus-admin
  experiments:
    - name: pod-delete
      spec:
        components:
          env:
            - name: TOTAL_CHAOS_DURATION
              value: "30"
            - name: CHAOS_INTERVAL
              value: "10"
            - name: FORCE
              value: "false"
```

---

## Implementation Timeline

```
Week 1-2: Security Architecture (CRITICAL)
  - TASK-151: Istio mTLS configuration
  - TASK-152: Keycloak OAuth2/OIDC setup
  - TASK-153: RBAC policies

Week 3-4: Security Architecture (continued)
  - TASK-154: ABAC implementation
  - TASK-155: Vault API key management
  - TASK-156: Secrets rotation
  - TASK-160: Vulnerability scanning pipeline

Week 5-6: Event Architecture
  - TASK-124: DLQ enhancements (Redis backend)
  - TASK-125: Event replay mechanism
  - TASK-127: Saga orchestration enhancements
  - TASK-128: Compensation transactions

Week 7: Event Architecture (continued)
  - TASK-129: Event monitoring dashboard
  - TASK-130: Event versioning

Week 8: API & Resilience
  - TASK-133: GraphQL production hardening
  - TASK-134: gRPC production hardening
  - TASK-135: Webhook enhancements
  - TASK-136: SSE enhancements
  - TASK-149: Chaos engineering tests

Week 9-10: Integration & Testing
  - Integration testing of all components
  - Performance testing
  - Security audit
  - Documentation updates
```

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Security vulnerabilities in production | HIGH | MEDIUM | Implement all security tasks before deployment |
| Event ordering issues | MEDIUM | MEDIUM | Implement event versioning early |
| Saga compensation failures | HIGH | LOW | Comprehensive compensation testing |
| Keycloak integration complexity | MEDIUM | MEDIUM | Use reference implementation, thorough testing |
| Vault HA configuration | MEDIUM | LOW | Follow HashiCorp best practices |

---

## Success Criteria

1. **Security:** All security scans pass with no critical/high vulnerabilities
2. **mTLS:** All service-to-service communication encrypted (Istio STRICT mode)
3. **Authentication:** 100% of API endpoints protected by OAuth2/OIDC
4. **Events:** DLQ processing < 100 pending entries, Saga success rate > 99%
5. **API:** GraphQL/gRPC latency p95 < 100ms
6. **Chaos:** System recovers from pod failures within 30 seconds

---

## File Locations Summary

| Component | Path |
|-----------|------|
| Istio mTLS | `infrastructure/k8s/istio/peer-authentication.yaml` |
| Keycloak Terraform | `terraform/modules/keycloak/main.tf` |
| OAuth2 Client | `greenlang/infrastructure/security/oauth2_client.py` |
| RBAC Engine | `greenlang/infrastructure/security/rbac.py` |
| ABAC Engine | `greenlang/infrastructure/security/abac.py` |
| Vault Client | `greenlang/infrastructure/security/vault_client.py` |
| Secrets Rotation | `greenlang/infrastructure/security/secrets_rotation.py` |
| Security Scanning | `.github/workflows/security-scanning.yml` |
| Event Replay | `greenlang/infrastructure/events/event_replay.py` |
| Compensation | `greenlang/infrastructure/events/compensation.py` |
| Event Versioning | `greenlang/infrastructure/events/event_versioning.py` |
| Event Dashboard | `infrastructure/monitoring/grafana/dashboards/event-monitoring.json` |
| Chaos Tests | `infrastructure/chaos/litmus-experiments.yaml` |

---

**Document Prepared By:** GL-DevOpsEngineer
**Review Required By:** Security Team, Platform Team
**Approval Required:** CTO, VP Engineering
