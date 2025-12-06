"""
Role-Based Access Control Manager for GreenLang Process Heat Agents
====================================================================

This module implements RBAC for the GreenLang platform, providing
fine-grained access control for Process Heat agents and resources.

Features:
    - Process Heat agent-specific permissions
    - Hierarchical role inheritance
    - Resource-based authorization
    - Comprehensive audit logging
    - Policy enforcement with caching
    - Integration with OAuth2Provider

Roles:
    - admin: Full access to all resources and agents
    - operator: Execute agents, view results, limited config changes
    - viewer: Read-only access to results and reports
    - auditor: Access to audit logs and compliance reports
    - agent-service: Service-to-service communication

Permissions:
    - agents:execute - Execute agent calculations
    - agents:view - View agent status and results
    - agents:configure - Modify agent configuration
    - results:view - View calculation results
    - results:export - Export results to files
    - config:view - View system configuration
    - config:modify - Modify system configuration
    - audit:view - View audit logs
    - provenance:view - View data provenance

OWASP Compliance:
    - A01:2021: Broken Access Control - Fine-grained RBAC
    - A09:2021: Security Logging - Comprehensive audit logging

Example:
    >>> from greenlang.infrastructure.auth import RBACManager, OAuth2Provider
    >>>
    >>> oauth2 = OAuth2Provider(config)
    >>> rbac = RBACManager(oauth2)
    >>>
    >>> # Check permission
    >>> if await rbac.can_execute_agent(token, "gl-010"):
    ...     result = await agent.execute()

Author: Security Team
Created: 2025-12-06
TASK-152: Implement OAuth2/OIDC (Keycloak) for Process Heat Agents
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

from greenlang.infrastructure.auth.oauth2_provider import (
    OAuth2Provider,
    TokenInfo,
    OAuth2Error,
    TokenValidationError,
)

logger = logging.getLogger(__name__)

# Separate audit logger for authorization decisions
audit_logger = logging.getLogger("greenlang.security.audit")


# =============================================================================
# Exceptions
# =============================================================================


class AuthorizationError(Exception):
    """Base exception for authorization errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "authorization_error",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class PermissionDeniedError(AuthorizationError):
    """Permission denied for the requested action."""

    def __init__(
        self,
        action: str,
        resource: str,
        reason: Optional[str] = None
    ):
        message = f"Permission denied: {action} on {resource}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "permission_denied")
        self.action = action
        self.resource = resource


class ResourceNotFoundError(AuthorizationError):
    """Resource not found for authorization check."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            f"Resource not found: {resource_type}/{resource_id}",
            "resource_not_found"
        )


# =============================================================================
# Enums and Constants
# =============================================================================


class Permission(str, Enum):
    """GreenLang permission definitions."""

    # Wildcard
    ALL = "*"

    # Agent permissions
    AGENTS_EXECUTE = "agents:execute"
    AGENTS_VIEW = "agents:view"
    AGENTS_CONFIGURE = "agents:configure"
    AGENTS_INTERNAL = "agents:internal"

    # Results permissions
    RESULTS_VIEW = "results:view"
    RESULTS_EXPORT = "results:export"
    RESULTS_DELETE = "results:delete"

    # Configuration permissions
    CONFIG_VIEW = "config:view"
    CONFIG_MODIFY = "config:modify"

    # Audit permissions
    AUDIT_VIEW = "audit:view"
    AUDIT_EXPORT = "audit:export"

    # Provenance permissions
    PROVENANCE_VIEW = "provenance:view"

    # Compliance permissions
    COMPLIANCE_VIEW = "compliance:view"
    COMPLIANCE_REPORT = "compliance:report"

    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"

    # Dashboard permissions
    DASHBOARDS_VIEW = "dashboards:view"
    DASHBOARDS_CREATE = "dashboards:create"
    DASHBOARDS_MODIFY = "dashboards:modify"

    # Reports permissions
    REPORTS_VIEW = "reports:view"
    REPORTS_CREATE = "reports:create"

    # Security permissions
    SECURITY_VIEW = "security:view"
    SECURITY_MANAGE = "security:manage"


class Role(str, Enum):
    """GreenLang role definitions."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    AUDITOR = "auditor"
    AGENT_SERVICE = "agent-service"


class ResourceType(str, Enum):
    """Resource types for authorization."""

    AGENT = "agent"
    RESULT = "result"
    CONFIG = "config"
    AUDIT_LOG = "audit_log"
    PROVENANCE = "provenance"
    DASHBOARD = "dashboard"
    REPORT = "report"


class Action(str, Enum):
    """Actions that can be performed on resources."""

    EXECUTE = "execute"
    VIEW = "view"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    EXPORT = "export"
    CONFIGURE = "configure"


class DecisionResult(str, Enum):
    """Authorization decision results."""

    PERMIT = "PERMIT"
    DENY = "DENY"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INDETERMINATE = "INDETERMINATE"


# =============================================================================
# Models
# =============================================================================


class RBACConfig(BaseModel):
    """Configuration for RBAC manager."""

    # Role hierarchy
    role_hierarchy: Dict[str, List[str]] = Field(
        default={
            "admin": ["operator", "viewer", "auditor"],
            "operator": ["viewer"],
            "viewer": [],
            "auditor": [],
            "agent-service": []
        },
        description="Role inheritance hierarchy"
    )

    # Role permissions
    role_permissions: Dict[str, List[str]] = Field(
        default={
            "admin": ["*"],
            "operator": [
                "agents:execute", "agents:view",
                "results:view", "results:export",
                "config:view",
                "dashboards:view",
                "reports:view"
            ],
            "viewer": [
                "results:view",
                "dashboards:view",
                "reports:view"
            ],
            "auditor": [
                "audit:view", "audit:export",
                "provenance:view",
                "compliance:view", "compliance:report",
                "security:view"
            ],
            "agent-service": [
                "agents:internal",
                "data:read", "data:write"
            ]
        },
        description="Permissions assigned to each role"
    )

    # Agent-specific access controls
    agent_access_control: Dict[str, Dict[str, List[str]]] = Field(
        default={
            # Default agent access by role
            "default": {
                "admin": ["execute", "view", "configure"],
                "operator": ["execute", "view"],
                "viewer": ["view"],
                "auditor": ["view"],
                "agent-service": ["execute", "view"]
            }
        },
        description="Agent-specific access control rules"
    )

    # Audit settings
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    audit_sensitive_actions: List[str] = Field(
        default=[
            "agents:configure",
            "config:modify",
            "data:delete",
            "security:manage"
        ],
        description="Actions that require enhanced audit logging"
    )

    # Caching settings
    decision_cache_ttl_seconds: int = Field(
        default=60,
        ge=0,
        description="Authorization decision cache TTL"
    )
    max_cached_decisions: int = Field(
        default=10000,
        ge=100,
        description="Maximum cached decisions"
    )

    # Policy settings
    default_deny: bool = Field(
        default=True,
        description="Deny by default if no policy matches"
    )
    require_explicit_permission: bool = Field(
        default=True,
        description="Require explicit permission grant"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class AuthorizationRequest(BaseModel):
    """Request for authorization decision."""

    subject: str = Field(..., description="Subject (user ID)")
    action: str = Field(..., description="Action to perform")
    resource_type: str = Field(..., description="Type of resource")
    resource_id: str = Field(..., description="Resource identifier")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for authorization"
    )

    @validator("action")
    def validate_action(cls, v: str) -> str:
        """Validate action format."""
        if not v or ":" not in v and v not in [a.value for a in Action]:
            # Allow both simple actions and namespaced permissions
            pass
        return v


class AuthorizationDecision(BaseModel):
    """Result of an authorization decision."""

    decision: DecisionResult = Field(..., description="Authorization decision")
    request: AuthorizationRequest = Field(..., description="Original request")
    reason: str = Field(..., description="Reason for decision")
    matched_policy: Optional[str] = Field(
        None,
        description="Policy that matched"
    )
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Evaluation timestamp"
    )
    evaluation_time_ms: float = Field(
        0.0,
        description="Evaluation time in milliseconds"
    )
    obligations: List[str] = Field(
        default_factory=list,
        description="Obligations that must be fulfilled"
    )

    def is_permitted(self) -> bool:
        """Check if the decision permits the action."""
        return self.decision == DecisionResult.PERMIT

    def get_provenance_hash(self) -> str:
        """Generate SHA-256 hash for audit trail."""
        data = (
            f"{self.request.subject}:{self.request.action}:"
            f"{self.request.resource_type}:{self.request.resource_id}:"
            f"{self.decision.value}:{self.evaluated_at.isoformat()}"
        )
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class AuditEntry:
    """Audit log entry for authorization decisions."""

    timestamp: datetime
    subject: str
    action: str
    resource_type: str
    resource_id: str
    decision: DecisionResult
    reason: str
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None
    provenance_hash: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON for logging."""
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "subject": self.subject,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "correlation_id": self.correlation_id,
            "provenance_hash": self.provenance_hash,
            "context": self.additional_context
        })


# =============================================================================
# RBAC Manager Implementation
# =============================================================================


class RBACManager:
    """
    Role-Based Access Control Manager for GreenLang.

    This class provides fine-grained access control for Process Heat agents
    and other GreenLang resources. It integrates with OAuth2Provider for
    authentication and implements comprehensive authorization policies.

    Attributes:
        oauth2_provider: OAuth2 provider for token validation
        config: RBAC configuration
        _decision_cache: Cached authorization decisions
        _audit_entries: Recent audit entries

    Example:
        >>> oauth2 = OAuth2Provider(oauth2_config)
        >>> rbac = RBACManager(oauth2, rbac_config)
        >>>
        >>> # Check if user can execute an agent
        >>> if await rbac.can_execute_agent(token, "gl-010"):
        ...     result = await agent.execute()
        >>>
        >>> # Check arbitrary permission
        >>> decision = await rbac.authorize(
        ...     token,
        ...     action="execute",
        ...     resource_type="agent",
        ...     resource_id="gl-010"
        ... )
        >>> if decision.is_permitted():
        ...     print("Access granted")
    """

    def __init__(
        self,
        oauth2_provider: OAuth2Provider,
        config: Optional[RBACConfig] = None
    ):
        """
        Initialize RBAC manager.

        Args:
            oauth2_provider: OAuth2 provider for token validation
            config: RBAC configuration
        """
        self.oauth2_provider = oauth2_provider
        self.config = config or RBACConfig()
        self._decision_cache: Dict[str, Tuple[AuthorizationDecision, float]] = {}
        self._audit_entries: List[AuditEntry] = []
        self._expanded_roles: Dict[str, Set[str]] = {}

        # Pre-compute role hierarchy
        self._expand_role_hierarchy()

        # Metrics
        self._authorization_count = 0
        self._permit_count = 0
        self._deny_count = 0
        self._cache_hits = 0

        logger.info("RBACManager initialized")

    def _expand_role_hierarchy(self) -> None:
        """Pre-compute expanded role hierarchy for efficient lookups."""
        for role in self.config.role_hierarchy:
            self._expanded_roles[role] = self._get_all_inherited_roles(role)

    def _get_all_inherited_roles(self, role: str) -> Set[str]:
        """
        Get all roles inherited by a given role.

        Args:
            role: Role to expand

        Returns:
            Set of all inherited roles (including the role itself)
        """
        result = {role}
        for inherited in self.config.role_hierarchy.get(role, []):
            result.update(self._get_all_inherited_roles(inherited))
        return result

    def _get_cache_key(
        self,
        subject: str,
        action: str,
        resource_type: str,
        resource_id: str
    ) -> str:
        """Generate cache key for authorization decision."""
        key_data = f"{subject}:{action}:{resource_type}:{resource_id}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def _check_decision_cache(
        self,
        subject: str,
        action: str,
        resource_type: str,
        resource_id: str
    ) -> Optional[AuthorizationDecision]:
        """Check if authorization decision is cached."""
        cache_key = self._get_cache_key(subject, action, resource_type, resource_id)
        cached = self._decision_cache.get(cache_key)

        if cached:
            decision, cached_at = cached
            cache_age = time.time() - cached_at

            if cache_age < self.config.decision_cache_ttl_seconds:
                self._cache_hits += 1
                return decision

            # Remove expired entry
            del self._decision_cache[cache_key]

        return None

    def _cache_decision(
        self,
        subject: str,
        action: str,
        resource_type: str,
        resource_id: str,
        decision: AuthorizationDecision
    ) -> None:
        """Cache authorization decision."""
        # Evict old entries if cache is full
        if len(self._decision_cache) >= self.config.max_cached_decisions:
            entries = sorted(
                self._decision_cache.items(),
                key=lambda x: x[1][1]
            )
            for key, _ in entries[:len(entries) // 10]:
                del self._decision_cache[key]

        cache_key = self._get_cache_key(subject, action, resource_type, resource_id)
        self._decision_cache[cache_key] = (decision, time.time())

    def _get_effective_permissions(
        self,
        token_info: TokenInfo
    ) -> Set[str]:
        """
        Get all effective permissions for a user based on their roles.

        Args:
            token_info: Validated token information

        Returns:
            Set of all effective permissions
        """
        permissions = set()

        # Add permissions from token
        permissions.update(token_info.greenlang_permissions)

        # Add permissions from roles
        for role in token_info.roles:
            # Get expanded roles (including inherited)
            expanded = self._expanded_roles.get(role, {role})
            for r in expanded:
                role_perms = self.config.role_permissions.get(r, [])
                permissions.update(role_perms)

        return permissions

    def _check_permission(
        self,
        permissions: Set[str],
        required_permission: str
    ) -> bool:
        """
        Check if required permission is in the permission set.

        Supports wildcard matching (e.g., "*" matches everything,
        "agents:*" matches "agents:execute").

        Args:
            permissions: Set of granted permissions
            required_permission: Permission to check

        Returns:
            True if permission is granted
        """
        # Check for exact match
        if required_permission in permissions:
            return True

        # Check for wildcard
        if "*" in permissions:
            return True

        # Check for namespace wildcard (e.g., "agents:*")
        if ":" in required_permission:
            namespace = required_permission.split(":")[0]
            if f"{namespace}:*" in permissions:
                return True

        return False

    def _audit_decision(
        self,
        decision: AuthorizationDecision,
        token_info: TokenInfo,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log authorization decision to audit log.

        Args:
            decision: Authorization decision
            token_info: User token information
            context: Additional context
        """
        if not self.config.audit_enabled:
            return

        entry = AuditEntry(
            timestamp=decision.evaluated_at,
            subject=decision.request.subject,
            action=decision.request.action,
            resource_type=decision.request.resource_type,
            resource_id=decision.request.resource_id,
            decision=decision.decision,
            reason=decision.reason,
            client_ip=context.get("client_ip") if context else None,
            user_agent=context.get("user_agent") if context else None,
            correlation_id=context.get("correlation_id") if context else None,
            provenance_hash=decision.get_provenance_hash(),
            additional_context={
                "user_email": token_info.email,
                "user_roles": token_info.roles,
                "evaluation_time_ms": decision.evaluation_time_ms
            }
        )

        # Store entry for in-memory access
        self._audit_entries.append(entry)
        if len(self._audit_entries) > 10000:
            self._audit_entries = self._audit_entries[-5000:]

        # Log to audit logger
        log_level = logging.WARNING if decision.decision == DecisionResult.DENY else logging.INFO
        audit_logger.log(log_level, entry.to_json())

        # Enhanced logging for sensitive actions
        if decision.request.action in self.config.audit_sensitive_actions:
            audit_logger.warning(
                f"SENSITIVE_ACTION: {entry.to_json()}"
            )

    async def authorize(
        self,
        token: str,
        action: str,
        resource_type: str,
        resource_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AuthorizationDecision:
        """
        Make an authorization decision.

        This is the main authorization method that validates the token,
        checks permissions, and returns a decision.

        Args:
            token: OAuth2 access token
            action: Action to perform
            resource_type: Type of resource
            resource_id: Resource identifier
            context: Additional context (client_ip, user_agent, etc.)

        Returns:
            Authorization decision

        Raises:
            AuthorizationError: If authorization check fails
        """
        start_time = time.time()
        self._authorization_count += 1

        try:
            # Validate token
            token_info = await self.oauth2_provider.validate_token(token)

            # Check cache
            cached_decision = self._check_decision_cache(
                token_info.subject, action, resource_type, resource_id
            )
            if cached_decision:
                return cached_decision

            # Build authorization request
            request = AuthorizationRequest(
                subject=token_info.subject,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                context=context or {}
            )

            # Evaluate policies
            decision = self._evaluate_policies(request, token_info)

            # Calculate evaluation time
            evaluation_time = (time.time() - start_time) * 1000
            decision.evaluation_time_ms = evaluation_time

            # Update metrics
            if decision.is_permitted():
                self._permit_count += 1
            else:
                self._deny_count += 1

            # Cache decision
            self._cache_decision(
                token_info.subject, action, resource_type, resource_id, decision
            )

            # Audit log
            self._audit_decision(decision, token_info, context)

            return decision

        except OAuth2Error as e:
            logger.error(f"Token validation failed during authorization: {e}")
            raise AuthorizationError(f"Authentication failed: {e}")
        except Exception as e:
            logger.error(f"Authorization check failed: {e}")
            raise AuthorizationError(f"Authorization check failed: {e}")

    def _evaluate_policies(
        self,
        request: AuthorizationRequest,
        token_info: TokenInfo
    ) -> AuthorizationDecision:
        """
        Evaluate authorization policies.

        Args:
            request: Authorization request
            token_info: Validated token info

        Returns:
            Authorization decision
        """
        # Get effective permissions
        permissions = self._get_effective_permissions(token_info)

        # Build required permission string
        if ":" in request.action:
            required_permission = request.action
        else:
            required_permission = f"{request.resource_type}s:{request.action}"

        # Check permission
        if self._check_permission(permissions, required_permission):
            return AuthorizationDecision(
                decision=DecisionResult.PERMIT,
                request=request,
                reason=f"Permission granted via {required_permission}",
                matched_policy="permission_check"
            )

        # Check resource-specific policies
        if request.resource_type == "agent":
            agent_decision = self._check_agent_access(
                request, token_info
            )
            if agent_decision:
                return agent_decision

        # Default deny
        if self.config.default_deny:
            return AuthorizationDecision(
                decision=DecisionResult.DENY,
                request=request,
                reason=f"No matching policy for {required_permission}",
                matched_policy="default_deny"
            )

        return AuthorizationDecision(
            decision=DecisionResult.NOT_APPLICABLE,
            request=request,
            reason="No applicable policy found",
            matched_policy=None
        )

    def _check_agent_access(
        self,
        request: AuthorizationRequest,
        token_info: TokenInfo
    ) -> Optional[AuthorizationDecision]:
        """
        Check agent-specific access control.

        Args:
            request: Authorization request
            token_info: Token information

        Returns:
            Authorization decision if policy matches, None otherwise
        """
        agent_id = request.resource_id
        action = request.action

        # Get agent-specific access control
        agent_acl = self.config.agent_access_control.get(
            agent_id,
            self.config.agent_access_control.get("default", {})
        )

        # Check each role
        for role in token_info.roles:
            allowed_actions = agent_acl.get(role, [])
            if action in allowed_actions or "*" in allowed_actions:
                return AuthorizationDecision(
                    decision=DecisionResult.PERMIT,
                    request=request,
                    reason=f"Agent access granted via role {role}",
                    matched_policy=f"agent_acl:{agent_id}:{role}"
                )

        return None

    async def can_execute_agent(
        self,
        token: str,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user can execute a specific agent.

        Args:
            token: OAuth2 access token
            agent_id: Agent identifier
            context: Additional context

        Returns:
            True if execution is permitted
        """
        decision = await self.authorize(
            token,
            action="execute",
            resource_type="agent",
            resource_id=agent_id,
            context=context
        )
        return decision.is_permitted()

    async def can_view_results(
        self,
        token: str,
        result_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user can view calculation results.

        Args:
            token: OAuth2 access token
            result_id: Result identifier
            context: Additional context

        Returns:
            True if viewing is permitted
        """
        decision = await self.authorize(
            token,
            action="view",
            resource_type="result",
            resource_id=result_id,
            context=context
        )
        return decision.is_permitted()

    async def can_modify_config(
        self,
        token: str,
        config_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user can modify configuration.

        Args:
            token: OAuth2 access token
            config_id: Configuration identifier
            context: Additional context

        Returns:
            True if modification is permitted
        """
        decision = await self.authorize(
            token,
            action="modify",
            resource_type="config",
            resource_id=config_id,
            context=context
        )
        return decision.is_permitted()

    async def can_view_audit_logs(
        self,
        token: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user can view audit logs.

        Args:
            token: OAuth2 access token
            context: Additional context

        Returns:
            True if viewing is permitted
        """
        decision = await self.authorize(
            token,
            action=Permission.AUDIT_VIEW.value,
            resource_type="audit_log",
            resource_id="*",
            context=context
        )
        return decision.is_permitted()

    async def can_export_data(
        self,
        token: str,
        data_type: str,
        data_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user can export data.

        Args:
            token: OAuth2 access token
            data_type: Type of data to export
            data_id: Data identifier
            context: Additional context

        Returns:
            True if export is permitted
        """
        decision = await self.authorize(
            token,
            action="export",
            resource_type=data_type,
            resource_id=data_id,
            context=context
        )
        return decision.is_permitted()

    async def require_permission(
        self,
        token: str,
        permission: Union[str, Permission],
        resource_type: str = "system",
        resource_id: str = "*",
        context: Optional[Dict[str, Any]] = None
    ) -> TokenInfo:
        """
        Require a specific permission, raising an error if not granted.

        Args:
            token: OAuth2 access token
            permission: Required permission
            resource_type: Type of resource
            resource_id: Resource identifier
            context: Additional context

        Returns:
            Token information if permission is granted

        Raises:
            PermissionDeniedError: If permission is denied
        """
        if isinstance(permission, Permission):
            permission = permission.value

        token_info = await self.oauth2_provider.validate_token(token)

        decision = await self.authorize(
            token,
            action=permission,
            resource_type=resource_type,
            resource_id=resource_id,
            context=context
        )

        if not decision.is_permitted():
            raise PermissionDeniedError(
                action=permission,
                resource=f"{resource_type}/{resource_id}",
                reason=decision.reason
            )

        return token_info

    def get_recent_audit_entries(
        self,
        limit: int = 100,
        subject: Optional[str] = None,
        action: Optional[str] = None,
        decision: Optional[DecisionResult] = None
    ) -> List[AuditEntry]:
        """
        Get recent audit entries with optional filtering.

        Args:
            limit: Maximum entries to return
            subject: Filter by subject
            action: Filter by action
            decision: Filter by decision result

        Returns:
            List of matching audit entries
        """
        entries = self._audit_entries[-limit:]

        if subject:
            entries = [e for e in entries if e.subject == subject]
        if action:
            entries = [e for e in entries if e.action == action]
        if decision:
            entries = [e for e in entries if e.decision == decision]

        return entries

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get RBAC manager metrics.

        Returns:
            Dictionary of metrics
        """
        total_decisions = self._permit_count + self._deny_count
        permit_rate = (
            self._permit_count / total_decisions * 100
            if total_decisions > 0 else 0
        )

        return {
            "authorization_count": self._authorization_count,
            "permit_count": self._permit_count,
            "deny_count": self._deny_count,
            "permit_rate_percent": round(permit_rate, 2),
            "cache_hits": self._cache_hits,
            "cached_decisions": len(self._decision_cache),
            "audit_entries_count": len(self._audit_entries)
        }

    def clear_cache(self) -> None:
        """Clear the authorization decision cache."""
        self._decision_cache.clear()
        logger.info("Authorization decision cache cleared")


# =============================================================================
# FastAPI Dependencies
# =============================================================================


def create_permission_dependency(
    rbac: RBACManager,
    required_permission: Union[str, Permission]
) -> Callable:
    """
    Create a FastAPI dependency that requires a specific permission.

    Args:
        rbac: RBAC manager instance
        required_permission: Required permission

    Returns:
        FastAPI dependency function

    Example:
        >>> rbac = RBACManager(oauth2_provider)
        >>> require_admin = create_permission_dependency(rbac, Permission.ALL)
        >>>
        >>> @app.get("/admin")
        >>> async def admin_endpoint(
        ...     token_info: TokenInfo = Depends(require_admin)
        ... ):
        ...     return {"admin": True}
    """
    try:
        from fastapi import Depends, HTTPException, Request, Security, status
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

        security = HTTPBearer(auto_error=True)

        if isinstance(required_permission, Permission):
            required_permission = required_permission.value

        async def permission_check(
            request: Request,
            credentials: HTTPAuthorizationCredentials = Security(security)
        ) -> TokenInfo:
            """Check required permission."""
            try:
                context = {
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                    "correlation_id": request.headers.get("x-correlation-id")
                }

                token_info = await rbac.require_permission(
                    credentials.credentials,
                    required_permission,
                    context=context
                )
                return token_info

            except PermissionDeniedError as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {e.message}"
                )
            except AuthorizationError as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Authorization failed: {e.message}"
                )
            except OAuth2Error as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Authentication failed: {e.message}",
                    headers={"WWW-Authenticate": "Bearer"}
                )

        return permission_check

    except ImportError:
        logger.warning("FastAPI not available, skipping dependency creation")
        return None


def create_agent_access_dependency(
    rbac: RBACManager,
    action: str = "execute"
) -> Callable:
    """
    Create a FastAPI dependency for agent access control.

    Args:
        rbac: RBAC manager instance
        action: Required action (execute, view, configure)

    Returns:
        FastAPI dependency function
    """
    try:
        from fastapi import Depends, HTTPException, Path, Request, Security, status
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

        security = HTTPBearer(auto_error=True)

        async def agent_access_check(
            request: Request,
            agent_id: str = Path(..., description="Agent identifier"),
            credentials: HTTPAuthorizationCredentials = Security(security)
        ) -> TokenInfo:
            """Check agent access permission."""
            try:
                context = {
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                    "correlation_id": request.headers.get("x-correlation-id")
                }

                decision = await rbac.authorize(
                    credentials.credentials,
                    action=action,
                    resource_type="agent",
                    resource_id=agent_id,
                    context=context
                )

                if not decision.is_permitted():
                    raise PermissionDeniedError(
                        action=action,
                        resource=f"agent/{agent_id}",
                        reason=decision.reason
                    )

                token_info = await rbac.oauth2_provider.validate_token(
                    credentials.credentials
                )
                return token_info

            except PermissionDeniedError as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: {e.message}"
                )
            except OAuth2Error as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Authentication failed: {e.message}",
                    headers={"WWW-Authenticate": "Bearer"}
                )

        return agent_access_check

    except ImportError:
        logger.warning("FastAPI not available, skipping dependency creation")
        return None
