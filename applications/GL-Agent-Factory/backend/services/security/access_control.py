"""
Access Control Service - Attribute-Based Access Control (ABAC)

This module provides a comprehensive attribute-based access control system
with support for time-based restrictions, IP allowlisting, device trust,
and session management with configurable timeouts.

SOC2 Controls Addressed:
    - CC6.1: Logical access security
    - CC6.2: Access restriction based on role
    - CC6.3: Role-based access control

ISO27001 Controls Addressed:
    - A.9.1.1: Access control policy
    - A.9.2.1: User registration and de-registration
    - A.9.4.1: Information access restriction

Example:
    >>> service = AccessControlService(config)
    >>> decision = await service.authorize(
    ...     context=AccessContext(
    ...         user_id="user-123",
    ...         resource_type="EmissionsReport",
    ...         action="READ",
    ...     )
    ... )
    >>> if decision.allowed:
    ...     print("Access granted")
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import uuid
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Action(str, Enum):
    """Standard CRUD actions for access control."""

    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXECUTE = "EXECUTE"
    APPROVE = "APPROVE"
    EXPORT = "EXPORT"
    ADMIN = "ADMIN"


class AccessDecisionResult(str, Enum):
    """Possible outcomes of an access decision."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    INDETERMINATE = "INDETERMINATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PolicyEffect(str, Enum):
    """Effect of a policy rule."""

    ALLOW = "ALLOW"
    DENY = "DENY"


class CombiningAlgorithm(str, Enum):
    """Algorithms for combining multiple policy decisions."""

    DENY_OVERRIDES = "DENY_OVERRIDES"  # Any deny = deny
    PERMIT_OVERRIDES = "PERMIT_OVERRIDES"  # Any permit = permit
    FIRST_APPLICABLE = "FIRST_APPLICABLE"  # Use first matching policy
    ONLY_ONE_APPLICABLE = "ONLY_ONE_APPLICABLE"  # Error if multiple match


class DeviceTrustLevel(str, Enum):
    """Trust levels for device-based access control."""

    UNTRUSTED = "UNTRUSTED"
    KNOWN = "KNOWN"
    TRUSTED = "TRUSTED"
    MANAGED = "MANAGED"  # Corporate-managed device


class SessionStatus(str, Enum):
    """Session lifecycle states."""

    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    LOCKED = "LOCKED"


class TimeWindow(BaseModel):
    """
    Time-based access restriction window.

    Defines when access is allowed based on day of week and time of day.
    """

    days: List[int] = Field(
        default=[0, 1, 2, 3, 4],  # Monday through Friday
        description="Days of week (0=Monday, 6=Sunday)",
    )
    start_time: time = Field(default=time(8, 0), description="Start of allowed window")
    end_time: time = Field(default=time(18, 0), description="End of allowed window")
    timezone: str = Field(default="UTC")

    def is_within_window(self, check_time: Optional[datetime] = None) -> bool:
        """Check if the given time falls within the access window."""
        if check_time is None:
            check_time = datetime.now(timezone.utc)

        # Check day of week
        if check_time.weekday() not in self.days:
            return False

        # Check time of day
        current_time = check_time.time()
        return self.start_time <= current_time <= self.end_time


class IPRestriction(BaseModel):
    """
    IP-based access restriction configuration.

    Supports individual IPs, CIDR ranges, and named allowlists.
    """

    # Allowlist mode
    allowlist_enabled: bool = Field(default=False)
    allowed_ips: List[str] = Field(default_factory=list)
    allowed_cidrs: List[str] = Field(default_factory=list)

    # Blocklist mode
    blocklist_enabled: bool = Field(default=False)
    blocked_ips: List[str] = Field(default_factory=list)
    blocked_cidrs: List[str] = Field(default_factory=list)

    # Geographic restrictions
    allowed_countries: List[str] = Field(default_factory=list)
    blocked_countries: List[str] = Field(default_factory=list)

    def is_allowed(self, ip_address: str) -> bool:
        """
        Check if an IP address is allowed.

        Args:
            ip_address: IP address to check

        Returns:
            True if allowed, False if blocked
        """
        try:
            ip = ipaddress.ip_address(ip_address)
        except ValueError:
            logger.warning(f"Invalid IP address: {ip_address}")
            return False

        # Check blocklist first (deny takes precedence)
        if self.blocklist_enabled:
            if ip_address in self.blocked_ips:
                return False
            for cidr in self.blocked_cidrs:
                try:
                    if ip in ipaddress.ip_network(cidr, strict=False):
                        return False
                except ValueError:
                    continue

        # Check allowlist
        if self.allowlist_enabled:
            # If allowlist is enabled, IP must be in allowlist
            if ip_address in self.allowed_ips:
                return True
            for cidr in self.allowed_cidrs:
                try:
                    if ip in ipaddress.ip_network(cidr, strict=False):
                        return True
                except ValueError:
                    continue
            # Not in allowlist
            return False

        # No restrictions, allow by default
        return True


class DeviceTrustPolicy(BaseModel):
    """
    Device trust policy for access control.

    Defines minimum trust requirements for accessing resources.
    """

    minimum_trust_level: DeviceTrustLevel = Field(default=DeviceTrustLevel.KNOWN)
    require_mfa: bool = Field(default=False)
    require_certificate: bool = Field(default=False)
    allowed_device_types: List[str] = Field(
        default_factory=lambda: ["desktop", "mobile", "tablet"]
    )
    require_encryption: bool = Field(default=False)
    max_device_age_days: Optional[int] = Field(default=None)


class SessionConfig(BaseModel):
    """
    Session management configuration.

    Defines timeout and security settings for user sessions.
    """

    # Timeout settings
    idle_timeout_minutes: int = Field(default=30, ge=5, le=480)
    absolute_timeout_hours: int = Field(default=8, ge=1, le=24)
    max_concurrent_sessions: int = Field(default=3, ge=1, le=10)

    # Security settings
    require_mfa_reauthentication: bool = Field(default=False)
    mfa_reauthentication_interval_hours: int = Field(default=4)
    invalidate_on_ip_change: bool = Field(default=False)
    invalidate_on_device_change: bool = Field(default=True)

    # Token settings
    refresh_token_enabled: bool = Field(default=True)
    refresh_token_rotation: bool = Field(default=True)


class Session(BaseModel):
    """
    User session with security metadata.

    Tracks session state for timeout and security enforcement.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    tenant_id: str
    status: SessionStatus = Field(default=SessionStatus.ACTIVE)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime

    # Security context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_id: Optional[str] = None
    device_trust_level: DeviceTrustLevel = Field(default=DeviceTrustLevel.UNKNOWN)
    mfa_verified: bool = Field(default=False)
    mfa_verified_at: Optional[datetime] = None

    # Token tracking
    access_token_hash: Optional[str] = None
    refresh_token_hash: Optional[str] = None

    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.status == SessionStatus.ACTIVE

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def is_idle(self, idle_timeout_minutes: int) -> bool:
        """Check if session is idle."""
        idle_threshold = datetime.now(timezone.utc) - timedelta(minutes=idle_timeout_minutes)
        return self.last_activity < idle_threshold


class AccessContext(BaseModel):
    """
    Context for an access control decision.

    Contains all attributes needed to evaluate access policies.
    """

    # Subject attributes (who)
    user_id: str = Field(..., description="User requesting access")
    tenant_id: str = Field(..., description="Organization tenant ID")
    roles: List[str] = Field(default_factory=list, description="User roles")
    groups: List[str] = Field(default_factory=list, description="User groups")
    department: Optional[str] = Field(None)
    job_title: Optional[str] = Field(None)

    # Resource attributes (what)
    resource_type: str = Field(..., description="Type of resource")
    resource_id: Optional[str] = Field(None, description="Specific resource ID")
    resource_owner: Optional[str] = Field(None)
    resource_classification: Optional[str] = Field(None)
    resource_tags: Dict[str, str] = Field(default_factory=dict)

    # Action attributes (how)
    action: Action = Field(..., description="Requested action")

    # Environment attributes (when/where)
    request_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: Optional[str] = Field(None)
    user_agent: Optional[str] = Field(None)
    device_id: Optional[str] = Field(None)
    device_trust_level: DeviceTrustLevel = Field(default=DeviceTrustLevel.UNTRUSTED)
    session_id: Optional[str] = Field(None)
    mfa_verified: bool = Field(default=False)

    # Custom attributes for flexible policies
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)


class PolicyCondition(BaseModel):
    """
    A condition that must be met for a policy to apply.

    Supports various comparison operators and value types.
    """

    attribute: str = Field(..., description="Attribute path to evaluate")
    operator: str = Field(
        ...,
        description="Comparison operator (eq, ne, gt, lt, in, contains, matches)",
    )
    value: Any = Field(..., description="Value to compare against")

    def evaluate(self, context: AccessContext) -> bool:
        """
        Evaluate this condition against the access context.

        Args:
            context: Access context to evaluate

        Returns:
            True if condition is met, False otherwise
        """
        # Get attribute value from context
        attr_value = self._get_attribute_value(context, self.attribute)

        # Evaluate based on operator
        if self.operator == "eq":
            return attr_value == self.value
        elif self.operator == "ne":
            return attr_value != self.value
        elif self.operator == "gt":
            return attr_value is not None and attr_value > self.value
        elif self.operator == "lt":
            return attr_value is not None and attr_value < self.value
        elif self.operator == "gte":
            return attr_value is not None and attr_value >= self.value
        elif self.operator == "lte":
            return attr_value is not None and attr_value <= self.value
        elif self.operator == "in":
            return attr_value in self.value
        elif self.operator == "not_in":
            return attr_value not in self.value
        elif self.operator == "contains":
            return self.value in attr_value if attr_value else False
        elif self.operator == "matches":
            return bool(re.match(self.value, str(attr_value))) if attr_value else False
        elif self.operator == "exists":
            return attr_value is not None
        elif self.operator == "not_exists":
            return attr_value is None
        else:
            logger.warning(f"Unknown operator: {self.operator}")
            return False

    def _get_attribute_value(self, context: AccessContext, path: str) -> Any:
        """
        Get attribute value from context using dot notation path.

        Args:
            context: Access context
            path: Attribute path (e.g., "user.department" or "resource_tags.env")

        Returns:
            Attribute value or None if not found
        """
        parts = path.split(".")
        value = context.dict()

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value


class PolicyRule(BaseModel):
    """
    A single rule within an access policy.

    Combines conditions with an effect (allow/deny).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable rule name")
    description: Optional[str] = Field(None)
    effect: PolicyEffect = Field(..., description="Effect if rule matches")
    conditions: List[PolicyCondition] = Field(default_factory=list)
    priority: int = Field(default=0, description="Rule priority (higher = more important)")

    def evaluate(self, context: AccessContext) -> Optional[PolicyEffect]:
        """
        Evaluate this rule against the access context.

        Args:
            context: Access context to evaluate

        Returns:
            PolicyEffect if all conditions match, None otherwise
        """
        # All conditions must be true for rule to apply
        for condition in self.conditions:
            if not condition.evaluate(context):
                return None

        return self.effect


class AccessPolicy(BaseModel):
    """
    Access control policy with multiple rules.

    Policies are evaluated against access contexts to make authorization decisions.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Policy name")
    description: Optional[str] = Field(None)
    version: str = Field(default="1.0.0")
    enabled: bool = Field(default=True)

    # Targeting
    target_tenants: List[str] = Field(
        default_factory=list,
        description="Tenants this policy applies to (empty = all)",
    )
    target_resource_types: List[str] = Field(
        default_factory=list,
        description="Resource types this policy applies to (empty = all)",
    )
    target_actions: List[Action] = Field(
        default_factory=list,
        description="Actions this policy applies to (empty = all)",
    )

    # Rules
    rules: List[PolicyRule] = Field(default_factory=list)
    combining_algorithm: CombiningAlgorithm = Field(default=CombiningAlgorithm.DENY_OVERRIDES)

    # Additional restrictions
    time_restrictions: Optional[TimeWindow] = Field(None)
    ip_restrictions: Optional[IPRestriction] = Field(None)
    device_policy: Optional[DeviceTrustPolicy] = Field(None)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = Field(None)

    def applies_to(self, context: AccessContext) -> bool:
        """
        Check if this policy applies to the given context.

        Args:
            context: Access context to check

        Returns:
            True if policy should be evaluated for this context
        """
        if not self.enabled:
            return False

        # Check tenant targeting
        if self.target_tenants and context.tenant_id not in self.target_tenants:
            return False

        # Check resource type targeting
        if self.target_resource_types and context.resource_type not in self.target_resource_types:
            return False

        # Check action targeting
        if self.target_actions and context.action not in self.target_actions:
            return False

        return True

    def evaluate(self, context: AccessContext) -> AccessDecisionResult:
        """
        Evaluate this policy against the access context.

        Args:
            context: Access context to evaluate

        Returns:
            Access decision result
        """
        if not self.applies_to(context):
            return AccessDecisionResult.NOT_APPLICABLE

        # Check time restrictions
        if self.time_restrictions and not self.time_restrictions.is_within_window(context.request_time):
            return AccessDecisionResult.DENY

        # Check IP restrictions
        if self.ip_restrictions and context.ip_address:
            if not self.ip_restrictions.is_allowed(context.ip_address):
                return AccessDecisionResult.DENY

        # Check device policy
        if self.device_policy:
            if context.device_trust_level.value < self.device_policy.minimum_trust_level.value:
                return AccessDecisionResult.DENY
            if self.device_policy.require_mfa and not context.mfa_verified:
                return AccessDecisionResult.DENY

        # Evaluate rules
        effects = []
        for rule in sorted(self.rules, key=lambda r: -r.priority):
            effect = rule.evaluate(context)
            if effect is not None:
                effects.append(effect)

        if not effects:
            return AccessDecisionResult.NOT_APPLICABLE

        # Combine effects based on algorithm
        return self._combine_effects(effects)

    def _combine_effects(self, effects: List[PolicyEffect]) -> AccessDecisionResult:
        """Combine multiple rule effects into a single decision."""
        if self.combining_algorithm == CombiningAlgorithm.DENY_OVERRIDES:
            if PolicyEffect.DENY in effects:
                return AccessDecisionResult.DENY
            if PolicyEffect.ALLOW in effects:
                return AccessDecisionResult.ALLOW
            return AccessDecisionResult.INDETERMINATE

        elif self.combining_algorithm == CombiningAlgorithm.PERMIT_OVERRIDES:
            if PolicyEffect.ALLOW in effects:
                return AccessDecisionResult.ALLOW
            if PolicyEffect.DENY in effects:
                return AccessDecisionResult.DENY
            return AccessDecisionResult.INDETERMINATE

        elif self.combining_algorithm == CombiningAlgorithm.FIRST_APPLICABLE:
            return (
                AccessDecisionResult.ALLOW
                if effects[0] == PolicyEffect.ALLOW
                else AccessDecisionResult.DENY
            )

        elif self.combining_algorithm == CombiningAlgorithm.ONLY_ONE_APPLICABLE:
            if len(effects) > 1:
                return AccessDecisionResult.INDETERMINATE
            return (
                AccessDecisionResult.ALLOW
                if effects[0] == PolicyEffect.ALLOW
                else AccessDecisionResult.DENY
            )

        return AccessDecisionResult.INDETERMINATE


class AccessDecision(BaseModel):
    """
    Result of an access control decision with audit information.

    Includes the decision outcome and all metadata needed for compliance auditing.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    allowed: bool = Field(..., description="Whether access is granted")
    result: AccessDecisionResult = Field(..., description="Detailed result")

    # Context
    context: AccessContext
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Policy information
    matching_policies: List[str] = Field(default_factory=list)
    denying_policies: List[str] = Field(default_factory=list)
    allowing_policies: List[str] = Field(default_factory=list)

    # Denial reasons for debugging/audit
    denial_reasons: List[str] = Field(default_factory=list)

    # Processing metadata
    processing_time_ms: float = Field(default=0.0)


class AccessControlConfig(BaseModel):
    """Configuration for the Access Control Service."""

    # Default behavior
    default_deny: bool = Field(default=True, description="Deny access if no policy matches")

    # Session management
    session_config: SessionConfig = Field(default_factory=SessionConfig)

    # Caching
    policy_cache_ttl_seconds: int = Field(default=300)
    decision_cache_enabled: bool = Field(default=True)
    decision_cache_ttl_seconds: int = Field(default=60)

    # Audit
    audit_all_decisions: bool = Field(default=True)
    audit_denied_only: bool = Field(default=False)


class AccessControlService:
    """
    Production-grade Attribute-Based Access Control (ABAC) service.

    Provides comprehensive access control with:
    - Attribute-based policies (ABAC)
    - Time-based access restrictions
    - IP allowlisting and blocklisting
    - Device trust policies
    - Session management with timeout

    The service evaluates policies against access contexts to make
    authorization decisions, supporting multiple combining algorithms
    for complex policy scenarios.

    Example:
        >>> config = AccessControlConfig()
        >>> service = AccessControlService(config)
        >>> await service.initialize()
        >>>
        >>> # Register a policy
        >>> policy = AccessPolicy(
        ...     name="Admin Access",
        ...     rules=[
        ...         PolicyRule(
        ...             name="Admins can do anything",
        ...             effect=PolicyEffect.ALLOW,
        ...             conditions=[
        ...                 PolicyCondition(
        ...                     attribute="roles",
        ...                     operator="contains",
        ...                     value="admin"
        ...                 )
        ...             ]
        ...         )
        ...     ]
        ... )
        >>> await service.register_policy(policy)
        >>>
        >>> # Make authorization decision
        >>> context = AccessContext(
        ...     user_id="user-123",
        ...     tenant_id="tenant-456",
        ...     roles=["admin"],
        ...     resource_type="Agent",
        ...     action=Action.DELETE,
        ... )
        >>> decision = await service.authorize(context)
        >>> print(f"Access {'granted' if decision.allowed else 'denied'}")

    Attributes:
        config: Service configuration
        policies: Registered access policies
        sessions: Active user sessions
    """

    def __init__(self, config: Optional[AccessControlConfig] = None):
        """
        Initialize the Access Control Service.

        Args:
            config: Service configuration
        """
        self.config = config or AccessControlConfig()
        self.policies: Dict[str, AccessPolicy] = {}
        self.sessions: Dict[str, Session] = {}
        self._decision_cache: Dict[str, AccessDecision] = {}
        self._initialized = False

        logger.info(
            "AccessControlService initialized",
            extra={
                "default_deny": self.config.default_deny,
                "session_idle_timeout": self.config.session_config.idle_timeout_minutes,
            },
        )

    async def initialize(self) -> None:
        """Initialize the access control service."""
        if self._initialized:
            logger.warning("AccessControlService already initialized")
            return

        # Load policies from storage (placeholder)
        await self._load_policies()

        # Load active sessions (placeholder)
        await self._load_sessions()

        self._initialized = True
        logger.info("AccessControlService initialization complete")

    async def authorize(
        self,
        context: AccessContext,
        skip_cache: bool = False,
    ) -> AccessDecision:
        """
        Make an access control decision.

        Evaluates all applicable policies against the context
        and returns an authorization decision.

        Args:
            context: Access context with all relevant attributes
            skip_cache: Whether to bypass decision cache

        Returns:
            Access decision with audit information
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Check cache first
            if not skip_cache and self.config.decision_cache_enabled:
                cached = self._get_cached_decision(context)
                if cached:
                    logger.debug(f"Cache hit for authorization decision")
                    return cached

            # Validate session if provided
            if context.session_id:
                session_valid = await self._validate_session(context)
                if not session_valid:
                    return self._create_denied_decision(
                        context,
                        start_time,
                        ["Session invalid or expired"],
                    )

            # Evaluate all applicable policies
            matching_policies = []
            allowing_policies = []
            denying_policies = []
            denial_reasons = []

            for policy in self.policies.values():
                if not policy.applies_to(context):
                    continue

                matching_policies.append(policy.id)
                result = policy.evaluate(context)

                if result == AccessDecisionResult.ALLOW:
                    allowing_policies.append(policy.id)
                elif result == AccessDecisionResult.DENY:
                    denying_policies.append(policy.id)
                    denial_reasons.append(f"Policy '{policy.name}' denied access")

            # Determine final decision
            if denying_policies:
                # Deny takes precedence
                decision = self._create_denied_decision(
                    context,
                    start_time,
                    denial_reasons,
                    matching_policies,
                    allowing_policies,
                    denying_policies,
                )
            elif allowing_policies:
                decision = self._create_allowed_decision(
                    context,
                    start_time,
                    matching_policies,
                    allowing_policies,
                )
            elif self.config.default_deny:
                decision = self._create_denied_decision(
                    context,
                    start_time,
                    ["No policy granted access (default deny)"],
                    matching_policies,
                )
            else:
                decision = self._create_allowed_decision(
                    context,
                    start_time,
                    matching_policies,
                )

            # Cache the decision
            if self.config.decision_cache_enabled:
                self._cache_decision(context, decision)

            # Log the decision
            log_level = logging.INFO if decision.allowed else logging.WARNING
            logger.log(
                log_level,
                f"Access {'granted' if decision.allowed else 'denied'}: "
                f"{context.action.value} on {context.resource_type}",
                extra={
                    "user_id": context.user_id,
                    "tenant_id": context.tenant_id,
                    "resource_type": context.resource_type,
                    "action": context.action.value,
                    "allowed": decision.allowed,
                    "matching_policies": len(matching_policies),
                },
            )

            return decision

        except Exception as e:
            logger.error(f"Authorization error: {e}", exc_info=True)
            # Fail closed on errors
            return self._create_denied_decision(
                context,
                start_time,
                [f"Authorization error: {str(e)}"],
            )

    async def register_policy(self, policy: AccessPolicy) -> None:
        """
        Register an access control policy.

        Args:
            policy: Policy to register
        """
        self.policies[policy.id] = policy
        self._decision_cache.clear()  # Invalidate cache

        logger.info(
            f"Policy registered: {policy.name}",
            extra={
                "policy_id": policy.id,
                "rule_count": len(policy.rules),
            },
        )

    async def unregister_policy(self, policy_id: str) -> None:
        """
        Unregister an access control policy.

        Args:
            policy_id: ID of policy to remove
        """
        if policy_id in self.policies:
            policy = self.policies.pop(policy_id)
            self._decision_cache.clear()

            logger.info(
                f"Policy unregistered: {policy.name}",
                extra={"policy_id": policy_id},
            )

    async def create_session(
        self,
        user_id: str,
        tenant_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
        device_trust_level: DeviceTrustLevel = DeviceTrustLevel.UNTRUSTED,
        mfa_verified: bool = False,
    ) -> Session:
        """
        Create a new user session.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            ip_address: Client IP address
            user_agent: Client user agent
            device_id: Device identifier
            device_trust_level: Trust level of the device
            mfa_verified: Whether MFA was verified

        Returns:
            Created session
        """
        # Check concurrent session limit
        user_sessions = [
            s for s in self.sessions.values()
            if s.user_id == user_id and s.is_active()
        ]
        if len(user_sessions) >= self.config.session_config.max_concurrent_sessions:
            # Revoke oldest session
            oldest = min(user_sessions, key=lambda s: s.created_at)
            await self.revoke_session(oldest.id)

        # Calculate expiration
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=self.config.session_config.absolute_timeout_hours
        )

        session = Session(
            user_id=user_id,
            tenant_id=tenant_id,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            device_id=device_id,
            device_trust_level=device_trust_level,
            mfa_verified=mfa_verified,
            mfa_verified_at=datetime.now(timezone.utc) if mfa_verified else None,
        )

        self.sessions[session.id] = session

        logger.info(
            f"Session created for user {user_id}",
            extra={
                "session_id": session.id,
                "expires_at": expires_at.isoformat(),
            },
        )

        return session

    async def validate_session(self, session_id: str) -> bool:
        """
        Validate a session is still active.

        Args:
            session_id: Session to validate

        Returns:
            True if session is valid, False otherwise
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check if expired
        if session.is_expired():
            session.status = SessionStatus.EXPIRED
            return False

        # Check if idle
        if session.is_idle(self.config.session_config.idle_timeout_minutes):
            session.status = SessionStatus.IDLE
            return False

        # Check if active
        if not session.is_active():
            return False

        return True

    async def refresh_session(self, session_id: str) -> Optional[Session]:
        """
        Refresh a session's activity timestamp.

        Args:
            session_id: Session to refresh

        Returns:
            Updated session or None if invalid
        """
        session = self.sessions.get(session_id)
        if not session or not await self.validate_session(session_id):
            return None

        session.last_activity = datetime.now(timezone.utc)
        return session

    async def revoke_session(self, session_id: str) -> bool:
        """
        Revoke a session.

        Args:
            session_id: Session to revoke

        Returns:
            True if session was revoked
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.status = SessionStatus.REVOKED

        logger.info(
            f"Session revoked: {session_id}",
            extra={"user_id": session.user_id},
        )

        return True

    async def revoke_all_sessions(self, user_id: str) -> int:
        """
        Revoke all sessions for a user.

        Args:
            user_id: User whose sessions should be revoked

        Returns:
            Number of sessions revoked
        """
        count = 0
        for session in self.sessions.values():
            if session.user_id == user_id and session.is_active():
                session.status = SessionStatus.REVOKED
                count += 1

        logger.info(
            f"All sessions revoked for user {user_id}",
            extra={"count": count},
        )

        return count

    def _create_allowed_decision(
        self,
        context: AccessContext,
        start_time: datetime,
        matching_policies: List[str] = None,
        allowing_policies: List[str] = None,
    ) -> AccessDecision:
        """Create an allowed access decision."""
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return AccessDecision(
            allowed=True,
            result=AccessDecisionResult.ALLOW,
            context=context,
            matching_policies=matching_policies or [],
            allowing_policies=allowing_policies or [],
            processing_time_ms=processing_time,
        )

    def _create_denied_decision(
        self,
        context: AccessContext,
        start_time: datetime,
        denial_reasons: List[str],
        matching_policies: List[str] = None,
        allowing_policies: List[str] = None,
        denying_policies: List[str] = None,
    ) -> AccessDecision:
        """Create a denied access decision."""
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return AccessDecision(
            allowed=False,
            result=AccessDecisionResult.DENY,
            context=context,
            matching_policies=matching_policies or [],
            allowing_policies=allowing_policies or [],
            denying_policies=denying_policies or [],
            denial_reasons=denial_reasons,
            processing_time_ms=processing_time,
        )

    async def _validate_session(self, context: AccessContext) -> bool:
        """Validate session from access context."""
        if not context.session_id:
            return True  # No session to validate

        session = self.sessions.get(context.session_id)
        if not session:
            return False

        # Validate session is active
        if not await self.validate_session(context.session_id):
            return False

        # Check IP change if configured
        if self.config.session_config.invalidate_on_ip_change:
            if context.ip_address and session.ip_address != context.ip_address:
                logger.warning(
                    f"IP address changed for session {context.session_id}",
                    extra={
                        "original_ip": session.ip_address,
                        "new_ip": context.ip_address,
                    },
                )
                await self.revoke_session(context.session_id)
                return False

        # Check device change if configured
        if self.config.session_config.invalidate_on_device_change:
            if context.device_id and session.device_id != context.device_id:
                logger.warning(f"Device changed for session {context.session_id}")
                await self.revoke_session(context.session_id)
                return False

        # Update last activity
        await self.refresh_session(context.session_id)
        return True

    def _get_cache_key(self, context: AccessContext) -> str:
        """Generate cache key for access decision."""
        key_data = f"{context.user_id}:{context.tenant_id}:{context.resource_type}:{context.action.value}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_cached_decision(self, context: AccessContext) -> Optional[AccessDecision]:
        """Get cached access decision if available."""
        key = self._get_cache_key(context)
        cached = self._decision_cache.get(key)

        if cached:
            # Check if cache entry is still valid
            age = (datetime.now(timezone.utc) - cached.evaluated_at).total_seconds()
            if age < self.config.decision_cache_ttl_seconds:
                return cached
            else:
                del self._decision_cache[key]

        return None

    def _cache_decision(self, context: AccessContext, decision: AccessDecision) -> None:
        """Cache an access decision."""
        key = self._get_cache_key(context)
        self._decision_cache[key] = decision

    async def _load_policies(self) -> None:
        """Load policies from storage (placeholder)."""
        # Would load from database in production
        pass

    async def _load_sessions(self) -> None:
        """Load active sessions from storage (placeholder)."""
        # Would load from Redis in production
        pass


# Default policies for common scenarios
def create_admin_policy() -> AccessPolicy:
    """Create a default admin access policy."""
    return AccessPolicy(
        name="Admin Full Access",
        description="Grants full access to users with admin role",
        rules=[
            PolicyRule(
                name="Admin allow all",
                effect=PolicyEffect.ALLOW,
                conditions=[
                    PolicyCondition(attribute="roles", operator="contains", value="admin")
                ],
                priority=100,
            )
        ],
    )


def create_read_only_policy(resource_type: str) -> AccessPolicy:
    """Create a read-only access policy for a resource type."""
    return AccessPolicy(
        name=f"Read-Only {resource_type} Access",
        description=f"Grants read access to {resource_type} resources",
        target_resource_types=[resource_type],
        target_actions=[Action.READ],
        rules=[
            PolicyRule(
                name="Allow read",
                effect=PolicyEffect.ALLOW,
                conditions=[],
            )
        ],
    )


def create_owner_policy() -> AccessPolicy:
    """Create a policy that allows owners full access to their resources."""
    return AccessPolicy(
        name="Owner Access",
        description="Grants full access to resource owners",
        rules=[
            PolicyRule(
                name="Owner full access",
                effect=PolicyEffect.ALLOW,
                conditions=[
                    PolicyCondition(
                        attribute="user_id",
                        operator="eq",
                        value="${resource_owner}",  # Dynamic reference
                    )
                ],
                priority=50,
            )
        ],
    )
