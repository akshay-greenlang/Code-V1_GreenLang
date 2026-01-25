"""
Attribute-Based Access Control (ABAC) Manager for GreenLang
============================================================

TASK-154: ABAC Implementation

This module provides attribute-based access control capabilities,
enabling fine-grained authorization based on subject, resource,
action, and environmental attributes.

Features:
- Attribute-based policy engine
- Context evaluation (time, location, resource state)
- Policy expression language
- Integration with rbac_manager.py
- Policy caching for performance
- Audit logging for policy decisions

Example:
    >>> from greenlang.infrastructure.auth import ABACManager, Policy
    >>> manager = ABACManager()
    >>>
    >>> # Define a policy
    >>> policy = Policy(
    ...     name="business-hours-access",
    ...     description="Allow access only during business hours",
    ...     condition="subject.role == 'operator' AND environment.hour >= 9 AND environment.hour <= 17"
    ... )
    >>> manager.register_policy(policy)
    >>>
    >>> # Evaluate access request
    >>> result = await manager.evaluate(subject, resource, action, context)

Author: GreenLang Security Team
Created: 2025-12-07
"""

import asyncio
import hashlib
import json
import logging
import operator
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("greenlang.security.abac.audit")


# =============================================================================
# Enums and Constants
# =============================================================================


class Effect(str, Enum):
    """Policy effect."""
    PERMIT = "permit"
    DENY = "deny"


class DecisionResult(str, Enum):
    """Authorization decision result."""
    PERMIT = "PERMIT"
    DENY = "DENY"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INDETERMINATE = "INDETERMINATE"


class CombiningAlgorithm(str, Enum):
    """Policy combining algorithms."""
    DENY_OVERRIDES = "deny_overrides"          # Any deny = deny
    PERMIT_OVERRIDES = "permit_overrides"       # Any permit = permit
    FIRST_APPLICABLE = "first_applicable"       # First matching policy decides
    ONLY_ONE_APPLICABLE = "only_one_applicable" # Exactly one policy must match
    DENY_UNLESS_PERMIT = "deny_unless_permit"   # Deny unless explicitly permitted


class AttributeCategory(str, Enum):
    """Attribute categories for ABAC."""
    SUBJECT = "subject"       # Who is making the request
    RESOURCE = "resource"     # What is being accessed
    ACTION = "action"         # What operation is being performed
    ENVIRONMENT = "environment"  # Context like time, location


class OperatorType(str, Enum):
    """Comparison operators."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex


# =============================================================================
# Attribute Models
# =============================================================================


class Attribute(BaseModel):
    """Attribute definition."""
    name: str = Field(..., description="Attribute name")
    category: AttributeCategory = Field(..., description="Attribute category")
    data_type: str = Field(default="string", description="Data type")
    description: str = Field(default="")
    required: bool = Field(default=False)
    default_value: Optional[Any] = Field(default=None)
    allowed_values: Optional[List[Any]] = Field(default=None)


class SubjectAttributes(BaseModel):
    """Subject (user) attributes."""
    user_id: str = Field(..., description="User identifier")
    username: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    roles: List[str] = Field(default_factory=list)
    groups: List[str] = Field(default_factory=list)
    department: Optional[str] = Field(default=None)
    clearance_level: int = Field(default=0)
    tenant_id: Optional[str] = Field(default=None)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_attributes.get(key, default)


class ResourceAttributes(BaseModel):
    """Resource attributes."""
    resource_id: str = Field(..., description="Resource identifier")
    resource_type: str = Field(..., description="Resource type")
    owner_id: Optional[str] = Field(default=None)
    classification: str = Field(default="public")  # public, internal, confidential, secret
    tags: List[str] = Field(default_factory=list)
    state: str = Field(default="active")
    created_at: Optional[datetime] = Field(default=None)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_attributes.get(key, default)


class ActionAttributes(BaseModel):
    """Action attributes."""
    action_id: str = Field(..., description="Action identifier")
    action_type: str = Field(default="read")  # read, write, delete, execute, admin
    requires_mfa: bool = Field(default=False)
    risk_level: str = Field(default="low")  # low, medium, high, critical
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_attributes.get(key, default)


class EnvironmentAttributes(BaseModel):
    """Environment/context attributes."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: Optional[str] = Field(default=None)
    user_agent: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)
    is_internal_network: bool = Field(default=False)
    is_business_hours: bool = Field(default=True)
    correlation_id: Optional[str] = Field(default=None)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)

    @property
    def hour(self) -> int:
        """Get current hour."""
        return self.timestamp.hour

    @property
    def day_of_week(self) -> int:
        """Get day of week (0=Monday, 6=Sunday)."""
        return self.timestamp.weekday()

    @property
    def is_weekend(self) -> bool:
        """Check if it's weekend."""
        return self.day_of_week >= 5

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_attributes.get(key, default)


class AccessRequest(BaseModel):
    """ABAC access request."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    subject: SubjectAttributes = Field(..., description="Subject attributes")
    resource: ResourceAttributes = Field(..., description="Resource attributes")
    action: ActionAttributes = Field(..., description="Action attributes")
    environment: EnvironmentAttributes = Field(
        default_factory=EnvironmentAttributes,
        description="Environment attributes"
    )

    def get_attribute(self, path: str) -> Any:
        """
        Get attribute by path (e.g., 'subject.roles', 'resource.classification').

        Args:
            path: Dot-separated attribute path

        Returns:
            Attribute value
        """
        parts = path.split(".")
        if len(parts) < 2:
            return None

        category = parts[0]
        attr_path = ".".join(parts[1:])

        if category == "subject":
            obj = self.subject
        elif category == "resource":
            obj = self.resource
        elif category == "action":
            obj = self.action
        elif category == "environment":
            obj = self.environment
        else:
            return None

        # Navigate nested path
        for part in parts[1:]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None

        return obj


# =============================================================================
# Policy Models
# =============================================================================


class Condition(BaseModel):
    """Policy condition."""
    attribute: str = Field(..., description="Attribute path (e.g., 'subject.role')")
    operator: OperatorType = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")

    def evaluate(self, request: AccessRequest) -> bool:
        """
        Evaluate the condition against a request.

        Args:
            request: Access request

        Returns:
            True if condition is satisfied
        """
        attr_value = request.get_attribute(self.attribute)

        if attr_value is None:
            return False

        try:
            if self.operator == OperatorType.EQUALS:
                return attr_value == self.value
            elif self.operator == OperatorType.NOT_EQUALS:
                return attr_value != self.value
            elif self.operator == OperatorType.GREATER_THAN:
                return attr_value > self.value
            elif self.operator == OperatorType.GREATER_THAN_OR_EQUAL:
                return attr_value >= self.value
            elif self.operator == OperatorType.LESS_THAN:
                return attr_value < self.value
            elif self.operator == OperatorType.LESS_THAN_OR_EQUAL:
                return attr_value <= self.value
            elif self.operator == OperatorType.IN:
                return attr_value in self.value
            elif self.operator == OperatorType.NOT_IN:
                return attr_value not in self.value
            elif self.operator == OperatorType.CONTAINS:
                return self.value in attr_value
            elif self.operator == OperatorType.STARTS_WITH:
                return str(attr_value).startswith(str(self.value))
            elif self.operator == OperatorType.ENDS_WITH:
                return str(attr_value).endswith(str(self.value))
            elif self.operator == OperatorType.MATCHES:
                return bool(re.match(str(self.value), str(attr_value)))
            else:
                return False
        except Exception as e:
            logger.warning(f"Condition evaluation error: {e}")
            return False


class Rule(BaseModel):
    """Policy rule combining multiple conditions."""
    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = Field(default="")
    effect: Effect = Field(..., description="Effect when rule matches")
    conditions: List[Condition] = Field(default_factory=list)
    condition_logic: str = Field(default="AND")  # AND, OR
    priority: int = Field(default=0)  # Higher = evaluated first

    def evaluate(self, request: AccessRequest) -> Optional[Effect]:
        """
        Evaluate the rule against a request.

        Args:
            request: Access request

        Returns:
            Effect if rule matches, None otherwise
        """
        if not self.conditions:
            return self.effect

        results = [c.evaluate(request) for c in self.conditions]

        if self.condition_logic == "AND":
            if all(results):
                return self.effect
        elif self.condition_logic == "OR":
            if any(results):
                return self.effect

        return None


class Policy(BaseModel):
    """ABAC policy."""
    policy_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Policy name")
    description: str = Field(default="")
    version: int = Field(default=1)
    effect: Effect = Field(default=Effect.PERMIT)
    target: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Target specifies when policy applies"
    )
    rules: List[Rule] = Field(default_factory=list)
    condition_expression: Optional[str] = Field(
        default=None,
        description="Expression-based condition (alternative to rules)"
    )
    combining_algorithm: CombiningAlgorithm = Field(
        default=CombiningAlgorithm.DENY_OVERRIDES
    )
    priority: int = Field(default=0)
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = Field(default=None)
    tags: List[str] = Field(default_factory=list)

    def matches_target(self, request: AccessRequest) -> bool:
        """
        Check if the policy's target matches the request.

        Args:
            request: Access request

        Returns:
            True if target matches
        """
        if not self.target:
            return True  # Policy applies to all requests

        for attr_path, expected_value in self.target.items():
            actual_value = request.get_attribute(attr_path)

            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False

        return True


class PolicySet(BaseModel):
    """Set of related policies."""
    policy_set_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Policy set name")
    description: str = Field(default="")
    policies: List[Policy] = Field(default_factory=list)
    combining_algorithm: CombiningAlgorithm = Field(
        default=CombiningAlgorithm.DENY_OVERRIDES
    )
    priority: int = Field(default=0)


# =============================================================================
# Decision Models
# =============================================================================


class Obligation(BaseModel):
    """Obligation to be fulfilled on permit."""
    obligation_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Obligation name")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    on_effect: Effect = Field(default=Effect.PERMIT)


class Advice(BaseModel):
    """Advice (non-mandatory obligation)."""
    advice_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Advice name")
    message: str = Field(..., description="Advice message")
    on_effect: Effect = Field(default=Effect.PERMIT)


class PolicyDecision(BaseModel):
    """Result of policy evaluation."""
    decision: DecisionResult = Field(..., description="Authorization decision")
    request_id: str = Field(..., description="Original request ID")
    evaluated_policies: List[str] = Field(
        default_factory=list,
        description="Policies that were evaluated"
    )
    matching_policy: Optional[str] = Field(default=None)
    reason: str = Field(default="")
    obligations: List[Obligation] = Field(default_factory=list)
    advice: List[Advice] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluation_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

    def is_permitted(self) -> bool:
        """Check if decision permits access."""
        return self.decision == DecisionResult.PERMIT

    def get_provenance_hash(self) -> str:
        """Generate provenance hash for audit."""
        data = f"{self.request_id}:{self.decision.value}:{self.evaluated_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()


# =============================================================================
# Expression Engine
# =============================================================================


class ExpressionEngine:
    """
    Evaluates policy condition expressions.

    Supports a simple expression language:
    - Attribute access: subject.role, resource.classification
    - Operators: ==, !=, >, >=, <, <=, in, contains
    - Logic: AND, OR, NOT
    - Functions: now(), date(), time()

    Example:
        subject.role == 'admin' OR (subject.clearance_level >= 3 AND resource.classification in ['public', 'internal'])
    """

    def __init__(self):
        """Initialize the expression engine."""
        self._operators = {
            "==": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
        }

    def evaluate(self, expression: str, request: AccessRequest) -> bool:
        """
        Evaluate an expression against a request.

        Args:
            expression: Condition expression
            request: Access request

        Returns:
            True if expression evaluates to true
        """
        try:
            # Tokenize and parse
            tokens = self._tokenize(expression)
            result = self._evaluate_tokens(tokens, request)
            return bool(result)
        except Exception as e:
            logger.error(f"Expression evaluation error: {e}")
            return False

    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize the expression."""
        # Simple tokenization - split on operators and parentheses
        pattern = r'(\s+|==|!=|>=|<=|>|<|\(|\)|AND|OR|NOT|in|contains)'
        tokens = re.split(pattern, expression)
        return [t.strip() for t in tokens if t.strip()]

    def _evaluate_tokens(self, tokens: List[str], request: AccessRequest) -> bool:
        """Evaluate tokenized expression."""
        # Simple recursive descent parser
        return self._parse_or(tokens, request, 0)[0]

    def _parse_or(
        self,
        tokens: List[str],
        request: AccessRequest,
        pos: int
    ) -> Tuple[bool, int]:
        """Parse OR expressions."""
        left, pos = self._parse_and(tokens, request, pos)

        while pos < len(tokens) and tokens[pos].upper() == "OR":
            pos += 1
            right, pos = self._parse_and(tokens, request, pos)
            left = left or right

        return left, pos

    def _parse_and(
        self,
        tokens: List[str],
        request: AccessRequest,
        pos: int
    ) -> Tuple[bool, int]:
        """Parse AND expressions."""
        left, pos = self._parse_not(tokens, request, pos)

        while pos < len(tokens) and tokens[pos].upper() == "AND":
            pos += 1
            right, pos = self._parse_not(tokens, request, pos)
            left = left and right

        return left, pos

    def _parse_not(
        self,
        tokens: List[str],
        request: AccessRequest,
        pos: int
    ) -> Tuple[bool, int]:
        """Parse NOT expressions."""
        if pos < len(tokens) and tokens[pos].upper() == "NOT":
            pos += 1
            value, pos = self._parse_primary(tokens, request, pos)
            return not value, pos

        return self._parse_primary(tokens, request, pos)

    def _parse_primary(
        self,
        tokens: List[str],
        request: AccessRequest,
        pos: int
    ) -> Tuple[bool, int]:
        """Parse primary expressions (comparisons, parentheses)."""
        if pos >= len(tokens):
            return True, pos

        token = tokens[pos]

        # Parentheses
        if token == "(":
            pos += 1
            result, pos = self._parse_or(tokens, request, pos)
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            return result, pos

        # Comparison: attr op value
        if pos + 2 < len(tokens):
            attr = token
            op = tokens[pos + 1]
            value = tokens[pos + 2]

            # Get attribute value
            attr_value = request.get_attribute(attr)

            # Parse value
            parsed_value = self._parse_value(value)

            # Evaluate comparison
            if op in self._operators:
                result = self._operators[op](attr_value, parsed_value)
            elif op.lower() == "in":
                result = attr_value in parsed_value
            elif op.lower() == "contains":
                result = parsed_value in attr_value if attr_value else False
            else:
                result = False

            return result, pos + 3

        return True, pos + 1

    def _parse_value(self, value: str) -> Any:
        """Parse a value from string."""
        # Remove quotes
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]

        # Try list
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            return [self._parse_value(item.strip()) for item in items]

        # Try number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Try boolean
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        return value


# =============================================================================
# Policy Cache
# =============================================================================


class PolicyCache:
    """Cache for policy decisions."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 10000):
        """Initialize the cache."""
        self._cache: Dict[str, Tuple[PolicyDecision, float]] = {}
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, cache_key: str) -> Optional[PolicyDecision]:
        """Get cached decision."""
        async with self._lock:
            if cache_key in self._cache:
                decision, cached_at = self._cache[cache_key]
                if time.time() - cached_at < self._ttl_seconds:
                    self._hits += 1
                    return decision
                del self._cache[cache_key]
            self._misses += 1
            return None

    async def set(self, cache_key: str, decision: PolicyDecision) -> None:
        """Cache a decision."""
        async with self._lock:
            # Evict if full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[cache_key] = (decision, time.time())

    async def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries."""
        async with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            keys_to_delete = [k for k in self._cache if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2)
        }


# =============================================================================
# ABAC Manager
# =============================================================================


class ABACManager:
    """
    Attribute-Based Access Control Manager for GreenLang.

    Provides fine-grained authorization based on subject, resource,
    action, and environmental attributes.

    Attributes:
        combining_algorithm: Default policy combining algorithm
        cache: Policy decision cache

    Example:
        >>> manager = ABACManager()
        >>>
        >>> # Register a policy
        >>> policy = Policy(
        ...     name="admin-full-access",
        ...     effect=Effect.PERMIT,
        ...     target={"subject.roles": ["admin"]}
        ... )
        >>> manager.register_policy(policy)
        >>>
        >>> # Evaluate request
        >>> request = AccessRequest(
        ...     subject=SubjectAttributes(user_id="user-1", roles=["admin"]),
        ...     resource=ResourceAttributes(resource_id="res-1", resource_type="document"),
        ...     action=ActionAttributes(action_id="read", action_type="read")
        ... )
        >>> decision = await manager.evaluate(request)
    """

    def __init__(
        self,
        combining_algorithm: CombiningAlgorithm = CombiningAlgorithm.DENY_OVERRIDES,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300
    ):
        """
        Initialize ABAC manager.

        Args:
            combining_algorithm: Default policy combining algorithm
            cache_enabled: Whether to enable decision caching
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.combining_algorithm = combining_algorithm
        self._policies: Dict[str, Policy] = {}
        self._policy_sets: Dict[str, PolicySet] = {}
        self._expression_engine = ExpressionEngine()
        self._cache = PolicyCache(ttl_seconds=cache_ttl_seconds) if cache_enabled else None
        self._audit_enabled = True
        self._rbac_manager = None  # Set via integration

        # Metrics
        self._evaluation_count = 0
        self._permit_count = 0
        self._deny_count = 0

        logger.info("ABACManager initialized")

    def set_rbac_manager(self, rbac_manager) -> None:
        """
        Set RBAC manager for integration.

        Args:
            rbac_manager: RBACManager instance
        """
        self._rbac_manager = rbac_manager
        logger.info("RBAC integration enabled")

    def register_policy(self, policy: Policy) -> None:
        """
        Register a policy.

        Args:
            policy: Policy to register
        """
        self._policies[policy.policy_id] = policy
        logger.info(f"Registered policy: {policy.name} ({policy.policy_id})")

    def unregister_policy(self, policy_id: str) -> bool:
        """
        Unregister a policy.

        Args:
            policy_id: Policy identifier

        Returns:
            True if policy was removed
        """
        if policy_id in self._policies:
            del self._policies[policy_id]
            logger.info(f"Unregistered policy: {policy_id}")
            return True
        return False

    def register_policy_set(self, policy_set: PolicySet) -> None:
        """
        Register a policy set.

        Args:
            policy_set: Policy set to register
        """
        self._policy_sets[policy_set.policy_set_id] = policy_set
        logger.info(f"Registered policy set: {policy_set.name}")

    async def evaluate(self, request: AccessRequest) -> PolicyDecision:
        """
        Evaluate an access request against all policies.

        Args:
            request: Access request

        Returns:
            Policy decision
        """
        start_time = time.time()
        self._evaluation_count += 1

        # Check cache
        if self._cache:
            cache_key = self._generate_cache_key(request)
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        # Evaluate policies
        decision = await self._evaluate_policies(request)

        # Update metrics
        if decision.is_permitted():
            self._permit_count += 1
        else:
            self._deny_count += 1

        # Set evaluation time
        decision.evaluation_time_ms = (time.time() - start_time) * 1000
        decision.provenance_hash = decision.get_provenance_hash()

        # Cache decision
        if self._cache:
            await self._cache.set(cache_key, decision)

        # Audit log
        await self._audit_decision(request, decision)

        return decision

    async def _evaluate_policies(self, request: AccessRequest) -> PolicyDecision:
        """Evaluate all applicable policies."""
        applicable_policies = []
        evaluated_policy_ids = []

        # Find applicable policies
        for policy in self._policies.values():
            if not policy.enabled:
                continue

            if policy.matches_target(request):
                applicable_policies.append(policy)
                evaluated_policy_ids.append(policy.policy_id)

        if not applicable_policies:
            return PolicyDecision(
                decision=DecisionResult.NOT_APPLICABLE,
                request_id=request.request_id,
                evaluated_policies=evaluated_policy_ids,
                reason="No applicable policies found"
            )

        # Sort by priority (higher first)
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)

        # Combine decisions
        return self._combine_decisions(
            applicable_policies,
            request,
            evaluated_policy_ids
        )

    def _combine_decisions(
        self,
        policies: List[Policy],
        request: AccessRequest,
        evaluated_ids: List[str]
    ) -> PolicyDecision:
        """Combine policy decisions using combining algorithm."""
        algorithm = self.combining_algorithm

        if algorithm == CombiningAlgorithm.FIRST_APPLICABLE:
            for policy in policies:
                effect = self._evaluate_policy(policy, request)
                if effect is not None:
                    return PolicyDecision(
                        decision=DecisionResult.PERMIT if effect == Effect.PERMIT else DecisionResult.DENY,
                        request_id=request.request_id,
                        evaluated_policies=evaluated_ids,
                        matching_policy=policy.policy_id,
                        reason=f"First applicable policy: {policy.name}"
                    )

        elif algorithm == CombiningAlgorithm.DENY_OVERRIDES:
            permit_found = False
            for policy in policies:
                effect = self._evaluate_policy(policy, request)
                if effect == Effect.DENY:
                    return PolicyDecision(
                        decision=DecisionResult.DENY,
                        request_id=request.request_id,
                        evaluated_policies=evaluated_ids,
                        matching_policy=policy.policy_id,
                        reason=f"Denied by policy: {policy.name}"
                    )
                if effect == Effect.PERMIT:
                    permit_found = True

            if permit_found:
                return PolicyDecision(
                    decision=DecisionResult.PERMIT,
                    request_id=request.request_id,
                    evaluated_policies=evaluated_ids,
                    reason="Permitted by deny-overrides algorithm"
                )

        elif algorithm == CombiningAlgorithm.PERMIT_OVERRIDES:
            deny_found = False
            for policy in policies:
                effect = self._evaluate_policy(policy, request)
                if effect == Effect.PERMIT:
                    return PolicyDecision(
                        decision=DecisionResult.PERMIT,
                        request_id=request.request_id,
                        evaluated_policies=evaluated_ids,
                        matching_policy=policy.policy_id,
                        reason=f"Permitted by policy: {policy.name}"
                    )
                if effect == Effect.DENY:
                    deny_found = True

            if deny_found:
                return PolicyDecision(
                    decision=DecisionResult.DENY,
                    request_id=request.request_id,
                    evaluated_policies=evaluated_ids,
                    reason="Denied by permit-overrides algorithm"
                )

        elif algorithm == CombiningAlgorithm.DENY_UNLESS_PERMIT:
            for policy in policies:
                effect = self._evaluate_policy(policy, request)
                if effect == Effect.PERMIT:
                    return PolicyDecision(
                        decision=DecisionResult.PERMIT,
                        request_id=request.request_id,
                        evaluated_policies=evaluated_ids,
                        matching_policy=policy.policy_id,
                        reason=f"Permitted by policy: {policy.name}"
                    )

            return PolicyDecision(
                decision=DecisionResult.DENY,
                request_id=request.request_id,
                evaluated_policies=evaluated_ids,
                reason="Denied by default (deny-unless-permit)"
            )

        # Default: NOT_APPLICABLE
        return PolicyDecision(
            decision=DecisionResult.NOT_APPLICABLE,
            request_id=request.request_id,
            evaluated_policies=evaluated_ids,
            reason="No applicable decision"
        )

    def _evaluate_policy(self, policy: Policy, request: AccessRequest) -> Optional[Effect]:
        """Evaluate a single policy."""
        # Expression-based evaluation
        if policy.condition_expression:
            if self._expression_engine.evaluate(policy.condition_expression, request):
                return policy.effect
            return None

        # Rule-based evaluation
        if policy.rules:
            # Sort rules by priority
            sorted_rules = sorted(policy.rules, key=lambda r: r.priority, reverse=True)

            for rule in sorted_rules:
                effect = rule.evaluate(request)
                if effect is not None:
                    return effect

            return None

        # No conditions - policy effect applies
        return policy.effect

    def _generate_cache_key(self, request: AccessRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            "subject_id": request.subject.user_id,
            "subject_roles": sorted(request.subject.roles),
            "resource_id": request.resource.resource_id,
            "resource_type": request.resource.resource_type,
            "action_id": request.action.action_id,
            "action_type": request.action.action_type,
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:32]

    async def _audit_decision(
        self,
        request: AccessRequest,
        decision: PolicyDecision
    ) -> None:
        """Log decision for audit."""
        if not self._audit_enabled:
            return

        audit_entry = {
            "timestamp": decision.evaluated_at.isoformat(),
            "request_id": request.request_id,
            "subject_id": request.subject.user_id,
            "subject_roles": request.subject.roles,
            "resource_id": request.resource.resource_id,
            "resource_type": request.resource.resource_type,
            "action": request.action.action_id,
            "decision": decision.decision.value,
            "matching_policy": decision.matching_policy,
            "reason": decision.reason,
            "evaluation_time_ms": decision.evaluation_time_ms,
            "provenance_hash": decision.provenance_hash
        }

        audit_logger.info(json.dumps(audit_entry))

    async def can_access(
        self,
        subject: SubjectAttributes,
        resource: ResourceAttributes,
        action: ActionAttributes,
        environment: Optional[EnvironmentAttributes] = None
    ) -> bool:
        """
        Convenience method to check access.

        Args:
            subject: Subject attributes
            resource: Resource attributes
            action: Action attributes
            environment: Environment attributes

        Returns:
            True if access is permitted
        """
        request = AccessRequest(
            subject=subject,
            resource=resource,
            action=action,
            environment=environment or EnvironmentAttributes()
        )

        decision = await self.evaluate(request)
        return decision.is_permitted()

    def get_policies(
        self,
        enabled_only: bool = True,
        tags: Optional[List[str]] = None
    ) -> List[Policy]:
        """
        Get registered policies.

        Args:
            enabled_only: Only return enabled policies
            tags: Filter by tags

        Returns:
            List of policies
        """
        policies = list(self._policies.values())

        if enabled_only:
            policies = [p for p in policies if p.enabled]

        if tags:
            policies = [
                p for p in policies
                if any(tag in p.tags for tag in tags)
            ]

        return policies

    def get_metrics(self) -> Dict[str, Any]:
        """Get ABAC metrics."""
        total = self._permit_count + self._deny_count
        permit_rate = (self._permit_count / total * 100) if total > 0 else 0

        cache_stats = self._cache.get_stats() if self._cache else {}

        return {
            "evaluation_count": self._evaluation_count,
            "permit_count": self._permit_count,
            "deny_count": self._deny_count,
            "permit_rate_percent": round(permit_rate, 2),
            "policy_count": len(self._policies),
            "policy_set_count": len(self._policy_sets),
            "cache": cache_stats
        }

    async def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries."""
        if self._cache:
            return await self._cache.invalidate(pattern)
        return 0


# =============================================================================
# FastAPI Router
# =============================================================================


def create_abac_router(manager: ABACManager):
    """
    Create FastAPI router for ABAC management.

    Args:
        manager: ABACManager instance

    Returns:
        FastAPI APIRouter
    """
    try:
        from fastapi import APIRouter, HTTPException, Query, status
    except ImportError:
        logger.warning("FastAPI not available, skipping router creation")
        return None

    router = APIRouter(prefix="/api/v1/abac", tags=["ABAC"])

    @router.post("/evaluate")
    async def evaluate_request(request: AccessRequest):
        """Evaluate an access request."""
        decision = await manager.evaluate(request)
        return decision.dict()

    @router.get("/policies")
    async def get_policies(
        enabled_only: bool = Query(True),
        tags: Optional[List[str]] = Query(None)
    ):
        """Get registered policies."""
        policies = manager.get_policies(enabled_only, tags)
        return {"policies": [p.dict() for p in policies]}

    @router.post("/policies")
    async def register_policy(policy: Policy):
        """Register a new policy."""
        manager.register_policy(policy)
        return {"message": "Policy registered", "policy_id": policy.policy_id}

    @router.delete("/policies/{policy_id}")
    async def delete_policy(policy_id: str):
        """Delete a policy."""
        success = manager.unregister_policy(policy_id)
        if not success:
            raise HTTPException(status_code=404, detail="Policy not found")
        return {"message": "Policy deleted"}

    @router.get("/metrics")
    async def get_metrics():
        """Get ABAC metrics."""
        return manager.get_metrics()

    @router.post("/cache/invalidate")
    async def invalidate_cache(pattern: Optional[str] = Query(None)):
        """Invalidate cache entries."""
        count = await manager.invalidate_cache(pattern)
        return {"invalidated_count": count}

    return router
