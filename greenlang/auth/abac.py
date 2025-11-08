"""
Attribute-Based Access Control (ABAC) for GreenLang

This module provides attribute-based access control with policy evaluation engine.
Supports complex policies based on user, resource, and environment attributes.

ABAC Policy Example:
    {
        "policy_id": "data-export-restriction",
        "effect": "deny",
        "actions": ["export"],
        "resources": ["data:*"],
        "conditions": {
            "user.department": {"eq": "finance"},
            "resource.classification": {"eq": "confidential"},
            "environment.time_of_day": {"between": [9, 17]}
        }
    }

Features:
    - Attribute providers (user, resource, environment)
    - Policy evaluation engine
    - JSON/YAML policy language
    - OPA integration support
    - Policy conflict resolution

Author: GreenLang Framework Team - Phase 4
Date: November 2025
Status: Production Ready
"""

import logging
import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from pathlib import Path
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.auth.permissions import PermissionEffect, PermissionAction

logger = logging.getLogger(__name__)


# ==============================================================================
# Attribute Providers
# ==============================================================================

class AttributeProvider(ABC):
    """Base class for attribute providers."""

    @abstractmethod
    def get_attributes(self, subject_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get attributes for a subject.

        Args:
            subject_id: Subject identifier
            context: Additional context

        Returns:
            Dictionary of attributes
        """
        pass


class UserAttributeProvider(AttributeProvider):
    """Provides user attributes for ABAC evaluation."""

    def __init__(self):
        """Initialize user attribute provider."""
        self._user_attributes: Dict[str, Dict[str, Any]] = {}

    def set_user_attributes(self, user_id: str, attributes: Dict[str, Any]):
        """Set attributes for a user."""
        self._user_attributes[user_id] = attributes

    def get_attributes(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get attributes for a user."""
        return self._user_attributes.get(user_id, {})

    def add_attribute(self, user_id: str, key: str, value: Any):
        """Add or update a single attribute for a user."""
        if user_id not in self._user_attributes:
            self._user_attributes[user_id] = {}
        self._user_attributes[user_id][key] = value


class ResourceAttributeProvider(AttributeProvider):
    """Provides resource attributes for ABAC evaluation."""

    def __init__(self):
        """Initialize resource attribute provider."""
        self._resource_attributes: Dict[str, Dict[str, Any]] = {}

    def set_resource_attributes(self, resource_id: str, attributes: Dict[str, Any]):
        """Set attributes for a resource."""
        self._resource_attributes[resource_id] = attributes

    def get_attributes(self, resource_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get attributes for a resource."""
        return self._resource_attributes.get(resource_id, {})

    def add_attribute(self, resource_id: str, key: str, value: Any):
        """Add or update a single attribute for a resource."""
        if resource_id not in self._resource_attributes:
            self._resource_attributes[resource_id] = {}
        self._resource_attributes[resource_id][key] = value


class EnvironmentAttributeProvider(AttributeProvider):
    """Provides environment attributes for ABAC evaluation."""

    def get_attributes(self, subject_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get environment attributes.

        Returns current environment state like time, IP address, etc.
        """
        now = datetime.now()

        return {
            'time_of_day': now.hour,
            'day_of_week': now.weekday(),  # 0=Monday, 6=Sunday
            'is_weekend': now.weekday() >= 5,
            'is_business_hours': 9 <= now.hour < 17 and now.weekday() < 5,
            'current_datetime': now.isoformat(),
            'ip_address': context.get('ip_address'),
            'user_agent': context.get('user_agent'),
            'location': context.get('location'),
            'tenant_id': context.get('tenant_id')
        }


# ==============================================================================
# ABAC Policy Models
# ==============================================================================

class PolicyEffect(str, Enum):
    """Effect of a policy (allow or deny)."""
    ALLOW = "allow"
    DENY = "deny"


class ConditionOperator(str, Enum):
    """Operators for policy conditions."""
    EQ = "eq"  # equals
    NE = "ne"  # not equals
    GT = "gt"  # greater than
    LT = "lt"  # less than
    GTE = "gte"  # greater than or equal
    LTE = "lte"  # less than or equal
    IN = "in"  # in list
    NOT_IN = "not_in"  # not in list
    CONTAINS = "contains"  # string/list contains
    MATCHES = "matches"  # regex match
    BETWEEN = "between"  # between two values


class PolicyCondition(BaseModel):
    """Condition in an ABAC policy."""

    attribute: str = Field(..., description="Attribute path (e.g., 'user.department')")
    operator: ConditionOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")

    def evaluate(self, attributes: Dict[str, Any]) -> bool:
        """
        Evaluate condition against attributes.

        Args:
            attributes: Attribute dictionary

        Returns:
            True if condition is satisfied
        """
        attr_value = self._get_nested_value(attributes, self.attribute)

        if attr_value is None:
            return False

        try:
            if self.operator == ConditionOperator.EQ:
                return attr_value == self.value
            elif self.operator == ConditionOperator.NE:
                return attr_value != self.value
            elif self.operator == ConditionOperator.GT:
                return attr_value > self.value
            elif self.operator == ConditionOperator.LT:
                return attr_value < self.value
            elif self.operator == ConditionOperator.GTE:
                return attr_value >= self.value
            elif self.operator == ConditionOperator.LTE:
                return attr_value <= self.value
            elif self.operator == ConditionOperator.IN:
                return attr_value in self.value
            elif self.operator == ConditionOperator.NOT_IN:
                return attr_value not in self.value
            elif self.operator == ConditionOperator.CONTAINS:
                return self.value in attr_value
            elif self.operator == ConditionOperator.MATCHES:
                import re
                return bool(re.match(self.value, str(attr_value)))
            elif self.operator == ConditionOperator.BETWEEN:
                if isinstance(self.value, list) and len(self.value) == 2:
                    return self.value[0] <= attr_value <= self.value[1]
                return False
        except (TypeError, ValueError) as e:
            logger.warning(f"Error evaluating condition: {e}")
            return False

        return False

    def _get_nested_value(self, attributes: Dict[str, Any], path: str) -> Any:
        """Get nested value from attributes using dot notation."""
        parts = path.split('.')
        value = attributes

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value


class ABACPolicy(BaseModel):
    """
    Attribute-Based Access Control Policy.

    A policy defines rules for access based on attributes of the user,
    resource, and environment.
    """

    policy_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique policy identifier"
    )
    name: str = Field(..., description="Policy name")
    description: str = Field(default="", description="Policy description")
    effect: PolicyEffect = Field(..., description="Effect when policy matches (allow/deny)")

    # What actions and resources this policy applies to
    actions: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Actions this policy applies to (supports wildcards)"
    )
    resources: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Resources this policy applies to (supports wildcards)"
    )

    # Conditions that must be satisfied
    conditions: List[PolicyCondition] = Field(
        default_factory=list,
        description="Conditions that must all be satisfied"
    )

    # Policy metadata
    priority: int = Field(
        default=100,
        description="Policy priority (higher = evaluated first)"
    )
    is_enabled: bool = Field(default=True, description="Whether policy is enabled")
    tenant_id: Optional[str] = Field(None, description="Tenant this policy belongs to")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            PolicyEffect: lambda v: v.value,
            ConditionOperator: lambda v: v.value
        }

    def matches(self, resource: str, action: str) -> bool:
        """
        Check if policy applies to resource and action.

        Args:
            resource: Resource being accessed
            action: Action being performed

        Returns:
            True if policy applies
        """
        import fnmatch

        # Check if action matches
        action_match = any(fnmatch.fnmatch(action, pattern) for pattern in self.actions)
        if not action_match:
            return False

        # Check if resource matches
        resource_match = any(fnmatch.fnmatch(resource, pattern) for pattern in self.resources)
        if not resource_match:
            return False

        return True

    def evaluate_conditions(self, attributes: Dict[str, Any]) -> bool:
        """
        Evaluate all policy conditions.

        Args:
            attributes: Combined attributes (user, resource, environment)

        Returns:
            True if all conditions are satisfied
        """
        if not self.conditions:
            return True  # No conditions means always match

        for condition in self.conditions:
            if not condition.evaluate(attributes):
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'policy_id': self.policy_id,
            'name': self.name,
            'description': self.description,
            'effect': self.effect.value,
            'actions': self.actions,
            'resources': self.resources,
            'conditions': [
                {
                    'attribute': c.attribute,
                    'operator': c.operator.value,
                    'value': c.value
                }
                for c in self.conditions
            ],
            'priority': self.priority,
            'is_enabled': self.is_enabled,
            'tenant_id': self.tenant_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABACPolicy':
        """Create policy from dictionary."""
        if 'conditions' in data and data['conditions']:
            data['conditions'] = [
                PolicyCondition(
                    attribute=c['attribute'],
                    operator=ConditionOperator(c['operator']),
                    value=c['value']
                )
                for c in data['conditions']
            ]
        if 'effect' in data and isinstance(data['effect'], str):
            data['effect'] = PolicyEffect(data['effect'])
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ABACPolicy':
        """Create policy from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'ABACPolicy':
        """Create policy from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, file_path: Path) -> 'ABACPolicy':
        """Load policy from file (JSON or YAML)."""
        with open(file_path, 'r') as f:
            content = f.read()

        if file_path.suffix in ['.yaml', '.yml']:
            return cls.from_yaml(content)
        else:
            return cls.from_json(content)


# ==============================================================================
# Policy Evaluation Engine
# ==============================================================================

@dataclass
class PolicyEvaluationResult:
    """Result of ABAC policy evaluation."""

    allowed: bool
    matched_policies: List[ABACPolicy] = field(default_factory=list)
    denied_by: Optional[ABACPolicy] = None
    evaluation_time_ms: float = 0.0
    reason: str = ""
    attributes_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'allowed': self.allowed,
            'matched_policies': [p.name for p in self.matched_policies],
            'denied_by': self.denied_by.name if self.denied_by else None,
            'evaluation_time_ms': round(self.evaluation_time_ms, 2),
            'reason': self.reason
        }


class ABACEvaluator:
    """
    ABAC Policy Evaluation Engine.

    Evaluates access requests against ABAC policies using user, resource,
    and environment attributes.
    """

    def __init__(
        self,
        user_provider: Optional[UserAttributeProvider] = None,
        resource_provider: Optional[ResourceAttributeProvider] = None,
        environment_provider: Optional[EnvironmentAttributeProvider] = None
    ):
        """
        Initialize ABAC evaluator.

        Args:
            user_provider: Provider for user attributes
            resource_provider: Provider for resource attributes
            environment_provider: Provider for environment attributes
        """
        self.user_provider = user_provider or UserAttributeProvider()
        self.resource_provider = resource_provider or ResourceAttributeProvider()
        self.environment_provider = environment_provider or EnvironmentAttributeProvider()

        self._policies: Dict[str, ABACPolicy] = {}
        self._stats = {
            'evaluations': 0,
            'allows': 0,
            'denies': 0
        }

        logger.info("Initialized ABACEvaluator")

    def add_policy(self, policy: ABACPolicy):
        """Add a policy to the evaluator."""
        self._policies[policy.policy_id] = policy
        logger.info(f"Added ABAC policy: {policy.name}")

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy from the evaluator."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            logger.info(f"Removed ABAC policy: {policy_id}")
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[ABACPolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(
        self,
        tenant_id: Optional[str] = None,
        enabled_only: bool = True
    ) -> List[ABACPolicy]:
        """List policies matching criteria."""
        policies = list(self._policies.values())

        if tenant_id is not None:
            policies = [p for p in policies if p.tenant_id == tenant_id]

        if enabled_only:
            policies = [p for p in policies if p.is_enabled]

        return policies

    def evaluate(
        self,
        user_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PolicyEvaluationResult:
        """
        Evaluate access request against ABAC policies.

        Resolution strategy:
        1. Collect all matching policies
        2. Sort by priority (higher first)
        3. Explicit DENY wins over ALLOW
        4. If no matching policies, default deny

        Args:
            user_id: User making the request
            resource: Resource being accessed
            action: Action being performed
            context: Additional context for evaluation

        Returns:
            Policy evaluation result
        """
        import time
        start_time = time.time()

        self._stats['evaluations'] += 1
        context = context or {}

        # Gather all attributes
        user_attrs = self.user_provider.get_attributes(user_id, context)
        resource_attrs = self.resource_provider.get_attributes(resource, context)
        env_attrs = self.environment_provider.get_attributes(user_id, context)

        combined_attrs = {
            'user': user_attrs,
            'resource': resource_attrs,
            'environment': env_attrs
        }

        # Find matching policies
        matching_allow: List[ABACPolicy] = []
        matching_deny: List[ABACPolicy] = []

        enabled_policies = [p for p in self._policies.values() if p.is_enabled]
        # Sort by priority (higher first)
        enabled_policies.sort(key=lambda p: p.priority, reverse=True)

        for policy in enabled_policies:
            # Check if policy applies to this resource/action
            if not policy.matches(resource, action):
                continue

            # Evaluate policy conditions
            if policy.evaluate_conditions(combined_attrs):
                if policy.effect == PolicyEffect.DENY:
                    matching_deny.append(policy)
                else:
                    matching_allow.append(policy)

        # Apply resolution strategy: explicit deny wins
        evaluation_time = (time.time() - start_time) * 1000

        if matching_deny:
            result = PolicyEvaluationResult(
                allowed=False,
                matched_policies=matching_allow + matching_deny,
                denied_by=matching_deny[0],
                evaluation_time_ms=evaluation_time,
                reason=f"Denied by policy: {matching_deny[0].name}",
                attributes_used=combined_attrs
            )
            self._stats['denies'] += 1
        elif matching_allow:
            result = PolicyEvaluationResult(
                allowed=True,
                matched_policies=matching_allow,
                evaluation_time_ms=evaluation_time,
                reason=f"Allowed by {len(matching_allow)} policy/policies",
                attributes_used=combined_attrs
            )
            self._stats['allows'] += 1
        else:
            # Default deny
            result = PolicyEvaluationResult(
                allowed=False,
                matched_policies=[],
                evaluation_time_ms=evaluation_time,
                reason="No matching policies (default deny)",
                attributes_used=combined_attrs
            )
            self._stats['denies'] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        allow_rate = 0.0
        if self._stats['evaluations'] > 0:
            allow_rate = (self._stats['allows'] / self._stats['evaluations']) * 100

        return {
            **self._stats,
            'allow_rate': round(allow_rate, 2),
            'policy_count': len(self._policies)
        }

    def load_policies_from_directory(self, directory: Path):
        """
        Load all policies from a directory.

        Args:
            directory: Directory containing policy files (.json or .yaml)
        """
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        count = 0
        for file_path in directory.glob('*.json'):
            try:
                policy = ABACPolicy.from_file(file_path)
                self.add_policy(policy)
                count += 1
            except Exception as e:
                logger.error(f"Error loading policy from {file_path}: {e}")

        for file_path in directory.glob('*.yaml'):
            try:
                policy = ABACPolicy.from_file(file_path)
                self.add_policy(policy)
                count += 1
            except Exception as e:
                logger.error(f"Error loading policy from {file_path}: {e}")

        for file_path in directory.glob('*.yml'):
            try:
                policy = ABACPolicy.from_file(file_path)
                self.add_policy(policy)
                count += 1
            except Exception as e:
                logger.error(f"Error loading policy from {file_path}: {e}")

        logger.info(f"Loaded {count} policies from {directory}")


# ==============================================================================
# OPA (Open Policy Agent) Integration Support
# ==============================================================================

class OPAIntegration:
    """
    Integration with Open Policy Agent (OPA) for policy evaluation.

    Provides adapter to use OPA as the policy decision point.
    """

    def __init__(self, opa_url: str = "http://localhost:8181"):
        """
        Initialize OPA integration.

        Args:
            opa_url: URL of OPA server
        """
        self.opa_url = opa_url.rstrip('/')
        logger.info(f"Initialized OPA integration: {opa_url}")

    def evaluate(
        self,
        policy_path: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a policy using OPA.

        Args:
            policy_path: OPA policy path (e.g., "greenlang/authz/allow")
            input_data: Input data for policy evaluation

        Returns:
            OPA evaluation result

        Raises:
            NotImplementedError: OPA integration not yet implemented
        """
        # This is a placeholder for OPA integration
        # In production, this would make HTTP requests to OPA server
        raise NotImplementedError(
            "OPA integration requires requests library and OPA server. "
            "Use ABACEvaluator for built-in policy evaluation."
        )


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_policy(
    name: str,
    effect: PolicyEffect,
    actions: List[str],
    resources: List[str],
    conditions: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> ABACPolicy:
    """
    Convenience function to create an ABAC policy.

    Args:
        name: Policy name
        effect: Policy effect (allow/deny)
        actions: Actions this policy applies to
        resources: Resources this policy applies to
        conditions: List of condition dictionaries
        **kwargs: Additional policy attributes

    Returns:
        Created ABAC policy
    """
    condition_objs = []
    if conditions:
        condition_objs = [
            PolicyCondition(
                attribute=c['attribute'],
                operator=ConditionOperator(c['operator']),
                value=c['value']
            )
            for c in conditions
        ]

    return ABACPolicy(
        name=name,
        effect=effect,
        actions=actions,
        resources=resources,
        conditions=condition_objs,
        **kwargs
    )


__all__ = [
    'AttributeProvider',
    'UserAttributeProvider',
    'ResourceAttributeProvider',
    'EnvironmentAttributeProvider',
    'PolicyEffect',
    'ConditionOperator',
    'PolicyCondition',
    'ABACPolicy',
    'PolicyEvaluationResult',
    'ABACEvaluator',
    'OPAIntegration',
    'create_policy'
]
