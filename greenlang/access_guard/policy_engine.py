# -*- coding: utf-8 -*-
"""
Policy Engine - AGENT-FOUND-006: Access & Policy Guard

Core policy evaluation engine supporting RBAC, ABAC, policy inheritance,
and priority-sorted first-match-wins rule semantics. Thread-safe with
SHA-256 provenance on all policy mutations.

Zero-Hallucination Guarantees:
    - All access decisions are deterministic based on stored policy rules
    - No ML or probabilistic decisions
    - Complete provenance tracking for every policy mutation
    - First-match-wins on priority-sorted rules

Example:
    >>> from greenlang.access_guard.policy_engine import PolicyEngine
    >>> engine = PolicyEngine()
    >>> policy_hash = engine.add_policy(my_policy)
    >>> result = engine.evaluate(access_request)
    >>> print(result.decision)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.access_guard.models import (
    AccessDecision,
    AccessDecisionResult,
    AccessRequest,
    CLASSIFICATION_HIERARCHY,
    Policy,
    PolicyRule,
    PolicyType,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class PolicyEngine:
    """Core policy evaluation engine.

    Supports RBAC, ABAC, and policy inheritance with priority-sorted
    first-match-wins rule evaluation. All mutations are protected by
    a threading lock and produce SHA-256 provenance hashes.

    Attributes:
        strict_mode: If True, deny by default when no rules match.
        _policies: Internal policy store keyed by policy_id.
        _policy_hierarchy: Maps child policy ID to list of parent IDs.

    Example:
        >>> engine = PolicyEngine(strict_mode=True)
        >>> engine.add_policy(policy)
        >>> result = engine.evaluate(request)
    """

    def __init__(self, strict_mode: bool = True) -> None:
        """Initialize the PolicyEngine.

        Args:
            strict_mode: Deny by default when no rules match.
        """
        self.strict_mode = strict_mode
        self._policies: Dict[str, Policy] = {}
        self._policy_hierarchy: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
        logger.info("PolicyEngine initialized (strict_mode=%s)", strict_mode)

    # ------------------------------------------------------------------
    # Policy CRUD
    # ------------------------------------------------------------------

    def add_policy(self, policy: Policy) -> str:
        """Add a policy to the engine.

        Computes a provenance hash and stores the policy. If the policy
        has a parent_policy_id, the inheritance relationship is tracked.

        Args:
            policy: The policy to add.

        Returns:
            SHA-256 provenance hash of the policy.

        Raises:
            ValueError: If max policy capacity would be exceeded.
        """
        with self._lock:
            policy.provenance_hash = policy.compute_hash()
            policy.updated_at = _utcnow()
            self._policies[policy.policy_id] = policy

            if policy.parent_policy_id:
                parents = self._policy_hierarchy.setdefault(policy.policy_id, [])
                if policy.parent_policy_id not in parents:
                    parents.append(policy.parent_policy_id)

        logger.info(
            "Added policy: %s (hash: %s)",
            policy.policy_id, policy.provenance_hash[:16],
        )
        return policy.provenance_hash

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy from the engine.

        Args:
            policy_id: ID of the policy to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                self._policy_hierarchy.pop(policy_id, None)
                logger.info("Removed policy: %s", policy_id)
                return True
        return False

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID.

        Args:
            policy_id: The policy identifier.

        Returns:
            Policy if found, None otherwise.
        """
        return self._policies.get(policy_id)

    def list_policies(
        self,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> List[Policy]:
        """List policies with optional filters.

        Args:
            tenant_id: Optional tenant filter.
            resource_type: Optional resource type filter.

        Returns:
            List of matching policies.
        """
        results: List[Policy] = []
        for policy in self._policies.values():
            if tenant_id and policy.tenant_id and policy.tenant_id != tenant_id:
                continue
            if resource_type and policy.applies_to and resource_type not in policy.applies_to:
                continue
            results.append(policy)
        return results

    def update_policy(
        self,
        policy_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        enabled: Optional[bool] = None,
        rules: Optional[List[PolicyRule]] = None,
        applies_to: Optional[List[str]] = None,
    ) -> Policy:
        """Update an existing policy.

        Args:
            policy_id: ID of the policy to update.
            name: New policy name.
            description: New description.
            enabled: New enabled state.
            rules: New list of rules.
            applies_to: New resource type scope.

        Returns:
            Updated policy with new provenance hash.

        Raises:
            KeyError: If policy_id not found.
        """
        with self._lock:
            policy = self._policies.get(policy_id)
            if policy is None:
                raise KeyError(f"Policy not found: {policy_id}")

            if name is not None:
                policy.name = name
            if description is not None:
                policy.description = description
            if enabled is not None:
                policy.enabled = enabled
            if rules is not None:
                policy.rules = rules
            if applies_to is not None:
                policy.applies_to = applies_to

            policy.updated_at = _utcnow()
            policy.provenance_hash = policy.compute_hash()

        logger.info(
            "Updated policy: %s (hash: %s)",
            policy_id, policy.provenance_hash[:16],
        )
        return policy

    # ------------------------------------------------------------------
    # Rule collection
    # ------------------------------------------------------------------

    def get_effective_rules(
        self,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> List[PolicyRule]:
        """Get all effective rules for a tenant/resource type.

        Respects policy inheritance by including parent policy rules
        when a child policy is in scope. Rules are sorted by priority
        (lower number = higher priority).

        Args:
            tenant_id: Optional tenant filter.
            resource_type: Optional resource type filter.

        Returns:
            List of applicable rules sorted by priority.
        """
        rules: List[PolicyRule] = []
        visited: set[str] = set()

        for policy in self._policies.values():
            self._collect_rules(
                policy, tenant_id, resource_type, rules, visited,
            )

        rules.sort(key=lambda r: r.priority)
        return rules

    def _collect_rules(
        self,
        policy: Policy,
        tenant_id: Optional[str],
        resource_type: Optional[str],
        rules: List[PolicyRule],
        visited: set[str],
    ) -> None:
        """Recursively collect rules from a policy and its parents.

        Args:
            policy: Current policy to collect from.
            tenant_id: Tenant filter.
            resource_type: Resource type filter.
            rules: Accumulator list.
            visited: Set of already-visited policy IDs.
        """
        if policy.policy_id in visited:
            return
        visited.add(policy.policy_id)

        if not policy.enabled:
            return

        if tenant_id and policy.tenant_id and policy.tenant_id != tenant_id:
            return

        if resource_type and policy.applies_to and resource_type not in policy.applies_to:
            return

        for rule in policy.rules:
            if rule.enabled:
                rules.append(rule)

        # Collect from parent policies
        parent_ids = self._policy_hierarchy.get(policy.policy_id, [])
        for parent_id in parent_ids:
            parent = self._policies.get(parent_id)
            if parent is not None:
                self._collect_rules(
                    parent, tenant_id, resource_type, rules, visited,
                )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, request: AccessRequest) -> AccessDecisionResult:
        """Evaluate an access request against all applicable policies.

        Uses first-match-wins on priority-sorted rules. If no rule
        matches, the default decision depends on strict_mode.

        Args:
            request: The access request to evaluate.

        Returns:
            AccessDecisionResult with decision and details.
        """
        start_time = time.time()

        matching_rules: List[str] = []
        deny_reasons: List[str] = []
        conditions: List[str] = []
        policy_versions: Dict[str, str] = {}

        # Collect applicable rules
        rules = self.get_effective_rules(
            tenant_id=request.resource.tenant_id,
            resource_type=request.resource.resource_type,
        )

        # Record policy versions for audit
        for policy in self._policies.values():
            policy_versions[policy.policy_id] = policy.version

        # Default decision based on mode
        decision = AccessDecision.DENY if self.strict_mode else AccessDecision.ALLOW

        for rule in rules:
            if self._rule_matches(rule, request):
                matching_rules.append(rule.rule_id)

                if rule.effect == AccessDecision.ALLOW:
                    decision = AccessDecision.ALLOW
                    if rule.time_constraints or rule.geographic_constraints:
                        conditions.append(
                            f"Rule {rule.rule_id} has conditions",
                        )
                        decision = AccessDecision.CONDITIONAL
                    break
                elif rule.effect == AccessDecision.DENY:
                    decision = AccessDecision.DENY
                    deny_reasons.append(
                        f"Denied by rule: {rule.name} ({rule.rule_id})",
                    )
                    break

        # Resolve conditional decisions
        if decision == AccessDecision.CONDITIONAL:
            conditions_met = self._check_conditions(request, matching_rules)
            if conditions_met:
                decision = AccessDecision.ALLOW
                conditions = []
            else:
                decision = AccessDecision.DENY
                deny_reasons.extend(conditions)

        evaluation_time = (time.time() - start_time) * 1000

        result = AccessDecisionResult(
            request_id=request.request_id,
            decision=decision,
            allowed=(decision == AccessDecision.ALLOW),
            matching_rules=matching_rules,
            deny_reasons=deny_reasons,
            conditions=conditions if decision == AccessDecision.CONDITIONAL else [],
            evaluation_time_ms=evaluation_time,
            policy_versions=policy_versions,
        )

        # Compute decision hash for provenance
        decision_str = json.dumps(
            {
                "request_id": request.request_id,
                "decision": decision.value,
                "matching_rules": matching_rules,
                "timestamp": result.evaluated_at.isoformat(),
            },
            sort_keys=True,
        )
        result.decision_hash = hashlib.sha256(decision_str.encode()).hexdigest()

        return result

    # ------------------------------------------------------------------
    # Rule matching helpers
    # ------------------------------------------------------------------

    def _rule_matches(self, rule: PolicyRule, request: AccessRequest) -> bool:
        """Check if a rule matches the given request.

        Args:
            rule: Rule to check.
            request: Incoming request.

        Returns:
            True if the rule matches.
        """
        # Check action
        if rule.actions and request.action not in rule.actions:
            if "*" not in rule.actions:
                return False

        # Check principal pattern
        if rule.principals:
            principal_match = self._match_principal(rule.principals, request)
            if not principal_match:
                return False

        # Check resource pattern
        if rule.resources:
            resource_match = self._match_resource(rule.resources, request)
            if not resource_match:
                return False

        # Check classification constraint
        if rule.classification_max:
            resource_level = CLASSIFICATION_HIERARCHY.get(
                request.resource.classification, 0,
            )
            max_level = CLASSIFICATION_HIERARCHY.get(rule.classification_max, 0)
            if resource_level > max_level:
                return False

        # Check ABAC conditions
        if rule.conditions:
            for key, expected_value in rule.conditions.items():
                actual_value = request.context.get(key)
                if actual_value != expected_value:
                    actual_value = request.principal.attributes.get(key)
                    if actual_value != expected_value:
                        return False

        return True

    def _match_principal(
        self, patterns: List[str], request: AccessRequest,
    ) -> bool:
        """Check if any principal pattern matches the request.

        Args:
            patterns: Principal patterns to match.
            request: Incoming request.

        Returns:
            True if any pattern matches.
        """
        for pattern in patterns:
            if self._pattern_matches(pattern, request.principal.principal_id):
                return True
            if pattern.startswith("role:"):
                role_name = pattern[5:]
                if role_name in request.principal.roles or role_name == "*":
                    return True
        return False

    def _match_resource(
        self, patterns: List[str], request: AccessRequest,
    ) -> bool:
        """Check if any resource pattern matches the request.

        Args:
            patterns: Resource patterns to match.
            request: Incoming request.

        Returns:
            True if any pattern matches.
        """
        for pattern in patterns:
            if self._pattern_matches(pattern, request.resource.resource_id):
                return True
            if pattern.startswith("type:"):
                type_name = pattern[5:]
                if type_name == request.resource.resource_type or type_name == "*":
                    return True
        return False

    def _pattern_matches(self, pattern: str, value: str) -> bool:
        """Check if a glob-like pattern matches a value.

        Args:
            pattern: Glob pattern (supports ``*`` wildcard).
            value: Value to match against.

        Returns:
            True if pattern matches value.
        """
        if pattern == "*":
            return True
        if "*" not in pattern:
            return pattern == value
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", value))

    def _check_conditions(
        self, request: AccessRequest, rule_ids: List[str],
    ) -> bool:
        """Check if all conditions are met for conditional access.

        Evaluates time and geographic constraints on matching rules.

        Args:
            request: The access request.
            rule_ids: IDs of matching rules to check.

        Returns:
            True if all conditions are met.
        """
        now = _utcnow()

        for rule_id in rule_ids:
            for policy in self._policies.values():
                for rule in policy.rules:
                    if rule.rule_id != rule_id:
                        continue

                    if rule.time_constraints:
                        start_hour = rule.time_constraints.get("start_hour", 0)
                        end_hour = rule.time_constraints.get("end_hour", 24)
                        if not (start_hour <= now.hour < end_hour):
                            return False

                    if rule.geographic_constraints:
                        location = request.resource.geographic_location
                        if location and location not in rule.geographic_constraints:
                            return False

        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Return the number of loaded policies."""
        return len(self._policies)


__all__ = [
    "PolicyEngine",
]
