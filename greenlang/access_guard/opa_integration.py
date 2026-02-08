# -*- coding: utf-8 -*-
"""
OPA Integration - AGENT-FOUND-006: Access & Policy Guard

Open Policy Agent (OPA) Rego policy management client. Stores Rego
policy source with SHA-256 hashing and version tracking. Provides
a local evaluation stub for environments without an OPA server.

Zero-Hallucination Guarantees:
    - All Rego policies are hashed with SHA-256
    - Version tracking for every Rego policy mutation
    - Basic syntax validation (package/rule detection)
    - No probabilistic evaluation

Example:
    >>> from greenlang.access_guard.opa_integration import OPAClient
    >>> client = OPAClient()
    >>> policy_hash = client.add_rego_policy("deny-export", rego_src)
    >>> valid, error = client.validate_rego_syntax(rego_src)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.access_guard.models import (
    AccessDecision,
    AccessDecisionResult,
    AccessRequest,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class OPAClient:
    """Open Policy Agent Rego policy management client.

    Manages Rego policy source storage, SHA-256 hashing, version
    tracking, and basic syntax validation. Provides an evaluation
    stub that can be replaced with real OPA server integration.

    Attributes:
        opa_endpoint: Optional OPA server URL.
        _policies: Rego source keyed by policy_id.
        _hashes: SHA-256 hashes keyed by policy_id.
        _versions: Version counters keyed by policy_id.

    Example:
        >>> client = OPAClient()
        >>> h = client.add_rego_policy("p1", 'package p1\\ndefault allow = false')
        >>> policies = client.list_rego_policies()
    """

    def __init__(self, opa_endpoint: Optional[str] = None) -> None:
        """Initialize the OPA client.

        Args:
            opa_endpoint: Optional OPA server endpoint URL. When None,
                evaluation operates in stub mode.
        """
        self.opa_endpoint = opa_endpoint
        self._policies: Dict[str, str] = {}
        self._hashes: Dict[str, str] = {}
        self._versions: Dict[str, int] = {}
        self._created_at: Dict[str, datetime] = {}
        self._updated_at: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        logger.info(
            "OPAClient initialized (endpoint=%s)",
            opa_endpoint or "stub-mode",
        )

    # ------------------------------------------------------------------
    # Rego policy CRUD
    # ------------------------------------------------------------------

    def add_rego_policy(self, policy_id: str, rego_source: str) -> str:
        """Add or update a Rego policy.

        Stores the Rego source, computes its SHA-256 hash, and
        increments the version counter.

        Args:
            policy_id: Unique policy identifier.
            rego_source: Rego policy source code.

        Returns:
            SHA-256 hash of the Rego source.
        """
        policy_hash = hashlib.sha256(rego_source.encode()).hexdigest()
        now = _utcnow()

        with self._lock:
            is_update = policy_id in self._policies
            self._policies[policy_id] = rego_source
            self._hashes[policy_id] = policy_hash
            self._versions[policy_id] = self._versions.get(policy_id, 0) + 1
            self._updated_at[policy_id] = now
            if not is_update:
                self._created_at[policy_id] = now

        action = "Updated" if is_update else "Added"
        logger.info(
            "%s Rego policy: %s v%d (hash: %s)",
            action, policy_id, self._versions[policy_id], policy_hash[:16],
        )
        return policy_hash

    def get_rego_policy(self, policy_id: str) -> Optional[str]:
        """Get the Rego source for a policy.

        Args:
            policy_id: The policy identifier.

        Returns:
            Rego source string if found, None otherwise.
        """
        return self._policies.get(policy_id)

    def list_rego_policies(self) -> List[Dict[str, Any]]:
        """List all registered Rego policies.

        Returns:
            List of dictionaries with policy_id, hash, version,
            created_at, and updated_at.
        """
        results: List[Dict[str, Any]] = []
        for policy_id in self._policies:
            results.append({
                "policy_id": policy_id,
                "hash": self._hashes.get(policy_id, ""),
                "version": self._versions.get(policy_id, 0),
                "created_at": self._created_at.get(policy_id, _utcnow()).isoformat(),
                "updated_at": self._updated_at.get(policy_id, _utcnow()).isoformat(),
            })
        return results

    def remove_rego_policy(self, policy_id: str) -> bool:
        """Remove a Rego policy.

        Args:
            policy_id: The policy identifier.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if policy_id not in self._policies:
                return False
            del self._policies[policy_id]
            self._hashes.pop(policy_id, None)
            self._versions.pop(policy_id, None)
            self._created_at.pop(policy_id, None)
            self._updated_at.pop(policy_id, None)

        logger.info("Removed Rego policy: %s", policy_id)
        return True

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_rego(
        self,
        request: AccessRequest,
        policy_id: str,
    ) -> Optional[AccessDecisionResult]:
        """Evaluate a request against a Rego policy.

        In stub mode (no OPA endpoint), this returns a deny-by-default
        result with a note that OPA evaluation is not connected.

        In live mode, this would POST the request to the OPA server
        and parse the response.

        Args:
            request: The access request.
            policy_id: The Rego policy to evaluate against.

        Returns:
            AccessDecisionResult, or None if the policy is not found.
        """
        rego_source = self._policies.get(policy_id)
        if rego_source is None:
            logger.error("Rego policy not found: %s", policy_id)
            return None

        start_time = time.time()

        if self.opa_endpoint:
            # Live OPA evaluation -- integration point
            result = self._evaluate_remote(request, policy_id, rego_source)
        else:
            # Stub mode: deny by default with provenance
            result = self._evaluate_stub(request, policy_id)

        evaluation_time = (time.time() - start_time) * 1000
        result.evaluation_time_ms = evaluation_time

        # Compute decision hash
        decision_str = json.dumps(
            {
                "request_id": request.request_id,
                "decision": result.decision.value,
                "policy_id": policy_id,
                "timestamp": result.evaluated_at.isoformat(),
            },
            sort_keys=True,
        )
        result.decision_hash = hashlib.sha256(decision_str.encode()).hexdigest()

        return result

    # ------------------------------------------------------------------
    # Syntax validation
    # ------------------------------------------------------------------

    def validate_rego_syntax(
        self, rego_source: str,
    ) -> Tuple[bool, Optional[str]]:
        """Validate basic Rego syntax.

        Checks for required ``package`` declaration and at least one
        rule definition. This is a lightweight check; full validation
        requires the OPA compiler.

        Args:
            rego_source: Rego source code to validate.

        Returns:
            Tuple of (valid, error_message). error_message is None
            when valid is True.
        """
        if not rego_source or not rego_source.strip():
            return False, "Rego source is empty"

        lines = rego_source.strip().splitlines()
        stripped_lines = [line.strip() for line in lines if line.strip()]

        # Check for package declaration
        has_package = any(
            line.startswith("package ") for line in stripped_lines
        )
        if not has_package:
            return False, "Missing 'package' declaration"

        # Check for at least one rule
        has_rule = any(
            self._looks_like_rule(line) for line in stripped_lines
        )
        if not has_rule:
            return False, "No rule definitions found"

        # Check for balanced braces
        open_braces = rego_source.count("{")
        close_braces = rego_source.count("}")
        if open_braces != close_braces:
            return False, (
                f"Unbalanced braces: {open_braces} opening, "
                f"{close_braces} closing"
            )

        return True, None

    @property
    def count(self) -> int:
        """Return the number of registered Rego policies."""
        return len(self._policies)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_stub(
        self, request: AccessRequest, policy_id: str,
    ) -> AccessDecisionResult:
        """Stub evaluation when no OPA server is connected.

        Args:
            request: The access request.
            policy_id: Policy ID for provenance.

        Returns:
            Deny-by-default AccessDecisionResult.
        """
        logger.info(
            "OPA stub evaluation for policy %s (no OPA server)",
            policy_id,
        )
        return AccessDecisionResult(
            request_id=request.request_id,
            decision=AccessDecision.DENY,
            allowed=False,
            deny_reasons=[
                f"OPA stub mode: policy '{policy_id}' not evaluated "
                f"(no OPA server configured)",
            ],
            policy_versions={policy_id: str(self._versions.get(policy_id, 0))},
        )

    def _evaluate_remote(
        self,
        request: AccessRequest,
        policy_id: str,
        rego_source: str,
    ) -> AccessDecisionResult:
        """Evaluate against a remote OPA server (integration point).

        In production, this would use httpx to POST the request
        input to the OPA data API and parse the decision.

        Args:
            request: The access request.
            policy_id: The policy to evaluate.
            rego_source: The Rego source (for context).

        Returns:
            AccessDecisionResult from the OPA server.
        """
        # NOTE: Replace this stub with actual OPA HTTP call:
        #   POST {self.opa_endpoint}/v1/data/{package_path}
        #   Body: {"input": request_input}
        logger.info(
            "Would evaluate Rego policy %s at %s",
            policy_id, self.opa_endpoint,
        )
        return AccessDecisionResult(
            request_id=request.request_id,
            decision=AccessDecision.DENY,
            allowed=False,
            deny_reasons=[
                f"OPA remote evaluation not implemented for policy "
                f"'{policy_id}'",
            ],
            policy_versions={policy_id: str(self._versions.get(policy_id, 0))},
        )

    @staticmethod
    def _looks_like_rule(line: str) -> bool:
        """Check if a line looks like a Rego rule definition.

        Args:
            line: Stripped source line.

        Returns:
            True if the line matches common Rego rule patterns.
        """
        rule_starters = (
            "default ", "allow", "deny", "violation",
            "authz", "permit", "forbid",
        )
        for starter in rule_starters:
            if line.startswith(starter):
                return True

        # Pattern: identifier = value or identifier { body }
        if "=" in line or (line.endswith("{") and " " in line):
            return True

        return False


__all__ = [
    "OPAClient",
]
