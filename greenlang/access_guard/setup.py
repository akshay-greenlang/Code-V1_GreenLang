# -*- coding: utf-8 -*-
"""
Access Guard Service Setup - AGENT-FOUND-006: Access & Policy Guard

Provides ``configure_access_guard(app)`` which wires up the Access &
Policy Guard SDK (policy engine, rate limiter, classifier, audit logger,
OPA client, provenance) and mounts the REST API.

Also exposes ``get_access_guard(app)`` for programmatic access and the
``AccessGuardService`` facade class.

The ``check_access()`` method orchestrates the full access decision
pipeline:
    1. Authentication check
    2. Tenant isolation check
    3. Data classification / clearance check
    4. Rate limiting check
    5. Policy engine evaluation
    6. Audit logging
    7. Simulation mode handling

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.access_guard.setup import configure_access_guard
    >>> app = FastAPI()
    >>> configure_access_guard(app)

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
from typing import Any, Dict, List, Optional, Tuple

from greenlang.access_guard.config import AccessGuardConfig, get_config
from greenlang.access_guard.policy_engine import PolicyEngine
from greenlang.access_guard.rate_limiter import RateLimiter
from greenlang.access_guard.classifier import DataClassifier
from greenlang.access_guard.audit_logger import AuditLogger
from greenlang.access_guard.opa_integration import OPAClient
from greenlang.access_guard.provenance import ProvenanceTracker
from greenlang.access_guard.models import (
    AccessDecision,
    AccessDecisionResult,
    AccessRequest,
    AuditEvent,
    AuditEventType,
    CLASSIFICATION_HIERARCHY,
    RateLimitConfig,
)
from greenlang.access_guard.metrics import (
    PROMETHEUS_AVAILABLE,
    record_decision,
    record_denial,
    record_rate_limit,
    record_tenant_violation,
    record_classification_check,
    record_cache_hit,
    record_cache_miss,
    update_policies_count,
    update_rules_count,
    update_audit_events_count,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# AccessGuardService facade
# ===================================================================

_singleton_lock = threading.Lock()
_singleton_instance: Optional["AccessGuardService"] = None


class AccessGuardService:
    """Unified facade over the Access & Policy Guard SDK.

    Orchestrates policy evaluation, rate limiting, data classification,
    audit logging, OPA integration, and provenance tracking through a
    single ``check_access()`` entry point.

    Attributes:
        policy_engine: PolicyEngine instance.
        rate_limiter: RateLimiter instance.
        classifier: DataClassifier instance.
        audit_logger: AuditLogger instance.
        opa_client: OPAClient instance.
        provenance: ProvenanceTracker instance.
        config: AccessGuardConfig instance.

    Example:
        >>> service = AccessGuardService()
        >>> result = service.check_access(access_request)
        >>> if result.allowed:
        ...     print("Access granted")
    """

    def __init__(
        self,
        config: Optional[AccessGuardConfig] = None,
    ) -> None:
        """Initialize the Access Guard Service facade.

        Args:
            config: Optional config. Uses global config if None.
        """
        self.config = config or get_config()

        # Build rate limit config from global config
        rate_config = RateLimitConfig(
            requests_per_minute=self.config.default_rate_rpm,
            requests_per_hour=self.config.default_rate_rph,
            requests_per_day=self.config.default_rate_rpd,
            burst_limit=self.config.burst_limit,
        )

        # Initialize sub-components
        self.policy_engine = PolicyEngine(strict_mode=self.config.strict_mode)
        self.rate_limiter = RateLimiter(rate_config)
        self.classifier = DataClassifier()
        self.audit_logger = AuditLogger(
            retention_days=self.config.audit_retention_days,
        )
        self.opa_client = OPAClient(opa_endpoint=self.config.opa_endpoint)
        self.provenance = ProvenanceTracker()

        # Decision cache
        self._decision_cache: Dict[str, Tuple[AccessDecisionResult, float]] = {}

        # Internal metrics
        self._total_requests = 0
        self._allowed_requests = 0
        self._denied_requests = 0
        self._rate_limited_requests = 0

        self._started = False
        logger.info("AccessGuardService facade created")

    # ------------------------------------------------------------------
    # Main access check orchestration
    # ------------------------------------------------------------------

    def check_access(self, request: AccessRequest) -> AccessDecisionResult:
        """Check if an access request should be allowed.

        Orchestrates the full decision pipeline:
            1. Authentication check
            2. Tenant isolation check
            3. Data classification / clearance check
            4. Rate limiting check
            5. Policy engine evaluation
            6. Audit logging
            7. Simulation mode handling

        Args:
            request: The access request to evaluate.

        Returns:
            AccessDecisionResult with the decision and details.
        """
        self._total_requests += 1
        start_time = time.time()

        # Check cache first
        cache_key = self._cache_key(request)
        cached = self._check_cache(cache_key)
        if cached is not None:
            record_cache_hit()
            return cached
        record_cache_miss()

        deny_reasons: List[str] = []

        # Step 1: Verify principal is authenticated
        if not request.principal.authenticated:
            deny_reasons.append("Principal is not authenticated")
            result = self._make_deny(request, deny_reasons, start_time)
            self._log_event(
                AuditEventType.ACCESS_DENIED, request,
                AccessDecision.DENY, {"reason": "unauthenticated"},
            )
            record_denial("unauthenticated")
            return result

        # Step 2: Check tenant isolation
        if self.config.strict_tenant_isolation:
            if request.principal.tenant_id != request.resource.tenant_id:
                deny_reasons.append(
                    f"Tenant boundary violation: principal tenant "
                    f"'{request.principal.tenant_id}' cannot access "
                    f"resource in tenant '{request.resource.tenant_id}'"
                )
                result = self._make_deny(request, deny_reasons, start_time)
                self._log_event(
                    AuditEventType.TENANT_BOUNDARY_VIOLATION, request,
                    AccessDecision.DENY,
                )
                record_tenant_violation()
                record_denial("tenant_violation")
                return result

        # Step 3: Check data classification / clearance
        classified_level = self.classifier.classify(request.resource)
        record_classification_check(classified_level.value)

        principal_clearance = CLASSIFICATION_HIERARCHY.get(
            request.principal.clearance_level, 0,
        )
        resource_classification = CLASSIFICATION_HIERARCHY.get(
            classified_level, 0,
        )

        if resource_classification > principal_clearance:
            deny_reasons.append(
                f"Insufficient clearance: principal has "
                f"'{request.principal.clearance_level.value}' but resource "
                f"requires '{classified_level.value}'"
            )
            result = self._make_deny(request, deny_reasons, start_time)
            self._log_event(
                AuditEventType.CLASSIFICATION_CHECK, request,
                AccessDecision.DENY,
                {"reason": "insufficient_clearance",
                 "classified_as": classified_level.value},
            )
            record_denial("classification")
            return result

        # Step 4: Check rate limits
        if self.config.rate_limiting_enabled:
            highest_role = self._get_highest_role(request)
            allowed, rate_reason = self.rate_limiter.check_rate_limit(
                request.principal.tenant_id,
                request.principal.principal_id,
                highest_role,
            )
            if not allowed:
                self._rate_limited_requests += 1
                deny_reasons.append(rate_reason or "Rate limited")
                result = self._make_deny(request, deny_reasons, start_time)
                self._log_event(
                    AuditEventType.RATE_LIMIT_EXCEEDED, request,
                    AccessDecision.DENY,
                    {"reason": rate_reason},
                )
                record_rate_limit(request.principal.tenant_id)
                record_denial("rate_limit")
                return result

        # Step 5: Evaluate policies
        result = self.policy_engine.evaluate(request)
        evaluation_time = (time.time() - start_time) * 1000
        result.evaluation_time_ms = evaluation_time

        # Step 6: Audit logging
        if self.config.audit_enabled:
            if self.config.audit_all_decisions or result.decision == AccessDecision.DENY:
                evt_type = (
                    AuditEventType.ACCESS_GRANTED
                    if result.allowed
                    else AuditEventType.ACCESS_DENIED
                )
                self._log_event(
                    evt_type, request, result.decision,
                    {"matching_rules": result.matching_rules},
                )

        # Update internal metrics
        if result.allowed:
            self._allowed_requests += 1
        else:
            self._denied_requests += 1
            for reason in result.deny_reasons:
                record_denial("policy")

        # Record Prometheus metrics
        record_decision(
            request.action,
            result.decision.value,
            evaluation_time / 1000,
        )

        # Update gauge metrics
        update_policies_count(self.policy_engine.count)
        update_audit_events_count(self.audit_logger.count)

        # Cache the result
        self._decision_cache[cache_key] = (result, time.time())

        # Step 7: Handle simulation mode
        if self.config.simulation_mode:
            logger.info(
                "SIMULATION: Request %s would be %s",
                request.request_id,
                "ALLOWED" if result.allowed else "DENIED",
            )
            return AccessDecisionResult(
                request_id=result.request_id,
                decision=AccessDecision.ALLOW,
                allowed=True,
                matching_rules=result.matching_rules,
                deny_reasons=[
                    f"[SIMULATED] {r}" for r in result.deny_reasons
                ],
                evaluation_time_ms=result.evaluation_time_ms,
                policy_versions=result.policy_versions,
                decision_hash=result.decision_hash,
            )

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the access guard service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("AccessGuardService already started; skipping")
            return

        logger.info("AccessGuardService starting up...")
        self._started = True
        logger.info("AccessGuardService startup complete")

    def shutdown(self) -> None:
        """Shutdown the access guard service and release resources."""
        if not self._started:
            return

        self._decision_cache.clear()
        self._started = False
        logger.info("AccessGuardService shut down")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get access guard service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        total = self._total_requests
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_requests": total,
            "allowed_requests": self._allowed_requests,
            "denied_requests": self._denied_requests,
            "rate_limited_requests": self._rate_limited_requests,
            "allow_rate": (
                self._allowed_requests / total * 100 if total > 0 else 0
            ),
            "policies_loaded": self.policy_engine.count,
            "audit_events": self.audit_logger.count,
            "rego_policies": self.opa_client.count,
            "provenance_entries": self.provenance.entry_count,
            "cache_size": len(self._decision_cache),
            "classifier_patterns": self.classifier.count,
            "rate_limiter_buckets": self.rate_limiter.count,
            "simulation_mode": self.config.simulation_mode,
            "strict_mode": self.config.strict_mode,
            "strict_tenant_isolation": self.config.strict_tenant_isolation,
        }

    def clear_cache(self) -> None:
        """Clear the decision cache."""
        self._decision_cache.clear()
        logger.info("Decision cache cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_deny(
        self,
        request: AccessRequest,
        deny_reasons: List[str],
        start_time: float,
    ) -> AccessDecisionResult:
        """Create a DENY AccessDecisionResult.

        Args:
            request: The original request.
            deny_reasons: Reasons for denial.
            start_time: Processing start time.

        Returns:
            AccessDecisionResult with DENY decision.
        """
        evaluation_time = (time.time() - start_time) * 1000
        self._denied_requests += 1

        result = AccessDecisionResult(
            request_id=request.request_id,
            decision=AccessDecision.DENY,
            allowed=False,
            deny_reasons=deny_reasons,
            evaluation_time_ms=evaluation_time,
        )

        # Compute decision hash
        decision_str = json.dumps(
            {
                "request_id": request.request_id,
                "decision": "deny",
                "deny_reasons": deny_reasons,
                "timestamp": result.evaluated_at.isoformat(),
            },
            sort_keys=True,
        )
        result.decision_hash = hashlib.sha256(decision_str.encode()).hexdigest()

        # Prometheus
        record_decision(
            request.action, "deny", evaluation_time / 1000,
        )

        return result

    def _log_event(
        self,
        event_type: AuditEventType,
        request: AccessRequest,
        decision: AccessDecision,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit event from an access request.

        Args:
            event_type: Type of audit event.
            request: The access request.
            decision: The access decision.
            details: Optional additional details.
        """
        if not self.config.audit_enabled:
            return

        event = AuditEvent(
            event_type=event_type,
            tenant_id=request.resource.tenant_id,
            principal_id=request.principal.principal_id,
            resource_id=request.resource.resource_id,
            action=request.action,
            decision=decision,
            details=details or {},
            source_ip=request.source_ip,
            user_agent=request.user_agent,
            retention_days=self.config.audit_retention_days,
        )
        self.audit_logger.log_audit_event(event)

    def _cache_key(self, request: AccessRequest) -> str:
        """Generate a cache key for a request.

        Args:
            request: The access request.

        Returns:
            MD5-based cache key string.
        """
        key_parts = [
            request.principal.principal_id,
            request.principal.tenant_id,
            request.resource.resource_id,
            request.resource.tenant_id,
            request.action,
            str(sorted(request.principal.roles)),
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()

    def _check_cache(
        self, cache_key: str,
    ) -> Optional[AccessDecisionResult]:
        """Check if a cached result exists and is still valid.

        Args:
            cache_key: The cache key to check.

        Returns:
            Cached AccessDecisionResult if valid, None otherwise.
        """
        if cache_key not in self._decision_cache:
            return None

        cached_result, cached_time = self._decision_cache[cache_key]
        if time.time() - cached_time < self.config.decision_cache_ttl_seconds:
            return cached_result

        # Expired
        del self._decision_cache[cache_key]
        return None

    def _get_highest_role(self, request: AccessRequest) -> Optional[str]:
        """Get the highest role from the request principal for rate limiting.

        Args:
            request: The access request.

        Returns:
            Highest role string, or None.
        """
        for role in request.principal.roles:
            if role in self.rate_limiter.config.role_overrides:
                return role
        return None


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> AccessGuardService:
    """Get or create the singleton AccessGuardService instance.

    Returns:
        The singleton AccessGuardService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = AccessGuardService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


def configure_access_guard(
    app: Any,
    config: Optional[AccessGuardConfig] = None,
) -> AccessGuardService:
    """Configure the Access Guard Service on a FastAPI application.

    Creates the AccessGuardService, stores it in app.state, mounts
    the access guard API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional access guard config.

    Returns:
        AccessGuardService instance.
    """
    global _singleton_instance

    service = AccessGuardService(config=config)

    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.access_guard_service = service

    # Mount API router
    try:
        from greenlang.access_guard.api.router import router as guard_router
        if guard_router is not None:
            app.include_router(guard_router)
            logger.info("Access Guard API router mounted")
    except ImportError:
        logger.warning("Access Guard router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("Access Guard service configured on app")
    return service


def get_access_guard(app: Any) -> AccessGuardService:
    """Get the AccessGuardService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        AccessGuardService instance.

    Raises:
        RuntimeError: If access guard service not configured.
    """
    service = getattr(app.state, "access_guard_service", None)
    if service is None:
        raise RuntimeError(
            "Access Guard service not configured. "
            "Call configure_access_guard(app) first."
        )
    return service


def get_router() -> Any:
    """Get the access guard API router.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.access_guard.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "AccessGuardService",
    "configure_access_guard",
    "get_access_guard",
    "get_router",
]
