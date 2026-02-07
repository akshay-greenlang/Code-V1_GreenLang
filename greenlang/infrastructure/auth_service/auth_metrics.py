# -*- coding: utf-8 -*-
"""
Auth Metrics - JWT Authentication Service (SEC-001)

Prometheus counters, gauges, and histograms for authentication observability.
Metrics are lazily initialized on first use so that the module can be imported
even when ``prometheus_client`` is not installed (metrics become no-ops).

Registered metrics:
    - gl_auth_login_total (Counter): Login attempts by result/method/tenant.
    - gl_auth_token_issued_total (Counter): Tokens issued by type/tenant.
    - gl_auth_token_validated_total (Counter): Token validations by result/method.
    - gl_auth_token_revoked_total (Counter): Token revocations by reason/tenant.
    - gl_auth_lockout_total (Counter): Account lockouts by tenant.
    - gl_auth_permission_denied_total (Counter): Permission denials by perm/tenant.
    - gl_auth_mfa_verification_total (Counter): MFA verifications by result/tenant.
    - gl_auth_password_change_total (Counter): Password changes by reason/tenant.
    - gl_auth_active_sessions (Gauge): Active session count by tenant.
    - gl_auth_login_duration_seconds (Histogram): Login latency by result/method.
    - gl_auth_token_validation_duration_seconds (Histogram): Validation latency.

Classes:
    - AuthMetrics: Singleton-style metrics manager.

Example:
    >>> metrics = AuthMetrics()
    >>> metrics.record_login("success", "password", "t-corp", duration_s=0.045)
    >>> metrics.record_token_issued("access", "t-corp")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: lazy Prometheus metric handles
# ---------------------------------------------------------------------------


class _PrometheusHandles:
    """Lazy-initialized Prometheus metric objects.

    Metrics are created on first call to :meth:`ensure_initialized`.
    If ``prometheus_client`` is not installed, all handles remain ``None``
    and recording methods become safe no-ops.
    """

    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()
    _available: bool = False

    # Counters
    login_total: Any = None
    token_issued_total: Any = None
    token_validated_total: Any = None
    token_revoked_total: Any = None
    lockout_total: Any = None
    permission_denied_total: Any = None
    mfa_verification_total: Any = None
    password_change_total: Any = None

    # Gauges
    active_sessions: Any = None

    # Histograms
    login_duration_seconds: Any = None
    token_validation_duration: Any = None

    @classmethod
    def ensure_initialized(cls) -> bool:
        """Create Prometheus metrics if the library is available.

        Thread-safe via a class-level lock.

        Returns:
            ``True`` if prometheus_client is available and metrics are
            registered, ``False`` otherwise.
        """
        if cls._initialized:
            return cls._available

        with cls._lock:
            if cls._initialized:
                return cls._available

            cls._initialized = True

            try:
                from prometheus_client import Counter, Gauge, Histogram
            except ImportError:
                logger.info(
                    "prometheus_client not installed; auth metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_auth"

            # -- Counters --------------------------------------------------

            cls.login_total = Counter(
                f"{prefix}_login_total",
                "Total login attempts",
                ["result", "method", "tenant_id"],
            )

            cls.token_issued_total = Counter(
                f"{prefix}_token_issued_total",
                "Total tokens issued",
                ["type", "tenant_id"],
            )

            cls.token_validated_total = Counter(
                f"{prefix}_token_validated_total",
                "Total token validations",
                ["result", "method"],
            )

            cls.token_revoked_total = Counter(
                f"{prefix}_token_revoked_total",
                "Total token revocations",
                ["reason", "tenant_id"],
            )

            cls.lockout_total = Counter(
                f"{prefix}_lockout_total",
                "Total account lockouts",
                ["tenant_id"],
            )

            cls.permission_denied_total = Counter(
                f"{prefix}_permission_denied_total",
                "Total permission denials",
                ["permission", "tenant_id"],
            )

            cls.mfa_verification_total = Counter(
                f"{prefix}_mfa_verification_total",
                "Total MFA verification attempts",
                ["result", "tenant_id"],
            )

            cls.password_change_total = Counter(
                f"{prefix}_password_change_total",
                "Total password changes",
                ["reason", "tenant_id"],
            )

            # -- Gauges ----------------------------------------------------

            cls.active_sessions = Gauge(
                f"{prefix}_active_sessions",
                "Number of active sessions",
                ["tenant_id"],
            )

            # -- Histograms ------------------------------------------------

            cls.login_duration_seconds = Histogram(
                f"{prefix}_login_duration_seconds",
                "Login processing duration in seconds",
                ["result", "method"],
                buckets=(
                    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                ),
            )

            cls.token_validation_duration = Histogram(
                f"{prefix}_token_validation_duration_seconds",
                "Token validation duration in seconds",
                ["result"],
                buckets=(
                    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1,
                ),
            )

            cls._available = True
            logger.info("Auth Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# AuthMetrics
# ---------------------------------------------------------------------------


class AuthMetrics:
    """Manages Prometheus metrics for the auth service.

    All recording methods are safe no-ops when ``prometheus_client`` is
    not installed.  Thread-safe.

    Example:
        >>> m = AuthMetrics()
        >>> m.record_login("success", "password", "t-1", duration_s=0.03)
        >>> m.record_lockout("t-1")
    """

    def __init__(self, prefix: str = "gl_auth") -> None:
        """Initialize auth metrics.

        Args:
            prefix: Metric name prefix (used for documentation only;
                actual prefix is fixed at registration time).
        """
        self._prefix = prefix
        self._available = _PrometheusHandles.ensure_initialized()

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    def record_login(
        self,
        result: str,
        method: str,
        tenant_id: str,
        duration_s: float,
    ) -> None:
        """Record a login attempt.

        Args:
            result: Outcome (``success``, ``failure``, ``locked``).
            method: Auth method (``password``, ``sso``, ``api_key``).
            tenant_id: UUID of the tenant.
            duration_s: Login processing duration in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.login_total.labels(
            result=result, method=method, tenant_id=tenant_id
        ).inc()

        _PrometheusHandles.login_duration_seconds.labels(
            result=result, method=method
        ).observe(duration_s)

    # ------------------------------------------------------------------
    # Tokens
    # ------------------------------------------------------------------

    def record_token_issued(self, token_type: str, tenant_id: str) -> None:
        """Record a token issuance.

        Args:
            token_type: Token type (``access``, ``refresh``).
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.token_issued_total.labels(
            type=token_type, tenant_id=tenant_id
        ).inc()

    def record_token_validated(
        self,
        result: str,
        method: str,
        duration_s: float,
    ) -> None:
        """Record a token validation.

        Args:
            result: Outcome (``valid``, ``expired``, ``revoked``, ``invalid``).
            method: Validation method (``local``, ``jwks``).
            duration_s: Validation duration in seconds.
        """
        if not self._available:
            return

        _PrometheusHandles.token_validated_total.labels(
            result=result, method=method
        ).inc()

        _PrometheusHandles.token_validation_duration.labels(
            result=result
        ).observe(duration_s)

    def record_token_revoked(self, reason: str, tenant_id: str) -> None:
        """Record a token revocation.

        Args:
            reason: Revocation reason (``logout``, ``admin``, ``rotation``,
                ``compromise``).
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.token_revoked_total.labels(
            reason=reason, tenant_id=tenant_id
        ).inc()

    # ------------------------------------------------------------------
    # Lockout
    # ------------------------------------------------------------------

    def record_lockout(self, tenant_id: str) -> None:
        """Record an account lockout event.

        Args:
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.lockout_total.labels(tenant_id=tenant_id).inc()

    # ------------------------------------------------------------------
    # Permission denied
    # ------------------------------------------------------------------

    def record_permission_denied(
        self,
        permission: str,
        tenant_id: str,
    ) -> None:
        """Record a permission denial.

        Args:
            permission: The permission that was required.
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.permission_denied_total.labels(
            permission=permission, tenant_id=tenant_id
        ).inc()

    # ------------------------------------------------------------------
    # MFA
    # ------------------------------------------------------------------

    def record_mfa_verification(
        self,
        result: str,
        tenant_id: str,
    ) -> None:
        """Record an MFA verification attempt.

        Args:
            result: Outcome (``success``, ``failure``).
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.mfa_verification_total.labels(
            result=result, tenant_id=tenant_id
        ).inc()

    # ------------------------------------------------------------------
    # Password change
    # ------------------------------------------------------------------

    def record_password_change(
        self,
        reason: str,
        tenant_id: str,
    ) -> None:
        """Record a password change.

        Args:
            reason: Change reason (``user_change``, ``admin_reset``, ``expiry``).
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.password_change_total.labels(
            reason=reason, tenant_id=tenant_id
        ).inc()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def set_active_sessions(self, tenant_id: str, count: int) -> None:
        """Set the active session gauge for a tenant.

        Args:
            tenant_id: UUID of the tenant.
            count: Current active session count.
        """
        if not self._available:
            return

        _PrometheusHandles.active_sessions.labels(
            tenant_id=tenant_id
        ).set(count)

    def inc_active_sessions(self, tenant_id: str) -> None:
        """Increment the active session gauge by 1.

        Args:
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.active_sessions.labels(
            tenant_id=tenant_id
        ).inc()

    def dec_active_sessions(self, tenant_id: str) -> None:
        """Decrement the active session gauge by 1.

        Args:
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.active_sessions.labels(
            tenant_id=tenant_id
        ).dec()
