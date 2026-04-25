# -*- coding: utf-8 -*-
"""
GreenLang Factors — release-profile feature gate.

Single source of truth for which surfaces are mounted at runtime in the
v0.1 Alpha launch (and beyond). Driven by the ``GL_FACTORS_RELEASE_PROFILE``
environment variable.

Profiles (ordered, lowest-to-highest):

    dev          - local development (ALL features enabled, used in tests).
    alpha-v0.1   - first public release; minimal surface only.
    beta-v0.5    - opens resolve/explain/batch/coverage/fqs/edition + signed
                   receipts + admin console + TS SDK + extended CLI.
    rc-v0.9      - adds GraphQL and ML resolve.
    ga-v1.0      - opens billing, OEM, SQL-over-HTTP, commercial packs,
                   real-time grid.

The five alpha-allowed v1 endpoints (always-on regardless of profile):

    GET /v1/healthz     (currently /v1/health — rename pending task #13)
    GET /v1/factors
    GET /v1/factors/{urn}
    GET /v1/sources
    GET /v1/packs

Default resolution:

    if GL_FACTORS_RELEASE_PROFILE is set      -> use that value.
    elif GL_ENV == "production" (or APP_ENV / ENVIRONMENT) -> alpha-v0.1.
    else                                      -> dev.

Usage:

    from greenlang.factors.release_profile import (
        ReleaseProfile, current_profile, is_alpha, feature_enabled,
    )
    if feature_enabled("graphql"):
        app.include_router(graphql_router)
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Dict, FrozenSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class ReleaseProfile(str, Enum):
    """Ordered release profiles. Order is established by ``_PROFILE_ORDER``."""

    DEV = "dev"
    ALPHA_V0_1 = "alpha-v0.1"
    BETA_V0_5 = "beta-v0.5"
    RC_V0_9 = "rc-v0.9"
    GA_V1_0 = "ga-v1.0"


# Profile total order. ``DEV`` is treated as "above GA" — it enables every
# feature so existing test suites keep passing without per-test env wiring.
_PROFILE_ORDER: Dict[ReleaseProfile, int] = {
    ReleaseProfile.ALPHA_V0_1: 0,
    ReleaseProfile.BETA_V0_5: 1,
    ReleaseProfile.RC_V0_9: 2,
    ReleaseProfile.GA_V1_0: 3,
    ReleaseProfile.DEV: 99,  # "all features on"
}


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------
#
# Each feature lists the *minimum* profile at which it becomes enabled. Any
# profile at or above that ordering threshold turns the feature ON. The DEV
# profile sits at rank 99 so every feature is on locally and in tests.

FEATURES: Dict[str, Dict[str, ReleaseProfile]] = {
    # Resolution / explain / batch / coverage / fqs / edition surfaces are
    # the "v0.5 Beta" public API.
    "resolve_endpoint":   {"min_profile": ReleaseProfile.BETA_V0_5},
    "explain_endpoint":   {"min_profile": ReleaseProfile.BETA_V0_5},
    "batch_endpoint":     {"min_profile": ReleaseProfile.BETA_V0_5},
    "edition_endpoint":   {"min_profile": ReleaseProfile.BETA_V0_5},
    "coverage_endpoint":  {"min_profile": ReleaseProfile.BETA_V0_5},
    "fqs_endpoint":       {"min_profile": ReleaseProfile.BETA_V0_5},
    "signed_receipts":    {"min_profile": ReleaseProfile.BETA_V0_5},
    # Method-pack coverage routes are part of the explain/coverage surface.
    "method_packs":       {"min_profile": ReleaseProfile.BETA_V0_5},
    "admin_console":      {"min_profile": ReleaseProfile.BETA_V0_5},
    "ts_sdk":             {"min_profile": ReleaseProfile.BETA_V0_5},
    "cli_extended":       {"min_profile": ReleaseProfile.BETA_V0_5},
    # GraphQL + ML resolve are RC-stage.
    "graphql":            {"min_profile": ReleaseProfile.RC_V0_9},
    "ml_resolve":         {"min_profile": ReleaseProfile.RC_V0_9},
    # GA-only commercial / infra surfaces.
    "sql_over_http":      {"min_profile": ReleaseProfile.GA_V1_0},
    "billing":            {"min_profile": ReleaseProfile.GA_V1_0},
    "oem":                {"min_profile": ReleaseProfile.GA_V1_0},
    "commercial_packs":   {"min_profile": ReleaseProfile.GA_V1_0},
    "real_time_grid":     {"min_profile": ReleaseProfile.GA_V1_0},
}


# Always-on alpha-allowed v1 paths. Routes whose path is in this set (or
# matches one of these via templated parameters) are NEVER gated, regardless
# of profile. This is the public surface for v0.1 Alpha.
ALPHA_ALLOWED_PATHS: FrozenSet[str] = frozenset({
    "/v1/healthz",
    "/v1/health",  # legacy — rename pending task #13.
    "/v1/factors",
    "/v1/factors/{factor_id}",
    "/v1/factors/{urn}",
    "/v1/sources",
    "/v1/packs",
})


# Map of feature name -> path prefixes that should be culled from the mounted
# api_v1_router when the feature is disabled. Used by ``filter_app_routes``
# to enforce alpha minimality without modifying api_v1_routes.py source.
_FEATURE_PATH_PREFIXES: Dict[str, tuple] = {
    "resolve_endpoint":  ("/v1/resolve",),
    "explain_endpoint":  ("/v1/explain", "/v1/factors/{factor_id}/explain"),
    "batch_endpoint":    ("/v1/batch",),
    "coverage_endpoint": ("/v1/coverage",),
    "fqs_endpoint":      ("/v1/quality/fqs", "/v1/quality"),
    "edition_endpoint":  ("/v1/editions",),
    # ``/v1/health/signing-status`` is part of the signed-receipts surface,
    # not the always-on liveness probe.
    "signed_receipts":   ("/v1/health/signing-status",),
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


_ENV_VAR = "GL_FACTORS_RELEASE_PROFILE"


def _normalize(value: str) -> str:
    """Lowercase, strip whitespace, and unify common aliases."""
    v = (value or "").strip().lower()
    aliases = {
        "alpha": "alpha-v0.1",
        "beta": "beta-v0.5",
        "rc": "rc-v0.9",
        "ga": "ga-v1.0",
        "production": "ga-v1.0",
        "prod": "ga-v1.0",
    }
    return aliases.get(v, v)


def current_profile() -> ReleaseProfile:
    """Return the active :class:`ReleaseProfile`.

    Resolution rules (in order):

    1. If ``GL_FACTORS_RELEASE_PROFILE`` is set and parses to a known
       profile, use it.
    2. Else if ``GL_ENV`` / ``APP_ENV`` / ``ENVIRONMENT`` indicates
       production, fall back to :data:`ReleaseProfile.ALPHA_V0_1`.
    3. Otherwise default to :data:`ReleaseProfile.DEV`.

    Unknown values are logged once and treated as ``alpha-v0.1`` in
    production-like envs and ``dev`` elsewhere — we never crash on a typo
    in the env var because that would brick the whole API.
    """
    raw = os.getenv(_ENV_VAR)
    if raw:
        normalized = _normalize(raw)
        try:
            return ReleaseProfile(normalized)
        except ValueError:
            logger.warning(
                "Unknown %s=%r; falling back to default profile selection.",
                _ENV_VAR, raw,
            )

    app_env = (
        os.getenv("GL_ENV")
        or os.getenv("APP_ENV")
        or os.getenv("ENVIRONMENT")
        or ""
    ).strip().lower()
    if app_env in {"production", "prod"}:
        return ReleaseProfile.ALPHA_V0_1
    return ReleaseProfile.DEV


def is_alpha() -> bool:
    """Return True iff the active profile is exactly ``alpha-v0.1``."""
    return current_profile() == ReleaseProfile.ALPHA_V0_1


def feature_enabled(feature: str) -> bool:
    """Return True iff ``feature`` is enabled in the current profile.

    Unknown features are conservatively reported as **disabled** — this
    makes a typo a "feature off" bug, which is safer than accidentally
    leaking a surface in alpha.
    """
    spec = FEATURES.get(feature)
    if spec is None:
        logger.debug("feature_enabled(%r): unknown feature, treating as off.", feature)
        return False
    min_profile: ReleaseProfile = spec["min_profile"]
    active = current_profile()
    return _PROFILE_ORDER[active] >= _PROFILE_ORDER[min_profile]


# ---------------------------------------------------------------------------
# Route filtering
# ---------------------------------------------------------------------------


def _path_is_alpha_allowed(path: str) -> bool:
    """True if ``path`` is in the always-on alpha allow-list.

    Matches by exact path (after parameter normalization). FastAPI exposes
    the templated path on ``route.path``, e.g. ``/v1/factors/{factor_id}``,
    so we accept either the canonical placeholder ``{urn}`` or the
    repository's current ``{factor_id}``.
    """
    return path in ALPHA_ALLOWED_PATHS


def _path_matches_feature(path: str, feature: str) -> bool:
    """True if ``path`` belongs to the named feature's prefix list."""
    prefixes = _FEATURE_PATH_PREFIXES.get(feature, ())
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "{"):
            return True
        # Handle templated suffixes: ``/v1/factors/{factor_id}/explain``
        if "{" in prefix and path.startswith(prefix.split("{", 1)[0]) and path.endswith(prefix.rsplit("}", 1)[-1]):
            return True
    return False


def filter_app_routes(app) -> None:  # type: ignore[no-untyped-def]
    """Cull routes from ``app.routes`` whose feature is disabled.

    Walks every mounted route and drops any whose path:

    * is NOT in :data:`ALPHA_ALLOWED_PATHS`, AND
    * matches a path-prefix of a feature for which
      :func:`feature_enabled` returns ``False``.

    This lets us keep ``api_v1_router`` un-touched while still presenting
    a minimal v0.1-Alpha surface at runtime. Idempotent.

    Routes without a ``.path`` attribute (e.g. websocket / mounts) are
    skipped untouched.
    """
    # If we're at DEV / GA / above-RC where every feature is on, fast-path.
    if all(feature_enabled(f) for f in _FEATURE_PATH_PREFIXES):
        return

    keep = []
    dropped = []
    for route in list(app.routes):
        path = getattr(route, "path", None)
        if not isinstance(path, str):
            keep.append(route)
            continue

        if _path_is_alpha_allowed(path):
            keep.append(route)
            continue

        gated = False
        for feature in _FEATURE_PATH_PREFIXES:
            if not feature_enabled(feature) and _path_matches_feature(path, feature):
                gated = True
                dropped.append((path, feature))
                break

        if not gated:
            keep.append(route)

    if dropped:
        logger.info(
            "release_profile=%s: dropped %d gated route(s): %s",
            current_profile().value,
            len(dropped),
            ", ".join(f"{p} (feature={f})" for p, f in dropped[:10]),
        )
    app.router.routes[:] = keep


__all__ = [
    "ReleaseProfile",
    "FEATURES",
    "ALPHA_ALLOWED_PATHS",
    "current_profile",
    "is_alpha",
    "feature_enabled",
    "filter_app_routes",
]
