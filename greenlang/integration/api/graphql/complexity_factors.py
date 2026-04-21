# -*- coding: utf-8 -*-
"""
Factors-specific complexity costs for the GraphQL query planner (GAP-12).

Kept as a separate module so :mod:`greenlang.api.graphql.complexity`
remains untouched by the Factors GAP.  The :func:`apply_factor_costs`
helper merges these costs into an existing :class:`ComplexityConfig`
without disturbing the base weights.

Per-tier complexity caps recommended by the CTO:
    * community   — 500
    * pro         — 2_000
    * consulting  — 5_000
    * enterprise  — 10_000
    * internal    — unlimited (enforced by policy, not the planner)

The resolvers themselves emit the tier error; this module just
surfaces the expected query cost so API-gateway rate-limiters can
meter intelligently.
"""

from __future__ import annotations

from typing import Dict, Optional

# Lazy import — the complexity module isn't needed at module-load time.
# ``apply_factor_costs`` will import it on first use.
ComplexityConfig = None  # type: ignore


# ==============================================================================
# Field-level costs
# ==============================================================================
#
# Costs are *per-request* base values.  The planner multiplies lists by
# the ``first`` / ``limit`` argument (see ``_get_field_multiplier`` in
# the base module), so a ``factors(first: 100)`` call ends up at roughly
# ``5 * 100 = 500`` complexity, not 5.

FACTOR_FIELD_COSTS: Dict[str, int] = {
    # Cheap — single-record lookups.
    "factor": 2,
    "edition": 1,
    "source": 1,
    "methodPack": 1,
    "editions": 2,
    "sources": 2,
    "methodPacks": 2,
    # Moderate — list / search / coverage endpoints.
    "factors": 5,
    "searchFactors": 10,
    "factorCoverage": 8,
    "factorDiff": 5,
    "factorAlternates": 5,
    # Expensive — full 7-step cascade touches many records.
    "resolveFactor": 25,
    "resolveFactorExplain": 40,
    "factorAuditBundle": 30,  # Enterprise tier already rate-limits
    # Mutations — variable cost; the list argument drives the multiplier.
    "setFactorOverride": 5,
    "removeFactorOverride": 3,
    "matchFactor": 15,
    "submitBatchResolution": 100,
}


# ==============================================================================
# Tier-based maxima (pre-computed so callers don't have to duplicate
# the mapping across the authz / rate-limiter layers).
# ==============================================================================

TIER_COMPLEXITY_CAP: Dict[str, int] = {
    "community": 500,
    "pro": 2_000,
    "consulting": 5_000,
    "enterprise": 10_000,
    "internal": 1_000_000,  # effectively unlimited
}


def apply_factor_costs(config: Optional[Any] = None) -> Any:
    """Merge Factors field costs into an existing :class:`ComplexityConfig`.

    If ``config`` is ``None``, a fresh config is created with the base
    defaults.  Never mutates the input in-place beyond its ``field_costs``
    dict so callers can share the object safely.
    """
    global ComplexityConfig
    if ComplexityConfig is None:
        try:
            from greenlang.integration.api.graphql.complexity import (
                ComplexityConfig as _Cfg,
            )

            ComplexityConfig = _Cfg  # type: ignore[assignment]
        except ImportError as exc:
            raise RuntimeError(
                "greenlang.integration.api.graphql.complexity is "
                "unavailable — cannot apply Factors complexity costs."
            ) from exc
    if config is None:
        config = ComplexityConfig()

    # In-place merge (field_costs is a plain dict).
    for field_name, cost in FACTOR_FIELD_COSTS.items():
        config.field_costs.setdefault(field_name, cost)

    return config


def cap_for_tier(tier: str) -> int:
    """Return the complexity cap for a tier string (case-insensitive)."""
    return TIER_COMPLEXITY_CAP.get((tier or "community").lower(), 500)


__all__ = [
    "FACTOR_FIELD_COSTS",
    "TIER_COMPLEXITY_CAP",
    "apply_factor_costs",
    "cap_for_tier",
]
