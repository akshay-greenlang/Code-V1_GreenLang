# -*- coding: utf-8 -*-
"""
Tier-based factor visibility enforcement (F030).

Controls which factors are visible based on API tier:
- Community: certified only (no preview, no connector_only)
- Pro: certified + preview (no connector_only)
- Enterprise: certified + preview + connector_only
- Internal: all (including deprecated)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from greenlang.data.emission_factor_record import EmissionFactorRecord

logger = logging.getLogger(__name__)


class Tier(str, Enum):
    COMMUNITY = "community"
    PRO = "pro"
    CONSULTING = "consulting"        # Phase F8 — white-label / multi-tenant reseller
    ENTERPRISE = "enterprise"
    INTERNAL = "internal"


@dataclass
class TierVisibility:
    """Visibility flags derived from a tier."""

    include_preview: bool
    include_connector: bool
    include_deprecated: bool
    max_export_rows: int
    audit_bundle_allowed: bool
    bulk_export_allowed: bool

    @classmethod
    def from_tier(cls, tier: str) -> "TierVisibility":
        t = tier.lower().strip() if tier else "community"
        if t == "internal":
            return cls(
                include_preview=True,
                include_connector=True,
                include_deprecated=True,
                max_export_rows=0,  # unlimited
                audit_bundle_allowed=True,
                bulk_export_allowed=True,
            )
        if t == "enterprise":
            return cls(
                include_preview=True,
                include_connector=True,
                include_deprecated=False,
                max_export_rows=100_000,
                audit_bundle_allowed=True,
                bulk_export_allowed=True,
            )
        if t == "consulting":
            # White-label / reseller: multi-tenant, higher limits than pro,
            # private-registry + override rights but NO connector-only + NO
            # audit bundles (customer-facing audit lives at enterprise).
            return cls(
                include_preview=True,
                include_connector=False,
                include_deprecated=False,
                max_export_rows=50_000,
                audit_bundle_allowed=False,
                bulk_export_allowed=True,
            )
        if t == "pro":
            return cls(
                include_preview=True,
                include_connector=False,
                include_deprecated=False,
                max_export_rows=10_000,
                audit_bundle_allowed=False,
                bulk_export_allowed=True,
            )
        # community (default)
        return cls(
            include_preview=False,
            include_connector=False,
            include_deprecated=False,
            max_export_rows=1_000,
            audit_bundle_allowed=False,
            bulk_export_allowed=False,
        )


def resolve_tier(user_context: Optional[Dict[str, Any]]) -> str:
    """Extract tier string from user context dict."""
    if not user_context:
        return Tier.COMMUNITY.value
    raw = user_context.get("tier", "community")
    try:
        return Tier(raw.lower().strip()).value
    except ValueError:
        logger.warning("Unknown tier %r, defaulting to community", raw)
        return Tier.COMMUNITY.value


def factor_visible_for_tier(
    factor_status: str,
    visibility: TierVisibility,
) -> bool:
    """Check if a factor with given status is visible under tier visibility rules."""
    st = (factor_status or "certified").lower()
    if st == "certified":
        return True
    if st == "preview":
        return visibility.include_preview
    if st == "connector_only":
        return visibility.include_connector
    if st == "deprecated":
        return visibility.include_deprecated
    return False


# ---------------------------------------------------------------------------
# N7 invariant: Community tier is forbidden from oem_redistributable and
# customer_private classes REGARDLESS of any per-caller grant. This is the
# final line of defense behind the licensing guard — if a mis-wired
# entitlement row accidentally carries a premium class for a community
# caller, this check blocks it at the tier layer.
# ---------------------------------------------------------------------------

# Redistribution classes the Community tier is never allowed to see.
COMMUNITY_FORBIDDEN_CLASSES = frozenset(
    {
        "oem_redistributable",
        "oem-redistributable",
        "customer_private",
        "customer-private",
        "private",
    }
)


def tier_allows_class(tier: str, redistribution_class: str) -> bool:
    """Return True iff ``tier`` is allowed to receive ``redistribution_class``.

    **N7 invariant**: Community tier can NEVER receive ``customer_private``
    or ``oem_redistributable`` factors, even if a per-caller grant (e.g. a
    mis-wired entitlement row) would otherwise allow it.

    This function is a pure, deterministic check used by
    :class:`LicensingGuardMiddleware` as a final safety net behind the
    class-rank grant logic.
    """
    if not redistribution_class:
        return True
    klass = redistribution_class.strip().lower()
    t = (tier or "").strip().lower() or Tier.COMMUNITY.value
    if t == Tier.COMMUNITY.value and klass in COMMUNITY_FORBIDDEN_CLASSES:
        return False
    return True


def assert_tier_can_read_class(tier: str, redistribution_class: str) -> None:
    """Raise PermissionError if the tier is forbidden from this class.

    Thin wrapper around :func:`tier_allows_class` for call-sites that
    prefer exceptions over booleans.
    """
    if not tier_allows_class(tier, redistribution_class):
        raise PermissionError(
            f"Tier {tier!r} is forbidden from reading factors of class "
            f"{redistribution_class!r} (N7 open-core boundary)."
        )


def enforce_tier_on_request(
    user_context: Optional[Dict[str, Any]],
    *,
    requested_preview: bool = False,
    requested_connector: bool = False,
) -> TierVisibility:
    """
    Resolve tier and clamp requested visibility flags.

    If user requests include_preview=True but tier doesn't allow it,
    the flag is silently clamped to False (no error).
    """
    tier = resolve_tier(user_context)
    tv = TierVisibility.from_tier(tier)
    # Clamp: user can't request more than tier allows
    tv.include_preview = requested_preview and tv.include_preview
    tv.include_connector = requested_connector and tv.include_connector
    logger.debug(
        "Tier enforcement: tier=%s preview=%s connector=%s",
        tier, tv.include_preview, tv.include_connector,
    )
    return tv


def filter_factors_by_tier(
    factors: Sequence[Any],
    visibility: TierVisibility,
) -> List[Any]:
    """Filter a sequence of factors (records or dicts) by tier visibility."""
    result = []
    for f in factors:
        if isinstance(f, dict):
            st = f.get("factor_status", "certified") or "certified"
        else:
            st = getattr(f, "factor_status", "certified") or "certified"
        if factor_visible_for_tier(st, visibility):
            result.append(f)
    return result
