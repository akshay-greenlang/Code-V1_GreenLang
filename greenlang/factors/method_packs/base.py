# -*- coding: utf-8 -*-
"""Method pack base classes (Phase F2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, FrozenSet, List, Optional, Tuple

from greenlang.data.canonical_v2 import (
    ElectricityBasis,
    FactorFamily,
    FormulaType,
    MethodProfile,
)


class CannotResolveAction(str, Enum):
    """What the resolver does when no candidate survives the selection gate.

    Values mirror the ``fallback.cannot_resolve_action`` field in
    ``docs/specs/method_pack_template.md`` §9.
    """

    RAISE_NO_SAFE_MATCH = "raise_no_safe_match"
    ALLOW_GLOBAL_DEFAULT = "allow_global_default"


class BiogenicTreatment(str, Enum):
    """How a method pack treats biogenic CO2."""

    EXCLUDED = "excluded"                  # Scope 1 fossil-only (GHG Protocol)
    REPORTED_SEPARATELY = "reported_separately"  # Most corporate reporting
    NET_ZERO_ASSUMED = "net_zero_assumed"  # Legacy convention
    INCLUDED = "included"                  # Full lifecycle products


class MarketInstrumentTreatment(str, Enum):
    """Handling of RECs / GOs / PPAs in market-based accounting."""

    NOT_APPLICABLE = "not_applicable"
    ALLOWED = "allowed"                    # market-based Scope 2
    REQUIRE_CERTIFICATE = "require_certificate"
    PROHIBITED = "prohibited"              # location-based Scope 2


@dataclass(frozen=True)
class SelectionRule:
    """Selection rule: which factor families + statuses apply.

    Evaluated at resolution time AFTER the candidate list is generated;
    rules filter the candidates down to what the pack will accept.
    """

    allowed_families: Tuple[FactorFamily, ...]
    allowed_formula_types: Tuple[FormulaType, ...]
    require_verification: bool = False
    require_primary_data: bool = False
    # Only factors in these statuses are eligible for *this* pack.
    allowed_statuses: Tuple[str, ...] = ("certified",)
    # Optional predicate — returns True to keep the candidate.
    custom_filter: Optional[Callable[[Any], bool]] = None
    # Structured activity-category allow / deny lists (MP3/MP4/MP5). Empty
    # frozenset = "no restriction". Populated per-pack in the three P0 method
    # modules — see docs/specs/method_pack_template.md §4.
    included_activity_categories: FrozenSet[str] = field(default_factory=frozenset)
    excluded_activity_categories: FrozenSet[str] = field(default_factory=frozenset)

    def accepts(self, record: Any) -> bool:
        status = str(getattr(record, "factor_status", "certified") or "certified")
        if status not in self.allowed_statuses:
            return False

        # Family check — if the record has no factor_family set, treat as
        # EMISSIONS family (legacy default) so existing YAML still matches.
        fam = getattr(record, "factor_family", None) or FactorFamily.EMISSIONS.value
        if self.allowed_families and fam not in {f.value for f in self.allowed_families}:
            return False

        # Formula-type check — records without explicit formula_type pass.
        ftype = getattr(record, "formula_type", None)
        if (
            self.allowed_formula_types
            and ftype is not None
            and ftype not in {t.value for t in self.allowed_formula_types}
        ):
            return False

        if self.require_verification:
            v = getattr(record, "verification", None)
            status_ok = v is not None and str(getattr(v, "status", "unverified")) in (
                "external_verified",
                "regulator_approved",
            )
            if not status_ok:
                return False

        if self.require_primary_data:
            pdf = str(getattr(record, "primary_data_flag", "") or "").lower()
            if pdf not in ("primary", "primary_modeled"):
                return False

        # Activity-category inclusion / exclusion gate. We look at either the
        # first-class ``activity_category`` attribute if present, or fall back
        # to the record's ``category`` / ``activity`` fields.
        #
        # Backwards-compat note: a record that carries NO category attribute
        # at all (legacy records that predate structured categorisation) is
        # allowed through even when the pack sets an inclusion allow-list.
        # The inclusion gate only rejects records that explicitly declare a
        # category outside the allow-list. This keeps the new contract from
        # breaking existing fixtures and gold-label records while still
        # providing a defensive denylist for the cases the auditor cares
        # about.
        has_category_attr = any(
            hasattr(record, name) for name in ("activity_category", "category", "activity")
        )
        rec_cat = (
            getattr(record, "activity_category", None)
            or getattr(record, "category", None)
            or getattr(record, "activity", None)
        )
        rec_cat_str = str(rec_cat).strip() if rec_cat is not None else ""
        if self.excluded_activity_categories and rec_cat_str in self.excluded_activity_categories:
            return False
        if self.included_activity_categories and has_category_attr and rec_cat_str:
            if rec_cat_str not in self.included_activity_categories:
                return False

        if self.custom_filter is not None and not self.custom_filter(record):
            return False

        return True


@dataclass(frozen=True)
class BoundaryRule:
    """Emission-boundary rule for the pack (e.g. Scope 1 fossil-only)."""

    allowed_scopes: Tuple[str, ...]        # e.g. ("1",) or ("2",) or ("1","2","3")
    allowed_boundaries: Tuple[str, ...]    # e.g. ("combustion",) or ("cradle_to_gate",)
    biogenic_treatment: BiogenicTreatment = BiogenicTreatment.REPORTED_SEPARATELY
    market_instruments: MarketInstrumentTreatment = MarketInstrumentTreatment.NOT_APPLICABLE
    include_transmission_losses: Optional[bool] = None


@dataclass(frozen=True)
class FallbackStep:
    """One step of the per-pack region-hierarchy fallback."""

    rank: int                              # 1 = most preferred (customer override)
    label: str                             # "customer_override", "supplier_specific", ...
    description: str


@dataclass(frozen=True)
class DeprecationRule:
    """Sunset policy for stale factors."""

    max_age_days: int                      # any certified factor older than this goes to preview
    grace_period_days: int = 180           # how long deprecated factors remain resolvable


@dataclass(frozen=True)
class MethodPack:
    """A commercial methodology profile.

    Instances are immutable (frozen dataclasses) + registered in the global
    :func:`~greenlang.factors.method_packs.registry.register_pack` registry.
    """

    profile: MethodProfile
    name: str
    description: str
    selection_rule: SelectionRule
    boundary_rule: BoundaryRule
    gwp_basis: str                         # e.g. "IPCC_AR6_100"
    region_hierarchy: Tuple[FallbackStep, ...]
    deprecation: DeprecationRule
    reporting_labels: Tuple[str, ...]      # e.g. ("GHG_Protocol", "IFRS_S2")
    audit_text_template: str               # Markdown-ish template for Explain
    # Default electricity basis when the profile implies one (Scope 2 packs).
    electricity_basis: Optional[ElectricityBasis] = None
    # Pack version — bumped on any material rule change.  Tracked in CI.
    pack_version: str = "1.0.0"
    # Free-form tags — used by SKU entitlement (Phase F8).
    tags: Tuple[str, ...] = field(default_factory=tuple)
    # --- cannot_resolve_safely contract --------------------------------
    # Strict default: certified packs MUST raise a structured error when
    # no candidate survives the selection gate; opting into
    # ``ALLOW_GLOBAL_DEFAULT`` preserves the pre-FY27 legacy behaviour
    # where the 7-tier cascade's global default was returned silently.
    # See docs/specs/method_pack_template.md §9.
    cannot_resolve_action: CannotResolveAction = CannotResolveAction.RAISE_NO_SAFE_MATCH
    # When false the resolver is not allowed to return a rank-7 global
    # default, even if a candidate exists at that tier. Certified
    # regulatory packs (CBAM, Battery, DPP) MUST keep this False.
    global_default_tier_allowed: bool = False
    # --- deprecation policy structured fields --------------------------
    # ``None`` = no successor declared yet (acceptable while pack is
    # active; required before a pack can be flipped to ``deprecated``).
    replacement_pack_id: Optional[str] = None
    # Advance notice period before a deprecated pack stops resolving.
    # Default 180 days matches docs/specs/method_pack_template.md §12.
    deprecation_notice_days: int = 180


# Standard 7-step fallback chain used by most packs.
DEFAULT_FALLBACK = (
    FallbackStep(1, "customer_override", "Tenant-supplied factor overlay"),
    FallbackStep(2, "supplier_specific", "Supplier or manufacturer disclosure"),
    FallbackStep(3, "facility_specific", "Facility / asset-specific measurement"),
    FallbackStep(4, "utility_or_grid_subregion", "Utility tariff or grid sub-region"),
    FallbackStep(5, "country_or_sector_average", "National / sectoral average"),
    FallbackStep(6, "method_pack_default", "Pack-defined default for this method profile"),
    FallbackStep(7, "global_default", "Global default (lowest quality, last resort)"),
)


__all__ = [
    "BiogenicTreatment",
    "BoundaryRule",
    "CannotResolveAction",
    "DeprecationRule",
    "DEFAULT_FALLBACK",
    "FallbackStep",
    "MarketInstrumentTreatment",
    "MethodPack",
    "SelectionRule",
]
