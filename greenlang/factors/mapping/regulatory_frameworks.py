# -*- coding: utf-8 -*-
"""Regulatory-framework tagger.

Factors in the catalog carry a narrow set of compliance framework hints
today: the CTO spec requires every factor to be queryable by the
framework(s) it supports (CBAM, ESRS E1, TCFD, CSRD, SBTi, PCAF, ISO,
CA SB 253, etc.).

This module centralises the applicability rules and exposes three
surfaces:

- :func:`tag_factor` — return the list of frameworks a single factor-like
  object satisfies.
- :func:`tag_factor_batch` — convenience wrapper for bulk tagging.
- :class:`FrameworkIndex` — invert the tagging so the API layer can
  answer "which factors apply to CBAM?" without re-scanning every row.

Rules are encoded as dataclasses for easy auditing and are intentionally
conservative: we only report applicability when at least one factor
attribute positively matches (never on absence).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Framework catalog
# ---------------------------------------------------------------------------


class RegulatoryFramework(str, Enum):
    """Frameworks we can tag factors against."""

    # GHG Protocol family
    GHG_PROTOCOL_CORPORATE = "ghg_protocol_corporate"
    GHG_PROTOCOL_SCOPE2_LB = "ghg_protocol_scope2_location_based"
    GHG_PROTOCOL_SCOPE2_MB = "ghg_protocol_scope2_market_based"
    GHG_PROTOCOL_SCOPE3 = "ghg_protocol_scope3"
    GHG_PROTOCOL_PRODUCT = "ghg_protocol_product"

    # ISO family
    ISO_14064_1 = "iso_14064_1"
    ISO_14064_2 = "iso_14064_2"
    ISO_14067 = "iso_14067"
    ISO_14083 = "iso_14083"

    # EU regulatory
    CBAM = "eu_cbam"
    CSRD = "eu_csrd"
    ESRS_E1 = "eu_esrs_e1"
    EU_TAXONOMY = "eu_taxonomy"
    PEF = "eu_pef"
    OEF = "eu_oef"

    # UK
    PAS_2050 = "uk_pas_2050"
    SECR = "uk_secr"

    # US / state
    SEC_CLIMATE = "us_sec_climate"
    CA_SB253 = "us_ca_sb253"
    CA_SB261 = "us_ca_sb261"

    # Global voluntary
    TCFD = "tcfd"
    IFRS_S2 = "ifrs_s2"
    CDP = "cdp"
    SBTi = "sbti"
    PCAF = "pcaf"


class FrameworkScope(str, Enum):
    """Applicable scope under the framework."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    PRODUCT = "product"       # product carbon / embodied / LCA
    ENTITY = "entity"         # corporate-wide disclosure, no single scope
    FINANCED = "financed"     # PCAF


ActivityFamily = str
"""Canonical activity family name (combustion, electricity, transport,
purchased_goods, waste, land_use, refrigerants, finance_proxy,
fugitive, process, ...)."""


# ---------------------------------------------------------------------------
# Applicability rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameworkApplicability:
    """Describes when a factor is in-scope for a given framework."""

    framework: RegulatoryFramework
    scopes: FrozenSet[FrameworkScope] = field(default_factory=frozenset)
    activity_families: FrozenSet[ActivityFamily] = field(default_factory=frozenset)
    jurisdictions: FrozenSet[str] = field(default_factory=frozenset)
    """Geography codes (ISO-3166 alpha-2 or regional codes like EU). Empty
    set means "any jurisdiction"."""
    method_profiles: FrozenSet[str] = field(default_factory=frozenset)
    """Method-pack ids that, when present on the factor, unambiguously
    confer this framework tag."""
    notes: str = ""

    def jurisdiction_matches(self, geography: Optional[str]) -> bool:
        if not self.jurisdictions:
            return True
        if not geography:
            return False
        geography = geography.strip().upper()
        if geography in self.jurisdictions:
            return True
        # EU includes any EU member state; we treat anything prefixed by
        # a known EU-27 code as satisfying EU.
        if "EU" in self.jurisdictions and geography in _EU_MEMBER_STATES:
            return True
        return False


_EU_MEMBER_STATES: FrozenSet[str] = frozenset(
    {
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    }
)


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


def _fs(*values: FrameworkScope) -> FrozenSet[FrameworkScope]:
    return frozenset(values)


def _af(*values: ActivityFamily) -> FrozenSet[ActivityFamily]:
    return frozenset(values)


_ALL_ACTIVITY_FAMILIES = _af(
    "combustion",
    "stationary_combustion",
    "mobile_combustion",
    "electricity",
    "steam",
    "cooling",
    "heating",
    "transport",
    "freight",
    "passenger_transport",
    "purchased_goods",
    "capital_goods",
    "fuel_and_energy_upstream",
    "waste",
    "waste_treatment",
    "business_travel",
    "commuting",
    "downstream_transport",
    "processing",
    "use_phase",
    "end_of_life",
    "leased_assets",
    "franchises",
    "investments",
    "land_use",
    "land_use_change",
    "refrigerants",
    "fugitive",
    "process",
    "finance_proxy",
    "agriculture",
    "livestock",
    "forestry",
)

BUILTIN_RULES: Tuple[FrameworkApplicability, ...] = (
    # GHG Protocol Corporate — universal for Scope 1/2/3
    FrameworkApplicability(
        framework=RegulatoryFramework.GHG_PROTOCOL_CORPORATE,
        scopes=_fs(FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2, FrameworkScope.SCOPE_3),
        activity_families=_ALL_ACTIVITY_FAMILIES,
        notes="Universal corporate inventory standard.",
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.GHG_PROTOCOL_SCOPE2_LB,
        scopes=_fs(FrameworkScope.SCOPE_2),
        activity_families=_af("electricity", "steam", "heating", "cooling"),
        method_profiles=frozenset({"ghg_corporate_scope2_location"}),
        notes="Location-based Scope 2 (grid average).",
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.GHG_PROTOCOL_SCOPE2_MB,
        scopes=_fs(FrameworkScope.SCOPE_2),
        activity_families=_af("electricity", "steam", "heating", "cooling"),
        method_profiles=frozenset({"ghg_corporate_scope2_market"}),
        notes="Market-based Scope 2 (PPAs, RECs, residual mix).",
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.GHG_PROTOCOL_SCOPE3,
        scopes=_fs(FrameworkScope.SCOPE_3),
        activity_families=_af(
            "purchased_goods", "capital_goods", "fuel_and_energy_upstream",
            "waste_treatment", "business_travel", "commuting",
            "downstream_transport", "processing", "use_phase",
            "end_of_life", "leased_assets", "franchises", "investments",
            "transport", "freight",
        ),
        method_profiles=frozenset({"ghg_corporate_scope3"}),
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.GHG_PROTOCOL_PRODUCT,
        scopes=_fs(FrameworkScope.PRODUCT),
        activity_families=_af("purchased_goods", "use_phase", "end_of_life", "processing"),
        method_profiles=frozenset({"product_carbon", "ghg_product"}),
    ),

    # ISO family
    FrameworkApplicability(
        framework=RegulatoryFramework.ISO_14064_1,
        scopes=_fs(FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2, FrameworkScope.SCOPE_3),
        activity_families=_ALL_ACTIVITY_FAMILIES,
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.ISO_14064_2,
        scopes=_fs(FrameworkScope.ENTITY),
        activity_families=_af("land_use", "forestry", "process"),
        notes="Project-level quantification (including removals).",
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.ISO_14067,
        scopes=_fs(FrameworkScope.PRODUCT),
        activity_families=_af("purchased_goods", "processing", "use_phase", "end_of_life"),
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.ISO_14083,
        scopes=_fs(FrameworkScope.SCOPE_3),
        activity_families=_af("transport", "freight", "passenger_transport", "downstream_transport"),
        method_profiles=frozenset({"freight_iso_14083"}),
    ),

    # EU frameworks
    FrameworkApplicability(
        framework=RegulatoryFramework.CBAM,
        scopes=_fs(FrameworkScope.PRODUCT, FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2),
        activity_families=_af(
            "purchased_goods", "process", "electricity", "processing",
        ),
        jurisdictions=frozenset({"EU"}),
        method_profiles=frozenset({"eu_cbam"}),
        notes="Carbon Border Adjustment — direct + embedded emissions of imported goods.",
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.CSRD,
        scopes=_fs(FrameworkScope.ENTITY),
        activity_families=_ALL_ACTIVITY_FAMILIES,
        jurisdictions=frozenset({"EU"}),
        notes="Umbrella directive; E1 covers climate.",
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.ESRS_E1,
        scopes=_fs(FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2, FrameworkScope.SCOPE_3),
        activity_families=_ALL_ACTIVITY_FAMILIES,
        jurisdictions=frozenset({"EU"}),
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.EU_TAXONOMY,
        scopes=_fs(FrameworkScope.FINANCED, FrameworkScope.ENTITY),
        activity_families=_af(
            "electricity", "transport", "freight", "purchased_goods",
            "finance_proxy", "process", "land_use",
        ),
        jurisdictions=frozenset({"EU"}),
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.PEF,
        scopes=_fs(FrameworkScope.PRODUCT),
        activity_families=_af("purchased_goods", "processing", "use_phase", "end_of_life"),
        jurisdictions=frozenset({"EU"}),
        method_profiles=frozenset({"eu_pef"}),
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.OEF,
        scopes=_fs(FrameworkScope.ENTITY),
        activity_families=_ALL_ACTIVITY_FAMILIES,
        jurisdictions=frozenset({"EU"}),
        method_profiles=frozenset({"eu_oef"}),
    ),

    # UK
    FrameworkApplicability(
        framework=RegulatoryFramework.PAS_2050,
        scopes=_fs(FrameworkScope.PRODUCT),
        activity_families=_af("purchased_goods", "processing", "use_phase", "end_of_life"),
        jurisdictions=frozenset({"GB", "UK"}),
        method_profiles=frozenset({"pas_2050"}),
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.SECR,
        scopes=_fs(FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2),
        activity_families=_af("combustion", "stationary_combustion", "mobile_combustion",
                              "electricity", "steam", "heating", "cooling"),
        jurisdictions=frozenset({"GB", "UK"}),
    ),

    # US / state
    FrameworkApplicability(
        framework=RegulatoryFramework.SEC_CLIMATE,
        scopes=_fs(FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2, FrameworkScope.ENTITY),
        activity_families=_ALL_ACTIVITY_FAMILIES,
        jurisdictions=frozenset({"US"}),
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.CA_SB253,
        scopes=_fs(FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2, FrameworkScope.SCOPE_3),
        activity_families=_ALL_ACTIVITY_FAMILIES,
        jurisdictions=frozenset({"US-CA", "US"}),
        notes="Applies to US entities with California operations above $1B revenue.",
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.CA_SB261,
        scopes=_fs(FrameworkScope.ENTITY),
        activity_families=_ALL_ACTIVITY_FAMILIES,
        jurisdictions=frozenset({"US-CA", "US"}),
    ),

    # Global voluntary
    FrameworkApplicability(
        framework=RegulatoryFramework.TCFD,
        scopes=_fs(FrameworkScope.ENTITY),
        activity_families=_ALL_ACTIVITY_FAMILIES,
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.IFRS_S2,
        scopes=_fs(FrameworkScope.ENTITY, FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2, FrameworkScope.SCOPE_3),
        activity_families=_ALL_ACTIVITY_FAMILIES,
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.CDP,
        scopes=_fs(FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2, FrameworkScope.SCOPE_3),
        activity_families=_ALL_ACTIVITY_FAMILIES,
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.SBTi,
        scopes=_fs(FrameworkScope.SCOPE_1, FrameworkScope.SCOPE_2, FrameworkScope.SCOPE_3),
        activity_families=_ALL_ACTIVITY_FAMILIES,
        notes="Target-setting framework — tagging is inventory-inclusive.",
    ),
    FrameworkApplicability(
        framework=RegulatoryFramework.PCAF,
        scopes=_fs(FrameworkScope.FINANCED),
        activity_families=_af("finance_proxy", "investments"),
        method_profiles=frozenset({"pcaf", "pcaf_finance_proxy"}),
    ),
)


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------


_SCOPE_ALIASES = {
    "1": FrameworkScope.SCOPE_1,
    1: FrameworkScope.SCOPE_1,
    "scope_1": FrameworkScope.SCOPE_1,
    "scope1": FrameworkScope.SCOPE_1,
    "2": FrameworkScope.SCOPE_2,
    2: FrameworkScope.SCOPE_2,
    "scope_2": FrameworkScope.SCOPE_2,
    "scope2": FrameworkScope.SCOPE_2,
    "3": FrameworkScope.SCOPE_3,
    3: FrameworkScope.SCOPE_3,
    "scope_3": FrameworkScope.SCOPE_3,
    "scope3": FrameworkScope.SCOPE_3,
    "product": FrameworkScope.PRODUCT,
    "financed": FrameworkScope.FINANCED,
    "entity": FrameworkScope.ENTITY,
}


def _normalise_scope(raw: Any) -> Optional[FrameworkScope]:
    if raw is None:
        return None
    if isinstance(raw, FrameworkScope):
        return raw
    key: Any = raw
    if isinstance(raw, str):
        key = raw.strip().lower()
    return _SCOPE_ALIASES.get(key)


def _extract_attr(obj: Any, name: str) -> Any:
    """Read a field off either a dataclass-ish record or a dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _normalise_activity_families(obj: Any) -> Set[str]:
    candidates: Set[str] = set()
    for field_name in ("activity_family", "activity_type", "activity_tag"):
        val = _extract_attr(obj, field_name)
        if isinstance(val, str) and val:
            candidates.add(val.strip().lower())
    for field_name in ("activity_tags", "sector_tags"):
        val = _extract_attr(obj, field_name)
        if isinstance(val, (list, tuple, set)):
            candidates.update(str(v).strip().lower() for v in val if v)
    return candidates


def _normalise_method_profile(obj: Any) -> Optional[str]:
    for name in ("method_profile", "method_pack_id", "methodology"):
        val = _extract_attr(obj, name)
        if isinstance(val, str) and val:
            return val.strip().lower()
    return None


def _normalise_geography(obj: Any) -> Optional[str]:
    for name in ("geography", "country", "jurisdiction", "region"):
        val = _extract_attr(obj, name)
        if isinstance(val, str) and val:
            return val.strip().upper()
    return None


def _normalise_scope_field(obj: Any) -> Set[FrameworkScope]:
    out: Set[FrameworkScope] = set()
    raw = _extract_attr(obj, "scope")
    s = _normalise_scope(raw)
    if s is not None:
        out.add(s)
    # Packs sometimes record multiple supported scopes.
    raw_multi = _extract_attr(obj, "applicable_scopes") or _extract_attr(obj, "scopes")
    if isinstance(raw_multi, (list, tuple, set)):
        for v in raw_multi:
            m = _normalise_scope(v)
            if m is not None:
                out.add(m)
    # Infer from factor_family when scope is absent.
    if not out:
        family = _extract_attr(obj, "factor_family")
        if family == "finance_proxy":
            out.add(FrameworkScope.FINANCED)
        elif family in ("material_embodied", "product_carbon"):
            out.add(FrameworkScope.PRODUCT)
    return out


def _rule_matches(
    rule: FrameworkApplicability,
    *,
    scopes: Set[FrameworkScope],
    activity_families: Set[str],
    geography: Optional[str],
    method_profile: Optional[str],
) -> bool:
    # Scope filter — if the rule declares scopes, at least one must match.
    if rule.scopes and scopes and not (rule.scopes & scopes):
        return False
    # Activity-family filter — same "at-least-one" rule.
    if rule.activity_families and activity_families and not (rule.activity_families & activity_families):
        return False
    # Jurisdiction filter.
    if not rule.jurisdiction_matches(geography):
        return False
    # Method-profile disambiguation.
    #
    # Rules with ``method_profiles`` are variant-specific (e.g. Scope 2
    # location-based vs market-based). When the factor declares an
    # explicit method_profile we require it to match the rule; otherwise
    # we'd tag both variants for every factor.
    if rule.method_profiles:
        if method_profile is None:
            # Allow rules that also match on activity families to pass
            # when no method is declared — the activity-family overlap is
            # still a positive signal.
            if not (rule.activity_families & activity_families):
                return False
        else:
            if method_profile not in rule.method_profiles:
                return False
    return True


def tag_factor(
    factor: Any,
    *,
    extra_rules: Optional[Sequence[FrameworkApplicability]] = None,
) -> List[str]:
    """Return the list of framework ids applicable to ``factor``.

    Accepts either an ``EmissionFactorRecord``-like object or a plain dict
    with at least one of: ``scope``, ``activity_family``, ``activity_tags``,
    ``sector_tags``, ``geography``, ``method_profile``, ``factor_family``.
    """
    scopes = _normalise_scope_field(factor)
    activity_families = _normalise_activity_families(factor)
    geography = _normalise_geography(factor)
    method_profile = _normalise_method_profile(factor)
    rules: Sequence[FrameworkApplicability] = (
        tuple(BUILTIN_RULES) + tuple(extra_rules)
        if extra_rules
        else BUILTIN_RULES
    )
    out: List[str] = []
    for rule in rules:
        if _rule_matches(
            rule,
            scopes=scopes,
            activity_families=activity_families,
            geography=geography,
            method_profile=method_profile,
        ):
            out.append(rule.framework.value)
    return out


def tag_factor_batch(
    factors: Iterable[Any],
    *,
    extra_rules: Optional[Sequence[FrameworkApplicability]] = None,
) -> List[List[str]]:
    return [tag_factor(f, extra_rules=extra_rules) for f in factors]


# ---------------------------------------------------------------------------
# Framework index — factor_id lookup by framework
# ---------------------------------------------------------------------------


FactorIdGetter = Callable[[Any], Optional[str]]


def _default_factor_id(factor: Any) -> Optional[str]:
    fid = _extract_attr(factor, "factor_id") or _extract_attr(factor, "id")
    return str(fid) if fid else None


@dataclass
class FrameworkIndex:
    """Bidirectional index between frameworks and factor ids."""

    _framework_to_ids: Dict[str, Set[str]] = field(default_factory=dict)
    _id_to_frameworks: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        factors: Iterable[Any],
        *,
        factor_id_getter: FactorIdGetter = _default_factor_id,
        extra_rules: Optional[Sequence[FrameworkApplicability]] = None,
    ) -> "FrameworkIndex":
        index = cls()
        for f in factors:
            fid = factor_id_getter(f)
            if not fid:
                continue
            frameworks = tag_factor(f, extra_rules=extra_rules)
            if not frameworks:
                continue
            index._id_to_frameworks[fid] = set(frameworks)
            for fw in frameworks:
                index._framework_to_ids.setdefault(fw, set()).add(fid)
        return index

    # -- query surface -----------------------------------------------------

    def factors_for(self, framework: str) -> List[str]:
        return sorted(self._framework_to_ids.get(framework, ()))

    def frameworks_for(self, factor_id: str) -> List[str]:
        return sorted(self._id_to_frameworks.get(factor_id, ()))

    def factor_count(self, framework: str) -> int:
        return len(self._framework_to_ids.get(framework, ()))

    def all_frameworks(self) -> List[str]:
        return sorted(self._framework_to_ids.keys())

    def summary(self) -> Dict[str, int]:
        return {fw: len(ids) for fw, ids in self._framework_to_ids.items()}


__all__ = [
    "ActivityFamily",
    "BUILTIN_RULES",
    "FrameworkApplicability",
    "FrameworkIndex",
    "FrameworkScope",
    "RegulatoryFramework",
    "tag_factor",
    "tag_factor_batch",
]
