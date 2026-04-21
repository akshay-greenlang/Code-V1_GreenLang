# -*- coding: utf-8 -*-
"""
GreenLang Factors - Method Pack Library (Phase F2, GAP-8 + GAP-9 closed).

Method packs are the commercial layer of GreenLang Factors.  The same
activity ("12,500 kWh, India, FY2027") resolves to *different* factors
depending on whether the caller is doing corporate inventory, product
carbon, freight, financed-emissions, or land-sector reporting.

Each :class:`MethodPack` wraps:

- factor-selection rules (which factor families apply, which statuses allowed)
- boundary rules (combustion / WTT / WTW / cradle-to-gate / cradle-to-grave)
- inclusion / exclusion logic (biogenic treatment, market instruments)
- gas-to-CO2e conversion basis (AR4 / AR5 / AR6, 100-yr or 20-yr)
- region hierarchy for fallback (facility > utility > country > GLOBAL)
- reporting labels (which framework(s) this satisfies)
- audit text templates (used by the Explain endpoint in Phase F3)
- deprecation policy (how long after a source update we keep an old version)

Quickstart::

    from greenlang.factors.method_packs import get_pack
    from greenlang.data.canonical_v2 import MethodProfile

    # Profile-level lookup (legacy):
    pack = get_pack(MethodProfile.CORPORATE_SCOPE2_LOCATION)

    # Variant-level lookup (GAP-8 / GAP-9):
    pcaf_mortgage = get_pack("pcaf_mortgages")
    lsr_removals = get_pack("lsr_removals")

Public surface re-exported below.
"""
from __future__ import annotations

from typing import Union

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs.base import (
    BoundaryRule,
    DeprecationRule,
    FallbackStep,
    MethodPack,
    SelectionRule,
)
from greenlang.factors.method_packs.registry import (
    MethodPackNotFound,
    get_pack as _get_pack_by_profile,
    list_packs,
    register_pack,
    registered_profiles,
)

# Import + register all built-in packs on module load.  Each module
# self-registers via ``register_pack(...)`` at import time.
from greenlang.factors.method_packs import (  # noqa: F401
    corporate,
    electricity,
    eu_policy,
    product_carbon,
    product_lca_variants,
    freight,
    land_removals,
    finance_proxy,
)

# Re-export the GAP-8 + GAP-9 variant-level helpers so callers can reach
# them without importing the submodules directly.
from greenlang.factors.method_packs.finance_proxy import (  # noqa: F401
    PCAFAssetClass,
    PCAFDataQualityScore,
    PCAFAttributionMethod,
    PCAFIntensityMode,
    PCAFPackMetadata,
    PCAFListedEquityPack,
    PCAFCorporateBondsPack,
    PCAFBusinessLoansPack,
    PCAFProjectFinancePack,
    PCAFCommercialRealEstatePack,
    PCAFMortgagesPack,
    PCAFMotorVehicleLoansPack,
    PCAF_DQS_RUBRIC,
    PCAF_ATTRIBUTION_HIERARCHY,
    get_pcaf_variant,
    get_pcaf_metadata,
    list_pcaf_variants,
)
from greenlang.factors.method_packs.product_lca_variants import (  # noqa: F401
    PAS_2050,
    PEF,
    OEF,
    get_product_lca_variant,
    list_product_lca_variants,
)
from greenlang.factors.method_packs.land_removals import (  # noqa: F401
    PermanenceClass,
    ReversalRiskLevel,
    BiogenicAccountingTreatment,
    RemovalCategory,
    RemovalType,
    VerificationStandard,
    ReportingFrequency,
    LSRPackMetadata,
    GHGLSRLandUseEmissionsPack,
    GHGLSRLandManagementPack,
    GHGLSRRemovalsPack,
    GHGLSRStoragePack,
    DEFAULT_BUFFER_POOL,
    RISK_BUFFER_MULTIPLIER,
    LSR_FALLBACK_HIERARCHY,
    get_lsr_variant,
    get_lsr_metadata,
    list_lsr_variants,
    compute_buffer_pool_pct,
)


def get_pack(key: Union[MethodProfile, str]) -> MethodPack:
    """Retrieve a registered method pack.

    Accepts either:
      * a :class:`MethodProfile` enum value (legacy behaviour — returns
        the umbrella pack registered under that profile), OR
      * a string variant name (GAP-8 / GAP-9 — returns the specific
        PCAF or LSR variant, e.g. ``"pcaf_listed_equity"`` or
        ``"lsr_removals"``).

    Raises :class:`MethodPackNotFound` if the key is unknown.
    """
    if isinstance(key, MethodProfile):
        return _get_pack_by_profile(key)

    if isinstance(key, str):
        # Try PCAF named variants.
        try:
            return get_pcaf_variant(key)
        except KeyError:
            pass
        # Try LSR named variants.
        try:
            return get_lsr_variant(key)
        except KeyError:
            pass
        # Try product-LCA variants (PAS 2050, PEF, OEF).
        try:
            return get_product_lca_variant(key)
        except KeyError:
            pass
        # Try as MethodProfile enum value string.
        try:
            return _get_pack_by_profile(MethodProfile(key))
        except (ValueError, MethodPackNotFound):
            pass

        raise MethodPackNotFound(
            "no method pack registered for key %r; "
            "try a MethodProfile or a registered variant name" % key
        )

    raise TypeError(
        "get_pack() expected MethodProfile or str, got %s" % type(key).__name__
    )


__all__ = [
    # Base types
    "BoundaryRule",
    "DeprecationRule",
    "FallbackStep",
    "MethodPack",
    "MethodPackNotFound",
    "SelectionRule",
    # Registry primitives
    "get_pack",
    "list_packs",
    "register_pack",
    "registered_profiles",
    # PCAF (GAP-8)
    "PCAFAssetClass",
    "PCAFDataQualityScore",
    "PCAFAttributionMethod",
    "PCAFIntensityMode",
    "PCAFPackMetadata",
    "PCAFListedEquityPack",
    "PCAFCorporateBondsPack",
    "PCAFBusinessLoansPack",
    "PCAFProjectFinancePack",
    "PCAFCommercialRealEstatePack",
    "PCAFMortgagesPack",
    "PCAFMotorVehicleLoansPack",
    "PCAF_DQS_RUBRIC",
    "PCAF_ATTRIBUTION_HIERARCHY",
    "get_pcaf_variant",
    "get_pcaf_metadata",
    "list_pcaf_variants",
    # LSR (GAP-9)
    "PermanenceClass",
    "ReversalRiskLevel",
    "BiogenicAccountingTreatment",
    "RemovalCategory",
    "RemovalType",
    "VerificationStandard",
    "ReportingFrequency",
    "LSRPackMetadata",
    "GHGLSRLandUseEmissionsPack",
    "GHGLSRLandManagementPack",
    "GHGLSRRemovalsPack",
    "GHGLSRStoragePack",
    "DEFAULT_BUFFER_POOL",
    "RISK_BUFFER_MULTIPLIER",
    "LSR_FALLBACK_HIERARCHY",
    "get_lsr_variant",
    "get_lsr_metadata",
    "list_lsr_variants",
    "compute_buffer_pool_pct",
    # Product-LCA variants (PAS 2050, PEF, OEF)
    "PAS_2050",
    "PEF",
    "OEF",
    "get_product_lca_variant",
    "list_product_lca_variants",
]
