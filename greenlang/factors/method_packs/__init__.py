# -*- coding: utf-8 -*-
"""
GreenLang Factors — Method Pack Library (Phase F2).

Method packs are the commercial layer of GreenLang Factors.  The same
activity ("12,500 kWh, India, FY2027") resolves to *different* factors
depending on whether the caller is doing corporate inventory, product
carbon, freight, or financed-emissions reporting.

Each :class:`MethodPack` wraps:

- factor-selection rules (which factor families apply, which statuses allowed)
- boundary rules (combustion / WTT / WTW / cradle-to-gate / cradle-to-grave)
- inclusion / exclusion logic (biogenic treatment, market instruments)
- gas-to-CO2e conversion basis (AR4 / AR5 / AR6, 100-yr or 20-yr)
- region hierarchy for fallback (facility → utility → country → GLOBAL)
- reporting labels (which framework(s) this satisfies)
- audit text templates (used by the Explain endpoint in Phase F3)
- deprecation policy (how long after a source update we keep an old version)

Callers never touch the raw catalog — they always pass a
:class:`~greenlang.data.canonical_v2.MethodProfile` to the resolution
engine (Phase F3), which in turn consults the registered pack.

Quickstart::

    from greenlang.factors.method_packs import get_pack
    from greenlang.data.canonical_v2 import MethodProfile

    pack = get_pack(MethodProfile.CORPORATE_SCOPE2_LOCATION)
    print(pack.name, pack.region_hierarchy)
    print(pack.gwp_basis, pack.boundary_rules)

Public surface re-exported below so callers ``from greenlang.factors.method_packs import ...``.
"""
from __future__ import annotations

from greenlang.factors.method_packs.base import (
    BoundaryRule,
    DeprecationRule,
    FallbackStep,
    MethodPack,
    SelectionRule,
)
from greenlang.factors.method_packs.registry import (
    get_pack,
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
    freight,
    land_removals,
    finance_proxy,
)

__all__ = [
    "BoundaryRule",
    "DeprecationRule",
    "FallbackStep",
    "MethodPack",
    "SelectionRule",
    "get_pack",
    "list_packs",
    "register_pack",
    "registered_profiles",
]
