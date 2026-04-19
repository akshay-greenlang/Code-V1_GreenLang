# -*- coding: utf-8 -*-
"""Applicability engine — decides which frameworks apply to an entity.

Rule-driven stub. Full OPA integration via greenlang/policy_graph/ lands in
COMPLY-APP 3 (task #17).
"""

from __future__ import annotations

from schemas.models import (
    ApplicabilityRequest,
    ApplicabilityResult,
    EntitySnapshot,
    FrameworkEnum,
)

# Default thresholds — overridable via policy_graph bundle at prod deploy.
_EU_CODES = {
    "EU", "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
    "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO",
    "SK", "SI", "ES", "SE",
}
_CSRD_REVENUE_EUR = 40_000_000
_CSRD_EMPLOYEES = 250


def evaluate(request: ApplicabilityRequest) -> ApplicabilityResult:
    e: EntitySnapshot = request.entity
    applicable: list[FrameworkEnum] = [FrameworkEnum.GHG_PROTOCOL, FrameworkEnum.ISO_14064]
    rationale: dict[FrameworkEnum, str] = {
        FrameworkEnum.GHG_PROTOCOL: "Universal GHG accounting baseline",
        FrameworkEnum.ISO_14064: "Verification-ready standard (applies universally)",
    }

    country = e.jurisdiction.split("-", 1)[0].upper()
    is_eu = country in _EU_CODES
    if is_eu and (e.revenue_eur or 0) >= _CSRD_REVENUE_EUR and (e.employees or 0) >= _CSRD_EMPLOYEES:
        applicable.append(FrameworkEnum.CSRD)
        rationale[FrameworkEnum.CSRD] = "EU-resident + revenue >= EUR 40M + employees >= 250"
        applicable.append(FrameworkEnum.EU_TAXONOMY)
        rationale[FrameworkEnum.EU_TAXONOMY] = "CSRD filers must report Taxonomy alignment"

    if e.imports_cbam_goods:
        applicable.append(FrameworkEnum.CBAM)
        rationale[FrameworkEnum.CBAM] = "Declared CBAM-scope goods import activity"

    if e.handles_eudr_commodities:
        applicable.append(FrameworkEnum.EUDR)
        rationale[FrameworkEnum.EUDR] = "Declared EUDR-scope commodity handling"

    if e.operates_in_us_ca or e.jurisdiction.upper() == "US-CA":
        applicable.append(FrameworkEnum.SB253)
        rationale[FrameworkEnum.SB253] = "California operations (SB 253 threshold)"

    if (e.revenue_eur or 0) >= 100_000_000:
        applicable.append(FrameworkEnum.SBTI)
        rationale[FrameworkEnum.SBTI] = "Revenue threshold for credible SBTi commitment"
        applicable.append(FrameworkEnum.TCFD)
        rationale[FrameworkEnum.TCFD] = "TCFD recommended for >$100M revenue"
        applicable.append(FrameworkEnum.CDP)
        rationale[FrameworkEnum.CDP] = "CDP disclosure recommended for investor relations"

    return ApplicabilityResult(
        applicable_frameworks=sorted(set(applicable), key=lambda f: f.value),
        rationale={k: v for k, v in rationale.items() if k in applicable},
    )
