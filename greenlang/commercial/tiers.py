# -*- coding: utf-8 -*-
"""Tier specifications — FY27 commercial model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    COMMUNITY = "community"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class TierSpec:
    tier: Tier
    daily_request_limit: int  # 0 = unlimited
    monthly_request_limit: int
    price_usd_monthly: int
    features: frozenset[str] = field(default_factory=frozenset)
    factor_visibility: frozenset[str] = field(default_factory=frozenset)


TIER_SPECS: dict[Tier, TierSpec] = {
    Tier.COMMUNITY: TierSpec(
        tier=Tier.COMMUNITY,
        daily_request_limit=100,
        monthly_request_limit=2_000,
        price_usd_monthly=0,
        features=frozenset({"factors.read", "scope_engine.compute.limited"}),
        factor_visibility=frozenset({"certified"}),
    ),
    Tier.PRO: TierSpec(
        tier=Tier.PRO,
        daily_request_limit=10_000,
        monthly_request_limit=250_000,
        price_usd_monthly=99,
        features=frozenset({
            "factors.read", "factors.preview", "scope_engine.compute",
            "comply.intake", "comply.report.json", "comply.report.xml",
        }),
        factor_visibility=frozenset({"certified", "preview"}),
    ),
    Tier.ENTERPRISE: TierSpec(
        tier=Tier.ENTERPRISE,
        daily_request_limit=0,
        monthly_request_limit=0,
        price_usd_monthly=0,  # custom; negotiated per account
        features=frozenset({
            "factors.read", "factors.preview", "factors.connector",
            "scope_engine.compute", "comply.intake", "comply.report.json",
            "comply.report.xml", "comply.report.pdf", "connect.erp",
            "sla.99_95", "support.dedicated",
        }),
        factor_visibility=frozenset({"certified", "preview", "connector"}),
    ),
}


def get_spec(tier: Tier) -> TierSpec:
    return TIER_SPECS[tier]


def feature_allowed(tier: Tier, feature: str) -> bool:
    return feature in TIER_SPECS[tier].features
