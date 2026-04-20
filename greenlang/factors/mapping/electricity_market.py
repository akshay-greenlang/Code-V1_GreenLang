# -*- coding: utf-8 -*-
"""Electricity market taxonomy — supplier / certificate / balancing area."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from greenlang.factors.mapping.base import (
    MappingConfidence,
    MappingResult,
    normalize_text,
)


class ElectricityMarketCategory(str, Enum):
    GRID_AVERAGE = "grid_average"                  # location-based default
    GRID_SUBREGION = "grid_subregion"              # eGRID, CEA regional, AIB zone
    SUPPLIER_SPECIFIC = "supplier_specific"        # utility tariff
    PPA = "ppa"                                    # power purchase agreement
    REC = "rec"                                    # Renewable Energy Certificate
    GO = "go"                                      # Guarantee of Origin
    GREEN_TARIFF = "green_tariff"                  # utility green tariff product
    RESIDUAL_MIX = "residual_mix"                  # AIB residual-mix fallback
    ONSITE_GENERATION = "onsite_generation"        # customer-owned solar / CHP


_PATTERNS = {
    ElectricityMarketCategory.PPA: [
        "ppa", "power purchase agreement", "virtual ppa", "vppa", "physical ppa",
    ],
    ElectricityMarketCategory.REC: [
        "rec", "renewable energy certificate", "i-rec", "irec", "ac-rec",
    ],
    ElectricityMarketCategory.GO: [
        "go", "guarantee of origin", "guarantees of origin", "aib go",
    ],
    ElectricityMarketCategory.GREEN_TARIFF: [
        "green tariff", "utility green tariff", "renewable tariff",
    ],
    ElectricityMarketCategory.RESIDUAL_MIX: [
        "residual mix", "eu residual mix", "aib residual",
    ],
    ElectricityMarketCategory.ONSITE_GENERATION: [
        "onsite solar", "behind the meter", "behind-the-meter", "onsite chp",
        "on-site solar", "captive generation",
    ],
    ElectricityMarketCategory.SUPPLIER_SPECIFIC: [
        "utility bill", "utility tariff", "supplier contract", "retail electricity",
    ],
    ElectricityMarketCategory.GRID_SUBREGION: [
        "egrid subregion", "egrid region", "cea region", "aib zone",
    ],
    ElectricityMarketCategory.GRID_AVERAGE: [
        "grid average", "grid electricity", "mains electricity", "location based",
    ],
}


def map_electricity_market(description: str) -> MappingResult:
    """Route an electricity line item to a market-attribution category."""
    needle = normalize_text(description)
    if not needle:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale="empty input",
            raw_input=description,
        )

    best: Optional[ElectricityMarketCategory] = None
    best_score = 0.0
    matched_pattern: Optional[str] = None
    # First, exact substring match ordered by category preference.
    for category, patterns in _PATTERNS.items():
        for pattern in patterns:
            if pattern in needle:
                score = min(1.0, len(pattern) / len(needle) + 0.5)
                if score > best_score:
                    best = category
                    best_score = score
                    matched_pattern = pattern

    if best is None:
        # Default to grid average if the caller just said "electricity".
        if "electricity" in needle or "power" in needle:
            return MappingResult(
                canonical={
                    "category": ElectricityMarketCategory.GRID_AVERAGE.value,
                    "electricity_basis": "location_based",
                    "requires_certificate": False,
                },
                confidence=0.5,
                band=MappingConfidence.LOW,
                rationale="Defaulted to grid_average (no market-basis indicator)",
                matched_pattern="electricity",
                raw_input=description,
            )
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=f"No electricity-market pattern in '{description}'",
            raw_input=description,
        )

    electricity_basis = {
        ElectricityMarketCategory.GRID_AVERAGE: "location_based",
        ElectricityMarketCategory.GRID_SUBREGION: "location_based",
        ElectricityMarketCategory.SUPPLIER_SPECIFIC: "market_based",
        ElectricityMarketCategory.PPA: "market_based",
        ElectricityMarketCategory.REC: "market_based",
        ElectricityMarketCategory.GO: "market_based",
        ElectricityMarketCategory.GREEN_TARIFF: "market_based",
        ElectricityMarketCategory.RESIDUAL_MIX: "residual_mix",
        ElectricityMarketCategory.ONSITE_GENERATION: "supplier_specific",
    }[best]

    canonical: Dict[str, Any] = {
        "category": best.value,
        "electricity_basis": electricity_basis,
        "requires_certificate": best in (
            ElectricityMarketCategory.PPA,
            ElectricityMarketCategory.REC,
            ElectricityMarketCategory.GO,
            ElectricityMarketCategory.GREEN_TARIFF,
        ),
    }

    return MappingResult(
        canonical=canonical,
        confidence=best_score,
        band=MappingConfidence.from_score(best_score),
        rationale=(
            f"Matched '{matched_pattern}' → {best.value} "
            f"(basis={electricity_basis})"
        ),
        matched_pattern=matched_pattern,
        raw_input=description,
    )


__all__ = ["ElectricityMarketCategory", "map_electricity_market"]
