# -*- coding: utf-8 -*-
"""Land-use and land-use-change (LUC) taxonomy.

Covers the commodities and production regions tracked by the GreenLang
Scope 3 Cat 1 / Cat 4 mapper and the EUDR pack. The data file
``luc_commodities.yaml`` holds regional LUC factors per commodity per
year plus a deforestation-risk tier (used to decide the method profile).

Key concepts:

- LandUseCategory (IPCC AFOLU): forest, cropland, grassland, wetland,
  settlement, other_land.
- LUCCommodity: the 8 EUDR-scoped commodities (palm, soy, beef, cocoa,
  coffee, rubber, pulp_paper, wood) plus extensions.
- ProductionRegion: ISO-3 country + optional ecoregion + risk tier.
- PermanenceClass: ephemeral (<10y) / short / medium / long (>100y)
  derived from IPCC 2019 Refinement Vol 4, Ch. 2.
"""
from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from greenlang.factors.mapping.base import (
    BaseMapping,
    MappingConfidence,
    MappingError,
    MappingResult,
    normalize_text,
)

logger = logging.getLogger(__name__)


try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LandUseCategory(str, Enum):
    """IPCC 2019 Refinement AFOLU land-use categories."""

    FOREST = "forest"
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    SETTLEMENT = "settlement"
    OTHER_LAND = "other_land"


class DeforestationRiskTier(str, Enum):
    """EUDR-style benchmarking risk tier."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"
    UNKNOWN = "unknown"


class PermanenceClass(str, Enum):
    """Permanence of the LUC carbon impact.

    Per IPCC 2019 Refinement Vol 4, Ch. 2, we band permanence as:

    - ephemeral : < 10 years
    - short     : 10-30 years
    - medium    : 30-100 years
    - long      : > 100 years
    """

    EPHEMERAL = "ephemeral"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Structured records
# ---------------------------------------------------------------------------


class ProductionRegion:
    """Production region = ISO-3 country + optional ecoregion + risk tier."""

    __slots__ = ("iso3", "label", "ecoregion", "risk_tier", "luc_tco2e_per_t_product", "notes")

    def __init__(
        self,
        iso3: str,
        label: str,
        risk_tier: DeforestationRiskTier,
        luc_tco2e_per_t_product: float,
        ecoregion: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        self.iso3 = iso3.upper().strip()
        if len(self.iso3) != 3:
            raise MappingError(
                "ProductionRegion requires an ISO-3 country code; got %r" % iso3
            )
        self.label = label
        self.risk_tier = risk_tier
        self.luc_tco2e_per_t_product = float(luc_tco2e_per_t_product)
        self.ecoregion = ecoregion
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iso3": self.iso3,
            "label": self.label,
            "risk_tier": self.risk_tier.value,
            "luc_tco2e_per_t_product": self.luc_tco2e_per_t_product,
            "ecoregion": self.ecoregion,
            "notes": self.notes,
        }


class LUCCommodity:
    """A commodity tracked for land-use-change emissions."""

    __slots__ = (
        "key",
        "eudr_covered",
        "cbam_covered",
        "base_luc_tco2e_per_t_product",
        "biogenic_share",
        "permanence_class",
        "regions",
    )

    def __init__(
        self,
        key: str,
        eudr_covered: bool,
        cbam_covered: bool,
        base_luc_tco2e_per_t_product: float,
        biogenic_share: float,
        permanence_class: PermanenceClass,
        regions: Dict[str, ProductionRegion],
    ):
        self.key = key
        self.eudr_covered = eudr_covered
        self.cbam_covered = cbam_covered
        self.base_luc_tco2e_per_t_product = float(base_luc_tco2e_per_t_product)
        self.biogenic_share = float(biogenic_share)
        self.permanence_class = permanence_class
        self.regions = regions

    def region_for(self, iso3: str) -> Optional[ProductionRegion]:
        return self.regions.get(iso3.upper().strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "eudr_covered": self.eudr_covered,
            "cbam_covered": self.cbam_covered,
            "base_luc_tco2e_per_t_product": self.base_luc_tco2e_per_t_product,
            "biogenic_share": self.biogenic_share,
            "permanence_class": self.permanence_class.value,
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
        }


# ---------------------------------------------------------------------------
# Commodity synonym taxonomy (free-text to canonical key)
# ---------------------------------------------------------------------------


LUC_COMMODITY_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "palm_oil": {
        "synonyms": ["palm", "palm oil", "cpo", "crude palm oil", "palm kernel oil"],
        "meta": {"eudr_scope": True},
    },
    "soy": {
        "synonyms": ["soy", "soya", "soybean", "soybeans", "soy meal", "soy oil"],
        "meta": {"eudr_scope": True},
    },
    "beef": {
        "synonyms": ["beef", "cattle", "bovine", "bovine meat", "live cattle"],
        "meta": {"eudr_scope": True},
    },
    "cocoa": {
        "synonyms": ["cocoa", "cacao", "cocoa beans", "cocoa paste"],
        "meta": {"eudr_scope": True},
    },
    "coffee": {
        "synonyms": ["coffee", "coffee beans", "arabica", "robusta", "green coffee"],
        "meta": {"eudr_scope": True},
    },
    "rubber": {
        "synonyms": ["rubber", "natural rubber", "latex", "hevea"],
        "meta": {"eudr_scope": True},
    },
    "pulp_paper": {
        "synonyms": ["pulp", "pulp and paper", "paper pulp", "market pulp"],
        "meta": {"eudr_scope": True},
    },
    "wood": {
        "synonyms": ["wood", "timber", "logs", "sawnwood", "roundwood"],
        "meta": {"eudr_scope": True},
    },
}


class LUCCommodityMapping(BaseMapping):
    """Free-text lookup for commodity aliases."""

    TAXONOMY = LUC_COMMODITY_TAXONOMY


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


_DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "taxonomies"
    / "luc_commodities.yaml"
)


@lru_cache(maxsize=1)
def load_luc_commodity_data() -> Dict[str, LUCCommodity]:
    """Parse ``luc_commodities.yaml`` into a dict of canonical LUCCommodity."""
    if yaml is None:  # pragma: no cover
        raise MappingError(
            "PyYAML is required for land_use loading; install pyyaml"
        )
    if not _DATA_PATH.exists():
        raise MappingError("luc_commodities.yaml not found at %s" % _DATA_PATH)
    with _DATA_PATH.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    commodities_raw = payload.get("commodities", {})
    out: Dict[str, LUCCommodity] = {}
    for key, body in commodities_raw.items():
        regions: Dict[str, ProductionRegion] = {}
        for iso3, region_body in (body.get("regions") or {}).items():
            try:
                risk_tier = DeforestationRiskTier(
                    region_body.get("risk_tier", "unknown")
                )
            except ValueError:
                risk_tier = DeforestationRiskTier.UNKNOWN
            regions[iso3.upper()] = ProductionRegion(
                iso3=iso3,
                label=region_body.get("label", iso3),
                risk_tier=risk_tier,
                luc_tco2e_per_t_product=float(
                    region_body.get("luc_tco2e_per_t_product", 0.0) or 0.0
                ),
                ecoregion=region_body.get("ecoregion"),
                notes=region_body.get("notes"),
            )
        try:
            permanence = PermanenceClass(body.get("permanence_class", "unknown"))
        except ValueError:
            permanence = PermanenceClass.UNKNOWN
        out[key] = LUCCommodity(
            key=key,
            eudr_covered=bool(body.get("eudr_covered", False)),
            cbam_covered=bool(body.get("cbam_covered", False)),
            base_luc_tco2e_per_t_product=float(
                body.get("base_luc_tco2e_per_t_product", 0.0) or 0.0
            ),
            biogenic_share=float(body.get("biogenic_share", 0.0) or 0.0),
            permanence_class=permanence,
            regions=regions,
        )
    logger.debug("Loaded %d LUC commodities from %s", len(out), _DATA_PATH)
    return out


def list_luc_commodities() -> List[str]:
    return sorted(load_luc_commodity_data().keys())


def list_regions_for(commodity_key: str) -> List[str]:
    data = load_luc_commodity_data()
    if commodity_key not in data:
        return []
    return sorted(data[commodity_key].regions.keys())


# ---------------------------------------------------------------------------
# Public mapping API
# ---------------------------------------------------------------------------


def map_luc_commodity(description: str) -> MappingResult:
    """Map a free-text commodity description to a canonical EUDR commodity key."""
    lookup = LUCCommodityMapping._lookup(description)
    if lookup is None:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale="No LUC commodity match for %r" % description,
            raw_input=description,
        )
    return lookup


def map_land_use(
    commodity: str,
    iso3_country: Optional[str] = None,
) -> MappingResult:
    """Return the LUC record for ``commodity`` + country.

    When ``iso3_country`` is given and matches a region, the result's
    ``canonical`` contains the region-specific LUC factor; otherwise the
    base factor is returned with a ``region=None`` flag.
    """
    commodity_lookup = map_luc_commodity(commodity)
    if commodity_lookup.canonical is None:
        return commodity_lookup
    key = commodity_lookup.canonical
    data = load_luc_commodity_data()
    record = data.get(key)
    if record is None:  # pragma: no cover - mapping + data drift
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale="Commodity %r not present in luc_commodities.yaml" % key,
            raw_input=commodity,
        )
    region: Optional[ProductionRegion] = None
    if iso3_country:
        region = record.region_for(iso3_country)
    factor = (
        region.luc_tco2e_per_t_product
        if region is not None
        else record.base_luc_tco2e_per_t_product
    )
    risk_tier = (
        region.risk_tier.value
        if region is not None
        else DeforestationRiskTier.UNKNOWN.value
    )
    canonical = {
        "commodity": key,
        "iso3_country": region.iso3 if region else (iso3_country.upper() if iso3_country else None),
        "region_label": region.label if region else None,
        "luc_tco2e_per_t_product": factor,
        "base_luc_tco2e_per_t_product": record.base_luc_tco2e_per_t_product,
        "biogenic_share": record.biogenic_share,
        "fossil_share": max(0.0, 1.0 - record.biogenic_share),
        "permanence_class": record.permanence_class.value,
        "risk_tier": risk_tier,
        "eudr_covered": record.eudr_covered,
        "cbam_covered": record.cbam_covered,
        "ecoregion": region.ecoregion if region else None,
    }
    confidence = commodity_lookup.confidence if region is None else 1.0
    return MappingResult(
        canonical=canonical,
        confidence=confidence,
        band=MappingConfidence.from_score(confidence),
        rationale=(
            "commodity=%s; country=%s; factor=%.2f tCO2e/t; risk=%s; "
            "permanence=%s; biogenic_share=%.2f"
            % (
                key,
                canonical["iso3_country"] or "n/a",
                factor,
                risk_tier,
                record.permanence_class.value,
                record.biogenic_share,
            )
        ),
        matched_pattern=commodity_lookup.matched_pattern,
        raw_input="%s|%s" % (commodity, iso3_country or ""),
    )


def get_risk_tier(commodity: str, iso3_country: str) -> DeforestationRiskTier:
    """Return the deforestation-risk tier for commodity x country."""
    mapped = map_luc_commodity(commodity)
    if mapped.canonical is None:
        return DeforestationRiskTier.UNKNOWN
    record = load_luc_commodity_data().get(mapped.canonical)
    if record is None:
        return DeforestationRiskTier.UNKNOWN
    region = record.region_for(iso3_country)
    return region.risk_tier if region else DeforestationRiskTier.UNKNOWN


# ---------------------------------------------------------------------------
# EUDR pack integration surface (import-optional)
# ---------------------------------------------------------------------------


def eudr_is_in_scope(commodity: str) -> bool:
    """Return True if the commodity is within the EU Deforestation Regulation."""
    mapped = map_luc_commodity(commodity)
    if mapped.canonical is None:
        return False
    record = load_luc_commodity_data().get(mapped.canonical)
    return bool(record and record.eudr_covered)


__all__ = [
    "DeforestationRiskTier",
    "LUCCommodity",
    "LUCCommodityMapping",
    "LUC_COMMODITY_TAXONOMY",
    "LandUseCategory",
    "PermanenceClass",
    "ProductionRegion",
    "eudr_is_in_scope",
    "get_risk_tier",
    "list_luc_commodities",
    "list_regions_for",
    "load_luc_commodity_data",
    "map_land_use",
    "map_luc_commodity",
]
