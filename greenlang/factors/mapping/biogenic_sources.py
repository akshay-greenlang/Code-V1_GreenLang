# -*- coding: utf-8 -*-
"""Biogenic feedstock source taxonomy.

Tracks the principal biomass / biogas / waste-derived bio-feedstocks used by
GreenLang bioenergy method profiles. Each source carries moisture content,
lower heating value, carbon content, and a CO2 accounting treatment
(carbon-neutral vs short-rotation sequestration) derived from IPCC 2019
Refinement, IEA Bioenergy, and RED III Annex VI.

The data lives in ``greenlang/factors/data/taxonomies/biogenic_sources.yaml``.
This module provides free-text lookup plus Pydantic-free structured access.
"""
from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class BiogenicCategory(str, Enum):
    """High-level categorisation of biogenic feedstocks."""

    WOOD = "wood"
    CROP_RESIDUE = "crop_residue"
    MANURE = "manure"
    MUNICIPAL_WASTE_ORGANIC = "municipal_waste_organic"
    DEDICATED_ENERGY_CROP = "dedicated_energy_crop"
    ALGAE = "algae"
    BIOGAS = "biogas"
    PULPING_BYPRODUCT = "pulping_byproduct"


class CO2AccountingTreatment(str, Enum):
    """How biogenic CO2 is treated for reporting."""

    CARBON_NEUTRAL = "carbon_neutral"
    SHORT_ROTATION_SEQUESTRATION = "short_rotation_sequestration"
    FOSSIL_EQUIVALENT = "fossil_equivalent"  # for e.g. peat


class SustainabilityCertification(str, Enum):
    """Recognised sustainability certifications for biogenic sources."""

    FSC = "FSC"
    PEFC = "PEFC"
    RSPO = "RSPO"
    BONSUCRO = "Bonsucro"
    RSB = "RSB"
    SBP = "SBP"
    UNCERTIFIED = "uncertified"


# ---------------------------------------------------------------------------
# Structured record
# ---------------------------------------------------------------------------


class BiogenicSource:
    """A biogenic feedstock source with physicochemical properties."""

    __slots__ = (
        "key",
        "category",
        "description",
        "moisture_content",
        "lhv_gj_per_t",
        "carbon_content",
        "co2_accounting",
        "biogenic_share",
        "typical_certs",
        "extras",
    )

    def __init__(
        self,
        key: str,
        category: BiogenicCategory,
        description: str,
        moisture_content: float,
        lhv_gj_per_t: float,
        carbon_content: float,
        co2_accounting: CO2AccountingTreatment,
        biogenic_share: float,
        typical_certs: Optional[List[SustainabilityCertification]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ):
        if not 0.0 <= moisture_content <= 1.0:
            raise MappingError(
                "moisture_content must be in [0, 1]; got %r" % moisture_content
            )
        if lhv_gj_per_t < 0.0:
            raise MappingError("lhv_gj_per_t must be non-negative")
        if not 0.0 <= carbon_content <= 1.0:
            raise MappingError(
                "carbon_content must be in [0, 1]; got %r" % carbon_content
            )
        if not 0.0 <= biogenic_share <= 1.0:
            raise MappingError(
                "biogenic_share must be in [0, 1]; got %r" % biogenic_share
            )
        self.key = key
        self.category = category
        self.description = description
        self.moisture_content = float(moisture_content)
        self.lhv_gj_per_t = float(lhv_gj_per_t)
        self.carbon_content = float(carbon_content)
        self.co2_accounting = co2_accounting
        self.biogenic_share = float(biogenic_share)
        self.typical_certs = typical_certs or []
        self.extras = extras or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "category": self.category.value,
            "description": self.description,
            "moisture_content": self.moisture_content,
            "lhv_gj_per_t": self.lhv_gj_per_t,
            "carbon_content": self.carbon_content,
            "co2_accounting": self.co2_accounting.value,
            "biogenic_share": self.biogenic_share,
            "typical_certs": [c.value for c in self.typical_certs],
            "extras": dict(self.extras),
        }


# ---------------------------------------------------------------------------
# Synonym table (free-text lookup only)
# ---------------------------------------------------------------------------


BIOGENIC_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "wood_pellets": {
        "synonyms": ["wood pellets", "industrial pellets", "en plus a1", "pellets"],
        "meta": {"category": "wood"},
    },
    "wood_chips": {
        "synonyms": ["wood chips", "green chips", "forest chips", "fuel chips"],
        "meta": {"category": "wood"},
    },
    "wood_firewood_dry": {
        "synonyms": ["firewood", "seasoned firewood", "split logs", "cordwood"],
        "meta": {"category": "wood"},
    },
    "short_rotation_coppice": {
        "synonyms": ["src", "short rotation coppice", "willow coppice", "poplar coppice"],
        "meta": {"category": "dedicated_energy_crop"},
    },
    "miscanthus": {
        "synonyms": ["miscanthus", "miscanthus grass", "giant miscanthus", "elephant grass"],
        "meta": {"category": "dedicated_energy_crop"},
    },
    "straw_cereal": {
        "synonyms": ["wheat straw", "barley straw", "cereal straw", "straw"],
        "meta": {"category": "crop_residue"},
    },
    "rice_husk": {
        "synonyms": ["rice husk", "rice hulls", "paddy husk"],
        "meta": {"category": "crop_residue"},
    },
    "bagasse": {
        "synonyms": ["bagasse", "sugarcane bagasse", "cane trash"],
        "meta": {"category": "crop_residue"},
    },
    "corn_stover": {
        "synonyms": ["corn stover", "maize stover", "corn stalks"],
        "meta": {"category": "crop_residue"},
    },
    "manure_cattle": {
        "synonyms": ["cattle manure", "cow manure", "dairy slurry", "bovine manure"],
        "meta": {"category": "manure"},
    },
    "manure_pig": {
        "synonyms": ["pig manure", "swine manure", "swine slurry", "hog manure"],
        "meta": {"category": "manure"},
    },
    "manure_poultry": {
        "synonyms": ["poultry litter", "chicken manure", "layer manure", "broiler litter"],
        "meta": {"category": "manure"},
    },
    "msw_organic": {
        "synonyms": ["organic msw", "food waste", "green waste", "biowaste"],
        "meta": {"category": "municipal_waste_organic"},
    },
    "msw_mixed": {
        "synonyms": ["mixed msw", "residual msw", "mixed municipal solid waste"],
        "meta": {"category": "municipal_waste_organic"},
    },
    "palm_empty_fruit_bunch": {
        "synonyms": ["empty fruit bunch", "efb", "palm efb", "oil palm efb"],
        "meta": {"category": "crop_residue"},
    },
    "algae_microalgae": {
        "synonyms": ["microalgae", "algae biomass", "chlorella", "spirulina"],
        "meta": {"category": "algae"},
    },
    "algae_macroalgae": {
        "synonyms": ["macroalgae", "seaweed", "kelp", "kombu"],
        "meta": {"category": "algae"},
    },
    "biogas_anaerobic": {
        "synonyms": ["biogas", "raw biogas", "landfill gas", "ad biogas"],
        "meta": {"category": "biogas"},
    },
    "biomethane": {
        "synonyms": ["biomethane", "upgraded biogas", "renewable natural gas", "rng"],
        "meta": {"category": "biogas"},
    },
    "black_liquor": {
        "synonyms": ["black liquor", "kraft black liquor", "pulping liquor"],
        "meta": {"category": "pulping_byproduct"},
    },
}


class BiogenicMapping(BaseMapping):
    TAXONOMY = BIOGENIC_TAXONOMY


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


_DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "taxonomies"
    / "biogenic_sources.yaml"
)


@lru_cache(maxsize=1)
def load_biogenic_sources() -> Dict[str, BiogenicSource]:
    """Parse ``biogenic_sources.yaml`` into a dict of BiogenicSource by key."""
    if yaml is None:  # pragma: no cover
        raise MappingError("PyYAML is required for biogenic_sources loading")
    if not _DATA_PATH.exists():
        raise MappingError("biogenic_sources.yaml not found at %s" % _DATA_PATH)
    with _DATA_PATH.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    out: Dict[str, BiogenicSource] = {}
    for key, body in (payload.get("sources") or {}).items():
        try:
            category = BiogenicCategory(body.get("category", "wood"))
        except ValueError:
            logger.warning("Unknown biogenic category for %s; defaulting to wood", key)
            category = BiogenicCategory.WOOD
        try:
            co2_treatment = CO2AccountingTreatment(
                body.get("co2_accounting", "carbon_neutral")
            )
        except ValueError:
            co2_treatment = CO2AccountingTreatment.CARBON_NEUTRAL
        certs: List[SustainabilityCertification] = []
        for raw_cert in body.get("typical_certs") or []:
            try:
                certs.append(SustainabilityCertification(raw_cert))
            except ValueError:
                logger.debug("Unknown certification %s for %s", raw_cert, key)
        extras: Dict[str, Any] = {
            k: v
            for k, v in body.items()
            if k not in {
                "category", "description", "moisture_content", "lhv_gj_per_t",
                "carbon_content", "co2_accounting", "biogenic_share", "typical_certs",
            }
        }
        out[key] = BiogenicSource(
            key=key,
            category=category,
            description=str(body.get("description", key)),
            moisture_content=float(body.get("moisture_content", 0.0) or 0.0),
            lhv_gj_per_t=float(body.get("lhv_gj_per_t", 0.0) or 0.0),
            carbon_content=float(body.get("carbon_content", 0.0) or 0.0),
            co2_accounting=co2_treatment,
            biogenic_share=float(body.get("biogenic_share", 1.0) or 1.0),
            typical_certs=certs,
            extras=extras,
        )
    logger.debug("Loaded %d biogenic sources from %s", len(out), _DATA_PATH)
    return out


@lru_cache(maxsize=1)
def load_certifications() -> Dict[str, Dict[str, Any]]:
    if yaml is None:  # pragma: no cover
        raise MappingError("PyYAML required")
    if not _DATA_PATH.exists():
        raise MappingError("biogenic_sources.yaml not found at %s" % _DATA_PATH)
    with _DATA_PATH.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return dict(payload.get("certifications") or {})


# ---------------------------------------------------------------------------
# Public mapping API
# ---------------------------------------------------------------------------


def list_biogenic_sources() -> List[str]:
    return sorted(load_biogenic_sources().keys())


def map_biogenic_source(description: str) -> MappingResult:
    """Map a free-text biomass description to a canonical biogenic key."""
    lookup = BiogenicMapping._lookup(description)
    if lookup is None:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale="No biogenic source match for %r" % description,
            raw_input=description,
        )
    sources = load_biogenic_sources()
    record = sources.get(lookup.canonical)
    if record is None:  # pragma: no cover
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale="Source key %r missing from data file" % lookup.canonical,
            raw_input=description,
        )
    canonical = record.to_dict()
    return MappingResult(
        canonical=canonical,
        confidence=lookup.confidence,
        band=lookup.band,
        rationale=(
            "%s (category=%s); lhv=%.2f GJ/t; carbon=%.3f kg/kg; co2=%s; "
            "biogenic_share=%.2f"
            % (
                record.description,
                record.category.value,
                record.lhv_gj_per_t,
                record.carbon_content,
                record.co2_accounting.value,
                record.biogenic_share,
            )
        ),
        matched_pattern=lookup.matched_pattern,
        raw_input=description,
    )


def get_certification_info(cert: str) -> Optional[Dict[str, Any]]:
    """Return the structured info for a :class:`SustainabilityCertification`."""
    name = cert.strip()
    return load_certifications().get(name) or load_certifications().get(
        name.upper()
    )


__all__ = [
    "BIOGENIC_TAXONOMY",
    "BiogenicCategory",
    "BiogenicMapping",
    "BiogenicSource",
    "CO2AccountingTreatment",
    "SustainabilityCertification",
    "get_certification_info",
    "list_biogenic_sources",
    "load_biogenic_sources",
    "load_certifications",
    "map_biogenic_source",
]
