# -*- coding: utf-8 -*-
"""Circular economy taxonomy - primary vs secondary material flows.

Covers recycled-content tracking, material lifecycle states, and the
embodied-carbon adjustment multiplier to apply when mapping a spend /
purchase line to a secondary (recycled / remanufactured) production route
instead of the virgin / primary route.

Data file: ``greenlang/factors/data/taxonomies/recycled_content_factors.yaml``.

Typical usage::

    from greenlang.factors.mapping import map_circular_flow

    flow = map_circular_flow(
        material="steel",
        recycled_content_pct=85.0,
        source="post_consumer_scrap",
    )
    # flow.canonical["adjusted_factor_kgco2e_per_kg"] -> 0.84
    # flow.canonical["lifecycle_state"] -> "recycled"

All loads are cached via ``functools.lru_cache``.
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


class MaterialLifecycle(str, Enum):
    """Lifecycle state of a material at the point of product manufacture."""

    VIRGIN = "virgin"
    RECYCLED = "recycled"
    REMANUFACTURED = "remanufactured"
    REUSED = "reused"
    BIODEGRADABLE = "biodegradable"
    UNKNOWN = "unknown"


class RecycledSource(str, Enum):
    """Where the recycled feed came from."""

    PRE_CONSUMER = "pre_consumer"
    POST_CONSUMER = "post_consumer"
    POST_INDUSTRIAL = "post_industrial"
    MIXED = "mixed"
    UNSPECIFIED = "unspecified"


class CircularRoute(str, Enum):
    """Manufacturing route encoding primary / secondary distinction."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    CHEMICAL_RECYCLED = "chemical_recycled"
    MECHANICAL_RECYCLED = "mechanical_recycled"
    REMANUFACTURED = "remanufactured"
    REUSED = "reused"


# ---------------------------------------------------------------------------
# Structured record classes
# ---------------------------------------------------------------------------


class RecycledContent:
    """Percentage of recycled input in a material, with source."""

    __slots__ = ("percentage", "source")

    def __init__(self, percentage: float, source: RecycledSource = RecycledSource.UNSPECIFIED):
        if not 0.0 <= float(percentage) <= 100.0:
            raise MappingError(
                f"recycled_content percentage must be in [0, 100]; got {percentage}"
            )
        self.percentage = float(percentage)
        self.source = source

    def as_fraction(self) -> float:
        return self.percentage / 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "percentage": self.percentage,
            "fraction": self.as_fraction(),
            "source": self.source.value,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"RecycledContent({self.percentage}%, {self.source.value})"


class CircularMaterialFlow:
    """A material flow carrying recycled-content and lifecycle attributes."""

    __slots__ = ("material", "route", "lifecycle", "recycled_content", "metadata")

    def __init__(
        self,
        material: str,
        route: CircularRoute,
        lifecycle: MaterialLifecycle,
        recycled_content: Optional[RecycledContent] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.material = material
        self.route = route
        self.lifecycle = lifecycle
        self.recycled_content = recycled_content
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material": self.material,
            "route": self.route.value,
            "lifecycle": self.lifecycle.value,
            "recycled_content": (
                self.recycled_content.to_dict() if self.recycled_content else None
            ),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Taxonomy (canonical keys + synonyms for free-text matching)
# ---------------------------------------------------------------------------


CIRCULAR_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "steel": {
        "synonyms": ["steel", "iron", "ferrous", "carbon steel", "rebar", "hrc", "crc"],
        "meta": {"primary_route": "bf_bof", "secondary_route": "eaf"},
    },
    "aluminium": {
        "synonyms": [
            "aluminium", "aluminum", "al", "aluminium ingot", "aluminum ingot",
            "aluminium cast", "aluminium wrought",
        ],
        "meta": {"primary_route": "hall_heroult", "secondary_route": "remelt"},
    },
    "plastic_pet": {
        "synonyms": ["pet", "polyethylene terephthalate", "rpet"],
        "meta": {"primary_route": "virgin_pet", "secondary_route": "mechanical_recycled_pet"},
    },
    "plastic_hdpe": {
        "synonyms": ["hdpe", "high-density polyethylene", "r-hdpe"],
        "meta": {"primary_route": "virgin_hdpe", "secondary_route": "mechanical_recycled_hdpe"},
    },
    "plastic_ldpe": {
        "synonyms": ["ldpe", "low-density polyethylene", "r-ldpe"],
        "meta": {"primary_route": "virgin_ldpe", "secondary_route": "mechanical_recycled_ldpe"},
    },
    "plastic_pp": {
        "synonyms": ["pp", "polypropylene", "r-pp"],
        "meta": {"primary_route": "virgin_pp", "secondary_route": "mechanical_recycled_pp"},
    },
    "plastic_ps": {
        "synonyms": ["ps", "polystyrene"],
        "meta": {"primary_route": "virgin_ps", "secondary_route": "mechanical_recycled_ps"},
    },
    "plastic_pvc": {
        "synonyms": ["pvc", "polyvinyl chloride"],
        "meta": {"primary_route": "virgin_pvc", "secondary_route": "mechanical_recycled_pvc"},
    },
    "paper": {
        "synonyms": ["paper", "kraft", "kraft paper"],
        "meta": {"primary_route": "virgin_kraft", "secondary_route": "recycled_fibre"},
    },
    "cardboard": {
        "synonyms": ["cardboard", "corrugated", "occ", "old corrugated cardboard"],
        "meta": {
            "primary_route": "virgin_kraft_corrugated",
            "secondary_route": "recycled_corrugated",
        },
    },
    "glass": {
        "synonyms": ["glass", "container glass", "float glass", "cullet"],
        "meta": {"primary_route": "virgin_soda_lime", "secondary_route": "cullet_route"},
    },
    "concrete": {
        "synonyms": [
            "concrete", "ready-mix concrete", "rmc",
            "scm concrete", "ggbs concrete", "fly ash concrete",
        ],
        "meta": {"primary_route": "opc_cement_100", "secondary_route": "scm_substituted"},
    },
    "cotton": {
        "synonyms": ["cotton", "cotton fibre", "recycled cotton"],
        "meta": {"primary_route": "virgin_cotton", "secondary_route": "recycled_cotton"},
    },
    "polyester": {
        "synonyms": ["polyester", "pet fibre", "rpet fibre", "recycled polyester"],
        "meta": {
            "primary_route": "virgin_polyester",
            "secondary_route": "recycled_polyester_rpet",
        },
    },
    "battery_li_ion": {
        "synonyms": [
            "li-ion battery", "lithium-ion battery", "nmc", "nmc battery",
            "ev battery", "cathode recycled",
        ],
        "meta": {
            "primary_route": "virgin_cathode_nmc",
            "secondary_route": "recycled_cathode",
        },
    },
    "rare_earth_neodymium": {
        "synonyms": ["neodymium", "nd", "rare earth neodymium", "ndfeb magnet"],
        "meta": {"primary_route": "virgin_nd", "secondary_route": "recycled_magnet_scrap"},
    },
    "rare_earth_dysprosium": {
        "synonyms": ["dysprosium", "dy", "rare earth dysprosium"],
        "meta": {"primary_route": "virgin_dy", "secondary_route": "recycled_magnet_scrap"},
    },
}


class CircularMapping(BaseMapping):
    """Free-text lookup against :data:`CIRCULAR_TAXONOMY`."""

    TAXONOMY = CIRCULAR_TAXONOMY


# ---------------------------------------------------------------------------
# YAML factor loader (lazy + cached)
# ---------------------------------------------------------------------------


_DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "taxonomies"
    / "recycled_content_factors.yaml"
)


@lru_cache(maxsize=1)
def load_recycled_content_factors() -> Dict[str, Any]:
    """Load and cache the recycled-content embodied-carbon factors.

    Returns:
        dict: The parsed ``recycled_content_factors.yaml`` contents with
        ``metadata`` and ``materials`` keys.

    Raises:
        MappingError: If PyYAML is unavailable or the data file is missing.
    """
    if yaml is None:  # pragma: no cover
        raise MappingError(
            "PyYAML is required for circular_economy factor loading; "
            "install pyyaml via `pip install pyyaml`"
        )
    if not _DATA_PATH.exists():
        raise MappingError(
            "recycled_content_factors.yaml not found at %s" % _DATA_PATH
        )
    with _DATA_PATH.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    materials = payload.get("materials", {})
    if not materials:
        logger.warning(
            "recycled_content_factors.yaml has no 'materials' block at %s", _DATA_PATH
        )
    logger.debug(
        "Loaded recycled_content_factors.yaml: %d material blocks", len(materials)
    )
    return payload


def list_circular_materials() -> List[str]:
    """Return the canonical material keys that have recycled-content factors."""
    return sorted(load_recycled_content_factors().get("materials", {}).keys())


# ---------------------------------------------------------------------------
# Public mapping API
# ---------------------------------------------------------------------------


def map_circular_flow(
    material: str,
    recycled_content_pct: float = 0.0,
    source: str = "unspecified",
    route_override: Optional[str] = None,
) -> MappingResult:
    """Map a material description + recycled-content spec to a circular flow.

    The result's ``canonical`` is a dict containing:

    - ``material``: canonical material key
    - ``route``: chosen :class:`CircularRoute`
    - ``lifecycle``: chosen :class:`MaterialLifecycle`
    - ``recycled_content_pct`` / ``recycled_source``
    - ``primary_factor_kgco2e_per_kg``: virgin-route factor
    - ``secondary_factor_kgco2e_per_kg``: recycled-route factor (if known)
    - ``adjusted_factor_kgco2e_per_kg``: linear blend based on
       ``recycled_content_pct`` (primary * (1-p) + secondary * p)
    """
    lookup = CircularMapping._lookup(material)
    if lookup is None:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale="No circular material match for %r" % material,
            raw_input=material,
        )
    canonical_material = lookup.canonical

    # Resolve lifecycle / route
    try:
        source_enum = RecycledSource(source)
    except ValueError:
        source_enum = RecycledSource.UNSPECIFIED

    rc_pct = max(0.0, min(100.0, float(recycled_content_pct or 0.0)))
    if route_override:
        try:
            route = CircularRoute(route_override)
        except ValueError as exc:
            raise MappingError(
                "Invalid route_override %r; expected one of %s"
                % (route_override, [r.value for r in CircularRoute])
            ) from exc
    elif rc_pct >= 90.0:
        route = CircularRoute.SECONDARY
    elif rc_pct > 0.0:
        route = CircularRoute.MECHANICAL_RECYCLED
    else:
        route = CircularRoute.PRIMARY

    if route == CircularRoute.PRIMARY:
        lifecycle = MaterialLifecycle.VIRGIN
    elif route == CircularRoute.REMANUFACTURED:
        lifecycle = MaterialLifecycle.REMANUFACTURED
    elif route == CircularRoute.REUSED:
        lifecycle = MaterialLifecycle.REUSED
    else:
        lifecycle = MaterialLifecycle.RECYCLED

    # Factor blend
    factors = load_recycled_content_factors().get("materials", {}).get(
        canonical_material, {}
    )
    primary_factor = None
    secondary_factor = None
    if factors:
        primary_factor = float(
            factors.get("primary", {}).get("emissions_kgco2e_per_kg", 0.0) or 0.0
        )
        secondary_factor = float(
            factors.get("secondary", {}).get("emissions_kgco2e_per_kg", 0.0) or 0.0
        )
    p = rc_pct / 100.0
    if primary_factor is not None and secondary_factor is not None and primary_factor > 0:
        adjusted = primary_factor * (1.0 - p) + secondary_factor * p
    else:
        adjusted = None

    canonical = {
        "material": canonical_material,
        "route": route.value,
        "lifecycle": lifecycle.value,
        "recycled_content_pct": rc_pct,
        "recycled_source": source_enum.value,
        "primary_factor_kgco2e_per_kg": primary_factor,
        "secondary_factor_kgco2e_per_kg": secondary_factor,
        "adjusted_factor_kgco2e_per_kg": adjusted,
        "meta": CIRCULAR_TAXONOMY[canonical_material]["meta"],
    }
    confidence = min(1.0, lookup.confidence)
    logger.debug(
        "map_circular_flow(%r, %.1f%%) -> %s (route=%s)",
        material, rc_pct, canonical_material, route.value,
    )
    return MappingResult(
        canonical=canonical,
        confidence=confidence,
        band=MappingConfidence.from_score(confidence),
        rationale=(
            "Matched '%s' -> %s; route=%s (rc=%.1f%%); primary=%s, secondary=%s"
            % (
                material, canonical_material, route.value, rc_pct,
                "n/a" if primary_factor is None else f"{primary_factor:.3f}",
                "n/a" if secondary_factor is None else f"{secondary_factor:.3f}",
            )
        ),
        matched_pattern=lookup.matched_pattern,
        raw_input=material,
    )


def get_reduction_ratio(material: str) -> Optional[float]:
    """Return the secondary/primary emissions ratio for a canonical material."""
    lookup = CircularMapping._lookup(material)
    if lookup is None:
        return None
    factors = (
        load_recycled_content_factors().get("materials", {}).get(lookup.canonical, {})
    )
    ratio = factors.get("reduction_ratio")
    return float(ratio) if ratio is not None else None


__all__ = [
    "CIRCULAR_TAXONOMY",
    "CircularMapping",
    "CircularMaterialFlow",
    "CircularRoute",
    "MaterialLifecycle",
    "RecycledContent",
    "RecycledSource",
    "get_reduction_ratio",
    "list_circular_materials",
    "load_recycled_content_factors",
    "map_circular_flow",
]
