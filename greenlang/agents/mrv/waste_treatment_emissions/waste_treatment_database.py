# -*- coding: utf-8 -*-
"""
WasteTreatmentDatabaseEngine - Emission Factors, DOC, MCF, NCV Reference Data (Engine 1 of 7)

AGENT-MRV-007: On-site Waste Treatment Emissions Agent

Provides the authoritative reference data repository for all IPCC/EPA/DEFRA
waste treatment emission factors, Degradable Organic Carbon (DOC) values,
Methane Correction Factors (MCF), carbon content and fossil carbon fractions,
composting and anaerobic digestion emission factors, incineration emission
factors by technology, wastewater MCF values, Net Calorific Values (NCV),
first-order decay half-life values, maximum methane producing capacity (Bo)
values, and Global Warming Potential (GWP) values across IPCC assessment
reports.

This engine is the single source of truth for numeric constants used by
the WasteTreatmentCalculatorEngine (Engine 2).  By centralizing all emission
factors and reference data in one module, we guarantee that every calculation
in the pipeline uses identical, auditable, peer-reviewed values.

Built-In Reference Data:
    - 15 waste treatment methods with full metadata
    - 19 waste categories with DOC values (IPCC 2006 Vol 5, Table 2.4)
    - MCF values by 8 landfill/disposal site types (IPCC 2006 Vol 5, Table 3.1)
    - Carbon content and fossil carbon fraction for 16 waste types (IPCC Vol 5 Ch 5 Table 5.2)
    - Incineration EFs by 5 technology types (IPCC Vol 5 Ch 5 Table 5.3)
    - Composting/AD emission factors for 6 biological treatment types (IPCC 2019 Table 5.1)
    - Wastewater MCF values for 10 treatment system types (IPCC Vol 5 Ch 6)
    - NCV by 12 waste types (GJ/tonne wet waste)
    - Half-life values by 4 climate zones and 5 waste categories (IPCC 2006 Vol 5 Table 3.4)
    - Bo values for 8 wastewater types (IPCC Vol 5 Ch 6 Table 6.2)
    - GWP values for CO2, CH4 (fossil & biogenic), N2O, CO across AR4/AR5/AR6/AR6_20YR
    - Open burning emission factors (IPCC 2006 Vol 5 Ch 5 Section 5.3)
    - DOC fraction that decomposes (DOCf) defaults
    - Oxidation factor defaults by landfill type
    - Custom emission factor registry with thread-safe locking

Zero-Hallucination Guarantees:
    - All factors are hard-coded from published IPCC/EPA/DEFRA tables.
    - All lookups are deterministic dictionary access.
    - No LLM involvement in any data retrieval path.
    - Every query result carries a SHA-256 provenance hash.

Thread Safety:
    All reference data is immutable after initialization.  The mutable
    custom factor registry is protected by a reentrant lock.

Example:
    >>> from greenlang.agents.mrv.waste_treatment_emissions.waste_treatment_database import (
    ...     WasteTreatmentDatabaseEngine,
    ... )
    >>> db = WasteTreatmentDatabaseEngine()
    >>> doc = db.get_doc_value("food_waste")
    >>> mcf = db.get_mcf_value("managed_anaerobic")
    >>> gwp = db.get_gwp("CH4", "AR6")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-007 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-007)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["WasteTreatmentDatabaseEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.waste_treatment_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.waste_treatment_emissions.metrics import (
        record_component_operation as _record_db_operation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_db_operation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


# ===========================================================================
# Enumerations
# ===========================================================================


class WasteCategory(str, Enum):
    """IPCC waste categories per 2006 Guidelines Vol 5.

    Covers all 19 waste stream types used for DOC values, carbon content,
    and NCV lookups.  Categories align with IPCC 2006 Vol 5 Table 2.4
    and national inventory categorisation.
    """

    FOOD_WASTE = "FOOD_WASTE"
    YARD_WASTE = "YARD_WASTE"
    PAPER = "PAPER"
    CARDBOARD = "CARDBOARD"
    WOOD = "WOOD"
    TEXTILES = "TEXTILES"
    TEXTILES_SYNTHETIC = "TEXTILES_SYNTHETIC"
    TEXTILES_NATURAL = "TEXTILES_NATURAL"
    RUBBER = "RUBBER"
    LEATHER = "LEATHER"
    PLASTIC = "PLASTIC"
    METAL = "METAL"
    GLASS = "GLASS"
    OTHER_INERT = "OTHER_INERT"
    MSW = "MSW"
    NAPPIES = "NAPPIES"
    SLUDGE = "SLUDGE"
    CLINICAL_WASTE = "CLINICAL_WASTE"
    INDUSTRIAL_WASTE = "INDUSTRIAL_WASTE"


class TreatmentMethod(str, Enum):
    """Waste treatment methods covered by this agent.

    15 treatment methods spanning landfill, biological, thermal,
    and other waste management pathways per IPCC 2006 Vol 5.
    """

    MANAGED_ANAEROBIC = "MANAGED_ANAEROBIC"
    MANAGED_SEMI_AEROBIC = "MANAGED_SEMI_AEROBIC"
    UNMANAGED_DEEP = "UNMANAGED_DEEP"
    UNMANAGED_SHALLOW = "UNMANAGED_SHALLOW"
    INCINERATION = "INCINERATION"
    OPEN_BURNING = "OPEN_BURNING"
    COMPOSTING = "COMPOSTING"
    ANAEROBIC_DIGESTION = "ANAEROBIC_DIGESTION"
    MBT = "MBT"
    PYROLYSIS = "PYROLYSIS"
    GASIFICATION = "GASIFICATION"
    WASTEWATER_TREATMENT = "WASTEWATER_TREATMENT"
    LANDFILL_GAS_RECOVERY = "LANDFILL_GAS_RECOVERY"
    RECYCLING = "RECYCLING"
    AUTOCLAVE = "AUTOCLAVE"


class IncineratorType(str, Enum):
    """Incineration technology types per IPCC Vol 5 Ch 5 Table 5.3.

    STOKER_GRATE: Mass-burn stoker grate incinerator (most common).
    FLUIDIZED_BED: Fluidized bed combustion system.
    ROTARY_KILN: Rotary kiln incinerator (hazardous/clinical waste).
    SEMI_CONTINUOUS: Semi-continuous (starved air) incinerator.
    BATCH_TYPE: Batch-type incinerator (small-scale, developing countries).
    """

    STOKER_GRATE = "STOKER_GRATE"
    FLUIDIZED_BED = "FLUIDIZED_BED"
    ROTARY_KILN = "ROTARY_KILN"
    SEMI_CONTINUOUS = "SEMI_CONTINUOUS"
    BATCH_TYPE = "BATCH_TYPE"


class CompostingType(str, Enum):
    """Biological treatment types per IPCC 2019 Refinement Table 5.1.

    WELL_MANAGED: Well-managed windrow or in-vessel composting.
    POORLY_MANAGED: Poorly managed or uncontrolled composting.
    AD_VENTED: Anaerobic digestion with biogas vented (no flare).
    AD_FLARED: Anaerobic digestion with biogas flared.
    MBT_AEROBIC: Mechanical-biological treatment (aerobic phase).
    MBT_ANAEROBIC: Mechanical-biological treatment (anaerobic phase).
    """

    WELL_MANAGED = "WELL_MANAGED"
    POORLY_MANAGED = "POORLY_MANAGED"
    AD_VENTED = "AD_VENTED"
    AD_FLARED = "AD_FLARED"
    MBT_AEROBIC = "MBT_AEROBIC"
    MBT_ANAEROBIC = "MBT_ANAEROBIC"


class LandfillType(str, Enum):
    """Landfill/disposal site types for MCF selection.

    Per IPCC 2006 Vol 5, Table 3.1.
    """

    MANAGED_ANAEROBIC = "MANAGED_ANAEROBIC"
    MANAGED_SEMI_AEROBIC = "MANAGED_SEMI_AEROBIC"
    UNMANAGED_DEEP = "UNMANAGED_DEEP"
    UNMANAGED_SHALLOW = "UNMANAGED_SHALLOW"
    UNCATEGORISED_SWDS = "UNCATEGORISED_SWDS"
    OPEN_DUMP_COVERED = "OPEN_DUMP_COVERED"
    OPEN_DUMP_UNCOVERED = "OPEN_DUMP_UNCOVERED"
    INDUSTRIAL_LANDFILL = "INDUSTRIAL_LANDFILL"


class WastewaterSystemType(str, Enum):
    """Wastewater treatment system types for MCF selection.

    Per IPCC 2006 Vol 5 Ch 6 Table 6.3.
    """

    AEROBIC_WELL_MANAGED = "AEROBIC_WELL_MANAGED"
    AEROBIC_OVERLOADED = "AEROBIC_OVERLOADED"
    ANAEROBIC_REACTOR = "ANAEROBIC_REACTOR"
    ANAEROBIC_SHALLOW_LAGOON = "ANAEROBIC_SHALLOW_LAGOON"
    ANAEROBIC_DEEP_LAGOON = "ANAEROBIC_DEEP_LAGOON"
    FACULTATIVE_LAGOON = "FACULTATIVE_LAGOON"
    SEPTIC_SYSTEM = "SEPTIC_SYSTEM"
    LATRINE_DRY = "LATRINE_DRY"
    LATRINE_WET = "LATRINE_WET"
    UNTREATED_DISCHARGE = "UNTREATED_DISCHARGE"


class ClimateZoneWaste(str, Enum):
    """Climate zone classification for waste decay half-life selection.

    Simplified four-zone scheme used in IPCC 2006 Vol 5 Table 3.4.
    """

    TROPICAL_WET = "TROPICAL_WET"
    TROPICAL_DRY = "TROPICAL_DRY"
    TEMPERATE_WET = "TEMPERATE_WET"
    TEMPERATE_DRY = "TEMPERATE_DRY"
    BOREAL_WET = "BOREAL_WET"
    BOREAL_DRY = "BOREAL_DRY"


class DecayCategory(str, Enum):
    """Waste decay rate categories for first-order decay half-life.

    Per IPCC 2006 Vol 5 Table 3.4.  Wastes are grouped by
    biodegradability speed.
    """

    RAPIDLY_DEGRADING = "RAPIDLY_DEGRADING"
    MODERATELY_DEGRADING = "MODERATELY_DEGRADING"
    SLOWLY_DEGRADING = "SLOWLY_DEGRADING"
    VERY_SLOWLY_DEGRADING = "VERY_SLOWLY_DEGRADING"
    NON_DEGRADING = "NON_DEGRADING"


class GWPSource(str, Enum):
    """IPCC Assessment Report editions for GWP values.

    AR4: Fourth Assessment Report (2007).
    AR5: Fifth Assessment Report (2014).
    AR6: Sixth Assessment Report (2021), 100-year horizon.
    AR6_20YR: Sixth Assessment Report (2021), 20-year horizon.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class WastewaterType(str, Enum):
    """Wastewater types for Bo (max CH4 producing capacity) lookup.

    Per IPCC 2006 Vol 5 Ch 6 Table 6.2.
    """

    DOMESTIC = "DOMESTIC"
    INDUSTRIAL_PULP_PAPER = "INDUSTRIAL_PULP_PAPER"
    INDUSTRIAL_FOOD_BEVERAGE = "INDUSTRIAL_FOOD_BEVERAGE"
    INDUSTRIAL_MEAT_POULTRY = "INDUSTRIAL_MEAT_POULTRY"
    INDUSTRIAL_DAIRY = "INDUSTRIAL_DAIRY"
    INDUSTRIAL_ALCOHOL_STARCH = "INDUSTRIAL_ALCOHOL_STARCH"
    INDUSTRIAL_ORGANIC_CHEMICALS = "INDUSTRIAL_ORGANIC_CHEMICALS"
    INDUSTRIAL_VEGETABLE_OIL = "INDUSTRIAL_VEGETABLE_OIL"


class EmissionFactorSource(str, Enum):
    """Sources for emission factor data.

    IPCC_2006: IPCC 2006 Guidelines for National GHG Inventories Vol 5.
    IPCC_2019: 2019 Refinement to the 2006 Guidelines.
    EPA: US EPA emission factors.
    DEFRA: UK DEFRA conversion factors.
    COUNTRY_SPECIFIC: National inventory-derived factors.
    SITE_MEASURED: Direct facility measurement values.
    CUSTOM: User-provided emission factors.
    """

    IPCC_2006 = "IPCC_2006"
    IPCC_2019 = "IPCC_2019"
    EPA = "EPA"
    DEFRA = "DEFRA"
    COUNTRY_SPECIFIC = "COUNTRY_SPECIFIC"
    SITE_MEASURED = "SITE_MEASURED"
    CUSTOM = "CUSTOM"


# ===========================================================================
# Dataclasses for structured lookups
# ===========================================================================


@dataclass(frozen=True)
class TreatmentMethodInfo:
    """Metadata for a waste treatment method.

    Attributes:
        code: Unique treatment method code.
        name: Human-readable name.
        description: Technical description.
        applicable_waste_types: Waste categories this method accepts.
        primary_gases: Greenhouse gases emitted.
        regulatory_refs: Applicable regulatory references.
        ipcc_chapter: IPCC Guidelines chapter reference.
    """

    code: str
    name: str
    description: str
    applicable_waste_types: Tuple[str, ...]
    primary_gases: Tuple[str, ...]
    regulatory_refs: Tuple[str, ...]
    ipcc_chapter: str


@dataclass(frozen=True)
class CarbonContentRecord:
    """Carbon content and fossil fraction for a waste type.

    Per IPCC 2006 Vol 5 Ch 5 Table 5.2.

    Attributes:
        waste_category: Waste type identifier.
        carbon_pct: Total carbon content as fraction of wet weight (0-1).
        fossil_fraction: Fraction of carbon that is fossil-derived (0-1).
        source: Reference citation.
    """

    waste_category: str
    carbon_pct: Decimal
    fossil_fraction: Decimal
    source: str


@dataclass(frozen=True)
class IncinerationEF:
    """Incineration emission factors per technology type.

    Per IPCC 2006 Vol 5 Ch 5 Table 5.3.
    Units: kg per Gg of waste incinerated.

    Attributes:
        technology: Incinerator technology type.
        n2o_kg_per_gg: N2O emission factor (kg/Gg waste).
        ch4_kg_per_gg: CH4 emission factor (kg/Gg waste).
        source: Reference citation.
    """

    technology: str
    n2o_kg_per_gg: Decimal
    ch4_kg_per_gg: Decimal
    source: str


@dataclass(frozen=True)
class CompostingEF:
    """Composting/biological treatment emission factors.

    Per IPCC 2019 Refinement Table 5.1.
    Units: g per kg of waste treated.

    Attributes:
        treatment_type: Biological treatment type.
        ch4_g_per_kg: CH4 emission factor (g/kg waste treated).
        n2o_g_per_kg: N2O emission factor (g/kg waste treated).
        source: Reference citation.
    """

    treatment_type: str
    ch4_g_per_kg: Decimal
    n2o_g_per_kg: Decimal
    source: str


@dataclass(frozen=True)
class OpenBurningEF:
    """Open burning emission factors per waste type.

    Per IPCC 2006 Vol 5 Ch 5 Section 5.3.
    Units: g per kg of waste burned.

    Attributes:
        waste_category: Waste category being burned.
        co2_g_per_kg: CO2 emission factor (g/kg waste burned).
        ch4_g_per_kg: CH4 emission factor (g/kg waste burned).
        n2o_g_per_kg: N2O emission factor (g/kg waste burned).
        source: Reference citation.
    """

    waste_category: str
    co2_g_per_kg: Decimal
    ch4_g_per_kg: Decimal
    n2o_g_per_kg: Decimal
    source: str


# ===========================================================================
# GWP Values (IPCC AR4/AR5/AR6/AR6_20YR)
# ===========================================================================

#: Global Warming Potential values for waste-sector gases.
#: Keys: (GWP_source, gas), Values: Decimal GWP factor.
#: CH4_FOSSIL and CH4_BIOGENIC distinguished per AR6 for waste sector.
GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": _D("1"),
        "CH4": _D("25"),
        "CH4_FOSSIL": _D("25"),
        "CH4_BIOGENIC": _D("25"),
        "N2O": _D("298"),
        "CO": _D("1.9"),
    },
    "AR5": {
        "CO2": _D("1"),
        "CH4": _D("28"),
        "CH4_FOSSIL": _D("30"),
        "CH4_BIOGENIC": _D("28"),
        "N2O": _D("265"),
        "CO": _D("1.9"),
    },
    "AR6": {
        "CO2": _D("1"),
        "CH4": _D("29.8"),
        "CH4_FOSSIL": _D("29.8"),
        "CH4_BIOGENIC": _D("27.0"),
        "N2O": _D("273"),
        "CO": _D("2.3"),
    },
    "AR6_20YR": {
        "CO2": _D("1"),
        "CH4": _D("82.5"),
        "CH4_FOSSIL": _D("82.5"),
        "CH4_BIOGENIC": _D("80.8"),
        "N2O": _D("273"),
        "CO": _D("2.3"),
    },
}


# ===========================================================================
# DOC Values - Degradable Organic Carbon (fraction of wet weight)
# IPCC 2006 Vol 5, Table 2.4
# ===========================================================================

#: DOC values by waste category.
#: Fraction of wet weight that is degradable organic carbon.
#: Source: IPCC 2006 Vol 5 Ch 2 Table 2.4.
DOC_VALUES: Dict[str, Decimal] = {
    "FOOD_WASTE": _D("0.15"),
    "YARD_WASTE": _D("0.20"),
    "PAPER": _D("0.40"),
    "CARDBOARD": _D("0.40"),
    "WOOD": _D("0.43"),
    "TEXTILES": _D("0.24"),
    "TEXTILES_SYNTHETIC": _D("0.24"),
    "TEXTILES_NATURAL": _D("0.24"),
    "RUBBER": _D("0.39"),
    "LEATHER": _D("0.39"),
    "PLASTIC": _D("0"),
    "METAL": _D("0"),
    "GLASS": _D("0"),
    "OTHER_INERT": _D("0"),
    "MSW": _D("0.18"),
    "NAPPIES": _D("0.24"),
    "SLUDGE": _D("0.05"),
    "CLINICAL_WASTE": _D("0.15"),
    "INDUSTRIAL_WASTE": _D("0.15"),
}


# ===========================================================================
# MCF - Methane Correction Factor by landfill/disposal site type
# IPCC 2006 Vol 5, Table 3.1
# ===========================================================================

#: MCF values by landfill/disposal site type.
#: Dimensionless fraction (0 to 1).
#: Source: IPCC 2006 Vol 5 Ch 3 Table 3.1.
MCF_VALUES: Dict[str, Decimal] = {
    "MANAGED_ANAEROBIC": _D("1.0"),
    "MANAGED_SEMI_AEROBIC": _D("0.5"),
    "UNMANAGED_DEEP": _D("0.8"),
    "UNMANAGED_SHALLOW": _D("0.4"),
    "UNCATEGORISED_SWDS": _D("0.6"),
    "OPEN_DUMP_COVERED": _D("0.5"),
    "OPEN_DUMP_UNCOVERED": _D("0.4"),
    "INDUSTRIAL_LANDFILL": _D("1.0"),
}


# ===========================================================================
# Carbon Content and Fossil Carbon Fraction
# IPCC 2006 Vol 5 Ch 5 Table 5.2
# ===========================================================================

#: Carbon content (fraction of wet weight) and fossil carbon fraction.
#: Source: IPCC 2006 Vol 5 Ch 5 Table 5.2.
CARBON_CONTENT: Dict[str, CarbonContentRecord] = {
    "FOOD_WASTE": CarbonContentRecord(
        waste_category="FOOD_WASTE",
        carbon_pct=_D("0.15"),
        fossil_fraction=_D("0"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "PAPER": CarbonContentRecord(
        waste_category="PAPER",
        carbon_pct=_D("0.375"),
        fossil_fraction=_D("0.01"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "CARDBOARD": CarbonContentRecord(
        waste_category="CARDBOARD",
        carbon_pct=_D("0.34"),
        fossil_fraction=_D("0.01"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "PLASTIC": CarbonContentRecord(
        waste_category="PLASTIC",
        carbon_pct=_D("0.675"),
        fossil_fraction=_D("1.0"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "TEXTILES_SYNTHETIC": CarbonContentRecord(
        waste_category="TEXTILES_SYNTHETIC",
        carbon_pct=_D("0.45"),
        fossil_fraction=_D("0.80"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "TEXTILES_NATURAL": CarbonContentRecord(
        waste_category="TEXTILES_NATURAL",
        carbon_pct=_D("0.45"),
        fossil_fraction=_D("0"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "TEXTILES": CarbonContentRecord(
        waste_category="TEXTILES",
        carbon_pct=_D("0.45"),
        fossil_fraction=_D("0.40"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "RUBBER": CarbonContentRecord(
        waste_category="RUBBER",
        carbon_pct=_D("0.50"),
        fossil_fraction=_D("0.20"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "LEATHER": CarbonContentRecord(
        waste_category="LEATHER",
        carbon_pct=_D("0.45"),
        fossil_fraction=_D("0"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "WOOD": CarbonContentRecord(
        waste_category="WOOD",
        carbon_pct=_D("0.465"),
        fossil_fraction=_D("0"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "YARD_WASTE": CarbonContentRecord(
        waste_category="YARD_WASTE",
        carbon_pct=_D("0.185"),
        fossil_fraction=_D("0"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "SLUDGE": CarbonContentRecord(
        waste_category="SLUDGE",
        carbon_pct=_D("0.10"),
        fossil_fraction=_D("0"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "NAPPIES": CarbonContentRecord(
        waste_category="NAPPIES",
        carbon_pct=_D("0.34"),
        fossil_fraction=_D("0.10"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "CLINICAL_WASTE": CarbonContentRecord(
        waste_category="CLINICAL_WASTE",
        carbon_pct=_D("0.30"),
        fossil_fraction=_D("0.40"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
    "MSW": CarbonContentRecord(
        waste_category="MSW",
        carbon_pct=_D("0.28"),
        fossil_fraction=_D("0.40"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2 (aggregate)",
    ),
    "INDUSTRIAL_WASTE": CarbonContentRecord(
        waste_category="INDUSTRIAL_WASTE",
        carbon_pct=_D("0.25"),
        fossil_fraction=_D("0.50"),
        source="IPCC 2006 Vol5 Ch5 Table 5.2",
    ),
}


# ===========================================================================
# Incineration Emission Factors by Technology
# IPCC 2006 Vol 5 Ch 5 Table 5.3
# Units: kg per Gg of waste incinerated
# ===========================================================================

#: Incineration emission factors by incinerator technology.
#: N2O and CH4 in kg per Gg (1000 tonnes) of waste incinerated.
#: Source: IPCC 2006 Vol 5 Ch 5 Table 5.3.
INCINERATION_EF: Dict[str, IncinerationEF] = {
    "STOKER_GRATE": IncinerationEF(
        technology="STOKER_GRATE",
        n2o_kg_per_gg=_D("50"),
        ch4_kg_per_gg=_D("0.2"),
        source="IPCC 2006 Vol5 Ch5 Table 5.3",
    ),
    "FLUIDIZED_BED": IncinerationEF(
        technology="FLUIDIZED_BED",
        n2o_kg_per_gg=_D("56"),
        ch4_kg_per_gg=_D("0.68"),
        source="IPCC 2006 Vol5 Ch5 Table 5.3",
    ),
    "ROTARY_KILN": IncinerationEF(
        technology="ROTARY_KILN",
        n2o_kg_per_gg=_D("50"),
        ch4_kg_per_gg=_D("0.2"),
        source="IPCC 2006 Vol5 Ch5 Table 5.3",
    ),
    "SEMI_CONTINUOUS": IncinerationEF(
        technology="SEMI_CONTINUOUS",
        n2o_kg_per_gg=_D("60"),
        ch4_kg_per_gg=_D("6.0"),
        source="IPCC 2006 Vol5 Ch5 Table 5.3",
    ),
    "BATCH_TYPE": IncinerationEF(
        technology="BATCH_TYPE",
        n2o_kg_per_gg=_D("60"),
        ch4_kg_per_gg=_D("60"),
        source="IPCC 2006 Vol5 Ch5 Table 5.3",
    ),
}


# ===========================================================================
# Composting / Biological Treatment Emission Factors
# IPCC 2019 Refinement Table 5.1
# Units: g per kg of waste treated
# ===========================================================================

#: Biological treatment emission factors.
#: CH4 and N2O in g per kg of waste treated.
#: Source: IPCC 2019 Refinement to 2006 GL, Table 5.1.
COMPOSTING_EF: Dict[str, CompostingEF] = {
    "WELL_MANAGED": CompostingEF(
        treatment_type="WELL_MANAGED",
        ch4_g_per_kg=_D("4.0"),
        n2o_g_per_kg=_D("0.24"),
        source="IPCC 2019 Refinement Table 5.1",
    ),
    "POORLY_MANAGED": CompostingEF(
        treatment_type="POORLY_MANAGED",
        ch4_g_per_kg=_D("10.0"),
        n2o_g_per_kg=_D("0.6"),
        source="IPCC 2019 Refinement Table 5.1",
    ),
    "AD_VENTED": CompostingEF(
        treatment_type="AD_VENTED",
        ch4_g_per_kg=_D("2.0"),
        n2o_g_per_kg=_D("0"),
        source="IPCC 2019 Refinement Table 5.1",
    ),
    "AD_FLARED": CompostingEF(
        treatment_type="AD_FLARED",
        ch4_g_per_kg=_D("0.8"),
        n2o_g_per_kg=_D("0"),
        source="IPCC 2019 Refinement Table 5.1",
    ),
    "MBT_AEROBIC": CompostingEF(
        treatment_type="MBT_AEROBIC",
        ch4_g_per_kg=_D("4.0"),
        n2o_g_per_kg=_D("0.3"),
        source="IPCC 2019 Refinement Table 5.1",
    ),
    "MBT_ANAEROBIC": CompostingEF(
        treatment_type="MBT_ANAEROBIC",
        ch4_g_per_kg=_D("2.0"),
        n2o_g_per_kg=_D("0.1"),
        source="IPCC 2019 Refinement Table 5.1",
    ),
}


# ===========================================================================
# Wastewater MCF Values
# IPCC 2006 Vol 5 Ch 6 Table 6.3
# ===========================================================================

#: MCF values by wastewater treatment system type.
#: Dimensionless fraction (0 to 1).
#: Source: IPCC 2006 Vol 5 Ch 6 Table 6.3.
WASTEWATER_MCF: Dict[str, Decimal] = {
    "AEROBIC_WELL_MANAGED": _D("0"),
    "AEROBIC_OVERLOADED": _D("0.3"),
    "ANAEROBIC_REACTOR": _D("0.8"),
    "ANAEROBIC_SHALLOW_LAGOON": _D("0.2"),
    "ANAEROBIC_DEEP_LAGOON": _D("0.8"),
    "FACULTATIVE_LAGOON": _D("0.2"),
    "SEPTIC_SYSTEM": _D("0.5"),
    "LATRINE_DRY": _D("0.1"),
    "LATRINE_WET": _D("0.7"),
    "UNTREATED_DISCHARGE": _D("0.1"),
}


# ===========================================================================
# Bo Values - Maximum CH4 Producing Capacity (kg CH4/kg COD)
# IPCC 2006 Vol 5 Ch 6 Table 6.2
# ===========================================================================

#: Maximum methane producing capacity by wastewater type.
#: Units: kg CH4 per kg COD (chemical oxygen demand).
#: Source: IPCC 2006 Vol 5 Ch 6 Table 6.2.
BO_VALUES: Dict[str, Decimal] = {
    "DOMESTIC": _D("0.25"),
    "INDUSTRIAL_PULP_PAPER": _D("0.25"),
    "INDUSTRIAL_FOOD_BEVERAGE": _D("0.25"),
    "INDUSTRIAL_MEAT_POULTRY": _D("0.25"),
    "INDUSTRIAL_DAIRY": _D("0.25"),
    "INDUSTRIAL_ALCOHOL_STARCH": _D("0.25"),
    "INDUSTRIAL_ORGANIC_CHEMICALS": _D("0.25"),
    "INDUSTRIAL_VEGETABLE_OIL": _D("0.25"),
}

#: Bo values by wastewater type in kg CH4 per kg BOD.
#: Some protocols use BOD instead of COD.
BO_VALUES_BOD: Dict[str, Decimal] = {
    "DOMESTIC": _D("0.60"),
    "INDUSTRIAL_PULP_PAPER": _D("0.55"),
    "INDUSTRIAL_FOOD_BEVERAGE": _D("0.58"),
    "INDUSTRIAL_MEAT_POULTRY": _D("0.60"),
    "INDUSTRIAL_DAIRY": _D("0.55"),
    "INDUSTRIAL_ALCOHOL_STARCH": _D("0.62"),
    "INDUSTRIAL_ORGANIC_CHEMICALS": _D("0.50"),
    "INDUSTRIAL_VEGETABLE_OIL": _D("0.58"),
}


# ===========================================================================
# NCV - Net Calorific Values by Waste Type
# Units: GJ per tonne of wet waste
# Source: IPCC 2006 Vol 5 Ch 5 Table 5.2 / DEFRA 2025
# ===========================================================================

#: Net Calorific Value by waste category.
#: Units: GJ per tonne of wet waste.
NCV_VALUES: Dict[str, Decimal] = {
    "FOOD_WASTE": _D("4.0"),
    "YARD_WASTE": _D("6.0"),
    "PAPER": _D("12.2"),
    "CARDBOARD": _D("11.0"),
    "WOOD": _D("15.4"),
    "TEXTILES": _D("16.0"),
    "TEXTILES_SYNTHETIC": _D("22.0"),
    "TEXTILES_NATURAL": _D("14.5"),
    "RUBBER": _D("23.0"),
    "LEATHER": _D("17.0"),
    "PLASTIC": _D("32.0"),
    "NAPPIES": _D("8.0"),
    "SLUDGE": _D("3.5"),
    "CLINICAL_WASTE": _D("15.0"),
    "MSW": _D("9.0"),
    "INDUSTRIAL_WASTE": _D("12.0"),
    "METAL": _D("0"),
    "GLASS": _D("0"),
    "OTHER_INERT": _D("0"),
}


# ===========================================================================
# Half-Life Values for First-Order Decay (years)
# IPCC 2006 Vol 5 Table 3.4
# By climate zone and waste decay category
# ===========================================================================

#: Half-life values (years) by (climate_zone, decay_category).
#: Source: IPCC 2006 Vol 5 Ch 3 Table 3.4.
HALF_LIFE_VALUES: Dict[str, Dict[str, Decimal]] = {
    "TROPICAL_WET": {
        "RAPIDLY_DEGRADING": _D("3"),
        "MODERATELY_DEGRADING": _D("5"),
        "SLOWLY_DEGRADING": _D("10"),
        "VERY_SLOWLY_DEGRADING": _D("25"),
        "NON_DEGRADING": _D("0"),
    },
    "TROPICAL_DRY": {
        "RAPIDLY_DEGRADING": _D("5"),
        "MODERATELY_DEGRADING": _D("8"),
        "SLOWLY_DEGRADING": _D("15"),
        "VERY_SLOWLY_DEGRADING": _D("35"),
        "NON_DEGRADING": _D("0"),
    },
    "TEMPERATE_WET": {
        "RAPIDLY_DEGRADING": _D("5"),
        "MODERATELY_DEGRADING": _D("8"),
        "SLOWLY_DEGRADING": _D("15"),
        "VERY_SLOWLY_DEGRADING": _D("35"),
        "NON_DEGRADING": _D("0"),
    },
    "TEMPERATE_DRY": {
        "RAPIDLY_DEGRADING": _D("8"),
        "MODERATELY_DEGRADING": _D("12"),
        "SLOWLY_DEGRADING": _D("23"),
        "VERY_SLOWLY_DEGRADING": _D("46"),
        "NON_DEGRADING": _D("0"),
    },
    "BOREAL_WET": {
        "RAPIDLY_DEGRADING": _D("8"),
        "MODERATELY_DEGRADING": _D("12"),
        "SLOWLY_DEGRADING": _D("23"),
        "VERY_SLOWLY_DEGRADING": _D("46"),
        "NON_DEGRADING": _D("0"),
    },
    "BOREAL_DRY": {
        "RAPIDLY_DEGRADING": _D("12"),
        "MODERATELY_DEGRADING": _D("17"),
        "SLOWLY_DEGRADING": _D("35"),
        "VERY_SLOWLY_DEGRADING": _D("69"),
        "NON_DEGRADING": _D("0"),
    },
}

#: Mapping from waste category to decay category.
#: Source: IPCC 2006 Vol 5 Ch 3 Table 3.4 footnotes.
WASTE_TO_DECAY_CATEGORY: Dict[str, str] = {
    "FOOD_WASTE": "RAPIDLY_DEGRADING",
    "YARD_WASTE": "MODERATELY_DEGRADING",
    "PAPER": "MODERATELY_DEGRADING",
    "CARDBOARD": "MODERATELY_DEGRADING",
    "WOOD": "SLOWLY_DEGRADING",
    "TEXTILES": "MODERATELY_DEGRADING",
    "TEXTILES_SYNTHETIC": "VERY_SLOWLY_DEGRADING",
    "TEXTILES_NATURAL": "MODERATELY_DEGRADING",
    "RUBBER": "VERY_SLOWLY_DEGRADING",
    "LEATHER": "SLOWLY_DEGRADING",
    "NAPPIES": "MODERATELY_DEGRADING",
    "SLUDGE": "RAPIDLY_DEGRADING",
    "CLINICAL_WASTE": "MODERATELY_DEGRADING",
    "MSW": "MODERATELY_DEGRADING",
    "INDUSTRIAL_WASTE": "MODERATELY_DEGRADING",
    "PLASTIC": "NON_DEGRADING",
    "METAL": "NON_DEGRADING",
    "GLASS": "NON_DEGRADING",
    "OTHER_INERT": "NON_DEGRADING",
}


# ===========================================================================
# DOCf - Fraction of DOC that Decomposes
# IPCC 2006 Vol 5 Ch 3 Section 3.2.3
# ===========================================================================

#: DOCf (fraction of DOC that can decompose under anaerobic conditions).
#: IPCC default = 0.5; site-specific values may differ.
DOCF_DEFAULT: Decimal = _D("0.5")

#: DOCf by waste category (where data is available from 2019 Refinement).
DOCF_VALUES: Dict[str, Decimal] = {
    "FOOD_WASTE": _D("0.70"),
    "YARD_WASTE": _D("0.50"),
    "PAPER": _D("0.50"),
    "CARDBOARD": _D("0.50"),
    "WOOD": _D("0.50"),
    "TEXTILES": _D("0.50"),
    "TEXTILES_SYNTHETIC": _D("0.50"),
    "TEXTILES_NATURAL": _D("0.50"),
    "RUBBER": _D("0.50"),
    "LEATHER": _D("0.50"),
    "NAPPIES": _D("0.50"),
    "SLUDGE": _D("0.70"),
    "CLINICAL_WASTE": _D("0.50"),
    "MSW": _D("0.50"),
    "INDUSTRIAL_WASTE": _D("0.50"),
    "PLASTIC": _D("0"),
    "METAL": _D("0"),
    "GLASS": _D("0"),
    "OTHER_INERT": _D("0"),
}


# ===========================================================================
# Oxidation Factor (OX)
# IPCC 2006 Vol 5 Ch 3 Table 3.2
# ===========================================================================

#: Fraction of CH4 oxidised in landfill cover soil.
#: Source: IPCC 2006 Vol 5 Ch 3 Table 3.2.
OXIDATION_FACTOR: Dict[str, Decimal] = {
    "MANAGED_ANAEROBIC": _D("0.10"),
    "MANAGED_SEMI_AEROBIC": _D("0.10"),
    "UNMANAGED_DEEP": _D("0"),
    "UNMANAGED_SHALLOW": _D("0"),
    "UNCATEGORISED_SWDS": _D("0"),
    "OPEN_DUMP_COVERED": _D("0.10"),
    "OPEN_DUMP_UNCOVERED": _D("0"),
    "INDUSTRIAL_LANDFILL": _D("0.10"),
}


# ===========================================================================
# Open Burning Emission Factors
# IPCC 2006 Vol 5 Ch 5 Section 5.3
# Units: g per kg of waste burned
# ===========================================================================

#: Open burning emission factors by waste category.
#: Source: IPCC 2006 Vol 5 Ch 5 Eq 5.7 / EPA AP-42 Ch 2.5.
OPEN_BURNING_EF: Dict[str, OpenBurningEF] = {
    "MSW": OpenBurningEF(
        waste_category="MSW",
        co2_g_per_kg=_D("1453"),
        ch4_g_per_kg=_D("6.5"),
        n2o_g_per_kg=_D("0.15"),
        source="IPCC 2006 Vol5 Ch5 Table 5.4 / EPA AP-42",
    ),
    "FOOD_WASTE": OpenBurningEF(
        waste_category="FOOD_WASTE",
        co2_g_per_kg=_D("1200"),
        ch4_g_per_kg=_D("10.0"),
        n2o_g_per_kg=_D("0.10"),
        source="IPCC 2006 Vol5 Ch5 / EPA AP-42 Ch 2.5",
    ),
    "YARD_WASTE": OpenBurningEF(
        waste_category="YARD_WASTE",
        co2_g_per_kg=_D("1515"),
        ch4_g_per_kg=_D("6.5"),
        n2o_g_per_kg=_D("0.15"),
        source="IPCC 2006 Vol5 Ch5 / EPA AP-42 Ch 2.5",
    ),
    "PAPER": OpenBurningEF(
        waste_category="PAPER",
        co2_g_per_kg=_D("1500"),
        ch4_g_per_kg=_D("3.0"),
        n2o_g_per_kg=_D("0.07"),
        source="IPCC 2006 Vol5 Ch5 / EPA AP-42 Ch 2.5",
    ),
    "PLASTIC": OpenBurningEF(
        waste_category="PLASTIC",
        co2_g_per_kg=_D("2200"),
        ch4_g_per_kg=_D("8.0"),
        n2o_g_per_kg=_D("0.20"),
        source="IPCC 2006 Vol5 Ch5 / EPA AP-42 Ch 2.5",
    ),
    "TEXTILES": OpenBurningEF(
        waste_category="TEXTILES",
        co2_g_per_kg=_D("1600"),
        ch4_g_per_kg=_D("5.0"),
        n2o_g_per_kg=_D("0.15"),
        source="IPCC 2006 Vol5 Ch5 / EPA AP-42 Ch 2.5",
    ),
    "WOOD": OpenBurningEF(
        waste_category="WOOD",
        co2_g_per_kg=_D("1580"),
        ch4_g_per_kg=_D("6.8"),
        n2o_g_per_kg=_D("0.20"),
        source="IPCC 2006 Vol5 Ch5 / EPA AP-42 Ch 2.5",
    ),
    "RUBBER": OpenBurningEF(
        waste_category="RUBBER",
        co2_g_per_kg=_D("1850"),
        ch4_g_per_kg=_D("10.0"),
        n2o_g_per_kg=_D("0.25"),
        source="IPCC 2006 Vol5 Ch5 / EPA AP-42 Ch 2.5",
    ),
    "CLINICAL_WASTE": OpenBurningEF(
        waste_category="CLINICAL_WASTE",
        co2_g_per_kg=_D("1500"),
        ch4_g_per_kg=_D("7.0"),
        n2o_g_per_kg=_D("0.50"),
        source="IPCC 2006 Vol5 Ch5 / WHO guidance",
    ),
}


# ===========================================================================
# Methane Fraction in Landfill Gas
# IPCC 2006 Vol 5 Ch 3 Section 3.2.3
# ===========================================================================

#: Default methane fraction in generated landfill gas (volume fraction).
METHANE_FRACTION_DEFAULT: Decimal = _D("0.50")

#: Methane fraction by waste composition (optional Tier 2).
METHANE_FRACTION: Dict[str, Decimal] = {
    "FOOD_WASTE": _D("0.50"),
    "YARD_WASTE": _D("0.50"),
    "PAPER": _D("0.50"),
    "CARDBOARD": _D("0.50"),
    "WOOD": _D("0.50"),
    "TEXTILES": _D("0.50"),
    "MSW": _D("0.50"),
    "SLUDGE": _D("0.55"),
    "INDUSTRIAL_WASTE": _D("0.50"),
}

#: Methane density at STP (kg/m3).
METHANE_DENSITY_KG_M3: Decimal = _D("0.6706")

#: CO2 to C conversion factor (12/44).
CO2_TO_C_RATIO: Decimal = _D("0.27273")

#: C to CO2 conversion factor (44/12).
C_TO_CO2_RATIO: Decimal = _D("3.66667")

#: Molecular weight of CH4 / C (16/12).
CH4_C_RATIO: Decimal = _D("1.33333")


# ===========================================================================
# Treatment Method Metadata (15 methods)
# ===========================================================================

#: Complete metadata for all 15 waste treatment methods.
TREATMENT_METHODS: Dict[str, TreatmentMethodInfo] = {
    "MANAGED_ANAEROBIC": TreatmentMethodInfo(
        code="MANAGED_ANAEROBIC",
        name="Managed Anaerobic Landfill",
        description="Sanitary landfill with controlled placement, compaction, "
                    "and daily/final cover material operating under anaerobic conditions.",
        applicable_waste_types=(
            "FOOD_WASTE", "YARD_WASTE", "PAPER", "CARDBOARD", "WOOD",
            "TEXTILES", "RUBBER", "LEATHER", "MSW", "NAPPIES", "SLUDGE",
            "INDUSTRIAL_WASTE",
        ),
        primary_gases=("CH4", "CO2"),
        regulatory_refs=("IPCC 2006 GL Vol.5 Ch.3", "40 CFR 98 Subpart HH"),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 3",
    ),
    "MANAGED_SEMI_AEROBIC": TreatmentMethodInfo(
        code="MANAGED_SEMI_AEROBIC",
        name="Managed Semi-Aerobic Landfill (Fukuoka Method)",
        description="Sanitary landfill with leachate collection pipes that allow "
                    "passive air intrusion, reducing methane generation.",
        applicable_waste_types=(
            "FOOD_WASTE", "YARD_WASTE", "PAPER", "CARDBOARD", "WOOD",
            "TEXTILES", "MSW", "INDUSTRIAL_WASTE",
        ),
        primary_gases=("CH4", "CO2"),
        regulatory_refs=("IPCC 2006 GL Vol.5 Ch.3",),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 3",
    ),
    "UNMANAGED_DEEP": TreatmentMethodInfo(
        code="UNMANAGED_DEEP",
        name="Unmanaged Disposal Site (Deep, >= 5m)",
        description="Unmanaged/uncontrolled disposal site with waste depth >= 5 metres, "
                    "predominantly anaerobic conditions.",
        applicable_waste_types=(
            "FOOD_WASTE", "YARD_WASTE", "PAPER", "CARDBOARD", "WOOD",
            "TEXTILES", "RUBBER", "PLASTIC", "MSW",
        ),
        primary_gases=("CH4", "CO2"),
        regulatory_refs=("IPCC 2006 GL Vol.5 Ch.3",),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 3",
    ),
    "UNMANAGED_SHALLOW": TreatmentMethodInfo(
        code="UNMANAGED_SHALLOW",
        name="Unmanaged Disposal Site (Shallow, < 5m)",
        description="Unmanaged/uncontrolled disposal site with waste depth < 5 metres, "
                    "partially aerobic conditions.",
        applicable_waste_types=(
            "FOOD_WASTE", "YARD_WASTE", "PAPER", "CARDBOARD", "MSW",
        ),
        primary_gases=("CH4", "CO2"),
        regulatory_refs=("IPCC 2006 GL Vol.5 Ch.3",),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 3",
    ),
    "INCINERATION": TreatmentMethodInfo(
        code="INCINERATION",
        name="Waste Incineration",
        description="Controlled combustion of solid waste in purpose-built "
                    "incinerators with or without energy recovery.",
        applicable_waste_types=(
            "FOOD_WASTE", "PAPER", "CARDBOARD", "WOOD", "TEXTILES",
            "TEXTILES_SYNTHETIC", "TEXTILES_NATURAL", "RUBBER", "LEATHER",
            "PLASTIC", "MSW", "NAPPIES", "SLUDGE", "CLINICAL_WASTE",
            "INDUSTRIAL_WASTE",
        ),
        primary_gases=("CO2", "CH4", "N2O"),
        regulatory_refs=(
            "IPCC 2006 GL Vol.5 Ch.5", "40 CFR 98 Subpart C",
            "EU Directive 2000/76/EC",
        ),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 5",
    ),
    "OPEN_BURNING": TreatmentMethodInfo(
        code="OPEN_BURNING",
        name="Open Burning of Waste",
        description="Uncontrolled combustion of waste in the open air, "
                    "including backyard burning and open dump fires.",
        applicable_waste_types=(
            "FOOD_WASTE", "YARD_WASTE", "PAPER", "WOOD", "TEXTILES",
            "RUBBER", "PLASTIC", "MSW", "CLINICAL_WASTE",
        ),
        primary_gases=("CO2", "CH4", "N2O", "CO"),
        regulatory_refs=("IPCC 2006 GL Vol.5 Ch.5 Section 5.3",),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 5",
    ),
    "COMPOSTING": TreatmentMethodInfo(
        code="COMPOSTING",
        name="Aerobic Composting",
        description="Biological decomposition of organic waste under controlled "
                    "aerobic conditions to produce compost.",
        applicable_waste_types=(
            "FOOD_WASTE", "YARD_WASTE", "PAPER", "CARDBOARD", "WOOD",
            "SLUDGE",
        ),
        primary_gases=("CH4", "N2O"),
        regulatory_refs=(
            "IPCC 2006 GL Vol.5 Ch.4", "IPCC 2019 Refinement Ch.5",
        ),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 4",
    ),
    "ANAEROBIC_DIGESTION": TreatmentMethodInfo(
        code="ANAEROBIC_DIGESTION",
        name="Anaerobic Digestion",
        description="Biological decomposition of organic waste in an enclosed "
                    "reactor under anaerobic conditions producing biogas.",
        applicable_waste_types=(
            "FOOD_WASTE", "YARD_WASTE", "SLUDGE", "MSW",
            "INDUSTRIAL_WASTE",
        ),
        primary_gases=("CH4",),
        regulatory_refs=(
            "IPCC 2006 GL Vol.5 Ch.4", "IPCC 2019 Refinement Ch.5",
        ),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 4",
    ),
    "MBT": TreatmentMethodInfo(
        code="MBT",
        name="Mechanical-Biological Treatment",
        description="Combined mechanical sorting and biological treatment "
                    "(aerobic stabilisation or anaerobic digestion) of residual waste.",
        applicable_waste_types=(
            "FOOD_WASTE", "PAPER", "CARDBOARD", "MSW",
            "INDUSTRIAL_WASTE",
        ),
        primary_gases=("CH4", "N2O"),
        regulatory_refs=(
            "IPCC 2019 Refinement Ch.5", "EU Landfill Directive 1999/31/EC",
        ),
        ipcc_chapter="IPCC 2019 Refinement Ch 5",
    ),
    "PYROLYSIS": TreatmentMethodInfo(
        code="PYROLYSIS",
        name="Pyrolysis",
        description="Thermal decomposition of waste in the absence of oxygen "
                    "producing syngas, bio-oil, and biochar.",
        applicable_waste_types=(
            "WOOD", "PAPER", "PLASTIC", "RUBBER", "TEXTILES", "MSW",
            "INDUSTRIAL_WASTE",
        ),
        primary_gases=("CO2", "CH4"),
        regulatory_refs=("IPCC 2006 GL Vol.5 Ch.5",),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 5",
    ),
    "GASIFICATION": TreatmentMethodInfo(
        code="GASIFICATION",
        name="Gasification",
        description="Partial oxidation of waste at elevated temperatures "
                    "producing synthesis gas (CO + H2).",
        applicable_waste_types=(
            "WOOD", "PAPER", "PLASTIC", "RUBBER", "MSW",
            "INDUSTRIAL_WASTE",
        ),
        primary_gases=("CO2", "CH4", "CO"),
        regulatory_refs=("IPCC 2006 GL Vol.5 Ch.5",),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 5",
    ),
    "WASTEWATER_TREATMENT": TreatmentMethodInfo(
        code="WASTEWATER_TREATMENT",
        name="On-site Wastewater Treatment",
        description="Treatment of wastewater from on-site industrial or "
                    "domestic processes generating CH4 and N2O.",
        applicable_waste_types=("SLUDGE",),
        primary_gases=("CH4", "N2O"),
        regulatory_refs=(
            "IPCC 2006 GL Vol.5 Ch.6", "40 CFR 98 Subpart II",
        ),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 6",
    ),
    "LANDFILL_GAS_RECOVERY": TreatmentMethodInfo(
        code="LANDFILL_GAS_RECOVERY",
        name="Landfill Gas Recovery and Utilisation",
        description="Capture and destruction or energy recovery of landfill gas "
                    "via flaring, electricity generation, or direct use.",
        applicable_waste_types=(
            "FOOD_WASTE", "YARD_WASTE", "PAPER", "CARDBOARD", "WOOD",
            "TEXTILES", "MSW",
        ),
        primary_gases=("CH4",),
        regulatory_refs=(
            "IPCC 2006 GL Vol.5 Ch.3", "40 CFR 60 Subpart WWW/XXX",
        ),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 3",
    ),
    "RECYCLING": TreatmentMethodInfo(
        code="RECYCLING",
        name="Material Recycling",
        description="Recovery and reprocessing of waste materials, accounted "
                    "as avoided landfill/incineration emissions.",
        applicable_waste_types=(
            "PAPER", "CARDBOARD", "PLASTIC", "METAL", "GLASS", "WOOD",
            "TEXTILES", "RUBBER",
        ),
        primary_gases=("CO2",),
        regulatory_refs=("GHG Protocol Scope 3 Category 5",),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 5",
    ),
    "AUTOCLAVE": TreatmentMethodInfo(
        code="AUTOCLAVE",
        name="Autoclave / Steam Sterilisation",
        description="High-pressure steam treatment of waste (primarily clinical "
                    "waste) for sterilisation before landfill or incineration.",
        applicable_waste_types=(
            "CLINICAL_WASTE", "SLUDGE",
        ),
        primary_gases=("CO2",),
        regulatory_refs=("WHO Healthcare Waste Management Guidelines",),
        ipcc_chapter="IPCC 2006 Vol 5 Ch 5",
    ),
}


# ===========================================================================
# Wastewater N2O Emission Factors
# IPCC 2006 Vol 5 Ch 6 Section 6.3
# ===========================================================================

#: N2O emission factor for wastewater (kg N2O-N per kg N in effluent).
#: Source: IPCC 2006 Vol 5 Ch 6, Equation 6.7.
WASTEWATER_N2O_EF_PLANT: Decimal = _D("0.005")

#: N2O emission factor for effluent discharge to water bodies.
#: kg N2O-N per kg N discharged.
WASTEWATER_N2O_EF_EFFLUENT: Decimal = _D("0.005")

#: N2O-N to N2O conversion factor (44/28).
N2O_N_RATIO: Decimal = _D("1.571429")

#: Fraction of industrial BOD in sewage.
INDUSTRIAL_BOD_FRACTION_DEFAULT: Decimal = _D("0.25")

#: Default per-capita BOD generation (g BOD/person/day).
PER_CAPITA_BOD_DEFAULT: Decimal = _D("60")

#: Default per-capita protein consumption (kg protein/person/year).
PER_CAPITA_PROTEIN_DEFAULT: Decimal = _D("25.55")

#: Fraction of nitrogen in protein.
NITROGEN_FRACTION_PROTEIN: Decimal = _D("0.16")

#: Non-consumed protein factor (added to sewage).
NON_CONSUMED_PROTEIN_FACTOR: Decimal = _D("1.0")

#: Factor for industrial/commercial co-discharged protein.
INDUSTRIAL_PROTEIN_FACTOR: Decimal = _D("1.25")


# ===========================================================================
# DEFRA Emission Factors (UK-specific, commonly used)
# Units: kg CO2e per tonne of waste
# Source: DEFRA 2025 Conversion Factors
# ===========================================================================

#: DEFRA emission factors by (waste_category, treatment_method).
#: Units: kg CO2e per tonne of waste.
DEFRA_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "FOOD_WASTE": {
        "LANDFILL": _D("587"),
        "INCINERATION": _D("21"),
        "COMPOSTING": _D("10"),
        "ANAEROBIC_DIGESTION": _D("10"),
    },
    "PAPER": {
        "LANDFILL": _D("1042"),
        "INCINERATION": _D("21"),
        "RECYCLING": _D("-519"),
    },
    "CARDBOARD": {
        "LANDFILL": _D("730"),
        "INCINERATION": _D("21"),
        "RECYCLING": _D("-519"),
    },
    "PLASTIC": {
        "LANDFILL": _D("21"),
        "INCINERATION": _D("2035"),
        "RECYCLING": _D("-1442"),
    },
    "WOOD": {
        "LANDFILL": _D("853"),
        "INCINERATION": _D("21"),
        "RECYCLING": _D("-427"),
    },
    "TEXTILES": {
        "LANDFILL": _D("467"),
        "INCINERATION": _D("21"),
        "RECYCLING": _D("-3845"),
    },
    "GLASS": {
        "LANDFILL": _D("21"),
        "INCINERATION": _D("21"),
        "RECYCLING": _D("-315"),
    },
    "METAL": {
        "LANDFILL": _D("21"),
        "INCINERATION": _D("21"),
        "RECYCLING": _D("-4297"),
    },
    "MSW": {
        "LANDFILL": _D("587"),
        "INCINERATION": _D("445"),
        "RECYCLING": _D("-519"),
    },
    "CLINICAL_WASTE": {
        "INCINERATION": _D("900"),
        "AUTOCLAVE": _D("50"),
    },
}


# ===========================================================================
# EPA Emission Factors (US-specific)
# Units: MTCO2E per short ton
# Source: EPA WARM Model v16
# ===========================================================================

#: EPA WARM model factors by (waste_category, treatment_method).
#: Units: metric tonnes CO2e per short ton (US).
EPA_WARM_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "FOOD_WASTE": {
        "LANDFILL": _D("0.56"),
        "INCINERATION": _D("0.08"),
        "COMPOSTING": _D("-0.05"),
        "ANAEROBIC_DIGESTION": _D("-0.04"),
    },
    "PAPER": {
        "LANDFILL": _D("1.22"),
        "INCINERATION": _D("0.08"),
        "RECYCLING": _D("-3.55"),
    },
    "PLASTIC_HDPE": {
        "LANDFILL": _D("0.04"),
        "INCINERATION": _D("1.24"),
        "RECYCLING": _D("-1.31"),
    },
    "PLASTIC_PET": {
        "LANDFILL": _D("0.04"),
        "INCINERATION": _D("1.24"),
        "RECYCLING": _D("-1.62"),
    },
    "YARD_WASTE": {
        "LANDFILL": _D("0.24"),
        "INCINERATION": _D("0.08"),
        "COMPOSTING": _D("-0.05"),
    },
    "WOOD": {
        "LANDFILL": _D("0.68"),
        "INCINERATION": _D("0.08"),
        "RECYCLING": _D("-2.52"),
    },
    "MSW": {
        "LANDFILL": _D("0.52"),
        "INCINERATION": _D("0.43"),
    },
}


# ===========================================================================
# Landfill Gas Collection Efficiency Defaults
# ===========================================================================

#: Default landfill gas collection efficiency by system type.
#: Source: EPA AP-42 / IPCC 2006 Vol 5 Ch 3.
LFG_COLLECTION_EFFICIENCY: Dict[str, Decimal] = {
    "ACTIVE_WELLS_WITH_CAP": _D("0.75"),
    "ACTIVE_WELLS_NO_CAP": _D("0.50"),
    "PASSIVE_VENTS": _D("0.20"),
    "NO_COLLECTION": _D("0"),
    "BEST_PRACTICE": _D("0.85"),
}

#: Default flare destruction efficiency.
FLARE_DESTRUCTION_EFFICIENCY: Decimal = _D("0.98")

#: Default engine destruction efficiency (landfill gas to energy).
ENGINE_DESTRUCTION_EFFICIENCY: Decimal = _D("0.995")


# ===========================================================================
# WasteTreatmentDatabaseEngine
# ===========================================================================


class WasteTreatmentDatabaseEngine:
    """Reference data repository for IPCC/EPA/DEFRA waste treatment emission factors.

    This engine provides deterministic lookups for all emission factors,
    DOC values, MCF values, carbon content, NCV, half-life data, Bo values,
    and GWP values needed by the WasteTreatmentCalculatorEngine (Engine 2).
    All data is hard-coded from published IPCC, EPA, and DEFRA tables.

    Thread Safety:
        Immutable reference data requires no locking.  The custom factor
        registry uses a reentrant lock for thread-safe mutations.

    Attributes:
        _custom_factors: User-provided custom emission factors.
        _lock: Reentrant lock protecting mutable state.
        _total_lookups: Counter of total lookup operations.
        _cache: In-memory cache for repeated lookups.

    Example:
        >>> db = WasteTreatmentDatabaseEngine()
        >>> doc = db.get_doc_value("food_waste")
        >>> assert doc == Decimal("0.15")
    """

    def __init__(self) -> None:
        """Initialize the WasteTreatmentDatabaseEngine with empty custom factor registry."""
        self._custom_factors: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._total_lookups: int = 0
        self._cache: Dict[str, Any] = {}
        self._created_at = _utcnow()

        logger.info(
            "WasteTreatmentDatabaseEngine initialized: "
            "waste_categories=%d, treatment_methods=%d, "
            "incinerator_types=%d, composting_types=%d, "
            "landfill_types=%d, wastewater_system_types=%d",
            len(WasteCategory),
            len(TreatmentMethod),
            len(IncineratorType),
            len(CompostingType),
            len(LandfillType),
            len(WastewaterSystemType),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_lookups(self) -> None:
        """Thread-safe increment of the lookup counter."""
        with self._lock:
            self._total_lookups += 1

    def _validate_waste_category(self, category: str) -> str:
        """Validate and normalise a waste category string.

        Args:
            category: Waste category name or enum value.

        Returns:
            Normalised category string (uppercase).

        Raises:
            ValueError: If the category is not recognized.
        """
        normalised = category.upper().replace(" ", "_")
        valid = {e.value for e in WasteCategory}
        if normalised not in valid:
            raise ValueError(
                f"Unknown waste category '{category}'. "
                f"Valid categories: {sorted(valid)}"
            )
        return normalised

    def _validate_landfill_type(self, landfill_type: str) -> str:
        """Validate and normalise a landfill type string.

        Args:
            landfill_type: Landfill type name or enum value.

        Returns:
            Normalised landfill type string.

        Raises:
            ValueError: If the landfill type is not recognized.
        """
        normalised = landfill_type.upper().replace(" ", "_")
        valid = {e.value for e in LandfillType}
        if normalised not in valid:
            raise ValueError(
                f"Unknown landfill type '{landfill_type}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    def _validate_incinerator_type(self, incinerator_type: str) -> str:
        """Validate and normalise an incinerator type string.

        Args:
            incinerator_type: Incinerator technology type.

        Returns:
            Normalised incinerator type string.

        Raises:
            ValueError: If the incinerator type is not recognized.
        """
        normalised = incinerator_type.upper().replace(" ", "_")
        valid = {e.value for e in IncineratorType}
        if normalised not in valid:
            raise ValueError(
                f"Unknown incinerator type '{incinerator_type}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    def _validate_composting_type(self, composting_type: str) -> str:
        """Validate and normalise a composting/biological treatment type string.

        Args:
            composting_type: Composting or biological treatment type.

        Returns:
            Normalised composting type string.

        Raises:
            ValueError: If the composting type is not recognized.
        """
        normalised = composting_type.upper().replace(" ", "_")
        valid = {e.value for e in CompostingType}
        if normalised not in valid:
            raise ValueError(
                f"Unknown composting type '{composting_type}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    def _validate_wastewater_system(self, system_type: str) -> str:
        """Validate and normalise a wastewater treatment system type string.

        Args:
            system_type: Wastewater treatment system type.

        Returns:
            Normalised system type string.

        Raises:
            ValueError: If the system type is not recognized.
        """
        normalised = system_type.upper().replace(" ", "_")
        valid = {e.value for e in WastewaterSystemType}
        if normalised not in valid:
            raise ValueError(
                f"Unknown wastewater system type '{system_type}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    def _validate_climate_zone(self, climate_zone: str) -> str:
        """Validate and normalise a waste-sector climate zone string.

        Args:
            climate_zone: Climate zone for half-life selection.

        Returns:
            Normalised climate zone string.

        Raises:
            ValueError: If the climate zone is not recognized.
        """
        normalised = climate_zone.upper().replace(" ", "_")
        valid = {e.value for e in ClimateZoneWaste}
        if normalised not in valid:
            raise ValueError(
                f"Unknown climate zone '{climate_zone}'. "
                f"Valid zones: {sorted(valid)}"
            )
        return normalised

    def _validate_wastewater_type(self, wastewater_type: str) -> str:
        """Validate and normalise a wastewater type for Bo lookup.

        Args:
            wastewater_type: Wastewater category for Bo selection.

        Returns:
            Normalised wastewater type string.

        Raises:
            ValueError: If the wastewater type is not recognized.
        """
        normalised = wastewater_type.upper().replace(" ", "_")
        valid = {e.value for e in WastewaterType}
        if normalised not in valid:
            raise ValueError(
                f"Unknown wastewater type '{wastewater_type}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    # ------------------------------------------------------------------
    # DOC Values
    # ------------------------------------------------------------------

    def get_doc_value(self, waste_category: str) -> Decimal:
        """Look up the Degradable Organic Carbon fraction for a waste category.

        DOC is expressed as a fraction of wet weight, per IPCC 2006 Vol 5
        Table 2.4.

        Args:
            waste_category: Waste category (e.g. "food_waste", "PAPER").

        Returns:
            DOC value as Decimal (fraction of wet weight).

        Raises:
            ValueError: If the waste category is not recognized.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_doc_value("food_waste")
            Decimal('0.15')
        """
        self._increment_lookups()
        cat = self._validate_waste_category(waste_category)

        cache_key = f"doc:{cat}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = DOC_VALUES[cat]
        self._cache[cache_key] = result

        logger.debug("DOC lookup: category=%s, doc=%s", cat, result)
        return result

    def get_docf_value(self, waste_category: str) -> Decimal:
        """Look up the fraction of DOC that decomposes (DOCf) for a waste category.

        DOCf is the fraction of DOC that can actually decompose under
        anaerobic landfill conditions.

        Args:
            waste_category: Waste category string.

        Returns:
            DOCf value as Decimal.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_docf_value("food_waste")
            Decimal('0.70')
        """
        self._increment_lookups()
        cat = self._validate_waste_category(waste_category)

        if cat in DOCF_VALUES:
            return DOCF_VALUES[cat]
        return DOCF_DEFAULT

    # ------------------------------------------------------------------
    # MCF Values (Landfill)
    # ------------------------------------------------------------------

    def get_mcf_value(self, landfill_type: str) -> Decimal:
        """Look up the Methane Correction Factor for a landfill/disposal site type.

        MCF reflects the methane-generating potential of a site based
        on its management conditions, per IPCC 2006 Vol 5 Table 3.1.

        Args:
            landfill_type: Landfill or disposal site type
                (e.g. "managed_anaerobic", "UNMANAGED_DEEP").

        Returns:
            MCF value as Decimal (0 to 1).

        Raises:
            ValueError: If the landfill type is not recognized.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_mcf_value("managed_anaerobic")
            Decimal('1.0')
        """
        self._increment_lookups()
        lt = self._validate_landfill_type(landfill_type)

        cache_key = f"mcf:{lt}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = MCF_VALUES[lt]
        self._cache[cache_key] = result

        logger.debug("MCF lookup: landfill_type=%s, mcf=%s", lt, result)
        return result

    def get_oxidation_factor(self, landfill_type: str) -> Decimal:
        """Look up the oxidation factor (OX) for a landfill type.

        OX is the fraction of generated CH4 that is oxidised in the
        landfill cover soil before escaping to the atmosphere.

        Args:
            landfill_type: Landfill type string.

        Returns:
            Oxidation factor as Decimal (0 to 0.1 typically).

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_oxidation_factor("managed_anaerobic")
            Decimal('0.10')
        """
        self._increment_lookups()
        lt = self._validate_landfill_type(landfill_type)
        return OXIDATION_FACTOR.get(lt, _D("0"))

    # ------------------------------------------------------------------
    # Carbon Content
    # ------------------------------------------------------------------

    def get_carbon_content(self, waste_category: str) -> Dict[str, Any]:
        """Look up carbon content and fossil carbon fraction for a waste type.

        Returns total carbon content as fraction of wet weight and the
        fraction of that carbon which is fossil-derived (relevant for
        incineration CO2 reporting).

        Args:
            waste_category: Waste category string.

        Returns:
            Dictionary with keys: carbon_pct, fossil_fraction, source.

        Raises:
            ValueError: If the waste category is not recognized.
            KeyError: If no carbon content data exists for the category.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> cc = db.get_carbon_content("plastic")
            >>> cc["fossil_fraction"]
            Decimal('1.0')
        """
        self._increment_lookups()
        cat = self._validate_waste_category(waste_category)

        cache_key = f"cc:{cat}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if cat not in CARBON_CONTENT:
            raise KeyError(
                f"No carbon content data for category '{waste_category}'. "
                f"Available: {sorted(CARBON_CONTENT.keys())}"
            )

        record = CARBON_CONTENT[cat]
        result = {
            "carbon_pct": record.carbon_pct,
            "fossil_fraction": record.fossil_fraction,
            "source": record.source,
        }
        self._cache[cache_key] = result

        logger.debug(
            "Carbon content lookup: category=%s, carbon_pct=%s, fossil_fraction=%s",
            cat, record.carbon_pct, record.fossil_fraction,
        )
        return result

    # ------------------------------------------------------------------
    # Composting / Biological Treatment EFs
    # ------------------------------------------------------------------

    def get_composting_ef(self, composting_type: str) -> Dict[str, Any]:
        """Look up emission factors for composting or biological treatment.

        Returns CH4 and N2O emission factors in g per kg of waste treated,
        per IPCC 2019 Refinement Table 5.1.

        Args:
            composting_type: Biological treatment type
                (e.g. "well_managed", "AD_FLARED", "MBT_AEROBIC").

        Returns:
            Dictionary with keys: ch4_g_per_kg, n2o_g_per_kg, source.

        Raises:
            ValueError: If the composting type is not recognized.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> ef = db.get_composting_ef("well_managed")
            >>> ef["ch4_g_per_kg"]
            Decimal('4.0')
        """
        self._increment_lookups()
        ct = self._validate_composting_type(composting_type)

        cache_key = f"comp:{ct}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        record = COMPOSTING_EF[ct]
        result = {
            "ch4_g_per_kg": record.ch4_g_per_kg,
            "n2o_g_per_kg": record.n2o_g_per_kg,
            "source": record.source,
        }
        self._cache[cache_key] = result

        logger.debug(
            "Composting EF lookup: type=%s, ch4=%s g/kg, n2o=%s g/kg",
            ct, record.ch4_g_per_kg, record.n2o_g_per_kg,
        )
        return result

    # ------------------------------------------------------------------
    # Incineration EFs
    # ------------------------------------------------------------------

    def get_incineration_ef(self, incinerator_type: str) -> Dict[str, Any]:
        """Look up emission factors for a specific incinerator technology.

        Returns N2O and CH4 emission factors in kg per Gg (1000 tonnes)
        of waste incinerated, per IPCC 2006 Vol 5 Ch 5 Table 5.3.

        Args:
            incinerator_type: Incinerator technology
                (e.g. "stoker_grate", "FLUIDIZED_BED").

        Returns:
            Dictionary with keys: n2o_kg_per_gg, ch4_kg_per_gg, source.

        Raises:
            ValueError: If the incinerator type is not recognized.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> ef = db.get_incineration_ef("stoker_grate")
            >>> ef["n2o_kg_per_gg"]
            Decimal('50')
        """
        self._increment_lookups()
        it = self._validate_incinerator_type(incinerator_type)

        cache_key = f"incin:{it}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        record = INCINERATION_EF[it]
        result = {
            "n2o_kg_per_gg": record.n2o_kg_per_gg,
            "ch4_kg_per_gg": record.ch4_kg_per_gg,
            "source": record.source,
        }
        self._cache[cache_key] = result

        logger.debug(
            "Incineration EF lookup: type=%s, n2o=%s kg/Gg, ch4=%s kg/Gg",
            it, record.n2o_kg_per_gg, record.ch4_kg_per_gg,
        )
        return result

    # ------------------------------------------------------------------
    # Open Burning EFs
    # ------------------------------------------------------------------

    def get_open_burning_ef(self, waste_category: str) -> Dict[str, Any]:
        """Look up open burning emission factors for a waste category.

        Returns CO2, CH4, and N2O emission factors in g per kg of waste
        burned, per IPCC 2006 Vol 5 Ch 5 / EPA AP-42.

        Args:
            waste_category: Waste category string.

        Returns:
            Dictionary with keys: co2_g_per_kg, ch4_g_per_kg,
            n2o_g_per_kg, source.

        Raises:
            KeyError: If no open burning EF exists for the category.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> ef = db.get_open_burning_ef("MSW")
            >>> ef["ch4_g_per_kg"]
            Decimal('6.5')
        """
        self._increment_lookups()
        cat = waste_category.upper().replace(" ", "_")

        if cat not in OPEN_BURNING_EF:
            raise KeyError(
                f"No open burning EF for category '{waste_category}'. "
                f"Available: {sorted(OPEN_BURNING_EF.keys())}"
            )

        record = OPEN_BURNING_EF[cat]
        result = {
            "co2_g_per_kg": record.co2_g_per_kg,
            "ch4_g_per_kg": record.ch4_g_per_kg,
            "n2o_g_per_kg": record.n2o_g_per_kg,
            "source": record.source,
        }

        logger.debug(
            "Open burning EF lookup: category=%s, co2=%s, ch4=%s, n2o=%s g/kg",
            cat, record.co2_g_per_kg, record.ch4_g_per_kg, record.n2o_g_per_kg,
        )
        return result

    # ------------------------------------------------------------------
    # Wastewater MCF
    # ------------------------------------------------------------------

    def get_wastewater_mcf(self, system_type: str) -> Decimal:
        """Look up the Methane Correction Factor for a wastewater treatment system.

        Per IPCC 2006 Vol 5 Ch 6 Table 6.3.

        Args:
            system_type: Wastewater treatment system type
                (e.g. "aerobic_well_managed", "SEPTIC_SYSTEM").

        Returns:
            MCF value as Decimal (0 to 1).

        Raises:
            ValueError: If the system type is not recognized.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_wastewater_mcf("septic_system")
            Decimal('0.5')
        """
        self._increment_lookups()
        st = self._validate_wastewater_system(system_type)

        cache_key = f"ww_mcf:{st}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = WASTEWATER_MCF[st]
        self._cache[cache_key] = result

        logger.debug("Wastewater MCF lookup: system=%s, mcf=%s", st, result)
        return result

    # ------------------------------------------------------------------
    # Bo Values (Max CH4 Producing Capacity)
    # ------------------------------------------------------------------

    def get_bo_value(
        self,
        wastewater_type: str,
        basis: str = "COD",
    ) -> Decimal:
        """Look up the maximum CH4 producing capacity (Bo) for a wastewater type.

        Bo is used in IPCC wastewater CH4 calculations:
        CH4 = Bo * S * MCF

        Args:
            wastewater_type: Wastewater category
                (e.g. "domestic", "INDUSTRIAL_DAIRY").
            basis: "COD" or "BOD" to select the appropriate Bo table.

        Returns:
            Bo value as Decimal (kg CH4 per kg COD or BOD).

        Raises:
            ValueError: If the wastewater type or basis is not recognized.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_bo_value("domestic", "COD")
            Decimal('0.25')
        """
        self._increment_lookups()
        wt = self._validate_wastewater_type(wastewater_type)
        basis_upper = basis.upper()

        if basis_upper == "COD":
            table = BO_VALUES
        elif basis_upper == "BOD":
            table = BO_VALUES_BOD
        else:
            raise ValueError(
                f"Unknown Bo basis '{basis}'. Must be 'COD' or 'BOD'."
            )

        result = table[wt]
        logger.debug(
            "Bo lookup: type=%s, basis=%s, bo=%s", wt, basis_upper, result,
        )
        return result

    # ------------------------------------------------------------------
    # NCV (Net Calorific Value)
    # ------------------------------------------------------------------

    def get_ncv(self, waste_category: str) -> Decimal:
        """Look up the Net Calorific Value for a waste category.

        NCV is used for incineration energy recovery calculations
        and CO2 emission estimation.

        Args:
            waste_category: Waste category string.

        Returns:
            NCV in GJ per tonne of wet waste as Decimal.

        Raises:
            ValueError: If the waste category is not recognized.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_ncv("plastic")
            Decimal('32.0')
        """
        self._increment_lookups()
        cat = self._validate_waste_category(waste_category)

        cache_key = f"ncv:{cat}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = NCV_VALUES.get(cat, _D("0"))
        self._cache[cache_key] = result

        logger.debug("NCV lookup: category=%s, ncv=%s GJ/t", cat, result)
        return result

    # ------------------------------------------------------------------
    # Half-Life Values
    # ------------------------------------------------------------------

    def get_half_life(
        self,
        climate_zone: str,
        waste_category: str,
    ) -> Decimal:
        """Look up the first-order decay half-life for a waste type in a climate zone.

        Half-life is used in the IPCC First-Order Decay (FOD) model
        for estimating methane generation from landfilled waste.

        Args:
            climate_zone: Climate zone for waste decay
                (e.g. "tropical_wet", "TEMPERATE_DRY").
            waste_category: Waste category string.

        Returns:
            Half-life in years as Decimal.  Returns 0 for non-degrading waste.

        Raises:
            ValueError: If the climate zone or waste category is not recognized.
            KeyError: If the specific combination is not found.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_half_life("tropical_wet", "food_waste")
            Decimal('3')
        """
        self._increment_lookups()
        zone = self._validate_climate_zone(climate_zone)
        cat = self._validate_waste_category(waste_category)

        cache_key = f"hl:{zone}:{cat}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Map waste category to decay category
        if cat not in WASTE_TO_DECAY_CATEGORY:
            raise KeyError(
                f"No decay category mapping for waste '{waste_category}'."
            )
        decay_cat = WASTE_TO_DECAY_CATEGORY[cat]

        if zone not in HALF_LIFE_VALUES:
            raise KeyError(f"No half-life data for zone '{climate_zone}'.")
        if decay_cat not in HALF_LIFE_VALUES[zone]:
            raise KeyError(
                f"No half-life for ({climate_zone}, {decay_cat})."
            )

        result = HALF_LIFE_VALUES[zone][decay_cat]
        self._cache[cache_key] = result

        logger.debug(
            "Half-life lookup: zone=%s, category=%s, decay_cat=%s, t_half=%s yr",
            zone, cat, decay_cat, result,
        )
        return result

    def get_decay_rate(
        self,
        climate_zone: str,
        waste_category: str,
    ) -> Decimal:
        """Calculate the first-order decay rate constant k from half-life.

        k = ln(2) / t_half

        Args:
            climate_zone: Climate zone string.
            waste_category: Waste category string.

        Returns:
            Decay rate constant k (yr^-1) as Decimal.
            Returns 0 for non-degrading waste.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> k = db.get_decay_rate("tropical_wet", "food_waste")
        """
        self._increment_lookups()
        t_half = self.get_half_life(climate_zone, waste_category)

        if t_half == _D("0"):
            return _D("0")

        # ln(2) = 0.693147...
        ln2 = _D("0.69314718")
        k = (ln2 / t_half).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "Decay rate: zone=%s, category=%s, t_half=%s, k=%s yr-1",
            climate_zone, waste_category, t_half, k,
        )
        return k

    # ------------------------------------------------------------------
    # GWP Values
    # ------------------------------------------------------------------

    def get_gwp(self, gas: str, gwp_source: str = "AR6") -> Decimal:
        """Look up the Global Warming Potential for a gas and assessment report.

        Supports separate GWP values for fossil and biogenic CH4
        per AR5/AR6 methodology.

        Args:
            gas: Gas name (CO2, CH4, CH4_FOSSIL, CH4_BIOGENIC, N2O, CO).
            gwp_source: IPCC assessment report (AR4, AR5, AR6, AR6_20YR).

        Returns:
            GWP value as Decimal.

        Raises:
            KeyError: If gas or GWP source is not recognized.

        Example:
            >>> db = WasteTreatmentDatabaseEngine()
            >>> db.get_gwp("CH4", "AR6")
            Decimal('29.8')
        """
        self._increment_lookups()
        source = gwp_source.upper()
        g = gas.upper()

        if source not in GWP_VALUES:
            raise KeyError(
                f"Unknown GWP source '{gwp_source}'. "
                f"Valid: {sorted(GWP_VALUES.keys())}"
            )
        if g not in GWP_VALUES[source]:
            raise KeyError(
                f"Unknown gas '{gas}' for source '{source}'. "
                f"Valid: {sorted(GWP_VALUES[source].keys())}"
            )

        return GWP_VALUES[source][g]

    # ------------------------------------------------------------------
    # Treatment Method Info
    # ------------------------------------------------------------------

    def get_treatment_method_info(
        self,
        treatment_method: str,
    ) -> Dict[str, Any]:
        """Look up metadata for a waste treatment method.

        Args:
            treatment_method: Treatment method code
                (e.g. "INCINERATION", "composting").

        Returns:
            Dictionary with method metadata including name, description,
            applicable waste types, primary gases, and regulatory refs.

        Raises:
            KeyError: If the treatment method is not recognized.
        """
        self._increment_lookups()
        key = treatment_method.upper().replace(" ", "_")

        if key not in TREATMENT_METHODS:
            raise KeyError(
                f"Unknown treatment method '{treatment_method}'. "
                f"Valid: {sorted(TREATMENT_METHODS.keys())}"
            )

        info = TREATMENT_METHODS[key]
        return {
            "code": info.code,
            "name": info.name,
            "description": info.description,
            "applicable_waste_types": list(info.applicable_waste_types),
            "primary_gases": list(info.primary_gases),
            "regulatory_refs": list(info.regulatory_refs),
            "ipcc_chapter": info.ipcc_chapter,
        }

    # ------------------------------------------------------------------
    # DEFRA / EPA Factor Lookups
    # ------------------------------------------------------------------

    def get_defra_factor(
        self,
        waste_category: str,
        treatment_method: str,
    ) -> Decimal:
        """Look up a DEFRA emission factor (kg CO2e/tonne waste).

        Args:
            waste_category: Waste category string.
            treatment_method: Treatment method string (LANDFILL, INCINERATION, etc.).

        Returns:
            DEFRA emission factor as Decimal (kg CO2e per tonne).

        Raises:
            KeyError: If the combination is not found.
        """
        self._increment_lookups()
        cat = waste_category.upper().replace(" ", "_")
        method = treatment_method.upper().replace(" ", "_")

        if cat not in DEFRA_FACTORS:
            raise KeyError(
                f"No DEFRA factor for category '{waste_category}'. "
                f"Available: {sorted(DEFRA_FACTORS.keys())}"
            )
        if method not in DEFRA_FACTORS[cat]:
            raise KeyError(
                f"No DEFRA factor for ({waste_category}, {treatment_method}). "
                f"Available methods for {cat}: {sorted(DEFRA_FACTORS[cat].keys())}"
            )

        result = DEFRA_FACTORS[cat][method]
        logger.debug(
            "DEFRA factor lookup: category=%s, method=%s, ef=%s kg CO2e/t",
            cat, method, result,
        )
        return result

    def get_epa_warm_factor(
        self,
        waste_category: str,
        treatment_method: str,
    ) -> Decimal:
        """Look up an EPA WARM model factor (MTCO2E/short ton).

        Args:
            waste_category: Waste category string.
            treatment_method: Treatment method string.

        Returns:
            EPA WARM factor as Decimal (metric tonnes CO2e per short ton).

        Raises:
            KeyError: If the combination is not found.
        """
        self._increment_lookups()
        cat = waste_category.upper().replace(" ", "_")
        method = treatment_method.upper().replace(" ", "_")

        if cat not in EPA_WARM_FACTORS:
            raise KeyError(
                f"No EPA WARM factor for category '{waste_category}'. "
                f"Available: {sorted(EPA_WARM_FACTORS.keys())}"
            )
        if method not in EPA_WARM_FACTORS[cat]:
            raise KeyError(
                f"No EPA WARM factor for ({waste_category}, {treatment_method}). "
                f"Available: {sorted(EPA_WARM_FACTORS[cat].keys())}"
            )

        result = EPA_WARM_FACTORS[cat][method]
        logger.debug(
            "EPA WARM factor lookup: category=%s, method=%s, ef=%s MTCO2E/st",
            cat, method, result,
        )
        return result

    # ------------------------------------------------------------------
    # Landfill Gas Collection
    # ------------------------------------------------------------------

    def get_lfg_collection_efficiency(self, system_type: str) -> Decimal:
        """Look up landfill gas collection efficiency by system type.

        Args:
            system_type: LFG collection system type
                (e.g. "ACTIVE_WELLS_WITH_CAP", "PASSIVE_VENTS").

        Returns:
            Collection efficiency as Decimal (0 to 1).

        Raises:
            KeyError: If the system type is not recognized.
        """
        self._increment_lookups()
        key = system_type.upper().replace(" ", "_")

        if key not in LFG_COLLECTION_EFFICIENCY:
            raise KeyError(
                f"Unknown LFG collection system '{system_type}'. "
                f"Valid: {sorted(LFG_COLLECTION_EFFICIENCY.keys())}"
            )

        return LFG_COLLECTION_EFFICIENCY[key]

    def get_methane_fraction(self, waste_category: Optional[str] = None) -> Decimal:
        """Get the methane fraction in landfill gas.

        Args:
            waste_category: Optional waste category for Tier 2 lookup.
                If None, returns the IPCC default of 0.50.

        Returns:
            Methane volume fraction as Decimal.
        """
        self._increment_lookups()
        if waste_category is None:
            return METHANE_FRACTION_DEFAULT

        cat = waste_category.upper().replace(" ", "_")
        return METHANE_FRACTION.get(cat, METHANE_FRACTION_DEFAULT)

    # ------------------------------------------------------------------
    # Wastewater N2O Parameters
    # ------------------------------------------------------------------

    def get_wastewater_n2o_ef(self, location: str = "plant") -> Decimal:
        """Get the wastewater N2O emission factor.

        Args:
            location: "plant" for treatment plant EF, "effluent" for
                discharge to waterbody EF.

        Returns:
            N2O emission factor as Decimal (kg N2O-N per kg N).

        Raises:
            ValueError: If location is not 'plant' or 'effluent'.
        """
        self._increment_lookups()
        loc = location.lower()
        if loc == "plant":
            return WASTEWATER_N2O_EF_PLANT
        elif loc == "effluent":
            return WASTEWATER_N2O_EF_EFFLUENT
        else:
            raise ValueError(
                f"Unknown N2O EF location '{location}'. "
                "Must be 'plant' or 'effluent'."
            )

    # ------------------------------------------------------------------
    # Convenience: All Factors for a Waste Category
    # ------------------------------------------------------------------

    def get_all_factors_for_waste(
        self,
        waste_category: str,
    ) -> Dict[str, Any]:
        """Get all available factors for a waste category in a single call.

        Convenience method that aggregates DOC, DOCf, carbon content,
        NCV, decay mapping, and methane fraction for a waste type.

        Args:
            waste_category: Waste category string.

        Returns:
            Dictionary with all available factor values and provenance hash.
        """
        self._increment_lookups()
        start_time = time.monotonic()
        cat = self._validate_waste_category(waste_category)

        doc = DOC_VALUES.get(cat, _D("0"))
        docf = DOCF_VALUES.get(cat, DOCF_DEFAULT)
        ncv = NCV_VALUES.get(cat, _D("0"))
        decay_cat = WASTE_TO_DECAY_CATEGORY.get(cat, "NON_DEGRADING")
        ch4_frac = METHANE_FRACTION.get(cat, METHANE_FRACTION_DEFAULT)

        cc_data: Dict[str, Any] = {}
        if cat in CARBON_CONTENT:
            record = CARBON_CONTENT[cat]
            cc_data = {
                "carbon_pct": str(record.carbon_pct),
                "fossil_fraction": str(record.fossil_fraction),
            }

        result: Dict[str, Any] = {
            "waste_category": cat,
            "doc": str(doc),
            "docf": str(docf),
            "ncv_gj_per_tonne": str(ncv),
            "decay_category": decay_cat,
            "methane_fraction": str(ch4_frac),
            "carbon_content": cc_data if cc_data else None,
            "processing_time_ms": round(
                (time.monotonic() - start_time) * 1000, 3
            ),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "All factors retrieved for %s: doc=%s, ncv=%s, time=%.3fms",
            cat, doc, ncv, result["processing_time_ms"],
        )
        return result

    # ------------------------------------------------------------------
    # List Available Factors
    # ------------------------------------------------------------------

    def list_available_factors(
        self,
        treatment_method: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all available emission factors, optionally filtered by treatment method.

        Args:
            treatment_method: Optional treatment method to filter by.
                If None, returns factors for all methods.

        Returns:
            List of factor description dictionaries.
        """
        self._increment_lookups()
        results: List[Dict[str, Any]] = []

        if treatment_method is not None:
            key = treatment_method.upper().replace(" ", "_")
            methods_to_list = [key] if key in TREATMENT_METHODS else []
        else:
            methods_to_list = list(TREATMENT_METHODS.keys())

        for method_key in methods_to_list:
            info = TREATMENT_METHODS[method_key]
            entry: Dict[str, Any] = {
                "treatment_method": method_key,
                "name": info.name,
                "applicable_waste_types": list(info.applicable_waste_types),
                "primary_gases": list(info.primary_gases),
                "factor_tables": [],
            }

            # Identify which factor tables apply
            if method_key in (
                "MANAGED_ANAEROBIC", "MANAGED_SEMI_AEROBIC",
                "UNMANAGED_DEEP", "UNMANAGED_SHALLOW",
            ):
                entry["factor_tables"].extend([
                    "DOC_VALUES", "MCF_VALUES", "HALF_LIFE_VALUES",
                    "OXIDATION_FACTOR", "DOCF_VALUES",
                ])

            if method_key == "INCINERATION":
                entry["factor_tables"].extend([
                    "INCINERATION_EF", "CARBON_CONTENT", "NCV_VALUES",
                ])

            if method_key in ("COMPOSTING", "ANAEROBIC_DIGESTION", "MBT"):
                entry["factor_tables"].append("COMPOSTING_EF")

            if method_key == "OPEN_BURNING":
                entry["factor_tables"].append("OPEN_BURNING_EF")

            if method_key == "WASTEWATER_TREATMENT":
                entry["factor_tables"].extend([
                    "WASTEWATER_MCF", "BO_VALUES",
                ])

            if method_key == "LANDFILL_GAS_RECOVERY":
                entry["factor_tables"].append("LFG_COLLECTION_EFFICIENCY")

            # Always available
            entry["factor_tables"].append("GWP_VALUES")

            results.append(entry)

        return results

    # ------------------------------------------------------------------
    # Custom Factor Management
    # ------------------------------------------------------------------

    def register_custom_factor(
        self,
        factor_type: str,
        key: str,
        value: Decimal,
        source: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Register a custom emission factor or reference value.

        Custom factors are stored in memory and can supplement or override
        IPCC defaults for site-specific or Tier 2/3 calculations.

        Args:
            factor_type: Type of factor (e.g. "DOC", "MCF", "NCV",
                "INCINERATION_EF", "COMPOSTING_EF").
            key: Lookup key (e.g. "FOOD_WASTE", "MANAGED_ANAEROBIC").
            value: Factor value as Decimal.
            source: Citation or source for the custom value.
            description: Optional description.

        Returns:
            Registration confirmation with provenance hash.
        """
        with self._lock:
            if factor_type not in self._custom_factors:
                self._custom_factors[factor_type] = {}

            record = {
                "value": str(value),
                "source": source,
                "description": description,
                "registered_at": _utcnow().isoformat(),
            }
            self._custom_factors[factor_type][key] = record

        result = {
            "status": "REGISTERED",
            "factor_type": factor_type,
            "key": key,
            "value": str(value),
            "source": source,
            "provenance_hash": _compute_hash(record),
        }

        logger.info(
            "Custom factor registered: type=%s, key=%s, value=%s, source=%s",
            factor_type, key, value, source,
        )
        return result

    def get_custom_factor(
        self,
        factor_type: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a registered custom factor.

        Args:
            factor_type: Type of factor.
            key: Lookup key.

        Returns:
            Custom factor record or None if not found.
        """
        with self._lock:
            if factor_type not in self._custom_factors:
                return None
            return self._custom_factors[factor_type].get(key)

    def list_custom_factors(self) -> Dict[str, Dict[str, Any]]:
        """List all registered custom factors.

        Returns:
            Dictionary of all custom factors keyed by (factor_type, key).
        """
        with self._lock:
            return {
                factor_type: dict(factors)
                for factor_type, factors in self._custom_factors.items()
            }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with lookup counts and engine metadata.
        """
        with self._lock:
            custom_count = sum(
                len(v) for v in self._custom_factors.values()
            )
            return {
                "engine": "WasteTreatmentDatabaseEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_lookups": self._total_lookups,
                "cache_size": len(self._cache),
                "custom_factors_registered": custom_count,
                "waste_categories": len(WasteCategory),
                "treatment_methods": len(TreatmentMethod),
                "incinerator_types": len(IncineratorType),
                "composting_types": len(CompostingType),
                "landfill_types": len(LandfillType),
                "wastewater_system_types": len(WastewaterSystemType),
                "doc_entries": len(DOC_VALUES),
                "mcf_entries": len(MCF_VALUES),
                "carbon_content_entries": len(CARBON_CONTENT),
                "incineration_ef_entries": len(INCINERATION_EF),
                "composting_ef_entries": len(COMPOSTING_EF),
                "wastewater_mcf_entries": len(WASTEWATER_MCF),
                "ncv_entries": len(NCV_VALUES),
                "open_burning_ef_entries": len(OPEN_BURNING_EF),
                "bo_entries": len(BO_VALUES),
                "half_life_zones": len(HALF_LIFE_VALUES),
                "defra_categories": len(DEFRA_FACTORS),
                "epa_warm_categories": len(EPA_WARM_FACTORS),
                "lfg_collection_types": len(LFG_COLLECTION_EFFICIENCY),
            }

    def reset(self) -> None:
        """Reset engine state (custom factors, cache, counters).

        Intended for testing teardown.
        """
        with self._lock:
            self._custom_factors.clear()
            self._cache.clear()
            self._total_lookups = 0
        logger.info("WasteTreatmentDatabaseEngine reset")
